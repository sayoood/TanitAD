"""Mutation check of the PREREG_V8F_SPEEDAVG instruments: every mutant must make `stack/tests/test_v8f_speedavg.py` go RED BY AN ASSERTION.

"Mutate, do not inspect" (CLAUDE.md, reachability rule 3): a test that is green on the right code says nothing until it is shown to go red on the wrong code.
For each mutant ONE exact-substring edit (count must be 1) is applied to ONE file of a private copy of the tree, the test file runs (`-x`), and the verdict is read from pytest's own summary:

  KILLED    the run failed and the first failure is an assertion (`assert ...` / AssertionError): the test caught the defect by testing for it
  UNATTRIBUTED  (the `...e` variants, which run the end-to-end tests ALONE) red, but not by the control the brief names
  CRASHED   the run failed but NOT on an assertion (an ImportError, a SystemExit, a KeyError ...): red for the wrong reason, it does not count
  SURVIVED  the run stayed green: the mutation is not pinned by any test (a finding, not a pass)
  BAD       the mutant does not compile, or its `old` text is not found exactly once (a bug in this script)

The unmutated tree must be green before the first mutant and again after the last (every mutated file is restored and its sha256 compared with the source).
Exit 0 only if the baseline is green, EVERY mutant is KILLED and the restored tree is green.

Usage:   python mutate_speedavg.py --work <SHORT dir> [--src <repo root>] [--only M01,M02] [--out <dir>]
CPU only (CUDA_VISIBLE_DEVICES=-1, 1 thread per run); ~25 s per mutant; refuses to run with < 2 GB free; the private copy lives under --work and is removed with --clean."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PKREL = Path("TanitAD Research Lab") / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
DRV = "stack/scripts/v8f_mini_eval_speedavg.py"
RD = (PKREL / "code" / "v8f_speedavg_read.py").as_posix()
RN = (PKREL / "code" / "devbox_ops" / "run_speedavg.sh").as_posix()
TEST = "stack/tests/test_v8f_speedavg.py"
MIN_FREE = 2 * 1024 ** 3

# ------------------------------------------------------------------------------------------------ the mutants: (id, file, [(old, new), ...], what it breaks, the test that should catch it)
MUTANTS = [
    # --- the four the brief names
    ("M01", DRV, [("    out[:, idx] = src[:, idx]\n", "    out[:, idx[:1]] = src[:, idx[:1]]\n")],
     "AVERAGE ONE CELL ONLY (replace_cells writes the first speed cell, not five)", "control_2 / the live average"),
    ("M02", DRV, [('    if mode == "self":\n        return model\n', '    if mode == "self":\n        return other\n')],
     "SWAP SEEDS: `self` (control 1) averages X_s with the OTHER seed instead of with itself", "control_1"),
    ("M03", RD, [('        out = "VARIANCE PAYS"\n', '        out = "@@"\n'), ('        out = "BIAS, NOT VARIANCE"\n', '        out = "VARIANCE PAYS"\n'), ('        out = "@@"\n', '        out = "BIAS, NOT VARIANCE"\n')],
     "SWAP OUTCOME ROWS: VARIANCE PAYS <-> BIAS, NOT VARIANCE in `decide`", "decision table"),
    ("M04", RD, [('return bool(g["point"] <= -F and g["separated"] and g["point"] < 0)', 'return bool(g["point"] <= -F and g["point"] < 0)')],
     "DROP THE SEPARATION TEST from PRIMARY", "primary / separation"),
    # --- the averaging rule and the cells
    ("M05", DRV, [("    return (a + b) * 0.5\n", "    return a * 0.6 + b * 0.4\n")], "averaging weights 0.6 / 0.4 instead of the plain mean", "mean_cells / avg cells are the mean"),
    ("M06", DRV, [('SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")', 'SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms")')],
     "the speed cell set loses v_hi_ms (four cells, not the PREREG's five)", "cell set pin"),
    ("M07", DRV, [("    out[:, idx] = src[:, idx]\n", "    out[:] = src\n")], "the hook averages ALL 60 cells, not only the five speed cells", "non-speed cells untouched"),
    ("M08", DRV, [('    if mode == "avg":\n        return other\n', '    if mode == "avg":\n        return model\n')], "`avg` averages X_s with ITSELF (the average moves nothing)", "avg is live"),
    ("M09", DRV, [("        return torch.where(has[:, None], gt, own)\n", "        return gt\n")], "`gt` has no fallback to T's own code where the window has no v9 row", "control_2 / gt fallback"),
    ("M10", DRV, [("            self.M.V8FMini.forward_diag = self.orig\n            self.orig = None\n", "            self.orig = None\n")],
     "the wrapped forward_diag is never restored", "restore"),
    ("M11", DRV, [("        if k not in s0 or not torch.equal(s0[k], s1[k]):", "        if False:")], "pair_violations no longer requires bitwise-equal cons_mu / cons_sd", "pair guard"),
    # --- the reader: constants, the table, the predicates
    ("M12", RD, [("F_BAR = 0.1207", "F_BAR = 0.1")], "F moved after the fact (0.1207 -> 0.1)", "F pin / boundary"),
    ("M13", RD, [('g["point"] <= -F and', 'g["point"] < -F and')], "PRIMARY boundary: exactly -F no longer meets it", "boundary at exactly -F"),
    ("M14", RD, [('        out = "PARTIAL"\n', '        out = "MIXED"\n')], "PARTIAL row routed to MIXED", "decision table"),
    ("M15", RD, [("    elif not any(sep.values()):\n", "    elif False:\n")], "BIAS, NOT VARIANCE row unreachable", "decision table"),
    ("M16", RD, [('    rows["lead_viol"] = lv\n', "")], "LEAD-VIOL dropped from SECONDARY", "secondary: lead_viol"),
    ("M17", RD, [('and g["lo"] > 0)', 'and g["hi"] > 0)')], "separated-worse reads the CI's upper end, not its lower end", "secondary: separated-worse"),
    ("M18", RD, [('    num = float(np.mean(own) - np.mean(avg))\n', '    num = float(np.mean(avg) - np.mean(own))\n')], "CAPTURE numerator sign flipped", "capture"),
    ("M19", RD, [('TIER = "T1-class open loop (own held outputs; no recorded future action)"', 'TIER = "T0-open-loop (teacher-forced)"')], "the tier stamp is the eval json's mislabelled T0 string", "no T0 string"),
    ("M20", RD, [('    "PARTIAL": ("PRIMARY < 0 and separated at both seeds but above \u2212F",', '    "PARTIAL": ("PRIMARY < 0 and separated at both seeds but above 0",')], "the PARTIAL condition text edited (the committed table is not the PREREG's)", "table equals PREREG"),
    # --- the refusals and the controls
    ("M21", RD, [("TOL_M = 1e-4", "TOL_M = 1.0")], "the controls' per-window tolerance loosened 1e-4 m -> 1 m", "control arms"),
    ("M22", RD, [('    if b.get("limit") and not allow_smoke:\n', "    if False:\n")], "a LIMITED (smoke) run is no longer refused", "limited-run refusal"),
    ("M23", RD, [('    if (b.get("x0_sha256"), b.get("x1_sha256")) != (sha0, sha1):\n', "    if False:\n")], "the checkpoint sha256 check removed", "wrong-sha refusal"),
    ("M24", RD, [('    if (b.get("mode"), b.get("seed"), b.get("tag"), j.get("tag")) != (mode, seed, tag, tag):\n', "    if False:\n")], "the mode / seed / tag check removed", "mislabelled-mode refusal"),
    ("M25", RD, [(' and r["control_3_own_equals_banked"]["PASS"]', "")], "control 3 no longer gates the controls' PASS", "control_3 arm"),
    ("M26", RD, [('    out["cross_seed_same_pair"] = x\n', '    x["PASS"] = True\n    out["cross_seed_same_pair"] = x\n')], "the cross-seed same-pair control forced to PASS", "cross-seed arm"),
    ("M27", RD, [('bool(d1 == 0.0 and r["control_1_self_average_is_identity"]["pred_equal"]', 'bool(d1 <= 1e-3 and r["control_1_self_average_is_identity"]["pred_equal"]')],
     "control 1 is no longer EXACTLY 0.0 per window (1e-3 m allowed)", "control_1 exactness"),
    # --- the runner
    ("M28", RN, [('stop() { say "ZZSPEEDAVG-STOP $*"; exit 3; }', 'stop() { say "ZZSPEEDAVG-HALT $*"; exit 3; }')], "the STOP marker renamed", "runner markers"),
    ("M29", RN, [(' || stop "KINDX\'s seed-1 read (', ' || say "KINDX\'s seed-1 read (')], "the runner no longer stops when KINDX's s1 read is absent (the arm is CONDITIONAL)", "runner refuses without KINDX s1"),
    ("M30", RN, [("MIN_FREE_KB=2097152", "MIN_FREE_KB=0")], "the free-space guard is off", "runner load-bearing lines"),
    ("M31", RN, [("c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250e")], "X_s0's committed sha256 edited", "runner load-bearing lines"),
    ("M32", RN, [(' || stop "X_s0 and X_s1 are the same file: no second seed"', " || true")], "the runner accepts X_s0 == X_s1 (no second seed)", "runner load-bearing lines"),
    ("M33", RN, [(' || stop "sha256 of $dst is $got, expected $want"', " || true")], "a pulled checkpoint's sha256 mismatch no longer stops the runner", "runner load-bearing lines"),
]


#: the brief's attribution claims, run on the END-TO-END tests alone (`-k "e2e or arm"`: the pure-piece tests are deselected), so it is the CONTROLS that must catch the defect, not a unit test:
#: (id, base mutant, the controls / guard blocks of which at least ONE must read red in the e2e assertion message)
E2E_ONLY = [
    ("M01e", "M01", ["control_2_gt_equals_kindx_speed_oracle"]),                          # averaging ONE cell: control 2 (GT in the five cells != KINDX's speed oracle)
    ("M02e", "M02", ["control_1_self_average_is_identity"]),                              # the seeds swapped in `self`: control 1
    ("M05e", "M05", ["control_1_self_average_is_identity", "guards"]),                    # weights 0.6 / 0.4
    ("M07e", "M07", ["control_2_gt_equals_kindx_speed_oracle", "guards"]),                # ALL cells averaged
    ("M08e", "M08", ["guards", "cross_seed_same_pair"]),                                  # `avg` with itself: the average moves nothing
    # (no M09e: the synthetic world has no window without a v9 row, so the end-to-end tests cannot see a missing gt fallback; the pure-piece test and the real-data smoke -- 17 such windows -- do)
]
E2E_K = "e2e or arm"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def free_bytes(p: Path) -> int:
    q = p
    while not q.exists():
        q = q.parent
    return shutil.disk_usage(q).free


def build_tree(src: Path, work: Path):
    """A private copy of exactly what the test file needs (no raw/ banks, no caches)."""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "raw", ".git")
    for rel in ("stack", "taniteval", "Project Steering"):
        if (src / rel).exists():
            shutil.copytree(src / rel, work / rel, ignore=ign, dirs_exist_ok=True)
    pk = src / PKREL
    for f in pk.glob("*"):
        if f.is_file():
            (work / PKREL).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, work / PKREL / f.name)
    if (pk / "code").exists():
        shutil.copytree(pk / "code", work / PKREL / "code", ignore=ign, dirs_exist_ok=True)


def red_controls(e_lines) -> list:
    """The controls / guard blocks whose `"PASS"` reads false in the e2e test's assertion message (it dumps the controls as json): which control caught the defect."""
    cur, red = None, []
    for ln in e_lines:
        m = re.search(r'"(control_\d_\w+|guards|cross_seed_same_pair)": \{', ln)
        if m:
            cur = m.group(1)
        if re.search(r'"(guards_)?PASS": false', ln):
            nm = "guards" if "guards_PASS" in ln else cur
            if nm and nm not in red:
                red.append(nm)
    return red


def run_tests(work: Path, timeout: int, k: str | None = None) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(work / "stack"), str(work / "taniteval")]))
    bt = work / "bt"
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short", "-rf", "-p", "no:cacheprovider", f"--basetemp={bt}", TEST]
    if k:
        cmd[cmd.index(TEST):cmd.index(TEST)] = ["-k", k]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"TIMEOUT after {timeout}s\n" + ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
    shutil.rmtree(bt, ignore_errors=True)
    failed = re.findall(r"^FAILED (\S+::[^\[\s]+(?:\[.*?\])?)(?: - (.*))?$", out, flags=re.M)
    e_all = [ln for ln in out.splitlines() if ln.startswith("E ")]
    errs = [ln.strip() for ln in e_all][:3]
    m = re.search(r"(\d+) passed", out)
    first = failed[0] if failed else None
    kind = None
    if first is not None:
        msg = first[1] or ""
        kind = "assertion" if (msg.startswith("assert") or msg.startswith("AssertionError")) else "other"
        if kind == "other" and (any(e.startswith("E   assert") or e.startswith("E   AssertionError") for e in e_all) or any("DID NOT RAISE" in e for e in e_all)):
            kind = "assertion"        # (`pytest.raises` -> "DID NOT RAISE" is the test failing by its own expectation)
    elif rc != 0:
        kind = "assertion" if any(e.startswith("E   assert") or e.startswith("E   AssertionError") or "DID NOT RAISE" in e for e in e_all) else "other"   # a FAILED line the regex missed, else a collection error / timeout
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "passed": int(m.group(1)) if m else 0, "first_failed": first[0] if first else None,
            "first_failed_msg": (first[1] if first else None), "kind": kind, "E_lines": errs, "red_controls": red_controls(e_all), "tail": out.strip().splitlines()[-4:]}


def apply_mutant(work: Path, rel: str, edits) -> str | None:
    p = work / rel
    text = p.read_text(encoding="utf-8")
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            return f"`old` text found {n} times (must be exactly 1): {old[:80]!r}"
        text = text.replace(old, new)
    if rel.endswith(".py"):
        try:
            compile(text, rel, "exec")
        except SyntaxError as e:
            return f"the mutant does not compile: {e}"
    p.write_text(text, encoding="utf-8", newline="\n")
    return None


def bash_n(work: Path, rel: str, bash: str | None) -> str | None:
    if not rel.endswith(".sh") or not bash:
        return None
    r = subprocess.run([bash, "-n", str(work / rel)], capture_output=True, text=True, timeout=60)
    return None if r.returncode == 0 else f"bash -n: {r.stderr.strip()[:200]}"


def find_bash():
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash")):
        if c and os.path.exists(c):
            try:
                r = subprocess.run([c, "-c", "echo ok; command -v df >/dev/null && echo df"], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.stdout.split() == ["ok", "df"]:
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="a SHORT directory for the private copy (Windows MAX_PATH)")
    ap.add_argument("--src", default=str(HERE.parents[6]), help="the repo root holding the files under test (default: derived from this file's place in the tree)")
    ap.add_argument("--only", default="", help="comma-separated mutant ids")
    ap.add_argument("--out", default=str(HERE.parent))
    ap.add_argument("--timeout", type=int, default=200)
    ap.add_argument("--clean", action="store_true", help="remove the private copy at the end")
    a = ap.parse_args()
    src, work, out = Path(a.src), Path(a.work), Path(a.out)
    if free_bytes(work) < MIN_FREE:
        print(f"ABORT: < 2 GB free on the drive of {work}")
        return 2
    only = {x for x in a.only.split(",") if x}
    muts = [m for m in MUTANTS if not only or m[0] in only]
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    build_tree(src, work)
    bash = find_bash()
    files = sorted({m[1] for m in MUTANTS})
    for rel in files + [TEST]:
        if not (work / rel).exists():
            print(f"ABORT: {rel} is not in {src}")
            return 2
    res = {"src": str(src), "tests": TEST, "n_mutants": len(muts), "baseline_before": None, "mutants": [], "baseline_after": None}
    print(f"mutation check of {len(muts)} mutants (+ the end-to-end-only variants); copy under {work}; python {sys.version.split()[0]}", flush=True)

    base = run_tests(work, a.timeout)
    res["baseline_before"] = base
    print(f"BASELINE (unmutated): rc {base['rc']}  {base['passed']} passed  {base['seconds']} s", flush=True)
    if base["rc"] != 0:
        print("ABORT: the unmutated tree is not green", base["tail"])
        (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 1

    base = {m[0]: m for m in MUTANTS}
    plan = [(m[0], m[1], m[2], m[3], m[4], None, None) for m in muts]
    plan += [(eid, base[bid][1], base[bid][2], base[bid][3] + "  [END-TO-END TESTS ONLY]", "the controls", E2E_K, must) for eid, bid, must in E2E_ONLY if (not only or eid in only or bid in only)]
    res["n_mutants"] = len(plan)
    for mid, rel, edits, what, hint, kexpr, must in plan:
        if free_bytes(work) < MIN_FREE:
            print("ABORT: < 2 GB free")
            return 2
        orig = (src / rel).read_bytes()
        err = apply_mutant(work, rel, edits) or bash_n(work, rel, bash)
        row = {"id": mid, "file": rel, "breaks": what, "expected_catcher": hint}
        if err:
            row.update(verdict="BAD", detail=err)
        else:
            r = run_tests(work, a.timeout, kexpr)
            row.update(r)
            row["verdict"] = "KILLED" if (r["rc"] != 0 and r["kind"] == "assertion") else ("CRASHED" if r["rc"] != 0 else "SURVIVED")
            if must is not None:
                row["must_be_red_one_of"] = must
                row["attributed"] = bool(set(must) & set(r["red_controls"]))
                if row["verdict"] == "KILLED" and not row["attributed"]:
                    row["verdict"] = "UNATTRIBUTED"                                 # red, but not by the control the brief names
        (work / rel).write_bytes(orig)                                           # restore at once
        assert sha(work / rel) == sha(src / rel)
        res["mutants"].append(row)
        print(f"{mid} {row['verdict']:12s} {row.get('seconds', '-')!s:>6} s  {what}\n      caught by: {row.get('first_failed')}  [{row.get('first_failed_msg')}]  red controls: {row.get('red_controls')}", flush=True)
        (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")   # bank incrementally

    for rel in files:
        assert sha(work / rel) == sha(src / rel), f"{rel} was not restored"
    after = run_tests(work, a.timeout)
    res["baseline_after"] = after
    print(f"BASELINE (restored): rc {after['rc']}  {after['passed']} passed  {after['seconds']} s", flush=True)
    counts = {v: sum(1 for m in res["mutants"] if m["verdict"] == v) for v in ("KILLED", "CRASHED", "SURVIVED", "UNATTRIBUTED", "BAD")}
    res["summary"] = counts
    ok = bool(after["rc"] == 0 and counts["KILLED"] == len(plan))
    res["ALL_KILLED_AND_RESTORED_GREEN"] = ok
    (out / "mutation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"SUMMARY {counts}  all killed and the restored tree green: {ok}", flush=True)
    if a.clean:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
