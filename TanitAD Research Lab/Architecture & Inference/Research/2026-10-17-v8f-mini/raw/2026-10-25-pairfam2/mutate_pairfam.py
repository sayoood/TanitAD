"""Mutation proof for `stack/scripts/v8f_paired_families.py`: every mutant must turn `stack/tests/test_v8f_paired_families.py` RED, and the unmutated tool must be GREEN before and after.

    python mutate_pairfam.py <root with stack/> <python exe> [--check] [--list] [--only <substring of a mutant's name>]

The tool is NEVER edited: each mutant is a copy of its source written to <root>/mutants/, and the suite is pointed at it with the environment variable
V8F_PAIRFAM_SRC (the test file loads the tool from there when it is set). The original file's sha256 is asserted before and after.
Each mutant is one (or, where stated, two) exact-substring replacement(s); an anchor that does not occur EXACTLY once aborts the run (`--check` verifies that and exits).
RED is judged from pytest's own summary: the mutant must make at least one test FAIL, and the line says whether it failed on an assertion (`AssertionError` / `Failed:`)
or on another exception (still RED, but flagged: a crash is a weaker signal than a wrong-value assertion).
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
CHECK = "--check" in sys.argv
LIST = "--list" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
TOOL = ROOT / "stack" / "scripts" / "v8f_paired_families.py"
TEST = "stack/tests/test_v8f_paired_families.py"
MUT_DIR = ROOT / "mutants"
BASETEMP = ROOT / "pytmp_mut"

PAIR = '    pair = paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed, reduce=reduce)'

MUTANTS = [
    # ---- pairing -----------------------------------------------------------------------------------------------------------------------------------
    ("pairing broken: B's windows are reversed before the paired bootstrap (B's episodes shuffled)",
     [(PAIR, PAIR.replace("(a, b, eid", "(a, b[::-1].copy(), eid"))]),
    ("quadrature: the interval is built by combining the two per-arm intervals",
     [('"delta": pair["delta"], "lo": pair["lo"], "hi": pair["hi"],',
       '"delta": pair["delta"], "lo": pair["delta"] - math.hypot(ra["ci95"], rb["ci95"]), "hi": pair["delta"] + math.hypot(ra["ci95"], rb["ci95"]),')]),
    ("separated is inverted", [('"separated": bool(pair["separated"])', '"separated": not bool(pair["separated"])')]),
    ("the bootstrap seed is not the requested one", [(PAIR, PAIR.replace("seed=seed", "seed=seed + 1"))]),
    # ---- per-resample F1 ---------------------------------------------------------------------------------------------------------------------------
    ("head macro-F1 computed ONCE (on the first call) instead of on every resample",
     [('        return float(C.macro_f1(v // (k + 1), v % (k + 1), k)["macro_f1"])',
       '        return _F1_ONCE.setdefault(k, float(C.macro_f1(v // (k + 1), v % (k + 1), k)["macro_f1"]))'),
      ("N_BOOT = 2000\n", "N_BOOT = 2000\n_F1_ONCE = {}\n")]),
    ("goal macro-F1 computed ONCE instead of on every resample",
     [('        f1, valid = self._f1_vec(v)\n        return float(f1[valid].mean()) if valid.any() else float("nan")',
       '        if not hasattr(self, "_once"):\n            f1, valid = self._f1_vec(v)\n            self._once = float(f1[valid].mean()) if valid.any() else float("nan")\n        return self._once')]),
    ("the goal decode reads the weight mask as the label", [("        return self.gy[idx], self.gw[idx], p", "        return self.gw[idx], self.gw[idx], p")]),
    # ---- direction ---------------------------------------------------------------------------------------------------------------------------------
    ("A and B swapped in the paired call", [(PAIR, PAIR.replace("(a, b, eid", "(b, a, eid"))]),
    ("A and B swapped when the dumps are loaded", [("    A, B = load_dump(a.a), load_dump(a.b)", "    A, B = load_dump(a.b), load_dump(a.a)")]),
    # ---- alignment ---------------------------------------------------------------------------------------------------------------------------------
    ("the alignment check is dropped",
     [("    facts = check_alignment(A, B, ctx)", '    facts = {"n_windows": len(ctx["eid"]), "n_episodes": int(len(np.unique(ctx["eid"])))}')]),
    ("the A-vs-B episode-id comparison is dropped", [("    if ne.size:\n        i = int(ne[0])", "    if False:\n        i = int(ne[0])")]),
    ("the per-window ADE6 recompute (the within-episode order check) is dropped", [("        if not dev <= TOL_ADE6:", "        if False:")]),
    ("the ck_pos_err NaN-mask comparison is dropped", [("        if not same:", "        if False:")]),
    # ---- the strategic family ----------------------------------------------------------------------------------------------------------------------
    ("strategic position error: the label mask is not applied (NaN -> 0 over every window)",
     [('_cell(A["ck_pos_err"][ok, m], B["ck_pos_err"][ok, m], eid[ok], "mean"',
       '_cell(np.nan_to_num(A["ck_pos_err"][:, m], nan=0.0), np.nan_to_num(B["ck_pos_err"][:, m], nan=0.0), eid, "mean"')]),
    ("strategic position error reads the next arc's column",
     [('A["ck_pos_err"][ok, m], B["ck_pos_err"][ok, m]', 'A["ck_pos_err"][ok, (m + 1) % 3], B["ck_pos_err"][ok, (m + 1) % 3]')]),
    ("the heading cells carry the POSITION error", [('eA["head_err_deg"][ok, m], eB["head_err_deg"][ok, m]', 'eA["pos_err_m"][ok, m], eB["pos_err_m"][ok, m]')]),
    ("the heading rebuild uses the wrong source frame", [('torch.as_tensor(ctx["pose_s"]), torch.as_tensor(ctx["pose_now"])).numpy()\n    return C.checkpoint_errors',
                                                         'torch.as_tensor(ctx["pose_now"]), torch.as_tensor(ctx["pose_now"])).numpy()\n    return C.checkpoint_errors')]),
    ("the heading gate is disabled", [("TOL_HEADING_GATE = 1e-4", 'TOL_HEADING_GATE = float("inf")')]),
    ("STRATEGIC is reported as paused", [('STRATEGIC_STATUS = "ACTIVE for v8F-mini (PI-V8F-2026-10-17; R72/R73): reported in full"', 'STRATEGIC_STATUS = "PAUSED (PI_DECISION_QUEUE ITEM 25)"')]),
    # ---- n and reasons -----------------------------------------------------------------------------------------------------------------------------
    ("the strategic reason's no-row count is not the mask's", [("        n_no_row = int((~has).sum())\n        n_row_bad", "        n_no_row = 0\n        n_row_bad")]),
    ("the head reason's IGNORE count is not the mask's", [("        n_ign = int((has & (cls < 0)).sum())", "        n_ign = 0")]),
    ("the bearing mask is dropped (a bearing is scored to a goal on the origin)", [("    move = rg > max(ff.MIN_DS_M, 0.5)", "    move = rg > -1.0")]),
    ("a goal token with no positive is no longer UNAVAILABLE", [("        if nlab == 0 or npos == 0:", "        if False:")]),
    # ---- the definitions ---------------------------------------------------------------------------------------------------------------------------
    ("kappa is halved", [('        return float("nan") if r is None else float(r)\n    return _kap', '        return float("nan") if r is None else float(r) * 0.5\n    return _kap')]),
    ("the lateral head decision is the argmin", [("ca = (true * (k + 1) + A[logits].argmax(-1)[m]).astype(np.float64)", "ca = (true * (k + 1) + A[logits].argmin(-1)[m]).astype(np.float64)")]),
    ("the goal head threshold is 0.6", [("return ((np.asarray(goal_prob) > 0.5).astype(np.int64)", "return ((np.asarray(goal_prob) > 0.6).astype(np.int64)")]),
    ("the lateral decision is scored against the longitudinal GT class", [('("lateral_decision", lat_g, la, lb, 3)', '("lateral_decision", lon_g, la, lb, 3)')]),
    # ---- the controls ------------------------------------------------------------------------------------------------------------------------------
    ("the identity control can never fail", [('"not_zero": bad, "ok": not bad}', '"not_zero": bad, "ok": True}')]),
    ("the identity control is not run", [("    if a.identity_control:\n        controls.append(identity_control", "    if False:\n        controls.append(identity_control")]),
    ("the JSON-reproduction tolerance is 1e9", [("TOL_REPRO = 1e-6", "TOL_REPRO = 1e9")]),
    ("the JSON-reproduction control is not run", [("        if a.json_controls and j is not None and isinstance(j.get(\"raw\"), dict):", "        if False:")]),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_suite(env_src=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=f"{ROOT / 'stack'}{os.pathsep}{ROOT / 'taniteval'}")
    env.pop("V8F_PAIRFAM_SRC", None)
    if env_src:
        env["V8F_PAIRFAM_SRC"] = str(env_src)
    cmd = [PY, "-m", "pytest", TEST, "-q", "-x", "-rf", "-p", "no:cacheprovider", f"--basetemp={BASETEMP}"]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    finally:
        shutil.rmtree(BASETEMP, ignore_errors=True)
    return r.returncode, r.stdout + r.stderr


def verdict(rc, out):
    """RED = the suite failed. The kind comes from the first `E ` line pytest prints under the failing test: an AssertionError / `Failed:` is a wrong-value
    assertion; anything else (KeyError, TypeError, ...) is a crash, which still turns the suite RED but is a weaker signal."""
    if rc == 0:
        return "SURVIVED", None, None
    fails = re.findall(r"^FAILED (\S+)", out, re.M)
    if not fails:
        return "ERROR (no test failed: collection / import problem)", None, out[-400:]
    e = re.search(r"^E\s+(\S.*)$", out, re.M)
    msg = e.group(1) if e else ""
    kind = "assertion" if msg.startswith(("AssertionError", "assert", "Failed")) else "exception"
    return f"RED ({kind})", fails[0].split("::")[-1], msg[:150]


def main():
    src = TOOL.read_text(encoding="utf-8")
    base_sha = sha(TOOL)
    bad = []
    for i, (name, edits) in enumerate(MUTANTS):
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                bad.append((i, name, old[:70], n))
            if old == new:
                bad.append((i, name, "old == new", 0))
    if bad:
        for b in bad:
            print("ANCHOR PROBLEM:", b)
        sys.exit(2)
    print(f"{len(MUTANTS)} mutants; every anchor occurs exactly once in the ORIGINAL bytes ({TOOL.name} sha256 {base_sha[:16]}...)")
    if LIST:
        for i, (name, _) in enumerate(MUTANTS):
            print(f"{i:2d}  {name}")
    if CHECK or LIST:
        return
    MUT_DIR.mkdir(exist_ok=True)
    rc, out = run_suite()
    print(f"BASELINE (unmutated): {'GREEN' if rc == 0 else 'NOT GREEN'}  {out.strip().splitlines()[-1] if out.strip() else ''}")
    if rc != 0:
        print(out[-2000:])
        sys.exit(1)
    summary = []
    for i, (name, edits) in enumerate(MUTANTS):
        if ONLY and ONLY not in name:
            continue
        s = src
        for old, new in edits:
            s = s.replace(old, new, 1)
        mp = MUT_DIR / f"mutant_{i:02d}.py"
        mp.write_text(s, encoding="utf-8", newline="\n")
        rc, out = run_suite(mp)
        v, test, msg = verdict(rc, out)
        summary.append((i, name, v, test))
        print(f"[{i:2d}] {v:22s} {name}\n       first failing test: {test}  {('| ' + msg) if msg else ''}", flush=True)
        mp.unlink()
    shutil.rmtree(MUT_DIR, ignore_errors=True)
    assert sha(TOOL) == base_sha, "the tool's bytes changed during the run"
    rc, out = run_suite()
    print(f"AFTER (unmutated, bytes identical: {sha(TOOL) == base_sha}): {'GREEN' if rc == 0 else 'NOT GREEN'}  {out.strip().splitlines()[-1] if out.strip() else ''}")
    n_red = sum(v.startswith("RED") for _, _, v, _ in summary)
    n_assert = sum(v == "RED (assertion)" for _, _, v, _ in summary)
    surv = [(i, n) for i, n, v, _ in summary if not v.startswith("RED")]
    print(f"\n{n_red} of {len(summary)} mutants RED ({n_assert} on an assertion); survived or errored: {surv if surv else 'none'}")
    sys.exit(0 if (not surv and rc == 0) else 1)


main()
