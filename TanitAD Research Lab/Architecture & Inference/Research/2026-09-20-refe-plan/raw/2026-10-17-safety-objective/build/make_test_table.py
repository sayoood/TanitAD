#!/usr/bin/env python3
"""make_test_table.py <PATCHED TREE> <PACKAGE DIR> : judge the sweep logs and write <PACKAGE DIR>/TEST_TABLE.md.
Inputs (all written by sweep.sh / run_tests.sh, one `SUMMARY.txt` line per test: `<name> rc=<n> t=<s>s tok=<ZZ... token>`):
  C:/so/logs_t1, C:/so/logs_t2   the PATCHED tree, this sweep (env SO_PATCHED_DIRS = comma list, later overrides earlier; logs_t3 / logs_t4 = the two tests that read test_rl_safeobj.py, rerun on the hardened tree)
  C:/so/logs_b, C:/so/logs_b2    the UNPATCHED tip (tree C:/so/b), run before the PC restart by the previous agent (a static tree: the result does not age)
  C:/so/logs_tb                  the UNPATCHED tree, rerun now for every test that failed on the patched tree
Verdict: PASS (rc 0 and token) | PRE-EXISTING (fails on both, same last line) | REGRESSION (passes on the baseline and fails patched, or the pinned count DROPPED) | CHECK (fails on both with different last lines)."""
import hashlib
import os
import re
import sys

PK_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ORDER = ["refe/test_rl_safeobj.py", "refe/test_rl_s1b.py", "refe/test_rl_r0c.py", "refe/test_rl_epo.py", "refe/test_launch_gate.py", "refe/test_rls1_chain.py", "refe/test_rls1_queue.py",
         "refe/test_dump_scores.py", "refe/test_aux_s1_wiring.py", "refe/test_s1aux_chain.py", "refe/test_s1c.py", "refe/test_gate_suite_prerun.py", "refe/test_gate_gpu_probes.py",
         "refe/test_p2r_chain.py", "refe/test_p2rep_chain.py", "refe/test_p6_chain.py", "refe/test_p6_levers.py", "refe/test_p6_cpu_probe.py", "refe/test_p6_overlap.py",
         "refe/test_rl_terms_label.py", "refe/test_rl_tlane_label.py", "refe/test_label_dir_refusal.py", "refe/test_flock_rw.py", "refe/test_aux_g12_bf16.py",
         "eval/test_aux_navtest_forward.py", "eval/test_rl_val_eval.py", "eval/test_safeobj_read.py", "eval/test_safeobj_gate.py"]
TOUCHED = ["refe/scorer_finetune.py", "refe/launch_gate.py", "refe/gate_checks.py", "refe/rls1_checks.py", "refe/rls1_pod.sh", "refe/test_rl_s1b.py", "refe/test_rl_r0c.py",
           "refe/test_rl_epo.py", "refe/test_launch_gate.py", "refe/test_rls1_chain.py", "refe/test_rls1_queue.py", "refe/test_dump_scores.py", "refe/test_aux_s1_wiring.py",
           "refe/test_rl_safeobj.py", "eval/rls1d_arms.json", "eval/safeobj_read.py", "eval/test_safeobj_read.py", "eval/test_safeobj_gate.py", "refe/rl_reward.py"]


def parse(path):
    out = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8", errors="replace"):
            m = re.match(r"(\S+) rc=(\d+) t=(\d+)s tok=(.*)$", ln.strip())
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)), m.group(4).rstrip("|").split("|")[-1].strip())
    return out


def merged(dirs):
    o, where = {}, {}
    for d in dirs:
        for k, v in parse(f"C:/so/{d}/SUMMARY.txt").items():
            o[k], where[k] = v, d
    return o, where


def sig(d, name):
    p = f"C:/so/{d}/{name}.log"
    if not os.path.exists(p):
        return None
    lines = [ln.rstrip() for ln in open(p, encoding="utf-8", errors="replace") if ln.strip()]
    s = lines[-1] if lines else "<empty log>"
    return re.sub(r"C:[\/]+so[\/]+[a-z]\b", "<TREE>", s).replace("\\", "/")[:900]


def count(tok):
    m = re.search(r"\b(\d+)\b", tok or "")
    return int(m.group(1)) if m else None


def blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main():
    tree, pkg = sys.argv[1], sys.argv[2]
    pk = os.path.join(tree, PK_REL)
    pdirs = os.environ.get("SO_PATCHED_DIRS", "logs_t1,logs_t2").split(",")          # later dirs override earlier ones (finish.sh adds the reruns on the hardened tree)
    pat, pwhere = merged(pdirs)
    base, bwhere = merged(["logs_b", "logs_b2"])
    bdirs = os.environ.get("SO_BASE_RERUN_DIRS", "logs_tb").split(",")
    rerun, rwhere = merged(bdirs)
    rows, n = [], {"PASS": 0, "PRE-EXISTING": 0, "REGRESSION": 0, "CHECK": 0, "NOT RUN": 0}
    for rel in ORDER:
        name = os.path.basename(rel)[:-3]
        p, b, rb = pat.get(name), base.get(name), rerun.get(name)
        cmd = f"cd <refe-plan dir> && python -X utf8 {rel}"
        if p is None:
            v, note = "NOT RUN", "no line in logs_t1/logs_t2"
        elif name in ("test_rl_safeobj", "test_safeobj_read", "test_safeobj_gate"):
            what = {"test_rl_safeobj": "NEW test; on the unpatched tree it cannot even be set up (`logs/red_before_patch.log`)",
                    "test_safeobj_read": "NEW test of the NEW reader eval/safeobj_read.py (bar S); neither exists on the unpatched tip",
                    "test_safeobj_gate": "NEW test (round 3, extended in round 4) of the gate's genuineness check and its re-read `rls1_checks.safeobj_gate`; on the round-2 tree the hand-written / edited / stale records it feeds the gate OPEN it (BUILD.md section 8)"}[name]
            v, note = ("PASS", what) if p[0] == 0 and p[2] else ("REGRESSION", "the new test fails")
        elif p[0] == 0 and p[2]:
            cb, cp = count(b[2]) if b else None, count(p[2])
            if b and b[0] != 0:
                v, note = "PASS", "baseline log FAILED (see its row)"
            elif cb is not None and cp is not None and cp < cb:
                v, note = "REGRESSION", f"pinned count dropped {cb} -> {cp}"
            else:
                v, note = "PASS", (f"count {cb} -> {cp}" if cb != cp else f"count unchanged ({cp})")
        else:
            sp, sb = sig(pwhere[name], name), (sig(rwhere[name], name) if rb else sig(bwhere.get(name, "logs_b"), name))
            bb = rb or b
            if bb is None:
                v, note = "CHECK", "fails patched; no baseline run exists"
            elif bb[0] == 0:
                v, note = "REGRESSION", "passes on the unpatched tree, fails patched"
            elif sp == sb:
                v, note = "PRE-EXISTING", "fails identically on the unpatched tree (same last line): " + (sp or "")[:260]
            else:
                v, note = "CHECK", f"fails on both with DIFFERENT last lines: patched `{(sp or '')[:200]}` / baseline `{(sb or '')[:200]}`"
        n[v] += 1
        fmt = lambda x: "-" if x is None else f"rc {x[0]}, {x[1]} s, `{x[2] or '(no token)'}`"
        pw = pwhere.get(name) or ""
        r3 = " [rerun on the ROUND-4 tree C:/so/t8]" if pw.startswith("logs_r4") else (" [rerun on the ROUND-3 tree C:/so/t6]" if pw.startswith("logs_r3") else "")
        ptree = {"logs_t3": " [rerun on the HARDENED tree]", "logs_t4": " [rerun on the HARDENED tree]", "logs_t5": " [rerun under bashshim.py on the HARDENED tree]",
                 "logs_t9": " [rerun on the FINAL tree C:/so/t4 (fixed queue test)]", "logs_t10": " [on the FINAL tree C:/so/t4]", "logs_t12": " [rerun on the FINAL tree C:/so/t4]", "logs_t14": " [rerun with the sitecustomize bash shim (every python process) on the FINAL tree C:/so/t4]"}.get(pwhere.get(name), r3)
        rtree = {"logs_tb2": " [under bashshim.py]", "logs_tb4": " [with the sitecustomize bash shim (every python process)]"}.get(rwhere.get(name), "")
        rows.append(f"| {len(rows) + 1} | `{rel}` | `{cmd}` | {fmt(b)}{'; rerun now: ' + fmt(rb) + rtree if rb else ''} | {fmt(p)}{ptree} | **{v}** | {note} |")
    out = ["# TEST TABLE (SAFEOBJ build, regression sweep on the clean patched tree)", "",
           f"Tree: the pristine tip `19785c8` of the refe-plan directory + `build/apply_all.sh` (patched files below). Run on the dev box, CPU (+ the local GPU where a test uses it), at most two test processes at a time (`build/sweep.sh`).",
           "Every test is run in its FULL form (the command column) from the refe-plan directory with the `driverl-eval` interpreter. Baseline = the same test on the UNPATCHED tip (tree `C:/so/b`, `logs/sweep/logs_b*`; failing tests rerun in this sweep, `logs/sweep/logs_tb`).",
           "Tests listed are `grep -l <edited file> refe/test_*.py eval/test_*.py` over scorer_finetune, launch_gate, gate_checks, rls1_checks, rls1_pod, rls1d_arms (25 files) + `eval/test_rl_val_eval.py` (defines the g5 bars; nothing in it changed) + round 2's new `eval/test_safeobj_read.py` + round 3's new `eval/test_safeobj_gate.py`. `rl_reward.py` is byte-unchanged. Round 2 changed one existing file (`refe/test_rls1_queue.py`) and added two (`eval/safeobj_read.py`, `eval/test_safeobj_read.py`); the full-form reruns of round 2 are the rows marked FINAL tree. **Round 3** (`BUILD.md` section 8) edits `eval/safeobj_read.py`, `refe/rls1_checks.py`, `refe/test_rl_safeobj.py`, `refe/test_rls1_queue.py` and adds `eval/test_safeobj_gate.py`; every test that names one of them (rls1_checks.py: 7 incl. the new one; test_rl_safeobj.py: launch_gate, r0c, s1b, rls1_chain; test_rls1_queue.py: rls1_chain) is rerun in FULL form on the round-3 tree C:/so/t6 and its row is marked ROUND-3 tree; the other rows ran on an earlier tree whose code for them is byte-identical to t6's (`logs/round3/t4_vs_t6.txt`). **Round 4** (`BUILD.md` sections 8.8-8.12; the gate re-reads the arm directory, the pod precheck lists the S-reader files) edits `eval/safeobj_read.py`, `refe/rls1_checks.py`, `refe/rls1_pod.sh`, `refe/test_rl_safeobj.py`, `refe/test_rls1_chain.py`, `refe/test_rls1_queue.py`, `eval/test_safeobj_read.py` and `eval/test_safeobj_gate.py`; every test that names one of them (11, the union of the greps in `build/sweep_round4.sh`; `refe/test_flock_rw.py` is new in that list because round 3 did not edit `rls1_pod.sh`) is rerun in FULL form on the round-4 tree C:/so/t8 and its row is marked ROUND-4 tree.", "",
           f"**Summary: {n['PASS']} PASS, {n['PRE-EXISTING']} PRE-EXISTING, {n['REGRESSION']} REGRESSION, {n['CHECK']} CHECK, {n['NOT RUN']} NOT RUN, of {len(ORDER)}.**", "",
           "| # | test | full command | baseline (unpatched tip) | patched | verdict | note |", "|---|---|---|---|---|---|---|"] + rows + ["", "## Patched-tree blobs the table is bound to (git blob sha1 over the LF bytes)", ""]
    for rel in TOUCHED:
        out.append(f"* `{rel}` `{blob(os.path.join(pk, rel))}`")
    first = parse("C:/so/logs_t1/SUMMARY.txt").get("test_launch_gate")
    firstb = parse("C:/so/logs_tb/SUMMARY.txt").get("test_launch_gate")
    out += ["", "## Notes", "",
            "* **`test_launch_gate` and the bare `bash`.** On this box a bare `bash` in a subprocess argument list resolves to `C:\\Windows\\System32\\bash.exe` (the WSL launcher: CreateProcess searches System32 before PATH), which drops the argument after `-c '<script>'`. "
            "`refe/test_launch_gate.py:339` (the only bare `bash` among these tests) then fails after 11 s, before its ~700 checks run, on the UNPATCHED tree as well. "
            f"First (unshimmed) runs: patched {first}, baseline {firstb} (logs `logs/sweep/logs_t1`, `logs/sweep/logs_tb`). The verdict row above is the rerun with a shim that only maps a bare `bash` to Git's POSIX bash and changes nothing else, on BOTH trees: round 1 used `build/bashshim.py` (outermost process only, so launch_gate's nested G-SUITE run of the same file still hit the WSL launcher and the test failed identically on both trees, `logs_t5` / `logs_tb2`); round 2 uses `build/sitecustomize_bash.py` on PYTHONPATH, which reaches EVERY python process, and the full file runs to its end on both trees (`logs_t14` / `logs_tb4`).",
            "* **Trees.** Rows marked *hardened* ran on `C:/so/t2` = the first sweep's tree `C:/so/t` + the hardened `refe/test_rl_safeobj.py` (see `HARDENING.md`); every other row without a mark ran on `C:/so/t`. "
            "Rows marked *FINAL tree* ran on `C:/so/t4`, rebuilt from the pristine tip by `build/apply_all.sh` AFTER the queue-test fix (`BUILD.md` section 7): `t4` differs from `t2` in exactly three files, "
            "`refe/test_rls1_queue.py` (the fix; no other test imports or names it) and the two NEW files `eval/safeobj_read.py` and `eval/test_safeobj_read.py` (`logs/t2_vs_t4.txt`), so every unmarked row's code is byte-identical to the final tree's.",
            "* **Round 3.** Rows marked *ROUND-3 tree* ran in FULL form on `C:/so/t6`, rebuilt from the pristine tip by `build/apply_all.sh` (now with `patch_genuine.py`) and byte-identical to a second fresh build `C:/so/t7` (`logs/round3/t6_vs_t7.txt`). "
            "`t6` differs from the round-2 tree `t4` in exactly `eval/safeobj_read.py`, `refe/rls1_checks.py`, `refe/test_rl_safeobj.py`, `refe/test_rls1_queue.py` (edited) and the NEW `eval/test_safeobj_gate.py` (`logs/round3/t4_vs_t6.txt`); the rows without that mark did not execute any of them (the grep over the tests that name them is in `build/sweep_round3.sh`). "
            "Counts that moved: `test_rl_safeobj` 58 -> 59 (one mutation, CH15), `test_rls1_queue` 226 -> 246 (the SAFEOBJ scenarios were re-cut, `BUILD.md` section 8.3). `test_launch_gate` stays at 719, `test_rls1_chain` at 512, `test_s1aux_chain` at 404, `test_rl_tlane_label` at 36, `test_rl_r0c` at 26, `test_rl_s1b` at 18, `test_safeobj_read` at 34. The run order and the 180 s offset of the second worker are in `build/sweep_round3.sh`.",
            "* **Round 4.** Rows marked *ROUND-4 tree* ran in FULL form on `C:/so/t8`, rebuilt from the pristine tip by `build/apply_all.sh` (now with `patch_reread.py`) and byte-identical to a second fresh build `C:/so/t9` (`logs/round4/t8_vs_t9.txt`). "
            "`t8` differs from the round-3 tree `t6` in exactly eight files (`logs/round4/t6_vs_t8.txt`): `eval/safeobj_read.py`, `refe/rls1_checks.py`, `refe/rls1_pod.sh`, `refe/test_rl_safeobj.py`, `refe/test_rls1_chain.py`, `refe/test_rls1_queue.py`, `eval/test_safeobj_read.py` (one expected-RED set loses a catcher the stronger gate made redundant) and `eval/test_safeobj_gate.py`. "
            "Counts that moved are in `BUILD.md` section 8.11. The run order and the 180 s offset of the second worker are in `build/sweep_round4.sh`.",
            "* **Known environmental failures** (CUDA checks of `test_aux_g12_bf16`, the `D:/Projects/TanitAD` path assertion of `eval/test_aux_navtest_forward`) fail identically on the unpatched tree; neither exercises code this build edited.",
            "", "Evidence class of every row: MEASURED (ours; the log is `logs/sweep/<dir>/<test>.log`)."]
    open(os.path.join(pkg, "TEST_TABLE.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print(n)
    return 1 if n["REGRESSION"] or n["CHECK"] or n["NOT RUN"] else 0


if __name__ == "__main__":
    sys.exit(main())
