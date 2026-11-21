"""freeze_n1dm.py COMMIT UTC REGRESSION_LINE LG_LINE -- MM 2026-11-21: write PREREG_V8F_N1DM.md (frozen) + its .sha256 from PREREG_V8F_N1DM_DRAFT.md.
Changes only the header (status, code of record) and section 7 (the build record as MEASURED by the MM); no lever, argv, cell, bar or outcome moves."""
import hashlib
import sys

C, T, REG, LG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
assert len(C) == 40 and all(ch in "0123456789abcdef" for ch in C), C
D = "F:/TanitAD-scratch/n1dc/next"
s = open(f"{D}/PREREG_V8F_N1DM_DRAFT.md", "rb").read().decode("utf-8").replace("\r\n", "\n")


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep("# PREREG_V8F_N1DM (DRAFT, not frozen; MM freezes it before any arm trains):",
    f"# PREREG_V8F_N1DM (FROZEN by the MM at {T}, before any N1DM arm trains):")
rep('Status: DRAFT written by the build agent from the MM brief (`BRIEF_BUILD_N1DM.md`). PI go, first-hand: "Build both" covers building and pre-registering this ARCHITECTURE change; RUNNING needs a second go with the cost (section 6). Nothing here has been run on a pod or GPU.',
    f'Status: FROZEN by the MM at {T} from the build agent\'s draft (`PREREG_V8F_N1DM_DRAFT.md`, written from `BRIEF_BUILD_N1DM.md`), after the MM checks in section 7. PI go, first-hand: "Build both" covers building and pre-registering this ARCHITECTURE change; RUNNING needs a second go with the cost (section 6). Nothing here has been run on a pod or GPU.')
rep("Code of record for the parent: **1f217bf0** (branch `agent/v8f-n1dh-20261120`). `file:line` below are in the build tree `build/n1dm_b1/tree` (= 1f217bf0 + `n1dm.patch`).",
    f"Code of record for the parent: **1f217bf0** (branch `agent/v8f-n1dh-20261120`). **Code of record for N1DM: {C[:8]}** ({C}, branch `agent/v8f-n1dm-20261121`, parent 1f217bf0, exactly the 14 files of `n1dm.patch`). `file:line` below are in the build tree `build/n1dm_b1/tree` (= 1f217bf0 + `n1dm.patch` = the tree of {C[:8]}).")
old7 = "See `build/n1dm_b1/HANDOFF_N1DM.md` (what ran, what did not) and `mutation_table_n1dm.json`. Nothing in this section changes a lever, argv, cell, bar or outcome; any code fact the build finds that does is an amendment before any arm trains."
new7 = (f"See `build/n1dm_b1/HANDOFF_N1DM.md` (what ran, what did not) and `mutation_table_n1dm.json`. Nothing in this section changes a lever, argv, cell, bar or outcome; any code fact the build finds that does is an amendment before any arm trains.\n\n"
        f"**MM checks before the freeze (dev box, CPU, MEASURED, {T}):**\n"
        "* `test_v8f_n1dm.py` 39 of 39 pass; `test_strat1_n1dm_check.py` passes on its re-run (its one red was a mutation clause the test could not reach, fixed before the re-run).\n"
        "* Mutation table: **21 of 21 mutants RED**, each by a FAILED test (none by a collection error), one anchored edit each (`mutation_table_n1dm.json`).\n"
        f"* Regression, whole files one at a time: {REG}\n"
        f"* Launch gate: {LG}\n"
        "* Post-training lines: the build's first dry run (`dry_post_lines.sh`) ran NOTHING and wrote rc 0: its `sed 's/\\//g'` is an unterminated s command, the pattern came out empty and `grep -F \"\"` matched line 1 (`#!/bin/bash`). The repaired `dry_post_lines2.sh` greps the literal command prefix, refuses a line that lacks it, fails on any non-zero rc and requires the outputs: on the REAL finished tiny arm (4 steps) the three post lines of `pod/n1dm_arm.sh` read post check PASS, zero-grad gate PASS (258 of 258 registered trained; controls route_head.weight and route_br.0/1/2.weight each trained), K-CLIP json written; `ZZ DRYPOST2 PASS ZZ` (`logs/dry_post_lines2.log`).\n"
        "* Read package: the build's dry check also reported rc 0 on a FAIL (2). Repairs, each with a `.bk1` copy: (i) `test_n1dm_paths.py` counted SOURCE echo lines against 10, but the five lines sit inside `for S in s0 s1`, which writes 10 (rendered: 10 dumps); (ii) the K-LIM runner and the eval-141 dry check named `stack/scripts/n1dm_klim.py`, a rename leftover: the tool of record is `n1dh_klim.py`, unchanged (section 4); (iii) the leftover-tag scan's `/tree` exclusion compared against forward slashes while Windows `os.walk` yields backslashes. The read tree `read_n1dm/tree` = `git archive` of this commit's tree + the hb wrapper 74fd8445: 3,443 of 3,443 blobs equal. **`ZZ N1DM-READ-DRYCHECK PASS ZZ`** (28 steps, incl. eval-141 `DRY CHECK PASS`).\n"
        "* **K-OWN-SPLIT (section 2.3) is a read-side diagnostic that is NOT built yet.** It is built before the read (it needs only the trained checkpoints and a probe batch); if it is not built by then it is reported NOT-COMPUTABLE with the reason. It has no bar and decides no outcome.\n"
        "* Not proved before the launch, said plainly: G-SUITE and the dry mint on the pod (both are steps of the chain, before any arm trains), the full 775-file suite on the dev box (scoped to 108 files, as N1DC / N1DH), and any number from a trained N1DM arm.")
rep(old7, new7)
out = s if s.endswith("\n") else s + "\n"
open(f"{D}/PREREG_V8F_N1DM.md", "w", encoding="utf-8", newline="\n").write(out)
h = hashlib.sha256(out.encode()).hexdigest()
open(f"{D}/PREREG_V8F_N1DM.sha256", "w", encoding="utf-8", newline="\n").write(f"{h} *PREREG_V8F_N1DM.md\n")
print("frozen", h)
