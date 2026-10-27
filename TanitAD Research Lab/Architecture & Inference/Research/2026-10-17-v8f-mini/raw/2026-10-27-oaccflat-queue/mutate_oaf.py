#!/usr/bin/env python3
"""mutate_oaf.py -- the MUTATION PROOF of the DAG-OACC / FLAT-DAG queue stages, helpers and BOTH box-2 chains. Each mutant is ONE defect put into a COPY of the queue or of the box2_ship directory (the real files are never touched);
selftest_oaf.py is re-run on the sections that cover that file and the mutant is KILLED iff the self-test goes RED (a FAIL or a MISSED line). A mutant whose anchor text is not found exactly N times is an ANCHOR ERROR (never counted
as killed). A control run (unmutated, EVERY section A,S,B,C,D,E,F) must be green first. A survivor is a defect of the INSTRUMENT (fix the self-test, re-run); it is never dropped.

  Q  mutants of the staged queue                       -> sections A,B (Q7, Q13 also S: an EXISTING stage changed is what the byte-identity sweep is for)
  K  mutants of v8f_oaf_chain_checks.py                -> sections C,D,F
  G  mutants of gate_oaf_assert.py                     -> section E
  C  mutants of chain_v8f_dagoacc.sh / oaf_lib.sh      -> section D
  F  mutants of chain_v8f_flatdag.sh                   -> section F

    python mutate_oaf.py [--only Q1,Q2] [--workers 4] [--out mutation_output.txt] [--json mutation.json] [--min-free-mb 150]
The mutated copy is selected through OAF_QUEUE (queue) and OAF_SHIP (the directory every helper, the gate asserter, the lib and the chain are read from).
PARALLEL: each worker owns a private scratch set (OAF_BT = <root>/mw/w<k>/bt with a junction pytree -> the shared read-only exported .py tree, OAF_FW = C:\\Users\\Admin\\oqf\\m<k>), so concurrent self-tests never share a file.
DISK: C: is nearly full and other jobs are writing to it; before every mutant the free space is read, a run below --min-free-mb ABORTS the proof (written as INCOMPLETE, never as killed), and every result records the free space it ran with.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oaccflat_lib as L  # noqa: E402

WORK = L.BT.parent                                                    # the session scratch work directory (holds the exported .py tree `pytree`)
SCR = Path(os.environ["OAF_SCRATCH"]) if os.environ.get("OAF_SCRATCH") else None     # OAF_SCRATCH: put ALL per-worker scratch (mutant copies, fake boxes, git repos) on another volume (C: is nearly full)
ROOT = (SCR / "mw") if SCR else WORK / "mw"                                          # per-worker scratch: mw/w<k>/{bt,mut,pytree}
FWROOT = (SCR / "oqf") if SCR else Path(r"C:\Users\Admin\oqf")
PY = L.PY
QUEUE = L.QUEUE
SHIP = HERE / "box2_ship"
CK, GA, CH, LIB, FCH = "v8f_oaf_chain_checks.py", "gate_oaf_assert.py", "chain_v8f_dagoacc.sh", "oaf_lib.sh", "chain_v8f_flatdag.sh"

# (id, kind, description, old, new, count)   -- queue anchors are written with LF; the runner turns them into CRLF
MUTANTS = [
    # ---- the queue
    ("Q1", "Q", "WRONG INIT: flatdag starts from M2b_s<S> instead of FLAT_s<S>", 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2_flat\\ckpt_last.pt" --steps 3000', 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000', 1),
    ("Q2", "Q", "WRONG INIT: dagoacc starts from DAG_s<S> instead of M2b_s<S>", 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif "%~3"=="" (echo REFUSED: dagoacc needs P',
     'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2_dag\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif "%~3"=="" (echo REFUSED: dagoacc needs P', 1),
    ("Q3", "Q", "w_acc FROM M2b: waccdag evaluates M2b_s0's checkpoint instead of DAG_s0's", '--ckpt "%RUNSM%\\M2b_s0_dag\\ckpt_last.pt" --expect-sha256 d186fd0b', '--ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt" --expect-sha256 d186fd0b', 1),
    ("Q4", "Q", "--o-interface cut MISSING from the flatdag argv", ' --ego-perturb-p 0 --o-interface cut"', ' --ego-perturb-p 0"', 1),
    ("Q5", "Q", "FLAT BANK FROM M2b: bankflat's generator is M2b_s<S>", 'set "DAGGEN=%RUNSM%\\M2b_s%~2_flat"', 'set "DAGGEN=%RUNSM%\\M2b_s%~2"', 1),
    ("Q6", "Q", "-0 ARM CARRYING A BANK: flatdag0 gets --dagger-bank", '  set "DAGF=%DAGF0% --dagger-p 0"\n  exit /b 0\n)\nif "%~3"=="" (echo REFUSED: flatdag needs P', '  set "DAGF=%DAGF0% --dagger-p 0 --dagger-bank "%DAGBANK%""\n  exit /b 0\n)\nif "%~3"=="" (echo REFUSED: flatdag needs P', 1),
    ("Q7", "Q", "AN EXISTING STAGE LINE CHANGED: DAG's own fine-tune lr 1e-4 -> 2e-4 (the iteration-1 DAGF0)", '--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif /i "%DAGS%"=="dag0" (', '--steps 3000 --lr 2e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif /i "%DAGS%"=="dag0" (', 1),
    ("Q8", "Q", "--w-acc missing from the dagoacc argv", ' --dagger-seed %~2 --w-acc %WACCDAG%"', ' --dagger-seed %~2"', 1),
    ("Q9", "Q", "dagoacc reads OACC's w_acc (the oacc directory) instead of DAG-OACC's", 'set /p WACCDAG=<"%RUNSM%\\dagoacc\\w_acc.txt"', 'set /p WACCDAG=<"%RUNSM%\\oacc\\w_acc.txt"', 1),
    ("Q10", "Q", "flatdag uses the ITERATION-1 (M2b) bank directory", 'if /i "%DAGS%"=="flatdag" set "DAGBANK=%RUNSM%\\dagger_bank_flat\\s%~2"', 'if /i "%DAGS%"=="flatdag" set "DAGBANK=%RUNSM%\\dagger_bank\\s%~2"', 1),
    ("Q11", "Q", "flatdag accepts P = 0.3", 'if not "%~3"=="0.1" (echo REFUSED: flatdag\'s P is fixed', 'if not "%~3"=="0.1" if not "%~3"=="0.3" (echo REFUSED: flatdag\'s P is fixed', 1),
    ("Q12", "Q", "waccdag writes into OACC's directory (%RUNSM%\\oacc)", 'set "OD=%RUNSM%\\dagoacc"', 'set "OD=%RUNSM%\\oacc"', 1),
    ("Q13", "Q", "AN EXISTING STAGE NOBODY LISTED changed (evalpgt's tag): only the all-stage sweep can see it", 'set "TG=M2b-pgt_s%~2"', 'set "TG=M2b-pgtX_s%~2"', 1),
    ("Q14", "Q", "bankflat accepts a half-trained FLAT (the summary.json guard removed)", 'if not exist "%DAGGEN%\\summary.json" (echo REFUSED: %DAGGEN%\\summary.json is missing - FLAT_s%~2 has not finished training & exit /b 2)', 'rem mutant', 1),
    ("Q15", "Q", "the :evaldag hand-off for flatdag removed (flatdag falls into the DAG eval, no oracle read)", 'if /i "%DAGS%"=="flatdag" goto :evalflatdag', 'rem mutant', 1),
    ("Q16", "Q", "flatdag0 runs at --dagger-p 0.1 (the -0 control is not a control)", '  set "DAGF=%DAGF0% --dagger-p 0"\n  exit /b 0\n)\nif "%~3"=="" (echo REFUSED: flatdag needs P', '  set "DAGF=%DAGF0% --dagger-p 0.1"\n  exit /b 0\n)\nif "%~3"=="" (echo REFUSED: flatdag needs P', 1),
    ("Q17", "Q", "--w-acc moved ahead of the dagger flags (the lever is not the LAST token)", '--dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2 --w-acc %WACCDAG%"', '--w-acc %WACCDAG% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"', 1),
    ("Q18", "Q", "an inherited WACCDAG can stand in for the rule in a real run (the clearing removed)", 'set "WACCDAG="\nif defined V8F_DRYRUN goto :dagoaccwacc_dry', 'rem mutant\nif defined V8F_DRYRUN goto :dagoaccwacc_dry', 1),
    ("Q19", "Q", "the :dagarm hand-off for dagoacc removed", 'if /i "%DAGS%"=="dagoacc" goto :dagoaccarm', 'rem mutant', 1),
    ("Q20", "Q", "dagoacc accepts P = 0.3", 'if not "%~3"=="0.1" (echo REFUSED: dagoacc\'s P is fixed', 'if not "%~3"=="0.1" if not "%~3"=="0.3" (echo REFUSED: dagoacc\'s P is fixed', 1),
    ("Q21", "Q", "waccdag drops the cross-check against DAG_s0's banked fit windows", ' --banked-fit-windows "%OD%\\fit_windows_DAG_s0.npz"', '', 1),
    ("Q22", "Q", "waccdag drops the checkpoint sha256 pin", ' --expect-sha256 d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1', '', 1),
    ("Q23", "Q", "the flatdag oracle eval (tacorc) dropped from the evaluation", 'if defined V8F_DRYRUN (echo %E0% & echo %E2% & exit /b 0)', 'if defined V8F_DRYRUN (echo %E0% & exit /b 0)', 1),
    ("Q24", "Q", "flatdag's --dagger-seed dropped", ' --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"\nexit /b 0\n\n:evalflatdag', ' --dagger-bank-sha256 %DAGSHA%"\nexit /b 0\n\n:evalflatdag', 1),
    ("Q25", "Q", "waccdag runs on a half-trained DAG_s0 (the summary.json guard removed)", 'if not exist "%RUNSM%\\M2b_s0_dag\\summary.json" (echo REFUSED: %RUNSM%\\M2b_s0_dag\\summary.json is missing - DAG_s0 has not finished training & exit /b 2)', 'rem mutant', 1),
    ("Q26", "Q", "bankflat reads M2b's argv.json instead of FLAT's (the TRAIN list is rebuilt from the wrong run)", '--m2b-argv "%DAGGEN%\\argv.json" --cache-dir "%CACHEFULL%" --tracks "%TRACKS%" --join-train "%RUNSM%\\join_train.npz" --eval-grid "%DAGGRID%" --batch 32 --threads 2 --device cuda --expect-clips 4512 --expect-windows 306566"\nif defined V8F_DRYRUN (echo %BX% & exit /b 0)\nif exist "%DAGBANK%\\bank.json" if exist "%DAGBANK%\\PROVENANCE.json" (echo [skip] bank flat',
     '--m2b-argv "%RUNSM%\\M2b_s%~2\\argv.json" --cache-dir "%CACHEFULL%" --tracks "%TRACKS%" --join-train "%RUNSM%\\join_train.npz" --eval-grid "%DAGGRID%" --batch 32 --threads 2 --device cuda --expect-clips 4512 --expect-windows 306566"\nif defined V8F_DRYRUN (echo %BX% & exit /b 0)\nif exist "%DAGBANK%\\bank.json" if exist "%DAGBANK%\\PROVENANCE.json" (echo [skip] bank flat', 1),
    # ---- the python helpers
    ("K1", "K", "C-O3 tolerance 1e-4 -> 1e-2", 'WACC_REL_TOL = 1e-4', 'WACC_REL_TOL = 1e-2', 1),
    ("K2", "K", "C-O3 does not require the model route's cross-check to pass", '    if not isinstance(bc, dict) or bc.get("pass") is not True or bc.get("same_windows") is not True:', '    if False:', 1),
    ("K3", "K", "C-O3 does not compare the checkpoint the rule was evaluated with", '    if expect_ckpt_sha and src.get("ckpt_sha256") != expect_ckpt_sha.lower():', '    if False:', 1),
    ("K4", "K", "C-O3 does not check that w_acc.txt is the record's number", "        if tv is not None and tv != f\"{b.get('w_acc')!r}\\n\":", '        if False:', 1),
    ("K5", "K", "F1: init-match accepts every trainer", 'return (key in v), (', 'return True, (', 1),
    ("K6", "K", "DISK FLOOR REMOVED: disk-check always passes", 'return (0 if after >= floor else 2), f"free=', 'return 0, f"free=', 1),
    ("K7", "K", "PREDECESSOR-STOP BRANCH INVERTED: a predecessor STOP of its own (box released) HALTs instead of going", 'return "GO", "the predecessor STOPPED on its own (', 'return "HALT", "the predecessor STOPPED on its own (', 1),
    ("K8", "K", "the predecessor's GUARD stop no longer halts", '        if guard:\n            return "HALT", "the predecessor was stopped by the GUARD (', '        if False:\n            return "HALT", "the predecessor was stopped by the GUARD (', 1),
    ("K9", "K", "a held BOX2_BUSY after the predecessor's DONE no longer halts", 'return ("HALT", "the predecessor ended DONE but BOX2_BUSY.txt is still there") if busy_exists else ("GO", done[-1][:200])', 'return ("GO", done[-1][:200])', 1),
    ("K10", "K", "C-O3 does not check the window count", '    if b.get("n_windows") != o.get("n_windows") or b.get("n_windows") != 9582:', '    if False:', 1),
    ("K11", "K", "a held BOX2_BUSY after a predecessor STOP reads GO (the box would be taken while held)", 'return "BUSY", "the predecessor stopped but its BOX2_BUSY.txt is not released (yet)"', 'return "GO", "the predecessor stopped but its BOX2_BUSY.txt is not released (yet)"', 1),
    # ---- the gate asserter
    ("G1", "G", "dagoacc's init tail is FLAT's", '"dagoacc": {"prereg": "PREREG_V8F_DAG_OACC", "p": True, "init_tail": "/M2b_s{seed}/ckpt_last.pt"', '"dagoacc": {"prereg": "PREREG_V8F_DAG_OACC", "p": True, "init_tail": "/M2b_s{seed}_flat/ckpt_last.pt"', 1),
    ("G2", "G", "flatdag's bank directory tail is the iteration-1 one", '"bank_tail": "/dagger_bank_flat/s{seed}/bank.json", "o_interface": "cut", "init_name": "FLAT"},\n    "flatdag0"', '"bank_tail": "/dagger_bank/s{seed}/bank.json", "o_interface": "cut", "init_name": "FLAT"},\n    "flatdag0"', 1),
    ("G3", "G", "K-B2 (the bank's generator is the init) not checked", '        if not (kb2.get("ckpt_sha256") == kb2.get("init_ckpt_sha256") == a.banked_init_sha):', '        if False:', 1),
    ("G4", "G", "the pin need not equal the planned training argv", '    if pin != tl:', '    if False:', 1),
    ("G5", "G", "the init checkpoint's sha256 is not compared", '    if ic.get("sha256") != a.banked_init_sha:', '    if False:', 1),
    ("G6", "G", "the acc term may be exactly zero", ' or live["terms"]["acc"] == 0.0:', ' or False:', 1),
    ("G7", "G", "ONE-LEVER: the dagoacc pin need not be DAG's pin + --w-acc W", '                if mine_[-2:] != ["--w-acc", a.w_acc] or mine_[:-2] != drop(dpin):', '                if False:', 1),
    ("G8", "G", "ONE-LEVER: box 2's DAG pin need not be DAG's banked argv.json", '                if dpin != ref:', '                if False:', 1),
    ("G9", "G", "the cut is not required to be built (o_interface may be full)", "    if lv.get(\"o_interface\") != \"cut\":", '    if False:', 1),
    ("G10", "G", "the swap need not reach the loss", '        if not (pr.get("loss_diff", 0.0) != 0.0 and pr.get("grad_diff_all", 0.0) > 0.0):', '        if False:', 1),
    ("G11", "G", "head_o's gradient from the acc term need not be non-zero", '    if not (finite(ap.get("head_o_grad_diff")) and ap["head_o_grad_diff"] > 0.0):', '    if False:', 1),
    ("G12", "G", "FLAT's every other flag is not compared", '            if flags(drop(pin, FINE_TUNE | {"--o-interface"})) != flags(drop(ref, FINE_TUNE | {"--o-interface"}))', '            if False and flags(drop(pin, FINE_TUNE | {"--o-interface"})) != flags(drop(ref, FINE_TUNE | {"--o-interface"}))', 1),
    # ---- the chain script and the lib
    ("C1", "C", "DISK FLOOR REMOVED from pull()", '  disk_floor "$d" "$sz" "$rel"\n', '  true\n', 1),
    ("C2", "C", "the chain's floor value 1 GiB -> 1 MiB", 'FLOOR_BYTES=1073741824 ', 'FLOOR_BYTES=1048576 ', 1),
    ("C3", "C", "pull() renames a short transfer (the size check removed)", '  [ "$(stat -c %s "$d/$n.part")" = "$sz" ] || { rm -f "$d/$n.part"; stop "pull $R/$rel: $(stat -c %s "$d/$n.part" 2>/dev/null) bytes on disk != box 2\'s $sz"; }\n', '', 1),
    ("C4", "C", "RUNS in the wrong order (seed 1 first)", 'RUNS=("dagoacc 0" "dagoacc 1")', 'RUNS=("dagoacc 1" "dagoacc 0")', 1),
    ("C5", "C", "PVAL 0.3", 'PVAL=0.1 ', 'PVAL=0.3 ', 1),
    ("C6", "C", "one hex digit of DAG_s0's checkpoint sha256 changed in the chain", 'CKPT_SHA=(d186fd0b', 'CKPT_SHA=(e186fd0b', 1),
    ("C7", "C", "PREDECESSOR-STOP BRANCH INVERTED in the shell: a HALT verdict returns GO", '      HALT) stop "predecessor verdict: $v" ;;', '      HALT) return 0 ;;', 1),
    ("C8", "C", "the wait loop treats BUSY as GO", '      BUSY) k=$((k+1)); [ $k -gt 3 ] && stop "predecessor verdict: $v (still held after $k checks)"; sleep $grace ;;', '      BUSY) return 0 ;;', 1),
    ("C9", "C", "the real entry does not wait for a predecessor at all (wait_predecessor removed)", '\nwait_predecessor\n', '\ntrue\n', 1),
    ("C10", "C", "C-O3's STOP is swallowed: a failing co3 falls through to `true` and the chain goes on after a mismatch (the first form of this mutant added `true;` INSIDE the STOP block, which changes nothing: an equivalent mutant)", 'co3 "$ART/wacc/w_acc.json" "$ART/wacc/w_acc.txt" || { echo "ZZDAGOACC-CO3-STOP', 'co3 "$ART/wacc/w_acc.json" "$ART/wacc/w_acc.txt" || true || { echo "ZZDAGOACC-CO3-STOP', 1),
    ("K12", "K", "K-B2 (the bank's checkpoint is the banked generator's) always passes in bank-judge", 'common.get("ckpt_sha256") == banked_sha, f"{str(common.get(\'ckpt_sha256\'))[:12]}', 'True, f"{str(common.get(\'ckpt_sha256\'))[:12]}', 1),
    ("K13", "K", "COVERAGE_MIN 0.5 -> 0.1 (a bank with 30 % eligible windows passes)", 'COVERAGE_MIN = 0.5', 'COVERAGE_MIN = 0.1', 1),
    ("K14", "K", "the BLOCKED outcome is never reached (a COVERAGE-blocked bank judged PASS)", '    elif blocked:\n        outcome, rc = "BLOCKED", 3', '    elif False:\n        outcome, rc = "BLOCKED", 3', 1),
    # ---- the FLAT-DAG chain
    ("F1", "F", "RUNS in the wrong order (the two FLATDAG arms before the two FLATDAG-0 arms)", 'RUNS=("flatdag 0" "flatdag0 0" "flatdag 1" "flatdag0 1")', 'RUNS=("flatdag 0" "flatdag 1" "flatdag0 0" "flatdag0 1")', 1),
    ("F2", "F", "PVAL 0.3", 'PVAL=0.1 ', 'PVAL=0.3 ', 1),
    ("F3", "F", "THE F1 CHECK REMOVED from the real run (the line deleted)", 'lever_f1 git "$T" || { echo "ZZFLATDAG-F1-STOP the tip\'s INIT_CKPT_MATCH does not name o_interface: a FLATDAG fine-tune could start from a checkpoint of the other interface (the F1 fix is not landed): nothing was touched on box 2"; stop "F1 not landed"; }\n', '', 1),
    ("F4", "F", "THE F1 CHECK NEUTERED in the real run (text kept, a failing check falls through to `true`)", 'lever_f1 git "$T" || { echo "ZZFLATDAG-F1-STOP the tip\'s INIT_CKPT_MATCH', 'lever_f1 git "$T" || true || { echo "ZZFLATDAG-F1-STOP the tip\'s INIT_CKPT_MATCH', 1),
    ("F5", "F", "F1 REMOVED from --check-tree (the check-only mode passes a tip trainer)", '  lever_f1 tree "$2" || { echo "ZZFLATDAG-F1-STOP', '  true || { echo "ZZFLATDAG-F1-STOP', 1),
    ("F6", "F", "one hex digit of FLAT_s0's checkpoint sha256 changed in the chain", 'CKPT_SHA=(8496934a', 'CKPT_SHA=(9496934a', 1),
    ("F7", "F", "BANK DIRECTORY -> dagger_bank (the iteration-1 M2b bank directory, in bank_one)", 'GATE_LOG="$R/dagger_bank_flat/s$1/bank_stage.log"', 'GATE_LOG="$R/dagger_bank/s$1/bank_stage.log"', 1),
    ("F8", "F", "GATES AFTER RUNS (the four gates run after the first training)", 'for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done\n# ---- 5. the four runs, in the PREREG\'s order; nothing is conditional on a read\nfor run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done', '# ---- 5. the four runs, in the PREREG\'s order; nothing is conditional on a read\nfor run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done\nfor run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done', 1),
    ("F9", "F", "PUBLISH BEFORE THE JUDGE in the real run (the banks are copied to the dev box before the controls and COVERAGE are judged)", 'for s in 0 1; do bank_one $s; done\njudge_banks', 'for s in 0 1; do bank_one $s; done\nfor s in 0 1; do publish_bank_one $s; done\njudge_banks', 1),
    ("F10", "F", "the chain's floor value 1 GiB -> 1 MiB", 'FLOOR_BYTES=1073741824 ', 'FLOOR_BYTES=1048576 ', 1),
    ("F11", "F", "the real entry does not wait for a predecessor at all (wait_predecessor removed)", '\nwait_predecessor\n', '\ntrue\n', 1),
    ("F12", "F", "the REACHED control (the swap reached the loss) is not enforced for FLATDAG: the 'swap-reached' FAIL exits 0", 'print("FLATDAG swap-reached:", "PASS" if ok else "FAIL", f"| flatdag s{seed} | dagger_n {n} / eligible {e} = {n / e if ok else \'n/a\'} (must lie in {p} +- 0.02)")\nsys.exit(0 if ok else 1)', 'print("FLATDAG swap-reached:", "PASS" if ok else "FAIL", f"| flatdag s{seed} | dagger_n {n} / eligible {e} = {n / e if ok else \'n/a\'} (must lie in {p} +- 0.02)")\nsys.exit(0)', 1),
    ("F13", "F", "the judge's REFUSED branch (a bank control failed) is reported as PASS (judge_banks swallows the rc)", '    *) echo "ZZFLATDAG-REFUSED a bank control failed:', '    *) return 0; echo "ZZFLATDAG-REFUSED a bank control failed:', 1),
    # ---- the post-run assertions embedded in runone (python heredocs): the record check and the REACHED control
    ("F14", "F", "FLATDAG run record: the built dagger_p need not equal P", 'if lv.get("dagger_p") != p: bad.append', 'if False: bad.append', 1),
    ("F15", "F", "FLATDAG run record: the cut need not be built (levers.o_interface)", '(sc.get("levers") or {}).get("o_interface")) != "cut": bad.append', '(sc.get("levers") or {}).get("o_interface")) != "cut" and False: bad.append', 1),
    ("F16", "F", "FLATDAG run record: FLATDAG-0 may have built a bank", 'if arm == "flatdag0" and lv.get("dagger_bank_sha256") is not None: bad.append', 'if False: bad.append', 1),
    ("F17", "F", "FLATDAG REACHED control: the swap-rate window +- 0.02 -> +- 0.5", 'p - 0.02 - 1e-12 <= n / e <= p + 0.02 + 1e-12', 'p - 0.5 - 1e-12 <= n / e <= p + 0.5 + 1e-12', 1),
    ("F18", "F", "FLATDAG run record: the pinned argv need not equal the planned one", 'if pinned != planned:', 'if False:', 1),
    ("F19", "F", "FLATDAG REACHED control: FLATDAG-0 may swap windows", 'ok = isinstance(n, int) and n == 0', 'ok = True', 1),
    ("C11", "C", "DAG-OACC run record: the built w_acc need not equal the argv's (C-O2)", 'if lv.get("w_acc") != wacc: bad.append', 'if False: bad.append', 1),
    ("C12", "C", "DAG-OACC REACHED control: the swap-rate window +- 0.02 -> +- 0.5", 'p - 0.02 - 1e-12 <= n / e <= p + 0.02 + 1e-12', 'p - 0.5 - 1e-12 <= n / e <= p + 0.5 + 1e-12', 1),
    ("C13", "C", "DAG-OACC run record: the init checkpoint need not be M2b's", 'if (lv.get("init_ckpt") or {}).get("sha256") != banked: bad.append', 'if False: bad.append', 1),
    ("C14", "C", "DAG-OACC run record: the built bank need not be DAG's", 'if lv.get("dagger_bank_sha256") != bsha: bad.append', 'if False: bad.append', 1),
]


LOCK = threading.Lock()


def free_mb() -> int:
    return shutil.disk_usage(WORK).free // (1 << 20)


def worker_dirs(k: int):
    wd = ROOT / f"w{k}"
    (wd / "bt").mkdir(parents=True, exist_ok=True)
    (wd / "mut").mkdir(parents=True, exist_ok=True)
    fw = FWROOT / f"m{k}"
    fw.mkdir(parents=True, exist_ok=True)
    if not (wd / "pytree").exists():
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(wd / "pytree"), str(WORK / "pytree")], capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"mklink /J failed: {r.stdout} {r.stderr}")
    return wd, fw


def run_selftest(sections: str, env: dict, wd: Path, fw: Path, tag: str, timeout=900):
    out = wd / "mut" / f"run_out_{tag}.txt"
    cmd = [PY, str(HERE / "selftest_oaf.py"), "--sections", sections, "--out", str(out)]
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "OAF_BT": str(wd / "bt"), "OAF_FW": str(fw), **env}
    for k in ("OAF_QUEUE", "OAF_SHIP"):
        if k not in env:
            e.pop(k, None)
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout, errors="replace", stdin=subprocess.DEVNULL)
    txt = (p.stdout or "") + (p.stderr or "")
    bad = [x for x in txt.splitlines() if x.startswith("FAIL") or x.startswith("MISSED") or "Traceback" in x]
    return p.returncode, bad, txt.splitlines()[-1:] if txt else [], time.time() - t, txt


def one_mutant(m, wd: Path, fw: Path, min_free: int):
    mid, kind, desc, old, new, count = m
    free0 = free_mb()
    if free0 < min_free:
        return {"id": mid, "kind": kind, "desc": desc, "status": "ABORTED_LOW_DISK", "free_mb": free0}
    d = wd / "mut" / mid
    d.mkdir(parents=True, exist_ok=True)
    env = {}
    if kind == "Q":
        raw = QUEUE.read_bytes()
        o, n = old.replace("\n", "\r\n").encode("utf-8"), new.replace("\n", "\r\n").encode("utf-8")
        c = raw.count(o)
        if c != count:
            return {"id": mid, "kind": kind, "desc": desc, "status": "ANCHOR_ERROR", "found": c}
        (d / "queue.cmd").write_bytes(raw.replace(o, n))
        env["OAF_QUEUE"] = str(d / "queue.cmd")
        sections = "A,S,B" if mid in ("Q7", "Q13") else "A,B"
    else:
        sd = d / "ship"
        shutil.copytree(SHIP, sd)
        target = {"K": CK, "G": GA, "C": CH, "F": FCH}[kind]
        if kind == "C" and mid in ("C1", "C3", "C7", "C8"):
            target = LIB
        raw = (sd / target).read_bytes().decode("utf-8")
        c = raw.count(old)
        if c != count:
            return {"id": mid, "kind": kind, "desc": desc, "status": "ANCHOR_ERROR", "found": c, "file": target}
        (sd / target).write_bytes(raw.replace(old, new).encode("utf-8"))
        env["OAF_SHIP"] = str(sd)
        sections = {"K": "C,D,F", "G": "E", "C": "D", "F": "F"}[kind]
    env["OAF_SKIP_F11"] = "1"                                  # F11 / F12 copy the real 644 MB banks: the control executes them once, the mutant runs do not (6 concurrent copies saturate the USB drive)
    rc, bad, last, dt, txt = run_selftest(sections, env, wd, fw, mid)
    free1 = free_mb()
    real = [b for b in bad if not b.startswith("Traceback")]
    killed = rc != 0 and bool(bad)
    res = {"id": mid, "kind": kind, "desc": desc, "status": ("KILLED" if real else "KILLED_BY_CRASH_ONLY") if killed else "SURVIVED", "sections": sections, "seconds": round(dt, 1), "rc": rc,
           "red_by": [b[:260] for b in bad[:3]], "n_red": len(bad), "free_mb_before": free0, "free_mb_after": free1, "result_line": (last[0][:200] if last else "")}
    shutil.rmtree(d, ignore_errors=True)                       # the mutant's own copy (queue / ship dir) only
    return res


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    min_free = int(sys.argv[sys.argv.index("--min-free-mb") + 1]) if "--min-free-mb" in sys.argv else 150
    outp = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "mutation_output.txt"
    jp = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else HERE / "mutation.json"
    log = []

    def say(s):
        with LOCK:
            log.append(s)
            print(s, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    dirs = [worker_dirs(k) for k in range(nw + 1)]                  # slot 0 = the control, slots 1..nw = the mutant workers
    say("MUTATION PROOF -- PREREG_V8F_DAG_OACC / PREREG_V8F_FLATDAG queue stages, helpers and BOTH chains")
    say(f"workers {nw}; C: free {free_mb()} MB at start; min-free {min_free} MB")
    rc, bad, last, dt, txt = run_selftest("A,S,B,C,D,E,F", {}, dirs[0][0], dirs[0][1], "control", timeout=1800)
    try:
        (HERE / "selftest_output.txt").write_text((dirs[0][0] / "mut" / "run_out_control.txt").read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as e:
        say(f"(could not save selftest_output.txt: {e})")
    say(f"CONTROL (unmutated, all sections A,S,B,C,D,E,F): rc {rc}, {'GREEN' if rc == 0 and not bad else 'RED ' + str(bad[:3])}  [{last}]  ({dt:.0f} s)")
    if rc != 0 or bad:
        say("the control is not green: the mutation proof is void")
        outp.write_text("\n".join(log) + "\n", encoding="utf-8")
        return 2
    todo = [m for m in MUTANTS if not only or m[0] in only]
    slots = list(range(1, nw + 1))
    free_slots = __import__("queue").Queue()
    for k in slots:
        free_slots.put(k)

    def job(m):
        k = free_slots.get()
        try:
            r = one_mutant(m, dirs[k][0], dirs[k][1], min_free)
        except Exception as e:                                      # a crashed runner is NOT a kill
            r = {"id": m[0], "kind": m[1], "desc": m[2], "status": "RUNNER_ERROR", "error": repr(e)[:300]}
        finally:
            free_slots.put(k)
        st = r["status"]
        if st.startswith("KILLED") or st == "SURVIVED":
            say(f"{r['id']}: {'KILLED ' if st.startswith('KILLED') else 'SURVIVED'} [{r['sections']}] ({r['seconds']:.0f} s, {r['n_red']} red, C: {r['free_mb_after']} MB) -- {r['desc']}" + (f"\n      by: {r['red_by'][0][:200]}" if r["red_by"] else ""))
        else:
            say(f"{r['id']}: {st} -- {r['desc']} {({k: v for k, v in r.items() if k in ('found', 'error', 'free_mb', 'file')})}")
        return r
    with ThreadPoolExecutor(max_workers=nw) as ex:
        results = list(ex.map(job, todo))                           # ex.map keeps MUTANTS order
    n = len(results)
    k = sum(r["status"].startswith("KILLED") for r in results)
    ae = sum(r["status"] == "ANCHOR_ERROR" for r in results)
    other = sum(r["status"] in ("ABORTED_LOW_DISK", "RUNNER_ERROR") for r in results)
    surv = sum(r["status"] == "SURVIVED" for r in results)
    crash = sum(r["status"] == "KILLED_BY_CRASH_ONLY" for r in results)
    say(f"\nSUMMARY: {k}/{n} mutants killed ({crash} of them by a crash only), {surv} survived, {ae} anchor errors, {other} not run (low disk / runner error); control green")
    for r in results:
        if r["status"] != "KILLED":
            say(f"  NOT A CLEAN KILL: {r['id']} {r['status']} -- {r['desc']}")
    outp.write_text("\n".join(log) + "\n", encoding="utf-8")
    jp.write_text(json.dumps({"control_green": True, "n_mutants": n, "killed": k, "killed_by_crash_only": crash, "survived": surv, "anchor_errors": ae, "not_run": other, "results": results}, indent=1), encoding="utf-8")
    return 0 if k == n and crash == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
