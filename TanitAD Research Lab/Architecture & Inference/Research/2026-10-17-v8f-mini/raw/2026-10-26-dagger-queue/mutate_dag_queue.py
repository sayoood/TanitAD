#!/usr/bin/env python3
"""Mutation proof for the new queue stages: put ONE defect into a COPY of the staged queue, run the selftest's sections A (rendering) and B (the fake box, non-dry) against that copy, and require the selftest to go RED.
A mutant that leaves the selftest green is a rule whose test is too weak. The staged queue itself is never edited. Writes mutation_output.txt.

Three families, all run against a COPY, never the staged files:
  MUTANTS          defects in the staged box-2 queue (.cmd)                                  -> the selftest's sections A (rendering) and B (the fake box)
  BANK_CHK_MUTANTS defects in box2_ship/v8f_dag_chain_checks.py's bank-publication checks     -> section G (bank_published / bank_verify / bank_plan, in-process)
  CHAIN_MUTANTS    defects in box2_ship/chain_v8f_dag.sh (publish order, markers, seal, sums) -> sections H (the offline --publish-only flow) and D (source order, dry run)
Each mutant must turn at least one check RED; the control run (no mutation) must read ALL PASS first.

    CUDA_VISIBLE_DEVICES=-1 python mutate_dag_queue.py [--list]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagq_lib as L  # noqa: E402

# (name, old text, new text): `old` must occur exactly once in the staged queue (CRLF removed for matching)
MUTANTS = [
    ("traindag: the byte comparison of the pin against the line it would build now is not made",
     'fc /b "%DAGPIN%" "%DAGPIN%.chk" >nul\nif errorlevel 1 (echo REFUSED: the training argv derived now differs', 'fc /b "%DAGPIN%" "%DAGPIN%.chk" >nul\nif errorlevel 99 (echo REFUSED: the training argv derived now differs'),
    ("gatedag: a pin that differs is silently replaced",
     '  fc /b "%DAGPIN%" "%DAGPIN%.new" >nul\n  if errorlevel 1 (echo REFUSED: %DAGPIN% pins another', '  fc /b "%DAGPIN%" "%DAGPIN%.new" >nul\n  if errorlevel 99 (echo REFUSED: %DAGPIN% pins another'),
    ("dagarm: a real run reads DAGSHA_DRY (the dry-run override leaks into a real run)",
     'set /p DAGSHA=<"%RUNSM%\\dagpin\\bank_sha_s%~2.txt"\ngoto :dagarm_have', 'set /p DAGSHA=<"%RUNSM%\\dagpin\\bank_sha_s%~2.txt"\nif defined DAGSHA_DRY set "DAGSHA=%DAGSHA_DRY%"\ngoto :dagarm_have'),
    ("dagarm: DAG-0 also names --dagger-seed",
     'set "DAGF=%DAGF0% --dagger-p 0"\n  exit /b 0', 'set "DAGF=%DAGF0% --dagger-p 0 --dagger-seed %~2"\n  exit /b 0'),
    ("dagarm: P is not validated",
     'if not "%~3"=="0.1" if not "%~3"=="0.3" (echo REFUSED: P must be 0.1 or 0.3', 'if "%~3"=="x" if not "%~3"=="0.3" (echo REFUSED: P must be 0.1 or 0.3'),
    ("dagarm: dag0 may be given a P",
     'if not "%~3"=="" (echo REFUSED: dag0 takes no P', 'if "%~3"=="zz" (echo REFUSED: dag0 takes no P'),
    ("dagname: any seed is accepted",
     'if not "%~2"=="0" if not "%~2"=="1" (echo REFUSED: SEED must be 0 or 1, got "%~2" & exit /b 2)\nset "DAGSEED=%~2"', 'set "DAGSEED=%~2"'),
    ("dagarm: --ego-perturb-p is not named (the flag's default disables the lever; the pair would no longer equal ORC-0 + --dagger-p 0)",
     'ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"', 'ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2"'),
    ("dagarm: DAG-0 also names --ego-perturb-p 0.3 (the arm would combine the two levers)",
     'set "DAGF=%DAGF0% --dagger-p 0"\n  exit /b 0', 'set "DAGF=%DAGF0% --dagger-p 0 --ego-perturb-p 0.3"\n  exit /b 0'),
    ("dagarm: the fine-tune is 3,001 steps",
     '--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif /i', '--steps 3001 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"\nif /i'),
    ("dagarm: DAG --dagger-seed is always 0 (not the run's seed)",
     '--dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"', '--dagger-bank-sha256 %DAGSHA% --dagger-seed 0"'),
    ("dagarm: the DAG bank is seed 0's for both seeds",
     'set "DAGBANK=%RUNSM%\\dagger_bank\\s%~2"', 'set "DAGBANK=%RUNSM%\\dagger_bank\\s0"'),
    ("gatedag: the gate line forgets --selfcheck-only",
     '%A% %DAGF% --selfcheck-only"', '%A% %DAGF%"'),
    ("gatedag: the gate's --out is the run directory",
     'set "GX="%PY%" "%TRAIN%" --arm M2b --seed %DAGSEED% --grid pool4x10 --out "%DAGGATE%"', 'set "GX="%PY%" "%TRAIN%" --arm M2b --seed %DAGSEED% --grid pool4x10 --out "%DAGOUT%"'),
    ("evaldag: the standard eval is run at --ages fresh",
     'set "E0="%PY%" "%EVAL%" run --ckpt "%DAGOUT%\\ckpt_last.pt" --tag %DAGTAG% %EC%', 'set "E0="%PY%" "%EVAL%" run --ckpt "%DAGOUT%\\ckpt_last.pt" --tag %DAGTAG% --ages fresh %EC%'),
    ("bankdag: two shards instead of four",
     'bank --n-shards 4 --out-dir', 'bank --n-shards 2 --out-dir'),
    ("bankdag: the bank is generated from seed 0's checkpoint whatever the seed (K-B2 would fail on box 2)",
     '-- --ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --m2b-argv', '-- --ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt" --m2b-argv'),
    ("dagarm: the DAG init checkpoint is seed 0's for both seeds",
     'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt"', 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s0\\ckpt_last.pt"'),
    ("bankdag: the eval grid is the repo path (ampersand) instead of the copy",
     '--eval-grid "%DAGGRID%"', '--eval-grid "%REPO%\\TanitAD Research Lab\\Architecture & Inference\\Research\\2026-10-17-v8f-mini\\raw\\eval_grid_windows.json"'),
    ("bankdag: the copy of the eval grid is made before DAGGRID is set (the order bug the fake box found)",
     'call :dagname dag %~2 || exit /b 2\ncall :prepdag || exit /b 2', 'call :prepdag || exit /b 2\ncall :dagname dag %~2 || exit /b 2'),
]


# ---- mutants of the chain's bank judge (box2_ship/v8f_dag_chain_checks.py): section C of the selftest must go RED on each (Master Mind O6: K-B4 / K-B5 are GATED)
CHK = HERE / "box2_ship" / "v8f_dag_chain_checks.py"
CHK_MUTANTS = [
    ("judge: K-B4 is not gated (the control is always True)",
     'ctl("K-B4 matcher synthetic cases (PROVENANCE.json controls.K-B4: PASS True and all six cases True)", why4 is None, why4 or "")', 'ctl("K-B4 matcher synthetic cases (PROVENANCE.json controls.K-B4: PASS True and all six cases True)", True, "")'),
    ("judge: K-B5 is not gated (the control is always True)",
     'ctl("K-B5 re-anchoring (PROVENANCE.json controls.K-B5: PASS True, both errors <= 1e-4 m)", why5 is None, why5 or "")', 'ctl("K-B5 re-anchoring (PROVENANCE.json controls.K-B5: PASS True, both errors <= 1e-4 m)", True, "")'),
    ("judge: a MISSING K-B4 / K-B5 record is tolerated",
     'return f"PROVENANCE.json carries no controls.{kid} record (the generator does not write it): the chain STOPS"', 'return None'),
    ("judge: K-B5 tolerance 1e-3 instead of 1e-4",
     'KB5_TOL_M = 1e-4', 'KB5_TOL_M = 1e-3'),
    ("judge: K-B5 is `<` instead of `<=` (the reader's rule is inclusive)",
     'c[k] <= KB5_TOL_M', 'c[k] < KB5_TOL_M'),
    ("judge: PASS is read as truthy, not the boolean True",
     'if c.get("PASS") is not True:', 'if not c.get("PASS"):'),
    ("judge: K-B4 checks five of the six cases (the `loop` case is skipped)",
     'miss = [k for k in KB4_CASES if cases.get(k) is not True]', 'miss = [k for k in KB4_CASES[:5] if cases.get(k) is not True]'),
    ("judge: K-B4's cases are never examined (only PASS)",
     'return f"every case must be recorded True; not True / absent: {miss}" if miss else None', 'return None'),
    ("judge: K-B5 checks only max_pose_now_err_m",
     'for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):', 'for k in ("max_pose_now_err_m",):'),
]

# ---- mutants of the BANK PUBLICATION checks (same file, bank-files / bank-verify): section G of the selftest must go RED on each
BANK_CHK_MUTANTS = [
    ("publish check: a shard npz is not hashed against bank.json's file_sha256",
     '        if got != by_k[k].get("file_sha256"):', '        if False:'),
    ("publish check: a shard provenance's file_sha256 is not compared with the npz's hash",
     '        if not isinstance(sp, dict) or sp.get("file_sha256") != got:', '        if False:'),
    ("publish check: bank.json is not hashed against PROVENANCE.bank_sha256",
     '    if prov.get("bank_sha256") != msha:', '    if False:'),
    ("publish check: SHA256SUMS.txt may list more or fewer names than the 11 (no exact set)",
     '            elif sorted(listed) != sorted(allnames):', '            elif False:'),
    ("publish check: the hashes in SHA256SUMS.txt are not compared with the files",
     '                    if h != listed[nm]:', '                    if False:'),
    ("publish check: an EMPTY published file is accepted",
     '            elif os.path.getsize(p) == 0:', '            elif False:'),
    ("publish check: an ABSENT published file is accepted",
     '            if not os.path.isfile(p):\n                r["bad"].append(f"{kind}: {nm} is absent")', '            if False:\n                r["bad"].append(f"{kind}: {nm} is absent")'),
    ("publish check: a bank of another shard count than the queue's four is accepted",
     '    if expect_shards is not None and n != expect_shards:', '    if False:'),
    ("publish names: the shard npz are not among the published names (only the small files would be)",
     '"shard_npz": [f"shard_{k:03d}of{n:03d}.npz" for k in range(n)], "stage_log": ["bank_stage.log"]}', '"shard_npz": [], "stage_log": ["bank_stage.log"]}'),
    ("publish names: bank.json is not among the published names",
     '"provenance": ["PROVENANCE.json"], "manifest": ["bank.json"],', '"provenance": ["PROVENANCE.json"], "manifest": [],'),
    ("publish check: a SHA256SUMS.txt without its final newline is accepted",
     '    if not text.endswith("\\n"):\n        return {}, "SHA256SUMS.txt does not end', '    if False:\n        return {}, "SHA256SUMS.txt does not end'),
    ("publish check: a name listed twice in SHA256SUMS.txt is accepted",
     '        if m.group(2).strip() in listed:', '        if False:'),
]

# ---- mutants of the CHAIN's publish code (box2_ship/chain_v8f_dag.sh): each is [(old, new), ...] applied to a COPY of the chain (its helper scripts copied beside it) that the selftest then runs (DAGQ_CHAIN)
CHAIN = HERE / "box2_ship" / "chain_v8f_dag.sh"
CHAIN_MUTANTS = [
    ("chain: the shard npz are never pulled by publish_bank_one", "H",
     [('    pull "dagger_bank/s$s/$f" "$d"\n    ls=$(sha256sum "$d/$f" | cut -c1-64); rs=$(rsha', '    true\n    ls=$(sha256sum "$d/$f" | cut -c1-64); rs=$(rsha')]),
    ("chain: box 2's own sha256 of a shard npz is not compared with the pulled file's", "H",
     [('    [ -n "$rs" ] && [ "$rs" = "$ls" ] || stop "bank s$s: box 2\'s $f sha256', '    true || stop "bank s$s: box 2\'s $f sha256')]),
    ("chain: the files are not verified BEFORE the seal is written", "H",
     [('v=$($PY "$(cygpath -m $CHECKS)" bank-verify "$(cygpath -m "$d")") || stop "bank s$s: the pulled files do not verify', 'v=$(true) || stop "bank s$s: the pulled files do not verify')]),
    ("chain: the seal is not re-verified after it is written", "H",
     [('v=$($PY "$(cygpath -m $CHECKS)" bank-verify "$(cygpath -m "$d")" --require-sums) || {', 'v=$(true) || {')]),
    ("chain: a seal that fails its re-verification is left on disk", "H",
     [('|| { rm -f "$d/SHA256SUMS.txt"; stop "bank s$s: SHA256SUMS.txt does not verify (removed)', '|| { stop "bank s$s: SHA256SUMS.txt does not verify (removed)')]),
    ("chain: an earlier attempt's SHA256SUMS.txt is not removed when the small files are fetched", "H",
     [('  rm -f "$d/SHA256SUMS.txt"\n  for f in bank.json PROVENANCE.json bank_stage.log; do', '  for f in bank.json PROVENANCE.json bank_stage.log; do')]),
    ("chain: the small-file fetch also pulls the shard npz (moved before the judge)", "H",
     [('for k in 0 1 2 3; do pull "dagger_bank/s$s/shard_00${k}of004.provenance.json" "$d"; done', 'for k in 0 1 2 3; do pull "dagger_bank/s$s/shard_00${k}of004.provenance.json" "$d"; pull "dagger_bank/s$s/shard_00${k}of004.npz" "$d"; done')]),
    ("chain: SHA256SUMS.txt is written over the small files only (the npz left out)", "H",
     [('bank-files "$(cygpath -m "$d")" --kind all)', 'bank-files "$(cygpath -m "$d")" --kind small)')]),
    ("chain: the real flow never calls publish_bank_one", "D",
     [('\nfor s in 0 1; do publish_bank_one $s; done\n# ---- 4.', '\n# ---- 4.')]),
    ("chain: the real flow publishes BEFORE the judge", "DH",
     [('\nfor s in 0 1; do publish_bank_one $s; done\n# ---- 4.', '\n# ---- 4.'),
      ('declare -a BSHA\nfor s in 0 1; do bank_one $s; done\n', 'declare -a BSHA\nfor s in 0 1; do bank_one $s; done\nfor s in 0 1; do publish_bank_one $s; done\n')]),
]


def main() -> int:
    if "--list" in sys.argv:
        for i, (n, _, _) in enumerate(MUTANTS):
            print(i, n)
        return 0
    base = L.QUEUE if not os.environ.get("DAGQ_QUEUE") else Path(os.environ["DAGQ_QUEUE"])
    text = base.read_bytes().decode("utf-8").replace("\r\n", "\n")
    out_dir = HERE / "bt" / "mutants"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, killed, survived, bad = [], 0, [], 0
    t0 = time.time()
    for i, (name, old, new) in enumerate(MUTANTS):
        if text.count(old) != 1:
            log.append(f"{i:2d} ANCHOR-ERROR (occurs {text.count(old)}x): {name}")
            bad += 1
            continue
        q = out_dir / f"queue_m{i}.cmd"
        q.write_bytes(text.replace(old, new).replace("\n", "\r\n").encode("utf-8"))
        outf = out_dir / f"out_m{i}.txt"
        env = {**os.environ, "DAGQ_QUEUE": str(q), "DAGQ_SECTIONS": "AB", "DAGQ_OUT": str(outf), "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run([sys.executable, str(HERE / "selftest_dag_queue.py")], env=env, capture_output=True, text=True, cwd=str(HERE), timeout=1800)
        o = outf.read_text(encoding="utf-8") if outf.exists() else p.stdout
        red = [l for l in o.splitlines() if l.startswith(("FAIL", "MISSED"))]
        if red:
            killed += 1
            log.append(f"{i:2d} KILLED   ({len(red)} red) {name}   <- {red[0][:150]}")
        else:
            survived.append(name)
            log.append(f"{i:2d} SURVIVED {name}")
        print(log[-1], flush=True)
    ctext = CHK.read_text(encoding="utf-8")
    for j, (name, old_c, new_c) in enumerate(CHK_MUTANTS):
        i = len(MUTANTS) + j
        if ctext.count(old_c) != 1:
            log.append(f"{i:2d} ANCHOR-ERROR (occurs {ctext.count(old_c)}x): {name}")
            bad += 1
            continue
        cd = out_dir / f"chk_m{j}"
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "v8f_dag_chain_checks.py").write_text(ctext.replace(old_c, new_c), encoding="utf-8", newline="")
        outf = out_dir / f"out_chk{j}.txt"
        env = {**os.environ, "DAGQ_CHECKS_DIR": str(cd), "DAGQ_SECTIONS": "C", "DAGQ_OUT": str(outf), "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, str(HERE / "selftest_dag_queue.py")], env=env, capture_output=True, text=True, cwd=str(HERE), timeout=1800)
        o = outf.read_text(encoding="utf-8") if outf.exists() else ""
        red = [l for l in o.splitlines() if l.startswith(("FAIL", "MISSED"))]
        if red:
            killed += 1
            log.append(f"{i:2d} KILLED   ({len(red)} red) {name}   <- {red[0][:150]}")
        else:
            survived.append(name)
            log.append(f"{i:2d} SURVIVED {name}")
        print(log[-1], flush=True)
    n_before = len(MUTANTS) + len(CHK_MUTANTS)
    for j, (name, old_c, new_c) in enumerate(BANK_CHK_MUTANTS):                      # the bank-publication checks: section G (in-process, so the mutated checks are the ones that run)
        i = n_before + j
        if ctext.count(old_c) != 1:
            log.append(f"{i:2d} ANCHOR-ERROR (occurs {ctext.count(old_c)}x): {name}")
            bad += 1
            continue
        cd = out_dir / f"bchk_m{j}"
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "v8f_dag_chain_checks.py").write_text(ctext.replace(old_c, new_c), encoding="utf-8", newline="")
        outf = out_dir / f"out_bchk{j}.txt"
        env = {**os.environ, "DAGQ_CHECKS_DIR": str(cd), "DAGQ_SECTIONS": "G", "DAGQ_OUT": str(outf), "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, str(HERE / "selftest_dag_queue.py")], env=env, capture_output=True, text=True, cwd=str(HERE), timeout=1800)
        o = outf.read_text(encoding="utf-8") if outf.exists() else ""
        red = [l for l in o.splitlines() if l.startswith(("FAIL", "MISSED"))]
        if red:
            killed += 1
            log.append(f"{i:2d} KILLED   ({len(red)} red) {name}   <- {red[0][:150]}")
        else:
            survived.append(name)
            log.append(f"{i:2d} SURVIVED {name}")
        print(log[-1], flush=True)
    chain_text = CHAIN.read_text(encoding="utf-8")
    n_before += len(BANK_CHK_MUTANTS)
    for j, (name, sections, pairs) in enumerate(CHAIN_MUTANTS):                      # the chain's publish code: the selftest runs the MUTATED chain (sections H = the offline publish flow, D = the source order)
        i = n_before + j
        mutated, anchor_bad = chain_text, [o_ for o_, _ in pairs if chain_text.count(o_) != 1]
        if anchor_bad:
            log.append(f"{i:2d} ANCHOR-ERROR (occurs {[chain_text.count(o_) for o_ in anchor_bad]}x): {name}")
            bad += 1
            continue
        for o_, n_ in pairs:
            mutated = mutated.replace(o_, n_)
        cd = out_dir / f"chain_m{j}"
        cd.mkdir(parents=True, exist_ok=True)
        for helper in ("v8f_dag_chain_checks.py", "gate_dag_assert.py"):
            shutil.copy(HERE / "box2_ship" / helper, cd / helper)
        (cd / "chain_v8f_dag.sh").write_text(mutated, encoding="utf-8", newline="")
        outf = out_dir / f"out_chain{j}.txt"
        env = {**os.environ, "DAGQ_CHAIN": str(cd / "chain_v8f_dag.sh"), "DAGQ_SECTIONS": sections, "DAGQ_OUT": str(outf), "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, str(HERE / "selftest_dag_queue.py")], env=env, capture_output=True, text=True, cwd=str(HERE), timeout=1800)
        o = outf.read_text(encoding="utf-8") if outf.exists() else ""
        red = [l for l in o.splitlines() if l.startswith(("FAIL", "MISSED"))]
        if red:
            killed += 1
            log.append(f"{i:2d} KILLED   ({len(red)} red) {name}   <- {red[0][:150]}")
        else:
            survived.append(name)
            log.append(f"{i:2d} SURVIVED {name}")
        print(log[-1], flush=True)
    n_all = len(MUTANTS) + len(CHK_MUTANTS) + len(BANK_CHK_MUTANTS) + len(CHAIN_MUTANTS)
    # control: the unmutated queue, judge, publication checks and chain are green under the same runner
    env = {**os.environ, "DAGQ_QUEUE": str(base), "DAGQ_SECTIONS": "ABCDGH", "DAGQ_OUT": str(out_dir / "out_control.txt"), "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, str(HERE / "selftest_dag_queue.py")], env=env, capture_output=True, text=True, cwd=str(HERE), timeout=1800)
    oc = (out_dir / "out_control.txt").read_text(encoding="utf-8")
    ctl_green = "RESULT: ALL PASS" in oc
    log.append(f"control (unmutated staged queue, judge, publication checks and chain, sections A, B, C, D, G and H): {'GREEN' if ctl_green else 'RED'}")
    summary = f"MUTATION: {killed}/{n_all} killed ({len(MUTANTS)} queue + {len(CHK_MUTANTS)} judge + {len(BANK_CHK_MUTANTS)} publication checks + {len(CHAIN_MUTANTS)} chain), {len(survived)} survived, {bad} anchor errors; control {'green' if ctl_green else 'RED'}; {time.time() - t0:.0f} s"
    log.append(summary)
    (HERE / "mutation_output.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(summary)
    return 0 if (not survived and not bad and ctl_green) else 1


if __name__ == "__main__":
    sys.exit(main())
