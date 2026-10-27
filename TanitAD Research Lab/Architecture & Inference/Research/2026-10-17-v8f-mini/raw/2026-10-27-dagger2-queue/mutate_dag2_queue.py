#!/usr/bin/env python3
"""mutate_dag2_queue.py -- the MUTATION PROOF of the DAGGER2 queue stages and chain. Each mutant is ONE defect put into a copy of the queue, a python helper or the chain script (the real files are never touched);
the self-test sections that cover that file are re-run against the copy and the mutant is KILLED iff the self-test goes RED. A mutant whose anchor text is not found exactly N times is an ANCHOR ERROR (never
counted as killed). A control run (unmutated) must be green first.

  Q  mutants of the staged queue          -> selftest sections A,S,B (S without the full sweep, except Q17 which only the sweep can see)
  K  mutants of v8f_dag2_chain_checks.py  -> section C
  G  mutants of gate_dag2_assert.py       -> section E
  C  mutants of chain_v8f_dag2.sh         -> sections D,F

    python mutate_dag2_queue.py [--only Q1,Q2] [--out mutation_output.txt] [--json mutation.json]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dag2q_lib as L  # noqa: E402

W = L.BT.parent / "mut"
PY = L.PY
QUEUE = L.QUEUE
SHIP = HERE / "box2_ship"

CK = "v8f_dag2_chain_checks.py"
GA = "gate_dag2_assert.py"
CH = "chain_v8f_dag2.sh"

# (id, kind, description, old, new, count)
MUTANTS = [
    ("Q1", "Q", "bank generator = M2b (not DAG): --ckpt / --m2b-argv read M2b_s<S>", 'set "DAGGEN=%RUNSM%\\M2b_s%~2_dag"', 'set "DAGGEN=%RUNSM%\\M2b_s%~2"', 1),
    ("Q2", "Q", "init = M2b: dag2 / dag20 --init-ckpt is M2b_s<S>\\ckpt_last.pt", 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2_dag\\ckpt_last.pt" --steps 3000', 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000', 1),
    ("Q3", "Q", "P != 0.1 accepted for dag2 (0.3 passes the check)", 'if not "%~3"=="0.1" (echo REFUSED: dag2\'s P', 'if not "%~3"=="0.1" if not "%~3"=="0.3" (echo REFUSED: dag2\'s P', 1),
    ("Q4", "Q", "dag20 carries a bank (--dagger-bank appended to its argv)", '  if not "%~3"=="" (echo REFUSED: dag20 takes no P, got "%~3" - a lever declared and inert & exit /b 2)\n  set "DAGF=%DAGF0% --dagger-p 0"',
     '  if not "%~3"=="" (echo REFUSED: dag20 takes no P, got "%~3" - a lever declared and inert & exit /b 2)\n  set "DAGF=%DAGF0% --dagger-p 0 --dagger-bank "%DAGBANK%""', 1),
    ("Q5", "Q", "dag2 uses the ITERATION-1 bank directory (dagger_bank\\s<S>)", 'if /i "%DAGS%"=="dag2" set "DAGBANK=%RUNSM%\\dagger_bank_dag2\\s%~2"', 'if /i "%DAGS%"=="dag2" set "DAGBANK=%RUNSM%\\dagger_bank\\s%~2"', 1),
    ("Q6a", "Q", "seeds swapped in the bank generator: every seed's bank is generated from DAG_s0", 'set "DAGGEN=%RUNSM%\\M2b_s%~2_dag"', 'set "DAGGEN=%RUNSM%\\M2b_s0_dag"', 1),
    ("Q6b", "Q", "seeds swapped in --dagger-seed: always 0", ':dag2arm_have\nif not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)\nset "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed %~2"',
     ':dag2arm_have\nif not defined DAGSHA (echo REFUSED: no bank sha256 & exit /b 2)\nset "DAGF=%DAGF0% --dagger-p %DAGP% --dagger-bank "%DAGBANK%" --dagger-bank-sha256 %DAGSHA% --dagger-seed 0"', 1),
    ("Q6c", "Q", "seeds swapped in --init-ckpt: every seed starts from DAG_s0", 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2_dag\\ckpt_last.pt" --steps 3000', 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s0_dag\\ckpt_last.pt" --steps 3000', 1),
    ("Q7", "Q", "an EXISTING stage line changed: DAG's own fine-tune lr 1e-4 -> 2e-4 (the iteration-1 DAGF0)", 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000 --lr 1e-4', 'set "DAGF0=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000 --lr 2e-4', 1),
    ("Q8", "Q", "dag2 is not an accepted arm any more (the :dagname insertion removed)", 'if /i "%~1"=="dag2" set "DAGS=dag2"', 'rem mutant', 1),
    ("Q9", "Q", "dag20 silently accepts a P (the refusal removed)", '  if not "%~3"=="" (echo REFUSED: dag20 takes no P, got "%~3" - a lever declared and inert & exit /b 2)', '  rem mutant: P ignored', 1),
    ("Q10", "Q", "bankdag2 accepts a half-trained DAG (the summary.json guard removed)", 'if not exist "%DAGGEN%\\summary.json" (echo REFUSED: %DAGGEN%\\summary.json is missing - DAG_s%~2 has not finished training & exit /b 2)', 'rem mutant', 1),
    ("Q11", "Q", "--ego-perturb-p 0 dropped from the dag2 / dag20 argv", '_dag\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0"', '_dag\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2"', 1),
    ("Q14", "Q", "bankdag2 writes into the ITERATION-1 bank directory", 'bank --n-shards 4 --out-dir "%DAGBANK%" -- --ckpt "%DAGGEN%\\ckpt_last.pt"', 'bank --n-shards 4 --out-dir "%RUNSM%\\dagger_bank\\s%~2" -- --ckpt "%DAGGEN%\\ckpt_last.pt"', 1),
    ("Q16", "Q", "bankdag2 rebuilds the TRAIN list from M2b's argv.json instead of DAG's", '--m2b-argv "%DAGGEN%\\argv.json"', '--m2b-argv "%RUNSM%\\M2b_s%~2\\argv.json"', 1),
    ("Q17", "Q", "an EXISTING stage nobody listed changed (evalpgt's tag): only the all-stage sweep can see it", 'set "OUT=%RUNSM%\\M2b-pgt_s%~2"\nset "TG=M2b-pgt_s%~2"\nset "E0="', 'set "OUT=%RUNSM%\\M2b-pgt_s%~2"\nset "TG=M2b-pgtX_s%~2"\nset "E0="', 1),
    ("Q18", "Q", "the :dagarm hand-off for dag20 removed (dag20 falls into the iteration-1 path)", 'if /i "%DAGS%"=="dag20" goto :dag2arm', 'rem mutant', 1),
    # ---- the python helpers
    ("K1", "K", "dag-predicate does not pin the file's sha256", '    if sha != str(expect_sha).lower():', '    if False:', 1),
    ("K2", "K", "dag-predicate does not require ROBUST", '    if name != "ROBUST":', '    if False:', 1),
    ("K3", "K", "judge: K-B2 compares nothing (the bank's generator is not checked against DAG's checkpoint)", 'common.get("ckpt_sha256") == banked_sha,', 'True,', 1),
    ("K4", "K", "judge: K-B4 not gated", 'why4 is None, why4 or "")', 'True, why4 or "")', 1),
    ("K5", "K", "judge: COVERAGE exactly 0.5 blocks (< becomes <=)", '"blocked": frac < COVERAGE_MIN}', '"blocked": frac <= COVERAGE_MIN}', 1),
    ("K6", "K", "dag-predicate accepts another PREREG's claim row", '    if d.get("claim_row") != "H-V8F-DAGGER-1":', '    if False:', 1),
    ("K7", "K", "dag-predicate tolerates a failed control", 'if not isinstance(ctl, dict) or not ctl or not all(v is True for v in ctl.values()):', 'if not isinstance(ctl, dict):', 1),
    ("K8", "K", "judge: K-B5 tolerance 1e-3", 'KB5_TOL_M = 1e-4', 'KB5_TOL_M = 1e-3', 1),
    ("G1", "G", "asserter: the init tail is M2b's, not DAG's", '"init_tail": "/M2b_s{seed}_dag/ckpt_last.pt"', '"init_tail": "/M2b_s{seed}/ckpt_last.pt"', 1),
    ("G2", "G", "asserter: the bank directory tail is the iteration-1 one", '"bank_dir_tail": "/dagger_bank_dag2/s{seed}/bank.json"', '"bank_dir_tail": "/dagger_bank/s{seed}/bank.json"', 1),
    ("G3", "G", "asserter: K-B2 (bank's checkpoint == init == the expected one) not checked", '        if not (kb2.get("ckpt_sha256") == kb2.get("init_ckpt_sha256") == a.banked_init_sha):', '        if False:', 1),
    ("G4", "G", "asserter: the pin need not equal the planned training argv", '    if pin != tl:', '    if False:', 1),
    ("G5", "G", "asserter: the init checkpoint's sha256 is not compared with the expected one", '    if ic.get("sha256") != a.banked_init_sha:', '    if False:', 1),
    # ---- the chain script
    ("C1", "C", "the real entry ignores the DAG-read predicate", 'dag_pred || { if [ $DAG_RC = 4 ]; then', 'dag_pred; true || { if [ $DAG_RC = 4 ]; then', 1),
    ("C2", "C", "the BLOCKED / REFUSED stop after the judge is disabled", 'if [ $jr != 0 ]; then', 'if false; then', 1),
    ("C3", "C", "the bank is fetched from the ITERATION-1 directory on box 2", 'for f in bank.json PROVENANCE.json bank_stage.log; do pull "dagger_bank_dag2/s$s/$f" "$d"; done', 'for f in bank.json PROVENANCE.json bank_stage.log; do pull "dagger_bank/s$s/$f" "$d"; done', 1),
    ("C4", "C", "one hex digit of DAG_s0's checkpoint sha256 changed in the chain", 'CKPT_SHA=(d186fd0b', 'CKPT_SHA=(e186fd0b', 1),
    ("C5", "C", "box-2 assertion skips DAG's argv.json", '    for f in ckpt_last.pt argv.json summary.json; do\n      case $f in', '    for f in ckpt_last.pt summary.json; do\n      case $f in', 1),
    ("C6", "C", "the export list drops raw/eval141_kin.npz (the eval's default --eval-kin)", '  "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval141_kin.npz"\n', '', 1),
    ("C7", "C", "the export size cap raised to 3 GB", 'EXPORT_MAX_BYTES=314572800 ', 'EXPORT_MAX_BYTES=3145728000 ', 1),
    ("C8", "C", "RUNS in the wrong order (seed 1 first)", 'RUNS=("dag2 0" "dag20 0" "dag2 1" "dag20 1")', 'RUNS=("dag2 1" "dag20 1" "dag2 0" "dag20 0")', 1),
    ("C9", "C", "PVAL 0.3", "PVAL=0.1      # DAG2's P", "PVAL=0.3      # DAG2's P", 1),
    ("C10", "C", "plan() hands P to dag20 as well", '[ "$2" = dag2 ] && parg=$PVAL', 'parg=$PVAL', 1),
    ("C11", "C", "the lever check no longer looks for the :bankdag2 stage on the tip", '  "$PK/code/v8f_mini_box2_queue.cmd|^:bankdag2"\n', '', 1),
    ("C12", "C", "the pinned DAG-read sha256 changed by one digit", 'DAG_READ_SHA=3393fdd5', 'DAG_READ_SHA=3393fdd6', 1),
    ("C13", "C", "seeds swapped in the judge's expected generator", '--banked-sha0 ${CKPT_SHA[0]} --banked-sha1 ${CKPT_SHA[1]}', '--banked-sha0 ${CKPT_SHA[1]} --banked-sha1 ${CKPT_SHA[0]}', 1),
    ("C14", "C", "the DAG-read re-check before the gates removed", 'o2=$(sha256sum "$DAG_READ" | cut -c1-64); [ "$o2" = "$DAGR_SHA" ] || stop', 'o2=x; [ "$o2" = "x" ] || stop', 1),
]


def run_selftest(sections: str, env: dict, sweep: bool, timeout=900):
    out = W / "run_out.txt"
    cmd = [PY, str(HERE / "selftest_dag2_queue.py"), "--sections", sections, "--out", str(out)] + ([] if sweep else ["--no-sweep"])
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", **env}
    for k in ("DAG2Q_QUEUE", "DAG2Q_CHAIN", "DAG2Q_CHECKS_DIR"):
        if k not in env:
            e.pop(k, None)
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout, errors="replace")
    txt = (p.stdout or "") + (p.stderr or "")
    bad = [x for x in txt.splitlines() if x.startswith("FAIL") or x.startswith("MISSED") or "Traceback" in x]
    return p.returncode, bad, txt.splitlines()[-1:] if txt else [], time.time() - t


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    outp = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "mutation_output.txt"
    jp = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else HERE / "mutation.json"
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    log = []

    def say(s):
        log.append(s)
        print(s, flush=True)
    say("MUTATION PROOF -- PREREG_V8F_DAGGER2 queue stages and chain")
    rc, bad, last, dt = run_selftest("A,S,B,C,D,E,F", {}, True, timeout=1500)
    try:
        shutil.copy(W / "run_out.txt", HERE / "selftest_output.txt")
    except OSError as e:
        say(f"(could not save selftest_output.txt: {e})")
    say(f"CONTROL (unmutated, all sections, full sweep): rc {rc}, {'GREEN' if rc == 0 and not bad else 'RED ' + str(bad[:3])}  [{last}]  ({dt:.0f} s)")
    if rc != 0 or bad:
        say("the control is not green: the mutation proof is void")
        outp.write_text("\n".join(log) + "\n", encoding="utf-8")
        return 2
    results = []
    for mid, kind, desc, old, new, count in MUTANTS:
        if only and mid not in only:
            continue
        d = W / mid
        d.mkdir(parents=True)
        env = {}
        if kind == "Q":
            raw = QUEUE.read_bytes()
            o, n = old.replace("\n", "\r\n").encode("utf-8"), new.replace("\n", "\r\n").encode("utf-8")
            c = raw.count(o)
            if c != count:
                say(f"{mid}: ANCHOR ERROR ({c} x, expected {count}) -- {desc}")
                results.append({"id": mid, "kind": kind, "desc": desc, "status": "ANCHOR_ERROR", "found": c})
                continue
            (d / "queue.cmd").write_bytes(raw.replace(o, n))
            env["DAG2Q_QUEUE"] = str(d / "queue.cmd")
            sections, sweep = ("A,S,B", mid == "Q17")
        else:
            for f in (CK, GA, CH, "chain_v8f_dag2.sh.in"):
                if (SHIP / f).exists():
                    shutil.copy(SHIP / f, d / f)
            target = {"K": CK, "G": GA, "C": CH}[kind]
            raw = (d / target).read_text(encoding="utf-8")
            c = raw.count(old)
            if c != count:
                say(f"{mid}: ANCHOR ERROR ({c} x, expected {count}) -- {desc}")
                results.append({"id": mid, "kind": kind, "desc": desc, "status": "ANCHOR_ERROR", "found": c})
                continue
            (d / target).write_bytes(raw.replace(old, new).encode("utf-8"))
            env["DAG2Q_CHECKS_DIR"] = str(d)
            if kind == "C":
                env["DAG2Q_CHAIN"] = str(d / CH)
            sections, sweep = {"K": ("C", False), "G": ("E", False), "C": ("D,F", False)}[kind]
        rc, bad, last, dt = run_selftest(sections, env, sweep)
        killed = rc != 0 and bool(bad)
        results.append({"id": mid, "kind": kind, "desc": desc, "status": "KILLED" if killed else "SURVIVED", "sections": sections, "seconds": round(dt, 1), "rc": rc, "red_by": [b[:220] for b in bad[:3]], "n_red": len(bad)})
        say(f"{mid}: {'KILLED ' if killed else 'SURVIVED'} [{sections}] ({dt:.0f} s, {len(bad)} red) -- {desc}" + (f"\n      by: {bad[0][:200]}" if bad else ""))
        shutil.rmtree(d, ignore_errors=True)
    n = len(results)
    k = sum(r["status"] == "KILLED" for r in results)
    ae = sum(r["status"] == "ANCHOR_ERROR" for r in results)
    say(f"\nSUMMARY: {k}/{n} mutants killed, {n - k - ae} survived, {ae} anchor errors; control green")
    outp.write_text("\n".join(log) + "\n", encoding="utf-8")
    jp.write_text(json.dumps({"control_green": True, "n_mutants": n, "killed": k, "survived": n - k - ae, "anchor_errors": ae, "results": results}, indent=1), encoding="utf-8")
    return 0 if k == n else 1


if __name__ == "__main__":
    sys.exit(main())
