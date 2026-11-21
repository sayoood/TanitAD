#!/usr/bin/env python3
"""mutate_oaf_r2.py -- the MUTATION PROOF of the r2 files (gate_oaf_assert_r2.py, oaf_r2_lib.sh, chain_v8f_dagoacc_r2.sh, the launcher, the waker). It extends mutate_oaf.py's method (which is not edited): each mutant is ONE defect put into a COPY of the
r2 ship directory (the real files are never touched); selftest_oaf_r2.py is re-run on the sections that cover that file and the mutant is KILLED iff the self-test goes RED (a FAIL or a MISSED line). A mutant whose anchor text is not found
exactly N times is an ANCHOR ERROR (never counted as killed). A control run (unmutated, every section E,D,RA,RB,RC) must be green first. A survivor is a defect of the INSTRUMENT (fix the self-test, re-run); it is never dropped.
For mutants of the CHAIN / LAUNCHER / WAKER the runner sets R2_SKIP_REGEN=1: the self-test's "equals a fresh regeneration" check (RB1) would kill every such mutant by construction, so it is skipped and the kill must come from a behavioural check.

    python mutate_oaf_r2.py [--only A1,L3] [--workers 2] [--out mutation_r2_output.txt] [--json mutation_r2.json] [--scratch F:/TanitAD-scratch/oaf_r2_tmp/mw]
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
SHIP = HERE / "box2_ship_r2"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
AS, LIB, CH, RUN, WK = "gate_oaf_assert_r2.py", "oaf_r2_lib.sh", "chain_v8f_dagoacc_r2.sh", "run_chain_v8f_dagoacc_r2.cmd", "wake_v8fdagoacc_r2.sh"
RD_LOOP = 'for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff"):'

# (id, file, description, old, new, count, sections)
MUTANTS = [
    # ---- the asserter's new branch  -> E (the original section, r2 asserter) + RA
    ("A1", AS, "the residual_zero_init condition is DROPPED: the exact-0.0 identity applies to every record (the real fine-tune gate would be refused again)", "    if zi is True:\n        if not (finite(rd) and rd == 0.0):", "    if True:\n        if not (finite(rd) and rd == 0.0):", 1, "E,RA"),
    ("A2", AS, "the identity is never applied: a from-scratch record (residual_zero_init True) is judged as a fine-tune", "    if zi is True:\n", "    if False:\n", 1, "E,RA"),
    ("A3", AS, "fine-tune branch: `> 0.0` -> `>= 0.0` (a read_o that receives NOTHING passes)", "        if not (finite(rd) and rd > 0.0):", "        if not (finite(rd) and rd >= 0.0):", 1, "E,RA"),
    ("A4", AS, "fine-tune branch: finite() dropped (inf and True pass)", "        if not (finite(rd) and rd > 0.0):", "        if not (rd is not None and rd > 0.0):", 1, "E,RA"),
    ("A5", AS, "from-scratch branch weakened: exactly 0.0 -> any non-negative finite value", "        if not (finite(rd) and rd == 0.0):", "        if not (finite(rd) and rd >= 0.0):", 1, "E,RA"),
    ("A6", AS, "the undecidable branch removed (a missing / non-bool residual_zero_init is silently accepted)", '        bad.append(f"levers.residual_zero_init = {zi!r} is neither True nor False: the read_o_grad_diff reading {rd!r} cannot be judged")', "        pass", 1, "E,RA"),
    ("A7", AS, "the FIRST ATTEMPT's rule is back: read_o_grad_diff among the exact-0.0 controls", RD_LOOP, RD_LOOP.replace('("control_repeat_total_diff"', '("read_o_grad_diff", "control_repeat_total_diff"'), 1, "E,RA"),
    ("A8", AS, "a registered control dropped from the loop (control_prior_grad_diff)", RD_LOOP, RD_LOOP.replace(' "control_prior_grad_diff",', ""), 1, "E,RA"),
    ("A9", AS, "`zi is True` -> `zi == True` (the int 1 is taken for True)", "    if zi is True:\n", "    if zi == True:\n", 1, "E,RA"),
    ("A10", AS, "`zi is False` -> `zi == False` (the int 0 is taken for False)", "    elif zi is False:\n", "    elif zi == False:\n", 1, "E,RA"),
    ("A11", AS, "residual_zero_init is read from acc_probe instead of levers", 'rd, zi = ap.get("read_o_grad_diff"), lv.get("residual_zero_init")', 'rd, zi = ap.get("read_o_grad_diff"), ap.get("residual_zero_init")', 1, "E,RA"),
    ("A12", AS, "from-scratch branch inverted: a NON-zero reading passes, exactly 0.0 is refused", "        if not (finite(rd) and rd == 0.0):", "        if not (finite(rd) and rd != 0.0):", 1, "E,RA"),
    ("A13", AS, "wacc_problems no longer returns the read_o failure (the rule's `bad` is built into a throw-away list)", '            bad.append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is False: a trained head', '            [].append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is False: a trained head', 1, "E,RA"),
    ("A14", AS, "flatdag logic touched (the cut check no longer requires exactly-zero perturbation effects) -- outside the one allowed change", '            if q.get(k) != 0.0:\n                bad.append(f"o_interface_probe {nm}.{k}', '            if False:\n                bad.append(f"o_interface_probe {nm}.{k}', 1, "E,RA"),
    # ---- the move-aside lib  -> RB + RC
    ("L1", LIB, "the box-2 job DELETES instead of renaming (ren -> rd /s /q)", 'ren "%~1" "%~nx1_r1_assert_stop"', 'rd /s /q "%~1"', 1, "RB,RC"),
    ("L2", LIB, "the 'both exist' refusal removed from the job (ren is left to fail on its own)", 'if exist "%~1_r1_assert_stop" (echo ASIDE-FAIL %~1 and its aside sibling %~nx1_r1_assert_stop both exist: refusing to overwrite or to guess & exit /b 1)\n', "", 1, "RB,RC"),
    ("L3", LIB, "the job never writes its marker", """    echo '>"%RS%\\r2_aside_done.txt" echo r2 aside done %DATE% %TIME%'\n""", "", 1, "RB,RC"),
    ("L4", LIB, "the job ignores its marker (a relaunch moves / refuses r2's own files)", """    echo 'if exist "%RS%\\r2_aside_done.txt" (echo ASIDE-ALREADY-DONE an earlier launch of r2 moved the first attempt artifacts aside, marker %RS%\\r2_aside_done.txt & goto :end)'\n""", "", 1, "RB,RC"),
    ("L5", LIB, "an item is missing from the job (the gate log of seed 0)", " 'gate_M2b-dagoacc_s0.log'", "", 1, "RB,RC"),
    ("L6", LIB, "an item is missing from the job (the seed-0 dagpin file: a moved pin could never refuse)", " 'dagpin\\M2b_s0_dagoacc.train.argv.txt'", "", 1, "RB,RC"),
    ("L7", LIB, "the w_acc record directory is added to the moved items (the stage that passed C-O3 would be recomputed / lost)", "printf '%s\\n' 'gate_M2b-dagoacc_s0'", "printf '%s\\n' 'dagoacc' 'gate_M2b-dagoacc_s0'", 1, "RB,RC"),
    ("L8", LIB, "a failed rename is reported as success (ASIDE-FAIL exits 0)", 'if errorlevel 1 (echo ASIDE-FAIL %~1 could not be renamed & exit /b 1)', 'if errorlevel 1 (echo ASIDE-FAIL %~1 could not be renamed & exit /b 0)', 1, "RB,RC"),
    ("L9", LIB, "the LOCAL aside deletes instead of renaming (mv -> rm -rf)", '  mv "$d" "$n" || stop', '  rm -rf "$d" || stop', 1, "RB,RC"),
    ("L10", LIB, "the local overwrite guard removed (an existing aside copy is moved into / over)", """  [ -e "$n" ] && { echo "ASIDE-KEPT $d : the first attempt's copy is already aside as $n"; return 0; }\n""", "", 1, "RB,RC"),
    ("L11", LIB, "the local aside is skipped in r2_aside_all", '  r2_aside_local "$ART"\n  r2_aside_job_text', "  r2_aside_job_text", 1, "RB,RC"),
    ("L12", LIB, "the count of ASIDE lines is no longer asserted", '  [ "$n" = "$want" ] || stop "the r2 aside job reported $n ASIDE lines, expected $want"', "  true", 1, "RB,RC"),
    ("L13", LIB, "the chain does not wait for the aside job", "  waitrc $j 40\n", "", 1, "RB,RC"),
    ("L14", LIB, "ALREADY-DONE is judged against six lines", "then want=1; else", "then want=6; else", 1, "RB,RC"),
    ("L15", LIB, "the job is shipped but never launched", '  launch $j || stop "launch the r2 aside job"\n', "", 1, "RB,RC"),
    # ---- the chain  -> RB + D (R2_SKIP_REGEN=1)
    ("C1", CH, "the r2_aside_all call is removed from the real run", "\nr2_aside_all\n", "\n", 1, "RB,D"),
    ("C2", CH, "a second r2_aside_all runs BEFORE BOX2_BUSY is taken", '( set -C; echo "v8F-mini PREREG_V8F_DAG_OACC r2', 'r2_aside_all\n( set -C; echo "v8F-mini PREREG_V8F_DAG_OACC r2', 1, "RB,D"),
    ("C3", CH, "the chain asserts with the FIRST attempt's asserter", "ASSERT=$HERE_SHIP/gate_oaf_assert_r2.py", "ASSERT=$HERE_SHIP/gate_oaf_assert.py", 1, "RB,D"),
    ("C4", CH, "the chain's logs dir is the first attempt's (its lines / runs / pulled evidence would be overwritten)", "L=$B/logs; E=$L/v8f_dagoacc_r2", "L=$B/logs; E=$L/v8f_dagoacc", 1, "RB,D"),
    ("C5", CH, "the guard log is the first attempt's", "GUARD_OUT=$L/v8fdagoacc_r2_guard.out", "GUARD_OUT=$L/v8fdagoacc_guard.out", 1, "RB,D"),
    ("C6", CH, "the guard watches the first attempt's chain log (it would never see r2's end marker)", "V8F_CHAIN_OUT=$L/chain_v8f_dagoacc_r2.out", "V8F_CHAIN_OUT=$L/chain_v8f_dagoacc.out", 1, "RB,D"),
    ("C7", CH, "the code archive overwrites the first attempt's tar name (local)", "git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fdagoacc_r2_tip.tar", "git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fdagoacc_tip.tar", 1, "RB,D"),
    ("C8", CH, "the 'r2 files not installed' refusal removed from the real entry", '[ -f "$ASSERT" ] && [ -f "$HERE_SHIP/oaf_r2_lib.sh" ] && type r2_aside_all', "true && type r2_aside_all", 1, "RB,D"),
    ("C9", CH, "oaf_r2_lib.sh is not sourced", 'source "$HERE_SHIP/oaf_r2_lib.sh"', ": # not sourced", 1, "RB,D"),
    ("C10", CH, "the aside runs AFTER the local mkdir of the chain's own dirs (not the first action)", "r2_aside_all\nmkdir -p $E/runs $E/lines      # only now:", "mkdir -p $E/runs $E/lines      # only now:", 1, "RB,D"),
    ("C11", CH, "the two runs are in the wrong order (seed 1 first)", 'RUNS=("dagoacc 0" "dagoacc 1")', 'RUNS=("dagoacc 1" "dagoacc 0")', 1, "RB,D"),
    ("C12", CH, "one hex digit of DAG_s0's checkpoint sha256 changed in the chain", "CKPT_SHA=(d186fd0b", "CKPT_SHA=(e186fd0b", 1, "RB,D"),
    ("C13", CH, "C-O3's STOP swallowed", 'co3 "$ART/wacc/w_acc.json" "$ART/wacc/w_acc.txt" || { echo "ZZDAGOACC-CO3-STOP', 'co3 "$ART/wacc/w_acc.json" "$ART/wacc/w_acc.txt" || true || { echo "ZZDAGOACC-CO3-STOP', 1, "RB,D"),
    # ---- the launcher and the waker  -> RB
    ("W1", RUN, "the launcher waits on the DAG2 chain's log, not FLAT-DAG's (it would start while FLAT-DAG holds box 2, or never wait at all)", "chain_v8f_flatdag.out", "chain_v8f_dag2.out", 1, "RB"),
    ("W2", RUN, "the launcher writes to the first attempt's chain log", "> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out", "> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc.out", 1, "RB"),
    ("W3", WK, "the waker polls the first attempt's log", "chain_v8f_dagoacc_r2.out", "chain_v8f_dagoacc.out", 2, "RB"),
]
LOCK = threading.Lock()


def run_selftest(sections, ship: Path, scratch: Path, env_extra: dict, tag: str, timeout=1800):
    out = scratch / f"out_{tag}.txt"
    cmd = [PY, str(HERE / "selftest_oaf_r2.py"), "--sections", sections, "--r2-ship", str(ship), "--scratch", str(scratch), "--out", str(out)]
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", **env_extra}
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout, errors="replace", stdin=subprocess.DEVNULL)
    txt = (p.stdout or "") + (p.stderr or "")
    bad = [x for x in txt.splitlines() if x.startswith("FAIL") or x.startswith("MISSED") or "Traceback" in x or "Error" in x[:12]]
    return p.returncode, bad, txt, time.time() - t


def one(m, scratch: Path):
    mid, f, desc, old, new, count, secs = m
    d = scratch / "mut" / mid
    if d.exists():
        shutil.rmtree(d)
    sd = d / "ship"
    shutil.copytree(SHIP, sd)
    raw = (sd / f).read_bytes().decode("utf-8")
    c = raw.count(old)
    if c != count:
        return {"id": mid, "file": f, "desc": desc, "status": "ANCHOR_ERROR", "found": c, "declared": count}
    (sd / f).write_bytes(raw.replace(old, new).encode("utf-8"))
    env = {"R2_SKIP_REGEN": "1"} if f in (CH, RUN, WK) else {}
    rc, bad, txt, dt = run_selftest(secs, sd, scratch / "st", env, mid)
    killed = rc != 0 and bool(bad)
    res = {"id": mid, "file": f, "desc": desc, "sections": secs, "seconds": round(dt, 1), "rc": rc, "status": "KILLED" if killed else "SURVIVED", "n_red": len(bad), "red_by": [b[:230] for b in bad[:3]]}
    shutil.rmtree(d, ignore_errors=True)
    return res


def main():
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    assert nw <= 2, "the brief caps the mutation workers at 2"
    outp = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "mutation_r2_output.txt"
    jp = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else HERE / "mutation_r2.json"
    root = Path(sys.argv[sys.argv.index("--scratch") + 1]) if "--scratch" in sys.argv else Path("F:/TanitAD-scratch/oaf_r2_tmp/mw")
    log = []

    def say(s):
        with LOCK:
            log.append(s)
            print(s, flush=True)
    say("MUTATION PROOF -- the r2 files of the DAG-OACC re-run (asserter r2, move-aside lib, r2 chain, launcher, waker)")
    say(f"workers {nw}; scratch {root}")
    c0 = root / "w0"
    c0.mkdir(parents=True, exist_ok=True)
    rc, bad, txt, dt = run_selftest("E,D,RA,RB,RC", SHIP, c0 / "st", {}, "control", timeout=3000)
    try:
        (HERE / "selftest_output_r2.txt").write_text((c0 / "st" / "out_control.txt").read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as e:
        say(f"(could not save selftest_output_r2.txt: {e})")
    last = txt.strip().splitlines()[-1] if txt.strip() else ""
    say(f"CONTROL (unmutated, all sections E,D,RA,RB,RC): rc {rc}, {'GREEN' if rc == 0 and not bad else 'RED ' + str(bad[:3])}  [{last}]  ({dt:.0f} s)")
    if rc != 0 or bad:
        say("the control is not green: the mutation proof is void")
        outp.write_text("\n".join(log) + "\n", encoding="utf-8")
        return 2
    todo = [m for m in MUTANTS if not only or m[0] in only]
    q = __import__("queue").Queue()
    for k in range(1, nw + 1):
        q.put(k)

    def job(m):
        k = q.get()
        try:
            r = one(m, root / f"w{k}")
        except Exception as e:                                           # a crashed runner is NOT a kill
            r = {"id": m[0], "file": m[1], "desc": m[2], "status": "RUNNER_ERROR", "error": repr(e)[:300]}
        finally:
            q.put(k)
        if r["status"] in ("KILLED", "SURVIVED"):
            say(f"{r['id']}: {r['status']:8s} [{r['sections']}] ({r['seconds']:.0f} s, {r['n_red']} red) -- {r['desc']}" + (f"\n      by: {r['red_by'][0][:200]}" if r["red_by"] else ""))
        else:
            say(f"{r['id']}: {r['status']} -- {r['desc']} {({k_: v for k_, v in r.items() if k_ in ('found', 'declared', 'error')})}")
        return r
    with ThreadPoolExecutor(max_workers=nw) as ex:
        results = list(ex.map(job, todo))
    n, k = len(results), sum(r["status"] == "KILLED" for r in results)
    say(f"\nSUMMARY: {k}/{n} mutants killed, {sum(r['status'] == 'SURVIVED' for r in results)} survived, {sum(r['status'] == 'ANCHOR_ERROR' for r in results)} anchor errors, "
        f"{sum(r['status'] == 'RUNNER_ERROR' for r in results)} runner errors; control green")
    for r in results:
        if r["status"] != "KILLED":
            say(f"  NOT A CLEAN KILL: {r['id']} {r['status']} -- {r['desc']}")
    outp.write_text("\n".join(log) + "\n", encoding="utf-8")
    jp.write_text(json.dumps({"control_green": True, "n_mutants": n, "killed": k, "survived": sum(r["status"] == "SURVIVED" for r in results), "anchor_errors": sum(r["status"] == "ANCHOR_ERROR" for r in results),
                              "results": results}, indent=1), encoding="utf-8")
    return 0 if k == n else 1


if __name__ == "__main__":
    sys.exit(main())
