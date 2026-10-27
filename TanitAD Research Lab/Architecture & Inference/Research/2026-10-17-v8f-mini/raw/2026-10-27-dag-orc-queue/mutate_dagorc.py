#!/usr/bin/env python3
"""Mutation proof of the DAG-ORC box-2 build. Each mutant is a COPY of one file (the staged queue, or one file of box2_ship/) with ONE defect; the self-test is pointed at the copy (DAGORC_QUEUE / DAGORC_SHIP) and the mutant is
KILLED iff the self-test turns RED. A control (the unmutated copy through the same pipeline) must stay GREEN. Rules (the programme's own mutation discipline):
  * an anchor that does not occur exactly once is an ANCHOR ERROR, never a kill; a harness failure (exit 2, no RESULT line) is a HARNESS ERROR, never a kill;
  * a crash whose traceback names the mutated file counts as a kill (flagged `crash`): the mutated program cannot run;
  * a survivor is reported as a survivor. Nothing is retried to turn a survivor into a kill.
  * the chain mutants run BOTH the offline section D and the end-to-end section E (restricted to the scenarios that exercise the mutated decision): the table says which section(s) killed it. The regeneration guard (D3) is OFF for them
    (DAGORC_SHIP set), so that a mutated chain is judged by what it DOES, not by `it differs from the template`.

    python mutate_dagorc.py [--only substr] [--workers 3]      writes mutation.json and prints the table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
QUEUE = HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
SHIP = HERE / "box2_ship"
WORK = Path(os.environ.get("DAGORC_MUT_WORK") or r"F:\TanitAD-scratch\dagorc_mut")
PYV = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
BS, CR, LF = chr(92), chr(13), chr(10)

# (name, kind, file, old, new[, mode]) ; mode: "all" (every occurrence, default needs count>=1), "one" (exactly one), "last" (the last occurrence)
QUEUE_MUT = [
    ("q01 A's vmax 8.0 -> 7.0 (both law lines)", b"--ego-perturb-vmax 8.0", b"--ego-perturb-vmax 7.0", "all"),
    ("q02 A's dose 0.3 -> 0.31", b'set "DAGORCP=0.3"', b'set "DAGORCP=0.31"', "one"),
    ("q03 B's dose 0.1 -> 0.2", b'set "DAGORCP=0.1"', b'set "DAGORCP=0.2"', "one"),
    ("q04 REP's trainer seed 1S -> S (no replicate)", b'set "DAGSEED=1%~2"', b'set "DAGSEED=%~2"', "one"),
    ("q05 REP also gets the perturbation (p 0.3)", b'if /i "%DAGS%"=="dagorcrep" set "DAGSEED=1%~2"', b'if /i "%DAGS%"=="dagorcrep" (set "DAGSEED=1%~2" & set "DAGORCP=0.3")', "one"),
    ("q06 --dagger-p 0 dropped (the flag's default disables nothing here, but the registered argv names it)", b"%DAGORCL% --dagger-p 0", b"%DAGORCL%", "one"),
    ("q07 --dagger-p 0 -> 0.1", b"%DAGORCL% --dagger-p 0", b"%DAGORCL% --dagger-p 0.1", "one"),
    ("q08 steps 3000 -> 3001", b"--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", b"--steps 3001 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", "one"),
    ("q09 lr 1e-4 -> 2e-4", b"--lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", b"--lr 2e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", "one"),
    ("q10 init is M2b's checkpoint, not DAG_s's", (BS + "M2b_s%~2_dag" + BS + "ckpt_last.pt\" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%").encode(), (BS + "M2b_s%~2" + BS + "ckpt_last.pt\" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%").encode(), "one"),
    ("q11 the law seed is 0 on every seed", b"--ego-perturb-seed %~2", b"--ego-perturb-seed 0", "all"),
    ("q12 yaw-rate std dropped from the law", b" --ego-perturb-yawrate-std 2.0", b"", "all"),
    ("q13 the no-extra-argument refusal is disabled", b'if not "%~3"=="" (echo REFUSED: %DAGS% takes no P', b'if "%~3"=="NEVER" (echo REFUSED: %DAGS% takes no P', "one"),
    ("q14 a PRE-EXISTING stage changes (flatdag0 -> flatdag1 in :dagname)", b'if /i "%~1"=="flatdag0" set "DAGS=flatdag0"', b'if /i "%~1"=="flatdag0" set "DAGS=flatdag1"', "one"),
    ("q15 the new stage exits 1", b"exit /b 0", b"exit /b 1", "last"),
    ("q16 dagorc01 is not a registered arm name", b'if /i "%~1"=="dagorc01" set "DAGS=dagorc01"', b"rem", "one"),
    ("q17 warmup 100 -> 50", b"--warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", b"--warmup 50 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", "one"),
    ("q18 p-gt-start 0.2 -> 0.3", b"--p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", b"--p-gt-start 0.3 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%", "one"),
    ("q19 dagorc01 is never handed off to :dagorcarm", b'if /i "%DAGS%"=="dagorc01" goto :dagorcarm', b"rem", "one"),
    ("q20 the perturbation law flags go only on A (B has no law)", b'if /i "%DAGS%"=="dagorc01" (set "DAGORCP=0.1" & set "DAGORCL=', b'if /i "%DAGS%"=="dagorc01" (set "DAGORCP=0.1" & set "DAGORCLX=', "one"),
]

# the asserter / checks / chain / guard / waker / launcher
ASSERT_MUT = [
    ("a01 LAW_VALUES yaw-rate std 2.0 -> 1.9", 'gate_dagorc_assert.py', '"ego_perturb_yawrate_std": 2.0}', '"ego_perturb_yawrate_std": 1.9}', "one"),
    ("a02 A's registered dose 0.3 -> 0.31", 'gate_dagorc_assert.py', 'ARMS = {"dagorc": 0.3,', 'ARMS = {"dagorc": 0.31,', "one"),
    ("a03 the pin == planned check removed", 'gate_dagorc_assert.py', "    if pin != tl:\n", "    if False:\n", "one"),
    ("a04 the probe's exactly-0.0 controls not enforced", 'gate_dagorc_assert.py', "            if pr.get(k) != 0.0:\n", "            if False:\n", "one"),
    ("a05 the DAgger probe may be anything", 'gate_dagorc_assert.py', ' or dp.get("dagger") != "off"', "", "one"),
    ("a06 the init checkpoint sha256 is not compared (K-INIT)", 'gate_dagorc_assert.py', '    if ic.get("sha256") != a.banked_init_sha:\n', "    if False:\n", "one"),
    ("a07 the ONE-LEVER comparison removed", 'gate_dagorc_assert.py', "                if drop(line) != exp_argv:\n", "                if False:\n", "one"),
    ("a08 gate argv == train argv not enforced", 'gate_dagorc_assert.py', '    if drop([x for x in gl if x != "--selfcheck-only"]) != drop(tl):\n', "    if False:\n", "one"),
    ("a09 --selfcheck-only count not enforced", 'gate_dagorc_assert.py', '    if gl.count("--selfcheck-only") != 1:\n', "    if False:\n", "one"),
    ("a10 second-lever flags allowed on the lines", 'gate_dagorc_assert.py', "        for fx in SECOND_LEVER:\n", "        for fx in ():\n", "one"),
    ("a11 a DAgger bank flag allowed on the lines", 'gate_dagorc_assert.py', "        for k in DAG_ONLY:\n", "        for k in ():\n", "one"),
    ("a12 a law flag allowed on REP's lines", 'gate_dagorc_assert.py', "            for k in EGO_LAW:\n", "            for k in ():\n", "one"),
    ("a13 the token binds the GATE argv, not the training argv", 'gate_dagorc_assert.py', '"argv_sha256": argv_sha256(t_argv), "data_sha256": data}', '"argv_sha256": argv_sha256(g_argv), "data_sha256": data}', "one"),
    ("a14 declared_vs_built not required", 'gate_dagorc_assert.py', '        bad.append("declared_vs_built")', "        pass", "one"),
    ("a15 dead parameters allowed (G-LIVE)", 'gate_dagorc_assert.py', '    if live.get("dead_params") != [] or live.get("declared_dead") != []:\n', "    if False:\n", "one"),
    ("a16 init not-applied / mismatches allowed", 'gate_dagorc_assert.py', ' or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"):', ' or not ic.get("n_tensors"):', "one"),
    ("a17 the TRAIN list need not equal DAG2-0's", 'gate_dagorc_assert.py', '        if i["train_sha256"] != b["train_sha256"]:\n', "        if False:\n", "one"),
    ("a18 REP's probe may be 'on'", 'gate_dagorc_assert.py', '    elif pr.get("ego_perturb") != "off":\n', "    elif False:\n", "one"),
    ("a19 REP's ONE-LEVER seed is 11+S", 'gate_dagorc_assert.py', "        out[ks + 1] = str(10 + seed)\n", "        out[ks + 1] = str(11 + seed)\n", "one"),
    ("a20 p = 0 need not be the identity object", 'gate_dagorc_assert.py', '        if pr.get("control_p0_same_object") is not True:\n', "        if False:\n", "one"),
    ("a21 the probe may see no eligible window", 'gate_dagorc_assert.py', '        if not (isinstance(pr.get("n_selected_at_p1"), int) and pr["n_selected_at_p1"] > 0):\n', "        if False:\n", "one"),
    ("a22 the probe's dose / mode not compared", 'gate_dagorc_assert.py', '        if pr.get("ego_perturb") == "off" or pr.get("p_declared") != p:\n', "        if False:\n", "one"),
    ("a23 built dagger_p not compared", 'gate_dagorc_assert.py', '    if lv.get("dagger_p") != 0.0:\n', "    if False:\n", "one"),
    ("a24 the step count not compared", 'gate_dagorc_assert.py', '    if sc.get("steps_total") != steps:\n', "    if False:\n", "one"),
    ("a25 p_gt not compared", 'gate_dagorc_assert.py', '    if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2):\n', "    if False:\n", "one"),
]
CHECKS_MUT = [
    ("k01 K-REACH tolerance 0.02 -> 0.5", 'v8f_dagorc_chain_checks.py', "REACH_TOL = 0.02", "REACH_TOL = 0.5", "one"),
    ("k02 K-REACH: REP's n may be any non-negative count", 'v8f_dagorc_chain_checks.py', "e > 0 and n == 0 and not isinstance(n, bool)", "e > 0 and n >= 0 and not isinstance(n, bool)", "one"),
    ("k03 K-REACH: REP's zero may rest on eligible = 0", 'v8f_dagorc_chain_checks.py', "isinstance(e, int) and e > 0 and n == 0", "isinstance(e, int) and n == 0", "one"),
    ("k04 run record: REP's trainer seed is the init seed", 'v8f_dagorc_chain_checks.py', '    tseed = 10 + seed if arm == "dagorcrep" else seed\n', "    tseed = seed\n", "one"),
    ("k05 run record: argv.json need not be the planned argv", 'v8f_dagorc_chain_checks.py', '    if aj.get("argv") != planned:\n', "    if False:\n", "one"),
    ("k06 run record: the init sha256 not compared (K-INIT)", 'v8f_dagorc_chain_checks.py', '    if ic.get("sha256") != a.banked_init_sha:\n', "    if False:\n", "one"),
    ("k07 run record: a swapped row (dagger_n) allowed", 'v8f_dagorc_chain_checks.py', '    if sm.get("dagger_n") != 0:\n', "    if False:\n", "one"),
    ("k08 run record: an unfinished run allowed", 'v8f_dagorc_chain_checks.py', '    if sm.get("done") is not True:\n', "    if False:\n", "one"),
    ("k09 run record: pin == planned not checked", 'v8f_dagorc_chain_checks.py', "    if pinned != planned:\n", "    if False:\n", "one"),
    ("k10 one-lever always true", 'v8f_dagorc_chain_checks.py', "    ok = G.drop(line) == exp\n", "    ok = True\n", "one"),
    ("k11 K-REACH: A / B need not be near their dose", 'v8f_dagorc_chain_checks.py', "and p - REACH_TOL - 1e-12 <= n / e <= p + REACH_TOL + 1e-12", "", "one"),
    ("k12 run record: the step count not compared", 'v8f_dagorc_chain_checks.py', '    if sm.get("steps") != steps:\n', "    if False:\n", "one"),
]
CHAIN_MUT = [      # name, anchor old, new, E scenarios that exercise the mutated decision
    ("c01 the gate's assertions are not a stop", '|| stop "gate DAG-ORC assertions $g"', "|| true", "E_gate_init,E_gate_probe"),
    ("c02 K-REACH is printed but not enforced", 'stop "reached-control fail $g"', "true", "E_reach0,E_rep_applied"),
    ("c03 the run record is not a stop", '|| stop "run DAG-ORC record $g"', "|| true", "E_argv"),
    ("c04 the tip need not carry the levers", 'lever_check git "$T" || stop "the tip does not carry the DAG-ORC queue stages / the perturbation levers (the greps above): land them first"', 'lever_check git "$T" || true', "E_noapply,E_oldtip"),
    ("c05 the stage grep is not a stop", '|| stop "the tip\'s queue has no $st stage"', "|| true", "E_oldtip"),
    ("c06 box 2's DAG_s checkpoint is not asserted before the gates", 'pinned_b "before the gates"', "true", "E_ckpt1"),
    ("c07 the v9 md5 is not a stop", '|| stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"', "|| true", "E_md5"),
    ("c08 box 2's key files need not equal the tip's", '|| stop "box 2 $f != tip $T"', "|| true", "E_keysha"),
    ("c09 run order A s0, A s1, REP s0, REP s1", 'AB_RUNS=("dagorc 0" "dagorcrep 0" "dagorc 1" "dagorcrep 1")', 'AB_RUNS=("dagorc 0" "dagorc 1" "dagorcrep 0" "dagorcrep 1")', "E_happy"),
    ("c10 the START disk floor removed", 'disk_floor "$ART" "$NEED_BYTES" "the six DAG-ORC runs"      #', "true      #", "E_happy"),
    ("c11 a default predecessor (AFTER_OUT may be unset)", 'if [ -z "${AFTER_OUT:-}" ]; then', "if false; then", "E_happy"),
    ("c12 the stage list does not require dagorcarm", "STAGES=(gatedag traindag evaldag dagname dagarm dagorcarm)", "STAGES=(gatedag traindag evaldag dagname dagarm)", "E_oldtip"),
    ("c13 the loop's ego.apply is not among the tip's required lines", "  'stack/scripts/v8f_mini_train.py|^ +b = ego" + BS + ".apply" + BS + "(b, step" + BS + ")'\n", "", "E_noapply"),
    ("c14 B runs even with DAGORC_SKIP_B=1", 'if [ "${DAGORC_SKIP_B:-0}" != 1 ]; then', "if true; then", "E_skipb"),
    ("c15 the busy marker is taken without noclobber", "( set -C; echo", "( echo", "E_happy"),
    ("c16 the end-of-chain box-2 pin check removed", 'pinned_b "after the last run"', "true", "E_happy"),
    ("c17 the pulled checkpoint's hash is not a stop", '[ "$l" = "$(ckget "${tg}_after_evals")" ] || stop "$g: the pulled checkpoint\'s sha256 $l != the value measured on box 2"', 'true', "E_pullcorrupt"),
    ("c18 a checkpoint that changed during the evals is not a stop", '[ "$h" = "$h0" ] || stop "$g\'s checkpoint changed during the evals: $h != $h0"', "true", "E_evaltouch"),
    ("c19 the one-lever reference (DAG2-0's argv.json) is not handed to the gate asserter", ' --ref-argv-json "$(cygpath -m $DAG2_DIR/M2b_s${s}_dag20/argv.json)")', ")", "-"),
    ("c20 the dry-run's ONE-LEVER verdict is ignored", '[ "$same" = SAME ] && [ "$ol" = ONE-LEVER ] ||', "true ||", "-"),
    ("c21 a stop does not release the busy marker (the chain never records that it took it)", "BUSY_TAKEN=1\n", "BUSY_TAKEN=0\n", "E_gate_init"),
]
OTHER_MUT = [
    ("g01 guard: the temperature trip count 4 -> 9", 'v8f_guard_dagorc.sh', '[ "$n61" -ge 4 ]', '[ "$n61" -ge 9 ]', "one"),
    ("w01 waker: the K-REACH stop marker dropped", 'wake_v8fdagorc.sh', "^ZZDAGORC-CONTROL-FAIL", "^ZZDAGORC-CONTROL-FAILX", "one"),
    ("w02 waker: gates-passed threshold 4 -> 2", 'wake_v8fdagorc.sh', "-ge 4", "-ge 2", "one"),
    ("l01 launcher: AFTER_OUT is the FLAT-DAG log", 'run_chain_v8f_dagorc.cmd', "chain_v8f_dagoacc_r2.out bash", "chain_v8f_flatdag.out bash", "one"),
    ("l02 launcher: stdin not closed (< NUL removed)", 'run_chain_v8f_dagorc.cmd', '2>&1" < NUL', '2>&1"', "one"),
]


def rm(p):
    """rmtree that also removes git's read-only object files; retries while a process of an earlier stage still holds a handle"""
    import stat

    def h(fn, path, *_):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    for k in range(6):
        try:
            try:
                shutil.rmtree(p, onexc=h)
            except TypeError:
                shutil.rmtree(p, onerror=h)
            return
        except OSError:
            if k == 5:
                raise
            time.sleep(1.5)


class AnchorError(Exception):
    pass


def apply(data: bytes, old: bytes, new: bytes, mode: str) -> bytes:
    n = data.count(old)
    if mode == "one" and n != 1:
        raise AnchorError(f"anchor occurs {n} times, need exactly 1: {old[:70]!r}")
    if n < 1:
        raise AnchorError(f"anchor absent: {old[:70]!r}")
    if mode == "last":
        i = data.rindex(old)
        return data[:i] + new + data[i + len(old):]
    return data.replace(old, new)


def run_selftest(name, sections, env, scratch):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "DAGORC_SCRATCH": str(scratch), "DAGORC_E_PAR": "2", **env}
    t0 = time.time()
    p = subprocess.run([PYV, str(HERE / "selftest_dagorc.py"), "--sections", sections], capture_output=True, env=e, timeout=3600)
    out = (p.stdout or b"").decode("utf-8", "replace").replace("\r", "")
    err = (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")
    red = [x for x in out.split("\n") if x.startswith(("FAIL ", "MISSED "))]
    has_result = any(x.startswith("RESULT:") for x in out.split("\n"))
    return {"sections": sections, "rc": p.returncode, "red": red[:4], "n_red": len(red), "result_line": next((x for x in out.split("\n") if x.startswith("RESULT:")), ""), "has_result": has_result, "stderr_tail": err[-600:], "stdout_tail": out[-400:], "sec": round(time.time() - t0)}


def classify(r, mutated_name=None):
    """killed | crash | survived | harness"""
    if r["has_result"]:
        return "killed" if r["n_red"] else "survived"
    if r["rc"] == 1 and mutated_name and mutated_name in r["stderr_tail"] + r["stdout_tail"]:
        return "crash"
    return "harness"


def do_mutant(spec):
    name, kind = spec["name"], spec["kind"]
    d = WORK / spec["id"]
    if d.exists():
        rm(d)
    d.mkdir(parents=True)
    env, mutated_file = {}, None
    try:
        if kind == "queue":
            q = apply(QUEUE.read_bytes(), spec["old"], spec["new"], spec["mode"])
            (d / "queue.cmd").write_bytes(q)
            env["DAGORC_QUEUE"] = str(d / "queue.cmd")
            mutated_file = "queue.cmd"
            stages = ["S,A", "B,C,D"]
        else:
            shutil.copytree(SHIP, d / "ship", ignore=shutil.ignore_patterns("__pycache__"))
            f = d / "ship" / spec["file"]
            raw = f.read_bytes()
            old = spec["old"] if isinstance(spec["old"], bytes) else spec["old"].encode()
            new = spec["new"] if isinstance(spec["new"], bytes) else spec["new"].encode()
            f.write_bytes(apply(raw, old, new, spec["mode"]))
            env["DAGORC_SHIP"] = str(d / "ship")
            mutated_file = spec["file"]
            stages = spec["stages"]
    except AnchorError as e:
        return {**spec_public(spec), "status": "anchor-error", "detail": str(e)}
    runs, status = [], None
    for k, sec in enumerate(stages):
        extra = {}
        if sec.startswith("E:"):
            sec, only = "E", sec[2:]
            extra["DAGORC_E_ONLY"] = only
        r = run_selftest(name, sec, {**env, **extra}, d / "scratch")
        r["stage"] = sec + (f"[{extra['DAGORC_E_ONLY']}]" if extra else "")
        runs.append(r)
        c = classify(r, mutated_file)
        r["verdict"] = c
        if spec["kind"] != "chain" and c in ("killed", "crash"):
            break
    verdicts = [r["verdict"] for r in runs]
    if "harness" in verdicts:
        status = "harness-error"
    elif any(v in ("killed", "crash") for v in verdicts):
        status = "killed"
    else:
        status = "SURVIVED"
    return {**spec_public(spec), "status": status, "killed_by": [r["stage"] for r in runs if r["verdict"] in ("killed", "crash")], "runs": runs}


def spec_public(spec):
    return {"id": spec["id"], "name": spec["name"], "kind": spec["kind"]}


def specs():
    out = []
    for nm, old, new, mode in QUEUE_MUT:
        out.append({"id": nm.split()[0], "name": nm, "kind": "queue", "old": old, "new": new, "mode": mode})
    for nm, f, old, new, mode in ASSERT_MUT:
        out.append({"id": nm.split()[0], "name": nm, "kind": "asserter", "file": f, "old": old, "new": new, "mode": mode, "stages": ["C", "D", "E:E_gate_init,E_gate_probe,E_happy"]})
    for nm, f, old, new, mode in CHECKS_MUT:
        out.append({"id": nm.split()[0], "name": nm, "kind": "checks", "file": f, "old": old, "new": new, "mode": mode, "stages": ["C", "D", "E:E_argv,E_reach0,E_rep_applied,E_happy"]})
    for nm, old, new, only in CHAIN_MUT:
        st = ["D"] + ([f"E:{only}"] if only != "-" else [])
        out.append({"id": nm.split()[0], "name": nm, "kind": "chain", "file": "chain_v8f_dagorc.sh", "old": old, "new": new, "mode": "one", "stages": st})
    for nm, f, old, new, mode in OTHER_MUT:
        out.append({"id": nm.split()[0], "name": nm, "kind": "other", "file": f, "old": old, "new": new, "mode": mode, "stages": ["D"]})
    return out


def control():
    """the unmutated copies through the identical pipeline: both must be GREEN"""
    res = {}
    d = WORK / "control"
    if d.exists():
        rm(d)
    d.mkdir(parents=True)
    (d / "queue.cmd").write_bytes(QUEUE.read_bytes())
    r = run_selftest("control-queue", "S,A,B,C,D", {"DAGORC_QUEUE": str(d / "queue.cmd")}, d / "scratch_q")
    res["queue copy through S,A,B,C,D"] = {"green": r["has_result"] and r["n_red"] == 0 and r["rc"] == 0, **r}
    shutil.copytree(SHIP, d / "ship", ignore=shutil.ignore_patterns("__pycache__"))
    r = run_selftest("control-ship", "C,D,E", {"DAGORC_SHIP": str(d / "ship"), "DAGORC_E_ONLY": "E_happy,E_gate_init,E_reach0,E_argv,E_noapply,E_skipb"}, d / "scratch_s")
    res["ship copy through C,D,E[subset]"] = {"green": r["has_result"] and r["n_red"] == 0 and r["rc"] == 0, **r}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--anchors-only", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    allspec = [s for s in specs() if not a.only or any(k and (s["id"] == k or k in s["name"]) for k in a.only.split(","))]
    if a.anchors_only:
        bad = 0
        for s in allspec:
            src = QUEUE if s["kind"] == "queue" else SHIP / s["file"]
            try:
                apply(src.read_bytes(), s["old"] if isinstance(s["old"], bytes) else s["old"].encode(), s["new"] if isinstance(s["new"], bytes) else s["new"].encode(), s["mode"])
            except AnchorError as e:
                bad += 1
                print("ANCHOR ERROR", s["name"], e)
        print(f"{len(allspec)} mutants, {bad} anchor errors")
        return 1 if bad else 0
    t0 = time.time()
    ctl = {} if (a.skip_control or a.only) else control()
    for k, v in ctl.items():
        print(("CONTROL GREEN  " if v["green"] else "CONTROL RED    ") + k + "  " + v["result_line"], flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = []
        for r in ex.map(do_mutant, allspec):
            res.append(r)
            print(f"{r['status']:13s} {r['name']}" + (f"   killed by {r['killed_by']}" if r.get("killed_by") else "") + (f"   {r.get('detail', '')}" if r["status"] == "anchor-error" else ""), flush=True)
    n = {k: sum(1 for r in res if r["status"] == k) for k in ("killed", "SURVIVED", "anchor-error", "harness-error")}
    print(f"MUTATION: {n['killed']} killed, {n['SURVIVED']} survived, {n['anchor-error']} anchor errors, {n['harness-error']} harness errors, of {len(res)} mutants; control green: {all(v['green'] for v in ctl.values()) if ctl else 'skipped'}  ({time.time() - t0:.0f} s)")
    out = HERE / "mutation.json"
    if a.only and out.exists():      # a re-run of a few mutants REPLACES their entries in the recorded run (same ids); the control of the full run is kept
        prev = json.loads(out.read_text(encoding="utf-8"))
        byid = {m["id"]: m for m in prev["mutants"]}
        for r in res:
            r["rerun"] = True
            byid[r["id"]] = r
        order = [s["id"] for s in specs()]
        res = [byid[i] for i in order if i in byid]
        ctl = ctl or prev.get("control", {})
        n = {k: sum(1 for r in res if r["status"] == k) for k in ("killed", "SURVIVED", "anchor-error", "harness-error")}
        print(f"MERGED INTO mutation.json: {n['killed']} killed, {n['SURVIVED']} survived, {n['anchor-error']} anchor errors, {n['harness-error']} harness errors, of {len(res)} mutants")
    out.write_text(json.dumps({"control": ctl, "mutants": res, "counts": n}, indent=1, default=str), encoding="utf-8")
    return 0 if (n["SURVIVED"] == 0 and n["anchor-error"] == 0 and n["harness-error"] == 0 and all(v["green"] for v in ctl.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
