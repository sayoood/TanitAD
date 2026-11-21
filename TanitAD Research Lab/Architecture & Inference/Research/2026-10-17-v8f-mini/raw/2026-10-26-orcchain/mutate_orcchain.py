"""mutate_orcchain.py: the deliberate-regression proof of test_v8f_orc_chain.py. For each mutant ONE edit (a list of (old, new) text pairs, EACH asserted to match exactly once) is applied to a COPY of the queue or the chain,
the new test file runs against the copies (env ORC_CHAIN_QUEUE / ORC_CHAIN_SCRIPT; nothing in the tree is touched) with -x, and the mutant is recorded RED (the suite caught it, naming the first failing test) or
SURVIVED (the proof is incomplete). A control run on the unmutated copies must be green first.

    python mutate_orcchain.py --tree <export root holding stack/ and the v8f-mini package> --out mutation.out [--only NAME ...] [--no-control]

The pytest cap of this machine is respected: a run starts only when `pytest_runs.py` prints 0 or 1."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CAP = "C:/Users/Admin/orcread/pytest_runs.py"

# (name, target, [(old, new), ...], what the mutation is)
M = [
    ("Q1 drop --init-ckpt", "queue", [('set "ORCF=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000', 'set "ORCF=--steps 3000')], "the fine-tune no longer starts from the banked M2b checkpoint"),
    ("Q2 swap p between the arms", "queue", [('(set "ORCP=0.3" & set "ORCS=orc")', '(set "ORCP=0" & set "ORCS=orc")'), ('(set "ORCP=0" & set "ORCS=orc0")', '(set "ORCP=0.3" & set "ORCS=orc0")')],
     "ORC trains without the perturbation and ORC-0 with it"),
    ("Q3 --ego-perturb-p not named", "queue", [('--ego-perturb-p %ORCP%%ORCL%"', '%ORCL%"')], "the flag's default disables the lever: nothing names it"),
    ("Q4 gate line carries a lever the run does not", "queue", [('%A% %ORCF% --selfcheck-only"', '%A% %ORCF% --p-gt-end 0.3 --selfcheck-only"')], "the gate gates another run than the one that trains"),
    ("Q5 perturbation seed fixed to 0", "queue", [('--ego-perturb-seed %~2"', '--ego-perturb-seed 0"')], "seed 1 perturbs with seed 0's draws"),
    ("Q6 init of the wrong seed", "queue", [('M2b_s%~2\\ckpt_last.pt" --steps 3000', 'M2b_s0\\ckpt_last.pt" --steps 3000')], "seed 1 fine-tunes seed 0's checkpoint"),
    ("Q7 the skip runs before the dry-run echo", "queue", [('if defined V8F_DRYRUN (echo %TX% & exit /b 0)\nif exist "%OUT%\\summary.json" (echo [skip] M2b_s%ORCSEED%_%ORCS% & exit /b 0)\n',
                                                         'if exist "%OUT%\\summary.json" (echo [skip] M2b_s%ORCSEED%_%ORCS% & exit /b 0)\nif defined V8F_DRYRUN (echo %TX% & exit /b 0)\n')], "a finished run prints [skip] instead of the line it ran"),
    ("Q8 the eval reads B's checkpoint", "queue", [('set "TG=M2b-%ORCS%_s%ORCSEED%"\nset "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt"', 'set "TG=M2b-%ORCS%_s%ORCSEED%"\nset "E0="%PY%" "%EVAL%" run --ckpt "%RUNSM%\\M2b_s%ORCSEED%\\ckpt_last.pt"')],
     "the eval scores the base, not the fine-tune"),
    ("Q9 an older stage edited", "queue", [("echo [ok] gate M2b-mixo\n", "echo [ok] gate M2b-mixo!\n")], "a label the PREREG did not touch changes"),
    ("Q10 yaw-rate std dropped from ORC's law flags", "queue", [(" --ego-perturb-yawrate-std 2.0", "")], "the lateral half of the lever is not named"),
    ("Q11 a law flag leaks onto ORC-0", "queue", [('if /i "%~1"=="orc" set "ORCL=', 'if defined ORCP set "ORCL=')], "ORC-0 carries the five law flags again (the shape the trainer's gate refused: Amendment 3 undone)"),
    ("Q12 a law flag missing on ORC", "queue", [(" --ego-perturb-adrop 1.5", "")], "ORC loses the adrop law flag"),
    ("Q13 ORC loses all five law flags", "queue", [('--ego-perturb-p %ORCP%%ORCL%"', '--ego-perturb-p %ORCP%"')], "ORCL is set but never read: both arms are p-only"),
    ("Q14 a law flag value moved on ORC", "queue", [("--ego-perturb-yawrate-std 2.0", "--ego-perturb-yawrate-std 1.0")], "ORC's registered yaw-rate std changes"),
    ("C1 a run is skipped", "chain", [('RUNS=("orc 0" "orc0 0" "orc 1" "orc0 1")', 'RUNS=("orc 0" "orc0 0" "orc 1")')], "ORC-0 s1 never runs"),
    ("C2 the runs are in another order", "chain", [('RUNS=("orc 0" "orc0 0" "orc 1" "orc0 1")', 'RUNS=("orc 0" "orc 1" "orc0 0" "orc0 1")')], "seed 1 ORC before seed 0 ORC-0"),
    ("C3 the token binds the gate argv, not the training argv", "chain", [("binding\": {\"commit\": commit, \"argv_sha256\": gate_argv_sha256(t_argv)", "binding\": {\"commit\": commit, \"argv_sha256\": gate_argv_sha256(g_argv)")],
     "orc_read.g_token refuses every run"),
    ("C4 a wrong banked checkpoint sha", "chain", [("601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f9")], "the chain would assert a checkpoint that is not the banked one"),
    ("C5 a default predecessor log", "chain", [('if [ -z "${AFTER_OUT:-}" ]; then', 'AFTER_OUT=${AFTER_OUT:-/c/Users/Admin/box2_ship/logs/chain_v8f_mixo.out}\nif [ -z "${AFTER_OUT:-}" ]; then')], "an OLD log's end marker satisfies the wait at once"),
    ("C6 the gate assertions skip the init sha", "chain", [('if ic.get("sha256") != banked:', "if False:")], "a fine-tune from the wrong checkpoint passes the gate"),
    ("C7 the gates run after the runs", "chain", [('for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done\n', ""),
                                                  ('for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done\n', 'for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done\nfor run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done\n')],
     "a gate that fails for the fourth run is found after three runs were trained"),
    ("C8 a failed gate assertion does not stop the chain", "chain", [("<<'EOF' || stop \"gate ORC assertions $g\"", "<<'EOF' || true")], "no PASS, yet training goes on"),
    ("C9 the busy marker is not taken atomically", "chain", [('( set -C; echo "v8F-mini PREREG_V8F_ORECOVER', '( echo "v8F-mini PREREG_V8F_ORECOVER')], "two chains can both arm"),
    ("C10 the end marker is renamed", "chain", [('echo "ZZV8FMINI-DONE $(date', 'echo "ZZV8FMINI-OK $(date')], "the guard and the next chain never see the end"),
    ("C11 the run record skips argv.json = planned", "chain", [('if aj.get("argv") != planned:', "if False:")], "a token binds an argv that did not run"),
    ("C12 stop does not release the box", "chain", [('echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }', 'echo "" > $CUR; exit 1; }')], "a stopped chain leaves the busy marker"),
    ("C13 the reach bound is widened", "chain", [("0.3 - 0.02 - 1e-12 <= n / e <= 0.3 + 0.02 + 1e-12", "0.3 - 0.2 - 1e-12 <= n / e <= 0.3 + 0.2 + 1e-12")], "a perturbed fraction of 0.15 passes"),
    ("C14 the wait does not stop on the predecessor's STOP", "chain", [('echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }', 'echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; }')],
     "a stopped predecessor is waited on forever"),
    ("C15 an empty AFTER_OUT is accepted", "chain", [('if [ -z "${AFTER_OUT:-}" ]; then', 'if [ -z "${AFTER_OUT+x}" ]; then')], "AFTER_OUT= passes the refusal"),
    ("C16 the pre-flight no longer looks for the apply line", "chain", [("git -C $MIRROR show \"$T:stack/scripts/v8f_mini_train.py\" | grep -q 'b = ego.apply(b, step)' || stop \"the tip's trainer does not apply the perturbation in its loop\"\n", "")], "a tip without the apply line is accepted"),
    ("C17 the guard is not started", "chain", [("V8F_CHAIN_OUT=$L/chain_v8f_orc.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8forc_guard.out 2>&1 &\n", "")], "the temperature / time guard is blind"),
    ("C18 the gate no longer requires the ORC-0 probe to read off", "chain", [("elif pr.get(\"ego_perturb\") != \"off\":", "elif False:")], "a perturbation leaking into the control passes"),
    ("C19 the gate/train identity is not checked", "chain", [("if drop_out(g2) != drop_out(t2): bad.append", "if False: bad.append")], "the gate gates another argv than the run trains"),
    ("C20 a law flag named on ORC-0 is not refused", "chain", [("            if k in fl: bad.append(f\"the queue's {name} line names {k} on ORC-0", "            if False: bad.append(f\"the queue's {name} line names {k} on ORC-0")], "ORC-0 may carry a law flag"),
    ("C21 ORC-0's built law values are not checked", "chain", [("**(LAW_ORC if p > 0.0 else LAW_DEFAULT)", "**(LAW_ORC if p > 0.0 else {})")], "a moved built law value passes on the control"),
    ("C22 ORC is not required to name the law flags", "chain", [("if p > 0.0: want.update(LAWF)", "if False: want.update(LAWF)")], "ORC may lose a law flag"),
    ("C23 ORC-0's expected built yaw-rate std is ORC's", "chain", [('"ego_perturb_yawrate_std": 0.0, "ego_perturb_seed": 0}', '"ego_perturb_yawrate_std": 2.0, "ego_perturb_seed": 0}')], "the control's built law is expected at ORC's value"),
]


def wait_cap():
    for _ in range(2000):
        n = subprocess.run([PY, CAP], capture_output=True, text=True).stdout.strip()
        if n in ("0", "1"):
            return
        time.sleep(4)
    raise SystemExit("the pytest cap never cleared")


def run_suite(tree, qf, cf, tag, pt):
    wait_cap()
    env = dict(os.environ, ORC_CHAIN_QUEUE=str(qf), ORC_CHAIN_SCRIPT=str(cf), CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", f"--basetemp={pt}/{tag}", "tests/test_v8f_orc_chain.py"], cwd=Path(tree) / "stack", env=env, capture_output=True, text=True, encoding="utf-8", timeout=1500)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pt", default="E:/Projects/TanitAD-artifacts/orcchain_build/pt")
    ap.add_argument("--work", default="E:/Projects/TanitAD-artifacts/orcchain_build/mut")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args()
    tree, work = Path(a.tree), Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    q0 = (tree / PK / "code" / "v8f_mini_box2_queue.cmd").read_bytes()
    c0 = (tree / PK / "code" / "box2_ops" / "chain_v8f_orc.sh").read_bytes()
    lines, red = [], 0
    ran = [m for m in M if not a.only or any(k in m[0] for k in a.only)]
    lines.append(f"# mutation proof of stack/tests/test_v8f_orc_chain.py ({time.strftime('%Y-%m-%d %H:%M')}): {len(ran)} mutants; each is ONE edit of a COPY (every substitution matched exactly once), the suite runs with -x")
    if not a.no_control:
        rc, out = run_suite(tree, work / "q0.cmd" if False else tree / PK / "code" / "v8f_mini_box2_queue.cmd", tree / PK / "code" / "box2_ops" / "chain_v8f_orc.sh", "control", a.pt)
        tail = [x for x in out.splitlines() if re.search(r"\d+ (passed|failed)", x)][-1:]
        lines.append(f"CONTROL (no mutation): {'GREEN' if rc == 0 else 'RED'}  {' '.join(tail)}")
        if rc != 0:
            lines.append(out[-3000:])
            Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
            raise SystemExit("the control is not green: the mutation proof means nothing")
    for k, (name, target, subs, what) in enumerate(ran):
        qf, cf = work / f"queue_{k}.cmd", work / f"chain_{k}.sh"
        text = (q0 if target == "queue" else c0).decode("utf-8").replace("\r\n", "\n")
        for old, new in subs:
            assert text.count(old) == 1, f"{name}: the anchor matches {text.count(old)} times: {old[:80]!r}"
            text = text.replace(old, new)
        if target == "queue":
            qf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            cf.write_bytes(c0)
        else:
            qf.write_bytes(q0)
            cf.write_bytes(text.encode("utf-8"))
        rc, out = run_suite(tree, qf, cf, f"m{k}", a.pt)
        first = next((x for x in out.splitlines() if x.startswith("FAILED") or x.startswith("ERROR")), "")
        tail = [x for x in out.splitlines() if re.search(r"\d+ (passed|failed|error)", x)][-1:]
        verdict = "RED" if rc != 0 else "SURVIVED"
        red += rc != 0
        lines.append(f"{verdict:8s} {name}  [{what}]\n         first failure: {first.split(' - ')[0][7:] if first else '-'}  | {' '.join(tail)}")
        print(lines[-1], flush=True)
    lines.append(f"SUMMARY: {red} of {len(ran)} mutants RED" + ("" if red == len(ran) else f"; {len(ran) - red} SURVIVED"))
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])
    return 0 if red == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
