"""D6 P1' CPU stage (SPEC_P1P2_A2_P1PRIME.md s5 steps 3-4).   tanitad venv launcher; scoring subprocesses run in the NAVSIM venv.  CPU only, <= 2 scorers, free-commit gate 6 GB.

Order of events (every wait reads an ARTIFACT, never a process status):
  1. wait for raw/p1x_gpu_done.json (the GPU export of all 5,912 scenes is complete);
  2. wait for the OFFICIAL step-50,400 R7_A1 navhard scoring: counts.json status PASS, csv_token_rows 5912 (not mine; the runner is still retrying on RAM_GUARD);
  3. build the token sets by RULE (d6_make_spec_sets_p1prime.py) -> raw/spec_p1x_token_sets.json; write raw/p1x_sets_ready.json and WAIT for raw/P1X_SETS_ACK.txt
     (the Master Mind receives the sets + sha256 BEFORE any fan is scored; the ack file is created after the message is sent);
  4. fan scoring of P1'_all (2 shards, exact controls on Cpass' u K1fail', k5 40) + K4s (+30 m STATE shift on Cpass', no Tier C) -> raw/p1x_scored_s{0,1}.jsonl, raw/p1x_scored_K4s.jsonl;
  5. raw/p1x_cpu_done.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_p1p2_cpu as C                                            # noqa: E402  (run_pool, free_gb, log, env, paths)

RAW = C.RAW
M50 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/scores_navhard")
COUNTS = f"{M50}/score_R7_A1__navhard_two_stage.counts.json"
HOOKS50 = f"{M50}/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_hooks.json"


def scoring_pass() -> bool:
    try:
        c = json.load(open(COUNTS, encoding="utf-8"))
    except Exception:                                               # noqa: BLE001
        return False
    return c.get("status") == "PASS" and int(c.get("csv_token_rows", -1)) == 5912 and os.path.exists(HOOKS50)


def main() -> int:
    C.log("P1X: waiting for the GPU export")
    while not os.path.exists(os.path.join(RAW, "p1x_gpu_done.json")):
        time.sleep(60)
    C.log("P1X: GPU export done; waiting for the OFFICIAL 50,400 navhard scoring (counts.json PASS)")
    while not scoring_pass():
        time.sleep(120)
    C.log("P1X: official scoring PASS -> building the token sets by rule")
    if not os.path.exists(os.path.join(RAW, "spec_p1x_token_sets.json")):
        subprocess.run([C.TPY, os.path.join(HERE, "d6_make_spec_sets_p1prime.py")], env=C.ENV, check=True)
    json.dump({"ready": time.strftime("%Y-%m-%dT%H:%M:%S"), "sets_file": "raw/spec_p1x_token_sets.json"}, open(os.path.join(RAW, "p1x_sets_ready.json"), "w"))
    C.log("P1X: sets built; waiting for raw/P1X_SETS_ACK.txt (sets must reach the Master Mind before any fan is scored)")
    while not os.path.exists(os.path.join(RAW, "P1X_SETS_ACK.txt")):
        time.sleep(30)
    fan = os.path.join(RAW, "p1x_fan", "fan_R7_A1.jsonl")
    ctrl = os.path.join(RAW, "spec_tokens_p1x_K1ctrl.txt")
    cp = [l.strip() for l in open(os.path.join(RAW, "spec_tokens_p1x_Cpass.txt")) if l.strip()]
    kf = [l.strip() for l in open(os.path.join(RAW, "spec_tokens_p1x_K1fail.txt")) if l.strip()]
    open(ctrl, "w").write("\n".join(sorted(set(cp) | set(kf))) + "\n")
    allp = os.path.join(RAW, "spec_tokens_p1x_all.txt")
    jobs = []
    for k in range(2):
        jobs.append((f"p1x_s{k}", [C.NPY, os.path.join(HERE, "d6_fan_score.py"), "--fan", fan, "--hooks", HOOKS50, "--out", os.path.join(RAW, f"p1x_scored_s{k}.jsonl"),
                                   "--shard", f"{k}/2", "--only-tokens", allp, "--exact-ids", ctrl, "--k5", "40"]))
    jobs.append(("p1x_K4s", [C.NPY, os.path.join(HERE, "d6_fan_score.py"), "--fan", fan, "--hooks", HOOKS50, "--out", os.path.join(RAW, "p1x_scored_K4s.jsonl"),
                             "--only-tokens", os.path.join(RAW, "spec_tokens_p1x_Cpass.txt"), "--shift-states", "30.0", "--no-tierc"]))
    C.run_pool(jobs)
    json.dump({"done": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(RAW, "p1x_cpu_done.json"), "w"))
    C.log("P1X: CPU stage DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
