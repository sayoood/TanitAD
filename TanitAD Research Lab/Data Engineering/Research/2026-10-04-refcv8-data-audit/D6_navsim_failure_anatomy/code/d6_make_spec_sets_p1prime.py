"""D6 -- build the P1' token sets (SPEC_P1P2_A2_P1PRIME.md s2) from the OFFICIAL step-50,400 R7_A1 navhard scoring.   tanitad venv (calls the navsim venv for the reference flag).

    python d6_make_spec_sets_p1prime.py

Refuses unless  .../step50400/scores_navhard/score_R7_A1__navhard_two_stage.counts.json  reads status PASS with csv_token_rows == 5912 AND the final-scores frame exists
(read from the artifact -- the runner's own exit status is not evidence).  Writes raw/spec_tokens_p1x_{S1,S1b,S2,Cpass,K1fail,all}.txt and raw/spec_p1x_token_sets.json
(sizes, sha256, N_pass', the source frame's sha256, the rule constants).  Deterministic: re-running on the same frame gives byte-identical files.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
M50 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/scores_navhard")
WR = f"{M50}/score_R7_A1__navhard_two_stage_wrapper"
FRAME = f"{WR}/R7_A1__navhard_two_stage_final_scores_frame.csv"
HOOKS = f"{WR}/R7_A1__navhard_two_stage_hooks.json"
COUNTS = f"{M50}/score_R7_A1__navhard_two_stage.counts.json"
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
SEED_CPASS, SEED_K1 = 20261004, 20261005


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    c = json.load(open(COUNTS, encoding="utf-8"))
    if c.get("status") != "PASS" or int(c.get("csv_token_rows", -1)) != 5912 or not os.path.exists(FRAME):
        sys.exit(f"REFUSED: the official 50,400 R7_A1 navhard scoring is not complete (status={c.get('status')}, rows={c.get('csv_token_rows')}, frame exists={os.path.exists(FRAME)})")
    df = pd.read_csv(FRAME)
    assert len(df) == 5912 and df["token"].is_unique
    df = df.set_index("token")
    for col in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"):
        assert col in df.columns, col
    NC, DAC, DDC, TLC = (df[k] for k in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"))
    dz = sorted(df.index[DAC == 0])
    tf = os.path.join(RAW, "_p1x_dac0_tokens.txt")
    open(tf, "w", newline="\n").write("\n".join(dz) + "\n")
    out_ref = os.path.join(RAW, "_p1x_refdac.json")
    env = dict(os.environ, PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    subprocess.run([NPY, os.path.join(HERE, "d6_ref_dac.py"), HOOKS, tf, out_ref], env=env, check=True)
    ref = json.load(open(out_ref))
    assert set(ref) == set(dz)
    s1 = sorted(t for t in dz if ref[t] == 1.0)
    s1b = sorted(t for t in dz if ref[t] != 1.0)
    s2 = sorted(df.index[NC == 0])
    pop = sorted(df.index[(NC == 1) & (DAC == 1) & (DDC == 1) & (TLC == 1)])
    cpass = sorted(np.random.default_rng(SEED_CPASS).choice(pop, 200, replace=False))
    k1f = sorted(np.random.default_rng(SEED_K1).choice(s1, min(100, len(s1)), replace=False))
    allp = sorted(set(s1) | set(s1b) | set(s2) | set(cpass))
    sets = {"S1": s1, "S1b": s1b, "S2": s2, "Cpass": cpass, "K1fail": k1f, "all": allp}
    info = {"source_frame": FRAME, "source_frame_sha256": sha(open(FRAME, "rb").read()), "counts_status": c.get("status"), "N_pass": len(pop),
            "seeds": {"Cpass": SEED_CPASS, "K1fail": SEED_K1}, "sets": {}}
    for k, v in sets.items():
        body = ("\n".join(v) + "\n").encode()
        open(os.path.join(RAW, f"spec_tokens_p1x_{k}.txt"), "wb").write(body)
        info["sets"][k] = {"n": len(v), "sha256": sha(body)}
    json.dump(info, open(os.path.join(RAW, "spec_p1x_token_sets.json"), "w"), indent=1)
    os.remove(tf)
    os.remove(out_ref)
    print(json.dumps({k: v["n"] for k, v in info["sets"].items()}), "N_pass", len(pop))


if __name__ == "__main__":
    main()
