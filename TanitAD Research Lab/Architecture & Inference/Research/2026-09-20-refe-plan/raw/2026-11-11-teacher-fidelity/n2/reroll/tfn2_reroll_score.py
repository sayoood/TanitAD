#!/usr/bin/env python3
"""TF-N2 re-roll, scoring stage (navtrain held-out ONLY, never navtest). One re-rolled arm -> Tc20s and P5-R conversion -> W-TRANSP scoring.

Exactly `n2/tfn2_probe.py`'s construction, with the teacher's 20 poses `t20` read from the arm's own roll rows instead of the bank:
  T      t20 as is (to_navsim 8 poses)
  C      Tc20s = iterate20(t20, cubic)                                        (N1-C form)
  R      P5-R, lambda* = 3.0, warm start Tc20s (p5r_labels.solve)
  human  the DB's logged future (W-TRANSP --human-arm form)
The scorer is W-TRANSP's own functions on the nuPlan-DB rebuild of the metric-cache parts (the navtrain route; no harness cache exists
for navtrain, so the A1 `--metric-cache-root` repair does NOT apply). Row format = W-TRANSP's, so `families_run.py` reads it unchanged.

  python tfn2_reroll_score.py --w <export dir with refe/ eval/ p5s/> --rolls "<glob of roll jsonl>" --arm <label> --out <jsonl> --shard i/N
Identity control (any arm, reported for R0): per token `bank_dxy_m` = max |dxy| of the re-rolled t20 against the BANKED t20 (TF-2 npz / TF-2b rows).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

N2 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2"
sys.path.insert(0, N2)
import tfn2_probe as P  # noqa: E402  (score_arm, mech, load_keys, DB_ROOT, P5R, LAM)


def load_rolls(pattern):
    out = {}
    for fp in sorted(glob.glob(pattern)):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" in r or "traj20" not in r:
                continue
            out[(r["log"], r["token"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--rolls", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for p in (os.path.join(a.w, "refe"), os.path.join(a.w, "eval"), os.path.join(a.w, "eval", "instruments"), P.P5R):
        sys.path.insert(0, p)
    os.environ["P5C_PKG"] = a.w
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    import transp_score as TS
    import p5r_labels as R
    from p4_tracker_comp import tracked8
    from p5_target_labels import iterate20
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    bank = P.load_keys(a.w)
    rolls = load_rolls(a.rolls)
    si, sn = (int(x) for x in a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add(r["token"])
    todo = [k for k in sorted(rolls) if k[1] not in done and int(hashlib.sha1(k[1].encode()).hexdigest(), 16) % sn == si]
    if a.limit:
        todo = todo[:a.limit]
    by = {}
    for lg, tok in todo:
        by.setdefault(lg, []).append(tok)
    dbs = NS.index_dbs(db_root=P.DB_ROOT)
    t0, n = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by):
            if not os.path.exists(dbs.get(lg, "")):
                for t in by[lg]:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], by[lg]):
                tok = sc._initial_lidar_token
                rr = rolls[(lg, tok)]
                t20 = np.asarray(rr["traj20"], np.float64)
                rec = {"log": lg, "token": tok, "src": bank[(lg, tok)][2], "arm_run": a.arm, "arms": {}, "knobs": rr["knobs"],
                       "roll_sec": rr["sec"]}
                ts = time.time()
                try:
                    rec["bank_dxy_m"] = float(np.abs(t20[:, :2] - bank[(lg, tok)][0][:, :2]).max())
                    parts = PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    ctl = {"knot_exact": True, "knot_checks": 0}
                    tc = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")[0]
                    track = lambda Pb: tracked8(ND, Pb, ego)                    # noqa: E731
                    rl = R.solve(t20, tc - t20, [P.LAM], track, ND.to_navsim)[P.LAM]["label"]
                    arms = {"T": t20, "C": tc, "R": rl}
                    hum = TS.human_rel(sc, ego)
                    sim_c = None
                    for nm, p20 in arms.items():
                        rec["arms"][nm], sim = P.score_arm(TS, parts, ego, s0, p20)
                        if nm == "C":
                            sim_c = sim
                    rec["arms"]["human"], _ = P.score_arm(TS, parts, ego, s0, hum[1::2][:20].copy(), hum[4::5][:8].copy())
                    v0 = ego.dynamic_car_state.rear_axle_velocity_2d
                    rec["shared"] = {"v0": float(math.hypot(v0.x, v0.y)), "human8": hum[4::5][:8].tolist(),
                                     "lead": TS.lead_track(sc, ego)}
                    rec["mech"] = P.mech(sc, parts, sim_c, t20)
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:                                           # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
    print("ZZTFN2_SCORE_DONE", a.arm, a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
