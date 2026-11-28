"""GEO10K step 6: the speed-limit sidecar rows (refcv6_speed_max_v8_train.jsonl schema) for the new clips.
The existing sidecar is a pure transform of s2_labels_v8 SPEED_BAND.v_hi_ms (build_refcv6_speed_max_window.build), and SPEED_BAND is
UNCONDITIONAL geometry (s2_geom_emit_v7.tactical_goals: max of the ego's realised speed over the 2-6 s band after t0 = 8 s). Here the
same function is called with cot=None, so no Alpamayo record is read. CONTROL: for the 20 existing-corpus clips the v_hi_ms from this
path must equal the SPEED_BAND.v_hi_ms stored in s2_labels_v8_train.jsonl.gz, exactly."""
import gzip, json, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/root/geo10k/code")
import geo_common as G
G.point_ego_lookup()
G.ES._index()
import s2_geom_emit_v7 as S2
from tanitad.data.v2_dataset import stable_episode_id
from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_KMH_V6, N_SPEED_MAX_BINS_V6, speed_max_bin


def vhi(cid):
    tr = G.ES.load(cid, hz=S2.HZ, max_s=G.ES.RAW_T0_S + S2.LOOKAHEAD_S + 5.0)
    key, poses = tr.key_index, tr.poses
    seq = S2.manoeuvre_sequence(poses, key, S2.HZ)
    goals, anchor, viol = S2.tactical_goals(poses, key, seq, None, S2.HZ, lat_action=None, clip_id_for_time=None, turn_suppressed=False)
    sb = goals.get("SPEED_BAND")
    return cid, (None if sb is None else float(sb["v_hi_ms"]))


def row(cid, v):
    sid = int(stable_episode_id(cid))
    if v is None:
        return {"sid": sid, "clip_id": cid, "v_hi_ms": None, "bin": None, "limit_kmh": None, "one_hot": [0.0] * N_SPEED_MAX_BINS_V6, "over_ceiling": None, "valid": 0}
    i, over = speed_max_bin(v)
    oh = [0.0] * N_SPEED_MAX_BINS_V6
    oh[i] = 1.0
    return {"sid": sid, "clip_id": cid, "v_hi_ms": round(v, 6), "bin": i, "limit_kmh": int(SPEED_MAX_STEPS_KMH_V6[i]), "one_hot": oh, "over_ceiling": bool(over), "valid": 1}


if __name__ == "__main__":
    sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]
    ctl = [l.strip() for l in open(f"{G.IN}/control_clips20.txt") if l.strip()]
    with Pool(24) as pool:
        res = dict(pool.map(vhi, ctl + sel, chunksize=25))
    recs = {}
    with gzip.open("/root/geo10k/ref/tanitad-v7-training-corpus/labels/s2_labels_v8_train.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if r["clip_id"] in set(ctl):
                    recs[r["clip_id"]] = r
    rows = []
    for c in ctl:
        ref = ((recs[c].get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND", {}).get("v_hi_ms")
        mine = res[c]
        rows.append({"clip": c, "ref_v_hi": ref, "mine_v_hi": mine, "equal": (ref == mine) or (ref is not None and mine is not None and float(ref) == float(mine))})
    # regression arm: the control must go RED when one clip reads another clip's value
    wrong = list(res[ctl[1]] for _ in [0])[0]
    red = not (wrong == recs[ctl[0]]["g_tac"]["goals"]["SPEED_BAND"]["v_hi_ms"])
    rep = {"n": len(rows), "n_equal": sum(r["equal"] for r in rows), "rows": rows, "regression_arm_went_red": bool(red)}
    json.dump(rep, open(f"{G.OUT}/control_speed.json", "w"), indent=1)
    print("[g6] control SPEED_BAND v_hi equal:", rep["n_equal"], "/", rep["n"], " regression arm red:", red, flush=True)
    n_none = 0
    with open(f"{G.OUT}/speed_sidecar_new10k.jsonl", "w") as fh:
        for c in sel:
            r = row(c, res[c])
            n_none += int(r["valid"] == 0)
            fh.write(json.dumps(r) + "\n")
    print(f"[g6] wrote {len(sel)} rows, valid=0: {n_none}")
    print("ZZ GEO10K-G6 DONE ZZ")
