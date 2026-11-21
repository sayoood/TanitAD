"""v2 BALANCED corpus selector — greedy water-filling toward an explicit target.

Phase-1 (selection design only). Reads the scored pool (score_v2_pool.py), keeps
moving clips (excludes parked/degenerate), then GREEDILY selects K clips so the
AGGREGATE per-timestep maneuver distribution + speed-regime hit an explicit target
that corrects the parity corpus's lane-keep skew.

METHOD — greedy marginal-gain water-filling. Maintain running class totals; at
each step add the remaining clip that minimizes
    WMAN * L1(maneuver_frac, TMAN) + WSPD * L1(speed_frac, TSPD).
Water-filling keeps the running distribution ON the target throughout, so at K
clips the aggregate equals the target whenever the target is feasible. Every clip
competes equally — the original 2,376 parity clips are re-selected under this same
objective, NOT grandfathered in (Sayed's directive: balance the WHOLE corpus, not
a balanced blob on a skewed base). Deterministic (argmin, no RNG).

NO country cap is applied: the candidate chunks were drawn country-stratified, so
the pool is already ~5 %/country and greedy keeps max share ~5.5 %.

Balancing is KINEMATIC (ego poses only). It cannot create semantic coverage
(traffic lights, roundabouts-as-class, pedestrians, merges) — that needs the VLM/
map track and is out of scope here.

Usage:
  python select_v2_corpus.py [--pool v2_pool_scored.parquet] [--k 9000]
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
import numpy as np, pandas as pd

# TARGET (per-timestep maneuver fractions) — halve lane-keep dominance, ~double
# turns (balanced L/R), keep accel/brake ~parity. Feasible: the turn ceiling for
# K=9000 is 36 % (top-by-turn), so 28 % leaves room to also hold the speed balance.
TMAN = np.array([0.45, 0.14, 0.14, 0.13, 0.14])   # lane_keep turn_left turn_right accelerate brake_stop
TSPD = np.array([0.10, 0.52, 0.38])               # stopped(<1) city(1-12) highway(>12) m/s
WMAN, WSPD = 1.0, 0.5

# parked/degenerate exclusion (keep highway; only drop non-driving recordings)
MIN_MEAN_V, MAX_STOP_FRAC, MIN_DIST_M = 1.0, 0.9, 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="v2_pool_scored.parquet")
    ap.add_argument("--k", type=int, default=9000)
    ap.add_argument("--out", default="r0_selection_v2.parquet")
    ap.add_argument("--meta", default="v2_selection_meta.json")
    ap.add_argument("--root", default=r"C:\Users\Admin\tanitad-data\physicalai",
                    help="physicalai root (for the catalog's canonical clip->chunk)")
    a = ap.parse_args()

    df = pd.read_parquet(a.pool)
    df = df[df.clip_is_valid & (df.split.astype(str) == "train")]
    pool = df[(df.mean_v >= MIN_MEAN_V) & (df.stop_frac < MAX_STOP_FRAC)
              & (df.dist_m >= MIN_DIST_M)].reset_index(drop=True)
    # A handful of clip_ids are bundled in TWO egomotion chunk zips (dataset quirk),
    # so the scorer emitted duplicate rows. The camera mp4 lives in the catalog's
    # CANONICAL chunk (clip_index has a unique clip_id index) — align `chunk` to it
    # so fetch-camera hits the right zip, then keep one row per clip_id.
    canon = pd.read_parquet(os.path.join(a.root, "clip_index.parquet"))["chunk"]
    pool["chunk"] = pool["clip_id"].map(canon).fillna(pool["chunk"]).astype(int)
    pool = pool.drop_duplicates("clip_id", keep="first").reset_index(drop=True)
    pool["turnfrac"] = (pool.tl + pool.tr) / pool.nlab
    N = len(pool)
    print(f"[sel] pool moving={N} ({N*20.1/3600:.1f}h); selecting K={a.k}", flush=True)

    man = pool[["lk", "tl", "tr", "ac", "bs"]].to_numpy(np.float64)   # [N,5] timestep counts
    nlab = pool["nlab"].to_numpy(np.float64)
    spd = pool[["stopped", "city", "hw"]].to_numpy(np.float64) * nlab[:, None]

    cur_m, cur_s, cur_nl = np.zeros(5), np.zeros(3), 0.0
    avail = np.ones(N, bool); sel = []; t0 = time.time()
    for step in range(a.k):
        nl_new = cur_nl + nlab
        d = (WMAN * np.abs((cur_m + man) / nl_new[:, None] - TMAN).sum(1)
             + WSPD * np.abs((cur_s + spd) / nl_new[:, None] - TSPD).sum(1))
        d[~avail] = np.inf
        i = int(np.argmin(d))
        sel.append(i); avail[i] = False
        cur_m += man[i]; cur_s += spd[i]; cur_nl += nlab[i]
        if step % 1500 == 0:
            print(f"[sel] {step}/{a.k} {time.time()-t0:.0f}s", flush=True)
    S = pool.iloc[sel].copy()

    amf = (S[["lk", "tl", "tr", "ac", "bs"]].sum() / S[["lk", "tl", "tr", "ac", "bs"]].sum().sum())
    asf = S[["stopped", "city", "hw"]].mean()
    ids = sorted(S.clip_id.astype(str))
    key = hashlib.sha1(json.dumps(
        {"ids": ids, "target": TMAN.tolist() + TSPD.tolist(), "k": a.k},
        sort_keys=True).encode()).hexdigest()[:12]
    corpus = f"physicalai-v2bal-{key}"

    cols = ["clip_id", "chunk", "country", "hour_of_day", "platform_class",
            "mean_v", "stop_frac", "dist_m", "net_head", "cum_head", "junction",
            "has_stop", "has_brake", "has_turn", "turnfrac", "stopped", "city",
            "hw", "lk", "tl", "tr", "ac", "bs", "nlab", "win_s"]
    S[cols].to_parquet(a.out)
    meta = dict(
        corpus_key=corpus, n_clips=int(len(S)), hours=round(len(S) * 20.1 / 3600, 2),
        target_maneuver=dict(zip(["lane_keep", "turn_left", "turn_right",
                                  "accelerate", "brake_stop"], TMAN.tolist())),
        target_speed=dict(zip(["stopped", "city", "highway"], TSPD.tolist())),
        achieved_maneuver={k: round(float(v), 4) for k, v in amf.items()},
        achieved_speed={k: round(float(v), 4) for k, v in asf.items()},
        junction_clip_frac=round(float(S.junction.mean()), 4),
        stop_clip_frac=round(float(S.has_stop.mean()), 4),
        n_camera_chunks=int(S.chunk.nunique()),
        camera_chunks_needed=sorted(S.chunk.astype(int).unique().tolist()))
    json.dump(meta, open(a.meta, "w"), indent=2)
    print(f"[sel] {corpus}  turns={amf['tl']+amf['tr']:.4f}  "
          f"junction={S.junction.mean():.3f} stop={S.has_stop.mean():.3f}", flush=True)
    print(f"[sel] WROTE {a.out} + {a.meta}", flush=True)


if __name__ == "__main__":
    main()
