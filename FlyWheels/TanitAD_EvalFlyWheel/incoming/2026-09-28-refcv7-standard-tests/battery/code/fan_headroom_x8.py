"""X8 follow-on lever for D-REFCV7-FANCEIL: how much of the oracle-in-fan gap is REACHABLE.

Two questions on the banked fan, both T0 DIAGNOSTICS (they use GT):
  (1) oracle among the TOP-K candidates by the model own score (anchor_logits, refined_logits,
      sel_score): the ceiling of ANY reranker that is restricted to the model own top-K.
      selected - oracle_topK = the headroom such a reranker could still win.
  (2) oracle among a RANDOM K-subset (R draws averaged per window): the order-statistic null.
      Min over K noisy candidates falls with K even for a worthless fan, so oracle_topK is only
      informative against oracle_randomK at the same K.
Estimator: FULL-SET mean + paired episode-cluster bootstrap (taniteval ci.py, n_boot 2000, seed 0).
CI answers ONLY: would another draw of EPISODES say this. Not training seed, not inference seed.
Usage: python fan_headroom_x8.py --fan-npz F --gt-dump D --out-json O [--rand-draws 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fan_control_x8 as X  # noqa: E402

KS = (1, 2, 4, 8, 16, 32, 64)
RAND_SEED_BASE = 20261111


def oracle_in_subset(de, mask):
    """de [B,N] per-candidate ADE; mask [B,N] bool -> [B] min ADE inside the subset."""
    return np.where(mask, de, np.inf).min(1)


def topk_mask(score, k):
    idx = np.argsort(-score, axis=1, kind="stable")[:, :k]
    m = np.zeros(score.shape, bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def rand_mask(shape, k, rng):
    r = rng.random(shape)
    idx = np.argsort(r, axis=1)[:, :k]
    m = np.zeros(shape, bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--gt-dump", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--rand-draws", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.fan_npz, allow_pickle=True)
    ci = np.asarray(z["clip_index"]).astype(np.int64)
    ws = np.asarray(z["ws"]).astype(np.int64)
    fan = z["anchor_traj"]
    sel = np.asarray(z["sel_idx"]).astype(np.int64).ravel()
    B, N = fan.shape[0], fan.shape[1]
    tab, n_ep_files = X.load_gt_dump(a.gt_dump)
    keys = [(int(c), int(w)) for c, w in zip(ci, ws)]
    assert len(set(keys)) == B and all(k in tab for k in keys) and B == len(tab)
    gt = np.stack([tab[k][0] for k in keys])
    eid = [tab[k][1] for k in keys]
    de = X._de_all(fan, gt)
    selv = de[np.arange(B), sel]
    orc = de.min(1)
    out = {"n_windows": B, "n_candidates": N, "n_episodes": len(set(eid)),
           "selected": X._boot(selv, eid, "selected"),
           "oracle_full": X._boot(orc, eid, "oracle_full"), "topk_oracle": {}, "rand_oracle": {}}
    for nm in ("anchor_logits", "refined_logits", "sel_score"):
        sc = np.asarray(z[nm])
        assert sc.shape == (B, N), nm
        rows = {}
        for k in KS:
            o = oracle_in_subset(de, topk_mask(sc, k))
            rows["K%d" % k] = {"oracle": X._boot(o, eid, "oracle_top%d_%s" % (k, nm)),
                               "selected_minus_oracle": X._paired(selv, o, eid, "selected - oracle_top%d_%s" % (k, nm))}
        out["topk_oracle"][nm] = rows
    for k in KS + (N,):
        per = []
        for r in range(a.rand_draws):
            rng = np.random.default_rng(RAND_SEED_BASE + 1000 * k + r)
            per.append(oracle_in_subset(de, rand_mask(de.shape, k, rng)))
        o = np.mean(per, axis=0)
        out["rand_oracle"]["K%d" % k] = {"oracle_mean_over_draws": X._boot(o, eid, "oracle_rand%d" % k),
                                         "n_draws": a.rand_draws}
    out["stamp"] = {"tier": "T0 diagnostic (every oracle uses GT)",
                    "estimator": "FULL-SET mean; episode_cluster_bootstrap / paired_episode_cluster_bootstrap n_boot=2000 seed=0",
                    "ci_answers": "ONLY: would another draw of EPISODES say this? NOT training seed, NOT inference seed",
                    "fan_npz": a.fan_npz, "fan_npz_sha256": X._sha256(a.fan_npz),
                    "rand_seed_base": RAND_SEED_BASE, "wall_s": round(time.time() - t0, 1),
                    "script": os.path.abspath(__file__)}
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE", a.out_json, "wall_s", out["stamp"]["wall_s"])


if __name__ == "__main__":
    main()
