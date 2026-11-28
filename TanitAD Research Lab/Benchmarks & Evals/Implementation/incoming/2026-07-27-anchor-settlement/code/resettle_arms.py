"""The OTHER half of the disqualification-lifting sentence: `+0.679`.

`COMMA_YAW_REISSUE.md` §3.3 lifted *"comma is disqualified for yaw"* on TWO
numbers — the deployed head at **+0.3308** and a RETRAINED head at **+0.679**.
`resettle_anchor.py` settles the first (2 of its 22 val episodes are, by content,
inside the deployed head's own training set). This settles the second.

⚠️ THE TWO ARE NOT THE SAME KIND OF PROBLEM, and saying so is the whole point:

  * the deployed head A0 trained on `comma2k19-val-61c46fca8f7f cm_[0:40]`,
    2 of which ARE (bit-identical) the anchor's val clips -> CONTAMINATED.
  * the retrained v3 arms (`R0`, `R0LEG`, `V2R`) trained on the v3 TRAIN split
    of `comma2k19-val-76b6e94a97a1`, which is disjoint from the v3 VAL split
    (verified by content: the cache has no duplicate episodes at all)
    -> NOT contaminated.

So the question for the retrained arms is not leakage but COMPOSITION: the same
2 episodes carry ~4x the yaw variance of the other 20 (gt_std 0.108 vs 0.025),
and R2 is a variance-weighted statistic. If a retrained arm's +0.65 also
evaporates on the other 20 episodes, then "comma can test yaw" rests on 2 clips
either way — a different mechanism reaching the same place.

⛔ Nothing retrained. The persisted per-arm predictions on the identical 4,195
windows are re-used, so every subset below is exact.
⚠️ `V3F` (the shipped composite, +0.6791 — the number actually quoted in §3.3)
is NOT in the persisted prediction file; re-running it needs the composite's
conditioning path. It is reported as UNVERIFIED here rather than inferred from
`R0`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HEAD_TANITEVAL = "/workspace/TanitAD-head/taniteval"
CI_MD5_HEAD = "c92618a02b36f8191a581fb74a491a8d"
sys.path.insert(0, HEAD_TANITEVAL)
from taniteval import ci as tci                                    # noqa: E402

OUT_DIR = Path("/workspace/idm3/out")
INTRAIN_TAGS = ("cm_00018", "cm_00039")
N_BOOT = 2000


def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(1 << 22)):
            h.update(b)
    return h.hexdigest()


assert md5_of(tci.__file__) == CI_MD5_HEAD, \
    f"⛔ ESTIMATOR PIN FAILED — {tci.__file__}"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def r2(p, g):
    p = np.asarray(p, np.float64); g = np.asarray(g, np.float64)
    return float(1.0 - ((p - g) ** 2).sum()
                 / max(((g - g.mean()) ** 2).sum(), 1e-12))


def boot_r2(p, g, eid):
    p = np.asarray(p, np.float64); g = np.asarray(g, np.float64)

    def _r2(idx):
        i = idx.astype(np.int64); gg, pp = g[i], p[i]
        return float(1.0 - ((pp - gg) ** 2).sum()
                     / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
    _r2.__name__ = "r2"
    return tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                         eid, reduce=_r2, n_boot=N_BOOT, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/arms_resettlement.json")
    args = ap.parse_args()

    gt = np.load(OUT_DIR / "val_gt_v3.npy", allow_pickle=True).item()
    S = gt["S"].astype(np.float64)
    eid = np.asarray(gt["eid"]); dom = np.asarray(gt["dom"])
    P = {}
    for f in ("arms_v3_preds.npy", "arms_v3c_preds.npy"):
        d = np.load(OUT_DIR / f, allow_pickle=True).item()
        for k, v in d.items():
            v = v.item() if isinstance(v, np.ndarray) and v.shape == () else v
            P[k] = np.asarray(v["S"], np.float64)
    P["A0_deployed"] = np.load(OUT_DIR / "a0_preds.npy",
                               allow_pickle=True).item()["S"].astype(np.float64)
    log(f"arms {sorted(P)}")

    is_cm = dom == "cm"
    intrain = np.isin(eid, INTRAIN_TAGS)
    subsets = {"cm_ALL22": is_cm,
               "cm_CLEAN20": is_cm & ~intrain,
               "cm_INTRAIN2": is_cm & intrain}

    res = {
        "what": "every persisted v3 arm's comma yaw R2, with and without the 2 "
                "episodes that are (by content) inside the DEPLOYED head's "
                "comma training set",
        "date": "2026-07-27", "agent": "anchor-settlement",
        "evidence_class": "MEASURED (ours; tanitad-eval, this script)",
        "tier": "decision-grade for the v3 val substrate; nothing retrained",
        "estimator": "taniteval.ci.episode_cluster_bootstrap, B=2000, unit = "
                     "the episode. overlapping_holdout_se NOT used.",
        "ci_py": tci.__file__, "ci_py_md5": md5_of(tci.__file__),
        "label_protocol": "repaired (hold_heading_through_standstill, v_min 0.5) "
                          "— the protocol every number below was published under",
        "contamination_status": {
            "A0_deployed": "CONTAMINATED — trained on 2 of these 22 val clips "
                           "(content-verified, raw/anchor_overlap.json)",
            "retrained_v3_arms": "NOT contaminated — trained on the v3 TRAIN "
                                 "split of the same cache, which is disjoint by "
                                 "content (no duplicate episodes in the cache). "
                                 "For these arms the question is COMPOSITION, "
                                 "not leakage.",
        },
        "gt_variance_by_subset": {
            k: {"gt_std": float(S[m, 1].std()), "n": int(m.sum()),
                "n_episodes": int(len(set(eid[m])))}
            for k, m in subsets.items()},
        "V3F_shipped_composite": {
            "published_comma_yaw_r2": 0.6791059738542922,
            "status": "COVERED BY THE R0 ROW — not inferred, IDENTIFIED. "
                      "`ship_v3.json -> composite.rotation_from = 'R0'`, and "
                      "`ship_v3.json -> V3F/yaw_rate/cm/r2` is bit-identical to "
                      "`compare_v3.json -> arms/R0/yaw_rate/cm/r2` "
                      "(0.6791059738542922). V3F's ROTATION HEAD IS R0, so the "
                      "R0 row below IS the +0.679 that "
                      "COMMA_YAW_REISSUE.md §3.3 quotes.",
            "source_files": ["results/ship_v3.json", "results/compare_v3.json"],
        },
        "arms": {},
    }

    for arm in sorted(P):
        row = {}
        for name, m in subsets.items():
            p, g = P[arm][m, 1], S[m, 1]
            row[name] = {"r2": r2(p, g), "rho": spearman(p, g),
                         "mae": float(np.abs(p - g).mean()),
                         "n": int(m.sum()),
                         "r2_ci": boot_r2(p, g, eid[m])}
        row["delta_r2_CLEAN20_minus_ALL22"] = (row["cm_CLEAN20"]["r2"]
                                               - row["cm_ALL22"]["r2"])
        res["arms"][arm] = row
        log(f"  {arm:12s} ALL22 {row['cm_ALL22']['r2']:+.4f}  "
            f"CLEAN20 {row['cm_CLEAN20']['r2']:+.4f}  "
            f"INTRAIN2 {row['cm_INTRAIN2']['r2']:+.4f}  "
            f"(delta {row['delta_r2_CLEAN20_minus_ALL22']:+.4f})")

    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
