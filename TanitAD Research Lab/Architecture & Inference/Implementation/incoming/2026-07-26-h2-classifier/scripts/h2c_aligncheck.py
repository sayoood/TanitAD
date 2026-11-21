"""H2 classifier — the ALIGNMENT ADMISSION DIAGNOSTIC (dev box, CPU).

`PRE_REGISTRATION.md §2.2` admits a clip only if the best-lag Pearson correlation between the two
ego-speed series reaches 0.99, and declares the run BLOCKED if more than 10 % of clips fail. That
rule has a KNOWN degeneracy that must be measured before it is either obeyed or amended: **Pearson
correlation is undefined for a constant series**, so a clip driven at a steady speed (or parked)
can be perfectly aligned and still score near zero.

This script answers, from the data, which of the two it is:

  * if the failures concentrate on LOW-VARIANCE speed series -> the statistic degenerated, the
    alignment did not fail, and the honest response is a MARKED AMENDMENT with this evidence;
  * if the failures are spread across high-variance clips -> the alignment method really is unfit
    and the pre-registration's BLOCKED branch applies.

It also reports what the dropped clips would have CONTRIBUTED (frames, trigger positives), so the
reader can see whether the drop can plausibly move the headline at all.

usage:  python h2c_aligncheck.py <align_summary.json> <l2tab dir> <bundle dir> <out.json>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "2026-07-26-h2-label-v2", "scripts")))
from l2_label import DEV_CHUNKS, response_r2, trigger_l2_percam  # noqa: E402

TAU = 0.5


def main():
    align_p, tab, bundle, outp = sys.argv[1:5]
    A = json.load(open(align_p))
    meta = json.load(open(os.path.join(bundle, "h2c_meta.json")))
    k2 = {m["k"]: m for m in meta}
    k2clip = json.load(open(os.path.join(bundle, "_LOCAL_ONLY_k2clip.json")))

    stats = {}
    for p in sorted(glob.glob(os.path.join(tab, "l2_*.parquet"))):
        D = pd.read_parquet(p)
        for cid, d in D.groupby("clip_id"):
            tL, tR = trigger_l2_percam(d, TAU)
            stats[cid] = {"v_std": float(d.ego_v.std()), "v_mean": float(d.ego_v.mean()),
                          "frames": int(len(d)), "pos": int((tL | tR).sum()),
                          "label_pos": int(((tL | tR) & response_r2(d)).sum())}

    rows = []
    for a in A["per_clip"]:
        cid = k2clip[str(a["k"])]
        s = stats.get(cid, {})
        rows.append({**{kk: a[kk] for kk in ("k", "chunk", "side", "lag", "corr", "admitted")},
                     **s})
    R = pd.DataFrame(rows)
    adm, drop = R[R.admitted], R[~R.admitted]
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    res = {
        "rule": "admit iff best-lag Pearson corr >= 0.99 (PRE_REGISTRATION.md 2.2)",
        "n_clips": len(R), "n_admitted": int(len(adm)), "n_dropped": int(len(drop)),
        "drop_frac": round(len(drop) / max(len(R), 1), 4),
        "preregistered_block_threshold": 0.10,
        "BLOCKED_by_letter_of_the_rule": bool(len(drop) / max(len(R), 1) > 0.10),
        "v_std_quantiles_admitted": {str(x): round(float(adm.v_std.quantile(x)), 4) for x in q},
        "v_std_quantiles_dropped": {str(x): round(float(drop.v_std.quantile(x)), 4) for x in q},
        "dropped_with_v_std_below_0.5_mps": int((drop.v_std < 0.5).sum()),
        "dropped_with_v_std_below_1.0_mps": int((drop.v_std < 1.0).sum()),
        "admitted_with_v_std_below_0.5_mps": int((adm.v_std < 0.5).sum()),
        "dropped_frames": int(drop.frames.sum()), "dropped_trigger_positives": int(drop.pos.sum()),
        "dropped_label_positives": int(drop.label_pos.sum()),
        "admitted_frames": int(adm.frames.sum()), "admitted_trigger_positives": int(adm.pos.sum()),
        "dropped_positive_share": round(float(drop.pos.sum())
                                        / max(float(R.pos.sum()), 1e-9), 4),
        "per_side_dropped": {s: int((drop.side == s).sum()) for s in ("TRAIN", "HELDOUT")},
        "per_side_admitted": {s: int((adm.side == s).sum()) for s in ("TRAIN", "HELDOUT")},
        "corr_vs_vstd_spearman": float(pd.Series(R.corr_ if hasattr(R, "corr_") else R["corr"])
                                       .rank().corr(R.v_std.rank())),
        "dropped_by_chunk": drop.chunk.value_counts().to_dict(),
    }
    # the discriminating number: among clips with a REAL speed profile, how many failed?
    real = R[R.v_std >= 1.0]
    res["among_clips_with_v_std_ge_1.0"] = {
        "n": int(len(real)), "n_dropped": int((~real.admitted).sum()),
        "drop_frac": round(float((~real.admitted).mean()), 4)}
    json.dump(res, open(outp, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
