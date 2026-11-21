"""H2 classifier — does the SUBSET I evaluate on still carry the decision-relevance that was gated?

The L2 GO rests on a lift of 2.41x [1.3998, 3.7041] over **1,415** CONFIRM clips. The classifier is
evaluated on the **subset** of those clips that are in the pod2 episode cache AND clear the
alignment floor — a few hundred, not 1,415. If that subset does not itself carry the lift, then a
perfect classifier would be predicting something with no measured decision-relevance, and the whole
exercise would be uninterpretable. This is the check for that, and it is run BEFORE the classifier's
own numbers are read.

Estimator: paired episode-cluster bootstrap, ratio form, B = 2000, seed 0, `taniteval.ci._draws`
via `2026-07-25-h2-e0-e1/scripts/h2e_stats.py` — the same machinery the label work used.
NEVER `overlapping_holdout_se`.

usage:  python h2c_subset_lift.py <l2tab> <bundle> <align_summary.json> <out.json>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
L2S = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-26-h2-label-v2", "scripts"))
E01 = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-25-h2-e0-e1", "scripts"))
sys.path.insert(0, L2S)
sys.path.insert(0, E01)
from h2e_stats import paired_cluster_diff, paired_cluster_lift  # noqa: E402
from l2_label import DEV_CHUNKS, response_r2, trigger_l2  # noqa: E402

TAU = 0.5


def main():
    tab, bundle, align_p, outp = sys.argv[1:5]
    meta = {m["k"]: m for m in json.load(open(os.path.join(bundle, "h2c_meta.json")))}
    k2clip = json.load(open(os.path.join(bundle, "_LOCAL_ONLY_k2clip.json")))
    A = json.load(open(align_p))
    admitted = {k2clip[str(a["k"])] for a in A["per_clip"] if a["admitted"]}
    side = {k2clip[str(k)]: m["side"] for k, m in meta.items()}

    res = {}
    for name, keep in (("HELDOUT_admitted", lambda c: c in admitted and side.get(c) == "HELDOUT"),
                       ("TRAIN_admitted", lambda c: c in admitted and side.get(c) == "TRAIN"),
                       ("ALL_26_chunks_full_table", lambda c: True)):
        parts = []
        for p in sorted(glob.glob(os.path.join(tab, "l2_*.parquet"))):
            D = pd.read_parquet(p)
            D = D[[keep(c) for c in D.clip_id]]
            if len(D):
                parts.append(D)
        if not parts:
            continue
        D = pd.concat(parts, ignore_index=True)
        g = trigger_l2(D, TAU)
        r = response_r2(D)
        eid = pd.factorize(D.clip_id)[0]
        res[name] = {
            "n_frames": int(len(D)), "n_clips": int(D.clip_id.nunique()),
            "n_trigger_pos": int(g.sum()),
            "n_trigger_pos_clips": int(D.assign(t=g).groupby("clip_id").t.max().sum()),
            "trigger_rate": float(g.mean()), "response_base_rate": float(r.mean()),
            "lift": paired_cluster_lift(r, g, eid),
            "risk_difference": paired_cluster_diff(r, g, eid),
        }
    res["published_reference"] = {
        "source": "2026-07-26-h2-label-v2/l2_confirm.json (INHERITED, not re-derived)",
        "lift": 2.41, "ci": [1.3998, 3.7041], "n_clips": 1415, "n_trigger_pos": 1192}
    json.dump(res, open(outp, "w"), indent=2, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "risk_difference"}
                      for k, v in res.items() if isinstance(v, dict) and "lift" in v},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
