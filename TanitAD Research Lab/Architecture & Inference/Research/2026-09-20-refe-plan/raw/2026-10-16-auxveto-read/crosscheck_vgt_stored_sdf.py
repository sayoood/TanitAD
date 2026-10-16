#!/usr/bin/env python3
"""crosscheck_vgt_stored_sdf.py -- POST-HOC PIPELINE CROSS-CHECK, NOT PART OF THE PREREG AND DECIDES NOTHING.

Does the V-GT arm's veto set (signed distance DERIVED FROM THE MASK, as §3 specifies) equal the veto set obtained from the label store's STORED int8 signed distance
(the field the feasibility package's `AGREEMENT.json` used)?  The two differ only by the store's 0.1 m quantisation, its +-12.7 m clip and its 13.2 m padding.
A disagreement beyond a handful of boundary proposals would point at a bug in `auxveto_read.margins_from_masks`.  Nothing here changes any number in AUXVETO_READ.json.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auxveto_read as av  # noqa: E402
import aux_label_store as A  # noqa: E402
import geodac as G  # noqa: E402

av.assert_prereg()
w, _ = av.load_world()
gt = av.load_gt_masks(w)
mine = av.margins_from_masks(w, gt, 0.5, G.GT_GEOM)
stored = np.stack([av.min_margins(G.gt_field_m(A.read_label(av.STORE, lg, t)), G.GT_GEOM, w.proposals[i]) for i, (t, lg) in enumerate(zip(w.token, w.log))])
out = {"note": "POST-HOC pipeline cross-check; decides nothing", "n_proposals": int(mine.size)}
for m in (0.0, 0.25, 0.5, 1.0, 2.0):
    a, b = av.vetoed(mine, m), av.vetoed(stored, m)
    out[f"m={m}"] = {"veto_sets_differ_proposals": int((a != b).sum()), "share_differ": float((a != b).mean()), "mine_vetoed": int(a.sum()), "stored_vetoed": int(b.sum())}
pk_a, pk_b = av.all_picks(w, av.vetoed(mine, av.MARGIN)), av.all_picks(w, av.vetoed(stored, av.MARGIN))
out["picks_differ_tokens_at_0.25"] = int((pk_a != pk_b).sum())
fin = np.isfinite(mine) & np.isfinite(stored)
out["margin_abs_diff_m"] = {"max": float(np.abs(mine - stored)[fin].max()), "p99": float(np.percentile(np.abs(mine - stored)[fin], 99)), "median": float(np.median(np.abs(mine - stored)[fin]))}
json.dump(out, open(HERE / "CROSSCHECK_VGT_STORED_SDF.json", "w"), indent=1)
print(json.dumps(out, indent=1))
