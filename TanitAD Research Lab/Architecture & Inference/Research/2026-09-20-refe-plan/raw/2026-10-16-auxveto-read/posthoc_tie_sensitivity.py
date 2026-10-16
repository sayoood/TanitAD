#!/usr/bin/env python3
"""posthoc_tie_sensitivity.py -- POST-HOC, NOT IN THE PREREG, DECIDES NOTHING.

The prereg's rule is the STRICT `signed distance < -0.25 m`.  On the 0.5 m GT grid a one-cell-wide non-drivable sliver has signed distance EXACTLY -0.25 at its cells, so some
proposals sit exactly on the threshold (199 proposals with margin == -0.25 in float, on 32 tokens).  This reports how much V-GT's primary Delta moves if those ties were VETOED
(`<=`) instead of kept.  V-PRED has no tie (0 proposals within 1e-6 of -0.25).  AUXVETO_READ.json is the strict-rule read and is NOT changed by this file.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auxveto_read as av  # noqa: E402
import geodac as G  # noqa: E402

av.assert_prereg()
w, _ = av.load_world()
ix, counts = av._log_counts(w)
gt = av.load_gt_masks(w)
pm, _ = av.load_pred_masks(w, av.UNIT_DIRS["V-PRED"])
out = {"note": "POST-HOC tie sensitivity; decides nothing; AUXVETO_READ.json (strict <) is unchanged"}
for arm, mg in (("V-GT", av.margins_from_masks(w, gt, 0.5, G.GT_GEOM)), ("V-PRED", av.margins_from_masks(w, pm, 2.0, G.PRED_GEOM))):
    ties = np.abs(mg + av.MARGIN) < 1e-9
    out[arm] = {"n_ties_within_1e-9": int(ties.sum()), "tokens_with_a_tie": int(ties.any(axis=1).sum())}
    for name, veto in (("strict_lt (prereg)", av.vetoed(mg, av.MARGIN)), ("ties_vetoed (<=, POST-HOC)", mg <= -av.MARGIN + 1e-9)):
        bl = av.arm_block(w, veto, av.all_picks(w, veto), ix, counts)
        out[arm][name] = {"d_pdms": bl["d_pdms"], "z_recovery_ncdac": bl["z_recovery"]["ncdac_count"], "false_veto_N": bl["false_veto_N"]["b_pick_vetoed_count"]}
json.dump(av.cj(out), open(HERE / "POSTHOC_TIE_SENSITIVITY.json", "w"), indent=1)
for arm in ("V-GT", "V-PRED"):
    for k, v in out[arm].items():
        if isinstance(v, dict) and "d_pdms" in v:
            d = v["d_pdms"]
            print(f"{arm:7s} {k:30s} dPDMS {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  Z rec {v['z_recovery_ncdac']}  false-veto {v['false_veto_N']}")
        else:
            print(arm, k, v)
