#!/usr/bin/env python3
"""E8: how far apart do training seeds 0 and 1 of H land?  On a SEMI-SYNTHETIC world (REAL tokens, SYNTHETIC labels) with the chosen recipe.  The head starts from P2's EP head and both losses are convex
in its 257 parameters, so the seed changes only the minibatch order; this measures how much that is."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import synthetic  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts/l2_offprop"
OUT = "C:/Users/Admin/par18/l2/out"


def main():
    import torch
    torch.set_num_threads(2)
    rec = l2lib.Recipe.from_json(f"{ART}/recipe_chosen.json")
    real = l2lib.load_real(f"{ART}/feats.npz")
    res = {"recipe": rec.as_dict(), "worlds": {}}
    for name, kw in (("sigA", dict(beta=0.5, alpha=0.25, seed=21)), ("null", dict(beta=0.0, alpha=0.4, seed=23))):
        D = synthetic.semi_synthetic(real, **kw)
        o = [l2lib.oof_scores(D, l2lib.ArmSpec(f"H{s}", "H", seed=s), rec) for s in (0, 1)]
        o0 = [l2lib.oof_scores(D, l2lib.ArmSpec(f"H0_{s}", "H0", seed=s), rec) for s in (0, 1)]
        ev = [l2lib.evaluate_arm(D, x["ep_in"], x["ep_p"]) for x in o]
        ev0 = [l2lib.evaluate_arm(D, x["ep_in"], x["ep_p"]) for x in o0]
        d_in, d_p = np.abs(o[0]["ep_in"] - o[1]["ep_in"]), np.abs(o[0]["ep_p"] - o[1]["ep_p"])
        pair_sign = lambda x: np.sign(x["ep_p"][np.arange(D.N)[:, None], np.arange(D.Q)[None, :]] - np.take_along_axis(x["ep_in"], D.anchor, 1))
        flips = float((pair_sign(o[0]) != pair_sign(o[1])).mean())
        res["worlds"][name] = {"A1_seed0": ev[0]["all"]["A1"]["point"], "A1_seed1": ev[1]["all"]["A1"]["point"], "A1_abs_diff": abs(ev[0]["all"]["A1"]["point"] - ev[1]["all"]["A1"]["point"]),
                               "infan_seed0": ev[0]["infan_top8_ep_concordance"]["point"], "infan_seed1": ev[1]["infan_top8_ep_concordance"]["point"],
                               "infan_h0_seed0": ev0[0]["infan_top8_ep_concordance"]["point"], "infan_h0_seed1": ev0[1]["infan_top8_ep_concordance"]["point"],
                               "infan_h_minus_h0_seed0": ev[0]["infan_top8_ep_concordance"]["point"] - ev0[0]["infan_top8_ep_concordance"]["point"],
                               "infan_h_minus_h0_seed1": ev[1]["infan_top8_ep_concordance"]["point"] - ev0[1]["infan_top8_ep_concordance"]["point"],
                               "max_abs_ep_score_diff_infan": float(d_in.max()), "mean_abs_ep_score_diff_infan": float(d_in.mean()), "max_abs_ep_score_diff_pert": float(d_p.max()),
                               "ep_score_std_infan": float(o[0]["ep_in"].std()), "share_of_pairs_whose_sign_differs_between_seeds": flips}
    json.dump(l2lib.jsonable(res), open(f"{OUT}/seed_spread.json", "w"), indent=1)
    print("ZZSEEDSPREAD_OK", json.dumps(l2lib.jsonable({k: v["A1_abs_diff"] for k, v in res["worlds"].items()})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
