#!/usr/bin/env python3
"""The margin of `test_semi_synthetic_hits_its_two_calibration_targets_and_reads_no_real_label` (MM addendum: 'if it still fails, report the margin; do not loosen the test'), under the SELECTED recipe and under
the chain's superseded one.  The test asserts  A1(H) > A1(P2) + 0.02  on a planted-signal semi-synthetic world (beta = 3).  SYNTHETIC only; reads no real label."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import synthetic  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts/l2_offprop"
OUT = "C:/Users/Admin/par18/l2/out"


def margin(rec):
    base = synthetic.make_data(n_tokens=72, n_logs=24, seed=9, gamma=0.8, rho=0.8)
    S2 = synthetic.semi_synthetic(base, beta=3.0, alpha=0.4, seed=1)
    p2 = l2lib.oof_scores(S2, l2lib.ArmSpec("P2", "P2"), rec)
    h = l2lib.oof_scores(S2, l2lib.ArmSpec("H", "H", seed=0), rec)
    a2 = l2lib.evaluate_arm(S2, p2["ep_in"], p2["ep_p"])["all"]["A1"]["point"]
    ah = l2lib.evaluate_arm(S2, h["ep_in"], h["ep_p"])["all"]["A1"]["point"]
    return {"a1_p2": a2, "a1_h": ah, "gain": ah - a2, "required_gain_strictly_above": 0.02, "margin_over_bar": ah - a2 - 0.02, "passes_as_written": bool(ah > a2 + 0.02)}


def main():
    import torch
    torch.set_num_threads(2)
    res = {"selected_recipe": margin(l2lib.Recipe.from_json(f"{ART}/recipe_chosen.json"))}
    sup = f"{OUT}/superseded/recipe_chosen_chain_lam10_lr3e-4_ep5.json"
    if os.path.exists(sup):
        res["superseded_chain_recipe_lam10_lr3e-4_ep5"] = margin(l2lib.Recipe.from_json(sup))
    res["source_default_REC_epochs12_used_by_the_other_tests"] = margin(l2lib.Recipe(epochs=12))
    json.dump(l2lib.jsonable(res), open(f"{OUT}/planted_margin.json", "w"), indent=1)
    print("ZZMARGIN_OK", json.dumps(l2lib.jsonable({k: v["margin_over_bar"] for k, v in res.items()})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
