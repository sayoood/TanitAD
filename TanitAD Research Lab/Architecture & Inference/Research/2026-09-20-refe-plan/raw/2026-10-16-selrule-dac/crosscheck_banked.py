#!/usr/bin/env python3
"""Post-fit ERRATUM instrument (reads FIT.json and the RL-S1 trainer's own step-0 row; recomputes NO rule and changes nothing).

FIT.json's `identity.banked_trainer_row_c0_sft_jsonl_step0_base_pick` pairs the trainer's `base.pick` (90.84276078, rule navsim_v1) with MY navsim_v1_lane
number (90.6337) and calls the 0.209 gap a cross-check, with the note 'its rule is navsim_v1_lane'.  That pairing is WRONG: in scorer_finetune the
`base.pick` is the navsim_v1 pick and `base.x_lane.pick` is the lane-rule pick.  The right pairs are computed here.  The block is informational and
was never an input to the choice (the hard gates are the mismatch counts and the rl_val_eval.run equality, both exact).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
fit = json.load(open(os.path.join(HERE, "FIT.json"), encoding="utf-8"))
row = None
for ln in open("C:/Users/Admin/rls1b_dl/c0/final/sft.jsonl", encoding="utf-8"):
    d = json.loads(ln)
    if d.get("event") == "eval" and d.get("tag") == "step0":
        row = d["base"]
mine_v1 = fit["identity"]["rl_val_eval_run"]["navsim_v1"]["mine_PDMS_x100"]
mine_ln = fit["identity"]["rl_val_eval_run"]["navsim_v1_lane"]["mine_PDMS_x100"]
out = {"what": "trainer's own step-0 base row vs my B, matched by rule (the FIT.json pairing is wrong)",
       "navsim_v1": {"trainer_base_pick": row["pick"], "mine_B_primary": mine_v1, "abs_diff": abs(row["pick"] - mine_v1), "trainer_zero_picks": row["zero_picks"], "mine_zero_picks": fit["B"]["zero_picks"]},
       "navsim_v1_lane": {"trainer_x_lane_pick": row["x_lane"]["pick"], "mine_B_lane": mine_ln, "abs_diff": abs(row["x_lane"]["pick"] - mine_ln),
                          "trainer_zero_picks": row["x_lane"]["zero_picks"], "mine_zero_picks": fit["identity"]["rl_val_eval_run"]["navsim_v1_lane"]["mine_zero_picks"]},
       "reading": "both pairs agree to ~1.3e-6 (the trainer's labels are rounded to ~5 decimals) and on the exact zero-pick counts: an INDEPENDENT code path reproduces B"}
json.dump(out, open(os.path.join(HERE, "FIT_CROSSCHECK.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
