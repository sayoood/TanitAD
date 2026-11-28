"""DRY-RUN CHECK (before any C3 proposal was scored): the BANKED u400 picks restricted to the 1,058-token C3 population, scored through the
subset harness (score_navtest_refe.py, label refe_c3_u400_dry_pick), must reproduce the banked full-navtest per-token score and sub-scores.
Establishes (i) the Hydra --tokens command line fits Windows' limit at 1,058 tokens, (ii) a 1,058-token subset run is score-identical to the 12,146-token
run, i.e. the pipeline's G2 gate is attainable. Not a C3 read: it scores only the banked picks."""
import csv, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
def rd(p): return {r["token"]: (float(r["score"]), [float(r[c]) for c in SUB]) for r in csv.DictReader(open(p, encoding="utf-8")) if r["token"] != "average"}
B = rd("D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv")
R = rd("D:/Projects/TanitAD/data/refe_navtest/score/refe_c3_u400_dry_pick/refe_c3_u400_dry_pick.csv")
toks = json.load(open(os.path.join(HERE, "tokens.json")))["tokens"]
d = np.array([abs(R[t][0] - B[t][0]) for t in toks]); s = np.array([np.abs(np.array(R[t][1]) - np.array(B[t][1])).max() for t in toks])
out = {"n_tokens": len(toks), "n_scored": len(R), "max_abs_score_diff": float(d.max()), "n_score_diff_gt_0": int((d > 0).sum()),
       "max_abs_subscore_diff": float(s.max()), "n_zero_score_tokens": int(sum(1 for t in toks if R[t][0] == 0)),
       "harness_counts": json.load(open("D:/Projects/TanitAD/data/refe_navtest/score/refe_c3_u400_dry_pick/refe_c3_u400_dry_pick.counts.json"))["wall_s"]}
json.dump(out, open(os.path.join(HERE, "dry_logs", "dry_check.json"), "w"), indent=1)
print(out)
