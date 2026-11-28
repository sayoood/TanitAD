"""per-token comparison: the control's harness csv (P2's banked seam re-scored on today's layout) vs P2's banked csv on the same 24 tokens; every column must be EXACTLY equal."""
import csv
import json

C = "E:/Projects/TanitAD-artifacts/p6read_layout_fix/harness_control"
NEW = f"{C}/scoreB/refe_layoutfix_ctlB/refe_layoutfix_ctlB.csv"
OLD = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv"
toks = json.load(open(f"{C}/tokens_b.json", encoding="utf-8"))["tokens"]
rd = lambda p: {r["token"]: r for r in csv.DictReader(open(p, encoding="utf-8", newline=""))}   # noqa: E731
new, old = rd(NEW), rd(OLD)
cols = ["valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]
bad = [(t, c, new[t][c], old[t][c]) for t in toks for c in cols if t in new and t in old and float(new[t][c] if c != "valid" else new[t][c] == "True") != float(old[t][c] if c != "valid" else old[t][c] == "True")]
miss = [t for t in toks if t not in new or t not in old]
n_tok_rows = sum(1 for t in new if t != "average")
res = {"n_tokens": len(toks), "rows_in_new_csv": n_tok_rows, "missing": miss, "n_cells_compared": len(toks) * len(cols), "n_cells_unequal": len(bad), "unequal_examples": bad[:5],
       "mean_pdms_x100_new": 100 * sum(float(new[t]["score"]) for t in toks if t in new) / max(1, len(toks)),
       "mean_pdms_x100_banked_p2": 100 * sum(float(old[t]["score"]) for t in toks if t in old) / max(1, len(toks)),
       "string_equal_score_cells": sum(1 for t in toks if t in new and t in old and new[t]["score"] == old[t]["score"])}
res["IDENTICAL"] = bool(not miss and not bad and n_tok_rows == len(toks))
print(json.dumps(res, indent=1))
print("ZZLAYOUT_CTL", "IDENTICAL" if res["IDENTICAL"] else "DIFFERENT", f"{len(toks)} tokens x {len(cols)} columns")
