"""compares SMOKE_ZPOST.json with SMOKE_ZPOST_MUTANT.json (labels shifted by one window): the decoded-plan and head accuracies must differ, else the mutant is NOT detected"""
import json
a = json.load(open("test/SMOKE_ZPOST.json"))["arms"]["SYNTH_s0"]
b = json.load(open("test/SMOKE_ZPOST_MUTANT.json"))["arms"]["SYNTH_s0"]
d = abs(a["decoded_plan_acc_T0"]["point"] - b["decoded_plan_acc_T0"]["point"])
rec = {"what": "Z-POST mutant: labels shifted one window; the tick-0 decoded-plan accuracy of the same windows must move", "clean_decoded_T0": a["decoded_plan_acc_T0"]["point"], "mutant_decoded_T0": b["decoded_plan_acc_T0"]["point"],
       "abs_diff": d, "clean_head_T0": a["route_head_acc_T0"]["point"], "mutant_head_T0": b["route_head_acc_T0"]["point"], "detected": bool(d > 0.05),
       "fixture": "SYNTHETIC route-head checkpoint (N1DC_s0 + seeded-random route_head / route_emb), eval-141 inputs, 96 windows: machinery smoke, NOT a result"}
json.dump(rec, open("test/MUTANT_RECORD_ZPOST.json", "w"), indent=1)
print(rec)
