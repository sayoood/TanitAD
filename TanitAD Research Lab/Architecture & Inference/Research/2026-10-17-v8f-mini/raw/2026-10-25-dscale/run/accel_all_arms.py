"""Zero-GPU: accel / speed MAE of every banked seed-0 arm against M2b_s0 and M2b_s1 (per-arm values from each eval JSON). EXPLORATORY."""
import json
L = "C:/Users/Admin/box2_ship/logs/"
A = {"B0": "v8f_mini/eval/eval_M2b_s0", "B1": "v8f_mini/eval/eval_M2b_s1", "G0": "v8f_grid/eval/eval_M2b-8x20_s0", "H0": "v8f_hist/eval/eval_M2b-h10_s0",
     "P0": "v8f_pgt/eval/eval_M2b-pgt_s0", "W0": "v8f_wc/eval/eval_M2b-wc_s0", "Q0": "v8f_dscale/eval/eval_M2b-df50_s0"}
J = {k: json.load(open(L + v + ".json", encoding="utf-8"))["raw"] for k, v in A.items()}
for m in ("accel_mae_mps2", "speed_mae_mps"):
    print(m, {k: round(J[k]["families"]["longitudinal"][m], 4) for k in J})
    print("   minus B0:", {k: round(J[k]["families"]["longitudinal"][m] - J["B0"]["families"]["longitudinal"][m], 4) for k in J if k != "B0"})
t = J["B0"].get("tactical", {}); print("tactical keys", list(t)[:12]); s = J["B0"].get("strategic", {}); print("strategic keys", list(s)[:12])
for k in ("B0", "Q0"):
    tl = J[k].get("tactical", {})
    print(k, "lat macroF1", (tl.get("lat") or {}).get("macro_f1"), "lon macroF1", (tl.get("lon") or {}).get("macro_f1"))
    print(k, "strategic", json.dumps(J[k].get("strategic"))[:300])
