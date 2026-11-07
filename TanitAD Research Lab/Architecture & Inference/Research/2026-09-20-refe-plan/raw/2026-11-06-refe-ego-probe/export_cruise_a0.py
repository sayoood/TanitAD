"""Per-token REFe u400 outputs on the 154 CRUISE tokens (A0 and GREY), for TanitScenarioFlyWheel's nuPlan round-trip comparison -> cruise_a0_per_token.json."""
import json, sys, numpy as np
sys.path.insert(0, "."); from analyze_ego_probe import gt_speed, step_speed
Z = np.load("full/results.npz", allow_pickle=False); arms = [str(x) for x in Z["arms"]]; ai = {x: i for i, x in enumerate(arms)}
ego = Z["ego_real"]; v0, ax0 = ego[:, 6], ego[:, 2]; gv = Z["gt_valid"].astype(bool)
vgt = np.where(gv[:, None], gt_speed(Z["gt_pose20"], Z["gt_t20"]), np.nan)
cruise = gv & (v0 > 5.0) & (np.abs(ax0) < 0.3) & (np.abs(vgt[:, 2] - v0) < 1.0)
P = Z["traj_pick"].astype(np.float64); vA, vG = step_speed(P[:, ai["A0"]]), step_speed(P[:, ai["GREY"]])
rows = []
for i in np.where(cruise)[0]:
    rows.append({"token": str(Z["token"][i]), "log": str(Z["log"][i]), "v0": round(float(v0[i]), 4), "ax0": round(float(ax0[i]), 4),
                 "A0_pick": int(Z["pick"][i, ai["A0"]]), "A0_v_1_2_4s": [round(float(x), 4) for x in vA[i]], "A0_v4_over_v0": round(float(vA[i, 2] / v0[i]), 4),
                 "GREY_v4_over_v0": round(float(vG[i, 2] / v0[i]), 4), "GT_v4": round(float(vgt[i, 2]), 4),
                 "A0_traj20_xyh": np.round(P[i, ai["A0"]], 4).tolist()})
meta = {"model": "model_sft4_B_u400.pt sha256 8b4dba4818b5cc96...", "tier": "T0", "speed_estimator": "analyze_ego_probe.step_speed (one-step chord of the 20 poses at 0.2 s)",
        "cruise_mask": "v0 > 5, |ax0| < 0.3, |GT v4 - v0| < 1 (analyze_ego_probe.py:96)", "n": len(rows),
        "frames": "4 cams CAM_F0/L0/R0/B0, 1920x1080 JPEG, model input = full-frame resize to 960x512 RGB /255 (refe/train.py:832)"}
json.dump({"meta": meta, "tokens": rows}, open("cruise_a0_per_token.json", "w"), indent=0)
print(len(rows), "mean A0 v4/v0", round(float(np.mean([r["A0_v4_over_v0"] for r in rows])), 4), "logs", len({r["log"] for r in rows}))
