"""D3 check 2b -- detail of the clips c2_poses.py flagged on the rules that are NOT explained by highway speed."""
import sys, json, hashlib
sys.path.insert(0, 'D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility/scripts')
import numpy as np, torch
import c2_poses as C
side = C.load_sidecar()
out = {}
for name, path in (("eval139", C.EVAL_MAN), ("train4369", C.TRAIN_MAN)):
    man = torch.load(path, map_location="cpu", weights_only=False)
    for i, cid in enumerate(man["clip_id"]):
        P = man["poses"][i].numpy().astype(np.float64)
        sc = side.get(int(man["episode_uid"][i])); dt = float(sc["dt_s"]) if sc else C.DT_NOM
        r = C.check_clip(P, man["actions"][i].numpy()[:, 1], dt)
        for k in ("jump_speedcons", "acc", "head", "rev", "jerk_sustained"):
            if r[k].any():
                idx = np.where(r[k])[0]
                # contiguous segments
                segs = []; s = idx[0]; p = idx[0]
                for q in idx[1:]:
                    if q != p + 1: segs.append((int(s), int(p))); s = q
                    p = q
                segs.append((int(s), int(p)))
                rec = {"split": name, "clip": C.sha12(cid), "rule": k, "n_rows": int(len(idx)), "segments_rows(start,end)": segs[:6],
                       "v_in_rows_min_max": [round(float(r["vbar"][idx].min()), 2), round(float(r["vbar"][idx].max()), 2)],
                       "d_in_rows_max": round(float(r["d"][idx].max()), 2)}
                if k == "jump_speedcons":
                    rec["d_minus_vdt"] = [round(float((r["d"] - r["vbar"] * dt)[j]), 2) for j in idx[:5]]
                if k == "acc":
                    rec["a_values"] = [round(float(r["a"][j]), 1) for j in idx[:5]]
                    rec["prov_accel_at_rows"] = [round(float(man["actions"][i][j + 1, 1]), 2) for j in idx[:5]]
                if k in ("rev", "head"):
                    rec["mismatch_deg_median_in_rows"] = round(float(np.median(r["mis"][idx])), 1)
                    # net displacement along heading over the segment: negative => truly backwards
                    a, b = segs[0]
                    yaw = P[a, 2]
                    disp = (P[b + 1, 0] - P[a, 0]) * np.cos(yaw) + (P[b + 1, 1] - P[a, 1]) * np.sin(yaw)
                    rec["seg0_net_disp_along_heading_m"] = round(float(disp), 2)
                    rec["seg0_speed_profile"] = [round(float(x), 2) for x in P[a:b + 2:max(1, (b - a) // 6 + 1), 3]]
                out.setdefault(k, []).append(rec)
json.dump(out, open('raw/c2b_anomaly_detail.json', 'w'), indent=1)
for k, v in out.items():
    print('==', k, len(v))
    for x in v[:20]: print('  ', json.dumps(x))
