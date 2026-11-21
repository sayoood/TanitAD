"""P1: does ANY derivable signal separate a real lane change from a false one?

Feature battery over the GATED lane-change event window, joined to the PI's
n=18 adjudication. Exact permutation tests, effect sizes, overlap. No tuning.
"""
import json
import math
import os
import sys
from itertools import combinations

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
EGO = os.path.join(SP, "s2_ego", "aug120")
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import refb_labels as rl                                      # noqa: E402
import s2_derive                                              # noqa: E402

T0_IDX = 80
DT = 0.1
H = rl.LAT_HORIZON_STEPS          # 40 steps = 4.0 s
STRIDE = 10                        # engine_a_summary stride


def unwrap(a):
    return np.unwrap(np.asarray(a, dtype=np.float64))


def win_features(poses_np, t):
    """Every candidate discriminator over the 4 s window starting at index t."""
    T = poses_np.shape[0]
    h = min(H, T - 1 - t)
    seg = poses_np[t:t + h + 1]
    x, y, yaw, v = seg[:, 0], seg[:, 1], unwrap(seg[:, 2]), seg[:, 3]

    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])          # arc length
    L = float(s[-1])
    net_yaw = float(yaw[-1] - yaw[0])

    # ego-frame lateral profile in the WINDOW-START frame (what latmaneuver uses)
    c, sn = math.cos(yaw[0]), math.sin(yaw[0])
    dx, dy = x - x[0], y - y[0]
    lon_p = c * dx + sn * dy
    lat_p = -sn * dx + c * dy
    lat_f = float(lat_p[-1])
    peak = float(np.abs(lat_p).max())

    f = {}
    f["L_m"] = L
    f["v_mean"] = float(v.mean())
    f["net_yaw"] = net_yaw
    f["abs_net_yaw"] = abs(net_yaw)
    f["lat_f"] = lat_f
    f["abs_lat_f"] = abs(lat_f)
    f["peak_lat"] = peak
    f["ret_frac"] = abs(lat_f) / peak if peak > 1e-6 else 0.0

    # --- dimensionless lateral rate; a constant-curvature arc gives net_yaw/2
    f["lat_over_L"] = lat_f / L if L > 1e-6 else 0.0
    # --- ARC RESIDUAL: subtract the lateral offset a pure circular arc with the
    #     SAME net heading change and the SAME arc length would have produced.
    if abs(net_yaw) > 1e-6:
        R = L / net_yaw
        lat_arc = R * (1.0 - math.cos(net_yaw))
    else:
        lat_arc = 0.0
    f["lat_arc_pred"] = lat_arc
    f["lat_resid_arc"] = lat_f - lat_arc
    f["abs_lat_resid_arc"] = abs(lat_f - lat_arc)
    f["resid_frac"] = ((lat_f - lat_arc) / lat_f) if abs(lat_f) > 1e-6 else 0.0

    # --- curvature over the window (NOT the whole-clip peak_kappa_per_m)
    ds = np.diff(s)
    ok = ds > 1e-3
    kap = np.zeros_like(ds)
    kap[ok] = np.diff(yaw)[ok] / ds[ok]
    f["kap_mean_abs"] = float(np.abs(kap).mean())
    f["kap_peak_abs"] = float(np.abs(kap).max()) if kap.size else 0.0

    # --- BIDIRECTIONALITY: a lane change MUST steer out and back, so the
    #     signed curvature integral has comparable positive and negative area.
    #     A curve has one sign only.
    pos = float(np.sum(np.clip(kap, 0, None) * ds))
    neg = float(-np.sum(np.clip(kap, None, 0) * ds))
    f["yaw_pos_area"] = pos
    f["yaw_neg_area"] = neg
    f["bidir"] = (min(pos, neg) / max(pos, neg)) if max(pos, neg) > 1e-9 else 0.0
    f["yaw_swing"] = pos + neg                       # total |heading| travelled
    f["swing_over_net"] = ((pos + neg) / abs(net_yaw)
                           if abs(net_yaw) > 1e-6 else float("inf"))

    # --- yaw DETREND: residual of yaw(s) vs the best constant-curvature line.
    if L > 1e-6 and s.size >= 3:
        A = np.vstack([s, np.ones_like(s)]).T
        coef, *_ = np.linalg.lstsq(A, yaw, rcond=None)
        res = yaw - A @ coef
        f["yaw_detrend_rms"] = float(np.sqrt((res ** 2).mean()))
        f["yaw_detrend_peak"] = float(np.abs(res).max())
        f["kap_fit"] = float(coef[0])
        # R^2 of the constant-curvature model on yaw(s): a CURVE fits well.
        ss_t = float(((yaw - yaw.mean()) ** 2).sum())
        f["yaw_lin_r2"] = 1.0 - float((res ** 2).sum()) / ss_t if ss_t > 1e-12 \
            else 1.0
    else:
        f["yaw_detrend_rms"] = f["yaw_detrend_peak"] = 0.0
        f["kap_fit"] = 0.0
        f["yaw_lin_r2"] = 1.0

    # --- STEP vs RAMP on the lateral profile.
    #     ramp  : lat = a*s^2      (constant-curvature arc, zero initial lat vel)
    #     step  : lat = A * logistic((s - s0)/w)  -> approximated by the best
    #             3-param logistic via coarse grid + least squares on A
    ss_t = float(((lat_p - lat_p.mean()) ** 2).sum())
    A2 = (s ** 2).reshape(-1, 1)
    c2, *_ = np.linalg.lstsq(A2, lat_p, rcond=None)
    r_ramp = lat_p - A2 @ c2
    f["ramp_sse"] = float((r_ramp ** 2).sum())
    best = (float("inf"), None)
    if L > 1e-6:
        for s0 in np.linspace(0.15 * L, 0.85 * L, 15):
            for w in np.linspace(0.04 * L, 0.35 * L, 12):
                g = 1.0 / (1.0 + np.exp(-(s - s0) / w))
                g = g - g[0]
                den = float((g * g).sum())
                if den < 1e-12:
                    continue
                a = float((g * lat_p).sum()) / den
                sse = float(((lat_p - a * g) ** 2).sum())
                if sse < best[0]:
                    best = (sse, (a, s0, w))
    f["step_sse"] = best[0]
    f["step_amp"] = best[1][0] if best[1] else 0.0
    f["ramp_r2"] = 1.0 - f["ramp_sse"] / ss_t if ss_t > 1e-12 else 1.0
    f["step_r2"] = 1.0 - f["step_sse"] / ss_t if ss_t > 1e-12 else 1.0
    f["step_minus_ramp_r2"] = f["step_r2"] - f["ramp_r2"]
    f["log_sse_ratio"] = math.log10((f["ramp_sse"] + 1e-9)
                                    / (f["step_sse"] + 1e-9))

    # --- lateral kinematics
    latv = np.gradient(lat_p, DT)
    f["lat_v_peak"] = float(np.abs(latv).max())
    f["lat_a_peak"] = float(np.abs(np.gradient(latv, DT)).max())
    f["yawrate_peak"] = float(np.abs(np.gradient(yaw, DT)).max())
    return f


def load(cid):
    z = np.load(os.path.join(EGO, f"{cid}.npz"))
    return np.asarray(z["poses"], dtype=np.float64)


# --------------------------------------------------------------------------- #
def main():
    ea_rows = [json.loads(l) for l in open(
        os.path.join(PKG, "labels", "engine_a_aug120.jsonl"),
        encoding="utf-8") if l.strip()]
    verd = json.load(open(os.path.join(PKG, "review",
                                       "PI_VERDICTS_2026-08-16.json"),
                          encoding="utf-8"))["verdicts"]

    out = []
    for r in ea_rows:
        cid, ea = r["clip_id"], r["engine_a"]
        ev = s2_derive._gated_lc_event(ea)
        rec = {"clip_id": cid, "gated": ev is not None}
        if ev is not None:
            rec["ev"] = ev
            poses = load(cid)
            t = T0_IDX + int(round(float(ev["t_start_s"]) / DT))
            rec["t_ev"] = t
            rec["feat"] = win_features(poses, t)
            # sanity: our lat_f must reproduce the banked event lat_m
            rec["lat_check"] = round(rec["feat"]["lat_f"] - float(ev["lat_m"]), 3)
        v = verd.get(cid)
        if v:
            note = (v.get("note") or "").lower()
            gt = v.get("v")
            if gt is None and "wrong" in note:
                gt = "wrong"          # 4bea0a51: note says wrong, v stayed null
            rec["pi"] = gt
            rec["pi_note"] = v.get("note")
            rec["pi_section"] = v.get("section")
        out.append(rec)

    json.dump(out, open(os.path.join(SP, "lc_feat.json"), "w",
                        encoding="utf-8"), indent=1)
    g = [r for r in out if r["gated"]]
    print(f"gated {len(g)}/{len(out)}")
    print("max |lat_f - banked lat_m| =",
          max(abs(r["lat_check"]) for r in g))
    lab = [r for r in g if r.get("pi") in ("wrong", "correct")]
    print(f"gated AND adjudicated: {len(lab)}  "
          f"wrong={sum(1 for r in lab if r['pi']=='wrong')}  "
          f"correct={sum(1 for r in lab if r['pi']=='correct')}")
    s1 = [c for c, v in verd.items() if v["section"] == "S1_LANE_TARGET"]
    gset = {r["clip_id"] for r in g}
    print(f"S1_LANE_TARGET rows: {len(s1)}; "
          f"in gated set: {len(set(s1) & gset)}; "
          f"gated but NOT reviewed: {sorted(gset - set(s1))}")


if __name__ == "__main__":
    main()
