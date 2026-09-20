#!/usr/bin/env python3
"""K2 / K3 / K9 (+ frame and curvature-channel checks) — TANITAD VENV.

K2  GT ROUND TRIP. Logged human futures (the devkit's own
    ``Scene.get_future_trajectory`` = exactly ``HumanAgent``'s output) are cut to
    the knot times refcv4b emits inside 4 s (0.5,1,1.5,2,3,4 s), pushed through the
    SAME ``knots_to_navsim`` the arms use, and compared with the full human poses.
    PASS: every KNOT pose reproduced < 1e-3 m. REPORTED: the 2.5 / 3.5 s
    interpolation error and the heading error (tangent vs logged yaw).
K3  ``taniteval/adapters/navsim.py::verify_frame`` (axis guard + y-sign guard) on
    the same human paths. PASS: status OK with >= 8 turning windows.
K9  NavSim command index order: argmax 0 must go LEFT (+y at 4 s), 2 RIGHT.
KF  FRAME CROSS-IMPLEMENTATION: the synthetic scenes' GLOBAL history poses
    (DataFlyWheel sidecar) mapped to the t0 ego frame by the TanitAD TRAINING
    convention (``refb_labels.ego_frame`` — the function that built refcv4b's GT)
    must equal the devkit's local ``AgentInput`` poses < 1e-3 m.
KC  curvature channel: sign agreement of kappa = ay/max(v,4)^2 with the human
    path's realised curvature over the next 0.5 s (reported, n stated).

Writes raw/K2_K3_K9_roundtrip.json.  ``python tests/test_gt_roundtrip.py``
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "code"))
import tanitad_navsim_bridge as B  # noqa: E402

sys.path.insert(0, os.path.join(B.REPO, "stack", "scripts"))      # refb_labels
RAW = os.path.join(PKG, "raw")
KNOT_IN_4S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
KNOT_IDX = (0, 1, 2, 3, 5, 7)            # indices of those times in NAVSIM_T_S
INTERP_IDX = (4, 6)                      # 2.5 s, 3.5 s


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def run() -> dict:
    z = np.load(os.path.join(RAW, "log_windows_human_future.npz"), allow_pickle=False)
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    s1 = {t: r for t, r in doc["tokens"].items() if r["stage"] == 1}
    human = np.asarray(z["human"], dtype=np.float64)                 # [N,8,3]
    hum_s1 = np.asarray([r["human_future_poses"] for r in s1.values()], dtype=np.float64)
    allh = np.concatenate([human, hum_s1], 0)
    rep = {"n_log_windows": int(human.shape[0]), "n_stage1_scenes": int(hum_s1.shape[0])}

    # ---- K2 -----------------------------------------------------------------
    rec = np.stack([B.knots_to_navsim(h[list(KNOT_IDX), :2], knot_t=KNOT_IN_4S)
                    for h in allh])
    d = np.linalg.norm(rec[:, :, :2] - allh[:, :, :2], axis=-1)       # [N,8]
    dk = d[:, list(KNOT_IDX)]
    di = d[:, list(INTERP_IDX)]
    moving = np.linalg.norm(allh[:, -1, :2], axis=-1) > 2.0
    hd = np.degrees(np.abs(_wrap(rec[:, :, 2] - allh[:, :, 2])))
    rep["K2"] = {
        "knot_err_max_m": float(dk.max()), "knot_err_mean_m": float(dk.mean()),
        "pass_knots_lt_1e-3_m": bool(dk.max() < 1e-3),
        "interp_2p5s_err_m": {"mean": float(di[:, 0].mean()), "p95": float(np.quantile(di[:, 0], .95)),
                              "max": float(di[:, 0].max())},
        "interp_3p5s_err_m": {"mean": float(di[:, 1].mean()), "p95": float(np.quantile(di[:, 1], .95)),
                              "max": float(di[:, 1].max())},
        "heading_err_deg_moving": {"n": int(moving.sum()),
                                   "mean": float(hd[moving].mean()),
                                   "p95": float(np.quantile(hd[moving], .95)),
                                   "max": float(hd[moving].max())},
        "heading_err_deg_all": {"n": int(len(hd)), "mean": float(hd.mean()),
                                "p95": float(np.quantile(hd, .95))},
        "_method": "knots_to_navsim (cubic spline, not-a-knot) on the 6 human poses at "
                   "0.5,1,1.5,2,3,4 s; compared at all 8 NavSim times",
        "_n": int(len(allh))}

    # ---- K3 -----------------------------------------------------------------
    import importlib.util
    ad_path = os.path.join(B.REPO, "taniteval", "adapters", "navsim.py")
    spec = importlib.util.spec_from_file_location("navsim_adapter_e2", ad_path)
    ad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ad)
    speed = np.linalg.norm(np.asarray(z["vel0"]), axis=-1)
    try:
        ev = ad.verify_frame(human[:, :, :2], dt_s=0.5, speed=speed, heading=human[:, :, 2])
        rep["K3"] = {"status": "OK", "axis_convention": ev["axis_convention"],
                     "lateral_sign": ev["lateral_sign"]}
    except Exception as ex:                 # noqa: BLE001 — recorded, never silent
        rep["K3"] = {"status": "RAISED", "error": f"{type(ex).__name__}: {ex}"}

    # ---- K9 -----------------------------------------------------------------
    cmd = np.argmax(np.asarray(z["cmd0"]), -1)
    y4 = human[:, 7, 1]
    turn = np.abs(y4) > 2.0
    tab = {}
    for k in range(4):
        m = (cmd == k)
        tab[str(k)] = {"n": int(m.sum()), "n_turning": int((m & turn).sum()),
                       "frac_y4_pos_among_turning": (float((y4[m & turn] > 0).mean())
                                                     if (m & turn).any() else None)}
    left_ok = tab["0"]["frac_y4_pos_among_turning"]
    right_ok = (None if tab["2"]["frac_y4_pos_among_turning"] is None
                else 1.0 - tab["2"]["frac_y4_pos_among_turning"])
    rep["K9"] = {"by_argmax": tab,
                 "left_is_index0": left_ok, "right_is_index2": right_ok,
                 "pass": bool(left_ok is not None and right_ok is not None and
                              left_ok >= 0.8 and right_ok >= 0.8),
                 "_rule": "turning = |y(4 s)| > 2 m; argmax 0 must turn LEFT (+y), 2 RIGHT"}

    # ---- KF -----------------------------------------------------------------
    import pandas as pd
    import torch
    from refb_labels import ego_frame
    side = pd.read_parquet(os.path.join(B.DEF_BANK if hasattr(B, "DEF_BANK") else
                                        "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus",
                                        "navsim_ego_sidecar.parquet"))
    by_scene = {st: g.sort_values("frame_idx") for st, g in side.groupby("scene_token")}
    errs, herrs, n = [], [], 0
    for tok, r in doc["tokens"].items():
        if r["stage"] != 2:
            continue
        g = by_scene[r["scene_token"]]
        gxy = torch.tensor(g[["ego_x", "ego_y"]].to_numpy(), dtype=torch.float64)
        gh = g["ego_heading"].to_numpy()
        loc = ego_frame(gxy - gxy[-1], torch.tensor(gh[-1], dtype=torch.float64)).numpy()
        dev = np.asarray([e["ego_pose"] for e in r["ego_statuses"]], dtype=np.float64)
        errs.append(np.abs(loc - dev[:, :2]).max())
        herrs.append(np.abs(_wrap((gh - gh[-1]) - dev[:, 2])).max())
        n += 1
    rep["KF"] = {"n_scenes": n, "max_abs_xy_err_m": float(max(errs)),
                 "max_abs_heading_err_rad": float(max(herrs)),
                 "pass_lt_1e-3": bool(max(errs) < 1e-3),
                 "_what": ("TanitAD training GT convention (refb_labels.ego_frame, "
                           "refb_labels.py:86) vs the devkit's local AgentInput poses "
                           "(convert_absolute_to_relative_se2_array) on the same global "
                           "history")}

    # ---- KC -----------------------------------------------------------------
    v = np.linalg.norm(np.asarray(z["vel0"]), axis=-1)
    ay = np.asarray(z["acc0"])[:, 1]
    k_ay = np.clip(ay / np.maximum(v, 4.0) ** 2, -0.12, 0.12)
    h1 = human[:, 0, 2]                                    # yaw change over 0.5 s
    s1d = np.linalg.norm(human[:, 0, :2], axis=-1)
    ok = (v > 4.0) & (s1d > 1.0) & (np.abs(h1) > math.radians(1.0))
    k_real = h1[ok] / s1d[ok]
    rep["KC"] = {"n": int(ok.sum()),
                 "sign_agree_frac": float((np.sign(k_ay[ok]) == np.sign(k_real)).mean()),
                 "pearson_r": float(np.corrcoef(k_ay[ok], k_real)[0, 1]),
                 "median_ratio": float(np.median(k_ay[ok] / k_real)),
                 "_rule": "windows with v0 > 4 m/s, >1 m travelled and >1 deg yaw in 0.5 s"}
    return rep


def test_k2_knots_exact():
    assert run()["K2"]["pass_knots_lt_1e-3_m"]


def test_k3_frame_ok():
    assert run()["K3"]["status"] == "OK"


def test_k9_command_order():
    assert run()["K9"]["pass"]


def test_kf_frame_cross_implementation():
    assert run()["KF"]["pass_lt_1e-3"]


if __name__ == "__main__":
    out = run()
    B.json_dump(out, os.path.join(RAW, "K2_K3_K9_roundtrip.json"))
    print(json.dumps({k: (v if k in ("n_log_windows", "n_stage1_scenes") else
                          {kk: vv for kk, vv in v.items() if not kk.startswith("_")})
                      for k, v in out.items()}, indent=1, default=str)[:6000])
