"""D1 analysis: reads the census tables (windows_<tag>.npz + clips_<tag>.json + record_<tag>.json) and writes
numbers_<tag>.json plus the COVERAGE tables. numpy only; runs on the dev box.

Every threshold below is written as a LITERAL from road geometry / the briefing (30 deg turn, 6 s and 10 s
horizons, the 4-step km/h ladder, the 6.0-10.0 s band) -- NOT imported from the builder or the trainer --
so a control that compares against the trainer's own column is an independent check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

IGN = -100
TURN_DEG = 30.0                      # literal (briefing + route-following SPEC sec. 3)
LADDER_KMH = np.array([30.0, 50.0, 100.0, 120.0])
LADDER_MS = LADDER_KMH / 3.6


def pct(a, b):
    return float("nan") if b == 0 else 100.0 * a / b


def load(d: Path, tag: str):
    z = np.load(d / f"windows_{tag}.npz")
    cols = {k: z[k] for k in z.files}
    clips = json.load(open(d / f"clips_{tag}.json", encoding="utf-8"))
    rec = json.load(open(d / f"record_{tag}.json", encoding="utf-8"))
    return cols, clips, rec


def real_bin(v_ms):
    """realised max speed (m/s) -> containing-window bin index of the 30/50/100/120 ladder (3 = clamp)."""
    v = np.asarray(v_ms, np.float64)
    b = np.full(v.shape, 3, np.int8)
    for i in range(3, -1, -1):
        b = np.where(v <= LADDER_MS[i] + 1e-9, i, b)
    return b


def cluster_boot(num, den, clip, B=2000, seed=0):
    """episode(clip)-cluster bootstrap of sum(num)/sum(den); num, den are per-window arrays."""
    ids = np.unique(clip)
    n_i = np.bincount(np.searchsorted(ids, clip), weights=num.astype(np.float64), minlength=len(ids))
    d_i = np.bincount(np.searchsorted(ids, clip), weights=den.astype(np.float64), minlength=len(ids))
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(ids), len(ids))
        dd = d_i[k].sum()
        out[b] = 100.0 * n_i[k].sum() / dd if dd > 0 else np.nan
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def analyse(d: Path, tag: str, boot=False):
    cols, clips, rec = load(d, tag)
    V = rec["vocab"]
    N = len(cols["t"])
    nclip = len(clips)
    c = cols
    clip = c["clip_ix"]
    tn = c["t_now_s"].astype(np.float64)
    out = {"tag": tag, "n_windows": int(N), "n_clips": int(nclip)}

    # ---------------- literal in-band control (c): a record anchored at 8.0 s -> in band iff 6.0 <= t_now <= 10.0
    t0s = {cl["t0_s"] for cl in clips}
    lit = (tn >= 6.0) & (tn <= 10.0) & c["has_record"]
    mism = int((lit != c["in_band"]).sum())
    near = np.abs(np.abs(tn - 8.0) - 2.0) < 1e-6
    out["control_in_band_literal"] = {
        "distinct_t0_s": sorted(float(x) for x in t0s if x is not None),
        "n_windows_literal_in_band": int(lit.sum()), "n_windows_trainer_in_band": int(c["in_band"].sum()),
        "n_mismatch": mism, "n_windows_within_1e-6_of_a_band_edge": int(near.sum()),
        "t_now_min_in_band": float(tn[c["in_band"]].min()) if c["in_band"].any() else None,
        "t_now_max_in_band": float(tn[c["in_band"]].max()) if c["in_band"].any() else None,
        "t_now_max_outside_below": float(tn[(~c["in_band"]) & (tn < 8)].max()),
        "t_now_min_outside_above": float(tn[(~c["in_band"]) & (tn > 8)].min()),
        "lat_ignore_iff_not_in_band": bool(((c["lat_v7"] == IGN) == (~c["in_band"])).all()),
        "lon_ignore_iff_not_in_band": bool(((c["lon_v7"] == IGN) == (~c["in_band"])).all())}

    # ---------------- T1 channel coverage ------------------------------------------------------------
    nv = c["nav_valid"]
    rows = {
        "windows": N,
        "joined to a v8 record": int(c["has_record"].sum()),
        "tactical lat/lon GT (in band)": int((c["lat_v7"] != IGN).sum()),
        ">=1 goal token SCORED (w>0), as the workers saw it": int((c["n_goal_scored"] > 0).sum()),
        ">=1 goal token POSITIVE": int((c["n_goal_pos"] > 0).sum()),
        "nav token fed (nav_valid)": int(nv.sum()),
        "nav token = left or right": int(((c["nav_cmd"] == 1) | (c["nav_cmd"] == 2)).sum()),
        "max-speed ceiling fed": int(c["vmax_valid"].sum()),
        "agent boxes labelled": int(c["agent_labelled"].sum()),
        "agent labelled AND >=1 agent": int((c["agent_labelled"] & (c["n_agents"] > 0)).sum()),
        "agent labelled and EMPTY (clear)": int((c["agent_labelled"] & (c["n_agents"] == 0)).sum()),
        "3-D cuboid on >=1 agent": int(c["box3d_labelled"].sum()),
        "SAM3 10 cm map label (trainer census 'ok')": int(c["map_label"].sum()),
        "full 6 s of future poses in the clip": int(c["full6"].sum()),
        "tactical GT AND full 6 s future": int(((c["lat_v7"] != IGN) & c["full6"]).sum()),
    }
    out["T1_channels"] = {k: {"n": v, "pct_of_windows": pct(v, N)} for k, v in rows.items()}
    clips_with_inband = np.unique(clip[c["in_band"]]).size
    out["clips_with_any_tactical_window"] = int(clips_with_inband)
    per_clip_inband = np.bincount(clip[c["in_band"]], minlength=nclip)
    out["in_band_windows_per_clip"] = {"min": int(per_clip_inband.min()), "median": float(np.median(per_clip_inband)),
                                       "max": int(per_clip_inband.max()), "mean": float(per_clip_inband.mean()),
                                       "n_clips_with_zero": int((per_clip_inband == 0).sum())}
    per_clip_n = np.bincount(clip, minlength=nclip)
    out["windows_per_clip"] = {"min": int(per_clip_n.min()), "median": float(np.median(per_clip_n)),
                               "max": int(per_clip_n.max())}
    if boot:
        out["T1_boot_ci95"] = {
            "tactical GT": cluster_boot(c["in_band"], np.ones(N), clip),
            "agent labelled": cluster_boot(c["agent_labelled"], np.ones(N), clip),
            "map label": cluster_boot(c["map_label"], np.ones(N), clip),
            "full6": cluster_boot(c["full6"], np.ones(N), clip)}

    # ---------------- T2 lat / lon classes ----------------------------------------------------------
    lat_n, lon_n = {}, {}
    for i, nm in enumerate(V["lat_classes"]):
        m = c["lat_v7"] == i
        lat_n[nm] = {"windows": int(m.sum()), "pct_of_all": pct(m.sum(), N),
                     "pct_of_in_band": pct(m.sum(), c["in_band"].sum()), "clips": int(np.unique(clip[m]).size)}
    for i, nm in enumerate(V["lon_classes"]):
        m = c["lon_v7"] == i
        lon_n[nm] = {"windows": int(m.sum()), "pct_of_all": pct(m.sum(), N),
                     "pct_of_in_band": pct(m.sum(), c["in_band"].sum()), "clips": int(np.unique(clip[m]).size)}
    out["T2_lat"], out["T2_lon"] = lat_n, lon_n

    # ---------------- T3 goal tokens ---------------------------------------------------------------
    tok = {}
    for i, nm in enumerate(V["goal_tokens"]):
        pos = ((c["goal_y_bits"] >> i) & 1).astype(bool)
        scB = ((c["goal_w_bits"] >> i) & 1).astype(bool)
        scA = ((c["goal_w_bits_censusstate"] >> i) & 1).astype(bool)
        tok[nm] = {"positive_windows": int(pos.sum()), "positive_clips": int(np.unique(clip[pos]).size),
                   "scored_windows_as_run": int(scB.sum()), "scored_windows_census_state": int(scA.sum()),
                   "pct_of_all_positive": pct(pos.sum(), N), "pct_of_all_scored": pct(scB.sum(), N),
                   "pos_over_scored_pct": pct(pos.sum(), scB.sum())}
    out["T3_goal_tokens"] = tok
    dscore = (c["n_goal_scored"] != c["n_goal_scored_censusstate"])
    out["goal_state_difference"] = {
        "n_windows_scored_count_differs": int(dscore.sum()),
        "windows_where_only_extra_cells": int((c["n_goal_scored"] > c["n_goal_scored_censusstate"]).sum()),
        "windows_where_fewer_cells": int((c["n_goal_scored"] < c["n_goal_scored_censusstate"]).sum()),
        "bits_differing": {nm: int((((c["goal_w_bits"] ^ c["goal_w_bits_censusstate"]) >> i) & 1).sum())
                           for i, nm in enumerate(V["goal_tokens"])
                           if (((c["goal_w_bits"] ^ c["goal_w_bits_censusstate"]) >> i) & 1).any()}}

    # ---------------- T4 in-band fraction as a function of t_now (1-s bins) --------------------------
    bins = {}
    for b in range(0, 20):
        m = (tn >= b) & (tn < b + 1)
        if m.sum():
            bins[f"[{b},{b + 1})"] = {"windows": int(m.sum()), "in_band": int((m & c["in_band"]).sum()),
                                      "pct_in_band": pct((m & c["in_band"]).sum(), m.sum()),
                                      "clips": int(np.unique(clip[m]).size)}
    out["T4_inband_by_tnow"] = bins
    out["t_now_range"] = [float(np.nanmin(tn)), float(np.nanmax(tn))]

    # ---------------- ego-future geometry sanity -----------------------------------------------------
    f6 = c["full6"]
    dy = c["dyaw6_deg"].astype(np.float64)
    turn = f6 & (np.abs(dy) >= TURN_DEG)
    left, right = turn & (dy > 0), turn & (dy < 0)
    gentle = f6 & (np.abs(dy) >= 10) & (np.abs(dy) < TURN_DEG)
    out["geometry"] = {
        "n_full6": int(f6.sum()), "pct_full6": pct(f6.sum(), N),
        "n_turn_windows_abs_dyaw6_ge_30": int(turn.sum()), "left": int(left.sum()), "right": int(right.sum()),
        "pct_of_full6_windows": pct(turn.sum(), f6.sum()),
        "n_turn_clips": int(np.unique(clip[turn]).size),
        "n_stop_windows_any_v_lt_0p5_in_0_6s": int((f6 & c["stop_0_6"]).sum()),
        "pct_stop_of_full6": pct((f6 & c["stop_0_6"]).sum(), f6.sum()),
        "v0_ms_pcts_5_50_95": [float(x) for x in np.nanpercentile(c["v0"], [5, 50, 95])],
        "lat6_m_abs_p50_p95_p99": [float(x) for x in np.nanpercentile(np.abs(c["lat6_m"][f6]), [50, 95, 99])]}

    # ---------------- T5 turn windows vs the tactical lateral label -----------------------------------
    sidelat = {}
    lat_names = V["lat_classes"]
    side_of = {}
    for i, nm in enumerate(lat_names):
        side_of[i] = 1 if nm.endswith("_L") else (-1 if nm.endswith("_R") else 0)
    latside = np.array([side_of.get(int(x), 0) if x != IGN else 99 for x in c["lat_v7"]], dtype=np.int8)
    def turn_block(m, name):
        n = int(m.sum())
        pres = m & (c["lat_v7"] != IGN)
        npres = int(pres.sum())
        gt = np.where(dy > 0, 1, -1)
        strict = np.array([lat_names[int(x)] if x != IGN else "IGNORE" for x in c["lat_v7"][m]]) if n else []
        agree = int((pres & (latside == gt)).sum())
        return {"n": n, "lat_GT_present": npres, "pct_present": pct(npres, n),
                "lat_IGNORE": n - npres, "pct_IGNORE": pct(n - npres, n),
                "side_agrees_where_present": agree, "pct_side_agree_where_present": pct(agree, npres),
                "lat_class_counts": {k: int((np.array(strict) == k).sum()) for k in sorted(set(strict))} if n else {},
                "lon_GT_present": int((m & (c["lon_v7"] != IGN)).sum())}
    out["T5_turn_windows"] = {"all_turn": turn_block(turn, "all"), "left": turn_block(left, "L"),
                              "right": turn_block(right, "R"),
                              "gentle_10_30": {"n": int(gentle.sum()),
                                               "pct_lat_present": pct((gentle & (c["lat_v7"] != IGN)).sum(), gentle.sum())},
                              "straight_lt_10": {"n": int((f6 & (np.abs(dy) < 10)).sum()),
                                                 "pct_lat_present": pct((f6 & (np.abs(dy) < 10) & (c["lat_v7"] != IGN)).sum(),
                                                                       (f6 & (np.abs(dy) < 10)).sum())}}
    # same, in the route-following package's own definition (terminal heading of the slot-50 -> 60 segment)
    th = c["theta_slot_deg"].astype(np.float64)
    pl = c["path_len_slot_m"].astype(np.float64)
    okc = np.isfinite(th) & (pl >= 5.0)
    gtL, gtR = okc & (th >= TURN_DEG), okc & (th <= -TURN_DEG)
    out["T5_route_definition_all_windows"] = {
        "GT-turn": int((gtL | gtR).sum()), "L": int(gtL.sum()), "R": int(gtR.sum()),
        "lat_IGNORE_on_GT-turn": int(((gtL | gtR) & (c["lat_v7"] == IGN)).sum()),
        "pct_lat_IGNORE_on_GT-turn": pct(((gtL | gtR) & (c["lat_v7"] == IGN)).sum(), (gtL | gtR).sum()),
        "straight_lt_10": int((okc & (np.abs(th) < 10)).sum()), "gentle": int((okc & (np.abs(th) >= 10) & (np.abs(th) < TURN_DEG)).sum()),
        "unclassified": int((~okc).sum())}

    # ---------------- T6 the fed nav token vs where a turn actually is --------------------------------
    nav = c["nav_cmd"]
    lr = (nav == 1) | (nav == 2)
    ttn = c["ttn_s"].astype(np.float64)
    inturn = c["in_turn_now"]
    nav_tbl = {"windows_nav_left": int((nav == 1).sum()), "windows_nav_right": int((nav == 2).sum()),
               "windows_nav_follow": int((nav == 0).sum()), "windows_nav_LR": int(lr.sum())}
    for H in (6.0, 10.0):
        starts = np.isfinite(ttn) & (ttn <= H)                       # a turn START within [now, now+H]
        overlap = starts | inturn                                    # a turn starts OR is in progress
        nav_tbl[f"LR_no_turn_start_within_{int(H)}s"] = int((lr & ~starts).sum())
        nav_tbl[f"pct_LR_no_turn_start_within_{int(H)}s"] = pct((lr & ~starts).sum(), lr.sum())
        nav_tbl[f"LR_no_turn_start_or_in_progress_within_{int(H)}s"] = int((lr & ~overlap).sum())
        nav_tbl[f"pct_LR_no_turn_start_or_in_progress_within_{int(H)}s"] = pct((lr & ~overlap).sum(), lr.sum())
        # same, but only windows that HAVE a full H-s future of labels (turn entries are raw-time; always defined)
    # side agreement: nav L but the next turn is R, etc. (labels)
    side = c["ttn_side"]
    nav_tbl["LR_with_upcoming_turn_in_labels"] = int((lr & np.isfinite(ttn)).sum())
    nav_tbl["LR_upcoming_turn_side_matches_token"] = int((lr & np.isfinite(ttn) & (((nav == 1) & (side == 1)) | ((nav == 2) & (side == -1)))).sum())
    nav_tbl["ttn_s_quantiles_5_25_50_75_95_on_LR_with_turn"] = [float(x) for x in np.nanpercentile(ttn[lr & np.isfinite(ttn)], [5, 25, 50, 75, 95])] if (lr & np.isfinite(ttn)).any() else None
    # from the clip's OWN realised poses (full-6 windows only): fed L/R, no realised turn >=30 deg in 6 s
    m_f6 = f6 & lr
    nav_tbl["realised_LR_windows_full6"] = int(m_f6.sum())
    nav_tbl["realised_LR_no_30deg_turn_in_6s"] = int((m_f6 & (np.abs(dy) < TURN_DEG)).sum())
    nav_tbl["pct_realised_LR_no_30deg_turn_in_6s"] = pct((m_f6 & (np.abs(dy) < TURN_DEG)).sum(), m_f6.sum())
    nav_tbl["realised_LR_no_10deg_in_6s"] = int((m_f6 & (np.abs(dy) < 10)).sum())
    nav_tbl["pct_realised_LR_no_10deg_in_6s"] = pct((m_f6 & (np.abs(dy) < 10)).sum(), m_f6.sum())
    # the other direction: a realised >=30 deg turn in the next 6 s while the fed token is 'follow'
    nav_tbl["realised_turn_windows_full6"] = int(turn.sum())
    nav_tbl["realised_turn_with_nav_follow"] = int((turn & (nav == 0)).sum())
    nav_tbl["pct_realised_turn_with_nav_follow"] = pct((turn & (nav == 0)).sum(), turn.sum())
    nav_tbl["realised_turn_with_wrong_side_token"] = int((turn & (((left) & (nav == 2)) | ((right) & (nav == 1)))).sum())
    nav_tbl["realised_turn_with_matching_token"] = int((turn & (((left) & (nav == 1)) | ((right) & (nav == 2)))).sum())
    nav_tbl["pct_realised_turn_with_matching_token"] = pct(nav_tbl["realised_turn_with_matching_token"], turn.sum())
    out["T6_nav"] = nav_tbl
    # per clip: nav token vs first turn timing
    cl_nav = {}
    for nm, k in (("follow", 0), ("left", 1), ("right", 2)):
        cl_nav[nm] = int(sum(1 for cl in clips if cl["nav_cmd"] == k))
    out["T6_clip_nav_counts"] = cl_nav

    # ---------------- T7 the per-clip ceiling vs the window's own realised speed --------------------
    vb = c["vmax_bin"].astype(np.int8)
    vmax26 = c["vmax26"].astype(np.float64)
    ok7 = f6 & c["vmax_valid"] & np.isfinite(vmax26)
    rb = real_bin(np.where(np.isfinite(vmax26), vmax26, 0.0))
    exceeds = ok7 & (vmax26 * 3.6 > LADDER_KMH[np.clip(vb, 0, 3)] + 1e-6)
    below = ok7 & (rb < vb)
    within = ok7 & (rb == vb)
    t7 = {"n_windows_full6_with_ceiling": int(ok7.sum()),
          "realised_max_26_exceeds_fed_ceiling": int(exceeds.sum()), "pct_exceeds": pct(exceeds.sum(), ok7.sum()),
          "realised_bin_below_fed_bin": int(below.sum()), "pct_below": pct(below.sum(), ok7.sum()),
          "realised_bin_equals_fed_bin": int(within.sum()), "pct_equal": pct(within.sum(), ok7.sum()),
          "per_fed_bin": {}}
    for b, nm in enumerate(LADDER_KMH.astype(int)):
        m = ok7 & (vb == b)
        t7["per_fed_bin"][str(nm)] = {
            "windows": int(m.sum()), "pct_of_ok": pct(m.sum(), ok7.sum()),
            "exceeds": int((m & exceeds).sum()), "pct_exceeds": pct((m & exceeds).sum(), m.sum()),
            "below": int((m & below).sum()), "pct_below": pct((m & below).sum(), m.sum()),
            "equal": int((m & within).sum()), "pct_equal": pct((m & within).sum(), m.sum()),
            "clips": int(np.unique(clip[m]).size)}
    over_kmh = np.where(ok7, vmax26 * 3.6 - LADDER_KMH[np.clip(vb, 0, 3)], 0.0)
    t7["exceeds_by_gt_5_kmh"] = {"n": int((ok7 & (over_kmh > 5.0)).sum()), "pct": pct((ok7 & (over_kmh > 5.0)).sum(), ok7.sum())}
    t7["exceeds_by_gt_10_kmh"] = {"n": int((ok7 & (over_kmh > 10.0)).sum()), "pct": pct((ok7 & (over_kmh > 10.0)).sum(), ok7.sum())}
    t7["realised_max_below_fed_ceiling_by_gt_20_kmh"] = {
        "n": int((ok7 & (over_kmh < -20.0)).sum()), "pct": pct((ok7 & (over_kmh < -20.0)).sum(), ok7.sum())}
    t7["median_realised_minus_ceiling_kmh"] = float(np.median(over_kmh[ok7])) if ok7.any() else None
    # restricted to the in-band windows (the span the ceiling's own definition describes)
    mb = ok7 & c["in_band"]
    t7["in_band_only"] = {"n": int(mb.sum()), "pct_exceeds": pct((mb & exceeds).sum(), mb.sum()),
                          "pct_below": pct((mb & below).sum(), mb.sum()), "pct_equal": pct((mb & within).sum(), mb.sum())}
    # and the current speed against the fed ceiling (no future needed): every window
    v0k = c["v0"].astype(np.float64) * 3.6
    ceil_k = LADDER_KMH[np.clip(vb, 0, 3)]
    okv = c["vmax_valid"] & np.isfinite(v0k)
    t7["v0_above_fed_ceiling"] = {"n": int((okv & (v0k > ceil_k + 1e-6)).sum()),
                                  "pct": pct((okv & (v0k > ceil_k + 1e-6)).sum(), okv.sum())}
    # windows that are stopped or nearly so (the ceiling is irrelevant there)
    t7["v0_lt_1ms"] = {"n": int((c["v0"] < 1.0).sum()), "pct": pct((c["v0"] < 1.0).sum(), N)}
    out["T7_ceiling"] = t7

    # ---------------- T8: everything that is a per-CLIP constant, as a function of t_now (2-s bins) ---------
    t8 = {}
    for b in range(0, 18, 2):
        m = (tn >= b) & (tn < b + 2)
        if not m.any():
            continue
        mf = m & f6
        mlr = m & lr
        mok = m & ok7
        mturn = m & turn
        starts6 = np.isfinite(ttn) & (ttn <= 6.0)
        t8[f"[{b},{b + 2})"] = {
            "windows": int(m.sum()), "full6_windows": int(mf.sum()),
            "pct_tactical_GT": pct((m & c["in_band"]).sum(), m.sum()),
            "pct_nav_LR": pct(mlr.sum(), m.sum()),
            "LR_windows": int(mlr.sum()),
            "pct_LR_with_turn_start_within_6s": pct((mlr & starts6).sum(), mlr.sum()),
            "pct_LR_no_30deg_turn_in_6s_poses": pct((mlr & f6 & (np.abs(dy) < TURN_DEG)).sum(), (mlr & f6).sum()),
            "turn_windows": int(mturn.sum()),
            "pct_turn_with_matching_token": pct((mturn & (((dy > 0) & (nav == 1)) | ((dy < 0) & (nav == 2)))).sum(), mturn.sum()),
            "pct_turn_with_follow_token": pct((mturn & (nav == 0)).sum(), mturn.sum()),
            "ceiling_windows": int(mok.sum()),
            "pct_exceeds": pct((mok & exceeds).sum(), mok.sum()),
            "pct_below": pct((mok & below).sum(), mok.sum()),
            "pct_same": pct((mok & within).sum(), mok.sum())}
    out["T8_by_tnow_2s"] = t8
    # in-band windows only: lateral label vs the window's OWN realised heading, by |t_now - 8|
    t5b = {}
    for lo_, hi_ in ((0, 1), (1, 2)):
        m = c["in_band"] & (np.abs(tn - 8.0) >= lo_) & (np.abs(tn - 8.0) < hi_ + (1e-9 if hi_ == 2 else 0))
        mt = m & turn
        pres = mt & (c["lat_v7"] != IGN)
        gt = np.where(dy > 0, 1, -1)
        t5b[f"abs(t_now-8) in [{lo_},{hi_}]"] = {
            "in_band_windows": int(m.sum()), "turn_windows": int(mt.sum()),
            "pct_label_is_TURN_x_on_turn": pct((mt & np.isin(c["lat_v7"], [i for i, nm in enumerate(lat_names) if nm.startswith("TURN")])).sum(), mt.sum()),
            "pct_label_is_LANE_KEEP_on_turn": pct((mt & (c["lat_v7"] == lat_names.index("LANE_KEEP"))).sum(), mt.sum()),
            "side_agree_pct": pct((mt & (latside == gt)).sum(), mt.sum())}
    out["T5b_label_vs_own_heading_by_distance_from_anchor"] = t5b

    # ---------------- control (f): the record's own v_hi vs my realised max at the window nearest to the anchor
    rows = []
    for k, cl in enumerate(clips):
        idx = np.nonzero(clip == k)[0]
        if not idx.size or cl.get("v_hi_ms_band") is None:
            continue
        j = idx[np.argmin(np.abs(tn[idx] - 8.0))]
        if abs(tn[j] - 8.0) <= 0.06 and f6[j] and np.isfinite(vmax26[j]):
            rows.append((cl["v_hi_ms_band"], float(vmax26[j])))
    if rows:
        a = np.array(rows)
        d = a[:, 1] - a[:, 0]
        out["control_vhi_vs_pose_vmax_at_anchor"] = {
            "n_clips": int(len(a)), "median_abs_diff_ms": float(np.median(np.abs(d))),
            "p90_abs_diff_ms": float(np.percentile(np.abs(d), 90)), "median_diff_ms": float(np.median(d)),
            "pearson_r": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
            "frac_within_0p5_ms": float(np.mean(np.abs(d) <= 0.5))}

    # ---------------- control (b): config.json stamps ---------------------------------------------
    out["control_config"] = {
        "n_windows": int(N), "agent_labelled": int(c["agent_labelled"].sum()),
        "agent_labelled_clear": int((c["agent_labelled"] & (c["n_agents"] == 0)).sum()),
        "map_ok": int(c["map_label"].sum()),
        "ceiling_fed": int(c["vmax_valid"].sum()),
        "nav_clip_counts": cl_nav}
    out["record_controls"] = {k: rec.get(k) for k in ("C2_direct_equals_getitem", "pose_controls", "n_windows", "n_clips")}
    return out, cols, clips, rec


def route_window_control(cols, clips, rec, d: Path):
    """control (a): the route-following package's EVAL grid -- 139 eps x 8 windows at (j+0.5)*n/8, episodes
    sorted by sha12 -- and its published counts."""
    clip = cols["clip_ix"]
    sha = [cl["sha12"] for cl in clips]
    order = sorted(range(len(clips)), key=lambda k: sha[k])
    win_sha, win_t, rows = [], [], []
    for k in order:
        idx = np.nonzero(clip == k)[0]
        ts = cols["t"][idx]
        o = np.argsort(ts, kind="stable")
        idx = idx[o]
        for j in range(8):
            r = int(idx[int((j + 0.5) * len(idx) / 8)])
            rows.append(r)
            win_sha.append(sha[k])
            win_t.append(int(cols["t"][r]))
    digest = hashlib.sha256(json.dumps(list(zip(win_sha, win_t))).encode()).hexdigest()
    rows = np.array(rows)
    th = cols["theta_slot_deg"][rows].astype(np.float64)
    pl = cols["path_len_slot_m"][rows].astype(np.float64)
    ok = np.isfinite(th) & (pl >= 5.0)
    L_, R_ = ok & (th >= 30.0), ok & (th <= -30.0)
    straight = ok & (np.abs(th) < 10.0)
    gentle = ok & (np.abs(th) >= 10.0) & (np.abs(th) < 30.0)
    lat = cols["lat_v7"][rows]
    names = rec["vocab"]["lat_classes"]
    side = np.array([(1 if names[int(x)].endswith("_L") else (-1 if names[int(x)].endswith("_R") else 0)) if x != IGN else 99
                     for x in lat])
    def row(m):
        return {"n": int(m.sum()), "ignore": int((m & (lat == IGN)).sum()),
                "label_side_L/S/R": [int((m & (side == 1)).sum()), int((m & (side == 0)).sum()), int((m & (side == -1)).sum())]}
    dy = cols["dyaw6_deg"][rows].astype(np.float64)
    f6 = cols["full6"][rows]
    t_dy = f6 & (np.abs(dy) >= 30.0)
    expect_digest = "92e36a1a74b8b699b69c799fba469376cafbcadd796e9a0737746f115ee92a1e"
    res = {"window_digest": digest, "published_digest": expect_digest, "digest_equal": digest == expect_digest,
           "n_windows": int(len(rows)), "n_episodes": int(len(set(win_sha))),
           "route_definition": {"GT-turn": int((L_ | R_).sum()), "L": int(L_.sum()), "R": int(R_.sum()),
                                "straight": int(straight.sum()), "gentle": int(gentle.sum()), "unclassified": int((~ok).sum()),
                                "turnL": row(L_), "turnR": row(R_), "turn": row(L_ | R_), "straight_row": row(straight),
                                "gentle_row": row(gentle),
                                "side_agree_where_present": int((L_ & (side == 1)).sum() + (R_ & (side == -1)).sum()),
                                "present": int(((L_ | R_) & (lat != IGN)).sum())},
           "published": {"GT-turn": 107, "L": 40, "R": 67, "straight": 588, "gentle": 105, "unclassified": 312,
                         "lat_IGNORE_on_turn": 72, "label_L/S/R_on_turnL(present)": [5, 5, 0],
                         "label_L/S/R_on_turnR(present)": [1, 12, 12], "IGNORE_turnL": 30, "IGNORE_turnR": 42,
                         "side_agree_present": "17/35 = 0.486"},
           "abs_dyaw6_ge_30_definition": {"n": int(t_dy.sum()), "L": int((t_dy & (dy > 0)).sum()), "R": int((t_dy & (dy < 0)).sum()),
                                         "lat_IGNORE": int((t_dy & (lat == IGN)).sum()),
                                         "n_with_label": int((t_dy & (lat != IGN)).sum())}}
    # overlap of the two definitions on this grid
    res["definitions_overlap"] = {"both": int(((L_ | R_) & t_dy).sum()), "route_only": int(((L_ | R_) & ~t_dy).sum()),
                                  "dyaw_only": int((~(L_ | R_) & t_dy).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    for tag in a.tags:
        out, cols, clips, rec = analyse(d, tag, boot=(tag.startswith("eval")))
        if tag.startswith("eval"):
            out["control_route_following_grid"] = route_window_control(cols, clips, rec, d)
        json.dump(out, open(d / f"numbers_{tag}.json", "w", encoding="utf-8"), indent=1)
        print("wrote", d / f"numbers_{tag}.json")


if __name__ == "__main__":
    main()
