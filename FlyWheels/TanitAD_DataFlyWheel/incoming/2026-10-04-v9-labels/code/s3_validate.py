"""WP-A Stage 3 validation of a v9 release split (SPEC §9.V, bars fixed in SPEC.md before the build).

V1 coverage · V2 agreement with the INDEPENDENT derivation (`v9_independent.py`, no builder import) · V3 constraint
errors · V5 lane change vs the Alpamayo lane-change TEXT (+ the mirror mutation) · V7 truncation rule (builder re-run
with h capped at 6 s — a builder self-consistency check, kept OUT of the independent module) · V8 nav bars ·
V9 VLM coverage · V10 class support · V11 join with the trainer's windows (D1 truth table, row == ds index).

usage: python s3_validate.py <eval139|train> <release.npz> <out.json> [--v7]
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v9_independent as IND  # noqa: E402

AUD = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
SRC = {"eval139": {"table": f"{AUD}/D1_label_census/tables/label_truth_eval139.npz",
                   "manifest": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
                   "labels": "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"},
       "train": {"table": f"{AUD}/D1_label_census/tables/label_truth_train.npz",
                 "manifest": "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt",
                 "labels": "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz"}}
LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")


def sha12(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def frac(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "wilson95": wilson(int(k), int(n))}


def main(split, rel_p, out_p, do_v7=False):
    import torch
    t0 = time.time()
    cfg = SRC[split]
    Z = np.load(rel_p, allow_pickle=True)
    R = {k[5:]: Z[k] for k in Z.files if k.startswith("row__")}
    C = {k[6:]: Z[k] for k in Z.files if k.startswith("clip__")}
    T = np.load(cfg["table"], allow_pickle=True)
    w_sha = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in T["clip_sha12"]])
    w_t = T["t"].astype(np.int64)
    w_now = T["t_now_s"].astype(np.float64)
    # ---------------------------------------------------------------- V11: join with the trainer's windows
    row0 = {str(s): (int(r0), int(n)) for s, r0, n in zip(C["sha12"], C["row0"], C["n_rows"])}
    rows = np.full(len(w_t), -1, np.int64)
    for i, (s, t) in enumerate(zip(w_sha, w_t)):
        if s in row0:
            r0, n = row0[s]
            k = t + 9
            j = k - 2
            if 0 <= j < n and int(R["k"][r0 + j]) == k:
                rows[i] = r0 + j
    ok = rows >= 0
    dnow = np.abs(R["now_s"][rows[ok]] - w_now[ok])
    out = {"split": split, "release": os.path.basename(rel_p),
           "V11_join": {"windows": int(len(w_t)), "joined": int(ok.sum()), "max_abs_now_diff_s": float(dnow.max()),
                        "pass": bool(ok.all() and dnow.max() <= 1e-6)}}
    W = {k: v[rows] for k, v in R.items()}           # window-level view (all joined)
    # ---------------------------------------------------------------- V1 coverage
    lat_cov = (W["lat_allowed_a"] > 0)
    lon_cov = (W["lon_allowed"] > 0)
    out["V1_coverage"] = {"lat_a_single_or_partial": frac(lat_cov.sum(), len(rows)),
                          "lat_b_single_or_partial": frac((W["lat_allowed_b"] > 0).sum(), len(rows)),
                          "lon_single_or_partial": frac(lon_cov.sum(), len(rows)),
                          "both": frac((lat_cov & lon_cov).sum(), len(rows)),
                          "lat_a_single_valued": frac((W["lat_cls_a"] >= 0).sum(), len(rows)),
                          "lon_single_valued": frac((W["lon_cls"] >= 0).sum(), len(rows))}
    out["V1_coverage"]["pass_95"] = bool(out["V1_coverage"]["both"]["rate"] >= 0.95)
    # ---------------------------------------------------------------- independent derivation per window
    m = torch.load(cfg["manifest"], map_location="cpu", weights_only=False)
    cid = {sha12(str(c)): str(c) for c in m["clip_id"]}
    th_i = np.full(len(rows), np.nan)
    th_i0 = np.full(len(rows), np.nan)   # as first registered: no stationary rule
    cusp_i = np.zeros(len(rows), bool)
    h_i = np.full(len(rows), np.nan)
    st_i = np.full(len(rows), np.nan)
    tst_i = np.full(len(rows), np.nan)
    dst_i = np.full(len(rows), np.nan)
    seg_match = {"dt_start": [], "dt_end": [], "ddyaw": [], "r_rel": []}
    nav_arc, nav_rel = [], []
    for s in np.unique(w_sha):
        L = IND.Log(EGO.format(cid[s]))
        segs = IND.segments(L, 0.0, L.t_end)
        for i in np.nonzero(w_sha == s)[0]:
            now = float(w_now[i])
            h = L.h_obs(now)
            h_i[i] = h
            th, obs = IND.band_turn(L, now, h)
            th_i[i] = th
            th_i0[i] = IND.band_turn(L, now, h, stationary_rule=False)[0]
            cusp_i[i] = L.cusp_in(now, now + 8.0)
            stp, ts, ds = IND.band_stop(L, now, h)
            st_i[i] = np.nan if stp is None else float(stp)
            tst_i[i], dst_i[i] = ts, ds
            # V3 turn timing / dyaw / radius: the builder's dominant segment vs an overlapping independent segment
            if W["lat_cls_b"][i] in (1, 2) and np.isfinite(W["turn_t_start_s"][i]):
                bs, be = now + W["turn_t_start_s"][i], now + W["turn_t_end_s"][i]
                sgn = 1 if W["turn_dyaw_deg"][i] > 0 else -1
                cand = [q for q in segs if q[1] > bs and q[0] < be and np.sign(q[2]) == sgn]
                if cand:
                    q = max(cand, key=lambda q: min(q[1], be) - max(q[0], bs))
                    seg_match["dt_start"].append(abs(q[0] - bs))
                    seg_match["dt_end"].append(abs(q[1] - be))
                    seg_match["ddyaw"].append(abs(q[2] - W["turn_dyaw_deg"][i]))
                    if np.isfinite(q[3]) and q[3] > 0:
                        seg_match["r_rel"].append(abs(W["turn_r_arc_m"][i] - q[3]) / q[3])
            # V3 nav distance: chord arc (builder) vs speed-integral arc (independent) over the same instants
            if W["nav_args_valid"][i] == 1 and np.isfinite(W["nav_t_next_s"][i]) and W["nav_t_next_s"][i] > 0:
                t_turn = now + W["nav_t_next_s"][i]
                if t_turn <= L.t_end:
                    d_ind = float(np.interp(t_turn, L.t, L.arc_v) - np.interp(now, L.t, L.arc_v))
                    nav_arc.append(abs(d_ind - W["nav_d_next_m"][i]))
                    nav_rel.append(abs(d_ind - W["nav_d_next_m"][i]) / max(d_ind, 1.0))
    # ---------------------------------------------------------------- V2 agreement
    ind_turn = np.isfinite(th_i) & (np.abs(th_i) >= IND.TURN_DEG)
    ind_side = np.where(th_i > 0, 1, 2)
    b_turn = np.isin(W["lat_cls_b"], (1, 2))
    a_turn = np.isin(W["lat_cls_a"], (1, 2))
    rec_b = ind_turn & (W["lat_cls_b"] == ind_side)
    prec_b = b_turn & ind_turn & (W["lat_cls_b"] == ind_side)
    margin = ind_turn & ((np.abs(th_i) < 27) | (np.abs(th_i) > 33))
    out["V2_turn"] = {"independent_turn_windows": int(ind_turn.sum()),
                      "recall_b_same_side": frac(rec_b.sum(), ind_turn.sum()),
                      "precision_b": frac(prec_b.sum(), b_turn.sum()),
                      "recall_b_excluding_27_33deg_margin": frac((rec_b & margin).sum(), margin.sum()),
                      "side_inverted_b": int((b_turn & ind_turn & (W["lat_cls_b"] != ind_side)).sum()),
                      "variant_a_recall_on_independent_turns": frac((ind_turn & (W["lat_cls_a"] == ind_side)).sum(), ind_turn.sum()),
                      "variant_a_precision": frac((a_turn & ind_turn & (W["lat_cls_a"] == ind_side)).sum(), a_turn.sum()),
                      "independent_turn_windows_unlabelled_b": int((ind_turn & (W["lat_allowed_b"] == 0)).sum())}
    it0 = np.isfinite(th_i0) & (np.abs(th_i0) >= IND.TURN_DEG)
    s0 = np.where(th_i0 > 0, 1, 2)
    out["V2_turn_AS_FIRST_RUN_no_stationary_rule"] = {"recall_b": frac((it0 & (W['lat_cls_b'] == s0)).sum(), it0.sum()),
        "precision_b": frac((b_turn & it0 & (W['lat_cls_b'] == s0)).sum(), b_turn.sum()),
        "side_inverted_b": int((b_turn & it0 & (W['lat_cls_b'] != s0)).sum())}
    nc = ~cusp_i
    out["V2_turn_excluding_cusp_windows"] = {"cusp_windows": int(cusp_i.sum()), "cusp_clips": int(len(np.unique(w_sha[cusp_i]))),
        "recall_b": frac((rec_b & nc).sum(), (ind_turn & nc).sum()),
        "precision_b": frac((prec_b & nc).sum(), (b_turn & nc).sum()),
        "side_inverted_b": int((b_turn & ind_turn & (W["lat_cls_b"] != ind_side) & nc).sum()),
        "builder_reversing_masked_windows": int((W["reversing"] == 1).sum()) if "reversing" in W else None}
    out["V2_turn"]["pass"] = bool(out["V2_turn"]["recall_b_same_side"]["rate"] >= 0.95 and out["V2_turn"]["precision_b"]["rate"] >= 0.95)
    # STOP: both defined (independent va > 0.5 and builder's v_a > 0.5), and STOP-ELIGIBLE by the SPEC's precedence
    # (HOLD and CREEP precede STOP by definition, so a stop from <= 2 m/s is CREEP, not a disagreement) — fixed
    # before this ran; the all-windows reading is printed beside it.
    lcc = W["lon_cls_computed"] if "lon_cls_computed" in W else W["lon_cls"]
    both_all = np.isfinite(st_i) & (W["v_a_ms"] > 0.5) & (W["lon_allowed"] > 0)
    both = both_all & ~np.isin(lcc, (0, 1))
    out["V2_stop_all_windows_incl_creep"] = {
        "recall": frac((both_all & (st_i == 1) & (W["lon_cls"] == 2)).sum(), (both_all & (st_i == 1)).sum())}
    b_stop = W["lon_cls"] == 2
    out["V2_stop"] = {"both_defined": int(both.sum()), "independent_stops": int((both & (st_i == 1)).sum()),
                      "recall": frac((both & (st_i == 1) & b_stop).sum(), (both & (st_i == 1)).sum()),
                      "precision": frac((both & b_stop & (st_i == 1)).sum(), (both & b_stop).sum())}
    out["V2_stop"]["pass"] = bool(out["V2_stop"]["recall"]["rate"] >= 0.95 and out["V2_stop"]["precision"]["rate"] >= 0.95)
    # ---------------------------------------------------------------- V3 constraint errors
    sm = both & b_stop & (st_i == 1)
    et = np.abs(W["lon_t_reach_s"][sm] - tst_i[sm])
    ed = np.abs(W["lon_d_reach_m"][sm] - dst_i[sm])

    def within(a, tol):
        a = np.asarray(a, np.float64)
        return {"n": int(a.size), "share_within": round(float((a <= tol).mean()), 4) if a.size else None,
                "median": round(float(np.median(a)), 4) if a.size else None, "p90": round(float(np.percentile(a, 90)), 4) if a.size else None, "tol": tol}
    out["V3"] = {"turn_t_start_s": within(seg_match["dt_start"], 0.5), "turn_t_end_s": within(seg_match["dt_end"], 0.5),
                 "turn_dyaw_deg": within(seg_match["ddyaw"], 3.0), "turn_r_arc_rel": within(seg_match["r_rel"], 0.10),
                 "stop_t_s": within(et, 0.3), "stop_d_m": within(ed, 1.0), "nav_d_next_m": within(nav_arc, 2.0),
                 "DIAG_nav_d_next_relative": within(nav_rel, 0.02)}
    out["V3"]["pass_90"] = {k: (v["share_within"] is not None and v["share_within"] >= 0.90) for k, v in out["V3"].items() if isinstance(v, dict)}
    # ---------------------------------------------------------------- V8 nav
    tok = W["nav_token"]
    lr = tok > 0
    inprog = W["nav_in_progress"] == 1
    soon = np.isfinite(W["nav_t_next_s"]) & (W["nav_t_next_s"] <= 6.0)
    viol = lr & ~(inprog | soon)
    near = np.isfinite(W["nav_d_next_m"]) & (W["nav_d_next_m"] <= 30.0) & ~inprog
    real = W["nav_args_valid"] == 1
    real &= inprog | (soon & (W["nav_t_next_s"] >= 0))
    match = real & (tok == np.where(W["nav_side_next"] > 0, 1, 2))
    ttime = W["nav_token_ttime"] > 0
    out["V8_nav"] = {"turn_commanded_windows": int(lr.sum()),
                     "i_as_written_no_turn_start_within_6s": frac(viol.sum(), lr.sum()),
                     "ii_excluding_ego_within_30m_of_turn": frac((viol & ~near).sum(), (lr & ~near).sum()),
                     "iii_true_time_token": frac((ttime & ~(inprog | soon)).sum(), ttime.sum()),
                     "realised_announced_turn_windows_matching_token": frac(match.sum(), real.sum()),
                     "unannounced_realised_turns_independent(nav FOLLOW, |theta|>=30)": int((ind_turn & (tok == 0)).sum()),
                     "refcv7_reference": "D1: 74.3 % of L/R windows had no turn start within 6 s (train)"}
    # ---------------------------------------------------------------- V9 VLM / goal coverage, V10 class support
    gy, gw = W["goal_y"].astype(np.int64), W["goal_w"].astype(np.int64)
    out["V9_goals_windows"] = {t: {"pos": int((((gy >> i) & 1) & ((gw >> i) & 1)).sum()),
                                   "neg": int(((1 - ((gy >> i) & 1)) & ((gw >> i) & 1)).sum()),
                                   "ignore": int((1 - ((gw >> i) & 1)).sum())} for i, t in enumerate(GOAL22)}
    out["V9_red_propagated"] = {"windows": int(W["red_propagated"].sum()),
                                "clips": int(len(np.unique(w_sha[W["red_propagated"] == 1])))}
    out["V9_vlm_frames_windows"] = int(W["vlm_frame"].sum())
    out["V10_support"] = {"lat_a": {c: [int((W["lat_cls_a"] == i).sum()), int(len(np.unique(w_sha[W["lat_cls_a"] == i])))] for i, c in enumerate(LAT9)},
                          "lat_b": {c: [int((W["lat_cls_b"] == i).sum()), int(len(np.unique(w_sha[W["lat_cls_b"] == i])))] for i, c in enumerate(LAT9)},
                          "lon": {c: [int((W["lon_cls"] == i).sum()), int(len(np.unique(w_sha[W["lon_cls"] == i])))] for i, c in enumerate(LON9)},
                          "lat_partial_windows": int(((W["lat_cls_a"] < 0) & (W["lat_allowed_a"] > 0)).sum()),
                          "lon_partial_windows": int(((W["lon_cls"] < 0) & (W["lon_allowed"] > 0)).sum()),
                          "lc_measurable_windows": int((W["lc_measurable"] == 1).sum())}
    # ---------------------------------------------------------------- V5 lane change vs the Alpamayo LC text
    recs = {}
    with gzip.open(cfg["labels"], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[sha12(r["clip_id"])] = r
    v5 = {"text_clips": 0, "text_side_correct": 0, "text_fired": 0, "no_text_clips": 0, "no_text_fired": 0,
          "status_counts": {}}
    rows_by = {str(s): (int(r0), int(n)) for s, r0, n in zip(C["sha12"], C["row0"], C["n_rows"])}
    for s, (r0, n) in rows_by.items():
        rec = recs.get(s)
        if rec is None or not int(C["has_lc"][list(C["sha12"]).index(s)]):
            continue
        lt = rec.get("lane_change_text") or {}
        st = str(lt.get("status"))
        v5["status_counts"][st] = v5["status_counts"].get(st, 0) + 1
        off = float((rec.get("alpamayo") or {}).get("anchor_offset_s", -2.9))
        w0 = float(rec["t0_s"]) + off
        sl = slice(r0, r0 + n)
        tc = R["now_s"][sl] + R["lc_t_cross_s"][sl]
        sd = R["lc_side"][sl]
        okc = np.isfinite(tc)
        cr = {(round(float(a), 2), int(b)) for a, b in zip(tc[okc], sd[okc])}
        inw = [b for a, b in cr if w0 <= a <= w0 + 6.0]
        if st == "EXECUTED" and lt.get("side") in ("left", "right", "L", "R"):
            v5["text_clips"] += 1
            want = 1 if str(lt.get("side")).lower().startswith("l") else -1
            if inw:
                v5["text_fired"] += 1
                dom = 1 if sum(inw) > 0 else (-1 if sum(inw) < 0 else 0)
                v5["text_side_correct"] += int(dom == want)
        elif not lt.get("asserted"):
            v5["no_text_clips"] += 1
            v5["no_text_fired"] += int(bool(inw))
    v5["side_correct_rate"] = round(v5["text_side_correct"] / v5["text_clips"], 4) if v5["text_clips"] else None
    v5["side_correct_given_fired"] = round(v5["text_side_correct"] / v5["text_fired"], 4) if v5["text_fired"] else None
    v5["no_text_firing_rate"] = round(v5["no_text_fired"] / v5["no_text_clips"], 4) if v5["no_text_clips"] else None
    v5["pass"] = bool(v5["side_correct_rate"] is not None and v5["side_correct_rate"] >= 0.80
                      and v5["no_text_firing_rate"] is not None and v5["no_text_firing_rate"] <= 0.10)
    out["V5_lane_change"] = v5
    # ---------------------------------------------------------------- V7 truncation (builder self-consistency)
    if do_v7:
        out["V7_truncation"] = v7(split, cfg, Z)
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(out_p, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in out.items() if k not in ("V9_goals_windows",)}, indent=1,
                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])


def v7(split, cfg, Z, n_clips=139, seed=20261004):
    """Rebuild a clip sample with h capped at 6.0 s and compare the labels on the FULL-band frames (h_obs = 8)."""
    sys.path.insert(0, r"C:\Users\Admin\r8_wpa\stack\scripts")
    import build_v9_labels as B
    import torch
    m = torch.load(cfg["manifest"], map_location="cpu", weights_only=False)
    cids = [str(c) for c in m["clip_id"]]
    poses = [p.double().numpy() for p in m["poses"]]
    side = B._read_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
    recs = {}
    with gzip.open(cfg["labels"], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["clip_id"]] = r
    idx = np.random.default_rng(seed).choice(len(cids), min(n_clips, len(cids)), replace=False)
    ch = {"lat_a": [0, 0], "lat_b": [0, 0], "lon": [0, 0]}
    for ci in idx:
        cid = cids[ci]
        g0, dt, _ = B.clip_clock(B.stable_episode_id(cid), side, poses[ci])
        k = np.arange(poses[ci].shape[0]) + 2
        now_k = g0 + k * dt
        rec = recs.get(cid)
        G = B.clip_geo(B.load_log(EGO.format(cid)), (rec or {}).get("turn_suppression"))
        Ff = B.clip_labels(G, now_k, k, rec, None, None)
        Fc = B.clip_labels(G, now_k, k, rec, None, None, h_cap=6.0)
        full = Ff["h_obs_s"] >= 8.0 - 1e-9
        for key, col in (("lat_a", "lat_cls_a"), ("lat_b", "lat_cls_b"), ("lon", "lon_cls_computed")):
            a, b = Ff[col][full], Fc[col][full]
            d = a >= 0
            ch[key][0] += int((d & (a != b)).sum())
            ch[key][1] += int(d.sum())
    res = {k: {"changed": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None} for k, v in ch.items()}
    res["pass_5pct"] = bool(all(v["rate"] is not None and v["rate"] <= 0.05 for v in res.values() if isinstance(v, dict)))
    res["note"] = "lon compares the class before the FOLLOW partial-label step (no agent data in this re-run)"
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--v7" in sys.argv)
