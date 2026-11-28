#!/usr/bin/env python3
"""t1k_read.py -- PREREG_V8F_T1KIN_HACC with Amendment 1 (landed 323624f2 / 15abe8d9, before any real-data number), applied verbatim.

Rows (Amendment 1): ADE2, ADE6, and the per-window components of ``taniteval/four_families.py`` (``_seq_geometry`` at dt 0.2 s and the
error definitions and masks of ``longitudinal`` / ``lateral``), for the executed tracks HACC (``cl``) and P10 (``fl``) against ``g``;
paired with ``taniteval.ci.paired_episode_cluster_bootstrap`` (B 2,000, seed 0) on episodes. Lower is better on every row.
Controls C-1..C-4 (PREREG section 4) and R-1..R-3 (Amendment 1) are read BEFORE any outcome; a failed one => REFUSED, no outcome row.

python t1k_read.py <export root (has taniteval/ and the v8f-mini raw/)> <run dir> <chain log> <out dir>
python t1k_read.py --selftest <export root>          (R-3 alone, synthetic)"""
import glob
import json
import math
import os
import sys

import numpy as np

SELF = sys.argv[1] == "--selftest"
ROOT = sys.argv[2] if SELF else sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT = 0.2
F = 0.1207
TWO_F = 0.2415
PK = os.path.join(ROOT, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
FLOOR_OL_ADE6 = 2.48699212                 # DESIGN_T1_V8F.md section 8: the banked open-loop floor, n = 2,399
ROWS = ("ade2_m", "ade6_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm",
        "cross_mae_m")
FAMILY = {"ade2_m": "ADE", "ade6_m": "ADE", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL",
          "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL", "cross_mae_m": "LATERAL"}


def comps(X, G):
    """Per-window components [n] of one arm against the GT, with four_families' definitions and masks (NaN where a mask is empty)."""
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    e = torch.linalg.norm(Xt - Gt, dim=-1)
    both, bp = P["valid"] & Q["valid"], P["pair_valid"] & Q["pair_valid"]
    dh = P["heading"] - Q["heading"]
    dh = ((dh + math.pi) % (2 * math.pi) - math.pi).abs()

    def mm(x, m):
        s, c = (x * m).sum(1), m.sum(1)
        return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))
    out = {"ade2_m": e[:, :10].mean(1), "ade6_m": e.mean(1),
           "speed_mae_mps": (P["speed"] - Q["speed"]).abs().mean(1), "along_mae_m": (P["along"] - Q["along"]).abs().mean(1),
           "accel_mae_mps2": (P["accel"] - Q["accel"]).abs().mean(1), "cross_mae_m": (P["cross"] - Q["cross"]).abs().mean(1),
           "heading_mae_deg": torch.rad2deg(mm(dh, both)), "yaw_rate_mae_degps": torch.rad2deg(mm((P["yaw_rate"] - Q["yaw_rate"]).abs(), bp)),
           "curvature_mae_1pm": mm((P["curvature"] - Q["curvature"]).abs(), bp)}
    res = {k: v.numpy().astype(np.float64) for k, v in out.items()}
    # the POOLED masked means four_families.lateral reports (sum over every valid step / count), for R-1 on the masked rows
    res["_pooled"] = {"heading_mae_deg": math.degrees(float(dh[both].mean())) if both.any() else float("nan"),
                      "yaw_rate_mae_degps": math.degrees(float((P["yaw_rate"] - Q["yaw_rate"]).abs()[bp].mean())) if bp.any() else float("nan"),
                      "curvature_mae_1pm": float((P["curvature"] - Q["curvature"]).abs()[bp].mean()) if bp.any() else float("nan")}
    return res


def paired(a, b, eid):
    m = np.isfinite(a) & np.isfinite(b)
    r = PB(a[m], b[m], eid[m], n_boot=2000, seed=0)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def selftest():
    rng = np.random.default_rng(0)
    v = rng.uniform(3, 15, size=(5, 1))
    t = np.arange(1, 31) * DT
    G = np.stack([v * t + 0.3 * np.sin(t), 0.4 * t ** 2 / (1 + t) * np.ones_like(v)], -1)        # [5, 30, 2] moving, curving
    c0 = comps(G, G)
    assert all(np.nanmax(np.abs(c0[k])) == 0.0 for k in ROWS), {k: float(np.nanmax(np.abs(c0[k]))) for k in ROWS}
    c1 = comps(G + np.array([0.0, 1.0]), G)
    assert np.allclose(c1["cross_mae_m"], 1.0, atol=1e-12) and np.allclose(c1["along_mae_m"], 0.0, atol=1e-12), (c1["cross_mae_m"], c1["along_mae_m"])
    c2 = comps(G + np.array([1.0, 0.0]), G)            # a constant along offset moves along by 1 m and nothing lateral at the waypoints
    assert np.allclose(c2["along_mae_m"], 1.0, atol=1e-12) and np.allclose(c2["cross_mae_m"], 0.0, atol=1e-12)
    return {"R-3": "PASS", "identical_track_all_rows_zero": True, "lateral_offset_1m": "cross 1.0, along 0.0", "along_offset_1m": "along 1.0, cross 0.0"}


if SELF:
    print(json.dumps(selftest()))
    sys.exit(0)

RUN, CHAIN, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)


def load(tag):
    fs = sorted(glob.glob(os.path.join(RUN, f"dump_{tag}", "ep*.npz")))
    assert fs, f"no dumps for {tag}"
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "eid", "clip_index", "ws")}
    d["json"] = json.load(open(os.path.join(RUN, f"t1v8f_{tag}.json"), encoding="utf-8"))
    return d


refuse = []
rep = {"prereg": "Project Steering/PREREG_V8F_T1KIN_HACC.md (323624f2) + Amendment 1 (15abe8d9)",
       "tier": "T1 (T1-kin: self-action OPEN loop; ego state = own executed motion; cameras logged; never 'closed loop')",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, on episodes; answers 'another draw of EPISODES'; the two seeds are the training replicate",
       "settings_primary": {"m": 10, "nav": "deadreckon", "scene": "logged", "limit": "logged", "schedule": "periodic"},
       "F_m": F, "two_F_m": TWO_F, "controls": {}, "seeds": {}}
rep["controls"]["R-3"] = selftest()
c4 = [ln for ln in open(CHAIN, encoding="utf-8") if " C-4 " in ln]
c4_ok = bool(c4) and " passed" in c4[-1] and "failed" not in c4[-1] and "error" not in c4[-1]
rep["controls"]["C-4"] = {"line": c4[-1].strip() if c4 else None, "PASS": c4_ok}
if not c4_ok:
    refuse.append("C-4")
sys.path.insert(0, os.path.join(ROOT, "stack"))
from tanitad.data import v8f_mini_data as Dd  # noqa: E402  (the canonical grid parser: [(ci into the join's clip list, t, 0, 0)])
wins = Dd.eval_grid_windows(os.path.join(PK, "raw", "eval_grid_windows.json"), Dd.load_join(os.path.join(PK, "raw", "join_eval141.npz")))
assert len(wins) == 2399
gidx = {(int(w[0]), int(w[1])): i for i, w in enumerate(wins)}
BANK = os.path.join(PK, "raw", "2026-10-25-hacc", "run")

for s in (0, 1):
    S = {}
    m10, m30, b10 = load(f"HACC_s{s}_m10"), load(f"HACC_s{s}_m30"), load(f"B_s{s}_m10")
    for d in (m30, b10):
        assert (d["clip_index"] == m10["clip_index"]).all() and (d["ws"] == m10["ws"]).all(), "window order differs between runs"
    # C-1 (Amendment 2): (i) m = 30 is open loop; (ii) the age-free identity: the floor equals HBLEND's banked P10 plan at the mapped windows
    bank = np.load(os.path.join(BANK, f"score_M2b_s{s}_perwindow.npz"))
    hbl = np.load(os.path.join(PK, "raw", "2026-10-25-hblend", f"score_M2b_s{s}_perwindow.npz"))
    gi = np.array([gidx[(int(c), int(w))] for c, w in zip(m30["clip_index"], m30["ws"])])
    assert (m30["eid"] == bank["eid"][gi]).all(), "episode ids differ from the banked grid"
    c1 = {"max_abs_cl_minus_ol_m": float(np.abs(m30["cl"] - m30["ol"]).max()),
          "max_abs_floor_vs_banked_P10_m": float(np.abs(m30["fl"] - hbl["P"][gi]).max()), "n_windows": int(len(gi)),
          "grid_indices_unique": bool(len(np.unique(gi)) == len(gi))}
    c1["PASS"] = c1["max_abs_cl_minus_ol_m"] <= 1e-9 and c1["max_abs_floor_vs_banked_P10_m"] <= 1e-4 and c1["grid_indices_unique"] and len(gi) == 2399
    dol = np.abs(m30["ol"] - bank["HACC"][gi]).max(axis=(1, 2))
    c1_beside = {"what": "Amendment 2 item 2, beside: |ol - banked HACC plan| per window (the eval's --ages cycle vs the T1 tick-0 age 0)",
                 "max_m": float(dol.max()), "median_m": float(np.median(dol)), "frac_windows_below_1e-4": float((dol <= 1e-4).mean())}
    # C-2: the floor reproduces its banked open-loop number
    fl6 = float(np.linalg.norm(m30["fl"] - m30["g"], axis=-1).mean())
    c2 = {"floor_m30_ade6_m": fl6, "banked_m": FLOOR_OL_ADE6, "PASS": abs(fl6 - FLOOR_OL_ADE6) <= 1e-5}
    # C-3: the post-map is in the loop
    c3v = float(np.abs(m10["cl"] - b10["cl"]).max())
    c3 = {"max_abs_HACC_minus_B_executed_m10_m": c3v, "PASS": c3v > 1e-2}
    # R-1 / R-2: the reader's rows are the canonical ones
    A, B = comps(m10["cl"], m10["g"]), comps(m10["fl"], m10["g"])
    ff = m10["json"]["arms"]
    lat_cross = [k for k in ff["cl"]["four_families"]["lateral"] if "cross" in k and "mae" in k]
    r1 = {}
    for arm, C in (("cl", A), ("fl", B)):
        lo = ff[arm]["four_families"]["longitudinal"]
        la = ff[arm]["four_families"]["lateral"]
        r1[arm] = {"speed_mae_mps": [float(C["speed_mae_mps"].mean()), lo["speed_mae_mps"]],
                   "along_mae_m": [float(C["along_mae_m"].mean()), lo["along_mae_m"]],
                   "accel_mae_mps2": [float(C["accel_mae_mps2"].mean()), lo["accel_mae_mps2"]]}
        if lat_cross:
            r1[arm]["cross_mae_m"] = [float(C["cross_mae_m"].mean()), la[lat_cross[0]]]
        for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm"):
            if isinstance(la.get(k), (int, float)):
                r1[arm][k] = [C["_pooled"][k], la[k]]
    r1_ok = all(abs(a - b) <= 6e-5 for arm in r1.values() for a, b in arm.values() if isinstance(b, (int, float)))
    eid = m10["eid"]
    p6 = paired(A["ade6_m"], B["ade6_m"], eid)
    js6 = m10["json"]["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"]
    r2 = {"reader_delta": p6["delta"], "analyze_delta": js6["delta"], "PASS": abs(p6["delta"] - float(js6["delta"])) <= 1e-6}
    S["C-1_beside_age_effect"] = c1_beside
    S["controls"] = {"C-1": c1, "C-2": c2, "C-3": c3, "R-1": {"PASS": r1_ok, "rows": r1, "tolerance": "6e-5 (four_families rounds to 4 dp)"}, "R-2": r2}
    for k, v in S["controls"].items():
        if not v["PASS"]:
            refuse.append(f"s{s}:{k}")
    # PRIMARY / DOMINANCE at m = 10
    rows = {k: paired(A[k], B[k], eid) for k in ROWS}
    for k in rows:
        rows[k]["family"] = FAMILY[k]
        rows[k]["separated_worse"] = rows[k]["separated"] and rows[k]["delta"] > 0
    prim = rows["ade6_m"]["delta"] <= -TWO_F and rows["ade6_m"]["hi"] < 0
    worse = [k for k in ROWS if rows[k]["separated_worse"]]
    S["m10_HACC_minus_P10"] = rows
    S["PRIMARY"] = {"met": bool(prim), "ade6": rows["ade6_m"], "bar": "point <= -2F = -0.2415 m AND the interval excludes 0"}
    S["DOMINANCE"] = {"met": bool(prim and not worse), "separated_worse_rows": worse}
    S["not_computed"] = {
        "TACTICAL": "NOT COMPUTED: the T1 dumps record no per-tick T code (Amendment 1 item 2); work item: dump the per-tick code",
        "STRATEGIC": "NOT COMPUTED: the T1 dumps record no per-tick S checkpoints (Amendment 1 item 2); work item: dump the per-tick checkpoints",
        "DISTANCE_KEEPING": "NOT COMPUTED: v8f_t1_eval passes no lead track to analyze(); four_families.longitudinal.distance_keeping in its output: "
                            + str(ff["cl"]["four_families"]["longitudinal"].get("distance_keeping", {}).get("status", "absent"))}
    # beside (decides nothing)
    bes = {}
    for tag in (f"HACC_s{s}_m30", f"HACC_s{s}_m5", f"HACC_s{s}_m10_frozen", f"HACC_s{s}_m10_limoff", f"B_s{s}_m10"):
        try:
            d = load(tag)
        except (AssertionError, FileNotFoundError) as e:
            bes[tag] = {"status": f"absent: {e}"}
            continue
        Ad, Bd = comps(d["cl"], d["g"]), comps(d["fl"], d["g"])
        e = {"model_minus_floor_ade6": paired(Ad["ade6_m"], Bd["ade6_m"], d["eid"]),
             "model_minus_floor_accel": paired(Ad["accel_mae_mps2"], Bd["accel_mae_mps2"], d["eid"]),
             "envelope_clipped_windows": d["json"]["t1v8f"]["windows_with_envelope_clip"],
             "t1_minus_open_ade": d["json"]["paired_decision_grade"]["paired_t1_minus_open"]["ade_m"],
             "settings": {k: d["json"]["t1v8f"][k] for k in ("replan_steps", "scene", "nav", "limit", "schedule")}}
        if (d["clip_index"] == m10["clip_index"]).all() and (d["ws"] == m10["ws"]).all() and tag != f"HACC_s{s}_m30" and tag != f"HACC_s{s}_m5":
            e["this_minus_default_HACC_m10_ade6"] = paired(Ad["ade6_m"], A["ade6_m"], d["eid"])
        bes[tag] = e
    bes["HACC_minus_B_m10_ade6"] = paired(A["ade6_m"], comps(b10["cl"], b10["g"])["ade6_m"], eid)
    bes["envelope_clipped_windows_m10"] = m10["json"]["t1v8f"]["windows_with_envelope_clip"]
    bes["mean_ade6_m10"] = {"HACC": float(A["ade6_m"].mean()), "P10": float(B["ade6_m"].mean())}
    S["beside"] = bes
    rep["seeds"][f"s{s}"] = S

if refuse:
    rep["OUTCOME"] = "REFUSED"
    rep["refused_controls"] = refuse
else:
    P = [rep["seeds"][f"s{s}"]["PRIMARY"]["met"] for s in (0, 1)]
    Dm = [rep["seeds"][f"s{s}"]["DOMINANCE"]["met"] for s in (0, 1)]
    if all(Dm):
        rep["OUTCOME"] = "T1 DOMINATES"
    elif all(P):
        rep["OUTCOME"] = "T1 BEATS THE FLOOR"
    elif not all(P):
        rep["OUTCOME"] = "MARGIN COLLAPSES"
    else:
        rep["OUTCOME"] = "MIXED"
json.dump(rep, open(os.path.join(OUT, "T1K_READ.json"), "w", encoding="utf-8"), indent=1)
print("OUTCOME", rep["OUTCOME"], "refused:", refuse)
