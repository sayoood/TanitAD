"""nav_feed_analyze.py -- read the nav-feed run and write NAV_FEED.json + NAV_FEED.md with the COMMITTED label (PREREG_NAV_FEED.md).

CPU only.  Reads the dumps the wrapper produced (``<out>/dump_<arm>_s<seed>/``, arm in clip | perframe_token | perframe_full, seed 0 | 1), the
sealed window selection and the archived step40000ema_unc dump (reference for controls C2 / C3).  Writes NAV_FEED.{json,md} next to the dumps.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 nav_feed_analyze.py \
        --out-dir C:/Users/Admin/refcv8_navfeed_out --windows <hold>/raw/windows_navfeed.json --prereg <hold>/PREREG_NAV_FEED.md

It REFUSES to run unless ``PREREG_NAV_FEED.md.sha256`` matches the bytes of ``PREREG_NAV_FEED.md`` (the seal) -- a label is only a label
if the rule was committed before the numbers.  Every number is T1 (self-action OPEN LOOP on logged eval139 frames; the nav inputs are
ORACLE-derived: the v9 per-frame token / args are built offline from the ego's future path) -- never "driving performance".

ESTIMATOR.  ``taniteval.ci.paired_episode_cluster_bootstrap``, B 2000, seed 0, over the episodes present in the set.  It answers "would
another draw of EPISODES say this?" at a FIXED checkpoint and a FIXED inference seed; it is blind to another TRAINING run (H-ESTIM-SEED-1)
and to another INFERENCE run -- so the two inference seeds are read SEPARATELY, never pooled, and the seed floor gates the control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAUNCH = Path(os.environ.get("R8_LAUNCH_TREE", "C:/lgs/Lfb47a58376"))
ARCHIVE_DUMP = "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc"       # root holding dump_s0 / dump_s1
ARMS = ("clip", "perframe_token", "perframe_full")
SEEDS = (0, 1)
N_BOOT = 2000
BOOT_SEED = 0
# ---- COMMITTED RULE CONSTANTS (PREREG_NAV_FEED.md sec 6) -------------------------------------------------------------------------------
PRIMARY_METRIC = "ade_6s"
FLOOR_MULT = 2.0
LABELS = ("FEED-FIX-MATTERS", "FEED-FIX-NULL", "MIXED")
# model slots 5,10,15,20,30,40,50,60 steps @0.1 s  ->  GT index h-1 ; S6 grid = instants 1..6 s
SLOT_STEPS = [5, 10, 15, 20, 30, 40, 50, 60]
SLOT_IDX = [h - 1 for h in SLOT_STEPS]
S6_SLOTS = [1, 3, 4, 5, 6, 7]                         # plan slots at 1,2,3,4,5,6 s
S6_GT_IDX = [9, 19, 29, 39, 49, 59]
TIER = ("T1 self-action OPEN LOOP on logged eval139 frames (NOT driving performance); the nav token / args are ORACLE-derived "
        "(v9 labels built offline from the ego's future path)")
ESTIMATOR = ("paired episode-cluster bootstrap (taniteval.ci.paired_episode_cluster_bootstrap), B 2000, seed 0, over the episodes present in "
             "the set: answers 'another draw of EPISODES' at a fixed checkpoint and a fixed inference seed; NOT another training run "
             "(H-ESTIM-SEED-1), NOT another inference run (seeds 0 and 1 are shown separately, never pooled)")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _bootstrap():
    for p in (str(LAUNCH / "stack"), str(LAUNCH / "taniteval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import taniteval.ci as ci
    return ci


def load_ra():
    """refav1_arm by path (its own bootstrap preflights taniteval): the programme's canonical per-window components."""
    for p in (str(LAUNCH / "stack"), str(LAUNCH / "taniteval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = LAUNCH / "taniteval" / "tools" / "refav1_arm.py"
    spec = importlib.util.spec_from_file_location("refav1_arm_for_navfeed", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refav1_arm_for_navfeed"] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================================================== dump reading
def ego_frame(gt_xy, pose_last):
    """World-frame GT [n,K,2] -> the ego frame of ``pose_last`` (x, y, yaw, v): the trace agent's transform (nav_natexp.ego_frame)."""
    c, s = np.cos(pose_last[:, 2]), np.sin(pose_last[:, 2])
    dx, dy = gt_xy[..., 0] - pose_last[:, None, 0], gt_xy[..., 1] - pose_last[:, None, 1]
    return np.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy], -1)


def load_dump(path) -> dict:
    """One dump -> flat per-window arrays, keyed by (episode_index, ws)."""
    d = Path(path)
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    cols = {k: [] for k in ("episode_index", "ws", "plan", "gt", "gt_valid", "pose_last", "v0", "g2s", "os2s", "nav_cmd", "nav_cmd_clip",
                            "r8_nav_known", "r8_nav_fed", "lat_pred", "lon_pred", "route_pred", "lat_label", "lon_label", "route_label")}
    has_clip = has_known = None
    for e in man["episodes"]:
        fi = int(e["file_index"])
        z = np.load(d / f"ep{fi:03d}.npz", allow_pickle=False)
        dec = np.load(d / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        n = len(dec["ws"])
        assert np.array_equal(z["ws"].astype(np.int64), dec["ws"].astype(np.int64))
        cols["episode_index"].append(np.full(n, int(e["episode_index"]), dtype=np.int64))
        cols["ws"].append(dec["ws"].astype(np.int64))
        cols["plan"].append(dec["plan_full_nav_true"].reshape(n, len(SLOT_STEPS), 2).astype(np.float64))
        cols["gt"].append(dec["gt_future_ext"].astype(np.float64))
        cols["gt_valid"].append(dec["gt_future_valid_ext"] > 0.5)
        cols["pose_last"].append(dec["pose_last"].astype(np.float64))
        cols["v0"].append(z["v0"].astype(np.float64).reshape(n))
        cols["g2s"].append(z["g"].astype(np.float64))
        cols["os2s"].append(z["os"].astype(np.float64))
        cols["nav_cmd"].append(dec["nav_cmd"].astype(np.int64))
        has_clip = "nav_cmd_clip" in dec.files
        has_known = "r8_nav_known" in dec.files
        cols["nav_cmd_clip"].append(dec["nav_cmd_clip"].astype(np.int64) if has_clip else np.full(n, -1, dtype=np.int64))
        cols["r8_nav_known"].append(dec["r8_nav_known"].astype(bool) if has_known else np.zeros(n, dtype=bool))
        cols["r8_nav_fed"].append(dec["r8_nav_fed"].reshape(n, 6).astype(np.float32) if "r8_nav_fed" in dec.files
                                  else np.zeros((n, 6), np.float32))
        cols["lat_pred"].append(dec["lat_pred_nav_true"].astype(np.int64))
        cols["lon_pred"].append(dec["lon_pred_nav_true"].astype(np.int64))
        cols["route_pred"].append(dec["route_pred_nav_true"].astype(np.int64))
        cols["lat_label"].append(dec["lat_label"].astype(np.int64))
        cols["lon_label"].append(dec["lon_label"].astype(np.int64))
        cols["route_label"].append(dec["route_label"].astype(np.int64))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["manifest"] = {"n_windows": int(len(out["ws"])), "nav_feed": man.get("nav_feed"), "inference_seed": man["inference_seed"]["seed"],
                       "model_step": (man.get("model") or {}).get("step"), "ckpt": (man.get("model") or {}).get("ckpt"),
                       "manifest_sha256": sha256_file(d / "manifest.json")}
    out["has_nav_feed_sidecar"] = bool(has_clip)
    out["key"] = list(zip(out["episode_index"].tolist(), out["ws"].tolist()))
    return out


# =============================================================================================================== per-window metrics
def gt_class(v0, v6, valid):
    """S6 GT longitudinal class (s6_lon_decomp.py:18, first matching rule wins); NaN-safe via ``valid``."""
    dv = v6 - v0
    cls = np.full(v0.shape, "unk", dtype=object)
    rules = [("start_from_stop", (v0 < 1.0) & (v6 >= 2.0)), ("stopping", (v0 >= 2.0) & (v6 < 1.0)), ("slow_both", (v0 < 2.0) & (v6 < 2.0)),
             ("accelerating", dv >= 1.5), ("decelerating", dv <= -1.5), ("cruising", np.abs(dv) < 1.5)]
    taken = np.zeros(v0.shape, dtype=bool)
    for name, m in rules:
        sel = m & ~taken & valid
        cls[sel] = name
        taken |= m
    return cls


def plan_class(P6, v0, dt=1.0):
    """decel / cruise / accel from the plan's speed over its LAST second minus the measured v0, +-1.5 m/s (s6_lon_decomp.plan_lon_class)."""
    v_end = np.linalg.norm(P6[:, -1] - P6[:, -2], axis=-1) / dt
    dv = v_end - v0
    return np.where(dv <= -1.5, "decel", np.where(dv >= 1.5, "accel", "cruise")).astype(object)


def window_metrics(dump, ra, ff):
    import torch
    n = len(dump["ws"])
    gt_ego = ego_frame(dump["gt"][:, SLOT_IDX, :2], dump["pose_last"])
    gv = dump["gt_valid"][:, SLOT_IDX]
    P = dump["plan"]
    err = np.linalg.norm(P - gt_ego, axis=-1)
    m = {"ade_2s": err[:, :4].mean(1),
         "ade_6s": np.where(gv, err, 0.0).sum(1) / np.maximum(gv.sum(1), 1)}
    all6 = dump["gt_valid"][:, S6_GT_IDX].all(1)
    P6 = P[:, S6_SLOTS]
    G6 = ego_frame(dump["gt"][:, S6_GT_IDX, :2], dump["pose_last"])
    comp = ra._components(P6.astype(np.float32), G6.astype(np.float32), 1.0)
    pt, gt = torch.as_tensor(P6).float(), torch.as_tensor(G6).float()
    Pg, Gg = ff._seq_geometry(pt, 1.0), ff._seq_geometry(gt, 1.0)
    both_last = (Pg["valid"] & Gg["valid"])[:, -1]
    dh = (Pg["heading"][:, -1] - Gg["heading"][:, -1] + math.pi) % (2 * math.pi) - math.pi
    head_final = torch.where(both_last, dh.abs() * 180.0 / math.pi, torch.full_like(dh, float("nan"))).numpy()
    along_signed = (Pg["along"][:, -1] - Gg["along"][:, -1]).numpy()
    v0 = dump["pose_last"][:, 3]
    v6 = dump["gt"][:, 59, 3]
    gcls = gt_class(v0, v6, all6)
    pcls = plan_class(P6, v0)
    cruise = gcls == "cruising"

    def s6(x):
        x = np.asarray(x, dtype=np.float64).copy()
        x[~all6] = np.nan
        return x
    m.update({
        "LON_speed_mae_mps": s6(comp["LON_speed_mae_mps"]),
        "LON_along_abs_final_m": s6(np.abs(along_signed)),
        "LON_along_signed_final_m": s6(along_signed),
        "LON_false_decel_on_gt_cruise": np.where(cruise, (pcls == "decel").astype(np.float64), np.nan),
        "LAT_cross_abs_final_m": s6((Pg["cross"][:, -1] - Gg["cross"][:, -1]).abs().numpy()),
        "LAT_cross_mae_m": s6(comp["LAT_cross_mae_m"]),
        "LAT_heading_err_final_deg": s6(head_final),
        "LAT_heading_mae_deg": s6(comp["LAT_heading_mae_deg"]),
        "TAC_traj_lat_correct": s6(comp["TAC_traj_lat_correct"]),
        "TAC_traj_lon_correct": s6(comp["TAC_traj_lon_correct"]),
    })
    def acc(pred, label):
        ok = (label != -100) & (pred >= 0)
        return np.where(ok, (pred == label).astype(np.float64), np.nan)
    m["TAC_head_lat_acc"] = acc(dump["lat_pred"], dump["lat_label"])
    m["TAC_head_lon_acc"] = acc(dump["lon_pred"], dump["lon_label"])
    m["STR_route_acc"] = acc(dump["route_pred"], dump["route_label"])
    m["_frame_check_max_abs_m"] = float(np.abs(gt_ego[:, :4] - dump["g2s"]).max())
    m["_os_vs_plan_max_abs_m"] = float(np.abs(dump["os2s"] - P[:, :4]).max())
    m["_gt_class"] = gcls
    return m


METRIC_FAMILY = [
    ("ade_2s", "ADE", "m"), ("ade_6s", "ADE", "m"),
    ("LON_speed_mae_mps", "LONGITUDINAL", "m/s"), ("LON_along_abs_final_m", "LONGITUDINAL", "m"),
    ("LON_along_signed_final_m", "LONGITUDINAL", "m (signed; <0 = plans short/slow)"),
    ("LON_false_decel_on_gt_cruise", "LONGITUDINAL", "rate"),
    ("LAT_cross_abs_final_m", "LATERAL", "m"), ("LAT_heading_err_final_deg", "LATERAL", "deg"),
    ("LAT_cross_mae_m", "LATERAL", "m"), ("LAT_heading_mae_deg", "LATERAL", "deg"),
    ("TAC_head_lat_acc", "TACTICAL", "accuracy"), ("TAC_head_lon_acc", "TACTICAL", "accuracy"),
    ("TAC_traj_lat_correct", "TACTICAL", "accuracy"), ("TAC_traj_lon_correct", "TACTICAL", "accuracy"),
    ("STR_route_acc", "STRATEGIC", "accuracy (the route head may ECHO the fed nav: read with the nav-null controls)"),
]


# =============================================================================================================== statistics
def paired(ci, a, b, eid, n_boot=N_BOOT):
    a, b, eid = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(eid)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or len(np.unique(eid[ok])) < 3:
        return {"n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(eid[ok]))), "note": "too few windows/episodes"}
    r = ci.paired_episode_cluster_bootstrap(a[ok], b[ok], eid[ok].tolist(), n_boot=n_boot, seed=BOOT_SEED)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]), "mean_a": float(a[ok].mean()), "mean_b": float(b[ok].mean()),
            "degenerate": bool(r.get("degenerate", False)), "n_dropped_nonfinite": int((~ok).sum())}


def decide(prim: dict, floor: dict, control: dict) -> dict:
    """The COMMITTED rule (PREREG sec 6).  ``prim[s]`` = paired cell perframe_full - clip, ADE 6 s, MISMATCH set; ``control[s]`` = |mean
    perframe_token - clip| on the CONTROL set; ``floor`` = F (the inference-seed floor).  Pure function: the unit tests pin it."""
    s0, s1 = prim.get(0), prim.get(1)
    if not all(isinstance(x, dict) and "delta" in x for x in (s0, s1)):
        return {"label": "NOT-EVALUABLE", "why": "a primary cell is absent for at least one inference seed", "control_ok": None}
    sep_neg = [bool(x["hi"] < 0) for x in (s0, s1)]
    sep_any = [bool(x["lo"] > 0 or x["hi"] < 0) for x in (s0, s1)]
    F = floor.get("F")
    if F is None or not all(isinstance(control.get(s), dict) and "abs_delta" in control[s] for s in SEEDS):
        ctrl_ok = None
    else:
        ctrl_ok = bool(all(control[s]["abs_delta"] <= FLOOR_MULT * F for s in SEEDS))
    # an effect no larger than the INFERENCE-seed floor is not an effect (CLAUDE.md, H-ESTIM-SEED-1): MATTERS needs |delta| > F at BOTH seeds
    over_floor = [bool(F is not None and abs(x["delta"]) > F) for x in (s0, s1)]
    if all(sep_neg) and ctrl_ok is True and all(over_floor):
        label = "FEED-FIX-MATTERS"
    elif not any(sep_any):
        label = "FEED-FIX-NULL"
    else:
        label = "MIXED"
    why = {"separated_negative_by_seed": sep_neg, "separated_any_by_seed": sep_any, "control_ok": ctrl_ok, "abs_delta_over_floor_by_seed": over_floor}
    admissible = ctrl_ok is True
    return {"label": label, **why, "label_admissible": admissible,
            "admissibility_note": (None if admissible else
                                   "CONTROL NOT PASSED (or not evaluable): the instrument did not read ~0 where nothing differs, so the label "
                                   "is the literal rule's output and NOT an admissible result")}


# =============================================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="C:/Users/Admin/refcv8_navfeed_out")
    ap.add_argument("--windows", default=str(HERE.parent / "raw" / "windows_navfeed.json"))
    ap.add_argument("--prereg", default=str(HERE.parent / "PREREG_NAV_FEED.md"))
    ap.add_argument("--archive", default=ARCHIVE_DUMP)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--result-stem", default="NAV_FEED")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out_dir)
    # ---- the seal ---------------------------------------------------------------------------------------------------------------------
    seal_p = Path(a.prereg + ".sha256")
    seal = {"prereg": a.prereg, "sealed_sha256": None, "actual_sha256": None, "ok": False}
    if seal_p.exists() and Path(a.prereg).exists():
        seal["sealed_sha256"] = seal_p.read_text(encoding="utf-8").split()[0].strip().lower()
        seal["actual_sha256"] = sha256_file(a.prereg)
        seal["ok"] = seal["sealed_sha256"] == seal["actual_sha256"] and len(seal["actual_sha256"]) == 64
    if not seal["ok"] and os.environ.get("NF_ANALYZE_TEST") != "1":
        raise SystemExit(f"[nav_feed_analyze] REFUSING: the prereg seal does not verify ({seal}). A label needs a rule committed before the numbers.")
    ci = _bootstrap()
    ra = load_ra()
    from taniteval import four_families as ff
    win = json.load(open(a.windows, encoding="utf-8"))
    sets = {"mismatch": {}, "control": {}}
    for e in win["episodes"]:
        for kind in ("mismatch", "control"):
            for w in e[f"{kind}_ws"]:
                sets[kind][(int(e["episode_index"]), int(w))] = e
    mism_v9 = {(e["episode_index"], w): (f, v) for e in win["episodes"] for w, f, v in zip(e["mismatch_ws"], e["mismatch_fed"], e["mismatch_v9"])}
    ctrl_v9 = {(e["episode_index"], w): v for e in win["episodes"] for w, v in zip(e["control_ws"], e["control_v9"])}

    dumps, mets, missing = {}, {}, []
    for arm in ARMS:
        for s in SEEDS:
            p = out / f"dump_{arm}_s{s}"
            if not (p / "manifest.json").exists():
                missing.append(f"{arm}_s{s}")
                continue
            dumps[(arm, s)] = load_dump(p)
            mets[(arm, s)] = window_metrics(dumps[(arm, s)], ra, ff)

    # ---- alignment: every dump holds exactly the union of the two sets, in the same order -------------------------------------------------
    union = set(sets["mismatch"]) | set(sets["control"])
    controls = {"missing_dumps": missing, "seal": seal}
    ref_key = None
    align_ok = True
    for k, d in dumps.items():
        keys = d["key"]
        if set(keys) != union or len(keys) != len(union):
            align_ok = False
            controls.setdefault("alignment_problems", []).append({"dump": f"{k[0]}_s{k[1]}", "n": len(keys), "expected": len(union),
                                                                   "missing": len(union - set(keys)), "extra": len(set(keys) - union)})
        if ref_key is None:
            ref_key = keys
        elif keys != ref_key:
            align_ok = False
            controls.setdefault("alignment_problems", []).append({"dump": f"{k[0]}_s{k[1]}", "problem": "window ORDER differs from the first dump"})
    controls["windows_identical_across_dumps_and_equal_to_the_sealed_union"] = bool(align_ok and dumps)

    def mask(keys, kind):
        return np.array([k in sets[kind] for k in keys])

    # ---- C3 / token controls --------------------------------------------------------------------------------------------------------------
    c3, tok = {}, {}
    for (arm, s), d in dumps.items():
        keys = d["key"]
        nm = f"{arm}_s{s}"
        c3[nm] = {"gt_ego_frame_vs_banked_g_max_abs_m": mets[(arm, s)]["_frame_check_max_abs_m"],
                  "os_2s_vs_plan_full_first4_max_abs_m": mets[(arm, s)]["_os_vs_plan_max_abs_m"]}
        exp_pf = np.array([mism_v9[k][1] if k in mism_v9 else ctrl_v9[k] for k in keys])
        exp_clip = np.array([mism_v9[k][0] if k in mism_v9 else ctrl_v9[k] for k in keys])      # control: fed == v9 by construction
        if arm == "clip":
            tok[nm] = {"nav_cmd_equals_the_banked_clip_token": bool(np.array_equal(d["nav_cmd"], exp_clip)),
                       "manifest_has_no_nav_feed_block": d["manifest"]["nav_feed"] is None}
        else:
            tok[nm] = {"nav_cmd_equals_the_v9_per_frame_token": bool(np.array_equal(d["nav_cmd"], exp_pf)),
                       "nav_cmd_clip_equals_the_banked_clip_token": bool(np.array_equal(d["nav_cmd_clip"], exp_clip)),
                       "manifest_mode_matches": (d["manifest"]["nav_feed"] or {}).get("mode") == arm,
                       "r8_nav_fed_nonzero_rows_equal_known_rows": (
                           bool(np.array_equal(np.abs(d["r8_nav_fed"]).sum(1) > 0, d["r8_nav_known"])) if arm == "perframe_full"
                           else bool(not np.any(d["r8_nav_fed"])))}
    controls["C3_gt_and_dump_consistency"] = c3
    controls["token_feed_as_designed"] = tok

    # ---- C2: re-roll vs the banked battery `os` on the same windows ----------------------------------------------------------------------
    c2 = {}
    banked = {}
    for s in SEEDS:
        bp = Path(a.archive) / f"dump_s{s}"
        if (bp / "manifest.json").exists():
            bd = load_dump(bp)
            idx = {k: i for i, k in enumerate(bd["key"])}
            if ref_key is not None and all(k in idx for k in ref_key):
                sel = np.array([idx[k] for k in ref_key])
                sub = {kk: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(bd["key"]) else v) for kk, v in bd.items()}
                banked[s] = window_metrics(sub, ra, ff)
    if banked:
        F_b = None
        if 0 in banked and 1 in banked:
            F_b = abs(float(np.nanmean(banked[1]["ade_6s"]) - np.nanmean(banked[0]["ade_6s"])))
        for s in SEEDS:
            if ("clip", s) in mets and s in banked:
                c2[f"clip_s{s}_vs_banked_os_s{s}"] = {
                    "mean_ade_6s_clip_reroll": float(np.mean(mets[("clip", s)]["ade_6s"])),
                    "mean_ade_6s_banked_os": float(np.mean(banked[s]["ade_6s"])),
                    "abs_diff_m": abs(float(np.mean(mets[("clip", s)]["ade_6s"]) - np.mean(banked[s]["ade_6s"]))),
                    "banked_seed_replicate_abs_diff_m": F_b}
    controls["C2_clip_reroll_reproduces_the_banked_battery"] = c2

    # ---- the inference-seed floor F, on the POOLED windows, clip arm, ADE 6 s -------------------------------------------------------------
    floor = {}
    if ("clip", 0) in mets and ("clip", 1) in mets:
        eid = dumps[("clip", 0)]["episode_index"]
        cell = paired(ci, mets[("clip", 1)][PRIMARY_METRIC], mets[("clip", 0)][PRIMARY_METRIC], eid, a.n_boot)
        if "delta" in cell:
            half = (cell["hi"] - cell["lo"]) / 2.0
            floor = {"metric": PRIMARY_METRIC, "cell_clip_s1_minus_clip_s0": cell, "abs_point_delta": abs(cell["delta"]),
                     "ci95_half_width": half, "F": max(abs(cell["delta"]), half),
                     "definition": "F = max(|point delta|, 95% half-width) of clip(seed 1) - clip(seed 0), ADE 6 s, all sealed windows"}
    controls["inference_seed_floor"] = floor

    # ---- the cells ------------------------------------------------------------------------------------------------------------------------
    cmp_pairs = {"perframe_token_minus_clip": ("perframe_token", "clip"), "perframe_full_minus_clip": ("perframe_full", "clip"),
                 "perframe_full_minus_perframe_token": ("perframe_full", "perframe_token")}
    cells = {}
    for kind in ("mismatch", "control"):
        for cname, (A, B) in cmp_pairs.items():
            for s in SEEDS:
                if (A, s) not in mets or (B, s) not in mets:
                    continue
                m = mask(dumps[(A, s)]["key"], kind)
                eid = dumps[(A, s)]["episode_index"][m]
                row = {}
                for met, fam, unit in METRIC_FAMILY:
                    row[met] = {"family": fam, "unit": unit, **paired(ci, mets[(A, s)][met][m], mets[(B, s)][met][m], eid, a.n_boot)}
                cells.setdefault(kind, {}).setdefault(cname, {})[f"s{s}"] = row
    # ---- control C1 and the primary --------------------------------------------------------------------------------------------------------
    prim = {s: cells.get("mismatch", {}).get("perframe_full_minus_clip", {}).get(f"s{s}", {}).get(PRIMARY_METRIC) for s in SEEDS}
    ctrl_cells = {}
    for s in SEEDS:
        c = cells.get("control", {}).get("perframe_token_minus_clip", {}).get(f"s{s}", {}).get(PRIMARY_METRIC)
        if c and "delta" in c:
            bit_identical = False
            if ("perframe_token", s) in mets:
                m = mask(dumps[("perframe_token", s)]["key"], "control")
                bit_identical = bool(np.array_equal(dumps[("perframe_token", s)]["plan"][m], dumps[("clip", s)]["plan"][m]))
            ctrl_cells[s] = {"abs_delta": abs(c["delta"]), "cell": c, "plans_bit_identical_to_clip": bit_identical}
    controls["C1_token_arm_on_matched_windows_must_read_zero"] = {str(s): v for s, v in ctrl_cells.items()}
    verdict = decide(prim, floor, ctrl_cells)
    for s in SEEDS:
        if isinstance(prim.get(s), dict) and "delta" in prim[s] and floor.get("F"):
            verdict.setdefault("effect_over_floor", {})[f"s{s}"] = abs(prim[s]["delta"]) / floor["F"]

    # ---- exploratory split (descriptive; NOT an input to the label) -------------------------------------------------------------------------
    split = {}
    for s in SEEDS:
        if ("perframe_full", s) in mets and ("clip", s) in mets:
            d = dumps[("perframe_full", s)]
            keys = d["key"]
            names = {(0, 1): "FOLLOW->LEFT", (0, 2): "FOLLOW->RIGHT", (1, 0): "LEFT->FOLLOW", (2, 0): "RIGHT->FOLLOW",
                     (1, 2): "LEFT->RIGHT (opposite)", (2, 1): "RIGHT->LEFT (opposite)"}
            grp = np.array([names.get(tuple(mism_v9[k]), "other") if k in mism_v9 else "control" for k in keys])
            for g in sorted(set(grp) - {"control"}):
                m = grp == g
                split.setdefault(g, {})[f"s{s}"] = paired(ci, mets[("perframe_full", s)][PRIMARY_METRIC][m], mets[("clip", s)][PRIMARY_METRIC][m],
                                                           d["episode_index"][m], a.n_boot)
    result = {
        "tool": "nav_feed_analyze.py", "when": time.strftime("%FT%T%z"), "tier": TIER, "estimator": ESTIMATOR,
        "evidence_class": "MEASURED (this run's dumps under out_dir; every cell cites n windows / n episodes)",
        "committed_label": verdict, "primary_cells_perframe_full_minus_clip_ade6s_mismatch": {f"s{s}": prim[s] for s in SEEDS},
        "controls": controls, "cells": cells, "exploratory_split_by_fed_to_v9_ade6s": split,
        "families_not_computable": {
            "LONGITUDINAL distance-keeping (headway / time-gap / TTC to the lead agent)": (
                "NOT COMPUTED here: the dumps carry no lead-block join (lead_block_common_grid needs the B1 lead block on a 0.2 s grid "
                "and the common instants with this 0.5 s plan grid); the v9 release's lead_gap columns are labels, not a plan-vs-GT "
                "distance-keeping instrument. Work item: join the B1 lead block to these dumps (taniteval.lead_metrics)."),
            "STRATEGIC route / goal-setting beyond the route-head accuracy": "NOT COMPUTED: needs the g_str goal-point head decode (not dumped as a row here)"},
        "inputs": {"out_dir": str(out), "windows": a.windows, "windows_sha256": sha256_file(a.windows), "n_boot": a.n_boot,
                   "dumps": {f"{k[0]}_s{k[1]}": v["manifest"] for k, v in dumps.items()},
                   "analysis_script_sha256": sha256_file(__file__), "wall_s": round(time.time() - t0, 1)},
    }
    stem = a.result_stem
    json.dump(result, open(out / f"{stem}.json", "w", encoding="utf-8"), indent=1, default=str)
    (out / f"{stem}.md").write_text(render_md(result), encoding="utf-8")
    print(json.dumps({"label": verdict["label"], "admissible": verdict.get("label_admissible"), "out": str(out / f'{stem}.json')}, indent=1))
    return 0


def _f(x, nd=4):
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def _cell(c):
    if not isinstance(c, dict) or "delta" not in c:
        return f"n/a ({(c or {}).get('note', 'absent')})"
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n {c['n_windows']}/{c['n_episodes']}"
            + (" **sep**" if c["separated"] else ""))


def render_md(r: dict) -> str:
    v = r["committed_label"]
    L = ["# NAV_FEED -- does feeding refcv8 the nav it was TRAINED on change the plan? (EMA-40k, os only)", "",
         f"**COMMITTED LABEL: `{v['label']}`**" + ("" if v.get("label_admissible") else "  --  **NOT ADMISSIBLE (" + str(v.get("admissibility_note")) + ")**"), "",
         f"tier: {r['tier']}", f"estimator: {r['estimator']}", f"evidence class: {r['evidence_class']}", "",
         f"rule inputs: separated-negative by seed {v.get('separated_negative_by_seed')}, separated (any sign) by seed "
         f"{v.get('separated_any_by_seed')}, control ok {v.get('control_ok')}, |delta| > F by seed {v.get('abs_delta_over_floor_by_seed')}, "
         f"effect/floor {v.get('effect_over_floor')}", ""]
    fl = r["controls"].get("inference_seed_floor") or {}
    L += ["## Primary: perframe_full - clip, ADE 6 s (m, lower = better), MISMATCH set", "",
          "| seed | delta [95 % CI] n windows/episodes | arm mean | clip mean |", "|---|---|---|---|"]
    for s in SEEDS:
        c = r["primary_cells_perframe_full_minus_clip_ade6s_mismatch"].get(f"s{s}")
        L.append(f"| {s} | {_cell(c)} | {_f((c or {}).get('mean_a'))} | {_f((c or {}).get('mean_b'))} |")
    L += ["", f"inference-seed floor F (clip s1 - clip s0, ADE 6 s, all sealed windows): {_f(fl.get('F'))} m "
          f"(|point| {_f(fl.get('abs_point_delta'))}, CI half-width {_f(fl.get('ci95_half_width'))})", "",
          "## Controls", ""]
    c1 = r["controls"].get("C1_token_arm_on_matched_windows_must_read_zero") or {}
    for s, x in c1.items():
        L.append(f"* C1 (perframe_token - clip on the 300 CONTROL windows, ADE 6 s), seed {s}: |delta| {_f(x['abs_delta'], 6)} vs 2F "
                 f"{_f(2 * fl['F']) if fl.get('F') is not None else 'n/a'} -> {'PASS' if fl.get('F') is not None and x['abs_delta'] <= 2 * fl['F'] else 'FAIL'}; "
                 f"plans bit-identical to clip: {x['plans_bit_identical_to_clip']}")
    for k in ("C2_clip_reroll_reproduces_the_banked_battery", "C3_gt_and_dump_consistency", "token_feed_as_designed"):
        L += ["", f"* {k}: `{json.dumps(r['controls'].get(k), default=str)}`"]
    L += ["", f"* windows identical across dumps and equal to the sealed union: {r['controls'].get('windows_identical_across_dumps_and_equal_to_the_sealed_union')}; "
          f"missing dumps: {r['controls'].get('missing_dumps')}", ""]
    for kind, title in (("mismatch", "MISMATCH set (clip token != per-frame token)"), ("control", "CONTROL set (tokens equal)")):
        for cname in ("perframe_full_minus_clip", "perframe_token_minus_clip", "perframe_full_minus_perframe_token"):
            blk = r["cells"].get(kind, {}).get(cname)
            if not blk:
                continue
            L += ["", f"## {title}: {cname} (A - B; error metrics lower = better, accuracy higher = better)", "",
                  "| family | metric | " + " | ".join(f"seed {s[1:]}" for s in sorted(blk)) + " |", "|---|---|" + "---|" * len(blk)]
            for met, fam, _u in METRIC_FAMILY:
                L.append(f"| {fam} | {met} | " + " | ".join(_cell(blk[s].get(met)) for s in sorted(blk)) + " |")
    L += ["", "## Exploratory split of the primary cell by (clip token -> per-frame token), perframe_full - clip, ADE 6 s (NOT an input to the label)", "",
          "| group | seed 0 | seed 1 |", "|---|---|---|"]
    for g, x in r["exploratory_split_by_fed_to_v9_ade6s"].items():
        L.append(f"| {g} | {_cell(x.get('s0'))} | {_cell(x.get('s1'))} |")
    L += ["", "## Families that cannot be computed from these dumps", ""]
    for k, why in r["families_not_computable"].items():
        L.append(f"* **{k}**: {why}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
