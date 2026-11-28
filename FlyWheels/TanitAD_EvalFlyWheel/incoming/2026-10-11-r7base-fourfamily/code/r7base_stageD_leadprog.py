"""r7base stage D (CPU only): the REGISTERED LEAD-VIOL progress control (LEAD-PROGRESS) on the refcv7-50,400 S6 panel.

Why this exists. Stage B's "progress ratio 1.108" was plan x(6 s) / human x(6 s) averaged over ALL 1,151 lead windows,
UNCLIPPED (r7base_stageB.py L98-L108). That is NOT the registered control. The registered control
(`taniteval/lead_metrics.py`: `lead_viol(..., plan_arc=, human_arc=)`, `lead_progress`, `path_arc_length`,
`lead_viol_verdict`; MM ruling 2026-10-11) is the mean over the AT-RISK windows (lead window AND human time gap >= 1.0 s)
of  plan_arc / human_arc  with POLYLINE ARC LENGTH, and the DECISION reading is `lead_progress_ratio_clipped`
(per-window ratio clipped at 1.0, then averaged) >= 0.90.

The join (lead_block_common_grid + join_lead_block) is copied call for call from r7base_stageB.py, so the
windows, the lead set and the time gaps are the SAME as the landed stage-B LEAD-VIOL row.

usage: python r7base_stageD_leadprog.py <work_root> <seed> [<seed> ...]
       <work_root> holds battery/ (r7_panel.py, r7_roll.py) and out/s6_s<seed>/ (rebuilt S6 panels, stage A);
       writes <work_root>/out/leadprog_s6_s<seed>.json
"""
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

work = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
BAT = work / "battery"
OUT = work / "out"
sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import tanitad  # noqa: E402
import r7_panel as P  # noqa: E402
import r7_roll as RR  # noqa: E402
from taniteval import lead_metrics as lm  # noqa: E402
from taniteval.ci import episode_cluster_bootstrap  # noqa: E402

print("tanitad from", tanitad.__file__, "| lead_metrics", lm.__file__, flush=True)
assert hasattr(lm, "path_arc_length") and hasattr(lm, "lead_viol_verdict"), "lead_metrics is not the registered tip"
LEAD12 = "C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz"
ra = RR.ra3()
N_BOOT = 2000
ARMS = ("os", "ha0", "ha0_ext", "ha", "stop")
ARM_NAME = {"os": "os (refcv7-50,400 deployed plan, nav_true)", "ha0": "hold-v0 (constant velocity, a=0 kappa=0)",
            "ha0_ext": "echo (constant a0 and k0 at t0)", "ha": "hold-action (a, steer at t0-1 held)",
            "stop": "STOP", "human": "human-as-plan (GT, regression control)"}


def join(panel_dir: str, man: dict):
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    fr = man["corpus"]["frames"]
    raw_off = int(fr["provider_to_raw_frame_offset"])
    dt, k = float(man["grid"]["dt_s"]), int(man["grid"]["k"])
    view, idx, meta, info = ra.lead_block_common_grid(LEAD12, dt, k)
    assert view is not None, info
    shim_dir = tempfile.mkdtemp(prefix="r7base_leadjoin_")
    shim = []
    for f in files:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = os.path.join(shim_dir, os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim.append(sf)
    lead = ra.ra.join_lead_block(shim, man, view, idx, k=info["k"], dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    return lead, cov, info, raw_off


def ci_of(x, ids):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"status": "NOT-APPLICABLE", "n": 0}
    r = episode_cluster_bootstrap(x, list(ids), reduce="mean", n_boot=N_BOOT, seed=0)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator") if k in r}


def compact(lvres: dict, eid: np.ndarray, lv: np.ndarray) -> dict:
    """The registered instrument's own numbers, verbatim, plus script-side sensitivity (labelled)."""
    pr = lvres["progress"]
    out = {k: lvres.get(k) for k in ("status", "n_windows", "n_lead_windows", "n_viol", "share", "n_at_risk",
                                     "share_of_at_risk", "n_plan_lost_lead", "share_incl_lost_lead",
                                     "n_human_tg_undefined", "n_human_tg_below_thr")}
    out["share_ci"] = lvres.get("ci")
    out["share_of_at_risk_ci"] = lvres.get("share_of_at_risk_ci")
    out["progress"] = {k: pr.get(k) for k in (
        "status", "n_at_risk", "n", "n_arc_nonfinite", "n_human_arc_degenerate", "min_progress", "min_human_arc_m",
        "decision_reading", "lead_progress_ratio", "lead_progress_ratio_clipped", "progress_ratio_of_means",
        "meets_min_progress", "meets_min_progress_mean", "overshoot_masks_crawl", "n_below_min", "reason")}
    out["progress"]["ci_unclipped_mean"] = pr.get("ci")                 # the INSTRUMENT's interval (unclipped mean)
    pw = np.asarray(lvres["_per_window"]["progress_ratio"], dtype=np.float64)
    use = np.isfinite(pw)
    out["script_side"] = {
        "_note": "computed in this script from the instrument's _per_window.progress_ratio, NOT emitted by lead_metrics",
        "clipped_ci": ci_of(np.minimum(pw[use], 1.0), eid[use]) if use.any() else None,
        "median_ratio": float(np.median(pw[use])) if use.any() else None,
        "n_ratio_gt_1": int((pw[use] > 1.0).sum()),
        "share_ratio_gt_1": float((pw[use] > 1.0).mean()) if use.any() else None,
        "mean_ratio_where_gt_1": float(pw[use][pw[use] > 1.0].mean()) if (pw[use] > 1.0).any() else None,
        "mean_clipped_deficit_where_lt_1": (float((1.0 - pw[use][pw[use] < 1.0]).mean()) if (pw[use] < 1.0).any() else None),
    }
    return out


def verdict_compact(v: dict) -> dict:
    return {k: v.get(k) for k in ("verdict", "bar_decision", "lead_progress_ratio_clipped", "lead_progress_ratio",
                                  "n_progress", "overshoot_masks_crawl", "min_progress", "reason",
                                  "baseline_lead_progress_ratio")}


def paired_compact(p: dict) -> dict:
    ar = p.get("at_risk") or {}
    pg = p.get("progress") or {}
    return {
        "x7_direction_met_at_risk": p.get("x7_direction_met_at_risk"),
        "at_risk": {k: ar.get(k) for k in ("status", "n_at_risk", "n_viol_a", "n_viol_b", "delta", "lo", "hi", "separated",
                                           "estimator", "n_episodes")},
        "progress_paired": {k: pg.get(k) for k in ("status", "n_used", "n_a_only", "n_b_only", "lead_progress_ratio_a",
                                                   "lead_progress_ratio_b", "delta", "lo", "hi", "separated")},
    }


def run_one(panel_dir: str, seed: int, other_seed_controls: dict | None):
    t0 = time.time()
    man, arrs, eid_f = P.load_panel(panel_dir)
    lead, cov, info, raw_off = join(panel_dir, man)
    cols = info["dump_cols"]
    dtv = info["dt_s"]
    state = np.asarray(lead["state"]).astype(str)
    lv = state == "LEAD"
    W = len(state)
    assert W == arrs["g"].shape[0] == len(eid_f), (W, arrs["g"].shape, len(eid_f))
    eid = np.asarray(eid_f)
    G = arrs["g"][:, cols].astype(np.float64)
    human_dk = lm.distance_keeping(G, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
    human_arc = lm.path_arc_length(G)
    res = {"label": "S6", "seed": seed, "panel_dir": panel_dir, "tier": "T1 (one-shot plan from t0, tier ruling open per panel manifest)",
           "estimator": "episode-cluster bootstrap (taniteval/ci.py), n_boot=%d, seed=0; answers 'another draw of EPISODES' only" % N_BOOT,
           "n_windows": int(W), "n_episodes": int(len(set(eid.tolist()))), "n_lead_windows": int(lv.sum()),
           "grid": {"instants_s": info["instants_s"], "dt_s": dtv, "k": info["k"], "dump_cols": [int(c) for c in cols]},
           "lead_block": LEAD12, "lead_coverage": cov["counts"],
           "arc_definition": ("path_arc_length: sum of chord lengths over (origin, wp1..wp6) on the 1 s S6 grid, ego origin implicit; "
                              "a sparse grid under-reads a curved path by the same small factor for plan and human (ratio used)"),
           "arms": {}, "verdict": {}, "paired_os_vs": {}, "verdict_os_vs": {}, "controls_internal": {}}
    # human-gap-conditioned at-risk set, for the old-ratio reproduction only
    htg = np.asarray(human_dk["_per_window"]["time_gap_min_s"] if "_per_window" in human_dk else human_dk["time_gap_min_s"],
                     dtype=np.float64)
    at_risk = lv & np.isfinite(htg) & (htg >= 1.0)
    prog_h = G[:, -1, 0]
    flags = {}
    plans = {}
    for arm in ARMS:
        plan = arrs[arm][:, cols].astype(np.float64)
        plans[arm] = plan
        dk = lm.distance_keeping(plan, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
        plan_arc = lm.path_arc_length(plan)
        lvres = lm.lead_viol(dk, human_dk, eid, lead_valid=lv, n_boot=N_BOOT, seed=0, plan_arc=plan_arc, human_arc=human_arc)
        flags[arm] = lvres
        c = compact(lvres, eid, lv)
        # reproduction of the OLD stage-B reading (plan x(6 s)/human x(6 s), unclipped, all lead windows) beside the registered one
        ratio_x = plan[:, -1, 0] / np.where(np.abs(prog_h) > 1e-6, prog_h, np.nan)
        c["old_stageB_x6s_ratio"] = {
            "what": "plan x(6 s) / human x(6 s), UNCLIPPED mean over ALL lead windows (stage B L98-L108) - NOT the registered control",
            "mean_all_lead_windows": float(np.nanmean(ratio_x[lv])) if lv.any() else None, "n": int(lv.sum()),
            "mean_at_risk_windows": float(np.nanmean(ratio_x[at_risk])) if at_risk.any() else None, "n_at_risk": int(at_risk.sum())}
        c["arm"] = ARM_NAME[arm]
        res["arms"][arm] = c
        res["verdict"][arm] = verdict_compact(lm.lead_viol_verdict(lvres))
    # paired os - control (at-risk bar key + paired progress), informational: os is the BASELINE here
    for b in ("stop", "ha0", "ha0_ext", "ha"):
        p = lm.paired_lead_viol(flags["os"], flags[b], eid, names=("os", b), n_boot=N_BOOT, seed=0)
        res["paired_os_vs"][b] = paired_compact(p)
        res["verdict_os_vs"][b] = verdict_compact(lm.lead_viol_verdict(flags["os"], paired=p))
    # --- internal controls that must read KNOWN values -------------------------------------------------------
    ci = res["controls_internal"]
    # (1) human-as-plan: ratio exactly 1, LEAD-VIOL exactly 0
    hv = lm.lead_viol(human_dk, human_dk, eid, lead_valid=lv, n_boot=200, seed=0, plan_arc=human_arc, human_arc=human_arc)
    ci["human_as_plan"] = {"lead_progress_ratio": hv["lead_progress_ratio"], "lead_progress_ratio_clipped": hv["lead_progress_ratio_clipped"],
                           "n_viol": hv["n_viol"], "expected": "ratio 1.0 exactly, n_viol 0",
                           "verdict": lm.lead_viol_verdict(hv)["verdict"]}
    # (2) half-speed plan (G scaled 0.5): ratio ~0.5 -> INADMISSIBLE-CRAWL
    half = 0.5 * G
    hv2 = lm.lead_viol(lm.distance_keeping(half, lead["leads"], lead["lead_lens"], lead["speeds"], dtv), human_dk, eid,
                       lead_valid=lv, n_boot=200, seed=0, plan_arc=lm.path_arc_length(half), human_arc=human_arc)
    ci["half_speed_plan"] = {"lead_progress_ratio": hv2["lead_progress_ratio"], "lead_progress_ratio_clipped": hv2["lead_progress_ratio_clipped"],
                             "expected": "ratio 0.5 exactly (a scaled polyline scales its arc), INADMISSIBLE-CRAWL",
                             "verdict": lm.lead_viol_verdict(hv2)["verdict"]}
    # (3) overshoot-masks-crawl: even episodes stand still, odd episodes run at 2x the human -> mean ratio ~1.0, clipped ~0.5
    odd = (eid % 2 == 1)
    mix = np.where(odd[:, None, None], 2.0 * G, 0.0 * G)
    hv3 = lm.lead_viol(lm.distance_keeping(mix, lead["leads"], lead["lead_lens"], lead["speeds"], dtv), human_dk, eid,
                       lead_valid=lv, n_boot=200, seed=0, plan_arc=lm.path_arc_length(mix), human_arc=human_arc)
    ci["crawl_in_half_overshoot_in_half"] = {
        "lead_progress_ratio": hv3["lead_progress_ratio"], "lead_progress_ratio_clipped": hv3["lead_progress_ratio_clipped"],
        "overshoot_masks_crawl": hv3["progress"].get("overshoot_masks_crawl"),
        "expected": "unclipped mean ~1.0 (passes), clipped ~0.5, overshoot_masks_crawl True, INADMISSIBLE-CRAWL",
        "verdict": lm.lead_viol_verdict(hv3)["verdict"]}
    # (4) model-free arms identical across seeds
    if other_seed_controls is not None:
        ci["model_free_arms_identical_to_other_seed"] = {b: bool(np.array_equal(plans[b], other_seed_controls[b]))
                                                         for b in ("ha0", "ha0_ext", "ha", "stop")}
    # content assertions (not just "ran"): the stop arm is all-zero, the others and os are not
    ci["content_asserts"] = {"stop_all_zero": bool(np.all(plans["stop"] == 0.0)),
                             "os_nonzero": bool(np.abs(plans["os"]).max() > 1.0),
                             "g_finite": bool(np.isfinite(G).all()), "os_finite": bool(np.isfinite(plans["os"]).all())}
    assert ci["content_asserts"]["stop_all_zero"] and ci["content_asserts"]["os_nonzero"] and ci["content_asserts"]["g_finite"]
    res["wall_s"] = round(time.time() - t0, 1)
    return res, plans


prev = None
for s in seeds:
    pdir = str(OUT / f"s6_s{s}")
    res, plans = run_one(pdir, s, prev)
    prev = {b: plans[b] for b in ("ha0", "ha0_ext", "ha", "stop")}
    o = res["arms"]["os"]["progress"]
    print("S6 seed", s, "n_lead", res["n_lead_windows"], "n_at_risk", o["n_at_risk"], "n_usable", o["n"],
          "ratio", o["lead_progress_ratio"], "clipped", o["lead_progress_ratio_clipped"],
          "meets", o["meets_min_progress"], "masks", o["overshoot_masks_crawl"], "verdict", res["verdict"]["os"]["verdict"], flush=True)
    json.dump(res, open(OUT / f"leadprog_s6_s{s}.json", "w", encoding="utf-8"), indent=1, default=str)
print("ZZSTAGED_DONE")
