#!/usr/bin/env python3
"""H-PROJ-CONTACT-1 -- the collision projection, tested at ZERO GPU on the banked fan.

Reads `fan_bank_base_240w.npz` (240 x 128 x 5 x 2) and the human GT extracted from the
same corpus at the same `wi`, applies `tanitad.refs.contact_projection`, and reports:
the five controls, the three-way taxonomy, and the contact-vs-ADE-vs-margin frontier.

Every contact number is reported in BOTH forms -- `fan_*` (the generator) and `sel_*`
(the selector's pick). RETRACTION #31: a `sel_*` metric may never stand alone.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\Admin\collproj"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, os.path.join(REPO, "taniteval", "tools"))

from tanitad.refs import contact_projection as CP           # noqa: E402
from tanitad.rl.rewards import _collision                    # noqa: E402
import taniteval.ci as CI                                    # noqa: E402
import importlib.util                                        # noqa: E402
_s = importlib.util.spec_from_file_location(
    "fan_safety", os.path.join(REPO, "taniteval", "tools", "fan_safety.py"))
FS = importlib.util.module_from_spec(_s)
sys.modules["fan_safety"] = FS
_s.loader.exec_module(FS)

BANK = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
        r"\Deployment & Optimization\Research\2026-09-05-veto-only-fan-safety"
        r"\raw\fan_bank_base_240w.npz")
GT = r"C:\Users\Admin\collproj\out\gt_240w.npz"
OUT = r"C:\Users\Admin\collproj\out\contact_projection_panel.json"

MARGINS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
DELTAS = [0.0, 0.5, 1.0, 2.0]           # worst-case agent-POSITION error, metres
N_BOOT = 4000
SEED = 11
R = CP.CONTACT_R_M


def contact_flag(paths, lead, r=R):
    """The SCORER's predicate, re-derived on whatever tensor is handed in."""
    ctx = {"lead_path": lead.expand_as(paths), "ego_radius_m": r / 2.0,
           "obs_radius_m": r / 2.0}
    return (_collision(paths, ctx) < 0)


def ade(paths, gt):
    """paths [B, N, 5, 2] -> [B, N] mean L2 over the 4 future slots vs gt [B, 4, 2]."""
    return (paths[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)


def boot(v, eid):
    r = CI.episode_cluster_bootstrap(np.asarray(v, dtype=np.float64),
                                     np.asarray(eid), n_boot=N_BOOT, seed=SEED)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}


def pboot(a, b, eid):
    r = CI.paired_episode_cluster_bootstrap(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
        np.asarray(eid), n_boot=N_BOOT, seed=SEED)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows",
                              "n_episodes", "estimator")}


def main():
    rec = {"_what": "H-PROJ-CONTACT-1: the collision projection on the banked refcv3 fan",
           "_tool": "TanitAD Research Lab/.../2026-09-05-collision-projection/raw/panel.py",
           "_evidence_class": "MEASURED (ours; rule-based, 0 GPU)",
           "_tier": "T0/T1-adjacent (the EMITTED fan on the eval clips); never a driving claim",
           "_predicate": "tanitad.rl.rewards._collision, moving-lead SWEPT branch, r = 2.0 m",
           "bank": BANK, "gt": GT}

    z = np.load(BANK, allow_pickle=True)
    gz = np.load(GT, allow_pickle=True)
    fan = torch.tensor(np.asarray(z["fan2"]), dtype=torch.float64)      # [240,128,5,2]
    lead = torch.tensor(np.asarray(z["lead5"]), dtype=torch.float64)[:, None]  # [240,1,5,2]
    has = torch.tensor(np.asarray(z["has_lead"]).astype(bool))
    v0 = torch.tensor(np.asarray(z["v0"], dtype=np.float64))
    sel = torch.tensor(np.asarray(z["sel_idx"]).astype(np.int64))
    eid = np.asarray(z["eid"]).astype(int).ravel()
    f_contact_banked = torch.tensor(np.asarray(z["f_contact"]).astype(bool))
    gt = torch.tensor(np.asarray(gz["gt"]), dtype=torch.float64)        # [240,4,2]
    B, N, S, _ = fan.shape
    assert bool(gz["join_ok"][0]), "the GT join control did not pass"

    rec["object"] = {
        "tensor": "fan_bank_base_240w.npz::fan2", "shape": list(fan.shape),
        "lead": "fan_bank_base_240w.npz::lead5", "gt_join_control": "PASS",
        "n_windows": B, "n_candidates_per_window": N,
        "n_lead_windows": int(has.sum()), "n_candidates_total": B * N}

    # ------------------------------------------------------------------ #
    # 0. THE DEFECT, reproduced from the banked tensor -- and RE-DERIVED  #
    #    with the CURRENT swept predicate, because the bank was built     #
    #    against a frozen clone while the swept fix was landing.          #
    # ------------------------------------------------------------------ #
    hit_in = contact_flag(fan, lead) & has[:, None]
    n_banked = int(f_contact_banked.sum())
    n_swept = int(hit_in.sum())
    rec["defect"] = {
        "banked_f_contact_count": n_banked,
        "banked_f_contact_rate": n_banked / (B * N),
        "banked_windows": int(f_contact_banked.any(1).sum()),
        "banked_worst_window": int(f_contact_banked.sum(1).max()),
        "rederived_swept_count": n_swept,
        "rederived_swept_rate": n_swept / (B * N),
        "rederived_windows": int(hit_in.any(1).sum()),
        "rederived_worst_window": int(hit_in.sum(1).max()),
        "banked_equals_swept": bool((f_contact_banked == hit_in).all()),
        "swept_minus_banked": n_swept - n_banked,
        "_note": ("the bank was written against a clone frozen BEFORE the swept-segment "
                  "fix (commit 9765634); a difference here means the banked 764 is the "
                  "POINT-test lower bound and the swept count is the real one")}
    rows_banked = sorted(torch.nonzero(f_contact_banked.any(1)).ravel().tolist())
    rows_swept = sorted(torch.nonzero(hit_in.any(1)).ravel().tolist())
    rec["defect"]["collider_rows_banked"] = rows_banked
    rec["defect"]["collider_rows_swept"] = rows_swept

    # the base ADE, on the input fan
    ade_in = ade(fan, gt)                                    # [240,128]
    orc_in = ade_in.min(dim=1).values
    sel_in = ade_in.gather(1, sel[:, None])[:, 0]
    sc_in = FS.score_paths(fan.float(), v0.float(), lead.float())
    rec["base"] = {
        "oracle_in_fan_ade_m": float(orc_in.mean()),
        "sel_ade_m": float(sel_in.mean()),
        "fan_contact": float(hit_in.double().mean()),
        "sel_contact": float(hit_in.gather(1, sel[:, None])[:, 0].double().mean()),
        "fan_contact_lead_pop": float(hit_in[has].double().mean()),
        "fan_peak_g_mean": float(sc_in["peak_g"].mean()),
        "fan_envelope": float(sc_in["envelope"].double().mean()),
        "fan_kamm_over": float(sc_in["kamm_over"].double().mean()),
        "fan_off_reach": float(sc_in["off_reach"].double().mean()),
        "sel_off_reach": float(sc_in["off_reach"].double().gather(1, sel[:, None])[:, 0].mean()),
        "fan_ttc_below": float(sc_in["ttc_below"][has].double().mean()),
    }

    # ------------------------------------------------------------------ #
    # 1. CONTROL C1 -- the DISABLED lever must return the input OBJECT    #
    # ------------------------------------------------------------------ #
    out_off, info_off = CP.project_contact_free(fan, lead, has, enabled=False)
    rec["controls"] = {}
    rec["controls"]["C1_disabled_bit_identical"] = {
        "known_value": "max|diff| == 0.0 exactly AND the returned object IS the input",
        "max_abs_diff_m": float((out_off - fan).abs().max()),
        "is_same_object": out_off is fan,
        "object_asserted": "fan_bank_base_240w.npz::fan2 (float64 view)",
        "PASS": bool(out_off is fan and float((out_off - fan).abs().max()) == 0.0),
        "_weak": ("this control is TRUE BY CONSTRUCTION (the arithmetic never ran) and "
                  "proves nothing about the projection -- C4 is the arithmetic's control")}

    # ------------------------------------------------------------------ #
    # 2. the projection, per MARGIN                                       #
    # ------------------------------------------------------------------ #
    rec["frontier"] = []
    per_margin_paths = {}
    for m in MARGINS:
        out, info = CP.project_contact_free(fan, lead, has, margin_m=m)
        per_margin_paths[m] = (out, info)
        hit_out = contact_flag(out, lead) & has[:, None]
        sc_out = FS.score_paths(out.float(), v0.float(), lead.float())
        ade_out = ade(out, gt)
        orc_out = ade_out.min(dim=1).values
        sel_out = ade_out.gather(1, sel[:, None])[:, 0]
        disp = (out - fan).norm(dim=-1).mean(dim=-1)          # [240,128]
        moved = (disp > 1e-12)
        d_min = CP.min_rel_distance(out, lead.expand_as(out))
        reason = info["reason"]
        tax = {CP.REASON_NAMES[k]: int((reason == k).sum()) for k in CP.REASON_NAMES}
        # robustness: a worst-case agent-POSITION error of delta is EXACTLY an
        # inflation of the contact radius by delta -- measured, not assumed.
        robust = {}
        for d in DELTAS:
            h = (d_min < (R + d)) & has[:, None]
            robust[f"fan_contact_at_r_plus_{d}"] = float(h.double().mean())
            robust[f"sel_contact_at_r_plus_{d}"] = float(
                h.gather(1, sel[:, None])[:, 0].double().mean())
        # a genuinely non-radial prediction error: the lead track TIME-SHIFTED by
        # one grid step (0.5 s) in each direction -- NOT reducible to a radius.
        for name, shift in (("lead_late_0.5s", 1), ("lead_early_0.5s", -1)):
            idx = torch.arange(S) + shift
            idx = idx.clamp(0, S - 1)
            lead_s = lead[:, :, idx, :]
            h = contact_flag(out, lead_s) & has[:, None]
            robust[f"fan_contact_{name}"] = float(h.double().mean())
        row = {
            "margin_m": m,
            "r_need_m": float(info["r_need_m"]),
            "fan_contact": float(hit_out.double().mean()),
            "fan_contact_count": int(hit_out.sum()),
            "sel_contact": float(hit_out.gather(1, sel[:, None])[:, 0].double().mean()),
            "fan_contact_lead_pop": float(hit_out[has].double().mean()),
            "windows_with_contact": int(hit_out.any(1).sum()),
            "min_clearance_m": float(d_min[has].min()),
            "taxonomy": tax,
            "oracle_in_fan_ade_m": float(orc_out.mean()),
            "d_oracle_ade_m": float((orc_out - orc_in).mean()),
            "sel_ade_m": float(sel_out.mean()),
            "d_sel_ade_m": float((sel_out - sel_in).mean()),
            "mean_disp_all_m": float(disp.mean()),
            "mean_disp_moved_m": float(disp[moved].mean()) if bool(moved.any()) else 0.0,
            "n_moved": int(moved.sum()),
            "n_moved_windows": int(moved.any(1).sum()),
            "sigma_min": float(info["sigma"][moved].min()) if bool(moved.any()) else 1.0,
            "sigma_mean_moved": float(info["sigma"][moved].mean()) if bool(moved.any()) else 1.0,
            "fan_peak_g_mean": float(sc_out["peak_g"].mean()),
            "fan_envelope": float(sc_out["envelope"].double().mean()),
            "fan_kamm_over": float(sc_out["kamm_over"].double().mean()),
            "fan_off_reach": float(sc_out["off_reach"].double().mean()),
            "sel_off_reach": float(sc_out["off_reach"].double().gather(1, sel[:, None])[:, 0].mean()),
            "fan_ttc_below": float(sc_out["ttc_below"][has].double().mean()),
            "roundtrip_max_m": float(info["roundtrip_max_m"]),
        }
        row.update(robust)
        row["paired_oracle_ade"] = pboot(orc_out.numpy(), orc_in.numpy(), eid)
        row["paired_sel_ade"] = pboot(sel_out.numpy(), sel_in.numpy(), eid)
        rec["frontier"].append(row)
        print(f"[panel] m={m:.2f}  fan_contact {row['fan_contact']:.6f} "
              f"({row['fan_contact_count']}) sel {row['sel_contact']:.6f} | "
              f"orcADE {row['oracle_in_fan_ade_m']:.4f} (d {row['d_oracle_ade_m']:+.4f}) | "
              f"selADE {row['sel_ade_m']:.4f} (d {row['d_sel_ade_m']:+.4f}) | "
              f"off_reach {row['fan_off_reach']:.4f} | tax {tax}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. CONTROLS C2 / C3 / C4 / C5 at the primary margin (m = 0)         #
    # ------------------------------------------------------------------ #
    out0, info0 = per_margin_paths[0.0]
    disp0 = (out0 - fan).norm(dim=-1).amax(dim=-1)            # [240,128] max over pts
    win_moved = (disp0 > 0).any(dim=1)
    no_agent_max = float(disp0[~has].max()) if bool((~has).any()) else 0.0
    rec["controls"]["C2_no_agent_untouched"] = {
        "known_value": "max|diff| == 0.0 exactly on the 175 windows with no agent track",
        "n_windows_no_agent": int((~has).sum()),
        "max_abs_diff_m": no_agent_max,
        "object_asserted": "the ~has_lead rows of fan_bank_base_240w.npz::fan2",
        "PASS": no_agent_max == 0.0}

    # C3 -- the windows that move must be EXACTLY the windows carrying colliders
    moved_rows = sorted(torch.nonzero(win_moved).ravel().tolist())
    rec["controls"]["C3_only_collider_windows_move"] = {
        "known_value": "the set of moved windows == the set of collider windows (swept)",
        "collider_windows_swept": rows_swept,
        "moved_windows": moved_rows,
        "identical": bool(moved_rows == rows_swept),
        "moved_not_collider": sorted(set(moved_rows) - set(rows_swept)),
        "collider_not_moved": sorted(set(rows_swept) - set(moved_rows)),
        "PASS": bool(moved_rows == rows_swept)}
    # and per CANDIDATE, which is stricter than per window
    moved_cand = (disp0 > 0)
    d_in = CP.min_rel_distance(fan, lead.expand_as(fan))
    r_need = float(info0["r_need_m"])
    should_move = (d_in < r_need) & has[:, None]
    extra = moved_cand & ~hit_in
    rec["controls"]["C3b_only_collider_candidates_move"] = {
        "known_value": ("the moved CANDIDATE set == the set violating the PROJECTION'S "
                        "OWN clearance r_need, which is r inflated by the numerical "
                        "margin -- NOT the scorer's r. The two differ by exactly the "
                        "candidates sitting inside the margin, and naming the scorer's "
                        "r here would be a control that is wrong about its own object."),
        "r_scorer_m": R, "r_need_m": r_need,
        "n_colliders_scorer_r": int(hit_in.sum()), "n_moved": int(moved_cand.sum()),
        "n_should_move_at_r_need": int(should_move.sum()),
        "moved_not_collider_at_scorer_r": int(extra.sum()),
        "moved_not_collider_all_inside_the_margin": bool(
            ((d_in >= R) & (d_in < r_need))[extra].all()) if bool(extra.any()) else True,
        "clearances_of_the_extra_m": sorted(float(x) for x in d_in[extra].tolist()),
        "collider_not_moved": int((hit_in & ~moved_cand).sum()),
        "PASS": bool(((moved_cand ^ should_move).sum()) == 0)}

    # C4 -- the ROUND-TRIP: the uniform (search-everywhere) form on untouched paths
    rt = float(info0["roundtrip_max_m"])
    uni = info0["uniform"]
    untouched = ~moved_cand
    rt_all = float((uni[untouched] - fan[untouched]).norm(dim=-1).max())
    rec["controls"]["C4_roundtrip"] = {
        "known_value": "< 1e-5 m (the arithmetic control C1 cannot give)",
        "max_residual_m": rt,
        "max_residual_over_all_untouched_candidates_m": rt_all,
        "n_untouched": int(untouched.sum()),
        "object_asserted": ("info['uniform'] -- the SEARCH-EVERYWHERE re-integration, a "
                            "DIFFERENT tensor from the shipped output. It is scoped to "
                            "the candidates the search left at sigma = 1, NOT to the "
                            "candidates the SCORER calls clear: those two sets differ by "
                            "the margin, and conflating them made this control read "
                            "2.18 m on its first pass -- a control wrong about its own "
                            "object, which is the failure it exists to catch."),
        "PASS": bool(rt < 1e-5 and rt_all < 1e-5)}

    # C5 -- friction non-regression, candidate-wise
    sc0 = FS.score_paths(out0.float(), v0.float(), lead.float())
    dpg = (sc0["peak_g"] - sc_in["peak_g"])
    rec["controls"]["C5_friction_non_regression"] = {
        "known_value": "peak_g(output) <= peak_g(input) for every candidate",
        "max_increase_g": float(dpg.max()),
        "n_increased": int((dpg > 1e-6).sum()),
        "PASS": bool(float(dpg.max()) <= 1e-6)}

    rec["controls"]["ALL_PASS"] = bool(all(
        v.get("PASS", True) for v in rec["controls"].values() if isinstance(v, dict)))

    # ------------------------------------------------------------------ #
    # 4. the UNAVOIDABLE residual -- named, per window                    #
    # ------------------------------------------------------------------ #
    reason0 = info0["reason"]
    unav = (reason0 == CP.REASON_UNAVOIDABLE)
    lead_only = CP.min_rel_distance(torch.zeros_like(fan), lead.expand_as(fan))
    rec["unavoidable"] = {
        "n_candidates": int(unav.sum()),
        "n_windows": int(unav.any(1).sum()),
        "windows": sorted(torch.nonzero(unav.any(1)).ravel().tolist()),
        "_definition": ("no member of the (sigma, dkappa) family clears the disc -- "
                        "including sigma = 0, the ego not moving at all"),
        "lead_enters_ego_disc_at_rest_windows": sorted(
            torch.nonzero((lead_only[:, 0] < R) & has).ravel().tolist()),
        "min_lead_distance_to_origin_m": {
            str(int(i)): float(lead_only[i, 0])
            for i in torch.nonzero(unav.any(1)).ravel().tolist()}}

    # per-window detail for the collider windows
    rec["per_collider_window"] = []
    for i in rows_swept:
        r0 = {"row": int(i), "wi": int(z["wi"][i]), "eid": int(eid[i]),
              "v0": float(v0[i]),
              "n_colliders_banked": int(f_contact_banked[i].sum()),
              "n_colliders_swept": int(hit_in[i].sum()),
              "n_moved": int(moved_cand[i].sum()),
              "taxonomy": {CP.REASON_NAMES[k]: int((reason0[i] == k).sum())
                           for k in CP.REASON_NAMES if int((reason0[i] == k).sum())},
              "sigma_min": float(info0["sigma"][i].min()),
              "lead_dist_at_rest_m": float(lead_only[i, 0]),
              "oracle_ade_base_m": float(orc_in[i]),
              "oracle_ade_proj_m": float(ade(out0, gt)[i].min()),
              "sel_contact_base": bool(hit_in[i, sel[i]]),
              "sel_ade_base_m": float(sel_in[i]),
              "sel_ade_proj_m": float(ade(out0, gt)[i, sel[i]])}
        rec["per_collider_window"].append(r0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[panel] -> {OUT}", flush=True)
    print("[panel] CONTROLS:", {k: v.get("PASS") for k, v in rec["controls"].items()
                                if isinstance(v, dict)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
