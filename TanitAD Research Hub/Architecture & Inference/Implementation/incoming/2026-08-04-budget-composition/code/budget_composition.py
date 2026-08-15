"""THE COMPOSITION — does the budget freed by the anchor band buy the off-fan operator?

Pre-registration: ``../PREREG_BUDGET_COMPOSITION.md``, git blob
``5fef562bbfbdc6e3357da9992ca7e2ff27b4a5b0`` (pinned before any statistic here existed;
re-verified at the end of the pass).

Joins two results that were produced independently and never composed:

  A  ``Research/2026-08-04-planner-hierarchy-sota`` (7d8ed27) — HAD's 5-point radial grid on
     the ALREADY-SELECTED trajectory, per-window ORACLE lambda: 0.4728 -> 0.3138 (base),
     0.4714 -> 0.3064 (XL). A CEILING, not a result.
  B  ``Implementation/incoming/2026-08-04-fan-width`` (be2da04) — filter the ANCHORS through
     the reachability band: 92/256, 46/128, 23/64, selection index identical on 881/881.

WHY THIS FILE EXISTS AND WHAT IT REFUSES TO ASSUME
==============================================================================
The two streams did NOT read the same dump. A read ``fan_emitted_*`` (s1-climbout, dumped
2026-08-04); B read ``taniteval/results/fan_*`` (2026-07-21). They are NOT bit-identical
(XL: fan maxabs 0.01373, logits maxabs 0.01234) even though ``v0`` is bit-identical, ``eid``
matches, and the selection index agrees on 881/881 — which is why both papers publish the
same 0.4714 / 0.4728. Composing two numbers computed on two different dumps would be a
category error, so BOTH operators are re-implemented here and run on ONE bank, with the
other bank carried as ``C-BANK``.

Every oracle built here is checked to BE an upper bound on every one of the 881 windows
(``C-upper-bound``) — one oracle in this programme fired under the null.

ESTIMATOR: ``taniteval.ci.paired_episode_cluster_bootstrap``, unit = episode, n_boot = 2000.
  overlapping_holdout_se is NEVER called.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# ---- verbatim from A's e_exp1_axis_reach.py, not re-parameterised ---------- #
LAMBDAS = np.array([0.92, 0.96, 1.00, 1.04, 1.08], dtype=np.float64)
DELTAS_DEG = np.array([-6.0, -3.0, 0.0, 3.0, 6.0], dtype=np.float64)
IDENT_LAM, IDENT_DEL = 2, 2                      # index of lambda=1.00 / delta=0

# ---- verbatim from B's reachable_budget.py -------------------------------- #
ACCEL_MAX, HORIZON_S = 2.5, 2.0
N_SUFF = {"refc-xl-30k": 92, "refc-base-30k": 46, "refc-small-30k": 23}

N_BOOT, SEED = 2000, 0
PREREG = ("TanitAD Research Hub/Architecture & Inference/Implementation/"
          "incoming/2026-08-04-budget-composition/PREREG_BUDGET_COMPOSITION.md")
PREREG_BLOB_AT_WRITE = "5fef562bbfbdc6e3357da9992ca7e2ff27b4a5b0"


# ============================================================================ #
# operators
# ============================================================================ #
def ade(traj, gt):
    """(..., T, 2) vs (W, T, 2) -> mean-over-T L2. VERBATIM from A."""
    return np.linalg.norm(traj - gt, axis=-1).mean(axis=-1)


def rotate(f, deg):
    """VERBATIM from A."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    x, y = f[..., 0], f[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def band_prefix_idx(keep: torch.Tensor, n: int) -> torch.Tensor:
    """VERBATIM from B — first ``n`` in FPS order surviving ``keep``, topped up."""
    return torch.argsort((~keep).long(), dim=1, stable=True)[:, :n].contiguous()


def three_sided(p: dict, *, signed_metric: bool = False) -> str:
    """better / worse / not separated.

    On a SIGNED component there is no such verdict: a bias shift is a DIRECTION.
    """
    if signed_metric:
        return ("n/a — SIGNED bias; the sign is a DIRECTION, not a quality verdict. "
                "Read it beside the |abs| row.")
    if not p["separated"]:
        return "not separated"
    return "better" if p["delta"] < 0 else "worse"


def _git(repo: Path, *a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                              text=True, timeout=90).stdout.strip()
    except Exception as exc:                                       # pragma: no cover
        return f"<git failed: {exc}>"


# ============================================================================ #
# four families, per window (for PAIRED intervals) — same geometry as the
# aggregate block, at the SAME derived dt, so they cannot drift apart.
# ============================================================================ #
def per_window_components(pred: torch.Tensor, gt: torch.Tensor, dt: float, ff) -> dict:
    P = ff._seq_geometry(pred, dt)
    G = ff._seq_geometry(gt, dt)
    both = P["valid"] & G["valid"]
    bp = P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    def rm(x, m):
        m = m.to(x.dtype)
        return (x.abs() * m).sum(1) / m.sum(1).clamp_min(1e-9)

    return {
        "LONGITUDINAL/speed_abs_err_mps": (P["speed"] - G["speed"]).abs().mean(1),
        "LONGITUDINAL/speed_signed_err_mps": (P["speed"] - G["speed"]).mean(1),
        "LONGITUDINAL/along_abs_err_m": (P["along"] - G["along"]).abs().mean(1),
        "LONGITUDINAL/along_signed_err_m": (P["along"] - G["along"]).mean(1),
        "LATERAL/cross_abs_err_m": (P["cross"] - G["cross"]).abs().mean(1),
        "LATERAL/heading_abs_err_deg": rm(dh, both) * 180.0 / math.pi,
        "LATERAL/curvature_abs_err_1pm": rm(P["curvature"] - G["curvature"], bp),
        "LATERAL/yaw_rate_abs_err_degps": rm(P["yaw_rate"] - G["yaw_rate"], bp)
                                          * 180.0 / math.pi,
    }


# ============================================================================ #
def run_arm(arm: str, bank: str, anchors_path: str, lead_npz: str, repo: Path,
            *, tag: str = "PRIMARY") -> dict:
    t0 = time.time()
    for p in (repo / "taniteval", repo / "stack"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from taniteval import ci, four_families, lead_metrics            # noqa: PLC0415
    from tanitad.refs import refc_select as sl                       # noqa: PLC0415

    d = torch.load(bank, map_location="cpu", weights_only=False)
    fan_t, logits, gt_t = d["fan"], d["logits"], d["gt"]
    W, NMAX = fan_t.shape[:2]
    eid = list(d["eid"])
    v0 = d["v0"].to(fan_t.dtype)
    bidx = torch.arange(W)

    V = torch.load(anchors_path, map_location="cpu", weights_only=False)
    if arm in V:
        anc = V[arm][:NMAX].to(fan_t.dtype)
        anchor_src = f"own buffer ({arm})"
    else:
        widest = max(V, key=lambda k: V[k].shape[0])
        anc = V[widest][:NMAX].to(fan_t.dtype)
        for other in V:                                # the nested-FPS-prefix identity
            m = min(NMAX, V[other].shape[0])
            assert torch.equal(V[widest][:m], V[other][:m]), (
                f"nested-FPS-prefix broken between {widest} and {other} — STOP")
        anchor_src = f"nested FPS prefix of {widest} (identity ASSERTED, not assumed)"

    keep_anc = sl.reachability_mask(anc[None].expand(W, -1, -1, -1).contiguous(), v0,
                                    accel_max=ACCEL_MAX, horizon_s=HORIZON_S)

    # ---------------- A0 : the incumbent, full fan --------------------------- #
    sel_full = logits.argmax(1)

    # ---------------- N_suff : RE-DERIVED here, not inherited ---------------- #
    order = torch.argsort((~keep_anc).long(), dim=1, stable=True)
    pos = (order == sel_full[:, None]).float().argmax(1)
    n_suff_derived = int(pos.max()) + 1
    n_suff = N_SUFF.get(arm, n_suff_derived)

    # ---------------- A1 : B alone, anchor-band prefix at N_suff ------------- #
    ix = band_prefix_idx(keep_anc, n_suff)
    j = logits.gather(1, ix).argmax(1)
    sel_band = ix.gather(1, j[:, None]).squeeze(1)

    fan = fan_t.double().numpy()
    gt = gt_t.double().numpy()
    pred0 = fan[np.arange(W), sel_full.numpy()]          # (W, T, 2)
    pred1 = fan[np.arange(W), sel_band.numpy()]

    ade0 = ade(pred0, gt)
    ade1 = ade(pred1, gt)

    # ---------------- the operator, on each base ----------------------------- #
    def lam_stack(base):                                 # (5, W)
        return np.stack([ade(base * L, gt) for L in LAMBDAS])

    def del_stack(base):                                 # (5, W)
        return np.stack([ade(rotate(base, D), gt) for D in DELTAS_DEG])

    def lamdel_stack(base):                              # (25, W)
        return np.stack([ade(rotate(base * L, D), gt)
                         for L in LAMBDAS for D in DELTAS_DEG])

    L0 = lam_stack(pred0)
    L1 = lam_stack(pred1)
    D1 = del_stack(pred1)
    LD1 = lamdel_stack(pred1)

    ade_A3 = L0.min(axis=0)                              # A alone   (full fan  + lambda)
    ade_A2 = L1.min(axis=0)                              # COMPOSED  (N_suff    + lambda)
    ade_A5 = LD1.min(axis=0)                             # COMPOSED  (N_suff    + lambda x delta)
    ade_Clat = D1.min(axis=0)                            # matched-DoF lateral control
    lam_star_idx = L1.argmin(axis=0)
    lam_star = LAMBDAS[lam_star_idx]

    # ---------------- band admissibility of the REFINED point ---------------- #
    # rotation preserves ||wp_last|| so it is band-neutral; only lambda moves the
    # implied mean speed. Both are MEASURED, not argued.
    scaled = np.stack([pred1 * L for L in LAMBDAS], axis=1)          # (W, 5, T, 2)
    keep_ref = sl.reachability_mask(
        torch.from_numpy(scaled).to(fan_t.dtype), v0,
        accel_max=ACCEL_MAX, horizon_s=HORIZON_S).numpy()            # (W, 5)
    rot = np.stack([rotate(pred1, D) for D in DELTAS_DEG], axis=1)
    keep_rot = sl.reachability_mask(
        torch.from_numpy(rot).to(fan_t.dtype), v0,
        accel_max=ACCEL_MAX, horizon_s=HORIZON_S).numpy()            # (W, 5)

    # ---------------- A4 : the composition, band-admissible only ------------- #
    L1_masked = np.where(keep_ref.T, L1, np.inf)
    ade_A4 = L1_masked.min(axis=0)
    lam4_idx = L1_masked.argmin(axis=0)
    lam4 = LAMBDAS[lam4_idx]

    # ---------------- A6 : refine EVERY survivor (headroom upper bound) ------ #
    keep_np = keep_anc.numpy()
    big = np.where(keep_np, 0.0, np.inf)                              # (W, N)
    cand_lam = np.stack([ade(fan * L, gt[:, None]) for L in LAMBDAS])  # (5, W, N)
    ade_A6 = (cand_lam + big[None]).min(axis=(0, 2))                   # survivors x lambda
    ade_oracle_fan = ade(fan, gt[:, None]).min(axis=1)
    ade_oracle_fan_lam = cand_lam.min(axis=(0, 2))                     # WHOLE fan x lambda

    B = lambda a, b: ci.paired_episode_cluster_bootstrap(              # noqa: E731
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
        eid, n_boot=N_BOOT, seed=SEED)

    # ======================= CONTROLS ======================================== #
    ident = rotate(pred1 * LAMBDAS[IDENT_LAM], DELTAS_DEG[IDENT_DEL])
    controls = {
        "C-identity": {
            "predicate": "lambda=1.00, delta=0 reproduces the selected trajectory BIT-identically",
            "bit_identical": bool(np.array_equal(ident, pred1)),
            "maxabs": float(np.abs(ident - pred1).max()),
            "verdict": "PASS" if np.array_equal(ident, pred1) else "FAIL — instrument void"},
        "C-sel-identity": {
            "predicate": "B's N_suff prefix reproduces the full-fan SELECTION INDEX on 881/881",
            "n_suff_used": n_suff, "n_suff_rederived_here": n_suff_derived,
            "n_suff_matches_B": bool(n_suff == n_suff_derived),
            "selection_identical_frac": round(float((sel_band == sel_full).double().mean()), 6),
            "n_windows_differing": int((sel_band != sel_full).sum()),
            "pred_bit_identical": bool(np.array_equal(pred1, pred0)),
            "verdict": ("PASS" if bool(np.array_equal(pred1, pred0))
                        else "FAIL — B is not reproduced on this bank")},
        "C-composition-exact": {
            "predicate": "A2 (composed) per-window realised ADE == A3 (A alone) EXACTLY",
            "array_equal": bool(np.array_equal(ade_A2, ade_A3)),
            "maxabs": float(np.abs(ade_A2 - ade_A3).max()),
            "n_windows_differing": int((ade_A2 != ade_A3).sum()),
            "verdict": ("PASS — B's bit-exactness SURVIVES A's operator"
                        if np.array_equal(ade_A2, ade_A3)
                        else "FAIL — outcome O-C fires")},
        "C-upper-bound": {
            "predicate": ("every oracle arm must be <= its own base on EVERY one of the "
                          "881 windows — n_windows_worse == 0, exactly. One oracle in this "
                          "programme fired under the null."),
            "A2_vs_A1_n_worse": int((ade_A2 > ade1).sum()),
            "A3_vs_A0_n_worse": int((ade_A3 > ade0).sum()),
            "A4_vs_A1_n_worse": int((ade_A4 > ade1).sum()),
            "A5_vs_A2_n_worse": int((ade_A5 > ade_A2).sum()),
            "Clat_vs_A1_n_worse": int((ade_Clat > ade1).sum()),
            "A6_vs_A2_n_worse": int((ade_A6 > ade_A2).sum()),
            "A6_finite_everywhere": bool(np.isfinite(ade_A6).all()),
            "A4_finite_everywhere": bool(np.isfinite(ade_A4).all()),
            "verdict": ("PASS" if max(int((ade_A2 > ade1).sum()), int((ade_A3 > ade0).sum()),
                                      int((ade_A4 > ade1).sum()), int((ade_A5 > ade_A2).sum()),
                                      int((ade_Clat > ade1).sum()),
                                      int((ade_A6 > ade_A2).sum())) == 0
                        else "FAIL — an 'oracle' is not an upper bound, STOP")},
        "C-band-admissibility": {
            "predicate": ("a refined point outside the reachability band is NOT admissible "
                          "under B's own guarantee. Reported with its DIRECTION."),
            "lambda_admissible_frac_per_lambda": {
                f"{L:.2f}": round(float(keep_ref[:, i].mean()), 6)
                for i, L in enumerate(LAMBDAS)},
            "identity_lambda_always_admissible": bool(keep_ref[:, IDENT_LAM].all()),
            "oracle_lambda_star_INADMISSIBLE_frac":
                round(float((~keep_ref[np.arange(W), lam_star_idx]).mean()), 6),
            "n_windows_oracle_lambda_inadmissible":
                int((~keep_ref[np.arange(W), lam_star_idx]).sum()),
            "direction_of_inadmissible_lambda_star": {
                "n_accelerating_lambda_gt_1":
                    int(((~keep_ref[np.arange(W), lam_star_idx]) & (lam_star > 1.0)).sum()),
                "n_decelerating_lambda_lt_1":
                    int(((~keep_ref[np.arange(W), lam_star_idx]) & (lam_star < 1.0)).sum())},
            "lambda_star_histogram": {f"{L:.2f}": int((lam_star_idx == i).sum())
                                      for i, L in enumerate(LAMBDAS)},
            "lambda_star_histogram_BAND_ADMISSIBLE_arm":
                {f"{L:.2f}": int((lam4_idx == i).sum()) for i, L in enumerate(LAMBDAS)}},
        "C-rotation-is-band-neutral": {
            "predicate": ("delta rotation preserves ||wp_last|| hence candidate_mean_speed "
                          "hence band membership — asserted numerically, not argued"),
            "all_rotations_match_identity_membership":
                bool((keep_rot == keep_rot[:, IDENT_DEL:IDENT_DEL + 1]).all()),
            "max_abs_norm_change_m": float(np.abs(
                np.linalg.norm(rot[:, :, -1, :], axis=-1)
                - np.linalg.norm(pred1[:, -1, :], axis=-1)[:, None]).max()),
            "verdict": ("PASS" if bool((keep_rot == keep_rot[:, IDENT_DEL:IDENT_DEL + 1]).all())
                        else "FAIL")},
        "C-shuffled": {
            "status": "DELIBERATELY NOT USED",
            "reason": ("permute-then-argmax over a candidate axis is a uniform random pick "
                       "and is VACUOUS by construction here — the same reason the fan-width "
                       "pass refused it (FAN_WIDTH §3.3)")},
    }

    # ======================= THE DECODE LEDGER =============================== #
    survivors = keep_np.sum(1)
    ledger = {
        "unit_note": ("DECODES are decoder forward passes over candidates. REFINE-EVALS are "
                      "closed-form geometric transforms of an ALREADY-DECODED (4,2) array "
                      "(scale and/or rotate). They are NOT the same currency and this ledger "
                      "never converts between them."),
        "latency_note": ("LATENCY IS NOT THE WIN. INHERITED (fan-width, MEASURED on Thor): "
                         "2.78x fewer decodes is worth 6.5 % end-to-end on XL / 3.3 % on base, "
                         "because the decoder is only 28.2 % of the frame and is fixed-cost "
                         "dominated. This ledger is BUDGET REALLOCATION, not speed."),
        "rows": [
            {"arm": "A0_full (incumbent)", "n_decodes": NMAX, "n_refine_evals": 0,
             "n_trajectories_scored": NMAX, "total_vs_incumbent_decodes": f"{NMAX} / {NMAX}"},
            {"arm": "A1_band (B alone)", "n_decodes": n_suff, "n_refine_evals": 0,
             "n_trajectories_scored": n_suff,
             "total_vs_incumbent_decodes": f"{n_suff} / {NMAX}"},
            {"arm": "A2_band_plus_L (COMPOSED)", "n_decodes": n_suff,
             "n_refine_evals": int(len(LAMBDAS)), "n_trajectories_scored": n_suff,
             "n_offfan_points_needing_a_score": int(len(LAMBDAS) - 1),
             "total_vs_incumbent_decodes": f"{n_suff} / {NMAX}"},
            {"arm": "A5_band_plus_LA (COMPOSED, full HAD grid)", "n_decodes": n_suff,
             "n_refine_evals": int(len(LAMBDAS) * len(DELTAS_DEG)),
             "n_trajectories_scored": n_suff,
             "n_offfan_points_needing_a_score": int(len(LAMBDAS) * len(DELTAS_DEG) - 1),
             "total_vs_incumbent_decodes": f"{n_suff} / {NMAX}"},
            {"arm": "A3_full_plus_L (A alone)", "n_decodes": NMAX,
             "n_refine_evals": int(len(LAMBDAS)), "n_trajectories_scored": NMAX,
             "total_vs_incumbent_decodes": f"{NMAX} / {NMAX}"},
        ],
        "variable_width_mean_decodes": round(float(survivors.mean()), 2),
        "variable_width_max_decodes": int(survivors.max()),
        "freed_decodes_fixed_Nsuff": int(NMAX - n_suff),
        "freed_decodes_variable": round(float(NMAX - survivors.mean()), 2),
        "refine_cost_in_freed_decodes_pct": round(
            100.0 * len(LAMBDAS) / max(1, NMAX - n_suff), 3),
        "⭐ CURRENCY MISMATCH (prereg outcome O-E)": {
            "shipped_score_shape": list(logits.shape),
            "shipped_score_is_anchor_indexed": bool(logits.shape == (W, NMAX)),
            "consequence": ("the shipped selector emits ONE logit PER ANCHOR. It is a "
                            "function of the anchor index, not of a trajectory, so it "
                            "CANNOT score a point that is not in the fan. Freed DECODES "
                            "therefore cannot be spent on the off-fan operator without a "
                            "TRAJECTORY-CONDITIONED scorer, which no amount of freed "
                            "decode budget provides.")},
    }

    # ======================= POINTS + PAIRED DELTAS ========================== #
    point = {
        "A0_full": float(ade0.mean()), "A1_band": float(ade1.mean()),
        "A2_band_plus_L": float(ade_A2.mean()), "A3_full_plus_L": float(ade_A3.mean()),
        "A4_band_plus_L_inband": float(ade_A4.mean()),
        "A5_band_plus_LA": float(ade_A5.mean()), "C_lat_band_plus_A": float(ade_Clat.mean()),
        "A6_refine_all_survivors_UPPER_BOUND": float(ade_A6.mean()),
        "oracle_in_fan": float(ade_oracle_fan.mean()),
        "oracle_whole_fan_x_lambda": float(ade_oracle_fan_lam.mean()),
    }
    paired = {
        "A1_minus_A2  (the operator, ON the cut fan)": B(ade1, ade_A2),
        "A0_minus_A3  (the operator, on the FULL fan = A's own number)": B(ade0, ade_A3),
        "A1_minus_A4  (band-ADMISSIBLE operator only)": B(ade1, ade_A4),
        "A2_minus_A4  (what the band admissibility COSTS the ceiling)": B(ade_A2, ade_A4),
        "A1_minus_C_lat  (matched-DoF LATERAL control)": B(ade1, ade_Clat),
        "A2_minus_A5  (adding the angular axis to the composition)": B(ade_A2, ade_A5),
        "A0_minus_A1  (B alone — identity, expected exactly 0)": B(ade0, ade1),
        "A2_minus_A6  (headroom IF an off-fan scorer existed; trivially monotone)":
            B(ade_A2, ade_A6),
    }
    d_long = paired["A1_minus_A2  (the operator, ON the cut fan)"]["delta"]
    d_lat = paired["A1_minus_C_lat  (matched-DoF LATERAL control)"]["delta"]
    d_inb = paired["A1_minus_A4  (band-ADMISSIBLE operator only)"]["delta"]
    ratios = {
        "longitudinal_over_lateral_matched_DoF": (round(d_long / d_lat, 3)
                                                  if d_lat else None),
        "band_admissible_share_of_the_ceiling":
            (round(d_inb / d_long, 4) if d_long else None),
        "share_of_this_arms_own_oracle_gap":
            round(d_long / (point["A0_full"] - point["oracle_in_fan"]), 4),
        "prereg_O-B_threshold": 0.50,
        "prereg_O-B_verdict": (
            "O-B′ NOT materially double-counted (>= 50 % survives the band)"
            if d_long and d_inb / d_long >= 0.50 else
            "O-B MATERIALLY DOUBLE-COUNTED (< 50 % survives the band)"),
        "prereg_O-A_threshold_m": 0.10,
        "prereg_O-A_verdict": (
            "O-A FIRES" if (controls["C-composition-exact"]["array_equal"]
                            and paired["A1_minus_A2  (the operator, ON the cut fan)"]["separated"]
                            and d_long >= 0.10) else "O-A does NOT fire"),
    }

    # ======================= P2 — THE FOUR FAMILIES ========================== #
    dt, dt_prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]), "dt_s": 0.1})
    preds = {
        "A0_full": pred0, "A1_band": pred1,
        "A2_band_plus_L": pred1 * lam_star[:, None, None],
        "A4_band_plus_L_inband": pred1 * lam4[:, None, None],
    }
    ref_cmp = per_window_components(torch.from_numpy(preds["A0_full"]).float(),
                                    gt_t, dt, four_families)

    z = np.load(lead_npz, allow_pickle=True)
    lead_ok = z["leads"].shape[0] == W
    lead_align = {
        "npz_rows": int(z["leads"].shape[0]), "bank_rows": W,
        "eid_agrees_as_a_partition": bool(
            len({(a, b) for a, b in zip(z["eid"].tolist(), eid)}) == len(set(eid))),
        "n_LEAD": int((z["state"] == "LEAD").sum()),
        "n_NO_LEAD": int((z["state"] == "NO_LEAD").sum()),
        "n_NO_LABEL": int((z["state"] == "NO_LABEL").sum()),
        "ts_rel_s": z["ts_rel_s"].tolist(), "wp_steps": list(d["wp_steps"]),
        "status": "OK" if lead_ok else "SKIPPED — row counts differ"}

    def dk_of(p):
        return lead_metrics.distance_keeping(np.asarray(p, dtype=np.float64), z["leads"],
                                             z["lead_lens"], z["speeds"], dt=dt)

    dk_ref = dk_of(preds["A0_full"]) if lead_ok else None

    families = {}
    for tag_a, p in preds.items():
        pt = torch.from_numpy(p).float()
        cmp = per_window_components(pt, gt_t, dt, four_families)
        row = {"n_decodes": (NMAX if tag_a == "A0_full" else n_suff),
               "ade_0_2s": round(float(ade(p, gt).mean()), 6),
               "LONGITUDINAL": {}, "LATERAL": {}}
        for k, v in cmp.items():
            fam, name = k.split("/")
            pp = ci.paired_episode_cluster_bootstrap(
                v.numpy().astype(np.float64), ref_cmp[k].numpy().astype(np.float64),
                eid, n_boot=N_BOOT, seed=SEED)
            row[fam][name] = {"value": round(float(v.mean()), 6),
                              "paired_vs_A0_full": pp,
                              "verdict": three_sided(pp, signed_metric="signed" in name)}
        if lead_ok:
            dk = dk_of(p)
            blk = {"status": dk["status"], "n_with_lead": dk["n"],
                   "n_windows": dk["n_windows"],
                   "mean_headway_min_m": dk.get("mean_headway_min_m"),
                   "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
                   "n_time_gap": dk.get("n_time_gap"),
                   "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
                   "n_closing": dk.get("n_closing"),
                   "censoring_note": dk.get("censoring_note")}
            if dk["status"] == "OK":
                blk["paired_vs_A0_full"] = lead_metrics.paired_distance_keeping(
                    dk, dk_ref, eid, names=(tag_a, "A0_full"), n_boot=N_BOOT)
                blk["by_speed_band"] = lead_metrics.distance_keeping_by_speed(
                    dk, z["speeds"], eid, states=z["state"], n_boot=N_BOOT)
            row["LONGITUDINAL"]["distance_keeping"] = blk
        else:
            row["LONGITUDINAL"]["distance_keeping"] = {
                "status": "UNAVAILABLE", "n": 0, "reason": lead_align["status"]}
        families[tag_a] = row

    # TACTICAL — the selection half is measurable; the decision half is not.
    de_all = ade(fan, gt[:, None])
    tactical = {}
    for tag_a, sidx, nn in (("A0_full", sel_full.numpy(), NMAX),
                            ("A1_band", sel_band.numpy(), n_suff)):
        s = de_all[np.arange(W), sidx]
        tactical[tag_a] = {
            "goal_anchor_selection_rank_acc": round(float((s <= de_all.min(1) + 1e-12).mean()), 6),
            "sel_gap_m": round(float((s - de_all.min(1)).mean()), 6),
            "frac_sel_2x_worse_than_best_in_fan":
                round(float((s > 2.0 * de_all.min(1)).mean()), 6),
            "n_decodes": nn}
    tactical["A2_band_plus_L / A4"] = {
        "status": "NOT-APPLICABLE", "n": 0,
        "reason": ("the refinement does not choose an anchor — it moves the already-chosen "
                   "trajectory off the fan. There is no goal/anchor selection to score and "
                   "no candidate set the refined point belongs to.")}
    tactical["manoeuvre_decision_half"] = {
        "status": "UNAVAILABLE", "n": 0,
        "reason": ("refc_rerank.dump stores no decoded manoeuvre logits — selected-vs-executed "
                   "and the 5-way confusion cannot be computed from a fan bank. "
                   "A WORK ITEM, not a pass.")}
    strategic = {
        "status": "UNAVAILABLE", "n": 0, "of_windows": W,
        "reason": ("no route/goal label in a fan bank and the decode ran "
                   "nav_mode='follow_constant', so the route input was never exercised. "
                   "A WORK ITEM, not a pass.")}

    out = {
        "bank_tag": tag, "arm": arm, "bank": str(bank), "bank_bytes": os.path.getsize(bank),
        "n_windows": W, "n_anchors": NMAX, "n_episodes": len(set(eid)),
        "ckpt_step": int(d.get("ckpt_step", -1)), "denoise_steps": int(d.get("steps", -1)),
        "anchor_source": anchor_src,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimator": {"paired": "paired_episode_cluster_bootstrap", "unit": "episode",
                      "n_boot": N_BOOT, "seed": SEED,
                      "⛔": "overlapping_holdout_se is NEVER called"},
        "operator": {"lambdas": LAMBDAS.tolist(), "deltas_deg": DELTAS_DEG.tolist(),
                     "source": "HAD arXiv 2604.03581, verbatim from E-EXP-1's code"},
        "band": (f"refc_select.reachability_mask, accel_max={ACCEL_MAX}, "
                 f"horizon_s={HORIZON_S} — NOT tuned here"),
        "N_suff": n_suff, "N_suff_rederived": n_suff_derived,
        "decode_ledger": ledger,
        "point_ade_0_2s": point,
        "paired_deltas": paired,
        "ratios_and_prereg_verdicts": ratios,
        "controls": controls,
        "lead_block_alignment": lead_align,
        "P2_four_families": {"_dt_s": dt, "_dt_provenance": dt_prov, **families},
        "TACTICAL": tactical, "STRATEGIC": strategic,
        "wall_s": round(time.time() - t0, 1),
    }
    return out


def _c(o):
    if isinstance(o, dict):
        return {k: _c(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_c(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 32 else f"<ndarray {o.shape}>"
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() < 32 else f"<tensor {tuple(o.shape)}>"
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--lead-npz", required=True)
    ap.add_argument("--only", default=None, help="comma-separated arm subset")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    hub = repo / "TanitAD Research Hub"
    primary = {
        "refc-xl-30k": str(repo / "taniteval/results/fan_refc-xl-30k.pt"),
        "refc-base-30k": str(repo / "taniteval/results/fan_refc-base-30k.pt"),
        "refc-small-30k": str(hub / "Benchmarks & Eval/Implementation/incoming/"
                             "2026-07-22-refc-small-30k/fan_refc-small-30k.pt"),
    }
    cbank = {
        "refc-xl-30k": str(hub / "Architecture & Inference/Implementation/incoming/"
                           "2026-08-03-s1-climbout/raw/fan_emitted_refc-xl-30k.pt"),
        "refc-base-30k": str(hub / "Architecture & Inference/Implementation/incoming/"
                             "2026-08-03-s1-climbout/raw/fan_emitted_refc-base-30k.pt"),
    }
    keep = set(a.only.split(",")) if a.only else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prereg_meta = {
        "path": PREREG, "blob_at_write_time": PREREG_BLOB_AT_WRITE,
        "staged_blob": (_git(repo, "ls-files", "-s", "--", PREREG).split()
                        or ["", "<not staged>"])[1],
        "worktree_blob": _git(repo, "hash-object", PREREG)}
    results = []
    for arm, bp in primary.items():
        if keep and arm not in keep:
            continue
        print(f"[compose] PRIMARY {arm}", flush=True)
        r = run_arm(arm, bp, a.anchors, a.lead_npz, repo, tag="PRIMARY")
        r["prereg"] = prereg_meta
        results.append(r)
        (out / f"budget_composition_{arm}.json").write_text(
            json.dumps(_c(r), indent=1, ensure_ascii=False), encoding="utf-8")
        p, q = r["point_ade_0_2s"], r["paired_deltas"]
        k = "A1_minus_A2  (the operator, ON the cut fan)"
        print(f"   A0 {p['A0_full']:.4f} | A1 {p['A1_band']:.4f} | A2 {p['A2_band_plus_L']:.4f}"
              f" | A3 {p['A3_full_plus_L']:.4f} | A4 {p['A4_band_plus_L_inband']:.4f}"
              f" | A5 {p['A5_band_plus_LA']:.4f}", flush=True)
        print(f"   A1-A2 {q[k]['delta']:+.4f} [{q[k]['lo']:+.4f},{q[k]['hi']:+.4f}] "
              f"sep={q[k]['separated']} | composed==A-alone: "
              f"{r['controls']['C-composition-exact']['array_equal']}", flush=True)

    for arm, bp in cbank.items():
        if keep and arm not in keep:
            continue
        print(f"[compose] C-BANK {arm}", flush=True)
        r = run_arm(arm, bp, a.anchors, a.lead_npz, repo, tag="C-BANK")
        r["prereg"] = prereg_meta
        results.append(r)
        (out / f"budget_composition_CBANK_{arm}.json").write_text(
            json.dumps(_c(r), indent=1, ensure_ascii=False), encoding="utf-8")

    ledger = {
        "what": "the decode ledger, all arms, PRIMARY bank",
        "prereg": prereg_meta,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {r["arm"]: {"bank_tag": r["bank_tag"], "N_max": r["n_anchors"],
                            "N_suff": r["N_suff"], **r["decode_ledger"]}
                 for r in results if r["bank_tag"] == "PRIMARY"},
        "cross_bank_stability": {
            r["arm"]: {"A3_full_plus_L": r["point_ade_0_2s"]["A3_full_plus_L"],
                       "A0_full": r["point_ade_0_2s"]["A0_full"]}
            for r in results if r["bank_tag"] == "C-BANK"},
    }
    (out / "decode_ledger.json").write_text(json.dumps(_c(ledger), indent=1,
                                                       ensure_ascii=False), encoding="utf-8")
    print(f"[compose] -> {out}", flush=True)
    return 0


if __name__ == "__main__":                                          # pragma: no cover
    raise SystemExit(main())
