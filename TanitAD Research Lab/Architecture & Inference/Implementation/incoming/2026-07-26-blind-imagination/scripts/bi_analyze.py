#!/usr/bin/env python3
"""E-IMAG analysis — every number in `BLIND_IMAGINATION.md` is produced here.

Runs on the per-window dumps ``bi_run.py`` writes. **No GPU is touched**: the
dumps hold the dense predicted and ground-truth paths, so any bar, any horizon
and any decomposition can be recomputed from them forever.

Estimator, everywhere: **paired episode-cluster bootstrap** (`taniteval/ci.py`,
B = 2000, seed 0, resampling unit = val episode) on identical windows.
``overlapping_holdout_se`` appears nowhere.

Outputs (all under ``--out``):
  ``horizon_curve.json``      per-arm ``de_N`` / ``ade_N`` at the reporting grid
  ``t_blind.json``            ⭐ the headline, with its bootstrapped interval
  ``decomposition.json``      lat/lon, drift/variance, OOD envelope per horizon
  ``duty_cycle.json``         E-IMAG-4 — peek policies, uniform vs oracle
  ``bi_perwindow_compact.pt`` the compact recomputable per-window dump
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
for _p in ("/root/taniteval", "/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci                      # noqa: E402
from taniteval import blindimag as bi                # noqa: E402

GRID = (5, 10, 20, 30, 45, 60, 90, 120, 185)
B_BOOT = 2000
SEED = 0
BARS_M = {"miss_2m": 2.0, "corridor_1p391m": 1.391, "lane_half_1m": 1.0}
# The four arms x the action regimes, named once so nothing is renamed later.
REGIMES = {
    "privileged_true_actions": {"a": "a_imagination__true",
                                "b": "b_frozenlast__true",
                                "c": "c_fullobs__true"},
    "deployable_own_actions": {"a": "a_imagination__own",
                               "b": "b_frozenlast__own",
                               "c": "c_fullobs__own"},
    "deployable_held_action": {"a": "a_imagination__hold",
                               "b": "b_frozenlast__hold",
                               "c": None},
    "convention_control_gt_actions": {"a": "a_imagination__gtkin",
                                      "b": "b_frozenlast__gtkin",
                                      "c": "c_fullobs__gtkin"},
    # A2 sensitivity — the ZERO-TRAINING readout-level lever. Same arms, same
    # windows; only which trained step-readout decodes the imagined transitions.
    "A2_readout_str_true_actions": {"a": "a_imagination__true__roSTR",
                                    "b": "b_frozenlast__true__roSTR",
                                    "c": None},
    "A2_readout_str_own_actions": {"a": "a_imagination__own__roSTR",
                                   "b": None, "c": None},
}


def _de(pred, gt):
    """Per-window per-step displacement error ``[N, K]`` (float64 numpy)."""
    return torch.linalg.norm(pred - gt, dim=-1).double().numpy()


def _boot_draws(eid):
    uniq, idx = _ci.episode_index(eid)
    return list(_ci._draws(uniq, idx, B_BOOT, SEED)), len(uniq)


def _ci_at(de, eid, n, draws, cumulative=False):
    """CI on ``de_N`` (or ``ade_N``) using PRE-DRAWN episode resamples so every
    arm and every horizon share the identical draws — that is what makes the
    paired contrasts and the T_blind bootstrap coherent."""
    v = de[:, :n].mean(axis=1) if cumulative else de[:, n - 1]
    point = float(v.mean())
    bs = np.array([v[s].mean() for s in draws])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": round(point, 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "ci95": round(float((hi - lo) / 2), 4),
            "n_windows": int(v.size), "n_episodes": int(len(set(eid))),
            "n_boot": B_BOOT, "estimator": "episode_cluster_bootstrap"}


def _paired(de_a, de_b, eid, n, draws, cumulative=False):
    """``reduce(b) - reduce(a)`` — POSITIVE means arm (a) is BETTER."""
    if cumulative:
        va, vb = de_a[:, :n].mean(axis=1), de_b[:, :n].mean(axis=1)
    else:
        va, vb = de_a[:, n - 1], de_b[:, n - 1]
    point = float(vb.mean() - va.mean())
    d = np.array([vb[s].mean() - va[s].mean() for s in draws])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_b_minus_a": round(point, 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "ci95": round(float((hi - lo) / 2), 4),
            "separated": bool(lo > 0 or hi < 0),
            "a_better": bool(lo > 0),
            "positive_means": "the FIRST argument has the smaller error",
            "p_a_better": round(float((d > 0).mean()), 4),
            "n_windows": int(va.size), "n_episodes": int(len(set(eid))),
            "n_boot": B_BOOT, "estimator": "paired_episode_cluster_bootstrap"}


#: Contiguity is evaluated from step 2, not step 1 — AMENDMENT A4.
#: Arms (a), (b) and (c) decode a BIT-IDENTICAL first transition by construction
#: (``test_blindimag.py::test_first_step_is_identical_across_state_sources``),
#: so the paired delta at step 1 is EXACTLY 0.0 and its bootstrap lower bound is
#: exactly 0.0 in every draw. The rule as pre-registered ("contiguously from
#: N = 1") is therefore UNSATISFIABLE BY CONSTRUCTION: it returns 0 for every
#: arm in every regime regardless of the data. That is a C13-class defect — a
#: criterion that cannot fire — in my own instrument, found by reading my own
#: result. The repair is the smallest one that makes the criterion evaluable:
#: anchor contiguity at the first horizon on which the arms can differ at all.
T_CONTIGUITY_START_STEP = 2


def _t_contiguous(delta_lo, start_idx=T_CONTIGUITY_START_STEP - 1):
    """Largest N such that ``delta_lo`` is > 0 contiguously from step
    ``T_CONTIGUITY_START_STEP``. Returns ``start_idx`` (i.e. 1 step) when the
    first evaluable step already fails — never 0, because step 1 is shared."""
    ok = delta_lo[start_idx:] > 0
    if ok.size == 0 or not ok[0]:
        return int(start_idx)
    bad = np.flatnonzero(~ok)
    return int(start_idx) + (int(bad[0]) if bad.size else int(ok.size))


def t_blind(de_a, de_b, eid, draws):
    """⭐ THE HEADLINE. ``T_blind`` = largest horizon at which imagination (a) is
    separated-better than frozen-last-frame (b) on ``de_N``, contiguously from
    N = 1 — with the interval obtained by re-deriving the WHOLE curve inside
    every episode resample, never asserted from the point curve."""
    K = de_a.shape[1]
    # point estimate: contiguous run where the paired lower bound clears 0
    d_boot = np.empty((len(draws), K))
    for i, s in enumerate(draws):
        d_boot[i] = de_b[s].mean(axis=0) - de_a[s].mean(axis=0)
    lo = np.percentile(d_boot, 2.5, axis=0)
    hi = np.percentile(d_boot, 97.5, axis=0)
    point_delta = de_b.mean(axis=0) - de_a.mean(axis=0)
    t_point = _t_contiguous(lo)
    # bootstrap distribution of T_blind itself: inside each draw, the horizon
    # up to which imagination is ahead
    t_dist = np.array([_t_contiguous(d) for d in d_boot])
    t_lo, t_hi = np.percentile(t_dist, [2.5, 97.5])
    return {
        "T_blind_steps": int(t_point),
        "T_blind_s": round(t_point * bi.DT, 2),
        "T_blind_ci95_steps": [int(round(t_lo)), int(round(t_hi))],
        "T_blind_ci95_s": [round(float(t_lo) * bi.DT, 2),
                           round(float(t_hi) * bi.DT, 2)],
        "T_blind_median_boot_steps": int(float(np.median(t_dist))),
        "frac_draws_T_blind_is_zero": round(float((t_dist == 0).mean()), 4),
        "frac_draws_T_blind_ge_10": round(float((t_dist >= 10).mean()), 4),
        "delta_at_step1_m": round(float(point_delta[0]), 4),
        "first_step_where_a_loses_point": (
            int(np.flatnonzero(point_delta[1:] <= 0)[0] + 2)
            if (point_delta[1:] <= 0).any() else None),
        "first_step_where_b_separated_better": (
            int(np.flatnonzero(hi[1:] < 0)[0] + 2)
            if (hi[1:] < 0).any() else None),
        "rule": ("largest N with paired CI lower bound > 0 contiguously from "
                 f"N={T_CONTIGUITY_START_STEP} (amendment A4: step 1 is "
                 "bit-identical across arms by construction, so a rule anchored "
                 "at N=1 CANNOT FIRE — C13); positive delta = imagination "
                 "better"),
        "contiguity_start_step": T_CONTIGUITY_START_STEP,
        "delta_at_step1_is_exactly_zero": bool(point_delta[0] == 0.0),
        # C14 — GRID END != MEASURED LIMIT. If T_blind lands on the sweep's own
        # terminus the instrument could not have reported a larger value, so the
        # number is a LOWER BOUND on our configuration, not a measurement.
        "C14_saturated_at_grid_terminus": bool(t_point >= K),
        "C14_note": ("T_blind == K_max means the instrument could not report "
                     "more; quote it as a LOWER BOUND, never as the horizon."
                     if t_point >= K else
                     "T_blind is interior to the swept range: the instrument "
                     "could have reported a larger value and did not."),
        "K_max_swept": int(K),
        "n_boot": B_BOOT, "n_episodes": int(len(set(eid))),
        "estimator": "paired_episode_cluster_bootstrap",
    }


def t_bar(de, eid, draws, bar):
    """Largest N at which the arm's ``de_N`` stays below ``bar`` metres, with a
    bootstrapped interval on that horizon."""
    K = de.shape[1]
    m = np.empty((len(draws), K))
    for i, s in enumerate(draws):
        m[i] = de[s].mean(axis=0)
    point = de.mean(axis=0)

    def _t(curve):
        ok = curve < bar
        if not ok[0]:
            return 0
        bad = np.flatnonzero(~ok)
        return int(bad[0]) if bad.size else int(ok.size)

    td = np.array([_t(c) for c in m])
    lo, hi = np.percentile(td, [2.5, 97.5])
    return {"bar_m": bar, "T_steps": _t(point),
            "T_s": round(_t(point) * bi.DT, 2),
            "T_ci95_s": [round(float(lo) * bi.DT, 2),
                         round(float(hi) * bi.DT, 2)]}


# =========================================================================== #
def analyse_sweep(d, out):
    pred, gt = d["pred"], d["gt"]
    eid = [str(x) for x in d["eid"]]
    draws, n_ep = _boot_draws(eid)
    K = gt.shape[1]
    de = {k: _de(v, gt) for k, v in pred.items()}
    de["d_constant_velocity"] = _de(d["cv"], gt)
    de["d2_hold_v0"] = _de(d["hold_v0"], gt)
    grid = [n for n in GRID if n <= K]

    curve = {"meta": {**d["meta"], "grid_steps": grid, "n_boot": B_BOOT,
                      "n_episode_clusters": n_ep,
                      "estimator": "episode_cluster_bootstrap",
                      "metric_defs": {
                          "de_N": "||pred_N - gt_N|| at step N (PRIMARY)",
                          "ade_N": "mean over steps 1..N of ||pred_j - gt_j||"}},
             "arms": {}}
    wp4 = [4, 9, 14, 19]
    for name, m in de.items():
        blk = {
            "de": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws) for n in grid},
            "ade": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws, True)
                    for n in grid}}
        if m.shape[1] > 19:
            # the PROGRAM's own ade_0_2s: mean over the 4 sparse waypoints.
            # Emitted for every arm so any arm can be put beside the committed
            # v1 values without re-deriving a convention.
            blk["ade_0_2s_sparse_4wp"] = _ci.episode_cluster_bootstrap(
                m[:, wp4].mean(axis=1), eid, n_boot=B_BOOT, seed=SEED)
        curve["arms"][name] = blk
    # ---- ⚠️ is this fixed window set REPRESENTATIVE? ----------------------- #
    # At K=185 the harness's own window rule leaves ~1 window per episode, and
    # it is the FIRST window of the episode. That is a biased subsample by
    # construction, so it is checked rather than assumed: the same sparse
    # ade_0_2s the registry publishes on the FULL window set is recomputed here.
    wp = [4, 9, 14, 19]
    rep = {"note": ("windows at K_max are episode-INITIAL by construction; "
                    "this compares the program's own ade_0_2s on THIS subsample "
                    "against the committed full-window-set value on the same "
                    "600-episode deployment"),
           "t0_distribution": {str(k): int(v) for k, v in
                               zip(*np.unique(np.asarray(d["t0"]),
                                              return_counts=True))},
           "arms": {}}
    for nm, committed in (("a_imagination__true", 0.4108),
                          ("d_constant_velocity", 0.6917)):
        if nm in de:
            v = de[nm][:, wp].mean(axis=1)
            c = _ci.episode_cluster_bootstrap(v, eid, n_boot=B_BOOT, seed=SEED)
            rep["arms"][nm] = {
                "ade_0_2s_this_window_set": round(float(v.mean()), 4),
                "ade_0_2s_full_window_set_committed_600ep": committed,
                "relative_shift": round(float(v.mean() / committed - 1.0), 4),
                "ci95_episode_cluster_bootstrap": c}
    curve["window_set_representativeness"] = rep
    (out / "horizon_curve.json").write_text(json.dumps(curve, indent=2, default=float),
                                            encoding="utf-8")

    # ---- T_blind, per action regime ---------------------------------------- #
    tb = {"meta": {"n_episode_clusters": n_ep, "n_windows": int(gt.shape[0]),
                   "K_max": K, "n_boot": B_BOOT,
                   "estimator": "paired_episode_cluster_bootstrap"},
          "regimes": {}}
    for rname, arms in REGIMES.items():
        if arms["a"] not in de or arms["b"] not in de:
            continue
        blk = {"arm_a": arms["a"], "arm_b": arms["b"],
               "t_blind": t_blind(de[arms["a"]], de[arms["b"]], eid, draws),
               "paired_de_at_grid": {
                   f"{n * bi.DT:g}s": _paired(de[arms["a"]], de[arms["b"]],
                                              eid, n, draws) for n in grid},
               "paired_ade_at_grid": {
                   f"{n * bi.DT:g}s": _paired(de[arms["a"]], de[arms["b"]],
                                              eid, n, draws, True)
                   for n in grid},
               "t_useful_bars": {k: t_bar(de[arms["a"]], eid, draws, v)
                                 for k, v in BARS_M.items()},
               "t_beats_cv": t_blind(de[arms["a"]], de["d_constant_velocity"],
                                     eid, draws)}
        if arms["c"] and arms["c"] in de:
            blk["ceiling_vs_imagination"] = {
                f"{n * bi.DT:g}s": _paired(de[arms["c"]], de[arms["a"]],
                                           eid, n, draws) for n in grid}
        tb["regimes"][rname] = blk
    # ---- A2: the readout-level lever, contrasted against op on the SAME arm -- #
    lever = {}
    for base, alt in (("a_imagination__true", "a_imagination__true__roTAC"),
                      ("a_imagination__true", "a_imagination__true__roSTR"),
                      ("a_imagination__own", "a_imagination__own__roSTR"),
                      ("a_imagination__hold", "a_imagination__hold__roSTR"),
                      ("b_frozenlast__true", "b_frozenlast__true__roSTR"),
                      ("c2_observedpair__true",
                       "c2_observedpair__true__roSTR")):
        if base in de and alt in de:
            lever[f"{alt}_vs_{base}"] = {
                f"{n * bi.DT:g}s": _paired(de[alt], de[base], eid, n, draws)
                for n in grid}
    tb["A2_readout_level_lever"] = {
        "note": ("HierarchicalGrounding trains THREE step readouts on the SAME "
                 "operative imagination rollout but over different lengths "
                 "(op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20 — trainer defaults, "
                 "not overridden by v1's launch command). Every grounded number "
                 "in the program uses step['op'], calibrated over 4 steps and "
                 "then read at k=20. Positive delta = the ALTERNATE readout is "
                 "better. Amendment A2: a sensitivity, never the primary."),
        "contrasts": lever}
    # the convention control's own read: gt_kinematic vs true_future on arm (a)
    if "a_imagination__gtkin" in de and "a_imagination__true" in de:
        tb["action_inverse_fidelity"] = {
            f"{n * bi.DT:g}s": _paired(de["a_imagination__true"],
                                       de["a_imagination__gtkin"], eid, n, draws)
            for n in grid}
    (out / "t_blind.json").write_text(json.dumps(tb, indent=2, default=float), encoding="utf-8")

    # ---- E-IMAG-3 probe: can the model TELL it is drifting? ---------------- #
    dec_extra = uncertainty_probe(d, de, grid)
    (out / "uncertainty_readout.json").write_text(
        json.dumps(dec_extra, indent=2, default=float), encoding="utf-8")

    # ---- decomposition ----------------------------------------------------- #
    dec = decompose_block(d, de, eid, draws, grid)
    (out / "decomposition.json").write_text(json.dumps(dec, indent=2, default=float),
                                            encoding="utf-8")
    return de, eid, draws, grid


def uncertainty_probe(d, de, grid):
    """E-IMAG-3 item 3: is there a SELF-SIGNAL — computable with no privileged
    information — that predicts how wrong the blind rollout has become?

    Two candidates, both free at inference:
      ``speed_drift``  |v_pred(j) - v0|, how far the model's own imagined speed
                       has wandered from the last OBSERVED speed;
      ``speed_jump``   |v_pred(j) - v_pred(j-1)|, the roughness of its own
                       imagined motion.
    Reported as Pearson AND Spearman against the true ``de_j`` across windows.
    A strong correlation is the trigger a peek policy would need; a weak one
    says the model cannot tell, and the peek must be scheduled rather than
    triggered."""
    out = {"note": ("no privileged information is used: both signals are "
                    "functions of the model's OWN decoded motion and the last "
                    "OBSERVED speed"), "arms": {}}
    if "pred_speed" not in d:
        return out
    v0 = d["speed"].numpy()
    for name, sp in d["pred_speed"].items():
        if name not in de:
            continue
        v = sp.numpy()
        drift = np.abs(v - v0[:, None])
        jump = np.abs(np.diff(v, axis=1, prepend=v0[:, None]))
        blk = {}
        for n in grid:
            j = n - 1
            e = de[name][:, j]
            row = {}
            for tag, sig in (("speed_drift", drift[:, j]),
                             ("speed_jump", jump[:, j])):
                if np.std(sig) < 1e-9 or np.std(e) < 1e-9:
                    row[tag] = {"pearson": None, "spearman": None}
                    continue
                pr = float(np.corrcoef(sig, e)[0, 1])
                rs = float(np.corrcoef(
                    np.argsort(np.argsort(sig)).astype(float),
                    np.argsort(np.argsort(e)).astype(float))[0, 1])
                row[tag] = {"pearson": round(pr, 4), "spearman": round(rs, 4)}
            row["mean_pred_speed"] = round(float(v[:, j].mean()), 4)
            row["mean_de"] = round(float(e.mean()), 4)
            blk[f"{n * bi.DT:g}s"] = row
        out["arms"][name] = blk
    out["mean_observed_v0_mps"] = round(float(v0.mean()), 4)
    return out


def decompose_block(d, de, eid, draws, grid):
    from taniteval import lateral as _lat
    from taniteval import ood as _ood
    gt, gty = d["gt"], d["gt_yaw"]
    K = gt.shape[1]
    prim = [k for k in ("a_imagination__true", "b_frozenlast__true",
                        "c_fullobs__true", "c2_observedpair__true",
                        "a_imagination__own", "b_frozenlast__own",
                        "a_imagination__hold", "b_frozenlast__hold",
                        "a_imagination__true__roSTR",
                        "a_imagination__own__roSTR")
            if k in d["pred"]]
    out = {"meta": {
        "axis_convention": "ego: axis0 = along-track, axis1 = cross-track "
                           "(taniteval.lateral)",
        "horizon_provenance": {
            "surface": "DENSE 10 Hz path, every step 1..K",
            "K": K, "dt_s": bi.DT, "horizon_s": round(K * bi.DT, 2),
            "note": "a horizon_s of 0.4 here would mean a sparse 4-knot "
                    "surface and therefore stale code — it is not."},
        "envelope": {"ENV_LAT_MAX_m": _ood.ENV_LAT_MAX,
                     "ENV_YAW_MAX_deg": _ood.ENV_YAW_MAX,
                     "extrapolation_note":
                         "The last horizon that is a genuine MEASUREMENT is "
                         "0.4 s. Every reading beyond it is EXTRAPOLATION. "
                         "The OOD RATIO is deliberately NOT quoted: "
                         "sup(ratio_arr)=1.298888 makes the <=1.30 test a "
                         "tautology (C13). ENV_YAW_MAX=12deg was never "
                         "measured; it is a grid terminus (C14)."}},
        "arms": {}}
    for name in prim:
        p = d["pred"][name]
        al, cr = _lat.decompose(p, gt, mode="ego")
        alf, crf = _lat.decompose(p, gt, mode="frenet")
        lat_abs, yaw_abs = bi.path_deviation(p, d["psi"][name], gt, gty)
        fr = _ood.envelope_fractions(lat_abs.numpy(), yaw_abs.numpy())
        blk = {"per_horizon": {}, "ood_envelope_full_K": fr,
               "EVIDENCE": "EXTRAPOLATION beyond 0.4 s"}
        for n in grid:
            j = n - 1
            a2 = float((al[:, j] ** 2).mean())
            c2 = float((cr[:, j] ** 2).mean())
            af2 = float((alf[:, j] ** 2).mean())
            cf2 = float((crf[:, j] ** 2).mean())
            frn = _ood.envelope_fractions(lat_abs[:, :n].numpy(),
                                          yaw_abs[:, :n].numpy())
            blk["per_horizon"][f"{n * bi.DT:g}s"] = {
                "de_mean": round(float(de[name][:, j].mean()), 4),
                "ego_along_abs_mean": round(float(al[:, j].abs().mean()), 4),
                "ego_cross_abs_mean": round(float(cr[:, j].abs().mean()), 4),
                "ego_longitudinal_energy_share":
                    round(a2 / max(a2 + c2, 1e-12), 4),
                "frenet_along_abs_mean": round(float(alf[:, j].abs().mean()), 4),
                "frenet_cross_abs_mean": round(float(crf[:, j].abs().mean()), 4),
                "frenet_longitudinal_energy_share":
                    round(af2 / max(af2 + cf2, 1e-12), 4),
                "frenet_cross_p90": round(float(np.percentile(
                    crf[:, j].abs().numpy(), 90)), 4),
                # DRIFT vs VARIANCE: a systematic bias vs a spread
                "drift_along_signed_mean": round(float(al[:, j].mean()), 4),
                "drift_cross_signed_mean": round(float(cr[:, j].mean()), 4),
                "variance_along_std": round(float(al[:, j].std()), 4),
                "variance_cross_std": round(float(cr[:, j].std()), 4),
                "drift_share_along": round(
                    float(al[:, j].mean() ** 2 / max(a2, 1e-12)), 4),
                "drift_share_cross": round(
                    float(cr[:, j].mean() ** 2 / max(c2, 1e-12)), 4),
                "frac_steps_out_of_envelope": frn["frac_steps_any"],
                "frac_windows_out_of_envelope":
                    frn["frac_windows_any_step_out_of_envelope"],
                "pred_speed_mean_mps": round(
                    float(d["pred_speed"][name][:, j].mean()), 4)
                if name in d.get("pred_speed", {}) else None,
            }
        out["arms"][name] = blk
    # the GT's own speed for reference (what the model should be doing)
    out["gt_speed_at_window_end_mps"] = round(float(d["speed"].mean()), 4)
    return out


def analyse_peek(dp, ds, out, eid_ref=None):
    gt = dp["gt"]
    eid = [str(x) for x in dp["eid"]]
    draws, n_ep = _boot_draws(eid)
    K = gt.shape[1]
    grid = [n for n in GRID if n <= K]
    de = {k: _de(v, gt) for k, v in dp["pred"].items()}
    # the two anchors the duty-cycle curve is read against
    base = {}
    for k in ("a_imagination__own", "a_imagination__hold",
              "c_fullobs__own", "c_fullobs__true", "a_imagination__true"):
        if k in ds["pred"]:
            base[k] = _de(ds["pred"][k], ds["gt"])
    res = {"meta": {**dp["meta"], "grid_steps": grid,
                    "n_episode_clusters": n_ep,
                    "duty_cycle_definition":
                        "realised fraction of rollout ticks on which a REAL "
                        "front-camera frame was encoded (measured from "
                        "peek_mask, never assumed)",
                    "disappointing_thresholds_preregistered": {
                        "duty_cycle_saving_vs_always_on": "< 2x",
                        "oracle_vs_uniform_gap": "< 15 % relative de reduction "
                                                 "at matched duty cycle"}},
           "policies": {}, "baselines": {}}
    for k, m in base.items():
        res["baselines"][k] = {
            "duty_cycle": 1.0 if k.startswith("c_fullobs") else 0.0,
            "de": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws) for n in grid},
            "ade": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws, True)
                    for n in grid}}
    for name, m in de.items():
        duty = float(dp["peek_mask"][name].float().mean()) \
            if name in dp.get("peek_mask", {}) else None
        cfg = dp["meta"]["peek"].get(name, {})
        blk = {"duty_cycle_realised": round(duty, 5) if duty is not None else None,
               "config": cfg,
               "de": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws) for n in grid},
               "ade": {f"{n * bi.DT:g}s": _ci_at(m, eid, n, draws, True)
                       for n in grid}}
        anchor = ("a_imagination__own"
                  if cfg.get("base_action_source") == "own_kinematic"
                  else "a_imagination__hold")
        if anchor in base:
            blk["vs_no_peek"] = {
                f"{n * bi.DT:g}s": _paired(m, base[anchor], eid, n, draws)
                for n in grid}
        res["policies"][name] = blk
    # ---- uniform vs ORACLE at matched duty cycle --------------------------- #
    res["oracle_vs_uniform"] = _oracle_vs_uniform(de, dp, eid, draws, grid)
    (out / "duty_cycle.json").write_text(json.dumps(res, indent=2, default=float),
                                         encoding="utf-8")
    return res


def _oracle_vs_uniform(de, dp, eid, draws, grid):
    """For each ORACLE policy, find the UNIFORM policy whose realised duty cycle
    is closest, and report the paired gap. That gap is the informative version of
    H2's efficiency claim: it is what a learned trigger could win."""
    duty = {k: float(v.float().mean()) for k, v in dp.get("peek_mask", {}).items()}
    out = {}
    for name in de:
        cfg = dp["meta"]["peek"].get(name, {})
        if cfg.get("oracle_bar_m") is None:
            continue
        base_a = cfg.get("base_action_source")
        cands = [(k, duty[k]) for k in de
                 if dp["meta"]["peek"].get(k, {}).get("uniform_period")
                 and dp["meta"]["peek"][k].get("base_action_source") == base_a]
        if not cands:
            continue
        match = min(cands, key=lambda kv: abs(kv[1] - duty[name]))
        out[name] = {
            "oracle_duty": round(duty[name], 5),
            "matched_uniform": match[0],
            "matched_uniform_duty": round(match[1], 5),
            "duty_ratio_oracle_over_uniform": round(
                duty[name] / max(match[1], 1e-9), 3),
            "paired_de": {f"{n * bi.DT:g}s":
                          _paired(de[name], de[match[0]], eid, n, draws)
                          for n in grid},
            "relative_de_reduction": {
                f"{n * bi.DT:g}s": round(
                    float(1.0 - de[name][:, n - 1].mean()
                          / max(de[match[0]][:, n - 1].mean(), 1e-9)), 4)
                for n in grid},
        }
    return out


def write_compact(ds, dp, out, grid):
    """The recompute-anything artifact that fits in the repo.

    * FULL dense per-window ``de`` for the headline arms (T_blind needs the
      dense curve);
    * reporting-grid ``de``/``along``/``cross`` for EVERY arm;
    * ``eid``/``speed``/``head_deg`` so any stratification can be redone.
    """
    from taniteval import lateral as _lat
    gt = ds["gt"]
    head = [k for k in ("a_imagination__true", "b_frozenlast__true",
                        "c_fullobs__true", "c2_observedpair__true",
                        "a_imagination__own", "b_frozenlast__own",
                        "c_fullobs__own", "a_imagination__hold",
                        "b_frozenlast__hold", "a_imagination__gtkin",
                        "b_frozenlast__gtkin", "a_imagination__true__roTAC",
                        "a_imagination__true__roSTR",
                        "a_imagination__own__roSTR",
                        "a_imagination__hold__roSTR",
                        "b_frozenlast__true__roSTR",
                        "c2_observedpair__true__roSTR",
                        "a_imagination__own_vupd") if k in ds["pred"]]
    dense = {k: torch.linalg.norm(ds["pred"][k] - gt, dim=-1).float()
             for k in head}
    dense["d_constant_velocity"] = torch.linalg.norm(ds["cv"] - gt, dim=-1).float()
    dense["d2_hold_v0"] = torch.linalg.norm(ds["hold_v0"] - gt, dim=-1).float()
    gidx = [n - 1 for n in grid]
    gridblk = {}
    for src, tag in ((ds, "sweep"), (dp, "peek")):
        for k, v in src["pred"].items():
            al, cr = _lat.decompose(v, src["gt"], mode="ego")
            gridblk[k] = {
                "de": torch.linalg.norm(v - src["gt"], dim=-1)[:, gidx].float(),
                "along": al[:, gidx].float(), "cross": cr[:, gidx].float(),
                "source": tag}
    torch.save({"dense_de_headline": dense, "grid_steps": list(grid),
                "grid_de_along_cross": gridblk,
                "eid": ds["eid"], "speed": ds["speed"],
                "head_deg": ds["head_deg"], "t0": ds["t0"],
                "peek_duty": {k: float(v.float().mean())
                              for k, v in dp.get("peek_mask", {}).items()},
                "meta_sweep": ds["meta"], "meta_peek": dp["meta"]},
               out / "bi_perwindow_compact.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--peek", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = torch.load(a.sweep, map_location="cpu", weights_only=False)
    dp = torch.load(a.peek, map_location="cpu", weights_only=False)
    de, eid, draws, grid = analyse_sweep(ds, out)
    analyse_peek(dp, ds, out)
    write_compact(ds, dp, out, grid)
    print("analysis written to", out, flush=True)


if __name__ == "__main__":
    main()
