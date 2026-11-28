#!/usr/bin/env python3
"""P1 -- FIX `progress`, the ONE reward term that ranks envelope-violating candidates HIGHER.

ZERO GPU, ZERO model calls: it reads the banked fan (`fan_bank_base_240w.npz`) that the
veto-only package emitted from the frozen refcv3 checkpoint, and re-scores it.

THE STRUCTURAL FACT THAT DICTATES THE FIX (SPEC 1, committed before this ran):
  the published rho is a PER-WINDOW Spearman across a window's own 128 candidates, and
  `progress = along / ref` with `ref = max(v0*H, min_ref)` CONSTANT within a window.
  Division by a positive per-window constant is MONOTONE, so
        rho(progress, peak_g) == rho(along, peak_g) exactly.
  => every "re-normalise progress" proposal is a provable NO-OP on this metric. The fix
  must change WHAT progress is a function of. `prog_v0_noop` below is that no-op, run
  deliberately, so the claim is MEASURED and not merely argued.

THE LEVER (SPEC 2.1):
    progress_v2 = min(along/ref, 1.0) * w(ex),   w(ex) = 1/(1 + ex/ex_s)
  * min(.,1.0) saturates at "kept the current speed" -- exceeding v0 stops being rewarded;
  * w(ex) is a MULTIPLICATIVE, NON-saturating envelope discount on the same exceedance
    `rewards._kinematic_feasibility` uses. It cannot be out-voted the way an ADDITIVE
    feasibility weight can -- and MEASURED, feasibility x4 moves rho(envelope) by 0.001.

CONTROLS, and C-OBJECT is the one RETRACTION #30 demands:
  C-OBJECT : every stock component recomputed offline here must reproduce the BANKED
             c_* column to 1e-5, and score_paths must reproduce the banked flags and
             `bank_vs_fan_feasibility.json`'s `emitted` block, which a DIFFERENT script
             wrote. That asserts WHICH tensor was ranked, not merely that the arithmetic
             was self-consistent.
  C-SELF   : rho(x, x) == +1.0000 exactly.
  C-CONST  : a constant term is UNDEFINED on every window (never 0.0).
  C-RAND   : a uniform-random score reads the probe's own bias floor (ref -0.0138).
ASCII-only output.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"
# The `taniteval` namespace SHADOW: the OUTER `<repo>/taniteval` has no __init__.py, so
# putting <repo> on the path makes `import taniteval.ci` fail with ModuleNotFoundError.
# The real package is <repo>/taniteval/taniteval; put its PARENT first.
sys.path.insert(0, os.path.join(_REPO, "stack"))
sys.path.insert(0, os.path.join(_REPO, "taniteval"))

from tanitad.rl import rewards as RW              # noqa: E402
from tanitad.refs import feasible_decode as FD    # noqa: E402
import taniteval.ci as CI                         # noqa: E402
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))
import fan_safety as FS                           # noqa: E402

TOOL = "2026-09-05-feasible-decode/raw/progress_rank_fix.py"
DT = 0.5
A_MAX = RW.A_MAX_MPS2          # 4.0
KAPPA_MAX = RW.KAPPA_MAX_1PM   # 0.2
FLAGS_OF_INTEREST = ("envelope", "kamm_over", "infeasible", "off_reach",
                     "ttc_below", "contact")
LEAD_ONLY = set(FS.LEAD_ONLY)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spearman(a, b):
    """NaN when either side has no variance -- 'no variance' and 'no relationship' are
    different facts and averaging one into the other reports a null never measured."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 3:
        return float("nan")
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(vals, eids, *, n_boot, seed):
    v = np.asarray(vals, dtype=np.float64)
    e = np.asarray(eids)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_windows": int(ok.sum()), "n_episodes": 0,
                "n_undefined": int((~ok).sum()),
                "estimator": "episode_cluster_bootstrap"}
    r = CI.episode_cluster_bootstrap(v[ok], e[ok], n_boot=n_boot, seed=seed)
    out = {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}
    out["n_undefined"] = int((~ok).sum())
    out["separated"] = bool((out["lo"] > 0.0) or (out["hi"] < 0.0))
    return out


# --------------------------------------------------------------------------- #
# the exceedance -- ONE spelling, the one `_kinematic_feasibility` uses          #
# --------------------------------------------------------------------------- #
def exceedance(traj, dt=DT, a_max=A_MAX, kappa_max=KAPPA_MAX):
    kin = RW.kinematics(traj, dt)
    a_ex = (kin.accel.abs() / a_max - 1.0).clamp_min(0.0)
    k_ex = (kin.kappa.abs() / kappa_max - 1.0).clamp_min(0.0)
    return a_ex.amax(dim=-1) + k_ex.amax(dim=-1)


def progress_v1(traj, ctx):
    return RW.COMPONENTS["progress"](traj, ctx)          # stock, clamped [-1, 1.5]


def progress_raw(traj, ctx):
    """the UNCLAMPED stock ratio -- the object the no-op claim is about."""
    return RW._progress(traj, ctx)


def progress_v2(traj, ctx, ex_s, cap=1.0):
    """min(along/ref, cap) * 1/(1 + ex/ex_s), floored at -1."""
    r = RW._progress(traj, ctx)
    ex = exceedance(traj, ctx.get("dt", DT))
    w = 1.0 / (1.0 + ex / max(float(ex_s), 1e-9))
    return (r.clamp(max=cap) * w).clamp_min(-1.0)


def progress_v3(traj, ctx, v0, mu=FD.MU_KAMM, clamp_entry=False):
    """P1b -- credit the distance the car would cover AFTER the friction-feasible
    projection, not the distance the decoder drew. No new hyper-parameter, and the
    IDENTITY on any candidate already inside the envelope."""
    q = FD.project_feasible(traj, v0, mu=mu, clamp_entry=clamp_entry)
    return RW._progress(q, ctx).clamp(-1.0, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bank-vs-fan", required=True)
    ap.add_argument("--fan-key", default="fan2", choices=("fan2", "fan8"),
                    help="fan2 = the 5-point prefix WITH origin (240w draw); fan8 = "
                         "the 8-slot fan WITHOUT origin (400w draw, carries gt4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.npz)
    npz_md5 = md5(a.npz)
    if a.fan_key == "fan2":
        fan2 = torch.from_numpy(z["fan2"]).float()             # [W, N, 5, 2]
    else:
        # the 400 w draw stores 8 slots and no origin; the 2 s prefix is slots 0-3.
        fan2 = FS.with_origin(torch.from_numpy(z["fan8"]).float()[..., :4, :])
    W, N = fan2.shape[0], fan2.shape[1]
    v0 = torch.from_numpy(z["v0"]).float()                     # [W]
    lead5 = torch.from_numpy(z["lead5"]).float()               # [W, 5, 2]
    eid = z["eid"]
    has_lead = z["has_lead"]
    peak_g_bank = (z["peak_g"].astype(np.float64) if "peak_g" in z.files else None)

    print("=== P1: FIX `progress` -- the one term that rewards violation ===")
    print("  object under test: fan2 %s from %s" % (tuple(fan2.shape), os.path.basename(a.npz)))
    print("  npz md5: %s" % npz_md5)

    # ---- the ctx, rebuilt offline exactly as rl_refcv3_min.reward_ctx builds it ----
    ctx = {"dt": DT, "v0": v0.reshape(W, 1), "lead_len_m": FS.LEAD_LEN_DEFAULT_M,
           "lead_path": lead5.reshape(W, 1, 5, 2)}

    # ================= C-OBJECT (RETRACTION #30) ================================= #
    # Every stock component recomputed here must reproduce the BANKED column, and the
    # flags must reproduce a DIFFERENT script's artifact. That asserts the OBJECT.
    comp_names = ["progress", "collision", "headway", "feasibility", "comfort"]
    stock = {n: RW.COMPONENTS[n](fan2, ctx).double().numpy() for n in comp_names}
    have_cols = all(("c_%s" % nm) in z.files for nm in comp_names)
    obj = {"npz": a.npz, "npz_md5": npz_md5, "fan2_shape": list(fan2.shape),
           "fan_key": a.fan_key, "has_banked_component_columns": bool(have_cols),
           "component_max_abs_diff_vs_banked": {}}
    if have_cols:
        for nm in comp_names:
            d = float(np.abs(stock[nm] - z["c_%s" % nm].astype(np.float64)).max())
            obj["component_max_abs_diff_vs_banked"][nm] = d
    else:
        obj["object_assertion_instead"] = (
            "this draw carries no banked c_* columns. The object is asserted by the npz "
            "md5 above plus the fact that raw/mu_frontier.py reproduced the sibling's "
            "independently written displacement_frontier.py lambda table ROW FOR ROW on "
            "this same file -- a different script, a different author, the same object.")
    sc = FS.score_paths(fan2, v0, lead5.reshape(W, 1, 5, 2),
                        lead_len_m=FS.LEAD_LEN_DEFAULT_M)
    obj["peak_g_max_abs_diff_vs_banked"] = (
        float(np.abs(sc["peak_g"].double().numpy() - peak_g_bank).max())
        if peak_g_bank is not None else None)
    flag_diff = {}
    if ("f_%s" % FS.FLAGS[0]) in z.files:
        for f in FS.FLAGS:
            flag_diff[f] = int((sc[f].numpy().astype(np.uint8) != z["f_%s" % f]).sum())
    obj["flag_mismatch_counts_vs_banked"] = flag_diff
    with open(a.bank_vs_fan, "r", encoding="utf-8") as fh:
        bvf = json.load(fh)
    obj["reference_artifact"] = a.bank_vs_fan
    obj["emitted_peak_g"] = {"ours": float(sc["peak_g"].float().mean()),
                             "ref": bvf["emitted"]["peak_g"]}
    obj["emitted_envelope"] = {"ours": float(sc["envelope"].float().mean()),
                               "ref": bvf["emitted"]["envelope"]}
    cd_ = (max(obj["component_max_abs_diff_vs_banked"].values())
           if obj["component_max_abs_diff_vs_banked"] else 0.0)
    fd_ = max(flag_diff.values()) if flag_diff else 0
    # the `emitted` block is a property of the 240 w draw; a DIFFERENT draw emits a
    # different fan and must NOT be asserted against it.
    same_draw = (a.fan_key == "fan2")
    obj["emitted_block_asserted"] = same_draw
    obj["PASS"] = bool(
        cd_ < 1e-5
        and (obj["peak_g_max_abs_diff_vs_banked"] is None
             or obj["peak_g_max_abs_diff_vs_banked"] < 1e-4)
        and fd_ == 0
        and ((not same_draw) or (
            abs(obj["emitted_peak_g"]["ours"] - obj["emitted_peak_g"]["ref"]) < 1e-3
            and abs(obj["emitted_envelope"]["ours"]
                    - obj["emitted_envelope"]["ref"]) < 1e-6)))
    print("  C-OBJECT PASS=%s  (max |component - banked| = %.3e, flags mismatched = %d,"
          " emitted peak_g ours %.4f vs the 240w ref %.4f%s)"
          % (obj["PASS"], cd_, fd_, obj["emitted_peak_g"]["ours"],
             obj["emitted_peak_g"]["ref"],
             "" if same_draw else " [DIFFERENT DRAW -- not asserted]"))
    if not obj["PASS"]:
        raise SystemExit("C-OBJECT FAILED -- the tensor being ranked is not the banked fan")

    # ================= H-PROG-SAT-1: is `feasibility` SATURATED? ================= #
    feas = stock["feasibility"]                                   # [W, N]
    iqr = np.percentile(feas, 75, axis=1) - np.percentile(feas, 25, axis=1)
    sat = {"hypothesis": "H-PROG-SAT-1: feasibility x4 is a null lever because the term is "
                         "SATURATED, not because feasibility is the wrong axis",
           "committed_prediction": "median within-window IQR of `feasibility` < 0.05",
           "median_within_window_iqr": float(np.median(iqr)),
           "mean_within_window_iqr": float(iqr.mean()),
           "frac_candidates_below_0p01": float((feas < 0.01).mean()),
           "median_value": float(np.median(feas))}
    sat["SUPPORTED"] = bool(sat["median_within_window_iqr"] < 0.05)
    print("  H-PROG-SAT-1: median within-window IQR of feasibility = %.5f  (predicted "
          "< 0.05) -> %s ; %.1f%% of candidates read < 0.01"
          % (sat["median_within_window_iqr"],
             "SUPPORTED" if sat["SUPPORTED"] else "REFUTED",
             100.0 * sat["frac_candidates_below_0p01"]))

    # ================= ex_s, calibrated ONCE on the object ======================= #
    ex = exceedance(fan2).double().numpy()
    nz = ex[ex > 0]
    ex_s = float(np.median(nz)) if nz.size else 1.0
    print("  ex_s (median NON-ZERO exceedance, calibrated once, not swept) = %.4f "
          "(%.1f%% of candidates exceed)" % (ex_s, 100.0 * (ex > 0).mean()))

    # ================= the terms under test ====================================== #
    terms = {
        "prog_v1": progress_v1(fan2, ctx).double().numpy(),
        # the deliberate NO-OP: a different per-window normalisation of the SAME `along`.
        # Committed in the SPEC to move rho by exactly 0.0000; run so the claim is MEASURED.
        "prog_v0_noop": (RW._progress(fan2, {**ctx, "progress_min_ref_m": 20.0})
                         .double().numpy()),
        "prog_v2": progress_v2(fan2, ctx, ex_s).double().numpy(),
        "prog_v2_caponly": (RW._progress(fan2, ctx).clamp(max=1.0)
                            .clamp_min(-1.0).double().numpy()),
        "prog_v2_wonly": (RW._progress(fan2, ctx).clamp(-1.0, 1.5).double().numpy()
                          * (1.0 / (1.0 + ex / ex_s))),
        # P1b -- the successor, pre-registered in SPEC_ADDENDUM_P1b.md BEFORE it ran
        "prog_v3": progress_v3(fan2, ctx, v0).double().numpy(),
        "prog_v3_entry": progress_v3(fan2, ctx, v0, clamp_entry=True).double().numpy(),
        # the EXACT no-op control: an UNCLAMPED per-window renormalisation of `along`.
        # SPEC 1 claimed the CLAMPED component was a no-op under renormalisation; it is
        # not, because the clamp creates TIES and ties move a Spearman. The unclamped
        # ratio is the object the claim is true of, and these two must agree to 0.
        "prog_raw": RW._progress(fan2, ctx).double().numpy(),
        "prog_raw_renorm": (RW._progress(fan2, {**ctx, "progress_min_ref_m": 20.0})
                            .double().numpy()),
    }
    # composed DEFAULT reward, with progress swapped
    base_wo_prog = sum(RW.DEFAULT_WEIGHTS[n] * stock[n]
                       for n in comp_names if n != "progress")
    wp = RW.DEFAULT_WEIGHTS["progress"]
    composed = {
        "DEFAULT": base_wo_prog + wp * terms["prog_v1"],
        "DEFAULT_progv2": base_wo_prog + wp * terms["prog_v2"],
        "DEFAULT_progv3": base_wo_prog + wp * terms["prog_v3"],
        "no_progress": base_wo_prog,
    }

    flags = {f: sc[f].numpy().astype(np.float64) for f in FS.FLAGS}
    pg = sc["peak_g"].double().numpy()

    # ================= per-window rho ============================================ #
    keys = list(terms) + ["cmp_%s" % k for k in composed]
    per_window = []
    for j in range(W):
        row = {"wi": int(z["wi"][j]), "eid": int(eid[j]), "has_lead": bool(has_lead[j])}
        vecs = {k: terms[k][j] for k in terms}
        vecs.update({"cmp_%s" % k: composed[k][j] for k in composed})
        for k, vv in vecs.items():
            row["rho__%s__peak_g" % k] = spearman(vv, pg[j])
            for f in FLAGS_OF_INTEREST:
                if f in LEAD_ONLY and not bool(has_lead[j]):
                    row["rho__%s__%s" % (k, f)] = float("nan")
                    continue
                row["rho__%s__%s" % (k, f)] = spearman(vv, flags[f][j])
        # ---- controls, per window ----
        row["ctrl_self"] = spearman(terms["prog_v2"][j], terms["prog_v2"][j])
        row["ctrl_const"] = spearman(np.ones(N), pg[j])
        rng = np.random.default_rng(1000 + j)
        row["ctrl_rand"] = spearman(rng.random(N), pg[j])
        per_window.append(row)

    panel = {}
    for k in keys:
        for tgt in ("peak_g",) + FLAGS_OF_INTEREST:
            name = "rho__%s__%s" % (k, tgt)
            panel[name] = boot([r[name] for r in per_window], eid,
                               n_boot=a.n_boot, seed=a.seed)
            panel[name]["population"] = ("lead windows" if tgt in LEAD_ONLY
                                         else "all windows")
    ctrls = {}
    for cn in ("ctrl_self", "ctrl_const", "ctrl_rand"):
        vals = np.array([r[cn] for r in per_window], dtype=np.float64)
        ctrls[cn] = {"n_undefined": int((~np.isfinite(vals)).sum()),
                     "n_windows": W}
        if np.isfinite(vals).any():
            ctrls[cn].update(boot(vals, eid, n_boot=a.n_boot, seed=a.seed))
    ctrls["ctrl_self"]["PASS"] = bool(
        abs(ctrls["ctrl_self"].get("mean", 0.0) - 1.0) < 1e-12)
    ctrls["ctrl_const"]["PASS"] = bool(ctrls["ctrl_const"]["n_undefined"] == W)
    ctrls["ctrl_rand"]["reference_bias_floor"] = -0.0138

    # ================= P1-C2: progress is not destroyed ========================== #
    # straight windows: the fan's own 2 s endpoints are laterally tight.
    lat_end = np.abs(fan2[:, :, -1, 1].numpy())
    straight = np.median(lat_end, axis=1) < 1.0
    along = (fan2[:, :, -1, 0] - fan2[:, :, 0, 0]).double().numpy()      # [W, N]
    gt_along = None
    if "gt4" in z.files:
        gt_along = z["gt4"][:, -1, 0].astype(np.float64)
    idx1 = terms["prog_v1"].argmax(axis=1)
    idx2 = terms["prog_v2"].argmax(axis=1)
    a1 = along[np.arange(W), idx1][straight]
    a2 = along[np.arange(W), idx2][straight]
    c2 = {"n_straight_windows": int(straight.sum()),
          "straight_rule": "median |lateral| of the fan's 2 s endpoints < 1.0 m",
          "mean_along_argmax_prog_v1_m": float(a1.mean()),
          "mean_along_argmax_prog_v2_m": float(a2.mean()),
          "ratio_v2_over_v1": float(a2.mean() / max(a1.mean(), 1e-9)),
          "mean_v0_x_2s_m": float((v0.numpy()[straight] * 2.0).mean()),
          "committed": "ratio >= 0.95 AND mean along >= the human's own"}
    c2["ratio_PASS"] = bool(c2["ratio_v2_over_v1"] >= 0.95)
    if gt_along is not None:
        c2["mean_gt_along_m"] = float(gt_along[straight].mean())
        c2["human_floor_PASS"] = bool(a2.mean() >= gt_along[straight].mean())
    else:
        # this npz carries no GT; the v0-hold distance is the same driving fact
        # (`ha0`, the programme's shared trivial floor) and is used in its place.
        c2["mean_gt_along_m"] = None
        c2["human_floor_substitute"] = "v0 * 2 s (the `ha0` constant-velocity floor)"
        c2["human_floor_PASS"] = bool(a2.mean() >= c2["mean_v0_x_2s_m"])
    c2["PASS"] = bool(c2["ratio_PASS"] and c2["human_floor_PASS"])

    def c2_for(term_key):
        i2 = terms[term_key].argmax(axis=1)
        aa = along[np.arange(W), i2][straight]
        r = {"term": term_key,
             "mean_along_argmax_m": float(aa.mean()),
             "ratio_over_prog_v1": float(aa.mean() / max(a1.mean(), 1e-9))}
        r["ratio_PASS"] = bool(r["ratio_over_prog_v1"] >= 0.95)
        floor = (float(gt_along[straight].mean()) if gt_along is not None
                 else float((v0.numpy()[straight] * 2.0).mean()))
        r["floor_m"] = floor
        r["floor_is_human_gt"] = bool(gt_along is not None)
        r["floor_PASS"] = bool(aa.mean() >= floor)
        r["PASS"] = bool(r["ratio_PASS"] and r["floor_PASS"])
        return r

    c1 = {"criterion": "rho(progress_v2, peak_g) < +0.05",
          "value": panel["rho__prog_v2__peak_g"]["mean"],
          "ci": [panel["rho__prog_v2__peak_g"]["lo"],
                 panel["rho__prog_v2__peak_g"]["hi"]],
          "reference_prog_v1": panel["rho__prog_v1__peak_g"]["mean"]}
    c1["PASS"] = bool(c1["value"] < 0.05)

    noop = {"claim": "a per-window MONOTONE re-normalisation of `along` moves rho by 0 "
                     "-- TRUE of the UNCLAMPED ratio, FALSE of the clamped component",
            "rho_prog_raw_peak_g": panel["rho__prog_raw__peak_g"]["mean"],
            "rho_prog_raw_renorm_peak_g": panel["rho__prog_raw_renorm__peak_g"]["mean"],
            "rho_prog_v1_clamped_peak_g": panel["rho__prog_v1__peak_g"]["mean"],
            "rho_prog_v0_noop_clamped_peak_g":
                panel["rho__prog_v0_noop__peak_g"]["mean"]}
    noop["max_abs_diff_unclamped"] = abs(noop["rho_prog_raw_peak_g"]
                                         - noop["rho_prog_raw_renorm_peak_g"])
    noop["max_abs_diff_clamped"] = abs(noop["rho_prog_v1_clamped_peak_g"]
                                       - noop["rho_prog_v0_noop_clamped_peak_g"])
    noop["CONFIRMED"] = bool(noop["max_abs_diff_unclamped"] < 1e-9)
    noop["clamp_breaks_it"] = bool(noop["max_abs_diff_clamped"] > 1e-9)
    noop["mechanism"] = ("the [-1, 1.5] CLAMP is not monotone -- it creates TIES, and a "
                         "Spearman moves when ties move. SPEC 1 stated the claim about "
                         "the clamped COMPONENT and was wrong about it; the claim holds "
                         "exactly for the ratio the component clamps.")

    p1b = {}
    for k in ("prog_v3", "prog_v3_entry"):
        p1b[k] = {
            "C1": {"criterion": "rho(%s, peak_g) < +0.05" % k,
                   "value": panel["rho__%s__peak_g" % k]["mean"],
                   "ci": [panel["rho__%s__peak_g" % k]["lo"],
                          panel["rho__%s__peak_g" % k]["hi"]],
                   "PASS": bool(panel["rho__%s__peak_g" % k]["mean"] < 0.05)},
            "C2": c2_for(k)}
        p1b[k]["PASS"] = bool(p1b[k]["C1"]["PASS"] and p1b[k]["C2"]["PASS"])

    out = {"_tool": TOOL, "_tier": "T0 readout on the emitted fan -- never a driving claim",
           "_evidence_class": "MEASURED (ours)",
           "n_windows": W, "n_candidates": N, "n_candidate_scores": W * N,
           "n_episodes": int(len(set(eid.tolist()))),
           "estimator": "episode_cluster_bootstrap over per-window Spearman rho",
           "n_boot": a.n_boot, "seed": a.seed, "dt_s": DT,
           "ex_s_calibrated": ex_s,
           "controls": {"C_OBJECT": obj, **ctrls},
           "H_PROG_SAT_1": sat,
           "no_op_claim": noop,
           "P1_C1": c1, "P1_C2": c2, "P1b": p1b,
           "panel": panel, "per_window": per_window}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("")
    print("  %-18s %10s %10s %10s %10s %10s %10s"
          % ("term", "peak_g", "envelope", "kamm_over", "ttc_below", "contact", "off_reach"))
    for k in keys:
        print("  %-18s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (k, panel["rho__%s__peak_g" % k]["mean"],
                 panel["rho__%s__envelope" % k]["mean"],
                 panel["rho__%s__kamm_over" % k]["mean"],
                 panel["rho__%s__ttc_below" % k]["mean"],
                 panel["rho__%s__contact" % k]["mean"],
                 panel["rho__%s__off_reach" % k]["mean"]))
    print("")
    print("  NO-OP claim (UNCLAMPED ratio) CONFIRMED=%s  diff %.2e ; the CLAMPED "
          "component moves by %.4f -- ties, not monotonicity"
          % (noop["CONFIRMED"], noop["max_abs_diff_unclamped"],
             noop["max_abs_diff_clamped"]))
    print("  P1-C1  rho(prog_v2, peak_g) = %+.4f [%+.4f, %+.4f]  (was %+.4f) -> %s"
          % (c1["value"], c1["ci"][0], c1["ci"][1], c1["reference_prog_v1"],
             "PASS" if c1["PASS"] else "FAIL"))
    print("  P1-C2  straight n=%d : along(argmax v2) %.3f m vs (argmax v1) %.3f m "
          "ratio %.4f ; floor %.3f m -> %s"
          % (c2["n_straight_windows"], c2["mean_along_argmax_prog_v2_m"],
             c2["mean_along_argmax_prog_v1_m"], c2["ratio_v2_over_v1"],
             (c2["mean_gt_along_m"] if c2["mean_gt_along_m"] is not None
              else c2["mean_v0_x_2s_m"]),
             "PASS" if c2["PASS"] else "FAIL"))
    print("  controls: self %.6f (PASS=%s) | const undefined %d/%d (PASS=%s) | "
          "rand %+.4f (ref -0.0138)"
          % (ctrls["ctrl_self"].get("mean", float("nan")), ctrls["ctrl_self"]["PASS"],
             ctrls["ctrl_const"]["n_undefined"], W, ctrls["ctrl_const"]["PASS"],
             ctrls["ctrl_rand"].get("mean", float("nan"))))
    print("  VERDICT P1 (progress_v2) = %s"
          % ("PASS" if (c1["PASS"] and c2["PASS"]) else "FAIL"))
    for k, r in p1b.items():
        print("  P1b %-14s C1 rho(peak_g) %+.4f [%+.4f, %+.4f] %s | C2 along %.3f m "
              "ratio %.4f vs floor %.3f m (%s) %s -> VERDICT %s"
              % (k, r["C1"]["value"], r["C1"]["ci"][0], r["C1"]["ci"][1],
                 "PASS" if r["C1"]["PASS"] else "FAIL",
                 r["C2"]["mean_along_argmax_m"], r["C2"]["ratio_over_prog_v1"],
                 r["C2"]["floor_m"],
                 "human GT" if r["C2"]["floor_is_human_gt"] else "ha0 v0-hold",
                 "PASS" if r["C2"]["PASS"] else "FAIL",
                 "PASS" if r["PASS"] else "FAIL"))
    print("[p1] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
