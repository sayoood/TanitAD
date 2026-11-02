#!/usr/bin/env python3
"""precheck_attribution.py -- FEAS2-T1 step 1: attribute KAPPA-T1's box harm to the accel clamp or the kappa clamp, OFFLINE, on SEEN data.

Data: the four KAPPA-T1 X runs' ticks (F:/TanitAD-scratch/kappa_t1/out/FEAS_*/ticks), field ``plan_s`` (the plan after S, before the projection) and ``v0`` (the tick's
measured ego speed), ``gt`` (the logged future in the tick's executed frame), ``P`` (the P10 prior's plan at the same state).  No rollout: every projection is recomputed
on the SAME plans, so a number here is a per-tick counterfactual of the plan the run issued, not a closed rollout (the executed path of a variant would differ).
Variants (feasible_decode.project_feasible, dt 0.2, mu=None):
  box     a_max 4, kappa_max 0.2, clamp_entry False     (KAPPA-T1's arm; must reproduce the logged ``plan``)
  kbox    a_max inf, kappa_max 0.2, clamp_entry False
  aonly   a_max 4, kappa_max inf, clamp_entry False     (the accel clamp alone, no entry bound)
  abox    a_max 4, kappa_max inf, clamp_entry True, v0 = tick v0  (lever (b) of the card)
  kabox   kbox, then abox
Evidence class MEASURED (ours), tier T1-kin plan-level proxy; estimator of the heading deltas: episode-cluster bootstrap B=1000 seed 0 over the 141 clips (this is a pre-check on
seen data: it decides nothing about the arms and is disclosed in PREREG_FROZEN.md).
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "F:/TanitAD-scratch/kappa_t1/tree_dag/stack")
from tanitad.refs import feasible_decode as FD  # noqa: E402

INF = float("inf")
DT = 0.2
M = 10                                   # executed knots per tick (--replan-steps 10)
OUT = "F:/TanitAD-scratch/kappa_t1/out"
PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
RUNS = ["FEAS_DAGOACC_s0", "FEAS_DAGOACC_s1", "FEAS_SOUP_s0", "FEAS_SOUP_s1"]


def proj(plan, v0=None, a_max=4.0, kappa=0.2, entry=False):
    p = np.asarray(plan, np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    v = None if v0 is None else torch.from_numpy(np.asarray(v0, np.float64))
    return FD.project_feasible(full, v0=v, dt=DT, a_max=a_max, kappa_max=kappa, mu=None, clamp_entry=entry)[:, 1:].numpy()


def variants(ps, v0):
    kb = proj(ps, a_max=INF, kappa=0.2)
    return {"box": proj(ps), "kbox": kb, "aonly": proj(ps, a_max=4.0, kappa=INF), "abox": proj(ps, v0, 4.0, INF, True), "kabox": proj(kb, v0, 4.0, INF, True)}


def steps(path):                         # path [N, K, 2] knots 1..K -> step vectors from the origin
    return np.diff(np.concatenate([np.zeros_like(path[:, :1]), path], 1), axis=1)


def heading_mae(plan, gt, ok):
    """Per tick: mean over the executed steps where both steps are > 0.1 m of |wrap(heading_plan - heading_gt)| in degrees (NaN if none or not gt_ok)."""
    a, b = steps(plan[:, :M]), steps(gt[:, :M])
    va = (np.linalg.norm(a, axis=-1) > 0.1) & (np.linalg.norm(b, axis=-1) > 0.1) & np.isfinite(b).all(-1)
    d = np.degrees(np.abs(np.arctan2(np.sin(np.arctan2(a[..., 1], a[..., 0]) - np.arctan2(b[..., 1], b[..., 0])), np.cos(np.arctan2(a[..., 1], a[..., 0]) - np.arctan2(b[..., 1], b[..., 0])))))
    d = np.where(va, d, np.nan)
    with np.errstate(all="ignore"):
        r = np.nanmean(d, axis=1)
    return np.where(ok, r, np.nan)


def mean_speed(plan):
    return np.linalg.norm(steps(plan[:, :M]), axis=-1).sum(1) / (M * DT)


def boot_delta(delta, grp, B=1000, seed=0):
    """Episode-cluster bootstrap of the mean of a per-tick delta (NaN dropped)."""
    ok = np.isfinite(delta)
    ids = np.unique(grp[ok])
    sums = np.array([delta[ok & (grp == g)].sum() for g in ids])
    cnt = np.array([(ok & (grp == g)).sum() for g in ids], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    bm = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(sums.sum() / cnt.sum()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def q(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)), "max": float(x.max()), "n": int(x.size)}


def main():
    res = {"tier": "T1-kin plan-level proxy (per-tick counterfactual on the issued S plans; not a rollout)", "evidence_class": "MEASURED (ours)", "runs": {}}
    pooled = {k: {} for k in ("box", "kbox", "aonly", "abox", "kabox")}
    for tag in RUNS:
        z = {k: v for k, v in np.load(f"{OUT}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True).items()}
        ps, plan, v0, gt, P, ci = z["plan_s"], z["plan"], z["v0"], z["gt"], z["P"], z["ci"]
        ok = z["gt_ok"].astype(bool) & np.isfinite(gt[:, :M]).all((1, 2))
        V = variants(ps, v0)
        r = {"n_ticks": int(len(ps)), "n_gt_ok": int(ok.sum()), "box_reproduces_logged_plan_max_abs_m": float(np.abs(V["box"].astype(np.float32).astype(np.float64) - plan).max())}
        # (1) displacement of the issued plan vs plan_s (executed knots 1..M and all 30 knots)
        disp = {}
        for nm, pl in V.items():
            d = np.linalg.norm(pl - ps, axis=-1)
            disp[nm] = {"exec_knots_max_per_tick": q(d[:, :M].max(1)), "all_knots_max_per_tick": q(d.max(1)), "exec_knots_mean_per_tick": q(d[:, :M].mean(1)),
                        "ticks_exec_max_gt_1m": int((d[:, :M].max(1) > 1).sum()), "ticks_exec_max_gt_10m": int((d[:, :M].max(1) > 10).sum()), "ticks_all_max_gt_50m": int((d.max(1) > 50).sum())}
        r["displacement_vs_plan_s"] = disp
        # (2) attribution of the large-displacement ticks of the box
        db = np.linalg.norm(V["box"] - ps, axis=-1).max(1)
        big = db > 5.0
        da = {nm: np.linalg.norm(V[nm] - ps, axis=-1).max(1) for nm in V}
        r["attribution_ticks_box_allknot_max_gt_5m"] = {"n": int(big.sum()),
            "kappa_clamp_carries_ge_half (kbox >= 0.5 box)": int((big & (da["kbox"] >= 0.5 * db)).sum()),
            "accel_clamp_carries_ge_half (aonly >= 0.5 box)": int((big & (da["aonly"] >= 0.5 * db)).sum()),
            "both_ge_half": int((big & (da["kbox"] >= 0.5 * db) & (da["aonly"] >= 0.5 * db)).sum()),
            "neither_ge_half": int((big & (da["kbox"] < 0.5 * db) & (da["aonly"] < 0.5 * db)).sum()),
            "sum_allknot_max_displacement_share_of_box": {nm: float(da[nm][big].sum() / max(db[big].sum(), 1e-12)) for nm in V}}
        # (3) heading MAE on the executed steps vs GT (deg), delta vs plan_s, episode-cluster bootstrap
        h0 = heading_mae(ps, gt, ok)
        r["heading_mae_deg"] = {"plan_s": float(np.nanmean(h0)), "P10_prior_P": float(np.nanmean(heading_mae(P, gt, ok)))}
        for nm, pl in V.items():
            m, lo, hi = boot_delta(heading_mae(pl, gt, ok) - h0, ci)
            r["heading_mae_deg"][f"{nm}_minus_plan_s"] = {"delta": m, "lo": lo, "hi": hi}
            pooled[nm].setdefault("h", []).append(m)
        # (4) position error at the executed endpoint (knot M) and mean over executed knots, vs GT (m)
        def perr(pl):
            e = np.linalg.norm(pl[:, :M] - gt[:, :M], axis=-1)
            return np.where(ok, e[:, M - 1], np.nan), np.where(ok, e.mean(1), np.nan)
        e0 = perr(ps)
        r["exec_position_error_m"] = {"plan_s_endpoint": float(np.nanmean(e0[0])), "plan_s_mean_knots": float(np.nanmean(e0[1]))}
        for nm, pl in V.items():
            e1 = perr(pl)
            r["exec_position_error_m"][f"{nm}_minus_plan_s"] = {"endpoint": float(np.nanmean(e1[0] - e0[0])), "mean_knots": float(np.nanmean(e1[1] - e0[1]))}
        # (5) the crawl proxy: share of ticks whose mean executed-knot plan speed is < GT's mean speed - 2 m/s (all ticks; and the 3-8 m/s v0 band)
        vg = np.linalg.norm(steps(gt[:, :M]), axis=-1).sum(1) / (M * DT)
        band = (v0 >= 3.0) & (v0 < 8.0) & ok
        sh = {}
        for nm, pl in {"plan_s": ps, "P10_prior_P": P, **V}.items():
            c = mean_speed(pl) < (vg - 2.0)
            sh[nm] = {"all": float(c[ok].mean()), "v0_3_8": float(c[band].mean())}
        s_ex, p_ex = sh["plan_s"]["all"] - sh["P10_prior_P"]["all"], sh["plan_s"]["v0_3_8"] - sh["P10_prior_P"]["v0_3_8"]
        r["crawl_proxy_share"] = {"shares": sh, "n_band_ticks": int(band.sum()), "excess_plan_s_minus_P_all": s_ex, "excess_plan_s_minus_P_v0_3_8": p_ex,
                                  "fraction_of_excess_closed_v0_3_8": {nm: (sh["plan_s"]["v0_3_8"] - sh[nm]["v0_3_8"]) / p_ex if abs(p_ex) > 1e-9 else None for nm in V},
                                  "fraction_of_excess_closed_all": {nm: (sh["plan_s"]["all"] - sh[nm]["all"]) / s_ex if abs(s_ex) > 1e-9 else None for nm in V}}
        # (5b) the same, per executed STEP (a plan step speed < the GT step speed - 2 m/s), share over the (tick, step) pairs of gt_ok ticks
        def stepcrawl(pl):
            sp, sg = np.linalg.norm(steps(pl[:, :M]), axis=-1) / DT, np.linalg.norm(steps(gt[:, :M]), axis=-1) / DT
            return (sp < sg - 2.0)
        sc = {nm: stepcrawl(pl) for nm, pl in {"plan_s": ps, "P10_prior_P": P, **V}.items()}
        r["crawl_proxy_step_share"] = {nm: {"all": float(c[ok].mean()), "v0_3_8": float(c[band].mean())} for nm, c in sc.items()}
        se = r["crawl_proxy_step_share"]["plan_s"]["v0_3_8"] - r["crawl_proxy_step_share"]["P10_prior_P"]["v0_3_8"]
        r["crawl_proxy_step_excess_plan_s_minus_P_v0_3_8"] = se
        r["crawl_proxy_step_fraction_closed_v0_3_8"] = {nm: ((r["crawl_proxy_step_share"]["plan_s"]["v0_3_8"] - r["crawl_proxy_step_share"][nm]["v0_3_8"]) / se if abs(se) > 1e-9 else None) for nm in V}
        # (6) the entry clamp: how often plan_s's first-step speed is outside v0 +- a_max*0.99*dt
        s0 = np.linalg.norm(ps[:, 0], axis=-1) / DT
        lim = 4.0 * 0.99 * DT
        out = np.abs(s0 - v0) > lim
        r["entry_clamp"] = {"first_step_speed_minus_v0": q(s0 - v0), "ticks_outside_v0_pm_0.792": int(out.sum()), "share": float(out.mean()),
                            "in_band_3_8_share": float(out[(v0 >= 3) & (v0 < 8)].mean()), "above_share": float((s0 - v0 > lim).mean()), "below_share": float((s0 - v0 < -lim).mean())}
        res["runs"][tag] = r
        print(tag, "box-repro", "%.2e" % r["box_reproduces_logged_plan_max_abs_m"], "big", r["attribution_ticks_box_allknot_max_gt_5m"]["n"],
              {k: v for k, v in r["attribution_ticks_box_allknot_max_gt_5m"].items() if k != "n" and not isinstance(v, dict)},
              "| head d", {nm: round(r["heading_mae_deg"][f"{nm}_minus_plan_s"]["delta"], 3) for nm in V}, flush=True)
    os.makedirs(f"{PK}/raw", exist_ok=True)
    json.dump(res, open(f"{PK}/raw/PRECHECK_ATTRIBUTION.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", f"{PK}/raw/PRECHECK_ATTRIBUTION.json")


if __name__ == "__main__":
    main()
