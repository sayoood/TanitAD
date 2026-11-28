#!/usr/bin/env python3
"""`H-RL-PROGRESS-LEADCAP-1` -- the A/B that scores the pre-registered prediction.

PRE-REGISTRATION (committed BEFORE `rewards.py` changed, commit 082606e02):
    `TanitAD Research Lab/Architecture & Inference/Research/
     2026-09-06-refcv4b-rl-repair/PREREG.md`

WHAT IT DOES. On the SAME RL-fit windows the banked `G-REWARD` panel used, it
rebuilds the three reference paths -- the HUMAN's logged 2 s future, the trivial
`hold-v0` constant-velocity path and the `frozen` do-nothing path -- and scores
every reward component THREE TIMES on identical geometry:

    off         the exact pre-repair `progress` (bit-identical, the CHANNEL control)
    achievable  the PRE-REGISTERED cap  min(ref_free, lead_x[-1] - standoff)
    lead        the SENSITIVITY variant, capping at the lead bound ONLY

⛔ ONE VARIABLE. The corpus, the window list, the lead model, the weights, the
three paths and every other component are byte-identical across the three
columns; the ONLY thing that moves is `ctx["progress_lead_cap"]`.

⛔ IT SELECTS NOTHING AND IT DOES NOT RE-SCORE `G-REWARD`. The Master Mind's
ruling stands: `G-REWARD` FAILED on its committed statistic (the RATE). The rate
appears here as a DIAGNOSTIC of the reward's geometry, never as a gate verdict.

THE PRE-REGISTERED STATISTICS, fixed before any number was computed:
  rate(r)      P(hold_v0 composed >= human composed)          -- G-REWARD's statistic
  mean(r)      mean(hold_v0 - human)
  DIV(r)       1 iff rate_ci_lo > 0.30 AND mean_ci_hi < 0     -- baseline 6 of 8 rungs
  SKEWSPLIT(r) median(gap) - mean(gap)                        -- baseline +0.008836 all
  P1-PRED-A    the `progress` per-term gap is NOT POSITIVE at <=3.0/<=2.0/<=1.5 s
  P1-PRED-B    sum DIV falls below 6, SKEWSPLIT falls at >=2 of the 3 conflict
               rungs, and rate(<=2.0 s) falls from 0.5482

CONTROLS THAT MUST READ KNOWN VALUES (CLAUDE.md 2026-08-22):
  CHANNEL      cap OFF must reproduce the banked all-window rate 0.441780259484316
               and mean gap -0.011239379601720929 to <= 1e-12
  ARITHMETIC   hold-v0's uncapped progress must be reproducible from v0 alone
  NON-BINDING  windows where the cap does not bind must be EXACTLY unchanged
  DEGENERACY   the `progress` spread must NOT collapse (baseline 0.088491 all-window,
               0.060303 at <= 2.0 s) -- a collapse is a DEGENERATE REPAIR, not a pass
  frozen       must stay far below the human at every rung (baseline 0.045492)

Estimator: episode-cluster bootstrap, n_boot 4000. It answers ONE question --
"would another draw of EPISODES say this?" -- not another training run
(`H-ESTIM-SEED-1`) and not another inference run. No arm is trained here and no
planner samples, so neither of those variances enters.

Tier: T0 instrument probe, NON-PARITY RL-fit windows. Evidence class: MEASURED
(ours). 0 GPU (CPU only).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("rl_refcv3_min_for_ab", os.path.join(HERE, "rl_refcv3_min.py"))
np = M.np
torch = M.torch

#: The banked values the CHANNEL control must reproduce.
BANKED_ALL_RATE = 0.441780259484316
BANKED_ALL_MEAN = -0.011239379601720929
BANKED_FROZEN_RATE = 0.04549187058630317
#: The banked baselines P1-PRED-B is scored against (PREREG.md 3.1).
BANKED_DIV_COUNT = 6
BANKED_RATE_AT_2S = 0.5482
BANKED_SKEWSPLIT = {"all": 0.008836, "<=3.0s": 0.008920,
                    "<=2.5s": 0.010966, "<=2.0s": 0.013674}
BANKED_SPREAD = {"all": 0.088491, "<=2.0s": 0.060303}

#: ⛔ FIXED IN ADVANCE, identical to `greward_power_curve.py`'s ladder.
LADDER = (float("inf"), 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0)
#: the repaired weighting whose rate/mean disagreement is under study
WEIGHTS = {"progress": 0.3, "collision": 1.0, "headway": 0.3}
#: (progress_lead_cap, headway_reduce) per column. ⛔ ONE VARIABLE PER COMPARISON:
#: `achievable`/`lead` move `progress` only against `off`; `hq25` moves `headway`
#: only against `off` -- that is `H-RL-HEADWAY-QUANTILE-1`'s pre-registered test.
#: `both` moves two and is reported for completeness, NEVER as the test.
MODES = ("off", "achievable", "lead", "hq25", "both")
CAP_CTX = {"off": (False, "min"), "achievable": ("achievable", "min"),
           "lead": ("lead", "min"), "hq25": (False, "q0.25"),
           "both": ("lead", "q0.25")}
#: the three rungs P1-PRED-A and P1-PRED-B(B2) are scored on
CONFLICT_RUNGS = ("<=3.0s", "<=2.5s", "<=2.0s")


def cluster_ci(vals, eids, n_boot=4000, seed=0, alpha=0.05, stat="mean"):
    """Episode-cluster bootstrap. ⚠️ Answers only: would another draw of EPISODES
    say this? -- not another training run, not another inference run."""
    vals = np.asarray(vals, dtype=np.float64)
    eids = np.asarray(eids)
    uniq = np.unique(eids)
    if len(uniq) == 0 or vals.size == 0:
        return None, None
    idx = {e: np.flatnonzero(eids == e) for e in uniq}
    rng = np.random.default_rng(seed)
    f = np.mean if stat == "mean" else np.median
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        draws[b] = f(vals[sel])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def build_rows(a):
    """Score every lead window under all three cap modes on IDENTICAL geometry."""
    M.LEAD_MODE = "track"
    _model, cfg, _targs, prov = M.load(a, "cpu")
    corp = M.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = M.load_lead_block(a.lead_block)
    if lead is None or lead.ts_rel is None:
        raise SystemExit("[leadcap-ab] a lead block with ts_rel_s is required")
    spec = M.RewardSpec(weights=dict(M.DEFAULT_WEIGHTS), dt=M.DT_REWARD_S)
    rows = []
    for wi in M.scoreable_windows(corp, lead):
        has, xy, _ln = M.lead_row(corp, lead, wi)
        if not has:
            continue
        e_i, t = corp.ds.index[wi]
        row = lead.idx.get((corp.clip_ids[e_i], int(t + corp.W - 1 + corp.raw_off)))
        human, v0 = M._human_future(corp, wi)
        xs = torch.tensor([v0 * s for s in M.GRID_S], dtype=torch.float32)
        hold = torch.stack([xs, torch.zeros_like(xs)], dim=-1).reshape(1, 1, 1, 5, 2)
        frozen = torch.zeros(1, 1, 1, 5, 2)
        batch = {"v0": torch.tensor([v0]),
                 "lead_xy": torch.tensor([xy], dtype=torch.float32),
                 "lead_track": M.lead_track(corp, lead, wi)[None]}
        base_ctx = M.reward_ctx(batch, S5=5)
        rec = {"wi": int(wi), "clip": corp.clip_ids[e_i], "eid": int(e_i), "v0": v0,
               "gt_time_gap_min_s": (float(lead.gt_time_gap[row])
                                     if lead.gt_time_gap is not None else float("nan"))}
        # the scene-level facts the cap is built from, banked so the panel is
        # auditable without re-opening the corpus
        rf = max(v0 * (len(M.GRID_S) - 1) * M.DT_REWARD_S, 5.0)
        rec["ref_free_m"] = float(rf)
        rec["lead_end_x_m"] = float(base_ctx["lead_path"].reshape(-1, 2)[-1, 0])
        rec["standoff_m"] = float(4.5 + 2.0 * v0)
        rec["ref_lead_m"] = rec["lead_end_x_m"] - rec["standoff_m"]
        for mode in MODES:
            cap, hred = CAP_CTX[mode]
            ctx = {**base_ctx, "progress_lead_cap": cap, "headway_reduce": hred}
            for name, traj in (("human", human), ("hold_v0", hold),
                               ("frozen", frozen)):
                parts = spec.per_component(traj, ctx)
                r = {k: float(v.reshape(-1)[0]) for k, v in parts.items()}
                r["composed"] = float(spec(traj, ctx).reshape(-1)[0])
                rec[f"{name}__{mode}"] = r
        rows.append(rec)
    return rows


def composed(side, weights=WEIGHTS):
    return sum(w * float(side.get(k, 0.0)) for k, w in weights.items())


def ladder_for(rows, mode, n_boot, seed):
    eid = np.array([r["eid"] for r in rows])
    tg = np.array([float(r.get("gt_time_gap_min_s", np.nan)) for r in rows])
    h = np.array([composed(r[f"human__{mode}"]) for r in rows])
    v = np.array([composed(r[f"hold_v0__{mode}"]) for r in rows])
    f = np.array([composed(r[f"frozen__{mode}"]) for r in rows])
    gap = v - h
    ge = (v >= h - 1e-12).astype(np.float64)
    fge = (f >= h - 1e-12).astype(np.float64)
    sterm = "headway" if mode in ("hq25", "both") else "progress"
    spread = np.array([abs(float(r[f"hold_v0__{mode}"][sterm])
                           - float(r[f"human__{mode}"][sterm])) for r in rows])
    per_term = {k: np.array([w * (float(r[f"hold_v0__{mode}"][k])
                                  - float(r[f"human__{mode}"][k])) for r in rows])
                for k, w in WEIGHTS.items()}
    out = []
    for thr in LADDER:
        m = (np.isfinite(tg) & (tg <= thr)) if np.isfinite(thr) else np.ones(len(rows), bool)
        n = int(m.sum())
        label = "all" if not np.isfinite(thr) else "<=%.1fs" % thr
        if n == 0:
            out.append({"label": label, "threshold_s": None if not np.isfinite(thr) else thr,
                        "n_windows": 0, "n_episodes": 0, "rate": None})
            continue
        rlo, rhi = cluster_ci(ge[m], eid[m], n_boot, seed)
        mlo, mhi = cluster_ci(gap[m], eid[m], n_boot, seed)
        med = float(np.median(gap[m]))
        mean = float(gap[m].mean())
        rec = {"label": label,
               "threshold_s": None if not np.isfinite(thr) else thr,
               "n_windows": n, "n_episodes": int(len(np.unique(eid[m]))),
               "RATE_hold_ge_human": float(ge[m].mean()), "RATE_ci": [rlo, rhi],
               "MEAN_gap_hold_minus_human": mean, "MEAN_ci": [mlo, mhi],
               "MEDIAN_gap": med, "SKEWSPLIT_median_minus_mean": med - mean,
               "DIV": bool(rlo is not None and rlo > 0.30 and mhi is not None and mhi < 0),
               "frozen_ge_human": float(fge[m].mean()),
               "progress_spread_mean": float(spread[m].mean()),
               "progress_spread_zero_frac": float((spread[m] <= 1e-12).mean()),
               "per_term_weighted_mean_gap": {},
               "per_term_ci": {}}
        for k, d in per_term.items():
            rec["per_term_weighted_mean_gap"][k] = float(d[m].mean())
            lo, hi = cluster_ci(d[m], eid[m], n_boot, seed)
            rec["per_term_ci"][k] = [lo, hi]
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=M.EXPECT_BASE_STEP)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = build_rows(a)
    if not rows:
        raise SystemExit("[leadcap-ab] no lead windows scored")
    eid = np.array([r["eid"] for r in rows])
    print("[leadcap-ab] %d windows / %d episodes" % (len(rows), len(np.unique(eid))))

    ladders = {m: ladder_for(rows, m, a.n_boot, a.seed) for m in MODES}

    # ---------------- CONTROLS -------------------------------------------------
    off = ladders["off"][0]
    h_off = np.array([composed(r["human__off"]) for r in rows])
    v_off = np.array([composed(r["hold_v0__off"]) for r in rows])
    f_off = np.array([composed(r["frozen__off"]) for r in rows])
    v0 = np.array([float(r["v0"]) for r in rows])
    ph = np.array([float(r["hold_v0__off"]["progress"]) for r in rows])
    pred = (v0 * 2.0) / np.maximum(v0 * 2.0, 5.0)
    nonbind = np.array([r["ref_lead_m"] >= r["ref_free_m"] for r in rows])
    err_nb = {}
    for mode in [m for m in MODES if m != "off"]:
        key = "headway" if mode in ("hq25", "both") else "progress"
        e = np.array([max(abs(float(r[f"{s}__{mode}"][key])
                              - float(r[f"{s}__off"][key]))
                          for s in ("human", "hold_v0", "frozen")) for r in rows])
        err_nb[mode] = {
            "n_nonbinding_windows": int(nonbind.sum()),
            "max_abs_progress_delta_on_nonbinding": (float(e[nonbind].max())
                                                     if nonbind.any() else None),
            "n_nonbinding_exactly_unchanged": int((e[nonbind] <= 1e-12).sum()),
            "max_abs_progress_delta_all": float(e.max())}
    controls = {
        "CHANNEL_all_window_rate": float((v_off >= h_off - 1e-12).mean()),
        "CHANNEL_banked_all_window_rate": BANKED_ALL_RATE,
        "CHANNEL_rate_abs_err": abs(float((v_off >= h_off - 1e-12).mean()) - BANKED_ALL_RATE),
        "CHANNEL_all_window_mean_gap": float((v_off - h_off).mean()),
        "CHANNEL_banked_all_window_mean_gap": BANKED_ALL_MEAN,
        "CHANNEL_mean_abs_err": abs(float((v_off - h_off).mean()) - BANKED_ALL_MEAN),
        "ARITHMETIC_hold_progress_from_v0_max_abs_err": float(np.abs(ph - pred).max()),
        "FROZEN_all_window_rate": float((f_off >= h_off - 1e-12).mean()),
        "FROZEN_banked": BANKED_FROZEN_RATE,
        "NONBINDING": err_nb,
        "HEADWAY_QUANTILE_BINDS_FRAC": {
            m: float(np.mean([abs(float(r[f"human__{m}"]["headway"])
                                  - float(r["human__off"]["headway"])) > 1e-12
                              for r in rows]))
            for m in ("hq25", "both")}}

    # ---------------- THE PRE-REGISTERED VERDICT -------------------------------
    def by_label(lad):
        return {r["label"]: r for r in lad if r.get("n_windows")}

    verdict = {}
    for mode in [m for m in MODES if m != "off"]:
        L, B = by_label(ladders[mode]), by_label(ladders["off"])
        pred_a = {}
        for lab in ("<=3.0s", "<=2.0s", "<=1.5s"):
            if lab not in L:
                continue
            g = L[lab]["per_term_weighted_mean_gap"]["progress"]
            ci = L[lab]["per_term_ci"]["progress"]
            pred_a[lab] = {"gap": g, "ci": ci,
                           "not_positive": bool(g <= 0.0 or (ci[0] is not None and ci[0] <= 0.0)),
                           "baseline_gap": B[lab]["per_term_weighted_mean_gap"]["progress"]}
        div_now = sum(1 for r in ladders[mode] if r.get("n_windows") and r["DIV"])
        skew_fell = [lab for lab in CONFLICT_RUNGS
                     if lab in L and abs(L[lab]["SKEWSPLIT_median_minus_mean"])
                     < abs(B[lab]["SKEWSPLIT_median_minus_mean"])]
        rate2 = L.get("<=2.0s", {}).get("RATE_hold_ge_human")
        spread_all = L["all"]["progress_spread_mean"]
        verdict[mode] = {
            "P1_PRED_A_progress_not_positive_at_conflict_rungs": pred_a,
            "P1_PRED_A_HELD": all(x["not_positive"] for x in pred_a.values()),
            "P1_PRED_B1_div_count": div_now,
            "P1_PRED_B1_baseline_div_count": BANKED_DIV_COUNT,
            "P1_PRED_B1_HELD": bool(div_now < BANKED_DIV_COUNT),
            "P1_PRED_B2_skewsplit_fell_at": skew_fell,
            "P1_PRED_B2_HELD": bool(len(skew_fell) >= 2),
            "P1_PRED_B_rate_at_2s": rate2,
            "P1_PRED_B_rate_at_2s_baseline": B["<=2.0s"]["RATE_hold_ge_human"],
            "P1_PRED_B_rate_fell": bool(rate2 is not None
                                        and rate2 < B["<=2.0s"]["RATE_hold_ge_human"]),
            "DEGENERACY_progress_spread_all": spread_all,
            "DEGENERACY_baseline_spread_all": B["all"]["progress_spread_mean"],
            "DEGENERACY_spread_collapsed": bool(spread_all < 0.1 * B["all"]["progress_spread_mean"]),
        }
        verdict[mode]["P1_PRED_B_HELD"] = bool(
            verdict[mode]["P1_PRED_B1_HELD"] and verdict[mode]["P1_PRED_B2_HELD"])

    out = {
        "_what": "H-RL-PROGRESS-LEADCAP-1 A/B: the pre-registered `progress` repair",
        "_prereg": ("TanitAD Research Lab/Architecture & Inference/Research/"
                    "2026-09-06-refcv4b-rl-repair/PREREG.md (commit 082606e02, "
                    "committed BEFORE rewards.py changed)"),
        "_evidence_class": "MEASURED (ours)",
        "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
        "_estimator": ("episode-cluster bootstrap, n_boot %d -- answers only "
                       "'would another draw of EPISODES say this?'" % a.n_boot),
        "_not_a_gate": ("G-REWARD remains FAILED on its committed statistic. The "
                        "rate appears here as a diagnostic of the reward's "
                        "geometry, never as a gate verdict."),
        "weights": WEIGHTS, "modes": list(MODES),
        "n_windows": len(rows), "n_episodes": int(len(np.unique(eid))),
        "controls": controls, "ladders": ladders, "verdict": verdict}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    if a.rows_out:
        with open(a.rows_out, "w", encoding="utf-8") as fh:
            json.dump({"_what": "per-window rows, three cap modes", "per_window": rows},
                      fh, indent=1)

    # ---------------- the printed panel ---------------------------------------
    print("")
    print("== CONTROLS (must read known values)")
    print("   CHANNEL rate  %.15f  banked %.15f  err %.3e"
          % (controls["CHANNEL_all_window_rate"], BANKED_ALL_RATE,
             controls["CHANNEL_rate_abs_err"]))
    print("   CHANNEL mean  %.15f  banked %.15f  err %.3e"
          % (controls["CHANNEL_all_window_mean_gap"], BANKED_ALL_MEAN,
             controls["CHANNEL_mean_abs_err"]))
    print("   ARITHMETIC hold-progress from v0 alone: max abs err %.3e"
          % controls["ARITHMETIC_hold_progress_from_v0_max_abs_err"])
    print("   frozen >= human  %.6f  (banked %.6f)"
          % (controls["FROZEN_all_window_rate"], BANKED_FROZEN_RATE))
    for mode in ("achievable", "lead"):
        d = controls["NONBINDING"][mode]
        print("   NON-BINDING[%s] n=%d  exactly unchanged %d  max|dprogress| %s"
              % (mode, d["n_nonbinding_windows"], d["n_nonbinding_exactly_unchanged"],
                 ("%.3e" % d["max_abs_progress_delta_on_nonbinding"])
                 if d["max_abs_progress_delta_on_nonbinding"] is not None else "n/a"))
    for mode in MODES:
        print("")
        print("== LADDER  cap=%s" % mode)
        print("%-8s %6s %5s %8s %-20s %10s %10s %10s %4s %9s %9s %9s"
              % ("rung", "nwin", "neps", "RATE", "RATE_ci", "MEAN", "MEDIAN",
                 "SKEWSPL", "DIV", "prog", "head", "spread"))
        for r in ladders[mode]:
            if not r.get("n_windows"):
                print("%-8s %6d  (empty)" % (r["label"], 0))
                continue
            pt = r["per_term_weighted_mean_gap"]
            print("%-8s %6d %5d %8.4f [%7.4f,%7.4f] %10.6f %10.6f %10.6f %4d %9.6f %9.6f %9.6f"
                  % (r["label"], r["n_windows"], r["n_episodes"], r["RATE_hold_ge_human"],
                     r["RATE_ci"][0], r["RATE_ci"][1], r["MEAN_gap_hold_minus_human"],
                     r["MEDIAN_gap"], r["SKEWSPLIT_median_minus_mean"], int(r["DIV"]),
                     pt["progress"], pt["headway"], r["progress_spread_mean"]))
    for mode in [m for m in MODES if m != "off"]:
        vd = verdict[mode]
        print("")
        print("== PRE-REGISTERED VERDICT  cap=%s" % mode)
        print("   P1-PRED-A (progress not positive at <=3.0/<=2.0/<=1.5 s): %s"
              % ("HELD" if vd["P1_PRED_A_HELD"] else "FAILED"))
        for lab, x in vd["P1_PRED_A_progress_not_positive_at_conflict_rungs"].items():
            print("      %-8s gap %+.6f  ci [%+.6f, %+.6f]  baseline %+.6f  %s"
                  % (lab, x["gap"], x["ci"][0], x["ci"][1], x["baseline_gap"],
                     "ok" if x["not_positive"] else "POSITIVE"))
        print("   P1-PRED-B1 sum DIV %d < baseline %d : %s"
              % (vd["P1_PRED_B1_div_count"], BANKED_DIV_COUNT,
                 "HELD" if vd["P1_PRED_B1_HELD"] else "FAILED"))
        print("   P1-PRED-B2 SKEWSPLIT fell at %s (need >=2 of 3): %s"
              % (vd["P1_PRED_B2_skewsplit_fell_at"],
                 "HELD" if vd["P1_PRED_B2_HELD"] else "FAILED"))
        print("   rate(<=2.0 s) %.4f vs baseline %.4f : %s"
              % (vd["P1_PRED_B_rate_at_2s"], vd["P1_PRED_B_rate_at_2s_baseline"],
                 "fell" if vd["P1_PRED_B_rate_fell"] else "DID NOT FALL"))
        print("   DEGENERACY progress spread %.6f vs %.6f : %s"
              % (vd["DEGENERACY_progress_spread_all"], vd["DEGENERACY_baseline_spread_all"],
                 "COLLAPSED" if vd["DEGENERACY_spread_collapsed"] else "still ranks"))
        print("   => P1-PRED-B %s" % ("HELD" if vd["P1_PRED_B_HELD"] else "FAILED"))
    print("")
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
