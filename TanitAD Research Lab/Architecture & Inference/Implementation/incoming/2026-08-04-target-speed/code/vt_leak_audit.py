#!/usr/bin/env python3
"""D-VT1 step 2 — PROVE THE GUARD, by measuring what it actually removes.

The excision in `vtarget_guarded` is disjointness by index arithmetic and that is
pinned by a unit test. This script measures the part arithmetic cannot settle:
**a disjoint window is not an independent window.** A speed track is
autocorrelated, so `v(t+3 s)` still predicts `v(t+1 s)` and a guarded label still
carries information about the scored horizon.

THE DECOMPOSITION. For each scored quantity y (the manoeuvre head's own label
`dv = v(t+2 s) - v(t)`, and the along-track displacement over 0-2 s that dominates
ADE), fit a ridge under **leave-one-episode-out** cross-fitting and report
out-of-fold R^2 for four feature sets:

  PAST          strictly causal ego speed over [t-0.7 s, t]  -> the HONEST baseline
  PAST + GUARD  + the guarded label   (reads [t+2.1 s, t+20 s])
  PAST + BAND   + the guarded label QUANTISED to its 23 VTARGET bands
  PAST + ORACLE + the unguarded label (reads [t+0.1 s, t+20 s]) -> the leak

`R^2(PAST+X) - R^2(PAST)` is the *future information the label injects beyond what
the model already legally holds*. On ORACLE that number is a leak. On GUARD it is
the residual the excision could not remove — the quantity the escalation needed
and nobody had measured.

⛔ Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` over the 40 val
episodes, on paired per-window squared errors. `overlapping_holdout_se` is never
called.
⚠️ Every number is stratified by ego speed as well as pooled: the tactical defect
is strongly speed-dependent (38.2 % lossy at 1-3 m/s vs 1.8 % at 10-15 m/s), so a
pooled figure hides the regime the lever is for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval.ci import paired_episode_cluster_bootstrap        # noqa: E402
from tanitad.lake.vocab import VTARGET_TOKENS                     # noqa: E402

RIDGE = 1e-3
#: ego-speed strata, m/s. The 1-3 / 10-15 pair is the one the tactical lossy-rate
#: study used, kept identical so the two studies are readable side by side.
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0),
         (15.0, np.inf)]


def band_mid(tok: str) -> float:
    """Midpoint of a VTARGET band token — what a model conditioned on the BAND
    can recover. Quantisation is itself a partial guard and this measures it."""
    if tok == "v_stop":
        return 0.0
    inner = tok[tok.index("(") + 1:tok.index("]")]
    lo, hi = inner.split("-")
    return (float(lo) + float(hi)) / 2.0


def past_block(rows) -> np.ndarray:
    """Strictly causal features from v[t-0.7 s .. t]: the level, the backward
    differences, and the mean. Nothing here reads past t."""
    p = np.array([r["past"] for r in rows], dtype=np.float64)
    d = np.diff(p, axis=1)
    return np.column_stack([p[:, -1], d, p.mean(axis=1), p.std(axis=1)])


def _ridge_fit(x, y, lam=RIDGE):
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    a = z.T @ z + lam * np.eye(z.shape[1])
    a[-1, -1] -= lam                                  # never penalise the bias
    return np.linalg.solve(a, z.T @ y), mu, sd


def _ridge_pred(w, mu, sd, x):
    return np.column_stack([(x - mu) / sd, np.ones(len(x))]) @ w


def loeo_predict(x: np.ndarray, y: np.ndarray, eid: np.ndarray) -> np.ndarray:
    """Out-of-fold prediction, one fold per val EPISODE. Windows inside one clip
    are strongly dependent, so the episode is the only admissible fold unit."""
    out = np.empty_like(y)
    for e in np.unique(eid):
        te = eid == e
        w, mu, sd = _ridge_fit(x[~te], y[~te])
        out[te] = _ridge_pred(w, mu, sd, x[te])
    return out


def r2(y, yh) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yh) ** 2).sum() / ss) if ss > 0 else float("nan")


def audit(rows, tag: str) -> dict:
    eid = np.array([r["eid"] for r in rows])
    past = past_block(rows)
    feats = {
        "PAST": past,
        "PAST+GUARD": np.column_stack([past, [r["vt_guarded"] for r in rows]]),
        "PAST+BAND": np.column_stack(
            [past, [band_mid(r["band_guarded"]) for r in rows]]),
        "PAST+ORACLE": np.column_stack([past, [r["vt_oracle"] for r in rows]]),
    }
    res = {"_n_windows": len(rows), "_n_episodes": int(len(np.unique(eid))),
           "targets": {}}
    for tname, key in (("dv_2s", "dv_2s"), ("along_2s", "along_2s")):
        y = np.array([r[key] for r in rows], dtype=np.float64)
        preds = {k: loeo_predict(v, y, eid) for k, v in feats.items()}
        se = {k: (y - p) ** 2 for k, p in preds.items()}
        blk = {"r2": {k: round(r2(y, p), 4) for k, p in preds.items()},
               "rmse": {k: round(float(np.sqrt(v.mean())), 4)
                        for k, v in se.items()},
               "delta_r2_over_past": {
                   k: round(r2(y, preds[k]) - r2(y, preds["PAST"]), 4)
                   for k in preds if k != "PAST"},
               "paired_mse_reduction_vs_PAST": {}}
        for k in preds:
            if k == "PAST":
                continue
            ci = paired_episode_cluster_bootstrap(se["PAST"], se[k], eid,
                                                  n_boot=2000, seed=0)
            ci["_reads"] = ("delta > 0 => adding this label REDUCES squared "
                            "error, i.e. it injects information the causal past "
                            "does not already hold")
            blk["paired_mse_reduction_vs_PAST"][k] = ci
        # ⭐ the decision-relevant comparison: is the GUARDED label measurably
        # less informative than the UNGUARDED one? If this does not separate, the
        # excision removed nothing detectable and "guarded" is not a weaker input.
        head = paired_episode_cluster_bootstrap(se["PAST+ORACLE"],
                                                se["PAST+GUARD"], eid,
                                                n_boot=2000, seed=0)
        head["_reads"] = ("delta > 0 => the GUARDED label predicts the scored "
                          "quantity BETTER than the unguarded one, i.e. the "
                          "excision removed nothing. separated=False means the "
                          "two are statistically indistinguishable — which is "
                          "itself the finding: the guard is not a weaker input.")
        blk["paired_ORACLE_minus_GUARD"] = head
        res["targets"][tname] = blk
    res["_tag"] = tag
    return res


def strat_pooled(rows, v0) -> dict:
    """ONE LOEO fit over all windows, its out-of-fold predictions sliced by ego
    speed. The right way to ask *where* the label helps: a per-stratum refit at
    n<200 is dominated by its own estimation noise."""
    eid = np.array([r["eid"] for r in rows])
    past = past_block(rows)
    feats = {
        "PAST": past,
        "PAST+GUARD": np.column_stack([past, [r["vt_guarded"] for r in rows]]),
        "PAST+ORACLE": np.column_stack([past, [r["vt_oracle"] for r in rows]]),
    }
    res = {}
    for tname in ("dv_2s", "along_2s"):
        y = np.array([r[tname] for r in rows], dtype=np.float64)
        preds = {k: loeo_predict(v, y, eid) for k, v in feats.items()}
        for lo, hi in BANDS:
            m = (v0 >= lo) & (v0 < hi)
            name = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
            blk = res.setdefault(name, {"n": int(m.sum()),
                                        "n_episodes": int(len(set(eid[m])))})
            if m.sum() < 30 or len(set(eid[m])) < 5:
                blk["status"] = "UNPOWERED"
                blk["reason"] = "<30 windows or <5 episode clusters"
                continue
            se = {k: (y[m] - p[m]) ** 2 for k, p in preds.items()}
            blk[tname] = {
                "rmse": {k: round(float(np.sqrt(v.mean())), 4)
                         for k, v in se.items()},
                "mse_reduction_vs_PAST": {
                    k: paired_episode_cluster_bootstrap(
                        se["PAST"], se[k], eid[m], n_boot=2000, seed=0)
                    for k in se if k != "PAST"},
            }
    return res


def lead_interaction(rows) -> dict:
    """⭐ Does the FREE-FLOW gate actually deliver a free-flow aspiration?

    VTARGET's whole semantic claim is that it is the speed the ego WOULD hold if
    unobstructed. If the label is systematically depressed whenever a lead vehicle
    is present, it is not an aspiration — it is a CONSEQUENCE of the traffic, and
    conditioning a planner on it teaches the planner to reproduce the very
    following behaviour we wanted it to reason about.

    Joined against the canonical val40 lead block
    (`…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`, 881 windows,
    the SAME grid). ⚠️ `NO_LABEL` is kept as its own state, never folded into
    `NO_LEAD` — that collapse is the bias `lead_source` exists to avoid.
    """
    blk = (REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
           / "incoming" / "2026-08-04-distance-keeping-arms" / "raw"
           / "val40_lead_block.npz")
    if not blk.exists():
        return {"status": "UNAVAILABLE", "reason": f"missing {blk}", "n": 0}
    z = np.load(blk, allow_pickle=True)
    state_all = z["state"].astype(str)
    speeds_all = z["speeds"].astype(np.float64)
    gap_all = z["gap0_m"].astype(np.float64)
    blk_eid = z["eid"].astype(str)                # FILE STEMS, not episode ids
    if len(state_all) != 881 or len(rows) != 881:
        return {"status": "UNAVAILABLE",
                "reason": (f"lead block {len(state_all)} rows / labels "
                           f"{len(rows)} rows; both must be the canonical 881"),
                "n": 0}
    # ⛔ THE JOIN, and why it is positional. Both tables are built from the SAME
    # `lead_source.window_last_indices` grid over the same 40 episodes in the same
    # file order, so row i is the same window in both. That is asserted, not
    # assumed: the file stems must agree row-for-row AND the two independently
    # derived ego speeds must agree to a rounding tolerance. A misalignment would
    # show O(1) speed differences, not O(1e-3).
    mine_stem = np.array([Path(r["file"]).stem for r in rows])
    if not np.array_equal(mine_stem, blk_eid):
        return {"status": "UNAVAILABLE",
                "reason": "row-wise episode stems disagree; refusing to join",
                "n": 0}
    vt_all = np.array([r["vt_guarded"] for r in rows])
    v0_all = np.array([r["v0"] for r in rows])
    speed_gap = float(np.abs(v0_all - speeds_all).max())
    if speed_gap > 0.01:
        return {"status": "UNAVAILABLE",
                "reason": f"positional join rejected: max |v0 - block speed| = "
                          f"{speed_gap:.5f} m/s > 0.01", "n": 0}
    keep = np.array([bool(r["vt_guarded_valid"] and r["vt_oracle_valid"])
                     for r in rows])
    st = np.where(keep, state_all, "EXCLUDED_INVALID_LABEL")
    vt, v0 = vt_all, v0_all
    idx = np.arange(881)
    out = {"n": int(keep.sum()), "n_rows_joined": 881,
           "_join": ("POSITIONAL on the canonical 881 grid; verified by "
                     "row-wise episode stem equality and by two independently "
                     "derived ego speeds agreeing to "
                     f"{speed_gap:.5f} m/s"),
           "_subset": "windows with BOTH labels valid, as everywhere else here",
           "by_state": {}}
    for s in ("LEAD", "NO_LEAD", "NO_LABEL"):
        m = st == s
        if m.sum() < 20:
            out["by_state"][s] = {"status": "UNPOWERED", "n": int(m.sum())}
            continue
        out["by_state"][s] = {
            "n": int(m.sum()),
            "mean_v0_mps": round(float(v0[m].mean()), 4),
            "mean_vt_guarded_mps": round(float(vt[m].mean()), 4),
            "mean_vt_minus_v0_mps": round(float((vt[m] - v0[m]).mean()), 4),
            "frac_vt_above_v0": round(float((vt[m] > v0[m]).mean()), 4),
        }
    m = st == "LEAD"
    if m.sum() >= 20:
        g = gap_all[idx[m]]
        fin = np.isfinite(g)
        out["lead_gap_corr"] = {
            "n": int(fin.sum()),
            "corr_gap_vs_vt_minus_v0": round(float(np.corrcoef(
                g[fin], (vt[m] - v0[m])[fin])[0, 1]), 4) if fin.sum() > 2 else None,
            "_reads": ("a POSITIVE correlation means the label rises as the road "
                       "ahead clears — i.e. the 'free-flow' gate did NOT remove "
                       "the lead's influence and the label is partly a following "
                       "behaviour, not an aspiration"),
        }
    return out


def main(labels_json: Path, out_json: Path):
    d = json.load(open(labels_json, encoding="utf-8"))
    rows = d["rows"]
    both = [r for r in rows if r["vt_guarded_valid"] and r["vt_oracle_valid"]]
    g_not_o = [r for r in rows if r["vt_guarded_valid"]
               and not r["vt_oracle_valid"]]

    out = {
        "_what": "leak audit — what the horizon guard actually removes",
        "_estimator": ("paired_episode_cluster_bootstrap (taniteval.ci), unit = "
                       "val episode, B=2000. NEVER overlapping_holdout_se"),
        "_guard_steps": d["_guard_steps"],
        "_guard_derivation": d["_guard_derivation"],
        "_canonical_881": d["canonical_881"], "_sha_ok_all": d["sha_ok_all"],
        "coverage": {
            "n_windows": d["n_windows"],
            "n_valid_oracle": sum(1 for r in rows if r["vt_oracle_valid"]),
            "n_valid_guarded": sum(1 for r in rows if r["vt_guarded_valid"]),
            "n_valid_both": len(both),
            "n_guarded_valid_but_not_oracle": len(g_not_o),
            "guard_cost_windows": sum(1 for r in rows if r["vt_oracle_valid"])
            - len(both),
            "_reads": ("guarded-valid is a SUBSET of oracle-valid by "
                       "construction (the guard only shortens the read window); "
                       "n_guarded_valid_but_not_oracle must be 0"),
        },
        "pooled_valid_both": audit(both, "valid_both"),
        "by_speed": {},
        "label_geometry": {},
    }

    v0 = np.array([r["v0"] for r in both])
    out["_by_speed_note"] = (
        "`by_speed` is the PRIMARY stratified view: ONE leave-one-episode-out fit "
        "over all valid windows, then its out-of-fold predictions SLICED by ego "
        "speed. `by_speed_refit` re-fits inside each stratum and is reported only "
        "as a secondary — at n=39..195 per band those fits are underpowered and "
        "their negative delta-R^2 are a fitting artefact, not a finding.")
    out["by_speed"] = strat_pooled(both, v0)
    out["by_speed_refit"] = {}
    for lo, hi in BANDS:
        sel = [r for r, k in zip(both, (v0 >= lo) & (v0 < hi)) if k]
        name = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
        if len(sel) < 30 or len({r["eid"] for r in sel}) < 5:
            out["by_speed_refit"][name] = {
                "status": "UNPOWERED", "n": len(sel),
                "n_episodes": len({r["eid"] for r in sel}),
                "reason": ("fewer than 30 windows or fewer than 5 episodes — an "
                           "episode-cluster bootstrap over <5 clusters is not an "
                           "interval, and reporting one would be the error this "
                           "programme keeps retracting")}
            continue
        out["by_speed_refit"][name] = audit(sel, name)
    out["lead_interaction"] = lead_interaction(rows)

    # how far apart the two labels actually are, and how often banding hides it
    g = np.array([r["vt_guarded"] for r in both])
    o = np.array([r["vt_oracle"] for r in both])
    bg = np.array([r["band_guarded"] for r in both])
    bo = np.array([r["band_oracle"] for r in both])
    out["label_geometry"] = {
        "n": len(both),
        "mean_abs_diff_mps": round(float(np.abs(g - o).mean()), 4),
        "median_abs_diff_mps": round(float(np.median(np.abs(g - o))), 4),
        "p90_abs_diff_mps": round(float(np.percentile(np.abs(g - o), 90)), 4),
        "guarded_minus_oracle_mean_mps": round(float((g - o).mean()), 4),
        "same_band_rate": round(float((bg == bo).mean()), 4),
        "corr_guarded_oracle": round(float(np.corrcoef(g, o)[0, 1]), 4),
        "corr_guarded_v0": round(float(np.corrcoef(g, v0)[0, 1]), 4),
        "corr_oracle_v0": round(float(np.corrcoef(o, v0)[0, 1]), 4),
        "n_bands": len(VTARGET_TOKENS),
        "_reads": ("same_band_rate is how often the 23-band quantisation ERASES "
                   "the guard — where it is high the guarded and unguarded "
                   "conditioning inputs are the same token and the excision "
                   "buys nothing at the model's interface"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    p = out["pooled_valid_both"]["targets"]
    print(f"n_valid_both={len(both)} of {d['n_windows']}")
    for t in ("dv_2s", "along_2s"):
        print(f"  {t}: R2 " + " ".join(
            f"{k}={v}" for k, v in p[t]["r2"].items()))
        print(f"        dR2 " + " ".join(
            f"{k}={v}" for k, v in p[t]["delta_r2_over_past"].items()))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
