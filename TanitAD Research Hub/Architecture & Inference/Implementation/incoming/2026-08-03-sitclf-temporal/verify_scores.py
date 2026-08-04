"""Re-derive the headline numbers from the BANKED SCORES, independently of the results JSON.

WHY. `run_temporal.py` writes both the per-row scores and the summary JSON in one process, so the
JSON agreeing with itself proves nothing. This script loads only `results_temporal.scores.npz` --
the raw per-row output -- and recomputes AP, AP-lift, the 5 % operating point and the paired
contrast against the reference from scratch. If any headline in the report is a transcription
error, a stale key or a mis-indexed situation column, the two will disagree here.

It also checks the two things a summary cannot show:
  * that the arms really were scored on IDENTICAL rows (the paired estimator's precondition);
  * that no arm's score column is constant, which would make its AP an artefact of tie order.

usage:
  python verify_scores.py --scores results_temporal.scores.npz --results results_temporal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (ap_lift, average_precision,           # noqa: E402
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf_deploy import precision_recall_at_budget     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results_temporal.scores.npz")
    ap.add_argument("--results", default="results_temporal.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=5e-5)
    a = ap.parse_args()

    z = np.load(a.scores)
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))
    ref = R["protocol"]["reference_arm"]
    sits = [str(s) for s in z["situations"]]
    y = z["y"].astype(np.int64)
    V = z["valid"].astype(bool)
    cc = z["clip_cluster"]
    arms = [k for k in z.files
            if k not in ("clip_cluster", "y", "valid", "ego", "cache_tag",
                         "folds", "situations", "t")]
    real = sorted(k for k in arms if not k.startswith("NEG_FEAT__"))
    print(f"bank: {y.shape[0]:,} rows, {len(arms)} score columns "
          f"({len(real)} real + {len(arms)-len(real)} nulls), {len(np.unique(cc))} clusters")

    bad = []
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = y[m, i]
        eid = cc[m]
        jr = R["per_situation"][s]
        # --- identical-rows precondition -----------------------------------
        assert int(m.sum()) == jr["n_scorable"], (s, int(m.sum()), jr["n_scorable"])
        assert int(yv.sum()) == jr["n_pos"], (s, int(yv.sum()), jr["n_pos"])
        print(f"\n{s}: {m.sum():,} rows, {yv.sum():,} pos, base {yv.mean():.5f}, "
              f"{len(np.unique(eid))} clusters "
              f"({len(np.unique(eid[yv > 0]))} with a positive)")
        rs = z[ref][m, i].astype(np.float64)
        for n in real:
            sc = z[n][m, i].astype(np.float64)
            got_ap = average_precision(yv, sc)
            got_lift = ap_lift(yv, sc)
            want = jr["arms"][n]
            d_ap = abs(got_ap - want["ap"])
            d_lift = abs(got_lift - want["point"])
            op = precision_recall_at_budget(yv, sc, np.ones(yv.size, bool))
            d_prec = abs(op["precision"] - want["op_5pct"]["precision"])
            flat = float(np.nanstd(sc)) < 1e-12
            tag = []
            if d_ap > a.tol or d_lift > 1e-3:
                tag.append("AP-MISMATCH")
            if d_prec > a.tol:
                tag.append("PRECISION-MISMATCH")
            if flat:
                tag.append("CONSTANT-SCORE")
            if op["n_alarm"] != want["op_5pct"]["n_alarm"]:
                tag.append("ALARM-COUNT-MISMATCH")
            if tag:
                bad.append((s, n, tag))
            print(f"  {n:>26}  AP {got_ap:.5f} (json {want['ap']:.5f}, d={d_ap:.2e})  "
                  f"lift {got_lift:.4f}  P@5% {op['precision']:.4f} "
                  f"fires {op['n_alarm']}/{op['n_pos']}  {'  '.join(tag) or 'OK'}")
        # --- one paired contrast recomputed from scratch --------------------
        peak = jr["peak_arm"]
        if peak != ref:
            d = paired_ap_episode_cluster_bootstrap(
                yv, z[peak][m, i].astype(np.float64), rs, eid, n_boot=a.n_boot, lift=True)
            w = jr["arms"][peak]["paired_vs_reference"]
            ok = (abs(d["delta"] - w["delta"]) < 1e-4 and abs(d["lo"] - w["lo"]) < 1e-3
                  and abs(d["hi"] - w["hi"]) < 1e-3 and d["separated"] == w["separated"])
            print(f"  PAIRED {peak} - {ref}: recomputed {d['delta']:+.5f} "
                  f"[{d['lo']:+.5f},{d['hi']:+.5f}] sep={d['separated']} | "
                  f"json {w['delta']:+.5f} [{w['lo']:+.5f},{w['hi']:+.5f}] "
                  f"sep={w['separated']} -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append((s, f"PAIRED {peak}", ["PAIRED-MISMATCH"]))

    print("\n" + ("VERIFIED: every recomputed number matches the banked JSON"
                  if not bad else f"MISMATCHES: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
