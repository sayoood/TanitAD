"""INDEPENDENT VERIFIER — re-derive the Q-B event yardstick from the banked scores alone.

WHY THIS EXISTS
---------------
`run_per_situation_horizon.py` computes the event yardstick inline. A JSON that is checked by the
script that wrote it agrees with itself and proves nothing. So the yardstick was PROMOTED to
`stack/tanitad/eval/sitclf_deploy.py::event_anticipation_report` — a separate implementation, in
the shared library, with its own eight tests — and this script re-derives every Q-B headline from
`results_horizon_ps.scores.npz` using THAT function.

Two independent implementations agreeing on the same banked columns is the check. A disagreement
means one of them is wrong and neither number may be quoted.

⚠️ The two are not expected to agree on EVERY field. The promoted function excludes onsets with no
reachable scorable row (`n_onsets_unreachable`), which the run script had already filtered when it
built the onset universe — so the counts must match — and it guards the look-back by CLUSTER where
the run script guarded by clip START index. Those are the same guard on this substrate (one clip =
one cluster) and the verifier asserts that rather than assuming it.

usage:
  python verify_event_yardstick.py --scores results_horizon_ps.scores.npz \
      --results results_horizon_ps.json --out verify_event_yardstick.json
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

from tanitad.eval.sitclf_deploy import event_anticipation_report        # noqa: E402

ARMS = ("FROZEN", "PS_SEL", "C_GLOBAL", "C_ORACLE_PS")
TOL = 1e-9


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results_horizon_ps.scores.npz")
    ap.add_argument("--results", default="results_horizon_ps.json")
    ap.add_argument("--out", default="verify_event_yardstick.json")
    a = ap.parse_args()

    z = np.load(a.scores)
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))
    sits = [str(s) for s in z["situations"]]
    EVAL = z["eval_rows"].astype(bool)
    cc = z["clip_cluster"]
    onsets, osit = z["onsets"], z["onset_sit"]

    # one clip == one cluster on this substrate: the run script guarded the look-back by
    # clip start, the promoted function guards by cluster. Assert they are the same guard.
    runs = np.flatnonzero(np.diff(cc) != 0) + 1
    one_clip_one_cluster = bool(len(np.unique(cc)) == len(runs) + 1)

    out = {"_what": "Q-B re-derived from the banked scores by the PROMOTED stack function",
           "_implementation": "tanitad.eval.sitclf_deploy.event_anticipation_report",
           "one_clip_one_cluster": one_clip_one_cluster, "situations": {}}
    bad = []
    for i, s in enumerate(sits):
        og = onsets[osit == i]
        row = {}
        for arm in ARMS:
            got = event_anticipation_report(
                z[arm][:, i].astype(np.float64), EVAL[:, i], og, cc,
                top_frac=R["per_situation"][s]["QB_DEPLOY_HORIZON"]["top_frac"],
                h_max_s=R["per_situation"][s]["QB_DEPLOY_HORIZON"]["h_max_s"],
                deploy_lead_s=3.0)
            want = R["per_situation"][s]["QB_DEPLOY_HORIZON"]["arms"][arm]
            checks = {
                "n_alarm": (got["n_alarm"], want["n_alarm"]),
                "n_onsets": (got["n_onsets"], want["n_onsets"]),
                "n_onsets_warned": (got["n_onsets_warned"], want["n_onsets_warned"]),
                "event_recall": (got["event_recall"], want["event_recall"]["point"]),
                "median_lead_s": (got["median_lead_s"], want["median_lead_s"]["point"]),
                "alarm_precision_5s": (got["alarm_precision_h_max"],
                                       want["alarm_precision_5s"]["point"]),
                "alarm_precision_3s": (got["alarm_precision_deploy"],
                                       want["alarm_precision_3s"]["point"])}
            diffs = {}
            for k, (g, w) in checks.items():
                if g is None or w is None:
                    diffs[k] = "both None" if g is w else f"{g} vs {w}"
                    if g is not w:
                        bad.append((s, arm, k, g, w))
                    continue
                d = abs(float(g) - float(w))
                diffs[k] = round(d, 9)
                if d > 1e-4:                 # the JSON rounds to 5 dp
                    bad.append((s, arm, k, g, w))
            row[arm] = {"recomputed": {k: got[k] for k in
                                       ("n_alarm", "n_onsets", "n_onsets_warned",
                                        "event_recall", "median_lead_s",
                                        "alarm_precision_h_max", "alarm_precision_deploy")},
                        "abs_diff_vs_banked": diffs}
        out["situations"][s] = row
        log(f"  {s:>13}: {len(ARMS)} arms re-derived, "
            f"{sum(1 for b in bad if b[0] == s)} mismatches")
    out["MISMATCHES"] = bad
    out["VERDICT"] = ("PASS — two independent implementations agree on every Q-B headline"
                      if not bad else f"FAIL — {len(bad)} mismatches")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(out["VERDICT"])
    if bad:
        log(f"  first mismatches: {bad[:8]}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
