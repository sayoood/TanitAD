#!/usr/bin/env python3
"""RE-RENDER attempt for the e1a horizon node — and the reason it CANNOT be done.

TARGET (named in `…/2026-07-27-vtband-wiring/VTBAND_WIRING.md` §4.3):
  artifact  …/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44.json
  node      paired_common_start / deltas_vs_K20 / longitudinal / 160 / d_closed_ade2s_m
  printed   delta 0.0  [0.0, 0.0]  separated=true  p_delta_gt0 0.975  n=80 / 21 ep

⛔ BLOCKED, AND THE BLOCK IS THE FINDING. A re-render needs the estimator's raw
draws, i.e. the two per-window ``ade2s`` vectors at K=160 and K=20 on the common
start set. ``e1a_horizon.py`` (committed beside the artifact) builds them inside
``main()`` from closed-loop GPU rollouts and writes ONLY the summary JSON — no
``torch.save``/``np.savez`` anywhere in the file. Probed at three locations
(CLAUDE.md rule 2 — absence at one location is not absence):
  1. the committed artifact itself — summary nodes only, no per-window arrays;
  2. the repo — no ``pw_*``/``perwindow_*`` dump belonging to this run;
     `…/2026-07-26-horizon-envelope-closeout/artifacts/perwindow_K{20,60,70,185}.pt`
     is a DIFFERENT design (K set 20/60/70/185, 41 common windows / 40 eps) and
     cannot substitute for this one (K 20/40/80/120/160, 175 common windows);
  3. the producing host `tanitad-pod3:/workspace/e1a_e2a/` — the JSONs and the
     scripts survive; no per-window tensor does.
⇒ Recovering the draws means re-running the closed-loop rollouts, which is a
RECOMPUTE (explicitly out of scope) and a GPU job. NOT DONE.

WHAT IS STILL DERIVABLE FROM THE COMMITTED ARTIFACT ALONE — recorded here
because it is what the marginal case actually rests on, and it needs no draws:
``p_delta_gt0`` is ``(d > 0).mean()`` over B=2000 draws, so 0.975 means EXACTLY
1950 draws > 0 and 50 draws ≤ 0. ``lo`` is ``np.percentile(d, 2.5)``, whose
0-based linear-interpolation index is 0.025·(2000−1) = 49.975 — i.e. ``lo`` is
fixed by the two order statistics d[49] (≤0) and d[50] (>0), weighted 0.975
toward d[50]. ``separated`` is therefore decided by a single order statistic at
the exact 2.5 % boundary. Both branches are demonstrated numerically below.

Run (dev box, no GPU, no pod):
  "C:/Users/Admin/venvs/tanitad/Scripts/python.exe" <this file>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
ART = (REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Implementation"
       / "incoming" / "2026-07-25-closedloop-horizon-and-shift" / "e1a_horizon_heldout44.json")
OUT = Path(__file__).resolve().parents[1] / "raw" / "rerender_e1a_horizon_BLOCKED.json"

B, ALPHA = 2000, 0.05


def main() -> int:
    d = json.loads(ART.read_text(encoding="utf-8"))
    node = d["paired_common_start"]["deltas_vs_K20"]["longitudinal"]["160"]["d_closed_ade2s_m"]

    idx = (ALPHA / 2) * (B - 1)
    lo_i, w = int(np.floor(idx)), float(idx - np.floor(idx))
    n_le0 = int(round((1.0 - node["p_delta_gt0"]) * B))

    # both branches, at the SAME p_delta_gt0 the artifact printed
    sep_case = np.concatenate([np.full(n_le0, -1e-9), np.full(B - n_le0, 1e-6)])
    nul_case = np.concatenate([np.full(n_le0, -1e-4), np.full(B - n_le0, 1e-6)])
    demo = {}
    for name, arr in (("separated_if_d49_is_barely_negative", sep_case),
                      ("NOT_separated_if_d49_is_more_negative", nul_case)):
        lo, hi = np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        demo[name] = {"p_delta_gt0": float((arr > 0).mean()), "lo": float(lo),
                      "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}

    res = {
        "_what": "RE-RENDER BLOCKED — the raw draws this node needs are not persisted anywhere.",
        "_evidence_class": "MEASURED (ours) for the probes and the arithmetic; the node's own "
                           "values are quoted from the committed artifact, unmodified.",
        "_source_artifact": str(ART.relative_to(REPO)).replace("\\", "/"),
        "_source_node": "paired_common_start/deltas_vs_K20/longitudinal/160/d_closed_ade2s_m",
        "_source_artifact_unmodified": True,
        "committed_rendering": node,
        "RE_RENDERED": False,
        "_blocked_because": "e1a_horizon.py persists only the summary JSON; the per-window "
                            "closed-loop ade2s vectors at K=160 and K=20 exist in no artifact.",
        "_probes": [
            {"where": "the committed artifact", "result": "summary nodes only; all_windows[K] "
             "holds bootstrapped blocks, not per-window arrays"},
            {"where": "repo-wide search for this run's per-window dumps",
             "result": "none; …/2026-07-26-horizon-envelope-closeout/artifacts/perwindow_K*.pt "
                       "is a different design (K 20/60/70/185, 41 common windows) and is NOT a "
                       "substitute"},
            {"where": "tanitad-pod3:/workspace/e1a_e2a/ (the producing host)",
             "result": "JSONs + scripts present; no per-window tensor"},
        ],
        "_what_recovery_would_cost": "re-running the closed-loop rollouts on a GPU = a RECOMPUTE, "
                                     "which this job explicitly excludes.",
        "derivable_without_the_draws": {
            "_rule": "p_delta_gt0 = (d > 0).mean() over B draws; lo = np.percentile(d, 2.5).",
            "n_boot": B,
            "n_draws_gt_0": B - n_le0,
            "n_draws_le_0": n_le0,
            "percentile_index_0based": idx,
            "lo_is_interpolated_between": [f"d[{lo_i}]", f"d[{lo_i + 1}]"],
            "weight_on_upper_order_statistic": w,
            "reading": "d[49] <= 0 < d[50], so `lo` sits on the exact 2.5 % boundary and "
                       "`separated` is decided by ONE order statistic. It holds only while "
                       "d[49] > -39 * d[50].",
            "bounds_from_the_printed_rounding": "|delta| < 5e-5 and |lo| < 5e-5, since both "
                                                "round to 0.0 at 4 dp.",
            "demonstration": demo,
        },
        "_conclusion_impact": "NONE. E1a_E2a_RESULTS.md's verdict rests on corridor-departure at "
                              "an 18.5 s horizon and states explicitly that ADE@2s 'CANNOT see 18 s "
                              "drift'; it quotes closed_ade2s only as the all-window 0.485->0.496 "
                              "move, not this paired node. No published conclusion leans on it.",
        "_escalation": "OWNER of 2026-07-25-closedloop-horizon-and-shift: this node cannot be "
                       "re-rendered from anything that exists. Either re-run e1a_horizon.py with a "
                       "per-window dump added (one np.savez), or stamp the node unquotable the way "
                       "legA_v5config_structural.json was stamped. Do not leave "
                       "'0.0 [0.0, 0.0] separated' standing.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(f"\n[e1a] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
