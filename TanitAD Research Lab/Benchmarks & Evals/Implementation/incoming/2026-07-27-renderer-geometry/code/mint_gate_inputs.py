"""Mint the two NON-corridor inputs ``run_gate.py check`` requires, from the
SAME rollout that produced the corridor block.

``check`` exits ``BLOCKED`` without ``--eval-json`` (it refuses to decide from a
train-log slope) and needs a ``--log`` to read the current step. Borrowing
another arm's eval JSON to make it proceed would attach one arm's ADE to
another arm's corridor — precisely the kind of splice this program keeps
retracting. So both are minted here from ``pw["ade2s"]``, the closed-loop
ADE@2s of the SAME policy on the SAME windows, with the SAME estimator the
corridor block uses (``taniteval.ci.episode_cluster_bootstrap``, B=2000).

⚠️ The resulting ``ade_0_2s`` is a CLOSED-LOOP number, not the open-loop
``eval_flagship_v4`` MODE-B one. It is stamped as such in the JSON. It enters
the gate only as the DEMOTED DIAGNOSTIC (``primary_role: diagnostic``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taniteval import ci as _ci


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perwindow", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--log-out", required=True)
    ap.add_argument("--gate-step", type=int, default=0)
    a = ap.parse_args(argv)

    pw = torch.load(a.perwindow, weights_only=False, map_location="cpu")
    ade = np.asarray(pw["ade2s"].numpy() if torch.is_tensor(pw["ade2s"])
                     else pw["ade2s"], dtype=float)
    eid = list(pw["eid"])
    node = _ci.episode_cluster_bootstrap(ade, eid, n_boot=2000, seed=0)

    ev = {
        "_what": (f"DIAGNOSTIC eval block for {a.run}, minted from the "
                  f"closed-loop rollout that produced the corridor co-primary "
                  f"— same policy, same windows, same estimator."),
        "_surface": "closed_loop (NOT eval_flagship_v4 MODE B open-loop dense)",
        "_NOT_A_MODEL_RESULT": (
            "the planner is a constant-velocity REFERENCE POLICY, not a v5 "
            "checkpoint. Quotable as a property of that policy only."),
        "run": a.run,
        "headline": {"ade_0_2s": node},
        "kill_secondaries": {
            "nonav_route_beats_majority": None,
            "_nonav_route_beats_majority_note": (
                "NOT REACHABLE on this arm — and VOID BY CONSTRUCTION per "
                "GATE_PROTOCOL 0.7: route_target is a lookup of the route "
                "input, so route_skill is 0.0 by construction. Adjudicate "
                "INSTRUMENT-FAIL, NEVER MODEL-FAIL."),
        },
    }
    Path(a.eval_out).write_text(json.dumps(ev, indent=2))

    rows = [{"step": a.gate_step, "loss": None,
             "_note": f"minted by mint_gate_inputs.py for {a.run}; the rollout "
                      f"carries no optimizer step (no training was run)."}]
    Path(a.log_out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps({"ade_0_2s": node, "eval_json": a.eval_out,
                      "log": a.log_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
