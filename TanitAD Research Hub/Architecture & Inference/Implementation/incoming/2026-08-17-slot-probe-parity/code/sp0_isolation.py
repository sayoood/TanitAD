"""SP0 — the FROZEN-TRUNK proof, at the LIVE run's production geometry.

Two independent statements, because either alone is weaker than the pair:

1. ``V6Stack.assert_isolation`` with the agent-slot head BUILT, at both
   ``slot_src`` arms — including the fourth edge ``perception_to_trunk``, whose
   forbidden set is every parameter that is not ``interp``. ``n_probed`` is
   reported for each edge: a probe over an absent module reports zero violations
   and has established NOTHING (the C13 family), so the count is the evidence
   that the check ran.

2. ⭐ The edge CAN FAIL, and that is what makes it a check. The deliberately
   mis-wired arm (``isolate_interp_from_encoder=False``) is run too and MUST
   raise ``IsolationViolation``. A guard that cannot fail is not a guard.

⚠️ This is the STRUCTURAL statement. The probe actually run by ``sp2_probe.py``
is strictly more isolated still: it trains on a BANKED, DETACHED latent tensor
that carries no autograd graph, so no gradient path to the trunk exists even in
principle. Both facts are reported.

Runs on CPU at the run's real widths — no checkpoint and no GPU needed, because
isolation is a property of the WIRING, not of the weights.
"""
from __future__ import annotations
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

STACK = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
for p in (str(STACK), str(STACK / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sp_common import merge_run_args
    from train_v6_staged import build_stack_from_args
    from tanitad.models.v6 import IsolationViolation

    base = json.loads(Path(a.config_json).read_text("utf-8"))["args"]
    rec: dict = {"_evidence_class": "MEASURED (ours; CPU, production widths)",
                 "config_source": str(a.config_json),
                 "n_slot_queries": int(a.n_queries),
                 "arms": {}}

    for src in ("cells", "tokens"):
        ns = merge_run_args(base, agent_slots=True, slot_src=src,
                            n_slot_queries=int(a.n_queries), device="cpu")
        stack = build_stack_from_args(ns)
        groups = {}
        for n, p in stack.named_parameters():
            g = stack.group_of(n)
            groups[g] = groups.get(g, 0) + p.numel()
        rep = stack.assert_isolation()
        rec["arms"][src] = {
            "total_params": int(sum(p.numel() for p in stack.parameters())),
            "per_group": {k: int(v) for k, v in sorted(groups.items())},
            "interp_params": int(groups.get("interp", 0)),
            "n_violations": rep["n_violations"],
            "n_probed": rep["n_probed"],
            "pass": rep["pass"],
            "config": rep["config"],
        }
        print(f"[sp0] {src}: n_violations {rep['n_violations']}  n_probed "
              f"{rep['n_probed']}  pass={rep['pass']}  interp="
              f"{groups.get('interp', 0):,}", flush=True)
        del stack

    # ---- the mis-wired control: the edge MUST fire -------------------------
    ns = merge_run_args(base, agent_slots=True, slot_src="cells",
                        n_slot_queries=int(a.n_queries), device="cpu",
                        no_isolate_interp=True)
    fired, detail = False, None
    try:
        stack = build_stack_from_args(ns)
        rep = stack.assert_isolation()
        detail = {"n_violations": rep["n_violations"], "pass": rep["pass"]}
    except IsolationViolation as ex:
        fired, detail = True, str(ex)[:400]
    rec["miswired_control"] = {
        "arm": "isolate_interp_from_encoder=False",
        "raised_IsolationViolation": fired, "detail": detail,
        "_read": ("the guard is only a guard if it can fail; this arm is the "
                  "demonstration that perception_to_trunk is non-vacuous")}
    rec["probe_isolation_note"] = (
        "sp2_probe.py trains on a BANKED, DETACHED latent tensor with no "
        "autograd graph — strictly stronger than the structural edge above: "
        "no gradient path to the trunk exists even in principle.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), "utf-8")
    print("[sp0] " + json.dumps({k: rec["arms"][k]["n_violations"]
                                 for k in rec["arms"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
