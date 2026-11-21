"""Q4 — information-disjointness and the DETACH, measured.

SPEC §4: *"Feeds the planner, detached: (a) the lat/lon posterior replaces the
image-only lat3/lon3 as the anchor prior (new zero-init 8->117); (b) the
valid-behaviour set gates selection."*
CLAUDE.md BINDING (PI 2026-08-03): a goal input is admissible but must NOT
carry the situation classifier's output; the goal path and the situation path
stay information-disjoint at inference.

Three MEASUREMENTS, none of them a code reading:

  D1  the DETACH is real — a planner-side loss must put ZERO gradient on the
      tactical decoder's parameters, while the tactical loss puts NON-ZERO
      gradient on them.  ⛔ A "0" from a backward that reached nothing is
      indistinguishable from a detach, so the control is the same backward
      reaching the TRUNK (R3 says it must) and the tactical loss reaching the
      same parameters that read 0 above.

  D2  the graft EXISTS and is zero-init — find the 8 -> n_anchors projection
      the spec promises and report its shape and its init.

  D3  DISJOINTNESS, interventionally: does the SITUATION half of the tactical
      output change the goal/anchor prior?  The five SITUATION_OUTPUT_TOKENS
      (`tac_SIT`, `YIELD`, the three TRAFFIC_LIGHT_*) must be structurally dead
      columns of the selection graft. Measured by MUTATING those columns of the
      behaviour vector and asserting the selection term does not move — with a
      NON-situation column mutated in the same breath, which MUST move it.
      ⛔ Without that second half the test passes on a graft that is dead
      everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402

from tanitad.refs import refcv6_tactical as v6tac   # noqa: E402
from tanitad.models.vocab_v7 import TACTICAL_GOAL_TOKENS_V7  # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def _census(model, prefix):
    n = nz = 0
    tot = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            continue
        n += 1
        if p.grad is not None:
            g = float(p.grad.abs().sum())
            tot += g
            nz += int(g > 0)
    return {"n": n, "nonzero": nz, "sum_abs": round(tot, 10)}


def _zero(m):
    for p in m.parameters():
        p.grad = None


def main():
    res = {}
    # ⛔ `graft_tac8_prior=True`, or the SPEC §4(a) seam is NOT BUILT and this
    # probe measures a detach on a graft that does not exist. (The first run of
    # this instrument did exactly that and found only the 3-wide lat3/lon3
    # pair; the finding was a scoping error in the PROBE, and it is recorded
    # here so the correction is legible.)
    model, cfg = B.build(tac8_prior=True)
    model.train()
    f, kw = B.batch(model, cfg, nav=1)

    # ---------------------------------------------------------------- D1
    _zero(model)
    out = model(f, **kw)
    (out["traj"].pow(2).mean()).backward()
    res["D1_planner_loss"] = {
        "tac_decoder_v6": _census(model, "tac_decoder_v6."),
        "CONTROL_trunk_must_be_nonzero": _census(model, "core.encoder."),
        "CONTROL_opdecoder_must_be_nonzero": _census(model, "core.decoder.layers."),
        "tac_behaviour_gate_v6": _census(model, "tac_behaviour_gate_v6"),
    }

    _zero(model)
    out = model(f, **kw)
    # the TACTICAL loss — the same surface `tactical_behaviour_losses` reads
    tl = (out["tacv6_goal_logits"].pow(2).mean()
          + out["tacv6_lat_logits"].pow(2).mean()
          + out["tacv6_lon_logits"].pow(2).mean())
    tl.backward()
    res["D1_tactical_loss"] = {
        "tac_decoder_v6_must_be_nonzero": _census(model, "tac_decoder_v6."),
        "trunk_R3_backprop_into_trunk": _census(model, "core.encoder."),
        "perception_branch": _census(model, "_perception."),
    }

    # ---------------------------------------------------------------- D2
    dec = model.core.decoder
    grafts = {}
    for nm in ("tac8_lat_to_anchor", "tac8_lon_to_anchor",
               "lat_to_anchor", "lon_to_anchor", "route_to_anchor",
               "maneuver_to_anchor"):
        g = getattr(dec, nm, None)
        grafts[nm] = (None if g is None else
                      {"shape": list(g.weight.shape),
                       "weight_all_zero": bool((g.weight == 0).all()),
                       "bias": (None if g.bias is None
                                else bool((g.bias == 0).all()))})
    bg = getattr(model, "tac_behaviour_gate_v6", None)
    res["D2_grafts"] = {
        "decoder_anchor_prior_grafts": grafts,
        "tac_behaviour_gate_v6": (
            None if bg is None else
            {"type": type(bg).__name__,
             "params": {n: {"shape": list(p.shape),
                            "all_zero": bool((p == 0).all())}
                        for n, p in bg.named_parameters()}}),
        "planner_feeds_keys": sorted(v6tac.planner_feeds({
            "lat_logits": torch.zeros(1, 8), "lon_logits": torch.zeros(1, 8),
            "goal_logits": torch.zeros(1, 22), "goal_conf": torch.zeros(1, 22),
        })),
        "planner_feeds_all_detached": all(
            not t.requires_grad for t in v6tac.planner_feeds({
                "lat_logits": torch.zeros(1, 8, requires_grad=True),
                "lon_logits": torch.zeros(1, 8, requires_grad=True),
                "goal_logits": torch.zeros(1, 22, requires_grad=True),
                "goal_conf": torch.zeros(1, 22, requires_grad=True),
            }).values()),
    }

    # ---------------------------------------------------------------- D3
    sit = sorted(v6tac.SITUATION_OUTPUT_TOKENS
                 & set(TACTICAL_GOAL_TOKENS_V7))
    sit_idx = [TACTICAL_GOAL_TOKENS_V7.index(t) for t in sit]
    non_sit_idx = [i for i in range(len(TACTICAL_GOAL_TOKENS_V7))
                   if i not in sit_idx]
    d3 = {"situation_tokens_in_vocab": sit,
          "situation_indices": sit_idx,
          "n_non_situation": len(non_sit_idx),
          "SITUATION_OUTPUT_TOKENS_declared": sorted(
              v6tac.SITUATION_OUTPUT_TOKENS)}
    if bg is not None:
        base = torch.full((2, len(TACTICAL_GOAL_TOKENS_V7)), 0.5)
        with torch.no_grad():
            # ⭐ the gate is zero-init by design, so it is given LIVE weights
            # first — otherwise every column is dead and the test passes on a
            # brick.
            for p in bg.parameters():
                p.data.normal_(0.0, 0.1)
            y0 = bg(base)
            m_sit = base.clone()
            m_sit[:, sit_idx] = 1.0
            y_sit = bg(m_sit)
            m_non = base.clone()
            m_non[:, non_sit_idx[0]] = 1.0
            y_non = bg(m_non)
        d3["delta_from_SITUATION_columns"] = float((y_sit - y0).abs().max())
        d3["delta_from_ONE_NONSITUATION_column_CONTROL"] = float(
            (y_non - y0).abs().max())
        d3["verdict"] = (
            "SITUATION columns STRUCTURALLY DEAD (0.0) and a non-situation "
            "column moves the term"
            if float((y_sit - y0).abs().max()) == 0.0
            and float((y_non - y0).abs().max()) > 0.0
            else "NOT ESTABLISHED")
    res["D3_disjointness"] = d3

    # -------------------------------------------- the declaration half
    try:
        roles = model.provenance_roles()
        res["D4_provenance_roles"] = {
            k: roles.get(k) for k in
            ("goal", "inference_inputs_of_goals", "selection_inputs",
             "situation_output")}
    except Exception as e:                                   # pragma: no cover
        res["D4_provenance_roles"] = f"ERROR {e}"

    p = OUT / "q4_detach_disjointness.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
