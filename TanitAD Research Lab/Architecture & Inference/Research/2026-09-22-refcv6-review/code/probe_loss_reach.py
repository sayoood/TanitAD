#!/usr/bin/env python
"""Q2 — every weighted loss term the refcv6 arm carries, and whether it
produces a NON-ZERO PARAMETER GRADIENT.

The registered `D-RC5-GROUND-DEAD` is the template: `agent_w_ground` reaches
`config.json`, adds `w * 0` to the total, and would read afterwards as
*"the ground prior does not help"*. That defect is invisible to a guard on the
flag and to a guard on the camera — **only a guard on the gradient can see it.**

Method (copied from `refc_v3_train.assert_ground_prior_is_supervised`, which is
the instrument that MEASURED the 1.164e-10):

  * one real `compute_losses_v3` forward on synthetic episodes, `retain_graph`;
  * for EVERY scalar tensor in the returned `losses` dict, `torch.autograd.grad`
    against **all trainable parameters**, and report `sum(|grad|)`;
  * ⛔ **a SAME-BREATH POSITIVE CONTROL** — `loss_traj`, the term the whole
    model exists to minimise — measured through the identical code path. If the
    control is flat, every reading is **INCONCLUSIVE, which is a refusal**,
    never a pass. A flat gradient and a probe that never ran are the same number.
  * `allow_unused=True` with `materialize_grads=False`: a parameter that is not
    in the term's graph contributes 0 and is COUNTED, so "reaches n of N
    tensors" is reported beside the magnitude.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch

TRAINER = pathlib.Path(r"D:/Projects/TanitAD/stack/scripts/refc_v3_train.py")
#: the floor `refc_v3_train` itself uses for "this gradient is zero"
FLOOR_FALLBACK = 1e-9


def _load():
    spec = importlib.util.spec_from_file_location("refc_v3_train", TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--extra", default="", help="extra argv, space separated")
    a = ap.parse_args()
    m = _load()
    from tanitad.refs import refc_v3 as v3

    argv = ["--smoke", "--arm", "hier", "--out", "unused"]
    if a.extra.strip():
        argv += a.extra.split()
    p = m.build_parser()
    args = p.parse_args(argv)
    cfg = m._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    # the weights that travel ON the model (compute_losses_v3 has no `args`)
    model._w_agent = float(getattr(args, "w_agent", 0.0))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    model._w_u0 = float(getattr(args, "w_u0", 0.0))
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0))
    for dest in ("w_tac_goal", "w_tac_v6", "w_map", "w_box3d",
                 "w_r7_wta", "w_r7_scorer"):
        setattr(model, f"_{dest}", float(getattr(args, dest, 0.0)))

    eps = m._synth_episodes(2, cfg.core, seed=0)
    ds = m.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                   channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])

    model.train()
    losses = m.compute_losses_v3(model, batch, "cpu", mode=args.mode)
    params = [q for q in model.parameters() if q.requires_grad]
    n_par = len(params)
    print(f"model: {n_par} trainable tensors, "
          f"{sum(q.numel() for q in params):,} params")
    print(f"losses dict: {len(losses)} keys\n")

    def reach(t):
        gs = torch.autograd.grad(t, params, retain_graph=True,
                                 allow_unused=True, materialize_grads=False)
        tot = 0.0
        hit = 0
        for g in gs:
            if g is None:
                continue
            s = float(g.abs().sum())
            tot += s
            if s > 0.0:
                hit += 1
        return tot, hit

    # ---- THE SAME-BREATH POSITIVE CONTROL ------------------------------- #
    ctrl_key = "traj" if "traj" in losses else None
    if ctrl_key is None:
        ctrl_key = next((k for k in ("loss_traj", "loss") if k in losses), None)
    if ctrl_key is None:
        raise SystemExit("INCONCLUSIVE: no control term in the losses dict.")
    ctrl_t = losses[ctrl_key]
    if not (torch.is_tensor(ctrl_t) and ctrl_t.requires_grad):
        raise SystemExit(f"INCONCLUSIVE: control {ctrl_key!r} carries no graph.")
    c_tot, c_hit = reach(ctrl_t)
    print(f"CONTROL  {ctrl_key:<26} grad_abs_sum={c_tot:.6e}  "
          f"reaches {c_hit}/{n_par} tensors")
    if c_tot <= FLOOR_FALLBACK:
        raise SystemExit(
            f"⛔ INCONCLUSIVE (= a refusal): the same-breath POSITIVE CONTROL "
            f"read {c_tot:.3e}. Every flat reading below would be "
            f"indistinguishable from a probe that never ran.")
    print()

    rows = []
    for k in sorted(losses):
        t = losses[k]
        if not (torch.is_tensor(t) and t.ndim == 0):
            rows.append({"term": k, "kind": "not-a-scalar-tensor",
                         "value": (float(t) if isinstance(t, (int, float, bool))
                                   else str(type(t).__name__))})
            continue
        if not t.requires_grad:
            rows.append({"term": k, "kind": "NO_GRAPH",
                         "value": float(t.detach()),
                         "grad_abs_sum": 0.0, "reaches": 0})
            continue
        tot, hit = reach(t)
        rows.append({"term": k, "kind": "live", "value": float(t.detach()),
                     "grad_abs_sum": tot, "reaches": hit,
                     "DEAD": tot <= FLOOR_FALLBACK})
    dead = [r for r in rows if r.get("kind") == "live" and r.get("DEAD")]
    nog = [r for r in rows if r.get("kind") == "NO_GRAPH"]
    live = [r for r in rows if r.get("kind") == "live" and not r.get("DEAD")]

    print(f"{'term':<30} {'value':>13} {'grad_abs_sum':>15} {'reaches':>10}")
    for r in sorted(rows, key=lambda x: -(x.get("grad_abs_sum") or -1)):
        if r["kind"] == "not-a-scalar-tensor":
            continue
        flag = "  <- DEAD" if r.get("DEAD") else ("  (no graph)"
                                                 if r["kind"] == "NO_GRAPH" else "")
        print(f"{r['term']:<30} {r['value']:>13.6g} "
              f"{r.get('grad_abs_sum', 0.0):>15.6e} "
              f"{r.get('reaches', 0):>6}/{n_par}{flag}")

    print(f"\n  live: {len(live)}   DEAD (grad <= {FLOOR_FALLBACK:.0e}): "
          f"{len(dead)}   NO_GRAPH: {len(nog)}")
    if dead:
        print("  DEAD terms: " + ", ".join(r["term"] for r in dead))
    if nog:
        print("  NO_GRAPH terms: " + ", ".join(r["term"] for r in nog))

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"argv": argv, "n_trainable_tensors": n_par,
             "control": {"term": ctrl_key, "grad_abs_sum": c_tot,
                         "reaches": c_hit},
             "floor": FLOOR_FALLBACK, "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"[artifact] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
