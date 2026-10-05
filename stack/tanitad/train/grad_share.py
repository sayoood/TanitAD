"""In-run GRADIENT-SHARE instrument (refcv8 WP-B DESIGN sec. 3.6 X4) -- WP-D's P-GRAD statistic, inside the trainer.

P-GRAD (``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/RESULT.md``
sec. 7.2, MEASURED on refcv7-50,400, frozen batch, all weights 1.0) found the trunk update paid for by agent 50.1 %,
box3d 35.7 %, the planner auxiliaries 14.2 %, the trajectory 0.02 %, the 10 cm map 0.01 %, the tactical decoder
-0.05 %. A loss-budget arm is sized and judged on THAT statistic -- never on a loss-value share -- so the trainer must
measure it in-run, on the run's own batches and weights.

The statistic (per parameter group, on ONE batch, at the weights the terms enter ``loss`` with):

* ``proj_t = <g_t, g_tot> / ||g_tot||^2`` -- the share of the update term ``t`` "pays for" (sums to 1 over the terms
  plus ``rest`` = total minus the named terms);
* ``norm_t = ||g_t||`` and ``norm_total``;
* the LINEARITY CONTROL ``lin_rel_err = ||sum_t g_t - g_tot|| / ||g_tot||`` (P-GRAD's bar: <= 1e-4) -- a reading whose
  control misses is logged with the control, never silently.

``torch.autograd.grad`` throughout: ``.grad`` is never touched, so the optimiser step that follows is the step the
trainer would have taken without the instrument (pinned: ``tests/test_refcv8_grad_share.py``). No RNG is consumed.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

import torch
from torch import Tensor

__all__ = ["TERMS", "GROUP_PREFIXES", "term_tensors", "param_groups", "measure", "bf16_levers", "fp32_replay"]

#: the named terms, in the order P-GRAD reports them; ``rest`` is derived
TERMS = ("traj", "agent", "box3d", "map_hires", "tac_v6", "r8")
#: parameter groups (name prefix on ``model.named_parameters()``); P-GRAD's ``trunk`` = the encoder stem..temporal fuse
GROUP_PREFIXES = {"trunk": ("core.encoder.",),
                  "bev025": ("_map_hires.lift.", "_map_hires.encoder."),
                  "bev_pool": ("_perception.bev_pool.",)}


def term_tensors(losses: Mapping[str, Tensor], weights: Mapping[str, float]) -> dict:
    """``{term: weighted attached tensor}`` for every term with a non-zero weight that is PRESENT and attached in this
    batch's loss dict, plus ``rest = loss - sum(named)``. A zero weight or an absent / detached key is not a term
    (its gradient is the zero vector, and a share of it is not a measurement)."""
    out: dict = {}
    for t in TERMS:
        w = float(weights.get(t, 0.0) or 0.0)
        v = losses.get(t)
        if w == 0.0 or v is None or not torch.is_tensor(v) or not v.requires_grad:
            continue
        out[t] = w * v
    total = losses["loss"]
    named = None
    for v in out.values():
        named = v if named is None else named + v
    out["rest"] = total if named is None else total - named
    return out


def param_groups(model, prefixes: Mapping[str, tuple] = GROUP_PREFIXES) -> dict:
    """``{group: [params]}`` over the parameters that REQUIRE grad, non-empty groups only."""
    g: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for name, pre in prefixes.items():
            if n.startswith(pre):
                g.setdefault(name, []).append(p)
                break
    return g


def _grads(t: Tensor, params: list) -> list:
    gr = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if x is None else x for x, p in zip(gr, params)]


def measure(terms: Mapping[str, Tensor], total: Tensor, groups: Mapping[str, list], *,
            prefix: str = "gs_") -> dict:
    """One reading -> flat ``{key: float}`` for the metrics row (exact, never rounded: a 1e-4 share rounded to 5 dp is
    the "no signal" misreading). Keys: ``{prefix}{group}_proj_{term}``, ``{prefix}{group}_norm_{term}``,
    ``{prefix}{group}_norm_total``, ``{prefix}{group}_lin_rel_err``, ``{prefix}n_terms``."""
    if not total.requires_grad:
        return {f"{prefix}skipped_no_graph": 1.0}
    flat, owner = [], []
    for gname, ps in groups.items():
        for p in ps:
            flat.append(p)
            owner.append(gname)
    if not flat:
        return {f"{prefix}skipped_no_params": 1.0}
    g_tot = _grads(total, flat)
    tot2 = {g: 0.0 for g in groups}
    for gname, x in zip(owner, g_tot):
        tot2[gname] += float(torch.sum(x.double() * x.double()))
    acc = [torch.zeros_like(x, dtype=torch.float64) for x in g_tot]
    row: dict = {f"{prefix}n_terms": float(len(terms))}
    for t, v in terms.items():
        if not v.requires_grad:
            continue
        gt = _grads(v, flat)
        dot = {g: 0.0 for g in groups}
        nrm = {g: 0.0 for g in groups}
        for i, (gname, x) in enumerate(zip(owner, gt)):
            xd = x.double()
            dot[gname] += float(torch.sum(xd * g_tot[i].double()))
            nrm[gname] += float(torch.sum(xd * xd))
            acc[i] += xd
        del gt
        for g in groups:
            row[f"{prefix}{g}_proj_{t}"] = (dot[g] / tot2[g]) if tot2[g] > 0 else float("nan")
            row[f"{prefix}{g}_norm_{t}"] = math.sqrt(nrm[g])
    err2 = {g: 0.0 for g in groups}
    for gname, a, x in zip(owner, acc, g_tot):
        d = a - x.double()
        err2[gname] += float(torch.sum(d * d))
    for g in groups:
        row[f"{prefix}{g}_norm_total"] = math.sqrt(tot2[g])
        row[f"{prefix}{g}_lin_rel_err"] = (math.sqrt(err2[g]) / math.sqrt(tot2[g])) if tot2[g] > 0 else 0.0
    return row


def weights_of(model, traj_weight: float) -> dict:
    """The weights each term enters ``loss`` with, read off the model the trainer set them on. ``r8`` is already the
    WEIGHTED refcv8 sum (``refcv8_train.r8_losses``), so it enters at 1.0."""
    return {"traj": float(traj_weight), "agent": float(getattr(model, "_w_agent", 0.0) or 0.0),
            "box3d": float(getattr(model, "_w_box3d", 0.0) or 0.0),
            "map_hires": float(getattr(model, "_w_map_hires", 0.0) or 0.0),
            "tac_v6": float(getattr(model, "_w_tac_v6", 0.0) or 0.0),
            "r8": 1.0 if bool(getattr(model, "r8_enabled", False)) else 0.0}


# ---------------------------------------------------------------------------------------------------------------- #
# the fp32 REPLAY (MEASURED 2026-10-05, the Thor L3 smoke: under `--trunk-bf16` the trunk group's linearity control  #
# read 5.9e-3 against the 1e-4 bar -- a bf16 backward makes the gradient of the sum differ from the sum of the        #
# gradients by ~ bf16 epsilon; the fp32 groups read 9.3e-5 / 6.0e-5). The statistic is read on a REPLAY of the SAME  #
# step's forward with every bf16 lever OFF; everything the replay could disturb is restored, so the training step  #
# that follows is the step the trainer would have taken without the instrument.                                    #
# ---------------------------------------------------------------------------------------------------------------- #
def bf16_levers(model) -> list:
    """Every module whose ``memory_levers['bf16']`` is ON (the timm trunk's autocast lever)."""
    out = []
    for m in model.modules():
        lv = getattr(m, "memory_levers", None)
        if isinstance(lv, dict) and bool(lv.get("bf16", False)):
            out.append(m)
    return out


def _dedicated_generators(model) -> list:
    """Every refcv8 dedicated generator reachable from the model (``R8Generator`` duck type: ``_g`` dict of
    torch.Generator) -- their states are snapshotted and restored around the replay."""
    seen, out = set(), []
    for obj in [model] + list(model.modules()):
        for v in list(vars(obj).values()):
            g = getattr(v, "_g", None)
            if isinstance(g, dict) and all(isinstance(x, torch.Generator) for x in g.values()) and id(v) not in seen:
                seen.add(id(v))
                out.append(v)
    return out


def fp32_replay(model, run_losses: Callable[[], dict], groups: Mapping[str, list], weights: Mapping[str, float], *,
                prefix: str = "gs_") -> dict:
    """``run_losses()`` re-runs THIS step's forward + losses; it is called with every bf16 lever OFF, inside a forked
    global RNG, with the dedicated generators' states and every BUFFER (BN running stats, ...) snapshotted and restored
    afterwards. -> ``measure(...)`` of that fp32 graph, plus ``{prefix}fp32_replay: 1.0`` and the number of levers
    switched. ``.grad`` is never touched (``torch.autograd.grad`` only)."""
    levers = bf16_levers(model)
    gens = _dedicated_generators(model)
    gstate = [{k: g.get_state() for k, g in v._g.items()} for v in gens]
    bufs = {n: b.detach().clone() for n, b in model.named_buffers()}
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        for m in levers:
            m.memory_levers["bf16"] = False
        with torch.random.fork_rng(devices=devs):
            losses = run_losses()
            row = measure(term_tensors(losses, weights), losses["loss"], groups, prefix=prefix)
        del losses
    finally:
        for m in levers:
            m.memory_levers["bf16"] = True
        for v, st in zip(gens, gstate):
            for k in [k for k in v._g if k not in st]:
                del v._g[k]              # a generator FIRST created by the replay: dropped, re-created fresh on use
            for k, s_ in st.items():
                v._g[k].set_state(s_)
        with torch.no_grad():
            for n, b in model.named_buffers():
                if n in bufs:
                    b.copy_(bufs[n])
    row[f"{prefix}fp32_replay"] = 1.0
    row[f"{prefix}fp32_replay_levers"] = float(len(levers))
    return row

