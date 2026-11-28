"""``--map-hires-trunk-grad-scale S`` -- a gradient scale on the trunk features that enter the banded map-hires head (K2DC's next lever,
``PREREG_K2DC02_2026-11-18.md`` section 4 step 2: "a stop-gradient from the map head into the trunk").

forward = identity; backward multiplies the gradient by ``S``. ``S == 1.0`` (the default) is OFF: NO op is inserted, no attribute is set,
the graph, the RNG draws, the loss and every gradient are the tip's, bit for bit. ``0 <= S < 1`` is ON; ``S == 0`` is the stop-gradient
(the backward returns EXACT zeros, so a trunk parameter fed only by the map term reads exactly 0.0). Anything outside [0, 1] (or NaN) is
refused, naming the flag.

WHERE the op sits (``MapHiresBranch._gs``, ``tanitad/models/map_head_hires.py``), and nowhere else -- the BANDED head's inputs:
  * ``src = fmap_s8``  : the stride-8 trunk map the banded head's near lift reads directly;
  * ``feats``          : the shared 0.25 m BEV encoder's output the banded decoders read (the planner / box BEV reads the UNSCALED ``feats``,
                         ``out["map_hires_bev"]``, so no other loss term sees the op);
  * ``bev`` into ``map_encoder`` (the M3 own-encoder builds only): the map's own encoder stays a full-gradient parameter group, its
                         input is scaled, so the gradient it hands back upstream (shared lift / trunk) is scaled.
CONSEQUENCE (declared, not hidden): the shared ``_map_hires.lift`` / ``_map_hires.encoder`` parameters sit UPSTREAM of the op under the shared
build, so they receive the map term's gradient scaled by S too (they keep the planner's / box's, unscaled). Only the parameters downstream
of the op (the banded head, ``refine``, ``near``, ``near_refine``, ``map_encoder``) keep the map term's full gradient.

No parameter, no buffer: ``state_dict`` is unchanged and a checkpoint / replay loads as before.
"""
from __future__ import annotations

import math

import torch

FLAG = "--map-hires-trunk-grad-scale"
DEST = "map_hires_trunk_grad_scale"
#: set on the model ONLY when the flag is ON (S < 1); an absent attribute is the tip's model (= 1.0)
MODEL_ATTR = "_map_trunk_gs"
OFF = 1.0


def add_arguments(g) -> None:
    g.add_argument(FLAG, type=float, default=OFF, dest=DEST,
                   help="K2DC next lever: scale the gradient the BANDED map-hires head sends into the trunk by S (forward = identity). "
                        "1.0 (the default) = OFF: no op is inserted, the tip's graph. 0 <= S < 1 = ON; 0 = stop-gradient. Applied only where "
                        "the trunk features enter the banded head (the planner / box BEV and every other term are unaffected). Needs "
                        "--r9-map-banded, --map-hires on and --w-map-hires > 0 when S < 1. Refused outside [0, 1].")


def scale_for(args) -> float:
    v = getattr(args, DEST, OFF)
    return OFF if v is None else float(v)


def is_on(args) -> bool:
    return scale_for(args) != OFF


def pin_args(args) -> None:
    """Argv refusals. OFF (S == 1.0): nothing is checked."""
    s = scale_for(args)
    if not (0.0 <= s <= 1.0) or math.isnan(s):
        raise SystemExit(f"[map-trunk-gs] {FLAG} {s!r} refused: the gradient scale is in [0, 1] (1 = off, 0 = stop-gradient)")
    if s == OFF:
        return
    bad = []
    if not bool(getattr(args, "r9_map_banded", False)):
        bad.append("--r9-map-banded is not passed: the op sits on the BANDED head's inputs, so on any other build it would be declared "
                   "and never built")
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: there is no map-hires branch to scale")
    if not float(getattr(args, "w_map_hires", 0.0) or 0.0) > 0.0:
        bad.append("--w-map-hires is 0: the map term sends no gradient to scale")
    if bad:
        raise SystemExit(f"[map-trunk-gs] {FLAG} {s:g} refused:\n  - " + "\n  - ".join(bad))


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, scale):
        ctx.scale = float(scale)
        return a.view_as(a)

    @staticmethod
    def backward(ctx, g):
        return (torch.zeros_like(g) if ctx.scale == 0.0 else g * ctx.scale), None


def grad_scale(x, s: float):
    """forward = ``x`` (a view); backward = ``s * grad`` (EXACT zeros at ``s == 0``). The caller never reaches this at ``s == 1``."""
    return _GradScale.apply(x, float(s))


def attach(model, args) -> None:
    """Called by train() right after the branch is built (and mirrored by the two replays). OFF: nothing is set."""
    s = scale_for(args)
    if s == OFF:
        return
    br = getattr(model, "_map_hires", None)
    if br is None or getattr(br, "banded", None) is None:
        raise SystemExit(f"[map-trunk-gs] {FLAG} {s:g} but no banded map-hires branch is built: the lever would be declared and never built")
    setattr(model, MODEL_ATTR, s)
    br.trunk_grad_scale = s
    print(f"[map-trunk-gs] banded map-hires head -> trunk gradient scale {s:g} ({'stop-gradient' if s == 0.0 else 'scaled'})", flush=True)


def dvb_check(model, args) -> list:
    """G-DVB ``built`` lever: the declared scale == what the forward READS (the branch attribute) == the model attribute."""
    from tanitad.train.declared_vs_built import Mismatch
    want = scale_for(args)
    br = getattr(model, "_map_hires", None)
    got_br = OFF if br is None else float(getattr(br, "trunk_grad_scale", OFF))
    got_m = float(getattr(model, MODEL_ATTR, OFF))
    out = []
    if got_br != want:
        out.append(Mismatch(DEST, want, got_br, "model._map_hires.trunk_grad_scale", "argv and the scale the banded head's forward reads disagree"))
    if got_m != want:
        out.append(Mismatch(DEST, want, got_m, f"model.{MODEL_ATTR}", "argv and the model attribute disagree"))
    return out


def register_dvb_levers(register) -> None:
    register(DEST, "built", dvb_check)
