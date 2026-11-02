"""``--r9-map-banded`` -- the refcv9 banded, channelled map head (C1; REFCV9_DESIGN 2.1 / 2.2). Argv refusals, G-DVB, the stamp.

The head, targets and losses live in ``tanitad/models/map_banded.py``; this module is the flag's contract with the trainer, in the
shape ``map_precision.py`` set: ``pin_args`` (refusals at argv time, a no-op when OFF), ``dvb_check`` (G-DVB ``built`` lever: the
argv's declaration equals what the BUILT model holds), ``register_dvb_levers`` (called by the trainer at import: the registry is
pinned two-way against its parser), ``stamp`` (the ``config.json["map_hires"]["banded"]`` block, absent when OFF).

Retired under the flag (2.1): ``--w-map-fp`` -- it is one-sided, not lift-masked, and 21.5 % of its corridor rows are camera-invisible
(D16); ``drv_sdf`` + ``sdw`` + ``unobs`` replace it. The two are REFUSED together.
"""
from __future__ import annotations

FLAG = "--r9-map-banded"
DEST = "r9_map_banded"
#: set on the model ONLY when the flag is on (an absent attribute is the tip's model)
MODEL_ATTR = "_r9_map_banded"
#: the near lift the near band reuses, narrowed 20 -> 10 m (2.1)
NEAR_LIFT_M = 10.0


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: nothing is checked."""
    if not is_on(args):
        return
    bad = []
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the banded head is a decoder of the 10 cm branch's 0.25 m encoder")
    if not float(getattr(args, "w_map_hires", 0.0) or 0.0) > 0.0:
        bad.append("--w-map-hires is 0: the banded map term (weight = --w-map-hires) would not exist")
    nl = getattr(args, "map_hires_near_lift_m", None)
    if nl is None or abs(float(nl) - NEAR_LIFT_M) > 1e-9:
        bad.append(f"--map-hires-near-lift-m is {nl!r}, not {NEAR_LIFT_M:g}: the near band (0-10 m @ 0.10 m) reuses the near "
                   f"lift narrowed 20 -> 10 m (a longer lift would cover rows the near band does not have)")
    if float(getattr(args, "w_map_fp", 0.0) or 0.0) != 0.0:
        bad.append("--w-map-fp is RETIRED under --r9-map-banded (REFCV9_DESIGN 2.1: one-sided, not lift-masked; drv_sdf + sdw + "
                   "unobs replace it)")
    if getattr(args, "map_lift_valid_mask", True) is False:
        bad.append("--no-map-lift-valid-mask: `unobs` IS the lift's validity mask; the banded losses always ignore it")
    if getattr(args, "map_hires_class_thresholds", None):
        bad.append("--map-hires-class-thresholds: the 8-way class-threshold rule reads logits the banded head does not emit "
                   "(per-channel, per-band thresholds are fitted on a FIT split by the eval harness)")
    if bool(getattr(args, "r8_critic_drivable", False)) or float(getattr(args, "w_r8_drivable", 0.0) or 0.0) > 0.0:
        bad.append("--r8-critic-drivable / --w-r8-drivable: the refcv8 drivable critic reads the 8-way logits; wiring it to the "
                   "banded channels is C8 (W-MAP-1), not C1")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        try:
            mods = _mof.trainable_modules(args)
        except SystemExit:
            mods = None                                        # a malformed --r9-train-only is refused by its own pin
        if mods is not None and not any(str(m).startswith("_map_hires.banded") or m == "_map_hires" for m in mods):
            bad.append(f"the declared freeze trains {list(mods)}: the NEW banded head would stay at its random init")
    if bad:
        raise SystemExit(f"[map-banded] {FLAG} refused:\n  - " + "\n  - ".join(bad))


def stamp(branch) -> dict:
    """The ``config.json["map_hires"]["banded"]`` block (read OFF the built branch)."""
    from tanitad.models import map_banded as MB
    h = branch.banded
    return {"flag": FLAG, "channels": list(MB.CHANNELS), "bands": [
                {"name": g.name, "x_m": [g.x0, g.x1], "cell_m": g.cell, "shape": [g.rows, g.cols], "weight": g.weight}
                for g in h.geom],
            "n_cells": sum(g.n_cells for g in h.geom), "band_weights": list(MB.BAND_WEIGHTS),
            "loss": "sigmoid focal (gamma 2) + soft Dice on drv/sdw/xwalk/arrow/hatch and on the dilated lane/edge_h; L1 on "
                    "drv_sdf and the lane offsets within 2 m; channels summed, per-band means, band weights 1/1/0.5; "
                    "max-entropy 1e-3 on unobs cells",
            "unobs": "the lift's own validity (~lift_valid), geometric, not learned; losses ignore it",
            "thin_width_m": 0.5, "sdf_clip_m": MB.SDF_CLIP_M, "maxent_w": MB.MAXENT_W,
            "refine_cls_built": getattr(branch.refine, "cls", None) is not None,
            "class_weights": "built and stamped but UNUSED by the banded loss (the 8-way CE is not computed)"}


def dvb_check(model, args) -> list:
    """G-DVB ``built`` lever: declared <=> the BUILT branch holds the banded head, the 8-way readout is absent, the near lift is the
    10 m one, and the loss's model attribute agrees."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = is_on(args)
    br = getattr(model, "_map_hires", None)
    built = br is not None and getattr(br, "banded", None) is not None
    attr = bool(getattr(model, MODEL_ATTR, False))
    out = []
    if declared != built:
        out.append(Mismatch(DEST, declared, built, "model._map_hires.banded", "argv and the build disagree"))
    if declared != attr:
        out.append(Mismatch(DEST, declared, attr, f"model.{MODEL_ATTR}", "argv and the attribute the LOSS reads disagree"))
    if built:
        if getattr(br.refine, "cls", None) is not None:
            out.append(Mismatch(DEST, "no 8-way readout", "refine.cls built", "model._map_hires.refine.cls",
                                "an untrained readout would sit in the optimiser"))
        if br.near is None or int(br.cfg.near_rows) != int(round(NEAR_LIFT_M / 0.1)):
            out.append(Mismatch(DEST, f"near lift {NEAR_LIFT_M:g} m", getattr(br.cfg, "near_lift_x_m", None),
                                "model._map_hires.cfg.near_lift_x_m", "the near band reuses the 10 m near lift"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
