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
#: the near lift the near band reuses, narrowed 20 -> 10 m (2.1) -- the DEFAULT (``--r9-map-banded-near-m`` unset); M1N20 widens it to 20
NEAR_LIFT_M = 10.0
#: E1-NEAR20 (PREREG_M1_NEAR20): where the near band (10 cm cells) ENDS. Unset = 10 m (M1 / M2: byte-identical to the tip); 20 = arm M1N20
#: (near 0-20 m = 200 x 600, C0's near lift). The pre-registered set is ``map_banded.NEAR_M_ADMITTED``; the near lift must EQUAL it.
NEAR_M_FLAG = "--r9-map-banded-near-m"
NEAR_M_DEST = "r9_map_banded_near_m"
NEAR_M_DEFAULT = 10.0
NEAR_M_ADMITTED = (10.0, 20.0)


def near_m(args) -> float:
    """The near band's end the argv DECLARES: ``--r9-map-banded-near-m`` when passed, else the default 10 m."""
    v = getattr(args, NEAR_M_DEST, None)
    return NEAR_M_DEFAULT if v is None else float(v)


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: nothing is checked -- except that ``--r9-map-banded-near-m`` WITHOUT ``--r9-map-banded`` is
    refused (a declared lever that nothing builds is a launch blocker, never a footnote)."""
    if not is_on(args):
        if getattr(args, NEAR_M_DEST, None) is not None:
            raise SystemExit(f"[map-banded] {NEAR_M_FLAG} {getattr(args, NEAR_M_DEST)!r} refused: it configures the banded head and "
                             f"{FLAG} is not passed (the lever would be declared and never built)")
        return
    bad = []
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the banded head is a decoder of the 10 cm branch's 0.25 m encoder")
    if not float(getattr(args, "w_map_hires", 0.0) or 0.0) > 0.0:
        bad.append("--w-map-hires is 0: the banded map term (weight = --w-map-hires) would not exist")
    nm = near_m(args)
    if not any(abs(nm - a) < 1e-9 for a in NEAR_M_ADMITTED):
        bad.append(f"{NEAR_M_FLAG} is {nm:g}: the pre-registered near-band ends are {list(NEAR_M_ADMITTED)} m (PREREG_M1_NEAR20)")
    nl = getattr(args, "map_hires_near_lift_m", None)
    if nl is None or abs(float(nl) - nm) > 1e-9:
        bad.append(f"--map-hires-near-lift-m is {nl!r}, not {nm:g}: the near band (0-{nm:g} m @ 0.10 m) reuses the near "
                   f"lift of exactly its own length (a longer lift would cover rows the near band does not have, a shorter one "
                   f"leaves near-band rows without it)")
    if float(getattr(args, "w_map_fp", 0.0) or 0.0) != 0.0:
        bad.append("--w-map-fp is RETIRED under --r9-map-banded (REFCV9_DESIGN 2.1: one-sided, not lift-masked; drv_sdf + sdw + "
                   "unobs replace it)")
    if getattr(args, "map_lift_valid_mask", True) is False:
        bad.append("--no-map-lift-valid-mask: `unobs` IS the lift's validity mask; the banded losses always ignore it")
    if getattr(args, "map_hires_class_thresholds", None):
        bad.append("--map-hires-class-thresholds: the 8-way class-threshold rule reads logits the banded head does not emit "
                   "(per-channel, per-band thresholds are fitted on a FIT split by the eval harness)")
    # refcv9 C8 (W-MAP-1): `--r8-critic-drivable` is LEGAL here -- the critic then reads the banded channels (P(drv), drv_sdf; `unobs` and
    # out-of-extent corners = no evidence), see tanitad/models/plan_wires_r9.py and refc_v3.py's E9 block.
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
            # E1-NEAR20: absent at the default 10 m, so an M1 / M2 stamp is byte-identical to the tip's
            **({"near_m": float(h.near_m)} if abs(float(h.near_m) - NEAR_M_DEFAULT) > 1e-9 else {}),
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
    drv = getattr(model, "r8_drv", None)               # refcv9 C8 W-MAP-1: the critic reads the banded channels iff both flags are declared
    if drv is not None and (int(drv.n_feats) == 5) != declared:
        out.append(Mismatch(DEST, "5-feature banded critic" if declared else "3-feature 8-way critic", f"{int(drv.n_feats)} features",
                            "model.r8_drv.n_feats", "the critic's input and the declared map head disagree"))
    if built:
        if getattr(br.refine, "cls", None) is not None:
            out.append(Mismatch(DEST, "no 8-way readout", "refine.cls built", "model._map_hires.refine.cls",
                                "an untrained readout would sit in the optimiser"))
        nm = near_m(args)                          # E1-NEAR20: BUILT near rows == the DECLARED near end / 0.1 m (10 m: 100 rows, 20 m: 200)
        if br.near is None or int(br.cfg.near_rows) != int(round(nm / 0.1)):
            out.append(Mismatch(DEST, f"near lift {nm:g} m", getattr(br.cfg, "near_lift_x_m", None),
                                "model._map_hires.cfg.near_lift_x_m", "the near band reuses the near lift of exactly its own length"))
    return out


def dvb_near_m(model, args) -> list:
    """G-DVB ``built`` lever ``r9_map_banded_near_m``: the declared near-band end (default 10 m when unset) <=> the BUILT head's near band --
    its first band's rows / end, the branch config's ``banded_near_m`` and the near lift's rows. A model with no banded head built is clean
    only when the flag is unset (the banded lever's own entry reports a missing head)."""
    from tanitad.train.declared_vs_built import Mismatch
    nm = near_m(args)
    br = getattr(model, "_map_hires", None)
    h = None if br is None else getattr(br, "banded", None)
    if h is None:
        return [] if getattr(args, NEAR_M_DEST, None) is None else [
            Mismatch(NEAR_M_DEST, nm, "no banded head", "model._map_hires.banded", "the near-band end is declared and no banded head is built")]
    g0 = h.geom[0]
    out = []
    if int(g0.rows) != int(round(nm / 0.1)) or abs(float(g0.x1) - nm) > 1e-9:
        out.append(Mismatch(NEAR_M_DEST, f"{int(round(nm / 0.1))} rows to {nm:g} m", f"{int(g0.rows)} rows to {float(g0.x1):g} m",
                            "model._map_hires.banded.geom[0]", "the built near band is not the declared one"))
    if int(br.cfg.near_rows) != int(round(nm / 0.1)):
        out.append(Mismatch(NEAR_M_DEST, int(round(nm / 0.1)), int(br.cfg.near_rows), "model._map_hires.cfg.near_rows",
                            "the near lift's rows are not the declared near band's"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
    register(NEAR_M_DEST, "built", dvb_near_m)               # E1-NEAR20: its own G-DVB entry (a flag with none is refused by the gate)
