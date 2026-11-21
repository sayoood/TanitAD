"""``--r9-map-own-enc`` -- the map's OWN 128-channel BEV encoder + the shared lift widened 64 -> 128 (refcv9 C2, arm M3; REFCV9_DESIGN 2.2
"Map BEV capacity (M3)", 4.3 row M3, 4.5). Argv refusals, G-DVB, the stamp and the PINNED warm-start rule.

The modules live in ``tanitad/models/map_head_hires.py`` (``MapHiresConfig.own_enc`` / ``d_map_enc``, ``MapHiresBranch.map_encoder``,
``MapHiresBranch._build_own_enc``); this module is the flag's contract with the trainer, in the shape ``map_banded_flag.py`` set:
``pin_args`` (refusals at argv time, a no-op when OFF), ``dvb_check`` (G-DVB ``built`` lever: the argv's declaration equals what the BUILT
branch holds), ``register_dvb_levers`` (called by the trainer at import; the registry is pinned two-way against its parser), ``stamp``
(``config.json["map_hires"]["own_enc"]``, absent when OFF) and ``apply_warm_start`` (called through ``refcv9_perc_rig.apply_refcv9_exemptions`` by
BOTH warm starts: the rig's ``warm_start_perc_rig`` and the full model's ``refcv8_train.warm_start_from``).

WHAT IT BUILDS (ON; OFF builds nothing and is bit-identical to the tip):
  * the shared lift's output is widened ``d_lift`` (64) -> ``d_map_enc`` (128);
  * the EXISTING ``d_model`` (64) channel encoder keeps feeding the planner BEV / box tokens / agents / tactical decoder, reading the first
    ``d_lift`` lift channels through its stem's input channels ``0..d_lift`` (refcv8's weights) and ZERO on ``d_lift..d_map_enc``;
  * a NEW ``d_map_enc``-channel encoder ``map_encoder`` (fresh init, same recipe) reads the widened lift and feeds ONLY the banded map head
    (``refine.inp`` is rebuilt to read 128 channels).

THE WARM-START RULE (pinned, BRIEF_C2 item 2) -- step 0 equals refcv8 on the box / agent / planner path:
  ``_map_hires.lift.proj.{weight,bias}`` / ``.lift.unobserved``: the first ``d_lift`` output rows are the source's, the extra rows keep their
  fresh init; ``_map_hires.encoder.stem.0.weight``: input channels ``0..d_lift`` are the source's, ``d_lift..d_map_enc`` are ZERO;
  ``_map_hires.refine.inp.*`` and ``_map_hires.map_encoder.*`` are NOT loaded (the source's ``refine.inp`` reads the 64-channel features of
  an encoder the map no longer uses). Every other key loads as before; nothing is a blanket ``strict=False``.
"""
from __future__ import annotations

FLAG = "--r9-map-own-enc"
DEST = "r9_map_own_enc"
#: the own encoder's width AND the widened lift's output width (REFCV9_DESIGN 2.2)
D_MAP_ENC = 128
#: key prefixes the refcv8 source cannot carry (fresh under the flag): the own encoder and the re-sized banded readout input
NEW_PREFIXES = ("_map_hires.map_encoder.", "_map_hires.refine.inp.")
#: source key -> the axis on which the source fills ``[:d_lift]`` of the widened tensor
WIDENED = {"_map_hires.lift.proj.weight": 0, "_map_hires.lift.proj.bias": 0, "_map_hires.lift.unobserved": 0,
           "_map_hires.encoder.stem.0.weight": 1}
#: the stem's extra INPUT channels are ZERO (the existing encoder must not see the extra lift channels at step 0)
ZERO_EXTRA = ("_map_hires.encoder.stem.0.weight",)
#: modules a declared freeze must TRAIN under the flag (else the widened lift rows / the own encoder would stay at random init)
REQUIRED_TRAINABLE = ("_map_hires.lift", "_map_hires.encoder", "_map_hires.map_encoder")


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: nothing is checked."""
    if not is_on(args):
        return
    bad = []
    if not bool(getattr(args, "r9_map_banded", False)):
        bad.append("--r9-map-banded is absent: the own encoder feeds ONLY the banded map head (M3 = the M2 winner + this flag)")
    # (FULL-WARM: the former rig-only `--init-from` refusal is LIFTED -- the pinned widening below is applied by BOTH warm starts, through
    #  refcv9_perc_rig.apply_refcv9_exemptions: the rig's warm_start_perc_rig and the full model's refcv8_train.warm_start_from)
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        try:
            mods = _mof.trainable_modules(args)
        except SystemExit:
            mods = None                                        # a malformed --r9-train-only is refused by its own pin
        if mods is not None:
            def covered(req):
                return any(req == m or req.startswith(str(m) + ".") for m in mods)
            lacking = [r for r in REQUIRED_TRAINABLE if not covered(r)]
            if lacking:
                bad.append(f"the declared freeze trains {list(mods)}, not {lacking}: the widened lift rows / the NEW own encoder "
                           f"would stay at their random init")
    if bad:
        raise SystemExit(f"[map-own-enc] {FLAG} refused:\n  - " + "\n  - ".join(bad))


def stamp(branch) -> dict:
    """The ``config.json["map_hires"]["own_enc"]`` block (read OFF the built branch)."""
    pb = branch.param_breakdown()
    return {"flag": FLAG, "d_map_enc": int(branch.cfg.d_map_enc), "lift_width_before": int(branch.cfg.d_lift),
            "lift_width": int(branch.lift.d_out), "box_encoder_d_model": int(branch.cfg.d_model),
            "map_encoder_params": int(pb["map_encoder"]), "branch_params": pb,
            "feeds": "map_encoder -> refine.inp -> the banded map head ONLY; box / agents / planner BEV / tactical keep `encoder` "
                     "(d_model channels) through `map_hires_bev`",
            "warm_start": "lift rows 0..d_lift and encoder.stem input channels 0..d_lift from the source, the stem's extra input "
                          "channels ZERO; map_encoder and refine.inp fresh (not loaded)"}


def apply_warm_start(model, loadable: dict) -> dict:
    """Called (via ``refcv9_perc_rig.apply_refcv9_exemptions``) by BOTH warm starts on the dict of source tensors they are about to load. Mutates ``loadable``: each
    WIDENED key becomes a full-shape tensor (the model's own tensor with the source's slice written into ``[:d_lift]``, the stem's extra
    input channels zeroed), the source's ``refine.inp.*`` is removed. Returns the record ``{}`` when the model has no own encoder (OFF)."""
    br = getattr(model, "_map_hires", None)
    if br is None or getattr(br, "map_encoder", None) is None:
        return {}
    own = model.state_dict()
    d0 = int(br.cfg.d_lift)
    widened = []
    for k, ax in WIDENED.items():
        if k not in loadable:
            continue                       # the rig's missing-key check reports it (the key is built, not in a NEW prefix)
        src, dst = loadable[k], own[k].detach().clone()
        want = list(dst.shape)
        want[ax] = d0
        if list(src.shape) != want:
            raise SystemExit(f"[map-own-enc] {k}: the source's shape {list(src.shape)} is not the {want} the widening rule needs "
                             f"(the built tensor is {list(dst.shape)}; the source must be the un-widened refcv8 branch)")
        if k in ZERO_EXTRA:
            dst.zero_()
        dst.narrow(ax, 0, d0).copy_(src.to(dst.dtype))
        loadable[k] = dst
        widened.append(k)
    dropped = sorted(k for k in loadable if k.startswith("_map_hires.refine.inp."))
    for k in dropped:
        del loadable[k]
    return {"d_lift": d0, "d_map_enc": int(br.cfg.d_map_enc), "widened_keys": widened, "dropped_source_keys": dropped,
            "new_prefixes": list(NEW_PREFIXES)}


def dvb_check(model, args) -> list:
    """G-DVB ``built`` lever: declared <=> the BUILT branch holds the own encoder, the lift is ``d_map_enc`` wide, the existing encoder's
    stem reads it, the own encoder is ``d_map_enc -> d_map_enc`` and ``refine.inp`` reads its output."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = is_on(args)
    br = getattr(model, "_map_hires", None)
    me = None if br is None else getattr(br, "map_encoder", None)
    built = me is not None and not bool(getattr(getattr(br, "cfg", None), "own_enc_lift64", False))   # E1-M3N20L64: that arm's map_encoder is not this flag's
    out = []
    if declared != built:
        out.append(Mismatch(DEST, declared, built, "model._map_hires.map_encoder", "argv and the build disagree"))
    cfg_on = bool(getattr(getattr(br, "cfg", None), "own_enc", False)) if br is not None else False
    if declared != cfg_on:
        out.append(Mismatch(DEST, declared, cfg_on, "model._map_hires.cfg.own_enc", "argv and the declared branch config disagree"))
    if built:
        w = int(br.cfg.d_map_enc)
        for what, want, got, where in (
                ("lift output channels", w, int(br.lift.d_out), "model._map_hires.lift.d_out"),
                ("lift proj rows", w, int(br.lift.proj.out_channels), "model._map_hires.lift.proj.out_channels"),
                ("box encoder stem input", w, int(br.encoder.stem[0].in_channels), "model._map_hires.encoder.stem.0.in_channels"),
                ("box encoder width", int(br.cfg.d_model), int(br.encoder.stem[0].out_channels),
                 "model._map_hires.encoder.stem.0.out_channels"),
                ("own encoder stem input", w, int(me.stem[0].in_channels), "model._map_hires.map_encoder.stem.0.in_channels"),
                ("own encoder width", w, int(me.stem[0].out_channels), "model._map_hires.map_encoder.stem.0.out_channels"),
                ("banded readout input", w, int(br.refine.inp[0].in_channels), "model._map_hires.refine.inp.0.in_channels")):
            if want != got:
                out.append(Mismatch(DEST, f"{what} {want}", got, where, "the widened build is inconsistent"))
        if getattr(br, "banded", None) is None:
            out.append(Mismatch(DEST, "banded head", "absent", "model._map_hires.banded", "the own encoder feeds only the banded head"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
