"""``--r9-map-own-enc-lift64`` -- the map's OWN 64-channel BEV encoder on the UNWIDENED lift, warm-started as a COPY of the source's shared encoder
(refcv9 E1 arm M3N20L64; ``PREREG_M3N20L64.md`` md5 114a28de). Argv refusals, G-DVB, the stamp and the PINNED warm-start rule.

The module lives in ``tanitad/models/map_head_hires.py`` (``MapHiresConfig.own_enc_lift64``, ``MapHiresBranch.map_encoder`` built by
``_build_own_enc_lift64``); this module is the flag's contract with the trainer, in the shape ``map_own_enc_flag.py`` set (M3, the 128-channel
widened-lift arm). Unlike M3 nothing is widened: ``lift`` (64 channels), ``encoder`` and ``refine.inp`` keep the OFF shapes.

WHAT IT BUILDS (ON; OFF builds nothing and is bit-identical to the tip):
  * a NEW ``map_encoder`` (``HiresBEVEncoder``, 64 -> 64, the recipe of ``encoder``) reads the shared lift UNCHANGED and feeds ONLY the banded map head;
  * the shared ``encoder`` keeps feeding the planner BEV / box tokens / agents / tactical decoder, so the box and planner losses can no longer move
    the map's features.

THE WARM-START RULE (pinned, the point of the arm): ``_map_hires.map_encoder.<k>`` is loaded as a COPY of the source's ``_map_hires.encoder.<k>`` for
EVERY ``k`` the built ``map_encoder`` holds, through ``refcv9_perc_rig.apply_refcv9_exemptions`` by BOTH warm starts. No widening, no zero-extension,
no fresh map weights. A source without that key REFUSES (no ``strict=False`` hole: ``map_encoder`` is not a NEW prefix, so a missing key is an
undeclared missing key).
"""
from __future__ import annotations

FLAG = "--r9-map-own-enc-lift64"
DEST = "r9_map_own_enc_lift64"
#: the built encoder's key prefix and the source prefix it is a copy of
TARGET_PREFIX = "_map_hires.map_encoder."
SOURCE_PREFIX = "_map_hires.encoder."
#: modules a declared freeze must TRAIN under the flag (else the own encoder would stay at the copied weights, a frozen map path)
REQUIRED_TRAINABLE = ("_map_hires.map_encoder",)


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def pin_args(args) -> None:
    """The flag's refusals, at argv time. OFF: nothing is checked."""
    if not is_on(args):
        return
    bad = []
    if not bool(getattr(args, "r9_map_banded", False)):
        bad.append("--r9-map-banded is absent: the own encoder feeds ONLY the banded map head")
    if bool(getattr(args, "r9_map_own_enc", False)):
        bad.append("--r9-map-own-enc is also given: the 128-channel own encoder on the widened lift (M3) and this 64-channel one on the "
                   "unwidened lift (M3N20L64) are exclusive")
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
                bad.append(f"the declared freeze trains {list(mods)}, not {lacking}: the own encoder would stay at its copied weights")
    if bad:
        raise SystemExit(f"[map-own-enc-lift64] {FLAG} refused:\n  - " + "\n  - ".join(bad))


def stamp(branch) -> dict:
    """The ``config.json["map_hires"]["own_enc_lift64"]`` block (read OFF the built branch)."""
    pb = branch.param_breakdown()
    me = branch.map_encoder
    return {"flag": FLAG, "lift_width": int(branch.lift.d_out), "lift_width_unchanged": int(branch.cfg.d_lift),
            "map_encoder_in": int(me.stem[0].in_channels), "map_encoder_width": int(me.stem[0].out_channels),
            "box_encoder_d_model": int(branch.cfg.d_model), "map_encoder_params": int(pb["map_encoder"]), "branch_params": pb,
            "feeds": "map_encoder -> refine.inp -> the banded map head ONLY; box / agents / planner BEV / tactical keep `encoder` through `map_hires_bev`",
            "warm_start": "map_encoder.* = a COPY of the source's _map_hires.encoder.* (every key); lift / refine.inp load as today; nothing widened, "
                          "zero-extended or fresh"}


def apply_warm_start(model, loadable: dict) -> dict:
    """Called (via ``refcv9_perc_rig.apply_refcv9_exemptions``) by BOTH warm starts on the dict of source tensors they are about to load. Mutates
    ``loadable``: every built ``_map_hires.map_encoder.<k>`` gets a CLONE of the source's ``_map_hires.encoder.<k>``. REFUSES a source lacking one, a
    source that already carries ``map_encoder`` keys (it is not the un-split refcv8 branch) and a shape that differs. ``{}`` when the model has no
    such encoder (OFF)."""
    br = getattr(model, "_map_hires", None)
    if br is None or getattr(br, "map_encoder", None) is None or not bool(getattr(br.cfg, "own_enc_lift64", False)):
        return {}
    own = model.state_dict()
    want = sorted(k for k in own if k.startswith(TARGET_PREFIX))
    if not want:
        raise SystemExit(f"[map-own-enc-lift64] the built model holds no {TARGET_PREFIX}* key")
    stray = sorted(k for k in loadable if k.startswith(TARGET_PREFIX))
    if stray:
        raise SystemExit(f"[map-own-enc-lift64] the source already carries {len(stray)} {TARGET_PREFIX}* key(s), e.g. {stray[:2]}: it is not the "
                         f"un-split refcv8 branch whose shared encoder this arm copies")
    copied = []
    for k in want:
        src_k = SOURCE_PREFIX + k[len(TARGET_PREFIX):]
        if src_k not in loadable:
            raise SystemExit(f"[map-own-enc-lift64] the source has no {src_k}: the map encoder cannot be warm-started as its copy (nothing is "
                             f"left fresh)")
        if tuple(loadable[src_k].shape) != tuple(own[k].shape):
            raise SystemExit(f"[map-own-enc-lift64] {src_k} is {tuple(loadable[src_k].shape)}, the built {k} is {tuple(own[k].shape)}")
        loadable[k] = loadable[src_k].detach().clone()
        copied.append(k)
    return {"copied_keys": len(copied), "source_prefix": SOURCE_PREFIX, "target_prefix": TARGET_PREFIX, "first_key": copied[0]}


def dvb_check(model, args) -> list:
    """G-DVB ``built`` lever: declared <=> the BUILT branch holds the own encoder, the lift is NOT widened, the own encoder is ``d_lift -> d_model``
    (the shared encoder's recipe) and the shared ``encoder`` / ``refine.inp`` keep their shapes."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = is_on(args)
    br = getattr(model, "_map_hires", None)
    me = None if br is None else getattr(br, "map_encoder", None)
    cfg_on = bool(getattr(getattr(br, "cfg", None), "own_enc_lift64", False)) if br is not None else False
    built = me is not None and cfg_on
    out = []
    if declared != built:
        out.append(Mismatch(DEST, declared, built, "model._map_hires.map_encoder", "argv and the build disagree"))
    if declared != cfg_on:
        out.append(Mismatch(DEST, declared, cfg_on, "model._map_hires.cfg.own_enc_lift64", "argv and the declared branch config disagree"))
    if built:
        d0, dm = int(br.cfg.d_lift), int(br.cfg.d_model)
        for what, want, got, where in (
                ("lift output channels (unwidened)", d0, int(br.lift.d_out), "model._map_hires.lift.d_out"),
                ("lift proj rows (unwidened)", d0, int(br.lift.proj.out_channels), "model._map_hires.lift.proj.out_channels"),
                ("box encoder stem input", d0, int(br.encoder.stem[0].in_channels), "model._map_hires.encoder.stem.0.in_channels"),
                ("own encoder stem input", d0, int(me.stem[0].in_channels), "model._map_hires.map_encoder.stem.0.in_channels"),
                ("own encoder width", dm, int(me.stem[0].out_channels), "model._map_hires.map_encoder.stem.0.out_channels"),
                ("banded readout input", dm, int(br.refine.inp[0].in_channels), "model._map_hires.refine.inp.0.in_channels")):
            if want != got:
                out.append(Mismatch(DEST, f"{what} {want}", got, where, "the build is inconsistent"))
        if bool(br.cfg.own_enc):
            out.append(Mismatch(DEST, "own_enc off", "on", "model._map_hires.cfg.own_enc", "the 128-channel own encoder and this one are exclusive"))
        if getattr(br, "banded", None) is None:
            out.append(Mismatch(DEST, "banded head", "absent", "model._map_hires.banded", "the own encoder feeds only the banded head"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
