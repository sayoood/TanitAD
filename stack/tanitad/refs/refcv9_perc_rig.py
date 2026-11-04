"""refcv9 C7 -- the E1 PERCEPTION-ONLY RIG (REFCV9_DESIGN 4.3): init from refcv8 EMA-48,852, the NEWEST window row only
(K = 3 stacked frames), the planner / tactical modules NOT BUILT, losses = the perception terms only (box3d, agent,
map_hires; weights exactly as the argv sets them).

Opt-in: ``refc_v3_train.py --r9-perc-rig``. OFF = nothing here is imported on the default path's hot loop and the build /
forward / loss are the tip's (pinned by tests/test_r9_perc_rig.py and the flags-absent digest).

* ``RIG_SKIPPED_MODULES`` is the NAMED list of the checkpoint's modules the rig does not build (read off the real
  refcv8 EMA-48,852 state_dict, 1,165 keys, MEASURED 2026-11-02); ``warm_start_perc_rig`` loads WITH THAT LIST and
  refuses any other missing / unexpected key -- never a blanket ``strict=False``.
* ``assert_rig_built`` is the check a rig that silently builds the planner fails.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: modules (state_dict key roots) the rig BUILDS -- trunk, the learned agent head, the perception branch (BEV lift,
#: map head, 3-D box head) and the 10 cm map branch.
RIG_BUILT_ROOTS = ("core.encoder", "core.agent_head", "_perception", "_map_hires")

#: modules the rig does NOT build: every other root of the refcv8 EMA-48,852 state_dict (27 roots).
RIG_SKIPPED_MODULES = (
    "goal_gate", "core.lat_log_prior", "core.lon_log_prior", "core.ego_hist", "core.strategic", "core.measurement",
    "core.decoder", "core.agent_embed", "core.tactical_trunk", "core.lat_head", "core.lon_head", "core.route_head",
    "core.law_head", "phi_tac", "str_goal_head", "gstr_embed", "gstr_film", "nav_inj", "nav_to_tac", "nav_to_str",
    "lat_head_tac", "lon_head_tac", "tac_decoder_v6", "tac_behaviour_gate_v6", "tac_goal_head", "tac_latent_proj",
    "scorer")

#: the planner subset an assertion of "the planner is not built" must find ABSENT as an attribute path.
PLANNER_ATTR_PATHS = ("core.decoder", "core.agent_embed", "core.strategic", "core.measurement", "core.law_head",
                      "tac_decoder_v6", "phi_tac", "core.tactical_trunk")

#: argv dests the rig REFUSES (each would be built / read only by the planner or tactical modules: silent drop).
PLANNER_FLAG_DESTS = (
    "tac_decoder_v6", "graft_behaviour_sel", "max_speed_input_v6", "graft_tac8_prior", "graft_nav_compliance",
    "speed_ceiling_filter", "anchors", "anchor_v0_conditioned", "bev_coupling", "ego_history", "refcv8",
    "r8_alloc_emit", "r8_no_rc", "f1_random_t", "f2_dd_step", "f3_per_layer", "f4_adaln", "f5_focal",
    "f5_emitting_conf", "f6_w_u0_zero", "nav_from_v7", "r8_nav_from_v9")
PLANNER_WEIGHT_DESTS = ("w_tac_v6", "w_r8_alloc_l1", "w_r8_cons", "w_r8_v9_cons", "w_r8_sat", "w_r8_subscore",
                        "w_tac_goal", "w_u0", "w_bev_aux")


def is_skipped_key(k: str) -> bool:
    return any(k == m or k.startswith(m + ".") for m in RIG_SKIPPED_MODULES)


def is_built_key(k: str) -> bool:
    return any(k == m or k.startswith(m + ".") for m in RIG_BUILT_ROOTS)


def _has_path(obj, path: str) -> bool:
    for part in path.split("."):
        if part not in getattr(obj, "_modules", {}) and part not in getattr(obj, "_parameters", {}) \
                and part not in getattr(obj, "_buffers", {}):
            return False
        obj = getattr(obj, part)
    return True


def pin_args(args) -> None:
    """Refuse every planner / tactical lever the rig cannot honour (a silent drop reads as "the lever did nothing")."""
    if not bool(getattr(args, "r9_perc_rig", False)):
        return
    bad = [d for d in PLANNER_FLAG_DESTS if bool(getattr(args, d, False))]
    bad += [d for d in PLANNER_WEIGHT_DESTS if float(getattr(args, d, 0.0) or 0.0) > 0.0]
    if str(getattr(args, "residual_prior", "off") or "off") not in ("off", "none"):
        bad.append("residual_prior")
    if bad:
        raise SystemExit("[r9-perc-rig] refused: planner / tactical flag(s) %s -- the rig builds no planner, so they "
                         "would be SILENTLY DROPPED." % sorted(bad))
    if float(getattr(args, "w_box3d", 0.0) or 0.0) <= 0.0 and float(getattr(args, "w_agent", 0.0) or 0.0) <= 0.0 \
            and str(getattr(args, "map_hires", "off")) != "on":
        raise SystemExit("[r9-perc-rig] refused: no perception term (w_box3d / w_agent / --map-hires on) -- nothing to train")


def assert_rig_built(model) -> dict:
    """The built model holds ONLY RIG_BUILT_ROOTS: no planner / tactical module as an attribute, no state_dict key outside
    the built roots. Returns the record config.json carries."""
    present = [p for p in PLANNER_ATTR_PATHS if _has_path(model, p)]
    if present:
        raise SystemExit("[r9-perc-rig] the planner / tactical module(s) %s were BUILT -- the rig must not build them" % present)
    keys = list(model.state_dict().keys())
    stray = [k for k in keys if not is_built_key(k)]
    if stray:
        raise SystemExit("[r9-perc-rig] %d state_dict key(s) outside the built roots, e.g. %s" % (len(stray), stray[:3]))
    n_params = sum(p.numel() for p in model.parameters())
    return {"built_roots": list(RIG_BUILT_ROOTS), "skipped_modules": list(RIG_SKIPPED_MODULES), "n_keys": len(keys),
            "n_params": int(n_params)}


def banded_exemptions(model) -> tuple:
    """(new key prefixes the source cannot carry, source key prefixes with no home) -- ``((), ())`` unless the model was built with
    ``--r9-map-banded`` (refcv9 C1), so the OFF warm start is the tip's, check for check."""
    br = getattr(model, "_map_hires", None)
    if br is None or getattr(br, "banded", None) is None:
        return (), ()
    return ("_map_hires.banded.",), ("_map_hires.refine.cls.",)


def b3_exemptions(model) -> tuple:
    """New key prefixes the source cannot carry for refcv9 B3 ``--r9-box-dn`` (``box_b3_flag.attach`` builds the DN query embeddings
    ``dn_embed`` / ``dn_cls_embed`` BEFORE the warm start, and refcv8 has no DN): named by their module path in THIS model, ``()``
    unless the model carries ``_r9_box_b3`` with ``dn`` on, so the OFF warm start is the tip's, check for check."""
    cfg = getattr(model, "_r9_box_b3", None)
    if cfg is None or not getattr(cfg, "dn", False):
        return ()
    names = tuple(sorted(n + "." for n, _ in model.named_modules() if n.rsplit(".", 1)[-1] in ("dn_embed", "dn_cls_embed")))
    if not names:
        raise SystemExit("[r9-perc-rig] --r9-box-dn is on but the model has no dn_embed / dn_cls_embed module")
    return names


def own_enc_exemptions(model) -> tuple:
    """New key prefixes the source cannot carry for refcv9 C2 ``--r9-map-own-enc`` (M3): the map's own encoder ``_map_hires.map_encoder.``
    and the re-sized banded readout input ``_map_hires.refine.inp.`` (the source's reads the 64-channel encoder the map no longer uses).
    NAMED; ``()`` unless the model was built with the own encoder, so the OFF warm start is the tip's, check for check. The WIDENED keys
    (lift rows, the 64-ch encoder's stem) are NOT exempt: ``map_own_enc_flag.apply_warm_start`` zero-extends them explicitly."""
    if getattr(getattr(model, "_map_hires", None), "map_encoder", None) is None:
        return ()
    from tanitad.train.map_own_enc_flag import NEW_PREFIXES
    return tuple(NEW_PREFIXES)


def box_depth_exemptions(model) -> tuple:
    """New key prefixes the source cannot carry for refcv9 C3 ``--r9-box-depth-aux`` (arm B1): the training-only depth head
    ``_perception.box_depth_aux.`` (``box_depth_aux_flag.attach`` builds it BEFORE the warm start; refcv8 has no such head). NAMED by its
    module path in THIS model; ``()`` unless the perception branch carries the head, so the OFF warm start is the tip's, check for check.
    An unnamed missing key still refuses (``warm_start_perc_rig`` raises on any missing key outside the named prefixes)."""
    br = getattr(model, "_perception", None)
    if br is None or getattr(br, "box_depth_aux", None) is None:
        return ()
    from tanitad.train.box_depth_aux_flag import KEY_PREFIX
    return (KEY_PREFIX,)


def persp_exemptions(model) -> tuple:
    """New key prefixes for refcv9 C4 (arm B2 ``--r9-box-persp-aux``, control B2c ``--r9-box-bev2-aux``): the training-only head
    ``_perception.box_persp_aux.`` or ``_perception.box_bev2_aux.`` (``persp3d_aux_flag.attach`` builds it BEFORE the warm start; refcv8 has
    neither). NAMED by its module path in THIS model; ``()`` unless the perception branch carries one, so the OFF warm start is the tip's,
    check for check. An unnamed missing key still refuses."""
    br = getattr(model, "_perception", None)
    if br is None:
        return ()
    from tanitad.train.persp3d_aux_flag import HEAD_B, HEAD_P, KEY_PREFIX_B, KEY_PREFIX_P
    return tuple(pfx for attr, pfx in ((HEAD_P, KEY_PREFIX_P), (HEAD_B, KEY_PREFIX_B)) if getattr(br, attr, None) is not None)


def bev_warp_exemptions(model) -> tuple:
    """New key prefixes the source cannot carry for refcv9 T1 ``--r9-bev-warp``: the zero-init fuse ``_map_hires.bev_warp.`` (``bev_warp_flag.attach`` builds
    it BEFORE the warm start; refcv8 has no such module). NAMED by its module path in THIS model; ``()`` unless the 10 cm branch carries the fuse, so the
    OFF warm start is the tip's, check for check. An unnamed missing key still refuses."""
    from tanitad.train.bev_warp_flag import warm_start_prefixes
    return warm_start_prefixes(model)


def plan_wires_exemptions(model) -> tuple:
    """New keys the source cannot carry for refcv9 C8 (``plan_wires_flag.attach`` builds them BEFORE the warm start; refcv8 has none):
    W-MAP-2 ``--r9-plan-map-channels`` -> the zero-init 1x1 ``map_chan_proj`` on the planner BEV pool; W-BOX ``--r9-plan-box-tokens`` -> the box
    tokens ``box_tok_r9`` and the zero-init ``box_gate`` of every consuming planner / tactical layer (each gate NAMED by its full key).
    FULL MODEL ONLY: the rig builds no planner (``pin_args`` refuses both flags there), so on the rig, and with both OFF, this is ``()`` and the
    warm start is the tip's, check for check. An unnamed missing key still refuses."""
    pool = getattr(getattr(model, "_perception", None), "bev_pool", None)
    core = getattr(model, "core", None)
    mods = [m for m in (getattr(pool, "map_chan_proj", None), getattr(core, "box_tok_r9", None)) if m is not None]
    if not mods:
        return ()
    names = sorted(n + "." for n, m in model.named_modules() if any(m is q for q in mods))
    if len(names) != len(mods):
        raise SystemExit("[r9-warm] a C8 plan-wire module is built but is not a registered submodule of the model")
    if getattr(core, "box_tok_r9", None) is not None:
        gates = sorted(n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1] == "box_gate")
        if not gates:
            raise SystemExit("[r9-warm] --r9-plan-box-tokens is on but the model has no box_gate parameter")
        names += gates
    return tuple(names)


def refcv9_warm_exemptions(model) -> tuple:
    """THE one list of what a refcv9 build may carry that the refcv8 source cannot -- used by BOTH ``--init-from`` warm starts
    (``warm_start_perc_rig`` here and ``refcv8_train.warm_start_from`` for the full model), so the two can never disagree about a lever.

    -> ``(new_pfx, drop_pfx, widen)``:
      * ``new_pfx``  -- key prefixes a missing key may sit under (each NAMED by its module path in THIS model, by declared flag: banded head,
        B3 DN embeddings, C2 own encoder, B1 / B2 / B2c aux heads, T1 fuse, and -- full model only -- the C8 plan wires);
      * ``drop_pfx`` -- source key prefixes with no home in this build (the banded head replaces the 8-way ``_map_hires.refine.cls.*``);
      * ``widen``    -- ``None``, or the own-encoder warm-start rule (``map_own_enc_flag.apply_warm_start``: zero-extends the widened lift / stem keys).
    Every lever OFF => ``((), (), None)``: the OFF warm start is the tip's, check for check. Nothing here is a blanket ``strict=False``."""
    new_pfx, drop_pfx = banded_exemptions(model)
    new_pfx = tuple(new_pfx) + b3_exemptions(model) + own_enc_exemptions(model) + box_depth_exemptions(model) + persp_exemptions(model) + bev_warp_exemptions(model)   # B3 DN embeddings; C2 own encoder; C3 depth head; T1 fuse (NAMED; OFF: ())
    new_pfx += plan_wires_exemptions(model)                # C8 plan wires (full model only; the rig and OFF: ())
    widen = None
    if own_enc_exemptions(model):                          # refcv9 C2 (--r9-map-own-enc); OFF: skipped
        from tanitad.train.map_own_enc_flag import apply_warm_start as widen
    return new_pfx, tuple(drop_pfx), widen


def apply_refcv9_exemptions(model, loadable: dict, own: dict = None) -> tuple:
    """Apply ``refcv9_warm_exemptions(model)`` to the dict of source tensors a warm start is about to load. MUTATES ``loadable``: the declared
    source keys with no home are removed (only those the build does not have), the own encoder's widening is applied. -> ``(new_pfx, dropped,
    own_enc_rec)``. ``own`` is ``model.state_dict()`` (passed by a caller that already holds it)."""
    new_pfx, drop_pfx, widen = refcv9_warm_exemptions(model)
    if own is None:
        own = model.state_dict()
    dropped = sorted(k for k in loadable if k not in own and any(k.startswith(q) for q in drop_pfx))
    for k in dropped:
        del loadable[k]
    own_enc_rec = {}
    if widen is not None:
        own_enc_rec = widen(model, loadable)               # zero-extends the widened lift / stem keys (the pinned rule); drops the source's refine.inp.*
    return new_pfx, dropped, own_enc_rec


def warm_start_perc_rig(model, path: str) -> dict:
    """``--init-from`` for the rig: load the refcv8 EMA into the perception modules with the NAMED skip list. Refuses (a)
    a source key that is neither loadable nor in RIG_SKIPPED_MODULES, (b) a rig key the source lacks, (c) a shape mismatch."""
    import torch
    p = Path(path)
    if not p.is_file():
        raise SystemExit("[r9-perc-rig] --init-from %s: no such file" % path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else None
    if not isinstance(sd, dict) or not sd:
        raise SystemExit("[r9-perc-rig] --init-from %s: no model state_dict in the file" % path)
    own = model.state_dict()
    skipped = {k: v for k, v in sd.items() if is_skipped_key(k)}
    loadable = {k: v for k, v in sd.items() if not is_skipped_key(k)}
    # refcv9 C1 (--r9-map-banded): the banded head's keys are NEW (the source cannot have them) and the 8-way readout
    # `_map_hires.refine.cls.*` is NOT BUILT (the source's copy has no home). Both are NAMED, never a blanket strict=False; OFF: ().
    # (the list + the drop + the own-encoder widening are `refcv9_warm_exemptions` / `apply_refcv9_exemptions`: ONE list, shared with the full model's warm start)
    new_pfx, dropped, own_enc_rec = apply_refcv9_exemptions(model, loadable, own)
    unexpected = sorted(k for k in loadable if k not in own)
    missing = sorted(k for k in own if k not in loadable and not any(k.startswith(q) for q in new_pfx))
    if unexpected or missing:
        raise SystemExit("[r9-perc-rig] --init-from %s: %d source key(s) neither built nor in RIG_SKIPPED_MODULES (e.g. %s); "
                         "%d rig key(s) the source lacks (e.g. %s)" % (path, len(unexpected), unexpected[:3], len(missing), missing[:3]))
    bad = [k for k in loadable if tuple(loadable[k].shape) != tuple(own[k].shape)]
    if bad:
        raise SystemExit("[r9-perc-rig] --init-from %s: shape mismatch on %s" % (path, bad[:3]))
    model.load_state_dict(loadable, strict=not new_pfx)
    if new_pfx:        # strict=False only because of the NAMED new prefixes: prove nothing else is missing
        miss = sorted(k for k in own if k not in loadable)
        if any(not any(k.startswith(q) for q in new_pfx) for k in miss):
            raise SystemExit("[r9-perc-rig] --init-from: keys missing beyond the NAMED new modules (banded head, B3 DN): %s" % miss[:3])
    per_mod = {m: sum(1 for k in skipped if k == m or k.startswith(m + ".")) for m in RIG_SKIPPED_MODULES}
    return {"path": str(p), "md5": h.hexdigest(), "source_step": (int(state["step"]) if isinstance(state, dict) and "step" in state else None),
            "n_loaded": len(loadable), "n_new_refcv8_keys": 0, "n_skipped_source_keys": len(skipped),
            **({"banded": {"new_prefixes": list(new_pfx), "n_new_keys": sum(1 for k in own if any(k.startswith(q) for q in new_pfx)),
                           "dropped_source_keys": dropped}} if new_pfx else {}),
            **({"own_enc": own_enc_rec} if own_enc_rec else {}),
            "skipped_modules_key_counts": per_mod, "optimizer": "fresh (not loaded)"}


def assert_seams_built_perc_rig(model, stamp: dict) -> None:
    """The rig's subset of ``refc_v3_train.assert_seams_are_built``: the record may only claim what the BUILT rig holds.
    The full check reads ``core.decoder`` / ``core.agent_embed`` / ``core.ego_hist``, none of which the rig builds, so it
    cannot run here; this one asserts the seams the rig DOES have (trunk, learned agent head, bottom-row equalisation) and
    that the stamp states no planner seam. Bidirectional, like the original."""
    core = getattr(model, "core", model)
    bad = []
    enc = getattr(core, "encoder", None)
    is_timm = type(enc).__name__ == "TimmResNetTrunk"
    if str(stamp.get("trunk", "refc")) == "timm" and not is_timm:
        bad.append("stamp says trunk='timm' but core.encoder is a %s" % type(enc).__name__)
    if str(stamp.get("trunk", "refc")) != "timm" and is_timm:
        bad.append("stamp says trunk='refc' but core.encoder IS the timm trunk")
    has_head = getattr(core, "agent_head", None) is not None
    if stamp.get("agents") is not None and not has_head:
        bad.append("stamp carries an `agents` block but core.agent_head is None")
    if stamp.get("agents") is None and has_head:
        bad.append("stamp says agents=None but core.agent_head WAS BUILT")
    if "trunk_equalize_bottom_rows" in stamp and is_timm:
        built = int(getattr(getattr(enc, "cfg", None), "equalize_bottom_rows", -1))
        if int(stamp["trunk_equalize_bottom_rows"]) != built:
            bad.append("stamp says trunk_equalize_bottom_rows=%s but the BUILT trunk zeroes %d rows"
                       % (stamp["trunk_equalize_bottom_rows"], built))
    if str(stamp.get("sampler", "none")) != "none" or float(stamp.get("w_u0", 0.0) or 0.0) > 0.0 or bool(stamp.get("cross_agent", False)):
        bad.append("stamp names a planner seam (sampler / w_u0 / cross_agent) the rig cannot hold")
    if bad:
        raise SystemExit("[r9-perc-rig] THE RUN RECORD DOES NOT MATCH THE MODEL: " + "; ".join(bad))


def dvb_check(model, args) -> list:
    """G-DVB `built` check for ``--r9-perc-rig``: declared on <=> the model holds only RIG_BUILT_ROOTS (and conversely OFF =>
    the planner IS built). Returns Mismatch rows."""
    from tanitad.train.declared_vs_built import Mismatch
    declared = bool(getattr(args, "r9_perc_rig", False))
    built = bool(getattr(model, "_perc_rig", False))
    out = []
    if declared != built:
        out.append(Mismatch("r9_perc_rig", declared, built, "model._perc_rig", "argv and the build disagree"))
    if declared:
        try:
            assert_rig_built(model)
        except SystemExit as e:
            out.append(Mismatch("r9_perc_rig", "planner / tactical NOT built", "built", "model.state_dict()", str(e)))
    return out


def register_dvb_levers(register) -> None:
    register("r9_perc_rig", "built", dvb_check)
