"""``--r9-box-layers`` -- refcv9 C5-REST, the box-head levers of arm B4 (REFCV9_DESIGN 2.4 rows *depth* / *size cap*, 4.2 item C5, 4.3 arm B4, D2).

B4 = "B2 winner + per-frame queries + 6 layers + s8 horizon tokens (needs D2)".  This module is the flags' contract with the trainer, in the
shape ``box_b3_flag.py`` set: ``pin_args`` (refusals at argv time), ``attach`` (build-time wiring on the BUILT perception branch), ``stamp``
(``config.json["r9_box_c5"]``, absent when every flag is OFF), the G-DVB ``built`` checks and ``register_dvb_levers``.

LEVERS IN THIS MODULE (bank order = the brief's priority order; each lever lands with its tests):
  --r9-box-layers N   (lever 2)  the box3d decoder's depth.  Default 0 = "not named" = today's depth (``TIP_DEPTH`` = 3), so every existing argv
                      builds and runs bit-identically.  B4 names 6.  The decoder is EXTENDED IN PLACE (``Box3DSlotDecoder.extend_depth``): the new
                      layers are deep copies of the template layer, which is exactly what ``nn.TransformerDecoder(layer, num_layers=N)`` builds, so
                      the result equals a from-scratch depth-N build and consumes no RNG (a test pins both).
  (lever 1, the D2 size cap, lives in ``box3d_head.BOX3D_PARAM_CAP`` -- it is a construction-time refusal, not a flag.)

REACHABILITY (CLAUDE.md "built, tested and unreachable from its caller"), one hop per line:
  argv ``--r9-box-layers`` (refc_v3_train.py build_parser, g8 group, default 0) -> ``args.r9_box_layers`` -> ``pin_args`` (``_pin_refcv8``) ->
  ``attach`` (train(), right after ``build_perception_branch``; and the eval loader) -> ``model._perception.box_dec.blocks.layers`` ->
  ``Box3DSlotDecoder.forward`` / ``box_b3.run_decoder`` / ``slot_query_select.anchored_forward`` (each loops over ``dec.blocks.layers``) ->
  ``dec.head`` -> the box3d loss (every layer's ``aux`` decode under ``--slot-deep-supervision``) -> ``.backward()``.
"""
from __future__ import annotations

FLAGS = {"layers": "--r9-box-layers"}
DESTS = {k: "r9_box_" + k for k in FLAGS}
#: the depth of the tip's box decoder (``Box3DSlotDecoder(depth=3)`` as ``PerceptionBranch`` builds it) -- what an unnamed flag means
TIP_DEPTH: int = 3
#: the key of the stamp in config.json
STAMP_KEY = "r9_box_c5"


def layers_named(args) -> int:
    """The depth the argv NAMES (0 = not named)."""
    return int(getattr(args, DESTS["layers"], 0) or 0)


def declared_depth(args) -> int:
    n = layers_named(args)
    return n if n > 0 else TIP_DEPTH


def is_any(args) -> bool:
    return layers_named(args) > 0


def pin_args(args) -> None:
    """The flags' refusals, at argv time.  All OFF: nothing is checked."""
    v = getattr(args, DESTS["layers"], 0)
    v = 0 if v is None else v
    if isinstance(v, bool) or not isinstance(v, int):
        raise SystemExit(f"[box-c5] {FLAGS['layers']} must be an integer, got {v!r}")
    if v < 0:
        raise SystemExit(f"[box-c5] {FLAGS['layers']} {v} < 0: a decoder depth is >= 1 (0 = not named = the tip's {TIP_DEPTH})")
    if not is_any(args):
        return
    bad = []
    if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
        bad.append("--w-box3d is 0: there is no box3d decoder for --r9-box-layers to deepen (a dead flag)")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        bad.append("a --r8-train-only-map / --r9-train-only freeze is declared: the NEW decoder layers would stay at their template init")
    if bad:
        raise SystemExit("[box-c5] --r9-box-layers refused:\n  - " + "\n  - ".join(bad))


def _box_dec(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, "box_dec", None)


def attach(model, args) -> None:
    """Build-time wiring, called by the trainer (and the eval loader) once the perception branch exists and BEFORE the optimizer is built and
    before any warm start.  OFF: touches nothing."""
    if not is_any(args):
        return
    dec = _box_dec(model)
    if dec is None:
        raise SystemExit("[box-c5] --r9-box-layers but the model has no box3d decoder (--w-box3d > 0 builds it)")
    dec.extend_depth(layers_named(args))             # raises ValueError above the D2 cap (BOX3D_PARAM_CAP)


def stamp(model, args) -> dict | None:
    """``config.json["r9_box_c5"]`` -- read OFF the built decoder; ``None`` (key absent) when every flag is OFF."""
    if not is_any(args):
        return None
    from tanitad.models.box3d_head import BOX3D_PARAM_CAP
    dec = _box_dec(model)
    out = {"flags": {FLAGS["layers"]: layers_named(args)}}
    out["layers"] = {"depth": None if dec is None else len(dec.blocks.layers), "tip_depth": TIP_DEPTH,
                     "new_layer_init": "deep copy of the template layer (= nn.TransformerDecoder(layer, num_layers=N)); consumes no RNG",
                     "box_decoder_params": None if dec is None else int(dec.n_params), "d2_cap": int(BOX3D_PARAM_CAP),
                     "d2_scope": "Box3DSlotDecoder (the decoder module); box_memory and the training-only aux heads are reported separately "
                                 "in perception.branch_params"}
    return out


# ---- G-DVB ("built" levers): the argv's declaration equals what the BUILT model holds --------------------------------------------- #
def _mm(dest, declared, built, where, why):
    from tanitad.train.declared_vs_built import Mismatch
    return Mismatch(dest, declared, built, where, why)


def dvb_check_layers(model, args) -> list:
    """declared <=> the decoder has that many layers.  Unnamed (0) means the tip's depth: a model with a box decoder must hold exactly
    ``TIP_DEPTH`` layers (a build that deepened the decoder without the argv naming it is a mismatch), and a model with none is fine."""
    dest = DESTS["layers"]
    want = declared_depth(args)
    dec = _box_dec(model)
    out = []
    if dec is None:
        if layers_named(args) > 0:
            out.append(_mm(dest, want, "no box3d decoder", "model._perception.box_dec", "argv names a depth and the model has no box3d decoder"))
        return out
    built = len(dec.blocks.layers)
    if built != want:
        out.append(_mm(dest, want, built, "model._perception.box_dec.blocks.layers", "argv and the BUILT decoder depth disagree"))
    if int(getattr(dec.blocks, "num_layers", built)) != built:
        out.append(_mm(dest, built, int(dec.blocks.num_layers), "model._perception.box_dec.blocks.num_layers",
                       "the stack's own num_layers attribute disagrees with its layer list"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DESTS["layers"], "built", dvb_check_layers)
