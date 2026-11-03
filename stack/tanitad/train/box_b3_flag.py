"""``--r9-box-quality`` / ``--r9-box-dn`` / ``--r9-box-class-sqrt`` / ``--r9-box-inloss-nms`` -- refcv9 arm B3 (REFCV9_DESIGN 2.4, R4, R11).

The lever code lives in ``tanitad/models/box_b3.py``; this module is the flags' contract with the trainer, in the shape
``map_banded_flag.py`` set: ``pin_args`` (refusals at argv time, a no-op when every flag is OFF), ``attach`` (build-time wiring: the
decoder's B3 state + the model attributes the loss reads), ``prepare`` (the DN pack, BEFORE the forward, training only),
``box3d_row`` (the loss dispatch: OFF calls ``refcv6_perception_branch.box3d_loss_row`` with the very arguments the trainer passed),
the four G-DVB ``built`` checks, ``register_dvb_levers`` and ``stamp`` (``config.json["r9_box_b3"]``, absent when OFF).

One flag per lever, so B3 (quality + DN + sqrt) vs B3' (+ in-loss NMS) is one variable.
"""
from __future__ import annotations

FLAGS = {"quality": "--r9-box-quality", "dn": "--r9-box-dn", "class_sqrt": "--r9-box-class-sqrt", "inloss_nms": "--r9-box-inloss-nms"}
DESTS = {k: "r9_box_" + k for k in FLAGS}
#: set on the model ONLY when a flag is on (an absent attribute is the tip's model)
MODEL_ATTR = "_r9_box_b3"
CLS_ATTR = "_r9_box_cls_weight"


def config(args):
    from tanitad.models.box_b3 import BoxB3Config
    return BoxB3Config(**{k: bool(getattr(args, d, False)) for k, d in DESTS.items()})


def is_any(args) -> bool:
    return any(bool(getattr(args, d, False)) for d in DESTS.values())


def pin_args(args) -> None:
    """The flags' refusals, at argv time. All OFF: nothing is checked."""
    if not is_any(args):
        return
    bad = []
    if not float(getattr(args, "w_box3d", 0.0) or 0.0) > 0.0:
        bad.append("--w-box3d is 0: there is no box3d head for a --r9-box-* lever to act on")
    if str(getattr(args, "slot_presence_loss", "bce") or "bce") != "focal":
        bad.append("--slot-presence-loss is not `focal`: B3 is defined on the refined focal presence path (the quality target and the "
                   "DN presence term are focal forms)")
    if str(getattr(args, "slot_query_select", "learned") or "learned") == "heatmap":
        bad.append("--slot-query-select heatmap: the A14 heatmap term and its top-K anchors are not built under B3")
    if getattr(args, "box3d_visible_filter", True) is False:
        bad.append("--no-box3d-visible-filter: the refined box loss always applies the field cut")
    if bool(getattr(args, "r9_box_class_sqrt", False)) and str(getattr(args, "agent_cls_weight", "off") or "off") == "off":
        bad.append("--r9-box-class-sqrt with --agent-cls-weight off: there is no inverse-frequency vector to temper (a silent no-op)")
    if bad:
        raise SystemExit("[box-b3] --r9-box-* refused:\n  - " + "\n  - ".join(bad))


def _box_dec(model):
    br = getattr(model, "_perception", None)
    return None if br is None else getattr(br, "box_dec", None)


def attach(model, args) -> None:
    """Build-time wiring, called by the trainer once the perception branch and ``model._cls_class_weight`` exist. OFF: touches nothing."""
    if not is_any(args):
        return
    from tanitad.models import box_b3
    cfg = config(args)
    dec = _box_dec(model)
    if dec is None:
        raise SystemExit("[box-b3] --r9-box-* but the model has no box3d decoder (--w-box3d > 0 builds it)")
    dec.enable_b3(dn=cfg.dn, nms_tau=(box_b3.NMS_TAU if cfg.inloss_nms else None))
    dec.to(next(dec.parameters()).device)
    setattr(model, MODEL_ATTR, cfg)
    if cfg.class_sqrt:
        w = getattr(model, "_cls_class_weight", None)
        if w is None:
            raise SystemExit("[box-b3] --r9-box-class-sqrt but model._cls_class_weight is None")
        setattr(model, CLS_ATTR, box_b3.sqrt_temper(w).to(w.device))


def prepare(model, batch: dict, device) -> None:
    """BEFORE the forward: build the DN pack from the batch's GT and hand it to the decoder (one-shot). A no-op unless ``--r9-box-dn``
    and the model is in training mode -- so the eval forward never sees a DN query."""
    cfg = getattr(model, MODEL_ATTR, None)
    if cfg is None or not cfg.dn or not model.training:
        return
    import torch
    from tanitad.models.box_b3 import build_dn_pack
    dec = _box_dec(model)
    if "agent_box" not in batch:
        raise SystemExit("[box-b3] --r9-box-dn but the batch carries no `agent_box`: no GT, no DN queries (refuse, do not skip)")
    tgt = {"box": batch["agent_box"].to(device), "yaw": batch["agent_yaw"].to(device), "cls": batch["agent_cls"].to(device),
           "valid": batch["agent_valid"].to(device)}
    if bool(getattr(model, "_vis1", False)):
        from tanitad.data.vis1 import vis1_split
        miss = [k for k in ("agent_vis_full", "agent_vis_px", "agent_vis_known") if k not in batch]
        if miss:
            raise SystemExit("[box-b3] VIS-1 is on and the batch carries no %s" % miss)
        pos = vis1_split(tgt, n_full=batch["agent_vis_full"].to(device), n_vis=batch["agent_vis_px"].to(device),
                         vis_known=batch["agent_vis_known"].to(device))["pos"]["valid"]
    else:
        from tanitad.refs.refc_agents import visible_target_filter
        pos = visible_target_filter(tgt)["valid"]
    keep = batch.get("agent_label")
    row_ok = (torch.ones(tgt["valid"].shape[0], dtype=torch.bool, device=device) if keep is None else keep.to(device).to(torch.bool))
    pack = build_dn_pack(tgt["box"], tgt["yaw"], tgt["cls"], pos, row_ok, x_fwd_m=float(dec.ranges.x_fwd_m),
                         y_half_m=float(dec.ranges.y_half_m), qpos_module=getattr(model._perception, "box_qpos", None))
    dec.set_dn(pack)


def box3d_row(model, perc, slots, tgt, *, cls_class_weight, visible_filter, presence_loss, vis1, vis, sel=None) -> dict:
    """The trainer's box3d loss row. No B3 lever on the loss (OFF, or ``--r9-box-class-sqrt`` alone): ``perc.box3d_loss_row`` with the
    SAME arguments the trainer passed (the sqrt vector replaces the class weight when that lever is on). In EVAL (``not model.training``) a DN
    config is read with ``dn=False``: the eval forward carries no DN queries, so there is no DN loss to compute (and none is demanded)."""
    cfg = getattr(model, MODEL_ATTR, None)
    if cfg is not None and cfg.dn and not getattr(model, "training", True):
        # DN is TRAINING-ONLY: ``prepare`` is a no-op in eval and ``Box3DSlotDecoder.b3_route`` returns the plain forward, so an eval slot dict
        # never holds a ``_dn_out``. ``refined_box3d_losses_b3`` REFUSES a DN config with no DN output (the train-time guard against a pack that
        # never reached the forward), which in eval made every in-training / EMA / final eval of a DN arm (B3, B3p) raise and be logged as an
        # ``eval_error`` row (MEASURED on the tiny path, BRIEF_BOX_B3). So the eval row is the row of the config with DN off -- the other levers
        # (quality, in-loss NMS, class sqrt) are unchanged, and the TRAINING path below never takes this branch.
        import dataclasses
        cfg = dataclasses.replace(cfg, dn=False)
    cw = getattr(model, CLS_ATTR, None) if (cfg is not None and cfg.class_sqrt) else cls_class_weight
    if cfg is None or not cfg.any_loss_change:
        return perc.box3d_loss_row(slots, tgt, cls_class_weight=cw, visible_filter=visible_filter,
                                   presence_loss=presence_loss, vis1=vis1, vis=vis)
    from tanitad.models.box_b3 import box3d_loss_row_b3
    if not visible_filter:
        raise ValueError("[box-b3] the refined box loss always applies the field cut (visible_filter=False is the pre-A9 regression arm)")
    return box3d_loss_row_b3(slots, tgt, cfg, cls_class_weight=cw, presence_loss=presence_loss, vis1=vis1, vis=vis, sel=sel)


def stamp(model, args) -> dict | None:
    """``config.json["r9_box_b3"]`` -- read OFF the built model; ``None`` (key absent) when every flag is OFF."""
    cfg = getattr(model, MODEL_ATTR, None)
    if cfg is None:
        return None
    from tanitad.models import box_b3 as B
    dec = _box_dec(model)
    return {"flags": {FLAGS[k]: bool(v) for k, v in cfg.as_dict().items()},
            "quality": {"target": "exp(-d_centre / %g m) * yawness on the matched slot; unmatched 0" % B.QUALITY_SIGMA_M,
                        "yawness": B.YAWNESS_FORM, "focal": "alpha_t * |t - p|^gamma * BCE (alpha 0.25, gamma 2)"} if cfg.quality else None,
            "dn": {"groups": B.DN_GROUPS, "neg_per_gt": B.DN_NEG_PER_GT, "max_gt": B.DN_MAX_GT, "range_scale_m": B.DN_RANGE_SCALE_M,
                   "pos_sigma_m": B.DN_POS_SIGMA_M, "neg_m": [B.DN_NEG_MIN_M, B.DN_NEG_MAX_M], "size_noise": B.DN_SIZE_NOISE,
                   "yaw_noise_rad": B.DN_YAW_NOISE_RAD, "loss_w": B.DN_LOSS_W,
                   "mask": "block-diagonal: matching | each DN group; both directions blocked",
                   "new_params": sum(p.numel() for n, p in dec.named_parameters() if n.startswith("dn_"))} if cfg.dn else None,
            "class_sqrt": {"rule": "sqrt(w_inverse_frequency) / mean"} if cfg.class_sqrt else None,
            "inloss_nms": {"tau": B.NMS_TAU, "p_floor": B.NMS_P_FLOOR, "form": "greedy BEV-NMS on the axis-aligned enclosing rectangle; "
                           "between layers: suppressed slots masked out of the others' self-attention keys; Hungarian over survivors"}
            if cfg.inloss_nms else None}


# ---- G-DVB ("built" levers): the argv's declaration equals what the BUILT model holds --------------------------------------------- #
def _mm(dest, declared, built, where, why):
    from tanitad.train.declared_vs_built import Mismatch
    return Mismatch(dest, declared, built, where, why)


def _dvb(key):
    dest = DESTS[key]

    def check(model, args) -> list:
        declared = bool(getattr(args, dest, False))
        cfg = getattr(model, MODEL_ATTR, None)
        attr = bool(getattr(cfg, key, False)) if cfg is not None else False
        out = []
        if declared != attr:
            out.append(_mm(dest, declared, attr, f"model.{MODEL_ATTR}.{key}", "argv and the attribute the LOSS reads disagree"))
        dec = _box_dec(model)
        if key == "dn":
            built = dec is not None and getattr(dec, "dn_embed", None) is not None
            if declared != built:
                out.append(_mm(dest, declared, built, "model._perception.box_dec.dn_embed", "argv and the BUILT DN parameters disagree"))
        elif key == "inloss_nms":
            tau = None if dec is None else getattr(dec, "b3_nms_tau", None)
            from tanitad.models.box_b3 import NMS_TAU
            if declared != (tau is not None) or (declared and abs(float(tau) - NMS_TAU) > 1e-12):
                out.append(_mm(dest, declared, tau, "model._perception.box_dec.b3_nms_tau", "argv and the BUILT decoder's NMS tau disagree"))
        elif key == "class_sqrt":
            w, base = getattr(model, CLS_ATTR, None), getattr(model, "_cls_class_weight", None)
            built = w is not None
            if declared != built:
                out.append(_mm(dest, declared, built, f"model.{CLS_ATTR}", "argv and the BUILT sqrt class weight disagree"))
            elif built:
                import torch
                from tanitad.models.box_b3 import sqrt_temper
                if base is None or not torch.allclose(w.cpu(), sqrt_temper(base).cpu(), atol=1e-6):
                    out.append(_mm(dest, "sqrt_temper(model._cls_class_weight)", "a different vector", f"model.{CLS_ATTR}",
                                   "the built weight is not the sqrt of the inverse-frequency vector"))
        return out
    check.__name__ = f"dvb_check_{key}"
    return check


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    for key, dest in DESTS.items():
        register(dest, "built", _dvb(key))
