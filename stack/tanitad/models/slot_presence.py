"""refcv7's REFINED slot losses (SPEC_REFCV7 §14, A9): focal presence (R1), per-layer deep supervision (R2) and the
VIS-1 IGNORE mask (R3) -- for BOTH slot heads, which share ``agent_slots.AgentSlotDecoder``:
the box3d head (``box3d_head.Box3DSlotDecoder``) and the planner's agent head (``refc_agents.build_agent_head``).

PI, verbatim (2026-09-27): *"based on the compariosn with proven refernce heads, refine our bb head and validate
it"*. Every constant below is the registered A9 literal; the evidence is the literature pass
(``…/2026-09-27-box-head-literature/RESULT.md`` §3 D1-D5) and the box-head audit (``…/2026-09-26-box-head-audit/``).

R1 -- FOCAL PRESENCE. The five nuScenes camera heads read (DETR3D, PETR, BEVFormer, StreamPETR, UniAD) all use a
sigmoid focal loss, alpha 0.25, gamma 2, weight 2.0, normalised by the number of matched GT (mmdet
``avg_factor``), a ``FocalLossCost`` (weight 2.0) in the matcher, and a 0.01 class-score prior. It REPLACES the
BCE with ``NO_OBJECT_W`` 0.1 averaged over all B*N slots, under which the 0.5 gate fires at a match belief of
pi = 1/11 = 0.091 (:func:`gate_match_belief`); under this focal loss the same gate needs pi = 0.75.
⛔ The audit's ln 10 tilt correction is NOT also applied: it is the zero-training readout of a BCE-0.1 head and
double-corrects a focal one.

R2 -- DEEP SUPERVISION. The SHARED ``norm`` + ``head`` read after EACH of the decoder's layers
(``AgentSlotDecoder.deep_supervision``, 0 new parameters); every layer is RE-MATCHED (Hungarian) against the same
targets and the per-layer losses are SUMMED with weight 1 (DETR / mmdet ``DETRHead``). Inference reads the LAST
layer only -- the dict every consumer already reads; the earlier layers ride in ``pred["aux"]``.

R3 -- VIS-1. ``tanitad.data.vis1.vis1_split`` makes the POSITIVE targets and the IGNORE rows (ONE function, used
here and by ``tanitad.eval.detection_metrics``). IGNORE rows are excluded from matching; an UNMATCHED slot whose
BEV centre is within 2 m of an IGNORE row gets presence weight 0.

⭐ THE DECISION RULE (declared, SPEC A9 BAR-B7-2 "focal, gate 0.5"): a slot is a DETECTION iff
``sigmoid(presence_logit) >= DETECTION_GATE = 0.5`` on the LAST decoder layer. Under R1's loss that is the slot's
own estimate that it is matched with probability >= 0.75 (:func:`gate_match_belief`).

⭐ WHAT THE PLANNER CONSUMES (unchanged except for the loss), in the LANDED files (tip cef9709 + this patch):
``refc.py:4717-4719`` runs the agent head and hands its output dict to ``refc_agents.AgentTokenEmbed.forward``
(``refc_agents.py:341-362``), which reads the LAST layer's ``presence_logit`` -> ``sigmoid`` (``:346``); that
probability is feature 15 of ``slot_features`` (``:289``, the entry at ``:317``) AND, in the default SOFT mode, the
multiplicative scale on every agent token (``:360``; the hard mode would pad below ``AgentSeamConfig.presence_gate``
0.5, ``:351-353``). ``refc.py:4727`` reads the last layer's ``box[..., :2]`` as the WP-B address. The per-layer ``"aux"`` list is read by NO consumer but the loss. Nothing about those reads changes
here; what changes is the DISTRIBUTION of the probability they read (focal is under-confident relative to BCE-0.1),
which is why G-DVB names the presence objective of the agent head explicitly.

⛔ THE LEGACY PATH IS NOT RE-IMPLEMENTED. :func:`refined_is_legacy` is True for the historical configuration
(BCE, no VIS-1, no per-layer outputs); the trainer then calls the ORIGINAL ``refc_agents.agent_losses`` /
``box3d_head.box3d_set_loss`` exactly as before, so every existing arm is bit-identical by construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "FOCAL_ALPHA", "FOCAL_GAMMA", "FOCAL_PRESENCE_W", "FOCAL_MATCH_W", "FOCAL_COST_EPS",
    "PRESENCE_PRIOR_FOCAL", "PRESENCE_PRIOR_LEGACY", "DETECTION_GATE", "PRESENCE_LOSSES",
    "G_LIVE_PRESENCE_MAX_CONFIDENT_FRAC", "sigmoid_focal_elementwise", "focal_presence_cost",
    "presence_expected_loss", "presence_optimum", "gate_match_belief", "layer_preds", "select_slots",
    "ignore_presence_weight", "presence_term", "refined_is_legacy", "refined_slot_losses",
    "refined_agent_losses", "refined_box3d_losses", "presence_sanity", "refine_stamp",
]

#: R1, the registered literals (SPEC_REFCV7 §14 A9; mmdet3d's shared recipe, all five configs read).
FOCAL_ALPHA: float = 0.25
FOCAL_GAMMA: float = 2.0
FOCAL_PRESENCE_W: float = 2.0            # loss_cls weight
FOCAL_MATCH_W: float = 2.0               # FocalLossCost weight
FOCAL_COST_EPS: float = 1e-12            # mmdet FocalLossCost eps
PRESENCE_PRIOR_FOCAL: float = 0.01       # RetinaNet / Deformable DETR / DETR3D class-score prior
PRESENCE_PRIOR_LEGACY: float = 0.05      # AgentSlotDecoder.PRESENCE_PRIOR before A9
DETECTION_GATE: float = 0.5              # the declared decision rule (last layer)
PRESENCE_LOSSES: tuple[str, ...] = ("bce", "focal")

#: G-LIVE-PRES (SPEC A9, fixed by A10 §15.2): at the end of the Thor smoke, on BOTH slot heads, the fraction of
#: slots with sigma >= 0.5 must be BELOW this literal, 0.5 (the gate profile owns the binding copy). refcv6's red
#: arm: 71-99 of 100 slots per window at its gate (the audit's S-conf). ⛔ The audit's G-LIVE-COUNT (sum of
#: calibrated presence / positives) is NOT APPLICABLE to a focal head (no calibration map, A10) and is not built.
G_LIVE_PRESENCE_MAX_CONFIDENT_FRAC: float = 0.5


# --------------------------------------------------------------------------------------------------------- #
# R1: the focal primitives                                                                                   #
# --------------------------------------------------------------------------------------------------------- #
def sigmoid_focal_elementwise(logit: Tensor, target: Tensor, alpha: float = FOCAL_ALPHA,
                              gamma: float = FOCAL_GAMMA) -> Tensor:
    """Sigmoid focal loss per element, no reduction -- ``torchvision.ops.sigmoid_focal_loss`` / mmdet
    ``py_sigmoid_focal_loss`` arithmetic: ``alpha_t * (1 - p_t)^gamma * BCE``.

    ``alpha`` weights the POSITIVE class (alpha_t = alpha for y = 1, 1 - alpha for y = 0), which is why the
    optimum sits at pi = 1 - alpha at p = 0.5 (:func:`gate_match_belief`)."""
    t = target.to(logit.dtype)
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    p_t = p * t + (1.0 - p) * (1.0 - t)
    loss = ce * (1.0 - p_t).pow(gamma)
    if alpha >= 0:
        loss = (alpha * t + (1.0 - alpha) * (1.0 - t)) * loss
    return loss


def focal_presence_cost(logit: Tensor, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                        eps: float = FOCAL_COST_EPS) -> Tensor:
    """mmdet ``FocalLossCost`` for "this slot is matched", unweighted: ``pos_cost - neg_cost``.

    Low (negative) for a confident slot, high for an empty one; ``FOCAL_MATCH_W`` multiplies it in the matcher."""
    p = torch.sigmoid(logit)
    neg_cost = -(1.0 - p + eps).log() * (1.0 - alpha) * p.pow(gamma)
    pos_cost = -(p + eps).log() * alpha * (1.0 - p).pow(gamma)
    return pos_cost - neg_cost


# --------------------------------------------------------------------------------------------------------- #
# R1: the ANALYTIC optimum (what a gate means under each loss)                                              #
# --------------------------------------------------------------------------------------------------------- #
def presence_expected_loss(p: float, pi: float, loss: str) -> float:
    """E[loss] of a slot that predicts ``p`` and is matched with probability ``pi``.

    ``"bce_noobj"`` -- the pre-A9 loss: weight 1 on a matched slot, NO_OBJECT_W 0.1 on an unmatched one;
    ``"focal"`` -- R1 (alpha 0.25, gamma 2). The normaliser (per slot or per matched GT) scales the loss and
    does not move the optimum."""
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    if loss == "bce_noobj":
        return -pi * math.log(p) - (1.0 - pi) * 0.1 * math.log(1.0 - p)
    if loss == "focal":
        a, g = FOCAL_ALPHA, FOCAL_GAMMA
        return (-pi * a * (1.0 - p) ** g * math.log(p)
                - (1.0 - pi) * (1.0 - a) * p ** g * math.log(1.0 - p))
    raise ValueError(f"loss {loss!r}")


def presence_optimum(pi: float, loss: str, tol: float = 1e-12) -> float:
    """``argmin_p E[loss]`` by golden-section on the logit (both losses are unimodal in p)."""
    lo, hi = -40.0, 40.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    def f(z):
        return presence_expected_loss(1.0 / (1.0 + math.exp(-z)), pi, loss)
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    while hi - lo > tol:
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    z = 0.5 * (lo + hi)
    return 1.0 / (1.0 + math.exp(-z))


def gate_match_belief(gate: float, loss: str) -> float:
    """The match belief pi at which the loss's optimum ``p*`` equals ``gate`` (bisection on pi).

    MEASURED ANALYTICALLY here and pinned by the tests: ``bce_noobj`` at 0.5 -> 1/11 = 0.0909...; ``focal`` at 0.5
    -> 0.75 exactly (d/dp E = 0 at p = 1/2 gives pi * alpha = (1 - pi)(1 - alpha))."""
    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if presence_optimum(mid, loss) < gate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------------------- #
# R2: per-layer outputs                                                                                      #
# --------------------------------------------------------------------------------------------------------- #
def layer_preds(pred: dict) -> list:
    """``[layer 0, ..., layer L-2, LAST]`` -- the decoded dict of every supervised layer, last one LAST.

    Without per-layer outputs (``"aux"`` absent) this is ``[pred]``: the historical single-layer loss."""
    return [*list(pred.get("aux") or []), pred]


def select_slots(slots: dict, sel: Tensor, batch: int) -> dict:
    """Batch-row selection of a slot-output dict INCLUDING its per-layer ``"aux"`` entries.

    ⛔ Exists because the trainer's selection comprehension (``v.index_select(0, sel) if torch.is_tensor(v)``)
    passes a LIST through unselected: the aux layers would keep the full batch while the last layer and the
    targets were narrowed to the labelled rows -- a silent row misalignment."""
    out = {}
    for k, v in slots.items():
        if k == "aux" and isinstance(v, (list, tuple)):
            out[k] = [select_slots(a, sel, batch) for a in v]
        elif torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == batch:
            out[k] = v.index_select(0, sel)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------------------------------------- #
# R3: the IGNORE presence weight                                                                             #
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def ignore_presence_weight(pred_box: Tensor, ign_box: Tensor, ignore: Tensor,
                           radius_m: float | None = None) -> Tensor:
    """``[B, N]`` in {0, 1}: 0 where the slot's BEV centre lies within ``radius_m`` of ANY IGNORE row.

    ``pred_box`` [B, N, >=2] and ``ign_box`` [B, A, >=2] in metres (cx, cy first); ``ignore`` [B, A] bool. The
    caller restores weight 1 on MATCHED slots (a positive's slot is never exempted). Not differentiated: it is a
    label-side mask, and the distance to an ignore row must not become a gradient path."""
    from tanitad.data.vis1 import VIS1_IGNORE_RADIUS_M
    r = float(VIS1_IGNORE_RADIUS_M if radius_m is None else radius_m)
    B, N = int(pred_box.shape[0]), int(pred_box.shape[1])
    w = torch.ones(B, N, dtype=pred_box.dtype, device=pred_box.device)
    if ignore is None or int(ignore.shape[1]) == 0 or not bool(ignore.any()):
        return w
    d = torch.cdist(pred_box[..., :2].float(), ign_box[..., :2].to(pred_box.device).float())   # [B, N, A]
    near = (d <= r) & ignore.to(pred_box.device)[:, None, :]
    w[near.any(-1)] = 0.0
    return w


def presence_term(logit: Tensor, match: dict, *, mode: str, ignore_w: Tensor | None = None) -> dict:
    """The presence loss of ONE layer: ``{"loss", "n_matched", "n_exempt"}``.

    * ``focal`` (R1): ``sum(focal * w) / max(n_matched, 1)`` over ALL slots of the batch, ``w`` = 1 except 0 on
      IGNORE-exempt unmatched slots. mmdet's ``avg_factor`` = the number of positives, clamped to 1.
    * ``bce`` (the pre-A9 regression arm): the historical ``slot_set_loss`` formula -- BCE, weight 1 matched /
      ``NO_OBJECT_W`` unmatched, mean over B*N -- with the IGNORE weights multiplied into the unmatched ones.
    """
    from tanitad.models.agent_slots import NO_OBJECT_W
    if mode not in PRESENCE_LOSSES:
        raise ValueError(f"presence loss {mode!r} not in {PRESENCE_LOSSES}")
    tgt = torch.zeros_like(logit)
    matched = torch.zeros_like(logit, dtype=torch.bool)
    for b, r in enumerate(match["rows"]):
        if r.numel():
            r = r.to(logit.device)
            tgt[b, r] = 1.0
            matched[b, r] = True
    n_m = int(matched.sum())
    w = torch.ones_like(logit) if ignore_w is None else ignore_w.to(logit.dtype).clone()
    w[matched] = 1.0
    n_ex = int(((w == 0) & ~matched).sum())
    if mode == "focal":
        el = sigmoid_focal_elementwise(logit, tgt)
        loss = (el * w).sum() / float(max(n_m, 1))
    else:
        wgt = torch.full_like(logit, NO_OBJECT_W) * w
        wgt[matched] = 1.0
        loss = F.binary_cross_entropy_with_logits(logit, tgt, weight=wgt)
    return {"loss": loss, "n_matched": n_m, "n_exempt": n_ex}


# --------------------------------------------------------------------------------------------------------- #
# the refined set losses                                                                                     #
# --------------------------------------------------------------------------------------------------------- #
def refined_is_legacy(presence_loss: str, vis1: bool, pred: dict | None = None) -> bool:
    """True for the historical configuration: the caller must run the ORIGINAL loss, untouched."""
    return (str(presence_loss) == "bce" and not bool(vis1)
            and (pred is None or not (pred.get("aux") or [])))


def refined_slot_losses(pred: dict, tgt_real: dict, *, head: str, presence_loss: str, vis1: bool,
                        vis: dict | None = None, visible_filter: bool = True, weights: dict | None = None,
                        cls_class_weight=None) -> dict:
    """The per-layer R1/R2/R3 set loss for one head. ``head`` is ``"agent"`` or ``"box3d"``.

    ``tgt_real`` carries ``valid`` = every REAL row (the batch block before any filter); ``vis`` =
    ``{"n_full", "n_vis", "known"}`` [B, A] when ``vis1``. Returns the LAST layer's terms under the historical
    keys (``loss_presence`` IS the refined presence term), ``total`` = the SUM over layers, and per-layer
    ``loss_layer{i}`` / ``loss_presence_layer{i}`` with ``n["layers"]`` = the number of supervised layers.
    """
    from tanitad.models.agent_slots import SLOT_LOSS_W, match_slots, slot_set_loss
    from tanitad.models.box3d_head import box3d_set_loss
    from tanitad.refs.refc_agents import visible_target_filter
    if head not in ("agent", "box3d"):
        raise ValueError(f"head {head!r}")
    if presence_loss not in PRESENCE_LOSSES:
        raise ValueError(f"presence loss {presence_loss!r} not in {PRESENCE_LOSSES}")
    ignore = None
    counts = {"target_prefilter": int(tgt_real["valid"].sum())}
    if vis1:
        from tanitad.data.vis1 import vis1_split
        if vis is None:
            raise ValueError("[slot-refine] vis1=True but no visibility block was passed: VIS-1 is never "
                             "guessed (REFUSE, do not skip)")
        sp = vis1_split(tgt_real, n_full=vis["n_full"], n_vis=vis["n_vis"], vis_known=vis["known"])
        tpos, ignore = sp["pos"], sp["ignore"]
        counts.update({f"vis1_{k}": v for k, v in sp["counts"].items()})
        counts["target_visible"] = int(sp["in_filter"].sum())
    else:
        tpos = visible_target_filter(tgt_real) if visible_filter else tgt_real
        counts["target_visible"] = int(tpos["valid"].sum())
    if head == "box3d" and tpos.get("zh_mask") is not None:
        tpos = {**tpos, "zh_mask": tpos["zh_mask"].to(torch.bool) & tpos["valid"].to(torch.bool)}
    cost = "focal" if presence_loss == "focal" else "sigmoid"
    w_pres = FOCAL_PRESENCE_W if presence_loss == "focal" else float(SLOT_LOSS_W["presence"])
    w0 = {**(weights or {}), "presence": 0.0}
    layers = layer_preds(pred)
    per = []
    for p in layers:
        m = match_slots(p, tpos, presence_cost=cost)
        if head == "box3d":
            base = box3d_set_loss(p, tpos, match=m, weights=w0, cls_class_weight=cls_class_weight,
                                  visible_filter=False)
        else:
            base = slot_set_loss(p, tpos, match=m, weights=w0, cls_class_weight=cls_class_weight)
        iw = None
        if ignore is not None:
            iw = ignore_presence_weight(p["box"], tgt_real["box"], ignore)
        pr = presence_term(p["presence_logit"], m, mode=presence_loss, ignore_w=iw)
        per.append({"base": base, "presence": pr, "match": m,
                    "total": base["total"] + w_pres * pr["loss"]})
    last = per[-1]
    out = {k: v for k, v in last["base"].items() if k.startswith("loss_")}
    out["loss_presence"] = last["presence"]["loss"]
    out["total"] = sum(x["total"] for x in per)
    for i, x in enumerate(per):
        out[f"loss_layer{i}"] = x["total"]
        out[f"loss_presence_layer{i}"] = x["presence"]["loss"]
    n = dict(last["base"].get("n") or {})
    n.update(counts)
    n["layers"] = len(per)
    n["presence_exempt"] = int(last["presence"]["n_exempt"])
    n["presence_matched"] = int(last["presence"]["n_matched"])
    out["n"] = n
    out["match"] = last["match"]
    out["_weights"] = {**dict(last["base"].get("_weights") or {}), "presence": w_pres}
    out["_refine"] = refine_stamp(presence_loss, vis1, n_layers=len(per))
    return out


def refined_agent_losses(slots: dict, tgt: dict, cfg, *, cam=None, weights: dict | None = None,
                         cls_class_weight=None, vis: dict | None = None) -> dict:
    """``refc_agents.agent_losses`` under R1/R2/R3 -- same keys, plus the per-layer ones.

    ⛔ The monocular terms (``w_project`` / ``w_ground``) read the LAST layer's boxes and match, exactly where
    ``agent_losses`` reads them; refcv6/refcv7 run both at 0.0."""
    from tanitad.refs.refc_agents import (RigCamera, ground_range_prior, monocular_projection_loss,
                                          row_cameras)
    out = refined_slot_losses(slots, tgt, head="agent", presence_loss=str(cfg.presence_loss),
                              vis1=bool(cfg.vis1), vis=vis, visible_filter=True, weights=weights,
                              cls_class_weight=cls_class_weight)
    total = out["total"]
    match = out["match"]
    _cams = row_cameras(cam, int(slots["box"].shape[0]))
    _n_cam = sum(1 for c in _cams if c is not None)
    out["cam_scope"] = ("none" if cam is None else ("single" if isinstance(cam, RigCamera) else "per-row"))
    out["n"]["rows_with_cam"] = int(_n_cam)
    out["n"]["rows_no_cam"] = int(len(_cams) - _n_cam)
    if _n_cam and float(cfg.w_project) > 0.0:
        pj = monocular_projection_loss(slots["box"], tgt["box"], cam, match)
        out["loss_project"] = pj["loss"]
        out["n"]["project"] = pj["n"]
        total = total + float(cfg.w_project) * pj["loss"]
    if _n_cam and float(cfg.w_ground) > 0.0:
        gp = ground_range_prior(slots["box"], cam)
        out["loss_ground"] = gp["loss"]
        out["n"]["ground"] = gp["n"]
        total = total + float(cfg.w_ground) * gp["loss"]
    out["total"] = total
    out["n_dropped"] = int(out["n"].get("dropped", 0))
    out["n_target"] = int(out["n"].get("target", 0))
    out["filter_visible"] = True
    return out


def refined_box3d_losses(slots: dict, tgt: dict, *, presence_loss: str, vis1: bool, vis: dict | None = None,
                         visible_filter: bool = True, weights: dict | None = None,
                         cls_class_weight=None) -> dict:
    """``box3d_head.box3d_set_loss`` under R1/R2/R3 -- same keys, plus the per-layer ones."""
    out = refined_slot_losses(slots, tgt, head="box3d", presence_loss=presence_loss, vis1=vis1, vis=vis,
                              visible_filter=visible_filter, weights=weights, cls_class_weight=cls_class_weight)
    n = out["n"]
    n["dropped_not_visible"] = int(n.get("target_prefilter", 0)) - int(n.get("target_visible", 0))
    out["_visible_filter"] = bool(visible_filter)
    return out


# --------------------------------------------------------------------------------------------------------- #
# G-LIVE presence sanity + the record                                                                        #
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def presence_sanity(presence_logit: Tensor, gate: float = DETECTION_GATE,
                    max_frac: float = G_LIVE_PRESENCE_MAX_CONFIDENT_FRAC) -> dict:
    """``{"frac_confident", "n_slots", "pass"}`` -- the fraction of slots with ``sigmoid >= gate`` (A10 §15.2
    G-LIVE-PRES: "the fraction of slots with sigma >= 0.5 is < 0.5", on BOTH slot heads)."""
    p = torch.sigmoid(presence_logit.detach().float())
    frac = float((p >= float(gate)).float().mean()) if p.numel() else float("nan")
    return {"frac_confident": frac, "n_slots": int(p.numel()), "gate": float(gate), "max_frac": float(max_frac),
            "pass": bool(frac == frac and frac < float(max_frac))}


def refine_stamp(presence_loss: str, vis1: bool, *, n_layers: int | None = None) -> dict:
    """The refinement as data, for ``config.json`` and the log."""
    focal = str(presence_loss) == "focal"
    return {"presence_loss": str(presence_loss),
            "focal_alpha": FOCAL_ALPHA if focal else None, "focal_gamma": FOCAL_GAMMA if focal else None,
            "presence_weight": FOCAL_PRESENCE_W if focal else 1.0,
            "presence_normaliser": "n_matched (clamped to 1)" if focal else "B*N slots (mean)",
            "match_presence_cost": ("focal x %.1f" % FOCAL_MATCH_W) if focal else "-sigmoid x 1.0",
            "vis1": bool(vis1), "n_supervised_layers": n_layers,
            "decision_rule": f"sigmoid(presence_logit) >= {DETECTION_GATE} on the LAST decoder layer",
            "gate_match_belief": 0.75 if focal else 1.0 / 11.0,
            "source": "SPEC_REFCV7.md §14 (A9) R1-R3"}
