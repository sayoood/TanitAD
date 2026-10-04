"""refcv7 NEW-2 -- the SAM3 map head at **10 cm** (``SPEC_REFCV7.md`` §6.2, §11).

WHAT IT IS (Amendment A6, PI 2026-09-27: *"do c"*)
---------------------------------------------------
ONE lift and ONE BEV encoder, attached by the trainer as ``model._map_hires`` behind
``--map-hires on``, run INSIDE the forward (``refc_v3.RefCV3Model._bev_hook``)::

    fmap_s8 [B, C8, H/8, W/8]  (the trunk's stride-8 map of the CURRENT frame;
       |                       `TimmResNetTrunk.enable_s8_tap`, passed through by
       |                       `RefCModel.forward` as out["fmap_s8"])
    lift @ 0.25 m    -> [B, d_lift, X4, Y4]   (the refcv6 lift, projection first)
    encoder @ 0.25 m -> [B, d_model, X4, Y4]  ("map_hires_bev": stem + dilated blocks)
       |-- (i)  the MAP: refine = 1x1 + GN -> bilinear x2.5 -> 2 x (3x3 + GN + GELU)
       |        -> 1x1 = [B, 8, X10, Y10] logits at 0.1 m, the ONLY map
       '-- (ii) the PLANNER's BEV: `refcv6_perception_branch.PlannerBEVPool` crops the
                planner's 60 m x +-16 m window, 2 x 2 average-pools it to 0.5 m and
                projects it (1x1 + GN) to the consumers' width -> [B, 96, 120, 64],
                which REPLACES the stride-16 lift for box3d, the 30 x 16 BEV tokens
                and the planner's BEV coupling (``--bev-source map_hires_pool``).

``(X4, Y4)`` / ``(X10, Y10)`` are the DECLARED extent (``MapHiresConfig.x_max_m`` /
``y_half_m`` -> :class:`semantic_map_gt_fine.MapExtent`) at 0.25 m / 0.1 m: 240 x 128 /
600 x 320 on today's 60 m x +-16 m, larger at the census extent (§11.2). Nothing in
this file types a grid size.

⛔ Removed in refcv7 (§11.1): the stride-16 0.5 m lift and the 0.5 m map head and
loss. Nothing at 0.5 m is supervised as a map. With ``--map-hires off`` nothing here
is built, the trunk's tap stays off and ``RefCModel.forward`` emits ``fmap_s8 = None``:
parameters, ``state_dict``, RNG draws and every tensor are the tip's (refcv6).

WHERE THE GEOMETRY COMES FROM -- nothing re-derived
---------------------------------------------------
* the camera model and the per-clip lift geometry: :func:`bev_lift.
  build_lift_geometry` through :class:`refcv6_perception_branch.LiftGeometryBank`,
  called with ``stride=8`` and the extent's 0.25 m grid -- both are that bank's own
  parameters. :class:`HiresLiftGeometryBank` only BOUNDS its cache (1.11 MB per clip
  at 60 m x +-16 m, MEASURED; 4,508 clips unbounded would be 5.0 GB);
* the sampler: :class:`BEVLiftProjectFirst` IS :class:`bev_lift.BEVLift` -- same
  parameters, same ``state_dict`` keys, same function -- with the 1x1 projection
  applied BEFORE the bilinear sampling instead of after. Both are linear, so they
  commute exactly; the test pins equality to the original on random inputs. The
  reason is memory: sampling 512 stride-8 channels at 4 heights x 240 x 128 cells
  is 62.9 M floats per sample, sampling 64 projected channels is 7.9 M;
* the channel count ``C8`` and the map shape: ``encoder.s8_dim`` / ``s8_shape``,
  read from timm's ``feature_info`` by ``enable_s8_tap`` -- never a literal.

THE LOSS
--------
:func:`hires_map_ce`: hard-label cross-entropy over the 8 classes at 0.1 m,
``ignore_index = 255`` (not seen), optional per-class weights (SPEC_REFCV7 §13, A8:
sqrt(median-frequency), ``stack/scripts/compute_map_class_weights.py`` id ``sqrt_mf``;
option (c) removed the 0.5 m auxiliary that kept drivable's gradient, and plain MF
would leave each big class 2.7 % of it at convergence), mean over supervised
cells with the weights in the denominator (``F.cross_entropy``'s rule, and the
rule ``bev_encoder.map_soft_ce`` already follows). A not-seen cell contributes
EXACTLY zero to the value and to the gradient (pinned, with a mutation arm).

⚠️ What 10 cm hard labels do NOT do (map-signal audit D7, correcting the design note
in ``…/2026-09-26-refcv7-map-hires/RESULT.md`` §4.1): they do not remove the
out-voting of thin classes "by construction". With localisation error sigma >= 0.2 m
the calibrated posterior at a line is < 0.5 and the edge ceiling falls to 0.04
(audit §2c). The levers are localisation first, then the decision rule -- hence the
DECLARED prior-corrected rule (:func:`decide`).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from tanitad.data.bev_raster import GRID_DEFAULT, BEVGrid
from tanitad.data.semantic_map_gt_fine import (
    BAND_ROWS_FINE, EXTENT_REFCV7, EXTENT_V2, FINE_CELL_M, FINE_CLASSES,
    N_FINE_CLASSES, NOT_SEEN_CODE, MapExtent, band_key,
)
from tanitad.models.bev_encoder import _DilatedBlock
from tanitad.models.bev_lift import HEIGHTS_M, BEVLift
from tanitad.models.refcv6_perception_branch import LiftGeometryBank, grad_abs_sum

__all__ = [
    "MapHiresConfig", "BEVLiftProjectFirst", "HiresBEVEncoder", "HiresRefine",
    "MapHiresBranch", "HiresLiftGeometryBank", "build_map_hires_branch",
    "hires_map_ce", "map_hires_loss_row", "lift_valid_to_fine",
    "load_class_weights", "CLASS_WEIGHT_SCHEMA", "CLASS_WEIGHT_CLIP_MAX",
    "grad_reach_report_hires", "built_state", "MapHiresMissingInput",
    "CLASS_KEYS", "BAND_KEYS", "LOG_PREFIX", "PER_CLASS_STATS", "per_class_key",
    "band_keys_for_rows", "band_keys_of", "per_class_signal", "per_class_log_values",
    "derived_per_class", "missing_per_class_keys", "per_class_liveness",
    "is_exact_log_key", "declared_extent",
    "DVB_KINDS", "dvb_check_map_hires", "dvb_check_w_map_hires",
    "dvb_check_class_weights", "dvb_check_decision_rule", "dvb_check_extent",
    "dvb_check_bev_source", "dvb_check_grad_ckpt", "dvb_check_planner_crop",
    "declared_grad_ckpt", "PLANNER_CROP_DEFAULT",
    "register_dvb_levers", "DECISION_RULES", "decide",
    # 2026-10-04 refcv7 inference-time fixes F1/F2 (OPT-IN; the default path is untouched)
    "CLASS_THRESHOLD_SCHEMA", "class_posterior_logits", "decide_masks",
    "load_class_thresholds", "dilate_chebyshev", "tolerant_iou_from_counts",
    "MONITOR_TOL_CELLS", "PER_CLASS_STATS_THR",
    # NEW-2 R2 (A12): the 0.1 m near-range lift
    "NearLiftSkip", "derive_near_geometry", "NEAR_LIFT_STEP_M", "declared_near_lift_m",
    "dvb_check_near_lift",
    # NEW-2 R3 (A15): the near refine block
    "NearRefineBlock", "NEAR_REFINE_MAX_BLOCKS", "declared_near_refine_blocks",
    "dvb_check_near_refine",
]

#: the JSON the weights script writes and the trainer reads.
CLASS_WEIGHT_SCHEMA = "tanitad.map_hires_class_weights/1"
#: SPEC_REFCV7 §6.2 / the NEW-2 brief: median-frequency weights clipped to <= 25.
CLASS_WEIGHT_CLIP_MAX = 25.0
#: the lift's cell (SPEC_REFCV7 §6.2) and the ratio to the 10 cm label cell.
LIFT_CELL_M = 0.25
#: the declared decision rules (``MapHiresConfig.decision_rule``); the first is the default.
#: ``class_threshold`` (refcv7 diagnostics F1, 2026-10-04) is OPT-IN and needs a thresholds
#: file (``--map-hires-class-thresholds``): see :func:`decide_masks`. Appending it leaves
#: ``DECISION_RULES[0]`` -- the default every caller reads -- exactly as it was.
DECISION_RULES: tuple[str, ...] = ("prior_corrected", "raw", "class_threshold")
#: the schema of the per-class thresholds JSON (:func:`load_class_thresholds`).
CLASS_THRESHOLD_SCHEMA = "tanitad.map_hires_class_thresholds/1"
#: F2: the boundary tolerance of the monitor's tolerant IoU, in 10 cm cells (2 cells = 0.2 m).
MONITOR_TOL_CELLS = 2
#: F2: the extra per-class x band statistics a row carries when thresholds are configured
#: (``interthr`` / ``unionthr`` / ``predthr`` under the THRESHOLDED rule; ``tppthr2`` /
#: ``tpgthr2`` = predicted cells with a GT cell within 2 cells / GT cells with a predicted
#: cell within 2 cells). New keys only: ``PER_CLASS_STATS`` is unchanged.
PER_CLASS_STATS_THR: tuple[str, ...] = ("interthr", "unionthr", "predthr",
                                        "tppthr%d" % MONITOR_TOL_CELLS,
                                        "tpgthr%d" % MONITOR_TOL_CELLS)
#: NEW-2 R2 (A12): the near lift's extent is a multiple of this, so it tiles the 0.25 m lift
#: grid, the 0.1 m label grid and the 0.5 m planner grid exactly.
NEAR_LIFT_STEP_M = 0.5
#: NEW-2 R3 (A15): at most this many near refine blocks (the registered arm uses 1).
NEAR_REFINE_MAX_BLOCKS = 4


def decide(logits: Tensor, rule: str, class_weight: Tensor | None = None,
           class_thresholds=None) -> Tensor:
    """``[B, 8, H, W]`` logits -> ``[B, H, W]`` class codes under a DECLARED rule.

    * ``"raw"``             -- ``argmax_c z_c``;
    * ``"prior_corrected"`` -- ``argmax_c (z_c - log w_c)``, ``w`` the loss's frozen
      class weights. ⛔ REFUSES ``class_weight=None``: correcting by "no weights"
      would silently BE the raw rule on a weighted model. Pass ``torch.ones(8)`` to
      state uniform weights (then the two rules are bit-identical, control C3);
    * ``"class_threshold"`` (OPT-IN, refcv7 diagnostics F1) -- the rule itself is
      MULTI-LABEL (:func:`decide_masks`: cell in class ``c`` iff ``p_hat_c >= tau_c``);
      this returns ONE code per cell for the consumers that need a partition (a
      renderer, an argmax-style caller): ``argmax_c (logit(p_hat_c) - tau_c)`` -- the
      class that clears its own threshold by the widest margin, or, where none clears
      it, the one closest to clearing. The IoU the monitor logs is computed on the
      multi-label masks, NOT on this partition. ⛔ REFUSES ``class_weight=None`` and
      ``class_thresholds=None``.

    ⛔ A class with weight 0 is NEVER decided (its score is ``-inf``): it was never
    supervised, so its logit carries no calibrated evidence. The naive formula would
    do the opposite -- ``z - log 0 = +inf`` decides it on EVERY cell (pinned, with a
    red arm). Negative or non-finite weights are refused."""
    if rule not in DECISION_RULES:
        raise ValueError(f"decision rule {rule!r} not in {DECISION_RULES}")
    if rule == "raw":
        return logits.argmax(dim=1)
    if rule == "class_threshold":
        tau = _check_class_thresholds(class_thresholds, logits.device)
        s = logits.float() - _decision_logw(class_weight, logits).view(1, -1, 1, 1)
        return (class_posterior_logits(s) - tau).argmax(dim=1)
    if class_weight is None:
        raise ValueError("the prior-corrected rule needs the loss's class weights "
                         "(torch.ones(8) for uniform); None would silently be `raw`")
    w = class_weight.detach().to(device=logits.device, dtype=torch.float32)
    if tuple(w.shape) != (logits.shape[1],):
        raise ValueError(f"class_weight {tuple(w.shape)} vs {logits.shape[1]} classes")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError(f"class weights must be finite and >= 0, got {w.tolist()}")
    pos = w > 0
    logw = torch.where(pos, torch.log(torch.where(pos, w, torch.ones_like(w))),
                       torch.full_like(w, float("inf")))
    return (logits.float() - logw.view(1, -1, 1, 1)).argmax(dim=1)


def _decision_logw(class_weight, logits: Tensor) -> Tensor:
    """``[8]`` ``log w`` for the calibrated posterior; ``+inf`` for a weight of 0 (that
    class was never supervised and is NEVER decided, :func:`decide`'s rule). Refuses
    ``None`` / wrong shape / negative / non-finite weights."""
    if class_weight is None:
        raise ValueError("the class-threshold rule needs the loss's class weights "
                         "(torch.ones(8) for uniform): p_hat = softmax(z - log w)")
    w = torch.as_tensor(class_weight).detach().to(device=logits.device, dtype=torch.float32)
    if tuple(w.shape) != (logits.shape[1],):
        raise ValueError(f"class_weight {tuple(w.shape)} vs {logits.shape[1]} classes")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError(f"class weights must be finite and >= 0, got {w.tolist()}")
    pos = w > 0
    if not bool(pos.any()):
        raise ValueError("all class weights are 0: no class can be decided")
    return torch.where(pos, torch.log(torch.where(pos, w, torch.ones_like(w))),
                       torch.full_like(w, float("inf")))


def _check_class_thresholds(class_thresholds, device) -> Tensor:
    """``[1, 8, 1, 1]`` float32 logit thresholds, or a loud refusal."""
    if class_thresholds is None:
        raise ValueError("decision rule 'class_threshold' needs the per-class thresholds "
                         "(--map-hires-class-thresholds <json>, load_class_thresholds): "
                         "without them there is no rule")
    t = torch.as_tensor(class_thresholds, dtype=torch.float32).detach().to(device).reshape(-1)
    if tuple(t.shape) != (N_FINE_CLASSES,) or not bool(torch.isfinite(t).all()):
        raise ValueError(f"class thresholds must be {N_FINE_CLASSES} finite logit values, "
                         f"got {t.tolist()}")
    return t.view(1, -1, 1, 1)


def class_posterior_logits(s: Tensor) -> Tensor:
    """``[B, C, H, W]`` scores -> ``logit(softmax(s)_c) = s_c - logsumexp_{j != c} s_j``
    per class, exact and stable (the diagnostics' ``diag_metrics.class_logits``,
    verbatim)."""
    outs = []
    for c in range(s.shape[1]):
        so = s.clone()
        so[:, c] = float("-inf")
        outs.append(s[:, c] - torch.logsumexp(so, dim=1))
    return torch.stack(outs, dim=1)


def decide_masks(logits: Tensor, class_weight, class_thresholds,
                 sup: Tensor | None = None) -> Tensor:
    """THE ``class_threshold`` RULE (refcv7 diagnostics F1, ``2026-10-04-refcv7-map-box-
    diagnostics`` RESULT sec. 1.2 / 3): ``[B, C, H, W]`` bool, one-vs-rest and
    MULTI-LABEL -- cell ``i`` is in class ``c`` iff the calibrated posterior
    ``p_hat_c = softmax(z - log w)_c`` is ``>= tau_c``, i.e. ``logit(p_hat_c) >= tau_c``
    in logit space (the space the thresholds were fitted in). A cell can be in several
    classes or in none: the diagnostics' ``thr_phat`` decision did not resolve either
    (``diag_metrics.decision_masks``), and the measured IoUs (lane 0.164, edge 0.042,
    hatched 0.062 ...) are of THESE masks. ``sup`` ``[B, H, W]`` bool restricts to the
    supervised cells, as the metric does."""
    tau = _check_class_thresholds(class_thresholds, logits.device)
    s = logits.float() - _decision_logw(class_weight, logits).view(1, -1, 1, 1)
    m = class_posterior_logits(s) >= tau
    return m if sup is None else (m & sup[:, None])


def dilate_chebyshev(mask: Tensor, k: int) -> Tensor:
    """``[B, C, H, W]`` bool -> Chebyshev dilation by ``k`` cells (the square
    ``(2k+1)^2``), as two separable max-pools -- the diagnostics' ``dilate``, verbatim
    (0/1 values, so fp16 on CUDA and fp32 on CPU give identical results)."""
    if k == 0:
        return mask
    x = mask.to(torch.float16) if mask.is_cuda else mask.to(torch.float32)
    B, C, H, W = x.shape
    x = x.reshape(B * C, 1, H, W)
    x = F.max_pool2d(x, kernel_size=(2 * k + 1, 1), stride=1, padding=(k, 0))
    x = F.max_pool2d(x, kernel_size=(1, 2 * k + 1), stride=1, padding=(0, k))
    return x.reshape(B, C, H, W) > 0.5


def tolerant_iou_from_counts(pred, gt, tpp, tpg):
    """The diagnostics' boundary-tolerant IoU from POOLED counts (``tolerant_summary``,
    verbatim): ``P_k = tpp / pred``, ``R_k = tpg / gt``, ``F_k = 2 P R / (P + R)``,
    ``IoU_k = F_k / (2 - F_k)`` (at k = 0 this is exactly the IoU). ``None`` when both
    ``pred`` and ``gt`` are 0; 0.0 when exactly one is."""
    P, G = float(pred), float(gt)
    prec = float(tpp) / P if P > 0 else None
    rec = float(tpg) / G if G > 0 else None
    if prec is None or rec is None:
        f = None if (P == 0 and G == 0) else (0.0 if (P == 0 or G == 0) else None)
    else:
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return None if f is None else f / (2.0 - f)


def load_class_thresholds(path, *, class_weight=None) -> tuple[Tensor, dict]:
    """Read the per-class thresholds JSON -> ``([8] float32 LOGIT thresholds, stamp)``.

    The file (``stack/tanitad/configs/refcv7_map_hires_class_thresholds_train.json``)
    carries ``tau_phat_logit`` (what the rule thresholds -- ``logit(p_hat_c) >= tau``),
    ``tau_phat_prob`` (= sigmoid of it, a cross-check), ``classes`` (= :data:`CLASS_KEYS`) and its provenance (source file,
    fit split, n windows, checkpoint step).

    ⛔ REFUSES: a different schema, a class order that is not ``CLASS_KEYS``, not 8
    finite values, a probability that is not the sigmoid of its logit (1e-9), and --
    when the file declares ``class_weight_values`` and ``class_weight`` is given --
    thresholds fitted under OTHER class weights: ``p_hat = softmax(z - log w)`` is a
    function of ``w``, so a threshold carried to another ``w`` means nothing. The stamp
    carries the file's sha256 (the trainer writes it to config.json)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"class thresholds file {str(p)!r} does not exist")
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if d.get("schema") != CLASS_THRESHOLD_SCHEMA:
        raise ValueError(f"{p.name}: schema {d.get('schema')!r} != {CLASS_THRESHOLD_SCHEMA!r}")
    if tuple(d.get("classes") or ()) != tuple(CLASS_KEYS):
        raise ValueError(f"{p.name}: class order {d.get('classes')!r} != {CLASS_KEYS}")
    lg = [float(v) for v in d.get("tau_phat_logit") or ()]
    if len(lg) != N_FINE_CLASSES or any(not math.isfinite(v) for v in lg):
        raise ValueError(f"{p.name}: need {N_FINE_CLASSES} finite tau_phat_logit values, got {lg}")
    pr = d.get("tau_phat_prob")
    if pr is not None:
        pr = [float(v) for v in pr]
        if len(pr) != N_FINE_CLASSES or any(
                abs(1.0 / (1.0 + math.exp(-a)) - b) > 1e-9 for a, b in zip(lg, pr)):
            raise ValueError(f"{p.name}: tau_phat_prob is not the sigmoid of tau_phat_logit")
    cwv = d.get("class_weight_values")
    if cwv is not None and class_weight is not None:
        have = torch.as_tensor(class_weight).detach().to("cpu", torch.float32).reshape(-1)
        want = torch.tensor([float(v) for v in cwv], dtype=torch.float32)
        if tuple(have.shape) != tuple(want.shape) or not bool(
                torch.allclose(have, want, rtol=1e-6, atol=0.0)):
            raise ValueError(
                f"{p.name}: these thresholds were fitted under class weights "
                f"{want.tolist()}, the model's frozen weights are {have.tolist()}: "
                f"p_hat = softmax(z - log w) depends on w, so they do not transfer")
    stamp = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
             "schema": CLASS_THRESHOLD_SCHEMA, "tau_phat_logit": lg,
             "provenance": d.get("provenance")}
    return torch.tensor(lg, dtype=torch.float32), stamp


class MapHiresMissingInput(ValueError):
    """The branch is built but an input it needs did not arrive."""


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapHiresConfig:
    """The branch's declared shape. Every IMAGE-side shape comes from the trunk, every
    BEV-side shape from the DECLARED extent (``x_max_m``, ``y_half_m``).

    ⛔ ``w_map_hires == 0`` is not a legal config: a branch with no live weight
    must not be BUILT (its parameters would enter the optimiser and the checkpoint
    and train on nothing -- the ``PerceptionBranchConfig`` rule, verbatim)."""

    w_map_hires: float
    #: ⭐⭐ THE EXTENT (SPEC_REFCV7 §11.2, A6): ``x`` in [0, x_max_m) ahead, ``y`` in
    #: [-y_half_m, +y_half_m). The lift grid, the decoder's output, the metric bands and
    #: the planner pool's crop all derive from it (``--map-hires-x-max-m`` /
    #: ``--map-hires-y-half-m``; the value comes from the GT-coverage census).
    x_max_m: float = 60.0
    y_half_m: float = 16.0
    lift_cell_m: float = LIFT_CELL_M
    stride: int = 8
    heights_m: tuple = HEIGHTS_M
    d_lift: int = 64
    d_model: int = 64
    #: +-(1 + 2 * 31) = +-63 cells at 0.25 m = +-15.75 m, the 0.5 m encoder's
    #: +-31 cells x 0.5 m = +-15.5 m: the SAME metric context, at twice the density.
    dilations: tuple = (1, 2, 4, 8, 16)
    d_up: int = 32
    norm_groups: int = 8
    n_classes: int = N_FINE_CLASSES
    #: recompute the encoder and the refine in backward (same function, less memory).
    grad_ckpt: bool = False
    #: ⭐ G-HYG / G-DVB: the DECLARED home of ``--map-hires-class-weights`` -- the sha256
    #: of the weights file the loss reads ("" = unweighted). The built branch carries
    #: it, config.json stamps it, and G-DVB compares it with the file argv names.
    class_weights_sha256: str = ""
    #: ⭐⭐ THE DECISION RULE, DECLARED (map-signal audit D1, accepted by the Master
    #: Mind 2026-09-26). Under a class-weighted CE the softmax learns q_c ∝ w_c P(c|x),
    #: so the RAW argmax is a rare-class detector (lane wins at a ~5 % posterior under
    #: median-frequency weights). ``"prior_corrected"`` = ``argmax_c (z_c - log w_c)``
    #: with the SAME frozen weights the loss used -- the calibrated decision, and the
    #: DEFAULT; ``"raw"`` = ``argmax_c z_c``, logged beside it as a diagnostic. With
    #: uniform (or no) weights the two are identical.
    decision_rule: str = "prior_corrected"
    #: ⭐ NEW-2 R2 (SPEC_REFCV7 A12; the map-signal audit's §9 lever 2, form (b)): THE 0.1 m
    #: NEAR-RANGE LIFT. ``> 0``: the stride-8 map is ALSO lifted at the 10 cm label cell over
    #: ``x`` in ``[0, near_lift_x_m)``, full width, and ADDED to the 10 cm decoder's upsampled
    #: input on those rows (:class:`NearLiftSkip`). MAP-ONLY: the shared 0.25 m encoder and the
    #: planner's pooled BEV are untouched. ``0`` (the DEFAULT) builds nothing: the branch is
    #: NEW-2 as landed, parameter for parameter. (``--map-hires-near-lift-m``.)
    near_lift_x_m: float = 0.0
    #: ⭐ NEW-2 R3 (SPEC_REFCV7 §20, A15; §9 lever 3, THE DECODER, stacked on the near lift):
    #: residual :class:`NearRefineBlock` s (3x3 dilation 2 -> GN -> GELU -> 3x3 dilation 4,
    #: last conv zero-initialised) on the NEAR rows at 0.1 m, after the near lift is added.
    #: Needs ``near_lift_x_m > 0``. ``0`` (the DEFAULT) builds nothing: R2 as landed.
    #: (``--map-hires-near-refine-blocks``.)
    near_refine_blocks: int = 0

    def __post_init__(self) -> None:
        if str(self.decision_rule) not in DECISION_RULES:
            raise ValueError(f"decision_rule {self.decision_rule!r} not in "
                             f"{DECISION_RULES}")
        if not float(self.w_map_hires) > 0.0 or not math.isfinite(float(self.w_map_hires)):
            raise ValueError(
                f"MapHiresConfig with w_map_hires {self.w_map_hires!r}: a map-hires "
                f"branch with no live weight would add parameters to the optimiser "
                f"and the checkpoint while training on nothing. Do not build it.")
        ext = self.extent                                   # validates x_max / y_half
        if int(self.n_classes) != N_FINE_CLASSES:
            raise ValueError(f"n_classes {self.n_classes} != {N_FINE_CLASSES}")
        if int(self.stride) != 8:
            raise ValueError("the map-hires lift samples the STRIDE-8 map (§6.2)")
        if abs(float(self.lift_cell_m) - LIFT_CELL_M) > 1e-12:
            raise ValueError(f"lift_cell_m {self.lift_cell_m} != {LIFT_CELL_M} (§6.2: the "
                             f"lift is 0.25 m, the planner pool is exactly 2 x 2 of it)")
        ext.grid(float(self.lift_cell_m))                    # tiles exactly
        for n in ("d_lift", "d_model", "d_up"):
            if int(getattr(self, n)) % int(self.norm_groups):
                raise ValueError(f"{n} {getattr(self, n)} must divide by "
                                 f"norm_groups {self.norm_groups}")
        if not self.dilations or any(int(d) < 1 for d in self.dilations):
            raise ValueError(f"dilations must be >= 1, got {self.dilations}")
        nx = float(self.near_lift_x_m)
        if not math.isfinite(nx) or nx < 0.0:
            raise ValueError(f"near_lift_x_m {self.near_lift_x_m!r} must be a finite value "
                             f">= 0 (0 = no near lift)")
        if nx > 0.0:
            k = nx / NEAR_LIFT_STEP_M
            if abs(k - round(k)) > 1e-9:
                raise ValueError(f"near_lift_x_m {nx} is not a multiple of "
                                 f"{NEAR_LIFT_STEP_M} m (it must tile the 0.25 m lift, the "
                                 f"0.1 m label and the 0.5 m planner grids exactly)")
            if nx > float(self.x_max_m) + 1e-9:
                raise ValueError(f"near_lift_x_m {nx} reaches past the map extent "
                                 f"({self.x_max_m} m ahead)")
        nb = self.near_refine_blocks
        if isinstance(nb, bool) or int(nb) != nb or not 0 <= int(nb) <= NEAR_REFINE_MAX_BLOCKS:
            raise ValueError(f"near_refine_blocks {nb!r} must be an integer in "
                             f"[0, {NEAR_REFINE_MAX_BLOCKS}]")
        if int(nb) > 0 and not nx > 0.0:
            raise ValueError("near_refine_blocks > 0 with near_lift_x_m 0: the block refines "
                             "the NEAR rows, and without the near lift there are none")

    @property
    def extent(self) -> MapExtent:
        return MapExtent(float(self.x_max_m), float(self.y_half_m))

    @property
    def lift_grid(self) -> BEVGrid:
        return self.extent.grid(float(self.lift_cell_m))

    @property
    def out_hw(self) -> tuple[int, int]:
        """The 10 cm logits' grid = the extent's label grid (derived, never typed)."""
        return tuple(self.extent.fine_shape)

    @property
    def band_keys(self) -> tuple[str, ...]:
        return self.extent.band_keys

    @property
    def near_rows(self) -> int:
        """10 cm rows the near lift covers (0 = none): ``near_lift_x_m / 0.1``."""
        return int(round(float(self.near_lift_x_m) / FINE_CELL_M))

    @property
    def receptive_field_m(self) -> float:
        return (1 + 2 * sum(int(d) for d in self.dilations)) * float(self.lift_cell_m)

    def as_dict(self) -> dict:
        return {"w_map_hires": float(self.w_map_hires),
                "x_max_m": float(self.x_max_m), "y_half_m": float(self.y_half_m),
                "lift_cell_m": float(self.lift_cell_m),
                "lift_grid_hw": list(self.lift_grid.shape), "stride": int(self.stride),
                "heights_m": list(self.heights_m), "d_lift": int(self.d_lift),
                "d_model": int(self.d_model), "dilations": list(self.dilations),
                "d_up": int(self.d_up), "norm_groups": int(self.norm_groups),
                "n_classes": int(self.n_classes), "out_hw": list(self.out_hw),
                "out_cell_m": FINE_CELL_M, "band_keys": list(self.band_keys),
                "grad_ckpt": bool(self.grad_ckpt),
                "class_weights_sha256": str(self.class_weights_sha256),
                "decision_rule": str(self.decision_rule),
                "receptive_field_m": self.receptive_field_m,
                # NEW-2 R2 (A12): 0 = no near lift (the NEW-2 branch as landed)
                "near_lift_x_m": float(self.near_lift_x_m),
                "near_lift_rows": int(self.near_rows),
                # NEW-2 R3 (A15): 0 = no near refine block (R2 as landed)
                "near_refine_blocks": int(self.near_refine_blocks)}


def declared_extent(args) -> MapExtent:
    """The extent argv DECLARES (``--map-hires-x-max-m`` / ``--map-hires-y-half-m``).
    Unset = the extent SPEC_REFCV7 §12 (A7) fixed, 100 m x +-30 m (the census rule's
    selection) -- never the ``/2`` grid by accident. ⚠️ ``MapHiresConfig``'s own
    defaults stay the ``/2`` grid (a module-neutral default); the trainer always
    passes the declared extent explicitly."""
    x = getattr(args, "map_hires_x_max_m", None)
    y = getattr(args, "map_hires_y_half_m", None)
    return MapExtent(float(EXTENT_REFCV7.x_max_m if x is None else x),
                     float(EXTENT_REFCV7.y_half_m if y is None else y))


# --------------------------------------------------------------------------- #
# the lift: BEVLift, projection first                                          #
# --------------------------------------------------------------------------- #
class BEVLiftProjectFirst(BEVLift):
    """:class:`BEVLift` with the 1x1 projection applied BEFORE the sampling.

    ``BEVLift.forward`` = ``proj(concat_z(valid_z * sample_z(fmap)))`` with
    ``proj.weight [d_out, Z*C]`` in height-major order. Sampling (bilinear, border
    padding) and the validity mask are linear per pixel, and a 1x1 conv is linear
    per pixel, so this equals ``sum_z valid_z * sample_z(W_z fmap) + bias`` with
    ``W_z = proj.weight[:, z*C:(z+1)*C]`` -- the same parameters (``proj``,
    ``unobserved``; the ``state_dict`` keys are BEVLift's), a different order of
    two linear maps. Pinned against ``BEVLift.forward`` by a test.
    """

    def forward(self, fmap: Tensor, grid: Tensor, valid: Tensor) -> Tensor:
        B, C, h, w = fmap.shape
        if C != self.d_in:
            raise ValueError(f"feature map has {C} channels, the lift expects {self.d_in}")
        if self.feat_hw is not None and (h, w) != self.feat_hw:
            raise ValueError(f"feature map is {h}x{w}, the lift geometry was built for "
                             f"{self.feat_hw[0]}x{self.feat_hw[1]} (wrong stride?)")
        if grid.dim() != 5 or grid.shape[0] != B or grid.shape[1] != self.n_heights \
                or grid.shape[-1] != 2:
            raise ValueError(f"grid must be [B={B}, Z={self.n_heights}, X, Y, 2], got "
                             f"{tuple(grid.shape)}")
        _, Z, X, Y, _ = grid.shape
        if tuple(valid.shape) != (B, Z, X, Y):
            raise ValueError(f"valid must be {(B, Z, X, Y)}, got {tuple(valid.shape)}")
        D = self.d_out
        # [D, Z*C] -> [Z*D, C]: row z*D + o = W[o, z*C : (z+1)*C]
        wz = self.proj.weight.reshape(D, Z, C).permute(1, 0, 2).reshape(Z * D, C, 1, 1)
        g = F.conv2d(fmap, wz)                                         # [B, Z*D, h, w]
        g = g.reshape(B * Z, D, h, w)
        gr = torch.where(valid.unsqueeze(-1), grid.to(g.dtype),
                         torch.zeros_like(grid, dtype=g.dtype))
        s = F.grid_sample(g, gr.reshape(B * Z, X, Y, 2), mode="bilinear",
                          padding_mode="border", align_corners=False)  # [B*Z, D, X, Y]
        s = s.reshape(B, Z, D, X, Y) * valid.to(g.dtype).unsqueeze(2)
        out = s.sum(dim=1)
        if self.proj.bias is not None:
            out = out + self.proj.bias.view(1, -1, 1, 1)
        unobs = (~valid.any(dim=1)).to(out.dtype).unsqueeze(1)
        return out + unobs * self.unobserved.view(1, -1, 1, 1)


class HiresLiftGeometryBank(LiftGeometryBank):
    """:class:`LiftGeometryBank` at stride 8 / 0.25 m over the DECLARED extent, with a
    BOUNDED cache.

    The geometry is the parent's, built by the parent's ``build_lift_geometry``
    call; only the cache is an LRU of ``max_cache`` clips (0.25 m costs 1.11 MB
    per clip at 60 m x +-16 m and 6.2 ms to build, MEASURED on the dev box)."""

    def __init__(self, extr_by_clip: dict, *, frame, cfg: MapHiresConfig,
                 equalize_bottom_rows: int = 0, max_cache: int = 512):
        super().__init__(extr_by_clip, frame=frame, stride=int(cfg.stride),
                         heights_m=tuple(cfg.heights_m), grid=cfg.lift_grid,
                         equalize_bottom_rows=equalize_bottom_rows)
        self.max_cache = int(max_cache)
        if self.max_cache < 1:
            raise ValueError("max_cache must be >= 1")
        self._cache = OrderedDict()

    def geometry(self, ep_id: int) -> tuple:
        k = int(ep_id)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        hit = super().geometry(k)                  # builds + stores in self._cache
        self._cache.move_to_end(k)
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return hit


# --------------------------------------------------------------------------- #
# encoder and refinement                                                       #
# --------------------------------------------------------------------------- #
class HiresBEVEncoder(nn.Module):
    """``[B, d_lift, X4, Y4] -> [B, d_model, X4, Y4]``: the 0.5 m encoder's recipe
    (3x3 stem + GN + GELU, the SAME ``_DilatedBlock`` at each dilation) at 0.25 m.
    Stride 1 everywhere; the grid is asserted on every forward.

    ⭐ Its output is SHARED (A6): the map decoder (:class:`HiresRefine`) and the
    planner's pooled BEV (``refcv6_perception_branch.PlannerBEVPool``) both read it."""

    def __init__(self, cfg: MapHiresConfig):
        super().__init__()
        self.cfg = cfg
        self.grid_hw = tuple(cfg.lift_grid.shape)
        g = int(cfg.norm_groups)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.d_lift), int(cfg.d_model), 3, padding=1, bias=False),
            nn.GroupNorm(g, int(cfg.d_model)), nn.GELU())
        self.blocks = nn.ModuleList(
            [_DilatedBlock(int(cfg.d_model), int(d), g) for d in cfg.dilations])

    def forward(self, x: Tensor) -> Tensor:
        if tuple(x.shape[2:]) != self.grid_hw:
            raise ValueError(f"lifted map is {tuple(x.shape[2:])}, the encoder was "
                             f"built for {self.grid_hw}")
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class HiresRefine(nn.Module):
    """The MAP decoder: ``[B, d_model, X4, Y4] -> [B, 8, X10, Y10]`` -- a 1x1 + GN to
    ``d_up``, bilinear x2.5, two 3x3 convs at 10 cm, a 1x1 to the class logits.

    ⭐ ``align_corners=False`` is METRICALLY EXACT here: output row ``j`` samples
    input coordinate ``(j + 0.5) * 0.1/0.25 - 0.5``, i.e. x = ``(j + 0.5) * 0.1 m``
    on the 0.25 m grid whose cell ``i`` is centred at ``(i + 0.5) * 0.25 m`` -- for
    every extent, because both grids tile the SAME extent from x = 0."""

    def __init__(self, cfg: MapHiresConfig):
        super().__init__()
        self.out_hw = tuple(int(v) for v in cfg.out_hw)
        d, g = int(cfg.d_up), int(cfg.norm_groups)
        self.inp = nn.Sequential(nn.Conv2d(int(cfg.d_model), d, 1, bias=False),
                                 nn.GroupNorm(g, d))
        self.conv1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g, d)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, d)
        self.act = nn.GELU()
        self.cls = nn.Conv2d(d, int(cfg.n_classes), 1)

    def forward(self, x: Tensor, near: Tensor | None = None, near_refine=None,
                near_block_zero_input: bool = False) -> Tensor:
        x = self.inp(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        if near is not None:
            # NEW-2 R2 (A12): the 0.1 m near lift, ADDED on its rows before the 10 cm convs
            n = int(near.shape[2])
            if tuple(near.shape[:2]) != tuple(x.shape[:2]) or int(near.shape[3]) != int(x.shape[3]) \
                    or n > int(x.shape[2]):
                raise ValueError(f"near-lift skip {tuple(near.shape)} does not fit the "
                                 f"decoder input {tuple(x.shape)}")
            top = x[:, :, :n] + near
            for blk in (near_refine or ()):         # NEW-2 R3 (A15): the near refine block(s)
                top = blk(top, zero_input=near_block_zero_input)
            x = torch.cat([top, x[:, :, n:]], dim=2)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return self.cls(x)


def _fine_to_coarse_index(n: int, ratio: float, device) -> Tensor:
    """Fractional COARSE index of the ``n`` FINE cell centres: fine cell ``a`` is centred at
    ``(a + 0.5) * fine``, i.e. at coarse index ``(a + 0.5) * fine/coarse - 0.5`` (cell centres
    on both grids; the decoder's ``align_corners=False`` upsample uses the same map).
    ⛔ The naive ``a * fine/coarse`` is off by 0.3 coarse cells (7.5 cm) -- a regression
    arm in the tests."""
    return (torch.arange(int(n), device=device, dtype=torch.float32) + 0.5) * float(ratio) - 0.5


def _near_valid(bad_weight: Tensor) -> Tensor:
    """A fine cell is valid iff NO invalid coarse neighbour carries interpolation weight
    (conservative). ⛔ A majority rule would sample a sanitised-zero coordinate into a
    'valid' cell -- a regression arm in the tests."""
    return bad_weight < 1e-6


def derive_near_geometry(grid: Tensor, valid: Tensor, *, near_rows: int, out_w: int,
                         lift_cell_m: float = LIFT_CELL_M,
                         fine_cell_m: float = FINE_CELL_M) -> tuple[Tensor, Tensor]:
    """The 0.25 m lift geometry -> the SAME camera's geometry at the 0.1 m cell centres of
    rows ``[0, near_rows)`` x all ``out_w`` columns: ``([B, Z, near_rows, out_w, 2],
    [B, Z, near_rows, out_w] bool)``.

    ⭐ DERIVED, not rebuilt: the sampling coordinates are a SMOOTH function of the BEV
    position for a fixed height (the rig -> cylinder projection), so they are interpolated
    bilinearly at the fine cell centres -- ``x_a = (a + 0.5) * 0.1`` sits at fractional
    coarse row ``(a + 0.5) * 0.4 - 0.5`` (the SAME metric alignment as the decoder's
    ``align_corners=False`` upsample) -- with LINEAR EXTRAPOLATION one coarse cell past
    each border. No new geometry crosses the forward, the hook or the eval loader; the
    error against the EXACT 0.1 m geometry is pinned by tests
    (``test_map_head_hires.py``).
    ⛔ Validity is CONSERVATIVE: a fine cell is valid only if EVERY coarse neighbour that
    carries interpolation weight is valid (an invalid coarse cell's coordinate is a
    sanitised 0, which must never leak into a valid sample)."""
    if grid.dim() != 5 or grid.shape[-1] != 2 or tuple(valid.shape) != tuple(grid.shape[:4]):
        raise ValueError(f"grid must be [B, Z, X, Y, 2] and valid [B, Z, X, Y]; got "
                         f"{tuple(grid.shape)} / {tuple(valid.shape)}")
    B, Z, X4, Y4, _ = grid.shape
    r = float(fine_cell_m) / float(lift_cell_m)
    if int(near_rows) < 1 or int(out_w) != int(round(Y4 / r)):
        raise ValueError(f"near_rows {near_rows} / out_w {out_w} do not tile a "
                         f"{X4} x {Y4} lift grid at {fine_cell_m} m")
    last = (int(near_rows) - 0.5) * r - 0.5              # the last fine row, coarse units
    R = max(2, min(X4, int(math.floor(last)) + 2))       # coarse rows that carry weight
    c = grid[:, :, :R].permute(0, 1, 4, 2, 3).reshape(B * Z, 2, R, Y4).float()
    v = valid[:, :, :R].reshape(B * Z, 1, R, Y4).float()

    def pad_lin(t: Tensor) -> Tensor:                    # 1 cell, linear extrapolation
        t = torch.cat([2 * t[:, :, :1] - t[:, :, 1:2], t, 2 * t[:, :, -1:] - t[:, :, -2:-1]], 2)
        return torch.cat([2 * t[..., :1] - t[..., 1:2], t,
                          2 * t[..., -1:] - t[..., -2:-1]], 3)

    def pad_rep(t: Tensor) -> Tensor:                    # validity: replicate the border
        return F.pad(t, (1, 1, 1, 1), mode="replicate")
    cp, vp = pad_lin(c), pad_rep(v)
    Hp, Wp = R + 2, Y4 + 2
    dev = grid.device
    rows = _fine_to_coarse_index(int(near_rows), r, dev) + 1.0     # +1: the padded border
    cols = _fine_to_coarse_index(int(out_w), r, dev) + 1.0
    gy = (2.0 * rows / (Hp - 1) - 1.0).view(-1, 1).expand(-1, int(out_w))
    gx = (2.0 * cols / (Wp - 1) - 1.0).view(1, -1).expand(int(near_rows), -1)
    pos = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B * Z, -1, -1, -1)
    g = F.grid_sample(cp, pos, mode="bilinear", align_corners=True)          # [BZ, 2, n, w]
    bad = F.grid_sample(1.0 - vp, pos, mode="bilinear", align_corners=True)  # [BZ, 1, n, w]
    ok = _near_valid(bad).reshape(B, Z, int(near_rows), int(out_w))
    g = g.reshape(B, Z, 2, int(near_rows), int(out_w)).permute(0, 1, 3, 4, 2)
    g = torch.where(ok.unsqueeze(-1), g, torch.zeros_like(g)).to(grid.dtype)
    return g.contiguous(), ok


class NearLiftSkip(nn.Module):
    """⭐ NEW-2 R2 (SPEC_REFCV7 A12): the stride-8 map lifted at the 10 cm label cell over
    ``x`` in ``[0, near_lift_x_m)``, full width -> ``[B, d_up, near_rows, W10]``, which
    :class:`HiresRefine` ADDS to its bilinear-upsampled input on those rows.

    WHY (the early G-MAP-OVERFIT record, map-signal audit §9 lever 2): a decoder that sees
    only 0.25 m class fractions reaches non-drivable edge IoU 0.328 on the gate's 16 frames
    (GT-only oracle; 1.000 at 0.1 m), below the 0.50 bar; the 3,000-step MAIN arm FINDS edges
    (0.2 m tolerance F1 0.906) but misplaces them by ~one cell (IoU 0.430), flat over range --
    the 0.25 m lift cell, not the stride, is the limit.
    ⛔ MAP-ONLY: nothing here reaches the shared 0.25 m encoder or the planner's pool.
    ⛔ ZERO-INITIALISED (projection, bias, unobserved embedding): at step 0 the branch IS the
    NEW-2 branch's function, and every other parameter initialises exactly as without it
    (the module is built last)."""

    def __init__(self, cfg: MapHiresConfig, *, d_image: int, image_hw: tuple):
        super().__init__()
        if int(cfg.near_rows) < 1:
            raise ValueError("NearLiftSkip with near_lift_x_m 0: nothing to lift")
        self.rows = int(cfg.near_rows)
        self.out_w = int(cfg.out_hw[1])
        self.lift = BEVLiftProjectFirst(d_in=int(d_image), d_out=int(cfg.d_up),
                                        n_heights=len(cfg.heights_m),
                                        feat_hw=(int(image_hw[0]), int(image_hw[1])))
        with torch.no_grad():
            self.lift.proj.weight.zero_()
            if self.lift.proj.bias is not None:
                self.lift.proj.bias.zero_()
            self.lift.unobserved.zero_()

    def forward(self, fmap: Tensor, grid: Tensor, valid: Tensor) -> Tensor:
        g, v = derive_near_geometry(grid, valid, near_rows=self.rows, out_w=self.out_w)
        return self.lift(fmap, g, v)


class NearRefineBlock(nn.Module):
    """⭐ NEW-2 R3 (SPEC_REFCV7 §20, A15; the map-signal audit's §9 lever 3, THE DECODER): one
    residual block at 0.1 m on the NEAR rows, applied after the near lift is added and before
    the decoder's own convs: ``x + conv_d4(GELU(GN(conv_d2(x))))``.

    WHY (the early records): the classes that lag are the thin LINES, not the rare or the
    low-weight ones -- at 1,000 steps lane / edge read CE ratio .280 / .290 against crosswalk
    .067 at nearly lane's weight (1.07 vs 0.94), and in the 3,000-step run crosswalk crosses
    its bar at step 200, lane at 1,200, edge never. The decoder's own 0.1 m receptive field is
    0.5 m (two 3x3 convs); dilations 2 and 4 add ~1.3 m, so line evidence can connect along a
    line. ⛔ MAP-ONLY. ⛔ The last conv is ZERO-initialised: at step 0 the branch is R2's
    function, and the block is built last so every other parameter initialises as in R2.
    ``zero_input`` (the G-MAP-OVERFIT ``near_block_zeros`` regression arm ONLY): the residual
    branch reads zeros, so the block can add nothing but what its constants give."""

    DILATIONS: tuple = (2, 4)

    def __init__(self, cfg: MapHiresConfig):
        super().__init__()
        d, g = int(cfg.d_up), int(cfg.norm_groups)
        d1, d2 = self.DILATIONS
        self.c1 = nn.Conv2d(d, d, 3, padding=d1, dilation=d1, bias=False)
        self.n1 = nn.GroupNorm(g, d)
        self.c2 = nn.Conv2d(d, d, 3, padding=d2, dilation=d2, bias=False)
        with torch.no_grad():
            self.c2.weight.zero_()

    def forward(self, x: Tensor, zero_input: bool = False) -> Tensor:
        h = torch.zeros_like(x) if zero_input else x
        return x + self.c2(F.gelu(self.n1(self.c1(h))))


class MapHiresBranch(nn.Module):
    """``fmap_s8`` (+ this batch's 0.25 m lift geometry) -> 10 cm map logits AND the
    shared 0.25 m BEV features the planner's pool reads."""

    def __init__(self, cfg: MapHiresConfig, *, d_image: int, image_hw: tuple):
        super().__init__()
        self.cfg = cfg
        self.d_image = int(d_image)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.lift = BEVLiftProjectFirst(d_in=self.d_image, d_out=int(cfg.d_lift),
                                        n_heights=len(cfg.heights_m),
                                        feat_hw=self.image_hw)
        self.encoder = HiresBEVEncoder(cfg)
        self.refine = HiresRefine(cfg)
        # ⭐ NEW-2 R2 (A12): built LAST, so every parameter above initialises exactly as in
        # a branch without it; None = the NEW-2 branch as landed.
        self.near = (NearLiftSkip(cfg, d_image=self.d_image, image_hw=self.image_hw)
                     if int(cfg.near_rows) > 0 else None)
        # ⭐ NEW-2 R3 (A15): built LAST (after the near lift), so every parameter above
        # initialises exactly as in R2; None = R2 as landed.
        self.near_refine = (nn.ModuleList([NearRefineBlock(cfg)
                                           for _ in range(int(cfg.near_refine_blocks))])
                            if int(cfg.near_refine_blocks) > 0 else None)

    def param_breakdown(self) -> dict:
        def n(m):
            return int(sum(p.numel() for p in m.parameters()))
        out = {"lift": n(self.lift), "encoder": n(self.encoder), "refine": n(self.refine)}
        if self.near is not None:
            out["near"] = n(self.near)
        if self.near_refine is not None:
            out["near_refine"] = n(self.near_refine)
        out["total"] = n(self)
        return out

    def _refine_near(self, feats: Tensor, src: Tensor, grid: Tensor,
                     valid: Tensor, near_block_zeros: bool = False) -> Tensor:
        return self.refine(feats, near=self.near(src, grid, valid),
                           near_refine=self.near_refine,
                           near_block_zero_input=bool(near_block_zeros))

    def _run(self, fn, x: Tensor) -> Tensor:
        if self.cfg.grad_ckpt and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, fmap_s8: Tensor | None, grid: Tensor | None,
                valid: Tensor | None, *, near_source: Tensor | None = None,
                near_block_zeros: bool = False) -> dict:
        """⛔ Vision tensors only: no parameter here can carry a label.

        ``near_source`` (NEW-2 R2): what the 0.1 m near lift samples instead of ``fmap_s8``.
        ⛔ ONLY the G-MAP-OVERFIT ``near_zeros`` regression arm passes it (zeros); the
        trainer never does.
        ``near_block_zeros`` (NEW-2 R3): the near refine block's residual branch reads zeros.
        ⛔ ONLY the G-MAP-OVERFIT ``near_block_zeros`` regression arm passes it."""
        if fmap_s8 is None:
            raise MapHiresMissingInput(
                "[map-hires] ⛔ fmap_s8 is None: the 10 cm branch is built but the "
                "trunk's stride-8 map did not reach it. The tap is off "
                "(TimmResNetTrunk.enable_s8_tap), or the forward did not pass "
                "`fmap_s8` through -- the D-REFCV6-F3-WHITELIST class. Refusing: "
                "a skipped branch would train nothing while config.json stamps it.")
        if grid is None or valid is None:
            raise MapHiresMissingInput(
                "[map-hires] ⛔ no 0.25 m lift geometry reached the branch; the map "
                "would be predicted from features sampled at no camera at all")
        if fmap_s8.shape[1] != self.d_image:
            raise ValueError(f"[map-hires] fmap_s8 has {fmap_s8.shape[1]} channels, "
                             f"the branch was built for {self.d_image}")
        bev = self.lift(fmap_s8, grid, valid)
        feats = self._run(self.encoder, bev)
        if near_block_zeros and self.near_refine is None:
            raise ValueError("near_block_zeros given to a branch with no near refine block")
        if self.near is None:
            if near_source is not None:
                raise ValueError("near_source given to a branch with no near lift")
            logits = self._run(self.refine, feats)
        else:
            src = fmap_s8 if near_source is None else near_source
            if tuple(src.shape) != tuple(fmap_s8.shape):
                raise ValueError(f"near_source {tuple(src.shape)} != fmap_s8 "
                                 f"{tuple(fmap_s8.shape)}")
            if self.cfg.grad_ckpt and torch.is_grad_enabled():
                logits = torch.utils.checkpoint.checkpoint(
                    self._refine_near, feats, src, grid, valid, bool(near_block_zeros),
                    use_reentrant=False)
            else:
                logits = self._refine_near(feats, src, grid, valid, bool(near_block_zeros))
        if tuple(logits.shape[1:]) != (int(self.cfg.n_classes),) + tuple(self.cfg.out_hw):
            raise RuntimeError(f"map-hires logits {tuple(logits.shape)} are not "
                               f"[B, 8, {self.cfg.out_hw[0]}, {self.cfg.out_hw[1]}]")
        return {"map_hires_logits": logits,
                "map_hires_lift_valid": valid.any(dim=1),        # [B, X4, Y4]
                "map_hires_bev": feats}                          # [B, d_model, X4, Y4]


def build_map_hires_branch(model, cfg: MapHiresConfig) -> MapHiresBranch:
    """Read the stride-8 shape off the BUILT trunk (its tap must be on)."""
    enc = model.core.encoder
    if not bool(getattr(enc, "s8_tap", False)):
        raise SystemExit(
            "[map-hires] ⛔ the trunk's stride-8 tap is OFF. Call "
            "model.core.encoder.enable_s8_tap() first (--trunk timm with frozen BN); "
            "the legacy REF-C encoder has no stride-8 map at all.")
    return MapHiresBranch(cfg, d_image=int(enc.s8_dim), image_hw=tuple(enc.s8_shape))


# --------------------------------------------------------------------------- #
# the loss                                                                     #
# --------------------------------------------------------------------------- #
def _ce_targets(codes: Tensor) -> Tensor:
    """uint8 codes -> long targets. ⛔ The ONE place the not-seen code is handled:
    it is passed through as ``NOT_SEEN_CODE`` and ignored by the loss."""
    return codes.long()


def lift_valid_to_fine(lift_valid: Tensor, out_hw: tuple | None = None) -> Tensor:
    """``[B, X4, Y4]`` bool (0.25 m) -> ``[B, X10, Y10]`` bool (0.1 m), each 10 cm cell
    taking the 0.25 m cell its CENTRE falls in (``nearest-exact``: fine row j ->
    coarse row ``floor((j + 0.5) * 0.4)``; plain ``nearest`` would be ``floor(0.4 j)``,
    off by one coarse row on 1 fine row in 5). ``out_hw`` defaults to the same extent
    at 0.1 m (``X4 * 2.5``, ``Y4 * 2.5``, which must be whole)."""
    if lift_valid.dim() != 3 or lift_valid.dtype != torch.bool:
        raise ValueError(f"lift_valid must be [B, X, Y] bool, got "
                         f"{tuple(lift_valid.shape)} {lift_valid.dtype}")
    if out_hw is None:
        r = LIFT_CELL_M / FINE_CELL_M
        hw = [lift_valid.shape[1] * r, lift_valid.shape[2] * r]
        if any(abs(v - round(v)) > 1e-9 for v in hw):
            raise ValueError(f"lift grid {tuple(lift_valid.shape[1:])} does not tile a "
                             f"10 cm grid")
        out_hw = (int(round(hw[0])), int(round(hw[1])))
    x = lift_valid[:, None].to(torch.float32)
    return F.interpolate(x, size=tuple(int(v) for v in out_hw),
                         mode="nearest-exact")[:, 0] > 0.5


def hires_map_ce(logits: Tensor, codes: Tensor, *, class_weight: Tensor | None = None,
                 lift_valid: Tensor | None = None) -> dict:
    """Weighted CE over the 8 classes on supervised cells. -> ``{loss, n_cells,
    n_cells_seen}``.

    ``logits`` ``[B, 8, H, W]`` and ``codes`` ``[B, H, W]`` uint8 (255 = not seen) on
    the SAME 10 cm grid (the declared extent's); ``lift_valid`` optional ``[B, H, W]``
    bool narrows the supervised cells to those the camera reaches at this instant
    (refcv6's D-3 rule, ``map_loss_row``). A batch with NO supervised cell returns an
    exact 0 attached to the graph, with ``n_cells`` 0."""
    if logits.dim() != 4 or int(logits.shape[1]) != N_FINE_CLASSES:
        raise ValueError(f"logits must be [B, 8, H, W], got {tuple(logits.shape)}")
    if tuple(codes.shape) != (logits.shape[0],) + tuple(logits.shape[2:]) \
            or codes.dtype != torch.uint8:
        raise ValueError(f"codes must be [B, H, W] uint8 on the logits' grid "
                         f"{tuple(logits.shape[2:])}, got {tuple(codes.shape)} {codes.dtype}")
    tgt = _ce_targets(codes.to(logits.device))
    n_seen = int((codes != NOT_SEEN_CODE).sum())
    if lift_valid is not None:
        if tuple(lift_valid.shape) != tuple(codes.shape) or lift_valid.dtype != torch.bool:
            raise ValueError(f"lift_valid must be {tuple(codes.shape)} bool")
        tgt = torch.where(lift_valid.to(tgt.device), tgt,
                          torch.full_like(tgt, NOT_SEEN_CODE))
    sup = tgt != NOT_SEEN_CODE
    n = int(sup.sum())
    w = None
    if class_weight is not None:
        w = class_weight.to(device=logits.device, dtype=logits.dtype)
        if tuple(w.shape) != (N_FINE_CLASSES,):
            raise ValueError(f"class_weight must be [8], got {tuple(w.shape)}")
    if n == 0:
        return {"loss": logits.sum() * 0.0, "n_cells": 0, "n_cells_seen": n_seen}
    loss = F.cross_entropy(logits, tgt, weight=w, ignore_index=NOT_SEEN_CODE,
                           reduction="mean")
    return {"loss": loss, "n_cells": n, "n_cells_seen": n_seen}


def map_hires_loss_row(logits: Tensor, codes: Tensor, *, class_weight=None,
                       lift_valid_025: Tensor | None = None,
                       with_metrics: bool = True,
                       decision_rule: str = "raw",
                       class_thresholds=None) -> dict:
    """:func:`hires_map_ce` + the counts and the PER-CLASS signal a log row carries.

    ⭐ Both counts, always (``n_map_hires_cells`` supervised, ``..._seen`` before
    the lift narrowing): a 0.0 loss on 0 cells and on 3 M cells are opposite
    findings. ⛔ NEVER DRIVABLE-ONLY (the refcv6 lesson: 4,621 metrics rows carried
    map quality for drivable alone and the thin-class collapse was invisible for
    38,000 steps): with ``with_metrics`` the row carries, for ALL 8 classes x EVERY
    20 m band of the extent, the labelled-cell count, the class's share of the loss,
    the gradient norm on its logit channel and the argmax intersection / union
    (:func:`per_class_signal`, keys :func:`per_class_key`), on the SAME cells as the
    loss. :func:`derived_per_class` turns a row -- or an eval mean of rows -- into
    pooled IoUs and loss shares.

    ``class_thresholds`` (OPT-IN, refcv7 diagnostics F1/F2; ``None`` = the row is
    byte-for-byte what it was): the ``[8]`` logit thresholds of :func:`decide_masks`. Given,
    the row ALSO carries the ``PER_CLASS_STATS_THR`` keys -- the thresholded rule's
    intersection / union / prediction counts and the 2-cell (0.2 m) boundary-tolerant
    counts -- and ``decision_rule="class_threshold"`` becomes legal. No existing key or
    value changes."""
    lv = (None if lift_valid_025 is None
          else lift_valid_to_fine(lift_valid_025, tuple(logits.shape[2:])))
    r = hires_map_ce(logits, codes, class_weight=class_weight, lift_valid=lv)
    row = {"loss": r["loss"], "n_map_hires_cells": float(r["n_cells"]),
           "n_map_hires_cells_seen": float(r["n_cells_seen"])}
    if with_metrics and r["n_cells"]:
        # ⛔ the prior-corrected rule with NO weights is uniform weights, stated
        cw = (class_weight if class_weight is not None or decision_rule == "raw"
              else torch.ones(N_FINE_CLASSES))
        sig = per_class_signal(logits, codes, class_weight=cw, lift_valid=lv,
                               decision_rule=decision_rule,
                               class_thresholds=class_thresholds)
        row.update(per_class_log_values(sig))
    return row


# --------------------------------------------------------------------------- #
# per-class instrumentation (the Master Mind's NEW-2 addition 1, 2026-09-26)   #
# --------------------------------------------------------------------------- #
#: short, stable class names for log keys, in code order 0..7
CLASS_KEYS: tuple[str, ...] = ("nocls", "drivable", "lane", "crosswalk", "arrow",
                               "edge", "hatched", "sidewalk")
#: the x bands of the ``/2`` extent (fine rows 0-199 / 200-399 / 400-599). A larger
#: extent adds one key per further 20 m (``MapHiresConfig.band_keys``).
BAND_KEYS: tuple[str, ...] = EXTENT_V2.band_keys
#: ⚠️ ONE place for every per-class log key. The map-signal audit's LOGGING_SPEC_MAP10
#: adopted these spellings; renaming happens here and nowhere else.
LOG_PREFIX = "map_hires_"
#: the per-class x band statistics a log row carries (LOGGING_SPEC_MAP10 §2):
#: ``inter``/``union`` under the DECLARED rule, ``interraw``/``unionraw`` under the raw
#: argmax (the diagnostic). ⚠️ ``ce`` is NOT logged (the harness reads it).
PER_CLASS_STATS: tuple[str, ...] = ("n", "lc", "gn", "gno", "inter", "union",
                                    "interraw", "unionraw")


def band_keys_for_rows(n_rows: int) -> tuple[str, ...]:
    """The band keys of a 10 cm grid with ``n_rows`` rows along x: every 200 rows =
    20 m from the nearest, a partial last band keeping its true edge."""
    n = int(n_rows)
    if n < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    out, r0 = [], 0
    while r0 < n:
        r1 = min(r0 + BAND_ROWS_FINE, n)
        out.append(band_key(r0 * FINE_CELL_M, r1 * FINE_CELL_M))
        r0 = r1
    return tuple(out)


def band_keys_of(model) -> tuple[str, ...]:
    """The band keys of the BUILT 10 cm branch (the ``/2`` bands without one)."""
    br = getattr(model, "_map_hires", None)
    return tuple(br.cfg.band_keys) if br is not None else BAND_KEYS


def per_class_key(stat: str, c: int, b, prefix: str = "") -> str:
    """``<prefix>map_hires_<stat>_<class>_<band>``, e.g. ``map_hires_iou_lane_0_20``.
    ``b`` is a band KEY (``"60_80"``) or an index into the ``/2`` :data:`BAND_KEYS`."""
    bk = b if isinstance(b, str) else BAND_KEYS[int(b)]
    return f"{prefix}{LOG_PREFIX}{stat}_{CLASS_KEYS[c]}_{bk}"


def is_exact_log_key(key: str) -> bool:
    """The log keys a trainer row keeps UNROUNDED: the 10 cm loss itself
    (``map_hires``) and every ``map_hires_*`` statistic.

    LOGGING_SPEC_MAP10 §6 check 2 compares ``sum(map_hires_lc_*)`` with ``map_hires``
    at 1e-5 RELATIVE; a loss rounded to 5 dp fails that as soon as the loss is below
    ~0.5, and a rare class's gradient norm (~1e-6) rounds to the 0.0 that reads as
    "no signal". Every other key keeps the trainer's 5 dp rounding."""
    return key == LOG_PREFIX[:-1] or key.startswith(LOG_PREFIX)


def per_class_signal(logits: Tensor, codes: Tensor, *, class_weight=None,
                     lift_valid: Tensor | None = None,
                     decision_rule: str = "raw", class_thresholds=None) -> dict:
    """PER CLASS x PER BAND signal of the 10 cm loss on one batch. PURE: no autograd.

    For the loss ``L = sum_i w_{y_i} CE_i / W`` (``W = sum_i w_{y_i}`` over
    supervised cells; :func:`hires_map_ce`'s rule) the logit gradient is analytic,
    ``dL/dz_{i,c} = w_{y_i} (p_{i,c} - [c = y_i]) / W`` on supervised cells and 0
    elsewhere. Returns ``[8, n_bands]`` float64 tensors (class, 20 m band of the
    logits' own grid, :func:`band_keys_for_rows` -> ``out["band_keys"]``):

    * ``n``     -- supervised cells LABELLED c in band b;
    * ``lc``    -- their contribution to ``L`` (``sum lc == L`` over all c, b);
    * ``gn``    -- ``|| dL/dz_{., c} ||_2`` over ALL supervised cells in band b (the
                   gradient norm on logit CHANNEL c);
    * ``gno``   -- the same norm restricted to cells labelled c (the pull UP);
    * ``inter`` / ``union`` -- intersection / union for class c in band b under the
                   DECLARED ``decision_rule`` (:func:`decide`);
    * ``interraw`` / ``unionraw`` -- the same under the RAW argmax (the diagnostic);
    * ``ce``    -- the UNWEIGHTED CE summed over cells labelled c (``ce / n`` is the
                   class's mean per-cell CE; G-MAP-OVERFIT's "every class learned").

    plus ``loss`` (= ``L``), ``W`` and ``n_supervised``. ``lift_valid`` ``[B, H, W]``
    narrows the supervised cells exactly as the loss does. ⚠️ ``decision_rule
    "prior_corrected"`` with ``class_weight=None`` is REFUSED (:func:`decide`).

    ``class_thresholds`` (OPT-IN; ``None`` = nothing below changes): the ``[8]`` logit
    thresholds of the ``class_threshold`` rule (:func:`decide_masks`, MULTI-LABEL).
    ``decision_rule="class_threshold"`` REQUIRES them (and ``class_weight``) -- a loud
    ValueError otherwise -- and then ``inter`` / ``union`` are of that rule's masks. Given
    with ANY rule, the signal also carries ``[8, n_bands]`` ``interthr`` / ``unionthr`` /
    ``predthr`` and the 2-cell tolerant ``tppthr2`` / ``tpgthr2`` of the thresholded
    masks (:data:`PER_CLASS_STATS_THR`): ``tpp`` = predicted cells with a GT cell within
    2 cells (Chebyshev), ``tpg`` = GT cells with a predicted cell within 2 cells, both on
    the supervised cells, exactly the diagnostics' ``tolerant_counts``."""
    with torch.no_grad():
        lg = logits.detach().float()
        B, C, Hh, Ww = lg.shape
        if C != N_FINE_CLASSES:
            raise ValueError(f"logits must be [B, 8, H, W], got {tuple(lg.shape)}")
        if tuple(codes.shape) != (B, Hh, Ww):
            raise ValueError(f"codes {tuple(codes.shape)} are not on the logits' grid "
                             f"{(B, Hh, Ww)}")
        tgt = _ce_targets(codes.to(lg.device))
        if lift_valid is not None:
            tgt = torch.where(lift_valid.to(tgt.device), tgt,
                              torch.full_like(tgt, NOT_SEEN_CODE))
        sup = tgt != NOT_SEEN_CODE                                 # [B,H,W]
        y = torch.where(sup, tgt, torch.zeros_like(tgt)).clamp(0, C - 1)
        w = (torch.ones(C, device=lg.device) if class_weight is None
             else class_weight.detach().to(device=lg.device, dtype=torch.float32))
        wi = w[y] * sup.float()                                     # [B,H,W]
        W = wi.sum()
        logp = torch.log_softmax(lg, dim=1)
        ce = -logp.gather(1, y[:, None])[:, 0]                      # [B,H,W]
        onehot = F.one_hot(y, C).permute(0, 3, 1, 2).float() * sup[:, None].float()
        keys = band_keys_for_rows(Hh)
        # row -> band membership [H, nb]: every 200 rows, the last band partial
        rb = torch.arange(Hh, device=lg.device) // BAND_ROWS_FINE
        M = F.one_hot(rb, len(keys)).to(torch.float32)

        def bands(t):                                # [B,C,H,W] -> [C, nb]
            return t.sum(dim=(0, 3)) @ M
        Wd = W if float(W) > 0 else torch.ones((), device=lg.device)
        g = wi[:, None] * (logp.exp() * sup[:, None].float() - onehot) / Wd

        def onehot_pred(codes_hw):
            return F.one_hot(codes_hw, C).permute(0, 3, 1, 2).float() \
                * sup[:, None].float()
        pred_raw = onehot_pred(lg.argmax(dim=1))
        thr_mask = None
        if decision_rule == "class_threshold" and class_thresholds is None:
            raise ValueError("decision_rule 'class_threshold' needs class_thresholds "
                             "(--map-hires-class-thresholds): there is no rule without them")
        if class_thresholds is not None:
            thr_mask = decide_masks(lg, class_weight, class_thresholds, sup)    # bool, sup-only
        if decision_rule == "class_threshold":
            pred = thr_mask.float()                      # MULTI-LABEL, as the diagnostics scored it
        else:
            pred = (pred_raw if decision_rule == "raw" else
                    onehot_pred(decide(lg, decision_rule, class_weight)))
        out = {"n": bands(onehot),
               "lc": bands(onehot * (wi * ce)[:, None]) / Wd,
               "gn": bands(g * g).sqrt(),
               "gno": bands(g * g * onehot).sqrt(),
               "inter": bands(onehot * pred),
               "union": bands(((onehot + pred) > 0).float()),
               "interraw": bands(onehot * pred_raw),
               "unionraw": bands(((onehot + pred_raw) > 0).float()),
               "ce": bands(onehot * ce[:, None])}
        if thr_mask is not None:                         # F1/F2: the thresholded rule's counts
            gt_m = onehot > 0.5
            k_tol = MONITOR_TOL_CELLS
            out["interthr"] = bands((thr_mask & gt_m).float())
            out["unionthr"] = bands((thr_mask | gt_m).float())
            out["predthr"] = bands(thr_mask.float())
            out["tppthr%d" % k_tol] = bands((thr_mask & dilate_chebyshev(gt_m, k_tol)).float())
            out["tpgthr%d" % k_tol] = bands((gt_m & dilate_chebyshev(thr_mask, k_tol)).float())
        out = {k: v.double() for k, v in out.items()}
        out["loss"] = float((wi * ce).sum() / Wd) if float(W) > 0 else 0.0
        out["W"] = float(W)
        out["n_supervised"] = int(sup.sum())
        out["band_keys"] = keys
    return out


def per_class_log_values(sig: dict, prefix: str = "") -> dict:
    """The ``[8, n_bands]`` statistics as flat ``{key: float}`` -- ONE device->host
    copy. The band keys are the signal's own (the logits' grid)."""
    keys = tuple(sig.get("band_keys") or BAND_KEYS)
    # F1/F2: the thresholded-rule statistics ride along ONLY when the signal carries them
    stats = PER_CLASS_STATS + (PER_CLASS_STATS_THR if PER_CLASS_STATS_THR[0] in sig else ())
    flat = torch.stack([sig[s] for s in stats]).cpu().tolist()
    out = {}
    for si, s in enumerate(stats):
        for c in range(N_FINE_CLASSES):
            for b, bk in enumerate(keys):
                out[per_class_key(s, c, bk, prefix)] = float(flat[si][c][b])
    return out


def derived_per_class(row: dict, prefix: str = "", band_keys=BAND_KEYS) -> dict:
    """Pooled IoU and loss SHARE per class x band from a row's raw keys.

    Works on one training row AND on an eval row that is the MEAN of per-batch rows:
    ``mean(inter) / mean(union)`` IS ``sum(inter) / sum(union)`` (the pooled IoU),
    and ``mean(lc_cb) / sum mean(lc)`` is the pooled loss share.

    ⭐ ALL 8 x n_bands keys are ALWAYS emitted (SPEC_REFCV7 A3's G-DVB check reads the
    key list; §11.2: every band, never dropped): an IoU whose union is 0 -- the class
    absent from GT AND prediction -- is UNDEFINED and written as ``None`` (JSON
    null), never 0 and never NaN; its raw ``inter`` / ``union`` = 0 stay in the row
    beside it. ``band_keys``: the DECLARED extent's (:func:`band_keys_of`).

    ``iou`` is the DECLARED decision rule's; ``iouraw`` the raw argmax's (the
    diagnostic, SPEC_REFCV7 A4 (1): both rules are logged, the declared one gates)."""
    out = {}
    lcs = {(c, bk): row.get(per_class_key("lc", c, bk, prefix))
           for c in range(N_FINE_CLASSES) for bk in band_keys}
    tot = sum(v for v in lcs.values() if v is not None)
    for c in range(N_FINE_CLASSES):
        for bk in band_keys:
            for s_iou, s_i, s_u in (("iou", "inter", "union"),
                                    ("iouraw", "interraw", "unionraw")):
                u = row.get(per_class_key(s_u, c, bk, prefix))
                i = row.get(per_class_key(s_i, c, bk, prefix))
                out[per_class_key(s_iou, c, bk, prefix)] = (
                    float(i) / float(u) if (u is not None and i is not None and u > 0)
                    else None)
            lc = lcs[(c, bk)]
            out[per_class_key("lshare", c, bk, prefix)] = (
                float(lc) / float(tot) if (lc is not None and tot > 0) else None)
            # F1/F2: ``iouthr`` = the thresholded rule's pooled IoU, ``iou2thr`` = its
            # boundary-tolerant IoU at MONITOR_TOL_CELLS cells -- only when the row carries
            # the thresholded counts (a default row emits exactly the keys it always did)
            if per_class_key(PER_CLASS_STATS_THR[0], c, bk, prefix) in row:
                g = lambda st: row.get(per_class_key(st, c, bk, prefix))     # noqa: E731
                u, i = g("unionthr"), g("interthr")
                out[per_class_key("iouthr", c, bk, prefix)] = (
                    float(i) / float(u) if (u is not None and i is not None and u > 0)
                    else None)
                vals = [g("predthr"), g("n"), g(PER_CLASS_STATS_THR[3]), g(PER_CLASS_STATS_THR[4])]
                out[per_class_key("iou%dthr" % MONITOR_TOL_CELLS, c, bk, prefix)] = (
                    None if any(v is None for v in vals) else tolerant_iou_from_counts(*vals))
    return out


def missing_per_class_keys(row: dict, prefix: str = "",
                           stats=("iou", "lshare"), band_keys=BAND_KEYS) -> list:
    """The per-class x band keys a row LACKS -- SPEC_REFCV7 A3's G-DVB logging check.
    Empty = all 8 classes x every declared band present for every stat. ⛔ A
    refcv6-shaped row (``map_iou_drivable`` at 0.5 m and nothing per class) returns
    all 48 keys of the default two stats as missing on the ``/2`` bands."""
    return [per_class_key(s, c, bk, prefix) for s in stats
            for c in range(N_FINE_CLASSES) for bk in band_keys
            if per_class_key(s, c, bk, prefix) not in row]


def per_class_liveness(row: dict, *, min_cells: float, prefix: str = "",
                       band_keys=BAND_KEYS) -> dict:
    """SPEC_REFCV7 A3's G-LIVE per-class check on one row: every (class, band) with at
    least ``min_cells`` labelled 10 cm cells must have a finite, NON-ZERO loss
    contribution (``lc``), a non-zero gradient on its logit channel (``gn``) and a
    non-zero pull from its own cells (``gno``).

    ⚠️ ``gno`` is the discriminating one: a class whose weight is 0 still has a
    non-zero ``gn`` (every OTHER cell pushes its logit down) but ``lc = gno = 0``.
    Returns ``{"checked": n, "violations": [...]}``; the ``min_cells`` value is the
    audit prereg's M."""
    viol, checked = [], 0
    for c in range(N_FINE_CLASSES):
        for bk in band_keys:
            n = row.get(per_class_key("n", c, bk, prefix))
            if n is None or n < float(min_cells):
                continue
            checked += 1
            for s in ("lc", "gn", "gno"):
                v = row.get(per_class_key(s, c, bk, prefix))
                if v is None or not math.isfinite(float(v)) or float(v) <= 0.0:
                    viol.append({"class": CLASS_KEYS[c], "band": bk,
                                 "stat": s, "value": v, "n": n})
    return {"checked": checked, "violations": viol}


# --------------------------------------------------------------------------- #
# the class weights                                                            #
# --------------------------------------------------------------------------- #
def load_class_weights(path, extent: MapExtent | None = None) -> tuple[Tensor, dict]:
    """Read the weights JSON -> ``([8] float32 tensor, stamp)``.

    ⛔ REFUSES: a different schema, not 8 weights, a class order that is not
    ``FINE_CLASSES``, a non-finite or non-positive weight, a weight above the
    pre-registered clip (25), a file marked ``dry_run`` (the dev-box DRY RUN on
    eval files is NOT the launch weights: those are computed on Thor's TRAIN GT), and
    -- when ``extent`` is given -- weights counted on ANOTHER extent (class
    frequencies change with the range; a file without an ``extent`` key was counted
    on the ``/2`` grid, 60 m x +-16 m). The stamp carries the file's sha256, which
    the trainer writes to config.json."""
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if d.get("schema") != CLASS_WEIGHT_SCHEMA:
        raise ValueError(f"{p.name}: schema {d.get('schema')!r} != {CLASS_WEIGHT_SCHEMA!r}")
    if tuple(d.get("classes") or ()) != tuple(FINE_CLASSES):
        raise ValueError(f"{p.name}: class order {d.get('classes')!r} != {FINE_CLASSES}")
    w = [float(v) for v in d.get("weights") or ()]
    if len(w) != N_FINE_CLASSES:
        raise ValueError(f"{p.name}: {len(w)} weights, need {N_FINE_CLASSES}")
    if any(not math.isfinite(v) or v <= 0.0 for v in w):
        raise ValueError(f"{p.name}: weights must be finite and > 0, got {w}")
    if max(w) > CLASS_WEIGHT_CLIP_MAX + 1e-9:
        raise ValueError(f"{p.name}: max weight {max(w)} > the pre-registered clip "
                         f"{CLASS_WEIGHT_CLIP_MAX}")
    if bool(d.get("dry_run", False)):
        raise ValueError(f"{p.name} is a DRY RUN ({d.get('dry_run_note', '')}) -- not "
                         f"the launch weights. Compute them on the TRAIN split's GT.")
    fe = d.get("extent") or {"x_max_m": EXTENT_V2.x_max_m, "y_half_m": EXTENT_V2.y_half_m}
    file_extent = MapExtent(float(fe["x_max_m"]), float(fe["y_half_m"]))
    if extent is not None and file_extent != extent:
        raise ValueError(
            f"{p.name}: weights counted on the {file_extent.x_max_m} m x "
            f"+-{file_extent.y_half_m} m extent, the DECLARED extent is "
            f"{extent.x_max_m} m x +-{extent.y_half_m} m: class frequencies change "
            f"with range. Recompute them at the declared extent.")
    stamp = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
             "weights": w, "split": d.get("split"),
             "inputs_sha256": d.get("inputs_sha256"),
             "definition": d.get("definition"),
             # SPEC_REFCV7 §13 (A8): the gate reads these two -- refcv7 uses `sqrt_mf`
             "definition_id": d.get("definition_id"),
             "pre_registered": d.get("pre_registered"),
             "extent": {"x_max_m": float(file_extent.x_max_m),
                        "y_half_m": float(file_extent.y_half_m)}}
    return torch.tensor(w, dtype=torch.float32), stamp


# --------------------------------------------------------------------------- #
# instruments                                                                  #
# --------------------------------------------------------------------------- #
def grad_reach_report_hires(model, branch: MapHiresBranch | None = None) -> dict:
    """``grad_abs_sum`` per part AFTER a backward, including the trunk's stride-8
    stage (the tap's parameters are the trunk's own ``layer1``/``layer2``...).

    ⚠️ Under A6 the ``lift`` and ``encoder`` are SHARED by the map and the planner's
    pooled BEV, so their reach includes the planner / box / tactical gradients; the
    ``refine`` is map-only, the pool (``ga_bev_pool``) is planner-only."""
    br = branch if branch is not None else getattr(model, "_map_hires", None)
    rep = {}
    if br is not None:
        parts = [("lift", br.lift), ("encoder", br.encoder), ("refine", br.refine)]
        if getattr(br, "near", None) is not None:
            parts.append(("near", br.near))          # NEW-2 R2 (A12): map-only
        if getattr(br, "near_refine", None) is not None:
            parts.append(("near_refine", br.near_refine))  # NEW-2 R3 (A15): map-only
        for name, m in parts:
            s, n, g = grad_abs_sum(m)
            rep[name] = {"grad_abs_sum": s, "n_params": n, "n_params_with_grad": g}
    enc = getattr(getattr(model, "core", None), "encoder", None)
    s8m = getattr(enc, "s8_module", None)
    if enc is not None and s8m is not None:
        s, n, g = grad_abs_sum(enc.net[s8m])
        rep["trunk_s8_stage"] = {"module": s8m, "grad_abs_sum": s, "n_params": n,
                                 "n_params_with_grad": g}
    return rep


def built_state(model) -> dict:
    """What G-DVB compares to argv: is the branch built, on which extent, fed by a
    live stride-8 tap, with which class weights, and where the planner's BEV comes
    from."""
    br = getattr(model, "_map_hires", None)
    enc = getattr(getattr(model, "core", None), "encoder", None)
    cw = getattr(model, "_map_hires_class_weight_stamp", None)
    pb = getattr(model, "_perception", None)
    return {"built": br is not None,
            "extent": None if br is None else [float(br.cfg.x_max_m),
                                               float(br.cfg.y_half_m)],
            "out_hw": None if br is None else list(br.cfg.out_hw),
            "lift_grid_hw": None if br is None else list(br.cfg.lift_grid.shape),
            "w_map_hires": float(getattr(model, "_w_map_hires", 0.0) or 0.0),
            "s8_tap": bool(getattr(enc, "s8_tap", False)),
            "s8_dim": getattr(enc, "s8_dim", None),
            "class_weights_sha256": None if not cw else cw.get("sha256"),
            "params": None if br is None else br.param_breakdown()["total"],
            "bev_source": (None if pb is None
                           else str(getattr(getattr(pb, "cfg", None), "bev_source", None))),
            "stride16_lift_built": pb is not None and getattr(pb, "lift", None) is not None,
            "bev_pool_built": pb is not None and getattr(pb, "bev_pool", None) is not None,
            "near_lift_x_m": None if br is None else float(br.cfg.near_lift_x_m),
            "near_lift_built": br is not None and getattr(br, "near", None) is not None,
            "near_refine_blocks": None if br is None else int(br.cfg.near_refine_blocks),
            "near_refine_built": (0 if br is None or getattr(br, "near_refine", None) is None
                                  else len(br.near_refine))}


# --------------------------------------------------------------------------- #
# G-DVB (tanitad/train/declared_vs_built.py): the NEW-2 levers                 #
# --------------------------------------------------------------------------- #
#: dest -> G-DVB kind. ⛔ Registered by the TRAINER (:func:`register_dvb_levers`),
#: NEVER at this module's import: the registry is pinned two-way against the
#: trainer's parser, and a tree whose parser lacks these flags must not carry them.
DVB_KINDS: dict = {"map_hires": "built", "w_map_hires": "loss",
                   "map_hires_class_weights": "built",
                   "map_hires_decision_rule": "built",
                   "map_hires_x_max_m": "built", "map_hires_y_half_m": "built",
                   "map_hires_grad_ckpt": "built",
                   "bev_source": "built", "bev_planner_crop_m": "built",
                   # NEW-2 R2 (A12)
                   "map_hires_near_lift_m": "built",
                   # NEW-2 R3 (A15)
                   "map_hires_near_refine_blocks": "built"}

#: ``--bev-planner-crop-m``'s default and the ONLY value the refcv6 consumers are built
#: for (SPEC_REFCV7 §12 item 2): the planner's 60 m x +-16 m window.
PLANNER_CROP_DEFAULT: tuple = (float(GRID_DEFAULT.x_fwd_m), float(GRID_DEFAULT.y_half_m))


def declared_grad_ckpt(args) -> bool:
    """``--map-hires-grad-ckpt {on,off}``; unset = ON under ``--map-hires on``
    (SPEC_REFCV7 §12 item 4: the 10 cm decoder's saved activations at b16 are
    ~21.4 GB unchecked at the A7 extent)."""
    v = getattr(args, "map_hires_grad_ckpt", None)
    if v is None:
        return str(getattr(args, "map_hires", "off") or "off") == "on"
    return str(v) == "on"


def declared_near_lift_m(args) -> float:
    """``--map-hires-near-lift-m``; unset = 0.0 (no near lift: NEW-2 as landed)."""
    v = getattr(args, "map_hires_near_lift_m", None)
    return 0.0 if v is None else float(v)


def declared_near_refine_blocks(args) -> int:
    """``--map-hires-near-refine-blocks``; unset = 0 (no near refine block: R2 as landed)."""
    v = getattr(args, "map_hires_near_refine_blocks", None)
    return 0 if v is None else int(v)


def _mm(lever, declared, built, read_from, why=""):
    from tanitad.train.declared_vs_built import Mismatch
    return Mismatch(lever, declared, built, read_from, why)


def _dvb_core(model):
    return getattr(model, "core", model)


def dvb_check_map_hires(model, args) -> list:
    """``--map-hires``: the branch, the trunk tap, the pass-through and the 0.25 m
    bank are BUILT iff ``on``, at the DECLARED extent's shapes."""
    on = str(getattr(args, "map_hires", "off") or "off") == "on"
    br = getattr(model, "_map_hires", None)
    core = _dvb_core(model)
    enc = getattr(core, "encoder", None)
    out = []
    if (br is not None) != on:
        out.append(_mm("--map-hires", on, br is not None, "model._map_hires is not None",
                       "a 10 cm branch no flag asked for" if br is not None else
                       "the 10 cm branch was never built"))
    tap = bool(getattr(enc, "s8_tap", False))
    if tap != on:
        out.append(_mm("--map-hires", on, tap, "core.encoder.s8_tap",
                       "the trunk's stride-8 tap disagrees with the branch"))
    if not on:
        return out
    passthrough = tuple(getattr(type(core), "DECODER_PASSTHROUGH", ()))
    if "fmap_s8" not in passthrough:
        out.append(_mm("--map-hires", "fmap_s8 in the forward output", "absent",
                       f"{type(core).__name__}.DECODER_PASSTHROUGH",
                       "the 10 cm loss would find no stride-8 map "
                       "(the D-REFCV6-F3-WHITELIST class)"))
    if br is None:
        return out
    ext = declared_extent(args)
    for name, want, got in (("out_hw", tuple(ext.fine_shape), tuple(br.cfg.out_hw)),
                            ("lift_grid", tuple(ext.grid(LIFT_CELL_M).shape),
                             tuple(br.cfg.lift_grid.shape)),
                            ("refine.out_hw", tuple(ext.fine_shape),
                             tuple(br.refine.out_hw)),
                            ("encoder.grid_hw", tuple(ext.grid(LIFT_CELL_M).shape),
                             tuple(br.encoder.grid_hw)),
                            ("stride", 8, int(br.cfg.stride))):
        if got != want:
            out.append(_mm("--map-hires", want, got, f"model._map_hires.{name}",
                           "the DECLARED extent's 10 cm head (SPEC_REFCV7 §6.2, §11.2)"))
    if int(getattr(br, "d_image", -1)) != int(getattr(enc, "s8_dim", -2) or -2):
        out.append(_mm("--map-hires", getattr(enc, "s8_dim", None), br.d_image,
                       "model._map_hires.d_image vs core.encoder.s8_dim",
                       "the branch was built for another stride-8 width"))
    bank = getattr(model, "_lift_bank_hires", None)
    if bank is None:
        out.append(_mm("--map-hires", "a 0.25 m / stride-8 lift bank", None,
                       "model._lift_bank_hires", "the branch would sample at no camera"))
    else:
        rows = int(getattr(args, "equalize_bottom_rows", 0) or 0)
        for name, want, got in (("stride", 8, int(bank.stride)),
                                ("cell_m", LIFT_CELL_M, float(bank.grid_spec.cell_m)),
                                ("grid_spec.shape", tuple(ext.grid(LIFT_CELL_M).shape),
                                 tuple(bank.grid_spec.shape)),
                                ("equalize_bottom_rows", rows,
                                 int(bank.equalize_bottom_rows))):
            if got != want:
                out.append(_mm("--map-hires", want, got, f"model._lift_bank_hires.{name}"))
    return out


def dvb_check_w_map_hires(model, args) -> list:
    """``--w-map-hires`` (a loss weight): the weight the LOSS reads, the head it
    supervises built iff it is > 0, and the built config's own copy."""
    w = float(getattr(args, "w_map_hires", 0.0) or 0.0)
    br = getattr(model, "_map_hires", None)
    got = getattr(model, "_w_map_hires", None)
    out = []
    try:
        ok = abs(float(got) - w) <= 1e-12
    except (TypeError, ValueError):
        ok = False
    if not ok:
        out.append(_mm("--w-map-hires", w, got, "model._w_map_hires (read by the loss)"))
    if (w > 0.0) != (br is not None):
        out.append(_mm("--w-map-hires", w > 0.0, br is not None,
                       "model._map_hires is not None",
                       "a built head with no live weight gets NO gradient, and a live "
                       "weight with no head supervises nothing"))
    if br is not None and abs(float(br.cfg.w_map_hires) - w) > 1e-12:
        out.append(_mm("--w-map-hires", w, float(br.cfg.w_map_hires),
                       "model._map_hires.cfg.w_map_hires"))
    return out


def dvb_check_class_weights(model, args) -> list:
    """``--map-hires-class-weights``: the loss's weight tensor is the FILE argv names --
    by sha256 on the built config, and value by value on the tensor."""
    path = getattr(args, "map_hires_class_weights", None)
    t = getattr(model, "_map_hires_class_weight", None)
    out = []
    if bool(path) != (t is not None):
        out.append(_mm("--map-hires-class-weights", path,
                       None if t is None else "a tensor",
                       "model._map_hires_class_weight",
                       "the loss would run unweighted (or weighted by no declared file)"))
        return out
    if not path:
        return out
    raw = Path(path).read_bytes()
    want_sha = hashlib.sha256(raw).hexdigest()
    br = getattr(model, "_map_hires", None)
    got_sha = None if br is None else str(br.cfg.class_weights_sha256)
    if got_sha != want_sha:
        out.append(_mm("--map-hires-class-weights", want_sha[:12], (got_sha or "")[:12],
                       "model._map_hires.cfg.class_weights_sha256 vs sha256(argv file)"))
    want_w = [float(v) for v in json.loads(raw.decode("utf-8")).get("weights") or ()]
    got_w = [float(v) for v in t.detach().cpu().reshape(-1).tolist()]
    if len(want_w) != len(got_w) or any(abs(a - b) > 1e-6 for a, b in zip(want_w, got_w)):
        out.append(_mm("--map-hires-class-weights", want_w, got_w,
                       "model._map_hires_class_weight (read by the loss)"))
    return out


def dvb_check_decision_rule(model, args) -> list:
    """``--map-hires-decision-rule``: the rule the in-run counts and every eval use is
    the one the BUILT branch config declares (``MapHiresConfig.decision_rule``)."""
    want = str(getattr(args, "map_hires_decision_rule", DECISION_RULES[0])
               or DECISION_RULES[0])
    br = getattr(model, "_map_hires", None)
    if br is None:
        return [] if want == DECISION_RULES[0] else [_mm(
            "--map-hires-decision-rule", want, "no 10 cm branch",
            "model._map_hires", "a decision rule for a branch that is not built")]
    got = str(getattr(br.cfg, "decision_rule", "<absent>"))
    return [] if got == want else [_mm("--map-hires-decision-rule", want, got,
                                       "model._map_hires.cfg.decision_rule")]


def dvb_check_extent(dest: str):
    """``--map-hires-x-max-m`` / ``--map-hires-y-half-m``: the BUILT branch config, its
    decoder's output grid and its lift bank's grid are the DECLARED extent's
    (SPEC_REFCV7 §11.2 item 4: "G-HYG and G-DVB check that the built grid equals the
    declared extent"). Without a branch, only the default is legal."""
    flag = "--" + dest.replace("_", "-")
    attr = dest[len("map_hires_"):]                      # x_max_m / y_half_m

    def chk(model, args) -> list:
        want = float(getattr(declared_extent(args), attr))
        br = getattr(model, "_map_hires", None)
        if br is None:
            explicit = getattr(args, dest, None)
            return [] if explicit is None else [_mm(
                flag, explicit, "no 10 cm branch", "model._map_hires",
                "an extent for a branch that is not built")]
        out = []
        got = float(getattr(br.cfg, attr))
        if abs(got - want) > 1e-9:
            out.append(_mm(flag, want, got, f"model._map_hires.cfg.{attr}"))
        ext = MapExtent(want if attr == "x_max_m" else float(br.cfg.x_max_m),
                        want if attr == "y_half_m" else float(br.cfg.y_half_m))
        if tuple(br.refine.out_hw) != tuple(ext.fine_shape):
            out.append(_mm(flag, list(ext.fine_shape), list(br.refine.out_hw),
                           "model._map_hires.refine.out_hw",
                           "the decoder predicts on another grid than the declared one"))
        bank = getattr(model, "_lift_bank_hires", None)
        if bank is not None:
            gs = bank.grid_spec
            gv = float(gs.x_fwd_m if attr == "x_max_m" else gs.y_half_m)
            if abs(gv - want) > 1e-9:
                out.append(_mm(flag, want, gv, f"model._lift_bank_hires.grid_spec.{attr}",
                               "the lift samples another extent than the declared one"))
        return out
    return chk


def dvb_check_bev_source(model, args) -> list:
    """``--bev-source`` (SPEC_REFCV7 §11.1, A6 option (c)): WHERE every consumer's BEV
    comes from. ``map_hires_pool``: the perception branch builds NO stride-16 lift and
    no 0.5 m encoder/head, its ``bev_pool`` reads the 10 cm branch's 0.25 m encoder
    (widths and source grid checked), the box memory reads that BEV, and no stride-16
    lift bank exists. ``s16_lift`` (refcv6): no pool, and never beside a 10 cm branch
    (two lifts). ⛔ The regression arm: a consumer still wired to a stride-16 lift."""
    # ⚠️ imported HERE, not at module level: `BEV_SOURCES` is part of the shared-file
    # edit to refcv6_perception_branch, and this module must import on a tree without it
    from tanitad.models.refcv6_perception_branch import BEV_SOURCES
    want = str(getattr(args, "bev_source", BEV_SOURCES[0]) or BEV_SOURCES[0])
    pb = getattr(model, "_perception", None)
    mh = getattr(model, "_map_hires", None)
    out = []
    if want not in BEV_SOURCES:
        return [_mm("--bev-source", want, None, "BEV_SOURCES", "not a declared source")]
    if pb is not None:
        got = str(getattr(getattr(pb, "cfg", None), "bev_source", "<absent>"))
        if got != want:
            out.append(_mm("--bev-source", want, got, "model._perception.cfg.bev_source"))
    s16_built = pb is not None and getattr(pb, "lift", None) is not None
    pool = None if pb is None else getattr(pb, "bev_pool", None)
    if want == "map_hires_pool":
        if mh is None:
            out.append(_mm("--bev-source", want, "no 10 cm branch", "model._map_hires",
                           "the pooled BEV has no source"))
        if s16_built:
            out.append(_mm("--bev-source", want, "a stride-16 lift is BUILT",
                           "model._perception.lift is None",
                           "A6: a consumer still wired to a stride-16 lift"))
        if pb is not None and getattr(pb, "map_branch", None) is not None:
            out.append(_mm("--bev-source", want, "a 0.5 m BEV encoder/map head is BUILT",
                           "model._perception.map_branch is None",
                           "A6: one BEV encoder, the 0.25 m one"))
        if getattr(model, "_lift_bank", None) is not None:
            out.append(_mm("--bev-source", want, "a stride-16 lift bank",
                           "model._lift_bank is None",
                           "A6: the forward's perception geometry is the 0.25 m bank's"))
        if pb is not None:
            if pool is None:
                out.append(_mm("--bev-source", want, "no pool",
                               "model._perception.bev_pool is not None",
                               "the consumers would read no BEV"))
            elif mh is not None:
                for name, w_, g_ in (("d_in", int(mh.cfg.d_model), int(pool.d_in)),
                                     ("src_grid", tuple(mh.cfg.lift_grid.shape),
                                      tuple(pool.src_grid.shape)),
                                     ("d_out", int(pb.bev_token_dim), int(pool.d_out))):
                    if w_ != g_:
                        out.append(_mm("--bev-source", w_, g_,
                                       f"model._perception.bev_pool.{name}",
                                       "the pool does not read the 10 cm branch's encoder "
                                       "at its own grid, or emits another width than the "
                                       "consumers were built for"))
            bm = getattr(pb, "box_mem", None)
            if bm is not None and not bool(getattr(bm, "use_bev", False)):
                out.append(_mm("--bev-source", want, "box memory without BEV",
                               "model._perception.box_mem.use_bev",
                               "box3d is a consumer of the pooled BEV (§11.1)"))
    else:
        if pool is not None:
            out.append(_mm("--bev-source", want, "a planner pool is BUILT",
                           "model._perception.bev_pool is None"))
        if mh is not None and s16_built:
            out.append(_mm("--bev-source", want, "a 10 cm branch AND a stride-16 lift",
                           "model._map_hires / model._perception.lift",
                           "A6: one lift; pass --bev-source map_hires_pool"))
    return out


def dvb_check_grad_ckpt(model, args) -> list:
    """``--map-hires-grad-ckpt``: the BUILT branch recomputes its encoder and decoder in
    backward iff declared (unset = on under ``--map-hires on``)."""
    want = declared_grad_ckpt(args)
    br = getattr(model, "_map_hires", None)
    if br is None:
        explicit = getattr(args, "map_hires_grad_ckpt", None)
        return [] if explicit is None else [_mm(
            "--map-hires-grad-ckpt", explicit, "no 10 cm branch", "model._map_hires",
            "checkpointing for a branch that is not built")]
    got = bool(br.cfg.grad_ckpt)
    return [] if got == want else [_mm("--map-hires-grad-ckpt", want, got,
                                       "model._map_hires.cfg.grad_ckpt")]


def dvb_check_planner_crop(model, args) -> list:
    """``--bev-planner-crop-m`` (SPEC_REFCV7 §12 item 2): the pool CROPS the declared
    window out of the map extent BEFORE pooling, and what it emits is the grid the
    consumers were built for (``GRID_DEFAULT``, 120 x 64 -> 30 x 16 tokens of 2 x 2 m).
    ⛔ The regression arm: an UNCROPPED pool over the 100 m x +-30 m extent emits a
    200 x 120 grid that every consumer would still accept silently (adaptive pooling
    to 30 x 16 tokens of 3.3 x 3.75 m; the coupling indexing the wrong cells)."""
    want = tuple(float(v) for v in (getattr(args, "bev_planner_crop_m", None)
                                    or PLANNER_CROP_DEFAULT))
    pb = getattr(model, "_perception", None)
    pool = None if pb is None else getattr(pb, "bev_pool", None)
    if pool is None:
        return [] if want == PLANNER_CROP_DEFAULT else [_mm(
            "--bev-planner-crop-m", list(want), "no planner pool",
            "model._perception.bev_pool", "a crop for a pool that is not built")]
    out = []
    got_cfg = tuple(float(v) for v in getattr(pb.cfg, "planner_crop_m", ()))
    if got_cfg != want:
        out.append(_mm("--bev-planner-crop-m", list(want), list(got_cfg),
                       "model._perception.cfg.planner_crop_m"))
    dg = pool.dst_grid
    got_dst = (float(dg.x_fwd_m), float(dg.y_half_m))
    if got_dst != want:
        out.append(_mm("--bev-planner-crop-m", list(want), list(got_dst),
                       "model._perception.bev_pool.dst_grid",
                       "the pool crops another window than the declared one"))
    if tuple(pool.dst_hw) != tuple(GRID_DEFAULT.shape):
        out.append(_mm("--bev-planner-crop-m", list(GRID_DEFAULT.shape), list(pool.dst_hw),
                       "model._perception.bev_pool.dst_hw",
                       "the consumers (BEV tokens, coupling, box memory) are built for "
                       "the refcv6 planner grid; an uncropped or re-cropped grid is "
                       "accepted by them SILENTLY"))
    return out


def dvb_check_near_lift(model, args) -> list:
    """``--map-hires-near-lift-m`` (NEW-2 R2, A12): the BUILT branch lifts the stride-8 map
    at 0.1 m over EXACTLY the declared rows, full width, into the 10 cm decoder ONLY (its
    width is the decoder's ``d_up``); ``0`` = no near lift. ⛔ The regression arm: a
    declared lever that is not built (or built at another range) reads as a real arm."""
    want = declared_near_lift_m(args)
    br = getattr(model, "_map_hires", None)
    if br is None:
        return [] if want == 0.0 else [_mm(
            "--map-hires-near-lift-m", want, "no 10 cm branch", "model._map_hires",
            "a near lift for a branch that is not built")]
    out = []
    got = float(br.cfg.near_lift_x_m)
    if got != want:
        out.append(_mm("--map-hires-near-lift-m", want, got,
                       "model._map_hires.cfg.near_lift_x_m"))
    nl = getattr(br, "near", None)
    if (nl is not None) != (want > 0.0):
        out.append(_mm("--map-hires-near-lift-m", want,
                       "built" if nl is not None else "not built", "model._map_hires.near",
                       "the near lift must exist iff it is declared"))
    if nl is not None:
        rows = int(round(want / FINE_CELL_M))
        if int(nl.rows) != rows or int(nl.out_w) != int(br.cfg.out_hw[1]):
            out.append(_mm("--map-hires-near-lift-m", [rows, int(br.cfg.out_hw[1])],
                           [int(nl.rows), int(nl.out_w)], "model._map_hires.near.rows/out_w",
                           "the near lift covers another window than the declared one"))
        if int(nl.lift.d_out) != int(br.cfg.d_up):
            out.append(_mm("--map-hires-near-lift-m", int(br.cfg.d_up), int(nl.lift.d_out),
                           "model._map_hires.near.lift.d_out",
                           "the skip must match the 10 cm decoder's width (map-only)"))
    return out


def dvb_check_near_refine(model, args) -> list:
    """``--map-hires-near-refine-blocks`` (NEW-2 R3, A15): the BUILT branch carries EXACTLY the
    declared number of near refine blocks, of the registered design (dilations 2 and 4, the
    decoder's width), on a branch that HAS the near lift; ``0`` = none. ⛔ The regression arm:
    a declared block that is not built reads as a real decoder arm."""
    want = declared_near_refine_blocks(args)
    br = getattr(model, "_map_hires", None)
    if br is None:
        return [] if want == 0 else [_mm(
            "--map-hires-near-refine-blocks", want, "no 10 cm branch", "model._map_hires",
            "a decoder block for a branch that is not built")]
    out = []
    got = int(br.cfg.near_refine_blocks)
    if got != want:
        out.append(_mm("--map-hires-near-refine-blocks", want, got,
                       "model._map_hires.cfg.near_refine_blocks"))
    blocks = getattr(br, "near_refine", None)
    n_built = 0 if blocks is None else len(blocks)
    if n_built != want:
        out.append(_mm("--map-hires-near-refine-blocks", want, n_built,
                       "model._map_hires.near_refine",
                       "the number of BUILT near refine blocks"))
    if want > 0 and getattr(br, "near", None) is None:
        out.append(_mm("--map-hires-near-refine-blocks", want, "no near lift",
                       "model._map_hires.near",
                       "the block refines the near rows, which only the near lift creates"))
    for i, blk in enumerate(blocks or ()):
        dil = (int(blk.c1.dilation[0]), int(blk.c2.dilation[0]))
        if dil != tuple(NearRefineBlock.DILATIONS) or int(blk.c1.in_channels) != int(br.cfg.d_up):
            out.append(_mm("--map-hires-near-refine-blocks",
                           [list(NearRefineBlock.DILATIONS), int(br.cfg.d_up)],
                           [list(dil), int(blk.c1.in_channels)],
                           f"model._map_hires.near_refine[{i}]",
                           "not the registered block (dilations 2 and 4 at the decoder width)"))
    return out


def register_dvb_levers(register=None) -> dict:
    """Register the NEW-2 levers in G-DVB's registry -> ``{dest: kind}``.

    ⛔ Called by the TRAINER at its import (it owns the parser the registry is pinned
    against), never here at import time: ``declared_vs_built.REGISTRY`` is a
    process-global, and a test run that imported this module on a tree without the
    flags would otherwise fail the two-way coverage pin."""
    if register is None:
        from tanitad.train.declared_vs_built import register
    checks = {"map_hires": dvb_check_map_hires, "w_map_hires": dvb_check_w_map_hires,
              "map_hires_class_weights": dvb_check_class_weights,
              "map_hires_decision_rule": dvb_check_decision_rule,
              "map_hires_x_max_m": dvb_check_extent("map_hires_x_max_m"),
              "map_hires_y_half_m": dvb_check_extent("map_hires_y_half_m"),
              "map_hires_grad_ckpt": dvb_check_grad_ckpt,
              "bev_source": dvb_check_bev_source,
              "bev_planner_crop_m": dvb_check_planner_crop,
              "map_hires_near_lift_m": dvb_check_near_lift,
              "map_hires_near_refine_blocks": dvb_check_near_refine}
    for dest, kind in DVB_KINDS.items():
        register(dest, kind, checks[dest])
    return dict(DVB_KINDS)
