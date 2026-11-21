"""WP-D — the REF-C BEV auxiliary HEAD and its loss. ⛔ TRAINING-ONLY.

⭐ **The question this exists to answer.** WP-A (`E-READOUT-CEILING-1`) MEASURED
that REF-C's feature map is a perfectly adequate *address space* (8x20 = 20
azimuth columns = 6.0°/column, retaining **71.6 %** of the 16x40 oracle ceiling)
but that a planning-trained trunk carries **no transferable agent localisation
at any resolution** (test AP 0.027-0.034 vs a 0.0325 marginal control).
⇒ a waypoint-indexed deformable attention (DiffusionDrive coupling (1)) built
today would index into a map with no agents in it. **This head puts agents into
the map**; the index is the NEXT work package, not this one.

⛔⛔ **REMOVABLE AT INFERENCE, PROVEN — NOT ASSERTED.**
``tests/test_bev_aux.py`` requires **bit-identical** planner outputs from a model
built WITH this head and one built WITHOUT it, and **bit-identical shared
parameters at initialisation**. Two design constraints make that true and both
are load-bearing:

1. ⛔ **The head is constructed LAST in ``RefCModel.__init__``.** Module
   construction draws from the global RNG, so inserting a module ANYWHERE
   earlier silently changes every subsequent module's initial weights — and the
   "aux on vs aux off" A/B would then differ in the *seed* as well as in the
   lever. That is a one-variable violation invisible in every log.
2. ⛔ **The head is called LAST in ``RefCModel.forward`` and consumes no RNG**
   (no dropout, no sampling), so it cannot perturb the decoder's draws.

The published precedent for the whole shape is PhyLatent's Physical State
Grounding — *"the physical targets supervise representation learning only during
training and are not required by the planner"* (`LIT-2`, banked `2608.05720`).

⛔⛔ **AND THE PROGRAMME HAS ALREADY RUN THIS EXPERIMENT ONCE, ON A DIFFERENT
STACK, AND IT FAILED — READ `E-DEC-18` BEFORE TUNING ANYTHING.** PSG (the same
labels, an 8-column azimuth target, on the v6/v7 world-model stack) produced
*"the largest environment gain the programme has measured"* and **destroyed the
predictor at every weight tested** (`E-DEC-18b`), with the damage done to the
**ENCODER** even when no gradient reached the predictor (`E-DEC-18c`). The
measured mechanism was scale: ``psg_enc``/``psg_pred`` sat at **~0.3-0.9** while
the objective ``o5_loss`` sat at **~0.03**, i.e. the aux term outweighed the
objective it was meant to support by **10-30x**. ⇒ :func:`assert_loss_parity`
below refuses a run whose weighted aux term is outside a pre-registered band
against the trajectory loss, **at step 0, before the GPU-days are spent**.
⚠️ REF-C has no predictor, so PSG's exact failure surface does not exist here —
but "an aux term that overwhelms the objective" does, and that is what is
guarded.

⛔ **No headline number from this head may be an ACCURACY.** MEASURED occupancy
is ~1-2 % of cells, so an all-zero predictor scores ~98 % accuracy.
:func:`bev_aux_metrics` reports AP / IoU / F1 and refuses to compute an accuracy;
:func:`constant_control_ap` returns the exact no-information value a control must
read.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["BEVAuxConfig", "BEVAuxHead", "bev_aux_loss", "bev_aux_metrics",
           "constant_control_ap", "assert_loss_parity", "HEAD_KINDS",
           "LOSS_PARITY_BAND"]

#: ``col`` — each feature-map COLUMN maps to a radial ray of range bins, with no
#: cross-column mixing. It exploits the cylindrical projection maximally (the
#: column already IS a bearing, so azimuth needs no learning) and is the CHEAP
#: FLOOR. ``xcol`` adds ONE self-attention block over the 20 column tokens, so
#: "does cross-column reasoning buy anything" is an ABLATION rather than an
#: assumption. ⚠️ Both are pre-registered arms; neither is a default to be
#: chosen after seeing a number.
HEAD_KINDS = ("col", "xcol")

#: The weighted-aux / trajectory-loss ratio band :func:`assert_loss_parity`
#: enforces at step 0. The lower bound stops a term that is present in
#: ``config.json`` and inert in the gradient (the ``w_agent`` defect verbatim);
#: the upper bound is the `E-DEC-18b` failure, whose MEASURED ratio was 10-30x.
LOSS_PARITY_BAND = (0.02, 3.0)


@dataclass
class BEVAuxConfig:
    """⛔ ``enable=False`` (the default) means the head is NEVER CONSTRUCTED —
    not constructed-and-disabled. A disabled-but-present module still consumes
    RNG at ``__init__`` and still lands in ``state_dict``, and both break the
    bit-identity the removability proof needs."""

    enable: bool = False
    kind: str = "col"
    n_rng: int = 24
    n_az: int = 20                 # MUST equal the encoder map's column count
    d_tok: int = 64                # per-token 1x1 compression before the ray MLP
    hidden: int = 256
    n_heads: int = 4               # xcol only
    w: float = 0.0                 # loss weight (0 with enable=True is REFUSED)
    #: BCE positive-class weight. ⭐ A pre-registered CONSTANT from the measured
    #: corpus base rate, never computed from the batch: a batch-derived weight
    #: makes the loss depend on batch composition, so two arms with identical
    #: flags optimise different objectives.
    pos_weight: float = 60.0
    occlusion: str = "mask"        # recorded here; applied in `data.bev_aux`
    #: ⛔ DELIBERATE REGRESSION when True: the head reads a DETACHED feature
    #: map, so it learns the target perfectly well and **teaches the trunk
    #: nothing**. Its own metric improves and the planner sees no gradient —
    #: the arm that proves the aux loss is actually reaching the trunk.
    detach_trunk: bool = False

    def __post_init__(self):
        if self.kind not in HEAD_KINDS:
            raise ValueError(f"kind must be one of {HEAD_KINDS}, "
                             f"got {self.kind!r}")
        if self.enable and self.w <= 0.0:
            raise ValueError(
                "BEVAuxConfig(enable=True, w=0) builds a head, stamps it into "
                "config.json and puts ZERO gradient into the trunk — the "
                "`w_agent` defect verbatim (a declared seam whose loss term is "
                "silently skipped). Set w > 0, or enable=False.")

    def to_dict(self) -> dict:
        return asdict(self)


class BEVAuxHead(nn.Module):
    """``fmap [B, F, gh, gw]`` -> BEV occupancy logits ``[B, n_rng, n_az]``.

    ⛔ **REFUSES ``gw != n_az``.** The whole reason this target is polar is that
    the corpus is CYLINDRICAL, so an image column IS an azimuth bin and the
    registration is exact with zero resampling. A head that silently interpolated
    ``gw`` columns onto ``n_az`` target columns would keep the loss curve looking
    healthy while mis-registering every agent — the mirrored-world failure in a
    different costume (`E-DEC-18`'s build measured the ego-frame convention for
    exactly this reason). If the geometries disagree, the SPEC is wrong, not the
    head.
    """

    def __init__(self, feat_dim: int, grid_shape, cfg: BEVAuxConfig):
        super().__init__()
        gh, gw = (int(grid_shape[0]), int(grid_shape[1]))
        if gw != int(cfg.n_az):
            raise ValueError(
                f"BEV aux: the encoder map has {gw} azimuth columns but the "
                f"target spec has n_az={cfg.n_az}. The polar target is "
                f"registered COLUMN-TO-COLUMN on purpose (cylindrical corpus: "
                f"the image column is linear in azimuth). Resampling here "
                f"would mis-register every agent silently. Fix the spec.")
        self.cfg = cfg
        self.grid_shape = (gh, gw)
        self.tok = nn.Conv2d(feat_dim, cfg.d_tok, 1)
        self.block = None
        if cfg.kind == "xcol":
            # ⭐ IMPORTED, never re-implemented: the same pre-norm
            # self-attention + MLP block the imagination graft uses. A second
            # local copy is how two geometries drift apart (the `advect`
            # precedent, retired 2026-07-27). Lazy, because `refc` imports this
            # module lazily in `RefCModel.__init__` and a module-level import
            # here would close the cycle.
            from tanitad.refs.refc import ImagBlock
            self.block = ImagBlock(cfg.d_tok * gh, cfg.n_heads, 2)
        self.ray = nn.Sequential(
            nn.Linear(cfg.d_tok * gh, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.n_rng))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, fmap: Tensor) -> Tensor:
        if self.cfg.detach_trunk:
            fmap = fmap.detach()
        z = self.tok(fmap)                                  # [B, d, gh, gw]
        b, d, gh, gw = z.shape
        # [B, gw, d*gh] — one token per AZIMUTH COLUMN, carrying its whole
        # elevation stack. Elevation is inverse depth on a ground plane, so the
        # rows are the evidence the ray MLP needs to place range.
        cols = z.permute(0, 3, 1, 2).reshape(b, gw, d * gh)
        if self.block is not None:
            cols = self.block(cols)
        return self.ray(cols).transpose(1, 2)               # [B, n_rng, n_az]


def bev_aux_loss(logits: Tensor, occ: Tensor, mask: Tensor,
                 pos_weight: float) -> dict:
    """Masked BCE-with-logits, per-term and with its ``n``.

    ``mask`` is REQUIRED and is the third state (True = supervised). A consumer
    that omitted it would supervise ``OCC_IGNORE`` cells as FREE — which is the
    exact defect ``data.bev_aux`` exists to remove — so there is no default.

    ⛔ Returns ``n_supervised`` and ``n_pos`` alongside the loss, always. A
    batch whose windows are all NO_LABEL yields ``n_supervised == 0``, and the
    loss is then an exact ``0.0`` **with ``n_supervised`` saying why** rather
    than a NaN or a silently-averaged nothing.
    """
    if mask.dtype != torch.bool:
        raise TypeError("bev_aux_loss: `mask` must be a bool tensor (True = "
                        "supervised); a float mask would silently weight the "
                        "IGNORE state instead of removing it.")
    if logits.shape != occ.shape or logits.shape != mask.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)}, "
                         f"occ {tuple(occ.shape)}, mask {tuple(mask.shape)}")
    n_sup = int(mask.sum())
    if n_sup == 0:
        return {"loss": logits.sum() * 0.0, "n_supervised": 0, "n_pos": 0,
                "base_rate": float("nan")}
    lg = logits[mask]
    tg = occ[mask].to(lg.dtype)
    pw = torch.as_tensor(float(pos_weight), device=lg.device, dtype=lg.dtype)
    loss = F.binary_cross_entropy_with_logits(lg, tg, pos_weight=pw)
    n_pos = int((tg > 0.5).sum())
    return {"loss": loss, "n_supervised": n_sup, "n_pos": n_pos,
            "base_rate": n_pos / n_sup}


def _average_precision(score, target) -> float:
    """Area under the precision-recall curve, ``AP = sum_g P(g) * dR(g)`` summed
    over **TIE GROUPS** (distinct scores), interpolation-free.

    ⭐ Written out rather than imported so the module has no sklearn dependency,
    and pinned in the tests against ANALYTIC values a reader can verify by hand:
    a perfect ranker reads **exactly 1.0**, a constant score reads **exactly the
    base rate**.

    ⛔⛔ **THE TIE GROUPING IS THE WHOLE POINT AND IT IS NOT COSMETIC.** The naive
    per-sample form (``sum_k P(k)*y_k / n_pos``) breaks ties by array order, so a
    CONSTANT score — which cannot rank at all — reads whatever the arbitrary tie
    order happens to give: on this module's own 480-cell fixture it read
    **0.010236 against a base rate of 0.008333, a 22.8 % overstatement**, and on
    a 1-positive fixture **0.009434 against 0.002083, 4.5x**. ⇒ a
    no-information control would have read HIGHER than the value it is supposed
    to define, and every arm would have been ranked against an inflated floor.
    ⭐ This defect was found by the control itself
    (``test_CONTROL_constant_score_reads_exactly_the_base_rate``) failing on the
    first run — which is the argument for writing controls that must read a
    LITERAL rather than controls that are merely "lower than the arm".
    ⚠️ It is worth checking any other AP in the programme for the same tie
    handling: WP-A's `all-zero` control reads **AP 0.016256 against a stated test
    base rate of 0.016124** (`E-READOUT-CEILING-1`), a +0.8 % gap of exactly this
    shape — small there, and the same mechanism.
    """
    import numpy as np
    s = np.asarray(score, dtype=np.float64).ravel()
    y = (np.asarray(target).ravel() > 0.5).astype(np.float64)
    n_pos = float(y.sum())
    if n_pos == 0.0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    # last index of each run of equal scores
    ends = np.flatnonzero(np.r_[np.diff(s) != 0.0, True])
    prec = tp[ends] / (ends + 1).astype(np.float64)
    d_rec = np.diff(np.r_[0.0, tp[ends]]) / n_pos
    return float((prec * d_rec).sum())


def constant_control_ap(occ, mask) -> float:
    """⭐ The value a CONSTANT-SCORE (all-zero / all-one / any constant) control
    MUST read: the base rate over the supervised cells, **exactly**.

    A control that does not read its known value means the metric is wrong, not
    that the model is good. This function returns the value so a test asserts a
    LITERAL relationship rather than eyeballing two numbers.
    """
    import numpy as np
    m = np.asarray(mask).ravel().astype(bool)
    y = (np.asarray(occ).ravel()[m] > 0.5)
    return float(y.sum() / m.sum()) if m.sum() else float("nan")


def bev_aux_metrics(logits, occ, mask, threshold: float = 0.5) -> dict:
    """AP / IoU / F1 over the SUPERVISED cells. ⛔ Never an accuracy.

    An all-zero predictor scores ~98 % accuracy on this target (MEASURED base
    rate ~1-2 %), so accuracy is not reported and cannot be derived from what is
    returned without also using ``n_supervised`` — which is the point.
    """
    import numpy as np
    lg = (logits.detach().cpu().numpy() if hasattr(logits, "detach")
          else np.asarray(logits))
    oc = (occ.detach().cpu().numpy() if hasattr(occ, "detach")
          else np.asarray(occ))
    mk = (mask.detach().cpu().numpy() if hasattr(mask, "detach")
          else np.asarray(mask)).astype(bool)
    n_sup = int(mk.sum())
    if n_sup == 0:
        return {"ap": float("nan"), "iou": float("nan"), "f1": float("nan"),
                "n_supervised": 0, "n_pos": 0, "base_rate": float("nan")}
    p = 1.0 / (1.0 + np.exp(-lg.ravel()[mk.ravel()]))
    y = oc.ravel()[mk.ravel()] > 0.5
    pred = p >= threshold
    inter = float((pred & y).sum())
    union = float((pred | y).sum())
    tp, fp, fn = inter, float((pred & ~y).sum()), float((~pred & y).sum())
    return {
        "ap": _average_precision(p, y),
        "iou": (inter / union) if union else 0.0,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "n_supervised": n_sup,
        "n_pos": int(y.sum()),
        "base_rate": float(y.sum() / n_sup),
        "threshold": float(threshold),
    }


def assert_loss_parity(w_bev: float, bev_loss: float, traj_loss: float,
                       band: tuple[float, float] = LOSS_PARITY_BAND) -> dict:
    """⛔ Refuse a run whose weighted aux term is outside ``band`` x the
    trajectory loss, **at step 0**.

    ⭐ THIS GUARD IS A MEASURED HISTORICAL DEFECT, NOT A STYLE RULE. `E-DEC-18b`:
    the PSG term sat 10-30x above the objective it was meant to support, at every
    weight tested, and destroyed the encoder — and the failure was only visible
    after the GPU-days. This is that check, moved to before them.

    ⚠️ It is a NECESSARY condition, not a sufficient one: a term inside the band
    can still be harmful, which is what the pre-registered planner-regression
    failure criterion is for. Returns the ratio so a run record can carry it.
    """
    lo, hi = band
    if traj_loss <= 0.0 or not (traj_loss == traj_loss):        # <=0 or NaN
        raise ValueError(f"assert_loss_parity: traj_loss must be finite and "
                         f"positive, got {traj_loss!r}")
    ratio = (float(w_bev) * float(bev_loss)) / float(traj_loss)
    ok = lo <= ratio <= hi
    if not ok:
        raise SystemExit(
            f"[WP-D] REFUSING: w_bev * bev_loss / traj_loss = {ratio:.4f}, "
            f"outside the pre-registered band [{lo}, {hi}]. "
            f"(w_bev={w_bev}, bev_loss={bev_loss:.5f}, "
            f"traj_loss={traj_loss:.5f}). Below the band the term is present "
            f"in config.json and inert in the gradient; above it, this is the "
            f"E-DEC-18b failure shape — an aux term outweighing the objective "
            f"it is meant to support by 10-30x, which destroyed the encoder at "
            f"EVERY weight tested. Re-scale `--w-bev-aux` and re-launch.")
    return {"ratio": ratio, "band": [lo, hi], "ok": True}
