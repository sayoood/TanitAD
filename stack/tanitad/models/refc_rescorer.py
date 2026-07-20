"""REF-C v1.2 — a LEARNED re-scorer over REF-C's REFINED trajectory fan.

THE MEASURED PROBLEM (do not re-derive — corpus figures, 881-window val)
------------------------------------------------------------------------
``refc.AnchoredDiffusionDecoder.forward`` denoises ALL N anchors, but the
denoise passes return ``_, off`` — their confidences are DISCARDED — and the
final ``argmax`` runs over the **t=0 classifier score computed on the UNREFINED
anchors**. Geometry is refined; ranking is not.

    baseline (selected)   0.4714 full-set / 0.4577 heldout
    oracle-in-fan         0.1640          <- min over all 256 refined proposals
    ranking gap           0.3075 m
    frac_sel_2x_worse     0.454

(Earlier drafts quoted 0.295 / 65 % — those are SINGLE-CLIP ep11 stride-1
figures and must not be used as corpus numbers.)

WHY THIS IS A TOP-K PROBLEM, NOT A 256-WAY PROBLEM
---------------------------------------------------
Two measurements from the REF-C v1.0 study reshape the objective:

* The 0.1640 full-fan oracle is partly a **lottery**: it is the min over 256
  draws whose TYPICAL member is 13.9 m off. Chasing it teaches a ranker to find
  needles in a haystack of garbage.
* **Oracle within the top-8 by frozen confidence is 0.2026 — 87 % of the gap**
  (top-4: 0.2506, 72 %). The reachable signal is dense there.
* The frozen confidence head is **strong**: Spearman 0.907 against ADE,
  capturing 68 % of the chance->oracle span, with sharp logits (top-1 prob
  0.654, top1-top2 gap 1.56 nats). It must be treated as a strong incumbent to
  be RESIDUALLY corrected, not replaced.
* A hand-written cost re-rank (REF-C v1.0) recovers **0.0 %** of the gap — the
  best blend weight is zero, and pure cost is -171 %. So the open question this
  arm answers is precisely: *can a LEARNED ranker do what a hand-written one
  provably cannot?*

Hence ``RescorerConfig.topk`` (default 8): the head scores only the top-K
candidates by frozen confidence and re-orders WITHIN them. K is swept.

TARGET DISCIPLINE — measured, not assumed
------------------------------------------
The target is the **joint** along-track + cross-track error (plain ADE over the
4 waypoints). It is deliberately NOT a speed/VTARGET objective: a GT-perfect
speed-matcher scores **1.1236, worse than the baseline**, and a GT-perfect
along-track-only ranker caps at 34 % of the gap. VTARGET sits +1.42 m/s above
``v0`` and is a 10-20 s free-flow aspiration; used at 2 s it is worse than
simply holding ``v0`` and it makes braking windows +0.51 m worse. Speed
quantities appear here only as INPUT FEATURES, never as a target.

THE v1.5 LESSON THIS MODULE EXISTS TO AVOID
-------------------------------------------
``flagship_v15`` already tried the obvious repair: keep the last denoise pass's
confidence as ``refined_logits``, select on it, and train it with a **hard
argmin CE** against the oracle index (``v15_losses.cls_refined``). It helped and
then DEGRADED as the fan sharpened — ``frac_sel_2x_worse`` 0.099 -> 0.40. The
mechanism: once many candidates converge to near-identical quality, the argmin
index becomes a coin flip between equally-good plans, the CE keeps paying full
loss for picking the "wrong" one of two 0.30 m plans, and the score is dragged
into a degenerate high-confidence mode. **The hard target is the bug.**

v1.2 therefore trains the score with objectives whose gradient is proportional
to how much WORSE a candidate actually is:

  ``soft``     ListNet-top-1 / distillation: cross-entropy against the soft
               label ``p_i = softmax(-ADE_i / tau)``. As ``tau -> 0`` this
               becomes EXACTLY the v1.5 hard-argmin CE, so the temperature sweep
               contains the known failure mode as its endpoint — the sweep is
               the experiment, not a hyper-parameter detail.
  ``pair``     distance-weighted pairwise margin: for every ordered pair the
               required score gap is ``margin_scale * |ADE_i - ADE_j|``, so
               near-ties contribute ~nothing and gross mis-rankings dominate.
  ``regress``  pointwise value learning: predict each candidate's ADE and select
               the argmin. No temperature at all — the natural control for
               "is the pathology inherent to ranking losses?".
  ``hard``     the v1.5 objective, reproduced deliberately as the control arm.

(pointwise / pairwise / listwise is the classical learning-to-rank triad; this
module runs all three legs plus the v1.5 control off one architecture.)

ARCHITECTURE — head-only, and IDENTITY AT INITIALISATION
--------------------------------------------------------
The entire ``RefCModel`` is frozen. The only trainable module is
:class:`RefCRescorer` (~1.7 M params), which consumes, per candidate:

  * ``q``          the frozen decoder's FINAL-denoise-pass query embedding
                   [N, d_q] — the representation that PRODUCED the discarded
                   refined confidence. This is the signal v1.0 (a training-free
                   hand-written cost) cannot see.
  * ``fan``        the refined trajectory [N, S, 2] -> explicit kinematic
                   features (per-segment speed, terminal speed, speed delta vs
                   ``v0``, heading change, lateral excursion).
  * ``base_logit`` the frozen selection score itself, as a feature.
  * context        pooled visual latent, the decoder condition, ``v0``.

``layers`` self-attention blocks run ACROSS the K candidates, so the score of a
proposal is computed in the context of its competitors — the listwise inductive
bias the frozen per-anchor ``conf_head`` (a plain ``Linear(d, 1)`` applied
independently) structurally cannot express.

The score is a RESIDUAL on the frozen ranking::

    score = base_gain * standardise(base_logit) + score_head(tokens)

with ``base_gain`` initialised to 1.0 and ``score_head`` zero-initialised, so an
untrained v1.2 reproduces ``refc-xl-30k``'s selection **exactly** (asserted by
tests). Every metre of movement from 0.4714 is therefore attributable to the
re-scorer and nothing else. (``regress`` is the one exception: it selects on a
predicted ADE and has no identity init — stated, not hidden.)

The standardisation matters: the frozen logits are SHARP, so a gentle additive
correction is a no-op — the same reason v1.0's blend degenerated to lambda=0.
Row-standardising is an increasing affine map (``argmax`` provably unchanged)
that puts the residual on a scale where confident re-orderings inside the top-K
are actually reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.refs.refc import NAV_COMMANDS

DT = 0.1                       # 10 Hz — the corpus/eval cadence
SEG_DT = 0.5                   # 5/10/15/20 -> 0.5 s between waypoint slots
SPEED_SCALE = 10.0             # the stack-wide v0 scaling contract
POS_SCALE = 10.0               # metres -> O(1) for the geometry features
N_GEOM = 20                    # width of :func:`geom_features` (pinned by tests)


# ============================================================================
# Instrumented frozen forward — the fan PLUS the discarded refined embeddings
# ============================================================================

@torch.no_grad()
def refc_forward_fan(model, frames: Tensor, nav_cmd: Tensor | None = None,
                     v0: Tensor | None = None, steps: int | None = None
                     ) -> dict:
    """Run a frozen :class:`~tanitad.refs.refc.RefCModel` and ALSO return the
    quantities its own ``forward`` throws away.

    Reproduces ``RefCModel.forward`` + ``AnchoredDiffusionDecoder.forward``
    step for step — reusing the model's own submodules, so there is no second
    copy of any weight or layer — and additionally returns:

      ``q``            [B, N, d]   FINAL-pass anchor query embedding
      ``q0``           [B, N, d]   t=0 (classifier-pass) query embedding
      ``refined_conf`` [B, N]      final-pass confidence (the DISCARDED score)
      ``cond``         [B, d]      decoder condition (measurement + ctx graft)
      ``pooled``       [B, F]      encoder pooled latent

    BOTH embeddings are returned because they are NOT interchangeable and the
    difference is measurable: ``conf_head(q0)`` IS the frozen selection score
    (Spearman 0.907 vs ADE), while ``conf_head(q_final)`` selects at 1.366 m —
    2.9x worse than the baseline — because REF-C never supervised the conf head
    at the denoise timesteps. ``q0`` is therefore a representation with a
    known-excellent linear readout; ``q_final`` is the representation OF the
    refined trajectory that selection actually ranks. Which one a learned
    re-scorer should consume is an empirical question (``RescorerConfig.
    q_source``), not an assumption.

    ``tests/test_refc_rescorer.py`` asserts bit-exact agreement with
    ``model(...)`` on ``anchor_logits`` / ``anchor_traj`` / ``traj`` / ``sel_idx``
    — if refc.py ever changes, this fails loud rather than drifting.

    NOTE the model must be in ``eval()`` mode: that is what zeroes the denoise
    noise and the ego-dropout, i.e. the deterministic decode TanitEval scores.
    """
    assert not model.training, "refc_forward_fan requires model.eval()"
    cfg = model.cfg
    dec = model.decoder
    if steps is None:
        steps = cfg.decoder.diffusion_steps
    b, w = frames.shape[:2]

    # ---- encoder (hierarchy=True -> all W frames; else the last only) ------
    if cfg.hierarchy:
        fmap_all, pooled_all = model.encoder(
            frames.reshape(b * w, *frames.shape[2:]))
        pooled_seq = pooled_all.reshape(b, w, -1)
        pooled = pooled_seq[:, -1]
        fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
        ctx = model.strategic(pooled_seq)
    else:
        fmap, pooled = model.encoder(frames[:, -1])
        ctx = None

    if cfg.graft_imagination:
        fmap, _ = model.imagination(fmap)

    # ---- measurement condition --------------------------------------------
    if nav_cmd is None:
        nav_cmd = torch.zeros(b, dtype=torch.long, device=frames.device)
    nav = F.one_hot(nav_cmd, len(NAV_COMMANDS)).to(pooled.dtype)
    v = (torch.zeros(b, 1, dtype=pooled.dtype, device=pooled.device)
         if v0 is None else (v0.to(pooled.dtype) / SPEED_SCALE).reshape(b, 1))
    m = model.measurement(torch.cat([v, nav], dim=-1))

    man_logits = model.maneuver_head(pooled)

    # ---- decoder (AnchoredDiffusionDecoder.forward, instrumented) ----------
    kv = dec.feat_proj(fmap.flatten(2).transpose(1, 2))
    cond = dec.cond_proj(m)
    if dec.ctx_to_cond is not None and ctx is not None:
        cond = cond + dec.ctx_to_cond(ctx)

    anchors = dec.anchors.to(fmap.dtype)
    n = anchors.shape[0]
    x0 = anchors[None].expand(b, n, dec.n_steps, 2)

    def _pass(x_est: Tensor, t_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """``AnchoredDiffusionDecoder._decode`` + the query embedding."""
        q = dec.traj_proj(x_est.reshape(b, n, -1))
        q = q + dec.time_embed.weight[t_idx][None, None]
        for layer in dec.layers:
            q = layer(q, kv, cond)
        conf = dec.conf_head(q).squeeze(-1)
        offset = dec.offset_head(q).reshape(b, n, dec.n_steps, 2)
        return conf, offset, q

    conf, offset, q0 = _pass(x0, 0)
    q = q0
    x = anchors[None] + offset
    if dec.maneuver_to_anchor is not None:
        conf = conf + dec.maneuver_to_anchor(
            torch.log_softmax(man_logits, dim=-1))

    refined_conf = conf
    for i in range(steps):
        t_idx = min(i + 1, dec.cfg.diffusion_steps)
        # eval mode -> zero noise (parity with refc.py's `if self.training`)
        refined_conf, off, q = _pass(x, t_idx)
        x = x + off

    score = conf + dec._grounded_score(x) if dec.grounded else conf
    idx = score.argmax(dim=1)
    traj = x[torch.arange(b, device=x.device), idx]
    return {"anchor_logits": conf, "anchor_traj": x, "traj": traj,
            "sel_idx": idx, "q": q, "q0": q0, "refined_conf": refined_conf,
            "cond": cond, "pooled": pooled, "maneuver_logits": man_logits}


# ============================================================================
# Candidate geometry features
# ============================================================================

def select_q(d: dict, q_source: str = "final") -> Tensor:
    """Pick the frozen embedding a head consumes, from a cache row or from
    :func:`refc_forward_fan`'s output. ``d`` must carry ``q`` and ``q0``."""
    if q_source == "final":
        return d["q"]
    if q_source == "t0":
        return d["q0"]
    if q_source == "both":
        return torch.cat([d["q0"], d["q"]], dim=-1)
    raise ValueError(f"unknown q_source {q_source!r}")


def q_width(d_q: int, q_source: str) -> int:
    """Head input width implied by ``q_source`` for a decoder of width ``d_q``."""
    return 2 * d_q if q_source == "both" else d_q


def topk_view(base_logit: Tensor, k: int, *tensors: Tensor
              ) -> tuple[Tensor, list[Tensor]]:
    """Restrict the fan to its top-``k`` candidates by FROZEN confidence.

    Returns ``(idx [B, k], [t[b, idx] for t in tensors])`` where ``idx`` maps
    back to global anchor indices — the eval path needs that mapping to report a
    selection in the decoder's own index space. ``k <= 0`` or ``k >= N`` is the
    identity (full fan), so every downstream call site is K-agnostic.
    """
    b, n = base_logit.shape
    if k <= 0 or k >= n:
        idx = torch.arange(n, device=base_logit.device).expand(b, n)
        return idx, list(tensors)
    idx = base_logit.topk(k, dim=1).indices                    # [B, k]
    ar = torch.arange(b, device=base_logit.device)[:, None]
    return idx, [t[ar, idx] for t in tensors]


def geom_features(fan: Tensor, v0: Tensor, seg_dt: float = SEG_DT) -> Tensor:
    """Explicit kinematics of every candidate: [B, N, S, 2] + [B] -> [B, N, 20].

    The frozen ``conf_head`` sees only the decoder's latent; the measured REF-C
    residual is dominantly LONGITUDINAL (the selector falls back to a
    constant-velocity plan and passes over a decelerating proposal already in
    the fan), so the re-scorer is handed speed structure explicitly rather than
    being asked to rediscover it from 8 raw coordinates.

    Layout (all O(1)-scaled): 8 raw waypoint coords · 4 per-segment speeds ·
    terminal speed · mean speed · terminal-minus-initial speed delta · terminal
    speed minus v0 · net heading change (rad) · |lateral| at the last horizon ·
    signed lateral at the last horizon · path length.
    """
    b, n, s, _ = fan.shape
    p = fan / POS_SCALE                                        # [B, N, S, 2]
    prev = torch.cat([torch.zeros_like(fan[:, :, :1]), fan[:, :, :-1]], dim=2)
    seg = fan - prev                                           # [B, N, S, 2]
    seg_speed = seg.norm(dim=-1) / seg_dt                      # [B, N, S] m/s
    v_term = seg_speed[..., -1]
    v_mean = seg_speed.mean(dim=-1)
    v_init = seg_speed[..., 0]
    head = torch.atan2(seg[..., -1, 1], seg[..., -1, 0].clamp_min(1e-6))
    lat = fan[..., -1, 1]
    plen = seg.norm(dim=-1).sum(dim=-1)
    v0e = (v0.reshape(b, 1).expand(b, n)).to(fan.dtype)
    return torch.cat([
        p.reshape(b, n, s * 2),                                # 8
        (seg_speed / SPEED_SCALE),                             # 4
        (v_term / SPEED_SCALE).unsqueeze(-1),                  # 1
        (v_mean / SPEED_SCALE).unsqueeze(-1),                  # 1
        ((v_term - v_init) / SPEED_SCALE).unsqueeze(-1),       # 1
        ((v_term - v0e) / SPEED_SCALE).unsqueeze(-1),          # 1
        head.unsqueeze(-1),                                    # 1
        (lat.abs() / POS_SCALE).unsqueeze(-1),                 # 1
        (lat / POS_SCALE).unsqueeze(-1),                       # 1
        (plen / POS_SCALE).unsqueeze(-1),                      # 1
    ], dim=-1)                                                 # -> 20


# ============================================================================
# The re-scorer
# ============================================================================

@dataclass
class RescorerConfig:
    """Head geometry. Widths default to the refc-xl-30k contract."""
    n_steps: int = 4              # waypoints per candidate (5/10/15/20)
    d_q: int = 512                # frozen decoder width (XL: 512)
    d_pooled: int = 992           # frozen encoder feat_dim (XL: 124*8)
    d_cond: int = 512             # decoder condition width (== d_q)
    d: int = 256                  # re-scorer width
    n_heads: int = 4
    layers: int = 2               # self-attention blocks ACROSS candidates
    ff_mult: int = 2
    use_q: bool = True            # consume the frozen decoder query embedding
    q_source: str = "final"
    """Which frozen embedding feeds ``q_proj``: ``final`` (the last denoise
    pass — the representation OF the refined trajectory), ``t0`` (the
    classifier pass, whose frozen LINEAR readout is the 0.907-Spearman
    selection score), or ``both`` (concatenated, ``d_q`` must be doubled).
    Empirical question, not an assumption — see :func:`select_q`."""
    use_geom: bool = True         # consume explicit candidate kinematics
    use_context: bool = True      # consume pooled visual + condition + v0
    dropout: float = 0.0
    topk: int = 8
    """Score only the top-K candidates by FROZEN confidence (0 = the full fan).

    Measured: the full-fan oracle (0.1640) is a lottery over 256 draws whose
    typical member is 13.9 m off, while the oracle within the top-8 is 0.2026 —
    **87 % of the ranking gap, in 3 % of the candidates.** Restricting is not a
    convenience: it removes the garbage tail that would otherwise dominate every
    listwise/pairwise sum and dilute the gradient with comparisons nobody will
    ever have to make."""
    normalize_base: bool = True
    """Per-window standardisation of the frozen logits before they enter the
    score. A trained REF-C is very confident, so its raw logit spread can be
    tens of units — a zero-init residual would need an implausible magnitude to
    overturn any pick, and the head would look inert for reasons that have
    nothing to do with the target. Standardising is an increasing affine map of
    each row, so ``argmax`` is provably unchanged and identity-at-init survives
    (asserted by tests)."""


class CandidateBlock(nn.Module):
    """Pre-norm self-attention over the N candidates + MLP. The attention is
    what makes the score LISTWISE: a candidate is scored relative to the fan it
    competes in, which a per-anchor Linear head cannot do."""

    def __init__(self, d: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff_mult * d, d))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class RefCRescorer(nn.Module):
    """The ONLY trainable module of REF-C v1.2 (~1.7 M params).

    ``forward`` returns ``score`` [B, N] — the quantity ``argmax`` consumes, and
    therefore the quantity the ranking loss MUST be applied to (argmax has no
    gradient; anything else leaves the head inert, the v1.5 finding).
    """

    def __init__(self, cfg: RescorerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        # Input LayerNorms: q / pooled / cond are RAW frozen-decoder activations
        # with arbitrary scale and no reason to be conditioned for a fresh head.
        # Normalising them is what lets the head train at a sane lr instead of
        # spending its budget learning the scale.
        if cfg.use_q:
            self.q_norm = nn.LayerNorm(cfg.d_q)
            self.q_proj = nn.Linear(cfg.d_q, d)
        if cfg.use_geom:
            self.geom_proj = nn.Linear(N_GEOM, d)
        self.logit_proj = nn.Linear(1, d)
        if cfg.use_context:
            self.ctx_norm = nn.LayerNorm(cfg.d_pooled + cfg.d_cond + 1)
            self.ctx_proj = nn.Sequential(
                nn.Linear(cfg.d_pooled + cfg.d_cond + 1, d), nn.GELU(),
                nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            CandidateBlock(d, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(d)
        # Residual score head: ZERO-init -> an untrained v1.2 selects EXACTLY
        # what refc-xl-30k selects. Any delta is the re-scorer's doing.
        self.score_head = nn.Linear(d, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        self.base_gain = nn.Parameter(torch.tensor(1.0))
        # Pointwise value head (``--target regress`` only): predicts each
        # candidate's ADE in metres. No identity init — selection becomes
        # argmin(ade_hat), which is a DIFFERENT selector by construction.
        self.ade_head = nn.Linear(d, 1)

    # ------------------------------------------------------------------ ---
    def base_score(self, base_logit: Tensor) -> Tensor:
        """The frozen selection score, optionally row-standardised (an
        increasing affine map per window -> the pick is bit-identical)."""
        if not self.cfg.normalize_base:
            return base_logit
        mu = base_logit.mean(dim=1, keepdim=True)
        # population std (unbiased=False): K=1 is a legal degenerate view
        sd = base_logit.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (base_logit - mu) / sd

    def tokens(self, base: Tensor, q: Tensor, fan: Tensor,
               pooled: Tensor, cond: Tensor, v0: Tensor) -> Tensor:
        cfg = self.cfg
        b, n = base.shape
        t = self.logit_proj(base.unsqueeze(-1))
        if cfg.use_q:
            t = t + self.q_proj(self.q_norm(q))
        if cfg.use_geom:
            t = t + self.geom_proj(geom_features(fan, v0))
        if cfg.use_context:
            c = torch.cat([pooled, cond,
                           (v0 / SPEED_SCALE).reshape(b, 1).to(pooled.dtype)],
                          dim=-1)
            t = t + self.ctx_proj(self.ctx_norm(c)).unsqueeze(1)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

    def forward(self, q: Tensor, base_logit: Tensor, fan: Tensor,
                pooled: Tensor, cond: Tensor, v0: Tensor,
                target: str = "soft") -> dict:
        """Score the top-``cfg.topk`` candidates of the fan.

        -> ``{"score" [B, K], "topk_idx" [B, K] (GLOBAL anchor indices),
        "sel_idx" [B] (global), "ade_hat" [B, K] (regress only)}``.

        The top-K gather lives INSIDE the module so K travels in the config and
        therefore in the checkpoint: the trainer and the eval adapter cannot
        drift apart on the single most consequential design choice.

        ``target='regress'`` swaps the selector to ``-ade_hat`` so the pointwise
        arm selects on its own predicted quality rather than on a residual.
        """
        # Gather FIRST, cast second: at K=8 the fan is 256 wide, so converting
        # the full [B, 256, d_q] embedding to fp32 before the gather is 32x the
        # necessary work and it dominated the sweep's wall-clock.
        idx_k, (q_k, fan_k, base_k) = topk_view(
            base_logit.float(), self.cfg.topk, q, fan, base_logit)
        dt = self.logit_proj.weight.dtype
        q_k, fan_k, base_k = q_k.to(dt), fan_k.to(dt), base_k.to(dt)
        pooled, cond, v0 = pooled.to(dt), cond.to(dt), v0.to(dt)
        base = self.base_score(base_k)          # standardise WITHIN the top-K
        t = self.tokens(base, q_k, fan_k, pooled, cond, v0)
        resid = self.score_head(t).squeeze(-1)
        out = {"resid": resid, "topk_idx": idx_k}
        if target == "regress":
            ade_hat = F.softplus(self.ade_head(t).squeeze(-1))
            out["ade_hat"] = ade_hat
            out["score"] = -ade_hat
        else:
            out["score"] = self.base_gain * base + resid
        out["sel_idx"] = idx_k.gather(
            1, out["score"].argmax(dim=1, keepdim=True)).squeeze(1)
        return out


# ============================================================================
# Targets — the point of the experiment
# ============================================================================

def rescorer_loss(out: dict, fan_ade: Tensor, target: str = "soft",
                  tau: float = 0.2, margin_scale: float = 2.0) -> Tensor:
    """Ranking loss over the scored candidate set.

    ``fan_ade`` [B, N] is every candidate's JOINT along+cross-track ADE in
    metres (see the module docstring on why the target is never a speed
    quantity); it is gathered down to the head's top-K view automatically.

    ``soft``     -CE against ``softmax(-ADE/tau)``. tau -> 0 IS ``hard``.
    ``hard``     the flagship-v1.5 objective — kept as the control arm.
    ``pair``     ``relu(margin_scale*|dADE| - (s_better - s_worse))`` over all
                 ordered pairs: near-ties demand ~zero gap, gross mis-rankings
                 demand a large one.
    ``regress``  smooth-L1 of the predicted ADE against the true one.
    """
    score = out["score"].float()
    ade = fan_ade.float()
    if "topk_idx" in out and ade.shape[1] != score.shape[1]:
        ade = ade.gather(1, out["topk_idx"])
    if target == "hard":
        return F.cross_entropy(score, ade.argmin(dim=1))
    if target == "soft":
        p = F.softmax(-ade / max(tau, 1e-6), dim=1)
        return -(p * F.log_softmax(score, dim=1)).sum(dim=1).mean()
    if target == "regress":
        return F.smooth_l1_loss(out["ade_hat"].float(), ade, beta=0.2)
    if target == "pair":
        d_ade = ade[:, :, None] - ade[:, None, :]              # [B, N, N]
        d_s = score[:, :, None] - score[:, None, :]
        # i is BETTER than j  <=>  d_ade < 0; required gap = scale * |d_ade|
        better = (d_ade < 0).float()
        viol = F.relu(margin_scale * d_ade.abs() - d_s) * better
        n = score.shape[1]
        return viol.sum(dim=(1, 2)).mean() / (n * (n - 1) / 2)
    raise ValueError(f"unknown target {target!r}")


@torch.no_grad()
def rank_metrics(out: dict, base_logit: Tensor, fan_ade: Tensor,
                 topk: int = 0) -> dict:
    """The G3 mechanism read — BEFORE (frozen) and AFTER (learned) on identical
    windows, so the comparison cannot drift.

    ``sel_ade``      ADE of the plan the re-scorer picks
    ``base_ade``     ADE of the plan the FROZEN refc-xl-30k score picks
    ``oracle_ade``   best ADE in the FULL fan — GT-informed and UNREACHABLE,
                     and partly a lottery (min over 256 draws whose typical
                     member is 13.9 m off). Report it; never target it.
    ``oracle_k_ade`` best ADE inside the top-K the head actually sees — the
                     honest ceiling for THIS design (top-8: 0.2026 = 87 % of
                     the gap)
    ``*_gap``        selected - full-fan oracle, in metres
    ``sel_gap_k``    selected - top-K oracle: how much of the reachable signal
                     is still on the table
    ``*_2x``         fraction of windows picking >2x worse than the full oracle
    """
    b = fan_ade.shape[0]
    ar = torch.arange(b, device=fan_ade.device)
    oracle = fan_ade.min(dim=1).values
    sel = fan_ade[ar, out["sel_idx"]]
    base = fan_ade[ar, base_logit.argmax(dim=1)]
    k_ade = fan_ade.gather(1, out["topk_idx"])
    oracle_k = k_ade.min(dim=1).values
    m = {"sel_ade": sel.mean().item(), "base_ade": base.mean().item(),
         "oracle_ade": oracle.mean().item(),
         "oracle_k_ade": oracle_k.mean().item(),
         # CHANCE floors — the expected ADE of a ranker with no skill, over the
         # full fan and inside the top-K. Without these the 0.3075 m "gap" is
         # uninterpretable: an oracle is a MINIMUM over many candidates scored
         # against ONE realised future, so part of it is the statistics of
         # taking a min, not headroom a ranker could reach.
         "chance_ade": fan_ade.mean(dim=1).mean().item(),
         "chance_k_ade": k_ade.mean(dim=1).mean().item(),
         "sel_gap": (sel - oracle).mean().item(),
         "base_gap": (base - oracle).mean().item(),
         "sel_gap_k": (sel - oracle_k).mean().item(),
         "sel_2x": (sel > 2.0 * oracle).float().mean().item(),
         "base_2x": (base > 2.0 * oracle).float().mean().item(),
         "rank_acc": (out["sel_idx"]
                      == fan_ade.argmin(dim=1)).float().mean().item(),
         "rank_acc_k": (out["score"].argmax(dim=1)
                        == k_ade.argmin(dim=1)).float().mean().item()}
    # fraction of the RANKING GAP recovered — the study's headline unit, the
    # same one v1.0 scored 0.0 % on.
    denom = (base - oracle).mean().item()
    m["gap_recovered"] = ((base - sel).mean().item() / denom
                          if abs(denom) > 1e-9 else 0.0)
    # what fraction of the chance -> oracle span each selector captures inside
    # the top-K: the incumbent's true strength, and the room actually left
    span = m["chance_k_ade"] - m["oracle_k_ade"]
    if abs(span) > 1e-9:
        m["base_span_k"] = (m["chance_k_ade"] - m["base_ade"]) / span
        m["sel_span_k"] = (m["chance_k_ade"] - m["sel_ade"]) / span
    return m


def fan_ade_from(fan: Tensor, tgt: Tensor) -> Tensor:
    """Per-candidate ADE over the 4 waypoints: [B, N, S, 2] x [B, S, 2] -> [B, N].

    The JOINT along-track + cross-track error — the same definition TanitEval's
    ADE uses (mean over waypoints of the L2 displacement error), so ``sel_ade``
    here is directly comparable to the harness's ``full_set`` ADE on the same
    windows. A GT-perfect along-track-ONLY ranker caps at 34 % of the gap and a
    GT-perfect speed-matcher is WORSE than the baseline, which is why the two
    axes are never separated in the objective.
    """
    return (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)


def fan_ade_axes(fan: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
    """Along-track / cross-track decomposition of the same error, for the
    report only: [B, N] each (ego frame -> x is along, y is cross)."""
    d = fan - tgt[:, None]
    return d[..., 0].abs().mean(dim=-1), d[..., 1].abs().mean(dim=-1)


def param_breakdown(head: RefCRescorer) -> dict[str, int]:
    cnt = lambda m: sum(p.numel() for p in m.parameters())     # noqa: E731
    cfg = head.cfg
    return {"q_proj": (cnt(head.q_proj) + cnt(head.q_norm)) if cfg.use_q else 0,
            "geom_proj": cnt(head.geom_proj) if cfg.use_geom else 0,
            "ctx_proj": (cnt(head.ctx_proj) + cnt(head.ctx_norm))
            if cfg.use_context else 0,
            "blocks": cnt(head.blocks), "heads": cnt(head.score_head)
            + cnt(head.ade_head) + 1,
            "total": sum(p.numel() for p in head.parameters())}
