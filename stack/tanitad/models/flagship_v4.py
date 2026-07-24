"""Flagship v4 — the OPERATIVE planner head (P1 of V4_FLAGSHIP_DESIGN §15).

``FlagshipV4Head`` is ``FlagshipV15Head`` (REF-C's ``AnchoredDiffusionDecoder``,
KV-swapped, selection flaw repaired) with the four v4-specific structural changes
that §6/§7 add on top of the v1.5 lineage — and NOTHING that touches the encoder
(v4 is at 2 of 2 encoder-touching levers; every change here is decode-side):

1. **Dense 20-step operative anchors** (``horizons = (1..20)``, §2.7/§3.3). A
   4-point head admits exactly one third difference; the dense emitted plan is the
   precondition for every smoothness term (§7) — and it is the path v4 SHIPS.

2. **Factorised LAT×LON×DIST selection** (§6.2). REF-C's single 5-way maneuver
   softmax mixes LATERAL and LONGITUDINAL modes into one prior (measured: it
   predicts ``accelerate`` on 0/881 windows), so there is no longitudinal signal
   anywhere in the ranking. v4 replaces it with three SEPARATE additive grafts —
   ``lat_to_anchor`` / ``lon_to_anchor`` / ``dist_to_anchor`` — reaching the
   RANKED score. All three are **zero-init**, so the selection path starts
   bit-identical to the graft-free baseline and the effect is attributable rather
   than confounded (``--lat/lon/dist-weight 0`` is the step-0 state, §16).

3. **In-graph norm clamp on the grafts** (§6.2 discipline 4). A graft that swamps
   the base score is not a prior, it is a second selector — the F3/ROUTE-seam
   failure mode (fired at 2.80x). Each graft's contribution norm is monitored
   against the base score every forward; the total is rescaled in-graph at
   ``seam_clamp`` (1.0x) and FAILS LOUD at ``seam_fail`` (1.5x). The KILL secondary
   ``seam_norm_ratio_max<=1.0`` reads the post-clamp ratio; a value above 1.0 means
   the clamp is broken (a code fault), which is exactly what that gate catches.

4. **λ_plan seam** (P2, O-20). The trunk->planner activation boundary (the readout
   state as it enters the head) passes through :func:`grad_scale`: forward is
   BIT-EXACTLY the state, the backward gradient into the trunk is multiplied by
   λ_plan. ``lambda_plan == 0`` lets the planner heads train at full strength on a
   trunk they cannot move — the LP phase of LP-FT (Phase A) — and reproduces the
   frozen-trunk v1.5 regime byte-identically, which is the attributability claim
   §9 rests on. ``lambda_plan == 1`` short-circuits to a strict no-op.

P5b (the learned-null row for the planner's own ``ego_dropout``) lives one level
up in :class:`~tanitad.models.flagship_v15.FlagshipV15Head` (gated by
``V4Config.ego_null_row = True``), because that is where ``condition`` lives and
v4 inherits it unchanged.

The TACTICAL 5 s instance (②) and the STRATEGIC subspace planner (①) are P5 / P6
and land in separate modules; ② is *this same class* instantiated at the coarse
horizons (:func:`tactical_config`), so nothing here is tactical-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tanitad.models.flagship_v15 import (FlagshipV15Head, V15Config,
                                         param_breakdown as _v15_breakdown)
from tanitad.models.metric_dynamics import grad_scale
from tanitad.refs.refc import DecoderConfig

# Factorised tactical-mode vocabulary widths (V3_FACTORIZED_TACTICAL_HEAD_SPEC.md
# §2, design §4.3/§6.2). These are the KINEMATICALLY-MINTABLE subsets plus a
# masked ``unknown`` sentinel — deliberately NARROWER than the 9-wide lake vocab,
# because a logit no label can ever train is a dead parameter that invites a
# shortcut (§6.5): LATMANEUVER keeps 7 kinematic modes, LONMODE 6 (the three
# lead-referenced modes are unmintable while lead_state is a None stub), DIST 8
# metric bands with a masked ``d_unknown``. The label->index maps live with the v3
# labels and the trainer (P4); the model only needs the widths.
N_LAT = 8                       # 7 kinematic LATMANEUVER + unknown sentinel
N_LON = 7                       # 6 kinematic LONMODE      + unknown sentinel
N_DIST = 8                      # 8 DIST_BAND_TOKENS (d_unknown masked in the CE)

DENSE_HORIZONS = tuple(range(1, 21))            # 20 steps @ 0.1 s = 2 s dense (③)
TACTICAL_HORIZONS = tuple(range(5, 51, 5))      # 10 knots @ 0.5 s = 5 s coarse (②)


# ============================================================================
# Config
# ============================================================================

@dataclass
class V4Config(V15Config):
    """v1.5 head geometry, re-pointed at the v4 operative instance.

    Inherits every V15Config field (frozen-trunk contract, conditioning switches,
    anti-shortcut discipline) and overrides only what §2.4/§2.7 change: the dense
    horizons, REF-C's 256-anchor / d384x4L decoder, the learned null row (P5b),
    the factorised heads and the seam clamp.
    """

    # --- dense operative geometry (§2.4/§3.3) -------------------------------
    horizons: tuple[int, ...] = DENSE_HORIZONS
    n_anchors: int = 256                    # REF-C-XL vocabulary size
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(
        d=384, n_heads=8, layers=4, ff_mult=4, aux_hidden=512,
        diffusion_steps=2, noise_std=0.1))  # d384x4L (§2.4), NOT v1.5's d512x8L

    # --- P5b: learned null row instead of the v0 zero-fill (X15) ------------
    ego_null_row: bool = True

    # --- factorised LAT x LON x DIST selection (§6.2) -----------------------
    factorised: bool = True                 # the operative instance ③ has it; the
                                            # tactical instance ② does NOT (§3.1)
    factor_hidden: int = 128                # per-head MLP hidden (a §14.4 O-12 knob)
    seam_clamp: float = 1.0                 # rescale the graft in-graph at this ratio
    seam_fail: float = 1.5                  # fail loud above this pre-clamp ratio


def v4_config() -> V4Config:
    """The v4 OPERATIVE planner (③): dense 2 s, factorised selection, null row."""
    return V4Config()


def tactical_config() -> V4Config:
    """The v4 TACTICAL instance (②, P5): the SAME head at 5 s coarse horizons.

    Two instances cost ~19.5 M combined vs one at 9.8 M because the decoder is the
    only sized part and the anchor vocabulary is a 0-parameter buffer (§3.3). The
    dense-vs-coarse split is nothing but the horizon tuple.
    """
    cfg = V4Config()
    cfg.horizons = TACTICAL_HORIZONS
    cfg.imag_read = TACTICAL_HORIZONS       # imagination read at the coarse steps
    cfg.factorised = False                  # factorised selection is operative-only
    return cfg


# ============================================================================
# The operative head
# ============================================================================

class FlagshipV4Head(FlagshipV15Head):
    """Operative planner ③ — see the module docstring for the four changes."""

    def __init__(self, cfg: V4Config):
        super().__init__(cfg)
        self.cfg: V4Config = cfg
        n = cfg.n_anchors

        if cfg.factorised:
            # Factorised tactical-mode heads, read from the current-frame readout
            # state. SEPARATE (not one Linear) so ``lon``/``dist`` are ablatable.
            def _head(n_out: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(cfg.state_dim, cfg.factor_hidden),
                    nn.ReLU(inplace=True), nn.Linear(cfg.factor_hidden, n_out))

            self.lat_head = _head(N_LAT)
            self.lon_head = _head(N_LON)
            self.dist_head = _head(N_DIST)

            # The three anchor grafts. ZERO-INIT (ReZero discipline, as REF-C's
            # ctx_to_cond) so the ranked score starts bit-identical to the
            # graft-free baseline — the attributability claim. bias=False mirrors
            # maneuver_to_anchor.
            self.lat_to_anchor = nn.Linear(N_LAT, n, bias=False)
            self.lon_to_anchor = nn.Linear(N_LON, n, bias=False)
            self.dist_to_anchor = nn.Linear(N_DIST, n, bias=False)
            for g in (self.lat_to_anchor, self.lon_to_anchor, self.dist_to_anchor):
                nn.init.zeros_(g.weight)

    # ------------------------------------------------------------- grafts ---
    def _factor_grafts(self, refined: Tensor, lat_logits: Tensor,
                       lon_logits: Tensor, dist_logits: Tensor
                       ) -> tuple[Tensor, dict]:
        """Add the three log-softmax grafts to the RANKED score, norm-clamped.

        ``refined`` [B, N] is the base score selection ranks on; the grafts are
        priors on which anchors a LAT/LON/DIST mode favours. Returns
        ``(grafted [B, N], telemetry)``. The clamp is in-graph (differentiable):
        the total graft is rescaled per-sample so its norm never exceeds
        ``seam_clamp`` x the base-score norm, and a pre-clamp ratio above
        ``seam_fail`` raises (the clamp itself would be broken).
        """
        lsm = torch.log_softmax
        g_lat = self.lat_to_anchor(lsm(lat_logits, dim=-1))       # [B, N]
        g_lon = self.lon_to_anchor(lsm(lon_logits, dim=-1))
        g_dist = self.dist_to_anchor(lsm(dist_logits, dim=-1))
        graft = g_lat + g_lon + g_dist                            # [B, N]

        base = refined.norm(dim=-1).clamp_min(1e-9)               # [B]
        tele = {f"{k}_over_conf": round(float(
                    (g.norm(dim=-1) / base).mean().detach()), 4)
                for k, g in (("lat", g_lat), ("lon", g_lon), ("dist", g_dist))}
        ratio = graft.norm(dim=-1) / base                        # [B]
        pre_max = float(ratio.max().detach())
        if pre_max > self.cfg.seam_fail:
            raise RuntimeError(
                f"factorised-selection seam norm ratio {pre_max:.3f} > "
                f"{self.cfg.seam_fail} (fail-loud): a graft is swamping the base "
                f"score — the in-graph clamp is not holding, i.e. a code fault.")
        # rescale in-graph so the EFFECTIVE ratio never exceeds seam_clamp
        scale = self.cfg.seam_clamp / ratio.clamp_min(self.cfg.seam_clamp)  # <=1
        graft = graft * scale[:, None]
        tele["seam_norm_ratio_max"] = round(min(pre_max, self.cfg.seam_clamp), 4)
        tele["seam_norm_ratio_preclamp_max"] = round(pre_max, 4)
        return refined + graft, tele

    # -------------------------------------------------------------- forward --
    def forward(self, states: Tensor, v0: Tensor,          # type: ignore[override]
                imagined: Tensor | None = None,
                vt_band: Tensor | None = None, route: Tensor | None = None,
                route_graded: Tensor | None = None,
                vt_speed: Tensor | None = None,
                steps: int | None = None, lambda_plan: float = 1.0) -> dict:
        """As :meth:`FlagshipV15Head.forward`, plus the λ_plan seam and the
        factorised grafts. ``lambda_plan`` scales the planner->trunk gradient
        (O-20): 1.0 is a strict no-op, 0.0 is the LP regime (heads train, trunk
        unmoved by the planner)."""
        if steps is None:
            steps = self.cfg.decoder.diffusion_steps
        # λ_plan seam: forward-identical, gradient into the trunk scaled by λ.
        states_p = grad_scale(states, lambda_plan)

        tokens = self.build_tokens(states_p, imagined)
        m, tele, vt_keep = self.condition(v0, vt_band, route, route_graded)
        out = self.decoder(tokens, m, steps=steps)

        seam_tele: dict = {}
        refined = out["refined_logits"]
        if self.cfg.factorised:
            # factorised LAT x LON x DIST -> the ranked score (§6.2). Operative ③
            # only; the tactical instance ② has no factorised selection (§3.1).
            cur = states_p[:, -1]                               # [B, state_dim]
            lat_logits = self.lat_head(cur)
            lon_logits = self.lon_head(cur)
            dist_logits = self.dist_head(cur)
            refined, seam_tele = self._factor_grafts(
                refined, lat_logits, lon_logits, dist_logits)
            out["refined_logits"] = refined
            out["lat_logits"] = lat_logits
            out["lon_logits"] = lon_logits
            out["dist_logits"] = dist_logits

        traj, idx, score, s_tele = self.select(
            out["anchor_traj"], refined, vt_speed, vt_keep, v0)
        out["traj"] = traj
        out["sel_idx"] = idx
        out["sel_score"] = score
        out["waypoints"] = {k: traj[:, i]
                            for i, k in enumerate(self.cfg.horizons)}
        out["wp_seq"] = traj
        out["telemetry"] = {**tele, **s_tele, **seam_tele}
        return out


def param_breakdown(head: FlagshipV4Head) -> dict[str, int]:
    """v1.5 breakdown + the factorised heads/grafts and the P5b null row."""
    cnt = lambda m: sum(p.numel() for p in m.parameters())     # noqa: E731
    out = _v15_breakdown(head)
    if head.cfg.factorised:
        out["factor_heads"] = (cnt(head.lat_head) + cnt(head.lon_head)
                               + cnt(head.dist_head))
        out["factor_grafts"] = (cnt(head.lat_to_anchor) + cnt(head.lon_to_anchor)
                                + cnt(head.dist_to_anchor))
    else:
        out["factor_heads"] = out["factor_grafts"] = 0
    return out
