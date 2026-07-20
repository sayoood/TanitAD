"""Flagship v1.5 — a REF-C-style anchored-diffusion planner head on the FROZEN
flagship-v1 world model, with VTARGET conditioning.

WHAT IS FROZEN, WHAT TRAINS
---------------------------
Frozen (loaded from ``flagship4b-speedjerk-30k``, the DEPLOYED v1 —
``speed_input=true``, ``action_dim=3``, ADE@2s 0.4522): the ViT encoder, the
spatial readout, and the operative predictor (the imagination). Nothing in the
trunk receives a gradient; nothing in the trunk is even in the optimizer.

Trains: ONLY :class:`FlagshipV15Head` (~32 M). It is the REF-C decoder algorithm
(``tanitad.refs.refc.AnchoredDiffusionDecoder``) REUSED, not reimplemented —
:class:`V15Decoder` subclasses it and overrides exactly one thing: the KV source.
REF-C cross-attends a conv feature map ``[B, F, g, g]``; v1.5 cross-attends a
heterogeneous TOKEN SET assembled from the frozen trunk. Everything else — the
anchor buffer, ``traj_proj``/``cond_proj``/``time_embed``, the cross-attention
stack, the per-anchor confidence + offset heads, the truncated-denoise loop, the
``load_anchors`` contract — is inherited unchanged.

THE THREE CONDITIONING SOURCES (the ablation axis)
--------------------------------------------------
(a) ``cond_states`` — the frozen encoder states. The v1 readout is a 4x4 spatial
    grid of ``d_readout=128`` cells flattened to ``state_dim=2048``
    (``SpatialGridReadout``: ``[B, G*G, d_r] -> flatten``), so a window of W
    states RE-EXPANDS losslessly to ``W * 16`` spatial-temporal tokens of width
    128. This is the drop-in stage: the head sees what v1 sees.

(b) ``cond_imagination`` — THE NOVEL PART. The frozen predictor is rolled
    forward under a fixed vocabulary of probe ACTION sequences
    (:func:`imagine_probes`), and the imagined future latents are fed as
    conditioning tokens. The decoder therefore sees the CONSEQUENCES of candidate
    controls before it denoises, instead of inferring them from the present
    frame. Latents are read at the same horizons the anchors live at.

(c) ``cond_vtarget`` / ``cond_route`` — the GOAL tokens. VTARGET is the tactical
    set-speed (V3, 23 non-uniform bands + one DROPPED slot); ROUTE is the
    strategic route class from the **v2.1** derivation
    (``refb_labels.route_target_v21``: adaptive horizon, explicit
    ``ROUTE_UNKNOWN`` instead of the silent straight-default, ``net_dyaw`` in
    the decision) plus its threshold-free graded companion. Both enter the
    measurement condition through independent ReZero gates (init 0.1) under
    **goal-dropout 0.5** (the H25/H26 anti-shortcut rule: without it the head
    reads the goal and ignores vision). Each gate's contribution norm is
    reported next to the measurement norm every log step — the H26 swamping bug
    class is a *monitored* quantity here, not an assumption.

    v1.5 is the first arm to deploy the label repairs, so the label set is a
    switchable input: ``ROUTE_UNKNOWN`` and an untrustworthy VTARGET both route
    to their DROPPED embedding rows rather than being silently coerced into a
    real class. Training on the legacy labels is still reachable (the trainer's
    ``--label-set legacy``) precisely so the label contribution can be measured
    rather than asserted.

LOSS — the DiffusionDrive recipe REF-C validated (refc_train.compute_losses):
anchor-classification CE against the GT-nearest anchor (weight 1.0) + L1
trajectory reconstruction FROM that assigned anchor (weight 1.0), with the
truncated-denoise refinement active in the same forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tanitad.refs.refc import (AnchoredDiffusionDecoder, DecoderConfig,
                               default_anchors, furthest_point_sample)

# VTARGET vocabulary width (tanitad.lake.vocab.VTARGET_TOKENS). Index
# ``N_VTARGET_BANDS`` is the extra DROPPED/unknown slot used by goal-dropout and
# by windows whose label is not trustworthy.
N_VTARGET_BANDS = 23
VT_DROPPED = N_VTARGET_BANDS

# ROUTE vocabulary as the v2.1 derivation emits it: 0 left / 1 straight / 2
# right / 3 ROUTE_UNKNOWN (refb_labels.ROUTE_UNKNOWN — the sentinel that replaced
# the silent straight-default), + 4 = DROPPED by goal-dropout. UNKNOWN and
# DROPPED are deliberately DIFFERENT rows: "the labeler could not judge this
# window" and "we withheld the goal from you" are different states.
N_ROUTE_CLASSES = 4
ROUTE_UNKNOWN = 3
ROUTE_DROPPED = 4

SPEED_SCALE = 10.0          # hard contract with the v1 trunk (flagship_losses)


# ============================================================================
# Config
# ============================================================================

@dataclass
class V15Config:
    """Head geometry + the conditioning switches that define the ablation."""

    # --- frozen-trunk contract (must match the loaded v1 checkpoint) --------
    state_dim: int = 2048           # v1 readout out_dim (4*4*128)
    readout_grid: int = 4           # -> 16 cells per frame
    d_cell: int = 128               # v1 d_readout
    window: int = 8                 # v1 predictor window
    action_dim: int = 3             # v1 speed_input: [steer, accel, v0/10]

    # --- decoder (REF-C AnchoredDiffusionDecoder) ---------------------------
    horizons: tuple[int, ...] = (5, 10, 15, 20)     # TanitEval WP_STEPS
    n_anchors: int = 256                            # REF-C-XL vocabulary size
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig(
        d=512, n_heads=8, layers=8, ff_mult=4, aux_hidden=512,
        diffusion_steps=2, noise_std=0.1))
    d_token: int = 512              # common KV width before the inherited feat_proj
    d_meas: int = 128               # measurement/condition width

    # --- conditioning switches (a) / (b) / (c) ------------------------------
    cond_states: bool = True
    cond_imagination: bool = True
    cond_vtarget: bool = True
    cond_route: bool = True         # part of (c): the v2.1 route goal token

    # --- (b) imagination probes --------------------------------------------
    n_probes: int = 8               # probe action-sequence vocabulary size
    probe_steps: int = 20           # roll length (2 s @ 10 Hz)
    imag_read: tuple[int, ...] = (5, 10, 15, 20)    # latents read at these steps

    # --- anti-shortcut discipline ------------------------------------------
    goal_dropout: float = 0.5       # (c) VTARGET dropout  (H25/H26 rule)
    ego_dropout: float = 0.5        # v0 dropout (REF-C convention)
    vt_gate_init: float = 0.1       # ReZero gate on the goal seams
    sel_gate_init: float = 0.0      # longitudinal selection term: starts OFF,
                                    # training decides (and the learned value is
                                    # the measurement of whether it helps)
    sel_accel_max: float = 2.5      # m/s^2 — the 2 s reachable-speed clamp
                                    # (P2's accel clamp; keeps an aspiration
                                    # 10 m/s away from demanding the impossible)

    def __post_init__(self):
        if not (self.cond_states or self.cond_imagination):
            raise ValueError("v1.5 needs at least one KV source: enable "
                             "cond_states and/or cond_imagination")
        if self.state_dim != self.readout_grid ** 2 * self.d_cell:
            raise ValueError(
                f"state_dim {self.state_dim} != grid^2*d_cell "
                f"{self.readout_grid ** 2 * self.d_cell} — the readout re-expand "
                "would scramble the spatial layout")


def v15_config() -> V15Config:
    """The v1.5 arm as trained (a+b+c)."""
    return V15Config()


def v15_ablation_config(states: bool = True, imagination: bool = True,
                        vtarget: bool = True,
                        route: bool | None = None) -> V15Config:
    """(a) / (a+b) / (a+b+c) ablation arms — identical geometry, switches only.

    ``route`` defaults to ``vtarget``: the two goal tokens together ARE the (c)
    stage, so the ablation isolates conditioning sources, not label plumbing.
    """
    cfg = V15Config()
    cfg.cond_states = states
    cfg.cond_imagination = imagination
    cfg.cond_vtarget = vtarget
    cfg.cond_route = vtarget if route is None else route
    return cfg


# ============================================================================
# Decoder — REF-C's, with the KV source swapped for a token set
# ============================================================================

class V15Decoder(AnchoredDiffusionDecoder):
    """:class:`AnchoredDiffusionDecoder` with the KV source swapped for a token
    set AND the REF-C selection flaw repaired.

    KV: the parent builds it as ``feat_proj(fmap.flatten(2).transpose(1, 2))`` —
    it only ever needs ``[B, P, feat_dim]``. v1.5 assembles P from several
    frozen-trunk sources of different widths, so it passes the tokens in
    directly. The denoise loop, anchor buffer and heads are the parent's.

    **THE SCORING FIX (do not silently revert this).** The parent refines every
    anchor through the denoise passes but throws their confidences away
    (``_, off = self._decode(...)``) and then selects with ``argmax`` over the
    t=0 classifier score — i.e. it ranks the REFINED fan using the UNREFINED
    anchor's score. Scoring and refinement are decoupled. Measured consequence
    on REF-C: selected ADE 1.110 m against an oracle-in-fan of 0.295 m clip-wide,
    with the selected plan more than 2x worse than the best available proposal in
    65 % of frames — a RANKING failure, not a coverage or refinement failure (the
    0.29 m plan was already in the fan).

    v1.5 therefore keeps the last denoise pass's confidence as
    ``refined_logits`` and selects on THAT. Both heads are supervised:
    ``anchor_logits`` (t=0) against the GT-nearest ORIGINAL anchor — the
    vocabulary-level classification the DiffusionDrive recipe trains — and
    ``refined_logits`` against the GT-nearest REFINED trajectory, which is the
    quantity selection actually uses. ``steps=0`` still reproduces the parent
    exactly (no denoise pass runs, so ``refined_logits is anchor_logits``).
    """

    def forward(self, tokens: Tensor, m: Tensor,          # type: ignore[override]
                ctx: Tensor | None = None,
                maneuver_logits: Tensor | None = None,
                target_latent: Tensor | None = None,
                steps: int = 0) -> dict:
        """``tokens`` [B, P, feat_dim] (already width-harmonised), ``m`` [B, d_meas]."""
        b = tokens.shape[0]
        kv = self.feat_proj(tokens)                       # [B, P, d]
        cond = self.cond_proj(m)                          # [B, d]
        if self.ctx_to_cond is not None and ctx is not None:
            cond = cond + self.ctx_to_cond(ctx)
        if self.tgt_film is not None and target_latent is not None:
            cond = self.tgt_film(cond, self.tgt_proj(target_latent))

        anchors = self.anchors.to(tokens.dtype)           # [N, S, 2]
        n = anchors.shape[0]
        x0 = anchors[None].expand(b, n, self.n_steps, 2)
        conf, offset = self._decode(kv, cond, x0, 0)      # classifier pass
        x = anchors[None] + offset                        # [B, N, S, 2]

        if self.maneuver_to_anchor is not None and maneuver_logits is not None:
            conf = conf + self.maneuver_to_anchor(
                torch.log_softmax(maneuver_logits, dim=-1))

        refined = conf                                    # steps=0 -> parent
        for i in range(steps):                            # truncated diffusion
            t_idx = min(i + 1, self.cfg.diffusion_steps)
            noise = (torch.randn_like(x) * self.cfg.noise_std
                     if self.training else torch.zeros_like(x))
            x_in = x + noise
            refined, off = self._decode(kv, cond, x_in, t_idx)   # KEEP the conf
            x = x_in + off

        return {"anchor_logits": conf, "refined_logits": refined,
                "anchor_traj": x, "offset": offset}


# ============================================================================
# The head
# ============================================================================

class FlagshipV15Head(nn.Module):
    """The ONLY trainable module of flagship v1.5."""

    def __init__(self, cfg: V15Config):
        super().__init__()
        self.cfg = cfg
        n_steps = len(cfg.horizons)
        d_tok = cfg.d_token

        # (a) frozen encoder states -> W*16 spatial-temporal cell tokens.
        n_cells = cfg.readout_grid ** 2
        self.n_state_tokens = cfg.window * n_cells if cfg.cond_states else 0
        if cfg.cond_states:
            self.state_proj = nn.Linear(cfg.d_cell, d_tok)
            self.state_pos = nn.Parameter(
                torch.zeros(cfg.window * n_cells, d_tok))
            nn.init.normal_(self.state_pos, std=0.02)

        # (b) imagined future latents -> n_probes * len(imag_read) tokens.
        self.n_imag_tokens = (cfg.n_probes * len(cfg.imag_read)
                              if cfg.cond_imagination else 0)
        if cfg.cond_imagination:
            self.imag_proj = nn.Linear(cfg.state_dim, d_tok)
            self.imag_pos = nn.Parameter(torch.zeros(self.n_imag_tokens, d_tok))
            nn.init.normal_(self.imag_pos, std=0.02)
            # source-type embedding: tells the decoder "this token is a
            # CONSEQUENCE, not an observation".
            self.src_embed = nn.Parameter(torch.zeros(2, d_tok))
            nn.init.normal_(self.src_embed, std=0.02)

        # measurement condition: the observed ego speed, ego-dropped.
        self.measurement = nn.Sequential(
            nn.Linear(1, cfg.d_meas), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_meas, cfg.d_meas), nn.ReLU(inplace=True))

        # (c) VTARGET token through a ReZero gate (init 0.1) — norm-parity
        # monitored (H26 swamping guard).
        if cfg.cond_vtarget:
            self.vtarget_emb = nn.Embedding(N_VTARGET_BANDS + 1, cfg.d_meas)
            nn.init.normal_(self.vtarget_emb.weight, std=0.02)
            self.vt_gate = nn.Parameter(torch.tensor(cfg.vt_gate_init))
        if cfg.cond_route:
            # class token (incl. the UNKNOWN sentinel and the DROPPED row) plus
            # the threshold-free graded value, which is defined on the windows
            # the 3-class label must mask.
            self.route_emb = nn.Embedding(N_ROUTE_CLASSES + 1, cfg.d_meas)
            nn.init.normal_(self.route_emb.weight, std=0.02)
            self.route_graded = nn.Linear(1, cfg.d_meas)
            self.rt_gate = nn.Parameter(torch.tensor(cfg.vt_gate_init))
        self.sel_gate = nn.Parameter(torch.tensor(cfg.sel_gate_init))

        self.decoder = V15Decoder(
            feat_dim=d_tok, n_steps=n_steps, d_meas=cfg.d_meas, d_ctx=cfg.d_meas,
            tac_latent_dim=cfg.d_meas,
            anchors=default_anchors(cfg.horizons, cfg.n_anchors, 4096, 0,
                                    device="cpu"),
            cfg=cfg.decoder, hierarchy=False, graft_maneuver=False,
            graft_target_latent=False, grounded_selector=False)

    # ---------------------------------------------------------------- utils --
    def load_anchors(self, anchors: Tensor) -> None:
        self.decoder.load_anchors(anchors)

    def build_tokens(self, states: Tensor, imagined: Tensor | None) -> Tensor:
        """states [B, W, S] (frozen), imagined [B, P_i, S] or None -> [B, P, d_tok]."""
        cfg = self.cfg
        toks = []
        if cfg.cond_states:
            b, w, s = states.shape
            cells = states.reshape(b, w * cfg.readout_grid ** 2, cfg.d_cell)
            t = self.state_proj(cells) + self.state_pos[None]
            if cfg.cond_imagination:
                t = t + self.src_embed[0][None, None]
            toks.append(t)
        if cfg.cond_imagination:
            if imagined is None:
                raise ValueError("cond_imagination is on but no imagined "
                                 "latents were supplied")
            t = self.imag_proj(imagined) + self.imag_pos[None]
            t = t + self.src_embed[1][None, None]
            toks.append(t)
        return torch.cat(toks, dim=1)

    def condition(self, v0: Tensor, vt_band: Tensor | None,
                  route: Tensor | None = None,
                  route_graded: Tensor | None = None
                  ) -> tuple[Tensor, dict, Tensor | None]:
        """-> (m [B, d_meas], telemetry, vt_keep [B] bool or None).

        ``vt_keep`` is False wherever the VTARGET was dropped by goal-dropout or
        was already the untrustworthy DROPPED band, so the selection term can be
        masked with the SAME mask the conditioning saw — a goal that was withheld
        from the decoder must not sneak back in through the ranking.

        Applies ego-dropout to v0 and INDEPENDENT goal-dropout to each goal
        token, all only in training. The DROPPED index is a real embedding row,
        so the head learns an explicit "no goal given" state rather than seeing
        a zero it could confuse with a class — and, for ROUTE, DROPPED is a
        different row from the v2.1 UNKNOWN sentinel.

        The telemetry is the H26 norm-parity monitor: ``*_over_m`` is each goal
        seam's contribution norm divided by the measurement norm. A seam that
        grows to swamp the rest of the condition is the documented failure mode,
        so it is logged rather than assumed away.
        """
        cfg = self.cfg
        v = (v0 / SPEED_SCALE).reshape(-1, 1).to(self.measurement[0].weight.dtype)
        if self.training and cfg.ego_dropout > 0:
            keep = (torch.rand(v.shape[0], 1, device=v.device)
                    >= cfg.ego_dropout).to(v.dtype)
            v = v * keep
        m = self.measurement(v)
        tele = {"m_norm": float(m.detach().norm(dim=-1).mean())}
        vt_keep = None
        if cfg.cond_vtarget:
            if vt_band is None:
                raise ValueError("cond_vtarget is on but no vt_band supplied")
            band = vt_band.clone()
            if self.training and cfg.goal_dropout > 0:
                drop = torch.rand(band.shape[0], device=band.device) < cfg.goal_dropout
                band = band.masked_fill(drop, VT_DROPPED)
            vt_keep = band != VT_DROPPED
            g = self.vt_gate * self.vtarget_emb(band)
            m = m + g
            tele["vt_norm"] = float(g.detach().norm(dim=-1).mean())
            tele["vt_gate"] = float(self.vt_gate.detach())
            tele["vt_over_m"] = round(tele["vt_norm"] / max(tele["m_norm"], 1e-9), 4)
        if cfg.cond_route:
            if route is None:
                raise ValueError("cond_route is on but no route supplied")
            r = route.clone()
            gr = (torch.zeros_like(v) if route_graded is None
                  else route_graded.reshape(-1, 1).to(v.dtype))
            if self.training and cfg.goal_dropout > 0:
                drop = torch.rand(r.shape[0], device=r.device) < cfg.goal_dropout
                r = r.masked_fill(drop, ROUTE_DROPPED)
                gr = gr * (~drop).to(gr.dtype).reshape(-1, 1)
            g = self.rt_gate * (self.route_emb(r) + self.route_graded(gr))
            m = m + g
            tele["rt_norm"] = float(g.detach().norm(dim=-1).mean())
            tele["rt_gate"] = float(self.rt_gate.detach())
            tele["rt_over_m"] = round(tele["rt_norm"] / max(tele["m_norm"], 1e-9), 4)
        return m, tele, vt_keep

    # ------------------------------------------------------------ selection --
    def terminal_speed(self, traj: Tensor) -> Tensor:
        """Implied terminal speed of each fan trajectory [B, N, S, 2] -> [B, N].

        The last inter-waypoint displacement over its 0.5 s stride. Terminal, not
        mean: VTARGET is a free-flow ASPIRATION read over 10-20 s (measured on
        this corpus to sit +1.46 m/s above the speed actually reached at +2 s),
        so scoring the 2 s MEAN speed against it would systematically demand
        over-speeding — which is precisely v1's known failure mode (+0.19 m/s
        speed bias overall, +0.66 m/s in the high-speed decile).
        """
        dt = (self.cfg.horizons[-1] - self.cfg.horizons[-2]) * 0.1
        return (traj[..., -1, :] - traj[..., -2, :]).norm(dim=-1) / dt

    def select(self, fan: Tensor, refined_logits: Tensor,
               vt_speed: Tensor | None, vt_keep: Tensor | None,
               v0: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Rank the REFINED fan.

        Returns ``(traj [B, S, 2], idx [B], score [B, N], telemetry)``. The
        SCORE is returned, not just the pick, because ``argmax`` is not
        differentiable: the ranking loss must be applied to the score itself or
        the longitudinal term's gate can never receive a gradient and would sit
        at its initial value forever, silently inert.

        Base score is ``refined_logits`` — the confidence of the trajectory that
        is actually emitted, not of the raw anchor (see :class:`V15Decoder`).

        On top of it, an optional LONGITUDINAL term: the measured REF-C failure
        is that the selector falls back to constant-velocity plans and passes
        over the decelerating proposal that was already in the fan, and the
        residual is dominantly longitudinal. A target-speed-aware score is
        exactly the signal that would rank the decelerating proposal first, so
        the term is ``-|v_term(i) - v_target_reachable|`` with a LEARNED scale
        initialised to ZERO: training decides whether it helps, and the learned
        value is itself the measurement. ``v_target`` is clamped to the 2 s
        reachable set around ``v0`` so an aspiration 10 m/s away cannot demand a
        physically impossible plan. The term is masked off wherever the goal was
        dropped or is untrustworthy, so it never leaks past goal-dropout.
        """
        score = refined_logits
        tele: dict = {}
        if self.cfg.cond_vtarget and vt_speed is not None:
            v_term = self.terminal_speed(fan)                     # [B, N]
            reach = self.cfg.sel_accel_max * self.cfg.horizons[-1] * 0.1
            v_goal = torch.max(torch.min(vt_speed, v0 + reach),
                               (v0 - reach).clamp_min(0.0))
            pen = -(v_term - v_goal[:, None]).abs()               # [B, N]
            if vt_keep is not None:
                pen = pen * vt_keep[:, None].to(pen.dtype)
            score = score + self.sel_gate * pen
            tele["sel_gate"] = float(self.sel_gate.detach())
            tele["sel_pen_span"] = float(
                (pen.max(dim=1).values - pen.min(dim=1).values).mean().detach())
        idx = score.argmax(dim=1)
        traj = fan[torch.arange(fan.shape[0], device=fan.device), idx]
        return traj, idx, score, tele

    # -------------------------------------------------------------- forward --
    def forward(self, states: Tensor, v0: Tensor,
                imagined: Tensor | None = None,
                vt_band: Tensor | None = None, route: Tensor | None = None,
                route_graded: Tensor | None = None,
                vt_speed: Tensor | None = None,
                steps: int | None = None) -> dict:
        """states [B, W, S] frozen · v0 [B] · imagined [B, P_i, S] · goal tokens.

        ``vt_speed`` [B] is the NUMERIC minted target speed (m/s) used only by
        the longitudinal selection term; the banded token is what conditions the
        decoder. ``steps`` selects the decoder mode: ``None`` -> the trained
        truncated-denoise refinement; ``0`` -> the classifier floor.
        """
        if steps is None:
            steps = self.cfg.decoder.diffusion_steps
        tokens = self.build_tokens(states, imagined)
        m, tele, vt_keep = self.condition(v0, vt_band, route, route_graded)
        out = self.decoder(tokens, m, steps=steps)
        traj, idx, score, s_tele = self.select(
            out["anchor_traj"], out["refined_logits"], vt_speed, vt_keep, v0)
        out["traj"] = traj
        out["sel_idx"] = idx
        out["sel_score"] = score
        out["waypoints"] = {k: traj[:, i]
                            for i, k in enumerate(self.cfg.horizons)}
        out["wp_seq"] = traj
        out["telemetry"] = {**tele, **s_tele}
        return out


# ============================================================================
# (b) imagination — roll the FROZEN predictor under the probe action vocabulary
# ============================================================================

@torch.no_grad()
def imagine_probes(predictor, states: Tensor, actions: Tensor, probes: Tensor,
                   read: tuple[int, ...], v0n: Tensor) -> Tensor:
    """Frozen-predictor consequence rollout -> conditioning latents.

    ``states`` [B, W, S], ``actions`` [B, W, A] (the OBSERVED window, A=3 with
    the v1 speed channel), ``probes`` [M, K, 2] the probe (steer, accel)
    sequences, ``v0n`` [B] the observed speed ALREADY divided by SPEED_SCALE.
    Returns [B, M*len(read), S].

    The roll is byte-identical in mechanism to
    ``metric_dynamics.rollout_transitions`` (the gate path): 1-step head, slide
    the window, append the next action. The v1 speed channel is HELD at the
    observed v0 — never a future speed (leakage-safe; matches every trainer and
    the P2 planner).
    """
    b, w, s = states.shape
    m, k, _ = probes.shape
    a_dim = actions.shape[-1]
    ws = states.unsqueeze(1).expand(b, m, w, s).reshape(b * m, w, s)
    wa = actions.unsqueeze(1).expand(b, m, w, a_dim).reshape(b * m, w, a_dim)
    pr = probes.unsqueeze(0).expand(b, m, k, 2).reshape(b * m, k, 2)
    v_col = v0n.reshape(b, 1, 1).expand(b, m, 1).reshape(b * m, 1)
    reads, k_max = [], max(read)
    for j in range(k_max):
        z = predictor(ws, wa)[1]                       # 1-step head -> z_{t+j+1}
        if (j + 1) in read:
            reads.append(z)
        if j < k_max - 1:
            a_next = torch.cat([pr[:, min(j, k - 1)], v_col], dim=-1) \
                if a_dim == 3 else pr[:, min(j, k - 1)]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
            wa = torch.cat([wa[:, 1:], a_next.unsqueeze(1)], dim=1)
    out = torch.stack(reads, dim=1)                    # [B*M, R, S]
    return out.reshape(b, m * len(read), s)


def build_probe_vocabulary(action_seqs: Tensor, n_probes: int,
                           seed: int = 0) -> Tensor:
    """FPS over REAL future action sequences -> [n_probes, K, 2].

    The same furthest-point-sampling argument as the anchor vocabulary: the
    corpus is ~74 % straight cruise, so k-means would collapse every probe onto
    "hold speed, go straight" and the imagination would only ever be asked one
    question. FPS spreads the probes over the manoeuvre space so the rolled-out
    consequences actually differ. ``action_seqs`` [M, K, 2].
    """
    return furthest_point_sample(action_seqs, n_probes, seed=seed).contiguous()


# ============================================================================
# Losses — the DiffusionDrive recipe (refc_train.compute_losses), head-only
# ============================================================================

TRAJ_WEIGHT = 1.0
ANCHOR_CLS_WEIGHT = 1.0
REFINED_CLS_WEIGHT = 1.0


def v15_losses(out: dict, anchors: Tensor, traj_tgt: Tensor) -> dict:
    """anchor-cls CE + traj-recon L1 + REFINED-rank CE.

    ``out`` from :meth:`FlagshipV15Head.forward`; ``anchors`` [N, S, 2];
    ``traj_tgt`` [B, S, 2] ego-frame waypoint targets
    (``refb_labels.waypoint_targets``).

    Two classification terms, because there are two different questions:

    * ``cls`` — the DiffusionDrive/REF-C term, verbatim: which ORIGINAL anchor is
      nearest the GT (flattened L2), scored by the t=0 classifier pass. This is
      the vocabulary-level assignment that also picks the reconstruction target.
    * ``cls_refined`` — which REFINED trajectory is nearest the GT, scored by
      ``sel_score``: the EXACT quantity ``argmax`` selects on, refined confidence
      plus the gated longitudinal term. REF-C never trained anything of the kind
      because it discarded the refined confidences, and the measured cost was a
      ranking failure: selected 1.110 m vs oracle-in-fan 0.295 m, with the pick
      worse than the fan's best by >2x in 65 % of frames. Supervising the score
      rather than the bare logits is also what gives the longitudinal gate a
      gradient — ``argmax`` has none.

    The returned ``oracle_ade`` / ``sel_gap`` are the diagnostic the fleet asked
    for as a standing metric: ``oracle_ade`` is the best trajectory available in
    the fan, ``ade`` is the one selected, and ``sel_gap = ade - oracle_ade``
    separates "cannot propose it" (both high) from "cannot rank it" (gap high).
    """
    b = traj_tgt.shape[0]
    ar = torch.arange(b, device=traj_tgt.device)
    dist = ((traj_tgt[:, None] - anchors[None].to(traj_tgt.dtype)) ** 2
            ).sum(dim=(-1, -2))                              # [B, N]
    a_star = dist.argmin(dim=1)
    loss_cls = nn.functional.cross_entropy(out["anchor_logits"].float(), a_star)
    recon = out["anchor_traj"][ar, a_star]
    loss_traj = (recon - traj_tgt).abs().mean()

    # the fan AFTER refinement — the thing selection ranks
    fan_err = (out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1).mean(dim=-1)
    r_star = fan_err.argmin(dim=1)                           # [B] oracle index
    loss_rcls = nn.functional.cross_entropy(out["sel_score"].float(),
                                            r_star.detach())

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + REFINED_CLS_WEIGHT * loss_rcls)
    with torch.no_grad():
        ade = (out["traj"] - traj_tgt).norm(dim=-1).mean()
        ade2s = (out["traj"][:, -1] - traj_tgt[:, -1]).norm(dim=-1).mean()
        acc = (out["anchor_logits"].argmax(dim=1) == a_star).float().mean()
        oracle = fan_err.min(dim=1).values.mean()
        sel_err = fan_err[ar, out["sel_idx"]]
        rank_acc = (out["sel_idx"] == r_star).float().mean()
        worse2x = (sel_err > 2.0 * fan_err.min(dim=1).values).float().mean()
    return {"loss": loss, "traj": loss_traj, "cls": loss_cls,
            "cls_refined": loss_rcls, "anchor_acc": acc, "ade": ade,
            "fde2s": ade2s, "oracle_ade": oracle,
            "sel_gap": ade - oracle, "rank_acc": rank_acc,
            "frac_sel_2x_worse_than_oracle": worse2x}


def param_breakdown(head: FlagshipV15Head) -> dict[str, int]:
    cnt = lambda m: sum(p.numel() for p in m.parameters())    # noqa: E731
    out = {"decoder": cnt(head.decoder), "measurement": cnt(head.measurement)}
    out["state_proj"] = (cnt(head.state_proj) + head.state_pos.numel()
                         if head.cfg.cond_states else 0)
    out["imag_proj"] = (cnt(head.imag_proj) + head.imag_pos.numel()
                        + head.src_embed.numel()
                        if head.cfg.cond_imagination else 0)
    out["vtarget"] = (cnt(head.vtarget_emb) + 1 if head.cfg.cond_vtarget else 0)
    out["route"] = (cnt(head.route_emb) + cnt(head.route_graded) + 1
                    if head.cfg.cond_route else 0)
    out["total"] = sum(p.numel() for p in head.parameters())
    return out
