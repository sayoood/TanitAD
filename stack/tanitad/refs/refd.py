"""REF-D — SimWAM's training-time prior, our hierarchy, one twentieth the size.

⭐ THE THESIS, and the one economic fact it rests on. SimWAM (2608.07468) reaches
91.5 PDMS on NAVSIM with **6 B parameters on 32 GPUs**, and its central trick is
an **isolated attention mask**: future-video prediction is a TRAINING signal, and
the video expert is **deleted at inference**. That means the expensive prior is a
*training* cost, not a deployment cost — so a sub-300 M deployment constraint
does not forbid a large prior, it only forbids **shipping** one.

REF-D takes that economics and spends it on our hierarchy instead of on scale:

    frozen driving-pretrained prior  ->  adapter  ->  3-rate goal-conditioned
    (Cosmos3-Edge, 4 B, TRAINING ONLY)              hierarchy with ACTION TOKENS
                                                     (ships; ~180 M)

## What is inherited, and from which measurement

| choice | source | evidence |
|---|---|---|
| isolated mask — no future token reaches the policy | SimWAM Tab. 3 | isolated 90.3 vs bidirectional 90.2 vs action->video 90.1 |
| ⭐ **action as TOKENS, not broadcast** | **ours, E-ACTSTREAM-1** | token beats concat **5.9x** and add **9.9x** at parameter parity, separated at 3 widths / 2 horizons / 2 targets / 3 seeds |
| **controls, never waypoints** | ours, H1 + v5f | a per-waypoint head amplified eps **25x** in acceleration; the v5f dense fan was **97.6 % infeasible steps / 100 % infeasible candidates** |
| **OU-correlated noise** | ours, F-15 | white noise on a 60-step control sequence integrates to near-cancelling jitter |
| feasibility by construction | ours, W4 | every sample is squashed and integrated through ``unicycle_rollout`` |
| goal conditioning by a **shared vocabulary** | ours, HIERARCHY_VOCABULARY §5 | ``head.vocab is cond.vocab`` — one table, two views |
| multi-horizon supervision | SimWAM Tab. 8 + ours | 4 s/1 Hz (90.2) ~ 4 s/2 Hz (90.3) >> 2 s/2 Hz (89.9): **coverage beats density** |

## ⛔ THE POLICY IS A GENERATOR, NOT A FAN — and this is the load-bearing change

v6 diffuses a **fan** of candidates and then a **selector** ranks them.
``assert_selector_admissible`` REFUSES every selector launch while SEL-1 stands
refused (E-WC2: sigma/ADE 9.9915 [7.4492, 13.5119] against the 3.0 line). So the
fan design created a selection problem that is now the programme's blocker.

**SimWAM has no selector because its flow model IS the policy** — sample once,
integrate, done. REF-D adopts that: the generator emits ONE control sequence.
⇒ This does not fix the selector. It removes the need for one, which is a much
cheaper way past SEL-1 than repairing it.

⚠️ **A REAL AND UNRESOLVED TENSION, stated rather than papered over.** Rectified
flow assumes an **isotropic Gaussian** base; our OU-correlated noise is not
isotropic. Plain flow matching tolerates that (it is a different coupling), but
Flow-GRPO's tractable transition likelihoods — the machinery SimWAM's +1.2 PDMS
of RL depends on — assume isotropic steps. ``noise`` is therefore a DECLARED
config axis with both arms buildable, and E-REFD-2 below is pre-registered to
settle it. Choosing silently would make the RL result uninterpretable.

## ⛔ RL IS DESIGNED IN BUT NOT REACHABLE YET, AND THE BLOCKER IS NOT COMPUTE

SimWAM's RL is worth **+1.2 PDMS** (90.3 -> 91.5) and its reward is the **NAVSIM
PDM score**. MEASURED 2026-08-19: this repo contains **zero ``import navsim``** —
the benchmark named in the Master Plan's Phase 1 was never implemented. So the
order is forced: **benchmark -> reward -> RL**, and REF-D ships the hooks
(``rl_ready``) with the loop deliberately absent.
⚠️ Building a reward out of our own four metric families instead would mean
optimising the quantity we also report — the "scoring a loop" failure the
goal/situation-disjointness rule exists to prevent.

## TIER

This module is MECHANISM. No number produced by calling anything here is
quotable; capability claims come from ``taniteval`` at **T1**.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.fourbrain import StrategicPolicy, TacticalPolicy
from tanitad.models.v6 import TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS
from tanitad.refs.refa_v1 import WideAdapter, _Block
from tanitad.refs.refa_v1p import ActionStreamPredictor

__all__ = ["RefDConfig", "FlowControlPolicy", "RefD", "PRIOR_GEOMETRY"]

#: The frozen prior. ⚠️ SIZES CORRECTED 2026-08-20 from the model card: the
#: Cosmos3 family is Super **64 B** / Nano **16 B** / Edge **4 B**. An earlier
#: note in this programme said "Cosmos3-Nano, 2 B" — that was **Cosmos-Dreams**
#: (ex `omni-dreams`), a 2 B model DISTILLED from Cosmos-Predict 2.5, conflated
#: with Nano. Edge is the relevant one: 4 B, and its recommended hardware line
#: names **Jetson Thor** explicitly.
#: ⚠️ It is also already a World-Action Model — Action is both an INPUT and an
#: OUTPUT modality (`Cosmos3-Edge-Policy-DROID`), so this is not a video prior
#: with an action head bolted on.
PRIOR_GEOMETRY: dict = {
    "model": "cosmos3-edge",
    "params_b": 4.0,
    "role": "FROZEN, pre-extracted, TRAINING-TIME ONLY — never shipped",
    "runs_on": "Jetson AGX Orin / Thor / RTX Pro 6000",
    #: ⚠️ NOT "Cosmos beats Wan". SimWAM Tab. 4 is Cosmos-Predict2.5 **90.4** vs
    #: Wan2.2-5B **90.3** — 0.1 PDMS with NO CI reported, i.e. a tie. The 1.7-point
    #: spread often quoted is LTX (88.7) -> Cosmos (90.4), which contrasts a weak
    #: backbone with a strong one. Our reason to prefer Edge is that it RUNS ON
    #: THOR, not that it scored higher.
    "why_not_wan": "Wan2.2-5B ties it at 90.3 vs 90.4; Edge is chosen for Thor",
}


@dataclass
class RefDConfig:
    """Every number that defines the arm, in one place."""

    # ---- the frozen prior ------------------------------------------------- #
    d_enc: int = 1024                 # width of the pre-extracted prior field
    n_tokens: int = 640
    #: ⛔ >= d_enc (change #3, inherited): FROST-Drive measured 8.17 -> 7.68 RFS
    #: on exactly this axis, so the state may not bottleneck below the encoder.
    d_state: int = 1024

    # ---- the three rates, each reaching exactly 6.0 s ---------------------- #
    op_dt: float = 0.2
    op_steps: int = 30
    op_layers: int = 6
    op_heads: int = 8
    op_window: int = 4
    tac_dt: float = 0.6
    tac_steps: int = 10
    tac_layers: int = 4
    tac_queries: int = 64
    str_dt: float = 1.5
    str_steps: int = 4

    #: action tokens per predictor. 2 is the PARAMETER-MATCHED value measured in
    #: E-ACTSTREAM-1; raising it buys capacity as well as structure.
    n_act_tokens: int = 2

    # ---- the policy -------------------------------------------------------- #
    a_dim: int = 2                    # (a, kappa) — controls, never waypoints
    plan_steps: int = 60              # 6.0 s at 10 Hz
    flow_steps: int = 10              # SimWAM Tab. 10: 5 recovers most, 10 best
    policy_hidden: int = 256
    #: ⛔ DECLARED, NOT CHOSEN — see the module docstring. "ou" is our measured
    #: preference for fan diversity; "iso" is what Flow-GRPO's likelihoods need.
    noise: str = "ou"                 # "ou" | "iso"
    ou_rho: float = 0.9
    a_max: float = 4.0
    kappa_max: float = 0.2

    # ---- multi-horizon future supervision (all ISOLATED from inference) ---- #
    #: ⭐ THE EXTENSION SIMWAM CANNOT EXPRESS. They have ONE action group and ONE
    #: future horizon. A hierarchy is a machine for covering several, and their
    #: own Tab. 8 says horizon COVERAGE is what matters. Each layer is supervised
    #: at its own rate; none of it is visible at inference.
    w_future_op: float = 1.0
    w_future_tac: float = 0.5
    w_future_str: float = 0.25
    w_policy: float = 1.0

    # ---- the brains -------------------------------------------------------- #
    tactical_cfg: TacticalPolicyConfig | None = None
    strategic_cfg: StrategicPolicyConfig | None = None

    #: hooks only; the loop is deliberately absent until a reward exists
    rl_ready: bool = True

    def sanity(self) -> None:
        if self.d_state < self.d_enc:
            raise ValueError(
                f"d_state ({self.d_state}) < d_enc ({self.d_enc}) — change #3 "
                f"forbids a bottleneck below the encoder width")
        for name, dt, steps in (("operative", self.op_dt, self.op_steps),
                                ("tactical", self.tac_dt, self.tac_steps),
                                ("strategic", self.str_dt, self.str_steps)):
            if abs(dt * steps - 6.0) > 1e-6:
                raise ValueError(f"{name} horizon must reach exactly 6.0 s, "
                                 f"got {dt} x {steps} = {dt * steps}")
        if self.noise not in ("ou", "iso"):
            raise ValueError(f"noise must be 'ou' | 'iso', got {self.noise!r}")
        if self.n_act_tokens < 1:
            raise ValueError("n_act_tokens >= 1: zero action tokens is a "
                             "predictor the action cannot reach")
        if self.noise == "ou" and self.rl_ready:
            # not an error — a declared, visible consequence
            pass


class FlowControlPolicy(nn.Module):
    """⭐ THE POLICY. Flow matching over a CONTROL sequence — one sample, no fan.

    SimWAM's action expert is a flow-matching DiT over 8 waypoints. Ours is the
    same objective over **60 x (a, kappa)** controls, because a per-waypoint head
    amplified eps 25x in acceleration and the v5f dense waypoint fan measured
    97.6 % infeasible steps. Controls integrated through the unicycle are
    feasible BY CONSTRUCTION.

    Training (rectified flow, SimWAM §3): with clean target ``x`` and noise
    ``eps``, ``x_tau = (1-tau) x + tau eps`` has constant velocity ``eps - x``,
    and the network regresses it. Sampling integrates the ODE from tau=1 to 0.
    """

    def __init__(self, cfg: RefDConfig, d_cond: int):
        super().__init__()
        self.cfg = cfg
        self.d_cond = int(d_cond)
        h = cfg.policy_hidden
        # +2 = the flow time tau and v0/SPEED_SCALE (the trunk's own speed column)
        self.cond = nn.Linear(d_cond + 2, h)
        self.net = nn.Sequential(
            nn.Conv1d(cfg.a_dim + h, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, cfg.a_dim, 1))
        # ZERO-INIT the velocity head: at init the field is zero, so the ODE
        # returns the (feasible) noise prior unchanged and anything the module
        # later does is LEARNED rather than handed to it by initialisation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _noise(self, b: int, dev, gen=None) -> Tensor:
        """[B, a_dim, T]. ⚠️ OU vs isotropic is the declared axis, not a detail."""
        T = self.cfg.plan_steps
        w = torch.randn(b, self.cfg.a_dim, T, device=dev, generator=gen)
        if self.cfg.noise == "iso":
            return w
        rho = self.cfg.ou_rho
        out = torch.empty_like(w)
        out[:, :, 0] = w[:, :, 0]
        for t in range(1, T):
            out[:, :, t] = rho * out[:, :, t - 1] + (1 - rho ** 2) ** 0.5 * w[:, :, t]
        return out

    def velocity(self, x_tau: Tensor, tau: Tensor, cond: Tensor,
                 v0: Tensor) -> Tensor:
        """The flow field v_theta(x_tau, tau, c) -> [B, a_dim, T]."""
        b, _, T = x_tau.shape
        c = self.cond(torch.cat([cond, tau.reshape(b, 1),
                                 (v0 / 10.0).reshape(b, 1)], dim=-1))
        return self.net(torch.cat([x_tau, c[:, :, None].expand(-1, -1, T)], 1))

    def loss(self, target: Tensor, cond: Tensor, v0: Tensor) -> Tensor:
        """Rectified-flow regression on the constant velocity ``eps - x``."""
        b, dev = target.shape[0], target.device
        eps = self._noise(b, dev)
        tau = torch.rand(b, device=dev)
        x_tau = (1 - tau[:, None, None]) * target + tau[:, None, None] * eps
        return ((self.velocity(x_tau, tau, cond, v0) - (eps - target)) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: Tensor, v0: Tensor, gen=None) -> Tensor:
        """Integrate tau: 1 -> 0. Returns ONE control sequence [B, a_dim, T].

        ⛔ ONE sample, not a fan: the generator IS the policy (see the module
        docstring on SEL-1).
        """
        b, dev = cond.shape[0], cond.device
        x = self._noise(b, dev, gen)
        n = max(1, int(self.cfg.flow_steps))
        for i in range(n):
            tau = torch.full((b,), 1.0 - i / n, device=dev)
            x = x - self.velocity(x, tau, cond, v0) / n
        a = torch.tanh(x[:, 0]) * self.cfg.a_max
        k = torch.tanh(x[:, 1]) * self.cfg.kappa_max
        return torch.stack([a, k], dim=1)


class RefD(nn.Module):
    """Frozen prior field -> adapter -> 3-rate action-token hierarchy -> policy.

    ``forward`` is TRAINING (multi-horizon future prediction + the policy's flow
    loss). ``act`` is DEPLOYMENT and touches no future token.
    """

    def __init__(self, cfg: RefDConfig | None = None):
        super().__init__()
        cfg = cfg or RefDConfig()
        cfg.sanity()
        self.cfg = cfg

        self.adapter = WideAdapter(cfg)
        intent_dim = (cfg.tactical_cfg.d_intent
                      if cfg.tactical_cfg is not None else None)
        # ⭐ ACTION-AS-TOKENS on both field predictors (E-ACTSTREAM-1)
        self.operative = ActionStreamPredictor(
            cfg, cfg.d_state, cfg.op_layers, cfg.op_heads, intent_dim,
            n_act_tokens=cfg.n_act_tokens)
        self.tactical = ActionStreamPredictor(
            cfg, cfg.d_state, cfg.tac_layers, cfg.op_heads, intent_dim,
            n_act_tokens=cfg.n_act_tokens)
        self.tac_queries = nn.Parameter(torch.zeros(1, cfg.tac_queries, cfg.d_state))
        nn.init.trunc_normal_(self.tac_queries, std=0.02)
        self.tac_pool = nn.MultiheadAttention(cfg.d_state, 8, batch_first=True)

        self.strategic_policy = (
            StrategicPolicy(cfg.strategic_cfg, cfg.d_state, cfg.op_window)
            if cfg.strategic_cfg is not None else None)
        self.tactical_policy = (
            TacticalPolicy(cfg.tactical_cfg, cfg.d_state, cfg.op_window,
                           d_cond=cfg.strategic_cfg.d_ctx)
            if cfg.tactical_cfg is not None else None)
        if (self.tactical_policy is None) != (self.strategic_policy is None):
            raise ValueError("the brains are a MATCHED SET — the tactical "
                             "policy is FiLM-conditioned on the strategic ctx")

        # factored lat x lon heads, importing v6's tuples BY IDENTITY so the
        # retired 5-way mixed softmax cannot come back through a copy
        self.n_lat, self.n_lon = len(TACTICAL_LAT_ACTIONS), len(TACTICAL_LON_ACTIONS)
        self.lat_head = nn.Sequential(nn.LayerNorm(cfg.d_state),
                                      nn.Linear(cfg.d_state, self.n_lat))
        self.lon_head = nn.Sequential(nn.LayerNorm(cfg.d_state),
                                      nn.Linear(cfg.d_state, self.n_lon))
        self.policy = FlowControlPolicy(cfg, d_cond=cfg.d_state)

    # ------------------------------------------------------------------ #
    def _pool(self, field: Tensor) -> Tensor:
        q = self.tac_queries.expand(field.shape[0], -1, -1)
        return self.tac_pool(q, field, field, need_weights=False)[0]

    def act(self, window: Tensor, v0: Tensor, *, gen=None) -> dict:
        """⛔ DEPLOYMENT. ``window`` is [B, W, N, d_enc] — a causal window of
        PAST prior fields, ending at the CURRENT frame.

        ⚠️ PAST frames are not future frames. The isolated-mask discipline
        forbids a PREDICTED field from reaching this path; it says nothing
        against observed history, which is what `op_window` has always been and
        what `WideAdapter` requires (it mixes temporally over W).

        No future token, no rollout, no fan, no selector — and
        :meth:`assert_no_future_at_inference` makes that checkable rather than
        merely asserted here.
        """
        if window.dim() != 4:
            raise ValueError(
                f"act expects [B, W, N, d_enc], got {tuple(window.shape)} — a "
                f"3-D input silently drops the temporal mixing WideAdapter "
                f"performs, which is a different model, not a reshaping")
        z = self.adapter(window)[:, -1]          # the CURRENT frame's field
        pooled = self._pool(z).mean(dim=1)
        return {"controls": self.policy.sample(pooled, v0, gen=gen),
                "lat_logits": self.lat_head(pooled),
                "lon_logits": self.lon_head(pooled),
                "cond": pooled}

    def assert_no_future_at_inference(self) -> None:
        """⛔ The isolated mask, as a CHECK rather than a convention.

        v6's imagination discipline lives in DEFAULTS (``mpc_w_consist=0.0``,
        ``fallback_trigger=False``) — and a default is not an invariant. This
        walks the deployment path's signature and refuses any parameter that
        could carry a future field or a rollout into it.
        """
        import inspect
        banned = ("future", "rollout", "imagin", "z_next", "horizon_field")
        sig = inspect.signature(self.act).parameters
        bad = [p for p in sig for b in banned if b in p.lower()]
        if bad:
            raise RuntimeError(
                f"REF-D deployment path accepts {bad} — future prediction is a "
                f"TRAINING signal only (SimWAM Tab. 3: isolated 90.3 vs "
                f"bidirectional 90.2; ours: open-loop 0.45 m -> closed-loop "
                f"1.69 m with imagination in the loop)")
