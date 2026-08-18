"""REF-A **v1** — the redesigned frozen-encoder arm.

⭐ WHAT CHANGED FROM REF-A, AND WHY EACH CHANGE HAS A NAMED SOURCE.

REF-A (`refa.py`) was: frozen **DINOv2-B/14**, 224x224 -> **256 tokens / 51.39
deg**, a **pooling/grid adapter into a compact state**, a **supervised** head on
top, **no test-time planning**. Measured: ADE@2s **2.1675 m** (T0), plateaued.

Two independent lines of evidence then landed on the same day:

* **Ours** (`…/Research/2026-08-18-encoder-localisation/`, verdict
  ``P2-PRESERVED``): the information REF-A was accused of losing is **present
  and preserved at five measured stages** — raw features 0.5285, trained adapter
  0.4751, predictor latent 0.4762 / 0.4863 — arriving intact at the exact latent
  the eval decoded. The trained adapter did **not** collapse (per-dim std 0.8011
  vs 0.220 random). ⇒ The deficit is on the **consumption** side.
* **The literature** (`…/Research/2026-08-18-frozen-encoder-literature/`):
  frozen encoders succeed in exactly two configurations — (A) a *very large*
  frozen VLM + wide visual interface + supervised head (FROST-Drive: frozen 14B
  **8.17 RFS / 1.04 m** beats the SAME encoder fine-tuned **8.13 / 1.47**, while
  a frozen *ImageNet* ViT is the worst arm in the table at **7.39 / 2.28**), or
  (B) a *moderate* frozen encoder + **future-feature prediction** + **test-time
  planning** (DINO-WM, V-JEPA 2-AC; in driving DeepSight and LAW). REF-A had
  configuration A's consumer with configuration B's encoder class **and neither
  one's compensating strength** — the one cell nothing succeeds in.

⇒ **v1 commits to configuration B, fully**, and fixes the interface defects that
were configuration-independent.

| # | change | source |
|---|---|---|
| 1 | encoder **DINOv3** (ViT-L/16, d=1024), still frozen, still cached | DeepSight uses DINOv3-ViT-L/16 as its world-state target; PI directive |
| 2 | **640 patch tokens, 120 deg HFOV, 256x640** (was 256 tokens / 51.39 deg) | DINOv2's H/14 map is documented-insufficient for small/distant objects; our own w120 geometry decision |
| 3 | ⛔ **no bottleneck**: adapter width >= encoder width (1024) | FROST-Drive interface-width ablation: 5120-d **8.17** vs 256-d **7.68** on the SAME frozen encoder |
| 4 | primary objective = **predict future PATCH features** (L2), not a supervised head | DINO-WM: latent L2, "no auxiliary reconstruction, reward, or terminal losses", no policy head |
| 5 | **patch tokens only, never CLS/pooled** for the predictive path | DINO-WM ablation: global R3M / ResNet18 / **DINOv2 CLS** "significantly degrades" |
| 6 | behaviour from **iCEM + MPC at test time** (`refa_v1_plan.py`) | DINO-WM / V-JEPA 2-AC / GPC; repairs C101 |
| 7 | hierarchy **kept**: strategic --FiLM--> tactical --FiLM--> operative | our `fourbrain.run_hierarchy`; PI directive |
| 8 | goals enter the **planning COST**, not only a head | v3 direction (`tanitad-v3-direction`): "target-speed + mode-switching become the PLANNING COST not a head" |
| 9 | **6 s predictive horizon** at three rates, strategic on its OWN predictor over a strategy-only subspace | PI directive + three-planner hierarchy directive |

⚠️ **WHAT THIS DESIGN DOES NOT CLAIM.** Nothing here is a result. The ranking
that motivates it is a hypothesis ranking, and the recipe imports our
known-worst component (the action search) — which is why `refa_v1_plan.py`
carries a structural floor *and* a cost-fidelity gate rather than trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.fourbrain import StrategicPolicy, TacticalPolicy
# ⛔ ONE VOCABULARY SOURCE. These tuples are IMPORTED, never re-declared — a
# second copy is a second vocabulary, and the programme has already paid for
# that once (see the defect note below).
from tanitad.models.v6 import TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS
from tanitad.refs.refa_v1_plan import PlanConfig, icem_plan, unicycle_paths

__all__ = ["RefAV1Config", "RefAV1", "DINOV3_GEOMETRY",
           "TACTICAL_LAT_ACTIONS", "TACTICAL_LON_ACTIONS"]

#: The frozen-encoder contract. Cached offline; the encoder never enters the
#: graph (REF-A stability item 2, preserved verbatim in v1).
DINOV3_GEOMETRY = {
    "model": "dinov3-vit-l-16",
    "d_enc": 1024,
    "patch": 16,
    "height": 256, "width": 640,          # 2.5 aspect — SAME as v6's 224x560,
    "grid_h": 16, "grid_w": 40,           # so this is a pure resolution scale
    "n_tokens": 640,                      # 16*40, vs REF-A's 256
    "hfov_deg": 120.0,                    # vs REF-A's 51.39
    "tokens_include_cls": False,          # ⛔ patch tokens ONLY (change #5)
}


@dataclass
class RefAV1Config:
    """Every number that defines the arm, in one place, so a launch can be
    diffed against a checkpoint."""

    # --- frozen visual interface (change #1, #2, #3) ---------------------- #
    d_enc: int = 1024
    n_tokens: int = 640
    d_state: int = 1024                   # ⛔ MUST be >= d_enc (change #3)

    # --- operative: dense, 6 s (change #9) -------------------------------- #
    op_dt: float = 0.2
    op_steps: int = 30                    # 30 * 0.2 = 6.0 s
    op_layers: int = 6
    op_heads: int = 8
    op_window: int = 4                    # observed frames fed to the predictor

    # --- tactical: coarse, 6 s, on learned query tokens ------------------- #
    tac_dt: float = 0.6
    tac_steps: int = 10                   # 10 * 0.6 = 6.0 s
    tac_queries: int = 64
    tac_layers: int = 4

    # --- strategic: its OWN predictor on a strategy-only subspace --------- #
    str_dt: float = 1.5
    str_steps: int = 4                    # 4 * 1.5 = 6.0 s
    str_dim: int = 256
    str_layers: int = 2

    # --- control / planning ----------------------------------------------- #
    a_dim: int = 2                        # (a, kappa) — Alpamayo-2 form
    plan_horizon_s: float = 2.0           # optimised window; cost spans 6 s
    goal_times_s: tuple = (2.0, 4.0, 6.0)

    #: ⛔ WHERE THE SEARCH ROLLS, AND WHY IT IS NOT THE OPERATIVE FIELD.
    #: MEASURED on an RTX 4060 at the real geometry: a 10-step rollout of the
    #: 640x1024 operative field costs **160 ms per candidate** and scales
    #: linearly (2553 / 5129 / 10519 ms at n = 16 / 32 / 64 — the GPU is already
    #: saturated at n=16), i.e. ~6 candidates/s. DINO-WM's published
    #: configuration (300 samples x 30 iterations, ~1975 rollouts after decay)
    #: therefore costs **325 s per MPC tick** here, ~54 s on an A40-class card.
    #: That is not a planner, it is a batch job.
    #:
    #: ⭐ The hierarchy already contains the fix: the TACTICAL field is 64 query
    #: tokens instead of 640, so the same search costs ~1/10th. Searching a
    #: manoeuvre against the coarse tactical world and then VERIFYING the winner
    #: on the full operative field is coarse-to-fine, and it is what the
    #: tactical level is for. ``"operative"`` stays reachable so the deviation
    #: from DINO-WM is a measurable ablation, not an unstated compromise.
    plan_level: str = "tactical"          # "tactical" | "operative"
    verify_on_operative: bool = True      # re-score the winner on 640 tokens

    # --- hierarchy brains (kept from REF-A / flagship) -------------------- #
    tactical_cfg: TacticalPolicyConfig | None = None
    strategic_cfg: StrategicPolicyConfig | None = None

    # --- loss weights: feature prediction is PRIMARY (change #4) ---------- #
    w_feat_op: float = 1.0
    w_feat_tac: float = 0.5
    w_feat_str: float = 0.25
    w_aux_head: float = 0.1               # imitation proposal: AUXILIARY only

    def sanity(self) -> None:
        if self.d_state < self.d_enc:
            raise ValueError(
                f"d_state ({self.d_state}) < d_enc ({self.d_enc}) — change #3 "
                "forbids a bottleneck below the encoder width; FROST-Drive "
                "measured 8.17 -> 7.68 RFS on exactly this axis")
        if abs(self.op_dt * self.op_steps - 6.0) > 1e-6:
            raise ValueError("operative horizon must reach exactly 6.0 s")
        if abs(self.tac_dt * self.tac_steps - 6.0) > 1e-6:
            raise ValueError("tactical horizon must reach exactly 6.0 s")
        if abs(self.str_dt * self.str_steps - 6.0) > 1e-6:
            raise ValueError("strategic horizon must reach exactly 6.0 s")
        if int(round(self.plan_horizon_s / self.op_dt)) > self.op_steps:
            raise ValueError("plan horizon exceeds the operative rollout")

    @property
    def plan_steps(self) -> int:
        return int(round(self.plan_horizon_s / self.op_dt))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    """Pre-norm transformer block. LayerNorm only — no BatchNorm, no dropout
    (REF-A stability item 5: I2 batch-consistency, preserved in v1)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class FeatureStandardizerV1(nn.Module):
    """Per-channel standardisation with FROZEN buffers, fit once over the train
    corpus (REF-A stability item 1, carried over unchanged). Refitting a loaded
    checkpoint raises — the stats are part of the parity contract."""

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, feats: Tensor) -> None:
        if bool(self.fitted):
            raise RuntimeError("standardizer already fitted — refusing to refit "
                               "(the stats are part of the parity contract)")
        flat = feats.reshape(-1, feats.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-3))
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class WideAdapter(nn.Module):
    """Token-preserving adapter: [B,T,N,d_enc] -> [B,T,N,d_state].

    ⛔ Deliberately NOT a readout/pooling head. REF-A's adapter mapped the token
    grid into a compact state, which is the bottleneck change #3 forbids and the
    surface DINO-WM's ablation says must stay spatial. Per-token MLP (shared
    across tokens, so parameter cost is independent of ``n_tokens``) + a learned
    spatial embedding + a depthwise temporal mix."""

    def __init__(self, cfg: RefAV1Config):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, cfg.n_tokens, cfg.d_state))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.d_enc),
            nn.Linear(cfg.d_enc, cfg.d_state), nn.GELU(),
            nn.Linear(cfg.d_state, cfg.d_state))
        self.tmix = nn.Conv1d(cfg.d_state, cfg.d_state, kernel_size=3,
                              padding=1, groups=cfg.d_state)
        self.out = nn.LayerNorm(cfg.d_state)

    def forward(self, feats: Tensor) -> Tensor:
        b, t, n, _ = feats.shape
        x = self.proj(feats) + self.pos
        y = x.permute(0, 2, 3, 1).reshape(b * n, -1, t)      # [B*N, d, T]
        x = x + self.tmix(y).reshape(b, n, -1, t).permute(0, 3, 1, 2)
        return self.out(x)


class TokenFieldPredictor(nn.Module):
    """⭐ THE ARCHITECTURAL HEART OF v1 — DINO-WM's predictor, on our field.

    Consumes a causal window of token fields plus per-step actions and predicts
    the **future patch-feature field**. Action embedding is broadcast over tokens
    and concatenated-then-projected, which is DINO-WM's exact conditioning
    scheme. ``intent`` (from the tactical brain) is ADDED to the action
    conditioning, which is how our hierarchy already closes onto the operative
    predictor (`fourbrain.run_hierarchy`) — so change #7 costs no new mechanism.
    """

    def __init__(self, cfg: RefAV1Config, d: int, layers: int, heads: int = 8,
                 intent_dim: int | None = None):
        super().__init__()
        self.d = d
        self.act = nn.Sequential(nn.Linear(cfg.a_dim, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.intent = nn.Linear(intent_dim, d) if intent_dim else None
        self.mix = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList([_Block(d, heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def step(self, field: Tensor, action: Tensor,
             intent: Tensor | None = None) -> Tensor:
        """One latent step. ``field`` [B,N,d], ``action`` [B,a_dim] -> [B,N,d].

        Residual by construction (``z_hat = z + delta``): the predictor learns
        the CHANGE, so a zero-action step is near-identity at init and the
        6 s rollout does not drift on the first gradient.
        """
        a = self.act(action)
        if intent is not None and self.intent is not None:
            a = a + self.intent(intent)
        a = a[:, None, :].expand(-1, field.shape[1], -1)
        x = self.mix(torch.cat([field, a], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        return field + self.head(x)

    def rollout(self, field: Tensor, actions: Tensor,
                intent: Tensor | None = None,
                last_only: bool = False) -> Tensor:
        """``actions`` [B,K,a_dim] -> predicted fields [B,K,N,d].

        ⛔ ``last_only`` IS A MEMORY REQUIREMENT, NOT AN OPTION, ON THE PLANNING
        PATH. MEASURED at the real geometry: one latent field is
        640 tokens x 1024 d = 1.31 MB in fp16, so a CEM population of 300 held
        for a 10-step rollout is **300 x 10 x 1.31 MB = 3.9 GB** of stored
        intermediates for a cost that only ever reads the FINAL field. Storing
        them would have made the arm un-runnable on anything but an 80 GB card
        and the failure would have surfaced only at planning time — the C111
        class (an analysis-time failure after the compute is paid).
        """
        z = field
        if last_only:
            for k in range(actions.shape[1]):
                z = self.step(z, actions[:, k], intent=intent)
            return z
        out = []
        for k in range(actions.shape[1]):
            z = self.step(z, actions[:, k], intent=intent)
            out.append(z)
        return torch.stack(out, dim=1)


class StrategicSubspacePredictor(nn.Module):
    """The strategic brain's OWN predictor, on a strategy-only latent subspace.

    Three-planner directive: *"strategic gets its OWN predictor on a
    strategy-only latent subspace"*. The subspace is a learned linear read of the
    pooled field — deliberately narrow (``str_dim``), because a route hypothesis
    at 1.5 s cadence should not carry lane-level texture, and because keeping it
    separate is what makes the strategic prediction falsifiable on its own.
    """

    def __init__(self, cfg: RefAV1Config):
        super().__init__()
        self.read = nn.Sequential(nn.LayerNorm(cfg.d_state),
                                  nn.Linear(cfg.d_state, cfg.str_dim))
        self.act = nn.Linear(cfg.a_dim, cfg.str_dim)
        self.blocks = nn.ModuleList(
            [_Block(cfg.str_dim, 4) for _ in range(cfg.str_layers)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.str_dim),
                                  nn.Linear(cfg.str_dim, cfg.str_dim))

    def subspace(self, field: Tensor) -> Tensor:
        return self.read(field.mean(dim=-2))          # pool tokens -> [B, str]

    def rollout(self, s: Tensor, actions: Tensor) -> Tensor:
        out = []
        for k in range(actions.shape[1]):
            x = s + self.act(actions[:, k])
            for blk in self.blocks:
                x = blk(x[:, None, :]).squeeze(1)
            s = s + self.head(x)
            out.append(s)
        return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- #
# The arm
# --------------------------------------------------------------------------- #
class RefAV1(nn.Module):
    """Frozen DINOv3 field -> wide adapter -> three-rate predictive hierarchy,
    with behaviour produced by planning rather than regression.

    ``forward`` is TRAINING (feature prediction + auxiliaries). ``plan`` is
    DEPLOYMENT (iCEM/MPC over the operative predictor, cost assembled from the
    tactical and strategic goals). They share every weight; nothing in the
    planning path is trained to imitate a trajectory.
    """

    def __init__(self, cfg: RefAV1Config | None = None):
        super().__init__()
        cfg = cfg or RefAV1Config()
        cfg.sanity()
        self.cfg = cfg

        self.std = FeatureStandardizerV1(cfg.d_enc)
        self.adapter = WideAdapter(cfg)

        # ⚠️ ``d_intent`` (not ``intent_dim``) — the field name on
        # TacticalPolicyConfig. Reading the wrong attribute would have silently
        # built the predictor WITHOUT intent conditioning and quietly deleted
        # change #7 (the hierarchy) while every shape still checked out.
        intent_dim = (cfg.tactical_cfg.d_intent
                      if cfg.tactical_cfg is not None else None)
        self.operative = TokenFieldPredictor(cfg, cfg.d_state, cfg.op_layers,
                                             cfg.op_heads, intent_dim)
        self.tac_queries = nn.Parameter(
            torch.zeros(1, cfg.tac_queries, cfg.d_state))
        nn.init.trunc_normal_(self.tac_queries, std=0.02)
        self.tac_pool = nn.MultiheadAttention(cfg.d_state, 8, batch_first=True)
        self.tactical = TokenFieldPredictor(cfg, cfg.d_state, cfg.tac_layers,
                                            cfg.op_heads, intent_dim)
        self.strategic = StrategicSubspacePredictor(cfg)

        # Hierarchy brains — the SAME classes the flagship holds, so the
        # conditioning chain is identical and comparisons stay on one axis.
        # Signature is (cfg, state_dim, window) — the brains compose on ANY
        # compact state, which is why the flagship and REF-A can share them.
        self.strategic_policy = (
            StrategicPolicy(cfg.strategic_cfg, cfg.d_state, cfg.op_window)
            if cfg.strategic_cfg is not None else None)
        self.tactical_policy = (
            TacticalPolicy(cfg.tactical_cfg, cfg.d_state, cfg.op_window,
                           d_cond=cfg.strategic_cfg.d_ctx)
            if cfg.tactical_cfg is not None else None)
        if (self.tactical_policy is not None) != (self.strategic_policy is not None):
            raise ValueError(
                "the brains are a MATCHED SET — the tactical policy is "
                "FiLM-conditioned on the strategic ctx (d_cond=d_ctx), so one "
                "without the other is a broken conditioning chain, not a "
                "smaller model")

        # ⛔ FACTORED LAT × LON TACTICAL HEADS — AND WHY THEY EXIST AT ALL.
        #
        # DEFECT FOUND 2026-08-18, after v1 was first committed: the shared
        # `TacticalPolicy` emits ONE `maneuver_logits [B, 5]` over
        # `refb.MANEUVER_CLASSES = (lane_keep, turn_left, turn_right,
        # accelerate, brake_stop)` — a softmax that MIXES the lateral and
        # longitudinal axes. `v6.py` names that mixing "the programme's single
        # largest known defect", retired BY DESIGN, and REF-C v3 already reads
        # `tac.N_LAT` / `tac.N_LON`. v1 silently inherited the retired form
        # because it reused the legacy brain with its DEFAULT config — every
        # shape checked out and nothing failed.
        #
        # MEASURED consequences of the mixed head (D-TAC1, 2026-08-03): shipped
        # 5-way decode accuracy 0.7581 / macro-recall 0.5313 with `accelerate`
        # NEVER PREDICTED, against 0.9348 / 0.8290 factored; and the 5-way label
        # destroys 9.68 % (132/1364) of the longitudinal decisions outright.
        #
        # ⇒ v1 decodes the tactical action on TWO independent heads over the
        # v6 vocabulary, imported from `v6.py` so there is exactly one source.
        # The legacy `maneuver_logits` is NOT consumed anywhere in v1.
        self.n_lat, self.n_lon = len(TACTICAL_LAT_ACTIONS), len(TACTICAL_LON_ACTIONS)
        d_int = intent_dim or cfg.d_state
        self.lat_head = nn.Sequential(nn.LayerNorm(d_int),
                                      nn.Linear(d_int, self.n_lat))
        self.lon_head = nn.Sequential(nn.LayerNorm(d_int),
                                      nn.Linear(d_int, self.n_lon))

        # ⚠️ AUXILIARY imitation proposal — NOT the behaviour source. It exists
        # only to seed the planner (GPC: "generative control proposes, MPC
        # disposes"). w_aux_head is 0.1 and it never gates a metric.
        self.proposal = nn.Sequential(
            nn.LayerNorm(cfg.d_state), nn.Linear(cfg.d_state, 512), nn.GELU(),
            nn.Linear(512, cfg.plan_steps * cfg.a_dim))

    # -- encoding ---------------------------------------------------------- #
    def encode(self, feats: Tensor) -> Tensor:
        """Cached DINOv3 patch features [B,T,N,d_enc] -> state field."""
        if feats.shape[-1] != self.cfg.d_enc:
            raise ValueError(f"expected d_enc={self.cfg.d_enc}, "
                             f"got {feats.shape[-1]}")
        if feats.shape[-2] != self.cfg.n_tokens:
            raise ValueError(f"expected {self.cfg.n_tokens} patch tokens, got "
                             f"{feats.shape[-2]} — v1 forbids a narrowed visual "
                             "interface (change #2/#3)")
        return self.adapter(self.std(feats))

    def _tac_field(self, field: Tensor) -> Tensor:
        q = self.tac_queries.expand(field.shape[0], -1, -1)
        return self.tac_pool(q, field, field, need_weights=False)[0]

    # -- training ---------------------------------------------------------- #
    def forward(self, feats: Tensor, actions: Tensor, *,
                future_feats: Tensor | None = None,
                nav_cmd: Tensor | None = None,
                ego: Tensor | None = None) -> dict:
        """``feats`` [B,W,N,d_enc] observed window, ``actions`` [B,K,a_dim].

        ``future_feats`` [B,K,N,d_enc] are the **targets** — future patch
        features in the SAME standardised space. That is the primary loss
        (change #4): the model is asked to carry the world forward, not to hit a
        trajectory label.
        """
        field = self.encode(feats)                       # [B,W,N,d]
        last = field[:, -1]
        # ⚠️ The brains take a STATE WINDOW [B, W, D], not a single state — the
        # window length is baked into their positional embeddings, so passing
        # [B,1,D] would be a silent shape-compatible wrong input.
        pooled_win = field.mean(dim=-2)                  # [B,W,D]
        pooled = pooled_win[:, -1]

        intent = None
        out: dict = {}
        if self.strategic_policy is not None and self.tactical_policy is not None:
            b = pooled.shape[0]
            nav = (torch.zeros(b, dtype=torch.long, device=pooled.device)
                   if nav_cmd is None else nav_cmd)
            strat = self.strategic_policy(pooled_win, nav, ego=ego)
            tac = self.tactical_policy(pooled_win, strat["ctx"], ego=ego)
            intent = tac["intent"]
            out.update({"ctx": strat["ctx"], "intent": intent,
                        "route_logits": strat.get("route_logits")})
            # ⭐ THE FACTORED DECODE — v1's tactical action. Two independent
            # softmaxes, so a longitudinal decision can never be outvoted by a
            # lateral one sharing its logit space.
            out["lat_logits"] = self.lat_head(intent)
            out["lon_logits"] = self.lon_head(intent)
            # The legacy mixed head is passed through under a name that says
            # what it is, so nothing downstream can consume it by accident
            # while looking like it read a tactical action.
            out["legacy_mixed_maneuver_logits_DO_NOT_USE"] = \
                tac.get("maneuver_logits")

        out["op_pred"] = self.operative.rollout(last, actions, intent=intent)
        tac_a = actions[:, ::max(1, int(round(self.cfg.tac_dt / self.cfg.op_dt)))]
        out["tac_pred"] = self.tactical.rollout(
            self._tac_field(last), tac_a[:, :self.cfg.tac_steps], intent=intent)
        str_a = actions[:, ::max(1, int(round(self.cfg.str_dt / self.cfg.op_dt)))]
        out["str_pred"] = self.strategic.rollout(
            self.strategic.subspace(last), str_a[:, :self.cfg.str_steps])
        out["proposal"] = self.proposal(pooled).reshape(
            -1, self.cfg.plan_steps, self.cfg.a_dim)

        if future_feats is not None:
            tgt = self.adapter(self.std(future_feats))
            k = min(tgt.shape[1], out["op_pred"].shape[1])
            out["loss_feat_op"] = F.mse_loss(out["op_pred"][:, :k], tgt[:, :k])
            tq = torch.stack([self._tac_field(tgt[:, i])
                              for i in range(tgt.shape[1])], dim=1)
            step = max(1, int(round(self.cfg.tac_dt / self.cfg.op_dt)))
            tq = tq[:, ::step][:, :out["tac_pred"].shape[1]]
            kt = min(tq.shape[1], out["tac_pred"].shape[1])
            out["loss_feat_tac"] = F.mse_loss(out["tac_pred"][:, :kt], tq[:, :kt])
            sstep = max(1, int(round(self.cfg.str_dt / self.cfg.op_dt)))
            st = self.strategic.subspace(tgt.flatten(0, 1)).reshape(
                tgt.shape[0], tgt.shape[1], -1)[:, ::sstep]
            ks = min(st.shape[1], out["str_pred"].shape[1])
            out["loss_feat_str"] = F.mse_loss(out["str_pred"][:, :ks],
                                              st[:, :ks])
            out["loss"] = (self.cfg.w_feat_op * out["loss_feat_op"]
                           + self.cfg.w_feat_tac * out["loss_feat_tac"]
                           + self.cfg.w_feat_str * out["loss_feat_str"])
        return out

    # -- deployment: behaviour by PLANNING, not regression ----------------- #
    @torch.no_grad()
    def plan(self, feats: Tensor, *, v0: float, goal_field: Tensor | None = None,
             target_speed: float | None = None, nav_cmd: Tensor | None = None,
             plan_cfg: PlanConfig | None = None, prev_elites: Tensor | None = None,
             cost_chunk: int = 64):
        """One MPC tick for ONE window (B must be 1).

        The cost is where the hierarchy earns its keep (change #8): the tactical
        target speed and the strategic goal field enter as **cost terms**, not as
        head outputs to be regressed. ``goal_field`` defaults to the tactical
        brain's own imagined 6 s field, which is what makes this
        goal-conditioning rather than goal-following.
        """
        if feats.shape[0] != 1:
            raise ValueError("plan() is a single-window API (B must be 1)")
        cfg = self.cfg
        pc = plan_cfg or PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt)
        if pc.horizon != cfg.plan_steps:
            raise ValueError(f"plan horizon {pc.horizon} != cfg.plan_steps "
                             f"{cfg.plan_steps}")
        field = self.encode(feats)
        last = field[:, -1]
        pooled_win = field.mean(dim=-2)
        pooled = pooled_win[:, -1]

        intent = None
        if self.strategic_policy is not None and self.tactical_policy is not None:
            nav = (torch.zeros(1, dtype=torch.long, device=pooled.device)
                   if nav_cmd is None else nav_cmd)
            strat = self.strategic_policy(pooled_win, nav)
            intent = self.tactical_policy(pooled_win, strat["ctx"])["intent"]

        proposal = self.proposal(pooled).reshape(cfg.plan_steps, cfg.a_dim)
        v0_t = torch.as_tensor([v0], dtype=torch.float32, device=feats.device)

        if cfg.plan_level not in ("tactical", "operative"):
            raise ValueError(f"plan_level must be tactical|operative, "
                             f"got {cfg.plan_level!r}")
        coarse = cfg.plan_level == "tactical"
        search_pred = self.tactical if coarse else self.operative
        search_z = self._tac_field(last) if coarse else last
        search_goal = (None if goal_field is None else
                       (self._tac_field(goal_field) if coarse else goal_field))

        def _cost_chunk(controls: Tensor, pred=None, z0=None,
                        goal=None) -> Tensor:
            pred = pred or search_pred
            z0 = search_z if z0 is None else z0
            goal = search_goal if goal is None else goal
            n = controls.shape[0]
            z = z0.expand(n, -1, -1)
            # last_only: the cost reads the terminal field only (see rollout).
            zk = pred.rollout(z, controls, intent=intent, last_only=True)
            c = torch.zeros(n, device=controls.device)
            if goal is not None:
                g = goal.expand(n, -1, -1)
                c = c + (1.0 - F.cosine_similarity(
                    zk.flatten(1), g.flatten(1), dim=-1))
            jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt
            c = c + 0.02 * jerk.pow(2).mean(-1)                    # comfort
            c = c + 0.05 * controls[..., 1].pow(2).mean(-1)        # curvature
            if target_speed is not None:
                v_end = v0 + controls[..., 0].sum(-1) * pc.dt
                c = c + 0.10 * (v_end - target_speed).pow(2)
            return c

        def cost_fn(controls: Tensor) -> Tensor:
            """Chunked so the population size is a SEARCH parameter and not a
            memory limit — DINO-WM's N=300 must remain reachable on a 24 GB
            card, which it is not if the whole population rolls at once."""
            if controls.shape[0] <= cost_chunk:
                return _cost_chunk(controls)
            return torch.cat([_cost_chunk(controls[i:i + cost_chunk])
                              for i in range(0, controls.shape[0], cost_chunk)])

        res = icem_plan(cost_fn, v0=v0, cfg=pc, proposal=proposal,
                        prev_elites=prev_elites, device=feats.device)

        # ⭐ COARSE-TO-FINE: the search ran on the tactical field; re-score the
        # WINNER (and the baselines it beat) on the full operative field, so the
        # reported cost is the fine-grained one and a coarse-level mistake shows
        # up as a rank flip rather than disappearing. This costs a handful of
        # rollouts, not a population.
        if coarse and cfg.verify_on_operative and self.tactical is not None:
            cands = {"plan": res.controls}
            if pc.inject_baselines:
                from tanitad.refs.refa_v1_plan import _baseline_controls
                cands.update(_baseline_controls(pc, v0, feats.device, proposal))
            names = list(cands)
            stack = torch.stack([cands[k] for k in names])
            fine = _cost_chunk(stack, pred=self.operative, z0=last,
                               goal=goal_field)
            res.fine_costs = {k: float(v) for k, v in zip(names, fine)}
            best = min(res.fine_costs, key=res.fine_costs.get)
            res.fine_best = best
            res.coarse_fine_agree = (best == "plan")
        return res

    # -- bookkeeping -------------------------------------------------------- #
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_encoder_parameters(self) -> int:
        """0 by construction — features are data tensors on disk. Kept as a
        method so a test can assert the invariant rather than a comment."""
        return 0
