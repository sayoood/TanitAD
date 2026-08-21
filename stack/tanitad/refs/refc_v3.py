"""REF-C v3 — the goal-mediated strategic/tactical/operative hierarchy on the
supervised arm (the 4B-dominance proof track).

Design + pre-registration: ``TanitAD Research Hub/Architecture & Inference/
Research/2026-08-18-refc-v3-design/`` (``REFC_V3_DESIGN.md`` carries the edge
list E1..E12 this module implements; ``PREREG_REFC_V3.md`` commits the
experiment). This docstring repeats only what a reader of the CODE needs.

WHAT THIS MODULE IS
============================================================================
:class:`RefCV3Model` composes an UNMODIFIED :class:`tanitad.refs.refc.RefCModel`
with a goal cascade built from components the programme already measured:

  * strategic state  = the core's own ``StrategicCtx`` GRU (existing, trained);
    a new 195-param head reads a PREDICTED geometric route goal ``g_str`` off it;
  * tactical state   = :class:`tanitad.models.tactical.PhiTac` — the causal-TCN
    window pool that C116 found *tested, trained, and never called*. v3 WIRES
    the orphaned component instead of rebuilding it (one-implementation rule,
    the ``refc_select`` precedent). Conditioned on ``g_str`` through a
    ZERO-INIT FiLM (E4), so the cascade starts bit-inert;
  * tactical outputs = factored lat(3)/lon(3) heads (the 5-way priority
    collapse provably destroys the longitudinal decision — ``refc_tactical``),
    geometric goals ``g_tac`` [K, 4] = (x, y, heading, speed) at {2, 4, 6} s
    (the E4.1 label layout, verbatim), and a tactical latent for the decoder;
  * conditioning     = REF-C's two EXISTING external-tactical-brain ports —
    ``maneuver_logits`` (H19 anchor prior) and ``target_latent`` (zero-init
    FiLM on the decoder condition) — fed through the ``hierarchy_hook``
    argument of ``RefCModel.forward`` (one gated hook; with ``None`` the core
    forward is byte-identical to the pre-hook file);
  * selection        = distance to the predicted tactical goal at the 2 s slot,
    through :class:`tanitad.models.v6.GoalDistanceScorer` — the ONE selection
    mechanism our measurements support (candidate-INDEPENDENT reference: no
    winner's curse; error-rank FALLS with N; requirement curve measured —
    σ=0.5 m beats the trained selector separated, σ=1.0 m loses separated,
    ⇒ admission gate σ ≤ 0.8 m @ 2 s, a gate not a hope). Zero-init
    ``goal_gate`` + ``refc_select.apply_seam_clamp`` keep the emission
    bit-identical to the core at init.

GRADIENT POLICY (the v6 ``_cut()`` discipline, adopted)
============================================================================
Goals travel DOWNWARD detached: ``g_str`` into the tactical FiLM (E4), the
tactical latent into the decoder port (E7), and the goal point into selection
(E9). Each level trains by its OWN supervision; the selection objective cannot
corrupt the goal head (the ``cons_detach`` frozen-predictor discipline, applied
to goals). ⛔ CONSEQUENCE (C120): gradient probes are structurally BLIND to
these forward paths — the audit for the PI's goal/situation disjointness ruling
is therefore the INTERVENTION probe (``tanitad.eval.goal_provenance``), wired
into the tests and the launch preflight, never a doc note.

WHAT IS DELIBERATELY NOT HERE
============================================================================
* No learned fan re-scorer (SEL-1 REFUSED: winner's curse; v1.2 NOT separated).
* No roll-consistency argmin (+5.9787 m WORSE, measured).
* No MPC/CEM (C101: 35.8 % worse than CV at T1; and ``law_head`` cannot be
  iterated — argued from source in ``refc_select.py``).
* No ego state into any goal head (edge E11, REFUSED — pinned by test + audit;
  matches the C120 finding on v6: *"every goal head is a function of frames
  alone"*).
* No supplied route at inference (E12): the LAN corridor is the TRAINING LABEL
  for ``g_str`` only (``refc_goal_config`` precedent).

PARITY NOTE (the 6 s horizon). ``V3_HORIZONS`` extends the plan to 6.0 s. The
window ENUMERATION must keep ``max_horizon=20`` and fetch steps 21..60 by
CLAMP + validity mask (``tanitad/data/_contract.py:120`` re-selects windows
otherwise — REFC_V3_DESIGN.md §3). The loss helpers here therefore all take a
mask and are pinned to contribute EXACTLY ZERO gradient at masked slots.

⚠️ The 2 s-band reachability numbers (72.08 % clipped / 3.58x) were measured at
horizon_s=2.0 and MUST NOT be quoted for the 6 s band — the band is re-derived
from ``max(horizons)`` and its statistics are a property of THESE anchors at
THIS horizon (re-measure, never inherit).

Evidence classes: every number above is MEASURED with its artifact named in
REFC_V3_DESIGN.md §0; param costs of this module are MEASURED by
:func:`param_breakdown_v3` and pinned in band by ``tests/test_refc_v3.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models.tactical import PhiTac
from tanitad.refs import refc
from tanitad.refs import refc_select as sl
from tanitad.refs import refc_tactical as tac

__all__ = [
    "V3_HORIZONS", "SEAM_SLOT", "GOAL_TAU_STEPS", "RefCV3Config",
    "refc_v3_flat_config", "refc_v3_hier_config", "refc_v3_smoke_config",
    "refc_v3_sized_config", "refc_v3_small_config", "refc_v3_xl_config",
    "V3_SIZES",
    "RefCV3Model", "config_delta", "param_breakdown_v3",
    "masked_goal_loss", "strategic_goal_loss", "selection_ce",
    "freeze_history_report",
]

#: 6.0 s @ 10 Hz — BINDING (PLAN_STEPS=60, DT=0.1). 0.5 s stride through the
#: operative band (0, 2], 1 s stride through the tactical band (2, 6].
V3_HORIZONS: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 60)
#: Index of step 20 (= 2.0 s) in V3_HORIZONS — the operative/tactical seam and
#: the SELECTION slot (the goal-distance admission curve was measured at 2 s
#: endpoints; the 6 s endpoint may not select until its own curve is measured).
SEAM_SLOT: int = 3
#: Tactical goal horizons in 10 Hz steps — MUST equal
#: ``refb_labels.GOAL_TAC_TAUS_STEPS`` / ``tactical.GOAL_TAC_TAUS_STEPS``
#: (asserted by tests; re-declared here because refs/ must not import scripts/).
GOAL_TAU_STEPS: tuple[int, ...] = (20, 40, 60)
#: Goal row layout, E4.1 verbatim: (x, y, heading, speed).
GOAL_DIMS: int = 4


# ============================================================================
# Configs — the dominance pair differs in EXACTLY the registered lever set,
# and config_delta() below is the instrument that PINS that (C122's rule:
# "an ablation's 'everything else identical' must be DERIVED and pinned").
# ============================================================================

@dataclass
class RefCV3Config:
    """v3 = a core RefCConfig + the goal-cascade switch + shared sizing.

    ``hier`` is THE dominance lever (one switch builds/withholds the cascade).
    Every other field holds the SAME value in both arms, so the derived config
    delta is {hier, core.graft_target_latent} and nothing else — the decoder's
    target-latent FiLM must be CONSTRUCTED on the H arm (a port that is silently
    ignored is the bug class the seam refuses; ``refc.py:1313`` skips a None
    film), and that construction is part of the registered, measured delta.
    """
    core: refc.RefCConfig = field(default_factory=refc.RefCConfig)
    hier: bool = False                # THE lever: goal cascade on/off
    d_tac: int = 512                  # PhiTac output width (tactical state)
    tac_hidden: int = 256             # PhiTac conv width (1.71 M at 512/256/512)
    d_gcond: int = 64                 # strategic-goal conditioning embed (E4)
    goal_tau_steps: tuple[int, ...] = GOAL_TAU_STEPS
    scorer_tau_m: float = 1.0         # GoalDistanceScorer distance temperature
    seam_clamp: float = 1.0           # S4 cap on the selection graft (E9)
    seam_fail: float = 1.5
    seam_fail_frac: float = 0.75
    seam_fail_patience: int = 50
    #: MEASURED admission (v6.GoalDistanceScorer requirement curve, 881 w /
    #: 40 ep): the goal head must reach ≤ this 1-sigma endpoint error at 2 s
    #: before goal-selection deltas may be READ as improvements (PREREG §5).
    #: An eval-read rule, not a runtime switch — the zero-init gate + CE learn
    #: freely either way, and the read discipline lives in the prereg.
    admission_sigma_m: float = 0.8

    @property
    def n_goal_taus(self) -> int:
        return len(self.goal_tau_steps)


def _v3_core_base() -> refc.RefCConfig:
    """The SHARED core both arms train — every choice cites REFC_V3_DESIGN §2.

    small-class trunk (fan quality measured equal at matched K), base-class
    decoder geometry (base≈XL tie; decoder is ~1.7 ms of the tick), 128 anchors
    (the measured fan lever), 6 s horizons (binding), factored tactical head
    (collapse defect + Alpamayo 40.62 % dual-axis), reach clamp (measured inert
    on ADE, precondition for per-candidate work).
    """
    cfg = refc.RefCConfig()
    cfg.encoder = refc.CNNEncoderConfig(in_channels=9, image_size=256,
                                        base_width=64, blocks=(3, 6, 16, 6))
    cfg.decoder = refc.DecoderConfig(d=384, n_heads=8, layers=4, ff_mult=4,
                                     aux_hidden=384, diffusion_steps=2,
                                     noise_std=0.1)
    cfg.anchors = refc.AnchorConfig(n_anchors=128, pool_size=4096)
    cfg.trajectory = refc.TrajectoryConfig(horizons=V3_HORIZONS)
    cfg.strategic = refc.StrategicCtxConfig(hidden=512, d_ctx=64)
    cfg.factored_maneuver = True      # action space: both arms, never a lever
    cfg.tactical_speed_input = False  # goal path stays vision-pure (E11)
    cfg.sel_reach_clamp = True        # precondition, measured inert on ADE @2s
    return cfg


#: ⭐ THE TWO SIZE RUNGS (PI 2026-08-21: "let have two versions of refcv3 small
#: and xl"). ONLY ``encoder.base_width``/``blocks`` differ — every other choice
#: in :func:`_v3_core_base` is shared, so the pair is a clean SCALE axis and not
#: a bundle of levers.
#:
#: ⛔ WHY XL EXISTS. ``D-008`` (2026-07-05, accepted by Sayed) sets model scale
#: **>= 250 M**, and it ties that to *"a scale where hierarchy is expressible"* —
#: which is precisely what v3 exists to test. The small rung MEASURES
#: 62,930,419, i.e. **4x under the decision**. The design justified small on
#: FAN-QUALITY grounds (*"small's fan is at least as tight as base's at every
#: matched K"*, *"base≈XL tie"*), and ⚠️ a tie on one metric does not retire a
#: SCALE decision.
#:
#: MEASURED 2026-08-21 by rebuilding at each rung:
#:     small  bw=64   core  60,882,074   hierarchy 2,048,345   TOTAL  62,930,419
#:     base   bw=88   core 104,707,298   hierarchy 2,097,497   TOTAL 106,804,795
#:     XL     bw=124  core 215,589,550   hierarchy 2,171,225   TOTAL 217,760,775
#: ⭐ The hierarchy cost is essentially CONSTANT across the ladder, so SCALE and
#: HIERARCHY are independent decisions — which is why they are separate configs.
V3_SIZES: dict[str, tuple[int, tuple[int, ...]]] = {
    "small": (64, (3, 6, 16, 6)),
    "base": (88, (3, 6, 16, 6)),
    "xl": (124, (3, 8, 20, 6)),
}


def refc_v3_sized_config(size: str = "small", *, hier: bool = True
                         ) -> RefCV3Config:
    """v3 at one rung of the REF-C size ladder.

    ⚠️ ``size`` moves the ENCODER ONLY. The decoder, anchors, horizons, tactical
    factoring and reach clamp are ``_v3_core_base``'s and do not vary — so a
    small-vs-XL comparison attributes to scale, not to a config bundle. That is
    the C122 lesson applied to the size axis.
    """
    if size not in V3_SIZES:
        raise ValueError(f"size must be one of {sorted(V3_SIZES)}, got {size!r}")
    bw, blocks = V3_SIZES[size]
    cfg = refc_v3_hier_config() if hier else refc_v3_flat_config()
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=cfg.core.encoder.in_channels,
        image_size=cfg.core.encoder.image_size,
        base_width=bw, blocks=blocks)
    return cfg


def refc_v3_small_config(hier: bool = True) -> RefCV3Config:
    """v3-small — the AS-REGISTERED rung (`PREREG_REFC_V3.md`). 62,930,419."""
    return refc_v3_sized_config("small", hier=hier)


def refc_v3_xl_config(hier: bool = True) -> RefCV3Config:
    """⭐ v3-XL — the ``D-008`` rung. 217,760,775 with the current hierarchy.

    ⚠️ Adopting this VOIDS the registered cost line in ``PREREG_REFC_V3.md``
    (~7-9 h A40/run at small). Amending a pre-registration BEFORE any read is
    legitimate; after a read it is not.
    """
    return refc_v3_sized_config("xl", hier=hier)


def refc_v3_flat_config() -> RefCV3Config:
    """v3-F — the flat arm. Incumbent REF-C seams only (ctx token + own-head
    H19), at the shared v3 sizing/horizon/action space."""
    return RefCV3Config(core=_v3_core_base(), hier=False)


def refc_v3_hier_config() -> RefCV3Config:
    """v3-H — the goal-cascade arm. Delta vs flat = {hier,
    core.graft_target_latent}, pinned by test_dominance_delta_is_pinned."""
    cfg = RefCV3Config(core=_v3_core_base(), hier=True)
    cfg.core.graft_target_latent = True
    return cfg


def refc_v3_smoke_config(hier: bool = True) -> RefCV3Config:
    """Tiny CPU pair for tests — same structure, same 8-slot horizon layout."""
    core = refc.refc_smoke_config()
    core.trajectory = refc.TrajectoryConfig(horizons=V3_HORIZONS)
    core.factored_maneuver = True
    core.sel_reach_clamp = True
    cfg = RefCV3Config(core=core, hier=hier, d_tac=32, tac_hidden=16,
                       d_gcond=8)
    if hier:
        cfg.core.graft_target_latent = True
    return cfg


# ============================================================================
# The model
# ============================================================================

class RefCV3Model(nn.Module):
    """Core RefCModel + (gated) goal cascade. With ``hier=False`` this class is
    a TRANSPARENT wrapper: forward defers to the core untouched, and the
    state_dict is the core's under the ``core.`` prefix — pinned by tests."""

    def __init__(self, cfg: RefCV3Config):
        super().__init__()
        self.cfg = cfg
        self.core = refc.RefCModel(cfg.core)
        if not cfg.hier:
            return
        if not cfg.core.hierarchy:
            raise ValueError("v3 hier needs core.hierarchy=True (pooled_seq)")
        if not cfg.core.graft_target_latent:
            raise ValueError(
                "v3 hier needs core.graft_target_latent=True — the decoder "
                "skips a None film (refc.py tgt_film), so feeding the port of "
                "a build that never constructed it would be a silently-ignored "
                "external prior, the exact bug class the seam refuses.")
        feat = self.core.encoder.feat_dim
        d_ctx = cfg.core.strategic.d_ctx
        k = cfg.n_goal_taus
        # E5 — the tactical state. THE existing, tested implementation
        # (tactical.py:99), first wiring (C116). Window = the core's window.
        self.phi_tac = PhiTac(d_op=feat, d_tac=cfg.d_tac,
                              window=cfg.core.window, hidden=cfg.tac_hidden)
        # E3 — strategic goal head (predicted geometric route goal off z_str).
        self.str_goal_head = nn.Linear(d_ctx, 3)
        # E4 — g_str conditions the tactical state. ZERO-INIT FiLM: at init the
        # cascade is exactly PhiTac (bit-inert conditioning), so the edge's
        # effect is attributable from step 0 (the H19/zero-init discipline).
        self.gstr_embed = nn.Linear(3, cfg.d_gcond)
        self.gstr_film = nn.Linear(cfg.d_gcond, 2 * cfg.d_tac)
        nn.init.zeros_(self.gstr_film.weight)
        nn.init.zeros_(self.gstr_film.bias)
        # E6 — factored tactical decision heads on z_tac (the H arm's decision
        # supplier; the core's own pooled-based heads keep training as the
        # shared aux surface in BOTH arms, so the supervision surface is
        # identical and only the DECISION SOURCE differs).
        self.lat_head_tac = nn.Linear(cfg.d_tac, tac.N_LAT)
        self.lon_head_tac = nn.Linear(cfg.d_tac, tac.N_LON)
        # E8 — tactical geometric goals, E4.1 layout (x, y, heading, speed)@tau.
        self.tac_goal_head = nn.Linear(cfg.d_tac, k * GOAL_DIMS)
        # E7 — tactical latent into the decoder's target-latent FiLM port.
        self.tac_latent_proj = nn.Linear(cfg.d_tac, cfg.core.tactical_latent_dim)
        # E9 — selection by distance to the predicted goal. THE v6 scorer,
        # imported (one implementation; its docstring carries the winner's-curse
        # measurements and the admission curve). Deferred import: v6 is a heavy
        # module and refs/ must not pay it unless the cascade is built.
        from tanitad.models.v6 import GoalDistanceScorer
        self.scorer = GoalDistanceScorer(d_goal_embed=cfg.d_tac,
                                         n_candidates=cfg.core.anchors.n_anchors,
                                         tau_m=cfg.scorer_tau_m)
        self.goal_gate = nn.Parameter(torch.zeros(()))   # zero-init: bit-inert
        self._seam = sl.SeamState()                      # not a buffer (no ckpt key)

    # --- provenance (the PI's admissibility ruling, as data + roles) --------
    @staticmethod
    def provenance_roles() -> dict:
        """Role map for ``tanitad.eval.goal_provenance.audit_arm``. GOAL nodes:
        g_str, g_tac, goal_point. SITUATION_OUTPUT nodes: NONE IN GRAPH — and
        the audit MEASURES that (positive control on frames, pinned negative
        edge on v0) rather than trusting this declaration."""
        return {
            "goal": ["g_str", "g_tac", "goal_point_tac"],
            "situation_output": [],
            "inference_inputs_of_goals": ["frames (via pooled_seq/ctx only)"],
            "refused_edges": ["v0 -> any goal node (E11)",
                              "lan -> inference (E12; label-only)"],
            "shared_trunk": "encoder (common ancestor, declared; zero-init "
                            "gates carry attributability)",
        }

    # --- the in-forward hierarchy supplier ----------------------------------
    def _hook(self, cache: dict):
        cfg = self.cfg

        def hook(pooled_seq: Tensor, ctx: Tensor) -> dict:
            b = pooled_seq.shape[0]
            z_tac_raw = self.phi_tac(pooled_seq)                  # [B, d_tac]
            g = self.str_goal_head(ctx)                           # [B, 3]
            bearing = g[:, :2] / torch.linalg.vector_norm(
                g[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
            g_str = torch.cat([bearing, torch.tanh(g[:, 2:3])], dim=-1)
            # E4: strategic goal conditions tactical — DETACHED downward.
            gcond = self.gstr_embed(g_str.detach())
            gamma, beta = self.gstr_film(gcond).chunk(2, dim=-1)
            z_tac = z_tac_raw * (1.0 + gamma) + beta              # zero-init
            lat = self.lat_head_tac(z_tac)
            lon = self.lon_head_tac(z_tac)
            man5 = tac.derive_man5_logprobs(lat, lon)             # exact push-fwd
            g_tac = self.tac_goal_head(z_tac).reshape(
                b, cfg.n_goal_taus, GOAL_DIMS)
            cache.update(z_tac=z_tac, g_str=g_str, g_str_raw=g,
                         lat_logits_tac=lat, lon_logits_tac=lon,
                         g_tac=g_tac)
            # E6 live (the H19 seam is live-from-step-0 by design);
            # E7 detached (the decoder cannot train the tactical pool).
            return {"maneuver_logits": man5,
                    "target_latent": self.tac_latent_proj(z_tac.detach())}

        return hook

    def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,
                v0: Tensor | None = None, steps: int = 0,
                lan: Tensor | None = None,
                nav_known: Tensor | None = None) -> dict:
        if not self.cfg.hier:
            return self.core(frames, nav_cmd, v0, steps=steps, lan=lan,
                             nav_known=nav_known)
        cache: dict = {}
        out = self.core(frames, nav_cmd, v0, steps=steps, lan=lan,
                        nav_known=nav_known, hierarchy_hook=self._hook(cache))
        # ---- E9: goal selection over the emitted fan (post-decoder) --------
        fan = out["anchor_traj"]                                  # [B, N, S, 2]
        b = fan.shape[0]
        # 2 s slot: the slot the admission curve was measured at. The goal is
        # DETACHED into selection (the winner's-curse firewall: selection can
        # never train the goal toward the fan).
        g2 = cache["g_tac"][:, self._tau_slot_2s(), :2].detach()  # [B, 2]
        sc = self.scorer(fan[:, :, SEAM_SLOT:SEAM_SLOT + 1],
                         cache["z_tac"].detach(), goal_point=g2)
        graft = self.goal_gate * sc["score"]                      # [B, N]
        blended, tele = sl.apply_seam_clamp(
            out["sel_score"], graft, clamp=self.cfg.seam_clamp,
            fail=self.cfg.seam_fail, fail_frac=self.cfg.seam_fail_frac,
            patience=self.cfg.seam_fail_patience, state=self._seam,
            surface="goal_sel")
        rank = blended
        if "reach_keep" in out:            # post-guard mask (dead rows full)
            rank = blended.masked_fill(~out["reach_keep"], float("-inf"))
        idx = rank.argmax(dim=1)
        traj = fan[torch.arange(b, device=fan.device), idx]
        out.update(cache)
        out.update(tele)
        out["traj_base"], out["sel_idx_base"] = out["traj"], out["sel_idx"]
        out["traj"], out["sel_idx"] = traj, idx
        out["wp_seq"] = traj
        out["sel_score_v3"] = blended
        out["goal_point_tac"] = g2
        out["goal_dist"] = sc["goal_dist"]
        if "goal_point_free" in sc:        # E-AG2-style free-decode control
            out["goal_point_free"] = sc["goal_point_free"]
        return out

    def _tau_slot_2s(self) -> int:
        """Index of the 2 s tau in goal_tau_steps (fails loudly if absent —
        the seam slot is load-bearing for the measured admission)."""
        try:
            return self.cfg.goal_tau_steps.index(20)
        except ValueError as e:
            raise ValueError("goal_tau_steps must contain 20 (= 2.0 s): the "
                             "selection admission curve is measured at the "
                             "2 s endpoint") from e


# ============================================================================
# The pinning instruments (C122: derive the delta; C115: prove sensitivity)
# ============================================================================

def _leaf_diffs(a, b, prefix: str = "") -> dict:
    out = {}
    for f in fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        name = f"{prefix}{f.name}"
        if is_dataclass(va) and is_dataclass(vb) and type(va) is type(vb):
            out.update(_leaf_diffs(va, vb, prefix=name + "."))
        elif va != vb:
            out[name] = (va, vb)
    return out


def config_delta(a: RefCV3Config, b: RefCV3Config) -> dict:
    """EVERY differing leaf field between two v3 configs, derived by walking
    the dataclasses — never asserted in prose. The dominance experiment's
    'everything else identical' is ``test_dominance_delta_is_pinned`` asserting
    this returns EXACTLY the registered lever set (C122's rule)."""
    if type(a) is not type(b):
        raise TypeError(f"config_delta compares like with like, got "
                        f"{type(a).__name__} vs {type(b).__name__}")
    return _leaf_diffs(a, b)


def param_breakdown_v3(model: RefCV3Model) -> dict[str, int]:
    """MEASURED per-module parameter costs (the capacity ledger the prereg
    pins). ``core`` is the unmodified RefCModel total; hierarchy lines are the
    cascade's own modules; sums to ``total`` exactly."""
    cnt = lambda m: sum(p.numel() for p in m.parameters())      # noqa: E731
    out = {"core": cnt(model.core)}
    if model.cfg.hier:
        out.update({
            "phi_tac": cnt(model.phi_tac),
            "str_goal_head": cnt(model.str_goal_head),
            "gstr_cond": cnt(model.gstr_embed) + cnt(model.gstr_film),
            "tac_heads": cnt(model.lat_head_tac) + cnt(model.lon_head_tac)
            + cnt(model.tac_goal_head),
            "tac_latent_proj": cnt(model.tac_latent_proj),
            "scorer": cnt(model.scorer) + model.goal_gate.numel(),
        })
    out["total"] = cnt(model)
    return out


# ============================================================================
# Loss helpers (masked — the parity-preserving 6 s design needs exact-zero
# gradient at invalid slots; pinned by tests)
# ============================================================================

def masked_goal_loss(g_pred: Tensor, g_tgt: Tensor, valid: Tensor) -> Tensor:
    """Smooth-L1 over (x, y, speed) + wrapped-angle L1 over heading, masked.

    ``g_pred``/``g_tgt`` [B, K, 4] in the E4.1 layout; ``valid`` [B, K] bool.
    Invalid rows contribute EXACTLY zero (mask multiplies the summand, so the
    gradient at a masked row is structurally zero, not merely small)."""
    if g_pred.shape != g_tgt.shape or g_pred.shape[:2] != valid.shape:
        raise ValueError(f"shape mismatch: {tuple(g_pred.shape)} vs "
                         f"{tuple(g_tgt.shape)} vs {tuple(valid.shape)}")
    m = valid.to(g_pred.dtype)
    xy_sp = F.smooth_l1_loss(g_pred[..., [0, 1, 3]], g_tgt[..., [0, 1, 3]],
                             reduction="none").sum(-1)
    hd = tac.wrap_to_pi(g_pred[..., 2] - g_tgt[..., 2]).abs()
    return ((xy_sp + hd) * m).sum() / m.sum().clamp_min(1.0)


def strategic_goal_loss(g_str: Tensor, bearing_tgt: Tensor, dist_tgt: Tensor,
                        valid: Tensor) -> Tensor:
    """Cosine bearing loss + L1 dist_pref, masked by route validity.

    ``g_str`` [B, 3] (unit bearing + tanh dist_pref, the model's own output
    layout); targets from ``refc.RefCModel.goal_targets`` (the leak-guarded LAN
    label — TRAIN ONLY, E12). The dist_pref half carries the DECLARED speed
    confound (REFC_V3_DESIGN §4.4) — separate terms, so it can be ablated."""
    m = valid.to(g_str.dtype)
    cos = (g_str[:, :2] * bearing_tgt).sum(-1)                   # [-1, 1]
    l_bear = ((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0)
    l_dist = ((g_str[:, 2] - dist_tgt).abs() * m).sum() / m.sum().clamp_min(1.0)
    return l_bear + l_dist


def selection_ce(blended: Tensor, fan_err: Tensor,
                 reach_keep: Tensor | None = None) -> Tensor:
    """CE for the blended selection score, normalised over EXACTLY the survivor
    set the argmax ranks over, target = best candidate IN that set (the S1c
    lesson: a full-fan softmax on a ~27 % problem is dominated by candidates no
    selector ever picks). ``fan_err`` [B, N] per-candidate error (masked slots
    already excluded upstream)."""
    if reach_keep is not None:
        blended = blended.masked_fill(~reach_keep, float("-inf"))
        fan_err = fan_err.masked_fill(~reach_keep, float("inf"))
    tgt = fan_err.argmin(dim=1)
    return F.cross_entropy(blended, tgt)


# ============================================================================
# The C115 gate — hierarchy is PROVEN, never asserted
# ============================================================================

@torch.no_grad()
def _freeze_history(frames: Tensor) -> Tensor:
    """[B, W, ...] -> the same window with every non-last frame replaced by the
    last (C115's probe: a window with ZERO temporal information)."""
    fz = frames.clone()
    fz[:, :-1] = frames[:, -1:].expand_as(frames[:, :-1])
    return fz


def freeze_history_report(model: RefCV3Model, frames: Tensor,
                          v0: Tensor | None = None) -> dict:
    """The pre-registered sensitivity gate (PREREG §6.4). Two mechanisms:

    1. INTERVENTION: freeze-history vs true window — z_tac / g_tac / g_str must
       MOVE (relative L2). ``pooled`` is the built-in NEGATIVE control: it is a
       last-frame function BY CONSTRUCTION, so it must be BIT-IDENTICAL — a
       probe that reports pooled moving is broken, not a finding.
    2. GRADIENT: d(random-projection of z_tac)/d(frame_t) per history slot —
       through a RANDOM PROJECTION, never ``.sum()`` (C116: LayerNorm makes the
       sum's gradient identically zero and the probe confirms C115 harder than
       the truth).

    Returns the numbers + verdicts; the caller (test / preflight) asserts.
    A FAIL here voids the dominance experiment (OUTCOME V), it does not score
    an arm.
    """
    if not model.cfg.hier:
        raise ValueError("freeze_history_report probes the H arm's cascade")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            o_true = model(frames, v0=v0)
            o_frozen = model(_freeze_history(frames), v0=v0)

        def rel(a: Tensor, b: Tensor) -> float:
            return float((a - b).norm() / a.norm().clamp_min(1e-9))

        report = {
            "z_tac_rel_move": rel(o_true["z_tac"], o_frozen["z_tac"]),
            "g_tac_rel_move": rel(o_true["g_tac"], o_frozen["g_tac"]),
            "g_str_rel_move": rel(o_true["g_str"], o_frozen["g_str"]),
            "pooled_rel_move": rel(o_true["pooled"], o_frozen["pooled"]),
        }
        # gradient mechanism (random projection — C116 instrument hazard)
        f = frames.clone().requires_grad_(True)
        cache: dict = {}
        _ = model.core(f, v0=v0, hierarchy_hook=model._hook(cache))
        z = cache["z_tac"]
        g = torch.Generator(device="cpu").manual_seed(0)
        proj = torch.randn(z.shape[-1], generator=g).to(z.device, z.dtype)
        (z @ proj).sum().backward()
        with torch.no_grad():
            per_frame = f.grad.reshape(f.shape[0], f.shape[1], -1).norm(dim=-1)
            gm = per_frame.mean(dim=0)                        # [W]
        report["grad_per_frame"] = [round(float(x), 8) for x in gm]
        report["history_grad_nonzero"] = bool((gm[:-1] > 0).all())
        report["pass"] = (report["z_tac_rel_move"] > 1e-6
                          and report["pooled_rel_move"] == 0.0
                          and report["history_grad_nonzero"])
        return report
    finally:
        if was_training:
            model.train()
