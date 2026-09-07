"""REF-C model: Anchored-Diffusion-C — a DiffusionDrive-style trajectory head.

REF-C was TCP-C (a two-branch GRU trajectory/control stack, arXiv 2206.08129).
This revision REPLACES the GRU trajectory + control branches with an ANCHORED
TRUNCATED-DIFFUSION trajectory decoder in the DiffusionDrive spirit (arXiv
2411.15139): a fixed vocabulary of trajectory ANCHORS whose queries cross-attend
the conv feature map, emitting a per-anchor confidence + a per-anchor offset, and
(optionally) refining the winning modes with a few truncated denoising steps. The
rest of the TCP-C stack is KEPT verbatim: the torchvision-free ResNet-34-style
encoder, the measurement encoder (TCP's FC-128 x 2, bit-for-bit), the LAW
latent-world-model aux, the strategic-ctx hierarchy graft, and the REF-C.1
target-speed class head.

⚠⚠ CORRECTION 2026-09-04 -- THIS DOCSTRING USED TO SAY "the measurement encoder
with per-sample ego-dropout" INSIDE THE "kept verbatim" LIST, AND THAT LAUNDERED
AN UNSWEPT HYPER-PARAMETER AS INHERITED, VALIDATED PRACTICE. The ENCODER is
TCP's. The **ego-dropout is not**: the string "dropout" occurs ZERO times in
arXiv 2206.08129 (two probes -- a full-text search for dropout/drop out/mask,
and TCP's own layer table, which lists the measurement encoder with no
regulariser). ``ego_dropout = 0.5`` is a TanitAD invention that nobody, upstream
or here, has ever swept; the two published planners that DID sweep it landed on
**0.5** (DRAMA, NAVSIM PDMS 0.835 -> 0.848) and **0.75** (PlanTF, nuPlan
closed-loop, at a measured cost of -1.48 OLS). Same root-cause class as the
registry's un-refined-anchor correction: a true statement about one component
extended over a neighbour it never covered.
⛔ AND IT IS NOT A FREE KNOB: with ``ego_valid_channel = False`` a withheld
speed is byte-identical to a genuine standstill -- MEASURED 22.5 : 1 withheld
vs genuine at train against 100 % genuine at eval (X15; see
``ego_valid_channel`` below and REF-C v4's PREREG section 4.2).

Why anchors + FPS (not k-means): comma2k19 is ~74 % straight, so k-means
collapses almost every centroid onto the straight mode and starves the turns.
Furthest-point sampling (FPS) deliberately SPREADS the vocabulary to cover the
rare sharp-curve / hard-brake trajectories — the modes that actually matter.

Decoder (``AnchoredDiffusionDecoder``), two inference modes off ONE weight set:
  - ``steps=0`` (classifier / 0-step floor): anchor queries cross-attend the
    8x8xF conv map (F = base_width * 8: 512 small / 704 base / 992 XL) through
    ``cfg.layers`` MHA layers at width ``cfg.d`` (small d=256 x 3L, base d=384 x
    4L, XL d=512 x 6L), FiLM(condition); emit per-anchor confidence + per-anchor
    [n_horizons, 2] offset; traj = selected anchor + offset. This is what the
    trainer optimises.
  - ``steps>0`` (truncated diffusion): the SAME offset head refines the anchor
    trajectories over a few timestep-embedded denoising passes around the
    anchors; ``steps=0`` reproduces the classifier byte-for-byte.

Graft seams (gated, zero-init / identity starts — byte-identical when off):
  - ``hierarchy`` (default True): a strategic-ctx GRU over the W window frames
    -> a d_ctx token added (zero-init) to the decoder CONDITION embedding.
  - ``graft_maneuver`` (default True, H19): the model's maneuver-head logits (or
    an external tactical brain's) reweight the anchor confidence PRIORS through a
    learned maneuver->anchor projection (LIVE from step 0 — the H19 coupling is
    the point; the zero-init discipline applies to the ctx / target-latent
    seams).
  - ``factored_maneuver`` (default False, D-TAC1): REPLACES the single 5-way
    tactical softmax with INDEPENDENT lateral (3) and longitudinal (3) heads,
    and splits the H19 graft into ``lat_to_anchor`` (default init, inherits
    today's role) + ``lon_to_anchor`` (ZERO-init, so the longitudinal prior's
    effect on selection is attributable from step 0). ``maneuver_logits`` [B, 5]
    is still emitted — derived exactly from the two heads through the priority
    collapse (``refc_tactical.derive_man5_logprobs``) — so every downstream
    reader keeps working. Companion: ``man_prior_tau`` (prior-corrected decode).
    MEASURED motivation and the algebra: see ``tanitad/refs/refc_tactical.py``.
  - ``tactical_speed_input`` (default False, D-TAC1 F1): the tactical head reads
    the ego speed alongside the image embedding. **INDEPENDENT of
    ``factored_maneuver``** — it applies to the SHIPPED 5-way head as well, so
    INPUT can be ablated without STRUCTURE. (It was coupled to
    ``factored_maneuver`` until 2026-08-03; the coupling made the pre-registered
    arms confound F1 with F2, since `dtac1-full` = F1+F2 and `dtac1-f2only` = F2
    left no arm isolating F1. See ``refc_f1only_config``.)
  - ``graft_target_latent`` (default False): a tactical GOAL latent [B, S] FiLMs
    (zero-init -> identity) the decoder condition. Off by default because it has
    no standalone supervision (it only activates when a real tactical brain
    feeds a ``target_latent``).
  - ``grounded_selector`` (default False, param-free): score decoded ego-frame
    endpoints by a progress/collision proxy and blend with the top-1 confidence.
  - ``graft_lan`` (default False, LAN): a LANE-ANCHORED ROUTE corridor — K
    arc-length route anchors x [cos bearing, sin bearing, lat_norm, valid] from
    ``tanitad.data.lan`` — enters on two surfaces: a zero-init projection into
    the decoder CONDITION, and ONE learned scalar gate on a param-free
    geometric compatibility between each anchor's terminal bearing and the
    route bearing (the SELECTION surface). Both start at exactly 0, so the
    model is unchanged at step 0. Exists because the 4-way ``nav_cmd`` is
    ``follow`` on ~75-79 % of windows and is a CONSTANT at eval (``nav_cmd=
    None``) — the C6 confound. Additive: nav_cmd is kept.
  - **D-SEL — the SELECTION surface** (all default False / 0, so an all-off
    build is byte-identical to pre-D-SEL REF-C). REF-C's separation from the
    flagship is entirely LATERAL and every one of its measured defects is about
    WHICH candidate is emitted, not which are proposed: the refined fan is
    ranked by the UNREFINED score (``sel_refined``), 72.08 % of that fan is not
    physically flyable (``sel_reach_clamp``), the consequences of the candidates
    never reach the ranking (``graft_cons``), the grafts that do reach it are
    uncapped (``seam_clamp``), and the strategic route can only warp the
    condition, never choose (``graft_route``). Rationale, the measurements, and
    the argued list of flagship levers that do NOT transfer: ``refc_select.py``.
    Preset: :func:`refc_select_config` (+385 parameters, MEASURED).
  - ``graft_goal`` (default False, S6): a **PREDICTED GEOMETRIC** goal — bearing
    + signed along-track preference, from the image embedding ALONE — reaching
    the ranked score through the same param-free geometric compatibility LAN
    uses, on TWO independent zero-init gates. Admissible under the PI ruling of
    2026-08-03; what it is computed from is declared by
    :meth:`RefCModel.goal_provenance` and written into every ``config.json``.
    Preset: :func:`refc_goal_config`.
  - ``ego_valid_channel`` (default False, X15): an explicit "v0 is present" flag
    beside the ego-dropped speed, for the measurement encoder and the tactical
    head. 0.0 m/s is in-distribution "stationary", so zero-filling a withheld
    speed is a confident lie the reader cannot detect.
  - ``graft_imagination`` (default False, H15): a belief field over the conv-map
    tokens — latent-advection prior (object permanence) + transformer refinement
    + per-cell epistemic log-variance gating a residual belief written back into
    the tokens the anchor decoder cross-attends. ON for REF-C-XL (the flagship-
    matched capacity control); the whole field sits in the trajectory-loss
    gradient path (no dead params). Absent from the state_dict when off.
  - LAW ``law_head`` is KEPT: gradients flow through the decoded trajectory.

Scale presets (SAME code + decoder algorithm, three sizes — the size cap was
lifted so the budget lands where the encoder has PROVEN value: Hydra-MDP went
86.6 -> 91.0 PDMS purely by swapping ResNet-34 -> V2-99, so the ENCODER is the
lever, and the deeper/wider trunk is data-appropriate for the full 2376-ep set):
  - ``refc_config`` -> REF-C-base 104,191,577 (primary): a widened/deepened
    ResNet trunk (90,458,632 encoder, 8x8xF map preserved) + a d=384 / 4-layer /
    128-anchor decoder. The data-appropriate reference.
  - ``refc_xl_config`` -> REF-C-XL 251,932,584: a much wider/deeper ResNet trunk
    (199,496,532 encoder, 8x8xF map preserved) + a d=512 / 6-layer / 256-anchor
    decoder + the gated H15 imagination field (20,986,339). Same-capacity control
    vs the 263 M flagship (removes the "REF-C is worse because smaller"
    confound). All counts MEASURED (MODEL_REGISTRY.md section 4), not estimated.
  - ``refc_smoke_config`` -> tiny CPU config (CI / tests / dry runs).

REF-C.1 (gated ``refc1``, default False): the trajectory targets become fixed-
DISTANCE path checkpoints at (2, 5, 10, 20) m (refb_labels.path_targets) and a
target-speed classification head is added (CE + expected-value decode).

Gated-flag discipline (REF-B convention): with a flag off the corresponding
module is NOT constructed — the model is byte-identical to one that never had the
feature (state_dict keys pinned by tests/test_refc.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models.kinematic import rollout_unicycle
from tanitad.refs import feasible_decode as _feas
# ⭐ S7 / E15 — the param-free metric-goal-point compatibility. Imported, NEVER
# re-derived beside the thing it scores: a guard/score re-implemented next to
# its own consumer is the exact defect the navpred RESULT retracted two
# published numbers for. `goal_point` imports `tanitad.data.lan` and (lazily,
# inside one function) `refc_v3` — so there is no module-level cycle with this
# file, which `refc_v3` imports.
from tanitad.refs import goal_point as gpm
# refcv5 WP-4. Cheap and cycle-free: `refc_sampler` imports only
# `tanitad.models.kinematic`, which this module already imports. WP-6's
# `refc_agents` is imported LAZILY inside `RefCModel.__init__` instead --
# it pulls the corpus-side raster/projection modules, and an OFF seam must
# not pay for them.
from tanitad.refs import refc_sampler as rs


def _feas_with_origin(x: Tensor) -> Tensor:
    """[..., S, 2] -> [..., S+1, 2] with the ego origin prepended. Same shape
    contract as ``fan_safety.with_origin``; duplicated here only because
    ``taniteval`` is a SIBLING of ``stack/`` and a hard import would make this
    module un-importable on a pod."""
    z = torch.zeros(*x.shape[:-2], 1, 2, dtype=x.dtype, device=x.device)
    return torch.cat([z, x], dim=-2)
from tanitad.refs import refc_select as sl
from tanitad.refs import refc_tactical as tac

# Strategic vocabulary — order pinned against scripts/refb_labels.py indices
# by tests/test_refc.py (same 4-wide interface as tanitad.refs.refb).
NAV_COMMANDS = ("follow", "left", "right", "straight")

# Maneuver vocabulary width — order matches refb_labels (lane_keep, turn_left,
# turn_right, accelerate, brake_stop). The model emits [B, N_MANEUVERS] logits;
# the trainer supplies the kinematic pseudo-label target (refc.py must not import
# from scripts/, so the count is pinned here and cross-checked by the tests).
N_MANEUVERS = 5
# Factorised tactical widths (D-TAC1). Sourced from the vocabulary module, never
# from a literal, so the head can never be sized apart from the collapse table
# it is projected through.
N_LAT_MAN = tac.N_LAT              # lane_keep / turn_left / turn_right
N_LON_MAN = tac.N_LON              # brake_stop / steady / accelerate
N_ROUTE = 3                        # route-heading aux (left / straight / right)
# LAN per-anchor feature width. Pinned here (refc.py stays import-light, the
# same rule as N_MANEUVERS) and cross-checked against tanitad.data.lan by
# tests/test_lan.py — a silent divergence would mis-slice the route tensor.
LAN_FEATS_PER_ANCHOR = 4


# ============================================================================
# Anchor vocabulary — FPS over ego-frame future trajectories
# ============================================================================

def synth_anchor_pool(horizons: tuple[int, ...], pool_size: int = 4096,
                      seed: int = 0, dt: float = 0.1,
                      device: str = "cpu") -> Tensor:
    """Synthesize a pool of ego-frame trajectories via random unicycle rollouts.

    Each rollout samples (v0, yaw_rate, accel) uniformly, integrates a unicycle
    from the ego origin (heading +x), and reads the position at every horizon
    step. Because the ego starts at the origin heading +x, the world positions
    ARE the ego-frame waypoints. Returns [pool_size, len(horizons), 2].

    Used for the model's DEFAULT anchor set (so REF-C builds without a data file)
    and by build_refc_anchors.py for the CPU-smoke / no-data path. Explicit
    ``device`` keeps this real even under a ``torch.device("meta")`` build.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    m = pool_size
    v = (torch.rand(m, generator=g, device=device) * 30.0)          # 0..30 m/s
    yaw_rate = (torch.rand(m, generator=g, device=device) - 0.5) * 0.7   # +-0.35
    accel = (torch.rand(m, generator=g, device=device) - 0.5) * 6.0     # +-3
    x = torch.zeros(m, device=device)
    y = torch.zeros(m, device=device)
    yaw = torch.zeros(m, device=device)
    max_h = max(horizons)
    pos = torch.zeros(m, max_h + 1, 2, device=device)
    for t in range(1, max_h + 1):
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + yaw_rate * dt
        v = (v + accel * dt).clamp_min(0.0)
        pos[:, t, 0] = x
        pos[:, t, 1] = y
    return torch.stack([pos[:, h] for h in horizons], dim=1)         # [m, S, 2]


def furthest_point_sample(pool: Tensor, n: int, seed: int = 0) -> Tensor:
    """Furthest-point sample ``n`` anchors from ``pool`` [M, S, 2] -> [n, S, 2].

    Greedy FPS in flattened-L2 space: seed one point, then repeatedly add the
    pool point maximally far from the current set (min-distance criterion). This
    SPREADS the vocabulary over the trajectory manifold (covering the rare
    curves) rather than concentrating on the dense straight majority the way
    k-means would. Deterministic given ``seed``.
    """
    m = pool.shape[0]
    if n > m:
        raise ValueError(f"cannot FPS {n} anchors from a pool of {m}")
    flat = pool.reshape(m, -1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(m, (1,), generator=g, device=flat.device))
    chosen = [first]
    dist = ((flat - flat[first]) ** 2).sum(dim=-1)                   # [M]
    for _ in range(n - 1):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((flat - flat[nxt]) ** 2).sum(dim=-1))
    return pool[torch.tensor(chosen, device=pool.device)]


#: ⭐ H-EGO-LIT-4 (2026-09-05): the admissible SOURCES of the speed a WITHHELD
#: row's anchor bank is rolled at. `AnchorDecoder.anchor_withheld_bank` holds
#: one of these; the trainer's `--withheld-bank` flag sets it and STAMPS it.
#:   fixed  — `anchor_ref_speed` (the shipped 10 m/s roll; the pre-flag
#:            arithmetic, bit-identical)
#:   pred   — the model's OWN detached, vision-only `g_tac` 2 s speed
#:            (SparseDrive's pattern), clamped to [0, anchor_withheld_speed_max]
#:   random — a draw from `anchor_random_speed_pool` (the training marginal of
#:            v0): the right distribution and ZERO information — the blindness
#:            CONTROL that must NOT gain if `pred` gains for the right reason
#:   none   — EVERY row (kept, withheld, eval) at `anchor_ref_speed`: the
#:            field's speed-blind fixed vocabulary
#: ⛔ The measured v0 of a withheld row reaches NONE of them: a speed bucketed on
#: it would put log2(k) bits of the withheld channel into the candidate
#: geometry (D-EGOLIT-WITHHELD1, refused as designed).
WITHHELD_BANK_MODES: tuple[str, ...] = ("fixed", "pred", "random", "none")


def default_anchors(horizons: tuple[int, ...], n_anchors: int,
                    pool_size: int = 4096, seed: int = 0,
                    device: str = "cpu") -> Tensor:
    """The model's built-in anchor vocabulary: FPS over a synthetic pool.

    Deterministic (fixed seed) so two independently-built models share anchors
    byte-for-byte. Overridden at train time by build_refc_anchors.py output via
    :meth:`AnchoredDiffusionDecoder.load_anchors`.
    """
    pool = synth_anchor_pool(horizons, pool_size, seed, device=device)
    return furthest_point_sample(pool, n_anchors, seed=seed).contiguous()


# ============================================================================
# Configs
# ============================================================================

@dataclass
class CNNEncoderConfig:
    """ResNet-34-style trunk (torchvision-free): stem /4, four stages /2 each
    -> stride 32; feat_dim = base_width * 8; grid = image_size // 32.

    The size cap was lifted to spend the budget on the encoder (the Hydra-MDP
    ResNet-34 -> V2-99 lever). REF-C-base WIDENS base_width to 88 (V2-99-class
    trunk, measured 90,458,632) and REF-C-XL to 124 with blocks (3, 8, 20, 6)
    (measured 199,496,532 — wider AND deeper); both KEEP the 8x8xF conv map the
    anchor decoder cross-attends (grid = 8 at 256 px; F = base_width * 8, and the
    decoder's feat_proj adapts to any F — the contract is [B, F, 8, 8], not a
    fixed 512). ``blocks`` is the per-stage BasicBlock depth (widened trunk keeps
    the deep-34 (3, 6, 16, 6) shape; XL deepens stage-3/4). Param counts are
    measured, not estimated — see MODEL_REGISTRY.md section 4."""
    in_channels: int = 9          # D-015 3-frame RGB stack (latest = [-3:])
    image_size: int = 256         # HEIGHT (and width, unless image_width is set)
    image_width: int | None = None   # non-square input; None == square (default)
    base_width: int = 88          # V2-99-class width (90.5 M trunk); XL -> 124
    blocks: tuple[int, ...] = (3, 6, 16, 6)    # deep-34 (8x8xF map preserved)

    @property
    def feat_dim(self) -> int:
        return self.base_width * 8

    @property
    def grid(self) -> int:
        """Square feature-map side. Raises on a non-square input ON PURPOSE —
        :attr:`grid_shape` is the general accessor."""
        gh, gw = self.grid_shape
        if gh != gw:
            raise ValueError(
                f"REF-C feature map is {gh}x{gw} (non-square) — this caller "
                f"still reads the scalar `grid`. Use `grid_shape`.")
        return gh

    def image_hw(self) -> tuple[int, int]:
        return (self.image_size,
                self.image_size if self.image_width is None else self.image_width)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Conv feature-map grid ``(rows, cols)`` = input // 32 per axis."""
        h, w = self.image_hw()
        return (h // 32, w // 32)


@dataclass
class MeasurementConfig:
    hidden: int = 128
    d_out: int = 128


@dataclass
class TrajectoryConfig:
    # 2 s @ 10 Hz in 0.5 s strides (the REF-B tactical horizons) — time-indexed
    # waypoints; under refc1 the SAME step slots are read as fixed-distance path
    # checkpoints (RefCConfig.path_dists). The anchor vocabulary lives in this
    # [len(horizons), 2] ego-frame trajectory space.
    horizons: tuple[int, ...] = (5, 10, 15, 20)


@dataclass
class AnchorConfig:
    n_anchors: int = 128          # FPS vocabulary size (base 128; XL 256; 20 smoke)
    pool_size: int = 4096         # synthetic pool the default anchors FPS over
    seed: int = 0
    # ⭐⭐ refcv4-b — THE v0-CONDITIONED VOCABULARY (2026-09-04).
    #
    # A FIXED bank of paths in absolute metres must spend its budget on the
    # SPEED axis before it can spend any of it on shape: over 6 s the corpus
    # covers ~0-216 m of along-track displacement, so most of 128 anchors go to
    # "how fast", not "which way". `ha` (hold-action) is v0-conditioned by
    # construction, which is why a fixed bank cannot reach it. MEASURED on the
    # banked 4,823-window surface: the shipped fixed-path set reads 0.3773 m
    # oracle-in-vocabulary (+0.0777 [+0.0528, +0.1044] SEPARATED WORSE than
    # ha = 0.2996), while a v0-conditioned constant-(accel, curvature) family of
    # 117 candidates reads 0.2610 (-0.0387 [-0.0652, -0.0104] BEATS ha).
    #
    # With this ON, `anchor_controls` [N, 2] = (accel m/s^2, curvature 1/m) is
    # the vocabulary and the emitted bank is rolled per window through
    # `rollout_unicycle` from the window's own v0. `anchors` [N, S, 2] still
    # holds the bank rolled at `ref_speed_ms` -- it is the checkpoint-visible
    # artifact and the surface `_lan_anchor_prior` / `_goal_along_prior` fall
    # back to, so a content check against the built file still works.
    #
    # ⛔ ego-dropout is NOT weakened by this. `v_ms` is the PRE-dropout speed;
    # rolling the bank from it on a window whose speed was WITHHELD would feed
    # the channel back in through the candidate GEOMETRY -- a more direct leak
    # than the ranking one S2 guards. Withheld rows are rolled at
    # `ref_speed_ms` instead, so the dropout regime stays genuinely speed-blind.
    v0_conditioned: bool = False
    ref_speed_ms: float = 10.0
    # ⭐ WHAT CHANNEL 1 OF `anchor_controls` MEANS.
    #   "kappa" — curvature 1/m, integrated as supplied (the literal control).
    #   "alat"  — LATERAL ACCELERATION m/s^2; curvature is DERIVED per window as
    #             `a_lat / max(v0, alat_v_floor_ms)^2`, clamped to `kappa_cap`.
    # MEASURED: a constant-CURVATURE family is unflyable at speed (a_lat =
    # v^2 * kappa, so kappa 0.06 at 27 m/s is 4.5 g and 104 of 117 anchors break
    # a mu = 0.7 circle). Under "alat" the control space IS the Kamm-circle
    # space, so a bound on the grid is a bound on the friction circle: the same
    # 117-anchor budget reads 0.1987 m oracle-in-vocabulary (-0.1009 [-0.1213,
    # -0.0813] vs ha) with 0/117 over mu = 0.7 and a 0.68 g peak.
    # `a_lat = 0` <=> `kappa = 0` exactly, so the pinned straight-ahead control
    # survives the reparameterisation.
    control_units: str = "kappa"
    alat_v_floor_ms: float = 4.0     # below this the clamp would explode
    kappa_cap: float = 0.12          # ~8.3 m turn radius; a parking-lot bound


@dataclass
class DecoderConfig:
    d: int = 384                  # decoder width (anchor queries + cross-attn)
    n_heads: int = 8
    layers: int = 4               # cross-attention layers (base 4; XL 6)
    ff_mult: int = 4
    aux_hidden: int = 384         # maneuver-head hidden
    diffusion_steps: int = 2      # truncated-denoise steps (0 == classifier)
    noise_std: float = 0.1        # train-time truncated-diffusion noise (metres)
    # ---- THE FEASIBILITY-AWARE DECODE (default OFF, byte-identical when off) --
    # MEASURED 2026-09-05: this decode emits a fan whose mean friction load is
    # 4.18 g while the frozen vocabulary it started from is drivable at 0.48 g --
    # an 8.6x blow-up manufactured DOWNSTREAM of the vocabulary, with 88.8 % of
    # candidates outside the (a, kappa) envelope. A feasibility REWARD moves the
    # ranking by 0.001 and a post-train VETO closed ~2.7 % of the gap while
    # regressing T1 ade_m +0.0362 m. Both are soft: they penalise a path the
    # decode can still emit. With this ON, an envelope- or Kamm-violating path is
    # UNREPRESENTABLE, and the projection is applied at EVERY pass that moves a
    # waypoint -- a penalty on `out["offset"]` reaches one of three.
    feasible_decode: bool = False
    feasible_mu: float = 0.7      # friction circle; None-like <= 0 disables it
    feasible_entry: bool = False  # also bind the first step's speed to v0 +- a*dt
    feasible_a_max: float = 4.0   # == rewards.A_MAX_MPS2 (the scorer's own)
    feasible_kappa_max: float = 0.2
    feasible_prefix_slots: int = 4  # the UNIFORM 2 s prefix the scorer differentiates
    # ---- refcv5 WP-4: THE CONTROL-SPACE DIFFUSION SAMPLER (default OFF) -----
    #
    # ⭐ THESE ARE REAL FIELDS, AND THAT IS THE POINT. Until 2026-09-06 the
    # trainer's `_pin_refcv5_seams` wrote `core.decoder.sampler = "ddim"` onto
    # an UNFROZEN dataclass that had no such field. Python accepted it, the
    # trainer's own guards read it back and passed, and `_seam_stamp` copied it
    # into `config.json` -- for a model that contained NO DENOISER. The run
    # record claimed a sampler the weights did not have: FALSE PROVENANCE, the
    # `--v2` conflation failure in a new costume. A declared field plus the
    # build below plus `assert_seams_are_built` (refc_v3_train) closes it at
    # all three levels: the config can hold it, the model must build it, and
    # the stamp is checked against the MODEL rather than against the config
    # that produced it.
    #
    # `"none"` constructs NOTHING -- not "constructs and gates" -- so RNG draw
    # order is unchanged and every pre-v5 checkpoint keeps loading strictly.
    sampler: str = "none"              # "none" | "ddim"
    # ⛔ "metre" is the DD-LITERAL DELIBERATE REGRESSION, pre-registered to FAIL
    # the flyability gate: it noises the WAYPOINTS, so a 0.73 m lateral
    # excursion over a 0.5 s slot implies ~5.8 m/s^2 (0.6 g) of lateral
    # acceleration and the fan fills with paths no car can drive. An arm that
    # cannot be RUN cannot fail, so it is a code path, not a paragraph.
    sampler_space: str = "control"     # "control" | "metre"
    sampler_train_t_max: int = 50      # DD's train draw, t ~ U[0, t_max)
    sampler_infer_t: int = 8           # DD's truncation point (sigma = 0.0316)
    sampler_steps: int = 2             # DDIM steps at inference -> ladder [10, 0]
    # ⛔ > 1 is REFUSED at forward time, not silently honoured: a [B, G*N, ...]
    # fan mis-indexes `loss_cls`'s a_star, the [B, N] priors and every `sel_idx`
    # dump, all of which still assume N.
    sampler_groups: int = 1
    # (a_lon, a_lat) m/s^2 -- the normaliser the anchored Gaussian and the x0
    # loss both divide by, so "sigma at t=8" means the same thing in both.
    control_norm: tuple[float, float] = (4.0, 3.0)
    # ⛔ DD's PUBLISHED PER-WAYPOINT SIGMA **AT THE TRUNCATION POINT**, in
    # METRES -- a TARGET, not a divisor, and the distinction is the whole
    # regression. Used as a divisor directly it delivers `sigma(8) * 0.90 =
    # 0.028 m` of noise: **31.7x too gentle**, and the DD-literal arm that is
    # pre-registered to FAIL the flyability gate would quietly PASS it. An arm
    # that cannot fail proves nothing, which is the defect this whole seam is
    # instrumented against. The normaliser is therefore DERIVED in `_sample` as
    # `metre_sigma_m / sqrt(1 - abar(sampler_infer_t))` (= 28.49 / 23.11), so
    # the emitted noise IS the published 0.90 m / 0.73 m.
    metre_sigma_m: tuple[float, float] = (0.90, 0.73)
    # ---- refcv5 WP-6: the agent seam on the decoder's cross-attention -------
    cross_agent: bool = False


@dataclass
class LawConfig:
    hidden: int = 2048            # latent-world-model aux MLP width


@dataclass
class StrategicCtxConfig:
    hidden: int = 512             # ctx GRU width
    d_ctx: int = 64               # strategic token -> decoder condition seam


@dataclass
class ImaginationConfig:
    """H15 belief field over the conv-map tokens (graft_imagination). Sized on
    REF-C-XL to ~22 M as the flagship-matched capacity control."""
    d: int = 512                  # belief-field width (in/out project feat_dim<->d)
    depth: int = 6                # self-attention refinement blocks
    n_heads: int = 8
    ff_mult: int = 4              # refinement-block MLP ratio
    head_hidden: int = 1024       # flow / log-variance head hidden width


@dataclass
class LanConfig:
    """LAN — Lane-Anchored Navigation route conditioning (graft_lan, default
    OFF). Shape MUST match ``tanitad.data.lan.LanConfig``: K route anchors x 4
    features [cos bearing, sin bearing, lat_norm, valid].

    Why this seam exists: the 4-way ``nav_cmd`` one-hot is ``follow`` on the
    large majority of windows (``nav_valid_frac`` 0.21-0.25, MEASURED across all
    four arms) and is a CONSTANT at eval (every published REF-C number decodes
    with ``nav_cmd=None`` -> index 0). LAN adds a dense, always-defined,
    leak-guarded route corridor ALONGSIDE nav_cmd — it never removes it.
    """
    k: int = 4                    # route anchors (arc-lengths, not timesteps)
    feats: int = 4                # per-anchor features (pinned by data.lan)
    hidden: int = 64              # route-feature MLP width
    d_out: int = 64               # route embedding -> decoder condition seam

    @property
    def dim(self) -> int:
        """Flat input width the model consumes."""
        return self.k * self.feats


@dataclass
class SelectionConfig:
    """D-SEL — the selection-surface policy the decoder carries.

    A projection of the ``RefCConfig`` D-SEL fields, built by
    :meth:`RefCConfig.selection`. It exists so the decoder owns its own ranking
    policy (every lever acts on the argmax this class performs) without the
    decoder needing the whole model config. Defaults are exactly today's
    behaviour, so a decoder built with ``SelectionConfig()`` is byte-identical
    to one built before D-SEL existed.
    """
    refined: bool = False             # S1 rank on the refined confidence
    score_emitted: bool = False       # S1b read that confidence FROM THE EMITTED
    #                                   fan (one extra conf-only pass), so the
    #                                   scored object IS the ranked object
    score_emitted_t: int = -1         # ...with WHICH timestep token. -1 continues
    #                                   the loop's own schedule; >=0 pins one.
    #                                   t=0 is the ONLY token `loss_cls` ever
    #                                   supervises, and "this estimate is clean"
    #                                   is arguably what a denoised fan IS.
    #                                   0 parameters either way (the embedding
    #                                   table already carries every index).
    reach_clamp: bool = False         # S2 bounded-acceleration candidate band
    accel_max: float = 2.5            # m/s^2 for that band
    # S2b: THE SAME BAND, MOVED BEFORE THE DECODE. S2 filters `anchors+offset`,
    # so by the time it runs all N decodes are paid and it can only narrow an
    # argmax. `v0` is known PRE-decode, so the band applies to the ANCHORS and
    # only survivors need decoding. INFERENCE-ONLY and no retrain: it changes
    # which candidates are computed, never any weight.
    # MEASURED (be2da04, re-measured 2026-08-04 on the 881 canonical val
    # speeds): 3.46x on the 256 vocabulary, 3.70x on 64, ZERO empty windows,
    # and the SELECTION INDEX identical on 881/881.
    # ⛔ EXACTNESS RESTS ON ONE STRUCTURAL FACT, verified in source rather than
    # assumed: `CrossAttnLayer` cross-attends q->kv with NO self-attention over
    # the candidate axis, and its MLP/LayerNorm are per-token. Candidates are
    # therefore INDEPENDENT and decoding a subset is bit-identical for that
    # subset. If a candidate-axis interaction is ever added to the decoder this
    # flag becomes UNSOUND and must be retired with it.
    anchor_prefilter: bool = False    # S2b decode only reachable anchors
    anchor_prefilter_guard: bool = True   # verify the winner survived
    graft_cons: bool = False          # S3 consequence score reaches the ranking
    cons_detach: bool = True          # ...with law_head under no_grad
    graft_route: bool = False         # S5 route readout reaches the ranking
    graft_goal: bool = False          # S6 predicted GEOMETRIC goal (bearing +
    #                                   along-track), two independent gates
    # ⭐⭐ S7 / E15 (GP-1, 2026-09-06): the PREDICTED METRIC GOAL POINT reaches
    # the ranked score through a PARAM-FREE geometric compatibility at a fixed
    # TIME slot (`goal_point.anchor_goal_prior_at_time`), behind ONE zero-init
    # gate. Registered in `…/Research/2026-09-06-goal-point/PREREG.md` §8.
    #
    # ⛔ WHY IT IS A SEPARATE SEAM FROM S6 AND NOT A THIRD S6 GATE. S6's two
    # terms are a BEARING (`_lan_anchor_prior`, terminal-direction cosine) and
    # an ALONG-TRACK PREFERENCE (`_goal_along_prior`, z-scored terminal reach).
    # MEASURED (PREREG §0): on the lateral axis a point and a bearing TIE TO
    # FOUR DECIMALS — an identity, because anchors compared at the same ARC all
    # sit at ~the same range — while at a fixed TIME the candidates differ in
    # RANGE, and the range half is separated WORSE by +2.3632 m when stripped.
    # A metric point at a fixed time is therefore a DIFFERENT quantity from
    # either S6 term, and mixing it into their gates would make the pre-
    # registered ablation ("set THIS gate to 0") unreadable.
    graft_gp_point: bool = False      # S7 the E15 metric goal point ranks
    gp_slot: int = -1                 # ...bank slot whose time == t_goal_s;
    #                                   DERIVED by goal_point.goal_slot_index,
    #                                   never typed (a wrong slot compares the
    #                                   goal at one time to the anchor at
    #                                   another and still trains)
    gp_scale_m: float = 40.0          # ...GoalPointConfig.range_norm_m, so the
    #                                   score is metres/metres = scale-free
    seam_clamp: float = 0.0           # S4 norm cap on the total graft (<=0 off)
    seam_fail: float = 1.5
    seam_fail_frac: float = 0.75
    seam_fail_patience: int = 50
    horizon_s: float = 2.0            # last horizon in seconds (band + speeds)

    @property
    def any_on(self) -> bool:
        return bool(self.refined or self.score_emitted or self.reach_clamp
                    or self.graft_cons or self.graft_route or self.graft_goal
                    or self.graft_gp_point
                    or self.seam_clamp > 0.0)


@dataclass
class RefCConfig:
    encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    window: int = 8               # shared state window (main stack: 8)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # ⭐ refcv5 WP-6 -- `refc_agents.AgentSeamConfig | None`. A DECLARED field,
    # for the same reason `decoder.sampler` is one: the trainer assigns
    # `core.agents = acfg`, and on a dataclass without the field that assignment
    # is an ad-hoc attribute the stamp would happily serialise for a model that
    # built no head. `None` (not a disabled config) is the OFF state, so
    # `_seam_stamp` can distinguish "off" from "absent".
    # ⚠️ Annotated loosely on purpose: `from __future__ import annotations` is
    # in force, so this is never evaluated, and naming the real type here would
    # force a module-level import of the corpus-side raster/projection stack.
    agents: "object | None" = None
    # ⭐ WP-D -- `refc_bev_aux.BEVAuxConfig | None`, the TRAINING-ONLY BEV
    # auxiliary head. Declared for the same reason `agents` is: the trainer
    # assigns `core.bev_aux = bcfg`, and on a dataclass without the field that
    # assignment is an ad-hoc attribute the stamp would serialise for a model
    # that built no head. `None` (not a disabled config) is the OFF state.
    #
    # ⛔⛔ WHEN IT IS SET, THE HEAD IS CONSTRUCTED **LAST** IN `__init__` AND
    # CALLED **LAST** IN `forward`, AND BOTH ARE LOAD-BEARING. Module
    # construction draws from the global RNG, so a head inserted anywhere
    # earlier silently changes every subsequent module's initial weights and the
    # "aux on vs aux off" A/B would differ in the SEED as well as in the lever --
    # a one-variable violation invisible in every log. Pinned by
    # `stack/tests/test_bev_aux.py::test_shared_params_bit_identical` and
    # `::test_planner_output_bit_identical`.
    # ⚠️ Annotated loosely on purpose, exactly like `agents` above.
    bev_aux: "object | None" = None
    law: LawConfig = field(default_factory=LawConfig)
    strategic: StrategicCtxConfig = field(default_factory=StrategicCtxConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    lan: LanConfig = field(default_factory=LanConfig)
    speed_hidden: int = 256       # refc1 target-speed class head width
    ego_dropout: float = 0.5      # per-sample Bernoulli zero of v0 (training)
    route_dropout: float = 0.5    # per-sample Bernoulli mask of the LAN route
    hierarchy: bool = True        # strategic ctx -> decoder condition (graft)
    # ⭐⭐ STRATEGIC BYPASS (`--no-strategic`, PI 2026-09-06, BINDING):
    #   *"remove the strategic layer in the next experiments, feed the nav
    #   command to tactical and operative planning, solve the driving task,
    #   then add the strategic layer. Include a flag to ignore the strategic
    #   layer in the next iterations."*
    #
    # ⛔ IT BYPASSES, IT NEVER DELETES. `hierarchy` stays True, so
    # `StrategicCtx`, `ctx_to_cond` and every strategic parameter are STILL
    # CONSTRUCTED and STILL IN `state_dict`. An ON build and an OFF build have
    # BIT-IDENTICAL state_dict KEYS AND SHAPES, so every banked checkpoint
    # loads strictly into both and no arm is stranded. That is the documented
    # precedent: a dead 1.57 M-param tensor was declared and kept in
    # `state_dict` because deleting it broke strict loads on 11/11 banked
    # checkpoints with exactly 6 unexpected keys. Setting `hierarchy=False`
    # would REMOVE keys and reintroduce exactly that failure -- it is NOT the
    # bypass and must never be used as one.
    #
    # ⛔ IT ADDS NO PARAMETER EITHER. The flag is a pure FORWARD gate: it
    # can only remove a term from a sum, never introduce one. That is what
    # makes "OFF is byte-identical to today" provable rather than asserted.
    #
    # The two seams it closes IN THIS FILE (the third is E4 in `refc_v3.py`):
    #   S-BYPASS-1  OPERATIVE: `ctx` is not handed to the decoder, so
    #               `cond = cond_proj(m)` alone. The nav command STILL reaches
    #               the operative planner -- it is INSIDE `m` (`meas_in`
    #               carries the nav one-hot), which is the whole point of the
    #               PI's staging: the route reaches the planner DIRECTLY
    #               instead of through a strategic goal.
    #   S-BYPASS-3  SELECTION: `route_prior` (S5 `graft_route`) is forced to
    #               None, so the strategic route READOUT cannot re-rank the fan.
    #
    # ⚠ WHAT IT DOES NOT TOUCH: `route_head` keeps running and
    # `route_logits` keeps being emitted, because that is the STRATEGIC family
    # metric every eval must report (route acc / kappa). With `graft_route`
    # False it is a pure READOUT with no path into the emitted plan -- which
    # the mutation proof ASSERTS rather than assumes.
    no_strategic: bool = False    # bypass the strategic layer (never delete it)
    graft_maneuver: bool = True   # maneuver logits reweight anchor priors (H19)
    # --- D-TAC1: the factorised tactical head (three INDEPENDENT ablations) ---
    factored_maneuver: bool = False   # F2 STRUCTURE: lat(3) x lon(3) heads +
    #                                   split lat/lon anchor grafts, replacing
    #                                   the mixed 5-way softmax. Off => the
    #                                   model is byte-identical to today.
    tactical_speed_input: bool = False  # F1 INPUT: the tactical head(s) read the
    #                                   ego speed alongside the image embedding.
    #                                   Deliberately the SPEED CHANNEL ONLY, not
    #                                   the full measurement: `nav_cmd` is a
    #                                   CONSTANT at eval (nav_cmd=None -> follow)
    #                                   so feeding it here would train the head
    #                                   on a signal that vanishes at eval — the
    #                                   C6 confound. INDEPENDENT of
    #                                   factored_maneuver: it widens the SHIPPED
    #                                   5-way head's input too, which is what
    #                                   makes an F1-only arm possible.
    man_prior_tau: float = 0.0        # F3 DECISION: logit-adjustment strength
    #                                   for the REPORTED class. 0.0 = argmax of
    #                                   the raw posterior (today's behaviour).
    graft_prior_center: bool = True   # feed the anchor grafts the log-likelihood
    #                                   RATIO rather than the raw log-posterior
    #                                   (conditioning, not expressivity — see
    #                                   refc_tactical.prior_centered_logprobs).
    tactical_prior_momentum: float = 0.99   # EMA on the label class priors the
    #                                   trainer feeds to update_tactical_prior()
    graft_target_latent: bool = False   # FiLM the condition on a goal latent
    grounded_selector: bool = False     # progress/collision proxy vs top-1 conf
    graft_imagination: bool = False     # H15 belief field over conv-map tokens
    graft_lan: bool = False             # LAN lane-anchored route conditioning
    # --- D-SEL: the SELECTION surface (see tanitad/refs/refc_select.py) ------
    # Every lever below acts on WHICH candidate is emitted, never on which are
    # proposed. All are zero-init or param-free; all default OFF, so the
    # state_dict of an all-off build is byte-identical to REF-C today.
    sel_refined: bool = False         # S1 rank the refined fan with the REFINED
    #                                   confidence (today: the t=0 classifier
    #                                   score ranks post-denoise trajectories).
    #                                   Inert at steps=0 BY CONSTRUCTION.
    # --- S1's CLIMB-OUT: two ZERO-PARAMETER distribution matches --------------
    # E-SEL-0 MEASURED that the refined readout, UNSUPERVISED, ranks 0.8372 m
    # (base) / 0.9187 m (XL) WORSE than the shipped t=0 score - separated, both
    # arms - while still scoring 8.7x / 16.6x chance. So it is off-distribution,
    # not uninformative, and S1 must CLIMB OUT (supervise it) rather than HARVEST
    # it. These two flags remove the two places where the object that is SCORED,
    # the object that is SUPERVISED and the object that is EMITTED are still
    # three different things. Both cost 0 parameters.
    sel_score_emitted: bool = False   # S1b the ranked confidence is read from the
    #                                   EMITTED fan. Today the last denoise pass
    #                                   scores its own INPUT `x_in` and the fan
    #                                   that leaves the decoder is `x_in + off` -
    #                                   the emitted trajectories are never scored
    #                                   by any head. Costs one extra conf-only
    #                                   decoder pass; 0 parameters. Inert at
    #                                   steps=0 BY CONSTRUCTION (as S1 is).
    sel_ce_reach: bool = False        # S1c the ranked-score CE normalises over
    #                                   EXACTLY the survivor set the argmax ranks
    #                                   over, and its target is the best candidate
    #                                   IN that set. Today the CE is a full-fan
    #                                   softmax while the selector solves a
    #                                   ~26-28 % sized problem: MEASURED 73.76 %
    #                                   (base) / 72.08 % (XL) of the fan is
    #                                   unreachable and never selected, and
    #                                   S3_DEPLOYABLE 3.2 measured that a
    #                                   statistic over the whole candidate axis is
    #                                   DOMINATED by candidates no selector ever
    #                                   picks. REQUIRES sel_reach_clamp (the mask
    #                                   is its survivor set); the trainer refuses
    #                                   the combination without it. 0 parameters.
    sel_score_emitted_t: int = -1     # ...and with WHICH timestep token (-1 =
    #                                   continue the loop's schedule). MEASURED
    #                                   POST-HOC: `loss_cls` supervises the conf
    #                                   head ONLY at t=0, so the token matters.
    # --- E-OBJ-1: the OBJECTIVE the ranked score is trained under --------------
    # "ce"      (default) the INCUMBENT one-hot cross-entropy. Bit-unchanged.
    # "softade" the EXPECTED fan error under the score's own softmax -- a
    #           METRIC-AWARE objective. ⭐ MEASURED (E-OBJ-1, frozen 30 k weights,
    #           881 windows, LOEO, paired episode-cluster bootstrap): swapping a
    #           fitted ranker's objective from the CE to this recovers -0.0974 m
    #           (base) / -0.1670 m (XL) of its deficit, separated, and the
    #           recovery is LONGITUDINAL (`speed_abs` -0.1102 / -0.1816).
    # "softce"  the CE form with a SOFTENED target `softmax(-fan_err/tau)`.
    #           ⚠️ MEASURED SEPARATED **WORSE** than the incumbent CE (+0.0909 m
    #           base) at every tau in {0.1, 0.25, 0.5}. It is implemented ONLY so
    #           that a `softade` arm has the control that separates
    #           METRIC-AWARENESS from TARGET-SOFTNESS in training. ⛔ Not a
    #           recommendation.
    # 0 parameters, all three: they change a scalar loss, never a module.
    # ⚠️ SCALE: "ce"/"softce" are in NATS, "softade" is in METRES, and
    # REFINED_CLS_WEIGHT was calibrated for the former. An arm that swaps the
    # objective MUST decide that weight explicitly -- the trainer says so loudly.
    sel_ce_objective: str = "ce"
    #: extra multiplier on `loss_rcls`, applied ONLY when `sel_ce_objective` is
    #: not "ce", so the incumbent path stays bit-identical. It exists because the
    #: objective swap changes the loss's UNITS, and a silent re-weighting of the
    #: selection term against LAW / maneuver / trajectory is exactly the kind of
    #: confound that makes an arm unattributable. The trainer refuses a value
    #: other than 1.0 on the "ce" path rather than recording an inert number.
    sel_ce_weight: float = 1.0
    # --- E-OBJ-1: the TARGET SHAPE of the ranked-score CE ----------------------
    # `loss_rcls` is a ONE-HOT cross-entropy over ~128 near-duplicate candidates:
    # one winner, 127 losers, and the loser that missed by a centimetre is
    # penalised exactly as hard as the one that missed by ten metres. MEASURED
    # (E-S1-0 3.1): under that objective EVERY fitted ranker is separated WORSE
    # than the incumbent selector - including feature sets that CONTAIN the
    # incumbent's own score - with a C-leak gap of -0.001 to -0.003 m, i.e. NOT
    # overfitting. Softening the TARGET to `softmax(-fan_err / tau)` keeps the CE
    # form (and with it the gradient path `cons_gate` / `route_to_anchor` depend
    # on - see the note at loss_rcls) and only stops the objective insisting on
    # one winner among near-duplicates.
    #   0.0 (default) => the INCUMBENT one-hot target, bit-unchanged.
    #   tau -> 0      => converges to the one-hot target BY CONSTRUCTION, which is
    #                    what makes this a continuous knob and not a new loss.
    # 0 parameters: it changes a target tensor, never a module.
    sel_ce_soft_tau: float = 0.0
    sel_reach_clamp: bool = False     # S2 bounded-acceleration band on the
    #                                   CANDIDATES (argmax only; the returned
    #                                   score stays unmasked so no -inf reaches
    #                                   a cross-entropy)
    sel_accel_max: float = 2.5        # m/s^2 for that band (flagship v1.5's)
    # S2b: the SAME band moved BEFORE the decode, so only reachable anchors are
    # decoded. INFERENCE-ONLY and OUTPUT-EXACT (candidates are independent —
    # `CrossAttnLayer` has no self-attention over the candidate axis), so it
    # needs NO retrain and changes no weight. See `SelectionConfig`.
    sel_anchor_prefilter: bool = False
    sel_anchor_prefilter_guard: bool = True
    graft_cons: bool = False          # S3 candidate-conditioned CONSEQUENCE
    #                                   scoring through law_head — the only
    #                                   form of cond_imagination REF-C's world
    #                                   model can express
    cons_detach: bool = True          # run law_head under no_grad in S3, so the
    #                                   ranking objective cannot corrupt the
    #                                   world model (the flagship's FROZEN-
    #                                   predictor discipline)
    graft_route: bool = False         # S5 the strategic route READOUT reaches
    #                                   the ranked score (zero-init), instead of
    #                                   only warping the decoder condition
    graft_goal: bool = False          # S6 a PREDICTED GEOMETRIC goal (bearing +
    #                                   along-track distance) reaches the ranked
    #                                   score through the SAME param-free
    #                                   geometric compatibility LAN uses. The
    #                                   PI's 2026-08-03 ruling: a goal input is
    #                                   admissible, but it must be GEOMETRIC and
    #                                   PREDICTED, never categorical-and-
    #                                   supplied, and it must not carry the
    #                                   situation classifier's output in any
    #                                   form. See RefCModel.goal_provenance().
    # ⭐⭐ S7 / E15 (GP-1): the PREDICTED METRIC GOAL POINT reaches the ranked
    # score. The VALUE arrives from the REF-C v3 hierarchy hook
    # (`hook_out["goal_point"]`, emitted by `refc_v3.py`) or from an explicit
    # `gp_point=` argument to `forward` — which is what the pre-registered
    # eval-time interventions (mirror / range x0.5 / straight / shuffle /
    # withheld) use, so the whole §2 panel rolls in ONE process on ONE surface.
    # `gp_slot` / `gp_scale_m` are set by the trainer from `GoalPointConfig`.
    graft_gp_point: bool = False
    gp_slot: int = -1
    gp_scale_m: float = 40.0
    seam_clamp: float = 0.0           # S4 in-graph norm cap on the TOTAL graft
    #                                   per surface, as a multiple of the base
    #                                   score norm. <=0 disables; below the cap
    #                                   the rescale is exactly 1.0 (bit-exact).
    seam_fail: float = 1.5            # ...fail-loud ceiling on the batch MEAN
    seam_fail_frac: float = 0.75      # ...AND this share of the batch clamped
    seam_fail_patience: int = 50      # ...AND sustained this many steps (0=off)
    ego_valid_channel: bool = False   # X15: never zero-fill a channel whose
    #                                   zero is a valid in-distribution value.
    #                                   Adds an explicit "v0 is present" flag
    #                                   alongside the ego-dropped speed, for the
    #                                   measurement encoder AND (when
    #                                   tactical_speed_input is on) the tactical
    #                                   head. 0.0 m/s is in-distribution
    #                                   "stationary"; masking to it is a
    #                                   confident lie.
    nav_known_channel: bool = False   # E1: the COMPANION BIT of the nav command.
    #                                   `nav_command_v21` collapses ROUTE_UNKNOWN
    #                                   and ROUTE_STRAIGHT onto the SAME
    #                                   NAV_FOLLOW token, so "the road goes
    #                                   straight" and "I could not judge the
    #                                   route" are BYTE-IDENTICAL at the model
    #                                   input. MEASURED: 1,985 of 3,179 (62.4 %)
    #                                   `follow` windows are a collapsed UNKNOWN.
    #                                   This adds `nav_known` in {0,1} beside the
    #                                   one-hot, exactly as `ego_valid_channel`
    #                                   adds the "v0 is present" flag beside the
    #                                   ego-dropped speed — the same X15 rule
    #                                   ("never zero-fill a channel whose zero is
    #                                   a valid in-distribution value"), applied
    #                                   to the route instead of the speed.
    #                                   ⛔ DEFAULT OFF. Turning it on changes what
    #                                   every arm is fed and is a PI decision.
    #                                   Feed it from `refb_labels.nav_input_v22`.
    tactical_latent_dim: int = 512      # external target_latent width (S)
    refc1: bool = False           # fixed-distance path + target-speed class
    path_dists: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)   # metres (refc1)
    speed_bins: int = 4           # refc1 target-speed classes over [0, max]
    speed_max: float = 30.0       # m/s

    def selection(self) -> SelectionConfig:
        """The D-SEL policy the decoder carries. ``horizon_s`` is DERIVED from
        the trajectory horizons (0.1 s steps), never a constant — the reachable
        band and the anchors must agree about how long the plan is."""
        return SelectionConfig(
            refined=self.sel_refined, score_emitted=self.sel_score_emitted,
            score_emitted_t=self.sel_score_emitted_t,
            reach_clamp=self.sel_reach_clamp,
            accel_max=self.sel_accel_max,
            anchor_prefilter=self.sel_anchor_prefilter,
            anchor_prefilter_guard=self.sel_anchor_prefilter_guard,
            graft_cons=self.graft_cons,
            cons_detach=self.cons_detach, graft_route=self.graft_route,
            graft_goal=self.graft_goal,
            graft_gp_point=self.graft_gp_point,
            gp_slot=self.gp_slot, gp_scale_m=self.gp_scale_m,
            seam_clamp=self.seam_clamp, seam_fail=self.seam_fail,
            seam_fail_frac=self.seam_fail_frac,
            seam_fail_patience=self.seam_fail_patience,
            horizon_s=max(self.trajectory.horizons) * 0.1)


def refc_config() -> RefCConfig:
    """REF-C-base ~110 M (the primary, data-appropriate for the full 2376-ep
    set): a V2-99-class ResNet trunk (base_width 88 -> ~90 M encoder, 8x8xF map)
    + the d=384 / 4-layer / 128-anchor anchored-diffusion decoder + LAW aux +
    strategic-ctx hierarchy. The budget lands in the ENCODER on purpose — the
    Hydra-MDP ResNet-34 -> V2-99 lever (86.6 -> 91.0 PDMS). Imagination graft OFF.
    Measured by param_breakdown at instantiation; tests pin the 90-130 M band."""
    return RefCConfig()


def refc_small_config() -> RefCConfig:
    """REF-C-small ~55 M — the RESEARCH-anchored size (DiffusionDrive's proven
    ~60 M scale: NAVSIM 88.1 PDMS). Low end of the size-vs-data scaling study
    (small 55 M / base 104 M / XL 252 M on the IDENTICAL 2376-ep set, read via
    the 5k/15k/20k/30k milestone gates, to see where bigger helps vs overfits on
    our data). Encoder base_width 64 + deeper blocks (3,6,16,6) -> ~48 M trunk
    (V2-class depth at ResNet-34 width) + the d=256 / 3-layer / 64-anchor
    decoder; same anchored-diffusion algorithm + LAW + strategic-ctx as base/XL,
    imagination OFF. Tests pin the 45-65 M band."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=9, image_size=256, base_width=64,
                                   blocks=(3, 6, 16, 6))
    cfg.decoder = DecoderConfig(d=256, n_heads=4, layers=3, ff_mult=4,
                                aux_hidden=256, diffusion_steps=2, noise_std=0.1)
    cfg.anchors = AnchorConfig(n_anchors=64, pool_size=2048)
    cfg.strategic = StrategicCtxConfig(hidden=512, d_ctx=64)
    return cfg


def refc_xl_config() -> RefCConfig:
    """REF-C-XL ~260 M (flagship-matched capacity control — same-capacity vs the
    261 M flagship, removing the "REF-C is worse because smaller" confound). SAME
    refc.py code + decoder algorithm as base; only the widths/depths grow:
      encoder   base_width 124, blocks (3, 8, 20, 6) -> ~200 M ResNet-L trunk
                (wider AND deeper than base; 8x8xF map preserved, F = 992, the
                decoder feat_proj adapts).
      decoder   d=512, 6 cross-attn layers, 256 FPS anchors.
      grafts    H15 imagination field ON (~22 M belief field over the conv-map
                tokens) — the extra budget beyond the wider trunk/decoder.
    Tests pin the 230-280 M band with a full encoder/decoder/imagination split."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=9, image_size=256, base_width=124,
                                   blocks=(3, 8, 20, 6))
    cfg.decoder = DecoderConfig(d=512, n_heads=8, layers=6, ff_mult=4,
                                aux_hidden=512, diffusion_steps=2, noise_std=0.1)
    cfg.anchors = AnchorConfig(n_anchors=256, pool_size=4096)
    cfg.strategic = StrategicCtxConfig(hidden=768, d_ctx=96)
    cfg.imagination = ImaginationConfig(d=512, depth=6, n_heads=8, ff_mult=4,
                                        head_hidden=1024)
    cfg.graft_imagination = True
    return cfg


def refc_factored_config() -> RefCConfig:
    """REF-C-base with the D-TAC1 factorised tactical head — the pre-registered
    arm (``Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md``).

    IDENTICAL to :func:`refc_config` except for the tactical seam, so the A/B is
    attributable to STRUCTURE and INPUT, not to capacity: the delta is two
    ``Linear(aux_hidden, 3)`` readouts off a SHARED trunk in place of one
    ``Linear(aux_hidden, 5)``, two ``Linear(3, n_anchors)`` grafts in place of
    one ``Linear(5, n_anchors)``, and one extra input scalar.
    **MEASURED** by ``param_breakdown`` (tests/test_refc_tactical.py):
    104,191,577 -> 104,192,474, i.e. **+897 parameters (+0.00086 %)**. The
    earlier spec's "~5 k" was an ESTIMATE and the first implementation here cost
    +272,001 because it built two full MLPs — the test is what caught both.

    All three levers ON together, because the pre-registered probes decide which
    are load-bearing BEFORE any GPU-day is spent; the single-lever ablations are
    the same function with one field flipped back.
    """
    cfg = refc_config()
    cfg.factored_maneuver = True     # F2 structure
    cfg.tactical_speed_input = True  # F1 input
    cfg.man_prior_tau = 1.0          # F3 decision rule (balanced-posterior)
    return cfg


def refc_f1only_config() -> RefCConfig:
    """REF-C-base with the SHIPPED 5-way tactical head, plus the ego speed —
    the INPUT-only arm (D-TAC1 F1 in isolation), added 2026-08-03.

    WHY IT EXISTS. The pre-registered arm set was `dtac1-full` (F1+F2+F3),
    `dtac1-f2only` (F2) and `dtac1-nolon-graft`. F1's only estimate would then
    have been `full − f2only`, which is confounded: those two arms differ not
    only in whether the speed is read but in WHICH head reads it (a shared trunk
    with two 3-way readouts vs a 2-layer MLP with one 5-way readout). This arm
    changes the input and NOTHING else — same head, same label, same loss, same
    decode — so a delta against `refc-base` is attributable to the input alone.

    **MEASURED capacity delta** (``param_breakdown``, pinned by
    ``tests/test_refc_tactical.py::test_f1only_is_not_a_capacity_change``):
    104,191,577 -> 104,191,961 = **+384 parameters (+0.00037 %)**, exactly one
    extra input column into ``maneuver_head.0`` (``aux_hidden = 384``). That is
    less than half the factored arm's +897 and ~1/700 of the +272,001 the first
    two-MLP implementation cost.

    ⚠️ Note the training/eval asymmetry it inherits: ``ego_dropout = 0.5`` zeroes
    ``v`` on half the TRAINING samples while eval always supplies it. That is the
    existing, documented guard (the same one the measurement encoder lives with),
    not something this arm introduces — but it does mean the head is trained to
    work without the channel half the time, so ``ego_dropout`` is the first knob
    to sweep if F1 lands weaker than the E-A2 lower bound.
    """
    cfg = refc_config()
    cfg.tactical_speed_input = True  # F1 input, on the UNCHANGED 5-way head
    return cfg


def refc_select_config() -> RefCConfig:
    """REF-C-base with the D-SEL SELECTION SURFACE — the pre-registered arm
    (``Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md``).

    IDENTICAL to :func:`refc_config` everywhere except how a candidate is
    CHOSEN. The five levers and what each answers:

      S1 ``sel_refined``      rank the refined fan with the REFINED confidence.
                              Today the post-denoise trajectories are ranked by
                              the t=0 classifier score; MEASURED on **REF-C-XL**
                              (256 anchors, 881 windows) the pick is >2x worse
                              than the fan's best in 45.4 % of windows against
                              an oracle-in-fan of 0.1640 m — base's own figures
                              are 41.09 % / 0.1914 m. 0 parameters.
                              ⛔ NOT a headroom claim: the registry's standing
                              caveat is that the oracle gap is ~92 % irreducible
                              and REF-C v1.2's re-scorer was NOT separated.
      S2 ``sel_reach_clamp``  delete candidates a bounded-acceleration ego
                              cannot fly. MEASURED on ``fan_refc-xl-30k.pt``:
                              72.08 % removed, oracle survives 100 %, empty sets
                              0.00 %, paired delta **exactly 0.0** — i.e. INERT
                              on ADE by measurement. It is a PRECONDITION: it
                              makes per-candidate compute 3.58x cheaper, which
                              is what lets S3 run. 0 parameters.
      S3 ``graft_cons``       the CONSEQUENCE of each candidate, through
                              ``law_head``, reaches the ranking — the only form
                              of the flagship's ``cond_imagination`` REF-C's
                              world model can express, and the only one with a
                              real candidate axis. +1 parameter.
      S4 ``seam_clamp``       cap the TOTAL graft against the base score. The
                              trainer already logs ``graft_lat_norm`` /
                              ``graft_lon_norm`` / ``conf_norm``; this is the
                              missing actuator. 0 parameters.
      S5 ``graft_route``      the strategic route READOUT reaches the ranking
                              instead of only warping the condition. +384
                              parameters (3 x 128 anchors, zero-init).
                              ⚠️ THE LOWEST-PRIOR LEVER OF THE FIVE, on purpose.
                              MEASURED (R-2026-08-03-l): ``route_head(pooled)``
                              is nav-blind BY ARCHITECTURE
                              (``nav_passthrough_rate`` 0.0000, logit std across
                              navs exactly 0.0), its junction-scene accuracy
                              0.7613 sits BELOW that scene's majority-class
                              baseline 0.7806, and an image-free NAV_ECHO lookup
                              beats it by 0.1724. Grafting a weak readout onto
                              selection is expected to be NULL; it is included
                              because it is the only lever that TESTS the
                              readout pathway, and zero-init means training —
                              not taste — decides. The higher-prior route ->
                              selection pathway already exists and is LAN's
                              param-free geometric ``lan_gate`` (route coverage
                              0.8801 vs ``nav_cmd``'s 0.2724).

    **MEASURED capacity delta** (``param_breakdown``, pinned by
    ``tests/test_refc_select.py::test_dsel_is_not_a_capacity_change``):
    104,191,577 -> 104,191,962 = **+385 parameters (+0.00037 %)**.

    ⚠️ ``graft_route`` REQUIRES a future-derived route target (``--labels
    v21``/``v3``). Under ``--labels v1`` the route target is
    ``route_target(nav_cmd)`` — circular with a model INPUT — and grafting that
    readout onto selection would pipe the nav echo into the ranking. The trainer
    refuses the combination.

    ``ego_valid_channel`` is NOT in this preset: it changes the measurement
    encoder's INPUT, so it is an input lever, not a selection lever, and mixing
    it in would make the arm non-attributable. It has its own arm.

    ``graft_goal`` (S6) is NOT in this preset either, and for a stronger reason:
    it is **SEQUENCED BEHIND** the temporal-feature question, not independent of
    it. See :func:`refc_goal_config`.
    """
    cfg = refc_config()
    cfg.sel_refined = True
    cfg.sel_reach_clamp = True
    cfg.graft_cons = True
    cfg.graft_route = True
    cfg.seam_clamp = 1.0
    return cfg


def refc_goal_config() -> RefCConfig:
    """D-SEL + S6, the **PREDICTED GEOMETRIC GOAL** arm (PI ruling 2026-08-03).

    *"yes a goal input is admissible, at the same time, we need to be careful
    not to include the result of the situation classification in the goal
    input."* — Sayed, 2026-08-03.

    WHAT THE GOAL IS COMPUTED FROM is declared in machine-readable form by
    :meth:`RefCModel.goal_provenance`, written into every run's ``config.json``,
    and asserted by ``tests/test_refc_select.py``. In one line: **the image
    embedding of the last frame, and nothing else.** No situation-classifier
    output exists in this graph in any form.

    WHY GEOMETRIC AND PREDICTED. With proper no-navigation controls, a
    CATEGORICAL command buys ~nothing (TransFuser perturbed to
    None/Random/Left/Right: PDMS flat 84.0-84.7; no-nav -> command-only **+0.2**)
    while a GEOMETRIC goal buys a lot (route path + turn-by-turn **+2.3**;
    GoalFlow goal POINT **+4.7**). And a SUPPLIED route is optimistic by
    construction on PhysicalAI, whose only route supplier is the ego's own
    future path. So S6 predicts a bearing + an along-track preference from
    vision and feeds them to the SAME param-free geometric compatibility LAN
    uses. ``--graft-lan`` is required because the corridor is the head's
    TRAINING LABEL — never a model input for this seam.

    ⛔ **SEQUENCING, NOT A STANDALONE WIN.** A predicted goal must predict
    along-track distance from latents in which ``long_accel`` was MEASURED
    unrecoverable across 17 head architectures (K7). Our own prior therefore
    predicts PARTIAL FAILURE — the bearing half should work, the along-track
    half should not. The two gates are separate precisely so that prediction is
    MEASURED rather than assumed.

    ⛔ **CORRECTED 2026-08-03 — two claims in the previous version of this
    paragraph were wrong, and the second one could have killed this arm for
    free.**

    1. *"REF-C is structurally single-instant"* is **imprecise**. It keeps one
       feature **map**, but that map is computed from a D-015 **3-frame stack**
       (``in_channels=9``), so it already carries **~300 ms** of motion — not
       zero. Verified numerically: ``frames_u8[t][6:9] == frames_u8[t+1][3:6]``,
       max|d| = 0.0. Two independent streams corroborated it on different files
       and corpora. What REF-C discards is *history beyond that stack*, which is
       a much weaker statement than "single-instant".
    2. ⭐ *"conditional on the sibling temporal-feature stream"* is **RETRACTED**.
       The two do **not** share an input path — REF-C keeps ONE feature map
       (``forward``, this file), while the situation classifier stacks **eight**
       via ``sitclf.causal_window``. A null in that stream is therefore **not
       evidence about S6**, and registering S6 as contingent on it would have
       let an unrelated result cancel a lever that was never tested.
       **S6 needs its own evidence.** It is an independent lever whose
       along-track half carries a stated adverse prior — not a dependent one.

    ⇒ **RULE: an arm may only be registered as conditional on another result
    when the two share the mechanism, not merely the topic.** Check the input
    path before writing "conditional on".

    Capacity, MEASURED: ``goal_head`` = ``Linear(feat, 3)`` = 2,115 on base,
    plus two scalar gates.
    """
    cfg = refc_select_config()
    cfg.graft_route = False       # the categorical pathway is NOT mixed in: two
    #                               route-ish levers in one arm is unattributable
    # ⛔ ``graft_lan`` STAYS OFF. It is the SUPPLIED corridor as a MODEL INPUT —
    # exactly the thing the ruling says a goal must not be. The trainer mints the
    # `lan` batch field for S6's LABEL without building this input pathway, so
    # the arm reads a PREDICTED goal and nothing supplied. (An earlier draft of
    # this preset set it True and would have contaminated the arm with both.)
    cfg.graft_lan = False
    cfg.graft_goal = True
    return cfg


def refc_smoke_config() -> RefCConfig:
    """Tiny CPU config (CI smoke / tests / dry runs) — same structure, same
    horizons/path_dists, shrunk widths. Episodes: 1-channel 64 px (grid 2, 4
    cross-attention positions); 20-anchor vocabulary."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=1, image_size=64, base_width=8,
                                   blocks=(1, 1, 1, 1))
    cfg.window = 4
    cfg.measurement = MeasurementConfig(hidden=32, d_out=16)
    cfg.anchors = AnchorConfig(n_anchors=20, pool_size=256)
    cfg.decoder = DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                                aux_hidden=32, diffusion_steps=2, noise_std=0.1)
    cfg.law = LawConfig(hidden=32)
    cfg.strategic = StrategicCtxConfig(hidden=16, d_ctx=8)
    cfg.speed_hidden = 32
    cfg.tactical_latent_dim = 32
    return cfg


# ============================================================================
# Encoder (KEEP verbatim from TCP-C)
# ============================================================================

class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + BN, identity/1x1 shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.down: nn.Module | None = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x: Tensor) -> Tensor:
        idn = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + idn)


class ResNetEncoder(nn.Module):
    """Torchvision-free ResNet-34-style image encoder.

    Stem (7x7 /2 + maxpool /2) then four BasicBlock stages at widths
    (w, 2w, 4w, 8w), strides (1, 2, 2, 2) -> total stride 32. forward returns
    (fmap [B, 8w, g, g], pooled [B, 8w]) with g = image_size // 32 — for the
    full config the TCP-shaped [B, 512, 8, 8] + 512-d pooled vector.
    """

    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        # ⚠️ The stride-32 constraint is PER AXIS. Written for a square input it
        # checked ONE number, so a 256x640 frame would have passed while the
        # other axis was silently mis-sized (found 2026-07-27).
        _h, _w = cfg.image_hw()
        if _h % 32 != 0 or _w % 32 != 0:
            raise ValueError(f"image height and width must each be divisible "
                             f"by 32, got {_h}x{_w}")
        w = cfg.base_width
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        widths = (w, 2 * w, 4 * w, 8 * w)
        stages, c_in = [], w
        for s_i, (c_out, depth) in enumerate(zip(widths, cfg.blocks)):
            blocks = [BasicBlock(c_in, c_out, stride=1 if s_i == 0 else 2)]
            blocks += [BasicBlock(c_out, c_out) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.feat_dim = cfg.feat_dim
        self.grid_shape = cfg.grid_shape      # (rows, cols) — non-square-safe

    @property
    def grid(self) -> int:
        """Square feature-map side — raises on a non-square build ON PURPOSE
        (mirrors :attr:`CNNEncoderConfig.grid`; :attr:`grid_shape` is the
        general accessor). Until 2026-09-01 this was an EAGER ``__init__`` read
        (``self.grid = cfg.grid``), which made ANY non-square construction
        crash at build time even though every in-repo consumer of the map is
        shape-agnostic (``flatten(2)`` / ``grid_shape``) — MEASURED as the one
        blocker for the B1 256x640cyl corpus. Moving the raise to READ time
        keeps the square contract for scalar callers and unblocks non-square
        builds without touching any consumer."""
        gh, gw = self.grid_shape
        if gh != gw:
            raise ValueError(
                f"REF-C feature map is {gh}x{gw} (non-square) — this caller "
                f"still reads the scalar `grid`. Use `grid_shape`.")
        return gh

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x, x.mean(dim=(2, 3))                # fmap [B,F,g,g], pooled


class StrategicCtx(nn.Module):
    """Hierarchy graft: tiny GRU over the W pooled window features -> d_ctx
    token (KEEP). Now feeds the decoder CONDITION rather than the measurement."""

    def __init__(self, feat_dim: int, hidden: int, d_ctx: int):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_ctx)

    def forward(self, pooled_seq: Tensor) -> Tensor:   # [B, W, F] -> [B, d_ctx]
        _, h = self.gru(pooled_seq)
        return self.proj(h[-1])


# ============================================================================
# Anchored truncated-diffusion decoder
# ============================================================================

class FiLM(nn.Module):
    """Feature-wise linear modulation. ``zero_init`` starts it as identity (for
    the graft seams that must not perturb the base); the core condition FiLM in
    the decoder layers is LIVE (default init) so the measurement/nav/v0 condition
    steers the decoder from step 0 rather than waiting for the FiLM to train."""

    def __init__(self, cond_dim: int, d: int, zero_init: bool = True):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)
        if zero_init:
            nn.init.zeros_(self.to_scale_shift.weight)
            nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class CrossAttnLayer(nn.Module):
    """Anchor-query cross-attention block: cross-attend the conv map, then a
    FiLM(condition)-modulated MLP (pre-norm, residual)."""

    def __init__(self, d: int, n_heads: int, cond_dim: int, ff_mult: int,
                 cross_agent: bool = False):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.film = FiLM(cond_dim, d, zero_init=False)   # live core conditioning
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))
        # ⭐ refcv5 WP-6: the AGENT seam. Constructed ONLY when asked, and
        # AFTER every pre-v5 submodule above, so an off build draws exactly the
        # RNG it drew before this seam existed.
        # `agent_gate` is ZERO-INIT (the `ctx_to_cond` discipline): the agent
        # branch contributes exactly nothing at step 0, so a fresh +agents run
        # starts bit-identical to the agent-free one and every subsequent
        # change is attributable to the seam -- while the gradient
        # (agent_out * dL/dq) is non-zero, so it is GATED, not dead.
        self.cross_agent: nn.MultiheadAttention | None = None
        self.norm_a: nn.LayerNorm | None = None
        self.agent_gate: nn.Parameter | None = None
        if cross_agent:
            self.norm_a = nn.LayerNorm(d)
            self.cross_agent = nn.MultiheadAttention(d, n_heads,
                                                     batch_first=True)
            self.agent_gate = nn.Parameter(torch.zeros(1))

    def forward(self, q: Tensor, kv: Tensor, cond: Tensor,
                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None) -> Tensor:
        h = self.norm_q(q)
        q = q + self.cross(h, kv, kv, need_weights=False)[0]
        if self.cross_agent is not None and agent_tokens is not None:
            q = q + self.agent_gate * self._attend_agents(q, agent_tokens,
                                                          agent_pad)
        q = q + self.mlp(self.film(self.norm_f(q), cond.unsqueeze(1)))
        return q

    def _attend_agents(self, q: Tensor, tok: Tensor,
                       pad: Tensor | None) -> Tensor:
        """Cross-attend the agent tokens. ``pad[b, n] == True`` means slot ``n``
        is PADDING (torch's convention, and `AgentTokenEmbed`'s).

        ⛔⛔ A FULLY-PADDED ROW MAKES `MultiheadAttention` RETURN NaN, NOT
        ZERO -- softmax over an all-`-inf` row is undefined. A frame with no
        detected agent is not an edge case here, it is the *common* case on an
        empty road, so an unguarded mask would poison the whole batch through
        the residual and read as an exploding planner rather than as an empty
        scene. The row is therefore UN-masked (so the attention is well-defined)
        and its contribution is zeroed afterwards, which is the same answer the
        maths would give if softmax-over-nothing were defined as 0.
        """
        empty = None
        if pad is not None:
            empty = pad.all(dim=1)                          # [B]
            if bool(empty.any()):
                pad = pad & ~empty[:, None]
        out = self.cross_agent(self.norm_a(q), tok, tok,
                               key_padding_mask=pad, need_weights=False)[0]
        if empty is not None:
            out = out.masked_fill(empty[:, None, None], 0.0)
        return out


class AnchoredDiffusionDecoder(nn.Module):
    """DiffusionDrive-style anchored trajectory decoder (two modes, one weight
    set). See the module docstring for the classifier / truncated-diffusion
    split and the gated graft seams.

    forward -> {anchor_logits [B, N], anchor_traj [B, N, S, 2], offset (base)
    [B, N, S, 2], traj [B, S, 2] (selected), sel_idx [B]}.
    """

    def __init__(self, feat_dim: int, n_steps: int, d_meas: int, d_ctx: int,
                 tac_latent_dim: int, anchors: Tensor, cfg: DecoderConfig,
                 hierarchy: bool, graft_maneuver: bool,
                 graft_target_latent: bool, grounded_selector: bool,
                 n_maneuvers: int = N_MANEUVERS,
                 graft_lan: bool = False, d_lan: int = 0,
                 factored_maneuver: bool = False,
                 sel: "SelectionConfig | None" = None,
                 horizons: tuple[int, ...] = (),
                 v0_conditioned: bool = False,
                 ref_speed_ms: float = 10.0,
                 control_units: str = "kappa",
                 alat_v_floor_ms: float = 4.0,
                 kappa_cap: float = 0.12):
        super().__init__()
        self.cfg = cfg
        self.n_steps = n_steps
        self.grounded = grounded_selector
        # D-SEL: the selection-surface policy travels with the decoder, because
        # every one of its levers acts on the ranking this class performs.
        self.sel = sel or SelectionConfig()
        # Sustained-saturation counters: PLAIN attributes, never buffers — they
        # must not enter state_dict and change checkpoint compatibility.
        self._seam_conf = sl.SeamState()
        self._seam_refined = sl.SeamState()
        self._seam_rank = sl.SeamState()
        # Anchor vocabulary — a persistent buffer (travels with the checkpoint).
        self.register_buffer("anchors", anchors)              # [N, S, 2]
        # ⭐ refcv4-b: the v0-CONDITIONED vocabulary. `anchor_controls` [N, 2]
        # = (accel m/s^2, curvature 1/m) held constant over the horizon; the
        # emitted bank is rolled per window from that window's own v0. `anchors`
        # keeps holding the SAME family rolled at `ref_speed_ms`, so it stays
        # the checkpoint-visible artifact a content check can compare, and it is
        # the surface the two param-free geometric priors fall back to.
        self.anchor_v0_cond = bool(v0_conditioned)
        self.anchor_ref_speed = float(ref_speed_ms)
        self.anchor_dt = 0.1
        if control_units not in ("kappa", "alat"):
            raise ValueError(f"control_units {control_units!r} not in "
                             f"('kappa', 'alat')")
        self.anchor_control_units = control_units
        self.anchor_alat_v_floor = float(alat_v_floor_ms)
        self.anchor_kappa_cap = float(kappa_cap)
        self.register_buffer("anchor_controls",
                             torch.zeros(anchors.shape[0], 2), persistent=True)
        # slot index of each horizon inside a dt-tick rollout. `rollout_unicycle`
        # returns the state AFTER each step, so horizon k lands at index k - 1.
        _h = tuple(horizons) or tuple(range(1, n_steps + 1))
        if len(_h) != n_steps:
            raise ValueError(f"horizons {_h} does not match n_steps {n_steps}")
        self.register_buffer("anchor_slots",
                             torch.tensor([k - 1 for k in _h],
                                          dtype=torch.long), persistent=False)
        self.anchor_roll_steps = int(max(_h))
        # The horizons themselves, in TICKS, as a plain attribute: the WP-4
        # sampler rolls a per-slot control SEQUENCE and needs the slot spans,
        # which `anchor_slots` (k - 1) does not carry unambiguously.
        self.anchor_horizons: tuple[int, ...] = tuple(int(k) for k in _h)
        # ⭐ H-EGO-LIT-4: what speed a WITHHELD row's bank is rolled at. Plain
        # attributes, never buffers (no state_dict change, so every checkpoint
        # before and after this flag loads strictly). See WITHHELD_BANK_MODES.
        # Kept rows are untouched by every mode but "none"; at eval nothing
        # is withheld, so "fixed"/"pred"/"random" leave eval byte-identical.
        self.anchor_withheld_bank: str = "fixed"
        self.anchor_withheld_speed_max: float = 35.0
        self.anchor_random_speed_pool: Tensor | None = None
        d = cfg.d
        self.feat_proj = nn.Linear(feat_dim, d)               # conv map -> KV
        self.traj_proj = nn.Linear(n_steps * 2, d)            # traj estimate -> Q
        self.cond_proj = nn.Linear(d_meas, d)                 # measurement -> cond
        self.time_embed = nn.Embedding(cfg.diffusion_steps + 1, d)  # 0..steps
        self.layers = nn.ModuleList(
            CrossAttnLayer(d, cfg.n_heads, d, cfg.ff_mult,
                           cross_agent=bool(getattr(cfg, "cross_agent", False)))
            for _ in range(cfg.layers))
        self.conf_head = nn.Linear(d, 1)                      # per-anchor conf
        self.offset_head = nn.Linear(d, n_steps * 2)          # per-anchor offset
        # Graft: strategic ctx -> condition (zero-init identity start).
        self.ctx_to_cond: nn.Linear | None = None
        if hierarchy:
            self.ctx_to_cond = nn.Linear(d_ctx, d)
            nn.init.zeros_(self.ctx_to_cond.weight)
            nn.init.zeros_(self.ctx_to_cond.bias)
        # Graft (H19): maneuver logits reweight anchor priors. LIVE from step 0
        # (default Linear init) — the coupling is the point of the seam.
        #
        # D-TAC1: with ``factored_maneuver`` the ONE rank-5 graft becomes TWO
        # additive rank-3 grafts. Why two summed terms and not one Linear over
        # the concatenation: (i) the anchor vocabulary is 2-D — every anchor is a
        # trajectory with BOTH a lateral shape and a speed profile, and a
        # lateral-dominated prior can only ever re-rank the lateral axis, so
        # ``lon_to_anchor`` is the surface on which "we are stopping" can
        # suppress every anchor that keeps rolling; (ii) two separable terms are
        # ABLATABLE — ``lon_to_anchor`` can be zeroed and the delta measured,
        # which the single concatenated graft never allowed. ``lon_to_anchor``
        # is ZERO-INIT (the ctx_to_cond discipline) so the selection path starts
        # with exactly today's lateral-only prior and every subsequent change is
        # attributable to the longitudinal seam; ``lat_to_anchor`` keeps the
        # default init because it inherits the LIVE H19 role.
        self.maneuver_to_anchor: nn.Linear | None = None
        self.lat_to_anchor: nn.Linear | None = None
        self.lon_to_anchor: nn.Linear | None = None
        if graft_maneuver and factored_maneuver:
            self.lat_to_anchor = nn.Linear(N_LAT_MAN, anchors.shape[0],
                                           bias=False)
            self.lon_to_anchor = nn.Linear(N_LON_MAN, anchors.shape[0],
                                           bias=False)
            nn.init.zeros_(self.lon_to_anchor.weight)
        elif graft_maneuver:
            self.maneuver_to_anchor = nn.Linear(n_maneuvers, anchors.shape[0],
                                                bias=False)
        # Graft: FiLM the condition on a tactical goal latent (zero-init).
        self.tgt_proj: nn.Linear | None = None
        self.tgt_film: FiLM | None = None
        if graft_target_latent:
            self.tgt_proj = nn.Linear(tac_latent_dim, d)
            self.tgt_film = FiLM(d, d)
        # Graft (LAN): the route corridor enters on TWO surfaces.
        #   1. ``lan_to_cond`` — zero-init, so the condition is unperturbed at
        #      step 0 (the ctx_to_cond discipline).
        #   2. ``lan_gate`` — ONE scalar on a PARAM-FREE geometric compatibility
        #      between each anchor's terminal bearing and the route's bearing.
        #      Selection among the fan is where a route can act at all, and a
        #      geometric score cannot become a route-shaped shortcut the way a
        #      learned ``route -> n_anchors`` matrix could. Initialised to 0, so
        #      the anchor priors are bit-unchanged at step 0 while the gradient
        #      (compat * dL/dconf) is non-zero — gated, not dead.
        self.lan_to_cond: nn.Linear | None = None
        self.lan_gate: nn.Parameter | None = None
        if graft_lan:
            self.lan_to_cond = nn.Linear(d_lan, d)
            nn.init.zeros_(self.lan_to_cond.weight)
            nn.init.zeros_(self.lan_to_cond.bias)
            self.lan_gate = nn.Parameter(torch.zeros(1))
        # --- D-SEL grafts on the RANKED score (gated; absent when off) -------
        # S5: the STRATEGIC route readout reaches SELECTION. Today the route
        # reaches the decoder only through the CONDITION (nav_cmd -> measurement,
        # LAN -> cond), i.e. it can WARP every candidate but never CHOOSE among
        # them — and MEASURED (LAN E0), feeding the ORACLE route makes the
        # cross-track separation WORSE, which is what a warp-only pathway looks
        # like. Same shape and same zero-init discipline as v4's lat/lon/dist
        # grafts and as ``lon_to_anchor``: bias-free, so a constant offset in the
        # log-posterior is absorbable, and zero at step 0 so the ranked score
        # starts bit-identical to the graft-free baseline.
        self.route_to_anchor: nn.Linear | None = None
        if self.sel.graft_route:
            self.route_to_anchor = nn.Linear(N_ROUTE, anchors.shape[0],
                                             bias=False)
            nn.init.zeros_(self.route_to_anchor.weight)
        # S3: ONE scalar on the candidate-conditioned consequence score. The
        # score itself costs zero parameters (refc_select.consequence_scores
        # reuses feat_proj + conf_head + a param-free layer_norm), so the whole
        # cond_imagination port is +1 parameter.
        self.cons_gate: nn.Parameter | None = None
        if self.sel.graft_cons:
            self.cons_gate = nn.Parameter(torch.zeros(1))
        # S6: the PREDICTED GEOMETRIC goal, on TWO INDEPENDENT zero-init gates.
        #
        # ⭐ THE SPLIT IS THE INSTRUMENT, not tidiness. The PI's ruling carries a
        # caveat that binds this lever: a predicted goal must predict ALONG-TRACK
        # distance from latents in which `long_accel` was measured UNRECOVERABLE
        # across 17 head architectures (K7), and REF-C is structurally
        # single-instant — `RefCModel.forward` cross-attends the LAST frame's
        # feature map only. The BEARING half is lateral topology and is
        # recoverable from one frame; the ALONG-TRACK half is exactly the half
        # our own prior says should fail until temporal features land.
        # Two gates therefore let the arm MEASURE that prediction instead of
        # being defeated by it: `goal_dist_gate` staying at ~0 while
        # `goal_gate` opens IS the K7 result, read off a training run, and it is
        # the cheapest available read on whether the sibling temporal-feature
        # stream is a precondition for goal conditioning.
        self.goal_gate: nn.Parameter | None = None
        self.goal_dist_gate: nn.Parameter | None = None
        if self.sel.graft_goal:
            self.goal_gate = nn.Parameter(torch.zeros(1))
            self.goal_dist_gate = nn.Parameter(torch.zeros(1))
        # ⭐⭐ S7 / E15 (GP-1, 2026-09-06) — THE METRIC GOAL POINT ON THE RANKED
        # SCORE. ONE zero-init scalar, so the ranked score is BIT-IDENTICAL to
        # the goal-free baseline at step 0 and the exact ablation is "set the
        # gate to 0". The score it multiplies costs ZERO parameters
        # (`goal_point.anchor_goal_prior_at_time` is param-free), so the whole
        # seam is +1 parameter — the same discipline as `cons_gate`.
        #
        # ⛔ THE SLOT IS REFUSED AT BUILD, NOT CLAMPED AT RUN. Comparing the
        # goal at t = 4 s against the anchor at some other slot is a
        # plausible-looking WRONG experiment that trains and converges — the
        # derived-constant trap (`HORIZON = round(6.0*10/STRIDE)`) in geometry
        # costume. `goal_slot_index` derives the slot from the model horizons;
        # this asserts the decoder it landed on actually HAS that slot.
        self.gp_point_gate: nn.Parameter | None = None
        self.gp_slot = int(self.sel.gp_slot)
        self.gp_scale_m = float(self.sel.gp_scale_m)
        if self.sel.graft_gp_point:
            if not (0 <= self.gp_slot < self.n_steps):
                raise ValueError(
                    f"S7/E15: gp_slot {self.gp_slot} is not a slot of this "
                    f"decoder's {self.n_steps}-step bank. The goal point and "
                    f"the anchor must be compared AT THE SAME TIME; a slot "
                    f"outside the bank means the seam would either crash or "
                    f"(clamped) silently score a different horizon. Derive it "
                    f"with goal_point.goal_slot_index(t_goal_s, horizons).")
            if self.gp_scale_m <= 0.0:
                raise ValueError(
                    f"S7/E15: gp_scale_m must be positive, got "
                    f"{self.gp_scale_m} — it is GoalPointConfig.range_norm_m "
                    f"and it is what makes the score scale-free.")
            self.gp_point_gate = nn.Parameter(torch.zeros(1))
        # ---- refcv5 WP-4: THE DENOISER ------------------------------------ #
        #
        # ⭐ THE MECHANISM, WHICH IS WHAT refcv3 DID NOT HAVE. refcv3 carried
        # the truncated-diffusion SKELETON -- a `noise_std` perturbation and a
        # "ranking" that read the t = 0 confidence -- and MEASURED, that
        # ranking was UNCHANGED ON 201/201 WINDOWS. A loop that cannot move its
        # own output is not a sampler. What is added here is the four things
        # that were missing and nothing else: a NOISE SCHEDULE, an ANCHORED
        # GAUSSIAN over the vocabulary, DDIM SAMPLING, and a real DENOISING
        # LOOP whose prediction leaves the decoder as `u0_hat` so a loss can
        # reach it.
        #
        # ⛔ CONSTRUCTED LAST, AND ONLY WHEN ASKED. `sampler == "none"` builds
        # nothing at all, so the RNG draw order is byte-identical to refcv4b
        # and every pre-v5 checkpoint keeps loading strictly.
        #
        # Only TWO new modules exist, because the sampler reuses the decoder it
        # is wired into: `traj_proj` -> `layers` -> `conf_head` are the same
        # weights the classifier pass uses, so the sampler cannot drift into a
        # second, differently-conditioned decoder.
        #   * `control_head` -- d -> S*2, the predicted CLEAN control x0.
        #     ZERO-INIT, so the first refinement pass is exactly the identity
        #     and the emitted fan starts AT the anchored Gaussian. That is what
        #     makes "the vocabulary is the prior" true rather than intended.
        #   * `time_mlp` -- DD's sinusoidal embedding of the CONTINUOUS t. The
        #     pre-v5 `time_embed` is a 3-row `nn.Embedding` and CANNOT represent
        #     t ~ U[0, 50); using it would quietly collapse the training
        #     timestep to one of three rows.
        self.control_head: nn.Linear | None = None
        self.time_mlp: nn.Module | None = None
        self.sched: "rs.DDIMSchedule | None" = None
        sampler = str(getattr(cfg, "sampler", "none"))
        if sampler not in ("none", "ddim"):
            raise ValueError(f"cfg.sampler {sampler!r} not in ('none', 'ddim')")
        space = str(getattr(cfg, "sampler_space", "control"))
        if space not in ("control", "metre"):
            raise ValueError(f"cfg.sampler_space {space!r} not in "
                             f"('control', 'metre')")
        if sampler == "ddim":
            self.control_head = nn.Linear(d, n_steps * 2)
            nn.init.zeros_(self.control_head.weight)
            nn.init.zeros_(self.control_head.bias)
            self.time_mlp = rs.build_time_mlp(d)
            # A PLAIN attribute, never a buffer or a module: the alpha table is
            # a constant of the published schedule, not a learned tensor, and
            # putting it in `state_dict` would change checkpoint compatibility
            # for a quantity no run can alter.
            self.sched = rs.DDIMSchedule()

    def anchor_control_seq(self, batch: int, dtype: torch.dtype) -> Tensor:
        """``[B, N, S, 2]`` — the anchor vocabulary as a per-slot control
        SEQUENCE, which is the state the sampler denoises.

        ⭐ The vocabulary holds ONE ``(a_lon, a_lat)`` pair per anchor; the
        sampler's state is a sequence of ``S``. This expansion is the bridge,
        and it is the reason a CONSTANT sequence must roll to exactly
        :meth:`roll_bank`'s bank (pinned in ``test_refc_sampler.py``): at
        ``sigma -> 0`` the sequence IS the constant pair, so if the two
        integrators disagreed, every anchored-Gaussian claim would be measured
        against a fan the vocabulary never emitted.
        """
        n = self.anchors.shape[0]
        return (self.anchor_controls.to(dtype)[None, :, None, :]
                .expand(batch, n, self.n_steps, 2))

    def load_anchors(self, anchors: Tensor,
                     controls: Tensor | None = None) -> None:
        """Install an externally-built anchor vocabulary (build_refc_anchors.py).
        Shape must match [N, n_steps, 2] of the constructed decoder.

        ``controls`` [N, 2] = the (accel, curvature) the paths were rolled from.
        Required when the decoder is ``v0_conditioned`` — without it the bank
        cannot be re-rolled at the window's speed, and a silent fallback to the
        fixed paths would be exactly the defect this vocabulary exists to fix,
        so it RAISES instead.
        """
        if tuple(anchors.shape) != tuple(self.anchors.shape):
            raise ValueError(f"anchor shape {tuple(anchors.shape)} != decoder "
                             f"{tuple(self.anchors.shape)}")
        self.anchors.copy_(anchors.to(self.anchors.dtype))
        if controls is not None:
            if tuple(controls.shape) != tuple(self.anchor_controls.shape):
                raise ValueError(
                    f"anchor controls {tuple(controls.shape)} != decoder "
                    f"{tuple(self.anchor_controls.shape)}")
            self.anchor_controls.copy_(controls.to(self.anchor_controls.dtype))
        elif self.anchor_v0_cond:
            raise ValueError(
                "v0_conditioned decoder loaded an anchor file with no "
                "`controls` [N, 2]. The bank is rolled per window from those "
                "controls; falling back to the fixed paths would silently "
                "restore the un-conditioned vocabulary this build exists to "
                "replace.")

    def _withheld_ref_speed(self, v: Tensor,
                            withheld_speed: Tensor | None) -> Tensor:
        """[B] the speed a WITHHELD row is rolled at, per ``anchor_withheld_bank``.

        Under ``"fixed"`` this is exactly the pre-flag expression
        ``torch.full_like(v, self.anchor_ref_speed)`` (pinned bit-identical by
        ``tests/test_withheld_bank.py``). ``"pred"`` without a supplied speed
        (warm-up, or a caller that has no goal head) falls back to the fixed
        roll — the trainer logs which one was live on every step.
        ``"random"`` without a pool is REFUSED: a silent fall-back would turn
        the blindness control into a second copy of the fixed arm.
        """
        mode = self.anchor_withheld_bank
        if mode not in WITHHELD_BANK_MODES:
            raise ValueError(f"anchor_withheld_bank {mode!r} not in "
                             f"{WITHHELD_BANK_MODES}")
        if mode == "pred" and withheld_speed is not None:
            # ⛔ DETACHED: the bank is geometry the anchor target is measured
            # against; a gradient from the target into the speed head would
            # let selection train the goal toward the fan (the SEL-1 refusal).
            return withheld_speed.detach().reshape(-1).to(v.dtype).clamp(
                0.0, self.anchor_withheld_speed_max)
        if mode == "random":
            pool = self.anchor_random_speed_pool
            if pool is None or pool.numel() == 0:
                raise ValueError(
                    "anchor_withheld_bank='random' needs a non-empty "
                    "anchor_random_speed_pool (the training marginal of v0); "
                    "refusing to fall back to the fixed roll silently")
            idx = torch.randint(pool.numel(), (v.shape[0],),
                                device=pool.device)
            return pool.reshape(-1)[idx].to(device=v.device, dtype=v.dtype)
        return torch.full_like(v, self.anchor_ref_speed)

    def roll_bank(self, v_ms: Tensor | None, ego_keep: Tensor | None,
                  batch: int, dtype: torch.dtype,
                  withheld_speed: Tensor | None = None) -> Tensor:
        """[B, N, S, 2] — the anchor bank THIS forward decodes.

        ``withheld_speed`` [B] (optional) is the model's OWN predicted speed,
        read only under ``anchor_withheld_bank == "pred"`` and only on rows
        whose ego channel was withheld (H-EGO-LIT-4).

        Fixed builds expand the stored paths, which is byte-identical to the
        pre-2026-09-04 ``anchors[None].expand(...)``. v0-conditioned builds roll
        ``anchor_controls`` through the programme's OWN integrator
        (:func:`tanitad.models.kinematic.rollout_unicycle`, the one
        ``refa_v1_plan.unicycle_paths`` calls with ``action_units="kappa"``)
        from each window's measured speed.

        ⛔ ``ego_keep`` binds here for the same reason it binds on the S2 band,
        only harder: ``v_ms`` is the PRE-dropout speed, and rolling the bank
        from it on a withheld row would put the withheld channel into the
        candidate GEOMETRY. Withheld rows are rolled at ``ref_speed_ms``, so the
        dropout regime is genuinely speed-blind.
        """
        n = self.anchors.shape[0]
        if not self.anchor_v0_cond:
            return self.anchors.to(dtype)[None].expand(
                batch, n, self.n_steps, 2)
        if self.anchor_withheld_bank not in WITHHELD_BANK_MODES:
            raise ValueError(f"anchor_withheld_bank "
                             f"{self.anchor_withheld_bank!r} not in "
                             f"{WITHHELD_BANK_MODES}")
        if v_ms is None or self.anchor_withheld_bank == "none":
            # "none": the speed-blind vocabulary — every row, kept or withheld,
            # train or eval, at the reference speed (the field's design).
            v = self.anchors.new_full((batch,), self.anchor_ref_speed)
        else:
            v = v_ms.reshape(-1).to(torch.float32)
            if ego_keep is not None:
                v = torch.where(ego_keep.reshape(-1),
                                v, self._withheld_ref_speed(v, withheld_speed))
        # ⚠️ rolled in float32 regardless of the AMP dtype: 60 sequential
        # integration steps in fp16 accumulate visible drift, and the bank is
        # the geometry every anchor target is measured against.
        h = self.anchor_roll_steps
        ctrl = self.anchor_controls.to(torch.float32)
        if self.anchor_control_units == "alat":
            # kappa = a_lat / v^2, clamped. The floor keeps a standing-start
            # window from asking for an infinite curvature, and the cap is a
            # geometric bound (~8.3 m radius) that no road manoeuvre needs.
            vv = v.clamp_min(self.anchor_alat_v_floor) ** 2          # [B]
            kap = (ctrl[None, :, 1] / vv[:, None]).clamp(
                -self.anchor_kappa_cap, self.anchor_kappa_cap)       # [B, N]
            ctrl = torch.stack(
                [ctrl[None, :, 0].expand(batch, n), kap], dim=-1)    # [B, N, 2]
            ctrl = ctrl[:, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)
        else:
            ctrl = ctrl[None, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)
        state0 = torch.zeros(batch * n, 4, device=ctrl.device,
                             dtype=torch.float32)
        state0[:, 3] = v[:, None].expand(batch, n).reshape(-1)
        path = rollout_unicycle(state0, ctrl, dt=self.anchor_dt)[..., :2]
        return path[:, self.anchor_slots].reshape(
            batch, n, self.n_steps, 2).to(dtype)

    def _feasible(self, x: Tensor, v_ms: Tensor | None) -> Tensor:
        """Project the emitted fan onto the friction-feasible set. No-op when
        ``cfg.feasible_decode`` is False, and then it returns the SAME OBJECT.

        ⛔ **The prefix dt is DERIVED from ``anchor_slots``, never assumed.** The
        scorer differentiates the 2 s prefix on a uniform 0.5 s grid; our horizons
        are ``[5, 10, 15, 20, 30, 40, 50, 60]`` ticks, i.e. uniform 0.5 s for the
        first four and 1.0 s after. A projection run at the wrong dt is
        *approximately* feasible and reports a residual that looks like noise --
        the `df` / `step_s` family in a geometry costume. If the prefix is not
        uniform this refuses rather than guessing.

        ⚠️ **SCOPE, STATED: only the uniform prefix is projected.** The 2-6 s tail
        is rigidly TRANSLATED so the path stays C0-continuous at the seam and the
        tail's own kinematics are unchanged. Velocity continuity across the seam
        is NOT enforced, and projecting the tail on its own 1 s grid is a named
        work item, not a silent omission.
        """
        cfg = self.cfg
        if not bool(getattr(cfg, "feasible_decode", False)):
            return x
        sl = self.anchor_slots.tolist()
        k = int(getattr(cfg, "feasible_prefix_slots", 4))
        k = min(k, len(sl))
        steps = [sl[0] + 1] + [sl[i] - sl[i - 1] for i in range(1, k)]
        if len(set(steps)) != 1:
            raise ValueError(
                f"feasible_decode needs a UNIFORM prefix grid; anchor_slots "
                f"{sl[:k]} give tick spacings {steps}. Refusing rather than "
                f"projecting at a dt the scorer does not use.")
        dt = float(steps[0]) * float(self.anchor_dt)
        pre = _feas.project_feasible(
            _feas_with_origin(x[..., :k, :].float()),
            None if v_ms is None else v_ms.float(),
            dt=dt, a_max=float(cfg.feasible_a_max),
            kappa_max=float(cfg.feasible_kappa_max),
            mu=(float(cfg.feasible_mu) if float(cfg.feasible_mu) > 0 else None),
            clamp_entry=bool(cfg.feasible_entry) and v_ms is not None,
        )[..., 1:, :].to(x.dtype)
        if x.shape[-2] == k:
            return pre
        shift = (pre[..., k - 1, :] - x[..., k - 1, :]).unsqueeze(-2)
        return torch.cat([pre, x[..., k:, :] + shift], dim=-2)

    def _decode(self, kv: Tensor, cond: Tensor, x_est: Tensor,
                t_idx: int, agents: Tensor | None = None,
                agent_pad: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """One decoder pass: current trajectory estimate + timestep -> queries;
        cross-attend the map; emit (conf [B, N], offset [B, N, S, 2]).

        ``agents`` / ``agent_pad`` (refcv5 WP-6) are forwarded to every
        layer. They are inert unless the layer was BUILT with ``cross_agent``,
        so passing them on an agent-free build is a no-op rather than a branch.
        """
        b, n = x_est.shape[:2]
        q = self.traj_proj(x_est.reshape(b, n, -1))           # [B, N, d]
        q = q + self.time_embed.weight[t_idx][None, None]     # timestep bias
        for layer in self.layers:
            q = layer(q, kv, cond, agents, agent_pad)
        conf = self.conf_head(q).squeeze(-1)                  # [B, N]
        offset = self.offset_head(q).reshape(b, n, self.n_steps, 2)
        return conf, offset

    # ---- refcv5 WP-4: the sampler's own pass and its denoising loop -------- #
    def _decode_ctrl(self, kv: Tensor, cond: Tensor, x_path: Tensor,
                     t: Tensor, agents: Tensor | None,
                     agent_pad: Tensor | None) -> tuple[Tensor, Tensor]:
        """One SAMPLER pass -> (conf [B, N], du [B, N, S, 2]).

        ⭐ It reuses ``traj_proj`` -> ``layers`` -> ``conf_head``, the SAME
        weights the classifier pass uses. A second, separately-conditioned
        decoder is how the sampler and the ranker would silently come to
        disagree about what they are looking at; the only new parameters in the
        whole seam are ``control_head`` and ``time_mlp``.

        ⛔ The timestep enters through ``time_mlp`` (DD's sinusoidal embedding
        of the CONTINUOUS ``t``), NOT through ``time_embed``. ``time_embed`` is
        an ``nn.Embedding`` with ``diffusion_steps + 1 == 3`` rows and cannot
        represent ``t ~ U[0, 50)`` -- it would quietly collapse the training
        timestep onto one of three rows, and the schedule would stop meaning
        anything.
        """
        b, n = x_path.shape[:2]
        q = self.traj_proj(x_path.reshape(b, n, -1))          # [B, N, d]
        te = self.time_mlp(t.reshape(-1).to(torch.float32)).to(q.dtype)
        q = q + te.reshape(-1, 1, q.shape[-1])                # per-sample t
        for layer in self.layers:
            q = layer(q, kv, cond, agents, agent_pad)
        conf = self.conf_head(q).squeeze(-1)                  # [B, N]
        du = self.control_head(q).reshape(b, n, self.n_steps, 2)
        return conf, du

    def _state_to_path(self, state: Tensor, v: Tensor,
                       metre: bool) -> Tensor:
        """The sampler's state -> the emitted fan ``[B, N, S, 2]`` in metres.

        In CONTROL space the state is ``(a_lon, a_lat|kappa)`` per slot and the
        path is INTEGRATED, so every sample is flyable by construction. In the
        ``metre`` regression the state already IS the path -- which is exactly
        why that arm is pre-registered to fail the flyability gate.
        """
        if metre:
            return state
        return rs.roll_controls(
            state, v, self.anchor_horizons,
            control_units=self.anchor_control_units, tick=self.anchor_dt,
            alat_v_floor=self.anchor_alat_v_floor,
            kappa_cap=self.anchor_kappa_cap)

    def _sample(self, kv: Tensor, cond: Tensor, bank: Tensor,
                v_ms: Tensor | None, steps: int,
                agents: Tensor | None = None,
                agent_pad: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """The truncated-diffusion sampler. -> (fan, u0_hat, conf, telemetry).

        ⭐⭐ THIS IS THE MECHANISM refcv3 DID NOT HAVE, and the four pieces are
        each here on purpose:

        1. **the anchored Gaussian** -- the state starts at the VOCABULARY's own
           control sequence, noised once at ``t = sampler_infer_t``. The prior
           is the anchor set, not a standard normal, so a sample is a
           perturbation of something the vocabulary can already express;
        2. **the noise schedule** -- DD's ``scaled_linear`` table, so
           ``sigma(8) = 0.0316`` means here what it means in the paper;
        3. **DDIM sampling** -- deterministic (``eta = 0``) steps down the
           published ``[10, 0]`` ladder;
        4. **a real denoising loop** -- each pass PREDICTS THE CLEAN CONTROL and
           the state moves toward it. refcv3's loop read the ``t = 0``
           confidence and was MEASURED UNCHANGED ON 201/201 WINDOWS; a loop
           whose output cannot move is not a sampler, and this one's output is
           ``u0_hat``, the tensor the x0 loss is computed on.
        """
        cfg = self.cfg
        b, n = bank.shape[:2]
        dev, dtype = bank.device, bank.dtype
        metre = str(cfg.sampler_space) == "metre"
        if metre:
            # DERIVED, never the raw sigma -- see `metre_sigma_m`. Dividing by
            # the sigma itself would make the DD-literal arm 31.7x too gentle
            # and it would PASS the gate it exists to fail.
            _s = float(self.sched.sqrt_one_minus_abar(int(cfg.sampler_infer_t)))
            norm = bank.new_tensor(tuple(cfg.metre_sigma_m)) / max(_s, 1e-12)
            x0_n = bank / norm
        else:
            norm = bank.new_tensor(tuple(cfg.control_norm))
            x0_n = self.anchor_control_seq(b, dtype) / norm
        # The speed the fan is rolled from. `roll_bank` has already applied the
        # ego-dropout policy to produce `bank`; a sampler with no speed at all
        # falls back to the SAME reference speed `roll_bank` uses, so the two
        # never integrate from different initial states.
        v = (bank.new_full((b,), self.anchor_ref_speed)
             if v_ms is None else v_ms.reshape(-1).to(torch.float32))
        t0 = int(cfg.sampler_infer_t)
        k = int(steps) if int(steps) > 0 else int(cfg.sampler_steps)
        k = max(k, 1)
        ladder = self.sched.infer_timesteps(t0, k)
        # ⭐ THE ANCHORED GAUSSIAN: ONE fresh eps at the truncation point.
        #
        # ⛔⛔ THIS IS STOCHASTIC AT EVAL, BY DESIGN AND NOT BY OVERSIGHT, and
        # any arm that runs it inherits an obligation. The pre-v5 refinement
        # loop zeroes its noise outside training so a decode is reproducible;
        # a SAMPLER cannot, because sampling is the mechanism. MEASURED on
        # refav1 (`D-REFAV1-SEED-GOAL-MISMATCH`): on a planner that samples,
        # the same checkpoint evaluated twice does not give the same answer,
        # and that rig's inference-seed floor was ~0.30 m ADE. ⇒ A refcv5
        # sampler arm must be replicated over INFERENCE seeds, and an effect
        # smaller than that floor is not an effect. The episode-cluster
        # bootstrap answers "would another draw of EPISODES say this?" and is
        # structurally blind to this variance.
        self.sched.to(dev)          # the alpha table must live where x0 does
        eps = torch.randn_like(x0_n)
        x_n = self.sched.add_noise(x0_n, eps,
                                   torch.tensor(t0, device=dev))
        x0_hat_n, conf = x_n, None
        for i, t in enumerate(ladder):
            t_prev = ladder[i + 1] if i + 1 < len(ladder) else 0
            x_path = self._state_to_path(x_n * norm, v, metre)
            tt = torch.full((b,), float(t), device=dev, dtype=torch.float32)
            conf, du = self._decode_ctrl(kv, cond, x_path, tt,
                                         agents, agent_pad)
            # x0-parameterised ("sample"), and `control_head` is ZERO-INIT, so
            # the first pass predicts exactly the current state: the fan starts
            # AT the anchored Gaussian and every later movement is learned.
            x0_hat_n = x_n + du
            x_n = self.sched.step(x0_hat_n, x_n,
                                  torch.tensor(t, device=dev),
                                  torch.tensor(t_prev, device=dev))
        u0_hat = x0_hat_n * norm
        fan = self._state_to_path(u0_hat, v, metre)
        tele = {"sampler": "ddim", "sampler_space": cfg.sampler_space,
                "sampler_ladder": [int(x) for x in ladder],
                "sampler_infer_t": t0,
                # ⚠ MEASURED 2026-09-06 and it is a CONFIG decision, not a
                # bug: `SelectionConfig.refined` defaults to FALSE, so the
                # ranked score is the CLASSIFIER surface -- which the sampler
                # deliberately does not touch (pinned by
                # `test_hfov_style_sanity...`). With this False the fan is
                # SAMPLED but RANKED by a head that never saw the sample, which
                # is the S1 defect one level up (MEASURED there as a 45.4 %
                # -of-windows ranking failure). It is stamped rather than
                # refused because "improve the geometry, keep the ranking" is a
                # legitimate arm -- but an arm that wants the sampler to reach
                # SELECTION must run `--sel-refined`, and this key is how a
                # reader tells which one they are looking at.
                "sampler_ranks_the_fan": bool(self.sel.refined)}
        return fan, u0_hat, conf, tele

    def _lan_anchor_prior(self, lan_dir: Tensor,
                          bank: Tensor | None = None) -> Tensor:
        """Param-free geometric route compatibility of every anchor. [B, N].

        ``lan_dir`` [B, 3] = (cos, sin, valid) of the route's bearing at its
        first admissible arc-length anchor — i.e. WHICH WAY the route goes from
        here. Each trajectory anchor is scored by ``cos(phi_anchor - theta_
        route)`` on its TERMINAL bearing, so the score is purely a lateral-
        topology agreement and carries no along-track information (the axis
        GOAL_INPUT.md measured at +83.7 % and which a route input must never
        supply). Invalid routes score 0 for every anchor.
        """
        if bank is None:                    # fixed vocabulary — legacy path,
            a = self.anchors.to(lan_dir.dtype)  # bit-identical to pre-refcv4-b
            end = a[:, -1]                                    # [N, 2]
            r = torch.linalg.vector_norm(end, dim=-1).clamp_min(1e-6)
            cos_a, sin_a = end[:, 0] / r, end[:, 1] / r       # [N]
            compat = (cos_a[None] * lan_dir[:, 0:1]
                      + sin_a[None] * lan_dir[:, 1:2])        # [B, N]
            return compat * lan_dir[:, 2:3]
        end = bank.to(lan_dir.dtype)[:, :, -1]                # [B, N, 2]
        r = torch.linalg.vector_norm(end, dim=-1).clamp_min(1e-6)
        cos_a, sin_a = end[..., 0] / r, end[..., 1] / r       # [B, N]
        compat = cos_a * lan_dir[:, 0:1] + sin_a * lan_dir[:, 1:2]
        return compat * lan_dir[:, 2:3]

    def _apply_grafts(self, base: Tensor, terms: list[Tensor],
                      state, surface: str, patience: int) -> tuple[Tensor, dict]:
        """``base`` + the graft terms, optionally norm-capped as a group (S4).

        ⚠️ THE UNCLAMPED PATH ACCUMULATES IN THE LEGACY ORDER, ON PURPOSE.
        Floating-point addition is not associative: pre-D-SEL the decoder wrote
        ``conf = conf + man; conf = conf + lat; ...``, and folding that into
        ``conf + (man + lat + ...)`` would change the last bits of every published
        REF-C decode. So with ``seam_clamp <= 0`` this reproduces the original
        left-fold exactly, and the summed form is built ONLY when the clamp is
        on — where the group norm is the quantity being controlled anyway.
        """
        out = base
        for t in terms:
            out = out + t                       # EXACT pre-D-SEL accumulation
        if self.sel.seam_clamp <= 0.0 or not terms:
            return out, {}
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return sl.apply_seam_clamp(
            base, total, clamp=self.sel.seam_clamp, fail=self.sel.seam_fail,
            fail_frac=self.sel.seam_fail_frac, patience=patience,
            state=state, surface=surface)

    def _goal_along_prior(self, dist_pref: Tensor,
                          bank: Tensor | None = None) -> Tensor:
        """Along-track compatibility of every anchor with a PREDICTED goal
        distance. ``dist_pref`` [B] in [-1, 1] -> [B, N].

        Each anchor's terminal FORWARD displacement, z-scored across the fan (so
        the term is scale-free and cannot become a second confidence), times a
        signed preference: +1 = "the goal is far, prefer plans that cover
        ground", -1 = "the goal is near, prefer plans that do not".

        ⚠️ THIS IS THE AXIS A *SUPPLIED* ROUTE MAY NEVER TOUCH.
        :meth:`_lan_anchor_prior` is deliberately along-track-free, because a
        supplied route's along-track content is the ego's own future progress —
        i.e. its speed — and feeding that back as an input is a leak
        (GOAL_INPUT.md measured that axis at +83.7 %). The distinction that makes
        this admissible is PROVENANCE, not shape: ``dist_pref`` is PREDICTED from
        the image embedding at inference and never read from the label. The
        label is used only to train the head, which is the sanctioned direction
        ("LABELS MAY USE EGO; INFERENCE IS VISION-ONLY").
        """
        if bank is None:              # fixed vocabulary — legacy path, exact
            end_x = self.anchors.to(dist_pref.dtype)[:, -1, 0]       # [N]
            z = (end_x - end_x.mean()) / end_x.std().clamp_min(1e-6)  # [N]
            return z[None] * dist_pref.reshape(-1, 1)               # [B, N]
        end_x = bank.to(dist_pref.dtype)[:, :, -1, 0]                # [B, N]
        z = ((end_x - end_x.mean(dim=1, keepdim=True))
             / end_x.std(dim=1, keepdim=True).clamp_min(1e-6))       # [B, N]
        return z * dist_pref.reshape(-1, 1)                          # [B, N]

    @staticmethod
    def _grounded_score(x: Tensor) -> Tensor:
        """Param-free progress/collision proxy over decoded endpoints [B,N,S,2]:
        reward forward reach, penalise lateral excursion (no obstacle map yet)."""
        end = x[:, :, -1]                                     # [B, N, 2]
        return 0.1 * end[..., 0] - 0.1 * end[..., 1].abs()

    def forward(self, fmap: Tensor, m: Tensor, ctx: Tensor | None = None,
                maneuver_logits: Tensor | None = None,
                target_latent: Tensor | None = None,
                steps: int = 0, lan_emb: Tensor | None = None,
                lan_dir: Tensor | None = None,
                lat_prior: Tensor | None = None,
                lon_prior: Tensor | None = None,
                route_prior: Tensor | None = None,
                cons_head=None, cons_ctx: Tensor | None = None,
                v_ms: Tensor | None = None,
                ego_keep: Tensor | None = None,
                goal_dir: Tensor | None = None,
                goal_dist_pref: Tensor | None = None,
                gp_point: Tensor | None = None,
                gp_valid: Tensor | None = None,
                withheld_speed: Tensor | None = None,
                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None) -> dict:
        """D-SEL adds five OPTIONAL ranking inputs; with all flags off the
        emitted ``traj`` / ``sel_idx`` are bit-identical to pre-D-SEL REF-C.

        ``route_prior`` [B, N_ROUTE] log-probs (S5) · ``cons_head`` + ``cons_ctx``
        [B, F] the model's ``law_head`` and pooled latent (S3) · ``v_ms`` [B] the
        RAW ego speed in m/s and ``ego_keep`` [B] bool for the reachability band
        (S2).

        ⚠️ ``ego_keep`` is not decoration. ``v_ms`` is the speed BEFORE
        ego-dropout, and using it to filter candidates on a sample whose speed
        was withheld from the conditioning would leak the channel back in through
        the ranking — the exact failure ``flagship_v15``'s ``vt_keep`` masking
        exists to prevent ("a goal that was withheld from the decoder must not
        sneak back in through the ranking"). The band is therefore applied only
        where the speed was KEPT. At eval nothing is dropped, so the band is
        always active there — which is the regime the 72.08 % was measured in.
        """
        b = fmap.shape[0]
        sel = self.sel
        # ---- refcv5 WP-4 PREFLIGHT -- refuse before the compute, not after -- #
        # Both of these describe a run that would TRAIN, CONVERGE and MEAN
        # NOTHING, which is worse than a crash because a crash gets fixed.
        if self.control_head is not None:
            if not self.anchor_v0_cond:
                raise ValueError(
                    "refcv5 WP-4: the sampler needs a v0-CONDITIONED anchor "
                    "vocabulary. The sampler's state IS the control sequence, "
                    "and a fixed-path bank carries `anchor_controls` of all "
                    "zeros -- so the anchored Gaussian would be centred on 'do "
                    "nothing' and the arm would be a plausible-looking WRONG "
                    "experiment. Build the anchors with controls "
                    "(`--anchor-file` from build_refc_anchors), or run "
                    "`sampler='none'`.")
            if int(getattr(self.cfg, "sampler_groups", 1)) > 1:
                raise NotImplementedError(
                    "refcv5 WP-4: `sampler_groups > 1` emits a [B, G*N, ...] "
                    "fan while THREE call sites still assume N and would be "
                    "silently MIS-INDEXED: (1) `loss_cls`'s `a_star` anchor "
                    "target, which indexes the fan by anchor id; (2) the "
                    "[B, N] anchor priors (`maneuver_to_anchor` / "
                    "`lat_to_anchor` / `lon_to_anchor` / `route_to_anchor`); "
                    "and (3) every `sel_idx` dump the eval harness joins on. "
                    "Widening the fan is not a knob until those three are "
                    "widened with it -- refusing rather than mis-indexing.")
        kv = self.feat_proj(fmap.flatten(2).transpose(1, 2))  # [B, P, d]
        cond = self.cond_proj(m)                              # [B, d]
        if self.ctx_to_cond is not None and ctx is not None:
            cond = cond + self.ctx_to_cond(ctx)
        if self.lan_to_cond is not None and lan_emb is not None:
            cond = cond + self.lan_to_cond(lan_emb)           # LAN (zero-init)
        if self.tgt_film is not None and target_latent is not None:
            cond = self.tgt_film(cond, self.tgt_proj(target_latent))

        anchors = self.anchors.to(fmap.dtype)                 # [N, S, 2]
        n = anchors.shape[0]
        # ⭐ refcv4-b: [B, N, S, 2]. For a fixed vocabulary this is exactly the
        # old `anchors[None].expand(...)`; for a v0-conditioned one it is the
        # per-window roll. `bank` — not `anchors` — is the geometry every
        # consumer below must read, including the anchor target in the trainer,
        # which is why it is also returned.
        bank = self.roll_bank(v_ms, ego_keep, b, fmap.dtype,
                              withheld_speed=withheld_speed)
        x0 = bank
        prior_bank = bank if self.anchor_v0_cond else None

        # ---- S2b: PRE-DECODE anchor band (opt-in, inference-only) ---------- #
        # Gather the survivors into a dense [B, K, S, 2], decode ONLY those, and
        # scatter back. Pruned rows come back with conf = -inf so they can never
        # win, and with offset 0 so the emitted fan still HAS N entries and every
        # downstream consumer keeps its shape.
        pre_keep = None
        pre_tele: dict = {}          # merged into `tele` once it exists (l.~1373)
        if sel.anchor_prefilter and v_ms is not None:
            pre_keep = sl.anchor_reachability_mask(
                bank, v_ms.to(bank.dtype), accel_max=sel.accel_max,
                horizon_s=sel.horizon_s)
            if ego_keep is not None:
                # ⚠️ STRONGER than the S2 version of this guard. `v_ms` is the
                # PRE-dropout speed; at S2 it could only bias a ranking, but
                # here it decides which candidates are COMPUTED AT ALL. A row
                # whose speed was withheld keeps its whole fan.
                pre_keep = pre_keep | (~ego_keep)[:, None]
            dead = ~pre_keep.any(dim=1)          # unreachable-everywhere window
            pre_keep = pre_keep | dead[:, None]  # ...keeps its whole fan
            k = int(pre_keep.sum(dim=1).max())
            if k < n:
                # dense top-k gather: `keep` sorts survivors first, and the
                # per-row pad re-uses a survivor index so the gather is valid;
                # padded slots are re-masked to -inf below.
                order = pre_keep.to(torch.int8).argsort(dim=1, descending=True,
                                                        stable=True)
                sub = order[:, :k]                            # [B, K]
                took = pre_keep.gather(1, sub)                # [B, K] bool
                xs = x0.gather(
                    1, sub[:, :, None, None].expand(b, k, self.n_steps, 2))
                c_s, o_s = self._decode(kv, cond, xs, 0,
                                        agent_tokens, agent_pad)
                conf0 = x0.new_full((b, n), float("-inf"))
                conf0.scatter_(1, sub, c_s.masked_fill(~took, float("-inf")))
                offset = x0.new_zeros(b, n, self.n_steps, 2)
                offset.scatter_(
                    1, sub[:, :, None, None].expand(b, k, self.n_steps, 2),
                    o_s * took[:, :, None, None].to(o_s.dtype))
                pre_tele["prefilter_k"] = int(k)
                pre_tele["prefilter_speedup"] = round(float(n) / max(k, 1), 3)
            else:                                 # nothing to save this batch
                conf0, offset = self._decode(kv, cond, x0, 0,
                                             agent_tokens, agent_pad)
                pre_tele["prefilter_k"] = int(n)
                pre_tele["prefilter_speedup"] = 1.0
        else:
            conf0, offset = self._decode(kv, cond, x0, 0,
                                         agent_tokens, agent_pad)  # classifier
        x = self._feasible(bank + offset, v_ms)               # [B, N, S, 2]

        # ---- priors on the CLASSIFIER surface (unchanged semantics) ---------
        terms: list[Tensor] = []
        # H19: maneuver prior reweights the anchor confidences (log-space).
        if self.maneuver_to_anchor is not None and maneuver_logits is not None:
            terms.append(self.maneuver_to_anchor(
                torch.log_softmax(maneuver_logits, dim=-1)))
        # D-TAC1: the factorised pair, summed. ``lat_prior`` / ``lon_prior`` are
        # already log-probabilities (optionally prior-centered) prepared by the
        # model, so the decoder keeps no policy of its own.
        if self.lat_to_anchor is not None and lat_prior is not None:
            terms.append(self.lat_to_anchor(lat_prior))
        if self.lon_to_anchor is not None and lon_prior is not None:
            terms.append(self.lon_to_anchor(lon_prior))
        # LAN: the route reweights the SAME anchor priors, geometrically.
        if self.lan_gate is not None and lan_dir is not None:
            terms.append(self.lan_gate
                         * self._lan_anchor_prior(lan_dir, prior_bank))
        conf, tele = self._apply_grafts(conf0, terms, self._seam_conf, "conf",
                                        sel.seam_fail_patience)

        # Truncated diffusion: refine the anchor trajectories a few steps. Noise
        # only in training (deterministic at eval so decoding is reproducible).
        # S1: the last pass's confidence is KEPT. Pre-D-SEL this was
        # ``_, off = self._decode(...)`` — the refined fan was then ranked by the
        # UNREFINED score, which is the measured 45.4 %-of-windows ranking
        # failure. ``steps == 0`` leaves ``refined is conf`` by construction, so
        # ``--mode classifier`` is provably unaffected by S1.
        refined = conf
        # ⭐ refcv5 WP-4: the sampler REPLACES this loop, it does not wrap it.
        # The loop below perturbs the fan in METRES by a fixed `noise_std` and
        # re-reads a confidence -- the SKELETON of truncated diffusion with none
        # of the mechanism, and MEASURED on refcv3 its ranking was UNCHANGED ON
        # 201/201 WINDOWS. Running both would put two different perturbations on
        # one fan and make the arm non-attributable, which is the `--v2`
        # conflation failure. Exactly one of them refines the fan.
        u0_hat = None
        smp_tele: dict = {}
        _loop_steps = 0 if self.control_head is not None else steps
        for i in range(_loop_steps):
            t_idx = min(i + 1, self.cfg.diffusion_steps)
            noise = (torch.randn_like(x) * self.cfg.noise_std
                     if self.training else torch.zeros_like(x))
            x_in = x + noise
            r_conf, off = self._decode(kv, cond, x_in, t_idx)
            # EVERY pass that moves a waypoint, not just the classifier pass.
            x = self._feasible(x_in + off, v_ms)
            # The refined readout carries the SAME priors as the classifier
            # surface: switching the ranking to `refined` without them would
            # silently DELETE the H19 coupling from selection, which is a
            # regression dressed as a fix. Patience 0 — the saturation counter is
            # advanced once per forward, on the classifier surface above.
            refined, _ = self._apply_grafts(r_conf, terms, self._seam_refined,
                                            "refined", 0)

        if self.control_head is not None:
            # The emitted fan comes out of the INTEGRATOR, and the predicted
            # clean CONTROL leaves the decoder as `u0_hat` -- it is the only
            # tensor the x0 loss can be computed on, and without that loss
            # `control_head` stays at its zero init forever while every log row
            # still says "sampler: ddim" (which is why the trainer REFUSES
            # `--sampler ddim --w-u0 0`).
            x, u0_hat, s_conf, smp_tele = self._sample(
                kv, cond, bank, v_ms, steps, agent_tokens, agent_pad)
            # Stage 0 applies to EVERY pass that moves a waypoint; a no-op
            # returning the same object when `feasible_decode` is off.
            x = self._feasible(x, v_ms)
            # The refined readout carries the SAME priors as the classifier
            # surface -- dropping them here would silently DELETE the H19
            # coupling from selection, a regression dressed as a fix.
            refined, _ = self._apply_grafts(s_conf, terms, self._seam_refined,
                                            "refined", 0)

        # S1b: THE READOUT ABOVE SCORES THE WRONG OBJECT, AND IT IS ONE LINE
        # OF SOURCE. `_decode(kv, cond, x_in, t)` returns the confidence OF
        # `x_in` alongside the offset that improves it, and the loop then emits
        # `x = x_in + off`. So `refined` is the confidence of the estimate the
        # LAST pass CONSUMED, never of the fan that leaves this method - the
        # emitted trajectories are scored by no head at all. That is D1 again,
        # one denoise step less severe: the shipped ranker is 2 passes stale and
        # S1's refined ranker is 1 pass stale.
        #
        # THE EMITTED FAN IS NOT TOUCHED. The extra pass keeps its confidence and
        # DISCARDS its offset, so `anchor_traj` - and therefore the published
        # oracle-in-fan (0.1914 base / 0.1640 XL) that every D-SEL contrast is
        # paired against - is bit-unchanged. The cost is one extra decoder pass
        # and ZERO parameters.
        #
        # `t_idx` continues the loop's own schedule (pass i used `i + 1`),
        # clamped to the embedding table exactly as the loop clamps it.
        prefinal = None
        if sel.score_emitted and steps > 0:
            t_e = (min(steps + 1, self.cfg.diffusion_steps)
                   if sel.score_emitted_t < 0
                   else min(sel.score_emitted_t, self.cfg.diffusion_steps))
            e_conf, _ = self._decode(kv, cond, x, t_e,
                                     agent_tokens, agent_pad)
            prefinal = refined
            refined, _ = self._apply_grafts(e_conf, terms, self._seam_refined,
                                            "refined", 0)

        # ---- the RANKED score ------------------------------------------------
        base = refined if sel.refined else conf
        r_terms: list[Tensor] = []
        cons_s = None
        if self.route_to_anchor is not None and route_prior is not None:
            r_terms.append(self.route_to_anchor(route_prior))
        # S6: the PREDICTED goal reaches SELECTION through the SAME param-free
        # geometric compatibility the SUPPLIED LAN route uses — identical
        # mechanism, different provenance — and through a separately-gated
        # along-track term. Two gates, so bearing and distance are individually
        # ablatable and the K7 prediction is readable off the learned values.
        if self.goal_gate is not None and goal_dir is not None:
            r_terms.append(self.goal_gate
                           * self._lan_anchor_prior(goal_dir, prior_bank))
        if self.goal_dist_gate is not None and goal_dist_pref is not None:
            r_terms.append(self.goal_dist_gate
                           * self._goal_along_prior(goal_dist_pref,
                                                    prior_bank))
        # ⭐⭐ S7 / E15 (GP-1) — THE METRIC GOAL POINT REACHES SELECTION.
        #
        # `-||bank[:, :, slot] - goal|| / scale_m`, param-free, times ONE
        # zero-init gate. Moving the goal moves the ranking BY CONSTRUCTION:
        # there is no between-row variance that can shrink to zero while a
        # useful mean survives, which is precisely the degree of freedom E13's
        # `Embedding(4) -> Linear -> add` collapsed into (content 2.3 %,
        # presence 97.7 %). ⛔ The GATE is still free to go to zero — that is
        # why value-sensitivity is a pre-registered GATE (PREREG §4, V1-V3)
        # with a committed minimum degradation under a corrupted goal, and not
        # an architectural promise.
        #
        # ⛔⛔ UNITS. The head emits a NORMALISED point (x/scale, y/scale) —
        # that is what the loss supervises — while `anchor_goal_prior_at_time`
        # compares against `bank`, which is in METRES.
        # `anchor_goal_prior_at_time_from_norm` is the ONE place the conversion
        # lives, and it is named for it. MEASURED 2026-09-06: without it a
        # 24 m goal enters as a 2.5 m one, the term stays finite and the seam
        # trains, and a MIRRORED goal moved the ranked score by 7.7e-4 and
        # flipped 0 of 4 picks on a rig whose anchors span ±66–82 m laterally
        # — i.e. the PREREG §4 value-sensitivity gate would have failed for a
        # UNITS reason and read as "the geometric prior ignores the goal".
        #
        # ⚠️ DEVIATION FROM PREREG §8's DRAFT, STATED RATHER THAN SILENT: the
        # draft passed `prior_bank`, which is `None` on a FIXED (non-v0-
        # conditioned) vocabulary — there it is the legacy bit-exactness path
        # for the two S6 terms, and `anchor_goal_prior_at_time` would raise on
        # it. `bank` is [B, N, S, 2] on BOTH paths and is EXACTLY `prior_bank`
        # in the registered arm (`--anchor-v0-conditioned`), so this is
        # identical where the arm is defined and functional where the draft
        # would have crashed. There is no pre-S7 checkpoint to stay bit-exact
        # against, so the legacy fallback has nothing to preserve here.
        if self.gp_point_gate is not None and gp_point is not None:
            gv = (torch.ones(gp_point.shape[0], device=gp_point.device,
                             dtype=gp_point.dtype)
                  if gp_valid is None else gp_valid)
            r_terms.append(self.gp_point_gate
                           * gpm.anchor_goal_prior_at_time_from_norm(
                               gp_point.to(bank.dtype), gv, bank,
                               self.gp_slot, self.gp_scale_m))
        if (self.cons_gate is not None and cons_head is not None
                and cons_ctx is not None):
            cons_s = sl.consequence_scores(x, cons_ctx, cons_head,
                                           self.feat_proj, self.conf_head,
                                           detach=sel.cons_detach)
            r_terms.append(self.cons_gate * cons_s)
        score, r_tele = self._apply_grafts(base, r_terms, self._seam_rank,
                                           "rank", sel.seam_fail_patience)
        if self.grounded:
            score = score + self._grounded_score(x)
        tele.update(r_tele)

        # S2: the reachability band filters the ARGMAX ONLY. ``score`` is
        # returned unmasked, so no ``-inf`` can reach a cross-entropy, and a row
        # whose survivor set is empty keeps its whole fan — an unreachable-
        # everywhere window is a measurement failure, not a licence to emit
        # nothing.
        rank = score
        reach_keep = None
        if sel.reach_clamp and v_ms is not None:
            keep = sl.reachability_mask(x, v_ms.to(x.dtype),
                                        accel_max=sel.accel_max,
                                        horizon_s=sel.horizon_s)
            if ego_keep is not None:                  # never leak past dropout
                keep = keep | (~ego_keep)[:, None]
            dead = ~keep.any(dim=1)
            keep = keep | dead[:, None]
            reach_keep = keep
            rank = score.masked_fill(~keep, float("-inf"))
            tele["reach_frac_candidates_clipped"] = round(
                float(1.0 - keep.to(score.dtype).mean().detach()), 4)
            tele["reach_frac_windows_empty"] = round(
                float(dead.to(score.dtype).mean().detach()), 4)
        idx = rank.argmax(dim=1)                              # [B] (detached)
        # S2b telemetry + THE RUNTIME GUARD. `be2da04` keeps two claims apart:
        # the VARIABLE-width policy is structurally exact, while a FIXED budget
        # is an EMPIRICAL CALIBRATION (XL's worst window had 102 survivors
        # against a budget of 92, and held only because the winner's rank never
        # exceeded 92 ON THAT CORPUS). So equivalence is ASSERTED PER BATCH and
        # never assumed: `winner_survives_frac < 1.0` means the prefilter would
        # have changed the emitted trajectory, which is a CORRECTNESS failure,
        # not a speed regression.
        if pre_tele:
            tele.update(pre_tele)
        if pre_keep is not None and sel.anchor_prefilter_guard:
            # ⛔ WHAT THIS CAN AND CANNOT CHECK. `idx` is ALREADY the restricted
            # argmax, so asking whether it survived the band is TAUTOLOGICAL —
            # it is a survivor by construction and would report 1.0000 forever,
            # whether or not the prefilter changed the emitted trajectory. That
            # is worse than no check, because it reads as a verified invariant.
            # Winner-survival is only answerable by ALSO decoding the full fan,
            # which is the cost this flag exists to avoid, so it is an OFFLINE
            # CALIBRATION (feed `anchor_prefilter_report` the FULL-fan idx from
            # a banked fan) and NOT a runtime guarantee. What IS honest at
            # runtime is the budget and the empty-row count.
            surv = pre_keep.sum(dim=1)
            tele["prefilter_survivors_mean"] = round(
                float(surv.to(torch.float32).mean()), 3)
            tele["prefilter_survivors_max"] = int(surv.max())
            tele["prefilter_rows_full_fan"] = int((surv == pre_keep.shape[1])
                                                  .sum())
        # refcv5 WP-4: stamped ONLY when the sampler ran, so a reader can
        # never mistake "the key is absent" for "the sampler was off" -- the
        # off/absent distinction lives in `config.json`'s seam stamp, which is
        # written unconditionally.
        tele.update(smp_tele)
        traj = x[torch.arange(b, device=x.device), idx]       # [B, S, 2]
        out = {"anchor_logits": conf, "refined_logits": refined,
               "anchor_traj": x, "anchor_bank": bank,
               "offset": offset, "sel_score": score,
               "traj": traj, "sel_idx": idx, "sel_tele": tele}
        if u0_hat is not None:
            # [B, N, S, 2] in the VOCABULARY's control units -- `alat` (m/s^2)
            # or `kappa` (1/m) per `anchor_control_units`. ⛔ The x0 loss MUST
            # convert its target into these units before comparing: the inverse
            # map returns CURVATURE, and at 36 m/s the two differ by ~1300x
            # while BOTH tables look plausible.
            out["u0_hat"] = u0_hat
        if cons_s is not None:
            out["cons_score"] = cons_s
        if prefinal is not None:
            # S1b's own control, carried in the SAME forward: the readout S1
            # ships today, next to the one that scores the emitted fan. A
            # cross-forward comparison would confound the change with float
            # non-determinism; this one cannot.
            out["prefinal_logits"] = prefinal
        if reach_keep is not None:
            # S1c consumes this in the trainer. The argmax above already ranks
            # over exactly this set, so exporting it is what lets the CROSS-
            # ENTROPY normalise over the same support instead of over a fan that
            # is 72-74 % unpickable.
            out["reach_keep"] = reach_keep
        return out


# ============================================================================
# H15 imagination field (gated graft — belief over the conv-map tokens)
# ============================================================================

# `advect` was a self-contained COPY of tanitad.models.imagination.advect. The
# copy is RETIRED (2026-07-27): two implementations of the same warp is how
# geometries drift apart, and the shared one is now grid-general (scalar side OR
# (rows, cols)). Re-exported so `from tanitad.refs.refc import advect` still works.
from tanitad.models.imagination import advect                    # noqa: E402,F401


class ImagBlock(nn.Module):
    """Pre-norm self-attention + FiLM-free MLP (belief-field refinement over the
    token grid — same shape as the encoder's ViT Block, kept local)."""

    def __init__(self, d: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ImaginationField(nn.Module):
    """H15 gated belief field over the [B, F, g, g] conv-map tokens.

    Latent-advection prior (object permanence; zero-init flow -> identity warp at
    start) -> transformer refinement -> a per-cell epistemic log-variance. The
    confidence sigmoid(-logvar) gates a residual belief written back into the
    tokens the anchor decoder cross-attends: refined = tokens + conf * out(z).
    Every parameter (in_proj, flow, blocks, norm, logvar, out_proj) sits in the
    trajectory-loss gradient path — no dead params. Gated: absent when off."""

    def __init__(self, feat_dim: int, grid_hw, cfg: ImaginationConfig):
        super().__init__()
        self.grid_hw = grid_hw          # scalar side OR (rows, cols)
        d = cfg.d
        self.in_proj = nn.Linear(feat_dim, d)                 # conv tokens -> d
        self.flow_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 2))
        nn.init.zeros_(self.flow_head[-1].weight)             # identity advection
        nn.init.zeros_(self.flow_head[-1].bias)               # (needs grid >= 4:
        # at grid 2 every zero-flow sample lands on the normalized boundary where
        # grid_sample's position gradient is clamped to 0 — real configs use 8)
        self.blocks = nn.ModuleList(
            ImagBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.logvar_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1))
        self.out_proj = nn.Linear(d, feat_dim)                # belief -> feat_dim

    def forward(self, fmap: Tensor) -> tuple[Tensor, Tensor]:
        """fmap [B, F, g, g] -> (refined [B, F, g, g], logvar [B, g*g])."""
        b, fdim, g, _ = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)              # [B, N, F]
        z = self.in_proj(tokens)                              # [B, N, d]
        z = advect(z, self.flow_head(z), self.grid_hw)        # object permanence
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        logvar = self.logvar_head(z).squeeze(-1).clamp(-10.0, 10.0)   # [B, N]
        conf = torch.sigmoid(-logvar).unsqueeze(-1)           # low var -> trust
        refined = tokens + conf * self.out_proj(z)            # residual belief
        return refined.transpose(1, 2).reshape(b, fdim, g, g), logvar


# ============================================================================
# Model
# ============================================================================

class RefCModel(nn.Module):
    """Anchored-Diffusion-C: ResNet encoder + anchored-diffusion trajectory
    decoder + LAW aux + maneuver/route aux heads + hierarchical conditioning."""

    def __init__(self, cfg: RefCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResNetEncoder(cfg.encoder)
        feat = self.encoder.feat_dim
        n_steps = len(cfg.trajectory.horizons)
        if cfg.refc1 and len(cfg.path_dists) != n_steps:
            raise ValueError(f"refc1 needs len(path_dists) == "
                             f"len(horizons): {len(cfg.path_dists)} != "
                             f"{n_steps}")
        # Hierarchy graft (gated): absent when off — byte-identical model.
        if cfg.hierarchy:
            self.strategic = StrategicCtx(feat, cfg.strategic.hidden,
                                          cfg.strategic.d_ctx)
        # Measurement encoder (KEEP): [v0, nav one-hot] with ego-dropout. The
        # strategic ctx now conditions the DECODER, not the measurement.
        # X15 (``ego_valid_channel``): an explicit "v0 is present" flag next to
        # the ego-dropped speed. REF-C zero-fills ``v`` under ego-dropout, but
        # 0.0 m/s is a perfectly in-distribution "stationary", so the zero-fill
        # is a CONFIDENT LIE — the reader cannot tell "we withheld the speed"
        # from "the car is stopped". The repo already encodes exactly this
        # distinction for the route (``LAN_FEATS_PER_ANCHOR`` carries a ``valid``
        # flag), so the validity CHANNEL is this codebase's own convention; the
        # flagship reaches the same rule with a learned null EMBEDDING ROW
        # (``V15Config.ego_null_row``). The channel is preferred here because it
        # also reaches the TACTICAL head, which reads ``v`` directly and would be
        # untouched by a row swapped in at the measurement OUTPUT.
        # E1 (``nav_known_channel``): the same rule one channel over. The nav
        # one-hot cannot express "this command is the UNKNOWN sentinel", because
        # `_ROUTE_TO_NAV.get(route, NAV_FOLLOW)` maps BOTH `ROUTE_STRAIGHT` and
        # `ROUTE_UNKNOWN` to `follow`. Without the bit the network's only options
        # are to trust a meaningless command on 62.4 % of `follow` windows or to
        # learn to ignore the command channel entirely — and evaluating REF-C
        # with ``nav_cmd=None`` is what the second one looks like from outside.
        d_meas_in = (1 + len(NAV_COMMANDS) + (1 if cfg.ego_valid_channel else 0)
                     + (1 if cfg.nav_known_channel else 0))
        self.measurement = nn.Sequential(
            nn.Linear(d_meas_in, cfg.measurement.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.measurement.hidden, cfg.measurement.d_out),
            nn.ReLU(inplace=True))
        # Anchored-diffusion trajectory decoder (replaces TCP traj+control).
        anchors = default_anchors(cfg.trajectory.horizons, cfg.anchors.n_anchors,
                                  cfg.anchors.pool_size, cfg.anchors.seed,
                                  device="cpu")
        self.decoder = AnchoredDiffusionDecoder(
            feat, n_steps, cfg.measurement.d_out, cfg.strategic.d_ctx,
            cfg.tactical_latent_dim, anchors, cfg.decoder,
            hierarchy=cfg.hierarchy, graft_maneuver=cfg.graft_maneuver,
            graft_target_latent=cfg.graft_target_latent,
            grounded_selector=cfg.grounded_selector,
            graft_lan=cfg.graft_lan, d_lan=cfg.lan.d_out,
            factored_maneuver=cfg.factored_maneuver,
            sel=cfg.selection(),
            horizons=cfg.trajectory.horizons,
            v0_conditioned=cfg.anchors.v0_conditioned,
            ref_speed_ms=cfg.anchors.ref_speed_ms,
            control_units=cfg.anchors.control_units,
            alat_v_floor_ms=cfg.anchors.alat_v_floor_ms,
            kappa_cap=cfg.anchors.kappa_cap)
        # ---- refcv5 WP-6: THE AGENT SEAM (default OFF, builds NOTHING) ----
        #
        # ⭐ `agent_head` produces SLOTS, `agent_embed` turns slots into the
        # tokens the planner decoder cross-attends. The oracle and the learned
        # detector emit the SAME dict shape, so `agent_embed` consumes either
        # without a branch and the two arms differ in exactly one object.
        #
        # ⛔ THE ORACLE IS NOT A CAPABILITY CLAIM. It reads a privileged label
        # at inference, which the vision-only rule forbids for any deployable
        # arm. Its ONLY job is to price the ceiling -- "if the detector were
        # perfect, would the planner move at all?" -- before a detector
        # GPU-day is spent. `AgentSeamConfig.oracle` is stamped into
        # config.json so no reader can mistake one arm for the other.
        self.agent_head: nn.Module | None = None
        self.agent_embed: nn.Module | None = None
        _ag = getattr(cfg, "agents", None)
        if _ag is not None and bool(getattr(_ag, "enable", False)):
            from tanitad.refs import refc_agents as _ra    # lazy; see imports
            if not cfg.decoder.cross_agent:
                raise ValueError(
                    "refcv5 WP-6: `agents.enable` is set but "
                    "`decoder.cross_agent` is False, so the head would be "
                    "BUILT, SUPERVISED and STAMPED while its tokens reached "
                    "no decoder layer -- a detector trained into a dead end, "
                    "and an arm that would read as 'agent tokens do not "
                    "help'. Set both, or neither.")
            _gh, _gw = cfg.encoder.grid_shape
            self.agent_head = (
                _ra.OracleAgentEmbed(_ag) if bool(_ag.oracle)
                else _ra.build_agent_head(_ag, feat, int(_gh) * int(_gw)))
            self.agent_embed = _ra.AgentTokenEmbed(cfg.decoder.d, _ag)
        # LAN route encoder (gated): [B, K*4] corridor features -> [B, d_out].
        # Lives at model level next to ``measurement`` because it is an INPUT
        # encoder, not part of the decoder; param_breakdown reports it as `lan`.
        if cfg.graft_lan:
            self.lan_enc = nn.Sequential(
                nn.Linear(cfg.lan.dim, cfg.lan.hidden), nn.ReLU(inplace=True),
                nn.Linear(cfg.lan.hidden, cfg.lan.d_out), nn.ReLU(inplace=True))
        # H15 imagination graft (gated): belief field over the conv-map tokens,
        # refining the [B, F, 8, 8] map the decoder cross-attends. Absent when off.
        if cfg.graft_imagination:
            self.imagination = ImaginationField(feat, cfg.encoder.grid_shape,
                                                cfg.imagination)
        # Aux heads: the tactical head feeds BOTH the tactical CE and the H19
        # anchor reweight; the route head is the strategic aux.
        #
        # D-TAC1 (gated): ``factored_maneuver`` REPLACES the mixed 5-way head
        # with two independent ones. The 5-way output survives as an exact
        # derivation (refc_tactical.derive_man5_logprobs), so nothing downstream
        # loses a field, and the priority collapse stops destroying the
        # longitudinal decision INSIDE the model.
        #
        # ``tactical_speed_input`` widens the head input by the ego-speed
        # channel. The 5-way head reads ``pooled`` — the IMAGE embedding alone —
        # while its own label is dv = v(t+2s) - v(t): it is asked a question
        # about speed while blind to speed. The refc1 target-speed head already
        # concatenates the measurement (see ``speed_cls`` below), so this is a
        # pattern the file already uses; only the SPEED channel is taken, never
        # the nav one-hot (constant at eval -> the C6 confound).
        #
        # ⚠️ It is DELIBERATELY INDEPENDENT of ``factored_maneuver`` (it was
        # coupled to it until 2026-08-03). The coupling looked conservative — it
        # "froze the 5-way head's input so published numbers stay reproducible" —
        # but reproducibility is already guaranteed by the flag DEFAULTING OFF
        # (pinned byte-identical by tests), and the coupling had a real cost: the
        # pre-registered arm set (`dtac1-full` = F1+F2, `dtac1-f2only` = F2) had
        # NO arm isolating F1, so a win by `dtac1-full` over `dtac1-f2only` was
        # the only available F1 estimate and it is confounded by the fact that
        # the two arms also differ in which head consumes the speed. With the
        # flag free, `refc_f1only_config()` is the missing INPUT-only arm.
        # The tactical head's speed channel carries its validity flag with it —
        # the flag is meaningless to a head that never reads the speed, so it is
        # conditioned on BOTH switches rather than on ``ego_valid_channel`` alone.
        d_tac = feat + (2 if (cfg.tactical_speed_input and cfg.ego_valid_channel)
                        else 1 if cfg.tactical_speed_input else 0)
        if cfg.factored_maneuver:
            # ONE shared trunk, TWO linear readouts — deliberately, and measured:
            # two independent MLPs cost +272,001 params (+0.261 % of REF-C-base)
            # and would let anyone attribute an A/B win to capacity. Sharing the
            # trunk makes the change a PURE READOUT change — same features, two
            # heads — for +897 params (+0.00086 %). Pinned by
            # tests/test_refc_tactical.py, which is what caught the original
            # two-MLP version and the spec's "~5 k parameters" estimate.
            self.tactical_trunk = nn.Sequential(
                nn.Linear(d_tac, cfg.decoder.aux_hidden), nn.ReLU(inplace=True))
            self.lat_head = nn.Linear(cfg.decoder.aux_hidden, N_LAT_MAN)
            self.lon_head = nn.Linear(cfg.decoder.aux_hidden, N_LON_MAN)
            # Class log-priors travel with the checkpoint. UNIFORM at init on
            # purpose: logit adjustment by a uniform prior is exactly the
            # identity, so a run that never calls update_tactical_prior() can
            # never silently alter a decode.
            self.register_buffer("lat_log_prior",
                                 torch.full((N_LAT_MAN,),
                                            -math.log(N_LAT_MAN)))
            self.register_buffer("lon_log_prior",
                                 torch.full((N_LON_MAN,),
                                            -math.log(N_LON_MAN)))
        else:
            # ``d_tac`` (not ``feat``) is the ONLY difference from the shipped
            # head, and it is ``feat`` exactly when tactical_speed_input is off —
            # so the default path builds the identical module it always did.
            self.maneuver_head = nn.Sequential(
                nn.Linear(d_tac, cfg.decoder.aux_hidden), nn.ReLU(inplace=True),
                nn.Linear(cfg.decoder.aux_hidden, N_MANEUVERS))
        self.route_head = nn.Linear(feat, N_ROUTE)
        # S6 (gated): the PREDICTED GEOMETRIC goal head. THREE outputs off the
        # image embedding alone — (cos, sin) of the route bearing at the first
        # admissible arc-length, and a signed along-track preference. ONE Linear,
        # not an MLP: the capacity control that caught a +272,001-parameter
        # tactical head applies here too, and a goal head that needs depth to
        # read a bearing off a trained trunk is a different claim than the one
        # being tested. Provenance is declared by :meth:`goal_provenance`.
        if cfg.graft_goal:
            self.goal_head = nn.Linear(feat, 3)
        # LAW aux (KEEP): decoded trajectory enters NON-detached — gradients flow.
        self.law_head = nn.Sequential(
            nn.Linear(feat + 2 * n_steps, cfg.law.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.law.hidden, feat))
        # REF-C.1 (gated): target-speed classification head (KEEP).
        if cfg.refc1:
            self.speed_cls = nn.Sequential(
                nn.Linear(feat + cfg.measurement.d_out, cfg.speed_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.speed_hidden, cfg.speed_bins))
        # ====================================================================
        # ⛔⛔ WP-D BEV AUXILIARY HEAD -- THIS BLOCK MUST STAY LAST IN __init__
        # ====================================================================
        # Everything above draws from the global RNG at construction. Building
        # this head here and NOWHERE ELSE is what makes an `aux on` model and an
        # `aux off` model share BIT-IDENTICAL initial weights for every
        # parameter they have in common -- i.e. what makes the A/B a
        # one-variable comparison at step 0 rather than a lever-plus-seed
        # change. ⛔ Do not insert a module after this one; add it above.
        # Pinned by `stack/tests/test_bev_aux.py::test_shared_params_bit_identical`.
        #
        # ⛔ TRAINING-ONLY. Its output reaches no planner tensor; removing it is
        # bit-identical at the planner's output
        # (`::test_planner_output_bit_identical`). The published precedent is
        # PhyLatent's Physical State Grounding, "used only during training and
        # not required by the planner" (LIT-2, banked 2608.05720).
        self.bev_aux_head: nn.Module | None = None
        _bv = getattr(cfg, "bev_aux", None)
        if _bv is not None and bool(getattr(_bv, "enable", False)):
            from tanitad.refs import refc_bev_aux as _rb   # lazy; see imports
            self.bev_aux_head = _rb.BEVAuxHead(
                feat, cfg.encoder.grid_shape, _bv)

    # --- S6 goal provenance (the PI's admissibility check, in code) ----------
    @staticmethod
    def goal_provenance() -> dict:
        """WHAT THE GOAL IS COMPUTED FROM — declared, not left implicit.

        Binding ruling (Sayed, 2026-08-03): *"yes a goal input is admissible, at
        the same time, we need to be careful not to include the result of the
        situation classification in the goal input."* The admissibility check is
        **"could this goal have been computed from the situation classifier's
        output?"** — and if yes, it is inadmissible until shown otherwise.

        REF-C's answer, from source:

        * **AT INFERENCE the goal is a function of ``pooled`` and NOTHING else.**
          ``pooled`` is the mean-pooled conv feature of the LAST frame. No
          situation-classifier output — posterior, argmax, embedding, or any
          feature derived from them — exists anywhere in ``RefCModel``'s graph.
          The situation classifier is a SEPARATE model (the sitclf stream's
          ``head_img``); REF-C neither imports it, loads it, nor receives its
          output as a batch field. There is no output to leak.
        * **THE SHARED TRUNK IS DECLARED, AND HERE IS WHY IT IS NOT A BACK
          DOOR.** ``goal_head``, ``route_head`` and the tactical head all read
          the SAME ``pooled``. That is a shared ENCODER, which is what the model
          IS; giving the goal its own trunk would be a capacity change, not an
          attribution fix (C34: match capacity before attributing an effect to
          information). Attributability is bought instead by the ZERO-INIT
          gates: ``goal_gate`` and ``goal_dist_gate`` start at exactly 0, so the
          ranked score is bit-identical to the goal-free baseline at step 0 and
          the exact ablation is "set the gate to 0". A shared trunk can only
          launder a signal that EXISTS in the graph — and the classifier's does
          not.
        * **GEOMETRIC, NOT CATEGORICAL.** The published evidence with proper
          no-navigation controls says a categorical command buys ~nothing
          (TransFuser perturbed to None/Random/Left/Right: PDMS flat 84.0-84.7;
          no-nav -> command-only **+0.2**) while a geometric goal buys a lot
          (route path + turn-by-turn **+2.3**; GoalFlow goal POINT **+4.7**).
          So the goal here is a BEARING and an along-track preference entering a
          param-free geometric compatibility — the same surface LAN uses — and
          NOT another class token. *(This is also why S5, which IS categorical,
          is registered as the lowest-prior lever.)*
        * **PREDICTED, NOT SUPPLIED.** A supplied route is optimistic by
          construction on PhysicalAI, whose only route supplier is the ego's own
          future path. ``lan`` (the supplied corridor) is used ONLY as the
          TRAINING LABEL for ``goal_head``; at inference the seam reads the
          HEAD's output. That is the sanctioned direction of the standing rule
          "LABELS MAY USE EGO; INFERENCE IS VISION-ONLY", and it is asserted by
          ``tests/test_refc_select.py``, which shows the goal terms are
          bit-unchanged when the ``lan`` field is withheld at eval.

        Returned as data so a run's ``config.json`` and any results file carry
        the declaration rather than a reader having to trust a docstring.
        """
        return {
            "inference_inputs": ["pooled (mean-pooled conv features, last frame)"],
            "contains_situation_classifier_output": False,
            "situation_classifier_in_graph": False,
            "shared_trunk_with": ["route_head", "maneuver_head/tactical_trunk"],
            "shared_trunk_justification":
                "shared ENCODER, not a shared signal; attributability comes "
                "from the zero-init gates, and a shared trunk cannot launder a "
                "signal that is absent from the graph",
            "form": "geometric (bearing + signed along-track preference)",
            "supplied_or_predicted": "predicted",
            "label_source": "tanitad.data.lan.lan_window_features (ego future "
                            "path, arc-length resampled, leak-guarded) — TRAIN "
                            "ONLY; never read at inference",
            "admissibility_check":
                "could this goal have been computed from the situation "
                "classifier's output? NO — that output is not in the graph, is "
                "not a batch field, and is not a label source here.",
        }

    # --- encode surface -----------------------------------------------------
    def encode_pooled(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> pooled latent [B, F] (LAW target path)."""
        return self.encoder(frames)[1]

    def _speed_bin_centers(self, device, dtype) -> Tensor:
        half = self.cfg.speed_max / (2 * self.cfg.speed_bins)
        return torch.linspace(half, self.cfg.speed_max - half,
                              self.cfg.speed_bins, device=device, dtype=dtype)

    # --- D-TAC1 tactical class prior ----------------------------------------
    @torch.no_grad()
    def update_tactical_prior(self, lat_idx: Tensor, lon_idx: Tensor,
                              momentum: float | None = None) -> None:
        """EMA the empirical class log-priors from a batch of LABELS.

        Called by the trainer, NOT by ``forward``: a buffer that mutates inside
        a forward pass would drift at eval and silently change a published
        decode. Uniform at init, so a run that never calls this leaves
        :func:`refc_tactical.logit_adjust` an exact identity at any ``tau``.
        """
        if not self.cfg.factored_maneuver:
            raise ValueError("update_tactical_prior needs factored_maneuver")
        mom = self.cfg.tactical_prior_momentum if momentum is None else momentum
        for buf, idx, n in ((self.lat_log_prior, lat_idx, N_LAT_MAN),
                            (self.lon_log_prior, lon_idx, N_LON_MAN)):
            batch = tac.class_log_prior(idx, n).to(buf.device, buf.dtype)
            # EMA in PROBABILITY space (an EMA of log-probs does not stay
            # normalised), then re-log.
            p = mom * buf.exp() + (1.0 - mom) * batch.exp()
            buf.copy_((p / p.sum().clamp_min(1e-12)).clamp_min(1e-6).log())

    # ------------------------------------------------------------------------
    @staticmethod
    def lan_direction(lan: Tensor, k: int) -> Tensor:
        """[B, K*4] LAN features -> [B, 3] (cos, sin, valid) route bearing.

        The bearing is read from the FIRST anchor whose ``valid`` flag is set —
        the nearest admissible arc-length, i.e. the earliest point of the route
        that survived the leak guard. Rows with no valid anchor return
        ``(1, 0, 0)``: bearing dead ahead but ``valid = 0``, so the geometric
        anchor prior multiplies out to exactly zero rather than silently voting
        "straight" (which on a 74 %-straight corpus is the base rate and is how
        an inert route input looks like a working one).
        """
        f = lan.reshape(lan.shape[0], k, LAN_FEATS_PER_ANCHOR)
        valid = f[..., 3] > 0.5                                   # [B, K]
        any_valid = valid.any(dim=-1)
        first = torch.argmax(valid.to(lan.dtype), dim=-1)         # [B]
        idx = first.reshape(-1, 1, 1).expand(-1, 1, LAN_FEATS_PER_ANCHOR)
        picked = torch.gather(f, 1, idx).squeeze(1)               # [B, 4]
        cos_b = torch.where(any_valid, picked[:, 0],
                            torch.ones_like(picked[:, 0]))
        sin_b = torch.where(any_valid, picked[:, 1],
                            torch.zeros_like(picked[:, 1]))
        return torch.stack([cos_b, sin_b, any_valid.to(lan.dtype)], dim=-1)

    @staticmethod
    def goal_targets(lan: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
        """LAN corridor -> ``(bearing [B, 2], dist_pref [B], valid [B])``.

        The TRAINING LABEL for :attr:`goal_head`, and **only** that: nothing in
        :meth:`forward` calls this. ``bearing`` is the unit route direction at
        the first admissible arc-length; ``dist_pref`` maps that anchor's INDEX
        (arc-lengths are ascending, so the index is monotone in distance) onto
        [-1, +1] — near goal -> -1, far goal -> +1.

        ⚠️ **DECLARED CONFOUND on ``dist_pref``, because it decides how the S6
        result may be read.** Which arc-length is "first admissible" depends on
        LAN's leak guard, which is ``2 s of path length + min_lead_m`` — i.e. it
        is partly a function of the ego's own SPEED. So a head trained on it is
        partly being asked to predict speed from a single frame, which is
        precisely the K7 quantity measured UNRECOVERABLE across 17 head
        architectures. That is not a reason to drop the term; it is the reason
        the gate is SEPARATE and the reason the pre-registration predicts it
        stays near zero. If ``goal_dist_gate`` opens anyway, the K7 prior is
        wrong on this substrate and that is the finding.

        The bearing half carries no such confound: lateral topology is
        recoverable from one frame, and it is the half the published
        goal-conditioning wins (route path +2.3, goal point +4.7) are about.
        """
        f = lan.reshape(lan.shape[0], k, LAN_FEATS_PER_ANCHOR)
        valid = f[..., 3] > 0.5                                   # [B, K]
        any_valid = valid.any(dim=-1)
        first = torch.argmax(valid.to(lan.dtype), dim=-1)         # [B]
        idx = first.reshape(-1, 1, 1).expand(-1, 1, LAN_FEATS_PER_ANCHOR)
        picked = torch.gather(f, 1, idx).squeeze(1)               # [B, 4]
        bearing = picked[:, :2] / torch.linalg.vector_norm(
            picked[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
        span = max(k - 1, 1)
        dist_pref = first.to(lan.dtype) / span * 2.0 - 1.0
        return bearing, dist_pref, any_valid

    def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,
                v0: Tensor | None = None,
                maneuver_logits: Tensor | None = None,
                target_latent: Tensor | None = None, steps: int = 0,
                lan: Tensor | None = None,
                nav_known: Tensor | None = None,
                hierarchy_hook=None,
                ego_keep: Tensor | None = None,
                withheld_speed: Tensor | None = None,
                gp_point: Tensor | None = None,
                gp_valid: Tensor | None = None,
                agent_gt: dict | None = None) -> dict:
        """frames [B, W, C, H, W'], nav_cmd [B] long (None -> `follow`), v0 [B]
        current ego speed (None -> zeros; scaled /10 inside). ``maneuver_logits``
        / ``target_latent`` are OPTIONAL external tactical-brain seams (else the
        model's own maneuver head drives the H19 reweight and the target-latent
        FiLM stays inactive). ``lan`` [B, K*4] is the LAN route corridor
        (``graft_lan``; None -> the seam is skipped entirely, so an unrouted
        window costs nothing). ``steps`` selects the decoder mode: 0 =
        classifier (default), >0 = truncated diffusion.

        ``hierarchy_hook`` (REF-C v3, default ``None`` == byte-identical to the
        pre-hook forward): an IN-FORWARD supplier for the two external-tactical-
        brain ports above, so a hierarchy built OUTSIDE this class can read the
        window it is conditioning on without encoding the frames a second time
        (the encoder is ~90 % of the tick — a second encode would be the cost,
        and a cached one would be a stale-latent hazard). Called as
        ``hook(pooled_seq [B, W, F], ctx [B, d_ctx]) -> dict`` and its
        ``"maneuver_logits"`` / ``"target_latent"`` entries fill the ports ONLY
        where the caller passed ``None`` — an explicitly supplied port always
        wins, so the hook can never silently override an experiment's input.
        Requires ``cfg.hierarchy`` (``pooled_seq`` exists only on that path).
        The supplier lives in ``tanitad/refs/refc_v3.py``.

        Returns dict: pooled [B, F], traj / wp_seq [B, n_steps, 2], waypoints
        {key: [B, 2]}, anchor_logits [B, N], anchor_traj [B, N, n_steps, 2],
        offset [B, N, n_steps, 2], sel_idx [B], maneuver_logits [B, 5],
        route_logits [B, 3], law_pred [B, F], measurement [B, d_m] (+ hierarchy:
        ctx [B, d_ctx]) (+ graft_imagination: imag_logvar [B, g*g]) (+ refc1:
        speed_logits, target_speed) (+ factored_maneuver: lat_logits [B, 3],
        lon_logits [B, 3], lat_decision / lon_decision / maneuver_decision [B];
        ``maneuver_logits`` is then the EXACT derived 5-way LOG-PROB vector, so
        every existing reader keeps its field and its semantics).
        """
        b, w = frames.shape[:2]
        if self.cfg.hierarchy:
            fmap_all, pooled_all = self.encoder(
                frames.reshape(b * w, *frames.shape[2:]))
            pooled_seq = pooled_all.reshape(b, w, -1)
            pooled = pooled_seq[:, -1]
            fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
            ctx = self.strategic(pooled_seq)
        else:                                    # last frame only (same values)
            fmap, pooled = self.encoder(frames[:, -1])
            ctx = None

        # REF-C v3 hierarchy hook (gated by the ARGUMENT, not by a config flag:
        # with hook=None this block is untouched dead code and the forward is
        # byte-identical to the pre-hook file — pinned by tests/test_refc_v3.py).
        if hierarchy_hook is not None:
            if not self.cfg.hierarchy:
                raise ValueError(
                    "hierarchy_hook requires cfg.hierarchy=True — pooled_seq "
                    "is only computed on that path, and running the hook on a "
                    "last-frame-only build would hand it a window that does "
                    "not exist (the C115 defect, manufactured).")
            hk = hierarchy_hook(pooled_seq, ctx)
            if maneuver_logits is None:
                maneuver_logits = hk.get("maneuver_logits")
            if target_latent is None:
                target_latent = hk.get("target_latent")
            # ⭐ H-EGO-LIT-4: the hook's OWN 2 s speed (detached) feeds the
            # decoder's `pred` withheld-bank mode. An explicitly supplied
            # `withheld_speed` wins (the eval-time SHUFFLE control passes a
            # permuted copy), exactly like the two ports above.
            if withheld_speed is None:
                withheld_speed = hk.get("bank_speed_pred")
            # ⭐⭐ S7 / E15 (GP-1): the PREDICTED METRIC GOAL POINT, emitted by
            # `refc_v3.py`'s hook. NO SECOND FORWARD — the value already exists
            # in that graph and the whole point of shipping it through the hook
            # was that this patch needs no extra encode.
            #
            # ⛔ AN EXPLICITLY SUPPLIED `gp_point` WINS, exactly like the two
            # ports above and `withheld_speed`. That is not symmetry for its own
            # sake: it is what makes PREREG §2's five eval-time interventions
            # (mirror / range x0.5 / straight / shuffle / withheld) run as ONE
            # process on ONE surface, which is the discipline a recent
            # cross-hardware reproduction violated and paid two argmax
            # tie-breaks and two missed controls for.
            if gp_point is None:
                gp_point = hk.get("goal_point")

        # H15 belief field refines the conv-map tokens before the decoder (gated).
        imag_logvar = None
        if self.cfg.graft_imagination:
            fmap, imag_logvar = self.imagination(fmap)

        nav_cmd_given, known = nav_cmd is not None, None
        if nav_cmd is None:                      # unlabeled -> follow (idx 0)
            nav_cmd = torch.zeros(b, dtype=torch.long, device=frames.device)
        nav = F.one_hot(nav_cmd, len(NAV_COMMANDS)).to(pooled.dtype)
        v = torch.zeros(b, 1, dtype=pooled.dtype, device=pooled.device) \
            if v0 is None else (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
        # ``keep`` is now MATERIAL, not an intermediate: it is the ego-validity
        # channel (X15) and it is what stops the reachability band (S2) from
        # leaking the speed back in on a sample whose speed was withheld.
        keep = torch.ones(b, 1, dtype=v.dtype, device=v.device) \
            if v0 is not None else torch.zeros(b, 1, dtype=v.dtype,
                                               device=v.device)
        # ⭐ REF-C v4 SEAM (2026-09-03): a CALLER-SUPPLIED withholding draw
        # overrides the internal one. Additive and gated — with `ego_keep=None`
        # (every pre-v4 caller) this branch is dead and the forward is
        # byte-identical to the pre-seam file.
        #
        # ⛔ WHY IT HAD TO EXIST RATHER THAN v4 DRAWING ITS OWN. v4 feeds the
        # measured ego state to the GOAL path as well as to the measurement
        # encoder, and the two must be withheld TOGETHER: this file's own
        # comment 30 lines down warns that a second, unsynchronised dropout is
        # the wrong fix ("`ego_dropout` is the knob to sweep — not a second
        # dropout here"), and it is right. The alternative — pre-multiplying
        # `v0` outside — is WORSE THAN WRONG: `keep` is derived from
        # `v0 is not None`, so a pre-zeroed v0 would arrive with keep = 1 and
        # assert "this really is a stationary car", which is EXACTLY the X15
        # zero-collision this seam exists to remove. One draw, one owner.
        if ego_keep is not None:
            keep = keep * ego_keep.to(v.dtype).reshape(b, 1)
            v = v * keep
        elif self.training and self.cfg.ego_dropout > 0:
            keep = keep * (torch.rand(b, 1, device=v.device)
                           >= self.cfg.ego_dropout).to(v.dtype)
            v = v * keep                         # per-sample Bernoulli zero
        # E1: the nav command's COMPANION BIT. Fail loud in BOTH directions —
        # a `nav_known` that is silently dropped is exactly the class of bug this
        # seam exists to remove, and a gate that is on while the caller forgets
        # the bit would quietly assert "this command is a real judgement" on
        # every window.
        if nav_known is not None and not self.cfg.nav_known_channel:
            raise ValueError(
                "nav_known was supplied but cfg.nav_known_channel is False — it "
                "would be silently dropped. Turn the gate on or stop passing it.")
        if self.cfg.nav_known_channel:
            if nav_known is None:
                if not nav_cmd_given:
                    # `nav_cmd=None` -> the `follow` fallback IS the sentinel, so
                    # the honest companion bit is 0.0 and needs no argument.
                    nav_known = torch.zeros(b, dtype=pooled.dtype,
                                            device=pooled.device)
                else:
                    raise ValueError(
                        "nav_known_channel is on and a nav_cmd was supplied, so "
                        "nav_known must be supplied too (refb_labels."
                        "nav_input_v22 returns the pair). Defaulting it to 1.0 "
                        "would assert a judgement the labeller never made.")
            known = nav_known.to(pooled.dtype).reshape(b, 1)
        meas_in = ([v, nav] + ([keep] if self.cfg.ego_valid_channel else [])
                   + ([known] if self.cfg.nav_known_channel else []))
        m = self.measurement(torch.cat(meas_in, dim=-1))

        # Aux heads (image branch): maneuver logits also drive the H19 reweight.
        route_logits = self.route_head(pooled)
        lat_logits = lon_logits = lat_prior = lon_prior = None
        # D-TAC1 F1. ``v`` is the SHARED, already-ego-dropped speed channel: one
        # dropout draw per sample across the whole model, so the tactical head
        # introduces no new stochastic surface and cannot be accused of weakening
        # the documented ego-dropout guard. (If speed turns out to be the binding
        # constraint, ``ego_dropout`` is the knob to sweep — not a second,
        # unsynchronised dropout here.) Built once and used by BOTH branches, so
        # the factored and 5-way heads see exactly the same input vector.
        tac_in = torch.cat([pooled, v] + ([keep] if self.cfg.ego_valid_channel
                                          else []), dim=-1) \
            if self.cfg.tactical_speed_input else pooled
        if self.cfg.factored_maneuver:
            h_tac = self.tactical_trunk(tac_in)
            lat_logits = self.lat_head(h_tac)
            lon_logits = self.lon_head(h_tac)
            man_logits = tac.derive_man5_logprobs(lat_logits, lon_logits)
            if self.cfg.graft_prior_center:
                lat_prior = tac.prior_centered_logprobs(lat_logits,
                                                        self.lat_log_prior)
                lon_prior = tac.prior_centered_logprobs(lon_logits,
                                                        self.lon_log_prior)
            else:
                lat_prior = torch.log_softmax(lat_logits, dim=-1)
                lon_prior = torch.log_softmax(lon_logits, dim=-1)
        else:
            man_logits = self.maneuver_head(tac_in)
        reweight = maneuver_logits if maneuver_logits is not None else man_logits
        if maneuver_logits is not None and self.cfg.factored_maneuver:
            # An EXTERNAL tactical brain speaks the 5-way surface. Factorise its
            # posterior through the exact inverse collapse rather than dropping
            # it on the floor — a silently-ignored external prior is exactly the
            # class of bug this seam exists to remove.
            log_lat, log_lon = tac.invert_man5(maneuver_logits)
            lat_prior, lon_prior = (
                (log_lat - self.lat_log_prior, log_lon - self.lon_log_prior)
                if self.cfg.graft_prior_center else (log_lat, log_lon))

        # LAN (gated): encode the route corridor, with per-sample route dropout
        # in training so the planner can never become route-DEPENDENT — a model
        # that collapses when the route is absent cannot be deployed on a corpus
        # whose route label is missing on ~75 % of windows.
        lan_emb = lan_dir = None
        if self.cfg.graft_lan and lan is not None:
            lan_in = lan.to(pooled.dtype)
            if self.training and self.cfg.route_dropout > 0:
                keep = (torch.rand(b, 1, device=lan_in.device)
                        >= self.cfg.route_dropout).to(lan_in.dtype)
                lan_in = lan_in * keep       # zeroes valid flags too -> masked
            lan_emb = self.lan_enc(lan_in)
            lan_dir = self.lan_direction(lan_in, self.cfg.lan.k)

        # D-SEL wiring. Each is None unless its flag is on, so a default build
        # calls the decoder with exactly the pre-D-SEL argument set.
        #   S5  the model's OWN strategic route readout, as log-probabilities —
        #       the same shape of coupling H19 already gives the tactical head.
        #       ⚠️ Admissible only when route_head is trained on a FUTURE-derived
        #       target (--labels v21/v3); under --labels v1 the target is
        #       route_target(nav_cmd) and grafting it would pipe the nav echo
        #       into selection (the C6 confound). The trainer enforces this.
        #   S3  law_head is REF-C's trajectory-conditioned world model, so it
        #       IS the consequence predictor; `pooled` is its context.
        #   S2  the RAW speed and the ego-dropout keep-mask.
        route_prior = torch.log_softmax(route_logits, dim=-1) \
            if (self.cfg.graft_route and not self.cfg.no_strategic) \
            else None
        cons_head = self.law_head if self.cfg.graft_cons else None
        cons_ctx = pooled if self.cfg.graft_cons else None
        v_ms = (v0.to(pooled.dtype) if (self.cfg.sel_reach_clamp
                                        and v0 is not None) else None)
        #   S6  the PREDICTED geometric goal. Read from `pooled` ALONE — never
        #       from `lan`, which is the training LABEL only. `valid` is pinned
        #       to 1 because a PREDICTION is always defined; the supplied
        #       corridor's validity flag belongs to the label, and letting it
        #       reach here would make the seam silently depend on the label at
        #       eval. Provenance: `RefCModel.goal_provenance()`.
        goal_dir = goal_dist_pref = None
        if self.cfg.graft_goal:
            g = self.goal_head(pooled)                        # [B, 3]
            bearing = g[:, :2] / torch.linalg.vector_norm(
                g[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
            goal_dir = torch.cat(
                [bearing, torch.ones_like(bearing[:, :1])], dim=-1)   # [B, 3]
            goal_dist_pref = torch.tanh(g[:, 2])              # [B] in (-1, 1)
            out_goal = {"goal_bearing": bearing, "goal_dist_pref": goal_dist_pref}
        else:
            out_goal = {}
        # ---- refcv5 WP-6: slots -> tokens ---------------------------------
        # ⛔ THE VISION-ONLY RULE IS ENFORCED BY WHERE THE TENSOR IS READ, not
        # by a comment. The LEARNED head takes `fmap` and nothing else --
        # `AgentSlotDecoder.forward` accepts exactly one argument, which is the
        # audit in signature form. `agent_gt` is read ONLY on the oracle path,
        # which is declared inadmissible as a capability claim and stamped as
        # such. The detection LABELS never enter here at all; they enter the
        # LOSS, in the trainer, from `batch["agent_box"]`.
        agent_slots = agent_tokens = agent_pad = None
        if self.agent_head is not None:
            if self.cfg.agents.oracle:
                if agent_gt is None:
                    raise ValueError(
                        "refcv5 WP-6: `agents.oracle` is set but no "
                        "`agent_gt` reached the forward. The oracle's tokens "
                        "ARE the ground-truth boxes; with none, the seam would "
                        "silently emit nothing and the arm would read as "
                        "'agent tokens do not help' while never having had "
                        "any. Wire --agent-join, or run --agents head.")
                agent_slots = self.agent_head(
                    agent_gt["box"], agent_gt["yaw"], agent_gt["cls"],
                    agent_gt["valid"], agent_gt.get("rates"))
            else:
                agent_slots = self.agent_head(
                    fmap.flatten(2).transpose(1, 2))          # [B, M, F]
            agent_tokens, agent_pad = self.agent_embed(agent_slots)
        # ⭐⭐ S-BYPASS-1 -- THE OPERATIVE SEAM. `ctx_to_cond` is skipped by
        # the decoder itself when `ctx is None` (its `cond` block reads
        # `ctx is not None`), so the bypass needs NO branch inside the decoder
        # and adds NO parameter.
        # ⛔ `ctx` ITSELF IS STILL COMPUTED AND STILL EMITTED in `out` -- the
        # strategic token stays a READABLE DIAGNOSTIC, it simply stops being an
        # INPUT to the plan. With the flag OFF `ctx_dec is ctx` exactly, so this
        # line cannot perturb a today-arm.
        ctx_dec = None if self.cfg.no_strategic else ctx
        dec = self.decoder(fmap, m, ctx=ctx_dec, maneuver_logits=reweight,
                           target_latent=target_latent, steps=steps,
                           lan_emb=lan_emb, lan_dir=lan_dir,
                           lat_prior=lat_prior, lon_prior=lon_prior,
                           route_prior=route_prior, cons_head=cons_head,
                           cons_ctx=cons_ctx, v_ms=v_ms,
                           ego_keep=keep.squeeze(-1) > 0.5,
                           goal_dir=goal_dir, goal_dist_pref=goal_dist_pref,
                           gp_point=gp_point, gp_valid=gp_valid,
                           withheld_speed=withheld_speed,
                           agent_tokens=agent_tokens, agent_pad=agent_pad)
        traj = dec["traj"]
        law_pred = self.law_head(torch.cat([pooled, traj.reshape(b, -1)],
                                           dim=-1))

        keys = self.cfg.path_dists if self.cfg.refc1 \
            else self.cfg.trajectory.horizons
        out = {"pooled": pooled, "traj": traj, "wp_seq": traj,
               "waypoints": {k: traj[:, i] for i, k in enumerate(keys)},
               "anchor_logits": dec["anchor_logits"],
               "refined_logits": dec["refined_logits"],
               "sel_score": dec["sel_score"], "sel_tele": dec["sel_tele"],
               "anchor_traj": dec["anchor_traj"],
               "anchor_bank": dec["anchor_bank"], "offset": dec["offset"],
               "sel_idx": dec["sel_idx"], "maneuver_logits": man_logits,
               "route_logits": route_logits, "law_pred": law_pred,
               "measurement": m, **out_goal}
        # ⭐ S7 / E15 CAVEAT-B INSTRUMENTATION. The gate is zero-init and must
        # LEARN to open; a 0.0000 reading at 40 k must be a LOGGED fact, not a
        # conclusion inferred from an eval three days later — that inference is
        # what produced Caveat-B. The gate ALONE cannot distinguish "has not
        # opened YET" from "will never open", so the score scale it multiplies
        # is emitted beside it. Both are detached diagnostics.
        if self.decoder.gp_point_gate is not None:
            out["gp_point_gate_value"] = self.decoder.gp_point_gate.detach()
            out["gp_point_injected"] = gp_point is not None
            if gp_point is not None:
                with torch.no_grad():
                    out["gp_point_score_absmean"] = (
                        gpm.anchor_goal_prior_at_time_from_norm(
                            gp_point.to(dec["anchor_bank"].dtype),
                            (torch.ones(gp_point.shape[0],
                                        device=gp_point.device,
                                        dtype=dec["anchor_bank"].dtype)
                             if gp_valid is None
                             else gp_valid.to(dec["anchor_bank"].dtype)),
                            dec["anchor_bank"], self.decoder.gp_slot,
                            self.decoder.gp_scale_m).abs().mean())
        if agent_slots is not None:
            # ⛔ The detection loss is guarded on `"agent_slots" in out`. With
            # the seam built but this key absent, `--w-agent 1.0` would parse,
            # be STAMPED into config.json, and supervise NOTHING -- the M18
            # defect the trainer's own `--agents off` guard names.
            out["agent_slots"] = agent_slots
        if "u0_hat" in dec:
            out["u0_hat"] = dec["u0_hat"]        # refcv5 WP-4's x0 loss target
        if "cons_score" in dec:
            out["cons_score"] = dec["cons_score"]
        for _k in ("prefinal_logits", "reach_keep"):
            # S1b's in-forward control and S1c's CE support, passed through
            # VERBATIM: `compute_losses` reads `reach_keep` and the probes read
            # both. Re-deriving either outside the decoder is how two
            # definitions of the same mask drift apart - the reason
            # `refc_select.reachability_mask` is a re-export and not a copy.
            if _k in dec:
                out[_k] = dec[_k]
        if ctx is not None:
            out["ctx"] = ctx
        if lat_logits is not None:
            # D-TAC1. Raw logits (what the CE trains) AND the prior-corrected
            # DECISION (what a report/HUD/closed-loop logger should read) are
            # both emitted, never conflated: an argmax over the raw posterior is
            # the estimate of "most likely class"; the adjusted argmax is the
            # estimate of "most surprising given the base rate", and only the
            # second can emit a 12 %-prior class at all. ``man_prior_tau = 0``
            # makes them identical, which is the default.
            out["lat_logits"] = lat_logits
            out["lon_logits"] = lon_logits
            tau = float(self.cfg.man_prior_tau)
            out["lat_decision"] = tac.logit_adjust(
                lat_logits, self.lat_log_prior, tau).argmax(dim=-1)
            out["lon_decision"] = tac.logit_adjust(
                lon_logits, self.lon_log_prior, tau).argmax(dim=-1)
            out["maneuver_decision"] = tac.collapse(out["lat_decision"],
                                                    out["lon_decision"])
        if lan_dir is not None:
            out["lan_dir"] = lan_dir                 # route bearing actually used
        if imag_logvar is not None:
            out["imag_logvar"] = imag_logvar         # H15 per-cell uncertainty
        if self.cfg.refc1:
            logits = self.speed_cls(torch.cat([pooled, m], dim=-1))
            centers = self._speed_bin_centers(logits.device, logits.dtype)
            out["speed_logits"] = logits
            out["target_speed"] = F.softmax(logits, dim=-1) @ centers
        # ⛔⛔ WP-D -- THIS BLOCK MUST STAY LAST IN forward, for the same reason
        # its constructor is last in __init__: it must not perturb any draw the
        # decoder makes. It consumes NO RNG (no dropout, no sampling) and writes
        # to no tensor the planner reads, so `out` minus this key is
        # bit-identical to an `aux off` build's `out`. TRAINING-ONLY: nothing
        # downstream of the planner may read `bev_logits`, and an eval path that
        # did would be reading a head that does not exist at deployment.
        if self.bev_aux_head is not None:
            out["bev_logits"] = self.bev_aux_head(fmap)   # [B, n_rng, n_az]
        return out


def param_breakdown(model: RefCModel) -> dict[str, int]:
    """Per-module trainable-parameter table (report + config.json row).

    ⭐ ``selection`` is the D-SEL capacity control and it is reported SEPARATELY
    from ``decoder`` even though both grafts are decoder submodules — a lever
    whose cost is buried inside a 40 M-parameter line is a lever nobody can
    audit. On REF-C-base the whole selection surface with every flag on is
    ``route_to_anchor`` (N_ROUTE x n_anchors, zero-init) + ``cons_gate`` (1):
    **+385 parameters**, i.e. +0.00037 % — the same order as the F1 arm's +384
    and ~1/700 of the +272,001 an earlier tactical attempt cost before its own
    capacity check caught it. S1/S2/S4 are structurally FREE (0 parameters).

    ``encoder`` (the proven lever — where the lifted budget goes) vs ``decoder``
    (the anchored-diffusion decoder INCLUDING its gated graft submodules ctx->
    cond, maneuver->anchor, target-latent FiLM) is the split that shows where the
    budget went; ``imagination`` is the H15 belief field (0 unless
    graft_imagination); ``aux`` is the maneuver + route heads; ``strategic`` is 0
    when hierarchy=False; ``speed`` is 0 unless refc1. REF-C-base lands ~110 M
    (tests pin 90-130 M), REF-C-XL ~260 M (230-280 M)."""
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    dec = model.decoder
    # D-SEL is CARVED OUT of `decoder`, never added on top of it: the breakdown
    # must keep summing to `total` exactly (pinned by tests/test_refc.py), and a
    # line that double-counted would make the capacity control unreadable.
    n_sel = ((cnt(dec.route_to_anchor) if dec.route_to_anchor is not None else 0)
             + (dec.cons_gate.numel() if dec.cons_gate is not None else 0)
             + (dec.goal_gate.numel() + dec.goal_dist_gate.numel()
                if dec.goal_gate is not None else 0)
             # S7/E15: +1 parameter. The score it multiplies is param-free
             # (`goal_point.anchor_goal_prior_at_time`), so the entire metric-
             # goal-point selection seam costs ONE scalar — carved out of
             # `decoder` here for the same reason every other D-SEL gate is:
             # the breakdown must keep summing to `total` exactly.
             + (dec.gp_point_gate.numel()
                if getattr(dec, "gp_point_gate", None) is not None else 0))
    # S6's head is a MODEL-level input head (like `lan_enc`), reported on its own
    # line: the goal is the lever under the PI's admissibility ruling and its
    # cost must be readable without unpicking a 40 M-parameter decoder row.
    n_goal = cnt(model.goal_head) if model.cfg.graft_goal else 0
    return {
        "encoder": cnt(model.encoder),
        "measurement": cnt(model.measurement),
        "strategic": cnt(model.strategic) if model.cfg.hierarchy else 0,
        "decoder": cnt(model.decoder) - n_sel,
        "imagination": cnt(model.imagination) if model.cfg.graft_imagination
        else 0,
        "lan": cnt(model.lan_enc) if model.cfg.graft_lan else 0,
        "aux": (cnt(model.tactical_trunk) + cnt(model.lat_head)
                + cnt(model.lon_head) if model.cfg.factored_maneuver
                else cnt(model.maneuver_head)) + cnt(model.route_head),
        "law": cnt(model.law_head),
        "speed": cnt(model.speed_cls) if model.cfg.refc1 else 0,
        "selection": n_sel,
        "goal": n_goal,
        # ⭐ WP-D: reported on its own line and NEVER folded into `encoder`, for
        # the same reason `selection` is carved out of `decoder` — this head is
        # a TRAINING-ONLY term whose cost a reader must be able to subtract when
        # comparing a deployed parameter count against an `aux on` run's.
        "bev_aux": (cnt(model.bev_aux_head)
                    if getattr(model, "bev_aux_head", None) is not None else 0),
        "total": cnt(model),
    }
