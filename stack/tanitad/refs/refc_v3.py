"""REF-C v3 — the goal-mediated strategic/tactical/operative hierarchy on the
supervised arm (the 4B-dominance proof track).

Design + pre-registration: ``TanitAD Research Lab/Architecture & Inference/
Research/2026-08-18-refc-v3-design/`` (``REFC_V3_DESIGN.md`` carries the edge
list E1..E12 this module implements; ``PREREG_REFC_V3.md`` commits the
experiment). This docstring repeats only what a reader of the CODE needs.

WHAT THIS MODULE IS
============================================================================
:class:`RefCV3Model` composes an UNMODIFIED :class:`tanitad.refs.refc.RefCModel`
with a goal cascade built from components the programme already measured:

  * strategic state  = the core's own ``StrategicCtx`` GRU (existing, trained);
    a new 771-param head reads a PREDICTED geometric route goal ``g_str`` off it
    (195 before the 2026-09-02 hierarchy rebalance widened ``d_ctx`` 64 -> 256
    to match flagship v7's ``d_str`` and refav1's ``d_ctx``);
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
* No ego state into any goal head — **TRUE FOR v3 ONLY, AND v4 REVERSES IT**
  (``ego_state_inject``). E11 refused ``v0 -> {z_tac, g_str, g_tac}`` on the
  2026-08-03 vision-only rule; the PI narrowed that rule twice (2026-09-02
  measured-``v``-at-cycle-time, 2026-09-03 *"the rule is ... anti-ECHO"*), and
  **E11'** makes the MEASURED t0 ego state a REQUIRED LIVE edge into all three
  goal nodes. The refusal is not deleted but MOVED: **E11'' refuses
  ``future_poses``/``future_actions`` into any goal node**, which is the edge
  the PI's *"and not the future one"* actually names. ⛔ With
  ``ego_state_inject=False`` (the default) this file still builds v3 and the
  original refusal still holds, pinned by
  ``test_T5b_the_ORIGINAL_E11_test_still_passes_on_a_v3_config``. See
  ``.../2026-09-03-refc-v4-design/PREREG_REFC_V4.md`` §3.
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
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models.tactical import PhiTac
from tanitad.refs import goal_point as gp
from tanitad.refs import refc
from tanitad.refs import refc_select as sl
from tanitad.refs import refc_tactical as tac

__all__ = [
    "V3_HORIZONS", "SEAM_SLOT", "GOAL_TAU_STEPS", "RefCV3Config",
    "refc_v3_flat_config", "refc_v3_hier_config", "refc_v3_smoke_config",
    "refc_v3_sized_config", "refc_v3_small_config", "refc_v3_xl_config",
    "V3_SIZES", "V3_RIG_SIZES",
    "RefCV3Model", "config_delta", "param_breakdown_v3",
    "masked_goal_loss", "strategic_goal_loss", "selection_ce",
    "freeze_history_report",
    # --- v4 (E11' + E14) ---
    "WHEELBASE_CONST2P9", "EGO_DIMS", "ego_state_at_t0",
    "kinematic_goal_extrapolation", "refc_v4_config",
    "REGISTERED_DELTA_KEYS_V4",
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
# ⭐⭐ REF-C v4 — THE MEASURED EGO STATE AT t0 (E11', PI 2026-09-03)
# ============================================================================
# PI, verbatim: *"allow it to use the measured ego state (measured current
# speed, measured current acceleration, measured current yaw rate) and not the
# future one"*, under the narrowed vision-only rule: *"the rule is saying that
# the semantic understanding and the trajectory planning must be based on image
# frames and should avoid ECHO of the ego dynamics"*.
#
# ⭐ THE FINDING THAT MADE THIS CHEAP: neither derived channel needs a finite
# difference. MEASURED at source, `tanitad/data/physicalai.py:597-632`:
#     actions[:, 1] = ax        the dataset's OWN longitudinal accel — its
#                               docstring says explicitly "NOT d/dt(v), which
#                               differentiates interpolation noise and lags"
#     actions[:, 0] = atan(WHEELBASE * curvature)   an INVERTIBLE encoding of
#                               the dataset's OWN curvature column
# and `_contract.py:130` already returns `ep.actions[t:t+w]` in every window.
# So the brief's worst-case hazard (differencing noisy poses) does not arise:
# v4 READS two measured signals and differentiates nothing.
#
# ⚠️ THE WHEELBASE CANCELS, AND THAT IS NOT AN ACCIDENT WE MAY ASSUME.
# `WHEELBASE = 2.9` is wrong for 98.2 % of clips, but the cache stored
# `atan(2.9 * curvature)`, so `tan(steer)/2.9` recovers `curvature` EXACTLY —
# provided the cache is the LEGACY `const2p9` regime, which the parity corpus
# `physicalai-train-e438721ae894` is (`physicalai.py:82`). A `per_clip_v1`
# cache needs its per-clip L and is REFUSED below rather than silently
# mis-inverted: a wrong wheelbase would scale every curvature by L_true/2.9 and
# read exactly like a working channel.
#: The legacy wheelbase the parity caches encoded `steer` with. Inverting with
#: any OTHER value silently rescales curvature — hence the explicit refusal.
WHEELBASE_CONST2P9: float = 2.9
#: Ego block layout: (v0, a_long, yaw_rate, curvature, ego_keep). FOUR value
#: channels + the X15 validity bit.
EGO_DIMS: int = 5
#: Nav-arg block layout: (distance_norm, time_norm, args_valid) -- TWO value
#: channels from `vocab_v7.NAV_ARG_SLOTS` + the validity bit, EXACTLY the
#: `EGO_DIMS` shape above and for exactly the same reason. ⛔ The bit is not a
#: nicety: `NAV_FOLLOW_ROAD` carries `args: {}` on 2,897/2,897 train and 96/96
#: eval records, and `NavEmitter._args_for_window` defaults a missing slot to
#: 0.0, so without it the model is told "the turn is HERE, NOW" on the
#: follow-road majority. ⛔ UNITS: metres and seconds BEFORE normalisation;
#: the normalisation is fit-split-only (`NavArgStats`) and is stamped into
#: `config.json` beside the slot names.
NAV_ARG_DIMS: int = 3
#: The raw units the two value channels carry BEFORE normalisation. Written
#: into `config.json`; a run whose record lacks them is not quotable.
NAV_ARG_UNITS: tuple[str, ...] = ("distance_m:metres", "time_s:seconds")
#: Per-channel scales, so the block enters at ~unit magnitude. MEASURED on the
#: val epcache (40 ep / 7,963 frames): v std 3.671, a std 0.931, r std 0.159,
#: k std 0.055 — these are ~2 sigma, not guesses.
EGO_SCALE_V: float = 10.0
EGO_SCALE_A: float = 3.0
EGO_SCALE_R: float = 0.5
EGO_SCALE_K: float = 0.1


def ego_state_at_t0(poses_win: Tensor, actions_win: Tensor, *,
                    wheelbase_mode: str = "const2p9",
                    keep: Tensor | None = None) -> Tensor:
    """The MEASURED ego state at t0, as the ``[B, 5]`` block the model reads.

    ``poses_win``  ``[B, W, 4]`` = (x, y, yaw, speed) over the OBSERVED window.
    ``actions_win`` ``[B, W, 2]`` = (steer, accel) over the same window.
    Returns ``[B, 5]`` = ``(v0, a_long, yaw_rate, curvature, keep)`` at
    ``t0 = W-1`` — the LAST OBSERVED frame, the same index ``pose_last`` uses
    (``tanitad/data/_contract.py:137``).

    ⛔⛔ EVERY READ IS AT ``[:, -1]`` OF THE **OBSERVED** WINDOW. There is no
    index into the future here and there must never be one: the obvious wrong
    implementation of this feature is ``a0 = (future[:, 0, 3] - v0) / dt``,
    which would look identical in a config diff and would satisfy any *"does
    the goal use acceleration?"* check while violating the PI's *"not the
    future one"*. That is why the refused edge is PINNED interventionally
    (``future_poses -> any goal node`` must leave every goal node
    bit-identical) rather than asserted in a docstring — including this one.

    ⚠️ ``curvature`` is fed ALONGSIDE ``yaw_rate`` and not instead of it:
    ``yaw_rate = v0 * curvature`` VANISHES at standstill, and MEASURED 16.50 %
    of frames have ``v < 0.5 m/s``. A stopped car with the wheel turned has a
    real path geometry and a zero yaw rate. The two are algebraically dependent
    given ``v0``; that is disclosed, not hidden.
    """
    if wheelbase_mode != "const2p9":
        raise ValueError(
            f"ego_state_at_t0 can only invert the LEGACY 'const2p9' steer "
            f"encoding, got {wheelbase_mode!r}. A per-clip regime stored "
            f"atan(L_clip * curvature); inverting it with 2.9 rescales every "
            f"curvature by L_clip/2.9 and reads exactly like a working "
            f"channel. Pass the per-clip L through explicitly instead.")
    if poses_win.dim() != 3 or poses_win.shape[-1] != 4:
        raise ValueError(f"poses_win must be [B, W, 4], got "
                         f"{tuple(poses_win.shape)}")
    if actions_win.dim() != 3 or actions_win.shape[-1] != 2:
        raise ValueError(f"actions_win must be [B, W, 2], got "
                         f"{tuple(actions_win.shape)}")
    if poses_win.shape[1] != actions_win.shape[1]:
        raise ValueError("poses_win and actions_win must cover the SAME "
                         f"observed window: {poses_win.shape[1]} != "
                         f"{actions_win.shape[1]}")
    v0 = poses_win[:, -1, 3]                                # MEASURED speed
    a0 = actions_win[:, -1, 1]                              # MEASURED ax
    k0 = torch.tan(actions_win[:, -1, 0]) / WHEELBASE_CONST2P9   # curvature
    r0 = v0 * k0                                            # yaw rate [rad/s]
    if keep is None:
        keep = torch.ones_like(v0)
    return torch.stack([v0, a0, r0, k0, keep.to(v0.dtype)], dim=-1)


def ego_state_from_batch(batch: dict, *, wheelbase_mode: str = "const2p9",
                         device=None) -> Tensor:
    """The ``[B, 5]`` ego block from a ``V3Dataset`` batch. THE trainer entry.

    Reads ``pose_last`` ``[B, 4]`` (already ``ep.poses[t + w - 1]``, i.e. t0 by
    construction — ``_contract.py:137``) and ``actions`` ``[B, W, 2]`` at
    ``[:, -1]``. ⭐ Both fields are ALREADY in every window the contract
    returns; v4 adds no dataset field and no cache rebuild.

    ⛔ ``future_poses`` / ``future_poses_ext`` are NOT touched here and must
    never be: that is the refused edge E11' pins, and the pin is an
    interventional bit-identity check, not this sentence.
    """
    pose_last = batch["pose_last"]
    actions = batch["actions"]
    if device is not None:
        pose_last, actions = pose_last.to(device), actions.to(device)
    if actions.dim() != 3 or actions.shape[-1] != 2:
        raise ValueError(
            f"batch['actions'] must be [B, W, 2] = (steer, accel) over the "
            f"OBSERVED window, got {tuple(actions.shape)}. If it is missing, "
            f"the cache predates the action contract and the ego channels "
            f"cannot be derived — REFUSE rather than substitute zeros, which "
            f"would train a dead channel and read as 'ego does not help'.")
    return ego_state_at_t0(pose_last.unsqueeze(1).expand(-1, actions.shape[1],
                                                         -1),
                           actions, wheelbase_mode=wheelbase_mode)


def kinematic_goal_extrapolation(v0: Tensor, a0: Tensor, k0: Tensor,
                                 taus_s: Sequence[float]) -> Tensor:
    """⭐ ``ha0_ext`` — THE ECHO, in closed form, in the goal label's own frame.

    Constant longitudinal acceleration ``a0`` and constant curvature ``k0``
    from the measured t0 state, integrated exactly. Returns ``[B, K, 4]`` =
    ``(x, y, heading, speed)`` in the EGO FRAME OF t0 (+x forward, +y left,
    heading = wrap(yaw(t0+tau) - yaw(t0))) — which is verbatim the layout
    ``refb_labels.goal_tac_targets`` produces, so base and label live in the
    same frame and the residual is meaningful.

    ⛔⛔ THIS FUNCTION IS THE TRIVIAL SOLUTION, AND IT IS NOT SMALL. MEASURED
    2026-09-03 on the val epcache (40 ep, 1,041 windows @2 s / 801 @6 s):

        horizon   ha0 (const v)   ha0_ext (this)   ext beats ha0 by
          2.0 s     0.7040 m        0.4449 m           +36.80 %
          6.0 s     5.3547 m        4.4652 m           +16.61 %

    **0.4449 m at 2 s from ZERO PIXELS** is the same league as flagship v1's
    deployed 0.452 m. ⇒ any arm handed (v0, a0, k0) can reproduce the
    programme's headline 2 s number without reading the image, and the
    anti-echo bar at 2 s is 0.4449, NOT the 0.7040 of the old ``ha0`` control.

    ⚠️ Those two numbers are a corpus-kinematic property of THOSE windows. They
    are the design input and the gate's construction; they are NOT decision
    numbers. ``echo_gate`` recomputes both controls on the SAME windows as the
    model, paired — quoting 0.4449 against a differently-windowed model would
    be the ``df``/``step_s`` scope error in a new costume.

    A vehicle that would reverse under constant ``a0`` is STOPPED instead of
    run backwards: ``tau`` is clamped at ``-v0/a0`` when ``a0 < 0``. Physical,
    and it keeps the arc length monotone (a negative arc would rotate the
    heading the wrong way through ``theta = k0 * s``).
    """
    tau = torch.as_tensor(list(taus_s), dtype=v0.dtype,
                          device=v0.device).reshape(1, -1)      # [1, K]
    v0, a0, k0 = (t.reshape(-1, 1) for t in (v0, a0, k0))       # [B, 1]
    # Stop time under a deceleration; +inf when a0 >= 0 (never binds).
    t_stop = torch.where(a0 < 0, -v0 / a0.clamp(max=-1e-6),
                         torch.full_like(a0, float("inf")))
    ts = torch.minimum(tau.expand_as(v0 * tau), t_stop)         # [B, K]
    ts = ts.clamp_min(0.0)
    s = (v0 * ts + 0.5 * a0 * ts * ts).clamp_min(0.0)            # arc length
    speed = (v0 + a0 * ts).clamp_min(0.0)
    theta = k0 * s                                              # heading @ tau
    # x = sin(theta)/k, y = (1 - cos(theta))/k, with the k->0 limits x = s,
    # y = k s^2 / 2 taken by a Taylor branch (a bare divide is NaN at k = 0,
    # and k IS exactly 0 on straight windows — 0.40 % of frames carry a zero
    # steer and every synthetic corpus is entirely straight).
    small = k0.abs() < 1e-6
    k_safe = torch.where(small, torch.ones_like(k0), k0)
    x = torch.where(small, s, torch.sin(theta) / k_safe)
    y = torch.where(small, 0.5 * k0 * s * s, (1.0 - torch.cos(theta)) / k_safe)
    return torch.stack([x, y, theta, speed], dim=-1)            # [B, K, 4]


# ============================================================================
# Configs — the dominance pair differs in EXACTLY the registered lever set,
# and config_delta() below is the instrument that PINS that (C122's rule:
# "an ablation's 'everything else identical' must be DERIVED and pinned").
# ============================================================================

@dataclass
class RefCV3Config:
    #: ⭐ v7 mandate default; the kinematic trainer passes "kin3"
    #: explicitly because its labels are the 3x3 kinematic classes.
    tac_vocab_version: str = "v7.0"
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

    # --- E13: nav to the tactical and strategic layers (PI 2026-09-02) ------
    # BINDING: *"No training without feeding nav command to all layers."*
    # Before this, nav_cmd reached `self.core` and nowhere else. Additive and
    # ZERO-INIT, so a v3 checkpoint trained without it is unchanged at step 0.
    # ⛔ Does NOT relax E11 — `v0` remains refused into every goal node; nav is
    # a route command, which the admissibility ruling permits.
    nav_inject: bool = True

    # --- E13b: the nav command's CONTINUOUS ARGS (D-GSTR-1 P3, 2026-09-06) ---
    # ⭐ THE FIELD ALREADY EXISTS IN THE LABEL AND WAS NEVER FED. MEASURED:
    # `vocab_v7.NAV_ARG_SLOTS = ("distance_m", "time_s")`, written by
    # `s2_geom_emit_v7.nav_command()` on EVERY turn token (train TURN_L median
    # 27.3 m / 7.2 s, TURN_R 36.6 m / 7.4 s, max ~690 m / 33.6 s). v6/v7f FEED
    # both through `NavConditioner.arg_proj = Linear(2, d_embed)`; refav1
    # DISCARDS them; refcv3/refcv4b NEVER READ them -- the join takes
    # `nav.get("token")` and stores one integer. So refcv4b's nav input is
    # exactly the stripped BEARING that is separated WORSE by +2.3632 m, while
    # a goal carrying RANGE recovers 57.1 % of the longitudinal ceiling.
    #
    # ⛔ THREE SLOTS, NOT TWO -- the third is a VALIDITY CHANNEL and it is
    # not optional. MEASURED: `NAV_FOLLOW_ROAD` carries `args: {}` -- EMPTY on
    # 2,897/2,897 train and 96/96 eval records -- and the consumer's
    # `_args_for_window` defaults a missing slot to 0.0. A silent 0.0 is a LIE
    # that reads "the turn is HERE, NOW"; it is the `_ensure_ego` default trap
    # in a nav costume.
    #
    # ⛔ UNITS ARE DECLARED, NOT REMEMBERED. `distance_m` is METRES,
    # `time_s` is SECONDS, and both arrive ALREADY NORMALISED by fit-split
    # statistics (`tanitad.models.nav_conditioning.NavArgStats`). Units,
    # constants and semantics are stamped into `config.json` by the trainer.
    #
    # ⚠️ Default OFF: no banked arm's recipe changes.
    nav_args_inject: bool = False
    n_nav_commands: int = 4           # == len(refb.NAV_COMMANDS), test-pinned
    d_nav: int = 64

    # --- CAVEAT-A lever: is the hierarchy CONDITIONED or OPTIMISED? ---------
    # The E4/E7 `.detach()` buys clean attribution and costs joint optimisation:
    # the strategic goal head never learns whether its goal HELPED. Design item
    # D-4 registers the non-detached variant as a lever; the PI asked for it to
    # be addressed, so it is a switch with both settings runnable.
    # ⛔ E9's goal detach is deliberately NOT under this flag — that one is the
    # winner's-curse firewall (SEL-1's refusal), a different failure mode.
    uplink_grad: bool = True

    # --- ⭐⭐ v4: E11' + E14 (PI 2026-09-03) --------------------------------
    # ⛔ BOTH DEFAULT FALSE. With both off this class builds today's refcv3
    # BIT-IDENTICALLY (pinned by tests/test_refc_v4.py::test_v3_parity) — the
    # live 40,284-step run resumes through this file, so a default that moved
    # would silently change a training in flight.
    #
    # E11' — the measured ego state at t0 reaches {z_tac, g_str, g_tac}. This
    # RELAXES E11, which refused it. The relaxation is the PI's, twice narrowed:
    # the line is TIME (measured at t0, never future) plus ANTI-ECHO, not
    # MODALITY. What stays refused is unchanged and is listed in
    # `provenance_roles`.
    ego_state_inject: bool = False
    d_ego: int = 32                   # ego embed width (~= d_nav's 64 / 2)

    # --- ⭐⭐ E15: the PREDICTED METRIC GOAL POINT (goal-point agent, 2026-09-06) ----
    # ⛔ DEFAULT FALSE. With it off this class builds today's refcv4b/refcv5 BIT-
    # IDENTICALLY (pinned by tests/test_goal_point_wiring.py::test_v3_parity_when_off),
    # which matters because a live run resumes through this file.
    #
    # WHY IT EXISTS, measured, not argued. E13's categorical nav is a PRESENCE-GATED
    # BIAS: inverting the token on all 1,743 commanded windows costs +0.0022 m
    # [-0.0006, +0.0052] ADE (NOT separated) while removing it costs +0.0961 m
    # separated — content 2.3 %, presence 97.7 %. And on the anchor fan an ORACLE 3-way
    # command's best deterministic decoding of the LATERAL index is "go straight" for
    # ALL THREE classes: it is bit-identical to the no-information goal and separated
    # WORSE than the model's own pick. A metric goal point at a fixed TIME beyond the
    # scored horizon recovers 78.4 % of the lateral and 57.1 % of the LONGITUDINAL
    # selection ceiling, both separated — and the longitudinal half is unreachable by a
    # bearing (separated worse by +2.36 m) or by a categorical command (which has no
    # longitudinal vocabulary at all).
    # `…/Research/2026-09-06-goal-point/{RESULT.md,PREREG.md}`.
    #
    # ⛔ The goal is PREDICTED from the strategic context (vision). It carries NO
    # situation-classifier output in any form — see goal_point.goal_point_provenance(),
    # which is written into the run's config.json rather than left in a docstring.
    goal_point_inject: bool = False
    goal_point_cfg: gp.GoalPointConfig = field(default_factory=gp.GoalPointConfig)
    # E14 — echo-quotiented goal supervision: g_tac = ha0_ext + delta, with a
    # ZERO-INIT delta head, so the model STARTS at the kinematic extrapolation
    # and every learned parameter is spent on what the ego state cannot
    # explain. Emits `g_tac_delta_absmean` / `echo_base_absmean` every step, so
    # "is it echoing?" is read off the LOG rather than inferred from an eval
    # three days later (the Caveat-B discipline applied to the echo).
    echo_base: bool = False

    @property
    def n_goal_taus(self) -> int:
        return len(self.goal_tau_steps)

    @property
    def goal_tau_seconds(self) -> tuple[float, ...]:
        """``goal_tau_steps`` in SECONDS at the binding 10 Hz (DT = 0.1).

        The E14 base is integrated in seconds while the taus are stored in
        10 Hz steps; deriving the conversion here (rather than at each call
        site) is the C-class rule that a derived constant is re-derived, never
        re-typed — a hardcoded 2/4/6 would silently become a different
        experiment the moment `goal_tau_steps` moves."""
        return tuple(float(s) * 0.1 for s in self.goal_tau_steps)


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
    # ⭐⭐ HIERARCHY REBALANCE (PI 2026-09-02, "very important"): the STRATEGIC
    # width is matched to the other arms so a cross-arm hierarchy comparison
    # attributes to DESIGN, not to CAPACITY.
    #   flagship v7 (train_v6_staged defaults):  d_tac 512 | d_str 256
    #   refav1 (RefAV1Config + brains):          d_model 512 / d_intent 256
    #                                            | d_ctx 256, str_dim 256
    #   refc v3 BEFORE this change:              d_tac 512 | d_ctx 64  <- 4x narrow
    # The tactical side already matched at 512; the strategic side did not, and
    # a 4x-narrower strategic context is a capacity confound sitting inside the
    # exact claim v3 exists to test (that the hierarchy earns its place).
    # ⚠️ `refc_v3_sized_config` moves the ENCODER ONLY, so this width is the
    # same at small/base/xl — the rebalance is a property of the arm, not of a
    # rung, which is what makes it comparable across the ladder too.
    cfg.strategic = refc.StrategicCtxConfig(hidden=512, d_ctx=256)
    cfg.factored_maneuver = True      # action space: both arms, never a lever
    # ⚠️ WHAT THIS FLAG MEANS CHANGED WHEN E11' LANDED, AND THE OLD COMMENT
    # ("goal path stays vision-pure") IS NOW FALSE IN A v4 BUILD. It is kept
    # False in BOTH v3 and v4, but for a different reason in each:
    #   v3: it IS the E11 refusal -- no ego reaches any goal head at all.
    #   v4: the ego reaches the hierarchy's goal path through its OWN embedding
    #       (`ego_state_inject`), so leaving this False keeps the CORE's own
    #       pooled aux lat/lon heads ego-free -- a control that sits inside the
    #       same arm -- and keeps the registered v4 delta at three keys instead
    #       of silently widening a second head's input.
    cfg.tactical_speed_input = False
    cfg.sel_reach_clamp = True        # precondition, measured inert on ADE
    # ⛔ AND `sel_accel_max` IS NOW RE-DERIVED AT THE PLANNED HORIZON, not
    # inherited. `refc.py:652` sets `horizon_s = max(horizons)*0.1`, so at
    # V3_HORIZONS the band is a*6.0. MEASURED 2026-09-04 on the refcv4
    # data-driven vocabulary, 19,602 held-out eval + 635,331 train windows:
    #   a=2.5 (the 2 s default) -> +-15.0 m/s, kills 18.02 %  <- near-vacuous
    #   a=2.0                   -> +-12.0 m/s, kills 26.23 %  <- the pick
    #   a=1.5                   -> +- 9.0 m/s, kills 38.18 %
    # The binding criterion is GT DELETION, not kill rate: a band that removes
    # the trajectory the ego actually flew is wrong, not conservative. At a=2.0
    # it deletes 0.000 % of eval and 0.007 % of train ground truth and moves the
    # oracle-in-vocabulary ADE by +0.00000 m. The trainer pins it through
    # `--sel-accel-max` (applied to BOTH arms, so the hier/flat delta is
    # unchanged). ⚠️ The 72.08 % / 77.28 % / 3.58x figures elsewhere in this
    # file are 2 s statistics and are NOT reproduced at 6 s -- see the module
    # docstring's standing warning, which this block is the discharge of.
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

#: ⭐ THE VALIDATION RIG RUNG — **deliberately NOT in** :data:`V3_SIZES`.
#:
#: ``TanitAD_ValidateAIDesign`` §2 requires a design change to be validated on a
#: tiny rig (~19 M, ~29 min/arm on the dev box) BEFORE it earns compute. REF-C's
#: smallest REGISTERED rung is 63 M and does not fit that budget on the dev
#: box's 8,187 MiB RTX 4060, so v4's gate needed a rung of its own.
#:
#: MEASURED by building: ``tiny`` = **16,989,725** params (``small`` 63,158,525
#: at the same v7.0 heads) — inside the skill's band, and it moves the
#: **ENCODER ONLY**, exactly like every registered rung.
#:
#: ⛔⛔ WHY IT IS A SEPARATE DICT AND NOT A FOURTH ENTRY ABOVE. ``V3_SIZES`` is
#: read by ``refc_v3_scale_matrix`` and by the D-008 package decision, and
#: ``tests/test_refc_v3_scale_matrix.py`` asserts *"the hierarchy cost is near
#: constant across the ladder"* (spread < 10 %) — an assertion about the
#: REGISTERED ladder whose ``aligned`` adapter tracks the encoder's ``feat_dim``.
#: Adding a 2x-narrower encoder to that dict would break a guard that exists to
#: catch exactly this kind of quiet geometry drift, and "the test went red so I
#: widened the band" is how a guard stops guarding. The rig rung is therefore
#: reachable by name and invisible to the ladder.
#:
#: ⛔ NOTHING MEASURED AT THIS RUNG IS A MODEL CLAIM. It validates the DESIGN
#: (does the wiring fire, does the gate fail what it must fail) and never the
#: architecture's quality. No registry row may cite it.
V3_RIG_SIZES: dict[str, tuple[int, tuple[int, ...]]] = {
    "tiny": (32, (2, 2, 4, 2)),
}


#: ⭐⭐ THE v4 LEVER SET — pinned, not asserted (the C122 rule). The launch
#: preflight refuses unless ``config_delta(cfg_v4, cfg_v3)`` equals EXACTLY
#: this. `core.ego_valid_channel` is in the set ON PURPOSE and is NOT a hidden
#: second lever: admitting four channels whose withheld state is
#: indistinguishable from a real physical state would multiply the X15 defect,
#: so the flag is a PRECONDITION of the lever — the same relationship
#: `graft_target_latent` has to `hier` in refcv3's own registered delta.
REGISTERED_DELTA_KEYS_V4: frozenset = frozenset({
    "ego_state_inject", "echo_base", "core.ego_valid_channel",
})


def refc_v4_config(size: str = "small", *, hier: bool = True,
                   echo_base: bool = True) -> RefCV3Config:
    """⭐⭐ REF-C v4 — v3 at the same rung, plus E11' and (by default) E14.

    ``config_delta(refc_v4_config(s), refc_v3_sized_config(s))`` is
    :data:`REGISTERED_DELTA_KEYS_V4` exactly — asserted by
    ``tests/test_refc_v4.py``, so the "everything else identical" claim of
    ``PREREG_REFC_V4.md`` is DERIVED from the dataclasses rather than believed.

    ``echo_base=False`` builds the one-variable intermediate arm ``A-ego``
    (ego admitted, no echo quotient), which is what makes v3 -> A -> v4 two
    single-variable steps instead of one bundle.
    """
    cfg = refc_v3_sized_config(size, hier=hier)
    cfg.ego_state_inject = True
    cfg.echo_base = bool(echo_base)
    cfg.core.ego_valid_channel = True        # X15 — precondition, not option
    return cfg


def refc_v3_sized_config(size: str = "small", *, hier: bool = True
                         ) -> RefCV3Config:
    """v3 at one rung of the REF-C size ladder.

    ⚠️ ``size`` moves the ENCODER ONLY. The decoder, anchors, horizons, tactical
    factoring and reach clamp are ``_v3_core_base``'s and do not vary — so a
    small-vs-XL comparison attributes to scale, not to a config bundle. That is
    the C122 lesson applied to the size axis.

    ⭐ ``size`` may also name a rung of :data:`V3_RIG_SIZES` (``tiny``) — the
    validation-rig geometry. It is resolved here so the rig runs through the
    SAME builder as every registered rung (one implementation), but it is not a
    member of the registered ladder and nothing measured at it is a model claim.
    """
    table = (V3_SIZES if size in V3_SIZES
             else V3_RIG_SIZES if size in V3_RIG_SIZES else None)
    if table is None:
        raise ValueError(
            f"size must be one of {sorted(V3_SIZES)} "
            f"(or the validation rig rung {sorted(V3_RIG_SIZES)}), got {size!r}")
    bw, blocks = table[size]
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
        # ⭐⭐ S7 / E15 (GP-1) — THE RANKED-SCORE SEAM'S SLOT AND SCALE ARE
        # DERIVED HERE, BEFORE THE CORE IS BUILT, AND NEVER TYPED.
        #
        # ⛔ The failure this closes is not hypothetical, it is the derived-
        # constant trap (`HORIZON = round(6.0 * 10 / STRIDE)`): a hand-set
        # `gp_slot` compares the goal at t = 4 s against the anchor at some
        # OTHER time, and that arm trains, converges, and reports a plausible
        # number. Deriving it from THIS core's OWN horizons and THIS config's
        # OWN `t_goal_s` makes the two un-desyncable, and `goal_slot_index`
        # raises when the time is not a slot at all.
        if cfg.goal_point_inject:
            cfg.core.gp_slot = gp.goal_slot_index(
                cfg.goal_point_cfg.t_goal_s, cfg.core.trajectory.horizons)
            cfg.core.gp_scale_m = float(cfg.goal_point_cfg.range_norm_m)
        elif cfg.core.graft_gp_point:
            # The gate would exist with nothing to feed it: `gp_point` stays
            # None, the `r_terms` entry is skipped on every forward, and the
            # arm reads as "the geometric goal prior does nothing" while never
            # having had a goal. Refuse rather than manufacture that negative.
            raise ValueError(
                "core.graft_gp_point=True needs goal_point_inject=True — the "
                "S7 ranking term is fed by the E15 head's own prediction "
                "through the hierarchy hook. Without the head the gate is "
                "built, the term is never appended, and the arm would report "
                "the seam as inert while it was never wired.")
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

        # E13 — nav into the TACTICAL and STRATEGIC states (PI 2026-09-02).
        # Its own embedding, never the core's: the core's nav path is a
        # measurement-encoder condition and sharing the table would couple two
        # unrelated conditioning surfaces through one gradient. ZERO-INIT
        # projections keep the arm bit-identical to a nav-less v3 at step 0, so
        # the delta this edge buys is attributable to training, not to init.
        # ⭐⭐ E15 — THE PREDICTED METRIC GOAL POINT. Structurally the E13 nav block
        # with the 4-row categorical table replaced by an MLP over a METRIC vector, and
        # deliberately so: same sites, same additive form, same zero-init, so the ONE
        # variable between the arms is the TYPE of the conditioning signal.
        # ⛔ The head is a bare Linear on purpose. Giving the goal more capacity than
        # E13's embedding would make the comparison a CAPACITY comparison (C34: match
        # capacity before attributing an effect to information).
        if cfg.goal_point_inject:
            self.gp_head = gp.GoalPointHead(cfg.core.strategic.d_ctx)
            self.gp_cond = gp.GoalPointConditioning(
                cfg.goal_point_cfg.d_goal, cfg.d_tac, cfg.core.strategic.d_ctx)
            # DERIVED from THIS core's horizons, not from the module-global
            # V3_HORIZONS: a build whose trajectory horizons were resized would
            # otherwise carry a slot that indexes a bank it does not have.
            self.gp_slot = gp.goal_slot_index(
                cfg.goal_point_cfg.t_goal_s, cfg.core.trajectory.horizons)
        else:
            self.gp_head = None
            self.gp_cond = None
            self.gp_slot = None

        if cfg.nav_inject:
            # ctx is the StrategicCtx token the hook receives — d_ctx (64),
            # read from the CORE'S OWN config so a resize cannot desync them.
            d_ctx = cfg.core.strategic.d_ctx
            self.nav_inj = nn.Embedding(cfg.n_nav_commands, cfg.d_nav)
            self.nav_to_tac = nn.Linear(cfg.d_nav, cfg.d_tac)
            self.nav_to_str = nn.Linear(cfg.d_nav, d_ctx)
            for lin in (self.nav_to_tac, self.nav_to_str):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            # ⭐ E13b -- the args ride the SAME embedding by ADDITION, which
            # is `NavConditioner.encode`'s contract verbatim
            # (`embed(token_id) + arg_proj(args)`), not a new one. Following
            # that class rather than importing it is deliberate: its
            # `arg_proj` is `Linear(2, d_embed)` and widening it to 3 would
            # change the shape every v6/v7f checkpoint loads.
            # ⛔ NOT zero-init, and that is the CORRECT reading of the
            # contract: in `NavConditioner` it is the OUTPUT projection
            # (`layer_proj`) that is zeroed, never `arg_proj`. Here
            # `nav_to_tac`/`nav_to_str` ARE the output projections and are
            # already zeroed above, so the whole nav term is exactly 0 at
            # step 0 with or without this line -- bit-identity is already
            # bought, and a second zero would only stall this projection's
            # gradient by one step for nothing.
            self.nav_arg_proj = (nn.Linear(NAV_ARG_DIMS, cfg.d_nav)
                                 if cfg.nav_args_inject else None)
        else:
            self.nav_inj = None
            self.nav_arg_proj = None

        # ⭐⭐ E11' — THE MEASURED EGO STATE INTO THE TACTICAL AND STRATEGIC
        # STATES (PI 2026-09-03). Structurally the E13 nav block one edge over,
        # and deliberately so: its own embedding (never the core's measurement
        # encoder — sharing would couple two unrelated conditioning surfaces
        # through one gradient) and ZERO-INIT projections.
        #
        # ⚠️ WHAT ZERO-INIT DOES AND DOES NOT BUY HERE — stated precisely,
        # because the first draft of this comment OVERCLAIMED and the tiny-rig
        # test caught it. It buys that **the EDGE is bit-inert at init**:
        # within one v4 model, changing `ego_state` leaves `z_tac`, `ctx`,
        # `g_str` and `traj` bit-identical, so any later delta is attributable
        # to TRAINING (`test_T6_the_ego_edge_is_bit_inert_at_init`).
        # ⛔ It does NOT buy that a v4 model equals a v3 model at step 0: the
        # required `core.ego_valid_channel=True` widens `d_meas_in` by one, so
        # the measurement Linear has a different SHAPE and every subsequent RNG
        # draw shifts. Seeding does not fix that, it HIDES it. The two arms are
        # compared by TRAINING them, never by an init identity that is false.
        #
        # ⛔ WHAT THIS DOES NOT RELAX. The situation classifier's output is
        # still refused into every goal node (PI 2026-08-03, UNCHANGED), the
        # LAN corridor is still label-only at inference (E12, UNCHANGED), and
        # ANY future ego quantity is refused — that last one is now the audit's
        # PINNED NEGATIVE EDGE, because E11's negative became a positive and a
        # provenance audit with no refused edge left has no teeth.
        if cfg.ego_state_inject:
            if not cfg.core.ego_valid_channel:
                raise ValueError(
                    "ego_state_inject requires core.ego_valid_channel=True. "
                    "With ego_dropout > 0 and no validity bit, a WITHHELD ego "
                    "channel is byte-identical to a genuine physical zero, and "
                    "the mechanism is DIFFERENT PER CHANNEL — MEASURED, val "
                    "epcache (census_val40.json / zero_probe_val40.json): "
                    "v0 is EXACTLY 0.0 on 11.00 % of frames and its [0, 0.1) "
                    "bin holds 12.63x the next (a hard atom: withheld reads as "
                    "a genuine standstill); curvature is exactly 0.0 on "
                    "0.387 % (reads as a genuinely straight wheel); yaw_rate "
                    "on 11.34 %. "
                    "⚠ a_long is NEVER exactly 0.0 (0.000 % of 7,963 frames) "
                    "— an earlier version of this message claimed 0 was its "
                    "MODE and that was an inherited plausibility, not a "
                    "measurement. It is still unsafe to zero-fill: 0.0 sits at "
                    "the CENTRE of its density (mean -0.1607, std 0.9312), so "
                    "a withheld value reads as an ordinary cruise and the "
                    "model cannot tell 'no reading' from 'not accelerating'. "
                    "Admitting three more channels whose withheld state is a "
                    "confident lie would multiply the X15 defect by four, so "
                    "the flag is a PRECONDITION of the lever, not an option "
                    "beside it.")
            self.ego_inj = nn.Linear(EGO_DIMS, cfg.d_ego)
            self.ego_to_tac = nn.Linear(cfg.d_ego, cfg.d_tac)
            self.ego_to_str = nn.Linear(cfg.d_ego, d_ctx)
            for lin in (self.ego_to_tac, self.ego_to_str):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            self.ego_inj = None
        # E6 — factored tactical decision heads on z_tac (the H arm's decision
        # supplier; the core's own pooled-based heads keep training as the
        # shared aux surface in BOTH arms, so the supervision surface is
        # identical and only the DECISION SOURCE differs).
        # ⭐ v7 mandate (PI 2026-08-27), with the SUPERVISION fact stated:
        # these heads TRAIN against `tac.window_factored_labels` — the
        # KINEMATIC 3x3 — so "kin3" is what today's trainer can supervise;
        # v7.0 (8x8, the FlyWheel space) is the go-forward head and NEEDS
        # v7 labels. The trainer PINS its version and `refc_v3_train`
        # refuses a head/label width mismatch loudly.
        _vv = getattr(cfg, "tac_vocab_version", "kin3")
        if _vv == "kin3":
            _nlat, _nlon = tac.N_LAT, tac.N_LON
        else:
            from tanitad.models.v6 import (tactical_lat_actions,
                                           tactical_lon_actions_v)
            _nlat = len(tactical_lat_actions(_vv))
            _nlon = len(tactical_lon_actions_v(_vv))
        self.tac_vocab_version = _vv
        self.lat_head_tac = nn.Linear(cfg.d_tac, _nlat)
        self.lon_head_tac = nn.Linear(cfg.d_tac, _nlon)
        # ⭐⭐ D-TACGOAL-1 — THE TACTICAL GOAL *TOKEN* HEAD (2026-09-06).
        # The v7 emitter mints a 22-token tactical goal SET on every clip
        # — including the traffic-light COLOUR — and until now nothing
        # was sized on it, so no gradient could reach it. That structural
        # absence is why the model never brakes for a red light and why
        # lane changes never activate: the labels existed, nothing taught
        # them (audit `2026-09-06-label-vocab-audit/AUDIT_RESULT.json`).
        #
        # ⚠️ NOT `tac_goal_head` — that name is TAKEN, by the GEOMETRIC
        # goal regressor two lines below. Two different objects called
        # "the tactical goal head" is how one arm's number gets quoted
        # for the other.
        #
        # ⚠️ It exists ONLY under a v7 vocabulary. `kin3` has no tactical
        # goal vocabulary at all, so building it there would be 22 dead
        # logits that can never be supervised — the same defect
        # `effective_mask` exists to prevent one layer down.
        self.tac_goal_tok_head = None
        if _vv != "kin3":
            from tanitad.models.v6 import TACTICAL_GOAL_VOCAB_VERSIONS
            from tanitad.refs.tac_goal_head import TacGoalTokenHead
            _goal_toks = TACTICAL_GOAL_VOCAB_VERSIONS[_vv]
            self.tac_goal_tok_head = TacGoalTokenHead(
                cfg.d_tac, n_tokens=len(_goal_toks))
            self.tac_goal_tokens = tuple(_goal_toks)
        # E8 — tactical geometric goals, E4.1 layout (x, y, heading, speed)@tau.
        self.tac_goal_head = nn.Linear(cfg.d_tac, k * GOAL_DIMS)
        # ⭐ E14 — under `echo_base` this head predicts the RESIDUAL over the
        # kinematic extrapolation, so it is ZERO-INIT: the model starts exactly
        # AT `ha0_ext` (a known, MEASURED 0.4449 m @2 s) instead of at random,
        # and every parameter it learns is spent on what the ego state cannot
        # explain — which is the scene. ⚠️ This changes the head's init, so a
        # v3 checkpoint must NOT be resumed into an `echo_base` build; v4 is a
        # fresh run and `ckpt_compat` refuses the cross-load.
        if cfg.echo_base:
            nn.init.zeros_(self.tac_goal_head.weight)
            nn.init.zeros_(self.tac_goal_head.bias)
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
    def provenance_roles(self) -> dict:
        """Role map for ``tanitad.eval.goal_provenance.audit_arm``. GOAL nodes:
        g_str, g_tac, goal_point. SITUATION_OUTPUT nodes: NONE IN GRAPH — and
        the audit MEASURES that (positive control on frames, pinned negative
        edge) rather than trusting this declaration.

        ⭐⭐ v4 MOVES THE PINNED NEGATIVE EDGE, AND THE AUDIT GETS STRONGER.
        Under v3 the refused edge was ``v0 -> goal``. E11' admits it, so an
        audit that only ever refused ``v0`` would be left with no teeth at
        exactly the moment new ego plumbing arrives. The replacement is the
        edge the PI's constraint actually names — *"and not the future one"*:

            ⛔ future_poses / future_actions -> ANY goal node : REFUSED

        ⚠️ Why that is the right one, mechanically. The obvious wrong
        implementation of this feature is ``a0 = (future[:, 0, 3] - v0) / dt``.
        It would look identical in a config diff, satisfy every *"does the goal
        use acceleration?"* check, produce a BETTER-looking result — and be a
        future read. Under v3 nothing probed that edge because no ego plumbing
        existed. v4 adds the plumbing, so v4 adds the probe.
        """
        v4 = bool(self.cfg.ego_state_inject)
        refused = ["lan -> inference (E12; label-only)",
                   "situation classifier output -> any goal node "
                   "(PI 2026-08-03, UNCHANGED)"]
        refused.append(
            "future_poses/future_actions -> any goal node (E11'; the PI's "
            "'not the future one', pinned interventionally)" if v4
            else "v0 -> any goal node (E11)")
        return {
            "goal": ["g_str", "g_tac", "goal_point_tac"],
            "situation_output": [],
            "inference_inputs_of_goals": (
                ["frames (via pooled_seq/ctx)",
                 "ego_state @ t0 = (v0, a_long, yaw_rate, curvature, keep) "
                 "— MEASURED at the last OBSERVED frame (E11')"]
                if v4 else ["frames (via pooled_seq/ctx only)"]),
            "required_live_edges": (
                ["ego_state -> {z_tac, g_str, g_tac}", "frames -> every goal"]
                if v4 else ["frames -> every goal"]),
            "refused_edges": refused,
            "shared_trunk": "encoder (common ancestor, declared; zero-init "
                            "gates carry attributability)",
            "_reads": ("`required_live_edges` is not decoration: an arm where "
                       "the SCENE edge does not fire is ECHOING, and an arm "
                       "where the EGO edge does not fire is v3 wearing a v4 "
                       "config. Both are failures and both are measured."),
        }

    # --- the in-forward hierarchy supplier ----------------------------------
    def _hook(self, cache: dict, nav_cmd: Tensor | None = None,
              ego_state: Tensor | None = None,
              nav_args: Tensor | None = None):
        cfg = self.cfg
        # ⭐⭐ STRATEGIC BYPASS (`--no-strategic`, PI 2026-09-06). Read from
        # the CORE config so there is exactly ONE source of truth for the flag
        # across `refc.py` and this file -- a second copy is how two halves of
        # one switch drift apart. `getattr` so an older pickled config without
        # the field still builds (it reads False = today's behaviour).
        bypass = bool(getattr(cfg.core, "no_strategic", False))

        def hook(pooled_seq: Tensor, ctx: Tensor) -> dict:
            b = pooled_seq.shape[0]
            z_tac_raw = self.phi_tac(pooled_seq)                  # [B, d_tac]
            # ⭐ E13 — NAV REACHES THE TACTICAL AND STRATEGIC LAYERS
            # (PI 2026-09-02, BINDING: "No training without feeding nav command
            # to all layers"). Before this, `nav_cmd` went to `self.core` and
            # NOWHERE else: PhiTac and StrategicCtx never saw the route.
            # ⛔ WHAT THIS DOES **NOT** RELAX: E11 still refuses `v0` into every
            # goal node — nav is a ROUTE COMMAND, not ego state, and the
            # admissibility ruling permits a goal/route input while forbidding
            # the situation classifier's output. The intervention audit that
            # pins the v0 edge is unchanged and still runs in the preflight.
            # ⚠️ EVAL OBLIGATION, inseparable from this edge: nav is CONSTANT
            # (`follow`) on ~75-79 % of windows and is `nav_cmd=None -> index 0`
            # at eval — the C6 confound. Any nav-conditioned result carries a
            # NAV-SHUFFLE control, or it is not evidence.
            nav_t = nav_s = None
            if self.nav_inj is not None and nav_cmd is not None:
                e = self.nav_inj(nav_cmd.reshape(-1).long())      # [B, d_nav]
                # ⭐ E13b -- RANGE AND TIME, ADDED TO THE TOKEN'S OWN CODE.
                # `NavConditioner.encode` is `embed(id) + arg_proj(args)`; this
                # is that line with a third, VALIDITY slot. The token still
                # says WHICH way; the args say HOW FAR and HOW SOON, which is
                # the half the published goal-conditioning wins are about.
                if self.nav_arg_proj is not None and nav_args is not None:
                    a = nav_args.to(e.dtype).reshape(e.shape[0], -1)
                    if a.shape[-1] != NAV_ARG_DIMS:
                        raise ValueError(
                            f"nav_args must be [B, {NAV_ARG_DIMS}] = "
                            f"(distance_norm, time_norm, args_valid), got "
                            f"{tuple(nav_args.shape)}. The validity slot is "
                            f"NOT optional: NAV_FOLLOW_ROAD carries empty "
                            f"args and a silent 0.0 says 'the turn is here, "
                            f"now'.")
                    # ⛔ THE BIT GATES THE VALUES, IN THE MODEL, NOT ONLY
                    # IN THE LOADER. X15's rule: the consumer re-applies the
                    # flag so a caller that forgot to zero an invalid row
                    # cannot leak a phantom range. The bit itself always
                    # passes, so "no range known" stays a distinct, learnable
                    # input rather than collapsing onto "0 m away".
                    a = torch.cat([a[:, :NAV_ARG_DIMS - 1]
                                   * a[:, NAV_ARG_DIMS - 1:],
                                   a[:, NAV_ARG_DIMS - 1:]], dim=-1)
                    e = e + self.nav_arg_proj(a)
                nav_t, nav_s = self.nav_to_tac(e), self.nav_to_str(e)
                z_tac_raw = z_tac_raw + nav_t                     # -> tactical
                ctx = ctx + nav_s                                 # -> strategic
            # ⭐⭐ E11' — THE MEASURED EGO STATE REACHES THE GOAL PATH (v4).
            # Additive and ZERO-INIT, exactly like E13 above, so a v4 build is
            # bit-identical to v3 at step 0. `ego_state` is [B, 5] =
            # (v0, a_long, yaw_rate, curvature, keep), every channel read at
            # the LAST OBSERVED frame by `ego_state_at_t0`.
            ego_e = None
            if self.ego_inj is not None and ego_state is not None:
                es = ego_state.to(pooled_seq.dtype)
                keep_b = es[:, 4:5]
                # Scale to ~unit, then RE-APPLY `keep`: a withheld block must
                # be exactly zeros next to a keep bit of 0, never a scaled
                # leftover. The X15 rule is that "withheld" and "genuinely
                # zero" differ in the FLAG, and that only works if the values
                # really are zero when the flag is.
                es = torch.cat([
                    es[:, 0:1] / EGO_SCALE_V, es[:, 1:2] / EGO_SCALE_A,
                    es[:, 2:3] / EGO_SCALE_R, es[:, 3:4] / EGO_SCALE_K,
                ], dim=-1) * keep_b
                ego_e = self.ego_inj(torch.cat([es, keep_b], dim=-1))
                z_tac_raw = z_tac_raw + self.ego_to_tac(ego_e)    # -> tactical
                ctx = ctx + self.ego_to_str(ego_e)                # -> strategic
            # ⭐⭐ E15 — the predicted metric goal point, read off the STRATEGIC
            # context BEFORE its own conditioning is added (a feed-forward residual, not
            # a cycle) and fed back as a CONTINUOUS, per-window, metric signal.
            # ⚠️ At train time the LABEL supervises `g_point`; at inference nothing else
            # is read — the ego's future path never enters the forward.
            g_point = None
            # S-BYPASS-4: the E15 metric goal point is read OFF `ctx`, so it
            # is a STRATEGIC readout and a "goal constraint" in exactly the
            # sense the PI's directive names. Bypassed with the rest of the
            # layer. (Inert on refcv4b, whose argv carries no goal-point flag.)
            if self.gp_cond is not None and not bypass:
                g_point = self.gp_head(ctx)                       # [B, 2] normalised
                feats = torch.cat(
                    [g_point.clamp(-cfg.goal_point_cfg.xy_clip,
                                   cfg.goal_point_cfg.xy_clip),
                     torch.ones_like(g_point[:, :1])], dim=-1)    # [B, 3], valid = 1
                gp_t, gp_s = self.gp_cond(feats)
                z_tac_raw = z_tac_raw + gp_t                      # -> tactical
                ctx = ctx + gp_s                                  # -> strategic
            g = self.str_goal_head(ctx)                           # [B, 3]
            bearing = g[:, :2] / torch.linalg.vector_norm(
                g[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
            g_str = torch.cat([bearing, torch.tanh(g[:, 2:3])], dim=-1)
            # E4: strategic goal conditions tactical.
            # ⭐ CAVEAT-A LEVER (PI 2026-09-02). The `.detach()` here and on E7
            # is the v6 `_cut()` discipline: it buys clean ATTRIBUTION (each
            # level trains only by its own supervision) at the cost of the
            # hierarchy never being OPTIMISED as one — the strategic goal head
            # is trained by its hindsight label, never by whether its goal
            # helped the trajectory. `uplink_grad=True` opens that path; it is
            # design item D-4, a pre-registered lever, not a new invention.
            # ⛔ E9's goal detach is NOT covered by this flag and stays hard —
            # it is a different mechanism (the winner's-curse firewall: letting
            # selection train the goal toward the fan is the failure SEL-1 was
            # refused for). Two detaches, two reasons, one flag.
            # ⭐⭐ S-BYPASS-2 -- THE TACTICAL SEAM. This FiLM is the ONLY
            # in-graph consumer of `g_str` (every other reference is the cache,
            # the aux loss or the intervention audit), so skipping it is what
            # makes "the strategic layer sets no goal constraint for tactical"
            # TRUE rather than merely intended.
            # ⛔ `g_str` is still COMPUTED and still cached: the strategic
            # head's opinion stays visible in the record -- which is the whole
            # reason to prefer a bypass over a deletion, since the route head's
            # kappa 0.4852 is exactly the quantity this arm is testing.
            # ⚠ The `else` branch below is the pre-flag code VERBATIM, so with
            # the flag OFF this block is byte-identical to today.
            if bypass:
                z_tac = z_tac_raw
            else:
                g_down = g_str if cfg.uplink_grad else g_str.detach()
                gcond = self.gstr_embed(g_down)
                gamma, beta = self.gstr_film(gcond).chunk(2, dim=-1)
                z_tac = z_tac_raw * (1.0 + gamma) + beta          # zero-init
            lat = self.lat_head_tac(z_tac)
            lon = self.lon_head_tac(z_tac)
            # ⛔⛔ DEFECT A — RESOLVED 2026-09-04 (option (a), pre-registered).
            # `derive_man5_logprobs` is DEFINED on [B, 3] x [B, 3] and indexes
            # the LAT_/LON_ constants POSITIONALLY. Fed the 8-wide v7 heads it
            # did not raise: it silently read the WRONG classes — `turn_left`
            # <- LANE_CHANGE_L, `turn_right` <- LANE_CHANGE_R, `accelerate` <-
            # YIELD_MERGE, `brake_stop` <- FOLLOW, and 10 of the 16 classes were
            # never read at all — and still returned a valid-looking
            # distribution, which then reweighted `refc.py:1405`, the LIVE H19
            # anchor prior. It ran that way for all of refcv3 and for refcv4's
            # first 6,400 steps.
            #
            # ⇒ The push-forward is now called ONLY on the kin3 vocabulary its
            # positional contract is defined on. Under a v7 vocabulary the hook
            # supplies NO `maneuver_logits`, so `refc.py` falls back to the
            # CORE's own 3-wide kin3-derived 5-way (`refc.py:2115-2117`, heads
            # sized `N_LAT_MAN`/`N_LON_MAN` = 3) — the documented intent, and no
            # invented 8->5 mapping is shipped.
            #
            # ⚠️ STATED HONESTLY, because it is a real reduction: under a v7
            # vocabulary the TACTICAL BRAIN no longer drives the anchor prior.
            # H19 stays live but is fed by the core's aux head, so the tactical
            # level reaches the decoder through E7 (`target_latent`) and E9
            # (goal selection) only. That is the price of removing a scrambled
            # input rather than adding an unvalidated mapping.
            man5 = (tac.derive_man5_logprobs(lat, lon)
                    if self.tac_vocab_version == "kin3" else None)
            g_delta = self.tac_goal_head(z_tac).reshape(
                b, cfg.n_goal_taus, GOAL_DIMS)
            # ⭐⭐ E14 — ECHO-QUOTIENTED GOAL SUPERVISION. The head predicts the
            # RESIDUAL over the kinematic self-extrapolation, so the trivial
            # solution is FREE and gradient descent has no incentive to re-derive
            # it. `echo_base` is multiplied by `keep`, which makes the withheld
            # regime (keep=0) an absolute-prediction regime — i.e. refcv4 trains
            # its OWN vision-only arm, and the eval reads it at keep=0 with no
            # second run and no second config.
            #
            # ⚠️ STATED HONESTLY: this does NOT make echoing impossible. A lazy
            # model outputs delta ~ 0 and scores exactly ha0_ext. What E14 buys
            # is that this outcome is VISIBLE FROM STEP 1 and attributable —
            # `g_tac_delta_absmean / echo_base_absmean` is the readout. The
            # thing that CATCHES echoing is the gate (`tanitad.eval.echo_gate`);
            # E14 is what makes its verdict interpretable. Both ship.
            echo_base = None
            if cfg.echo_base and ego_state is not None:
                es_raw = ego_state.to(z_tac.dtype)
                echo_base = kinematic_goal_extrapolation(
                    es_raw[:, 0], es_raw[:, 1], es_raw[:, 3],
                    cfg.goal_tau_seconds) * es_raw[:, 4].reshape(-1, 1, 1)
                g_tac = echo_base + g_delta
            else:
                g_tac = g_delta
            # ⭐ D-TACGOAL-1: the 22-token goal-SET logits ride the same
            # `z_tac` the two action heads read, so a goal and the action
            # that serves it are predicted from ONE tactical latent.
            if self.tac_goal_tok_head is not None:
                cache["tac_goal_logits"] = self.tac_goal_tok_head(z_tac)
            cache.update(z_tac=z_tac, g_str=g_str, g_str_raw=g,
                         lat_logits_tac=lat, lon_logits_tac=lon,
                         g_tac=g_tac, g_tac_delta=g_delta,
                         ego_injected=bool(ego_e is not None),
                         nav_injected=bool(nav_t is not None),
                         goal_point=g_point,
                         goal_point_injected=bool(g_point is not None))
            if echo_base is not None:
                cache.update(
                    echo_base=echo_base,
                    echo_base_absmean=echo_base.detach().abs().mean(),
                    g_tac_delta_absmean=g_delta.detach().abs().mean())
            # E6 live (the H19 seam is live-from-step-0 by design); E7 detached
            # unless the Caveat-A lever is open.
            z_up = z_tac if cfg.uplink_grad else z_tac.detach()
            hook_out = {"target_latent": self.tac_latent_proj(z_up)}
            if g_point is not None:
                # ⛔ Emitted, not yet CONSUMED: the param-free geometric ranking term
                # (`goal_point.anchor_goal_prior_at_time`) needs one gated `r_terms`
                # entry in `refc.py`'s ranked-score block, which is a different owner's
                # file. Shipping the value here means that patch is one line and needs
                # no second forward. Named as an integration item in PREREG.md §8.
                hook_out["goal_point"] = g_point
            if man5 is not None:
                hook_out["maneuver_logits"] = man5
            # ⭐ H-EGO-LIT-4: the model's OWN 2 s speed, DETACHED, for the
            # decoder's `pred` withheld-bank mode (`refc.py`). On a withheld
            # row `g_tac` is a function of the image, nav and constants only:
            # the ego block is multiplied by keep = 0 before `ego_inj` (above)
            # and `tactical_speed_input` is False, so the measured v0 cannot
            # reach the bank through it. Emitted on EVERY forward and stored
            # in the output; the decoder ignores it unless its mode is "pred".
            bank_speed_pred = g_tac[:, self._tau_slot_2s(), 3].detach()
            cache["bank_speed_pred"] = bank_speed_pred
            hook_out["bank_speed_pred"] = bank_speed_pred
            return hook_out

        return hook

    def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,
                v0: Tensor | None = None, steps: int = 0,
                lan: Tensor | None = None,
                nav_known: Tensor | None = None,
                ego_state: Tensor | None = None,
                withheld_speed: Tensor | None = None,
                gp_point: Tensor | None = None,
                gp_valid: Tensor | None = None,
                agent_gt: dict | None = None,
                nav_args: Tensor | None = None) -> dict:
        """``ego_state`` is the v4 block ``[B, 5]`` from :func:`ego_state_at_t0`
        — (v0, a_long, yaw_rate, curvature, keep) at the LAST OBSERVED frame.

        ⭐ ``gp_point`` [B, 2] / ``gp_valid`` [B] (E15, S7): an EXTERNALLY
        SUPPLIED metric goal point, in the SAME normalised units the head
        emits, overriding the model's own prediction on the ranked-score seam.
        This exists for exactly one purpose — PREREG §2's five eval-time
        interventions (mirror ``y -> -y``, range ``x0.5``, straight, shuffled,
        withheld) — so the whole panel rolls in ONE process on ONE surface
        instead of five monkeypatched copies of the harness.
        ⛔ It is a DIAGNOSTIC PORT, never a training input: the deployable arm
        predicts its own goal from vision, and a run that fed this from a label
        would be supplying a route, which is optimistic by construction on
        PhysicalAI.

        ⛔ Fails loud when supplied to a build that would silently drop it: a
        measured ego block quietly discarded is exactly the class of bug the
        E13 nav seam and the X15 flag exist to remove, and it would look like
        "the ego channels do not help" in a result table.
        """
        # ⛔ refcv5 WP-6, the SAME rule as the `ego_state` guard below: a
        # privileged tensor that is SILENTLY DROPPED reads as "agent tokens do
        # not help" in a result table, which is a refutation manufactured by a
        # wiring gap. Both directions refuse.
        _ag = getattr(self.cfg.core, "agents", None)
        if agent_gt is not None and (_ag is None or not _ag.enable):
            raise ValueError(
                "agent_gt was supplied but this build has no agent seam "
                "(`core.agents` is off) -- it would be SILENTLY DROPPED and "
                "the arm would report as +agents while running without them. "
                "Pass --agents oracle/head, or stop passing agent_gt.")
        if agent_gt is None and _ag is not None and _ag.enable and _ag.oracle:
            raise ValueError(
                "this build is `--agents oracle` but no agent_gt reached the "
                "forward. The oracle's tokens ARE the ground-truth boxes; "
                "with none the seam emits nothing and the arm would read as "
                "'agent tokens do not help' while never having had any.")
        if ego_state is not None and not self.cfg.ego_state_inject:
            raise ValueError(
                "ego_state was supplied but cfg.ego_state_inject is False — it "
                "would be SILENTLY DROPPED and the arm would report as a v4 "
                "while running v3. Turn the lever on or stop passing it.")
        if ego_state is not None and ego_state.shape[-1] != EGO_DIMS:
            raise ValueError(f"ego_state must be [B, {EGO_DIMS}] = (v0, "
                             f"a_long, yaw_rate, curvature, keep), got "
                             f"{tuple(ego_state.shape)}")
        # ⭐ ONE WITHHOLDING DRAW, ONE OWNER (E11'/X15). In training v4 draws
        # `keep` HERE and hands the SAME vector to both consumers: the goal
        # path (through `ego_state[:, 4]`) and the core's measurement encoder
        # (through the `ego_keep` seam). `refc.py` warns that a second,
        # unsynchronised dropout is the wrong fix, and it is right — so there
        # is exactly one, at the level where both consumers are visible.
        ego_keep = None
        if ego_state is not None:
            ego_state = ego_state.clone()
            if self.training and self.cfg.core.ego_dropout > 0:
                k = (torch.rand(ego_state.shape[0], device=ego_state.device)
                     >= self.cfg.core.ego_dropout).to(ego_state.dtype)
                ego_state[:, 4] = ego_state[:, 4] * k
            ego_keep = ego_state[:, 4]
        # ⛔ SAME REFUSAL AS `ego_state` ABOVE, AND FOR THE SAME REASON: a
        # supplied goal point that this build has no seam for would be SILENTLY
        # DROPPED, and the intervention arm (`gp_geo_mirror`, `_rng05`, …)
        # would read as "corrupting the goal costs nothing" — i.e. it would
        # manufacture a FAILED value-sensitivity gate out of a wiring gap,
        # which is the one outcome that must never be produced by accident.
        if gp_point is not None and not self.cfg.core.graft_gp_point:
            raise ValueError(
                "gp_point was supplied but this build has no S7 seam "
                "(`core.graft_gp_point` is False) — it would be SILENTLY "
                "DROPPED and the intervention would read as 'the goal value "
                "does not matter', which is exactly the PREREG §4 gate "
                "failing for the wrong reason. Build with "
                "--goal-point-geo-prior, or stop passing gp_point.")
        # ⛔ SAME REFUSAL AS `ego_state` AND `gp_point` ABOVE, AND FOR
        # THE SAME REASON. A build without the E13b seam would SILENTLY DROP
        # the range and the arm would read as "the nav args do not help"
        # while never having had them -- the exact shape of the finding this
        # seam exists to overturn (refav1 binds them to `_args` and never
        # uses them again; refcv3 never read them at all).
        if nav_args is not None and not self.cfg.nav_args_inject:
            raise ValueError(
                "nav_args was supplied but this build has no E13b seam "
                "(`cfg.nav_args_inject` is False) - it would be SILENTLY "
                "DROPPED and the arm would report as +nav-args while running "
                "on the bare 3-way token. Build with --nav-args, or stop "
                "passing nav_args.")
        if nav_args is None and self.cfg.nav_args_inject \
                and nav_cmd is not None:
            raise ValueError(
                "this build is --nav-args but no nav_args reached the "
                "forward while a nav token did. Refusing rather than "
                "defaulting: a silent zero would say 'the turn is here, now' "
                "on every window and the channel would be measured as noise.")
        if not self.cfg.hier:
            return self.core(frames, nav_cmd, v0, steps=steps, lan=lan,
                             nav_known=nav_known, ego_keep=ego_keep,
                             withheld_speed=withheld_speed,
                             gp_point=gp_point, gp_valid=gp_valid,
                             agent_gt=agent_gt)
        cache: dict = {}
        out = self.core(frames, nav_cmd, v0, steps=steps, lan=lan,
                        nav_known=nav_known, ego_keep=ego_keep,
                        hierarchy_hook=self._hook(cache, nav_cmd, ego_state,
                                                  nav_args),
                        withheld_speed=withheld_speed,
                        gp_point=gp_point, gp_valid=gp_valid,
                        agent_gt=agent_gt)
        # the per-row withholding draw, for diagnostics that split kept from
        # withheld rows (the trainer's `withheld_speed_mae`); `ego_keep_frac`
        # below is its mean.
        if ego_keep is not None:
            out["ego_keep"] = ego_keep.detach()
        # ---- E9: goal selection over the emitted fan (post-decoder) --------
        fan = out["anchor_traj"]                                  # [B, N, S, 2]
        b = fan.shape[0]
        # 2 s slot: the slot the admission curve was measured at. The goal is
        # DETACHED into selection (the winner's-curse firewall: selection can
        # never train the goal toward the fan).
        g2 = cache["g_tac"][:, self._tau_slot_2s(), :2].detach()  # [B, 2]
        sc = self.scorer(fan[:, :, SEAM_SLOT:SEAM_SLOT + 1],
                         cache["z_tac"].detach(), goal_point=g2)
        # ⭐ CAVEAT-B INSTRUMENTATION (PI 2026-09-02). The gate is zero-init and
        # must LEARN to open; if it never does, E9 contributed exactly nothing
        # and the run cannot claim goal-selection. It is NOT structurally
        # stuck — d(graft)/d(gate) = score != 0, so gradient reaches it — and a
        # 0.0000 reading early is expected under the 2000-step LR warmup (lr
        # was 5e-8 at step 1, MEASURED). ⇒ the honest instrument is to EMIT the
        # gate and the score scale every step, so "did it open" is read off the
        # log at 30k instead of inferred from a 14-step glance (which is the
        # mistake that produced this caveat in the first place).
        graft = self.goal_gate * sc["score"]                      # [B, N]
        out["goal_gate_value"] = self.goal_gate.detach()
        out["goal_score_absmean"] = sc["score"].detach().abs().mean()
        # ⭐ THE ANTI-ECHO READOUT. `g_tac_delta_absmean / echo_base_absmean` is
        # "how much goal does VISION contribute beyond the ego echo". It is
        # emitted every step for exactly the Caveat-B reason: a 0.0000 read at
        # 30k must be a LOGGED fact, not a conclusion inferred three days later
        # from an eval — that inference is what produced Caveat-B in the first
        # place.
        if "echo_base_absmean" in cache:
            eb = cache["echo_base_absmean"].clamp_min(1e-9)
            out["echo_ratio"] = cache["g_tac_delta_absmean"] / eb
        # ⭐ THE REALISED WITHHOLDING RATE, per step. `ego_dropout` is a
        # CONFIGURED probability; this is what the batch actually got. They
        # differ whenever the draw is not where you think it is — and the
        # X15 measurement (a withheld zero being indistinguishable from a
        # genuine standstill) is the reason we may not infer one from the
        # other. In eval() no draw happens and this reads exactly 1.0, which
        # is itself the train/eval asymmetry made visible in the log.
        if ego_keep is not None:
            out["ego_keep_frac"] = ego_keep.detach().float().mean()
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
        # ⭐ E11' ego injection — accounted EXPLICITLY, for the same reason E13
        # is: `test_param_breakdown_smoke_sums` asserts the lines sum to the
        # total, so an unaccounted module is a TEST FAILURE rather than a
        # silent capacity confound inside the exact claim v4 exists to test.
        if getattr(model, "ego_inj", None) is not None:
            out["ego_inject"] = (cnt(model.ego_inj) + cnt(model.ego_to_tac)
                                 + cnt(model.ego_to_str))
        # ⭐ E13 nav injection — accounted EXPLICITLY. Caught by
        # `test_param_breakdown_smoke_sums`, whose sum-equals-total assertion
        # fired the moment these parameters existed but had no ledger line: a
        # capacity ledger that silently under-reports is worse than none,
        # because the prereg quotes it as the arm's cost.
        if model.nav_inj is not None:
            out["nav_inject"] = (cnt(model.nav_inj) + cnt(model.nav_to_tac)
                                 + cnt(model.nav_to_str))
            # E13b rides the same ledger line's rule: parameters that exist
            # and are not accounted break `test_param_breakdown_smoke_sums`.
            if getattr(model, "nav_arg_proj", None) is not None:
                out["nav_inject"] += cnt(model.nav_arg_proj)
        # ⭐ D-TACGOAL-1 — the 22-token goal-SET head, accounted EXPLICITLY and
        # SEPARATELY from `tac_heads`. ⚠️ It is deliberately its own line: the
        # prereg quotes this ledger as the arm's capacity cost, and the arm's
        # whole question is what the goal-SET head buys, so folding it into the
        # ACTION heads' line would hide exactly the number being tested.
        # ⛔ `test_param_breakdown_smoke_sums` fired the moment these parameters
        # existed without a line — as it did for E11' and E13 before them — and
        # that is the check working, not a nuisance.
        if getattr(model, "tac_goal_tok_head", None) is not None:
            out["tac_goal_tok_head"] = cnt(model.tac_goal_tok_head)
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
