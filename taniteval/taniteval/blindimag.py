"""TanitEval — BLIND-IMAGINATION driving: the horizon sweep and its controls.

THE ONE FACT THIS MODULE IS BUILT ON, AND IT WAS VERIFIED IN THE SOURCE
----------------------------------------------------------------------
``tanitad.models.metric_dynamics.rollout_decode`` advances its latent window by
appending **the model's own predicted latent**::

    win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)   # :241

**No frame is ever encoded after the initial window.** So every number the
program calls a "grounded operative rollout" — ``taniteval.rollout.collect``'s
``ade_0_2s``, ``train_flagship_v4.canary_rollout``'s ``wm_canary_ade_2s``,
``train_flagship_v16.canary_rollout`` — is ALREADY a blind-imagination drive.
The program has been measuring the PI's question all along and only ever reading
it at ``k = 20`` (2 s) under the expert's TRUE future actions.

This module does not rebuild that. It **generalises** it along the two axes that
were frozen, and adds the control that turns it into an experiment:

  * ``k``            already free in ``rollout_decode`` — swept here to 185.
  * ``state_source`` what gets appended to the latent window each step.
  * ``action_source`` where the action fed to the predictor comes from.

``blind_rollout(state_source="imagination", action_source="true_future")`` is
**bit-identical** to ``rollout_decode`` — asserted in
``taniteval/tests/test_blindimag.py::test_imagination_is_bit_identical_to_rollout_decode``.
That equivalence is the certification (M3): if it ever breaks, every arm below
is measuring something other than the program's own instrument.

THE FOUR ARMS (``state_source``)
--------------------------------
``imagination``   (a) the predictor's own ``z_hat`` is appended. THE THING UNDER
                  TEST. Identical to ``rollout_decode``.
``frozen_last``   (b) the encoding of the LAST REAL FRAME is appended, every
                  step — "the world stopped". **THE CRITICAL CONTROL.** The
                  decode path, the actions and the predictor call are byte-for-
                  byte those of (a); the ONLY difference is which latent enters
                  the window. If (a) does not beat (b), the world model's
                  *dynamics* contribute nothing blind and one may as well hold
                  the last percept. This is the analogue of H2's
                  random-at-matched-rate control.
``full_obs``      (c) the encoding of the TRUE next frame is appended — a real
                  frame every step. THE CEILING.
``observed_pair`` (c2) diagnostic ceiling: the step readout decodes
                  ``(z_true_t, z_true_{t+1})`` — no prediction at all, pure
                  latent visual odometry. Separates readout error from predictor
                  error. Not one of the four arms; reported as a diagnostic.

Constant velocity — arm (d), the floor — needs no rollout and is
:func:`cv_dense_path`.

THE ACTION REGIMES (``action_source``)
--------------------------------------
``true_future``   the expert's logged future actions. This is what the existing
                  canary uses. **A PRIVILEGED UPPER BOUND, not deployable
                  capability**, and it is worded that way everywhere.
``own_kinematic`` the model's OWN action, derived from its OWN decoded motion by
                  the exact inverse of the corpus's own steer definition
                  (:func:`kinematic_action_from_dpose`). **The deployable
                  condition** — this is the one that answers the PI.
``gt_kinematic``  the same inverse applied to the TRUE per-step Δpose. This is
                  the **convention control**: it isolates how much of any
                  own-action penalty is the inverse map rather than the model.
                  Without it, an own-action result cannot be attributed.
``hold_last``     zero-order hold of the last observed action — ``rollout_decode``'s
                  own ``future_actions=None`` branch. A no-policy floor.

THE ACTION FILTERS (``action_source`` modifier suffix) — T_BLIND RUNG 1
-----------------------------------------------------------------------
``T_blind`` under the model's own actions is **25 steps**; under a zero-order hold
of the last observed action it is **115** (matched comparators, `str` readout,
`…/2026-07-26-tblind-ladder/artifacts/rung0c_matched_tblind.json`). The 90-step
gap is a property of the ACTION FED BACK, not of the weights — so it is
attackable with **no retraining** by filtering that tensor. A filter is declared
as a suffix on ``action_source`` so that every driver, dump and meta block
carries it verbatim with no change to the sweep machinery::

    "own_kinematic"                  the deployed inverse, UNFILTERED (default)
    "own_kinematic|blend=0.25"       a_fed = (1-a)*a_own + a*a_hold0
    "own_kinematic|ema=0.8"          a_fed = b*a_fed_prev + (1-b)*a_own
    "own_kinematic|every=5"          recompute every 5 steps, zero-order hold between
    "own_kinematic|steer_clip=0.02"  tighter steer band (rad)
    "own_kinematic|accel_clip=0.3"   tighter accel band (m/s^2)
    "own_kinematic|chan=steer"       steer from the model, accel held (diagnostic)
    "own_kinematic|own_before=20"    own actions for 20 steps, then held (diagnostic)
    "own_kinematic|own_after=20"     held for 20 steps, then own (diagnostic)

``a_hold0`` is the last OBSERVED action — the same constant ``hold_last`` feeds
forever — so ``blend=1.0`` reduces algebraically to ``action_source="hold_last"``
and ``blend=0.0`` / ``every=1`` reduce to the unfiltered arm. All three
identities are asserted in ``test_blindimag.py``; they are what makes the sweep's
endpoints checkable against arms that already exist instead of by eyeball.

THE PLANNER ACTION SOURCE (``"planner"``) — T_BLIND RUNG 1, THE R1 ROW
----------------------------------------------------------------------
Rung 1 swept the FILTER axis and explicitly did not run the ladder's R1 PLANNER
row; it recorded a prediction for it instead (``TBLIND_RUNG1.md`` §7.2). This is
that row. ``action_source="planner"`` replaces the kinematic inverse with **v1's
deployed tactical planner + pure-pursuit controller**::

    w_look = tactical_policy(win_s, strategic_policy(win_s, follow)["ctx"])
                 ["waypoints"][LOOKAHEAD_STEP]        # the 0.5 s ego-frame target
    steer, accel = closedloop.wp_to_control(w_look, v)

⭐ ``wp_to_control`` is **IMPORTED from** :mod:`taniteval.closedloop`, never
copied — it is the same function ``closed_loop_rollout`` deploys, so this arm
cannot drift from the deployed controller. It differs from
:func:`kinematic_action_from_dpose` in exactly two gains: a **0.5 s pure-pursuit
lookahead** instead of a one-tick yaw increment, and ``(v_target − v)/SPEED_TC``
with ``SPEED_TC = 0.5 s`` instead of ``(v − v_prev)/0.1 s`` — a **5× lower
longitudinal gain**. Rung 1 measured that the own-action failure is the
acceleration command's AMPLITUDE (mean 2.058 m/s², at the ±3 clamp 46.4 % of the
time), so the gain is the mechanism this arm tests.

The planner needs the whole model, not just the predictor, so it is supplied as
``plan_fn(win_s, v) -> [B,2]`` rather than by importing ``fourbrain`` here. Two
modifiers, both planner-only:

  ``vsrc=ctrl``     (default) ``v`` is the CONTROLLER's own integrated speed,
                    ``v <- clamp_min(v + accel*DT, 0)`` — byte-for-byte
                    ``closed_loop_rollout``'s bookkeeping.
  ``vsrc=decoded``  ``v`` is the model's DECODED speed ``|dxy|/DT``. Isolates
                    which speed estimate the controller stands on.
  ``look=plan``     (default) the target comes from ``plan_fn``.
  ``look=gt``       ⛔ DIAGNOSTIC, PRIVILEGED — the target is the TRUE pose
                    ``LOOKAHEAD_STEP`` ahead, transformed into the current
                    IMAGINED ego frame. The planner's analogue of
                    ``gt_kinematic``: it holds the controller fixed and replaces
                    the intent with a perfect one, so a controller fault and a
                    tactical-head fault can be told apart. It reads the future
                    and may never be quoted as deployable.

⚠️ As with every other arm the ``v0`` action channel is held CONSTANT by default.
``closedloop.build_action`` feeds ``v_tracked / SPEED_SCALE`` instead; that
variant is available as ``update_speed_channel=True`` and is reported separately,
never mixed into the primary — Rung 1 measured the decoded-speed version of that
lever to be catastrophic (``de@2s`` 1.8165 → 23.9351).

⚠️ A filter rewrites **only the fed action's (steer, accel) channels**. It does
NOT touch the speed channel, and it does NOT touch ``v_prev`` — the internal
speed bookkeeping continues to track the model's own DECODED speed, because a
deployed rate-limiter filters the command, not the odometry.

THE SPEED CHANNEL — a deliberate convention, stated because it is load-bearing
------------------------------------------------------------------------------
v1 (``action_dim = 3``) receives ``v0 = poses[last, 3] / SPEED_SCALE`` as a
CONSTANT third action channel, broadcast across the window AND the whole future
(``taniteval.rollout.append_ego``; ``tanitad/train/flagship_losses.py:228``).
It is constant in training too. The faithful long-horizon extension is to keep
it constant, and that is the default. ``update_speed_channel=True`` feeds the
model's own predicted speed instead — an E-IMAG-3 lever, reported separately and
never mixed into the primary.

WHEELBASE — 2.9, and NOT 2.7, and this matters
----------------------------------------------
The corpus's steer channel is ``atan(WHEELBASE * curvature)`` with
``WHEELBASE = 2.9`` (``stack/tanitad/data/physicalai.py:51``). The closed-loop
harness (``taniteval.clhorizon``/``closedloop``) uses **2.7**. Inverting the
corpus's own definition requires the corpus's own constant, so this module uses
2.9 and says so. (The discrepancy is a live program finding — see
``…/incoming/2026-07-26-wheelbase-impact/``.)

HONEST LIMITS
-------------
1. ``full_obs`` under ``own_kinematic`` actions is **not self-consistent**: the
   percepts come from the logged trajectory while the ego claims to steer
   itself. It is a teacher-forced-percept ceiling and is labelled as one. Under
   ``true_future`` actions it IS self-consistent and is a genuine ceiling.
2. The accumulated path is the SE(2) dead-reckoning of the step readout's
   Δposes — the same construction as every grounded number in the program. It is
   not a bicycle integration, so it is not comparable to
   ``closedloop.closed_bicycle``.
3. Nothing here is a safety metric. There is no map, no agent boxes, no
   drivable area (PhysicalAI-AV ships none). This is a drift measurement.
"""
from __future__ import annotations

import math

import torch

from tanitad.models.metric_dynamics import accumulate_se2

BLOCK = "taniteval.blindimag/blind_horizon"
VERSION = "1.0.0"

W_DEFAULT = 8                  # predictor window (every trainer, every eval)
DT = 0.1                       # 10 Hz — MEASURED on every trainer and eval path
SPEED_SCALE = 10.0             # hard contract with the v1 checkpoint
WHEELBASE = 2.9                # the CORPUS's constant (physicalai.py:51)
STEER_CLAMP = 0.05             # clhorizon/closedloop convention
ACCEL_CLAMP = 3.0              # clhorizon/closedloop convention
V_EPS = 0.5                    # m/s floor when inverting steer (avoids /0)

#: The LATENT-ABLATION state sources (2026-07-27). Each replaces ONLY the latent
#: appended to the predictor's window; the action channel — including the
#: CONSTANT true ``v0`` carried in the action template — is untouched. They exist
#: to answer whether the blind horizon is the imagined latent's content or the
#: integration of that action channel.
#:
#: ``frozen_other``  the last real percept of a DIFFERENT window, held constant.
#:                   Same constant-window shape as ``frozen_last``, wrong content
#:                   — it separates "a constant window is off-distribution" from
#:                   "the content matters".
#: ``shuffled``      at each step, the IMAGINED latent of a different window. The
#:                   per-step marginal is the same multiset; correspondence gone.
#: ``shuffled_obs``  the same, on the TRUE observed latents.
#: ``mean_latent``   the batch mean of the last real percept — marginally
#:                   central, zero per-window information.
#: ``zero_latent``   all zeros. The strongest, deliberately off-distribution.
_ABLATION_SOURCES = ("frozen_other", "shuffled", "shuffled_obs",
                     "mean_latent", "zero_latent")
STATE_SOURCES = (("imagination", "frozen_last", "full_obs", "observed_pair")
                 + _ABLATION_SOURCES)
#: Ablations that draw their latent from a permutation over the batch.
_PERMUTED_SOURCES = ("frozen_other", "shuffled", "shuffled_obs")
ACTION_SOURCES = ("true_future", "own_kinematic", "gt_kinematic", "hold_last",
                  "planner")
#: Sources whose action is produced by the kinematic inverse (the filter axis).
_KINEMATIC_SOURCES = ("own_kinematic", "gt_kinematic")
#: ⭐ PLUMBING SELF-TEST HOOK. ``seed = IDENTITY_PERM_SEED`` makes every permuted
#: source use the IDENTITY permutation, under which ``shuffled`` reduces
#: algebraically to ``imagination``, ``shuffled_obs`` to ``full_obs`` and
#: ``frozen_other`` to ``frozen_last``. A filter knob that is silently a no-op
#: produces a flat, confident, wrong table; this is how that is caught, and it is
#: the only reason this sentinel exists. It is never used for a reported arm.
IDENTITY_PERM_SEED = -1


# --------------------------------------------------------------------------- #
# SE(2) accumulation that also returns the heading (positions bit-identical)   #
# --------------------------------------------------------------------------- #
def accumulate_se2_pose(step_dpose: torch.Tensor):
    """``step_dpose [B,K,3]`` -> ``(pos [B,K,2], psi [B,K])``.

    The position output is **bit-identical** to
    :func:`tanitad.models.metric_dynamics.accumulate_se2` (same op order, same
    dtype); ``psi`` is the cumulative heading that function discards. Pinned by
    ``test_blindimag.py::test_accumulate_matches_metric_dynamics``.
    """
    b, k, _ = step_dpose.shape
    pos = torch.zeros(b, 2, device=step_dpose.device, dtype=step_dpose.dtype)
    psi = torch.zeros(b, device=step_dpose.device, dtype=step_dpose.dtype)
    out, psis = [], []
    for j in range(k):
        c, s = torch.cos(psi), torch.sin(psi)
        dx, dy = step_dpose[:, j, 0], step_dpose[:, j, 1]
        pos = pos + torch.stack([c * dx - s * dy, s * dx + c * dy], dim=-1)
        psi = psi + step_dpose[:, j, 2]
        out.append(pos)
        psis.append(psi)
    return torch.stack(out, dim=1), torch.stack(psis, dim=1)


# --------------------------------------------------------------------------- #
# The action inverse — the EXACT inverse of the corpus's own steer definition  #
# --------------------------------------------------------------------------- #
def kinematic_action_from_dpose(dpose: torch.Tensor, v_prev: torch.Tensor):
    """One step's Δpose -> the (steer, accel) that would have produced it.

    ``dpose [B,3]`` = (Δx, Δy, Δyaw) over one 0.1 s tick in the ego frame at the
    tick's start. ``v_prev [B]`` the speed entering the tick. Returns
    ``(steer [B], accel [B], v [B])``.

    ``steer = atan(WHEELBASE * kappa)`` with ``kappa = Δyaw / (v * DT)`` is
    **the corpus's own label definition read backwards** (``physicalai.py:412``
    builds ``steer`` from ``atan(WHEELBASE * curvature)`` and curvature is
    ``dyaw/ds``). ``accel = (v - v_prev) / DT`` is NOT: the corpus takes the
    dataset's measured longitudinal ``ax``, deliberately avoiding a finite
    difference of interpolated speed. That mismatch is real and it is why the
    ``gt_kinematic`` control arm exists — it feeds this inverse the TRUE Δposes,
    so the whole convention cost is measured rather than argued.

    Both outputs are clamped to the harness's own limits, so an unstable rollout
    cannot manufacture an action the corpus never contains.
    """
    dxy = dpose[..., :2]
    dyaw = dpose[..., 2]
    v = dxy.norm(dim=-1) / DT
    accel = ((v - v_prev) / DT).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    kappa = dyaw / (v.clamp_min(V_EPS) * DT)
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    return steer, accel, v


#: The action-filter knobs (T_blind Rung 1). Every one is a pure function of the
#: action the deployed inverse would already have produced — no weight changes.
ACTION_MOD_KEYS = ("blend", "ema", "every", "steer_clip", "accel_clip", "chan",
                   "own_before", "own_after")
#: Planner-only knobs (T_blind Rung 1, the R1 row). They select which speed the
#: controller stands on and where its lookahead target comes from; they are NOT
#: filters and are refused on any other base.
PLANNER_MOD_KEYS = ("vsrc", "look")


def parse_state_source(spec: str):
    """``"shuffled|seed=7"`` -> ``("shuffled", {"seed": 7})``.

    A bare source parses to an EMPTY modifier dict, so every pre-ablation call
    site is bit-identical by construction. ``seed`` is the only key; it is
    refused on a source that does not permute, because a silently-ignored knob
    is how a sweep produces a flat, confident, wrong table.
    """
    base, _, rest = str(spec).partition("|")
    base = base.strip()
    mod: dict = {}
    for part in (p for p in rest.split("|") if p.strip()):
        k, _, v = part.partition("=")
        k = k.strip()
        if k != "seed":
            raise ValueError(f"unknown state modifier {k!r}; expected 'seed'")
        mod[k] = int(v)
    if mod and base not in _PERMUTED_SOURCES:
        raise ValueError(
            f"state modifier 'seed' is only defined for a permuted latent "
            f"ablation {_PERMUTED_SOURCES}; got base {base!r}")
    return base, mod


def _derangement(b: int, seed: int, j: int, device) -> torch.Tensor:
    """A permutation of ``range(b)`` with **no fixed point**, deterministic in
    ``(seed, j)``.

    A random ROLL is used rather than a rejection-sampled random derangement:
    it is a guaranteed derangement for any ``b >= 2``, it preserves the batch's
    multiset of latents EXACTLY (so the per-step marginal statistics of the
    ablated arm are identical to the intact one's), and it is reproducible from
    two integers. ``seed = IDENTITY_PERM_SEED`` returns the identity — the
    plumbing self-test, and the ONLY way a fixed point can occur.
    """
    idx = torch.arange(b, device=device)
    if int(seed) == IDENTITY_PERM_SEED:
        return idx
    if b < 2:
        raise ValueError(
            "a permuted latent ablation needs batch >= 2 (a batch of 1 has no "
            "derangement, so the ablation would silently be a no-op)")
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + int(j))
    off = int(torch.randint(1, b, (1,), generator=g).item())
    return (idx + off) % b


def parse_action_source(spec: str):
    """``"own_kinematic|blend=0.25"`` -> ``("own_kinematic", {"blend": 0.25})``.

    A bare source parses to an EMPTY modifier dict, so every pre-Rung-1 call site
    is bit-identical by construction. Unknown keys and out-of-range values raise
    rather than being ignored — a silently-dropped knob is how a sweep produces a
    flat, confident, wrong curve.
    """
    base, _, rest = str(spec).partition("|")
    base = base.strip()
    mod: dict = {}
    for part in (p for p in rest.split("|") if p.strip()):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in ACTION_MOD_KEYS + PLANNER_MOD_KEYS:
            raise ValueError(f"unknown action modifier {k!r}; expected one of "
                             f"{ACTION_MOD_KEYS + PLANNER_MOD_KEYS}")
        if k == "chan":
            v = v.strip()
            if v not in ("steer", "accel"):
                raise ValueError("chan must be 'steer' or 'accel', got %r" % v)
            mod[k] = v
        elif k == "vsrc":
            v = v.strip()
            if v not in ("ctrl", "decoded"):
                raise ValueError("vsrc must be 'ctrl' or 'decoded', got %r" % v)
            mod[k] = v
        elif k == "look":
            v = v.strip()
            if v not in ("plan", "gt"):
                raise ValueError("look must be 'plan' or 'gt', got %r" % v)
            mod[k] = v
        elif k in ("every", "own_before", "own_after"):
            iv = int(v)
            if iv < 1:
                raise ValueError(f"{k} must be >= 1, got {iv}")
            mod[k] = iv
        else:
            fv = float(v)
            if k in ("blend", "ema") and not (0.0 <= fv <= 1.0):
                raise ValueError(f"{k} must be in [0,1], got {fv}")
            if k in ("steer_clip", "accel_clip") and fv < 0.0:
                raise ValueError(f"{k} must be >= 0, got {fv}")
            mod[k] = fv
    filt = {k for k in mod if k in ACTION_MOD_KEYS}
    plan = {k for k in mod if k in PLANNER_MOD_KEYS}
    if filt and base not in _KINEMATIC_SOURCES:
        raise ValueError(
            f"action modifiers are only defined for a kinematic action source "
            f"(they filter the action the inverse produced); got base {base!r}")
    if plan and base != "planner":
        raise ValueError(
            f"planner modifiers {sorted(plan)} are only defined for "
            f"action_source='planner'; got base {base!r}")
    return base, mod


def apply_action_filter(a_next, mod: dict, *, j: int, a_hold0, a_prev_fed):
    """Rewrite the (steer, accel) channels of one fed action row.

    ``a_next [B,A]``    what the unfiltered kinematic source would feed.
    ``a_hold0 [B,A]``   the last OBSERVED action (what ``hold_last`` feeds forever).
    ``a_prev_fed``      the previously FED action, or ``None`` at the first step.
    ``j``               index of the fed action, 0-based.

    Order of application is fixed and documented because it is load-bearing for
    any combined config: ``chan`` -> clips -> ``every`` -> ``ema`` -> ``blend`` ->
    the ``own_before`` / ``own_after`` switches. Every experiment in Rung 1 uses
    ONE knob at a time, so the order only binds future combinations.
    """
    if not mod:
        return a_next
    x = a_next.clone()
    if "chan" in mod:                       # amputate the OTHER channel
        x[..., 1 if mod["chan"] == "steer" else 0] = \
            a_hold0[..., 1 if mod["chan"] == "steer" else 0]
    if "steer_clip" in mod:
        c = float(mod["steer_clip"])
        x[..., 0] = x[..., 0].clamp(-c, c)
    if "accel_clip" in mod:
        c = float(mod["accel_clip"])
        x[..., 1] = x[..., 1].clamp(-c, c)
    if "every" in mod and a_prev_fed is not None and (j % int(mod["every"])):
        x = a_prev_fed.clone()              # zero-order hold between updates
    if "ema" in mod:
        b = float(mod["ema"])
        base = a_hold0 if a_prev_fed is None else a_prev_fed
        x[..., 0] = b * base[..., 0] + (1.0 - b) * x[..., 0]
        x[..., 1] = b * base[..., 1] + (1.0 - b) * x[..., 1]
    if "blend" in mod:
        w = float(mod["blend"])
        x[..., 0] = (1.0 - w) * x[..., 0] + w * a_hold0[..., 0]
        x[..., 1] = (1.0 - w) * x[..., 1] + w * a_hold0[..., 1]
    if (("own_before" in mod and j >= int(mod["own_before"]))
            or ("own_after" in mod and j < int(mod["own_after"]))):
        x[..., 0] = a_hold0[..., 0]
        x[..., 1] = a_hold0[..., 1]
    return x


def _pack_action(steer, accel, template):
    """(steer, accel) -> a full action row shaped like ``template [B, A]``.

    Channels beyond the first two (the v1 speed channel) are carried over from
    ``template`` unchanged — the harness convention is that ``v0`` is constant
    across the whole future (``taniteval.rollout.append_ego``).
    """
    a = template.clone()
    a[..., 0] = steer
    a[..., 1] = accel
    return a


# --------------------------------------------------------------------------- #
# THE ROLLOUT — one loop, four state sources, four action sources              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def blind_rollout(predictor, states: torch.Tensor, actions: torch.Tensor,
                  step_readout, k: int, *,
                  state_source: str = "imagination",
                  action_source: str = "true_future",
                  future_actions: torch.Tensor | None = None,
                  obs_states: torch.Tensor | None = None,
                  gt_step_dpose: torch.Tensor | None = None,
                  v_last: torch.Tensor | None = None,
                  update_speed_channel: bool = False,
                  plan_fn=None,
                  gt_pos: torch.Tensor | None = None,
                  peek_period: int | None = None,
                  peek_oracle_bar: float | None = None,
                  latent_perm_seed: int = 0,
                  latent_stats: bool = False) -> dict:
    """Roll ``predictor`` forward ``k`` steps and decode a metric path.

    ``states [B,W,S]``      encoded REAL frames of the observed window.
    ``actions [B,W,A]``     the window's actions (ego channels already appended).
    ``future_actions [B,H,A]``  H >= k-1, the expert's logged future actions.
    ``obs_states [B,k,S]``  encodings of the TRUE frames at last+1 … last+k
                            (required for ``full_obs`` / ``observed_pair``).
    ``gt_step_dpose [B,k,3]``  the TRUE per-step Δposes (required for
                            ``gt_kinematic``).
    ``v_last [B]``          speed at the window's last frame (m/s), required for
                            the kinematic action sources.

    THE PEEK POLICIES (E-IMAG-4) — both re-anchor by appending the TRUE frame's
    encoding instead of the imagined latent, and both RECORD what they did:
    ``peek_period=T'``      uniform: re-anchor every ``T'`` steps. Duty cycle
                            ``1/T'`` of the front-camera budget.
    ``peek_oracle_bar=e``   ORACLE: re-anchor only on ticks where the model's own
                            per-step decode error already exceeds ``e`` metres.
                            Needs ``gt_step_dpose``. **Privileged by
                            construction** — it reads the true error. It exists
                            to bound what a learned trigger could win, and may
                            never be reported as deployable.
    The realised duty cycle is returned as ``peek_mask``, never assumed.

    Returns ``{"waypoints" [B,k,2], "psi" [B,k], "step_dpose" [B,k,3],
    "fed_actions" [B,k,A], "pred_speed" [B,k], "peek_mask" [B,k]}``.

    ``action_source`` may carry an ACTION FILTER suffix (Rung 1), e.g.
    ``"own_kinematic|blend=0.25"`` — see :func:`parse_action_source`. A bare
    source is bit-identical to the pre-Rung-1 behaviour.

    ``state_source`` may carry a ``|seed=N`` suffix for the permuted LATENT
    ABLATIONS (2026-07-27) — see :func:`parse_state_source`. ``latent_perm_seed``
    is the default when the string carries none.

    ``latent_stats=True`` additionally returns the FIXED-POINT probe:
    ``lat_dz`` (``||z_j - z_{j-1}||``, with ``z_{-1}`` the last real percept),
    ``lat_d0`` (``||z_j - z_0||``), ``lat_cos0`` (``cos(z_j, z_0)``) and
    ``lat_norm`` (``||z_j||``), each ``[B,k]``, all measured on the latent the
    predictor EMITTED (``z_hat``) — not on the ablated substitute, so the probe
    reads the imagination itself.

    ⚠️ With ``state_source="imagination"``, ``action_source="true_future"`` and no
    peek policy the loop is byte-for-byte ``metric_dynamics.rollout_decode``
    (test-pinned).
    """
    state_source, state_mod = parse_state_source(state_source)
    if state_source not in STATE_SOURCES:
        raise ValueError(f"state_source must be one of {STATE_SOURCES}, "
                         f"got {state_source!r}")
    perm_seed = int(state_mod.get("seed", latent_perm_seed))
    if state_source in ("full_obs", "observed_pair", "shuffled_obs") \
            and obs_states is None:
        raise ValueError(f"state_source={state_source!r} needs obs_states "
                         f"[B,k,S] (the TRUE next-frame encodings)")
    action_source, action_mod = parse_action_source(action_source)
    if action_source not in ACTION_SOURCES:
        raise ValueError(f"action_source must be one of {ACTION_SOURCES}, "
                         f"got {action_source!r}")
    if action_source == "true_future" and future_actions is None:
        raise ValueError("action_source='true_future' needs future_actions")
    if action_source == "gt_kinematic" and gt_step_dpose is None:
        raise ValueError("action_source='gt_kinematic' needs gt_step_dpose")
    if action_source in _KINEMATIC_SOURCES + ("planner",) and v_last is None:
        raise ValueError(f"action_source={action_source!r} needs v_last [B]")
    look = action_mod.get("look", "plan")
    if action_source == "planner":
        if look == "plan" and plan_fn is None:
            raise ValueError(
                "action_source='planner' needs plan_fn(win_s, v) -> [B,2], the "
                "0.5 s ego-frame lookahead target from the model's tactical "
                "brain; blindimag deliberately does not import fourbrain")
        if look == "gt" and gt_pos is None:
            raise ValueError(
                "action_source='planner|look=gt' needs gt_pos [B,k,2] (it is an "
                "ORACLE target — it reads the true future by design)")
    peeking = peek_period is not None or peek_oracle_bar is not None
    if peeking and obs_states is None:
        raise ValueError("a peek policy needs obs_states [B,k,S] to re-anchor")
    if peek_oracle_bar is not None and gt_step_dpose is None:
        raise ValueError("peek_oracle_bar needs gt_step_dpose [B,k,3] (it is an "
                         "ORACLE policy — it reads the true error by design)")
    if peek_period is not None and peek_oracle_bar is not None:
        raise ValueError("choose ONE peek policy: uniform or oracle, not both")

    win_s, win_a = states, actions
    z_frozen = states[:, -1]                       # the last REAL percept
    a_hold0 = actions[:, -1]                       # what hold_last feeds forever
    a_prev_fed = None                              # for `ema` / `every`
    v_prev = v_last
    dposes, fed, speeds, peeks = [], [], [], []
    b = states.shape[0]
    # ---- the LATENT-ABLATION substitutes, built once ----------------------- #
    z_const = None                                 # a CONSTANT ablated latent
    if state_source == "frozen_other":
        # one permutation for the whole rollout: a constant, WRONG percept
        z_const = z_frozen[_derangement(b, perm_seed, 0, states.device)]
    elif state_source == "mean_latent":
        z_const = z_frozen.mean(dim=0, keepdim=True).expand_as(z_frozen)
    elif state_source == "zero_latent":
        z_const = torch.zeros_like(z_frozen)
    lat_dz, lat_d0, lat_cos0, lat_nrm = [], [], [], []
    if latent_stats:
        _z0 = z_frozen
        _z0n = _z0.flatten(1).norm(dim=-1).clamp_min(1e-12)
        _z_prev = _z0
    if action_source == "planner":
        # the controller's own speed bookkeeping, byte-for-byte
        # closedloop.closed_loop_rollout's `v = (v + accel*DT).clamp_min(0)`
        v_ctrl = v_last.clone()
        # running imagined pose, ONLY for the `look=gt` oracle target. Same op
        # order as accumulate_se2_pose, pinned equal by test_blindimag.py.
        cur_pos = torch.zeros(b, 2, device=states.device, dtype=states.dtype)
        cur_psi = torch.zeros(b, device=states.device, dtype=states.dtype)

    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]         # 1-step head -> z_{t+j+1}
        if latent_stats:
            # ⭐ THE FIXED-POINT PROBE. Measured on what the predictor EMITTED,
            # so it reads the imagination itself and is well defined for every
            # state source (including the ablations, where it reports what the
            # predictor does when its context is corrupted).
            _f, _fp, _f0 = z_hat.flatten(1), _z_prev.flatten(1), _z0.flatten(1)
            _n = _f.norm(dim=-1)
            lat_nrm.append(_n)
            lat_dz.append((_f - _fp).norm(dim=-1))
            lat_d0.append((_f - _f0).norm(dim=-1))
            lat_cos0.append((_f * _f0).sum(-1) / (_n.clamp_min(1e-12) * _z0n))
            _z_prev = z_hat
        if state_source == "observed_pair":
            dpose = step_readout(win_s[:, -1], obs_states[:, j])
        else:
            dpose = step_readout(win_s[:, -1], z_hat)
        dposes.append(dpose)

        if action_source in _KINEMATIC_SOURCES:
            src = dpose if action_source == "own_kinematic" else gt_step_dpose[:, j]
            steer, accel, v_now = kinematic_action_from_dpose(src, v_prev)
            speeds.append(v_now)
        else:
            v_now = dpose[..., :2].norm(dim=-1) / DT
            speeds.append(v_now)

        if action_source == "planner":
            # advance the imagined pose by the step just decoded
            c, s = torch.cos(cur_psi), torch.sin(cur_psi)
            dx, dy = dpose[:, 0].to(cur_pos.dtype), dpose[:, 1].to(cur_pos.dtype)
            cur_pos = cur_pos + torch.stack([c * dx - s * dy,
                                             s * dx + c * dy], dim=-1)
            cur_psi = cur_psi + dpose[:, 2].to(cur_psi.dtype)

        if j < k - 1:
            # ---- the action fed at step j+1 -------------------------------- #
            if action_source == "true_future":
                a_next = future_actions[:, j]
            elif action_source == "hold_last":
                a_next = win_a[:, -1]
            elif action_source == "planner":
                # ⭐ v1's DEPLOYED planner + controller. `wp_to_control` is the
                # closedloop function itself, imported, not re-derived.
                from taniteval.closedloop import (LOOKAHEAD_STEP,
                                                  wp_to_control)
                v_ct = v_now if action_mod.get("vsrc") == "decoded" else v_ctrl
                if look == "gt":
                    tgt = gt_pos[:, min(j + LOOKAHEAD_STEP, k - 1)]
                    d = (tgt.to(cur_pos.dtype) - cur_pos)
                    cc, ss = torch.cos(-cur_psi), torch.sin(-cur_psi)
                    w_look = torch.stack([d[:, 0] * cc - d[:, 1] * ss,
                                          d[:, 0] * ss + d[:, 1] * cc], dim=-1)
                else:
                    w_look = plan_fn(win_s, v_ct)
                steer, accel = wp_to_control(w_look.to(v_ct.dtype), v_ct)
                a_next = _pack_action(steer, accel, win_a[:, -1])
                if update_speed_channel and a_next.shape[-1] >= 3:
                    a_next = a_next.clone()
                    a_next[..., 2] = v_ctrl / SPEED_SCALE
                # the controller integrates its OWN command — closed_loop_rollout
                v_ctrl = (v_ctrl + accel * DT).clamp_min(0.0)
                a_prev_fed = a_next
                v_prev = v_now
            else:                                   # own_/gt_kinematic
                a_next = _pack_action(steer, accel, win_a[:, -1])
                if update_speed_channel and a_next.shape[-1] >= 3:
                    a_next = a_next.clone()
                    a_next[..., 2] = v_now / SPEED_SCALE
                # the action FILTER (Rung 1) — a no-op when `action_mod` is empty
                a_next = apply_action_filter(a_next, action_mod, j=j,
                                             a_hold0=a_hold0,
                                             a_prev_fed=a_prev_fed)
                a_prev_fed = a_next
                # ⚠️ v_prev tracks the model's DECODED speed, not the filtered
                # command: a deployed rate-limiter filters the command, not the
                # odometry. Unchanged by any filter.
                v_prev = v_now
            # ---- the latent appended at step j+1 --------------------------- #
            # ⚠️ ONLY this line differs between the latent ablations. The action
            # tensor above — including the CONSTANT true `v0` channel — is
            # produced identically for every state source, which is what makes
            # these arms a test of the latent rather than of the action loop.
            if state_source == "imagination":
                z_next = z_hat
            elif state_source == "frozen_last":
                z_next = z_frozen
            elif z_const is not None:               # frozen_other / mean / zero
                z_next = z_const
            elif state_source == "shuffled":
                z_next = z_hat[_derangement(b, perm_seed, j, states.device)]
            elif state_source == "shuffled_obs":
                z_next = obs_states[:, j][
                    _derangement(b, perm_seed, j, states.device)]
            else:                                   # full_obs / observed_pair
                z_next = obs_states[:, j]
            # ---- the peek policy overrides the latent source ---------------- #
            did_peek = torch.zeros(b, dtype=torch.bool, device=states.device)
            if peek_period is not None:
                if (j + 1) % int(peek_period) == 0:
                    z_next = obs_states[:, j]
                    did_peek |= True
            elif peek_oracle_bar is not None:
                # trigger on the INSTANTANEOUS per-step decode error: "the
                # imagined dynamics are drifting RIGHT NOW". Local, so it resets
                # naturally; a cumulative trigger would latch on forever after
                # the first exceedance and its duty cycle would be meaningless.
                # ⚠️ A peek re-anchors the LATENT only. It does NOT correct the
                # accumulated pose: this architecture has no localisation, the
                # step readout emits relative motion only, so a camera frame
                # buys perception, never a position fix.
                err = (dpose[..., :2] - gt_step_dpose[:, j, :2]).norm(dim=-1)
                did_peek = err > float(peek_oracle_bar)
                z_next = torch.where(did_peek[:, None], obs_states[:, j], z_next)
            peeks.append(did_peek)
            win_s = torch.cat([win_s[:, 1:], z_next.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
            fed.append(a_next)
        else:
            fed.append(win_a[:, -1])
            peeks.append(torch.zeros(b, dtype=torch.bool, device=states.device))

    step_dpose = torch.stack(dposes, dim=1)                       # [B,k,3]
    wp, psi = accumulate_se2_pose(step_dpose)
    out = {"waypoints": wp, "psi": psi, "step_dpose": step_dpose,
           "fed_actions": torch.stack(fed, dim=1),
           "pred_speed": torch.stack(speeds, dim=1),
           "peek_mask": torch.stack(peeks, dim=1)}
    if latent_stats:
        out.update({"lat_dz": torch.stack(lat_dz, dim=1),
                    "lat_d0": torch.stack(lat_d0, dim=1),
                    "lat_cos0": torch.stack(lat_cos0, dim=1),
                    "lat_norm": torch.stack(lat_nrm, dim=1)})
    return out


# --------------------------------------------------------------------------- #
# Reconstructing what the loop FED, from what a sweep dump already stores       #
# --------------------------------------------------------------------------- #
def reconstruct_kinematic_actions(psi: torch.Tensor, pred_speed: torch.Tensor,
                                  v_last: torch.Tensor):
    """``(steer [B,k], accel [B,k])`` — the RAW own-kinematic action per step.

    ``bi_run._run_arms`` stores ``psi`` and ``pred_speed`` per arm but not
    ``fed_actions``, so the action statistics that separate *drift* from
    *saturation* from *feedback instability* would otherwise need a second
    600-episode rollout. They do not: the inverse in
    :func:`kinematic_action_from_dpose` is a function of the per-step heading
    increment and the decoded speed alone, and ``psi`` is their cumulative sum.

    ``psi [B,k]`` cumulative heading, ``pred_speed [B,k]`` the decoded speed
    ``|dxy|/DT`` at each step, ``v_last [B]`` the speed entering the rollout.

    ⚠️ This reconstructs the action the inverse PRODUCED. On a filtered arm the
    action actually fed differs by exactly that filter. Asserted equal to
    ``blind_rollout(...)["fed_actions"]`` on the unfiltered arm by
    ``test_blindimag.py::test_reconstruct_kinematic_actions_matches_fed_actions``.
    """
    dyaw = torch.cat([psi[:, :1], psi[:, 1:] - psi[:, :-1]], dim=1)
    v = pred_speed
    v_prev = torch.cat([v_last.reshape(-1, 1).to(v.dtype), v[:, :-1]], dim=1)
    accel = ((v - v_prev) / DT).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    kappa = dyaw / (v.clamp_min(V_EPS) * DT)
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    return steer, accel


# --------------------------------------------------------------------------- #
# Ground truth and the trivial floor, at ARBITRARY horizon                     #
# --------------------------------------------------------------------------- #
def _ego(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gt_dense_path(poses: torch.Tensor, last: torch.Tensor, k: int):
    """``(pos [B,k,2], yaw [B,k])`` — the logged future in the ego frame at
    ``last``. ``pos`` reproduces ``driving_diagnostic.gt_ego_waypoints`` at
    ``wp_steps=range(1, k+1)`` exactly."""
    p0 = poses[last, :2]
    yaw0 = poses[last, 2]
    idx = last[:, None] + torch.arange(1, k + 1)[None]
    pos = _ego(poses[idx][..., :2] - p0[:, None], yaw0[:, None])
    yaw = _wrap(poses[idx][..., 2] - yaw0[:, None])
    return pos, yaw


def gt_step_dposes_dense(poses: torch.Tensor, last: torch.Tensor, k: int):
    """TRUE per-step Δposes ``[B,k,3]`` for the ``gt_kinematic`` action source.

    Row j is ``relative_ego_pose(pose_{last+j}, pose_{last+j+1})`` — identical in
    definition to ``metric_dynamics.gt_step_dposes``, computed here directly from
    the episode's pose table so no future-pose tensor has to be materialised."""
    idx = last[:, None] + torch.arange(0, k + 1)[None]
    P = poses[idx]                                                # [B,k+1,4]
    dxy_w = P[:, 1:, :2] - P[:, :-1, :2]
    dxy = _ego(dxy_w, P[:, :-1, 2])
    dyaw = _wrap(P[:, 1:, 2] - P[:, :-1, 2]).unsqueeze(-1)
    return torch.cat([dxy, dyaw], dim=-1)


def cv_dense_path(poses: torch.Tensor, last: torch.Tensor, k: int):
    """Arm (d): CONSTANT VELOCITY, dense, ``[B,k,2]``.

    ``driving_diagnostic.baseline_waypoints`` defines CV as the last one-step
    world velocity rotated into the ego frame and extrapolated linearly, i.e.
    ``cv[k] = ego_v * k``. Reproduced here verbatim for arbitrary k (CV is exactly
    linear in the step index, so nothing is approximated)."""
    p0, pm1 = poses[last, :2], poses[last - 1, :2]
    yaw0 = poses[last, 2]
    ego_v = _ego(p0 - pm1, yaw0)                                  # [B,2]
    ks = torch.arange(1, k + 1, dtype=poses.dtype).view(1, k, 1)
    return ego_v[:, None, :] * ks


def hold_v0_dense_path(poses: torch.Tensor, last: torch.Tensor, k: int):
    """The 'go straight at the current speed' floor, dense ``[B,k,2]``
    (``baseline_waypoints['go_straight']`` at arbitrary k)."""
    p0, pm1 = poses[last, :2], poses[last - 1, :2]
    speed = (p0 - pm1).norm(dim=-1)
    ks = torch.arange(1, k + 1, dtype=poses.dtype).view(1, k)
    out = torch.zeros(last.shape[0], k, 2, dtype=poses.dtype)
    out[..., 0] = speed[:, None] * ks
    return out


# --------------------------------------------------------------------------- #
# Path-relative deviation — the input the P1 envelope / OOD accounting wants   #
# --------------------------------------------------------------------------- #
def path_deviation(pred_pos: torch.Tensor, pred_yaw: torch.Tensor,
                   gt_pos: torch.Tensor, gt_yaw: torch.Tensor):
    """``(lat_abs [B,k], yaw_abs_deg [B,k])`` of the predicted path from the
    logged one, using the SAME nearest-reference-point construction as
    ``clhorizon.corridor_rollout`` (which is ``e1a_horizon.rollout``): for each
    step, find the nearest logged reference pose, take the signed lateral offset
    in that pose's frame and the wrapped heading error.

    Both series are already in the ego frame of the window's last real frame, so
    no extra transform is needed."""
    b, k, _ = pred_pos.shape
    ref = torch.cat([torch.zeros(b, 1, 2, dtype=gt_pos.dtype), gt_pos], dim=1)
    ref_yaw = torch.cat([torch.zeros(b, 1, dtype=gt_yaw.dtype), gt_yaw], dim=1)
    lat = torch.zeros(b, k, dtype=pred_pos.dtype)
    dpsi = torch.zeros(b, k, dtype=pred_pos.dtype)
    ar = torch.arange(b)
    for j in range(k):
        d = (ref - pred_pos[:, j][:, None]).norm(dim=-1)           # [B,k+1]
        m = d.argmin(dim=1)
        pref, yref = ref[ar, m], ref_yaw[ar, m]
        dx = pred_pos[:, j, 0] - pref[:, 0]
        dy = pred_pos[:, j, 1] - pref[:, 1]
        lat[:, j] = -torch.sin(yref) * dx + torch.cos(yref) * dy
        dpsi[:, j] = _wrap(pred_yaw[:, j] - yref)
    return lat.abs(), dpsi.abs() * 180.0 / math.pi


# --------------------------------------------------------------------------- #
# Episode encoding — done ONCE, reused by every arm                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_episode_states(model, ep, device, batch: int = 32,
                          t_max: int | None = None) -> torch.Tensor:
    """Encode every frame of an episode -> ``[T, S]`` float32 on CPU.

    ``WorldModel.encode_window`` is ``encode`` applied per frame and reshaped
    (``fourbrain.py``: ``flat = frames.reshape(b*w, ...)``), so per-frame
    encoding here is EXACTLY what the windowed path produces — pinned by
    ``test_blindimag.py::test_episode_encoding_matches_encode_window``. Doing it
    once per episode is what makes the ``full_obs`` arm free instead of
    ``K`` extra encoder passes per window."""
    fr = ep.feats
    T = int(fr.shape[0]) if t_max is None else min(int(fr.shape[0]), t_max)
    out = []
    for i in range(0, T, batch):
        x = torch.as_tensor(fr[i:i + batch]).to(device)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
        out.append(model.encode(x).float().cpu())
    return torch.cat(out)


def window_starts(T: int, k: int, window: int = W_DEFAULT, stride: int = 8):
    """``range(0, T - W - k, stride)`` — the harness's own window rule
    (``taniteval.rollout.collect``, ``clhorizon.horizon_windows``). At k = 185 on
    this corpus (T = 188…205) this yields ~1 window per episode; **n must be
    quoted with every long-horizon number.**"""
    return list(range(0, max(0, int(T) - int(window) - int(k)), int(stride)))


# --------------------------------------------------------------------------- #
# The window builder — ONE fixed window set, shared by every arm and horizon    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_windows(model, episodes, device, k: int, *, window: int = W_DEFAULT,
                  stride: int = 8, speed_input: bool = True,
                  states_cache: dict | None = None, verbose: bool = False):
    """Everything every arm needs, computed ONCE: the observed window's states,
    its actions, the true future actions, the true next-frame encodings, the GT
    dense path, the trivial floors and the episode id.

    ``states_cache`` maps ``ep_index -> [T, S]`` (from
    :func:`encode_episode_states`). When it is ``None`` the observed window is
    encoded with ``model.encode_window`` **exactly as** ``taniteval.rollout.collect``
    does — that path is what the reproduction gate uses, so the gate cannot be
    passed by a different encoding route than the one under test. The cached
    path additionally supplies ``obs_states`` (the ``full_obs`` arm), which the
    windowed path cannot.

    Returns a list of per-batch dicts, each holding CPU tensors except
    ``states``/``actions``/``obs_states`` which stay on ``device``.
    """
    out = []
    for ep_i, ep in enumerate(episodes):
        feats = ep.feats
        T = min(int(feats.shape[0]), int(ep.actions.shape[0]),
                int(ep.poses.shape[0]))
        starts = window_starts(T, k, window, stride)
        if not starts:
            continue
        st_full = None if states_cache is None else states_cache[ep_i].to(device)
        for t in starts:
            last = t + window - 1
            rec = {"ep_i": ep_i, "eid": str(ep.episode_id), "t0": t,
                   "last": last, "T": T}
            if st_full is None:
                fw = torch.as_tensor(feats[t:t + window])[None].to(device)
                fw = (fw.float().div_(255.0) if fw.dtype == torch.uint8
                      else fw.float())
                rec["states"] = model.encode_window(fw)              # [1,W,S]
                rec["obs_states"] = None
            else:
                rec["states"] = st_full[t:t + window][None]
                rec["obs_states"] = st_full[last + 1:last + 1 + k][None]
            aw = ep.actions[t:t + window][None].to(device).float()
            fa = ep.actions[last + 1:last + 1 + k][None].to(device).float()
            if speed_input:
                # taniteval.rollout.append_ego: v0 = pose_last.v / SPEED_SCALE,
                # broadcast CONSTANT across the window AND the whole future.
                v0c = (ep.poses[last, 3:4].float() / SPEED_SCALE).to(device)
                aw = torch.cat(
                    [aw, v0c.view(1, 1, 1).expand(1, aw.shape[1], 1)], dim=-1)
                fa = torch.cat(
                    [fa, v0c.view(1, 1, 1).expand(1, fa.shape[1], 1)], dim=-1)
            rec["actions"], rec["future_actions"] = aw, fa
            out.append(rec)
        if verbose and ep_i % 50 == 0:
            print(f"[blindimag] windows: episode {ep_i}/{len(episodes)} "
                  f"({len(out)} so far)", flush=True)
        if st_full is not None:
            del st_full
    return out


def batch_windows(recs, poses_by_ep, k: int, batch: int = 32):
    """Group per-window records into GPU batches, attaching the GT tensors.

    Yields dicts with ``states``/``actions``/``future_actions``/``obs_states``
    stacked, plus ``gt_pos``/``gt_yaw``/``gt_dpose``/``cv``/``hold_v0``/``v_last``
    and the bookkeeping (``eid``, ``t0``, ``ep_i``, ``speed``, ``head_deg``)."""
    for i in range(0, len(recs), batch):
        ch = recs[i:i + batch]
        by_ep: dict[int, list] = {}
        for r in ch:
            by_ep.setdefault(r["ep_i"], []).append(r)
        gt_pos, gt_yaw, gt_dp, cv, hv, vlast, hdeg = [], [], [], [], [], [], []
        for r in ch:
            p = poses_by_ep[r["ep_i"]]
            last = torch.tensor([r["last"]])
            gp, gy = gt_dense_path(p, last, k)
            gt_pos.append(gp)
            gt_yaw.append(gy)
            gt_dp.append(gt_step_dposes_dense(p, last, k))
            cv.append(cv_dense_path(p, last, k))
            hv.append(hold_v0_dense_path(p, last, k))
            vlast.append(p[r["last"], 3:4])
            hdeg.append((_wrap(p[r["last"] + 20, 2] - p[r["last"], 2]).abs()
                         * 180.0 / math.pi).reshape(1))
        yield {
            "states": torch.cat([r["states"] for r in ch]),
            "actions": torch.cat([r["actions"] for r in ch]),
            "future_actions": torch.cat([r["future_actions"] for r in ch]),
            "obs_states": (None if ch[0]["obs_states"] is None
                           else torch.cat([r["obs_states"] for r in ch])),
            "gt_pos": torch.cat(gt_pos), "gt_yaw": torch.cat(gt_yaw),
            "gt_dpose": torch.cat(gt_dp), "cv": torch.cat(cv),
            "hold_v0": torch.cat(hv),
            "v_last": torch.cat(vlast), "head_deg": torch.cat(hdeg),
            "speed": torch.cat(vlast),
            "eid": [r["eid"] for r in ch], "t0": [r["t0"] for r in ch],
            "ep_i": [r["ep_i"] for r in ch],
            "_n": len(ch),
        }
