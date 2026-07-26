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

STATE_SOURCES = ("imagination", "frozen_last", "full_obs", "observed_pair")
ACTION_SOURCES = ("true_future", "own_kinematic", "gt_kinematic", "hold_last")


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
                  peek_period: int | None = None,
                  peek_oracle_bar: float | None = None) -> dict:
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

    ⚠️ With ``state_source="imagination"``, ``action_source="true_future"`` and no
    peek policy the loop is byte-for-byte ``metric_dynamics.rollout_decode``
    (test-pinned).
    """
    if state_source not in STATE_SOURCES:
        raise ValueError(f"state_source must be one of {STATE_SOURCES}, "
                         f"got {state_source!r}")
    if action_source not in ACTION_SOURCES:
        raise ValueError(f"action_source must be one of {ACTION_SOURCES}, "
                         f"got {action_source!r}")
    if state_source in ("full_obs", "observed_pair") and obs_states is None:
        raise ValueError(f"state_source={state_source!r} needs obs_states "
                         f"[B,k,S] (the TRUE next-frame encodings)")
    if action_source == "true_future" and future_actions is None:
        raise ValueError("action_source='true_future' needs future_actions")
    if action_source == "gt_kinematic" and gt_step_dpose is None:
        raise ValueError("action_source='gt_kinematic' needs gt_step_dpose")
    if action_source in ("own_kinematic", "gt_kinematic") and v_last is None:
        raise ValueError(f"action_source={action_source!r} needs v_last [B]")
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
    v_prev = v_last
    dposes, fed, speeds, peeks = [], [], [], []
    b = states.shape[0]

    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]         # 1-step head -> z_{t+j+1}
        if state_source == "observed_pair":
            dpose = step_readout(win_s[:, -1], obs_states[:, j])
        else:
            dpose = step_readout(win_s[:, -1], z_hat)
        dposes.append(dpose)

        if action_source in ("own_kinematic", "gt_kinematic"):
            src = dpose if action_source == "own_kinematic" else gt_step_dpose[:, j]
            steer, accel, v_now = kinematic_action_from_dpose(src, v_prev)
            speeds.append(v_now)
        else:
            v_now = dpose[..., :2].norm(dim=-1) / DT
            speeds.append(v_now)

        if j < k - 1:
            # ---- the action fed at step j+1 -------------------------------- #
            if action_source == "true_future":
                a_next = future_actions[:, j]
            elif action_source == "hold_last":
                a_next = win_a[:, -1]
            else:                                   # own_/gt_kinematic
                a_next = _pack_action(steer, accel, win_a[:, -1])
                if update_speed_channel and a_next.shape[-1] >= 3:
                    a_next = a_next.clone()
                    a_next[..., 2] = v_now / SPEED_SCALE
                v_prev = v_now
            # ---- the latent appended at step j+1 --------------------------- #
            if state_source == "imagination":
                z_next = z_hat
            elif state_source == "frozen_last":
                z_next = z_frozen
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
    return {"waypoints": wp, "psi": psi, "step_dpose": step_dpose,
            "fed_actions": torch.stack(fed, dim=1),
            "pred_speed": torch.stack(speeds, dim=1),
            "peek_mask": torch.stack(peeks, dim=1)}


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
