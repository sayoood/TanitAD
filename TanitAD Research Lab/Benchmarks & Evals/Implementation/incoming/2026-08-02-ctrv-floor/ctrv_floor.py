"""CTRV — the third trivial floor, and the one the driving block is missing.

WHY THIS MODULE EXISTS
----------------------
``taniteval/driving.py`` (the canonical tier-0 driving-capability block) scores
every arm against exactly two floors::

    FLOORS = ("cv", "holdv0")

``cv`` extrapolates the last velocity *vector* linearly; ``holdv0`` goes
straight at the entry speed. **Both are straight-line predictors.** Neither can
represent a turn. Every "the model beats the floor" verdict that lives on a
turning stratum — ``sustained_turn``, ``by_curvature.gentle/sharp``,
``lat_abs_2s_m``, ``pathgeom_crosstrack_m``, ``curv_sign_agree`` — is therefore
measured against a family that is structurally unable to compete there.

CTRV (constant turn rate + constant velocity) is the standard third member of
that family. It is admissible under exactly the same information budget as the
other two — it reads ``poses[last]`` and ``poses[last-1]`` and nothing else, no
future, no privileged state — and it is **already computed on every window**:
``stack/scripts/driving_diagnostic.baseline_waypoints`` returns it as
``constant_yaw_rate``, and ``taniteval/rollout.collect`` keeps only
``constant_velocity`` and discards it (``rollout.py``, the ``CV.append(...)``
line). Adding it back costs no compute.

This matters because it was already measured, on other corpora, that CTRV is
the DOMINANT floor member: ``Implementation/incoming/2026-07-15-baseline-floor``
found CTRV winning **55-58 %** of anchors over CV/go-straight on 26 132 real
anchors, and CV overstating the floor **4.6x on curves**.

WHAT THIS MODULE PROVIDES
-------------------------
``ctrv_waypoints``        the floor itself (self-contained; a test pins it to
                          an independent heading-integration implementation).
``hold_v0_waypoints``     the ``driving.hold_v0`` floor, re-derived here so the
                          module is testable without the stack on sys.path.
``window_starts``         the window enumeration of ``rollout.collect``,
                          replicated so a legacy ``windows_<arm>.pt`` dump can
                          be BACKFILLED with a ``ctrv`` tensor.
``build_floors``          poses -> aligned floor tensors for a val cache.
``verify_alignment``      ⛔ THE PRECONDITION. Rebuilt ``cv``/``gt`` must match
                          the persisted tensors elementwise, or the backfill is
                          misaligned and NO verdict may be reported.

⚠️ ``verify_alignment`` exists because of C63 (``RETRACTION_LOG``): an imported
metric was published before its precondition was measured on our stack and the
numbers were an artifact. A backfilled floor is only as trustworthy as the
window alignment it assumes — so the alignment is measured, not assumed, and it
is measured against data that was persisted by the very code path being
replicated.

⚠️ THE LOW-SPEED CAVEAT, MEASURED BEFORE (2026-07-15). ``omega`` is a one-step
finite difference of yaw; at ``v -> 0`` the yaw signal is noise and an ungated
CTRV curls the prediction away for free. ``ctrv_waypoints(..., v_gate=...)``
zeroes ``omega`` below ``v_gate`` m/s. Both variants are reported; the gated one
is the honest floor, the ungated one is kept so the effect of the gate is
visible rather than assumed.

NO stack imports, torch + numpy only — so ``pytest`` runs this package
standalone (hub protocol, gate E).
"""
from __future__ import annotations

import math

import torch

WP_STEPS = (5, 10, 15, 20)          # 0.5 / 1 / 1.5 / 2 s @ 10 Hz
K_MAX = max(WP_STEPS)
WINDOW = 8                          # rollout.collect default
STRIDE = 8                          # rollout.collect default
DEFAULT_V_GATE = 2.0                # m/s — the speed gate from 2026-07-15


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ego(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """World displacement -> ego frame at heading ``yaw``."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


# --------------------------------------------------------------------------- #
# the floors                                                                   #
# --------------------------------------------------------------------------- #
def ctrv_waypoints(poses: torch.Tensor, last: torch.Tensor,
                   wp_steps=WP_STEPS, v_gate: float | None = None
                   ) -> torch.Tensor:
    """CTRV floor, ego frame, ``[b, H, 2]``.

    Constant turn rate + constant speed, forward-Euler: hold the one-step speed
    ``|p[last] - p[last-1]|`` and the one-step yaw rate ``wrap(yaw[last] -
    yaw[last-1])``, and integrate a circular arc for ``k`` sub-steps.

    ``v_gate`` (m/s per STEP-speed, i.e. metres per 0.1 s tick) zeroes ``omega``
    on windows whose entry speed is below it, which collapses CTRV onto
    go-straight there. Pass ``None`` for the ungated form.

    Identical in construction to
    ``stack/scripts/driving_diagnostic.baseline_waypoints()["constant_yaw_rate"]``
    (``test_matches_reference_arc`` pins it to an independent implementation).
    """
    if int(last.min()) < 1:
        raise ValueError("last must be >= 1 (the floor needs poses[last-1])")
    p0, pm1 = poses[last, :2], poses[last - 1, :2]
    yaw0, yawm1 = poses[last, 2], poses[last - 1, 2]
    v_world = p0 - pm1
    speed = v_world.norm(dim=-1)                       # metres per 0.1 s tick
    omega = _wrap(yaw0 - yawm1)
    if v_gate is not None:
        omega = torch.where(speed * 10.0 < v_gate, torch.zeros_like(omega),
                            omega)
    out = []
    for k in wp_steps:
        js = torch.arange(k, dtype=poses.dtype, device=poses.device)
        ang = js[None, :] * omega[:, None]              # heading at each sub-step
        out.append(torch.stack([speed * ang.cos().sum(dim=1),
                                speed * ang.sin().sum(dim=1)], dim=-1))
    return torch.stack(out, dim=1)


def cv_waypoints(poses: torch.Tensor, last: torch.Tensor,
                 wp_steps=WP_STEPS) -> torch.Tensor:
    """Constant-velocity floor, ego frame ``[b, H, 2]`` — the persisted ``cv``."""
    p0, pm1 = poses[last, :2], poses[last - 1, :2]
    ego_v = _ego(p0 - pm1, poses[last, 2])
    return torch.stack([ego_v * k for k in wp_steps], dim=1)


def hold_v0_waypoints(v0: torch.Tensor, n: int = 4, dt_wp: float = 0.5
                      ) -> torch.Tensor:
    """hold-v0 floor: go straight at the observed entry speed ``[b, n, 2]``.

    Mirrors ``taniteval.driving.hold_v0``. ``v0`` is metres/second (the
    persisted ``speed`` channel is ``poses[last, 3]``)."""
    t = torch.arange(1, n + 1, dtype=torch.float32) * dt_wp
    return torch.stack([v0[:, None] * t[None, :], torch.zeros(len(v0), n)], -1)


def gt_ego_waypoints(poses: torch.Tensor, last: torch.Tensor,
                     wp_steps=WP_STEPS) -> torch.Tensor:
    """Ground-truth ego-frame displacement to each horizon ``[b, H, 2]``."""
    p0, yaw0 = poses[last, :2], poses[last, 2]
    return torch.stack([_ego(poses[last + k, :2] - p0, yaw0) for k in wp_steps],
                       dim=1)


# --------------------------------------------------------------------------- #
# window enumeration — replicates rollout.collect exactly                      #
# --------------------------------------------------------------------------- #
def window_starts(T: int, window: int = WINDOW, stride: int = STRIDE,
                  k_max: int = K_MAX) -> list[int]:
    """``rollout.collect``'s window starts for an episode of usable length T.

    ``collect`` does ``starts = list(range(0, T - window - K_MAX, stride))`` and
    scores the window ending at ``last = t + window - 1``. Replicated verbatim,
    including the (deliberate) exclusive upper bound."""
    return list(range(0, T - window - k_max, stride))


def build_floors(episodes, window: int = WINDOW, stride: int = STRIDE,
                 k_max: int = K_MAX, wp_steps=WP_STEPS,
                 v_gate: float = DEFAULT_V_GATE) -> dict:
    """Aligned floors for a list of episodes, in ``rollout.collect`` order.

    ``episodes`` is a sequence of ``(episode_id, poses[T,4], T_usable)`` — the
    caller resolves ``T_usable`` the way ``collect`` does
    (``min(feats, actions, poses)``) so the enumeration cannot silently drift.

    Returns ``gt / cv / ctrv / ctrv_gated [N,H,2]``, ``holdv0 [N,H,2]``,
    ``eid [N] (list)``, ``speed [N]``.
    """
    GT, CV, CT, CTG, EID, SPD = [], [], [], [], [], []
    for eid, poses, T in episodes:
        poses = poses.float()
        starts = window_starts(T, window, stride, k_max)
        if not starts:
            continue
        last = torch.tensor([t + window - 1 for t in starts])
        GT.append(gt_ego_waypoints(poses, last, wp_steps))
        CV.append(cv_waypoints(poses, last, wp_steps))
        CT.append(ctrv_waypoints(poses, last, wp_steps, v_gate=None))
        CTG.append(ctrv_waypoints(poses, last, wp_steps, v_gate=v_gate))
        SPD.append(poses[last, 3])
        EID.extend([eid] * len(starts))
    speed = torch.cat(SPD).float()
    return {"gt": torch.cat(GT).float(), "cv": torch.cat(CV).float(),
            "ctrv": torch.cat(CT).float(), "ctrv_gated": torch.cat(CTG).float(),
            "holdv0": hold_v0_waypoints(speed, n=len(wp_steps)),
            "eid": EID, "speed": speed, "v_gate_mps": v_gate}


def _partition(eid) -> list[int]:
    """Canonical relabelling of an episode-id sequence to 0,1,2,... in order.

    Two id sequences describe the SAME clustering iff their canonical forms are
    equal — which is what an episode-cluster bootstrap actually depends on. The
    literal id VALUES are only a label."""
    seen: dict = {}
    out = []
    for e in eid:
        k = str(e)
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return out


def verify_alignment(built: dict, win: dict, tol: float = 1e-4) -> dict:
    """⛔ PRECONDITION GATE — rebuilt floors must land on the persisted windows.

    Compares the rebuilt ``cv`` and ``gt`` against the tensors persisted by
    ``rollout.collect`` in ``windows_<arm>.pt``. Both are produced by the code
    path being replicated, so an exact match is proof that window ``i`` here is
    window ``i`` there. Returns a record; ``["aligned"]`` is the predicate.

    ⛔ If this is False the backfilled ``ctrv`` is scored against the wrong
    ground truth and NOTHING derived from it may be reported.

    ⚠️ ``eid`` is checked as a **partition**, not as literal values. MEASURED
    2026-08-02 on the banked eval-pod dumps: ``windows_flagship-v4.1-10k.pt``
    and ``windows_flagship-v4.2-step4000.pt`` label the SAME 40 episodes with
    ``808464434, …`` (the episode's string uid reinterpreted as an int) where
    every other dump uses ``0..39`` — while their ``gt``/``cv``/``speed``
    tensors are bit-identical to the others. Requiring literal equality
    refuses two arms whose windows are provably aligned; requiring the
    partition is both correct for a cluster bootstrap and still strict.
    ``eid_labels_equal`` keeps the literal comparison visible, because a
    harness that mixes two id encodings will silently mis-join arms.
    """
    rec = {"n_built": int(built["gt"].shape[0]),
           "n_persisted": int(win["gt"].shape[0]),
           "eid_labels_equal": list(built["eid"]) == list(win["eid"]),
           "tol": tol}
    rec["n_equal"] = rec["n_built"] == rec["n_persisted"]
    if rec["n_equal"]:
        rec["eid_partition_equal"] = (_partition(built["eid"])
                                      == _partition(win["eid"]))
        rec["max_abs_diff_cv"] = float(
            (built["cv"] - win["cv"].float()).abs().max())
        rec["max_abs_diff_gt"] = float(
            (built["gt"] - win["gt"].float()).abs().max())
        rec["aligned"] = bool(rec["eid_partition_equal"]
                              and rec["max_abs_diff_cv"] < tol
                              and rec["max_abs_diff_gt"] < tol)
    else:
        rec["eid_partition_equal"] = False
        rec["max_abs_diff_cv"] = rec["max_abs_diff_gt"] = float("nan")
        rec["aligned"] = False
    return rec
