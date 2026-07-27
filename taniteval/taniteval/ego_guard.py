"""taniteval.ego_guard — REFUSE to score an ego-TRAINED checkpoint ego-BLIND.

THE LIVE BUG THIS MODULE CLOSES (MEASURED 2026-07-27/28)
--------------------------------------------------------
``TacticalPolicy.forward(states, ctx, ego=None)`` and
``StrategicPolicy.forward(states, nav_cmd, ego=None)`` guard their ego term with

    if self.ego_emb is not None and ego is not None:

— a **build-time** flag AND a **call-time** argument
(``tanitad/models/fourbrain.py:330``). Every audit before 2026-07-28 checked the
build flag (``cfg.v2_ego_to_planners``, ``config.py:196``) and concluded "the
lever is off". It is — **and the call-site half is independently, silently false
in 100 % of evaluation paths**: ``ego=`` is passed at exactly three call sites in
the whole repo (``train/flagship_losses.py:245,246,351``), **all three in the
TRAINER**. Not one evaluation path passes it.

⇒ A checkpoint TRAINED with the ego lever is EVALUATED without it, silently, with
no error and no log line. Its trained ``ego_emb`` weights are simply never
exercised and the arm is mis-attributed as *"the ego lever does nothing"*.
``flagship-v2corpus-30k`` is training with ``v2_ego_to_planners = true`` right
now, so this is a live bug on a run in progress, not a hypothetical.

WHY IT REFUSES RATHER THAN WARNS  (the decision, and the reason)
---------------------------------------------------------------
Default is :data:`MODE_REFUSE`. The alternative — log a warning and score anyway
— was rejected for four reasons, each of which this program has already paid for:

1. **A warning still produces a number, and a number that exists gets quoted.**
   Refusal produces nothing to quote. The failure being prevented is not a crash,
   it is a *silently wrong attribution* of a GPU-week arm.
2. **"Recorded somewhere" is a known-failed control here.** An integration
   request in a README sat unread for 10 days; a stale absence-claim propagated
   into >=7 documents. A stderr line in a multi-hour eval log is weaker than both.
3. **Refusing costs ZERO published numbers.** Every arm in the 2026-07-27 panel
   and every checkpoint in ``MODEL_REGISTRY`` has ``ego_emb is None``, so the
   guard is a *provable* no-op for all of them (pinned by
   ``test_guard_is_a_NO_OP_for_every_arm_in_the_published_panel``). It can only
   fire on the new capability class it was written for.
4. **The escape hatch is better than a warning, because it is semantic.** An
   ego-ABLATED arm is a legitimate experiment — you get it by passing an explicit
   ``ego = 0`` (IN-distribution when the run used ``v2_ego_dropout``), which is a
   *different object* from ``ego=None`` (that skips the ``ego_emb`` bias too).
   Making the ablation say so in code is the whole point.

:data:`MODE_WARN` exists for one case only — a batch re-score of legacy arms
where a stop would strand the whole sweep — and it is deliberately NOT free: it
must be turned on explicitly (``TANITEVAL_EGO_GUARD=warn``), it emits a
``EgoInputDroppedWarning``, and it stamps ``ego_input_DROPPED = True`` into the
provenance node so the artifact carries the defect. A number produced in warn
mode is identifiable as such from the JSON alone.

WHAT A CALLER MUST DO
---------------------
::

    from taniteval.ego_guard import assert_planner_ego, ego_from_poses

    ego = ego_from_poses(ep.poses, last, pose_scale, device)   # or None
    prov = assert_planner_ego(model, ego, where="myeval.py:123")
    ctx = model.strategic_policy(states, nav, ego=ego)["ctx"]
    wp = model.tactical_policy(states, ctx, ego=ego)["waypoints"]
    node["ego_input"] = prov          # <- the fact rides with the number

``prov`` is returned in every mode and must be stamped into the emitted node.
``taniteval/tests/test_ego_guard.py::test_every_shipped_eval_call_site_is_guarded``
scans the shipped eval modules and FAILS if a new planner call site appears
without a guard above it — the guard cannot be forgotten by the next author,
which is the only thing that makes it different from the last three "just
remember to call it" controls.

⚠️ Not every eval path CAN feed the vector. ``closed_loop_rollout`` plans on
IMAGINED latents from tick 1 on and has no pose history in scope, so it can only
supply proprioception at tick 0. That is a real limitation and the guard's
refusal is the correct output there: scoring an ego-trained checkpoint on a
harness that structurally cannot supply its input is exactly the
mis-attribution being prevented.
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "stack"), "/root/TanitAD/stack"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = [
    "MODE_REFUSE", "MODE_WARN", "ENV_VAR", "DEFAULT_MODE",
    "EgoInputDroppedWarning", "guard_mode", "planner_ego_capability",
    "assert_planner_ego", "assert_adapter_declares_ego", "ego_from_poses",
    "POSE_SCALE_DEFAULT", "DT", "ADAPTER_PROVENANCE_ATTR",
]

#: An adapter that wraps a 4-brain model for the pseudo-simulation ``.traj``
#: protocol declares its ego handling by setting this attribute to the dict
#: returned by :func:`assert_planner_ego`.
ADAPTER_PROVENANCE_ATTR = "ego_provenance"

#: Refuse to evaluate (raise ``tanitad.ego_plan.EgoInputDropped``). THE DEFAULT.
MODE_REFUSE = "refuse"
#: Score anyway, but warn AND stamp ``ego_input_DROPPED`` into the node.
MODE_WARN = "warn"
ENV_VAR = "TANITEVAL_EGO_GUARD"
DEFAULT_MODE = MODE_REFUSE

#: Pose-history tick, matching the trainer's ``/ 0.1`` yaw-rate divisor
#: (``train/flagship_losses.py:207``) and ``taniteval.clhorizon.DT``.
DT = 0.1
#: ``--pose-scale`` default in every 4-brain trainer
#: (``stack/scripts/train_flagship4b.py:713``). ⚠️ READ IT FROM THE CHECKPOINT
#: when one is available — this constant is the fallback, not the contract, and
#: a wrong scale feeds the brain a wrong number rather than no number.
POSE_SCALE_DEFAULT = 10.0


class EgoInputDroppedWarning(UserWarning):
    """Warn-mode counterpart of ``tanitad.ego_plan.EgoInputDropped``."""


def _ego_plan():
    from tanitad import ego_plan                       # noqa: PLC0415
    return ego_plan


def guard_mode() -> str:
    """``MODE_REFUSE`` unless ``TANITEVAL_EGO_GUARD=warn`` is set explicitly.

    An unrecognised value is an ERROR, not a silent fallback to the permissive
    mode — a typo'd env var must never be the thing that disables a guard."""
    raw = os.environ.get(ENV_VAR, DEFAULT_MODE).strip().lower()
    if raw not in (MODE_REFUSE, MODE_WARN):
        raise ValueError(
            f"{ENV_VAR}={raw!r} is not a mode. Use {MODE_REFUSE!r} (default) or "
            f"{MODE_WARN!r}. Refusing to fall back to the permissive mode on a "
            f"typo: that is how a guard silently stops guarding.")
    return raw


def planner_ego_capability(model) -> dict:
    """What the CHECKPOINT can do with an ego vector — read off the modules.

    Reported for every arm whether or not the guard fires, so ``MODEL_REGISTRY``
    and every eval node can carry ``ego_input_on_planners`` as a MEASURED fact
    instead of an inherited one (escalation E7 of the 2026-07-28 report)."""
    out = {}
    for name in ("strategic_policy", "tactical_policy"):
        pol = getattr(model, name, None)
        emb = getattr(pol, "ego_emb", None) if pol is not None else None
        out[name] = {
            "present": pol is not None,
            "has_trained_ego_emb": emb is not None,
            "shape": (None if emb is None
                      else [int(emb.in_features), int(emb.out_features)]),
        }
    out["ego_input_on_planners"] = any(
        v.get("has_trained_ego_emb") for v in out.values()
        if isinstance(v, dict))
    return out


def assert_planner_ego(model, ego, *, where: str, mode: str | None = None,
                       ego_source: str = "unspecified") -> dict:
    """⛔ THE CALL. Run it immediately before every planner forward.

    Delegates the predicate to ``tanitad.ego_plan.assert_ego_is_fed`` — the
    guard is IMPORTED, never reimplemented, so taniteval and tanitad cannot
    drift on what "fed" means.

    Returns the provenance node in EVERY mode; stamp it into whatever the caller
    emits. Raises ``tanitad.ego_plan.EgoInputDropped`` in ``refuse`` mode when a
    policy owns trained ``ego_emb`` weights and ``ego is None``.
    """
    mode = guard_mode() if mode is None else mode
    ep = _ego_plan()
    cap = planner_ego_capability(model)
    prov = {
        "_what": ("whether the planner brains' ego port was FED at this call "
                  "site. A checkpoint trained with v2_ego_to_planners and "
                  "evaluated with ego=None scores as though the lever did "
                  "nothing — silently."),
        "where": where,
        "guard": "tanitad.ego_plan.assert_ego_is_fed",
        "guard_mode": mode,
        "capability": cap,
        "ego_fed": ego is not None,
        "ego_source": ego_source,
        "ego_shape": (None if ego is None else list(ego.shape)),
    }
    dropped = bool(cap["ego_input_on_planners"]) and ego is None
    prov["ego_input_DROPPED"] = dropped
    if not dropped:
        return prov
    if mode == MODE_WARN:
        warnings.warn(
            f"{where}: planner brains own TRAINED ego weights but ego=None — "
            f"they are silently unused and this arm is being scored EGO-BLIND. "
            f"Running anyway because {ENV_VAR}={MODE_WARN}; the node is stamped "
            f"ego_input_DROPPED=True and the number is NOT comparable to an "
            f"ego-fed one.", EgoInputDroppedWarning, stacklevel=2)
        return prov
    for name in ("strategic_policy", "tactical_policy"):
        pol = getattr(model, name, None)
        if pol is not None:
            ep.assert_ego_is_fed(pol, ego, where=f"{where} [{name}]")
    raise ep.EgoInputDropped(                     # pragma: no cover - defensive
        f"{where}: capability says the ego port is trained but neither brain "
        f"raised; the capability probe and the guard disagree.")


def assert_adapter_declares_ego(planner, *, where: str,
                                mode: str | None = None) -> dict:
    """⛔ THE PSEUDO-SIMULATION HOOK — the surface v5 is gated on.

    ``pseudo_evaluate`` never touches the policy brains itself; it calls an
    adapter's ``.traj(frames, v0, goal)``. So the guard cannot sit at a policy
    call site there — it has to ask the ADAPTER whether it handled the ego port.

    An adapter that wraps a 4-brain whose planner brains own trained ``ego_emb``
    weights MUST set ``planner.ego_provenance`` (the dict
    :func:`assert_planner_ego` returns). An adapter that does not is refused:
    that is precisely the ``panel_run.py`` shape that would have scored
    ``flagship-v2corpus-30k`` ego-blind.

    Adapters over a v4/v5 ``FlagshipV4Head`` are unaffected — the head takes
    ``v0`` as a first-class argument and owns no ``ego_emb``.
    """
    mode = guard_mode() if mode is None else mode
    declared = getattr(planner, ADAPTER_PROVENANCE_ATTR, None)
    caps = []
    for attr in ("model", "world", "wm"):
        obj = getattr(planner, attr, None)
        if obj is not None and (hasattr(obj, "tactical_policy")
                                or hasattr(obj, "strategic_policy")):
            caps.append((attr, planner_ego_capability(obj)))
    if hasattr(planner, "tactical_policy") or hasattr(planner, "strategic_policy"):
        caps.append(("self", planner_ego_capability(planner)))
    needs = [a for a, c in caps if c["ego_input_on_planners"]]
    prov = {
        "_what": ("whether the pseudo-simulation ADAPTER handled the planner "
                  "brains' ego port. pseudo_evaluate calls .traj(); it cannot "
                  "see the policy call inside."),
        "where": where,
        "guard_mode": mode,
        "adapter": type(planner).__name__,
        "wrapped_models_with_trained_ego_emb": needs,
        "declared": declared,
    }
    dropped = bool(needs) and declared is None
    prov["ego_input_DROPPED"] = dropped
    if not dropped:
        return prov
    msg = (f"{where}: adapter {type(planner).__name__} wraps a model whose "
           f"planner brains own TRAINED ego weights ({', '.join(needs)}) but "
           f"declares no `{ADAPTER_PROVENANCE_ATTR}`. It would be scored "
           f"EGO-BLIND and mis-attributed as 'the ego lever does nothing'. Pass "
           f"ego= inside .traj() and set "
           f"planner.{ADAPTER_PROVENANCE_ATTR} = assert_planner_ego(...), or "
           f"declare an explicit zero-ego ablation.")
    if mode == MODE_WARN:
        warnings.warn(msg, EgoInputDroppedWarning, stacklevel=2)
        return prov
    raise _ego_plan().EgoInputDropped(msg)


def ego_from_poses(poses, last, pose_scale: float = POSE_SCALE_DEFAULT,
                   device=None, dt: float = DT):
    """``[n, 2]`` ``[v0 / pose_scale, yr0]`` from OBSERVED poses at ``t``/``t-1``.

    VERBATIM the trainer's construction (``train/flagship_losses.py:202-210``),
    delegating the packing to ``tanitad.ego_plan.ego_vector`` so the contract
    lives in exactly one place. ``poses`` is ``[T, >=4]`` with columns
    ``(x, y, yaw, speed)``; ``last`` is the index of the last OBSERVED frame of
    each window.

    ⚠️ ``pose_scale`` must come from the checkpoint's own training args when one
    is available. Feeding the port at the WRONG scale is worse than the bug this
    module exists to close: ``ego=None`` merely wastes the weights, a mis-scaled
    ego actively decodes garbage (registry §1.2).

    ``last == 0`` has no ``t-1``; the trainer's own fallback for a cache with no
    ``pose_prev`` is ``yr0 = 0`` (``flagship_losses.py:209``) and this reproduces
    it rather than inventing a rate.
    """
    p = torch.as_tensor(poses, dtype=torch.float32)
    idx = torch.as_tensor(last, dtype=torch.long).reshape(-1)
    prev = (idx - 1).clamp_min(0)
    v0 = p[idx, 3]
    dyaw = p[idx, 2] - p[prev, 2]
    yr0 = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)) / float(dt)
    yr0 = torch.where(idx > 0, yr0, torch.zeros_like(yr0))
    ego = _ego_plan().ego_vector(v0, yr0, float(pose_scale))
    return ego if device is None else ego.to(device)
