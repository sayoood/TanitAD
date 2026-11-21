"""`ANCHOR_GOAL(anchor_id, t_reach_s)` LABEL DERIVATION — the label side only.

⛔ **ADMISSIBILITY, and why this module is deliberately small.**

*"Labels may use ego; inference is vision-only"* (Sayed 2026-08-03). This is the
**LABEL** side, so ego — including FUTURE poses — is admissible here and nowhere
else. Nothing in this module runs at inference.

⛔ **AND THE SECOND, SHARPER RULE, ENFORCED RATHER THAN DOCUMENTED.** *"The goal
input must not carry the situation classifier's output."* The hazard for a label
deriver is subtle: ``ph0_pilot.engine_a_summary`` computes a ``situations`` block
from the very pose track an `ANCHOR_GOAL` label comes from, and a deriver that
consumed that summary wholesale would train the goal head to reproduce a function
of the classifier's own detectors — at which point a planner gain is unassignable
between the goal and the situation (the `--v2` conflation, verbatim).

⇒ **This module NEVER reads ``tanitad.data.situations``, and takes ego-frame
ENDPOINTS rather than a summary dict, so there is no channel through which a
classifier output could arrive.** Pinned by
``tests/test_anchor_goal_labels.py::test_module_has_no_situation_classifier_path``.
*(The same load-bearing omission as ``ph0_pilot._fmt_engine_a``, whose own comment
says the omission is deliberate.)*

⭐ **THE ECHO TEST, for the record.** The label is a function of ``poses[t : t+h]``
— the FUTURE. At inference the goal point is ``anchor_table[argmax logits]``, a
FROZEN buffer indexed by a vision-derived logit; no inference input is a function
of those future poses. The anchor table is shared by every window and therefore
carries zero per-window information — which is exactly what the goal-echo control
(goal ← the vocabulary's centroid) measures, at **13.5553 m** against live arms at
0.79–9.49 m.

⚠️ **THIS DERIVER REFUSES MORE THAN IT EMITS TODAY, ON PURPOSE.** Two refusals are
findings, not defects:

1. **No anchor vocabulary in the programme reaches the tactical band.** Every one
   built by ``build_refc_anchors.py`` stops at 20 steps = **2.0 s**
   (`refc_anchors_full_REBUILD.pt` horizons ``[5,10,15,20]``;
   `flagship_v4_anchors_dense.pt` ``[1..20]``), while ``TAC_BAND_S`` is
   **(2.0, 6.0)** and the v6f selector scores the **6 s** endpoint. Emitting a
   2 s "tactical" goal would look exactly like a label.
2. **``anchor_id`` is CATEGORICAL and the goal vocabulary's arg slots are not.**
   ``v6.GOAL_ARG_NAMES`` is eight slots of *"PHYSICAL UNITS (m, s, m/s)"*, consumed
   by ``GoalVocabulary.arg_proj = nn.Linear(n_args, d_embed)``. With an FPS-ordered
   vocabulary, anchor 5 is not "between" anchors 4 and 6 in any geometry, so
   writing an index into a physical-units slot is a type error. ⇒ this deriver
   emits ``anchor_id`` as its **own integer field** and leaves the continuous slot
   **NaN with mask 0** — the `IGNORE` discipline — so the gap stays visible instead
   of being papered over with a plausible float.

MEASURED context for anyone reading this before building on it
(`…/incoming/2026-08-16-anchor-goal-supervision/`): the quantisation floor of the
shipped 256-anchor vocabulary is **σ(2 s) = 0.5637 m [0.5085, 0.6185]**, inside the
FUNDED band — but a K-way classifier on frozen REF-C vision latents reaches
σ **9.4868 m**, **+4.7502 [+3.0514, +6.3981] separated WORSE** than the free ridge
that was already refused. ⇒ **the labels are necessary and not sufficient**, and
the surface is what has to move.
"""
from __future__ import annotations

import torch
from torch import Tensor

#: mirrors ``tanitad.models.v6.GOAL_ARG_SLOTS`` — pinned equal by a test rather
#: than imported, so a vocabulary change fires here instead of drifting silently.
ARG_SLOTS = 8
#: ``HIERARCHY_VOCABULARY.md:84`` — ``ANCHOR_GOAL(anchor_id, t_reach_s)``, in order.
ARG_LAYOUT: dict[str, int] = {"anchor_id": 0, "t_reach_s": 1}
#: ``v6.TAC_BAND_S`` — the band this goal is allowed to ground in.
TAC_BAND_S: tuple[float, float] = (2.0, 6.0)
TOKEN = "ANCHOR_GOAL"

__all__ = ["ARG_SLOTS", "ARG_LAYOUT", "TAC_BAND_S", "TOKEN",
           "anchor_endpoints", "assign_anchor", "anchor_goal_labels"]


def anchor_endpoints(anchors: Tensor, horizons, step: int,
                     dt: float = 0.1) -> Tensor:
    """``anchors`` [K, S, 2] + its ``horizons`` -> the [K, 2] endpoint at ``step``.

    ⛔ REFUSES unless ``horizons`` actually contains ``step``. ``anchors[:, -1]``
    silently means "2.0 s" for both a 4-point and a 20-point vocabulary, so a 6 s
    ground truth scored against either would produce a number rather than an
    error — the failure this check exists to make impossible.
    """
    a = torch.as_tensor(anchors)
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f"anchors must be [K, S, 2], got {tuple(a.shape)}")
    hs = [int(h) for h in horizons]
    if len(hs) != a.shape[1]:
        raise ValueError(f"{len(hs)} horizons for {a.shape[1]} anchor points")
    if int(step) not in hs:
        raise ValueError(
            f"⛔ this vocabulary covers horizons {hs} ({[h * dt for h in hs]} s) "
            f"and does NOT contain step {step} ({step * dt:g} s) — refusing "
            f"rather than reading a different horizon off it")
    return a[:, hs.index(int(step)), :]


def assign_anchor(endpoints: Tensor, anchor_ends: Tensor) -> dict:
    """Nearest-anchor assignment. ``endpoints`` [n, 2], ``anchor_ends`` [K, 2].

    Returns ``ids`` [n], ``residual`` [n, 2] (= endpoint − chosen anchor), and
    ``residual_second`` [n, 2] for the runner-up — the near-miss reference the
    required-accuracy algebra needs, and which a bare argmin would not preserve.
    """
    e = torch.as_tensor(endpoints, dtype=torch.float64)
    A = torch.as_tensor(anchor_ends, dtype=torch.float64)
    if e.ndim != 2 or e.shape[-1] != 2:
        raise ValueError(f"endpoints must be [n, 2], got {tuple(e.shape)}")
    if A.ndim != 2 or A.shape[-1] != 2:
        raise ValueError(f"anchor_ends must be [K, 2], got {tuple(A.shape)}")
    d2 = ((e[:, None, :] - A[None, :, :]) ** 2).sum(-1)              # [n, K]
    order = d2.argsort(dim=1)
    ids = order[:, 0]
    second = order[:, 1] if A.shape[0] > 1 else ids
    return {"ids": ids, "residual": e - A[ids],
            "residual_second": e - A[second], "d2": d2}


def anchor_goal_labels(gt_endpoint: Tensor, valid: Tensor, anchors: Tensor,
                       horizons, step: int, *, dt: float = 0.1,
                       band_s: tuple[float, float] = TAC_BAND_S,
                       allow_operative_band: bool = False,
                       vocab_meta: dict | None = None) -> dict:
    """``ANCHOR_GOAL`` labels for a batch of windows.

    ``gt_endpoint`` [n, 2] is the **ego-frame displacement at ``step``**, x forward
    / y left, in the frame of the window origin — i.e. exactly one horizon of
    ``driving_diagnostic.gt_ego_waypoints(poses, last, wp_steps=[step])``. Taking
    the endpoint rather than a pose track or a summary dict is what keeps this
    module free of any channel a situation-classifier output could travel down.

    ``valid`` [n] bool marks windows whose horizon fits inside the episode.
    ⛔ Invalid rows are labelled ``valid=False`` with a reason and are **never
    imputed**; the caller reports ``n`` per horizon.

    ⛔ Refuses when the anchor endpoint's time is outside ``band_s`` — see the
    module docstring. ``allow_operative_band=True`` runs it anyway and stamps the
    departure into the provenance, so a 2 s diagnostic can never be mistaken for a
    tactical label.
    """
    t_reach = float(step) * float(dt)
    lo, hi = float(band_s[0]), float(band_s[1])
    off_band = not (lo < t_reach <= hi)
    if off_band and not allow_operative_band:
        raise ValueError(
            f"⛔ t_reach = {t_reach:g}s is outside the tactical band "
            f"({lo:g}, {hi:g}] — this would be an OPERATIVE-band point wearing a "
            f"tactical goal's name. No anchor vocabulary in the programme reaches "
            f"the band; pass allow_operative_band=True for a stamped diagnostic.")

    ends = anchor_endpoints(anchors, horizons, int(step), dt=dt)
    e = torch.as_tensor(gt_endpoint, dtype=torch.float64)
    v = torch.as_tensor(valid).bool().reshape(-1)
    if e.ndim != 2 or e.shape[-1] != 2:
        raise ValueError(f"gt_endpoint must be [n, 2], got {tuple(e.shape)}")
    if v.shape[0] != e.shape[0]:
        raise ValueError(f"valid is [{v.shape[0]}] for [{e.shape[0]}] endpoints")
    finite = torch.isfinite(e).all(dim=-1)
    ok = v & finite

    n = e.shape[0]
    ids = torch.full((n,), -1, dtype=torch.long)
    resid = torch.full((n, 2), float("nan"), dtype=torch.float64)
    resid2 = torch.full((n, 2), float("nan"), dtype=torch.float64)
    if bool(ok.any()):
        a = assign_anchor(e[ok], ends)
        ids[ok] = a["ids"]
        resid[ok] = a["residual"]
        resid2[ok] = a["residual_second"]

    # ⛔ arg0 (anchor_id) is CATEGORICAL and this slot is not. NaN + mask 0.
    args = torch.full((n, ARG_SLOTS), float("nan"), dtype=torch.float64)
    mask = torch.zeros((n, ARG_SLOTS), dtype=torch.float64)
    args[:, ARG_LAYOUT["t_reach_s"]] = t_reach
    mask[ok, ARG_LAYOUT["t_reach_s"]] = 1.0

    sq = (resid[ok] ** 2).sum(-1) if bool(ok.any()) else torch.zeros(0)
    return {
        "token": TOKEN,
        "anchor_id": ids,                       # ⭐ the categorical, its own field
        "t_reach_s": t_reach,
        "args": args, "arg_mask": mask,
        "valid": ok,
        "invalid_reason": ("endpoint_valid False (the horizon runs past the end "
                           "of the episode) or a non-finite endpoint — excluded "
                           "with n, NEVER imputed"),
        "residual": resid, "residual_second": resid2,
        "n": int(n), "n_valid": int(ok.sum()),
        "sigma_perax_m": (float((sq.mean() / 2.0).sqrt()) if sq.numel() else None),
        "provenance": {
            "step": int(step), "dt": float(dt), "horizon_s": t_reach,
            "band_s": [lo, hi],
            "off_band": bool(off_band),
            "off_band_stamp": (
                "⚠️ OPERATIVE-BAND DIAGNOSTIC — t_reach is outside the tactical "
                "band and this is NOT a tactical goal label" if off_band else None),
            "vocab": dict(vocab_meta or {}),
            "frame": "ego frame at the window origin; x forward, y left, metres",
            "arg0_is_unset_because":
                "anchor_id is CATEGORICAL and GOAL_ARG_NAMES holds PHYSICAL UNITS "
                "only; writing an index there is a type error, so the slot is NaN "
                "with mask 0 and the id travels in its own field",
            "situation_disjoint":
                "derived from ego-frame FUTURE displacement and a frozen anchor "
                "table only. tanitad.data.situations is never read.",
        },
    }
