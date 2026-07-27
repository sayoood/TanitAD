"""C8 — the horizon-banked readout selection rule.

THE RULE
--------
**``op`` for lead times <= 0.5-0.8 s, ``str`` (or ``tac``) beyond.**

The ``op``/``tac``/``str`` step-displacement readouts
(:class:`~tanitad.models.metric_dynamics.HierarchicalGrounding`) are structurally
identical and differ only in the rollout length they were trained at (``op`` 4,
``tac`` 16, ``str`` 20 steps). Every production caller hard-wires
``grounding.step["op"]`` for the WHOLE path. MEASURED on v1 (``flagship-30k``,
step 29999; 599 windows / 596 episode clusters; paired episode-cluster bootstrap
B=2000), that is the wrong choice past ~1 s:

=========  ==========================  ============
lead       paired ``str`` - ``op``     separated?
=========  ==========================  ============
0.5 s      **+0.0449** [+0.0350, +0.0549]  yes (``op`` better)
1.0 s      +0.0254                          **no**
1.5 s      -0.0666                          yes (``str`` better)
2.0 s      -0.3414                          yes
18.5 s     -29.82                           yes
=========  ==========================  ============

⇒ the crossover sits between 0.5 s and 1.5 s with 1.0 s indistinguishable, which
is why the shipped switch is a **range-centred constant**, not a point estimate.

⛔ **DO NOT FIT THE SWITCH TO ADE.** MEASURED two-objective sweep (deployable
``own_kinematic`` regime):

============================  ============  ==========
switch at                     ``ade_0_2s``  ``T_blind``
============================  ============  ==========
0.0-0.3 s (= ``always_str``)  0.8710        2.5 s
**0.5 s**                     **0.8597**    **2.5 s**
0.8 s                         0.8597        2.5 s
1.0 s  <- the ADE optimum     0.8534        **0.8 s**  <- collapse
>= 2.0 s (= ``always_op``)    0.9554        0.8 s
============================  ============  ==========

The ADE-greedy switch buys **0.0063 m (0.7 %)** of ADE and pays **3.1x of
deployable ``T_blind``** (2.5 s -> 0.8 s). The 0.5-0.8 s rule is **Pareto-dominant
over the flat ``always_str`` swap** — strictly better ADE (0.8597 vs 0.8710) at
identical ``T_blind``.

**This module therefore ships the RULE and contains NO FITTER.** There is no
function here that takes ground truth, and
``test_the_module_ships_a_rule_and_CANNOT_fit_one`` enforces it by inspecting
every public signature. A constant that a future run could re-tune on held-out
ADE would reintroduce exactly the trade the measurement rejects.

THE SEMANTICS — and the artefact they carry
---------------------------------------------
C8 selects **which head's trajectory is READ at lead time j**, with every
specialist decoding the SAME latent rollout. It is **NOT** a per-step splice of
Δposes inside one SE(2) accumulation — that object was never validated and no
published system does it. So:

    roll the predictor ONCE (:func:`rollout_transitions`), decode the FULL path
    with each needed head (:func:`decode_transitions`), then take path
    ``r[j]``'s waypoint at index ``j``.

Cost is therefore one extra decode per additional level and **zero extra
rollout** — the "free" in the study's claim.

⚠️ **THE CONCATENATED PATH CAN BE DISCONTINUOUS AT THE SWITCH STEP.** Two heads
disagree about where the ego is at ``j = switch``, and the splice does not
interpolate. :func:`calibrated_rollout_decode` **measures that jump and returns
it** (``switch_discontinuity_m``) rather than leaving a downstream consumer to
discover a metre-scale step in a "smooth" trajectory. A consumer that needs a
continuous path must not use this function.

⚠️ **SCOPE.** C8's evidence is **v1 (`flagship-30k`) only**; the study says
verbatim it *"cannot support: v4 or any other arm — those dumps do not exist"*.
And the rule is a 12.5 % improvement on a number still **41 % worse than
``hold_v0`` (0.5933)**: *readout selection is real and free; it does not rescue
the deployable regime.*

⚠️ **REF-A/REF-B/REF-C arms have NO BANK** — they carry a single bare
``StepDisplacementReadout``. :func:`resolve_readout_bank` degrades to
one-readout-for-all-lead-times and SAYS SO in the returned telemetry, rather than
raising or silently pretending a bank exists.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tanitad.models.metric_dynamics import decode_transitions, rollout_transitions

__all__ = ["C8_SWITCH_STEP", "C8_SWITCH_RANGE_S", "C8_EARLY_LEVEL",
           "C8_LATE_LEVEL", "C8_DO_NOT_FIT", "DT",
           "readout_plan", "resolve_readout_bank", "calibrated_rollout_decode"]

DT = 0.1

#: The registered switch step. 5 steps = 0.5 s, the low end of the measured
#: 0.5-0.8 s range (0.5 s and 0.8 s score IDENTICALLY: 0.8597 at T_blind 2.5 s).
C8_SWITCH_STEP = 5
C8_SWITCH_RANGE_S = (0.5, 0.8)
C8_EARLY_LEVEL = "op"
C8_LATE_LEVEL = "str"

C8_DO_NOT_FIT = (
    "The switch step is a CONSTANT from a measured two-objective sweep, not a "
    "hyperparameter. Fitting it to ade_0_2s moves it to 1.0 s, which buys "
    "0.0063 m (0.7 %) of ADE and pays 3.1x of deployable T_blind (2.5 s -> "
    "0.8 s). Any code that re-tunes this on held-out error is making the trade "
    "the measurement rejected.")

#: Travels with every number this module produces.
C8_PROVENANCE = {
    "rule": "op for lead <= 0.5-0.8 s, str/tac beyond",
    "switch_step": C8_SWITCH_STEP, "switch_s": C8_SWITCH_STEP * DT,
    "measured_on": "v1 flagship-30k step 29999 (599 windows / 596 clusters)",
    "estimator": "paired episode-cluster bootstrap, B=2000, seed 0",
    "pareto": "0.8597 ade_0_2s at T_blind 2.5 s vs always_str 0.8710 at 2.5 s",
    "scope_limit": "v1 ONLY — no v4 dump exists; do not quote for another arm",
    "not_a_rescue": "still 41 % worse than hold_v0 (0.5933); selection is free, "
                    "not a fix",
    "evidence_class": "MEASURED (ours; X1_LATENT_METRIC.md §7, "
                      "artifacts/c8_selection_rule.json)",
    "do_not_fit": C8_DO_NOT_FIT,
}


def readout_plan(k: int, *, switch_step: int = C8_SWITCH_STEP,
                 early: str = C8_EARLY_LEVEL, late: str = C8_LATE_LEVEL,
                 available=("op", "tac", "str")) -> list[str]:
    """Per-lead-time readout level names for lead indices ``0 .. k-1``.

    Index ``j`` is lead time ``(j + 1) * 0.1 s``, so ``switch_step = 5`` means the
    first FIVE decoded steps (0.1-0.5 s) read ``early`` and everything from 0.6 s
    reads ``late``. Falls back to ``early`` for the whole path when ``late`` is
    not in ``available`` — a single-readout arm gets one readout, not a KeyError.
    """
    if late not in available:
        late = early
    if early not in available:
        early = late = next(iter(available))
    return [early if j < switch_step else late for j in range(int(k))]


def resolve_readout_bank(grounding) -> tuple[dict, dict]:
    """``(bank, telemetry)`` from a grounding object OR a bare readout module.

    ``bank`` maps level name -> readout module. A hierarchical grounding gives
    ``{"op":…, "tac":…, "str":…}``; anything else (REF-A/B/C's single
    ``StepDisplacementReadout``, or a plain callable) gives ``{"op": it}`` and a
    telemetry note that C8 is INACTIVE on this arm — degrading loudly in the
    record rather than quietly in the code.
    """
    step = getattr(grounding, "step", None)
    if step is not None and hasattr(step, "keys"):
        bank = {lvl: step[lvl] for lvl in step.keys()}
        return bank, {"bank_levels": sorted(bank), "c8_available": True}
    return ({"op": grounding},
            {"bank_levels": ["op"], "c8_available": False,
             "note": "this arm has a SINGLE step readout (no op/tac/str bank), "
                     "so C8 cannot apply; one readout is used for every lead "
                     "time and the result is NOT a C8 number."})


@torch.no_grad()
def calibrated_rollout_decode(predictor, states: Tensor, actions: Tensor,
                              future_actions: Tensor | None, grounding, k: int, *,
                              switch_step: int = C8_SWITCH_STEP,
                              early: str = C8_EARLY_LEVEL,
                              late: str = C8_LATE_LEVEL) -> tuple[Tensor, dict]:
    """C8 rollout: one roll, one decode per level, waypoints read per lead time.

    Returns ``(waypoints [B, k, 2], telemetry)``. The predictor is rolled EXACTLY
    ONCE and shared, so the only added cost over a single-readout decode is one
    extra ``decode_transitions`` per additional level used.

    ⚠️ The returned path may be DISCONTINUOUS at ``switch_step``; the telemetry's
    ``switch_discontinuity_m`` is the mean jump and exists so nobody has to
    discover it downstream.
    """
    bank, tele = resolve_readout_bank(grounding)
    plan = readout_plan(k, switch_step=switch_step, early=early, late=late,
                        available=tuple(bank))
    trans = rollout_transitions(predictor, states, actions, future_actions, k)
    decoded = {lvl: decode_transitions(bank[lvl], trans, k)[0]
               for lvl in sorted(set(plan))}

    wp = torch.stack([decoded[plan[j]][:, j] for j in range(k)], dim=1)

    tele.update({
        "c8": dict(C8_PROVENANCE),
        "switch_step": int(switch_step),
        "switch_s": round(switch_step * DT, 2),
        "levels_used": sorted(set(plan)),
        "per_step_level": plan,
        "levels_decoded": len(decoded),
        "rollouts_executed": 1,
        "_cost": "one predictor roll shared by every level; the extra cost is "
                 f"{max(len(decoded) - 1, 0)} decode pass(es), no extra rollout",
    })
    if 0 < switch_step < k and len(set(plan)) > 1:
        jump = (decoded[plan[switch_step]][:, switch_step]
                - decoded[plan[switch_step - 1]][:, switch_step]).norm(dim=-1)
        tele["switch_discontinuity_m"] = round(float(jump.mean()), 4)
        tele["switch_discontinuity_max_m"] = round(float(jump.max()), 4)
        tele["_discontinuity_note"] = (
            "C8 reads a DIFFERENT head's path either side of the switch; the "
            "concatenated path is not interpolated and can step. This is the "
            "known artefact of the validated semantics, reported not hidden. A "
            "consumer needing a continuous path must not use this function.")
    else:
        tele["switch_discontinuity_m"] = 0.0
    return wp, tele
