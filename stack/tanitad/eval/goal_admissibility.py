"""GOAL ADMISSIBILITY — the check the PI made binding, as an instrument.

WHY THIS EXISTS
---------------
Two rulings from 2026-08-03 govern every goal / route / target-speed signal:

  * *"a goal input is admissible, at the same time, we need to be careful not to
    include the result of the situation classification in the goal input."*
  * labels may use ego, other agents, maps and future poses; **inference** is the
    constrained side.

Both were being checked by reading prose. Prose lost, twice:

  * **the nav echo.** flagship-v1's route head is an EXACT BIJECTION of the nav it
    is fed — 369/369 and 81/81, 181/181 under manipulation — and scored
    ``1.0000``. An echo of its own input was read as skill for as long as nobody
    computed the mapping.
  * **a supplied route on PhysicalAI**, whose only route supplier is the ego's own
    future path, is optimistic by construction.

``route_cf.py`` already asks *does the model USE its route input*. This module
asks the orthogonal question — **is the model ALLOWED to have this input** — and
returns a verdict a gate can act on. Every check is mechanical: a rate, a set
operation, or an out-of-fold R², never a promise about anybody's state of mind.

THE THREE CHECKS, and what each can and cannot catch
----------------------------------------------------
1. :func:`echo_score` — is the scored OUTPUT a deterministic function of an
   INPUT? Catches the nav echo exactly. **Cannot** catch a leak that is
   statistical rather than functional.
2. :func:`horizon_disjoint` — does the signal's derivation read the scored
   horizon? Catches the VTARGET overlap. **Cannot** catch autocorrelation across
   a disjoint boundary, which is why check 3 exists and is not optional.
3. :func:`incremental_information` — how much does the signal add ABOVE what the
   model already legally holds? The substantive residual. ⚠️ A large value on a
   privileged signal is a LEAK; the same value on a PREDICTED signal is the
   lever working. The module reports the number and the caller must state which
   side the signal sits on — :func:`audit_goal_signal` forces that by requiring
   ``supplied_at_inference``.

⛔ NOT A SUBSTITUTE FOR PROVENANCE. Whether a signal is derived from the
situation classifier's output is a fact about the computation graph, not
something a correlation can settle: a shared trunk can make an independent signal
look correlated and a laundered one look clean. :func:`situation_disjoint` takes
the DECLARED provenance and checks it structurally, in the same spirit as
``RefCModel._goal_provenance``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["echo_score", "horizon_disjoint", "incremental_information",
           "situation_disjoint", "audit_goal_signal", "ECHO_FLAG_RATE"]

#: A functional agreement at or above this rate is reported as an ECHO. 0.98 and
#: not 1.0 because a real bijection can be broken by a handful of tie-breaks and
#: still be an echo; 369/369 and 181/181 (the measured nav case) are far above it.
ECHO_FLAG_RATE = 0.98


def echo_score(inputs, outputs) -> dict:
    """Is ``outputs`` a deterministic function of ``inputs``? -> the nav-echo test.

    Both are 1-D label arrays over the SAME windows. For each distinct input
    value, take the modal output; the agreement rate is the best accuracy any
    lookup table on this input could achieve. ``1.0`` means the output is exactly
    recoverable from the input — the head is echoing, and any score it earns is a
    property of its own input rather than of the world.

    Also returns ``bijection`` (the map is one-to-one in both directions), which
    is the flagship-v1 route-head shape specifically.
    """
    a = np.asarray(inputs).reshape(-1)
    b = np.asarray(outputs).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"echo_score needs aligned arrays: {a.shape} vs {b.shape}")
    if a.size == 0:
        return {"n": 0, "status": "UNPOWERED",
                "reason": "no windows — absence of an echo was NOT established"}
    hit, table = 0, {}
    for u in np.unique(a):
        m = a == u
        vals, cnt = np.unique(b[m], return_counts=True)
        table[str(u)] = str(vals[cnt.argmax()])
        hit += int(cnt.max())
    rate = hit / a.size
    n_in, n_out = len(np.unique(a)), len(np.unique(b))
    one_to_one = n_in == len(set(table.values())) == n_out
    return {
        "n": int(a.size),
        "n_input_values": int(n_in),
        "n_output_values": int(n_out),
        "functional_agreement": round(float(rate), 6),
        "bijection": bool(one_to_one and rate >= 1.0),
        "mapping": table,
        "is_echo": bool(rate >= ECHO_FLAG_RATE),
        "_reads": ("functional_agreement is the accuracy of the BEST possible "
                   "lookup table from this input to this output. At 1.0 the "
                   "output carries no information the input did not already "
                   "carry, so any score it earns is an echo, not skill."),
    }


def horizon_disjoint(read_lo: int, read_hi: int, scored_lo: int,
                     scored_hi: int) -> dict:
    """Does a signal's derivation window touch the scored horizon?

    Half-open ``[lo, hi)`` on both sides, in the SAME index space (pose steps).
    This is the mechanical half of the guard: it is decided by set arithmetic and
    cannot be argued with.
    """
    read = set(range(int(read_lo), int(read_hi)))
    scored = set(range(int(scored_lo), int(scored_hi)))
    overlap = sorted(read & scored)
    return {
        "read_window": [int(read_lo), int(read_hi)],
        "scored_window": [int(scored_lo), int(scored_hi)],
        "n_overlap_steps": len(overlap),
        "disjoint": not overlap,
        "_reads": ("disjoint=False means the signal is computed from a superset "
                   "of the thing being measured. Admissible as a LABEL; "
                   "INADMISSIBLE as an input at inference."),
    }


def incremental_information(y, x_legal, x_extra, eid, *, ridge: float = 1e-3
                            ) -> dict:
    """Out-of-fold R^2 gained by adding ``x_extra`` to what the model already has.

    ``x_legal`` [N, d] is everything the model legally holds at inference (e.g. the
    causal past); ``x_extra`` [N, k] is the candidate signal; ``y`` [N] is the
    scored quantity. Folds are val EPISODES — windows inside a clip are strongly
    dependent and any other fold unit inflates both arms.

    ⚠️ Interpretation depends entirely on which side the signal sits on, which is
    why this function does not emit a verdict. See :func:`audit_goal_signal`.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xl = np.asarray(x_legal, dtype=np.float64).reshape(len(y), -1)
    xe = np.asarray(x_extra, dtype=np.float64).reshape(len(y), -1)
    eid = np.asarray(eid).reshape(-1)
    if not (len(xl) == len(xe) == len(eid) == len(y)):
        raise ValueError("y / x_legal / x_extra / eid must align")

    def _oof(x):
        out = np.empty_like(y)
        for e in np.unique(eid):
            te = eid == e
            tr = ~te
            mu = x[tr].mean(0)
            sd = np.where(x[tr].std(0) < 1e-9, 1.0, x[tr].std(0))
            a = np.column_stack([(x[tr] - mu) / sd, np.ones(int(tr.sum()))])
            m = a.T @ a + ridge * np.eye(a.shape[1])
            m[-1, -1] -= ridge
            w = np.linalg.solve(m, a.T @ y[tr])
            out[te] = np.column_stack([(x[te] - mu) / sd,
                                       np.ones(int(te.sum()))]) @ w
        return out

    ss = float(((y - y.mean()) ** 2).sum())

    def _r2(p):
        return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")

    p_l, p_b = _oof(xl), _oof(np.column_stack([xl, xe]))
    return {
        "n": int(len(y)), "n_episodes": int(len(np.unique(eid))),
        "r2_legal_only": round(_r2(p_l), 4),
        "r2_with_extra": round(_r2(p_b), 4),
        "delta_r2": round(_r2(p_b) - _r2(p_l), 4),
        "se_legal_only": (y - p_l) ** 2,
        "se_with_extra": (y - p_b) ** 2,
        "_reads": ("delta_r2 is the information the candidate adds ABOVE what "
                   "the model already legally holds. On a SUPPLIED privileged "
                   "signal that is the leak magnitude; on a PREDICTED signal it "
                   "is the lever. The caller must say which."),
    }


def situation_disjoint(goal_inputs, situation_outputs) -> dict:
    """⛔ The PI's second clause: the goal path must not carry the situation
    classifier's output in ANY form — posterior, argmax, embedding, or a feature
    derived from them.

    Both arguments are DECLARED provenance (iterables of symbol names), because
    this is a fact about the computation graph. A name-set intersection is a weak
    check on purpose: it is a *tripwire* that fails loud on the obvious violation
    and forces the author to write the provenance down at all. It cannot detect
    laundering, and it says so.
    """
    g, s = set(map(str, goal_inputs)), set(map(str, situation_outputs))
    shared = sorted(g & s)
    derived = sorted(a for a in g for b in s if a != b and b in a)
    return {
        "goal_inputs": sorted(g), "situation_outputs": sorted(s),
        "shared_symbols": shared, "name_derived_symbols": derived,
        "disjoint": not shared and not derived,
        "_limitation": ("DECLARED provenance only. A shared trunk is not a "
                        "shared signal and this check cannot tell them apart; "
                        "it fails loud on the obvious violation and forces the "
                        "provenance to be written down."),
    }


def audit_goal_signal(*, name: str, supplied_at_inference: bool,
                      echo: dict | None = None, horizon: dict | None = None,
                      increment: dict | None = None,
                      provenance: dict | None = None) -> dict:
    """Combine the checks into ONE verdict, with the reason attached.

    ``supplied_at_inference`` is required and has no default: whether a
    privileged derivation is a leak or a legitimate label depends on it, and a
    default would let the most important fact about a signal go unstated.
    """
    fails, warns = [], []
    if echo and echo.get("is_echo"):
        fails.append(f"ECHO: output recoverable from input at "
                     f"{echo['functional_agreement']}")
    if horizon and not horizon.get("disjoint"):
        (fails if supplied_at_inference else warns).append(
            f"HORIZON OVERLAP: {horizon['n_overlap_steps']} scored steps are "
            f"inside the derivation window")
    if increment and supplied_at_inference and increment.get("delta_r2", 0) > 0:
        fails.append(f"PRIVILEGED INCREMENT: a supplied signal adds "
                     f"delta_r2={increment['delta_r2']} over the legal inputs")
    if provenance and not provenance.get("disjoint"):
        fails.append("SITUATION-CLASSIFIER OUTPUT is inside the goal path")
    return {
        "signal": name,
        "supplied_at_inference": bool(supplied_at_inference),
        "verdict": "INADMISSIBLE" if fails else "ADMISSIBLE",
        "failures": fails, "warnings": warns,
        "_note": ("a signal that is INADMISSIBLE as an inference input may still "
                  "be a perfectly good LABEL — PI ruling 2026-08-03: labels may "
                  "use ego, other agents, maps and future poses. Re-run with "
                  "supplied_at_inference=False to score it as a label."),
        "checks": {"echo": echo, "horizon": horizon, "increment":
                   None if increment is None else
                   {k: v for k, v in increment.items()
                    if not k.startswith("se_")},
                   "provenance": provenance},
    }
