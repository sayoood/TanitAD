"""H2 L2 — THE LABEL. One definition, imported by both the DEV and the CONFIRM driver.

`PRE_REGISTRATION_L2.md` Sec 1. This module contains no analysis and no thresholds beyond the ones
passed in, so DEV and CONFIRM provably evaluate the same predicate.

    L2_trigger(X,t) = 1  iff  a_req_off(X,t) >= tau   AND   a_req_seen(t) < tau
    R2(t)           = 1  iff  the ego applies a >= |brake| m/s^2 deceleration in (t, t+4 s]
    L2_label(X,t)   = L2_trigger AND R2

`a_req` = the smallest constant deceleration the ego must apply, ON ITS OWN REALISED PATH, to keep
its real oriented footprint clear of an agent extrapolated at CONSTANT VELOCITY, over (0, 4 s].
Built by `l2_build.py`; this module only thresholds and aggregates it.
"""
import numpy as np

A_UNRESOLVABLE = 8.0          # sentinel: no deceleration in the grid clears the conflict
TAU_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
D_L1 = 3.0                    # the refuted L1_gate conflict radius, for the head-to-head

# --- the split, fixed in PRE_REGISTRATION_L2.md Sec 2, before any L2 number existed -------------
DEV_CHUNKS = ["0036", "0170", "0174", "0834", "0868", "0928", "1852", "1870", "2433", "2503"]
CONFIRM_CHUNKS = ["0181", "0617", "0840", "0852", "0906", "0919", "0931", "1573",
                  "1860", "1864", "1880", "1900", "2498", "2500", "2820", "2838"]
assert not (set(DEV_CHUNKS) & set(CONFIRM_CHUNKS))


# --------------------------------------------------------------------------- the RESPONSE
def response_r2(D, v_min=3.0, brake=-2.0, freeflow=None):
    """Behavioural response: a genuine brake application inside the 4 s horizon.

    `freeflow` is the PRE-REGISTERED `alon_pre >= -0.5` precondition. It is passed as `None`
    (disabled) by the amended definition — see `H2_LABEL_V2_RESULTS.md` amendment A1: it excluded
    ~48 % of trigger frames *because the ego was already responding*, which is an anti-correlation
    built into the definition, the same trap as scoring conflict on a realised trajectory.
    Pass `freeflow=-0.5` to reproduce the pre-registered form.
    """
    r = (D.ego_v.to_numpy() >= v_min) & (D.alon_fut_min.to_numpy() <= brake)
    if freeflow is not None:
        r = r & (D.alon_pre.to_numpy() >= freeflow)
    return r


def response_l1(D, dv=-1.0):
    """The REFUTED L1 response: v(t+4 s) - v(t) <= -1 m/s. Fires on ~23-25 % of frames."""
    return D.ego_dv4.to_numpy() <= dv


# --------------------------------------------------------------------------- the TRIGGER
def _off_seen(D, resolvable, scope, ego="ps", agent="cv"):
    if ego == "ps" and agent == "cv":                       # PRIMARY
        base_L, base_R, base_S = "areq_off_L", "areq_off_R", "areq_seen"
        if scope == "residual":
            base_L, base_R = "areq_off_Lr", "areq_off_Rr"
        if resolvable:
            base_L, base_R, base_S = base_L + "_res", base_R + "_res", base_S + "_res"
    elif ego == "cv":                                       # sensitivity S1 (isolates M2)
        base_L, base_R, base_S = "areq_cv_off_L", "areq_cv_off_R", "areq_cv_seen"
    elif agent == "real":                                   # sensitivity S2 (isolates M1)
        base_L, base_R, base_S = "areq_re_off_L", "areq_re_off_R", "areq_re_seen"
    else:
        raise ValueError((ego, agent))
    return (np.maximum(D[base_L].to_numpy(), D[base_R].to_numpy()), D[base_S].to_numpy(),
            D[base_L].to_numpy(), D[base_R].to_numpy())


def trigger_l2(D, tau, resolvable=True, scope="crop", ego="ps", agent="cv"):
    """`L2_trigger`: an off-front agent forces >= tau m/s^2 AND nothing the encoder sees does.

    The second clause IS the agent-removal counterfactual (HOIST, lifted from objects to sensors):
    delete every off-front agent and the ego's current behaviour becomes feasible again, so the
    off-front agent is the BINDING constraint and a second camera is the only remedy.

    `scope="crop"`     -> outside the 51.4 deg encoder crop  (comparable to `L1_gate`)
    `scope="residual"` -> outside the FULL 120.5 deg front field (E0's genuine off-front 36.4 %)
    """
    off, seen, _, _ = _off_seen(D, resolvable, scope, ego, agent)
    return (off >= tau) & (seen < tau)


def trigger_l2_percam(D, tau, resolvable=True, scope="crop"):
    """(left, right) trigger masks — the head is a PER-CAMERA independent Bernoulli, never a
    softmax over mixed axes (`H2_SUBSTRATE` C.1: the 5-way maneuver softmax defect)."""
    _, seen, oL, oR = _off_seen(D, resolvable, scope)
    return (oL >= tau) & (seen < tau), (oR >= tau) & (seen < tau)


def trigger_l1(D, d=D_L1):
    """The REFUTED `L1_gate`, replicated on this table for a same-sample head-to-head."""
    return (D.dmin_l1_off.to_numpy() <= d) & ~(D.dmin_l1_seen.to_numpy() <= d)


# --------------------------------------------------------------------------- the LABEL
def label_l2(D, tau, **kw):
    """`L2_label` — the BEHAVIOURAL slice: conflict trigger AND the ego actually braked."""
    return trigger_l2(D, tau, **kw) & response_r2(D)
