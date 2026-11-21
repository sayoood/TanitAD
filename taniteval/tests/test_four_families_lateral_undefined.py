"""An UNDEFINED lateral term is REFUSED, never null — and `cross_mae_m` says what it is.

⛔ THE TWO FAILURES THIS FILE PINS, both MEASURED 2026-09-19 by W1 while scoring the NavSim
STOP floor (an all-zero plan):

1. `four_families.lateral` returned **`None`** for heading / yaw-rate / curvature whenever every
   step of the plan was below `min_ds_m` — a stationary path has no tangent, so the terms are
   genuinely undefined — and `tools/criteria_check.py` reads a null exactly as it reads a MISSING
   key. Three BINDING lateral criteria were therefore reported as SILENT OMISSIONS of an
   instrument that had honestly declined. A null is the one shape that cannot carry its reason;
   a 0.0 would be worse (perfect lateral agreement, from a car that never moved).
2. ⚠️ **`cross_mae_m` is a pure y-difference** (`_seq_geometry` returns `"cross": p[..., 1][:, 1:]`
   — the ego-frame y column), so it is a lateral OFFSET at matched time index, not a distance to
   the GT path. On W1's warmup run a constant-velocity arm and an all-zero STOP plan both read
   **1.0658 m**: a LATERAL row quoted alone cannot tell a moving arm from a parked one. That is
   not a NavSim artefact — the same reducer produces every LATERAL row in the programme.

The mechanism of (2) is REPRODUCED here from scratch rather than inherited: two arms with y == 0
score identically however differently they drive.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE)
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import four_families as ff                 # noqa: E402

DT = 0.5
K = 8


def _straight(n=6, v=8.0, k=K, dt=DT):
    """A constant-velocity plan: x advances, y is EXACTLY 0 — like the devkit's CV agent."""
    t = torch.arange(1, k + 1, dtype=torch.float64) * dt
    p = torch.zeros(n, k, 2, dtype=torch.float64)
    p[..., 0] = v * t
    return p


def _stopped(n=6, k=K):
    """The STOP floor: an all-zero plan. y is also EXACTLY 0."""
    return torch.zeros(n, k, 2, dtype=torch.float64)


def _curving(n=6, v=8.0, k=K, dt=DT, omega=0.12):
    """A human-like arc: it BOTH advances and moves laterally."""
    out = torch.zeros(n, k, 2, dtype=torch.float64)
    for i in range(n):
        x = y = h = 0.0
        for j in range(k):
            x += v * dt * math.cos(h)
            y += v * dt * math.sin(h)
            h += omega * dt * (1 if i % 2 == 0 else -1)
            out[i, j, 0], out[i, j, 1] = x, y
    return out


# ========================================================================== #
# 1. An undefined term REFUSES with a reason and an n                        #
# ========================================================================== #
def test_a_stationary_plan_REFUSES_heading_curvature_and_yaw_rate():
    lat = ff.lateral(_stopped(), _curving(), dt=DT)
    for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "curvature_bias_1pm"):
        v = lat[k]
        assert isinstance(v, dict), f"{k} is {v!r} — a null cannot carry its reason"
        assert v["status"] == "UNAVAILABLE"
        assert v["n"] == 0 and v["n_steps_total"] > 0
        assert "NO TANGENT" in v["reason"] and "min_ds_m" in v["reason"]
        assert v["min_ds_m"] == pytest.approx(ff.MIN_DS_MPS * DT)


def test_DELIBERATE_REGRESSION_the_refusal_is_never_a_null_and_never_a_zero():
    """The two shapes this fix exists to prevent, named: a null (silently absent) and a
    0.0 (perfect lateral agreement from an arm that never moved)."""
    lat = ff.lateral(_stopped(), _curving(), dt=DT)
    for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm"):
        assert lat[k] is not None
        assert lat[k] != 0.0 and not isinstance(lat[k], float)


def test_a_MOVING_plan_still_returns_the_plain_numbers_it_always_did():
    """The control: the defined branch is untouched — no banked number moves. A plan
    identical to the GT has exactly zero heading, curvature and yaw-rate error."""
    gt = _curving()
    lat = ff.lateral(gt.clone(), gt, dt=DT)
    for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm"):
        assert isinstance(lat[k], float), (k, lat[k])
        assert lat[k] == 0.0
    assert lat["n_steps_heading"] == 6 * K
    assert lat["cross_mae_m"] == 0.0


def test_one_moving_step_is_enough_to_define_heading():
    """The boundary, from the other side: the refusal fires on the DATA, not on the arm's
    name. A plan with a single step above the gate is defined again."""
    p = _stopped()
    p[:, -1, 0] = 5.0                                  # one long step at the end
    lat = ff.lateral(p, _curving(), dt=DT)
    assert isinstance(lat["heading_mae_deg"], float)
    assert lat["n_steps_heading"] == 6
    # ...while curvature still needs a PAIR of valid steps and stays refused
    assert isinstance(lat["curvature_mae_1pm"], dict)


def _alternating(n=6, k=K, step=5.0):
    """Advances on ALTERNATE steps only: single steps clear the gate, no consecutive PAIR does.
    This is the construction that reaches `n_head > 0 and n_curv == 0`."""
    p = torch.zeros(n, k, 2, dtype=torch.float64)
    x = 0.0
    for j in range(k):
        if j % 2 == 0:
            x += step
        p[:, j, 0] = x
    return p


def test_the_yaw_rate_NaN_BRANCH_is_reachable_and_now_REFUSES():
    """⛔ THE WORSE HALF OF THE SAME DEFECT, MEASURED 2026-09-20 (W2) — and it is WORSE than
    the null: `yaw_rate_mae_degps` is masked by `both_pair` but was emitted under the HEADING
    count, so an arm with valid single steps and NO valid pair emitted **NaN** — a `float`,
    which passes every `isinstance(v, (int, float))` type guard downstream (summarize.py writes
    it straight into `metrics`), while strict JSON refuses to serialise it.
    ⭐ The construction below PROVES the branch is reachable rather than arguing it."""
    lat = ff.lateral(_alternating(), _curving(), dt=DT)
    assert lat["n_steps_heading"] > 0 and lat["n_steps_curvature"] == 0, \
        "the arm must reach the mixed state, or this test proves nothing"
    assert isinstance(lat["heading_mae_deg"], float), "heading IS defined here — the control"

    v = lat["yaw_rate_mae_degps"]
    assert isinstance(v, dict) and v["status"] == "UNAVAILABLE", f"NaN regression: {v!r}"
    assert v["n"] == 0 and "step PAIRS" in v["reason"]
    # the shape the old code produced, named so a regression is unmistakable
    assert not (isinstance(v, float) and math.isnan(v))
    import json
    json.dumps(v, allow_nan=False)                 # the old NaN made this raise


def test_the_yaw_rate_DEFINED_branch_is_bit_for_bit_what_it_always_was():
    """The guard moved from `n_head` to `n_curv`; n_curv > 0 implies n_head > 0, so every case
    that produced a real number still produces the IDENTICAL number. Recomputed here from the
    raw geometry rather than trusted."""
    gt, pred = _curving(), _curving(omega=0.05)
    lat = ff.lateral(pred, gt, dt=DT)
    P, G = ff._seq_geometry(pred, DT), ff._seq_geometry(gt, DT)
    both_pair = P["pair_valid"] & G["pair_valid"]
    expect, n = ff._masked((P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair)
    assert n > 0 and lat["n_steps_yaw_rate"] == n
    assert lat["yaw_rate_mae_degps"] == round(math.degrees(expect), 4)
    assert ff.lateral(gt.clone(), gt, dt=DT)["yaw_rate_mae_degps"] == 0.0


# ========================================================================== #
# 2. What `cross_mae_m` is — and when it cannot rule                          #
# ========================================================================== #
def test_cross_mae_m_CANNOT_distinguish_a_moving_arm_from_a_parked_one():
    """⭐ THE FINDING, reproduced from scratch: `cross` is the ego-frame y column, so two
    arms with y == 0 — a constant-velocity straight line and an all-zero STOP plan — produce
    the IDENTICAL lateral row against a curving human, while their along-track errors differ
    by an order of magnitude. A LATERAL number quoted alone is blind to that."""
    gt = _curving()
    cv, stop = ff.lateral(_straight(), gt, dt=DT), ff.lateral(_stopped(), gt, dt=DT)
    assert cv["cross_mae_m"] == stop["cross_mae_m"]
    assert cv["cross_bias_m"] == stop["cross_bias_m"]
    assert cv["cross_final_mae_m"] == stop["cross_final_mae_m"]
    # and the along-track context that MAKES the difference visible is emitted beside it
    assert cv["_along_mae_m_for_context"] > 5 * stop["_along_mae_m_for_context"] or \
        stop["_along_mae_m_for_context"] > 5 * cv["_along_mae_m_for_context"]
    assert "lateral OFFSET at matched time index" in cv["_cross_is"]
    assert "frenet_dense" in cv["_projection_based_alternative"]


def test_cross_mae_m_equals_the_GTs_own_lateral_excursion_when_the_arm_has_no_y():
    """The mechanism in one line: with y_pred == 0 the metric is mean |y_gt| — a property of
    the HUMAN's path, not of the arm."""
    gt = _curving()
    lat = ff.lateral(_stopped(), gt, dt=DT)
    assert lat["cross_mae_m"] == pytest.approx(float(gt[..., 1].abs().mean()), abs=5e-5)


def test_the_qualifier_travels_with_every_lateral_block():
    """It must be emitted for a MOVING arm too — the defect is quoting the row alone, and a
    qualifier that only appears in the degenerate case would never be read."""
    lat = ff.lateral(_curving(), _curving(omega=0.05), dt=DT)
    for k in ("_cross_is", "_along_mae_m_for_context", "_projection_based_alternative"):
        assert k in lat
    assert "NOT a distance to the GT path" in lat["_cross_is"]


# ========================================================================== #
# 3. The checker half: a refusal is a WORK ITEM, a null is a DIAGNOSED gap    #
# ========================================================================== #
def _cc():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "criteria_check_uut", os.path.join(_REPO, "tools", "criteria_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _registry():
    import json
    with open(os.path.join(_REPO, "products", "P7-TanitEval", "CRITERIA_REGISTRY.json"),
              encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("cid,key", [("lat.heading", "heading_mae_deg"),
                                     ("lat.curvature", "curvature_mae_1pm"),
                                     ("lat.yaw_rate", "yaw_rate_mae_degps")])
def test_the_checker_reads_the_refusal_as_a_WORK_ITEM_and_a_null_as_a_DIAGNOSED_gap(cid, key):
    cc, reg = _cc(), _registry()
    crit = next(c for c in reg["families"]["LATERAL"]["criteria"] if c["id"] == cid)
    block = ff.lateral(_stopped(), _curving(), dt=DT)
    art = {"block": "taniteval.driving/tier1", "four_families": {"lateral": block}}
    state, detail = cc.classify(art, crit)
    assert state == cc.REFUSED, (cid, state, detail)
    assert "UNDEFINED" in detail and "n=0" in detail

    # ⛔ THE MUTATION: put the OLD null back. It must be ABSENT — and the detail must say
    # WHICH gap it is, because a null needs a refusal while a missing key needs an emitter.
    art["four_families"]["lateral"][key] = None
    state, detail = cc.classify(art, crit)
    assert state == cc.ABSENT
    assert "present but NULL" in detail and "not a refusal" in detail

    # ...and a genuinely missing key still reads as the other gap
    del art["four_families"]["lateral"][key]
    state, detail = cc.classify(art, crit)
    assert state == cc.ABSENT and detail == "no key present and not refused"


# ========================================================================== #
# 4. The downstream renderer that indexes the block DIRECTLY                  #
# ========================================================================== #
def _report_mod():
    """`taniteval/tools/refav1_openloop_report.py` reads
    `A[arm]["four_families"]["lateral"].get(k)` and formats it with `_f`. It is the ONE live
    consumer that neither type-guards nor bootstraps — everything else either requires
    `isinstance(v, (int, float))` (summarize.py, benchreport/render.py, release_gate._num)
    or already crashed on the old `None` (lan_probe.py), so the refusal changes nothing there."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "refav1_openloop_report_uut",
        os.path.join(_REPO, "taniteval", "tools", "refav1_openloop_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_openloop_report_renders_a_REFUSAL_as_a_short_token_not_a_600_char_dict():
    m = _report_mod()
    block = ff.lateral(_stopped(), _curving(), dt=DT)["heading_mae_deg"]
    assert len(str(block)) > 400, "the guard below is only meaningful for a LONG refusal"
    cell = m._f(block)
    assert cell.startswith("REFUSED (n=0)") and len(cell) < 120, cell
    assert "|" not in cell, "a raw pipe would break the markdown table it is written into"
    # the control: the shapes it already handled must be untouched
    assert m._f(None) == "n/a" and m._f(1.5) == "1.5000" and m._f(True) == "yes"
    # ...and a dict that is NOT a refusal still falls through to the old behaviour
    assert m._f({"mean": 2.0}) == str({"mean": 2.0})
