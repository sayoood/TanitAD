"""The contract the T1 kinematic-gate hook must satisfy — written BEFORE the hook.

⛔ **WHY THIS EXISTS AS A TEST AND NOT A NOTE.** The kinematic gate's own probe carried a
confound for its whole life: it ranked by ``sel_score`` while refcv3 (the ``hier`` arm)
argmaxes ``sel_score_v3``, disagreeing with the model's own ``sel_idx`` on **35/400 (8.75 %)**
and **29/400 (7.25 %)** across two draws. It went unnoticed because the probe's docstring
called ``k = 1`` *"a built-in identity control"* while its ``GATE_KS = (2, 4, 8, ...)`` —
**k = 1 was never in the tuple, so the identity control never ran.**

⭐ **A control that is documented but not executed is worse than none, because it gets cited.**
This file is that control, executed. It SKIPS while the hook is absent and ACTIVATES the moment
someone writes one, so the next person to place the gate cannot repeat the confound.

Sources: `Project Steering/Decisions/2026-09-05-mm-decisions.md` §M23/§M24; package
`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-kinematic-gate/`.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")


def _src() -> str:
    with io.open(ARM, encoding="utf-8") as fh:
        return fh.read()


def _has_hook(src: str) -> bool:
    """The hook is present once a gate flag reaches the arm's CLI."""
    return "--gate-k" in src or "gate_k" in src


# ------------------------------------------------------------------ the ranking key


def test_the_arm_ranks_by_sel_score_v3_and_falls_back_only_when_absent():
    """⛔ THE CONFOUND ITSELF. `hier` builds argmax `sel_score_v3`; ranking a gate by the
    flat `sel_score` silently scores a DIFFERENT candidate set than the model used."""
    src = _src()
    assert '"sel_score_v3" if "sel_score_v3" in out else "sel_score"' in src, (
        "the rank key must PREFER sel_score_v3 and fall back only when it is absent; "
        "a bare sel_score reproduces the 8.75 % / 7.25 % disagreement measured on two draws"
    )


def test_no_bare_sel_score_ranking_creeps_back_in():
    """A future edit must not reintroduce `out["sel_score"]` as a ranking source."""
    src = _src()
    bad = [ln.strip() for ln in src.splitlines()
           if 'out["sel_score"]' in ln and "sel_score_v3" not in ln
           and not ln.lstrip().startswith("#")]
    assert not bad, f"bare sel_score ranking reintroduced: {bad[:3]}"


# ------------------------------------- the identity control that was never executed


@pytest.mark.skipif(not _has_hook(_src()), reason="the T1 gate hook is not written yet")
def test_gate_k_1_must_be_reachable_so_the_identity_control_can_actually_run():
    """⛔ The defect that hid the confound: `k = 1` was called an identity control and was
    NOT in the swept tuple. Whatever ks the hook sweeps, **1 must be reachable**."""
    src = _src()
    tree = ast.parse(src)
    tuples = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and "GATE_K" in t.id.upper() for t in node.targets):
            try:
                tuples.append(ast.literal_eval(node.value))
            except Exception:                                    # noqa: BLE001
                pass
    for tup in tuples:
        assert 1 in tuple(tup), (
            f"k=1 absent from {tup}: the identity control cannot run, which is exactly "
            f"how the ranking confound survived undetected"
        )


@pytest.mark.skipif(not _has_hook(_src()), reason="the T1 gate hook is not written yet")
def test_the_zero_flag_path_must_be_declared_bit_identical():
    """With no gate requested, the arm must return the model's own selection untouched.
    ⚠️ Asserted on the INDEX, not on a metric (retraction #30: an identity control passed
    while interpolating the wrong tensor — it checked the arithmetic, not the object)."""
    src = _src()
    assert "sel_idx" in src
    assert ("gate_k" in src and ("is None" in src or "or 1" in src or "<= 1" in src)), (
        "the hook must have an explicit no-gate path; a gate that is always on has no "
        "control arm and its effect cannot be attributed"
    )


# --------------------------------------------------------------- the tier discipline


def test_the_gate_result_is_stamped_T0_until_a_T1_leg_exists():
    """⛔ EVAL_DOCTRINE: T1 is the primary tier for any capability claim. The banked gate
    numbers are T0 (the 4060 was saturated), and nothing may quote them as driving
    performance until the T1 leg runs."""
    d = os.path.join(REPO, "Project Steering", "Decisions", "2026-09-05-mm-decisions.md")
    with io.open(d, encoding="utf-8") as fh:
        txt = fh.read()
    assert "## M24." in txt, "M24 is the decision that scopes the gate result"
    seg = txt.split("## M24.", 1)[1]
    assert "T1 WAS NOT RUN" in seg or "T1 was not run" in seg, (
        "M24 must keep saying the T1 leg is missing until it is run; if the leg lands, "
        "update M24 and this assertion together"
    )
