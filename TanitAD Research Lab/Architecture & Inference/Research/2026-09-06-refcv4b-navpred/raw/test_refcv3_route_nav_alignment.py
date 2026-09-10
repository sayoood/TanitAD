"""ROUTE_CLASSES and NAV_COMMANDS are DIFFERENT SPACES -- pinned, with the
deliberate regression the pre-2026-09-06 statistic could not catch.

Why this file exists (MEASURED 2026-09-06 on the refcv4b landing dump):
``taniteval/tools/refcv3_arm.py`` compared ``route_pred`` (a ``ROUTE_CLASSES``
index, 3-wide) DIRECTLY against a ``NAV_COMMANDS`` index (4-wide), on a comment
that asserted the two are "both 3-wide and index-aligned in refb". They are not.
It published ``nav_echo_index`` 0.1621 where the value is 0.6405, and
``route_follows_SHUFFLED_NAV`` 0.2264 where the value is 0.3370.
The sibling harness ``taniteval/tools/refav1_arm.py`` has always mapped before
comparing, so this was a REGRESSION against an existing correct implementation.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARM = os.path.join(_REPO, "taniteval", "tools", "refcv3_arm.py")
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "stack", "scripts"),
           os.path.join(_REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_arm():
    if not os.path.exists(_ARM):
        pytest.skip("refcv3_arm.py not present")
    spec = importlib.util.spec_from_file_location("_refcv3_arm_under_test", _ARM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_two_label_spaces_are_different_widths():
    """The fact the old comment denied. If this ever fails, refb changed."""
    from tanitad.refs.refb import NAV_COMMANDS, ROUTE_CLASSES
    assert len(NAV_COMMANDS) == 4, NAV_COMMANDS
    assert len(ROUTE_CLASSES) == 3, ROUTE_CLASSES
    assert NAV_COMMANDS == ("follow", "left", "right", "straight")
    assert ROUTE_CLASSES == ("route_left", "route_straight", "route_right")


def test_maps_are_imported_not_re_derived_and_are_not_the_identity():
    import refb_labels
    arm = _load_arm()
    assert arm._route_to_nav_map() == dict(refb_labels._ROUTE_TO_NAV)
    assert arm._nav_to_route_map() == dict(refb_labels._NAV_TO_ROUTE)
    # the declared module constant must agree with the label module
    assert dict(arm._ROUTE_TO_NAV) == dict(refb_labels._ROUTE_TO_NAV)
    r2n = arm._route_to_nav_map()
    assert r2n != {k: k for k in r2n}, "the map is NOT the identity"
    assert r2n == {0: 1, 1: 0, 2: 2}
    n2r = arm._nav_to_route_map()
    assert n2r == {0: 1, 1: 0, 2: 2}


# --------------------------------------------------------------------------- #
# THE DELIBERATE REGRESSION: a route head that is a PERFECT ECHO of its nav     #
# input. The corrected statistic must read its KNOWN value (exactly 1.0); the   #
# pre-correction one must MISS it -- that is the whole reason the fix exists.   #
# --------------------------------------------------------------------------- #
def _echo_indices(route_pred, nav_cmd, n2r):
    """(corrected, legacy-raw-index) -- the two forms the tool now emits."""
    implied = np.array([n2r.get(int(x), -1) for x in nav_cmd])
    return (float((route_pred == implied).mean()),
            float((route_pred == nav_cmd).mean()))


def test_deliberate_regression_a_perfect_nav_echo():
    arm = _load_arm()
    n2r = arm._nav_to_route_map()
    rng = np.random.default_rng(0)
    # a realistic refcv4b-shaped nav marginal: follow-dominated, no NAV_STRAIGHT
    nav = rng.choice([0, 1, 2], size=4000, p=[0.64, 0.09, 0.27])
    # THE REGRESSION: the head simply re-emits the route its nav token implies.
    echo_pred = np.array([n2r[int(x)] for x in nav])
    corrected, legacy = _echo_indices(echo_pred, nav, n2r)
    assert corrected == pytest.approx(1.0), (
        "a PERFECT echo must read exactly 1.0 -- this is the control's known value")
    assert legacy < 0.45, (
        "the pre-correction raw-index form MUST FAIL to see a perfect echo; if it "
        "did see it, this test would not prove the fix was load-bearing")
    assert corrected - legacy > 0.5


def test_a_nav_independent_head_reads_low_on_the_corrected_index():
    """The other side of the control: a head that ignores nav must NOT read 1.0."""
    arm = _load_arm()
    n2r = arm._nav_to_route_map()
    rng = np.random.default_rng(1)
    nav = rng.choice([0, 1, 2], size=4000, p=[0.64, 0.09, 0.27])
    const_pred = np.full(4000, 1)          # always `route_straight`
    corrected, _ = _echo_indices(const_pred, nav, n2r)
    # a constant head agrees with the implied route only at the `follow` rate
    assert corrected == pytest.approx(float((nav == 0).mean()), abs=1e-12)
    assert corrected < 0.75


def test_the_shipped_analyze_uses_the_mapped_operand():
    """A source guard BESIDE the arithmetic tests: the primary key must compare
    against the mapped token, and the raw-index form must survive only under the
    LEGACY name."""
    src = open(_ARM, encoding="utf-8").read()
    assert '"route_follows_SHUFFLED_NAV_under_shuffle":\n' in src
    assert "(ps_ == _imp_s).astype(float)[changed]" in src
    assert "route_follows_SHUFFLED_NAV_RAW_INDEX_LEGACY" in src
    assert "nav_echo_index_RAW_INDEX_LEGACY" in src
    # the refuted claim must not be shipped as fact any more
    assert "both 3-wide and index-aligned in refb" not in src
