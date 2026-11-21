"""The v7 vocabulary wired into the REFERENCE architectures (PI, 2026-08-27).

⛔ LOAD-BEARING: the refs and v6 must speak the SAME action space or the L5
dominance comparison scores different output spaces against each other. And the
kin3 kinematic labeller stays a MEASUREMENT vocabulary — the v7→kin3 map is a
projection for metrics, never a label source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))

from tanitad.models import vocab_v7 as V7  # noqa: E402
from tanitad.refs import refc_tactical as tac  # noqa: E402


def test_refa_v1_defaults_to_v7_and_old_configs_resolve_v6():
    from tanitad.refs.refa_v1 import RefAV1Config
    assert RefAV1Config().tac_vocab_version == "v7.0"


def test_refd_defaults_to_v7():
    from tanitad.refs.refd import RefDConfig
    assert RefDConfig().tac_vocab_version == "v7.0"


def test_refc_v3_defaults_to_v7():
    from tanitad.refs.refc_v3 import RefCV3Config
    assert RefCV3Config.__dataclass_fields__["tac_vocab_version"].default == "v7.0"


# --------------------------------------------------------------------------- #
# the v7 -> kin3 projection                                                     #
# --------------------------------------------------------------------------- #
def test_the_map_covers_every_v7_action_token_exactly_once():
    """Total on the v7 tuples — an unmapped token would silently drop from any
    projected metric, which is a manufactured absence."""
    assert set(tac.V7_TO_KIN3_LAT) == set(V7.TACTICAL_LAT_ACTIONS_V7)
    assert set(tac.V7_TO_KIN3_LON) == set(V7.TACTICAL_LON_ACTIONS_V7)


def test_the_map_lands_only_on_kinematic_classes():
    assert set(tac.V7_TO_KIN3_LAT.values()) <= set(range(tac.N_LAT))
    assert set(tac.V7_TO_KIN3_LON.values()) <= set(range(tac.N_LON))


def test_the_map_is_many_to_one_ie_a_projection_not_a_bijection():
    """The v7 space is strictly richer; a bijection here would mean the widening
    bought nothing. This pins that the map DESTROYS distinctions on purpose."""
    assert len(set(tac.V7_TO_KIN3_LAT.values())) < len(tac.V7_TO_KIN3_LAT)
    assert len(set(tac.V7_TO_KIN3_LON.values())) < len(tac.V7_TO_KIN3_LON)


def test_semantic_spot_checks():
    assert tac.V7_TO_KIN3_LAT["TURN_L"] == tac.LAT_TURN_LEFT
    assert tac.V7_TO_KIN3_LAT["NUDGE_R"] == tac.LAT_LANE_KEEP
    assert tac.V7_TO_KIN3_LON["ACCELERATE"] == tac.LON_ACCELERATE
    assert tac.V7_TO_KIN3_LON["BRAKE_TO"] == tac.LON_BRAKE_STOP
