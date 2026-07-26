"""Option B (PI-approved 2026-07-26): per-clip wheelbase, fix-FORWARD only.

These tests exist to hold two things that a comment cannot:

  1. PARITY IS PRESERVED BY CONSTRUCTION. The legacy regime must contribute
     NOTHING to the build params, so `physicalai-train-e438721ae894` keeps its
     exact current meaning. A test that only checked "the new mode works" would
     have passed while silently re-keying every cache in the program.
  2. THE REGIMES CANNOT COLLIDE. A corrected cache must be unable to share a key
     with a legacy one, whatever the clip list.

Both directions are asserted: a FIDELITY check (legacy == today, corrected uses
the real value) and a DELIBERATELY-FAILING check (the guard raises on an
unresolvable clip, and the two regimes' keys differ). A guard that cannot be
shown to fail is not a guard.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tanitad.data.epcache import cache_key
from tanitad.data.physicalai import (DEFAULT_WHEELBASE_MODE, WHEELBASE,
                                     WHEELBASE_MODE_LEGACY,
                                     WHEELBASE_MODE_PER_CLIP, label_params,
                                     signals_at, wheelbase_for_clip)

# The exact params dict that minted the parity key, copied from
# scripts/build_pai_cache.py / rebuild_pai_rolling.py as of 2026-07-26.
CANONICAL_PARAMS = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2"}


def _ego(curv: float, n: int = 40) -> pd.DataFrame:
    t = np.arange(n, dtype=np.float64) * 0.1
    return pd.DataFrame({
        "timestamp": t, "x": t, "y": np.zeros(n), "vx": np.full(n, 10.0),
        "vy": np.zeros(n), "ax": np.zeros(n), "curvature": np.full(n, curv),
        "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n),
        "qw": np.ones(n),
    })


# --------------------------------------------------------------------------- #
# 1. PARITY — the load-bearing direction                                        #
# --------------------------------------------------------------------------- #

def test_legacy_regime_contributes_no_build_param():
    """The whole option-B parity guarantee reduces to this one assertion."""
    assert label_params(WHEELBASE_MODE_LEGACY) == {}
    assert label_params() == {}, "the DEFAULT must be the parity-preserving regime"
    assert DEFAULT_WHEELBASE_MODE == WHEELBASE_MODE_LEGACY


def test_legacy_params_dict_is_bit_identical():
    """Splatting label_params into the canonical dict must not perturb it."""
    merged = {**CANONICAL_PARAMS, **label_params(WHEELBASE_MODE_LEGACY)}
    assert merged == CANONICAL_PARAMS
    srcs = [{"clip_id": f"clip{i:04d}"} for i in range(50)]
    assert cache_key(srcs, merged) == cache_key(srcs, CANONICAL_PARAMS)


def test_signals_at_default_is_the_legacy_constant():
    """signals_at's new parameter must default to byte-identical behaviour."""
    ego = _ego(curv=0.05)
    t = ego["timestamp"].to_numpy(np.float64)
    a_default, _ = signals_at(ego, t)
    a_explicit, _ = signals_at(ego, t, wheelbase=WHEELBASE)
    assert np.array_equal(a_default, a_explicit)
    assert np.allclose(a_default[:, 0], math.atan(WHEELBASE * 0.05), atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. SEPARATION — the corrected regime cannot collide with a legacy cache       #
# --------------------------------------------------------------------------- #

def test_per_clip_regime_changes_the_key():
    srcs = [{"clip_id": f"clip{i:04d}"} for i in range(50)]
    legacy = {**CANONICAL_PARAMS, **label_params(WHEELBASE_MODE_LEGACY)}
    corrected = {**CANONICAL_PARAMS, **label_params(WHEELBASE_MODE_PER_CLIP)}
    assert corrected != legacy
    assert cache_key(srcs, corrected) != cache_key(srcs, legacy)


def test_unknown_regime_is_refused():
    with pytest.raises(ValueError):
        label_params("const2p7")


# --------------------------------------------------------------------------- #
# 3. THE LABEL ITSELF                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("wb", [2.730, 2.850, 3.135, 3.165, 3.216])
def test_steer_uses_the_supplied_wheelbase(wb):
    """steer = atan(L * curvature) for the corpus's five real wheelbases."""
    ego = _ego(curv=0.05)
    t = ego["timestamp"].to_numpy(np.float64)
    actions, _ = signals_at(ego, t, wheelbase=wb)
    assert np.allclose(actions[:, 0], math.atan(wb * 0.05), atol=1e-5)


def test_gain_matches_the_measured_range():
    """The per-clip gain vs the 2.9 constant must reproduce the measured
    0.941-1.109 envelope (WHEELBASE_IMPACT.md 2.1) -- a fidelity check on the
    direction of the correction, not just that it is non-zero."""
    ego = _ego(curv=0.01)
    t = ego["timestamp"].to_numpy(np.float64)
    base = signals_at(ego, t, wheelbase=WHEELBASE)[0][0, 0]
    gains = []
    for wb in (2.730, 2.850, 3.135, 3.165, 3.216):
        s = signals_at(ego, t, wheelbase=wb)[0][0, 0]
        gains.append(math.tan(s) / math.tan(base))
    assert min(gains) == pytest.approx(2.730 / 2.9, rel=1e-6)
    assert max(gains) == pytest.approx(3.216 / 2.9, rel=1e-6)
    assert 0.94 < min(gains) < 0.95 and 1.10 < max(gains) < 1.11


# --------------------------------------------------------------------------- #
# 4. THE GUARD CAN FAIL — the direction most easily left untested               #
# --------------------------------------------------------------------------- #

def test_strict_resolution_raises_when_unresolvable(tmp_path):
    """A corrected BUILD must refuse to silently fall back to 2.9."""
    with pytest.raises(RuntimeError, match="refusing to mint"):
        wheelbase_for_clip("no-such-clip", tmp_path, strict=True)


def test_nonstrict_resolution_falls_back_loudly(tmp_path, capsys):
    wb = wheelbase_for_clip("another-missing-clip", tmp_path, strict=False)
    assert wb == pytest.approx(WHEELBASE)
    assert "APPROXIMATION" in capsys.readouterr().out


def test_local_csv_table_is_preferred(tmp_path, monkeypatch):
    """Source (1) of the documented resolution order, and proof the join keys on
    clip_id rather than on anything positional."""
    csv = tmp_path / "wb.csv"
    pd.DataFrame({"clip_id": ["aaa", "bbb"],
                  "wheelbase": [3.165, 2.730]}).to_csv(csv, index=False)
    monkeypatch.setenv("TANITAD_PAI_WHEELBASE", str(csv))
    assert wheelbase_for_clip("aaa") == pytest.approx(3.165)
    assert wheelbase_for_clip("bbb") == pytest.approx(2.730)
