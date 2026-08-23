"""Contract tests for the pool column-semantics registry.

The registry exists because a column was misread (`lk` as a rate) and produced
`needed_in_val = 48330` for a 600-clip split. These tests pin the semantics that make that
misread impossible: COUNT columns are declared non-stratifiable, the identities that PROVE
they are counts are checked, and a deliberately corrupted frame must fail validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pool_columns import (  # noqa: E402
    COLUMNS, LABEL_HORIZON, MANEUVER_CLASSES, MANEUVER_CLASSES_V2,
    rate_columns, to_rates, validate_pool,
)

NLAB = 179  # T=199 - LABEL_HORIZON, the value every real clip carries


def synth(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A synthetic pool frame that satisfies every declared identity."""
    rng = np.random.default_rng(seed)
    T = np.full(n, NLAB + LABEL_HORIZON)
    # maneuver counts: a random composition summing to nlab
    parts = rng.multinomial(NLAB, [0.45, 0.14, 0.14, 0.13, 0.14], size=n)
    parts2 = rng.multinomial(NLAB, [0.5, 0.12, 0.12, 0.13, 0.13], size=n)
    stop_frac = rng.uniform(0, 0.3, n)
    stopped = stop_frac + rng.uniform(0, 0.1, n)              # loose stop ⊇ strict stop
    city = rng.uniform(0, 1 - stopped, n)
    hw = 1.0 - stopped - city                                  # bands partition the window
    net = rng.uniform(0, 90, n)
    df = pd.DataFrame({
        "clip_id": [f"c{i:04d}" for i in range(n)], "chunk": rng.integers(0, 200, n),
        "T": T, "nlab": np.full(n, NLAB), "win_s": np.full(n, 20.1),
        "mean_v": rng.uniform(0.5, 25, n), "stop_frac": stop_frac, "stopped": stopped,
        "city": city, "hw": hw, "dist_m": rng.uniform(10, 500, n),
        "net_head": net, "cum_head": net + rng.uniform(0, 60, n),
        "junction": rng.integers(0, 2, n),
    })
    for i, c in enumerate(MANEUVER_CLASSES):
        df[c] = parts[:, i]
    for i, c in enumerate(MANEUVER_CLASSES_V2):
        df[c] = parts2[:, i]
    df["has_turn"] = ((df.tl + df.tr) > 0).astype(int)
    df["has_brake"] = (df.bs > 0).astype(int)
    df["has_stop"] = (df.stop_frac > 0).astype(int)
    return df


def test_synthetic_frame_validates():
    rep = validate_pool(synth())
    assert rep.ok, [c for c in rep.checks if not c.ok]
    assert rep.n_rows == 40


def test_count_columns_are_not_directly_stratifiable():
    """The A3 defect, encoded: a COUNT may not be used as a stratum axis as it stands."""
    for c in MANEUVER_CLASSES + MANEUVER_CLASSES_V2:
        assert COLUMNS[c].kind == "count"
        assert COLUMNS[c].stratifiable_directly is False
        assert COLUMNS[c].denom == "nlab"
    for c in ("stopped", "city", "hw", "stop_frac"):
        assert COLUMNS[c].kind == "fraction"          # NOT binary flags
        assert COLUMNS[c].stratifiable_directly is True


def test_to_rates_divides_by_nlab():
    df = synth()
    r = to_rates(df)
    for c, rate in rate_columns().items():
        assert rate in r.columns
        np.testing.assert_allclose(r[rate], df[c] / df["nlab"])
    # the five v1 rates form a distribution
    np.testing.assert_allclose(sum(r[f"{c}_rate"] for c in MANEUVER_CLASSES), 1.0, atol=1e-12)


def test_to_rates_handles_zero_denominator():
    df = synth(4)
    df.loc[0, "nlab"] = 0
    r = to_rates(df)
    assert np.isnan(r.loc[0, "lk_rate"])              # NaN, never inf
    assert np.isfinite(r.loc[1, "lk_rate"])


@pytest.mark.parametrize("mutate,check", [
    (lambda d: d.assign(lk=d.lk + 1), "identity:sum(v1)==nlab"),
    (lambda d: d.assign(nlab=d.nlab + 1), "identity:nlab==T-20"),
    (lambda d: d.assign(hw=d.hw + 0.1), "identity:stopped+city+hw==1"),
    (lambda d: d.assign(stop_frac=d.stopped + 0.05), "identity:stop_frac<=stopped"),
    (lambda d: d.assign(cum_head=d.net_head - 1.0), "identity:cum_head>=net_head"),
    (lambda d: d.assign(has_turn=1 - d.has_turn), "identity:has_turn==(tl+tr>0)"),
    (lambda d: d.assign(junction=2), "domain:junction"),
    (lambda d: d.assign(city=d.city + 5.0), "domain:city"),
])
def test_corruptions_are_caught(mutate, check):
    """Each identity must actually fire — a check that never fails is not a check."""
    rep = validate_pool(mutate(synth()))
    assert not rep.ok
    failed = {c.name for c in rep.checks if not c.ok}
    assert check in failed, f"expected {check} to fail, got {failed}"


def test_raise_if_bad_raises_and_reports():
    with pytest.raises(ValueError, match="pool column semantics violated"):
        validate_pool(synth().assign(lk=lambda d: d.lk + 1)).raise_if_bad()
    assert validate_pool(synth()).raise_if_bad().ok


def test_every_column_declares_a_family_or_is_bookkeeping():
    bookkeeping = {"T", "nlab", "win_s"}
    for name, spec in COLUMNS.items():
        if name in bookkeeping:
            continue
        assert spec.family in {"longitudinal", "lateral", "tactical", "strategic"}, name


def test_label_horizon_matches_refb_labels():
    """The constant is duplicated for standalone-runnability; it must not drift."""
    root = Path(__file__).resolve().parents[6] / "stack" / "scripts" / "refb_labels.py"
    if not root.exists():
        pytest.skip(f"refb_labels.py not reachable at {root}")
    text = root.read_text(encoding="utf-8", errors="replace")
    line = next(ln for ln in text.splitlines() if ln.startswith("LABEL_HORIZON"))
    assert int(line.split("=")[1].split("#")[0].strip()) == LABEL_HORIZON
