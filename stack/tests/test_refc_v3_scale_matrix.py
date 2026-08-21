"""The REF-C v3 SCALE x HIERARCHY matrix is MEASURED, and stays measured.

⛔ WHY THESE TESTS EXIST. The A-vs-B package decision was carried as hand
arithmetic — a core count from one build plus a rung count from another. The
arithmetic was WRONG BY 540,672 parameters, because it used
`ALIGNED_TACTICAL.d_in`'s default (2048) instead of the encoder's ACTUAL
`feat_dim` at XL (992). Small in absolute terms; the point is that nobody could
have seen it without building, and a D-008 decision is not a place for a number
nobody built.
"""
from __future__ import annotations

import pytest

from tanitad.models.hierarchy import ALIGNED_TACTICAL
from tanitad.refs import refc_v3 as R
from scripts.refc_v3_scale_matrix import (D008_MIN_PARAMS,
                                          REFA_V1_TACTICAL_BLOCKS, TIERS,
                                          build_rungs, cell)


@pytest.fixture(scope="module")
def matrix() -> dict[tuple[str, str], dict]:
    return {(s, t): cell(s, t) for s in R.V3_SIZES for t in TIERS}


def test_exactly_one_cell_clears_D008(matrix):
    """⭐ THE FINDING. Scale and hierarchy are separable in COST and NOT in the
    CONSTRAINT: only `xl` + `aligned` reaches D-008's 250 M."""
    ok = {k for k, v in matrix.items() if v["d008_ok"]}
    assert ok == {("xl", "aligned")}, (
        f"D-008 (>= {D008_MIN_PARAMS:,}) is cleared by {sorted(ok)}. If this "
        f"changed, a size or a rung geometry moved — update "
        f"`Project Steering/MODEL_REGISTRY.md` and the REF-C v3 prereg in the "
        f"same commit, because the package decision reads these numbers.")


def test_aligned_body_REPRODUCES_refa_v1_measured_blocks(matrix):
    """The tier claims REF-A v1's tactical capacity. Prove it, per cell."""
    for (size, tier), v in matrix.items():
        if tier != "aligned":
            continue
        assert v["body"] == REFA_V1_TACTICAL_BLOCKS, (
            f"{size}/aligned body is {v['body']:,}, REF-A v1's MEASURED "
            f"tactical blocks are {REFA_V1_TACTICAL_BLOCKS:,} — the tier is "
            f"misnamed, not merely mis-sized")


def test_rung_d_in_FOLLOWS_the_built_encoder_not_the_config_default():
    """⛔ THE BUG THIS FILE WAS WRITTEN FOR.

    `ALIGNED_TACTICAL.d_in` defaults to 2048 (v6's operative width). REF-C v3's
    encoder is NARROWER at every size. Taking the default is how the hand
    arithmetic drifted, and it is silent: both numbers look plausible."""
    assert ALIGNED_TACTICAL.d_in == 2048, "v6's operative width moved"
    for size in R.V3_SIZES:
        c = cell(size, "aligned")
        assert c["d_in"] < ALIGNED_TACTICAL.d_in
        tac, _ = build_rungs("aligned", c["d_in"])
        assert tac.adapter[0].in_features == c["d_in"], (
            f"{size}: the rung was built at d_in={tac.adapter[0].in_features} "
            f"but the encoder emits {c['d_in']}")


def test_hierarchy_cost_is_near_constant_across_the_ladder(matrix):
    """The incumbent docstring's independence claim, checked rather than
    repeated. It holds for COST — the adapter's first layer is the only part
    that tracks the encoder — and the spread must stay small enough that the
    claim is not quietly false."""
    for tier in ("thin", "v6", "aligned"):
        vals = [matrix[(s, tier)]["hierarchy"] for s in R.V3_SIZES]
        spread = (max(vals) - min(vals)) / min(vals)
        assert spread < 0.10, (
            f"tier {tier!r} hierarchy cost spans {min(vals):,}..{max(vals):,} "
            f"({spread:.1%}) across the size ladder — it is no longer 'near "
            f"constant' and the independence claim needs restating")


def test_aligned_REPLACES_the_thin_cascade_it_does_not_stack(matrix):
    """A cell's total is core + ONE hierarchy. Stacking would double-count the
    tactical state and would inflate every package number."""
    for size in R.V3_SIZES:
        for tier in TIERS:
            v = matrix[(size, tier)]
            assert v["total"] == v["core"] + v["hierarchy"], (
                f"{size}/{tier}: {v['total']:,} != core {v['core']:,} + "
                f"hierarchy {v['hierarchy']:,}")


def test_matrix_carries_NO_training_claim(matrix):
    """⚠️ Parameter counts are not results. Nothing in a cell may be read as
    evidence about behaviour — the keys are checked so a future edit cannot
    smuggle a metric in beside the capacity ledger."""
    allowed = {"size", "tier", "d_in", "core", "hierarchy", "hierarchy_tac",
               "hierarchy_str", "body", "total", "d008_ok",
               "body_matches_refa_v1"}
    for k, v in matrix.items():
        assert set(v) <= allowed, f"{k} carries unexpected keys: {set(v) - allowed}"
