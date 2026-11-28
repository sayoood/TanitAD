"""Contract tests for the clean v2-line validation selector.

What must hold for a split to be usable at all:
  * it is drawn ONLY from the remainder and is clip-disjoint from train (C64's rule),
  * the disjointness proof is emitted even when the intersection is empty,
  * it is reproducible from (seed, inputs) and hashed,
  * the balancer actually balances — measured against the quota and uniform controls,
  * feasibility is reported PER AXIS SET, since headroom falls as axes are added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_pool_columns import synth  # noqa: E402
from pool_columns import to_rates  # noqa: E402
from clean_val_select import (  # noqa: E402
    ALL_AXES, BALANCE_BAR, FAMILY_AXES, assert_disjoint, balance_summary, balanced_select,
    cell_labels, census_fraction, feasibility, manifest_sha256, build_manifest,
    speed_edges, standardised_diffs, stratified_select,
)


@pytest.fixture(scope="module")
def corpus():
    """A train corpus and a REMAINDER deliberately skewed away from it.

    The skew reproduces the real situation: the remainder is the residue of a
    manoeuvre-balanced selection, so it is faster, straighter and less junction-heavy.
    """
    train = to_rates(synth(600, seed=1))
    train["junction"] = (np.arange(600) % 10 < 6).astype(int)     # 60 % junction
    train["mean_v"] = np.linspace(2, 18, 600)
    rem = to_rates(synth(3000, seed=2))
    rem["clip_id"] = [f"r{i:04d}" for i in range(3000)]           # ids disjoint from train
    rem["junction"] = (np.arange(3000) % 10 < 2).astype(int)      # 20 % junction
    rem["mean_v"] = np.linspace(6, 26, 3000)
    return train, rem


def test_selection_is_disjoint_and_from_the_remainder(corpus):
    train, rem = corpus
    val = balanced_select(train, rem, 100, seed=0)
    assert len(val) == 100
    assert set(val.clip_id) <= set(rem.clip_id)
    proof = assert_disjoint(val.clip_id, train.clip_id)
    assert proof["disjoint"] and proof["n_intersection"] == 0
    assert proof["granularity"] == "clip_id"
    # the proof is emitted EVEN WHEN EMPTY, and it does not overclaim drive-level cleanliness
    assert proof["drive_level_provable"] is False and proof["drive_level_reason"]


def test_disjointness_proof_detects_a_real_overlap(corpus):
    train, _ = corpus
    proof = assert_disjoint(train.clip_id[:5], train.clip_id)
    assert not proof["disjoint"] and proof["n_intersection"] == 5


def test_selection_is_reproducible_and_hashed(corpus):
    train, rem = corpus
    a = balanced_select(train, rem, 80, seed=3)
    b = balanced_select(train, rem, 80, seed=3)
    assert list(a.clip_id) == list(b.clip_id)
    assert manifest_sha256(a.clip_id) == manifest_sha256(b.clip_id)
    assert manifest_sha256(a.clip_id) != manifest_sha256(rem.clip_id[:80])
    # order must not change the hash — it is a set identity
    assert manifest_sha256(a.clip_id) == manifest_sha256(list(a.clip_id)[::-1])


def test_balancer_beats_the_uniform_and_quota_controls(corpus):
    """The claim under test: balancing on the four-family axes actually balances."""
    train, rem = corpus
    rng = np.random.default_rng(0)
    uniform = rem.iloc[rng.choice(len(rem), 200, replace=False)]
    quota = stratified_select(train, rem, 200, seed=0)
    bal = balanced_select(train, rem, 200, seed=0)

    d_u = balance_summary(standardised_diffs(train, uniform))["max_abs_d"]
    d_q = balance_summary(standardised_diffs(train, quota))["max_abs_d"]
    d_b = balance_summary(standardised_diffs(train, bal))["max_abs_d"]
    assert d_b < d_q < d_u
    assert d_b < BALANCE_BAR


def test_every_metric_family_is_represented_in_the_axis_set():
    assert set(FAMILY_AXES) == {"longitudinal", "lateral", "tactical", "strategic"}
    for fam, axes in FAMILY_AXES.items():
        assert axes, fam
        assert all(a in ALL_AXES for a in axes)


def test_balance_summary_reports_per_family(corpus):
    train, rem = corpus
    val = balanced_select(train, rem, 120, seed=0)
    s = balance_summary(standardised_diffs(train, val))
    assert set(s["per_family_max_abs_d"]) == set(FAMILY_AXES)
    assert s["max_abs_d"] == pytest.approx(max(s["per_family_max_abs_d"].values()))


def test_headroom_falls_as_axes_are_added(corpus):
    """The correction to the prior pass: feasibility is a property of the axis SET."""
    train, rem = corpus
    coarse = feasibility(train, rem, 300, cell_axes=("junction",))
    fine = feasibility(train, rem, 300, cell_axes=("junction", "has_turn", "speed"))
    assert fine.n_cells > coarse.n_cells
    assert fine.binding_headroom <= coarse.binding_headroom
    assert coarse.cell_axes == ("junction",)          # the axis set travels with the number


def test_feasibility_flags_an_impossible_request(corpus):
    train, rem = corpus
    f = feasibility(train, rem, len(rem) * 2, cell_axes=("junction", "has_turn", "speed"))
    assert f.n_cells_infeasible > 0 and f.binding_headroom < 1


def test_cell_quota_matches_cells_and_costs_pool_depth(corpus):
    """Neither arm dominates — the quota arm buys the cell match with pool depth."""
    train, rem = corpus
    edges = speed_edges(train)
    tp = cell_labels(train, edges).value_counts(normalize=True)

    def cell_l1(v):
        vp = cell_labels(v, edges).value_counts(normalize=True)
        return float((tp - vp.reindex(tp.index).fillna(0)).abs().sum())

    bal = balanced_select(train, rem, 200, seed=0)
    hyb = balanced_select(train, rem, 200, seed=0, cell_quota=True)
    assert cell_l1(hyb) < cell_l1(bal)
    assert max(r["census_fraction"] for r in census_fraction(hyb, rem, edges)) >= \
           max(r["census_fraction"] for r in census_fraction(bal, rem, edges))


def test_manifest_is_complete_and_self_describing(corpus):
    train, rem = corpus
    val = balanced_select(train, rem, 150, seed=0)
    man = build_manifest(train, rem, val, n_val=150, seed=0)
    assert man["n_val"] == 150 and len(man["clip_ids"]) == 150
    assert man["clip_ids"] == sorted(man["clip_ids"])
    assert man["sha256"] == manifest_sha256(val.clip_id)
    assert man["disjointness"]["disjoint"]
    assert set(man["balance"]["per_axis_d"]) == set(ALL_AXES)
    assert man["pool_depth"]["per_cell"] and "census_fraction" in man["pool_depth"]["per_cell"][0]


def test_exclusion_removes_a_third_corpus_from_the_remainder(tmp_path, corpus):
    """A val that must host BOTH arms has to be disjoint from BOTH corpora.

    Measured on the real bytes: 62 clips of a v2-only-clean 600-draw were in v1's TRAIN split.
    """
    from clean_val_select import load
    train, rem = corpus
    pool = pd.concat([train, rem], ignore_index=True)
    pool_p, sel_p, ex_p = (tmp_path / f"{n}.parquet" for n in ("pool", "sel", "ex"))
    # `load` re-validates and re-derives rates, so write the raw (pre-rate) columns
    raw = pool[[c for c in pool.columns if not c.endswith("_rate")]]
    raw.to_parquet(pool_p)
    train[["clip_id"]].to_parquet(sel_p)
    banned = list(rem.clip_id[:1500])
    pd.DataFrame({"clip_id": banned}).to_parquet(ex_p)

    _, rem_open = load(str(pool_p), str(sel_p))
    _, rem_excl = load(str(pool_p), str(sel_p), [str(ex_p)])
    assert len(rem_excl) == len(rem_open) - len(banned)
    assert not (set(rem_excl.clip_id) & set(banned))
    val = balanced_select(train, rem_excl, 50, seed=0)
    assert not (set(val.clip_id) & set(banned))


def test_cannot_draw_more_than_the_remainder(corpus):
    train, rem = corpus
    with pytest.raises(ValueError, match="cannot draw"):
        balanced_select(train, rem, len(rem) + 1, seed=0)
