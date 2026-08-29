"""The GATE-FACING open-loop print, migrated off the deprecated estimator.

`closedloop.py`, `driving.py` and `hierarchy.py` were all migrated to the
episode-cluster bootstrap. `runner.run_one`'s summary line — the number a human
actually reads when gating an arm — was not: it printed
``res["heldout"]["model"]``, i.e. `overlapping_holdout_se`.

Why that was worse than a width problem (MEASURED 2026-07-25 over the 27
committed ``results/windows_*.pt`` fixtures, dev box, no GPU):

  * CI width ratio primary/legacy: **1.11-3.10x**, median 1.50x — the legacy
    interval was too narrow by that factor, in the documented 1.28-2.06x band.
  * The POINT estimate moves as well: ``ade_0_2s`` shifts **-10.5 % … +7.1 %**,
    because `_agg` averages 8 overlapping 20 % holdouts while the primary point
    estimate is the full-set metric.
  * The cross-arm **RANKING changes in 10 of 27 positions**.
  * On v1 ``flagship-30k`` the line printed ``0.452±0.031`` where the primary is
    ``0.427 [0.368, 0.487]`` — and 0.4522 is exactly the registry headline.

The legacy block is KEPT, numerically unchanged, under a self-labelling key, for
published-number reproduction only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from taniteval import bench

N_WP = 4
SRC = Path(__file__).resolve().parents[1] / "taniteval"


def _win(n_ep=10, per=9, seed=0):
    """Synthetic rollout windows with per-episode structure, so the episode
    cluster bootstrap and the holdout block genuinely disagree."""
    g = torch.Generator().manual_seed(seed)
    n = n_ep * per
    eid = [i // per for i in range(n)]
    # a per-EPISODE offset: this is the between-cluster variance the overlapping
    # holdout block under-counts and the cluster bootstrap does not.
    off = torch.rand(n_ep, generator=g) * 2.0
    base = torch.stack([off[e].expand(N_WP * 2) for e in eid]).reshape(n, N_WP, 2)
    gt = torch.zeros(n, N_WP, 2)
    return {
        "pred": base + 0.1 * torch.rand(n, N_WP, 2, generator=g),
        "gt": gt,
        "cv": base * 1.3,
        "speed": torch.rand(n, generator=g) * 10,
        "head_deg": torch.randn(n, generator=g) * 5,
        "eid": eid,
        "wp_steps": (5, 10, 15, 20),
    }


@pytest.fixture(scope="module")
def res():
    return bench.run(_win(), n_boot=400)


# --------------------------------------------------------------------------- #
# 1. the quarantine                                                             #
# --------------------------------------------------------------------------- #
def test_legacy_block_is_emitted_under_a_self_labelling_key(res):
    lg = res[bench.LEGACY_BLOCK]
    assert bench.LEGACY_BLOCK == "legacy_overlapping_holdout_se"
    assert lg["_estimator"] == bench.DEPRECATED_ESTIMATOR == "overlapping_holdout_se"
    assert "NOT admissible" in lg["_why_kept"]
    assert "1.28-2.06x" in lg["estimator_note"]


def test_heldout_key_survives_as_a_TOMBSTONE_not_as_the_numbers(res):
    """⭐ SUPERSEDED 2026-08-28 (PI ruling, ``ESTIMATOR_CLOSEOUT.md``).

    The old assertion here was ``res["heldout"]["model"] is
    res[LEGACY_BLOCK]["model"]`` — the alias was the SAME dict, so a
    decision-grade-looking ``{"mean", "ci95"}`` sat under a bare key that
    ``gate_guard`` cannot see (it carries no verdict word). Five consumers read
    it as the arm's headline: the dashboard ORDERED on it,
    ``runner.regression`` fell back to it, and ``refc_rerank`` /
    ``efficiency`` / ``generalization`` published it.

    Both halves of the requirement are pinned here, because they pull opposite
    ways and dropping either one is a silent regression:

    * the KEY must survive — ``run_gate._deprecated_present`` searches
      ``("heldout", "model")`` and deleting it disarms the gate's own refusal;
    * the NUMBERS must not — nothing may be quotable out of it.
    """
    ho = res["heldout"]["model"]
    assert ho is not res[bench.LEGACY_BLOCK]["model"], \
        "the alias must no longer BE the legacy dict"
    assert set(ho) == set(res[bench.LEGACY_BLOCK]["model"]), \
        "same metric keys, so every presence check keeps working"
    for m in ("ade_0_2s", "fde@2s", "miss_rate@2m"):
        node = ho[m]
        # armed: this is the branch `_deprecated_present` fires on
        assert node["estimator"] == bench.DEPRECATED_ESTIMATOR
        assert node["deprecated"] is True and node["withdrawn"] is True
        # withdrawn: nothing to quote
        assert not {"mean", "ci95", "std"} & set(node), \
            f"a number is still quotable out of the bare heldout key: {node}"
        assert node["_moved_to"] == f"{bench.LEGACY_BLOCK}.model.{m}"


def test_the_withdrawn_number_is_still_reproducible_one_key_over(res):
    """History is closed, not deleted: the value lives under the self-labelling
    key, bit-identical, so every published figure stays traceable."""
    lg = res[bench.LEGACY_BLOCK]["model"]
    for m in ("ade_0_2s", "fde@2s", "miss_rate@2m"):
        assert isinstance(lg[m]["mean"], float)
        assert lg[m]["estimator"] == bench.DEPRECATED_ESTIMATOR


def test_legacy_numbers_are_bit_identical_to_the_pre_migration_block(res):
    """Published-number reproduction: the quarantine must WRAP the old block,
    never recompute it."""
    import numpy as np
    from taniteval.bench import _agg, _suite
    from tanitad.eval.gates import split_by_episode
    w = _win()
    splits = [split_by_episode(w["eid"], 0.2, s) for s in range(0, 8)]
    expect = _agg([_suite(w["pred"][va], w["gt"][va]) for _t, va in splits])
    got = res[bench.LEGACY_BLOCK]["model"]      # the bare alias is a tombstone
    for m in expect:
        assert np.isclose(got[m]["mean"], expect[m]["mean"], atol=1e-12)
        assert np.isclose(got[m]["ci95"], expect[m]["ci95"], atol=1e-12)


def test_each_run_RE_MEASURES_its_own_narrowing_factor(res):
    """The 1.28-2.06x finding must be re-measured per artifact, not cited from a
    doc — the program rule that a number carries its construction."""
    r = res[bench.LEGACY_BLOCK]["ci_width_ratio_new_over_legacy"]
    assert r["ade_0_2s"] is not None and r["ade_0_2s"] > 1.0, (
        "on clustered data the honest interval must be WIDER")
    for m in ("fde@2s", "miss_rate@2m"):
        assert m in r


def test_the_point_estimate_shift_is_reported_not_just_the_width(res):
    """The half of the problem that was invisible: `_agg`'s mean-over-holdouts
    is a DIFFERENT point estimate from the full-set metric."""
    s = res[bench.LEGACY_BLOCK]["point_estimate_shift_primary_minus_legacy"]
    a = s["ade_0_2s"]
    assert a["primary_mean"] == res["cluster_bootstrap"]["model"]["ade_0_2s"]["mean"]
    assert a["legacy_mean"] == \
        res[bench.LEGACY_BLOCK]["model"]["ade_0_2s"]["mean"]
    assert a["delta"] == pytest.approx(a["primary_mean"] - a["legacy_mean"],
                                       abs=1e-9)


def test_the_primary_point_estimate_is_the_full_set_metric(res):
    """...and the legacy one is not. If these ever coincide the fixture stopped
    exercising the difference and this suite stops proving anything."""
    prim = res["cluster_bootstrap"]["model"]["ade_0_2s"]["mean"]
    assert prim == pytest.approx(res["full_set"]["model"]["ade_0_2s"], abs=1e-4)
    assert prim != pytest.approx(
        res[bench.LEGACY_BLOCK]["model"]["ade_0_2s"]["mean"], abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. the print itself                                                           #
# --------------------------------------------------------------------------- #
def _run_one_src() -> str:
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    body = src.split("def run_one(", 1)[1].split("\ndef ", 1)[0]
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_the_gate_facing_print_no_longer_reads_the_deprecated_block():
    """⭐ the migration, asserted at the only place it can be: `run_one`'s own
    body. The deprecated block must not be dereferenced there at all."""
    body = _run_one_src()
    assert not re.search(r'res\[\s*["\']heldout["\']\s*\]', body), (
        "runner.run_one still reads the DEPRECATED heldout block for the "
        "gate-facing print")
    assert 'res["cluster_bootstrap"]["model"]' in body


def test_the_gate_facing_print_names_its_estimator():
    """A number printed without its estimator is how an unlabelled interval got
    published for months."""
    body = _run_one_src()
    assert "estimator=" in body and "['estimator']" in body
    assert "n_boot" in body


def test_the_gate_facing_print_carries_the_paired_separation():
    """`beats CV` is a decision; an unseparated win is a tie, not a win — so the
    paired interval belongs on the same line as the point estimate."""
    body = _run_one_src()
    assert "model_vs_cv_paired" in body
    assert "SEPARATED" in body and "separated" in body


def test_the_run_stamps_which_val_cache_produced_it():
    """The blast-radius fix: no result JSON used to record its val provenance,
    so the audit had to INFER the corpus from ``n_windows == 881``."""
    body = _run_one_src()
    assert 'res["val_parity"] = data.last_val_parity()' in body


def test_regression_gate_reads_the_primary_estimate():
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    body = src.split("def regression(", 1)[1].split("\ndef ", 1)[0]
    assert '"cluster_bootstrap"' in body
    assert "DEPRECATED" in body, (
        "the pre-cluster_bootstrap fallback must label itself")
