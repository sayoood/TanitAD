"""Pre-decode ANCHOR filtering — the 2.78x that S2 cannot deliver.

WHAT THIS PINS, and why each part is separate:

(a) S2 filters the DECODED fan, so it saves NOTHING: by the time
    ``anchors + offset`` exists, N decodes have already been paid. ``v0`` is
    known PRE-decode, so the identical band applies to the anchors themselves
    and only the survivors need decoding. That is the whole mechanism.
(b) ONE IMPLEMENTATION. ``anchor_reachability_mask`` must be
    ``reachability_mask`` applied to anchors -- bit-identical, not merely close.
    The 72.08 % measurement is tied to that specific function.
(c) THE GUARD, and the claim it refuses to make. `be2da04` separates a
    structurally-exact VARIABLE policy from a FIXED budget that is an EMPIRICAL
    CALIBRATION (XL's worst window: 102 survivors against a budget of 92). So
    equivalence is ASSERTED PER RUN by ``anchor_prefilter_report``, never
    assumed -- ``winner_survives_frac < 1.0`` is a correctness failure.
(d) THE EGO-DROPOUT LEAK, which is WORSE here than at S2. ``v0`` is the speed
    before ego-dropout; at S2 it could bias a ranking, but pre-decode it decides
    which candidates EXIST. The band must be defeatable per-sample.

Survivor counts are MEASURED against the committed PRODUCTION anchor sets and
the 881 canonical val speeds, not synthesised -- so they are the counts the arms
would actually see. ⚠️ Which fixture you use decides the answer: the 20-step
`anchors_dev256.pt` gives 2.28x with 132 empty windows, the production 4-step
sets give 3.46-3.70x with none. That is pinned, not glossed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tanitad.refs import refc_select as sl

REPO = Path(__file__).resolve().parents[2]
HUB = REPO / "TanitAD Research Hub"
#: THE PRODUCTION vocabularies (4-step). These are what the arms decode.
ANCHORS_FULL = (HUB / "Data Engineering" / "Implementation" / "incoming"
                / "2026-08-04-instrument-durability"
                / "refc_anchors_full_REBUILD.pt")
ANCHORS_SMALL = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
                 / "2026-07-22-refc-small-30k" / "refc_anchors_small64.pt")
#: A 20-step DEV set. NOT the production vocabulary -- see the coverage test.
ANCHORS_DEV = (HUB / "Architecture & Inference" / "Implementation" / "incoming"
               / "2026-07-27-percandidate-labels" / "raw" / "anchors_dev256.pt")
WINDOWS = REPO / "taniteval" / "results" / "windows_flagship-30k.pt"


def _load(p):
    if not p.exists():
        pytest.skip(f"anchor fixture absent: {p.name}")
    d = torch.load(p, map_location="cpu", weights_only=False)
    a = d["anchors"] if isinstance(d, dict) else d
    assert a.dim() == 3 and a.shape[-1] == 2, a.shape
    return a.float()


def _anchors():
    return _load(ANCHORS_FULL)


def _val_v0():
    """The canonical 881-window val speeds -- a REAL v0 distribution, so the
    survivor counts below are the ones the arms would actually see."""
    if not WINDOWS.exists():
        pytest.skip("windows_flagship-30k.pt absent")
    w = torch.load(WINDOWS, map_location="cpu", weights_only=False)
    return torch.as_tensor(w["speed"]).float()


# --- (b) one implementation ------------------------------------------------ #

def test_anchor_mask_IS_the_reachability_mask_applied_to_anchors():
    """Not 'equivalent'. The same function, so the geometry cannot drift."""
    a = _anchors()
    v0 = torch.tensor([0.0, 5.0, 12.0, 30.0])
    got = sl.anchor_reachability_mask(a, v0)
    want = sl.reachability_mask(a[None].expand(v0.shape[0], *a.shape), v0)
    assert torch.equal(got, want)


def test_expanded_and_unexpanded_anchors_agree():
    a = _anchors()
    v0 = torch.tensor([3.0, 18.0])
    flat = sl.anchor_reachability_mask(a, v0)
    pre = sl.anchor_reachability_mask(a[None].expand(2, *a.shape), v0)
    assert torch.equal(flat, pre)


# --- (a) the mechanism: it actually prunes, and it depends on v0 ----------- #

def test_the_band_prunes_the_real_anchor_set():
    a = _anchors()
    n = a.shape[0]
    v0 = torch.full((16,), 8.0)
    keep = sl.anchor_reachability_mask(a, v0)
    frac = keep.float().mean().item()
    assert 0.0 < frac < 1.0, (
        f"a band that keeps {frac:.3f} of {n} anchors prunes nothing or "
        "everything -- it cannot be the mechanism")


def test_the_survivor_set_MOVES_with_v0():
    """A band independent of v0 would be a fixed subset, not reachability."""
    a = _anchors()
    slow = sl.anchor_reachability_mask(a, torch.tensor([1.0]))
    fast = sl.anchor_reachability_mask(a, torch.tensor([28.0]))
    assert not torch.equal(slow, fast)


def test_a_wider_accel_band_never_removes_a_survivor():
    """Monotonicity: relaxing the bound can only admit candidates."""
    a = _anchors()
    v0 = torch.tensor([10.0, 20.0])
    tight = sl.anchor_reachability_mask(a, v0, accel_max=1.0)
    loose = sl.anchor_reachability_mask(a, v0, accel_max=6.0)
    assert bool((loose | tight).eq(loose).all()), "tighter band admitted extras"


# --- (c) the runtime guard ------------------------------------------------- #

def test_report_counts_and_speedup_are_consistent():
    keep = torch.tensor([[True, True, False, False],
                         [True, False, False, False]])
    rep = sl.anchor_prefilter_report(keep, torch.tensor([0, 0]))
    assert rep["n_candidates"] == 4
    assert rep["survivors_mean"] == pytest.approx(1.5)
    assert rep["decode_speedup"] == pytest.approx(4 / 1.5)
    assert rep["winner_survives_frac"] == 1.0
    assert rep["rows_empty"] == 0


def test_guard_FIRES_when_the_full_fan_winner_is_pruned():
    """The failure that matters: the prefilter would change what is emitted."""
    keep = torch.tensor([[True, False], [False, True]])
    rep = sl.anchor_prefilter_report(keep, torch.tensor([0, 0]))
    assert rep["winner_survives_frac"] == pytest.approx(0.5)
    assert rep["winner_survives_frac"] < 1.0


def test_guard_reports_empty_rows_rather_than_dividing_by_zero():
    keep = torch.zeros(3, 8, dtype=torch.bool)
    rep = sl.anchor_prefilter_report(keep, torch.zeros(3, dtype=torch.long))
    assert rep["rows_empty"] == 3
    assert rep["survivors_mean"] == 0.0
    assert rep["decode_speedup"] == float("inf")


def test_report_rejects_a_shape_it_cannot_interpret():
    with pytest.raises(ValueError):
        sl.anchor_prefilter_report(torch.ones(4, dtype=torch.bool),
                                   torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        sl.anchor_prefilter_report(torch.ones(2, 4, dtype=torch.bool),
                                   torch.zeros(3, dtype=torch.long))


def test_guard_passes_on_the_real_anchor_set_when_the_winner_is_reachable():
    a = _anchors()
    v0 = torch.full((8,), 10.0)
    keep = sl.anchor_reachability_mask(a, v0)
    # pick, per row, a survivor as the "winner" -- the regime the 881/881
    # bit-identity was measured in
    idx = keep.float().argmax(dim=1)
    rep = sl.anchor_prefilter_report(keep, idx)
    assert rep["winner_survives_frac"] == 1.0
    assert rep["decode_speedup"] > 1.0


# --- (d) the ego-dropout leak --------------------------------------------- #

def test_the_band_is_defeatable_per_sample_for_ego_dropout():
    """``v0`` is the PRE-dropout speed. On a sample whose speed was withheld the
    band must be neutralised, or the withheld channel decides which candidates
    exist -- a stronger leak than the ranking one S2 guards. This pins that
    ``keep | ~ego_keep`` restores the full fan for exactly those rows."""
    a = _anchors()
    v0 = torch.tensor([9.0, 9.0])
    ego_keep = torch.tensor([True, False])
    keep = sl.anchor_reachability_mask(a, v0)
    keep = keep | (~ego_keep)[:, None]
    assert not bool(keep[0].all()), "row with speed KEPT should still be banded"
    assert bool(keep[1].all()), "row with speed WITHHELD must keep its whole fan"


# --- replication of `be2da04` on the REAL vocabularies + REAL v0 ----------- #

def test_replicates_the_measured_survivor_counts_on_the_canonical_val():
    """`be2da04` reports the VARIABLE policy at "17.3 / 36.0 / 74.0 mean
    (3.46-3.70x)". Re-measured here on the production anchor sets against the
    881 canonical val speeds, rather than inherited from the commit message."""
    v0 = _val_v0()
    for path, n_expect, surv_expect in ((ANCHORS_SMALL, 64, 17.3),
                                        (ANCHORS_FULL, 256, 74.0)):
        a = _load(path)
        keep = sl.anchor_reachability_mask(a, v0)
        rep = sl.anchor_prefilter_report(
            keep, torch.zeros(v0.shape[0], dtype=torch.long))
        assert rep["n_candidates"] == n_expect
        assert rep["survivors_mean"] == pytest.approx(surv_expect, abs=0.1)
        assert 3.4 < rep["decode_speedup"] < 3.8
        assert rep["rows_empty"] == 0, (
            "a production vocabulary must cover every val speed; an empty "
            "survivor set means the fan falls back to full width there")


def test_the_20_step_DEV_set_has_a_HIGH_SPEED_COVERAGE_HOLE():
    """⚠️ A property of that fixture, recorded so it is not mistaken for a
    failure of the mechanism -- and so nobody re-derives it the hard way.

    MEASURED: `anchors_dev256.pt` reaches a mean speed of only ~18 m/s, so on
    the canonical val (v0 up to 36.55 m/s) **132 of 881 windows have ZERO
    reachable anchors** and degrade to the full fan. Its speedup is 2.28x, not
    the 3.46x of the production set. The empty rows are the FAST windows
    (v0 mean 30.32 vs 12.64 overall) -- the opposite of the intuition that a
    stopped ego is the hard case.
    """
    a = _load(ANCHORS_DEV)
    v0 = _val_v0()
    keep = sl.anchor_reachability_mask(a, v0)
    empty = keep.sum(dim=1) == 0
    assert int(empty.sum()) > 0, "the coverage hole is the point of this test"
    assert v0[empty].mean() > v0.mean(), "empty rows should be the FAST ones"
    rep = sl.anchor_prefilter_report(
        keep, torch.zeros(v0.shape[0], dtype=torch.long))
    assert rep["decode_speedup"] < 3.4, "dev set should underperform production"
