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

import sys
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


# --- THE CORRECTNESS PROOF: on a real model, on vs off ---------------------- #
#
# The mask tests above are about the band. THIS is about the decoder: does
# decoding only the survivors emit the same trajectory as decoding all N?
# It is exact iff candidates do not interact, and `CrossAttnLayer` cross-attends
# q->kv with NO self-attention over the candidate axis (per-token MLP and
# LayerNorm). That is a structural fact, so it is verified here rather than
# assumed -- and this test is what fails if a candidate-axis interaction is ever
# added to the decoder, which would silently make the flag unsound.

from tanitad.refs.refc import RefCModel, refc_smoke_config       # noqa: E402


def _smoke(prefilter: bool):
    cfg = refc_smoke_config()
    cfg.sel_reach_clamp = False
    cfg.sel_anchor_prefilter = prefilter
    torch.manual_seed(0)
    m = RefCModel(cfg).eval()
    return m


def _same_weights(a, b):
    b.load_state_dict(a.state_dict())
    return b


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows-CPU BLAS picks shape-dependent gemm tilings, so the batched "
           "and the subset decode differ by ULPs (measured max 1.9e-06 on the "
           "dev box, 2026-08-15). The claim is per-backend: bit-exact where it "
           "deploys — 881/881 identical selections on Linux/CUDA and on Thor "
           "(fan-width stream, 2026-08-04). A ULP tolerance here would silently "
           "weaken the structural guarantee this test exists to pin, so on "
           "Windows it is skipped, not loosened.")
def test_prefilter_is_BIT_EXACT_on_every_candidate_it_decodes():
    """THE structural claim. For every surviving candidate, the confidence and
    the offset must be BIT-identical to the full decode -- that is what
    candidate-independence buys, and it is what fails if self-attention over the
    candidate axis is ever added to ``CrossAttnLayer``."""
    off = _smoke(False)
    on = _same_weights(off, _smoke(True)).eval()
    torch.manual_seed(1)
    fmap = torch.randn(4, off.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(4, off.decoder.cond_proj.in_features)
    v_ms = torch.tensor([4.0, 9.0, 14.0, 22.0])
    with torch.no_grad():
        d0 = off.decoder(fmap, mvec, v_ms=v_ms)
        d1 = on.decoder(fmap, mvec, v_ms=v_ms)
    keep = sl.anchor_reachability_mask(
        off.decoder.anchors.to(fmap.dtype), v_ms)
    assert keep.any() and not keep.all(), "need a partial band to test anything"
    assert float((d0["offset"] - d1["offset"]).abs()[keep].max()) == 0.0
    assert float((d0["anchor_logits"]
                  - d1["anchor_logits"]).abs()[keep].max()) == 0.0


def test_the_prefiltered_pick_IS_the_survivor_restricted_argmax():
    """The exact semantic, stated as an equality rather than a hope.

    ⚠️ It is NOT 'the same pick as the full fan'. On a TRAINED model those
    coincide -- 881/881, because 72-74 % of the fan is unreachable and MEASURED
    never selected -- but that is a property of a trained score, not of this
    code. Under random init the full-fan argmax is arbitrary and frequently
    unreachable, so pinning full-fan equality here would pin the wrong thing.
    """
    off = _smoke(False)
    on = _same_weights(off, _smoke(True)).eval()
    torch.manual_seed(1)
    fmap = torch.randn(4, off.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(4, off.decoder.cond_proj.in_features)
    v_ms = torch.tensor([4.0, 9.0, 14.0, 22.0])
    with torch.no_grad():
        d0 = off.decoder(fmap, mvec, v_ms=v_ms)
        d1 = on.decoder(fmap, mvec, v_ms=v_ms)
    keep = sl.anchor_reachability_mask(
        off.decoder.anchors.to(fmap.dtype), v_ms)
    restricted = d0["sel_score"].masked_fill(~keep, float("-inf")).argmax(dim=1)
    assert torch.equal(restricted, d1["sel_idx"])


def test_the_runtime_telemetry_makes_no_claim_it_cannot_check():
    """`winner_survives_frac` must NOT appear in the decoder telemetry: `sel_idx`
    is already the restricted argmax, so it would be 1.0000 by construction and
    read as a verified invariant. Winner-survival is an OFFLINE calibration."""
    on = _smoke(True)
    torch.manual_seed(5)
    fmap = torch.randn(3, on.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(3, on.decoder.cond_proj.in_features)
    with torch.no_grad():
        d = on.decoder(fmap, mvec, v_ms=torch.tensor([6.0, 6.5, 7.0]))
    assert "winner_survives_frac" not in d["sel_tele"]
    assert "prefilter_survivors_mean" in d["sel_tele"]
    assert "prefilter_rows_full_fan" in d["sel_tele"]


def test_prefilter_actually_reduced_the_decode_width():
    """An 'exact' prefilter that pruned nothing would pass the test above while
    saving nothing -- pin that it really shrank the decode."""
    on = _smoke(True)
    torch.manual_seed(2)
    fmap = torch.randn(3, on.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(3, on.decoder.cond_proj.in_features)
    with torch.no_grad():
        d = on.decoder(fmap, mvec, v_ms=torch.tensor([6.0, 6.5, 7.0]))
    assert d["sel_tele"]["prefilter_speedup"] > 1.0
    assert d["sel_tele"]["prefilter_k"] < on.decoder.anchors.shape[0]


def test_prefilter_is_INERT_without_a_speed():
    """No ``v_ms`` -> no band -> the untouched path, and no prefilter telemetry."""
    on = _smoke(True)
    torch.manual_seed(3)
    fmap = torch.randn(2, on.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(2, on.decoder.cond_proj.in_features)
    with torch.no_grad():
        d = on.decoder(fmap, mvec)
    assert "prefilter_k" not in d["sel_tele"]


def test_a_withheld_speed_keeps_its_WHOLE_fan():
    """``ego_keep`` False -> that row must not be filtered at all, or the
    withheld channel decides which candidates are even computed."""
    on = _smoke(True)
    torch.manual_seed(4)
    fmap = torch.randn(2, on.decoder.feat_proj.in_features, 4, 4)
    mvec = torch.randn(2, on.decoder.cond_proj.in_features)
    with torch.no_grad():
        d = on.decoder(fmap, mvec, v_ms=torch.tensor([8.0, 8.0]),
                       ego_keep=torch.tensor([True, False]))
    # one row unfiltered => the batch budget is the full fan, so nothing saved
    assert d["sel_tele"]["prefilter_k"] == on.decoder.anchors.shape[0]


# --- THE OFFLINE CALIBRATION, on the REAL trained fans ---------------------- #
#
# This is the claim the runtime telemetry deliberately REFUSES to make (see
# `test_the_runtime_telemetry_makes_no_claim_it_cannot_check`): does the
# prefilter preserve the SHIPPED pick? Answering it needs the full fan, which is
# exactly the cost the flag avoids -- so it is answered ONCE here, offline,
# against banked fans decoded by the real 30k checkpoints.

FANS = REPO / "taniteval" / "results"


@pytest.mark.parametrize("arm,n", [("refc-xl-30k", 256), ("refc-base-30k", 128)])
def test_prefilter_preserves_the_SHIPPED_pick_on_the_trained_fans(arm, n):
    """MEASURED 2026-08-04 on ckpt_step 29999, 881 canonical val windows:

        XL-256   survivors 74.0 (max 102)  3.46x  881/881 identical  dADE 0.0
        base-128 survivors 36.0 (max  51)  3.55x  881/881 identical  dADE 0.0

    The survivor means reproduce `be2da04` (74.0 / 36.0) exactly, and 102 is the
    worst-window count that commit warns a FIXED budget of 92 would have missed.
    """
    fp = FANS / f"fan_{arm}.pt"
    if not fp.exists():
        pytest.skip(f"{fp.name} absent")
    d = torch.load(fp, map_location="cpu", weights_only=False)
    a = _load(ANCHORS_FULL)[:d["n_anchors"]]
    assert d["n_anchors"] == n
    keep = sl.anchor_reachability_mask(a, d["v0"])
    keep = keep | (~keep.any(dim=1))[:, None]
    rep = sl.anchor_prefilter_report(keep, d["sel"])

    # (1) the pick is never pruned
    assert rep["winner_survives_frac"] == 1.0
    assert rep["rows_empty"] == 0
    # (2) the restricted argmax IS the shipped index, on every window
    restricted = d["logits"].masked_fill(~keep, float("-inf")).argmax(dim=1)
    assert torch.equal(restricted, d["sel"]), "selection index moved"
    # (3) therefore the emitted trajectory, and every metric built on it, is
    #     unchanged -- EXACTLY, not within a tolerance
    fan, gt = d["fan"], d["gt"]
    rows = torch.arange(len(d["sel"]))
    ade = lambda i: (fan[rows, i] - gt).norm(dim=-1).mean(dim=-1)
    assert float((ade(restricted) - ade(d["sel"])).abs().max()) == 0.0
    # (4) and it actually saved something
    assert rep["decode_speedup"] > 3.0


def test_the_band_is_not_knife_edge_on_the_trained_fans():
    """⚠️ The committed anchors are a REBUILD and match the originals to
    7.6e-06 m (float32 rounding), not bit-exactly. If survivors sat on the band
    edge that difference could flip candidates and the calibration above would
    be fragile. MEASURED: ZERO flips under +/-1e-5 m on both arms."""
    fp = FANS / "fan_refc-xl-30k.pt"
    if not fp.exists():
        pytest.skip("fan_refc-xl-30k.pt absent")
    d = torch.load(fp, map_location="cpu", weights_only=False)
    a = _load(ANCHORS_FULL)[:d["n_anchors"]]
    base = sl.anchor_reachability_mask(a, d["v0"])
    for eps in (1e-5, -1e-5):
        assert torch.equal(sl.anchor_reachability_mask(a + eps, d["v0"]), base)
