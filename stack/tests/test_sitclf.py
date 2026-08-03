"""Situation-classifier head + late fusion: causality, leakage, and that the
fusion actually fixes the swamping defect it was written for."""

import numpy as np
import pytest
import torch

from tanitad.eval.ap_ci import average_precision
from tanitad.eval.sitclf import (
    CausalSitHead,
    causal_window,
    clip_runs,
    cluster_folds,
    late_fuse_scores,
    predict_sit_head,
    train_sit_head,
)


# --------------------------------------------------------------------------- #
# clip geometry                                                               #
# --------------------------------------------------------------------------- #
def test_clip_runs_splits_on_id_change():
    st, en = clip_runs(np.array([7, 7, 7, 9, 9, 4]))
    assert list(st) == [0, 3, 5] and list(en) == [3, 5, 6]


def test_causal_window_is_causal_and_never_crosses_a_clip():
    """The load-bearing property: row t must contain frames t-win+1..t of its OWN
    clip and nothing later. A window that leaked one future frame would make the
    3 s anticipation target partly readable, exactly the failure the stream's
    'STRICTLY CAUSAL' comments were caught overstating."""
    feats = np.arange(20, dtype=np.float32).reshape(20, 1)
    st, en = clip_runs(np.array([0] * 10 + [1] * 10))
    X, ok = causal_window(feats, st, en, win=4)
    # first 3 rows of each clip lack a full history
    assert list(np.flatnonzero(~ok)) == [0, 1, 2, 10, 11, 12]
    assert list(X[3]) == [0, 1, 2, 3]          # offsets -3..0, past-and-present
    assert list(X[13]) == [10, 11, 12, 13]     # clip 2 never reads clip 1
    assert X[9].max() == 9                     # no frame from the future


def test_causal_window_zero_fills_invalid_rows():
    feats = np.ones((6, 2), dtype=np.float32)
    st, en = clip_runs(np.zeros(6, dtype=int))
    X, ok = causal_window(feats, st, en, win=3)
    assert not ok[0] and np.all(X[0] == 0)
    assert ok[5] and np.all(X[5] == 1)


def test_short_clips_are_dropped_not_padded():
    feats = np.ones((4, 1), dtype=np.float32)
    st, en = clip_runs(np.array([0, 0, 1, 1]))
    _, ok = causal_window(feats, st, en, win=3)
    assert not ok.any()


def test_cluster_folds_never_split_a_cluster():
    """Row-level splitting would put frames of one clip on both sides and leak."""
    cc = np.repeat(np.arange(30), 7)
    f = cluster_folds(cc, n_folds=2, seed=0)
    for c in np.unique(cc):
        assert len(np.unique(f[cc == c])) == 1
    assert set(np.unique(f)) == {0, 1}


# --------------------------------------------------------------------------- #
# the head                                                                    #
# --------------------------------------------------------------------------- #
def test_head_shapes_and_window_guard():
    m = CausalSitHead(in_dim=3, win=8, d=32, n_out=3)
    assert m(torch.zeros(5, 8, 3)).shape == (5, 3)
    with pytest.raises(ValueError):
        m(torch.zeros(5, 9, 3))                 # wrong window must fail loud


def test_head_window_length_is_honoured():
    for win in (4, 16):
        m = CausalSitHead(in_dim=2, win=win, d=16, n_out=1)
        assert m.pos.shape == (win, 16)
        assert m(torch.zeros(2, win, 2)).shape == (2, 1)


def test_head_learns_a_signal_that_needs_a_LONG_window():
    """Smoke contract AND the window mechanism, with its own negative control:
    the target is decidable ONLY from a frame 12 steps back, so a win=16 head
    must fit it and a win=4 head must sit at chance. If win=4 scores above its
    base rate the window plumbing is leaking, and if win=16 does not fit, the
    architecture cannot use history at all -- either way the WIN sweep run on
    real data would be meaningless."""
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=(n, 1)).astype(np.float32)
    cc = np.repeat(np.arange(n // 100), 100)
    st, en = clip_runs(cc)
    # label at t = (value 12 frames earlier exceeded 1.0), within-clip
    y = np.zeros((n, 1), np.float32)
    for a, b in zip(st, en):
        y[a + 12:b, 0] = (x[a:b - 12, 0] > 1.0).astype(np.float32)
    res = {}
    for win in (4, 16):
        X, ok = causal_window(x, st, en, win)
        V = ok[:, None].astype(np.float32)
        m = train_sit_head(X, y, V, win=win, in_dim=1, epochs=25, d=64,
                           pos_weight=3.0, batch=256, seed=0)
        s = predict_sit_head(m, X, in_dim=1)[:, 0]
        res[win] = average_precision(y[ok, 0], s[ok])
    base = float(y.mean())
    assert res[16] > 0.90, res
    assert res[4] < 3 * base, (res, base)          # short window == chance
    assert res[16] > res[4] + 0.5, res


# --------------------------------------------------------------------------- #
# late fusion — the fix                                                       #
# --------------------------------------------------------------------------- #
def _swamp_fixture(seed=0, n_clip=120, per=100):
    """A strong 1-dim signal and a weak-but-wide noisy one, the shape that made
    early concat lose to its own ego ablation."""
    rng = np.random.default_rng(seed)
    n = n_clip * per
    cc = np.repeat(np.arange(n_clip), per)
    y = (rng.random(n) < 0.08).astype(np.int64)
    strong = rng.normal(size=n) + 2.0 * y            # the "ego" score
    weak = rng.normal(size=n) + 0.5 * y              # the "image" score
    return cc, y, strong, weak


def test_late_fusion_beats_both_unimodal_arms():
    cc, y, strong, weak = _swamp_fixture()
    valid = np.ones(y.size, bool)
    folds = cluster_folds(cc, 2, seed=0)
    fused = late_fuse_scores(np.stack([weak, strong], 1), y, valid, folds)
    ap_f = average_precision(y, fused)
    assert ap_f > average_precision(y, strong)
    assert ap_f > average_precision(y, weak)


def test_late_fusion_is_out_of_fit():
    """Every returned score must come from a combiner fitted without that row.
    Verified by construction: refitting per fold on the complement reproduces it."""
    from sklearn.linear_model import LogisticRegression

    cc, y, strong, weak = _swamp_fixture(seed=3, n_clip=40, per=50)
    valid = np.ones(y.size, bool)
    folds = cluster_folds(cc, 2, seed=1)
    F = np.stack([weak, strong], 1)
    got = late_fuse_scores(F, y, valid, folds)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = F[tr].mean(0), np.maximum(F[tr].std(0), 1e-9)
        lr = LogisticRegression(C=1.0, max_iter=300).fit((F[tr] - mu) / sd, y[tr])
        want = lr.decision_function((F[te] - mu) / sd)
        assert np.allclose(got[te], want)


def test_late_fusion_with_an_uninformative_column_does_not_collapse():
    """NEGATIVE CONTROL: adding pure noise as a second modality must not destroy
    the good arm — which is precisely what the early concat did."""
    cc, y, strong, _ = _swamp_fixture(seed=5)
    rng = np.random.default_rng(6)
    noise = rng.normal(size=y.size)
    valid = np.ones(y.size, bool)
    folds = cluster_folds(cc, 2, seed=0)
    fused = late_fuse_scores(np.stack([noise, strong], 1), y, valid, folds)
    assert average_precision(y, fused) > 0.9 * average_precision(y, strong)


def test_late_fusion_masks_invalid_rows_to_minus_inf():
    cc, y, strong, weak = _swamp_fixture(seed=7, n_clip=20, per=40)
    valid = np.ones(y.size, bool)
    valid[:37] = False
    folds = cluster_folds(cc, 2, seed=0)
    out = late_fuse_scores(np.stack([weak, strong], 1), y, valid, folds)
    assert np.all(np.isneginf(out[:37])) and np.all(np.isfinite(out[valid]))


def test_late_fusion_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        late_fuse_scores(np.zeros((10, 2)), np.zeros(9), np.ones(10, bool),
                         np.zeros(10))
