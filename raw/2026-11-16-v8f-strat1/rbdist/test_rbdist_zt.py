import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbdist_zt as R  # noqa: E402


def _clip(n=30, seed=0):
    rng = np.random.default_rng(seed)
    is_rb = np.zeros(n, bool); is_rb[10:25] = True
    flag = np.zeros(n, np.uint8)
    ex = np.full(n, -1, np.int32); ex[12:16] = 1; ex[16:20] = 2; ex[20:25] = 3
    dist = rng.uniform(10, 150, n).astype(np.float64); dist[22] = np.nan
    return is_rb, flag, ex, dist


def test_label_rule():
    y = R.label_rule([1, 1, 0, 1], [0, 2, 1, -1])
    assert y.tolist() == [0, 2, R.IGNORE, R.IGNORE]


def test_features_values_and_nan_distance():
    is_rb, flag, ex, dist = _clip()
    f = R.features(is_rb, flag, ex, dist, 18, 0)
    assert f[0] == 1.0 and f[1] == 2.0 and f[2] == 2.0 and f[4] == 1.0
    assert abs(f[3] - np.clip(dist[18] / 100, 0, 3)) < 1e-12
    f = R.features(is_rb, flag, ex, dist, 22, 0)
    assert f[3] == 0.0 and f[4] == 0.0                       # absent distance -> 0 with valid flag 0
    f = R.features(is_rb, flag, ex, dist, 5, 0)
    assert f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0       # before the roundabout: no state, no exit number
    flag2 = flag.copy(); flag2[18] = 1
    assert R.features(is_rb, flag2, ex, dist, 18, 0)[0] == 0.0   # flag != 0 -> state off


def test_exit_last_is_latest_numbered_row_at_or_before_r():
    is_rb, flag, ex, dist = _clip()
    ex2 = ex.copy(); ex2[18] = -1                              # the current text carries no number
    f = R.features(is_rb, flag, ex2, dist, 18, 0)
    assert f[1] == 0.0 and f[2] == 2.0                         # latest earlier numbered row


def test_admissibility_future_garbage_leaves_features_unchanged():
    is_rb, flag, ex, dist = _clip()
    for r in range(len(flag)):
        base = R.features(is_rb, flag, ex, dist, r, 0)
        a, b, c, d = is_rb.copy(), flag.copy(), ex.copy(), dist.copy()
        a[r + 1:] = ~a[r + 1:]; b[r + 1:] = 3; c[r + 1:] = 9; d[r + 1:] = 777.0
        assert R.features(a, b, c, d, r, 0) == base, f"row {r} reads a later row"


def test_mutant_reading_a_future_row_goes_red():
    is_rb, flag, ex, dist = _clip()
    red = 0
    for r in range(len(flag) - 6):
        base = R.features(is_rb, flag, ex, dist, r, 0)
        a, b, c, d = is_rb.copy(), flag.copy(), ex.copy(), dist.copy()
        a[r + 1:] = ~a[r + 1:]; b[r + 1:] = 3; c[r + 1:] = 9; d[r + 1:] = 777.0
        if R.features(a, b, c, d, r, 0, _shift=5) != R.features(is_rb, flag, ex, dist, r, 0, _shift=5):
            red += 1
    assert red > 0, "the future-read mutant stayed GREEN: the probe cannot see a future read"


def test_feature_function_takes_no_label_or_ego():
    names = set(inspect.signature(R.features).parameters) | set(inspect.signature(R.feature_matrix).parameters)
    for bad in ("y", "lat", "lat_cls", "label", "track", "speed", "ego", "nav", "g"):
        assert bad not in names


def test_verdict_rule_thresholds():
    assert R.verdict_rule(0.0383, True) == "TEXT-CARRIES-SIDE"
    assert R.verdict_rule(0.0382, True) == "TEXT-SILENT"
    assert R.verdict_rule(0.20, False) == "TEXT-SILENT"
    assert R.suggestive_unseparated(0.2, False) and not R.suggestive_unseparated(0.2, True)


def test_state_only_classifier_is_the_constant_majority():
    rng = np.random.default_rng(0)
    y = rng.choice(5, 500, p=[.6, .2, .1, .05, .05])
    Xtr = np.ones((500, 5)); Xte = np.ones((80, 5))
    p = R.fit_predict(Xtr, y, Xte, R.FEATS["S"])
    assert set(p.tolist()) == {int(np.bincount(y).argmax())}


def test_full_features_decode_a_planted_side_and_shuffle_does_not():
    rng = np.random.default_rng(1)
    n = 600
    ex = rng.integers(1, 4, n).astype(float)
    y = np.where(ex == 1, 2, np.where(ex == 3, 1, 0))
    X = np.column_stack([np.ones(n), ex, ex, rng.uniform(0, 3, n), np.ones(n)])
    p = R.fit_predict(X[:400], y[:400], X[400:], R.FEATS["ED"])
    assert (p == y[400:]).mean() > 0.99
    Xs = X.copy(); Xs[:, 1:] = X[rng.permutation(n)][:, 1:]
    p = R.fit_predict(X[:400], y[:400], Xs[400:], R.FEATS["ED"])
    assert (p == y[400:]).mean() < 0.6


def test_macro_f1_perfect_and_constant():
    y = np.array([0, 1, 2, 3, 4]); assert R.macro_f1(y, y) == 1.0
    assert R.macro_f1(y, np.zeros(5, int)) < 0.2


def test_real_table_future_garbage_probe_on_eval_rb_rows():
    """the same admissibility probe on real pinned table rows (first 40 eval roundabout rows)"""
    S1 = R.S1
    tab = np.load(S1 + "/read/inputs/navtok_table.npz", allow_pickle=False)
    ex_t = R.text_arrays(HERE + "/src/texts.json")
    rb, fl, ex, di = R.split_arrays(tab, "eval", ex_t)
    off = np.repeat(np.asarray(tab["eval_off"]), np.asarray(tab["eval_n"]))
    rows = np.where((fl == 0) & rb)[0][::25][:40]
    assert len(rows) > 10
    for r in rows:
        o = int(off[r]); a, b, c, d = rb.copy(), fl.copy(), ex.copy(), di.copy()
        a[r + 1:] = ~a[r + 1:]; b[r + 1:] = 3; c[r + 1:] = 9; d[r + 1:] = 777.0
        assert R.features(a, b, c, d, int(r), o) == R.features(rb, fl, ex, di, int(r), o)
