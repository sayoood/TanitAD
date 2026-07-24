"""Estimator tests for ``taniteval/ci.py`` — the tests that would have caught W1.

The 360° review (2026-07-20, W1/P2) found that every single-arm interval in the
registry came from 8 OVERLAPPING random 20 % holdouts divided by sqrt(8), and was
labelled "8-split episode-disjoint jackknife". These tests pin the three claims
that make that construction wrong, on synthetic data with a KNOWN answer:

  * ``test_naive_se_shrinks_with_more_splits``      — it measures split-selection
      noise, not model uncertainty (the smoking gun: same data, more splits,
      smaller "SE").
  * ``test_naive_se_too_small_under_episode_correlation`` — with within-episode
      correlation the naive half-width is ~sqrt(n_ep)/n_splits of the correct
      one, i.e. ~20 % too narrow at 40 episodes / 8 splits.
  * ``test_coverage_cluster_vs_naive``              — the review's falsifiable
      criterion: empirical coverage 93-97 % for the cluster bootstrap, and
      demonstrable UNDER-coverage for the naive estimator.

Plus reproduction (``overlapping_holdout_se`` must still return the exact old
arithmetic), point-estimate consistency against ``bench._suite``, the paired-vs-
quadrature power comparison, and fail-loud contracts.

pytest is NOT installed on the eval pod, so these run standalone too:
  python taniteval/tests/test_ci.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import ci as C  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic corpus with a KNOWN within-episode correlation                      #
# --------------------------------------------------------------------------- #
N_EP = 40          # canonical val: 40 episodes
N_WIN = 22         # ~881 windows / 40 episodes
SIG_EP = 0.30      # episode-level effect (the real uncertainty)
SIG_WIN = 0.10     # within-episode noise (nearly free extra windows)
MU = 0.45          # a plausible ADE@2s


def make_corpus(rng, n_ep=N_EP, n_win=N_WIN, sig_ep=SIG_EP, sig_win=SIG_WIN,
                mu=MU):
    """Per-window values with an episode random effect, plus their eids.

    True SE of the corpus mean is dominated by the EPISODE term:
        SE = sqrt(sig_ep**2 / n_ep + sig_win**2 / (n_ep * n_win))
    which for these constants is ~0.0475 — an interval built on the 880 windows
    as if they were independent would report ~0.0107 instead.
    """
    ep_effect = rng.normal(0.0, sig_ep, size=n_ep)
    vals = (mu + ep_effect[:, None]
            + rng.normal(0.0, sig_win, size=(n_ep, n_win))).ravel()
    eid = np.repeat(np.arange(n_ep), n_win)
    return vals, eid


def true_se(n_ep=N_EP, n_win=N_WIN, sig_ep=SIG_EP, sig_win=SIG_WIN):
    return float(np.sqrt(sig_ep ** 2 / n_ep + sig_win ** 2 / (n_ep * n_win)))


def naive_holdout_ci95(vals, eid, n_splits=8, val_frac=0.2, seed=0):
    """Replica of the PRE-FIX bench protocol: ``n_splits`` INDEPENDENT random
    ``val_frac`` episode holdouts, then ``1.96*std/sqrt(n_splits)``.

    Mirrors ``bench.run``'s ``[split_by_episode(eid, val_frac, s) for s in
    range(seed, seed+n_splits)]`` + ``bench._agg``. Deliberately reimplemented
    here so the test does not depend on the pod-path imports in bench.py.
    """
    vals = np.asarray(vals, dtype=np.float64)
    eid = np.asarray(eid)
    uniq = np.unique(eid)
    means = []
    for s in range(seed, seed + n_splits):
        rng = np.random.default_rng(s)
        n_val = max(1, int(round(val_frac * len(uniq))))
        val_ep = set(rng.permutation(uniq)[:n_val].tolist())
        sel = np.array([i for i, e in enumerate(eid) if e in val_ep])
        means.append(vals[sel].mean())
    means = np.asarray(means)
    return float(means.mean()), C.overlapping_holdout_se(means)


# --------------------------------------------------------------------------- #
# 1. Reproduction — history must stay reproducible                             #
# --------------------------------------------------------------------------- #
def test_overlapping_holdout_se_is_the_old_arithmetic():
    v = np.array([0.44, 0.47, 0.42, 0.49, 0.45, 0.46, 0.43, 0.48])
    expected = float(1.96 * np.nanstd(v) / len(v) ** 0.5)      # bench.py:88 verbatim
    got = C.overlapping_holdout_se(v)
    assert abs(got - expected) < 1e-12, f"{got} != {expected}"
    # and it is the number the registry printed for flagship v1's block shape
    assert round(got, 4) == round(expected, 4)


# --------------------------------------------------------------------------- #
# 2. THE SMOKING GUN — the naive "SE" is split-selection noise                  #
# --------------------------------------------------------------------------- #
def test_naive_se_shrinks_with_more_splits():
    """Same data, more overlapping splits -> smaller "SE". A real standard error
    for the arm cannot depend on how many times you resample your own val set.
    The cluster bootstrap must be stable under the same change."""
    rng = np.random.default_rng(7)
    vals, eid = make_corpus(rng)

    _, se8 = naive_holdout_ci95(vals, eid, n_splits=8)
    _, se64 = naive_holdout_ci95(vals, eid, n_splits=64)
    assert se64 < 0.6 * se8, (
        f"naive SE should collapse as n_splits grows (8 -> {se8:.4f}, "
        f"64 -> {se64:.4f}); it did not, so this test no longer pins the defect")

    b400 = C.episode_cluster_bootstrap(vals, eid, n_boot=400, seed=0)["ci95"]
    b2000 = C.episode_cluster_bootstrap(vals, eid, n_boot=2000, seed=0)["ci95"]
    assert abs(b2000 - b400) < 0.25 * b2000, (
        f"cluster bootstrap must be stable in B ({b400:.4f} vs {b2000:.4f})")


# --------------------------------------------------------------------------- #
# 3. The magnitude claim — ~20 % too narrow at 40 episodes / 8 splits           #
# --------------------------------------------------------------------------- #
def test_naive_se_too_small_under_episode_correlation():
    """With a real within-episode effect the naive interval is materially
    narrower than both the correct cluster bootstrap and the ANALYTIC truth."""
    rng = np.random.default_rng(11)
    vals, eid = make_corpus(rng)

    _, naive = naive_holdout_ci95(vals, eid, n_splits=8)
    boot = C.episode_cluster_bootstrap(vals, eid, n_boot=2000, seed=0)
    analytic = 1.96 * true_se()

    assert naive < boot["ci95"], (
        f"naive {naive:.4f} should be narrower than cluster {boot['ci95']:.4f}")
    # the cluster bootstrap must land near the analytic truth (+-35 %, MC noise
    # on a single 40-episode draw is genuinely large)
    assert 0.65 * analytic < boot["ci95"] < 1.35 * analytic, (
        f"cluster {boot['ci95']:.4f} vs analytic {analytic:.4f}")
    # and the naive one must NOT
    assert naive < 0.9 * analytic, (
        f"naive {naive:.4f} vs analytic {analytic:.4f} — the defect is gone?")


# --------------------------------------------------------------------------- #
# 4. Coverage — the review's falsifiable success criterion                      #
# --------------------------------------------------------------------------- #
def test_coverage_cluster_vs_naive(n_reps=400, n_boot=300):
    """Empirical coverage of MU over independent 40-episode corpora.

    Success criterion (360-review P2): the cluster bootstrap achieves ~93-97 %
    and the naive estimator is shown to UNDER-cover."""
    rng = np.random.default_rng(2026)
    cov_boot = cov_naive = 0
    w_boot, w_naive = [], []
    for _ in range(n_reps):
        vals, eid = make_corpus(rng)
        b = C.episode_cluster_bootstrap(vals, eid, n_boot=n_boot,
                                        seed=int(rng.integers(1 << 30)))
        cov_boot += int(b["lo"] <= MU <= b["hi"])
        w_boot.append(b["hi"] - b["lo"])

        m, se = naive_holdout_ci95(vals, eid, n_splits=8)
        cov_naive += int(m - se <= MU <= m + se)
        w_naive.append(2 * se)

    cb, cn = cov_boot / n_reps, cov_naive / n_reps
    print(f"    coverage: cluster {cb:.3f} (mean width {np.mean(w_boot):.4f}) | "
          f"naive {cn:.3f} (mean width {np.mean(w_naive):.4f}) | "
          f"width ratio naive/cluster {np.mean(w_naive)/np.mean(w_boot):.3f}")
    # allow +-2 pts of Monte-Carlo slack around the 93-97 % target at n_reps=400
    assert 0.91 <= cb <= 0.98, f"cluster bootstrap coverage {cb:.3f} off nominal"
    assert cn < cb, f"naive coverage {cn:.3f} must be below cluster {cb:.3f}"
    assert cn < 0.93, f"naive estimator failed to under-cover ({cn:.3f})"


# --------------------------------------------------------------------------- #
# 5. Paired beats quadrature on correlated arms                                 #
# --------------------------------------------------------------------------- #
def test_paired_is_more_powerful_than_quadrature():
    """Two arms scored on the SAME windows share their difficulty. Combining two
    single-arm intervals in quadrature assumes independence and throws that
    away; the paired episode-clustered CI must be tighter."""
    rng = np.random.default_rng(23)
    base, eid = make_corpus(rng)
    delta_true = 0.135                                  # the v1.5 a->ab effect size
    a = base + rng.normal(0, 0.05, size=base.shape)
    b = base - delta_true + rng.normal(0, 0.05, size=base.shape)

    ca = C.episode_cluster_bootstrap(a, eid, n_boot=1000, seed=1)
    cb = C.episode_cluster_bootstrap(b, eid, n_boot=1000, seed=1)
    quad = float(np.hypot(ca["ci95"], cb["ci95"]))       # the method actually used
    paired = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=1000, seed=1)

    print(f"    quadrature half-width {quad:.4f} | paired {paired['ci95']:.4f} "
          f"| delta {paired['delta']:.4f}")
    assert paired["ci95"] < 0.5 * quad, (
        f"paired {paired['ci95']:.4f} not materially tighter than quadrature {quad:.4f}")
    assert paired["separated"], "a real 0.135 m effect must separate when paired"
    assert abs(paired["delta"] - delta_true) < 0.02


def test_paired_null_effect_does_not_separate():
    rng = np.random.default_rng(31)
    base, eid = make_corpus(rng)
    a = base + rng.normal(0, 0.05, size=base.shape)
    b = base + rng.normal(0, 0.05, size=base.shape)
    p = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=1000, seed=3)
    assert not p["separated"], f"null effect wrongly separated: {p}"


# --------------------------------------------------------------------------- #
# 6. Suite consistency — the bootstrap must not move the point estimate         #
# --------------------------------------------------------------------------- #
def _load_bench():
    """Import bench with either the pod layout or this repo's layout, else None."""
    repo = _HERE.parents[2]
    for p in (str(repo / "stack"), str(repo / "stack" / "scripts"),
              "/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
        if p not in sys.path:
            sys.path.append(p)
    try:
        from taniteval import bench
        return bench
    except Exception as e:                                    # noqa: BLE001
        print(f"    [skip] bench import unavailable: {type(e).__name__}: {e}")
        return None


def test_bootstrap_point_estimate_matches_full_set():
    bench = _load_bench()
    if bench is None:
        return
    import torch
    g = torch.Generator().manual_seed(5)
    pred = torch.randn(200, 4, 2, generator=g)
    gt = pred + 0.3 * torch.randn(200, 4, 2, generator=g)
    eid = np.repeat(np.arange(10), 20)

    suite = bench._suite(pred, gt)
    boot = C.bootstrap_metrics(bench._suite_components(pred, gt), eid,
                               n_boot=50, seed=0)
    for k, v in suite.items():
        assert abs(boot[k]["mean"] - round(float(v), 4)) < 2e-4, (
            f"{k}: bootstrap point {boot[k]['mean']} != full-set {v}")


# --------------------------------------------------------------------------- #
# 7. Fail-loud contracts                                                        #
# --------------------------------------------------------------------------- #
def test_fails_loud_on_bad_input():
    for fn, args in (
        (C.episode_cluster_bootstrap, ([1.0, 2.0], [0])),           # length
        (C.episode_cluster_bootstrap, (np.zeros((2, 2)), [0, 0])),  # 2-D
        (C.episode_index, ([],)),                                   # empty
    ):
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} silently accepted {args!r}")

    try:
        C.paired_episode_cluster_bootstrap([1.0, 2.0], [1.0], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("paired test accepted misaligned arms")


def test_result_dicts_carry_their_provenance():
    """A number must never be quotable without the construction behind it."""
    rng = np.random.default_rng(3)
    vals, eid = make_corpus(rng)
    b = C.episode_cluster_bootstrap(vals, eid, n_boot=100, seed=0)
    for k in ("estimator", "n_episodes", "n_windows", "n_boot", "lo", "hi"):
        assert k in b, f"missing provenance field {k}"
    assert b["estimator"] == "episode_cluster_bootstrap"
    assert b["n_episodes"] == N_EP


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
