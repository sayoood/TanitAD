"""Tests for ``taniteval/selgap.py`` — the generalised oracle-vs-selected gap.

Pins, on synthetic data with a KNOWN answer:
  * perfect selector      -> gap 0 and a CI that covers 0
  * anti-selector         -> gap == mean(max - min) exactly
  * episode clustering    -> the episode-cluster CI on the gap is WIDER than a
      naive iid (per-window) bootstrap when the gap is episode-correlated —
      i.e. the module inherits ci.py's estimator, not a per-window one
  * top-k monotonicity    -> oracle_top4 >= oracle_top8 >= oracle_top16 >= oracle
      (both rankings: static corpus-mean and selector-scores)
  * torch input path      -> torch tensors give bit-identical results to numpy

pytest is NOT installed on the eval pod, so these run standalone too:
  python taniteval/tests/test_selgap.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import selgap as SG  # noqa: E402


def make_fan(rng, n=200, c=8, n_ep=10):
    """Random positive fan errors + episode ids (contiguous blocks)."""
    fan = rng.uniform(0.1, 2.0, size=(n, c))
    eid = np.repeat(np.arange(n_ep), n // n_ep)
    return fan, eid


# --------------------------------------------------------------------------- #
# 1. Perfect selector — gap 0, CI covering 0                                    #
# --------------------------------------------------------------------------- #
def test_perfect_selector_gap_zero_ci_covers_zero():
    rng = np.random.default_rng(0)
    fan, eid = make_fan(rng)
    sel = fan.argmin(axis=1)                       # the oracle pick, every window
    r = SG.selgap(fan, sel, eid, n_boot=300, seed=0)
    assert r["selected"] == r["oracle"], r
    assert r["gap"] == 0.0, r
    assert r["gap_frac"] == 0.0, r
    ci = r["gap_ci"]
    assert ci["lo"] <= 0.0 <= ci["hi"], f"CI must cover 0: {ci}"
    assert ci["estimator"] == "episode_cluster_bootstrap", ci
    # the pick is always rank 0 (strictly-better count), pct 0
    assert r["sel_rank_mean"] == 0.0 and r["sel_rank_pct_median"] == 0.0, r


# --------------------------------------------------------------------------- #
# 2. Anti-selector — gap == mean(max - min)                                     #
# --------------------------------------------------------------------------- #
def test_anti_selector_gap_is_mean_max_minus_min():
    rng = np.random.default_rng(1)
    fan, eid = make_fan(rng)
    sel = fan.argmax(axis=1)                       # the worst pick, every window
    r = SG.selgap(fan, sel, eid, n_boot=100, seed=0)
    expected = float((fan.max(axis=1) - fan.min(axis=1)).mean())
    assert abs(r["gap"] - expected) <= 1.01e-4, (   # 4 dp display rounding only
        f"gap {r['gap']} != mean(max-min) {expected}")
    # with continuous errors the worst pick has C-1 strictly-better candidates
    assert r["sel_rank_pct_mean"] == 1.0 and r["sel_rank_pct_median"] == 1.0, r


# --------------------------------------------------------------------------- #
# 3. Episode clustering — cluster CI wider than a naive iid bootstrap           #
# --------------------------------------------------------------------------- #
def _naive_iid_bootstrap_ci(gap_pw, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI resampling WINDOWS iid — the estimator selgap must NOT
    use, reimplemented here as the comparison arm."""
    rng = np.random.default_rng(seed)
    n = gap_pw.size
    boots = np.array([gap_pw[rng.integers(0, n, size=n)].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def test_cluster_ci_wider_than_iid_on_episode_correlated_gap():
    """2 episodes with DIFFERENT mean gaps (0.2 vs 0.8), tight within-episode
    noise: 400 windows look informative iid, but there are only 2 independent
    units. The episode-cluster CI must be materially wider than the iid one."""
    rng = np.random.default_rng(2)
    n_per, c = 200, 4
    gap_a = 0.2 + rng.normal(0, 0.02, size=n_per)      # episode A's gap level
    gap_b = 0.8 + rng.normal(0, 0.02, size=n_per)      # episode B's gap level
    gap_pw = np.clip(np.concatenate([gap_a, gap_b]), 0.01, None)
    n = 2 * n_per
    # build a fan that REALISES exactly this per-window gap: candidate 0 is the
    # oracle (error 0.1), candidate 1 is oracle + gap, the rest are far worse.
    fan = np.full((n, c), 5.0)
    fan[:, 0] = 0.1
    fan[:, 1] = 0.1 + gap_pw
    sel = np.ones(n, dtype=int)                        # always pick candidate 1
    eid = np.repeat(["epA", "epB"], n_per)

    r = SG.selgap(fan, sel, eid, n_boot=2000, seed=0)
    assert abs(r["gap"] - gap_pw.mean()) <= 1.01e-4, r
    cl_lo, cl_hi = r["gap_ci"]["lo"], r["gap_ci"]["hi"]
    iid_lo, iid_hi = _naive_iid_bootstrap_ci(gap_pw, n_boot=2000, seed=0)
    cl_w, iid_w = cl_hi - cl_lo, iid_hi - iid_lo
    print(f"    cluster width {cl_w:.4f} vs iid width {iid_w:.4f} "
          f"(ratio {cl_w / iid_w:.1f}x)")
    assert cl_w > 3.0 * iid_w, (
        f"episode-cluster CI ({cl_w:.4f}) must be much wider than iid "
        f"({iid_w:.4f}) on episode-correlated gaps — selgap is not using the "
        f"cluster estimator?")
    assert r["gap_ci"]["n_episodes"] == 2, r["gap_ci"]


# --------------------------------------------------------------------------- #
# 4. Top-k monotonicity                                                         #
# --------------------------------------------------------------------------- #
def test_topk_oracle_monotone_static_ranking():
    rng = np.random.default_rng(3)
    fan, eid = make_fan(rng, n=300, c=32, n_ep=10)
    sel = rng.integers(0, 32, size=300)
    r = SG.selgap(fan, sel, eid, n_boot=50, seed=0)
    assert r["topk_ranking"] == "corpus_mean_error", r
    assert (r["oracle_top4"] >= r["oracle_top8"]
            >= r["oracle_top16"] >= r["oracle"]), (
        f"top-k oracle must be monotone: {r['oracle_top4']} / "
        f"{r['oracle_top8']} / {r['oracle_top16']} / {r['oracle']}")
    # full-fan top-k (k >= C) must equal the full oracle
    r32 = SG.selgap(fan, sel, eid, n_boot=50, seed=0, topk=(4, 32, 64))
    assert r32["oracle_top32"] == r32["oracle"] == r32["oracle_top64"], r32


def test_topk_oracle_monotone_selector_scores():
    rng = np.random.default_rng(4)
    fan, eid = make_fan(rng, n=300, c=32, n_ep=10)
    scores = -fan + rng.normal(0, 0.5, size=fan.shape)   # noisy selector
    sel = scores.argmax(axis=1)
    r = SG.selgap(fan, sel, eid, n_boot=50, seed=0, scores=scores)
    assert r["topk_ranking"] == "selector_scores", r
    assert (r["oracle_top4"] >= r["oracle_top8"]
            >= r["oracle_top16"] >= r["oracle"]), r
    # top-1 under the selector's own ranking IS the selected error
    r1 = SG.selgap(fan, sel, eid, n_boot=50, seed=0, scores=scores, topk=(1,))
    assert r1["oracle_top1"] == r1["selected"], r1


# --------------------------------------------------------------------------- #
# 5. Torch input path                                                           #
# --------------------------------------------------------------------------- #
def test_torch_inputs_match_numpy():
    try:
        import torch
    except ImportError:
        print("    [skip] torch not installed on this host")
        return
    rng = np.random.default_rng(5)
    fan, eid = make_fan(rng, n=120, c=16, n_ep=6)
    sel = rng.integers(0, 16, size=120)
    r_np = SG.selgap(fan, sel, eid, n_boot=200, seed=0)
    r_th = SG.selgap(torch.as_tensor(fan), torch.as_tensor(sel), eid,
                     n_boot=200, seed=0)
    for k in ("selected", "oracle", "gap", "gap_frac", "oracle_top4",
              "oracle_top8", "oracle_top16", "sel_rank_mean",
              "sel_rank_pct_mean", "sel_rank_pct_median"):
        assert r_np[k] == r_th[k], f"{k}: numpy {r_np[k]} != torch {r_th[k]}"
    assert r_np["gap_ci"] == r_th["gap_ci"], "CI must be bit-identical"


# --------------------------------------------------------------------------- #
# 6. Report block + fail-loud contracts                                         #
# --------------------------------------------------------------------------- #
def test_report_block_carries_estimator_and_level():
    rng = np.random.default_rng(6)
    fan, eid = make_fan(rng)
    sel = fan.argmin(axis=1)
    txt = SG.selgap_report(fan, sel, eid, n_boot=100, seed=0, level="operative")
    assert "sel_gap [operative]" in txt, txt
    assert "episode_cluster_bootstrap" in txt, txt
    assert "never pool across levels" in txt, txt
    assert "overlapping_holdout_se" in txt, txt          # the ⛔ line names it


def test_fails_loud_on_bad_input():
    fan = np.ones((4, 3))
    eid = ["a", "a", "b", "b"]
    for args in (
        (np.ones(4), np.zeros(4, int), eid),             # 1-D fan
        (fan, np.zeros(3, int), eid),                    # sel length
        (fan, np.array([0, 1, 2, 3]), eid),              # sel out of range
        (fan, np.zeros(4, int), ["a", "b"]),             # eid length
    ):
        try:
            SG.selgap(*args, n_boot=10)
        except ValueError:
            continue
        raise AssertionError(f"selgap silently accepted {args!r}")
    try:
        SG.selgap(fan, np.zeros(4, int), eid, n_boot=10,
                  scores=np.ones((4, 2)))                # scores shape
    except ValueError:
        pass
    else:
        raise AssertionError("selgap accepted mis-shaped scores")


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
