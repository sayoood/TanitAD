"""``reference_reached_at`` robustness — the 2026-07-21 "step 450" defect.

The old rule was a 3-point rolling median. ``g_op_fwd_ade_m`` is a PER-BATCH
(B=16) train metric whose adjacent logged rows swing ~2x, and a 3-point median
does not survive that: on v1's raw log it first crossed 0.4101 at step 450,
while v1's bucket means only reach ~0.41 in the 2k-4k range. The reported figure
was ~5x too early and fed a "~23x more step-efficient" claim into
MODEL_REGISTRY.md 1.4 (retracted).

Every test below is built so that the RETIRED rule fails it — each adversarial
case asserts the 3-point median is fooled before asserting the shipped function
is not, so the fixture can never silently stop being adversarial.

Source: Research/2026-07-21-flagship-v3enc-postmortem.md 7.1.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gate as rg                                          # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _rows(values, metric="g_op_fwd_ade_m", step0=50, every=50):
    return [{"step": step0 + i * every, metric: float(x)}
            for i, x in enumerate(values)]


def _old_rule_step(rows, metric, value):
    """The RETIRED 3-point rolling median, kept verbatim as the adversary."""
    s, v = rg.series(rows, metric)
    med = np.array([np.median(v[max(0, i - 1):i + 2]) for i in range(v.size)])
    hit = np.flatnonzero(med <= value)
    return int(s[hit[0]]) if hit.size else None


# v1's REAL logged g_op_fwd_ade_m rows at steps 300-550 (raw train_log.jsonl,
# flagship4b-speedjerk-30k). Two isolated draws (0.404, 0.384) sit below the
# 0.4101 target while the level around them is ~0.7.
V1_REAL_300_550 = [0.758, 0.616, 0.404, 0.687, 0.384, 0.816]
TARGET = 0.4101


def _v1_like_series(n_tail=60, seed=0):
    """v1's real noisy early rows, then a level that genuinely reaches TARGET
    only much later — i.e. the true crossing is late, the dips are early."""
    gen = np.random.default_rng(seed)
    head = [0.95, 0.88, 0.83] + V1_REAL_300_550 + [0.72, 0.69, 0.66, 0.61]
    # true level decays 0.60 -> 0.22; multiplicative per-batch noise up to ~2x
    lvl = np.linspace(0.60, 0.22, n_tail)
    tail = lvl * gen.uniform(0.62, 1.60, n_tail)
    return _rows(list(head) + list(tail))


# --------------------------------------------------------------------------- #
# the defect itself                                                            #
# --------------------------------------------------------------------------- #
def test_isolated_dips_do_not_fabricate_an_early_crossing():
    """THE regression test: v1's real 300-550 rows must not report step 450."""
    rows = _rows(V1_REAL_300_550, step0=300)

    assert _old_rule_step(rows, "g_op_fwd_ade_m", TARGET) == 450, \
        "fixture is no longer adversarial — the retired rule must be fooled here"

    out = rg.reference_reached_at(rows, "g_op_fwd_ade_m", TARGET)
    assert out["reached_at_step"] is None, (
        "two isolated sub-target draws among ~0.7 rows are not a crossing; "
        f"got step {out['reached_at_step']}")


def test_noisy_series_crossing_is_not_reported_before_the_true_level():
    """A 2x-noisy series whose TRUE level reaches the target late."""
    rows = _v1_like_series()
    metric = "g_op_fwd_ade_m"
    old = _old_rule_step(rows, metric, TARGET)
    out = rg.reference_reached_at(rows, metric, TARGET)

    assert old is not None and old <= 450, \
        "fixture is no longer adversarial — the retired rule must fire early"
    assert out["reached_at_step"] is not None
    assert out["reached_at_step"] > old, (
        "robust rule must not fire as early as the 3-point median "
        f"({out['reached_at_step']} vs {old})")

    # the reported step must sit where the TRUE level is at/below target,
    # not where a lucky draw was
    s, v = rg.series(rows, metric)
    i = int(np.flatnonzero(s == out["reached_at_step"])[0])
    assert float(np.mean(v[i:i + 10])) <= TARGET * 1.15


def test_k_consecutive_is_actually_enforced():
    """k-1 sub-target rows never trigger; k in a row do."""
    metric = "g_op_fwd_ade_m"
    k = rg.REACHED_K
    hi, lo = 0.90, 0.10

    near = _rows([hi] * 4 + [lo] * (k - 1) + [hi] * 12)
    assert rg.reference_reached_at(near, metric, TARGET)["reached_at_step"] is None

    just = _rows([hi] * 4 + [lo] * k + [hi] * 12)
    got = rg.reference_reached_at(just, metric, TARGET)
    assert got["reached_at_step"] == 50 + 4 * 50          # first row of the run


def test_clean_monotone_series_still_reports_the_obvious_crossing():
    """Robustness must not break the easy case."""
    metric = "g_op_fwd_ade_m"
    vals = list(np.linspace(1.2, 0.05, 40))
    out = rg.reference_reached_at(_rows(vals), metric, TARGET)
    s, v = rg.series(_rows(vals), metric)
    first_true = int(s[np.flatnonzero(v <= TARGET)[0]])
    assert out["reached_at_step"] == first_true
    assert out["rules_agree"] is True


# --------------------------------------------------------------------------- #
# the statistic can never be quoted without its rule                           #
# --------------------------------------------------------------------------- #
def test_contract_keys_are_preserved_and_rule_is_explicit():
    out = rg.reference_reached_at(_v1_like_series(), "g_op_fwd_ade_m", TARGET)

    for key in ("target_value", "reached_at_step", "reference_final", "smoothing"):
        assert key in out, f"caller/JSON contract dropped {key!r}"

    for key in ("estimator", "k_consecutive", "bucket_steps",
                "reached_in_bucket", "rules_agree"):
        assert key in out, f"provenance field {key!r} missing"

    rule = out["estimator"]
    assert isinstance(rule, str) and len(rule) > 40
    assert "CONSECUTIVE" in rule and "MEAN" in rule
    assert "NOT a rolling median" in rule, \
        "estimator must explicitly disclaim the retired rule, not just omit it"
    assert out["smoothing"] == rule, "both provenance keys must state one rule"
    assert out["k_consecutive"] == rg.REACHED_K


def test_bucket_view_is_an_interval_not_a_point():
    out = rg.reference_reached_at(_v1_like_series(), "g_op_fwd_ade_m", TARGET)
    lo, hi = out["reached_in_bucket"]
    assert hi - lo == rg.REACHED_BUCKET
    assert out["bucket_mean_at_crossing"] <= TARGET
    assert lo <= out["reached_at_step"] <= hi or not out["rules_agree"]


def test_reference_final_is_a_bucket_mean_not_a_single_last_row():
    """`s.max()` rarely lands on a bucket edge; the final level must still be an
    average. Last row is a deliberate outlier — it must not become the answer."""
    metric = "g_op_fwd_ade_m"
    vals = [0.9] * 10 + [0.20, 0.22, 0.18, 0.21, 0.19, 0.20, 5.0]
    rows = _rows(vals)                       # steps 50..850, bucket edge is 1000
    out = rg.reference_reached_at(rows, metric, TARGET)
    assert out["reference_final"] != pytest.approx(5.0), \
        "fell back to the single last row"
    assert out["reference_final"] == pytest.approx(float(np.mean(vals)), abs=1e-4)


def test_rules_agree_flag_is_false_when_the_two_views_diverge():
    """A late isolated k-run inside a bucket that never means below target."""
    metric = "g_op_fwd_ade_m"
    k = rg.REACHED_K
    vals = [0.9] * 20 + [0.05] * k + [0.9] * 20
    out = rg.reference_reached_at(_rows(vals), metric, TARGET, bucket=100)
    assert out["reached_at_step"] is not None
    assert out["rules_agree"] is False


# --------------------------------------------------------------------------- #
# degenerate input                                                             #
# --------------------------------------------------------------------------- #
def test_too_few_points_is_explicit_and_still_carries_the_rule():
    out = rg.reference_reached_at(_rows([0.2, 0.2]), "g_op_fwd_ade_m", TARGET)
    assert out["reached_at_step"] is None
    assert "too few" in out["note"]
    assert "estimator" in out and "smoothing" in out


def test_target_never_reached_returns_none():
    out = rg.reference_reached_at(_rows([2.0] * 30), "g_op_fwd_ade_m", TARGET)
    assert out["reached_at_step"] is None
    assert out["reached_in_bucket"] is None


@pytest.mark.parametrize("seed", range(8))
def test_robust_rule_never_beats_the_median_to_the_crossing(seed):
    """Across seeds the robust rule is never EARLIER than the retired one."""
    rows = _v1_like_series(seed=seed)
    metric = "g_op_fwd_ade_m"
    old = _old_rule_step(rows, metric, TARGET)
    new = rg.reference_reached_at(rows, metric, TARGET)["reached_at_step"]
    if old is not None and new is not None:
        assert new >= old
