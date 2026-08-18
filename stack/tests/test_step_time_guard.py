"""Tests for the step-time abort criterion — pinned in BOTH directions.

⛔ C112: the criterion this replaces was **structurally unable to fire**.
``train_v6_staged.py``'s ``step_s`` is a cumulative mean since process start, so
at the intended trip point it never reaches the threshold at any duration, and a
2-hour pilot would have reported "safe" no matter what happened.

⚠️ C95/C97: this programme built a **rejects-everything** guard and a
**passes-everything** guard within one day. So every test here comes in a pair —
*it fires on a real slowdown* **and** *it stays quiet on measured normal
variation*. A guard tested in one direction only is not tested.

The "normal variation" side is not invented: it is the **live v6F-SW-30k log**,
banked in the repo and named in ``LIVE_LOG`` below.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from step_time_guard import (  # noqa: E402
    INSUFFICIENT, OK, TRIP, check, load_rows, marginal_rates,
    steps_this_process,
)

REPO = Path(__file__).resolve().parents[2]
LIVE_LOG = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
            "Implementation" / "incoming" / "2026-08-18-o2-live-and-ridge-reread" /
            "raw" / "v6F-SW-30k_train_log.jsonl")

#: MEASURED from LIVE_LOG, segment 1, n > 900 (110 marginal intervals).
STEADY_MEDIAN_S = 26.3594


def cum_rows(rates: list[float], *, start_n: int = 0, start_elapsed: float = 0.0,
             log_every: int = 50, start_step: int = 0) -> list[dict]:
    """Synthesise log rows carrying ONLY the cumulative field.

    This is the shape of every banked log written before ``step_s_interval``
    existed, so the recovery path is exercised on the format that actually
    matters.
    """
    rows, n, elapsed, step = [], start_n, start_elapsed, start_step
    for rate in rates:
        n += log_every
        step += log_every
        elapsed += rate * log_every
        rows.append({"step": step, "step_s": elapsed / n,
                     "steps_this_process": n})
    return rows


# ==========================================================================
# 1. THE C112 PROOF — the old criterion cannot fire, the new one can
# ==========================================================================
def test_the_cumulative_field_cannot_fire_but_the_marginal_one_does():
    """⛔ The headline. Same catastrophe, two instruments, opposite answers.

    A process 6300 steps in at 26.4749 s/step suddenly runs at **40 s/step**
    (+51 %, catastrophic) for the ~2 h a pilot lasts. The cumulative mean is so
    heavily damped by 166,791 s of history that it stays well under a +5 %
    threshold; the marginal rate trips on the second interval.
    """
    base_n, base_elapsed = 6300, 166791.9
    threshold = STEADY_MEDIAN_S * 1.05

    # 2 hours at 40 s/step = 180 steps = 3 logged intervals of 50.
    rows = cum_rows([40.0] * 3, start_n=base_n, start_elapsed=base_elapsed,
                    start_step=base_n)

    # -- the OLD criterion, applied exactly as it was written --------------
    worst_cumulative = max(r["step_s"] for r in rows)
    assert worst_cumulative < threshold, (
        "the premise of this test is that the cumulative mean stays under the "
        "trip point during a catastrophic slowdown")
    assert worst_cumulative == pytest.approx(26.789, abs=0.01)

    # -- the NEW criterion, on the same rows --------------------------------
    verdict = check(marginal_rates(rows), baseline_s=STEADY_MEDIAN_S,
                    warmup_steps=0)
    assert verdict["verdict"] == TRIP
    assert verdict["trip_run"][0]["s_per_step"] == pytest.approx(40.0)
    assert verdict["max_excess_frac"] == pytest.approx(0.5175, abs=1e-3)


def test_the_cumulative_mean_needs_hours_to_reach_the_trip_point():
    """⛔ Quantifies "structurally unable to fire" rather than asserting it.

    At the intended +5 % trip point the cumulative mean NEVER gets there — its
    asymptote IS that rate. Even at 40 s/step it needs hours, not the ~2 h a
    pilot runs.
    """
    base_n, base_elapsed = 6300, 166791.9
    threshold = STEADY_MEDIAN_S * 1.05

    # (a) at the intended trip rate the asymptote is BELOW the threshold, so no
    #     number of steps suffices — 10,000 intervals (~139 days) is a proxy.
    at_trip = cum_rows([threshold] * 10_000, start_n=base_n,
                       start_elapsed=base_elapsed, start_step=base_n)
    assert max(r["step_s"] for r in at_trip) < threshold

    # (b) at a catastrophic 40 s/step it takes >6 h of intervals to cross.
    k = next(i for i, r in enumerate(
        cum_rows([40.0] * 400, start_n=base_n, start_elapsed=base_elapsed,
                 start_step=base_n), start=1) if r["step_s"] >= threshold)
    assert k > 12, k
    assert k * 50 * 40.0 / 3600 > 6.0, "should need >6 hours of wall clock"


# ==========================================================================
# 2. THE OTHER DIRECTION — measured normal variation must stay quiet
# ==========================================================================
@pytest.mark.skipif(not LIVE_LOG.is_file(), reason="banked live log absent")
def test_real_steady_state_does_not_trip_at_five_percent():
    """The passes-everything half of the C95/C97 pair, on REAL data.

    110 marginal intervals of the live run, worst case +2.589 % over the
    steady median. A +5 % guard must stay silent on all of them.
    """
    rates = [r for r in marginal_rates(load_rows(LIVE_LOG))
             if r["segment"] == 1]
    verdict = check(rates, baseline_s=STEADY_MEDIAN_S, tol_frac=0.05)
    assert verdict["verdict"] == OK, verdict["note"]
    assert verdict["n_admissible"] == 110
    assert verdict["n_excluded_warmup"] == 17
    # the measured worst case — this is the number the tolerance is set against
    assert verdict["max_excess_frac"] == pytest.approx(0.02589, abs=5e-4)


@pytest.mark.skipif(not LIVE_LOG.is_file(), reason="banked live log absent")
def test_where_the_warmup_exclusion_becomes_load_bearing():
    """⛔ C95's failure mode, located precisely rather than asserted loosely.

    ⚠️ A first draft of this file claimed +3 % was a rejects-everything guard
    outright. It is not, and the test caught it. The true statement is narrower
    and more useful:

    * +5 % is OK on this run **with or without** the exclusion — at that
      tolerance the exclusion buys robustness, it is not load-bearing.
    * +3 % is OK on steady state but TRIPS on the warm-up ⇒ **without the
      exclusion it fires on every resume**, on 0.41 pp of margin with it.
    """
    rates = [r for r in marginal_rates(load_rows(LIVE_LOG))
             if r["segment"] == 1]

    def verdict(tol, warmup):
        return check(rates, baseline_s=STEADY_MEDIAN_S, tol_frac=tol,
                     warmup_steps=warmup)["verdict"]

    assert verdict(0.05, 900) == OK
    assert verdict(0.05, 0) == OK          # warm-up does NOT defeat +5 %
    assert verdict(0.03, 900) == OK
    assert verdict(0.03, 0) == TRIP        # ⛔ it DOES defeat +3 %

    # and the margin a +3 % guard would run on, which is why +5 % is the setting
    steady_max_excess = check(
        rates, baseline_s=STEADY_MEDIAN_S)["max_excess_frac"]
    assert 0.03 - steady_max_excess == pytest.approx(0.0041, abs=5e-4)


@pytest.mark.skipif(not LIVE_LOG.is_file(), reason="banked live log absent")
def test_the_live_log_recovers_two_segments_and_the_warmup_transient():
    """``n`` resets at the resume; the first ~900 steps after it run +3.2 %.

    MEASURED: warm-up median 27.1252 vs steady 26.3594 — sustained across 17
    CONSECUTIVE rows, which is why persistence alone cannot exclude it.
    """
    import statistics as st

    rates = marginal_rates(load_rows(LIVE_LOG))
    assert {r["segment"] for r in rates} == {0, 1}
    # the fallback path is what reads this log: it has no first-class n key
    assert {r["source"] for r in rates} == {"derived", "unusable"}

    warm = [r["s_per_step"] for r in rates
            if r["segment"] == 1 and r["s_per_step"] is not None
            and r["n"] <= 900]
    steady = [r["s_per_step"] for r in rates
              if r["segment"] == 1 and r["s_per_step"] is not None
              and r["n"] > 900]
    assert len(warm) == 17 and len(steady) == 110
    assert st.median(warm) == pytest.approx(27.1252, abs=1e-3)
    assert st.median(steady) == pytest.approx(STEADY_MEDIAN_S, abs=1e-3)
    # the warm-up alone clears a +3 % guard — a persistence-only rule is not
    # enough, only knowing where the process restarted is
    assert max(warm) / st.median(steady) - 1 > 0.03


def test_warmup_is_excluded_but_the_same_slowdown_later_trips():
    """The exclusion must be a WINDOW, not a way to ignore a real slowdown."""
    slow = [30.0] * 6                      # +13.8 %, well over threshold

    inside = cum_rows(slow, start_n=0, start_step=0)          # n = 50..300
    assert check(marginal_rates(inside), baseline_s=STEADY_MEDIAN_S
                 )["verdict"] == INSUFFICIENT

    outside = cum_rows(slow, start_n=5000, start_elapsed=5000 * 26.36,
                       start_step=5000)                       # n = 5050..5300
    assert check(marginal_rates(outside), baseline_s=STEADY_MEDIAN_S
                 )["verdict"] == TRIP


# ==========================================================================
# 3. THE ARITHMETIC — recovery from banked logs must be EXACT
# ==========================================================================
def test_first_differencing_recovers_the_marginal_rate_exactly():
    """The cumulative series is invertible, so banked logs are not lost."""
    truth = [26.3, 26.4, 31.7, 26.35, 26.5]
    rates = marginal_rates(cum_rows(truth, start_n=1000,
                                    start_elapsed=1000 * 26.36))
    got = [r["s_per_step"] for r in rates]
    assert got[0] is None                      # nothing to difference against
    assert got[1:] == pytest.approx(truth[1:])
    assert [r["source"] for r in rates[1:]] == ["derived"] * 4


def test_native_and_derived_agree():
    """The trainer's own field and the recovery path must not disagree."""
    truth = [26.3, 27.9, 26.4]
    rows = cum_rows(truth, start_n=1000, start_elapsed=1000 * 26.36)
    derived = [r["s_per_step"] for r in marginal_rates(rows)][1:]
    for row, rate in zip(rows, truth):
        row["step_s_interval"] = rate
    native = [r["s_per_step"] for r in marginal_rates(rows)][1:]
    assert native == pytest.approx(derived)


def test_a_rate_is_never_differenced_across_a_process_restart():
    """⛔ The elapsed clock resets, so the naive difference is meaningless.

    ⚠️ And it is meaningless in the WORST possible way. I expected a negative
    rate — an obvious absurdity a reader would catch. MEASURED here: the naive
    value is **+26.29 s/step**, i.e. an entirely plausible healthy rate, while
    the process is actually running at **40 s/step**. Both terms of the quotient
    are negative at a reset (the new process has less elapsed time AND fewer
    steps), so it is *always* a plausible positive number.

    ⇒ A wrong number that looks wrong is a bug; a wrong number that looks right
    is the `df`-on-a-pod family. Hence: refuse, never estimate.
    """
    before = cum_rows([26.4] * 3, start_n=6000, start_elapsed=6000 * 26.4,
                      start_step=6000)
    after = cum_rows([40.0] * 3, start_n=0, start_step=6150)
    rates = marginal_rates(before + after)

    boundary = rates[len(before)]
    assert boundary["segment"] == 1
    assert boundary["s_per_step"] is None, "must refuse, not produce a number"

    naive = ((after[0]["step_s"] * after[0]["steps_this_process"]
              - before[-1]["step_s"] * before[-1]["steps_this_process"])
             / (after[0]["steps_this_process"]
                - before[-1]["steps_this_process"]))
    assert naive == pytest.approx(26.29, abs=0.05), (
        "the number that would have been reported")
    assert naive < 26.4 * 1.05, "and it would have passed a +5 % guard"

    # while the truth it conceals is a 51 % slowdown, which IS caught
    verdict = check(rates, baseline_s=26.4, warmup_steps=0)
    assert verdict["verdict"] == TRIP
    assert verdict["max_s_per_step"] == pytest.approx(40.0)
    assert verdict["n_segments"] == 2


def test_a_trip_run_never_spans_a_restart():
    """One over-threshold interval each side of a restart is not persistence."""
    before = cum_rows([26.4, 26.4, 30.0], start_n=6000,
                      start_elapsed=6000 * 26.4, start_step=6000)
    after = cum_rows([30.0, 26.4, 26.4], start_n=0, start_step=6150)
    verdict = check(marginal_rates(before + after), baseline_s=26.4,
                    warmup_steps=0, persist=2)
    assert verdict["verdict"] == OK, verdict["note"]


# ==========================================================================
# 4. NOT-A-RESULT MUST NOT READ AS A CLEAN RESULT
# ==========================================================================
def test_no_admissible_points_is_insufficient_not_ok():
    """⛔ The dead-pod shape: absence of a finding is not a finding of absence."""
    verdict = check(marginal_rates(cum_rows([26.4])), baseline_s=26.36)
    assert verdict["verdict"] == INSUFFICIENT
    assert verdict["verdict"] != OK
    assert "proves NOTHING" in verdict["note"]


def test_a_single_spike_is_not_an_abort_but_two_in_a_row_are():
    """A checkpoint write must not abort a 5-day run; a real slowdown must."""
    base = 26.36
    one = cum_rows([base, base, 40.0, base, base], start_n=5000,
                   start_elapsed=5000 * base, start_step=5000)
    two = cum_rows([base, base, 40.0, 40.0, base], start_n=5000,
                   start_elapsed=5000 * base, start_step=5000)
    assert check(marginal_rates(one), baseline_s=base)["verdict"] == OK
    assert check(marginal_rates(two), baseline_s=base)["verdict"] == TRIP


def test_unusable_rows_are_counted_not_silently_dropped():
    """A shrinking denominator is how a monitor quietly stops monitoring."""
    rows = cum_rows([26.4] * 3, start_n=5000, start_elapsed=5000 * 26.4,
                    start_step=5000)
    rows.insert(2, {"step": 5100, "step_s": 26.4})   # no n, no note
    verdict = check(marginal_rates(rows), baseline_s=26.4)
    assert verdict["n_unusable"] >= 1
    assert verdict["n_rows"] == len(rows)


def test_steps_this_process_prefers_the_key_and_falls_back_to_the_note():
    """The prose note is a fallback for banked logs, never the contract."""
    assert steps_this_process({"steps_this_process": 6300}) == 6300
    assert steps_this_process(
        {"step_s_note": "elapsed/step over the 6300 steps THIS process ran"}
    ) == 6300
    # a key present but not a count must not be coerced into one
    assert steps_this_process({"steps_this_process": True}) is None
    assert steps_this_process({"step_s": 26.4}) is None


def test_auto_baseline_says_so_and_flags_what_it_cannot_see():
    """An auto baseline hides a uniformly slow run — it must announce that."""
    rows = cum_rows([40.0] * 6, start_n=5000, start_elapsed=5000 * 40.0,
                    start_step=5000)
    verdict = check(marginal_rates(rows))
    assert verdict["verdict"] == OK          # every point IS the baseline
    assert "CANNOT detect a uniformly slow run" in verdict["baseline_source"]


# ==========================================================================
# 5. THE TRAINER WIRING — the field must exist, and step_s must NOT move
# ==========================================================================
TRAINER = REPO / "stack" / "scripts" / "train_v6_staged.py"


def _train_fn_source():
    """AST of ``train()`` from source — no torch import, no corpus, no GPU.

    ⚠️ The real loop needs the parity corpus, which this box does not hold, so
    the emission is pinned structurally rather than left unverified. This is the
    check that survives a future refactor quietly dropping the field.
    """
    import ast
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node, ast
    raise AssertionError("train() not found in train_v6_staged.py")


def test_the_trainer_emits_the_marginal_field_alongside_the_cumulative_one():
    fn, ast = _train_fn_source()
    rec = next(
        (d for d in ast.walk(fn) if isinstance(d, ast.Dict)
         and any(isinstance(k, ast.Constant) and k.value == "step_s"
                 for k in d.keys)), None)
    assert rec is not None, "the log record dict is gone"
    keys = {k.value for k in rec.keys if isinstance(k, ast.Constant)}
    assert {"step_s", "step_s_interval", "steps_this_process"} <= keys, keys


def test_step_s_keeps_its_cumulative_definition():
    """⛔ ADDITIVE ONLY. Banked logs and the ~5.3-day ETA arithmetic key off
    this field, and this module's recovery path first-differences it. Redefining
    it as a marginal rate would silently corrupt all three."""
    fn, ast = _train_fn_source()
    rec = next(d for d in ast.walk(fn) if isinstance(d, ast.Dict)
               and any(isinstance(k, ast.Constant) and k.value == "step_s"
                       for k in d.keys))
    pairs = {k.value: v for k, v in zip(rec.keys, rec.values)
             if isinstance(k, ast.Constant)}
    cumulative = ast.unparse(pairs["step_s"])
    marginal = ast.unparse(pairs["step_s_interval"])
    assert "t0" in cumulative and "n_proc" in cumulative, cumulative
    assert "last_log_t" in marginal and "d_step" in marginal, marginal
    assert "t0" not in marginal, "the marginal field must not measure from t0"


def test_the_interval_window_is_advanced_inside_the_emission_block():
    """Without this the "interval" silently becomes a second cumulative mean.

    Checked structurally: the advance must live in the SAME ``if step %
    log_every`` block that builds the record, so a row that is never emitted
    widens the next interval instead of losing the time it covered.
    """
    fn, ast = _train_fn_source()

    emit_if = next(
        node for node in ast.walk(fn) if isinstance(node, ast.If)
        and any(isinstance(d, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == "step_s"
                        for k in d.keys)
                for d in ast.walk(node)))
    advances = [
        n for n in ast.walk(emit_if) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple)
        and [getattr(e, "id", None) for e in n.targets[0].elts]
        == ["last_log_t", "last_log_step"]]
    assert len(advances) == 1, "the interval window is never advanced"

    # and it must be INITIALISED from the resume point, not from zero
    src = ast.unparse(fn)
    assert "last_log_t = t0" in src and "last_log_step = start_step" in src


def test_load_rows_ignores_records_without_a_step_time():
    """Spectrum records carry `step` but no `step_s`; pairing them is nonsense."""
    path = Path(__file__).resolve().parent / "_tmp_step_guard.jsonl"
    path.write_text(
        json.dumps({"run_start": {"run": "v6-staged-S-W"}}) + "\n" +
        json.dumps({"step": 50, "spectrum": {"rank": 3}}) + "\n" +
        json.dumps({"step": 50, "step_s": 26.4, "steps_this_process": 50}) +
        "\n" + "not json\n", encoding="utf-8")
    try:
        rows = load_rows(path)
        assert len(rows) == 1 and rows[0]["step_s"] == 26.4
    finally:
        path.unlink()
