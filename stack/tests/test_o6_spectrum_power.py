"""The O6 collapse guard, PROVED to fire — and proved unable to fire at n=48.

⛔ WHY THIS FILE EXISTS. The S-W gate's O6 criterion is
``"O6_rank_retention": ">= 0.8x effective rank across phases"``. The number it
compares came from ONE training batch: on the live v6F S-W run
``z_op_win`` -> ``[B*W, d_op]`` = **48 x 2048**, i.e. 8 windows x 6 CONSECUTIVE
frames drawn from only **4 episodes** (``--batch 8 --window 6
--eps-per-batch 4``, read off the run's own argv).

A centred covariance from 48 rows has rank <= 47. So the banked
``effective_rank`` mean of 15.13 is **15 of 47, not 15 of 2048**, and the
existing collapse test in ``test_v6_staged.py`` exercises the estimator at
``n=64, d=16`` — the well-conditioned regime, which is exactly NOT the regime it
runs in. A guard tested only where it works is a hypothesis.

MEASURED in ``SIGREG_GATE_POWER.md`` (2026-08-16, CPU, seeded):

* at n=48 an ISOTROPIC d=2048 population (true effective rank 2048) reads
  **46.86**, and one collapsed 7.3x to true rank 281 still reads **22.6**;
* pooling the SAME healthy population to 1536 rows moves the reading
  **14 -> 122** — the demonstration that ~15 was a CEILING ARTIFACT;
* the ``>= 0.8x`` criterion fires when nothing changed between **9 %**
  (model-based null) and **38 %** (the run's own banked spread), with power
  **0.11** against a 1.43x true collapse.

The tests below pin all of that as executable facts:

1. the ceiling is STAMPED IN THE RECORD (§1) — the original sin was that it
   was not, so the number got read against ``d``;
2. ⭐ the shipped estimator FIRES on a synthetically collapsed representation
   and does NOT fire on a healthy one (§2);
3. ⭐ the SAME collapse at n=48 is INCONCLUSIVE — the power deficit, executable
   (§3);
4. the default build is UNCHANGED against a CONTENT-anchored reference, never
   ``HEAD`` (§4). v6F S-W is training from this file with ~6.8 days to run.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))

from tanitad.models.v6 import (  # noqa: E402
    O6_ADMISSIBLE_CEILING, O6_RANK_FLOOR, SpectrumAccumulator,
    o6_rank_verdict, spectrum_report)

#: The live run's spectrum geometry, from ``RESTART_v6F_SW.sh``'s argv.
LIVE_B, LIVE_W, LIVE_EPS, LIVE_D = 8, 6, 4, 2048
LIVE_N = LIVE_B * LIVE_W                                   # 48


# =========================================================================== #
# helpers — a latent whose rows carry the REAL sampler's nested correlation
# =========================================================================== #
def _rows(d: int, *, calls: int, gen: torch.Generator, keep: int | None = None,
          floor: float = 1e-3, n_eps: int = LIVE_EPS, n_win: int = LIVE_B,
          w: int = LIVE_W) -> torch.Tensor:
    """``[calls*n_win*w, d]``.

    ``keep=None`` is HEALTHY (isotropic). ``keep=k`` is COLLAPSED: every
    direction past the k-th is scaled by ``floor`` — a squeeze, not a hard
    truncation, because a hard truncation is the easiest possible thing to
    detect and would flatter the instrument.

    Rows are NOT iid: 6 consecutive frames share a window factor and the
    windows share ``n_eps`` episode factors, which is what the real
    ``InteractionSampler`` produces with ``--eps-per-batch 4``.
    """
    scale = torch.ones(d, dtype=torch.float64)
    if keep is not None:
        scale[keep:] = floor
    out = []
    for _ in range(calls):
        ge = torch.randn(n_eps, d, generator=gen, dtype=torch.float64)
        gw = torch.randn(n_win, d, generator=gen, dtype=torch.float64)
        gr = torch.randn(n_win * w, d, generator=gen, dtype=torch.float64)
        idx = torch.arange(n_win) % n_eps
        g = (0.6 * ge[idx].repeat_interleave(w, 0)
             + 0.7 * gw.repeat_interleave(w, 0) + 0.4 * gr)
        out.append(g * scale)
    return torch.cat(out, 0).float()


# =========================================================================== #
# 1. THE CEILING IS IN THE RECORD — the original sin, closed
# =========================================================================== #
def test_record_stamps_its_own_rank_ceiling():
    """⛔ "15 of 2048" must be impossible to write from this record."""
    torch.manual_seed(0)
    r = spectrum_report(torch.randn(LIVE_N, LIVE_D))
    assert r["n"] == LIVE_N and r["d"] == LIVE_D
    assert r["rank_ceiling"] == LIVE_N - 1 == 47
    assert r["effective_rank"] <= r["rank_ceiling"] + 1e-9
    assert r["effective_rank_frac"] == pytest.approx(
        r["effective_rank"] / r["rank_ceiling"])
    assert r["rank_admissible"] is False
    assert "rank <= 47" in r["ceiling_note"] and "d=2048" in r["ceiling_note"]


def test_ceiling_is_the_row_count_not_the_width():
    """A wide, healthy latent still cannot read above n-1. This is the fact the
    whole power deficit rests on, so it is pinned rather than argued."""
    torch.manual_seed(1)
    r = spectrum_report(torch.randn(LIVE_N, LIVE_D))
    # isotropic d=2048: the TRUE effective rank is 2048, the reading is ~47
    assert r["effective_rank"] < 47.0
    assert r["effective_rank"] > 46.0            # pinned AT the ceiling
    assert r["effective_rank_frac"] > 0.97


def test_accumulator_raises_the_ceiling_and_the_reading():
    """The fix, end to end: the SAME healthy population read at two pool sizes.

    This is the executable form of "14 -> 122": pooling does not make the
    representation better, it makes the ESTIMATOR able to see it."""
    gen = torch.Generator().manual_seed(2)
    d = 512
    one = spectrum_report(_rows(d, calls=1, gen=gen))
    acc = SpectrumAccumulator(capacity=8, block=LIVE_W)
    for _ in range(8):
        acc.push(_rows(d, calls=1, gen=gen))
    pooled = acc.report()
    assert one["rank_ceiling"] == 47
    assert pooled["rank_ceiling"] == 383 and pooled["n"] == 384
    assert pooled["pooled_steps"] == 8 and pooled["pool_block_rows"] == LIVE_W
    # the reading rises by a large factor on an UNCHANGED population
    assert pooled["effective_rank"] > 3.0 * one["effective_rank"]


def test_accumulator_ring_is_bounded():
    acc = SpectrumAccumulator(capacity=3, block=2)
    gen = torch.Generator().manual_seed(3)
    for _ in range(10):
        acc.push(torch.randn(4, 16, generator=gen))
    assert len(acc) == 3 and acc.n_rows == 12
    acc.clear()
    assert len(acc) == 0
    with pytest.raises(ValueError, match="empty"):
        acc.report()
    with pytest.raises(ValueError, match="capacity"):
        SpectrumAccumulator(capacity=0)


# =========================================================================== #
# 2. ⭐ THE GUARD FIRES — and does not fire on a healthy representation
# =========================================================================== #
#: Test-scale admissibility. The SHIPPED constants are 1024 / 64 and are
#: asserted separately below; these keep the eigendecompositions small enough
#: to run in the default suite while exercising the identical logic.
T_CEIL, T_FLOOR, T_D = 128, 16.0, 256


def _reading(d, calls, gen, keep=None, ci=48):
    return spectrum_report(_rows(d, calls=calls, gen=gen, keep=keep),
                           ci_reps=ci, block=LIVE_W, generator=gen)


def test_guard_FIRES_on_a_synthetically_collapsed_representation():
    """⭐ THE PROOF. A representation squeezed onto 8 of 256 directions must
    make the verdict FAIL — at a pool where the estimator is admissible."""
    gen = torch.Generator().manual_seed(10)
    ref = _reading(T_D, 8, gen)                       # healthy phase start
    cur = _reading(T_D, 8, gen, keep=8)               # collapsed
    v = o6_rank_verdict(cur, ref, floor=T_FLOOR, ceiling_min=T_CEIL)
    assert v["status"] == "FAIL" and v["pass"] is False
    assert v["rank_ceiling"] >= T_CEIL
    # it must fail for a STATED reason, one of the two pre-registered clauses
    assert ("clause 2" in v["reason"]) or ("clause 3" in v["reason"])


def test_guard_FIRES_on_clause_2_alone_with_the_floor_disarmed():
    """⭐ THE OTHER FIRING PATH. The test above short-circuits on clause 3 (the
    absolute floor), so on its own it would leave RETENTION unproven. Here the
    floor is set to 1.0 so it cannot trip, and the verdict must still FAIL —
    from the interval, not from the constant."""
    gen = torch.Generator().manual_seed(40)
    ref = _reading(T_D, 8, gen)
    cur = _reading(T_D, 8, gen, keep=64)              # a 3x squeeze
    v = o6_rank_verdict(cur, ref, floor=1.0, ceiling_min=T_CEIL)
    assert v["status"] == "FAIL" and "clause 2" in v["reason"]
    assert v["retention_ci95"]["hi"] < 0.8            # the interval EXCLUDES it
    assert v["effective_rank"] > 1.0                  # clause 3 genuinely idle


def test_guard_DOES_NOT_fire_on_a_healthy_representation():
    """The other half — and it must reach a real **PASS**, not merely avoid a
    FAIL. A guard that can only ever say INCONCLUSIVE is not a guard either."""
    gen = torch.Generator().manual_seed(11)
    ref = _reading(T_D, 8, gen)
    cur = _reading(T_D, 8, gen)                       # same population
    v = o6_rank_verdict(cur, ref, floor=T_FLOOR, ceiling_min=T_CEIL)
    assert v["status"] == "PASS" and v["pass"] is True, v["reason"]
    assert v["retention"] == pytest.approx(1.0, abs=0.10)
    lo, hi = v["retention_ci95"]["lo"], v["retention_ci95"]["hi"]
    assert lo >= 0.8 and lo < 1.0 < hi                # a real interval, not a point


def test_guard_fires_on_the_absolute_floor_without_any_reference():
    """Clause 3: retention alone cannot see a representation that was ALREADY
    collapsed when the reference was taken."""
    gen = torch.Generator().manual_seed(12)
    cur = _reading(T_D, 8, gen, keep=4, ci=0)
    v = o6_rank_verdict(cur, None, floor=T_FLOOR, ceiling_min=T_CEIL)
    assert v["status"] == "FAIL" and "clause 3" in v["reason"]


def test_guard_refuses_a_point_ratio_with_no_interval():
    """⛔ The defect this replaces: a bare ratio deciding a gate."""
    gen = torch.Generator().manual_seed(13)
    ref = _reading(T_D, 8, gen, ci=0)
    cur = _reading(T_D, 8, gen, keep=8, ci=0)
    v = o6_rank_verdict(cur, ref, floor=1.0, ceiling_min=T_CEIL)
    assert v["pass"] is None and v["status"] == "INCONCLUSIVE"
    assert "effective_rank_ci95" in v["reason"]


def test_shipped_constants_are_the_preregistered_ones():
    """The numbers in ``SIGREG_GATE_POWER.md`` §5 and the code are one thing."""
    assert O6_ADMISSIBLE_CEILING == 1024
    assert O6_RANK_FLOOR == 64.0


# =========================================================================== #
# 3. ⭐ THE POWER DEFICIT, EXECUTABLE — the old reading cannot see the collapse
# =========================================================================== #
def test_at_n48_the_verdict_is_INCONCLUSIVE_by_construction():
    """⛔ THE HEADLINE. The live run's own reading cannot pass OR fail the
    gate — and the guard now says so instead of producing a number."""
    gen = torch.Generator().manual_seed(20)
    cur = spectrum_report(_rows(LIVE_D, calls=1, gen=gen))
    v = o6_rank_verdict(cur, None)
    assert v["pass"] is None and v["status"] == "INCONCLUSIVE"
    assert v["rank_ceiling"] == 47
    assert "cannot resolve rank" in v["reason"]


def _true_effective_rank(d: int, keep: int, floor: float) -> float:
    """The ESTIMAND — the same entropy functional on the POPULATION singular
    values, which for :func:`_rows` are exactly the per-coordinate scales."""
    from tanitad.eval.spectral import effective_rank
    s = torch.ones(d, dtype=torch.float64)
    s[keep:] = floor
    return effective_rank(s)


def test_a_true_collapse_THROUGH_the_0_8_threshold_reads_as_no_change():
    """⭐ THE POWER DEFICIT, EXECUTABLE.

    Build a population whose TRUE effective-rank ratio is BELOW 0.8 — exactly
    the collapse the criterion exists to catch — and show the n=48 reading
    reports a ratio comfortably ABOVE 0.8. The criterion is looking straight at
    a collapse it was designed to catch and calling it healthy."""
    gen = torch.Generator().manual_seed(21)
    keep, floor = 64, 0.25
    true_ratio = (_true_effective_rank(T_D, keep, floor)
                  / _true_effective_rank(T_D, T_D, 1.0))
    assert true_ratio < 0.8, "the fixture must be a collapse the gate targets"
    healthy = torch.tensor([spectrum_report(
        _rows(T_D, calls=1, gen=gen))["effective_rank"] for _ in range(24)])
    mild = torch.tensor([spectrum_report(
        _rows(T_D, calls=1, gen=gen, keep=keep, floor=floor)
    )["effective_rank"] for _ in range(24)])
    observed = float(mild.mean() / healthy.mean())
    assert observed > 0.9, (
        f"true rank ratio {true_ratio:.3f} (BELOW the 0.8 criterion) reads as "
        f"{observed:.3f} at n=48 — the collapse is invisible to the gate")
    # and no single pairing of readings would have fired either
    fired = float(((mild / healthy) < 0.8).double().mean())
    assert fired == 0.0


def test_pooling_separates_what_n48_could_not():
    """And the fix closes it: the same mild collapse becomes separable once the
    estimator is pooled. Healthy and collapsed reading bands must not touch."""
    gen = torch.Generator().manual_seed(22)
    healthy = torch.tensor([spectrum_report(
        _rows(T_D, calls=8, gen=gen))["effective_rank"] for _ in range(6)])
    mild = torch.tensor([spectrum_report(
        _rows(T_D, calls=8, gen=gen, keep=64, floor=0.25)
    )["effective_rank"] for _ in range(6)])
    assert float(mild.max()) < float(healthy.min()), (
        f"pooled bands still overlap: healthy {healthy.min():.2f}.."
        f"{healthy.max():.2f} vs collapsed {mild.min():.2f}..{mild.max():.2f}")
    assert float(mild.mean() / healthy.mean()) < 0.8


# =========================================================================== #
# 4. ⛔ THE NO-CHANGE PROOF — CONTENT-anchored, never HEAD (C75)
# =========================================================================== #
#: The marker that separates the pre-change ``spectrum_report`` from this one.
#: Any revision of ``v6.py`` WITHOUT it is a revision the live v6F S-W run
#: could be executing.
_PRE_CHANGE_MARKER = "O6_ADMISSIBLE_CEILING"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_change_module():
    """``tanitad.models.v6`` as it was BEFORE this change, imported side by side.

    ⚠️ NOT ``HEAD``. HEAD moves under us while the file is being written — a
    HEAD comparison then compares the module with itself and passes by
    construction. Walk ``v6.py``'s own history for the NEWEST revision that
    does not yet carry the new symbol; that reference is stable no matter how
    many commits land afterwards, and it IS the code the live run is running.

    Returns ``None`` when git cannot answer; the caller skips. A skipped test is
    honest, a self-comparison dressed as a real one is not.
    """
    root = _STACK.parent
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", _V6_REL],
                             cwd=root, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        src = None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{_V6_REL}"], cwd=root,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if _PRE_CHANGE_MARKER.encode() in r.stdout:
                continue                        # already carries the change
            src = r.stdout
            break
        if src is None:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_spectrum.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_spectrum", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_spectrum"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def test_default_spectrum_report_is_UNCHANGED_vs_the_pre_change_revision():
    """⛔ Every shared key must be BIT-EQUAL on identical input, and the new
    keys must be purely additive. The live run is reading this function."""
    prev = _pre_change_module()
    if prev is None:
        pytest.skip("git could not supply a pre-change v6.py revision")
    for shape in [(LIVE_N, LIVE_D), (64, 16), (12, 3), (2, 5)]:
        torch.manual_seed(7)
        z = torch.randn(*shape)
        old = prev.spectrum_report(z)
        new = spectrum_report(z)
        assert set(old) <= set(new), f"a key VANISHED at {shape}"
        for k, v in old.items():
            assert new[k] == v, f"{k} moved at {shape}: {v} -> {new[k]}"
    # and the additions are exactly the ones documented
    added = set(spectrum_report(torch.randn(8, 4))) - set(
        prev.spectrum_report(torch.randn(8, 4)))
    assert added == {"rank_ceiling", "effective_rank_frac", "rank_admissible",
                     "ceiling_note"}


def test_default_spectrum_report_consumes_NO_global_rng():
    """⚠️ Switching the CI on must not be able to move the run's loss. The
    default path draws nothing, and the bootstrap path draws only from the
    generator it is handed."""
    z = torch.randn(48, 32)
    torch.manual_seed(99)
    before = torch.randn(3)
    torch.manual_seed(99)
    spectrum_report(z)                                   # default: no draws
    assert torch.equal(torch.randn(3), before)
    torch.manual_seed(99)
    g = torch.Generator().manual_seed(5)
    spectrum_report(z, ci_reps=8, block=6, generator=g)  # bootstrap: own gen
    assert torch.equal(torch.randn(3), before)


def test_interval_is_reported_with_its_kind_and_unit():
    """An interval with no estimator named is the thing CLAUDE.md forbids.

    ⛔ And the estimator is the JACKKNIFE, not the bootstrap — the one measured
    to cover (0.85 / 0.867 vs 0.25 / 0.00). The bootstrap survives only as a
    labelled diagnostic."""
    gen = torch.Generator().manual_seed(30)
    r = spectrum_report(_rows(64, calls=4, gen=gen), ci_reps=32, block=LIVE_W,
                        generator=gen)
    ci = r["effective_rank_ci95"]
    assert ci["kind"] == "leave-one-cluster-out jackknife"
    assert ci["block_rows"] == LIVE_W and ci["n_blocks"] == 32
    assert ci["lo"] < r["effective_rank"] < ci["hi"]     # it brackets the point
    assert "measured_coverage" in ci


def test_the_bootstrap_is_kept_ONLY_as_a_labelled_diagnostic():
    """⛔ The reason the bootstrap is not the interval, pinned as a fact: for a
    RANK functional it sits BELOW the point estimate, because resampling with
    replacement duplicates blocks and duplicated rows are rank-deficient."""
    gen = torch.Generator().manual_seed(32)
    r = spectrum_report(_rows(64, calls=4, gen=gen), ci_reps=48, block=LIVE_W,
                        generator=gen)
    diag = r["effective_rank_ci95"]["bootstrap_DIAGNOSTIC_do_not_quote"]
    assert diag["percentile_hi"] < r["effective_rank"], (
        "the bootstrap is supposed to be biased DOWN here — if it is not, the "
        "reason given in the record is wrong and must be re-derived")
    assert "rank-deficient" in diag["why_not_used"]


def test_the_pooling_window_ends_AT_the_emission():
    """⚠️ Off-by-one here silently changes what the pooled spectrum is measured
    over, and the record would still look well-formed. The block must be the
    ``accum-1`` steps BEFORE an emission plus the emission step itself — so the
    pool spans ``accum`` CONSECUTIVE steps ending where it is read."""
    sys.path.insert(0, str(_STACK / "scripts"))
    from train_v6_staged import in_spectrum_window
    every, accum = 200, 32
    hits = [s for s in range(1, 401)
            if in_spectrum_window(s, every, accum)]
    assert hits == list(range(169, 201)) + list(range(369, 401))
    assert len(hits) == 2 * accum                    # exactly accum per cycle
    assert 200 in hits and 400 in hits               # the emission steps
    assert 168 not in hits and 201 not in hits       # and nothing either side
    # the incumbent path never pools
    assert not any(in_spectrum_window(s, every, 1) for s in range(1, 401))
    # accum larger than the period degenerates to "every step", not to a gap
    assert all(in_spectrum_window(s, 4, 10) for s in range(1, 20))


def test_interval_refuses_too_few_blocks():
    gen = torch.Generator().manual_seed(31)
    r = spectrum_report(torch.randn(12, 8, generator=gen), ci_reps=16, block=6,
                        generator=gen)
    assert r["effective_rank_ci95"]["status"] == "n/a"
    assert "blocks" in r["effective_rank_ci95"]["reason"]
