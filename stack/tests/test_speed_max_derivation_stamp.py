"""E16 — THE PROVENANCE STAMP IS A PRECONDITION, and the v8 builder's ladder.

⭐⭐ THE PI'S RULING (2026-09-10): *"regarding to max speed as input, stick to the
labels we created in the data set with the logic of minimal speed etc..."* — so
the channel is built from ``g_tac.goals.SPEED_BAND``, which is **ego-future**
(max of the ego's own realised speed over [t0+2 s, +6 s], confirmed at three
independent source sites). The PI has decided to use it, and that decision is
consistent with the programme's existing position: refcv5-v2 already feeds an
oracle nav command and declares it in ``nav_cmd_derivation``.

⛔⛔ **THEREFORE THE BINDING REQUIREMENT IS DECLARATION, NOT REFUSAL — AND A
DECLARATION THAT CAN BE SILENTLY DROPPED IS NOT ONE.** This file pins that the
stamp is a *precondition* of starting, in both directions, with deliberate
regressions that must go RED. ⛔ NO CAPABILITY CLAIM MAY BE CREDITED TO THIS
CHANNEL WITHOUT THE STAMP BESIDE IT.

⛔ THE CROSS-CHECKS HERE ARE INDEPENDENTLY AUTHORED. The ladder is re-derived
from the km/h INTEGERS in **exact rational arithmetic**, never by re-running the
builder's own float derivation — *"a cross-check must be derived independently of
the value it checks; re-running the producer's derivation measures determinism,
not correctness."* MEASURED 2026-09-07: a builder that verified its buckets
against its own 4-dp-rounded ladder read **0 % moved** and passed its own gate
while **57.5 % of the corpus (2,631/4,572)** was one step off.

⚠️ Every expectation below is written as a **LITERAL**, never as an expression
over the code under test.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from fractions import Fraction

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import max_speed_input as msi             # noqa: E402

# ⛔ TYPED FROM ROAD LAW, NOT IMPORTED. If the module re-pins its ladder this
# file does NOT follow, and the disagreement is the finding.
LADDER_KMH = (20, 30, 50, 70, 80, 100, 120, 130)

#: ⛔ The four things a reader who opens `config.json` IN ISOLATION must learn
#: without going to find the code. Literals.
REQUIRED_TOKENS = ("oracle", "ego-future", "SPEED_BAND.v_hi_ms",
                   "[t0+2 s, +6 s]")


def _trainer():
    """``refc_v3_train.py`` by PATH — the convention in
    ``test_max_speed_wiring.py`` and ``test_refc_v3_rollability``."""
    p = os.path.join(_STACK, "scripts", "refc_v3_train.py")
    if not os.path.exists(p):
        pytest.skip(f"trainer not present at {p}")
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_stamp", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_stamp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _builder():
    p = os.path.join(_STACK, "scripts", "build_v8_speed_max_labels.py")
    if not os.path.exists(p):
        pytest.skip(f"builder not present at {p}")
    spec = importlib.util.spec_from_file_location("build_v8_for_test", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_v8_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    return types.SimpleNamespace(**kw)


def _rec(v_hi: float, v_lo: float, clip="clip-A", held=False) -> dict:
    return {"clip_id": clip,
            "g_tac": {"goals": {"SPEED_BAND": {"v_hi_ms": v_hi,
                                               "v_lo_ms": v_lo,
                                               "held": held,
                                               "band_s": [2.0, 6.0]}}}}


# ==========================================================================
# 1 — THE STAMP IS A PRECONDITION (deliberate regressions, both directions)
# ==========================================================================
def test_the_real_stamp_declares_all_four_things_a_reader_needs():
    """⭐ The constant itself, against LITERALS."""
    tr = _trainer()
    stamp = tr.SPEED_MAX_DERIVATION
    assert isinstance(stamp, str) and stamp.strip()
    for tok in REQUIRED_TOKENS:
        assert tok in stamp, f"the stamp does not declare {tok!r}: {stamp!r}"
    # same-breath NEGATIVE control: a token that must NOT be claimed. The
    # channel is an input, never a training signal, and the stamp says so.
    assert "training signal" in stamp and "never a training signal" in stamp


def test_a_missing_stamp_REFUSES_THE_RUN_not_merely_logs():
    """⛔ THE DELIBERATE REGRESSION. A config without the stamp must not start."""
    tr = _trainer()
    a = _args(max_speed_input=True)
    with pytest.raises(SystemExit) as e:
        tr._assert_speed_max_stamp({"arm": "hier"}, a)          # stamp absent
    msg = str(e.value)
    assert "speed_max_derivation" in msg and "Refusing to start" in msg
    # ⭐ SAME-BREATH POSITIVE CONTROL: the identical call WITH the stamp must
    # NOT raise. Without this, a guard that raised unconditionally would pass.
    tr._assert_speed_max_stamp(
        {"speed_max_derivation": tr.SPEED_MAX_DERIVATION}, a)


@pytest.mark.parametrize("stamp", [None, "", "   ", 42, {"on": True},
                                   "max speed: on"])
def test_a_stamp_that_declares_nothing_is_REFUSED(stamp):
    """⛔ A PRESENCE CHECK IS NOT ENOUGH. ``"max speed: on"`` satisfies "the key
    exists" and tells a reader nothing, so the guard asserts on CONTENT."""
    tr = _trainer()
    with pytest.raises(SystemExit):
        tr._assert_speed_max_stamp({"speed_max_derivation": stamp},
                                   _args(max_speed_input=True))


@pytest.mark.parametrize("drop", REQUIRED_TOKENS)
def test_dropping_ANY_ONE_required_token_is_REFUSED(drop):
    """⛔ Each of the four is individually load-bearing — a stamp that names the
    field but not the window, or the window but not the word `oracle`, is a
    partial declaration and is refused."""
    tr = _trainer()
    maimed = tr.SPEED_MAX_DERIVATION.replace(drop, "")
    assert maimed != tr.SPEED_MAX_DERIVATION, "the mutation changed nothing"
    with pytest.raises(SystemExit) as e:
        tr._assert_speed_max_stamp({"speed_max_derivation": maimed},
                                   _args(max_speed_input=True))
    assert drop in str(e.value), "the refusal must NAME what is missing"


def test_the_MIRROR_failure_is_also_refused_a_control_must_not_be_stamped():
    """⛔ An OFF run carrying the stamp would read as a max-speed-conditioned
    arm. That is how a control becomes a result."""
    tr = _trainer()
    off = _args(max_speed_input=False)
    with pytest.raises(SystemExit) as e:
        tr._assert_speed_max_stamp(
            {"speed_max_derivation": tr.SPEED_MAX_DERIVATION}, off)
    assert "OFF" in str(e.value)
    # same-breath control: an OFF run with no stamp is fine.
    tr._assert_speed_max_stamp({"speed_max_derivation": None}, off)
    tr._assert_speed_max_stamp({}, off)


def test_the_guard_is_WIRED_into_the_config_write_not_merely_defined():
    """⛔ A guard nobody calls is a comment. Assert the call site exists in the
    trainer's source, immediately around the config.json write."""
    src = open(os.path.join(_STACK, "scripts", "refc_v3_train.py"),
               encoding="utf-8").read()
    assert "_assert_speed_max_stamp(_run_config, args)" in src
    assert '"speed_max_derivation": (SPEED_MAX_DERIVATION' in src
    # the guard must run BEFORE the file is written, or a refused run still
    # leaves an unstamped artifact on disk.
    i_guard = src.index("_assert_speed_max_stamp(_run_config, args)")
    i_write = src.index('(out_dir / "config.json").write_text')
    assert i_guard < i_write, "the stamp guard runs AFTER config.json is written"


# ==========================================================================
# 2 — THE LADDER, cross-checked in EXACT RATIONAL ARITHMETIC
# ==========================================================================
def _snap_up_exact(v_ms_text: str) -> int:
    v = Fraction(v_ms_text) * Fraction(18, 5)      # 1 m/s = 18/5 km/h, exactly
    for k in LADDER_KMH:
        if v <= k:
            return k
    return LADDER_KMH[-1]


def _snap_down_exact(v_ms_text: str) -> int:
    v = Fraction(v_ms_text) * Fraction(18, 5)
    best = None
    for k in LADDER_KMH:
        if Fraction(k) <= v:
            best = k
    return LADDER_KMH[0] if best is None else best


def test_the_pinned_ladder_is_the_one_this_file_typed_from_road_law():
    assert msi.POSTED_LIMIT_STEPS_KMH == LADDER_KMH


def test_the_ladder_in_ms_is_NOT_rounded():
    """⛔⛔ THE 57.5 % DEFECT'S ROOT. ``round(50/3.6, 4) = 13.8889`` is strictly
    GREATER than the true step, so a rounded ladder shipped as a value and
    re-snapped moves a step. Assert the module keeps FULL precision."""
    for k, s in zip(msi.POSTED_LIMIT_STEPS_KMH, msi.POSTED_LIMIT_STEPS_MS):
        assert s == k / 3.6
        assert abs(s - round(s, 4)) < 1e-3        # they are close...
        if k in (20, 50, 100):
            assert round(s, 4) > s, (
                f"{k} km/h: the 4-dp rounding is supposed to be ABOVE the true "
                f"step — this is the mechanism, and if it stops being true the "
                f"regression arm below is measuring nothing")


@pytest.mark.parametrize("v_hi,v_lo", [
    ("0.00", "0.00"), ("5.55", "0.00"), ("5.56", "5.55"), ("11.19", "8.64"),
    ("13.88", "13.88"), ("13.89", "5.00"), ("27.77", "19.44"),
    ("36.11", "33.33"), ("37.80", "30.00"),
])
def test_the_builder_agrees_with_the_INDEPENDENT_exact_ladder(v_hi, v_lo):
    """⭐ The builder snaps in FLOAT m/s space; this asserts against EXACT
    rational km/h space. Two spaces, two authors — agreement is evidence."""
    b = _builder()
    blk = b.build_block(_rec(float(v_hi), float(v_lo)))
    assert blk["v_max_bucket_kmh"] == _snap_up_exact(v_hi)
    assert blk["v_min_bucket_kmh"] == _snap_down_exact(v_lo)


def test_the_DELIBERATE_REGRESSION_a_4dp_bucket_moves_the_corpus():
    """⛔⛔ REINTRODUCE THE REAL 2026-09-07 DEFECT AND PROVE IT IS DETECTABLE.

    Ship the bucket as a 4-dp-rounded m/s value and re-snap it: the three steps
    whose rounding lands ABOVE the true value each move one rung. A check that
    cannot go RED has never been shown to work.
    """
    moved = []
    for k in LADDER_KMH:
        shipped = round(k / 3.6, 4)               # the defect
        again = next((s for s in LADDER_KMH if shipped <= s / 3.6),
                     LADDER_KMH[-1])
        if again != k:
            moved.append((k, again))
    assert moved == [(20, 30), (50, 70), (100, 120)], (
        f"the historical defect no longer reproduces: {moved}. Either the "
        f"ladder changed or this regression arm is inert — either way the "
        f"builder's cross-check is no longer proven.")


def test_the_builder_ships_NO_ms_bucket_so_the_defect_has_no_surface():
    b = _builder()
    blk = b.build_block(_rec(13.89, 5.0))
    assert "v_max_bucket_ms" not in blk and "v_min_bucket_ms" not in blk
    # same-breath control: the km/h bucket IS there and IS an int.
    assert isinstance(blk["v_max_bucket_kmh"], int)
    assert isinstance(blk["v_min_bucket_kmh"], int)


# ==========================================================================
# 3 — UNITS ON THE WIRE, and the block the consumer actually reads
# ==========================================================================
def test_every_emitted_block_declares_its_units_and_the_reader_accepts_it():
    b = _builder()
    blk = b.build_block(_rec(11.19, 8.64))
    assert blk["units"] == "m_s" and blk["control_units"] == "m_s"
    got = msi.read_max_speed_field(blk, mode="quantized")
    assert got["valid"] is True and got["units_source"] == "artifact"
    assert got["raw_ms"] == 11.19
    # ⛔ THE CONSUMER'S OWN CONTENT ASSERTION (refc_v3_train.py:1574-1584):
    # round(quantized_ms * 3.6) must equal the shipped km/h bucket.
    assert round(float(got["quantized_ms"]) * 3.6) == blk["v_max_bucket_kmh"]


def test_a_block_with_its_units_STRIPPED_is_REFUSED_by_the_reader():
    """⛔ The 396 g incident: m/s vs km/h vs mph is a 1.61x spread."""
    b = _builder()
    blk = dict(b.build_block(_rec(11.19, 8.64)))
    blk.pop("units"), blk.pop("control_units")
    with pytest.raises(msi.MaxSpeedUnitsMissing):
        msi.read_max_speed_field(blk, mode="quantized")


def test_the_block_carries_the_ORACLE_declaration_on_every_record():
    b = _builder()
    blk = b.build_block(_rec(11.19, 8.64))
    assert blk["oracle"] is True
    assert blk["provenance"] == "ego-future"
    assert "SPEED_BAND.v_hi_ms" in blk["source_field"]
    assert "SPEED_BAND.v_lo_ms" in blk["source_field"]
    for tok in REQUIRED_TOKENS:
        assert tok in blk["derivation"]


def test_the_ladder_shipped_on_the_record_is_the_pinned_one():
    """⛔ The trainer refuses a record whose `bucket_steps_kmh` differs — two
    ladders is two experiments."""
    b = _builder()
    blk = b.build_block(_rec(11.19, 8.64))
    assert tuple(blk["bucket_steps_kmh"]) == LADDER_KMH


# ==========================================================================
# 4 — THE FLOOR: the PI's "logic of minimal speed"
# ==========================================================================
def test_the_bucket_MINIMUM_is_20_kmh_and_there_are_NO_NULLS():
    """⚠️ ALREADY SETTLED, PINNED SO IT CANNOT ROT: the PI's earlier premise
    that a 30 km/h floor was needed FAILS. A fully stopped ego snaps UP to the
    ladder's bottom step, so every clip receives a ceiling and none is null."""
    b = _builder()
    assert LADDER_KMH[0] == 20
    stopped = b.build_block(_rec(0.0, 0.0))
    assert stopped["v_max_bucket_kmh"] == 20
    assert stopped["v_max_ms"] is not None
    assert stopped["under_floor"] is True        # 0 m/s is below the bottom step
    assert stopped["over_ceiling"] is False


def test_the_floor_snaps_DOWN_and_the_ceiling_snaps_UP():
    """⭐ The two directions are the point: a ceiling admits the speed observed,
    a floor is the largest posted step at or below it."""
    b = _builder()
    blk = b.build_block(_rec(15.0, 15.0))        # 54 km/h, between 50 and 70
    assert blk["v_max_bucket_kmh"] == 70         # UP
    assert blk["v_min_bucket_kmh"] == 50         # DOWN
    assert blk["band_width_ms"] == 0.0


def test_the_FLOOR_IS_RECORDED_BUT_NOT_WIRED_so_refcv6_stays_single_lever():
    """⚠️ Widening the model block would make the arm multi-lever. The seam is
    3 dims and the floor is not one of them — pinned so a later change is a
    decision rather than a drift."""
    assert msi.MAX_SPEED_DIMS == 3
    assert msi.artifact_meta("quantized")["dim_names"] == [
        "value_norm", "over_ceiling", "valid"]
    b = _builder()
    blk = b.build_block(_rec(11.19, 8.64))
    assert blk["v_min_ms"] == 8.64               # present on the RECORD...
    blkc = msi.read_max_speed_field(blk, mode="quantized")
    assert "v_min_ms" not in blkc                # ...and absent from the READ


# ==========================================================================
# 5 — THE BUILDER REFUSES rather than fabricating
# ==========================================================================
def test_a_record_with_no_SPEED_BAND_is_REFUSED_not_defaulted():
    b = _builder()
    with pytest.raises(SystemExit) as e:
        b.build_block({"clip_id": "clip-X", "g_tac": {"goals": {}}})
    assert "SPEED_BAND" in str(e.value)
    # same-breath control: with the band, it builds.
    assert b.build_block(_rec(11.19, 8.64))["v_max_ms"] == 11.19


@pytest.mark.parametrize("bad", [None, "fast", float("nan"), float("inf")])
def test_a_nonfinite_bound_is_REFUSED(bad):
    b = _builder()
    with pytest.raises(SystemExit):
        b.build_block(_rec(bad, 0.0))


def test_an_INVERTED_band_is_REFUSED_rather_than_silently_swapped():
    b = _builder()
    with pytest.raises(SystemExit) as e:
        b.build_block(_rec(5.0, 20.0))           # floor above ceiling
    assert "inverted" in str(e.value)


# ==========================================================================
# 6 — the synthetic corpus can now carry a REAL label id
# ==========================================================================
def test_synth_episodes_stamps_real_clip_ids_when_asked_and_not_otherwise():
    """⛔ MEASURED 2026-09-10: `--synth-episodes --v7-labels` died on
    ``int('synth-000')``, so NO v7-label channel could be smoked at all. The
    default must be unchanged; the opt-in must produce joinable INTEGER ids."""
    from tanitad.data.v2_dataset import stable_episode_id
    from tanitad.refs import refc_v3 as v3
    tr = _trainer()
    cfg = v3.refc_v3_smoke_config(hier=True)

    default = tr._synth_episodes(2, cfg.core, seed=0)
    assert [e.episode_id for e in default] == ["synth-000", "synth-001"]

    clips = ["aaaa-1111", "bbbb-2222"]
    stamped = tr._synth_episodes(2, cfg.core, seed=0, clip_ids=clips)
    assert [e.episode_id for e in stamped] == [stable_episode_id(c)
                                               for c in clips]
    for e in stamped:
        assert isinstance(e.episode_id, int)     # the join does int(...)
    # ⛔ and it refuses to reuse a clip id, which would collapse two episodes
    # into one label while the join count still read 100 %.
    with pytest.raises(SystemExit):
        tr._synth_episodes(3, cfg.core, seed=0, clip_ids=clips)


def test_a_two_arg_shim_for_synth_episodes_STILL_WORKS_no_kwarg_leaks():
    """⛔⛔ THE REGRESSION THIS PINS ACTUALLY HAPPENED, AND THE SUITE CAUGHT IT.

    MEASURED 2026-09-10: passing ``clip_ids=None`` unconditionally from
    ``train()`` broke **three** tests in ``test_refc_v3_save_before_eval.py``
    which monkeypatch ``_synth_episodes`` with a shim of the OLD signature --
    ``TypeError: _train_eps() got an unexpected keyword argument 'clip_ids'``.
    The shims were right; the call site was wrong. ⇒ a NEW optional argument
    must not reach a caller that never asked for it.

    ⭐ Asserted on the trainer's SOURCE at the call site, because that is where
    the defect lived -- a test that only called ``_synth_episodes`` directly
    would pass either way and pin nothing.
    """
    src = open(os.path.join(_STACK, "scripts", "refc_v3_train.py"),
               encoding="utf-8").read()
    assert '_synth_kw = ({"clip_ids": synth_clip_ids_from_labels}' in src
    assert "if synth_clip_ids_from_labels else {})" in src
    assert "seed=args.seed,\n                              **_synth_kw)" in src
    # ⛔ and the unconditional form must be GONE, or both could coexist.
    assert "clip_ids=synth_clip_ids_from_labels)" not in src

    # the behavioural half: a shim with the OLD signature must survive the
    # kwargs the production call site builds when no labels were supplied.
    def old_shim(n, cfg, seed=0, min_frames=40):
        return [f"ep{i}" for i in range(n)]

    for value in (None, [], ""):                 # every falsy carrier
        kw = {"clip_ids": value} if value else {}
        assert old_shim(2, None, seed=0, **kw) == ["ep0", "ep1"]
    # same-breath control: a TRUTHY carrier DOES pass the kwarg, so a shim that
    # cannot take it fails -- proving the guard above is not vacuous.
    with pytest.raises(TypeError):
        kw = {"clip_ids": ["a"]} if ["a"] else {}
        old_shim(1, None, seed=0, **kw)


def test_the_frames_stay_synthetic_this_makes_the_JOIN_real_not_the_pixels():
    """⚠️ A guard against the worst possible misreading of the change above."""
    from tanitad.refs import refc_v3 as v3
    tr = _trainer()
    cfg = v3.refc_v3_smoke_config(hier=True)
    a = tr._synth_episodes(1, cfg.core, seed=0, clip_ids=["aaaa-1111"])[0]
    b = tr._synth_episodes(1, cfg.core, seed=0, clip_ids=["zzzz-9999"])[0]
    assert a.episode_id != b.episode_id          # the id follows the clip...
    assert a.frames.shape == b.frames.shape
    assert bool((a.frames == b.frames).all())    # ...the pixels do NOT
    assert math.isfinite(float(a.poses.sum()))
