"""``tanitad.refs.anchor_twoseg`` — the vocabulary can express a lane change.

⛔ WHY THIS FILE EXISTS. MEASURED 2026-09-06 (commit ``56dc078``): all 117
refcv4b candidates are constant-curvature arcs, **yaw monotone on 1.000000 of
564,291 (window, candidate) pairs**, so none is an S-shape — while **1,297 of
18,615** windows over **79 of 141** eval clips execute a lane-change shape. A
vocabulary is a CEILING. This module adds the cheapest family that contains one.

⛔⛔ AND PARITY IS SACRED. Adding candidates changes the fan every banked arm was
scored against, so the extension is a FLAG that DEFAULTS OFF, and the OFF path is
pinned here BY COMPARISON — not asserted:

(a) LIMIT — ``t_split_s = horizon_s`` never flips, so the two-segment integrator
    IS the constant integrator, ``torch.equal`` over a 0-36 m/s speed sweep. With
    a MUTATION control: a split INSIDE the horizon must DIFFER, or the check is
    vacuous. *(An AST census once read 0 suspects on BOTH the fixed and the
    broken trainer.)*
(b) AGAINST THE LIVE DECODER — the 2-column path is ``torch.equal`` to
    ``refc.AnchoredDiffusionDecoder.roll_bank``, so this is the incumbent's own
    arithmetic rather than a second implementation that happens to agree.
(c) STRUCTURAL ZERO — every default split is at t >= 2.0 s, so the 2 s scored
    grid (slots 0-3) is bit-identical. ⛔ That is also the honest scope statement:
    a capability gained here CANNOT appear in ``ade_0_2s``.
(d) BUILDER — ``scripts/build_twoseg_anchors.py`` without ``--two-segment``
    reproduces its source's tensors sha256-identically, and ``--assert-parity``
    refuses otherwise.
(e) DECLARATION — a 3-column ``controls`` file that declares no
    ``control_schedule`` is REFUSED, the same rule one column to the right of the
    units one that ``anchor_meta`` exists for (396 g vs 0.31 g).
(f) KAMM — with a MUTATION control, because a feasibility gate that cannot FAIL
    is not a gate.

CPU-only, no data, no checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.models.kinematic import rollout_unicycle           # noqa: E402
from tanitad.refs import anchor_meta as am                      # noqa: E402
from tanitad.refs import anchor_twoseg as ts                    # noqa: E402
from tanitad.refs import refc_v3 as v3                          # noqa: E402

HZ = v3.V3_HORIZONS                     # (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [h - 1 for h in HZ]
HORIZON_S = max(HZ) * 0.1               # 6.0
SPEEDS = torch.tensor([0.0, 1.0, 4.0, 8.0, 10.0, 16.0, 22.0, 30.0, 36.0])
#: the re-roll constants. ``ROLL`` is what an INTEGRATOR takes; ``CONST`` adds
#: ``ref_speed_ms``, which only the ARTIFACT declares. Keeping them apart is the
#: same discipline as the units: a constant handed to the wrong consumer is how
#: a number that looks like an answer gets produced.
ROLL = dict(kappa_cap=0.12, alat_v_floor=4.0)
CONST = dict(ref_speed_ms=10.0, **ROLL)


def _controls(n: int = 24) -> torch.Tensor:
    """[N, 2] (a_lon, a_lat) with the pinned straight-ahead control present."""
    g = torch.Generator().manual_seed(7)
    c = torch.rand(n, 2, generator=g) * torch.tensor([6.0, 6.0]) - 3.0
    c[0] = 0.0
    return c


def _anchors(n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return torch.randn(n, len(HZ), 2, generator=g)


# --------------------------------------------------------------- the family --
def test_default_family_is_six_and_ordering_is_fixed():
    """⛔ The COUNT is MEASURED, not chosen (out_p1h_two_segment.json): 24 new
    candidates buy 0.0009 m more than 6, and one split point loses 81.5 % of the
    gain even with all eight magnitudes. And the ORDER is a contract: the index
    of a candidate is a class label, so a reordering would permute a trained
    head's logits silently."""
    c = ts.two_segment_controls()
    assert tuple(c.shape) == (6, 3)
    assert c[:, 0].abs().max() == 0.0                     # a_lon = 0 on all six
    assert sorted({float(x) for x in c[:, 1]}) == [-0.75, 0.75]
    assert sorted({float(x) for x in c[:, 2]}) == [2.0, 3.0, 4.0]
    # (t_split outer, a_lat inner) — pinned literally
    assert [(round(float(r[1]), 4), round(float(r[2]), 4)) for r in c] == [
        (-0.75, 2.0), (0.75, 2.0), (-0.75, 3.0),
        (0.75, 3.0), (-0.75, 4.0), (0.75, 4.0)]


def test_two_segment_controls_refuses_an_empty_family():
    with pytest.raises(ValueError, match="EMPTY"):
        ts.two_segment_controls(a_lat=(), t_split_s=(2.0,))


def test_extend_is_append_only():
    """⛔ APPEND, never interleave: a checkpoint trained on the incumbent names
    candidates by integer index."""
    c = _controls(17)
    e = ts.extend_controls(c, HORIZON_S)
    assert tuple(e.shape) == (17 + 6, 3)
    assert torch.equal(e[:17, :2], c)
    assert float(e[:17, 2].min()) == float(e[:17, 2].max()) == HORIZON_S


# ------------------------------------------------------------------ (a) LIMIT
def test_single_segment_limit_is_bit_identical():
    """t_split = horizon_s never flips ⇒ the two-segment roll IS the constant
    roll. With the MUTATION control that makes the assertion non-vacuous."""
    c = _controls()
    base = ts.roll_bank(c, SPEEDS, control_units="alat", steps=max(HZ),
                        slots=SLOTS, **ROLL)
    lim = ts.roll_bank(ts.as_three_column(c, HORIZON_S), SPEEDS,
                       control_units="alat", steps=max(HZ), slots=SLOTS,
                       **ROLL)
    assert torch.equal(base, lim), "the single-segment limit is not the limit"

    # ⛔ MUTATION CONTROL — a split INSIDE the horizon MUST differ, or the
    # equality above would hold for a roll that ignores column 2 entirely.
    mid = ts.as_three_column(c, HORIZON_S)
    mid[:, 2] = 3.0
    moved = ts.roll_bank(mid, SPEEDS, control_units="alat", steps=max(HZ),
                         slots=SLOTS, **ROLL)
    assert not torch.equal(base, moved)
    assert float((base - moved).abs().max()) > 1.0

    # and a split BEYOND the horizon is the limit too (>= , not ==)
    beyond = ts.as_three_column(c, HORIZON_S)
    beyond[:, 2] = 99.0
    assert torch.equal(base, ts.roll_bank(beyond, SPEEDS,
                                          control_units="alat",
                                          steps=max(HZ), slots=SLOTS, **ROLL))


def test_kappa_units_limit_is_bit_identical_too():
    """The same limit under ``control_units='kappa'`` — the other unit domain
    must not have its own arithmetic."""
    c = _controls() * 0.02                       # plausible curvatures
    base = ts.roll_bank(c, SPEEDS, control_units="kappa", steps=max(HZ),
                        slots=SLOTS, **ROLL)
    lim = ts.roll_bank(ts.as_three_column(c, HORIZON_S), SPEEDS,
                       control_units="kappa", steps=max(HZ), slots=SLOTS,
                       **ROLL)
    assert torch.equal(base, lim)


# --------------------------------------------------- (b) AGAINST THE DECODER
def test_two_column_path_matches_the_live_decoder_roll_bank():
    """⛔ The claim is that this is the INCUMBENT's arithmetic, so it is checked
    against ``refc.AnchoredDiffusionDecoder.roll_bank`` itself — not against a
    hand-written copy of the same formula, which would agree with a shared bug."""
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(True)
    cfg.core.anchors.v0_conditioned = True
    cfg.core.anchors.control_units = "alat"
    cfg.core.anchors.ref_speed_ms = CONST["ref_speed_ms"]
    cfg.core.anchors.kappa_cap = CONST["kappa_cap"]
    cfg.core.anchors.alat_v_floor_ms = CONST["alat_v_floor"]
    dec = v3.RefCV3Model(cfg).core.decoder
    n = dec.anchors.shape[0]
    ctrl = _controls(n)
    dec.load_anchors(dec.anchors.clone(), ctrl)

    v = SPEEDS[: 5].clone()
    keep = torch.ones(v.shape[0], dtype=torch.bool)
    want = dec.roll_bank(v, keep, v.shape[0], torch.float32)
    got = ts.roll_bank(ctrl, v, control_units="alat",
                       steps=int(dec.anchor_roll_steps),
                       slots=dec.anchor_slots.tolist(),
                       dt=float(dec.anchor_dt),
                       alat_v_floor=float(dec.anchor_alat_v_floor),
                       kappa_cap=float(dec.anchor_kappa_cap))
    assert got.shape == want.shape
    assert torch.equal(got, want), (
        "anchor_twoseg.roll_bank is not the decoder's own arithmetic; "
        f"max |diff| = {float((got - want).abs().max()):.3e}")


def test_roll_matches_an_independent_integration():
    """A third, deliberately naive implementation — so a shared refactor cannot
    move both the module and the decoder together."""
    c = torch.tensor([[0.0, 1.5, 2.0]])
    v = torch.tensor([12.0])
    got = ts.roll_bank(c, v, control_units="alat", steps=60, slots=SLOTS,
                       **ROLL)
    kap = min(max(1.5 / max(12.0, 4.0) ** 2, -0.12), 0.12)
    seq = torch.tensor([[0.0, kap if k < 20 else -kap] for k in range(60)])
    s0 = torch.zeros(1, 4)
    s0[0, 3] = 12.0
    want = rollout_unicycle(s0, seq[None], dt=0.1)[..., :2][:, SLOTS]
    assert torch.allclose(got, want, atol=1e-6)


# ---------------------------------------------------- (c) THE STRUCTURAL ZERO
def test_2s_prefix_is_a_structural_zero_for_the_default_family():
    """Every default split is at t >= 2.0 s and slot 3 IS 2.0 s, so no default
    candidate can differ from a constant arc on the 2 s scored grid.
    ⛔ This is also the SCOPE statement: the capability cannot show up in
    ``ade_0_2s``, and that is reported, not buried."""
    c = _controls()
    ext = ts.extend_controls(c, HORIZON_S)
    s2 = SLOTS[:4]
    a = ts.roll_bank(ts.as_three_column(c, HORIZON_S), SPEEDS,
                     control_units="alat", steps=max(HZ), slots=s2, **ROLL)
    b = ts.roll_bank(ext[: c.shape[0]], SPEEDS, control_units="alat",
                     steps=max(HZ), slots=s2, **ROLL)
    assert torch.equal(a, b)
    # each NEW candidate equals its own constant counterpart over the 2 s grid
    new = ext[c.shape[0]:]
    flat = new.clone()
    flat[:, 2] = HORIZON_S                       # the same rows, never flipping
    assert torch.equal(
        ts.roll_bank(new, SPEEDS, control_units="alat", steps=max(HZ),
                     slots=s2, **ROLL),
        ts.roll_bank(flat, SPEEDS, control_units="alat", steps=max(HZ),
                     slots=s2, **ROLL))
    # ...and MUST differ over the full 6 s grid, or the family is inert
    assert not torch.equal(
        ts.roll_bank(new, SPEEDS, control_units="alat", steps=max(HZ),
                     slots=SLOTS, **ROLL),
        ts.roll_bank(flat, SPEEDS, control_units="alat", steps=max(HZ),
                     slots=SLOTS, **ROLL))


# ------------------------------------------------------------- the tick guard
def test_split_off_the_tick_grid_is_refused():
    """``k * dt`` is not exact in binary, so a float comparison at the boundary
    is a coin-flip; the split is snapped and an unsnappable one is REFUSED."""
    with pytest.raises(ts.SplitOffTick, match="rounding"):
        ts.split_ticks(torch.tensor([2.03]), 0.1, 60)
    assert ts.split_ticks(torch.tensor([0.0, 2.0, 6.0, 99.0]), 0.1,
                          60).tolist() == [0, 20, 60, 60]


# --------------------------------------------------------------- (f) THE KAMM
def test_kamm_passes_the_family_and_can_still_fail():
    """0 violations for the default family — and a MUTATION control proving the
    gate can fire. ⚠️ A zero here is only meaningful beside the manoeuvre rate;
    MEASURED, one zero-violation result was bought by a ``turn_left`` recall of
    exactly 0.0000."""
    speeds = (0.0, 4.0, 10.0, 20.0, 30.0, 36.0)
    ok = ts.kamm_report(ts.two_segment_controls(), speeds, mu=0.7, **ROLL)
    assert ok["n_violations"] == 0
    assert ok["violating_candidates"] == []
    assert not ok["kappa_cap_reached"]
    assert ok["peak_total_g"] == pytest.approx(0.75 / 9.81, rel=1e-3)

    bad = ts.kamm_report(torch.tensor([[9.0, 9.0, 3.0]]), speeds, mu=0.7,
                         **ROLL)
    assert bad["n_violations"] > 0 and bad["violating_candidates"] == [0]


def test_kamm_reports_the_REALISED_alat_not_the_declared_one():
    """⛔ The clamp binds at LOW speed, not high — ``kappa = a_lat / max(v,
    v_floor)^2`` is LARGEST when the floor binds — and there the candidate stops
    delivering the ``a_lat`` its column claims. Scoring the declared value would
    describe a vehicle the integrator never drives, which is the units incident
    in a kinematics costume.

    At ``v = 4`` (the floor) a declared ``a_lat = 3.0`` asks for
    ``kappa = 3.0 / 16 = 0.1875``, is clamped to ``0.12``, and REALISES
    ``0.12 * 16 = 1.92 m/s^2`` — 64 % of what the column says.
    ⭐ MEASURED consequence: the incumbent 117-candidate bank DOES reach the cap
    at low speed; the two-segment family (|a_lat| = 0.75 ⇒ kappa <= 0.0469)
    never does, which is why its declared and realised values coincide."""
    r = ts.kamm_report(torch.tensor([[0.0, 3.0]]), (4.0,), mu=0.7, **ROLL)
    assert r["kappa_cap_reached"]
    assert r["peak_abs_kappa_inv_m"] == pytest.approx(0.12)
    assert r["peak_abs_a_lat_realised_ms2"] == pytest.approx(1.92)
    # at HIGH speed the same declared value is NOT clamped: 3.0 / 36^2 = 0.0023
    hi = ts.kamm_report(torch.tensor([[0.0, 3.0]]), (36.0,), mu=0.7, **ROLL)
    assert not hi["kappa_cap_reached"]
    assert hi["peak_abs_a_lat_realised_ms2"] == pytest.approx(3.0, rel=1e-5)


# -------------------------------------------------------- (e) THE DECLARATION
def test_artifact_declares_the_schedule_and_reads_it_back():
    a = _anchors(6 + 4)
    c = ts.extend_controls(_controls(4), HORIZON_S)
    art = am.build_anchor_artifact(
        a, c, control_units="alat", horizons=HZ, dt=0.1,
        control_schedule=am.TWO_SEGMENT_SCHEDULE, builder=__file__, **CONST)
    assert art["control_schedule"] == am.TWO_SEGMENT_SCHEDULE
    assert art["controls_columns"] == ["a_lon_ms2", "a_lat_ms2", "t_split_s"]
    assert art["n_two_segment"] == 6 and art["n_constant"] == 4
    assert art["t_split_s_values"] == [2.0, 3.0, 4.0]
    read = am.read_anchor_artifact(art)
    assert read.control_schedule == am.TWO_SEGMENT_SCHEDULE
    assert "schedule=two_segment_alat_flip" in am.describe(read)


def test_a_three_column_file_that_declares_nothing_is_refused():
    """⛔ The units rule, one column to the right. There is no override here:
    no legacy 3-column file exists, so a silent guess would invent one."""
    a = _anchors(4)
    c = ts.as_three_column(_controls(4), HORIZON_S)
    with pytest.raises(am.AnchorScheduleMissing, match="control_schedule"):
        am.read_anchor_artifact({"anchors": a, "controls": c,
                                 "control_units": "alat"})
    # ...and a 2-column file that declares nothing is STILL fine: `constant` is
    # a fact about the shape, not a guess.
    two = am.read_anchor_artifact({"anchors": a, "controls": _controls(4),
                                   "control_units": "alat"})
    assert two.control_schedule == am.CONSTANT_SCHEDULE


def test_schedule_and_column_count_must_agree():
    a = _anchors(4)
    with pytest.raises(am.AnchorScheduleConflict, match="control_schedule"):
        am.read_anchor_artifact(
            {"anchors": a, "controls": _controls(4), "control_units": "alat",
             "control_schedule": am.TWO_SEGMENT_SCHEDULE})
    with pytest.raises(am.AnchorScheduleConflict, match="not one of"):
        am.read_anchor_artifact(
            {"anchors": a, "controls": _controls(4), "control_units": "alat",
             "control_schedule": "wishful"})
    with pytest.raises(ValueError, match=r"controls must be \[4, 3\]"):
        am.build_anchor_artifact(
            a, _controls(4), control_units="alat", horizons=HZ,
            control_schedule=am.TWO_SEGMENT_SCHEDULE, builder=__file__,
            **CONST)


def test_a_fixed_path_artifact_declares_NO_schedule():
    """⛔ present-but-``None`` where it does not apply, like the three re-roll
    constants. A fixed-path file has no ``controls``, so a schedule would be a
    statement about a tensor the file does not hold — and `build_refc_anchors.py`
    writes exactly such a file."""
    art = am.build_anchor_artifact(_anchors(4), None,
                                   control_units=am.PATHS_ONLY, horizons=HZ,
                                   builder=__file__)
    assert art["control_schedule"] is None
    read = am.read_anchor_artifact(art)
    assert read.control_units == am.PATHS_ONLY
    assert "schedule=" not in am.describe(read)
    # ...and a fixed-path file that DOES declare one is refused
    with pytest.raises(am.AnchorScheduleConflict, match="no `controls`"):
        am.read_anchor_artifact({"anchors": _anchors(4),
                                 "control_units": am.PATHS_ONLY,
                                 "control_schedule": am.CONSTANT_SCHEDULE})


def test_legacy_two_column_artifacts_are_untouched():
    """⛔ PARITY: a 2-column build declares ``constant``, keeps the two-name
    ``controls_columns``, and its tensor sha256s are the tensors' own."""
    a, c = _anchors(4), _controls(4)
    art = am.build_anchor_artifact(a, c, control_units="alat", horizons=HZ,
                                   builder=__file__, **CONST)
    assert art["control_schedule"] == am.CONSTANT_SCHEDULE
    assert art["controls_columns"] == ["a_lon_ms2", "a_lat_ms2"]
    assert "n_two_segment" not in art
    assert art["controls_sha256"] == am.sha256_of_tensor(c)
    assert art["anchors_sha256"] == am.sha256_of_tensor(a)
    assert am.read_anchor_artifact(art).control_schedule == am.CONSTANT_SCHEDULE


# ------------------------------------------------------------- (d) THE BUILDER
def _write_source(tmp_path, n=9):
    """⛔ A COHERENT source: `anchors` really IS `controls` rolled at
    `ref_speed_ms`, as the live refcv4b file is (verified 2026-09-06 — rolling
    its own `anchor_controls` at 10.0 m/s reproduces its `anchors` sha256
    `51f930dc…a66df`). A source whose two tensors disagree would trip the
    builder's append-only guard, which is the guard doing its job."""
    c = _controls(n)
    a = ts.roll_bank(c, torch.tensor([CONST["ref_speed_ms"]]),
                     control_units="alat", steps=max(HZ), slots=SLOTS,
                     **ROLL)[0]
    p = tmp_path / "src.pt"
    torch.save(am.build_anchor_artifact(a, c, control_units="alat",
                                        horizons=HZ, builder=__file__,
                                        **CONST), p)
    return p, am.sha256_of_tensor(a), am.sha256_of_tensor(c)


def test_builder_default_off_is_bit_identical(tmp_path):
    """⛔⛔ THE PARITY BAR. Without ``--two-segment`` the written tensors are the
    source's, proved by sha256 — the flag must be inert, not nearly inert."""
    import build_twoseg_anchors as B
    src, sa, sc = _write_source(tmp_path)
    out = tmp_path / "off.pt"
    assert B.main(["--from-anchors", str(src), "--out", str(out),
                   "--assert-anchors-sha", sa, "--assert-controls-sha", sc,
                   "--assert-parity"]) == 0
    got = am.read_anchor_artifact(str(out))
    assert got.controls.shape[1] == 2
    assert got.control_schedule == am.CONSTANT_SCHEDULE
    assert am.sha256_of_tensor(got.anchors) == sa
    assert am.sha256_of_tensor(got.controls) == sc


def test_builder_two_segment_appends_and_keeps_the_prefix(tmp_path):
    import build_twoseg_anchors as B
    src, sa, sc = _write_source(tmp_path)
    out = tmp_path / "on.pt"
    assert B.main(["--from-anchors", str(src), "--out", str(out),
                   "--two-segment"]) == 0
    got = am.read_anchor_artifact(str(out))
    assert tuple(got.controls.shape) == (9 + 6, 3)
    assert got.control_schedule == am.TWO_SEGMENT_SCHEDULE
    assert am.sha256_of_tensor(got.controls[:9, :2]) == sc
    assert am.sha256_of_tensor(got.anchors[:9].contiguous()) == sa
    assert got.meta["kamm"]["n_violations"] == 0
    assert got.meta["n_base_candidates"] == 9


def test_builder_refuses_the_incoherent_requests(tmp_path):
    import build_twoseg_anchors as B
    src, sa, sc = _write_source(tmp_path)
    out = tmp_path / "x.pt"
    with pytest.raises(SystemExit, match="meaningless"):
        B.main(["--from-anchors", str(src), "--out", str(out),
                "--two-segment", "--assert-parity"])
    with pytest.raises(SystemExit, match="MISMATCH"):
        B.main(["--from-anchors", str(src), "--out", str(out),
                "--assert-controls-sha", "0" * 64])
    with pytest.raises(SystemExit, match="INCONCLUSIVE"):
        B.main(["--from-anchors", str(src), "--out", str(out),
                "--assert-controls-sha", "deadbeef"])
    # extending an already-extended bank would append the family twice
    ext = tmp_path / "on2.pt"
    B.main(["--from-anchors", str(src), "--out", str(ext), "--two-segment"])
    with pytest.raises(SystemExit, match="already declares"):
        B.main(["--from-anchors", str(ext), "--out", str(out), "--two-segment"])
    # a split off the tick grid
    with pytest.raises(ts.SplitOffTick):
        B.main(["--from-anchors", str(src), "--out", str(out), "--two-segment",
                "--t-split", "2.03"])
