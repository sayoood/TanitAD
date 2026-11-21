"""v7F R1 EQUIVALENT FOR refcv8 -- the max speed "must not be exceeded" holds on the EMITTED plan.

PI R1 (2026-09-27): *"both of them should get the max speed, which it must not be exceeded."* v7F built a HARD
cap on the emitted plan (`plan_speed_cap.cap_fan`). refcv8 has the INPUT (`--max-speed-input-v6`) and an ARGMAX
FILTER (`--speed-ceiling-filter`: `refc.py:3941`, `refc_v3.e9_rank`) -- and the filter only CHOOSES among the fan.
`SpeedCeilingFilter` keeps the whole fan on a row with no compliant candidate (`refcv6_selection.py`, "the ONLY
reason the obedience bar is 99 % and not 100 %"), and the existing refcv8 test
(`test_refcv8_speed_input.py::test_the_EMITTED_plan_obeys...`) therefore asserts obedience on `picked[~exempt]` --
the EXEMPT rows are outside every refcv8 bound. `--plan-vmax-clamp` closes them: a projection of the SELECTED plan,
inference-only, bit-identical on non-binding rows, default OFF.

THE CHAIN, one hop per test (flag -> namespace -> cfg -> model -> forward):
  argv `--plan-vmax-clamp` -> `refc_v3_train.py` argparse -> `_pin_trainer_cfg` sets `cfg.plan_vmax_clamp`
  -> `RefCV3Model.cfg` -> scene hook exports `plan_v_lim_ms` -> E9 block `cap_plan_to_limit` -> `out["traj"]`.

DELIBERATE-REGRESSION ARMS (each must go RED, and does): the clamp replaced by identity; the limit export removed;
the G-DVB entry's built value flipped; the pin's refusal removed.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
for _p in (str(_HERE), str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv6_selection as v6sel  # noqa: E402
from tanitad.refs.refcv6_selection import cap_plan_to_limit, planned_max_speed  # noqa: E402

HOR = (5, 10, 15, 20)            # waypoint TICKS at 0.1 s -> 0.5 s apart (NOT 0.1 s)


def _straight(speeds):
    """A straight plan along +x whose interval i runs at speeds[i] m/s (0.5 s intervals)."""
    return torch.cumsum(torch.tensor([[s * 0.5, 0.0] for s in speeds]), 0)[None]


# =========================================================================== #
# 1. the pure function, LITERAL expectations                                   #
# =========================================================================== #
def test_literal_a_binding_plan_is_compressed_to_the_ceiling():
    out, d = cap_plan_to_limit(_straight([10, 20, 30, 40]), torch.tensor([25.0]), horizons=HOR)
    # intervals 5, 10, 15, 20 m -> allowed 12.5 m each -> 5, 10, 12.5, 12.5 -> cumulative 5, 15, 27.5, 40
    assert out[0, :, 0].tolist() == pytest.approx([5.0, 15.0, 27.5, 40.0], abs=1e-5)
    assert out[0, :, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert float(d["plan_cap_speed_before"]) == pytest.approx(40.0, abs=1e-5)
    assert float(d["plan_cap_speed_after"]) == pytest.approx(25.0, abs=1e-4)
    assert bool(d["plan_cap_binding"][0]) and float(d["plan_cap_excess_ms"]) <= 1e-4


def test_literal_headings_are_kept():
    tr = torch.tensor([[[6.0, 0.0], [12.0, 2.0], [18.0, 8.0], [20.0, 20.0]]])
    out, _ = cap_plan_to_limit(tr, torch.tensor([6.0]), horizons=HOR)     # 6 m/s -> 3 m per 0.5 s interval
    seq = torch.cat([torch.zeros(1, 1, 2), out], 1)
    d_new = seq[:, 1:] - seq[:, :-1]
    seq0 = torch.cat([torch.zeros(1, 1, 2), tr], 1)
    d_old = seq0[:, 1:] - seq0[:, :-1]
    assert torch.allclose(d_new.norm(dim=-1), torch.full((1, 4), 3.0), atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(d_new, d_old, dim=-1)
    assert torch.allclose(cos, torch.ones(1, 4), atol=1e-5)               # same direction per interval


def test_literal_non_binding_and_infinite_rows_are_BIT_identical():
    tr = torch.cat([_straight([5, 5, 5, 5]), _straight([10, 20, 30, 40]), _straight([10, 20, 30, 40])])
    lim = torch.tensor([25.0, float("inf"), 25.0])
    out, d = cap_plan_to_limit(tr, lim, horizons=HOR)
    assert torch.equal(out[0], tr[0]) and torch.equal(out[1], tr[1])      # under the ceiling / no ceiling
    assert not torch.equal(out[2], tr[2])                                 # the binding control moved
    assert d["plan_cap_binding"].tolist() == [False, False, True]


def test_literal_a_ceiling_exactly_at_the_plan_max_is_the_identity():
    tr = _straight([10, 20, 30, 40])
    out, d = cap_plan_to_limit(tr, torch.tensor([40.0]), horizons=HOR)
    assert torch.equal(out, tr) and not bool(d["plan_cap_binding"][0])


def test_literal_v0_above_the_ceiling_brakes_and_is_REPORTED():
    # v0 = 20 m/s, ceiling 10, brake 2.5 m/s^2: allowed speed in interval i is max(10, 20 - 2.5 * t_mid_i),
    # t_mid = 0.25, 0.75, 1.25, 1.75 s -> 19.375, 18.125, 16.875, 15.625 m/s -> metres 9.6875, 9.0625, 8.4375, 7.8125
    tr = _straight([25, 25, 25, 25])
    out, d = cap_plan_to_limit(tr, torch.tensor([10.0]), horizons=HOR, v0=torch.tensor([20.0]), brake_ms2=2.5)
    got = torch.diff(torch.cat([torch.zeros(1), out[0, :, 0]]))
    assert got.tolist() == pytest.approx([9.6875, 9.0625, 8.4375, 7.8125], abs=1e-4)
    assert bool(d["plan_cap_v0_over_limit"][0])
    strict, d2 = cap_plan_to_limit(tr, torch.tensor([10.0]), horizons=HOR, v0=torch.tensor([20.0]), brake_ms2=0.0)
    assert float(planned_max_speed(strict[:, None], horizons=HOR)[0, 0]) == pytest.approx(10.0, abs=1e-4)
    assert "plan_cap_v0_over_limit" not in d2                              # strict mode has no allowance to report


def test_property_the_cap_holds_on_random_plans_under_the_CEIL_OBEY_statistic():
    g = torch.Generator().manual_seed(7)
    tr = torch.cumsum(torch.randn(256, 4, 2, generator=g) * 6.0, 1)
    lim = torch.where(torch.rand(256, generator=g) < 0.2, torch.full((256,), float("inf")),
                      4.0 + 30.0 * torch.rand(256, generator=g))
    out, _ = cap_plan_to_limit(tr, lim, horizons=HOR)
    vm = planned_max_speed(out[:, None], horizons=HOR)[:, 0]
    finite = torch.isfinite(lim)
    assert bool((vm[finite] <= lim[finite] * (1 + 1e-5)).all())
    v0 = planned_max_speed(tr[:, None], horizons=HOR)[:, 0]
    assert int((v0[finite] > lim[finite]).sum()) > 20                     # NOT VACUOUS: many plans were over
    assert torch.equal(out[~finite], tr[~finite])


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -3.0])
def test_a_non_limit_REFUSES(bad):
    with pytest.raises(ValueError, match="not a speed limit"):
        cap_plan_to_limit(_straight([5, 5, 5, 5]), torch.tensor([bad]), horizons=HOR)


def test_a_mismatched_grid_REFUSES():
    with pytest.raises(ValueError, match="mismatched grid"):
        cap_plan_to_limit(_straight([5, 5, 5, 5]), torch.tensor([9.0]), horizons=(5, 10, 15))


# =========================================================================== #
# 2. the REAL RefCV3Model: the filter alone leaves exempt rows over the ceiling #
# =========================================================================== #
V0_GRID = [9.0, 10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 13.8]


def _speed_batch(cfg, v0s, seed=4):
    b = len(v0s)
    enc = cfg.core.encoder
    h, w = enc.image_hw()
    W = int(cfg.core.window)
    g = torch.Generator().manual_seed(seed)
    v_s = torch.tensor(v0s, dtype=torch.float32)
    t = torch.arange(W, dtype=torch.float32)
    ph = torch.zeros(b, W, 4)
    ph[:, :, 3] = v_s[:, None]
    ph[:, :, 0] = v_s[:, None] * 0.1 * t[None]
    return {"frames": torch.rand(b, W, int(enc.in_channels), h, w, generator=g),
            "nav_cmd": torch.zeros(b, dtype=torch.long), "v0": v_s.clone(), "pose_hist": ph}


@pytest.fixture(scope="module")
def rig():
    T = R.trainer()
    cfg, m = R.build(T, True, vmax=True, speed_input="n2", n_alloc=8, alloc_emit=True)
    return T, cfg, m


def _run(rig, kmh, clamp, train=False, valid=1.0):
    """``kmh``: one ceiling for every row, or a per-row list."""
    T, cfg, m = rig
    cfg.plan_vmax_clamp = bool(clamp)
    bt = _speed_batch(cfg, V0_GRID)
    b = len(V0_GRID)
    ks = list(kmh) if isinstance(kmh, (list, tuple)) else [kmh] * b
    out = R.forward(cfg, m, bt, train=train, v_max_ms=torch.tensor([k / 3.6 for k in ks]),
                    v_max_valid=torch.full((b,), float(valid)))
    dec = m.core.decoder
    vm = planned_max_speed(out["traj"][:, None], horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)[:, 0]
    return out, vm


#: rows 0-3 sit under a 30 km/h ceiling while the ego drives 9-12 m/s (every candidate is rolled from v0, so the
#: fan has nothing compliant: EXEMPT rows); rows 4-7 sit under a 120 km/h ceiling (never binding)
MIXED = [30, 30, 30, 30, 120, 120, 120, 120]


def test_the_filter_alone_leaves_EXEMPT_rows_over_the_ceiling_and_the_clamp_closes_them(rig):
    """The decisive A/B: SAME windows, SAME sampler seed, clamp OFF vs ON.

    On this rig an exempt row IS an ego already above the ceiling (every candidate is rolled from v0, so a
    constant-speed anchor is compliant whenever v0 <= ceiling). v7F's rule for that case applies unchanged: the
    plan brakes at `a_max` until under the ceiling, and the residual is REPORTED (`plan_cap_v0_over_limit`), not
    hidden -- so the bound on such a row is the braking envelope, not the ceiling."""
    T, cfg, m = rig
    lim = torch.tensor([k / 3.6 for k in MIXED])
    v0 = torch.tensor(V0_GRID)
    off, vm_off = _run(rig, MIXED, clamp=False)
    on, vm_on = _run(rig, MIXED, clamp=True)
    exempt = vm_off > lim + 1e-5                                          # the emitted plan is over the ceiling
    assert bool(exempt.any()), "no exempt row -- the A/B would prove nothing (raise v0 or lower the ceiling)"
    assert bool((~exempt).any()), "no non-binding row -- the bit-identity clause would be vacuous"
    dec = m.core.decoder
    brake = float(dec.sel.accel_max)
    env = torch.maximum(lim, v0 - brake * (0.5 * float(dec.anchor_horizons[0]) * float(dec.anchor_dt)))
    # (1) with the clamp ON no row exceeds its envelope; a row with v0 <= ceiling is bounded by the CEILING itself
    assert bool((vm_on <= env * (1 + 1e-4)).all()), (vm_on.tolist(), env.tolist())
    strict = v0 <= lim
    assert bool((vm_on[strict] <= lim[strict] * (1 + 1e-4)).all())
    # (2) the clamp MOVED every exempt row (the filter alone had left them above the ceiling) and only those
    assert bool((vm_on[exempt] < vm_off[exempt]).all()), (vm_on.tolist(), vm_off.tolist())
    assert torch.equal(on["traj"][~exempt], off["traj"][~exempt])         # bit-identical where it is not needed
    # (3) the residual is reported, per v7F: exactly the rows whose ego is above the ceiling
    assert on["plan_cap_v0_over_limit"].tolist() == (v0 > lim).tolist()
    # (4) the clamp changed ONLY the plan: the candidate, its index and the pre-clamp plan are the filter's
    assert torch.equal(on["sel_idx"], off["sel_idx"]) and torch.equal(on["traj_precap"], off["traj"])
    assert "traj_precap" not in off and "plan_cap_binding" not in off


def test_the_clamp_is_inference_only_and_the_default_is_off(rig):
    T, cfg, m = rig
    assert [f for f in dataclasses.fields(type(cfg)) if f.name == "plan_vmax_clamp"][0].default is False
    tr, _ = _run(rig, MIXED, clamp=True, train=True)
    # TRAINING forward: the clamp did not run (no pre-clamp plan, no diagnostics) and the emitted plan is the raw
    # fan member. (Not a cross-run equality: the refcv8 seams draw from a DEDICATED generator that advances.)
    assert "traj_precap" not in tr and "plan_cap_binding" not in tr
    ar = torch.arange(tr["traj"].shape[0])
    assert torch.equal(tr["traj"], tr["anchor_traj"][ar, tr["sel_idx"]])


def test_an_unknown_ceiling_row_is_never_capped(rig):
    _o, vm = _run(rig, MIXED, clamp=True, valid=0.0)                          # channel fed but unknown on every row
    _f, vm_off = _run(rig, MIXED, clamp=False, valid=0.0)
    assert torch.equal(_o["traj"], _f["traj"]) and torch.equal(vm, vm_off)  # +inf = no cap, no fabricated limit


def test_REGRESSION_ARM_the_clamp_replaced_by_identity_goes_RED(rig, monkeypatch):
    """The obedience assertion of the A/B above, run with the clamp neutered: it must FAIL."""
    monkeypatch.setattr(v6sel, "cap_plan_to_limit",
                        lambda traj, lim, **kw: (traj, {"plan_cap_binding": torch.zeros(traj.shape[0], dtype=torch.bool)}))
    T, cfg, m = rig
    lim = torch.tensor([k / 3.6 for k in MIXED])
    v0 = torch.tensor(V0_GRID)
    dec = m.core.decoder
    env = torch.maximum(lim, v0 - float(dec.sel.accel_max) * (0.5 * float(dec.anchor_horizons[0]) * float(dec.anchor_dt)))
    _on, vm_on = _run(rig, MIXED, clamp=True)
    assert not bool((vm_on <= env * (1 + 1e-4)).all()), "the neutered clamp still obeyed -- the A/B proves nothing"


def test_REGRESSION_ARM_a_missing_limit_export_REFUSES_instead_of_silently_not_running(rig, monkeypatch):
    """The hook exports `plan_v_lim_ms` only while the flag reads True AT HOOK TIME; flip it off for the core
    forward and on again for the E9 block: the clamp finds no limit and must REFUSE, not skip."""
    T, cfg, m = rig
    real = m.core.forward

    def flip(*a, **k):
        cfg.plan_vmax_clamp = False
        try:
            return real(*a, **k)
        finally:
            cfg.plan_vmax_clamp = True
    monkeypatch.setattr(m.core, "forward", flip)
    with pytest.raises(ValueError, match="exported no limit"):
        _run(rig, MIXED, clamp=True)


# =========================================================================== #
# 3. the BUILD guards                                                          #
# =========================================================================== #
def test_build_refuses_a_clamp_with_no_ceiling_channel():
    T = R.trainer()
    cfg, _m = R.build(T, False)                                           # vmax=False: no max_speed_onehot_v6
    cfg.plan_vmax_clamp = True
    with pytest.raises(ValueError, match="INERT"):
        T.v3.RefCV3Model(cfg)


def test_build_refuses_a_clamp_beside_the_refcv7_scorer_pick():
    T = R.trainer()
    cfg, _m = R.build(T, False, vmax=True)
    cfg.plan_vmax_clamp = True
    cfg.refcv7_select = True
    with pytest.raises(ValueError, match="refcv7_select"):
        T.v3.RefCV3Model(cfg)


# =========================================================================== #
# 4. the TRAINER chain: argv -> namespace -> cfg -> G-DVB                        #
# =========================================================================== #
BASE = ["--arm", "hier", "--size", "tiny", "--out", "X", "--v7-labels", "labels.jsonl.gz",
        "--agents", "head", "--w-agent", "1.0", "--agent-join", "join.jsonl.xz", "--agent-join-verify", "off",
        "--tac-decoder-v6", "--w-tac-v6", "1.0", "--max-speed-input-v6", "--speed-max-sidecar-v6", "s.jsonl"]


def _pin(argv):
    T = R.trainer()
    args = T.build_parser().parse_args(argv)
    cfg = T.v3.RefCV3Config(hier=(args.arm == "hier"))
    try:
        T._pin_trainer_cfg(cfg, args)
    except SystemExit as e:
        return False, str(e), cfg, args
    return True, "", cfg, args


def test_GREEN_the_flag_reaches_the_cfg():
    ok, msg, cfg, args = _pin(BASE + ["--plan-vmax-clamp"])
    assert ok, msg
    assert args.plan_vmax_clamp is True and cfg.plan_vmax_clamp is True


def test_the_default_is_off_as_a_LITERAL():
    ok, msg, cfg, args = _pin(BASE)
    assert ok, msg
    assert args.plan_vmax_clamp is False and cfg.plan_vmax_clamp is False


def test_RED_the_flag_without_the_channel_refuses():
    argv = [t for t in BASE if t not in ("--max-speed-input-v6",)]
    i = argv.index("--speed-max-sidecar-v6")
    del argv[i:i + 2]
    ok, msg, _c, _a = _pin(argv + ["--plan-vmax-clamp"])
    assert not ok and "--plan-vmax-clamp without --max-speed-input-v6" in msg, msg[:200]


def test_GDVB_has_an_entry_and_reads_the_BUILT_value(rig):
    from tanitad.train import declared_vs_built as dvb
    T, cfg, m = rig
    parser = T.build_parser()
    assert "plan_vmax_clamp" in dvb.REGISTRY and dvb.coverage(parser) == []
    args = parser.parse_args(["--arm", "hier", "--out", "X", "--plan-vmax-clamp"])
    cfg.plan_vmax_clamp = False                                            # the model was BUILT without it
    got = [x for x in dvb.check(m, args, parser) if x.lever == "--plan-vmax-clamp"]
    assert len(got) == 1 and got[0].declared is True and got[0].built is False       # declared-but-not-built: RED
    cfg.plan_vmax_clamp = True
    assert [x for x in dvb.check(m, args, parser) if x.lever == "--plan-vmax-clamp"] == []   # built: clean
    cfg.plan_vmax_clamp = False


def test_REGRESSION_ARM_the_gdvb_entry_removed_goes_RED(monkeypatch):
    from tanitad.train import declared_vs_built as dvb
    T = R.trainer()
    monkeypatch.delitem(dvb.REGISTRY, "plan_vmax_clamp")
    assert "plan_vmax_clamp" in dvb.coverage(T.build_parser())            # the gate refuses a flag with no entry
