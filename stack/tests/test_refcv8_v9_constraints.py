"""refcv8 Q2 (MM ruling, SPEC_REFCV8 sec. 13a; MM item 2, 2026-10-04) -- the v9 CONSTRAINT vectors are SUPERVISED, and
MM item 4 (Q8) -- R8-1-REACH's in-run key.

The v9 release carries per-frame constraint vectors (`lat_c` 12, `lon_c` 10, `speed_goal` 4: turn start / end time and
distance, stop distance, lead gap and time gap, target speed and time / distance to reach, ...). refcv8 now regresses
them with per-QUERY heads on the behaviour decoder (`--w-r8-v9-cons`), on the GT-active class's query, masked where the
v9 row is PARTIAL or absent, where a field is undefined, and (turn fields) where the class is not a TURN.

Pinned here:
1. the field order is the reader's, the scales and the clamp are the registered literals;
2. the targets: normalisation by hand, the masks (partial / absent / NaN / turn-only), the clamp;
3. the loss reads ONLY the GT-active query, and an all-masked batch is an attached zero;
4. ANALYTIC CONTROL (learning, the real decoder module): a synthetic STOP at distance d -- d visible in the scene --
   must be regressed (held-out R^2 >= 0.9 and MAE <= 3 m, bars fixed before the first run);
5. DELIBERATE REGRESSION: the same with the targets SHUFFLED across windows must not learn (held-out R^2 <= 0.1);
6. the rig: the heads exist iff the weight is on, the term reaches the heads AND the decoder's shared layers, a built
   head with no target in training refuses, the tac roll carries the constraint targets with the classes;
7. the dataset emits the three vectors from the join;
8. Q8: `tactical_rows` literals (exact or non-empty partial, lateral AND longitudinal);
9. the argv: dead flag, no release refused, the effective-weight gate, G-DVB, G-LIVE.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
import test_refcv8_v9_wiring as W9  # noqa: E402
from tanitad.data import v9_labels as V9  # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

NAN = float("nan")
BRAKE_TO, CRUISE, TURN_L, LANE_KEEP = 3, 1, 6, 0


# =========================================================================== #
# 1. field order, scales, clamp                                                #
# =========================================================================== #
def test_the_fields_are_the_readers_and_the_scales_are_the_registered_literals():
    assert C.V9_LAT_FIELDS == V9.LAT_CONSTRAINTS and C.V9_LON_FIELDS == V9.LON_CONSTRAINTS
    assert C.V9_SPEED_FIELDS == V9.SPEED_GOAL
    assert C.V9_LAT_SCALE == (90.0, 8.0, 8.0, 50.0, 50.0, 90.0, 50.0, 50.0, 50.0, 90.0, 8.0, 50.0)
    assert C.V9_LON_SCALE == (15.0, 8.0, 50.0, 8.0, 50.0, 50.0, 50.0, 8.0, 50.0, 8.0)
    assert C.V9_SPEED_SCALE == (15.0,) * 4 and C.V9_CLAMP == 6.0
    assert C.V9_TURN_COLS == (1, 2, 3, 4, 5, 6, 7, 8, 9) and C.V9_TURN_V7 == (6, 7)
    # INTEGRATION's rule restated independently: the unit is the field name's suffix
    unit = {"deg": 90.0, "s": 8.0, "m": 50.0, "ms": 15.0}
    for f, s in zip(C.V9_LAT_FIELDS + C.V9_LON_FIELDS + C.V9_SPEED_FIELDS,
                    C.V9_LAT_SCALE + C.V9_LON_SCALE + C.V9_SPEED_SCALE):
        assert unit[f.rsplit("_", 1)[1]] == s, f


# =========================================================================== #
# 2. the targets and masks                                                     #
# =========================================================================== #
def _raw(b):
    lat = torch.full((b, 12), NAN)
    lon = torch.full((b, 10), NAN)
    sp = torch.full((b, 4), NAN)
    return lat, lon, sp


def test_the_targets_are_normalised_by_hand_and_masked_by_the_rules():
    lat, lon, sp = _raw(4)
    lat[:, 0] = 45.0                       # Theta 45 deg on every row
    lat[:, 4] = 100.0                      # a turn length on every row
    lon[:, 4] = 20.0                       # stop_x 20 m on every row
    lon[:, 7] = 1000.0                     # a lead time gap far beyond the clamp
    sp[:, 0] = 7.5
    lat_cls = torch.tensor([TURN_L, LANE_KEEP, -100, TURN_L])
    lon_cls = torch.tensor([BRAKE_TO, -100, BRAKE_TO, CRUISE])
    t = C.v9_constraint_targets(lat, lon, sp, lat_cls, lon_cls)
    assert t["lat"][0, 0] == pytest.approx(0.5) and t["lat"][0, 4] == pytest.approx(2.0)
    assert t["lon"][0, 4] == pytest.approx(0.4) and t["speed"][0, 0] == pytest.approx(0.5)
    assert t["lon"][0, 7] == pytest.approx(6.0)                             # clamped (1000 s / 8 -> 6)
    assert float(t["lat"][0, 1]) == 0.0                                     # a NaN target is a 0, masked
    # lateral: exact class + finite; turn fields only on a TURN row; a PARTIAL row (-100) supervises nothing
    assert t["lat_m"][0].tolist() == [True, False, False, False, True] + [False] * 7
    assert t["lat_m"][1].tolist() == [True] + [False] * 11                   # LANE_KEEP: Theta yes, turn length no
    assert not bool(t["lat_m"][2].any())                                     # PARTIAL / absent
    # longitudinal + speed: exact lon class + finite
    assert t["lon_m"][0, 4] and not bool(t["lon_m"][1].any()) and t["lon_m"][2, 4]
    assert t["speed_m"][0, 0] and not bool(t["speed_m"][1].any())
    assert not bool(t["lon_m"][0, 0])                                        # NaN field: masked


# =========================================================================== #
# 3. the loss reads the GT-active query only                                    #
# =========================================================================== #
def test_the_loss_reads_only_the_gt_active_query_and_an_all_masked_batch_is_an_attached_zero():
    lat, lon, sp = _raw(2)
    lon[:, 4] = 20.0
    t = C.v9_constraint_targets(lat, lon, sp, torch.tensor([-100, -100]), torch.tensor([BRAKE_TO, BRAKE_TO]))
    pl = torch.zeros(2, 8, 12, requires_grad=True)
    po = torch.zeros(2, 8, 10)
    po[:, BRAKE_TO, 4] = 0.4                                                  # the GT query holds the exact target
    po = po.requires_grad_()
    ps = torch.zeros(2, 8, 4, requires_grad=True)
    loss, tele = C.v9_constraint_loss(pl, po, ps, t)
    assert float(loss) == 0.0 and tele["n_v9_lon_c"] == 2 and tele["n_v9_lat_c"] == 0
    po2 = po.detach().clone()
    po2[:, CRUISE, 4] = 9.0                                                   # a NON-GT query: no effect
    assert float(C.v9_constraint_loss(pl, po2, ps, t)[0]) == 0.0
    po3 = po.detach().clone()
    po3[:, BRAKE_TO, 4] = 0.9                                                 # the GT query: the Huber value
    l3 = float(C.v9_constraint_loss(pl, po3, ps, t)[0])
    assert l3 == pytest.approx(0.5 - 0.5 * C.V9_HUBER_BETA, abs=1e-6)
    none = C.v9_constraint_targets(*_raw(2), torch.tensor([-100, -100]), torch.tensor([-100, -100]))
    z, _ = C.v9_constraint_loss(pl, po, ps, none)
    assert float(z) == 0.0 and z.requires_grad                                # attached, never guarded away


# =========================================================================== #
# 4 + 5. the analytic control and its deliberate regression (the real module)   #
# =========================================================================== #
def _decoder(seed=0):
    torch.manual_seed(seed)
    dec = v6tac.TacticalBehaviourDecoder(v6tac.TacticalDecoderConfig(
        d_model=32, n_layers=1, n_heads=4, ff_mult=2, d_agent=8, d_bev=0, sources=("agent",)))
    dec.attach_refcv8_heads(10, v9_dims=(C.V9_LAT_DIMS, C.V9_LON_DIMS, C.V9_SPEED_DIMS))
    return dec


def _stop_batch(g, b):
    d = torch.rand(b, generator=g) * 75.0 + 5.0                              # a STOP at 5 ... 80 m
    agents = torch.randn(b, 3, 8, generator=g) * 0.1
    agents[:, 0, 0] = d / 50.0                                                # the scene carries d
    cond = v6tac.build_condition(torch.eye(4)[torch.zeros(b, dtype=torch.long)], torch.zeros(b, 4),
                                 torch.full((b, 1), 5.0), torch.zeros(b, 1), v_scale=30.0, a_scale=3.0)
    lat, lon, sp = _raw(b)
    lon[:, 4] = d                                                             # stop_x = d
    lon[:, 5] = 0.0                                                           # stop_y = 0
    return cond, agents, lat, lon, sp, d


def _fit(shuffle: bool, steps=300, b=64):
    dec = _decoder()
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    lat_cls = torch.full((b,), -100)
    lon_cls = torch.full((b,), BRAKE_TO)
    for _ in range(steps):
        cond, ag, lat, lon, sp, _d = _stop_batch(g, b)
        if shuffle:                                                           # another window's targets
            lon = lon[torch.randperm(b, generator=g)]
        out = dec(cond, agent_tokens=ag, cond_extra=torch.zeros(b, 10))
        t = C.v9_constraint_targets(lat, lon, sp, lat_cls, lon_cls)
        loss, _ = C.v9_constraint_loss(out["v9_lat_c"], out["v9_lon_c"], out["v9_speed"], t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dec.eval()
    gh = torch.Generator().manual_seed(99)
    cond, ag, lat, lon, sp, d = _stop_batch(gh, 256)
    with torch.no_grad():
        pred = dec(cond, agent_tokens=ag, cond_extra=torch.zeros(256, 10))["v9_lon_c"][:, BRAKE_TO, 4] * 50.0
    r2 = 1.0 - float(((pred - d) ** 2).sum() / ((d - d.mean()) ** 2).sum())
    return r2, float((pred - d).abs().mean())


def test_ANALYTIC_a_synthetic_stop_at_d_is_regressed_to_d():
    r2, mae = _fit(shuffle=False)
    assert r2 >= 0.9 and mae <= 3.0, (r2, mae)


def test_DELIBERATE_REGRESSION_shuffled_constraint_targets_do_not_learn():
    r2, mae = _fit(shuffle=True)
    assert r2 <= 0.1, (r2, mae)


# =========================================================================== #
# 6. the rig                                                                   #
# =========================================================================== #
@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("timm")
    T = R.trainer()
    cfg, m = R.build(T, True, v9_cons=True, w_v9_cons=0.05)
    bt = R.batch(cfg)
    b = bt["frames"].shape[0]
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    return T, cfg, m, bt, b, traj


def _cons_batch(b):
    lat, lon, sp = _raw(b)
    lat[:, 0] = 30.0
    lon[:, 0] = 8.0
    sp[:, :] = 6.0
    return {"r8_v9_lat_c": lat, "r8_v9_lon_c": lon, "r8_v9_speed": sp,
            "lat_v7": torch.tensor([LANE_KEEP, TURN_L, -100][:b]), "lon_v7": torch.tensor([CRUISE, -100, BRAKE_TO][:b]),
            "lat_allowed_v7": torch.tensor([[1] + [0] * 7, [0] * 6 + [1, 0], [1, 0, 0, 0, 0, 0, 1, 1]][:b]).bool(),
            "lon_allowed_v7": torch.tensor([[0, 1] + [0] * 6, [0] * 8, [0, 0, 0, 1] + [0] * 4][:b]).bool()}


def test_the_heads_exist_iff_asked_and_the_term_trains_the_heads_and_the_shared_decoder(rig):
    T, cfg, m, bt, b, traj = rig
    td = m.tac_decoder_v6
    assert td.r8_v9_lat_c is not None and td.r8_v9_lon_c.out_features == 10 and td.r8_v9_speed.out_features == 4
    _c0, m0 = R.build(T, True)
    assert m0.tac_decoder_v6.r8_v9_lat_c is None                             # the default build is unchanged
    m.train()
    m._r8_rc_dropout, m._r8_rc_noise, m._r8_nav_args_dropout = 0.3, (2.0, 0.75), 0.5
    prep = W9._prep(m, bt, b, traj, _cons_batch(b))
    out = R.forward(cfg, m, bt, train=True, **prep["fwd"])
    m.zero_grad()
    total, tele = RT.r8_losses(m, out, prep, traj, torch.ones(b, 8, dtype=torch.bool))
    assert "r8_v9_cons" in tele and tele["n_v9_lat_c"] == 2 and tele["n_v9_lon_c"] == 2       # partial rows masked
    lv, _ = C.v9_constraint_loss(out["r8_v9_lat_c"], out["r8_v9_lon_c"], out["r8_v9_speed"], prep["v9c"])
    lv.backward()
    assert float(td.r8_v9_lon_c.weight.grad.abs().sum()) > 0.0
    shared = [p for n, p in td.named_parameters() if n.startswith("layers.") and p.grad is not None]
    assert shared and sum(float(p.grad.abs().sum()) for p in shared) > 0.0  # the tactical trunk is shaped too
    assert tele["r8v9_tac_rows"] == pytest.approx(2 / 3)                     # Q8 key in the telemetry


def test_a_built_head_with_no_target_in_training_refuses(rig):
    T, cfg, m, bt, b, traj = rig
    m.train()
    prep = W9._prep(m, bt, b, traj, {})
    out = R.forward(cfg, m, bt, train=True, **prep["fwd"])
    with pytest.raises(SystemExit, match="r8_v9_cons"):
        RT.r8_losses(m, out, prep, traj, torch.ones(b, 8, dtype=torch.bool))
    m.eval()
    with torch.no_grad():
        out = R.forward(cfg, m, bt, train=False, **W9._prep(m, bt, b, traj, {})["fwd"])
    RT.r8_losses(m, out, W9._prep(m, bt, b, traj, {}), traj, torch.ones(b, 8, dtype=torch.bool))   # eval: allowed


def test_the_tac_roll_carries_the_constraint_targets_with_the_classes():
    assert {"r8_v9_lat_c", "r8_v9_lon_c", "r8_v9_speed"} <= set(RT.ROLL_FAMILIES["tac"])
    bt = _cons_batch(3)
    bt["r8_v9_lon_c"][:, 4] = torch.tensor([1.0, 2.0, 3.0])
    r = RT.roll_targets(bt, "tac")
    assert r["r8_v9_lon_c"][:, 4].tolist() == [3.0, 1.0, 2.0] and r["lon_v7"].tolist() == [BRAKE_TO, CRUISE, -100]


# =========================================================================== #
# 7. the dataset                                                                #
# =========================================================================== #
def test_the_dataset_emits_the_three_vectors_from_the_join(tmp_path):
    T = R.trainer()
    cfg = T.v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(1, cfg.core, seed=0, min_frames=140)
    sid = 616161
    eps[0].episode_id = sid
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    g0, dt = 0.113, 0.1007
    ds._label_clock = {sid: (g0, dt, "test")}
    off = ds._raw_offset(eps[0])
    n = int(eps[0].poses.shape[0])

    def mut(rows):
        k = len(rows["k"])
        for f in V9.LAT_CONSTRAINTS + V9.LON_CONSTRAINTS + V9.SPEED_GOAL:
            rows[f] = np.full(k, np.nan, np.float32)
        rows["stop_x_m"] = np.arange(k, dtype=np.float32)
        rows["v_a_ms"] = np.full(k, 4.5, np.float32)
    p, md5 = W9._write(tmp_path, sid, n=n, k0=off, g0=g0, dt=dt, mutate=mut, name="cons")
    j = RT.load_v9_join(p, expect_md5=md5)
    ds.enable_r8_v9(j)
    it0 = ds[0]
    assert "r8_v9_lat_c" not in it0                                           # off by default
    ds.r8_v9_cons = True
    it = ds[0]
    r = int(ds._r8_rows[0])
    assert tuple(it["r8_v9_lat_c"].shape) == (12,) and tuple(it["r8_v9_lon_c"].shape) == (10,)
    assert float(it["r8_v9_lon_c"][4]) == float(j.rel.rows["stop_x_m"][r]) and float(it["r8_v9_speed"][0]) == 4.5
    assert bool(torch.isnan(it["r8_v9_lat_c"]).all())


# =========================================================================== #
# 8. Q8: R8-1-REACH's in-run key                                               #
# =========================================================================== #
def test_Q8_tactical_rows_counts_exact_or_partial_on_BOTH_axes():
    lat = torch.tensor([0, -100, -100, 6, -100])
    lat_al = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1], [0, 1]]).bool()
    lon = torch.tensor([1, 3, 1, -100, -100])
    lon_al = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 0], [1, 0]]).bool()
    # row 0 exact/exact, 1 partial/exact, 2 ABSENT lat, 3 exact/ABSENT lon, 4 partial/partial -> 3 of 5
    assert RT.tactical_rows(lat, lat_al, lon, lon_al) == pytest.approx(0.6)
    assert RT.tactical_rows(torch.tensor([-100]), torch.zeros(1, 2).bool(), torch.tensor([1]), torch.ones(1, 2).bool()) == 0.0


# =========================================================================== #
# 9. the argv                                                                  #
# =========================================================================== #
T = None


def _pin(argv):
    tr = R.trainer()
    args = tr.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=C.R8Config())
    tr._pin_refcv8(cfg, args)
    return cfg.refcv8


ON = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05"]


def test_the_argv_pins_the_weight_and_refuses_a_dead_or_unsupervised_head():
    r = _pin(ON + ["--r8-v9-labels", "v9.npz", "--w-r8-v9-cons", "0.05"])
    assert r.v9_cons is True and r.w_v9_cons == 0.05
    assert _pin(ON).v9_cons is False
    with pytest.raises(SystemExit, match="without --r8-v9-labels"):
        _pin(ON + ["--w-r8-v9-cons", "0.05"])
    with pytest.raises(SystemExit, match="without --refcv8"):
        _pin(["--arm", "hier", "--out", "X", "--w-r8-v9-cons", "0.05"])
    tr = R.trainer()
    assert {a.dest: a.default for a in tr.build_parser()._actions}["w_r8_v9_cons"] == 0.0     # the zero default
    assert "w_r8_v9_cons" in tr.REFC_WEIGHT_GATES


def test_the_effective_weight_audit_and_G_LIVE_know_the_term():
    tr = R.trainer()
    ap = tr.build_parser()

    def audit(argv):
        a = ap.parse_args(argv)
        a._ew_parser, a._ew_argv = ap, list(argv)
        try:
            tr.check_effective_weights(a)
            return None
        except SystemExit as e:
            return str(e)
    assert audit(ON + ["--r8-v9-labels", "v9.npz", "--w-r8-v9-cons", "0.05"]) is None
    got = audit(ON + ["--w-r8-v9-cons", "0.05"])                             # the gate is shut (no v9 release)
    assert got is not None and "--w-r8-v9-cons" in got
    import launch_gate as LG
    a = ap.parse_args(ON + ["--r8-v9-labels", "v9.npz", "--w-r8-v9-cons", "0.05"])
    a._ew_parser, a._ew_argv = ap, []
    terms, _p = LG.declared_terms(tr, a)
    assert {t["id"]: t["keys"] for t in terms}.get("w_r8_v9_cons") == ["r8_v9_cons"]
    assert "--w-r8-v9-cons" in LG.PROFILES["refcv8"]["required_positive"]


def test_G_DVB_checks_the_heads_against_the_weight():
    from tanitad.train import declared_vs_built as dvb
    assert dvb.REGISTRY["w_r8_v9_cons"].kind == "loss"
