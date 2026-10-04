"""refcv8 WP-B -- the v9 label release wired into the TRAINER (WP-A INTEGRATION.md; MM binding 2026-10-04 items 1-6).

What is pinned here, each with a literal expectation and, where a check could pass vacuously, a deliberate regression:

1. the input scaling (``scale_nav`` / ``scale_rc``) on hand-computed values; the model widths agree (6 + 4);
2. the route-checkpoint TRAINING NOISE in the route-tangent frame -- per-axis sigmas measured on 200k draws at
   psi = 0 and psi = 90 deg (a swapped-axis noise goes RED) -- and the dedicated generator (the global stream is
   untouched);
3. the training treatment in ``r8_before_forward``: RC dropout >= 0.3 and noise >= the certified sigmas are REFUSED
   below; nav-args dropout clears the known bit and the args, never the token; the LEGAL eval row; ``--r8-no-rc``;
4. ``partial_label_loss`` / ``v9_partial_correction`` on literal logits (an exact row IS cross-entropy);
5. ``v9_integrity_census``: clean = all zero; each contract violation (LC exact action, reversing with an action or a
   geometry goal, an absence claim on a short band, SPEED_BAND supervised) is COUNTED -- one mutation per row;
6. the join on the trainer's own clock through ``V3Dataset.enable_r8_v9`` / ``__getitem__``: k = t + w - 1 +
   raw_offset, a 1 ms clock shift is REFUSED at enable time, the v9 targets replace v7.2's, a G3-excluded clip stays
   IGNORED;
7. the NavSim legal mapping (INTEGRATION sec. 5) as a literal table, and the v9 nav token -> nav_cmd index;
8. the REAL releases (dev box: D:/Projects/TanitAD-artifacts/v9labels): md5 + a census with ZERO violations.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.data import v9_labels as V9  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

REAL = Path("D:/Projects/TanitAD-artifacts/v9labels")


# =========================================================================== #
# a synthetic release, written by THIS test (independent of WP-A's builder)   #
# =========================================================================== #
def _rows(n, k0, g0, dt):
    k = np.arange(n) + k0
    r = {"k": k.astype(np.int16), "now_s": g0 + k * dt, "h_obs_s": np.full(n, 8.0), "reversing": np.zeros(n, np.int8),
         "lat_cls_a": np.zeros(n, np.int16), "lat_cls_b": np.zeros(n, np.int16),
         "lat_allowed_a": np.full(n, 1, np.uint8), "lat_allowed_b": np.full(n, 1, np.uint8),
         "lat_v7id_a": np.zeros(n, np.int16), "lat_v7id_b": np.zeros(n, np.int16),          # LANE_KEEP exact
         "lat_allowed_v7_a": np.full(n, 1, np.uint8), "lat_allowed_v7_b": np.full(n, 1, np.uint8),
         "lon_cls": np.full(n, -100, np.int16), "lon_allowed": np.full(n, 0b1001, np.uint8),
         "lon_v7id": np.full(n, -100, np.int16), "lon_allowed_v7": np.full(n, (1 << 5) | (1 << 0), np.uint8),
         "goal_y": np.full(n, 0b1, np.uint32), "goal_w": np.full(n, 0b1, np.uint32),      # FOLLOW_LANE positive
         "nav_token": np.full(n, 1.0), "nav_side_next": np.full(n, 1.0), "nav_d_next_m": np.full(n, 50.0),
         "nav_d_end_m": np.full(n, 100.0), "nav_dyaw_next_deg": np.full(n, 45.0), "nav_args_valid": np.ones(n),
         "nav_lookahead_m": np.full(n, 2000.0), "nav_t_next_s": np.full(n, 3.0), "nav_token_ttime": np.ones(n)}
    for v in V9.RC_VARIANTS:
        r[f"rc_{v}_x"], r[f"rc_{v}_y"], r[f"rc_{v}_psi"] = np.full(n, 25.0), np.full(n, -10.0), np.full(n, 90.0)
        r[f"rc_{v}_valid"] = np.ones(n)
    return r


def _write(tmp_path, sid, n=40, k0=2, g0=0.113, dt=0.1007, mutate=None, name="train"):
    rows = _rows(n, k0, g0, dt)
    if mutate:
        mutate(rows)
    clips = {"sid": np.array([sid], np.int64), "sha12": np.array(["abcdef012345"]), "row0": np.array([0]),
             "n_rows": np.array([n]), "k0": np.array([k0])}
    p = tmp_path / f"v9_labels_{name}.npz"
    np.savez_compressed(p, **{"row__" + a: b for a, b in rows.items()}, **{"clip__" + a: b for a, b in clips.items()})
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    (tmp_path / f"v9_labels_{name}.manifest.json").write_text(
        json.dumps({"schema": V9.SCHEMA, "split": name, "npz_md5": md5}), encoding="utf-8")
    return str(p), md5


# =========================================================================== #
# 1. scaling and widths                                                        #
# =========================================================================== #
def test_scale_nav_and_scale_rc_on_hand_computed_values():
    raw = torch.tensor([[50.0, 100.0, 45.0, 1.0, 2000.0], [50.0, 100.0, 45.0, 1.0, 2000.0]])
    got = RT.scale_nav(raw, torch.tensor([True, False]))
    want = [math.log(2.0), math.log(3.0), 0.5, 1.0, math.log(21.0), 1.0]      # lookahead clipped at 1000 m -> 20
    assert got[0].tolist() == pytest.approx(want, abs=1e-6)
    assert got[1].tolist() == [0.0] * 6                                       # unknown: exactly zeros next to a 0
    rc = RT.scale_rc(torch.tensor([[25.0, -10.0, 90.0], [25.0, -10.0, 90.0]]), torch.tensor([True, False]))
    assert rc[0].tolist() == pytest.approx([0.5, -0.2, 1.0, 1.0], abs=1e-7)
    assert rc[1].tolist() == [0.0] * 4


def test_the_model_reads_the_widths_the_trainer_writes():
    from tanitad.refs import refc_v3 as v3
    assert (v3.R8_NAV_DIMS, v3.R8_RC_DIMS) == (6, 4)
    assert (RT.NAV_IN_DIMS, RT.RC_IN_DIMS) == (6, 4)


# =========================================================================== #
# 2. the route-checkpoint training noise                                       #
# =========================================================================== #
def _noise_stats(psi_deg, sa=2.0, sl=0.75, n=200_000):
    from tanitad.refs import refcv8_conditioning as C
    g = C.R8Generator(7)
    raw = torch.zeros(n, 3)
    raw[:, 2] = psi_deg
    out = RT.rc_training_noise(raw, torch.ones(n, dtype=torch.bool), g, sigma_along_m=sa, sigma_lat_m=sl)
    return float(out[:, 0].std()), float(out[:, 1].std()), out


def _noise_is_in_the_tangent_frame(sx0, sy0, sx90, sy90) -> bool:
    return (abs(sx0 - 2.0) < 0.03 and abs(sy0 - 0.75) < 0.012 and abs(sx90 - 0.75) < 0.012 and abs(sy90 - 2.0) < 0.03)


def test_the_noise_is_along_and_lateral_in_the_route_tangent_frame():
    sx0, sy0, out0 = _noise_stats(0.0)
    sx90, sy90, out90 = _noise_stats(90.0)
    assert _noise_is_in_the_tangent_frame(sx0, sy0, sx90, sy90), (sx0, sy0, sx90, sy90)
    assert torch.equal(out0[:, 2], torch.zeros(len(out0)))                    # psi is not noised
    # DELIBERATE REGRESSION: the sigmas swapped (lateral 2 m, along 0.75 m) must fail the same check
    a, b, _ = _noise_stats(0.0, sa=0.75, sl=2.0)
    c, d, _ = _noise_stats(90.0, sa=0.75, sl=2.0)
    assert not _noise_is_in_the_tangent_frame(a, b, c, d)


def test_invalid_rows_are_untouched_and_the_global_stream_is_not_consumed():
    from tanitad.refs import refcv8_conditioning as C
    raw = torch.zeros(8, 3)
    valid = torch.tensor([True, False] * 4)
    st = torch.get_rng_state()
    out = RT.rc_training_noise(raw, valid, C.R8Generator(1))
    assert torch.equal(torch.get_rng_state(), st)
    assert torch.equal(out[~valid], torch.zeros(4, 3)) and bool((out[valid, :2] != 0).all())


# =========================================================================== #
# 3. the training treatment in r8_before_forward                               #
# =========================================================================== #
@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("timm")
    T = R.trainer()
    cfg8, m8 = R.build(T, True, n_alloc=4)
    bt = R.batch(cfg8)
    b = bt["frames"].shape[0]
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    fut = torch.zeros(b, 60, 4)
    fut[:, :, 3] = 6.0
    return cfg8, m8, bt, b, traj


def _prep(m, bt, b, traj, batch):
    return RT.r8_before_forward(m, batch, "cpu", traj, torch.ones(b, 8, dtype=torch.bool), bt["pose_hist"][:, -1],
                                torch.zeros(b, 60, 4), torch.ones(b, 60, dtype=torch.bool), bt["v0"])


def _v9_batch(b):
    return {"r8_nav_raw": torch.tensor([[50.0, 100.0, 45.0, 1.0, 200.0]] * b),
            "r8_nav_known": torch.ones(b, dtype=torch.bool),
            "r8_rc_raw": torch.tensor([[25.0, -10.0, 0.0]] * b), "r8_rc_valid": torch.ones(b, dtype=torch.bool)}


def test_eval_feeds_the_clean_scaled_inputs_and_the_legal_row_drops_both(rig):
    cfg8, m8, bt, b, traj = rig
    m8.eval()
    m8._r8_legal_row = False
    f = _prep(m8, bt, b, traj, _v9_batch(b))["fwd"]
    assert f["r8_rc"][0].tolist() == pytest.approx([0.5, -0.2, 0.0, 1.0], abs=1e-7)    # no noise at eval
    assert f["r8_nav"][:, -1].tolist() == [1.0] * b
    m8._r8_legal_row = True
    f = _prep(m8, bt, b, traj, _v9_batch(b))["fwd"]
    assert f["r8_nav"].abs().sum() == 0 and f["r8_rc"].abs().sum() == 0              # NavSim LEGAL: known 0, RC invalid
    m8._r8_legal_row = False


def test_training_drops_and_noises_with_the_dedicated_generator_only(rig):
    cfg8, m8, bt, b, traj = rig
    m8.train()
    m8._r8_nav_args_dropout, m8._r8_rc_dropout, m8._r8_rc_noise = 1.0, 0.3, (2.0, 0.75)
    st = torch.get_rng_state()
    f = _prep(m8, bt, b, traj, _v9_batch(b))["fwd"]
    assert torch.equal(torch.get_rng_state(), st)                                     # the refcv7 stream untouched
    assert f["r8_nav"].abs().sum() == 0                                               # every row's args 'unknown'
    kept = f["r8_rc"][:, -1] == 1.0
    if bool(kept.any()):                                                              # kept rows are NOISED, never clean
        assert not torch.allclose(f["r8_rc"][kept, :2], torch.tensor([0.5, -0.2]).expand(int(kept.sum()), 2))
    m8._r8_rc_dropout = 1.0
    f = _prep(m8, bt, b, traj, _v9_batch(b))["fwd"]
    assert f["r8_rc"].abs().sum() == 0


@pytest.mark.parametrize("attr,val,needle", [
    ("_r8_rc_dropout", 0.29, "< 0.3"),
    ("_r8_rc_noise", (1.0, 0.75), "CERTIFIED"),
    ("_r8_rc_noise", (2.0, 0.5), "CERTIFIED"),
])
def test_below_the_binding_values_the_training_step_refuses(rig, attr, val, needle):
    cfg8, m8, bt, b, traj = rig
    m8.train()
    m8._r8_rc_dropout, m8._r8_rc_noise, m8._r8_nav_args_dropout = 0.3, (2.0, 0.75), 0.5
    setattr(m8, attr, val)
    with pytest.raises(SystemExit) as e:
        _prep(m8, bt, b, traj, _v9_batch(b))
    assert needle in str(e.value)
    m8._r8_rc_dropout, m8._r8_rc_noise = 0.3, (2.0, 0.75)


def test_the_no_rc_switch_never_feeds_the_checkpoint(rig):
    cfg8, m8, bt, b, traj = rig
    m8.eval()
    m8._r8_no_rc = True
    f = _prep(m8, bt, b, traj, _v9_batch(b))["fwd"]
    assert "r8_rc" not in f and "r8_nav" in f
    m8._r8_no_rc = False


# =========================================================================== #
# 4. the partial-label loss                                                    #
# =========================================================================== #
def test_partial_label_loss_literals_and_the_exact_row_is_cross_entropy():
    z = torch.zeros(3, 4, requires_grad=True)
    al = torch.tensor([[True, True, False, False], [False, False, True, False], [False] * 4])
    loss, n = RT.partial_label_loss(z, al)
    assert n == 2
    assert float(loss) == pytest.approx((math.log(2.0) + math.log(4.0)) / 2.0, abs=1e-6)   # -log(1/2), -log(1/4)
    loss.backward()
    assert z.grad[2].abs().sum() == 0                                                  # an all-false row: nothing
    z2 = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    one = torch.zeros(1, 4, dtype=torch.bool)
    one[0, 1] = True
    assert float(RT.partial_label_loss(z2, one)[0]) == pytest.approx(float(torch.nn.functional.cross_entropy(
        z2, torch.tensor([1]))), abs=1e-6)


def test_the_correction_turns_the_exact_ce_into_the_combined_partial_mean():
    from tanitad.refs import refcv6_tactical as v6tac
    torch.manual_seed(0)
    lat, lon = torch.randn(4, 8), torch.randn(4, 8)
    lat_t = torch.tensor([6, -100, 0, -100])
    lon_t = torch.tensor([-100, 1, -100, -100])
    lat_al = torch.zeros(4, 8, dtype=torch.bool)
    lat_al[0, 6] = lat_al[2, 0] = True
    lat_al[1, [0, 1]] = True                                                           # partial {LANE_KEEP, LC_L}
    lon_al = torch.zeros(4, 8, dtype=torch.bool)
    lon_al[1, 1] = True
    lon_al[0, [0, 5]] = True
    W = v6tac.TacticalLossWeights()
    corr, tele = RT.v9_partial_correction(lat, lon, lat_t, lon_t, lat_al, lon_al, W)
    ce = lambda z, t: RT._exact_ce(z, t, -100)                                        # noqa: E731
    total = W.lat_ce * ce(lat, lat_t) + W.lon_ce * ce(lon, lon_t) + corr
    want = W.lat_ce * RT.partial_label_loss(lat, lat_al)[0] + W.lon_ce * RT.partial_label_loss(lon, lon_al)[0]
    assert float(total) == pytest.approx(float(want), abs=1e-6)
    assert (tele["r8v9_n_lat_partial"], tele["r8v9_n_lon_partial"]) == (1.0, 1.0)


# =========================================================================== #
# 5. the release contract census                                               #
# =========================================================================== #
def _census(tmp_path, mutate=None, name="c"):
    p, md5 = _write(tmp_path, 7, mutate=mutate, name=name)
    return RT.v9_integrity_census(V9.load_v9_release(p))["violations"]


def test_a_clean_release_has_zero_violations(tmp_path):
    assert set(_census(tmp_path).values()) == {0}


def _set(field, idx, value):
    def f(rows):
        rows[field][idx] = value
    return f


def _both(*fs):
    def f(rows):
        for g in fs:
            g(rows)
    return f


@pytest.mark.parametrize("mutate,key", [
    (_both(_set("lat_v7id_a", 3, 1), _set("lat_allowed_v7_a", 3, 1 << 1)), "lc_exact_action"),
    (_set("reversing", 5, 1), "reversing_with_action_label"),
    (_both(_set("reversing", 6, 1), _set("lat_allowed_v7_a", 6, 0), _set("lon_allowed_v7", 6, 0)),
     "reversing_with_geometry_goal"),
    (_set("h_obs_s", 7, 5.0), "absence_class_on_short_band"),
    (_both(_set("h_obs_s", 8, 5.0), _set("lat_v7id_a", 8, -100), _set("lat_allowed_v7_a", 8, 0b11)),
     "absence_goal_on_short_band"),
    (_set("lat_allowed_v7_a", 9, 0b11), "exact_label_mask_mismatch"),
    (_set("goal_w", 10, 1 | (1 << 7)), "speed_band_supervised"),
])
def test_each_contract_violation_is_counted(tmp_path, mutate, key):
    v = _census(tmp_path, mutate, name=key)
    assert v[key] >= 1, v
    with pytest.raises(SystemExit):
        p, md5 = _write(tmp_path, 7, mutate=mutate, name=key + "_l")
        RT.load_v9_join(p, expect_md5=md5)


def test_a_turn_negative_entailed_by_the_other_side_is_not_an_absence_claim(tmp_path):
    """TURN_L negative on a short band: a violation when nothing turns, NOT when TURN_R is positive (entailed)."""
    tl, tr = 1 << 1, 1 << 2                                                   # GOAL22 TURN_L / TURN_R bits

    def lone(rows):                                                           # TURN_L negative, no turn at all
        rows["h_obs_s"][4] = 6.0
        rows["goal_y"][4], rows["goal_w"][4] = 0, tl
    assert _census(tmp_path, lone, name="lone")["absence_goal_on_short_band"] == 1

    def entailed(rows):                                                       # TURN_L negative beside TURN_R positive
        rows["h_obs_s"][4] = 6.0
        rows["goal_y"][4], rows["goal_w"][4] = tr, tl | tr
    assert _census(tmp_path, entailed, name="ent")["absence_goal_on_short_band"] == 0


def test_a_wrong_md5_is_refused(tmp_path):
    p, md5 = _write(tmp_path, 7)
    with pytest.raises(SystemExit):
        RT.load_v9_join(p, expect_md5="0" * 32)


# =========================================================================== #
# 6. the join on the trainer's own clock, through V3Dataset                    #
# =========================================================================== #
def _dataset(tmp_path, clock_shift=0.0, excluded=False):
    T = R.trainer()
    cfg = T.v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(1, cfg.core, seed=0, min_frames=140)
    sid = 424242
    eps[0].episode_id = sid
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    g0, dt = 0.113, 0.1007
    ds._label_clock = {sid: (g0, dt, "test")}
    ds.tac_goal_targets = True
    off = ds._raw_offset(eps[0])
    n = int(eps[0].poses.shape[0])

    def mut(rows):
        rows["now_s"][5] += clock_shift
    p, md5 = _write(tmp_path, sid, n=n, k0=off, g0=g0, dt=dt, mutate=mut if clock_shift else None, name="ds")
    if excluded:
        ds.tactical_excluded_sids = frozenset({sid})
    return T, ds, RT.load_v9_join(p, expect_md5=md5), off


def test_every_window_joins_on_k_equal_t_plus_w_minus_1_plus_raw_offset(tmp_path):
    T, ds, j, off = _dataset(tmp_path)
    c = ds.enable_r8_v9(j)
    assert c["n_joined"] == c["n_windows"] == len(ds.index) and c["n_clips"] == 1
    w = int(ds.window)
    e0, t0 = ds.index[0]
    assert int(j.rel.rows["k"][int(c["rows"][0])]) == int(t0) + w - 1 + off


def test_a_one_millisecond_clock_shift_is_refused_at_enable_time(tmp_path):
    T, ds, j, off = _dataset(tmp_path, clock_shift=1e-3)
    with pytest.raises(V9.V9LabelError, match="clock mismatch"):
        ds.enable_r8_v9(j)


def test_the_item_carries_raw_inputs_and_the_v9_targets_replace_v7(tmp_path):
    T, ds, j, off = _dataset(tmp_path)
    ds.enable_r8_v9(j)
    ds.r8_nav_from_v9 = True
    it = ds[0]
    assert it["r8_nav_raw"].tolist() == [50.0, 100.0, 45.0, 1.0, 1000.0]          # the reader clips lookahead
    assert bool(it["r8_nav_known"]) and bool(it["r8_rc_valid"])
    assert it["r8_rc_raw"].tolist() == [25.0, -10.0, 90.0]
    assert int(it["nav_cmd"]) == 1 and "r8_nav_token" not in it                     # TURN_L -> refc 'left'
    assert int(it["lat_v7"]) == 0 and int(it["lon_v7"]) == -100
    assert it["lat_allowed_v7"].tolist() == [True] + [False] * 7
    assert it["lon_allowed_v7"].tolist() == [True, False, False, False, False, True, False, False]
    assert it["tac_goal_y"][0] == 1.0 and it["tac_goal_w"][0] == 1.0 and float(it["tac_goal_w"][1:].sum()) == 0.0
    assert not any(k.startswith("v9_") for k in it)


def test_a_G3_excluded_clip_keeps_its_inputs_and_loses_its_targets(tmp_path):
    T, ds, j, off = _dataset(tmp_path, excluded=True)
    ds.enable_r8_v9(j)
    it = ds[0]
    assert int(it["lat_v7"]) == -100 and not bool(it["lat_allowed_v7"].any()) and float(it["tac_goal_w"].sum()) == 0
    assert bool(it["r8_rc_valid"])


# =========================================================================== #
# 7. the NavSim legal mapping and the nav index                                #
# =========================================================================== #
def test_the_navsim_legal_mapping_is_the_integration_table():
    assert [RT.navsim_legal_nav_cmd(v) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])] == [1, 0, 2, 0]
    with pytest.raises(ValueError):
        RT.navsim_legal_nav_cmd([1, 1, 0, 0])


def test_the_v9_nav_token_lands_on_the_right_nav_command_row():
    T = R.trainer()
    from tanitad.refs import refc
    assert T.R8_V9_NAV_TO_NAV_CMD == {0: 0, 1: 1, 2: 2}
    assert [refc.NAV_COMMANDS[T.R8_V9_NAV_TO_NAV_CMD[i]] for i in (0, 1, 2)] == ["follow", "left", "right"]


# =========================================================================== #
# 8. the REAL releases                                                         #
# =========================================================================== #
@pytest.mark.parametrize("split", ["eval139", "train"])
def test_the_real_release_loads_md5_checked_with_zero_contract_violations(split):
    p = REAL / f"v9_labels_{split}.npz"
    if not p.is_file():
        pytest.skip(f"the v9 {split} release is not on this host ({p})")
    j = RT.load_v9_join(str(p), expect_md5=RT.V9_RELEASE_MD5[split])
    c = j.manifest["census"]
    assert set(c["violations"].values()) == {0}, c["violations"]
    assert c["n_rows"] == {"eval139": 27_664, "train": 869_278}[split]
