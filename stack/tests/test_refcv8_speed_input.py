"""refcv8 X3 (SPEC_REFCV8 sec. 8.3; MM ruling Q5; MM item 1, 2026-10-04) -- the PAST-ONLY speed input.

refcv7 fed the 4-way set-speed channel from the v8 sidecar: the window's FUTURE max over [NOW + 2 s, NOW + 6 s], a
future-ego oracle input (D4: 23.8 % of the missing future-speed information). refcv8 feeds the v9 release's past-only
proxy N2 (or N3 behind the same flag), trains the "unknown" row, and keeps the ceiling on the EMITTED plan.

Pinned here:
1. the LEAK instrument's known-value controls (a constant block recovers exactly 0, the target itself exactly 1, a
   cross-clip shuffle of a strong input ~0 while the input itself is large) -- synthetic, always run;
2. the REAL reproduction (data-gated, dev box): D4's N2 0.0346 recomputed, WP-A's shipped N2 column, D4's N2u 0.0093,
   the oracle 0.5698, refcv7's v8 sidecar 0.238 -- and the channel refcv8 FEEDS under the 0.05 bar, its V-VSHUF roll
   at ~0. Tolerances fixed BEFORE the first run: +-0.00005 where the computation is D4's own (same inputs, same
   code path), +-0.002 where the input is another stream's artifact (the v9 column, the v8 sidecar);
3. the encoding: every N2 / N3 value lands on its LITERAL 4-way bin through the item path, in float32 and bf16;
4. the dataset (`enable_r8_speed`, the item) and its refusals;
5. the per-batch treatment (`speed_input_treatment`): the trained unknown row on the dedicated generator only, the
   V-VSHUF roll in training only, VMAX-OFF / VMAX-SHUF at eval, the LEGAL row; the forward wrapper lets it REPLACE
   the batch's value;
6. the model: with a past-only source the ceiling is never below the fed value; with the inherited source unchanged;
7. the EMITTED plan obeys the fed ceiling on the EXTENDED fan (allocation + prior-free, both emitted) -- and the same
   windows without a ceiling violate it on at least one row (the test is not vacuous);
8. the config stamp, both directions.
The gate side (the v8 sidecar refused by name, the mutation that re-attaches it goes RED) is in
`test_refcv8_launch_profile.py`; the argv refusals in `test_refcv8_trainer_pin.py`.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
import test_refcv8_v9_wiring as W9  # noqa: E402  (the synthetic-release writer, independent of WP-A's builder)
from tanitad.data import v9_labels as V9  # noqa: E402
from tanitad.eval import speed_leak as SL  # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

KMH_N2 = (50, 70, 80, 100, 120, 130)


# =========================================================================== #
# 1. the instrument's known-value controls (synthetic)                         #
# =========================================================================== #
def _synth(n_clips=300, per=40, seed=0):
    rng = np.random.default_rng(seed)
    clip = np.repeat(np.arange(n_clips), per)
    folds = clip % 5
    v0 = rng.uniform(2.0, 30.0, len(clip))
    z = rng.integers(0, 4, n_clips)                                  # a per-clip latent the input carries
    y = v0 + 2.5 * z[clip] + rng.normal(0.0, 1.0, len(clip))
    return y, v0, folds, z, clip, rng


def test_the_instrument_reads_its_known_values_exactly():
    y, v0, folds, z, clip, rng = _synth()
    shuf = rng.permutation(len(z))
    rows = SL.recovered(y, v0, folds, {"const": np.ones((len(y), 1)), "target": y[:, None],
                                       "input": SL.onehot(z[clip], 4), "input_shuffled": SL.onehot(z[shuf][clip], 4)})
    assert abs(rows["const"]["recovered"]) < 1e-6                 # nothing beyond v0: 0
    assert rows["target"]["recovered"] > 1.0 - 1e-6              # the target itself: 1
    assert rows["input"]["recovered"] > 0.5                       # the real input carries the latent
    assert abs(rows["input_shuffled"]["recovered"]) < 0.01         # ... another clip's does not


def test_the_fed_block_is_the_models_own_binning_and_zero_where_unknown():
    blk = SL.fed_bin4(np.array([50, 70, 130, 50]) / 3.6, np.array([True, True, True, False]))
    assert blk.tolist() == [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]


# =========================================================================== #
# 2. the REAL reproduction (dev box; data-gated)                               #
# =========================================================================== #
_MAN = os.environ.get("R8_TRAIN_V2MANIFEST", "D:/Projects/TanitAD-artifacts/refcv8_wpb/inputs/train_v2manifest.pt")
_V9 = os.environ.get("R8_V9_TRAIN", "D:/Projects/TanitAD-artifacts/v9labels/v9_labels_train.npz")
_V8 = os.environ.get("R8_V8_SIDECAR", "D:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl")


@pytest.fixture(scope="module")
def real():
    missing = [p for p in (_MAN, _V9, _V8) if not Path(p).is_file()]
    if missing:
        pytest.skip(f"the X3 leak artifacts are not on this host: {missing}")
    return SL.x3_leak_table(_MAN, _V9, _V8)


def test_REAL_D4s_numbers_reproduce_and_the_fed_channel_is_under_the_bar(real):
    rows = real["rows"]
    rec = {k: v.get("recovered") for k, v in rows.items()}
    assert real["n_windows"] == 576_555 and real["v9_join_frac"] == 1.0
    assert rows["_base"]["oof_r2"] == pytest.approx(0.8926, abs=5e-5)                 # D4: v0 alone
    # D4's own computation (manifest poses -> N2) and the oracle: the same code path -> D4's 4 dp
    assert rec["D4_N2_from_poses_bin8"] == pytest.approx(0.0346, abs=5e-5)
    assert rec["ORACLE_window_0_6s_bin8"] == pytest.approx(0.5698, abs=5e-5)
    # the OTHER streams' artifacts: WP-A's shipped N2 column, D4's N2u, refcv7's v8 sidecar (USAGE_AUDIT 23.8 %)
    assert rec["v9_N2_bin8"] == pytest.approx(0.0346, abs=2e-3)
    assert rec["v9_N2u_bin8_unknown_p0.45"] == pytest.approx(0.0093, abs=2e-3)
    assert rec["v8_sidecar_bin4_D4"] == pytest.approx(0.238, abs=2e-3)
    # the known-value controls on the real windows
    assert rec["CONTROL_constant"] == pytest.approx(0.0, abs=1e-6)
    assert rec["CONTROL_target_itself"] == pytest.approx(1.0, abs=1e-6)
    # X3 bar 1 (LEAK <= 0.05) on the channel refcv8 FEEDS -- at eval (known everywhere) and in training (unknown row)
    assert rec["v9_N2_fed_bin4"] <= 0.05 and rec["v9_N2_fed_bin4_unknown_p0.45"] <= 0.05
    # the v8 sidecar FAILS the same bar (the reason it is refused)
    assert rec["v8_sidecar_bin4_as_fed"] > 0.05
    # V-VSHUF's information control: another window's N2 carries nothing about this window's future
    assert abs(rec["v9_N2_fed_bin4_ROLLED"]) < 0.002
    # refcv8 (B) `--r8-speed-enc8`: the 4-way + 8-step block recovers N2's full information, still under the bar
    assert rec["v9_N2_fed_bin4+enc8"] <= 0.05 and rec["v9_N2_fed_bin4+enc8_unknown_p0.45"] <= 0.05
    assert rec["v9_N2_fed_bin4+enc8"] > rec["v9_N2_fed_bin4"]                    # it adds what the 4 bins lose
    assert abs(rec["v9_N2_fed_bin4+enc8_ROLLED"]) < 0.002


# =========================================================================== #
# 3. the encoding: literal 4-way bins through the item path                    #
# =========================================================================== #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_N2_and_N3_value_lands_on_its_literal_4way_bin(dtype):
    # the item's arithmetic: python float64 km/h / 3.6, cast once (RT.R8LabelJoin.item)
    ms = torch.tensor([float(k) / 3.6 for k in KMH_N2], dtype=torch.float32).to(dtype)
    idx, over = v6ms.speed_max_bin_tensor(ms)
    assert idx.tolist() == [1, 2, 2, 2, 3, 3]                     # {30, 50, 100, 120}: 50 | 70 80 100 | 120 130
    assert over.tolist() == [False] * 5 + [True]                  # only 130 is above the top step
    n3 = torch.tensor([RT.R8_SPEED_N3_KMH[c] / 3.6 for c in (0, 1, 2)], dtype=torch.float32).to(dtype)
    i3, o3 = v6ms.speed_max_bin_tensor(n3)
    assert i3.tolist() == [1, 2, 3] and o3.tolist() == [False, False, True]


def test_the_N3_table_and_the_N2_ladder_are_the_registered_literals():
    assert RT.R8_SPEED_N3_KMH == {0: 50.0, 1: 100.0, 2: 130.0}
    assert RT.R8_SPEED_N2_LADDER_KMH == (20, 30, 50, 70, 80, 100, 120, 130)
    assert RT.R8_SPEED_UNKNOWN_P == 0.45
    assert RT.R8_SPEED_COLUMN == {"n2": "speed_n2_kmh", "n3": "speed_n3"}


# =========================================================================== #
# 4. the dataset                                                               #
# =========================================================================== #
def _dataset(tmp_path, n2=None, n3=None, drop_col=None):
    T = R.trainer()
    cfg = T.v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(1, cfg.core, seed=0, min_frames=140)
    sid = 515151
    eps[0].episode_id = sid
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    g0, dt = 0.113, 0.1007
    ds._label_clock = {sid: (g0, dt, "test")}
    off = ds._raw_offset(eps[0])
    n = int(eps[0].poses.shape[0])

    def mut(rows):
        k = len(rows["k"])
        rows["speed_n2_kmh"] = (np.resize(np.asarray(n2 if n2 is not None else KMH_N2, np.int16), k))
        rows["speed_n3"] = (np.resize(np.asarray(n3 if n3 is not None else (0, 1, 2), np.int8), k))
        if drop_col:
            rows.pop(drop_col)
    p, md5 = W9._write(tmp_path, sid, n=n, k0=off, g0=g0, dt=dt, mutate=mut, name="spd")
    return T, ds, RT.load_v9_join(p, expect_md5=md5)


def test_the_item_feeds_the_past_only_value_and_the_census_counts_the_bins(tmp_path):
    T, ds, j = _dataset(tmp_path)
    ds.enable_r8_v9(j)
    rep = ds.enable_r8_speed("n2")
    assert rep["column"] == "speed_n2_kmh" and rep["n_known"] == rep["n_windows"] == len(ds.index)
    assert abs(sum(rep["bin4_shares_of_known"]) - 1.0) < 1e-5 and rep["over_ceiling_frac_of_known"] > 0.0
    for i in range(6):
        it = ds[i]
        r = int(ds._r8_rows[i])
        kmh = float(j.rel.rows["speed_n2_kmh"][r])
        assert it["v_max_ms"].dtype == torch.float32 and float(it["v_max_valid"]) == 1.0
        assert torch.equal(it["v_max_ms"], torch.tensor(kmh / 3.6, dtype=torch.float32))   # exact, not 1 ulp off
        assert not any(k.startswith("r8_speed_") for k in it)                               # the raw keys do not travel


def test_N3_feeds_its_registered_kmh(tmp_path):
    T, ds, j = _dataset(tmp_path)
    ds.enable_r8_v9(j)
    ds.enable_r8_speed("n3")
    it = ds[0]
    c = int(j.rel.rows["speed_n3"][int(ds._r8_rows[0])])
    assert torch.equal(it["v_max_ms"], torch.tensor(RT.R8_SPEED_N3_KMH[c] / 3.6, dtype=torch.float32))


def test_a_not_computable_row_is_the_unknown_row_with_no_value(tmp_path):
    T, ds, j = _dataset(tmp_path, n2=(-1, 50))
    ds.enable_r8_v9(j)
    rep = ds.enable_r8_speed("n2")
    assert 0 < rep["n_known"] < rep["n_windows"]
    seen = set()
    for i in range(4):
        it = ds[i]
        seen.add(float(it["v_max_valid"]))
        if float(it["v_max_valid"]) == 0.0:
            assert float(it["v_max_ms"]) == 0.0                                       # zeros next to a 0 (X15)
    assert seen == {0.0, 1.0}


@pytest.mark.parametrize("case,needle", [
    ("no_join", "before enable_r8_v9"),
    ("sidecar_on", "one source"),
    ("off_ladder", "off its ladder"),
    ("all_unknown", "NOT ONE"),
    ("no_column", "no `speed_n2_kmh` column"),
    ("bad_mode", "not one of"),
])
def test_the_dataset_refuses(tmp_path, case, needle):
    n2 = {"off_ladder": (50, 55), "all_unknown": (-1,)}.get(case)
    T, ds, j = _dataset(tmp_path, n2=n2, drop_col="speed_n2_kmh" if case == "no_column" else None)
    if case != "no_join":
        ds.enable_r8_v9(j)
    if case == "sidecar_on":
        ds.max_speed_enabled = True
    with pytest.raises(SystemExit) as e:
        ds.enable_r8_speed("n4" if case == "bad_mode" else "n2")
    assert needle in str(e.value), str(e.value)


# =========================================================================== #
# 5. the per-batch treatment                                                   #
# =========================================================================== #
def _vm(b):
    return torch.tensor([float(KMH_N2[i % 6]) / 3.6 for i in range(b)]), torch.ones(b)


def test_training_trains_the_unknown_row_at_p_on_the_dedicated_generator_only():
    vm, vv = _vm(20_000)
    st = torch.get_rng_state()
    g = C.R8Generator(3)
    om, ov = RT.speed_input_treatment(vm, vv, g, training=True, unknown_p=0.45)
    assert torch.equal(torch.get_rng_state(), st)                                     # refcv7's stream untouched
    assert abs(float((ov == 0).float().mean()) - 0.45) < 0.015
    assert float(om[ov == 0].abs().sum()) == 0.0 and torch.equal(om[ov == 1], vm[ov == 1])


def test_the_VVSHUF_roll_is_another_windows_input_in_training_only():
    vm, vv = _vm(12)
    plain = RT.speed_input_treatment(vm, vv, C.R8Generator(5), training=True, unknown_p=0.45)
    rolled = RT.speed_input_treatment(vm, vv, C.R8Generator(5), training=True, unknown_p=0.45, roll=True)
    for a, b in zip(rolled, plain):
        assert torch.equal(a, torch.roll(b, 1, 0))
    assert not torch.equal(rolled[0], plain[0])
    ev = RT.speed_input_treatment(vm, vv, C.R8Generator(5), training=False, roll=True)
    assert torch.equal(ev[0], vm) and torch.equal(ev[1], vv)                           # eval: its own input


def test_eval_draws_nothing_and_the_interventions_and_the_legal_row():
    vm, vv = _vm(6)
    g = C.R8Generator(9)
    s0 = g.get("cpu").get_state()
    om, ov = RT.speed_input_treatment(vm, vv, g, training=False)
    assert torch.equal(g.get("cpu").get_state(), s0) and torch.equal(om, vm) and torch.equal(ov, vv)
    off = RT.speed_input_treatment(vm, vv, g, training=False, eval_intervention="off")
    assert float(off[0].abs().sum()) == 0.0 and float(off[1].sum()) == 0.0              # VMAX-OFF
    sh = RT.speed_input_treatment(vm, vv, g, training=False, eval_intervention="shuf")
    assert torch.equal(sh[0], torch.roll(vm, 1, 0))                                      # VMAX-SHUF
    lg = RT.speed_input_treatment(vm, vv, g, training=False, legal=True)
    assert float(lg[1].sum()) == 0.0 and float(lg[0].abs().sum()) == 0.0                 # NavSim LEGAL: unknown
    with pytest.raises(ValueError, match="batch of one"):
        RT.speed_input_treatment(vm[:1], vv[:1], g, training=False, eval_intervention="shuf")
    with pytest.raises(ValueError):
        RT.speed_input_treatment(vm, vv, g, training=False, eval_intervention="bogus")


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1])
def test_an_unknown_rate_outside_0_1_refuses_in_training(p):
    vm, vv = _vm(4)
    with pytest.raises(SystemExit, match=r"\(0, 1\)"):
        RT.speed_input_treatment(vm, vv, C.R8Generator(1), training=True, unknown_p=p)


def test_the_forward_wrapper_lets_the_treated_value_REPLACE_the_batchs():
    """compute_losses_v3 passes v_max_ms explicitly; refcv8's `fwd` must override it. The OLD wrapper form
    (`model(*_a, **_k, **fwd)`) raises on the duplicate -- the deliberate regression of the wrapper."""
    T = R.trainer()
    import inspect
    src = inspect.getsource(T.compute_losses_v3)
    assert "model(*_a, **{**_k, **_r8_prep[\"fwd\"]})" in src

    def f(**k):
        return k
    fwd = {"v_max_ms": "treated"}
    new = (lambda *_a, **_k: f(*_a, **{**_k, **fwd}))
    old = (lambda *_a, **_k: f(*_a, **_k, **fwd))
    assert new(v_max_ms="batch", x=1) == {"v_max_ms": "treated", "x": 1}
    with pytest.raises(TypeError):
        old(v_max_ms="batch", x=1)


@pytest.fixture(scope="module")
def rig8():
    pytest.importorskip("timm")
    T = R.trainer()
    cfg8, m8 = R.build(T, True, vmax=True, speed_input="n2", n_alloc=4)
    bt = R.batch(cfg8)
    b = bt["frames"].shape[0]
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    return cfg8, m8, bt, b, traj


def test_r8_before_forward_carries_the_treated_speed_only_with_a_past_only_source(rig8):
    cfg8, m8, bt, b, traj = rig8
    batch = {"v_max_ms": torch.full((b,), 50 / 3.6), "v_max_valid": torch.ones(b)}
    m8.eval()
    f = W9._prep(m8, bt, b, traj, batch)["fwd"]
    assert torch.equal(f["v_max_valid"], torch.ones(b)) and torch.equal(f["v_max_ms"], torch.full((b,), 50 / 3.6))
    m8._r8_legal_row = True
    f = W9._prep(m8, bt, b, traj, batch)["fwd"]
    assert float(f["v_max_valid"].sum()) == 0.0                                          # LEGAL: unknown
    m8._r8_legal_row = False
    m8.cfg.refcv8.speed_input = ""
    try:
        f = W9._prep(m8, bt, b, traj, batch)["fwd"]
        assert "v_max_ms" not in f                                                       # the inherited path: untouched
    finally:
        m8.cfg.refcv8.speed_input = "n2"


# =========================================================================== #
# 6. the model: the ceiling is never below the FED value with a past-only source #
# =========================================================================== #
def _speed_batch(cfg, v0s, seed=1):
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


def _limits(monkeypatch, cfg, m, kmh, valid):
    from tanitad.refs import refcv6_selection as v6sel
    seen = []
    real = v6sel.SpeedCeilingFilter.forward

    def spy(self, cand, v_limit_ms):
        seen.append(v_limit_ms.detach().clone())
        return real(self, cand, v_limit_ms)
    monkeypatch.setattr(v6sel.SpeedCeilingFilter, "forward", spy)
    bt = _speed_batch(cfg, [8.0] * len(kmh))
    R.forward(cfg, m, bt, v_max_ms=torch.tensor([k / 3.6 for k in kmh], dtype=torch.float32),
              v_max_valid=torch.tensor(valid, dtype=torch.float32))
    assert seen, "the ceiling filter never ran -- the rig is not the --speed-ceiling-filter build"
    return seen[-1]


def test_a_fed_130_is_the_ceiling_and_the_inherited_source_is_unchanged(monkeypatch):
    pytest.importorskip("timm")
    T = R.trainer()
    kmh, valid = [50, 70, 130, 50], [1.0, 1.0, 1.0, 0.0]
    cfg8, m8 = R.build(T, True, vmax=True, speed_input="n2")
    lim8 = _limits(monkeypatch, cfg8, m8, kmh, valid)
    assert lim8[:3].tolist() == pytest.approx([50 / 3.6, 100 / 3.6, 130 / 3.6], abs=1e-5)
    assert math.isinf(float(lim8[3]))                                                     # unknown: no ceiling
    cfg7, m7 = R.build(T, True, vmax=True)                                                # speed_input "" (inherited)
    lim7 = _limits(monkeypatch, cfg7, m7, kmh, valid)
    assert lim7[:3].tolist() == pytest.approx([50 / 3.6, 100 / 3.6, 120 / 3.6], abs=1e-5)   # refcv7's clamp, as was


# =========================================================================== #
# 7. the EMITTED plan obeys the fed ceiling on the EXTENDED fan                 #
# =========================================================================== #
V0_GRID = [9.0, 10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 13.8]


def _emitted(cfg, m, kmh, valid):
    bt = _speed_batch(cfg, V0_GRID, seed=4)
    b = len(V0_GRID)
    out = R.forward(cfg, m, bt, v_max_ms=torch.full((b,), kmh / 3.6), v_max_valid=torch.full((b,), float(valid)))
    dec = m.core.decoder
    from tanitad.refs.refcv6_selection import planned_max_speed
    vmax = planned_max_speed(out["anchor_traj"], horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)
    ar = torch.arange(b)
    assert torch.equal(out["traj"], out["anchor_traj"][ar, out["sel_idx"]])
    return out, vmax, vmax[ar, out["sel_idx"]]


def test_the_EMITTED_plan_obeys_the_N2_ceiling_on_the_extended_fan_and_the_check_is_not_vacuous():
    pytest.importorskip("timm")
    T = R.trainer()
    cfg, m = R.build(T, True, vmax=True, speed_input="n2", n_alloc=8, alloc_emit=True, prior_free_group=True,
                     prior_free_emit=True)
    lim = 50 / 3.6
    out, vmax, picked = _emitted(cfg, m, 50, 1.0)
    nb = int(out["r8_n_base"])
    assert out["anchor_traj"].shape[1] > nb, "the fan is not extended -- the test would not cover the extras"
    compliant = vmax <= lim
    keep = compliant if "reach_keep" not in out else compliant & out["reach_keep"]
    exempt = ~keep.any(dim=1)                                    # no compliant (reachable) candidate: counted rows
    assert bool((picked[~exempt] <= lim + 1e-5).all()), (picked, lim, exempt)
    # NOT VACUOUS: the SAME windows with the ceiling unknown emit at least one plan above it, which the ceiling moved
    _o, _v, natural = _emitted(cfg, m, 50, 0.0)
    moved = (natural > lim + 1e-5) & ~exempt
    assert bool(moved.any()), "no window's natural pick exceeds 50 km/h -- the obedience check proved nothing"


# =========================================================================== #
# 8. the stamp                                                                 #
# =========================================================================== #
def test_the_stamp_declares_the_source_and_refuses_both_directions():
    for mode in ("n2", "n3"):
        txt = RT.R8_SPEED_DERIVATION[mode]
        assert all(t in txt for t in RT.R8_SPEED_STAMP_REQUIRED[mode])
        RT.assert_r8_speed_stamp({"r8_speed_derivation": txt}, mode)
        with pytest.raises(SystemExit, match="does not declare"):
            RT.assert_r8_speed_stamp({"r8_speed_derivation": txt.replace("past-only", "past")}, mode)
        with pytest.raises(SystemExit, match="carry no"):
            RT.assert_r8_speed_stamp({"r8_speed_derivation": None}, mode)
    RT.assert_r8_speed_stamp({"r8_speed_derivation": None}, None)
    with pytest.raises(SystemExit, match="must not be stamped"):
        RT.assert_r8_speed_stamp({"r8_speed_derivation": RT.R8_SPEED_DERIVATION["n2"]}, None)
    # neither stamp can satisfy the other's guard: the past-only text is not the v6 ORACLE declaration and vice versa
    for mode in ("n2", "n3"):
        with pytest.raises(v6ms.SpeedMaxStampError):
            v6ms.assert_speed_max_stamp_v6({"speed_max_derivation_v6": RT.R8_SPEED_DERIVATION[mode]}, True)
        with pytest.raises(SystemExit, match="does not declare"):
            RT.assert_r8_speed_stamp({"r8_speed_derivation": v6ms.SPEED_MAX_DERIVATION_V6}, mode)


# =========================================================================== #
# 9. SPEED-ONLY: an arm WITHOUT --refcv8 feeds N2 too (SPEC_WPB_LADDER V0)       #
# =========================================================================== #
V0_ARGV = ["--arm", "hier", "--out", "X", "--max-speed-input-v6", "--r8-speed-input", "n2", "--r8-v9-labels", "v9.npz"]


def _pin0(argv):
    tr = R.trainer()
    args = tr.build_parser().parse_args(argv)
    cfg = __import__("types").SimpleNamespace(refcv8=C.R8Config())
    tr._pin_refcv8(cfg, args)
    return cfg.refcv8


def test_SPEED_ONLY_pins_the_source_without_the_refcv8_seams():
    r = _pin0(V0_ARGV)
    assert r.speed_input == "n2" and r.enable is False
    assert _pin0(V0_ARGV + ["--r8-speed-unknown-p", "0.45", "--r8-roll-speed-input", "--batch", "4"]).speed_input == "n2"
    assert _pin0(["--arm", "hier", "--out", "X"]).speed_input == ""            # refcv7 argv: unchanged


@pytest.mark.parametrize("argv,needle", [
    (V0_ARGV + ["--speed-max-sidecar-v6", "S.jsonl"], "FUTURE-MAX sidecar"),
    (["--arm", "hier", "--out", "X", "--r8-speed-input", "n2", "--r8-v9-labels", "v9.npz"], "--max-speed-input-v6"),
    (["--arm", "hier", "--out", "X", "--max-speed-input-v6", "--r8-speed-input", "n2"], "pass --r8-v9-labels"),
    (V0_ARGV + ["--eval-cache", "E"], "no --r8-v9-labels-eval"),
    (V0_ARGV + ["--r8-speed-unknown-p", "1.0"], "(0, 1)"),
    (V0_ARGV + ["--r8-roll-speed-input", "--batch", "1"], "--batch >= 2"),
    (V0_ARGV + ["--r8-nav-from-v9"], "without --refcv8"),                        # every OTHER refcv8 flag stays dead
    (V0_ARGV + ["--r8-speed-enc8"], "without --refcv8"),
    (["--arm", "hier", "--out", "X", "--r8-v9-labels", "v9.npz"], "without --refcv8"),   # v9 without a speed source
])
def test_SPEED_ONLY_refusals(argv, needle):
    with pytest.raises(SystemExit) as e:
        _pin0(argv)
    assert needle in str(e.value), str(e.value)


def test_SPEED_ONLY_the_treatment_runs_on_a_model_level_generator():
    m = types_ns = __import__("types").SimpleNamespace(cfg=__import__("types").SimpleNamespace(refcv8=C.R8Config()),
                                                     training=True)
    m._r8_speed_unknown_p = 0.45
    vm, vv = _vm(20_000)
    st = torch.get_rng_state()
    f = RT.speed_only_fwd(m, {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    assert torch.equal(torch.get_rng_state(), st) and isinstance(m._r8_speed_gen, C.R8Generator)
    assert abs(float((f["v_max_valid"] == 0).float().mean()) - 0.45) < 0.015
    m.training = False
    e = RT.speed_only_fwd(m, {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    assert torch.equal(e["v_max_ms"], vm) and torch.equal(e["v_max_valid"], vv)
    del types_ns


def test_SPEED_ONLY_compute_losses_feeds_the_treated_value_and_skips_the_refcv8_losses():
    import inspect
    src = inspect.getsource(R.trainer().compute_losses_v3)
    assert "_r8_prep = {\"fwd\": r8train.speed_only_fwd(model, batch, device), \"speed_only\": True}" in src
    assert "if _r8_prep is not None and not _r8_prep.get(\"speed_only\"):" in src


def test_SPEED_ONLY_the_ceiling_rule_holds_without_the_seams(monkeypatch):
    pytest.importorskip("timm")
    T = R.trainer()
    cfg, m = R.build(T, False, vmax=True)
    m.cfg.refcv8.speed_input = "n2"
    lim = _limits(monkeypatch, cfg, m, [50, 70, 130, 50], [1.0, 1.0, 1.0, 0.0])
    assert lim[:3].tolist() == pytest.approx([50 / 3.6, 100 / 3.6, 130 / 3.6], abs=1e-5) and math.isinf(float(lim[3]))


def test_SPEED_ONLY_the_join_feeds_the_input_and_leaves_the_targets(tmp_path):
    T, ds, j = _dataset(tmp_path)
    ds.enable_r8_v9(j, targets=False)
    ds.enable_r8_speed("n2")
    it = ds[0]
    assert ds.r8_v9_targets is False and "lat_allowed_v7" not in it and "tac_goal_y" not in it
    assert float(it["v_max_valid"]) == 1.0


def test_SPEED_ONLY_G_DVB_reads_the_model_config():
    from tanitad.train import declared_vs_built as dvb
    chk = dvb.REGISTRY["r8_speed_input"].check
    m = __import__("types").SimpleNamespace(cfg=__import__("types").SimpleNamespace(refcv8=C.R8Config(speed_input="n2")))
    ok = __import__("types").SimpleNamespace(r8_speed_input="n2", refcv8=False)
    bad = __import__("types").SimpleNamespace(r8_speed_input="n3", refcv8=False)
    assert chk(m, ok) == [] and len(chk(m, bad)) == 1


# =========================================================================== #
# 10. MM I-0 assertion 2: V0 and V-R8 feed a BYTE-IDENTICAL speed channel        #
# =========================================================================== #
def test_the_refcv8_and_speed_only_paths_feed_byte_identical_channels(rig8):
    import types as _t
    cfg8, m8, bt, b, traj = rig8
    v = torch.tensor([50 / 3.6, 70 / 3.6, 130 / 3.6][:b])
    batch = {"v_max_ms": v, "v_max_valid": torch.ones(b)}
    m8.train()
    m8._r8_rc_dropout, m8._r8_rc_noise, m8._r8_nav_args_dropout, m8._r8_speed_unknown_p = 0.3, (2.0, 0.75), 0.5, 0.45
    crcs8, crcs0 = [], []
    m8._r8_speed_gen = None                                                     # fresh, seeded cfg.refcv8.seed
    m0 = _t.SimpleNamespace(cfg=_t.SimpleNamespace(refcv8=C.R8Config(speed_input="n2", seed=m8.cfg.refcv8.seed)),
                            training=True, _r8_speed_unknown_p=0.45)
    for _ in range(20):                                                         # 20 steps of the same batches
        f8 = W9._prep(m8, bt, b, traj, batch)["fwd"]
        f0 = RT.speed_only_fwd(m0, batch, "cpu")
        crcs8.append(RT.speed_crc(f8["v_max_ms"], f8["v_max_valid"]))
        crcs0.append(RT.speed_crc(f0["v_max_ms"], f0["v_max_valid"]))
    assert crcs8 == crcs0 and len(set(crcs8)) > 1                             # identical AND not constant


def test_the_speed_draw_never_touches_the_decoders_r8_generator(rig8):
    cfg8, m8, bt, b, traj = rig8
    m8.train()
    m8._r8_rc_dropout, m8._r8_rc_noise, m8._r8_nav_args_dropout = 0.3, (2.0, 0.75), 0.5
    batch = {"v_max_ms": torch.full((b,), 50 / 3.6), "v_max_valid": torch.ones(b)}
    m8.core.decoder.r8_gen = C.R8Generator(9)
    W9._prep(m8, bt, b, traj, {})
    s_without = m8.core.decoder.r8_gen.get("cpu").get_state()
    m8.core.decoder.r8_gen = C.R8Generator(9)
    W9._prep(m8, bt, b, traj, batch)
    assert torch.equal(m8.core.decoder.r8_gen.get("cpu").get_state(), s_without)


def test_the_crc_is_the_bytes():
    import zlib
    v, k = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0])
    want = zlib.crc32(k.numpy().tobytes(), zlib.crc32(v.numpy().tobytes()))
    assert RT.speed_crc(v, k) == want and RT.speed_crc(v, k) != RT.speed_crc(v + 1e-6, k)


def test_the_RC_shuffled_eval_row(rig8):
    cfg8, m8, bt, b, traj = rig8
    raw = {"r8_rc_raw": torch.tensor([[10.0, 1.0, 0.0], [20.0, -2.0, 30.0], [30.0, 3.0, -30.0]][:b]),
           "r8_rc_valid": torch.ones(b, dtype=torch.bool)}
    m8.eval()
    plain = W9._prep(m8, bt, b, traj, raw)["fwd"]["r8_rc"]
    m8._r8_rc_eval = "shuf"
    try:
        sh = W9._prep(m8, bt, b, traj, raw)["fwd"]["r8_rc"]
        assert torch.equal(sh, torch.roll(plain, 1, 0))
    finally:
        m8._r8_rc_eval = None
