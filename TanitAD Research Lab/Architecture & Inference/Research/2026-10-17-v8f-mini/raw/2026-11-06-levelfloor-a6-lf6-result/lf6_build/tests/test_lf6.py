"""Unit tests of the LF6 tool (PREREG_LEVELFLOOR_A6 sha256 fbf5de02...: ``--floor-lift-cap DELTA``, the bounded-lift floor F_B).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf6_build/pt/run <this file>
  LF6_TREE  the read tree (default F:/TanitAD-scratch/motion_c/lf6_read/tree);  LF6_TOOL  the tool under test (default <tree>/stack/scripts/v8f_t1_eval_ri.py; code/run_mutations_lf6.py points it at a mutated copy)
The 64-window known answers (KA-1, KA-2, KA-3) are code/ka_lf6.py; these tests pin the operator behind them against an INDEPENDENT fine-grid integration of the card's formula."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
TREE = Path(os.environ.get("LF6_TREE", "F:/TanitAD-scratch/motion_c/lf6_read/tree"))
for _p in (TREE / "stack", TREE / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import importlib.util  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)
TOOL = Path(os.environ.get("LF6_TOOL", str(TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py")))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T1 = _load("v8f_t1_eval_lf6_under_test", TOOL)
OP = _load("level_floor_lf6_under_test", TREE / "stack" / "scripts" / "level_floor.py")
_REQ = ["run", "--ckpt", "c.pt", "--tag", "t", "--tracks", "t.pt", "--eval-grid", "g.json", "--eval-join", "j.npz", "--out-dir", "o"]
DT = 0.2


def _args(extra):
    return T1.build_parser().parse_args(_REQ + extra)


def _plans(n, seed):
    """n random plans [n, 30, 2]: a speed profile with trends and jumps (first step included), a curving heading."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 9.0, (n, 1))
    trend = rng.uniform(-0.15, 0.25, (n, 1)) * np.arange(30)[None, :]
    noise = np.cumsum(rng.normal(0, 0.35, (n, 30)), axis=1) * rng.uniform(0, 1, (n, 1))
    v = np.clip(base + trend + noise, 0.0, 16.0)
    v[:, 0] = rng.uniform(0.0, 12.0, n)                                    # a large or small FIRST step: v_O's interval 1 starts at the origin
    th = np.cumsum(rng.normal(0, 0.03, (n, 30)), axis=1)
    pts = np.cumsum(v[:, :, None] * DT * np.stack([np.cos(th), np.sin(th)], -1), axis=1)
    return pts.astype(np.float32)


def _inputs(n, seed):
    rng = np.random.default_rng(seed + 1000)
    return (rng.uniform(0.0, 11.0, n), rng.uniform(1.0, 16.0, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n))        # v0, v_hat, pi, gap_hat


def ref_sp(plan, v0, v_hat, pi, gap, kappa, delta, sub=2000):
    """INDEPENDENT reference of the card: s' = max(s, int_0^t min(kappa v_hat [lead cap], v0 + tau, v_O + delta)) on a fine midpoint grid; None where there is no floor (v_hat < 3)."""
    p = np.asarray(plan, np.float64)
    full = np.vstack([np.zeros((1, 2)), p])
    seg = np.linalg.norm(np.diff(full, axis=0), axis=1)
    S, vO = np.cumsum(seg), seg / DT
    if not v_hat >= 3.0:
        return S, False
    vf = kappa * v_hat
    if pi >= 0.5:
        vf = min(vf, max(0.0, (40.0 * gap - 2.0) / 1.5))
    h = DT / sub
    tau = (np.arange(30 * sub) + 0.5) * h
    k = np.arange(30 * sub) // sub
    f = np.minimum(np.minimum(vf, max(v0, 0.0) + tau), vO[k] + delta)
    sf = np.cumsum(f * h)[sub - 1::sub]
    return np.maximum(S, sf), True


def test_hand_case_constant_speed_plan_ramps_to_v_o_plus_delta_then_holds():
    plan = np.stack([np.arange(1, 31) * 1.0, np.zeros(30)], 1)[None].astype(np.float32)          # v_O = 5 m/s throughout
    v0 = np.array([5.0])
    out, info = T1.apply_floor_batch_bounded(OP, plan, v0, np.array([12.0]), np.array([0.0]), np.array([0.5]), 1.0, 0.5)
    t = OP.T_KNOTS
    # v_f(tau) = min(12, 5 + tau, 5 + 0.5): the ramp from 5 reaches 5.5 at tau = 0.5, then 5.5, so s_f = 5.5 t - 0.125 and s' - s = 0.5 t - 0.125 (< 0.5 t)
    s_f = np.where(t <= 0.5, 5 * t + 0.5 * t * t, 5 * 0.5 + 0.5 * 0.25 + 5.5 * (t - 0.5))
    s = np.arange(1, 31) * 1.0
    assert np.allclose(info["sp"][0], np.maximum(s, s_f), atol=1e-9) and np.allclose(info["s"][0], s)
    assert info["bind"][0] and info["dterm"][0] and info["dterm_knots"][0] > 0
    assert np.all(info["sp"][0] - s <= 0.5 * t + 1e-9)
    assert np.allclose(out[0, :, 1], 0.0) and np.allclose(out[0, :, 0], info["sp"][0], atol=1e-5)       # a straight plan is resampled along itself


def test_matches_an_independent_fine_grid_integral_of_the_card_formula():
    n = 240
    plans, (v0, vh, pi, gap) = _plans(n, 3), _inputs(n, 3)
    for delta in (0.0, 0.25, 0.5, 1.0, 3.0):
        _, info = T1.apply_floor_batch_bounded(OP, plans, v0, vh, pi, gap, 1.0, delta)
        worst = 0.0
        for i in range(n):
            ref, has = ref_sp(plans[i], v0[i], vh[i], pi[i], gap[i], 1.0, delta)
            worst = max(worst, float(np.abs(info["sp"][i] - ref).max()))
        assert worst < 2e-5, (delta, worst)


def test_the_bound_s_prime_minus_s_le_delta_t_holds_on_every_knot_and_s_prime_ge_s():
    n = 400
    plans, (v0, vh, pi, gap) = _plans(n, 5), _inputs(n, 5)
    t = OP.T_KNOTS[None, :]
    for delta in (0.0, 0.1, 0.5, 1.0, 2.0):
        _, info = T1.apply_floor_batch_bounded(OP, plans, v0, vh, pi, gap, 1.0, delta)
        assert np.all(info["sp"] - info["s"] <= delta * t + 1e-6) and np.all(info["sp"] >= info["s"])
        assert info["bind"].sum() > 0 or delta == 0.0


def test_delta_zero_is_an_identity_the_plan_object_comes_back_and_nothing_binds():
    n = 500
    plans, (v0, vh, pi, gap) = _plans(n, 7), _inputs(n, 7)
    out, info = T1.apply_floor_batch_bounded(OP, plans, v0, vh, pi, gap, 1.0, 0.0)
    assert out is plans and not info["bind"].any() and not info["dterm"].any() and np.array_equal(info["sp"], info["s"])
    assert not OP.apply_floor_batch(plans, v0, vh, pi, gap, 1.0)[1]["bind"].all()                    # the control: A5's F does bind on these plans


def test_delta_huge_is_the_a5_operator_bit_for_bit():
    n = 500
    plans, (v0, vh, pi, gap) = _plans(n, 9), _inputs(n, 9)
    for kappa in (1.0, 0.75):
        out_b, ib = T1.apply_floor_batch_bounded(OP, plans, v0, vh, pi, gap, kappa, 1e9)
        out_o, io = OP.apply_floor_batch(plans, v0, vh, pi, gap, kappa)
        assert io["bind"].sum() > 20
        assert np.array_equal(out_b, out_o) and out_b.dtype == out_o.dtype
        for k in io:
            assert np.array_equal(ib[k], io[k], equal_nan=True), k
        assert not ib["dterm"].any()


def test_bind_count_is_monotone_in_delta_and_partial_in_between():
    n = 600
    plans, (v0, vh, pi, gap) = _plans(n, 11), _inputs(n, 11)
    nb = [int(T1.apply_floor_batch_bounded(OP, plans, v0, vh, pi, gap, 1.0, d)[1]["bind"].sum()) for d in (0.0, 0.25, 0.5, 1.0, 4.0, 1e9)]
    assert nb == sorted(nb) and nb[0] == 0 and 0 < nb[2] < nb[-1], nb


def test_v_o_starts_at_the_origin_the_first_interval_is_the_first_knots_distance():
    plan = np.stack([np.concatenate([[6.0], 6.0 + 0.2 * np.arange(1, 30)]), np.zeros(30)], 1)[None].astype(np.float32)      # a 6 m first step (30 m/s), then 1 m/s
    S = OP.plan_arc(plan.astype(np.float64))[1]
    s_f, dterm = T1.bounded_progress(OP, np.array([10.0]), np.array([1.0]), S, 0.5)
    assert abs((s_f[0, 0]) - (1.0 * 0.2 + 0.5 * 0.2 ** 2)) < 1e-12 and not dterm[0, 0]               # interval 1 is capped by v0 + tau (the delta term is 30.5 m/s there), not by the later 1 m/s speed
    assert dterm[0, 1:].any()                                                                          # later, v_O + delta = 1.5 m/s < v_f and < v0 + t_k


def test_the_delta_term_share_dterm_only_where_it_is_the_strict_minimum():
    plan = np.stack([np.arange(1, 31) * 1.0, np.zeros(30)], 1)[None].astype(np.float32)
    _, i_small = T1.apply_floor_batch_bounded(OP, plan, np.array([5.0]), np.array([12.0]), np.array([0.0]), np.array([0.5]), 1.0, 0.5)
    _, i_big = T1.apply_floor_batch_bounded(OP, plan, np.array([5.0]), np.array([12.0]), np.array([0.0]), np.array([0.5]), 1.0, 20.0)       # v_O + 20 > v_f = 12: the delta term never binds
    _, i_off = T1.apply_floor_batch_bounded(OP, plan, np.array([5.0]), np.array([2.5]), np.array([0.0]), np.array([0.5]), 1.0, 0.5)         # below the 3 m/s band: no floor at all
    assert i_small["dterm"][0] and not i_big["dterm"][0] and i_big["bind"][0] and not i_off["bind"][0] and not i_off["dterm"][0]


def test_flag_refusals():
    ck = Path(os.environ.get("TEMP", ".")) / "lf6_dummy_floor.pt"
    ck.write_bytes(b"x")
    ON = ["--level-floor", "on", "--floor-kappa", "1.0", "--floor-ckpt", str(ck)]
    for good in ("0", "0.5", "1.0", "1e9"):
        a = _args(ON + ["--floor-lift-cap", good])
        T1.lf_check_args(a)
        assert a.floor_lift_cap == float(good)
    T1.lf_check_args(_args(ON + ["--floor-vhat-source", "logged", "--floor-lift-cap", "0.5"]))
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(["--floor-lift-cap", "0.5"]))                                          # floor off: silently nothing, so refused
    for bad in ("-0.1", "nan", "inf"):
        with pytest.raises(SystemExit, match="REFUSED"):
            T1.lf_check_args(_args(ON + ["--floor-lift-cap", bad]))
    c = _args(ON)
    T1.lf_check_args(c)
    assert c.floor_lift_cap is None                                                                    # absent = the LF5 form
    with pytest.raises(ValueError):
        T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, lift_cap=-1.0)
    with pytest.raises(ValueError):
        T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, lift_cap=float("inf"))


def _lf(cap=None, source="heads", v_log=None):
    n = 4
    v_log = np.array([6.0, np.nan, 7.5, 5.0] if v_log is None else v_log)
    kw = dict(vhat_source="logged", oracle=lambda s: v_log.copy()) if source == "logged" else {}
    lf = T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, lift_cap=cap, **kw)
    lf.heads = lambda batch: (np.full(n, 12.0, np.float32), np.full(n, -6.0, np.float32), np.full(n, 0.5, np.float32))
    lf.states = [object()] * n
    plan = np.stack([OP.T_KNOTS * 0.3, np.zeros(30)], 1)[None].repeat(n, 0).astype(np.float32)
    return lf, plan, np.full(n, 0.3)


def test_without_the_flag_the_level_floor_is_the_lf5_path_and_with_it_the_bounded_one():
    for source in ("heads", "logged"):
        l0, plan, v0 = _lf(None, source)
        o0, i0 = l0({}, plan, v0)
        assert l0.lift_cap is None and "dterm" not in i0 and "dterm_bound" not in l0.stats and "floor_lift_cap" not in l0.meta() and "floor_lift_cap" not in l0.detail()
        ref, iref = OP.apply_floor_batch(plan, np.full(4, 0.3), np.asarray(i0["vhat"]), OP.sigmoid(np.full(4, -6.0, np.float32)), np.full(4, 0.5, np.float32), 1.0)
        assert np.array_equal(o0, ref) and np.array_equal(i0["bind"], iref["bind"])
        l1, plan, v0 = _lf(0.5, source)
        o1, i1 = l1({}, plan, v0)
        m, d = l1.meta(), l1.detail()
        assert m["floor_lift_cap"] == 0.5 and d["floor_lift_cap"] == 0.5 and "DELTA" in d["operator"] and "a6_prereg" in d["prereg"]
        assert i1["dterm"].dtype == bool and i1["s"].shape == (4, 30) and i1["sp"].shape == (4, 30) and l1.stats["dterm_bound"] == int(i1["dterm"].sum())
        assert np.all(i1["sp"] - i1["s"] <= 0.5 * OP.T_KNOTS[None, :] + 1e-6)
        assert i1["bind"].sum() <= i0["bind"].sum()


def test_the_ticks_keys_omit_a_none_seed_and_the_file_loads_without_pickle(tmp_path):
    lf, *_ = _lf(0.5, "logged")                                                                        # the LFOB arms: source logged, sigma 0, no seed
    m = lf.meta()
    assert m["floor_vhat_noise"] == 0.0 and "floor_vhat_noise_seed" not in m and m["label"] == "CEILING"
    np.savez(tmp_path / "t.npz", win=np.arange(3), **m)
    z = np.load(tmp_path / "t.npz", allow_pickle=False)                                                # the pinned loader (A5_NOTE_READER_FIX): this refused the LF5 tool's file
    assert float(z["floor_lift_cap"]) == 0.5 and "floor_vhat_noise_seed" not in z.files
    noisy = T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, vhat_source="logged", oracle=lambda s: np.ones(2), noise=0.1, noise_seed=5)
    assert noisy.meta()["floor_vhat_noise_seed"] == 5
    assert "floor_vhat_noise" not in _lf(None, "heads")[0].meta() and "floor_lift_cap" not in _lf(None, "heads")[0].meta()


def test_recorder_fields_header_and_call_site_pins():
    on = T1.TickRecorder(None, None, None, level_floor=True, floor_oracle=False, floor_bound=True)
    off = T1.TickRecorder(None, None, None, level_floor=True, floor_oracle=False)
    nofloor = T1.TickRecorder(None, None, None, floor_bound=True)
    assert T1.TickRecorder.LF6_FIELDS == ("lf_dterm", "lf_dterm_knots", "lf_s", "lf_sp")
    assert all(k in on.rows for k in T1.TickRecorder.LF6_FIELDS) and not any(k in off.rows for k in T1.TickRecorder.LF6_FIELDS) and not any(k in nofloor.rows for k in T1.TickRecorder.LF6_FIELDS)
    code = TOOL.read_text(encoding="utf-8")
    assert 'header["floor_lift_cap"] = floor.lift_cap' in code
    assert "floor_bound=floor is not None and floor.lift_cap is not None" in code
    assert 'lift_cap=getattr(a, "floor_lift_cap", None)' in code
    assert 'r["lf_sp"].append(np.asarray(fi["sp"][b], np.float64))' in code
