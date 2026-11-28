"""Unit tests of the LF5 tool (PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...: ``--floor-vhat-noise SIGMA`` / ``--floor-vhat-noise-seed N`` on the logged-speed source).
Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf5_build/pt/run <this file>
  LF5_TREE  the read tree (default F:/TanitAD-scratch/motion_c/lf5_read/tree);  LF5_TOOL  the tool under test (default <tree>/stack/scripts/v8f_t1_eval_ri.py; code/run_mutations_lf5.py points it at a mutated copy)
The 64-window known answers (KA-N1, KA-N2, KA-N3) are code/ka_lf5.py; these tests pin the unit behaviour behind them."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
TREE = Path(os.environ.get("LF5_TREE", "F:/TanitAD-scratch/motion_c/lf5_read/tree"))
for _p in (TREE / "stack", TREE / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import importlib.util  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)
TOOL = Path(os.environ.get("LF5_TOOL", str(TREE / "stack" / "scripts" / "v8f_t1_eval_ri.py")))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T1 = _load("v8f_t1_eval_lf5_under_test", TOOL)
OP = _load("level_floor_lf5_under_test", TREE / "stack" / "scripts" / "level_floor.py")
_REQ = ["run", "--ckpt", "c.pt", "--tag", "t", "--tracks", "t.pt", "--eval-grid", "g.json", "--eval-join", "j.npz", "--out-dir", "o"]


def _args(extra):
    return T1.build_parser().parse_args(_REQ + extra)


def _lf(noise=None, seed=None, v_log=None):
    v_log = np.array([6.0, np.nan, 7.5, 5.0] if v_log is None else v_log)
    lf = T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, vhat_source="logged", oracle=lambda s: v_log.copy(), noise=noise, noise_seed=seed)
    n = len(v_log)
    lf.heads = lambda batch: (np.full(n, 12.0, np.float32), np.full(n, -6.0, np.float32), np.full(n, 0.5, np.float32))
    lf.states = [object()] * n
    plan = np.stack([OP.T_KNOTS * 0.3, np.zeros(30)], 1)[None].repeat(n, 0).astype(np.float32)
    return lf, plan, np.full(n, 0.3), v_log


def test_noise_is_v_log_times_exp_sigma_eps_from_one_generator_one_draw_per_window_per_call():
    lf, plan, v0, v_log = _lf(noise=0.2, seed=7)
    rng = np.random.default_rng(7)
    for _ in range(3):                                                      # three replans: the stream continues across calls, one vector draw each
        _, info = lf({}, plan, v0)
        eps = rng.standard_normal(len(v_log))
        assert np.array_equal(info["eps"], eps)
        assert np.array_equal(info["vlog"], v_log, equal_nan=True)
        assert np.array_equal(info["vhat"], v_log * np.exp(0.2 * eps), equal_nan=True)
    assert lf.stats["noise_draws"] == 12


def test_a_draw_is_taken_whether_or_not_a_floor_applies():
    lf, plan, v0, v_log = _lf(noise=0.2, seed=7)                             # window 1 has no logged route (NaN) = no floor; its draw is still consumed
    _, info = lf({}, plan, v0)
    assert np.isnan(info["vhat"][1]) and np.isnan(info["vf"][1]) and not info["bind"][1]
    assert np.array_equal(info["eps"], np.random.default_rng(7).standard_normal(4)) and np.isfinite(info["eps"]).all()


def test_sigma_zero_with_a_seed_is_the_oracle_exactly_and_no_flags_is_the_lfo_path():
    lf, plan, v0, v_log = _lf(noise=0.0, seed=7)
    out, info = lf({}, plan, v0)
    assert np.array_equal(info["vhat"], v_log, equal_nan=True) and "eps" in info
    lo, plan, v0, v_log = _lf()
    out0, info0 = lo({}, plan, v0)
    assert lo.rng is None and "eps" not in info0 and "noise_draws" not in lo.stats and np.array_equal(out0, out, equal_nan=True) and np.array_equal(info0["vf"], info["vf"], equal_nan=True)


def test_flag_refusals(tmp_path):
    ck = tmp_path / "f.pt"
    ck.write_bytes(b"x")
    ON = ["--level-floor", "on", "--floor-kappa", "1.0", "--floor-ckpt", str(ck)]
    a = _args(ON + ["--floor-vhat-source", "logged", "--floor-vhat-noise", "0.1", "--floor-vhat-noise-seed", "20261106"])
    T1.lf_check_args(a)
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(ON + ["--floor-vhat-source", "heads", "--floor-vhat-noise", "0.1", "--floor-vhat-noise-seed", "1"]))      # noise on the heads: silently nothing, so refused
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(ON + ["--floor-vhat-noise-seed", "1"]))                                                                   # default source = heads
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(["--floor-vhat-noise", "0.1", "--floor-vhat-noise-seed", "1"]))                                           # floor off
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(ON + ["--floor-vhat-source", "logged", "--floor-vhat-noise", "0.1"]))                                      # sigma > 0 needs the seed
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.lf_check_args(_args(ON + ["--floor-vhat-source", "logged", "--floor-vhat-noise", "-0.1", "--floor-vhat-noise-seed", "1"]))
    with pytest.raises(ValueError):
        T1.LevelFloor(None, 1.0, "x.pt", "0" * 64, vhat_source="heads", noise=0.1, noise_seed=1)
    c = _args(ON + ["--floor-vhat-source", "logged"])
    T1.lf_check_args(c)                                                                                                                    # the LFO100 form: no noise flags
    assert c.floor_vhat_noise is None and c.floor_vhat_noise_seed is None


def test_header_and_meta_record_both_and_the_ticks_gain_the_noise_fields():
    lf, *_ = _lf(noise=0.1, seed=20261106)
    m, d = lf.meta(), lf.detail()
    assert m["floor_vhat_noise"] == 0.1 and m["floor_vhat_noise_seed"] == 20261106 and m["label"] == "CEILING"
    assert d["floor_vhat_noise"] == 0.1 and d["floor_vhat_noise_seed"] == 20261106 and "N(0,1)" in d["noise"]
    lh = T1.LevelFloor(None, 1.0, "x.pt", "0" * 64)
    assert "floor_vhat_noise" not in lh.meta() and "floor_vhat_noise" not in lh.detail()
    code = TOOL.read_text(encoding="utf-8")
    assert 'header["floor_vhat_noise"], header["floor_vhat_noise_seed"] = floor.noise, floor.noise_seed' in code
    assert T1.TickRecorder.LF5_FIELDS == ("lf_vlog", "lf_eps")
