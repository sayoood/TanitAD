"""PREREG_V8F_CKPT120 (part 2b), end to end on the synthetic world (CPU): the far-output wrappers run the REAL shared tools and

  * write the far output of the SAME forward call that produced the plan (a far-only checkpoint: every other column of the tool's own dump is bit-identical to a plain run of the tool),
  * feed nav_s as declared (the table's row at NOW; T1: dead-reckoned from the own executed arc; variants zero / donor move TOGETHER with the tool's nav variants, the donor windows being the tool's),
  * refuse a flag that disagrees with the checkpoint's cfg.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "scripts"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "taniteval"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_v8f_t1 as T1T  # noqa: E402
from test_v8f_t1 import world  # noqa: E402,F401  (the synthetic-world fixture of the T1 suite)
import v8f_mini_eval as E  # noqa: E402
import v8f_mini_eval_far as W0  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402
import v8f_t1_eval_far as W1  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402


def _far_ckpt(world, path, far=True, nav=True):
    tracks, J, store, wins = T1T.setup(world)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(0)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():
        for q in prior.parameters():
            q.mul_(0.02)
    cfg = M.V8FMiniConfig(**T1T.TINY, **({"s_far": "on", "w_ckpt_far": 0.3333} if far else {}), **({"nav_strat": "on"} if nav else {}))
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.02)
        if far:
            torch.nn.init.normal_(model.s_far_out.weight, std=0.05)          # the zero init would make the far head the constant (120, 0, 0)
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)
    return J


def _nav_s_table(root, J, seed=0):
    rng = np.random.default_rng(seed)
    N = int(J["off"][-1] + J["n"][-1])
    nav = np.zeros((N, 5), np.float32)
    nav[np.arange(N), rng.integers(0, 3, N)] = 1.0
    nav[:, 3], nav[:, 4] = rng.uniform(0.3, 2.5, N).astype(np.float32), 1.0
    p = str(Path(root) / "nav_s_eval.npz")
    np.savez(p, eval_clip=np.asarray(J["clip"]).astype(str), eval_off=np.asarray(J["off"]), eval_n=np.asarray(J["n"]), eval_nav=nav, eval_nav_d=(nav[:, 3] * 100).astype(np.float32))
    return p, nav


def _sha_f(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _t1_args(world, ck, extra):
    root = Path(world["root"])
    return ["run", "--ckpt", ck, "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"),
            "--replan-steps", "5", "--batch", "6", "--max-windows", "6", "--n-boot", "20", "--dump-heads"] + extra


def test_t1_far_wrapper_end_to_end(world, tmp_path):
    ck = str(tmp_path / "ck.pt")
    J = _far_ckpt(world, ck, far=True, nav=True)
    tbl, nav = _nav_s_table(tmp_path, J)
    d1 = tmp_path / "t1"
    d1.mkdir()
    assert W1.main(_t1_args(world, ck, ["--dump-nav", "--tag", "far", "--out-dir", str(tmp_path / "o1"), "--dump-ticks", str(d1), "--far-out", "on", "--nav-strat", "on", "--nav-strat-npz", tbl,
                                        "--nav-strat-sha256", _sha_f(tbl)])) == 0
    z = np.load(d1 / "ticks_far.npz")
    N = len(z["tick"])
    assert z["s_far"].shape == (N, 3) and z["s_far_now"].shape == (N, 3) and z["s_far_w0"].shape == (N, 3) and z["nav_s_in"].shape == (N, 5) and np.isfinite(z["s_far"]).all()
    assert not np.allclose(z["s_far"], [120.0, 0.0, 0.0])                       # the head is not the constant: the output is the model's, not a placeholder
    t0rows = np.flatnonzero(z["tick"] == 0)
    o = J["off"][z["ci"][t0rows]] + z["t0"][t0rows]
    np.testing.assert_array_equal(z["nav_s_in"][t0rows], nav[o])               # tick 0: the logged row at NOW
    for w in np.unique(z["win"]):                                              # later ticks: dead-reckoned (non-increasing countdown, token held), never the logged future
        r = np.flatnonzero(z["win"] == w)
        r = r[np.argsort(z["tick"][r])]
        assert (np.diff(z["nav_s_in"][r, 3]) <= 1e-9).all() and (z["nav_s_in"][r, :3] == z["nav_s_in"][r[0], :3]).all() and z["nav_s_in"][r[-1], 3] < z["nav_s_in"][r[0], 3]


def test_t1_far_wrapper_changes_no_other_output_for_a_far_only_checkpoint(world, tmp_path):
    ck = str(tmp_path / "ck.pt")
    _far_ckpt(world, ck, far=True, nav=False)
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    assert T1.main(_t1_args(world, ck, ["--tag", "p", "--out-dir", str(tmp_path / "oa"), "--dump-ticks", str(tmp_path / "a")])) == 0
    assert W1.main(_t1_args(world, ck, ["--tag", "p", "--out-dir", str(tmp_path / "ob"), "--dump-ticks", str(tmp_path / "b"), "--far-out", "on"])) == 0
    a, b = np.load(tmp_path / "a" / "ticks_p.npz"), np.load(tmp_path / "b" / "ticks_p.npz")
    assert "s_far" not in a.files and "s_far" in b.files and "s_far_now" in b.files
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)                     # every column of the tool's dump is bit-identical


def test_t1_far_wrapper_refuses_a_flag_the_checkpoint_disagrees_with(world, tmp_path):
    ck = str(tmp_path / "ck.pt")
    _far_ckpt(world, ck, far=True, nav=False)
    for extra in ([], ["--far-out", "on", "--nav-strat", "on", "--nav-strat-npz", "x", "--nav-strat-sha256", "y"]):
        with pytest.raises(SystemExit, match="REFUSED"):
            W1.main(_t1_args(world, ck, ["--tag", "q", "--out-dir", str(tmp_path / "c")] + extra))


def _t0_args(world, ck):
    root = Path(world["root"])
    return ["run", "--ckpt", ck, "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"),
            "--n-boot", "20", "--batch", "16", "--lru", "4", "--tracks", str(root / "native_tracks.pt")]


def test_t0_far_wrapper_end_to_end_and_donor_zero_variants(world, tmp_path):
    ck = str(tmp_path / "ck.pt")
    J = _far_ckpt(world, ck, far=True, nav=True)
    tbl, nav = _nav_s_table(tmp_path, J)
    args = _t0_args(world, ck)
    far = ["--far-out", "on", "--nav-strat", "on", "--nav-strat-npz", tbl, "--nav-strat-sha256", _sha_f(tbl)]
    assert W0.main(args + ["--tag", "r", "--out-dir", str(tmp_path / "r")] + far) == 0
    z = np.load(tmp_path / "r" / "eval_r_far.npz")
    ev = np.load(tmp_path / "r" / "eval_r.npz")
    N = len(z["win_ci"])
    assert z["ckpt_far"].shape == (N, 3) and z["nav_s_now"].shape == (N, 5) and str(z["nav_s_variant"]) == "real" and np.isfinite(z["ckpt_far"]).all() and ev["ckpt"].shape[0] == N and z["ckpt_far_now"].shape == (N, 3)
    np.testing.assert_array_equal(z["nav_s_now"], nav[J["off"][z["win_ci"]] + z["win_t"]])          # the fed NOW row is the table's row at the window
    assert W0.main(args + ["--tag", "z", "--out-dir", str(tmp_path / "z"), "--nav-withheld"] + far) == 0
    zz = np.load(tmp_path / "z" / "eval_z_far.npz")
    assert str(zz["nav_s_variant"]) == "zero" and (zz["nav_s_now"] == 0).all() and not np.array_equal(zz["ckpt_far"], z["ckpt_far"])       # S reads nav_s: zeroing it moves the far point
    assert W0.main(args + ["--tag", "d", "--out-dir", str(tmp_path / "d"), "--nav-shuffle"] + far) == 0
    zd = np.load(tmp_path / "d" / "eval_d_far.npz")
    from tanitad.models.hjepa import donor_indices
    perm = donor_indices(N, episode_ids=torch.as_tensor(zd["win_ci"]), seed=0).numpy()
    assert str(zd["nav_s_variant"]) == "donor" and (perm != np.arange(N)).all()
    np.testing.assert_array_equal(zd["nav_s_now"], nav[J["off"][zd["win_ci"][perm]] + zd["win_t"][perm]])      # nav_s moved TOGETHER with nav: the same donor windows


def test_t0_far_wrapper_leaves_the_tools_outputs_bit_identical_for_a_far_only_checkpoint(world, tmp_path):
    ck = str(tmp_path / "ck.pt")
    _far_ckpt(world, ck, far=True, nav=False)
    args = _t0_args(world, ck)
    assert E.main(args + ["--tag", "p", "--out-dir", str(tmp_path / "a")]) == 0
    assert W0.main(args + ["--tag", "p", "--out-dir", str(tmp_path / "b"), "--far-out", "on"]) == 0
    a, b = np.load(tmp_path / "a" / "eval_p.npz"), np.load(tmp_path / "b" / "eval_p.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert not (tmp_path / "a" / "eval_p_far.npz").exists() and (tmp_path / "b" / "eval_p_far.npz").exists()
    with pytest.raises(SystemExit, match="REFUSED"):                                         # a head the checkpoint lacks
        W0.main(args + ["--tag", "q", "--out-dir", str(tmp_path / "c"), "--far-out", "on", "--nav-strat", "on", "--nav-strat-npz", "x", "--nav-strat-sha256", "y"])
    with pytest.raises(SystemExit, match="REFUSED"):                                         # a head the flags drop
        W0.main(args + ["--tag", "q", "--out-dir", str(tmp_path / "c")])
    assert W0.main(args + ["--tag", "f", "--out-dir", str(tmp_path / "f"), "--far-out", "on", "--ages", "fresh"]) == 0
    f = np.load(tmp_path / "f" / "eval_f_far.npz")
    np.testing.assert_allclose(f["ckpt_far"], f["ckpt_far_now"], atol=1e-3)                          # fresh ages: S's tick frame IS the NOW frame
