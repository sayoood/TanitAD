"""Tests of the SLOWWM beside readings B1 (decodability) and B2 (WMACT audit). CPU, tier T0. Run with the env of README_B1B2.md:
  python -m pytest -q --basetemp F:/TanitAD-scratch/slowwm_beside_build/b1b2/pt <this file>   (the basetemp's parent must exist)"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
BESIDE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, BESIDE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import decod_lib as DL  # noqa: E402
import decod_run as DR  # noqa: E402
import decod_build_train as DB  # noqa: E402

LR, LP, W = DL.LR, DL.LP, DL.W


def synth(n_clip=40, per=20, D=16, seed=0):
    g = np.random.default_rng(seed)
    eid = np.repeat([f"c{i:03d}" for i in range(n_clip)], per)
    X = g.standard_normal((len(eid), D))
    w = g.standard_normal(D)
    y = X @ w
    ok = np.ones(len(eid), bool)
    return dict(Z=X, eid=eid, y_speed=y, y_yaw=y, m_speed=ok, m_yaw=ok, n=len(eid), k=10)


def test_k_is_ten_steps():
    assert DL.k_steps() == 10 and LR.k_of(1, 2.0, 1) == (10, True)


def test_yaw_mask_drops_stopped_windows():
    tracks = {"a": np.column_stack([np.zeros(60), np.zeros(60), np.zeros(60), np.r_[np.full(30, 5.0), np.full(30, 0.1)]])}
    eid, t = np.array(["a", "a"]), np.array([15 + 0, 15 + 12])                # frames g = 0 and 12; target frame 10 and 22 -> rows 20 and 44
    A = np.zeros((2, 16, 2))
    ys, yy, ms, my = DL.targets_at(tracks, eid, t, A, 10)
    assert ms.all() and my.tolist() == [True, False], (ys, my)                 # the second target sits at 0.1 m/s: no heading tangent
    ys2, yy2, ms2, my2 = DL.targets_at(tracks, np.array(["a", "zz"]), t, A, 10)
    assert my2.tolist() == [True, False] and ms2.tolist() == [True, False]     # an unknown clip has no target


def test_probe_recovers_planted_and_rejects_permuted():
    a = synth()
    tr, ev = DL.pseudo_split(a)
    c = DL.controls_on_arrays(tr, ev)
    assert c["i_permuted_speed"]["ok"] and c["ii_planted"]["ok"] and c["red_arms_go_red"] and c["pass"], c
    r = DL.b1_rows(tr, ev)["speed_m_s"]
    assert r["eval"]["R2"]["point"] > 0.99 and r["eval"]["R2"]["lo"] <= r["eval"]["R2"]["point"] <= r["eval"]["R2"]["hi"]


def test_cv_folds_are_grouped_by_clip():
    a = synth()
    f = DL.fold_of(a["eid"])
    for e in np.unique(a["eid"]):
        assert len(set(f[a["eid"] == e])) == 1
    assert set(f) == set(range(DL.NFOLD))


def test_bootstrap_draws_are_the_episode_cluster_draws():
    a = synth()
    uniq, Wt = DL.boot_for(a["eid"])
    assert Wt.shape == (DL.B_BOOT, len(uniq)) and (Wt.sum(1) == len(uniq)).all() and W.check_draws_equal_ci(a["eid"], uniq, Wt)


def test_permutation_red_arm_goes_red_on_leaky_scoring():
    tr, ev = DL.pseudo_split(synth())
    assert DL.control_permuted(tr, ev, mutate="score_on_train")["ok"] is False
    assert DL.control_permuted(tr, ev)["ok"] is True


def test_runner_refuses_without_passing_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "CONTROLS_JSON", str(tmp_path / "none.json"))
    assert DR.main(["--out", str(tmp_path), "X_s0|nowhere"]) == 2
    (tmp_path / "c.json").write_text(json.dumps({"all_pass": False}))
    monkeypatch.setattr(DL, "CONTROLS_JSON", str(tmp_path / "c.json"))
    assert DR.main(["--out", str(tmp_path), "X_s0|nowhere"]) == 2


def test_banked_controls_file_reads_all_pass():
    z = json.load(open(os.path.join(BESIDE, "CONTROLS_B1B2.json"), encoding="utf-8"))
    assert z["all_pass"] is True and all(z[k]["pass"] is True for k in ("b1_i", "b1_ii", "b1_iii", "b2"))


def test_build_train_clip_inputs_equal_the_banked_eval_frames():
    """``clip_inputs`` is the shared per-clip function of the train builder: on an EVAL clip it reproduces the banked frames file (zmean within 1e-5, act and ok bit for bit)."""
    model, *_ = LR.load_arm(f"{LR.PULLS}/ladder_R35H_s0")
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    D = LR.D
    store = D.ClipStore(f"{LR.SF}/cache", "eval", lru=2, grid="pool4x10", kin_mode="yaw", tracks=D.load_native_tracks(f"{LR.SF}/inputs/native_tracks.pt"))
    c = fr["clips"][3]
    z, u, act, ok = DB.clip_inputs(store.get(c), {0: model}, {})
    assert float(np.abs(z[0] - fr["zmean"][3]).max()) < 1e-5 and np.array_equal(act, fr["act"][3])    # zmean: float32 thread drift ~2e-6 (MEASURED 1, 2, 4 threads); act and ok exact and np.array_equal(ok, fr["ok"][3]) and u == {}


def test_adapter_model_refuses_the_basis_collection():
    import wmact_arm as A
    model, *_ = LR.load_arm(f"{LR.PULLS}/ladder_R35H_s0")
    with pytest.raises(ValueError):
        model.wm.z1_from_basis(torch.zeros(1, 4, 256))
    assert A.collect_arm_z1.__doc__ and "basis" in A.collect_arm_z1.__doc__
