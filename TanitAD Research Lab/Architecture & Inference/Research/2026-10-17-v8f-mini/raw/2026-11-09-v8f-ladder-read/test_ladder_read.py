"""Tests and deliberate regressions of the ladder T0 read. Run: pytest -q test_ladder_read.py (CPU, a few minutes: three tests load a banked ladder run or the M2b models)."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import lr_lib as L

NE, M, D = 60, 8, 16


def world(seed=0):
    """Clips with M anchors; ora = z0 + drift (the drift is what a model could predict)."""
    rng = np.random.default_rng(seed)
    pos = np.repeat(np.arange(NE), M)
    z0 = rng.normal(size=(NE * M, D)) + np.repeat(rng.normal(size=(NE, D)), M, 0)
    drift = rng.normal(size=(NE * M, D))
    ora = z0 + drift
    return pos, z0, ora, drift, rng


W = L.IL.boot_weights(np.arange(NE))


def test_copy_last_against_itself_reads_exactly_zero():
    pos, z0, ora, _, _ = world()
    r = L.skill_mom(z0, z0, ora, pos, NE).row(W)
    assert r["point"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0


def test_regression_shifted_copy_reference_is_not_zero():
    """A Copy-Last reference shifted by one anchor against the true Copy-Last row must NOT read 0 (the check goes RED)."""
    pos, z0, ora, _, _ = world()
    r = L.skill_mom(z0, np.roll(z0, 1, axis=0), ora, pos, NE).row(W)
    assert abs(r["point"]) > 1e-3


def test_cells_against_states_check_goes_red_on_a_shifted_copy():
    pos = np.repeat(np.arange(3), 5)
    frame = np.tile(np.arange(5), 3)
    X = np.random.default_rng(1).normal(size=(15, 4))
    states = {1: {"X": X, "pos": pos, "frame": frame}}
    idx = np.flatnonzero(frame + 1 < 5)
    cells = {(1, 1, "built"): {"pos": pos[idx], "frame": frame[idx], "copy": X[idx], "ora": X[idx + 1]}}
    assert L.check_cells_against_states(cells, states, [1, 1, 1]) == 1
    bad = {(1, 1, "built"): {"pos": pos[idx], "frame": frame[idx], "copy": X[np.maximum(idx - 1, 0)], "ora": X[idx + 1]}}
    with pytest.raises(RuntimeError):
        L.check_cells_against_states(bad, states, [1, 1, 1])
    off = {(1, 1, "built"): {"pos": pos[idx], "frame": frame[idx], "copy": X[idx], "ora": X[idx]}}                  # the Oracle one step early
    with pytest.raises(RuntimeError):
        L.check_cells_against_states(off, states, [1, 1, 1])


def test_planted_fixed_direction_shift_is_not_skill():
    """pred = the present plus a FIXED direction unrelated to the future (IMAG1's failure, planted): never 'above'."""
    pos, z0, ora, drift, rng = world()
    u = rng.normal(size=D)
    r = L.skill_stats(z0 + 1.5 * u / np.linalg.norm(u), z0, ora, pos, W, (9, 9), R=39)
    assert r["readable"] and not r["above"]


def test_a_genuine_prediction_separates():
    pos, z0, ora, drift, rng = world()
    pred = z0 + 0.8 * drift + 0.3 * rng.normal(size=drift.shape)
    r = L.skill_stats(pred, z0, ora, pos, W, (9, 8), R=39)
    assert r["above"] and r["lo"] > 0 and r["point"] > r["thr_hi"]


def test_copy_as_prediction_is_not_separated():
    pos, z0, ora, _, _ = world()
    r = L.skill_stats(z0.copy(), z0, ora, pos, W, (9, 7), R=39)
    assert r["point"] == 0.0 and not r["above"] and not r["below"]


def test_health_separates_a_real_prediction_and_not_a_random_one():
    pos, z0, ora, drift, rng = world()
    good = L.health_stats(ora + 0.1 * rng.normal(size=ora.shape), ora, pos, W, 1, R=39)
    bad = L.health_stats(rng.normal(size=ora.shape), ora, pos, W, 1, R=39)
    assert good["separated_two_part"] and not bad["separated_two_part"]


def test_k_of_marks_off_grid_horizons():
    assert L.k_of(3, 12.0, 15) == (4, True)
    assert L.k_of(3, 6.0, 15) == (2, True)
    assert L.k_of(3, 6.0, 4) == (8, False)                      # C0: 6.0 s is 7.5 steps of 0.8 s, reported as 8 steps = 6.4 s
    assert L.k_of(2, 6.0, 2) == (15, True)
    p = L.horizon_plan([1, 2, 4])
    assert [(h, k, round(a, 1), on) for h, k, a, on in p[3]] == [(6.0, 8, 6.4, False), (9.0, 11, 8.8, False), (12.0, 15, 12.0, True), (15.0, 19, 15.2, False), (18.0, 23, 18.4, False)]


def test_regression_a_plan_built_on_the_wrong_strides_gives_other_reaches():
    assert [k for _, k, _, _ in L.horizon_plan([1, 3, 15])[3]] != [k for _, k, _, _ in L.horizon_plan([1, 2, 4])[3]]


def test_band_outcome_labels():
    def P(b, h=True):
        return {lv: {"beats": b[i], "healthy": h, "worse": False, "skill": 0.1, "lo": 0, "hi": 1} for i, lv in enumerate("123")}
    assert L.band_outcome({0: P([True] * 3), 1: P([True] * 3)})["label"] == "BAND-MET"
    assert L.band_outcome({0: P([True, False, True]), 1: P([True, False, True])})["levels_beating"] == ["1", "3"]
    assert L.band_outcome({0: P([False] * 3), 1: P([False] * 3)})["label"] == "BAND-MISS"
    assert L.band_outcome({0: P([True] * 3), 1: P([True, True, False])})["label"] == "BAND-PARTIAL"           # both seeds required
    assert L.band_outcome({0: P([True] * 3, h=False)})["label"] == "BAND-PARTIAL"                              # unhealthy: not MET


RUNS = {a: f"{L.PULLS}/ladder_{a}" for a in ("C0_s0", "R35_s0", "R35H_s0")}


@pytest.mark.skipif(not all(os.path.exists(p + "/ckpt_last.pt") for p in RUNS.values()), reason="banked runs absent")
def test_built_model_matches_argv_and_a_wrong_argv_is_red():
    expect = {"C0_s0": ([2, 2], [0, 0, 0], 1), "R35_s0": ([3, 5], [10, 10, 6], 1), "R35H_s0": ([3, 5], [10, 10, 6], 3)}
    for name, (st, k, hk) in expect.items():
        model, cfg, argv, ck = L.load_arm(RUNS[name])
        built = L.built_values(model)
        assert built["strides"] == st and built["trained_k"] == k and built["hist_k"] == [hk] * 3
        L.check_built_vs_argv(built, argv)
        for flag, bad in (("wm-strides", ["2", "2"] if st == [3, 5] else ["3", "5"]), ("wm-k", ["3"] if hk == 1 else ["1"]),
                          ("wm-rollout-k", ["1", "1", "1"] if k != [0, 0, 0] else ["10", "10", "6"]), ("wm-ctx", ["1", "1", "1"])):
            wrong = dict(argv)
            wrong[flag] = bad
            with pytest.raises(L.LadderMismatch):
                L.check_built_vs_argv(built, wrong)


@pytest.mark.skipif(not os.path.exists(RUNS["C0_s0"] + "/ckpt_last.pt"), reason="banked run absent")
def test_the_adapter_hash_tells_the_seeds_apart():
    m0, _, _, _ = L.load_arm(RUNS["C0_s0"])
    m1, _, _, _ = L.load_arm(RUNS["C0_s0"].replace("_s0", "_s1"))
    assert L.adapter_sha(m0) != L.adapter_sha(m1)


@pytest.mark.skipif(not os.path.exists(os.path.join(L.THEORY, "wmact", "latents.pt")), reason="IMAG1B inputs absent")
def test_m2b_control_passes_and_the_wrong_seed_regression_fails():
    import control_m2b as C
    assert C.run(False, R=5)["pass"]
    assert not C.run(True, R=5)["pass"]
