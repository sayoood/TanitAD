"""Tests of s0e_read.py on SYNTHETIC data only (64-wide cells). Run: python -m pytest -q test_s0e_read.py (CUDA_VISIBLE_DEVICES=-1)."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import pytest

import s0e_read as E

S0, S0C, RP = E.S0, E.S0C, E.RP


def synth(n_clips=30, frames=60, seed=0, planted_cell=None, planted_dir=3, amp=2.0):
    """Synthetic probe set: tokens noise (sd 1) [N, 40, 64], smooth lead-gap tracks, ego (v, a, yaw). If planted_cell is given, the standardised gap rides
    on ONE direction of ONE cell (so the mean over cells dilutes it 40 times)."""
    rng = np.random.default_rng(seed)
    N = n_clips * frames
    beid = np.repeat([f"clip{c:03d}" for c in range(n_clips)], frames)
    bg = np.tile(np.arange(frames), n_clips)
    epi = np.repeat(np.arange(n_clips), frames)
    gap = np.zeros(N)
    for c in range(n_clips):
        w = np.cumsum(rng.standard_normal(frames) * 0.6) + rng.uniform(15, 40)
        gap[c * frames:(c + 1) * frames] = np.clip(w, 2.0, 58.0)
    gap = gap.astype(np.float32).astype(np.float64)
    X = rng.standard_normal((N, 40, 64)).astype(np.float32)
    if planted_cell is not None:
        z = (gap - gap.mean()) / gap.std()
        X[:, planted_cell, planted_dir] += amp * z
    kin = np.stack([rng.uniform(0, 20, N), rng.standard_normal(N), 0.1 * rng.standard_normal(N)], 1)
    D = {"X": X.astype(np.float16), "kin": kin, "beid": beid, "bg": bg, "epi": epi, "uep": [f"clip{c:03d}" for c in range(n_clips)], "gap": gap.copy(),
         "pres": np.ones(N), "gap_loaded": gap.copy(), "pres_loaded": np.ones(N), "frames_index": np.full(n_clips, frames), "rows_per_clip": np.full(n_clips, frames)}
    return D


def synth_join(D, shift=0):
    """A join whose rows hold the synthetic labels; the join is padded in front so that off != 0; shift moves every offset (the mutation)."""
    clips = D["uep"]; n_clips = len(clips); frames = int(D["frames_index"][0])
    pad = 7
    lv = np.zeros(pad + n_clips * frames, np.int8); lg = np.full_like(lv, 0, dtype=np.float32)
    lv[pad:] = 1; lg[pad:] = D["gap"].astype(np.float32)
    return {"clip": np.array(clips), "off": pad + np.arange(n_clips) * frames + shift, "n": np.full(n_clips, frames), "lead_valid": lv, "lead_gap": lg}


# ----------------------------------------------------------------------------- labels: the 0 to 60 m rule
def test_gap_rule_zero_to_sixty():
    g, p = E.derive_labels(np.array([1, 1, 1, 1, 1, 0, 1]), np.array([0.0, 60.0, 60.01, -0.1, np.nan, 20.0, 30.0]))
    assert np.isnan(g[[2, 3, 4, 5]]).all()
    assert g[0] == 0.0 and g[1] == 60.0 and g[6] == 30.0
    assert list(p) == [1, 1, 1, 1, 1, 0, 1]


def test_sub_rows_cap_and_determinism():
    a = np.arange(50000)
    s1, s2 = E.sub_rows(a), E.sub_rows(a)
    assert len(s1) == E.PROBE_CAP and np.array_equal(s1, s2) and (np.diff(s1) > 0).all()
    assert len(E.sub_rows(np.arange(10))) == 10


# ----------------------------------------------------------------------------- e4: alignment, and a shifted offset must fail
def test_e4_join_alignment_and_shift_fails():
    D = synth(n_clips=10, frames=40)
    J = synth_join(D)
    ok = E.check_e4(D, J, n_rand=30)
    assert ok["pass"], ok
    assert ok["labels_vs_join"]["n_checked"] == 30 and ok["labels_vs_join"]["bad"] == 0
    bad = E.check_e4(D, synth_join(D, shift=1), n_rand=30)
    assert not bad["pass"] and bad["labels_vs_join"]["bad"] > 0


def test_e4_eval_clip_count_and_frames_mismatch():
    D = synth(n_clips=6, frames=30)
    J = synth_join(D)
    r = E.check_e4(D, J, eval_clips=["clip002", "other"], expect_clips=6)
    assert not r["pass"] and r["eval_clips_in_probe_set"]["count"] == 1
    J2 = synth_join(D); J2["n"] = J2["n"].copy(); J2["n"][0] = 31
    assert not E.check_e4(D, J2)["join_n_equals_index_frames"]["pass"]
    assert not E.check_e4(D, J, expect_clips=7)["clip_count"]["pass"]


def test_e4_kinematics_reference_exact():
    D = synth(n_clips=4, frames=30)
    J = synth_join(D)
    ref = {(str(D["beid"][i]), int(D["bg"][i])): D["kin"][i].copy() for i in (3, 50, 99)}
    assert E.check_e4(D, J, kin_ref=ref)["kinematics_vs_cache"]["pass"]
    k = (str(D["beid"][50]), int(D["bg"][50]))
    ref[k] = ref[k] + 1e-9
    assert not E.check_e4(D, J, kin_ref=ref)["kinematics_vs_cache"]["pass"]
    assert not E.check_e4(D, J, require_kin_ref=True)["pass"]


# ----------------------------------------------------------------------------- the fold-leak guard and its mutation
def test_fold_guard_green_then_red_under_mutation():
    D = synth(n_clips=20, frames=60)
    folds = RP.make_folds(D["beid"])
    tr = np.flatnonzero(folds != 0)
    assert E.guard_fold_leak(D["X"], tr, n_probe=1000)["pass"]
    S0.LEAK_PCA_FIT = True
    try:
        assert not E.guard_fold_leak(D["X"], tr, n_probe=1000)["pass"]
    finally:
        S0.LEAK_PCA_FIT = False
    assert E.guard_fold_leak(D["X"], tr, n_probe=1000)["pass"]


# ----------------------------------------------------------------------------- the lead gap in ONE cell: grid reads it, mean does not
def test_planted_gap_in_one_cell_grid_beats_mean():
    D = synth(n_clips=40, frames=50, planted_cell=7)
    folds = RP.make_folds(D["beid"])
    M = S0.mean_token(D["X"])
    out = {}
    for f in (0, 1):
        tr = np.flatnonzero(folds != f)
        St = {"g": S0.grid_state(D["X"], tr), "m": S0.mean_state(M, tr, E.TOKD)}
        assert St["g"].shape == (len(D["X"]), 640) and St["m"].shape == (len(D["X"]), 64)
        E.ra_fold(f, St, D["kin"], D["gap"], D["pres"], D["epi"], folds, out, "main")
    te = np.isfinite(out["ra|main|g|gap"])
    r2g, r2m = RP.r2_pooled(D["gap"], out["ra|main|g|gap"]), RP.r2_pooled(D["gap"], out["ra|main|m|gap"])
    assert te.sum() > 0 and r2g > 0.4 and r2g > r2m + 0.25, (r2g, r2m)


# ----------------------------------------------------------------------------- planted position and planted motion on 64-wide cells
def _e2_inputs(D):
    folds = RP.make_folds(D["beid"])
    M = S0.mean_token(D["X"])
    pl = S0.planted_setup(D)
    side = (np.random.default_rng(6).random(len(D["X"])) < 0.5).astype(np.float64)
    return folds, M, pl, side


def test_planted_position_grid_reads_mean_does_not():
    D = synth(n_clips=40, frames=50)
    folds, M, pl, side = _e2_inputs(D)
    assert pl["vec_f5"].shape == (64,)
    sg, sm = np.full(len(D["X"]), np.nan), np.full(len(D["X"]), np.nan)
    for f in (0, 1, 2):
        o = E.e2_position_fold(f, D["X"], M, folds, D["epi"], pl, side)
        for s, a in (("g", sg), ("m", sm)):
            m = np.isfinite(o[s]); a[m] = o[s][m]
    ag, am = RP.auc_pooled(side, sg), RP.auc_pooled(side, sm)
    assert ag > 0.95 and am <= 0.60, (ag, am)
    assert S0.check_s2(ag, am)["pass"]


def test_planted_motion_grid_two_frames_reads_mean_and_one_frame_do_not():
    D = synth(n_clips=40, frames=50)
    folds, M, pl, side_unused = _e2_inputs(D)
    D["gap"][:] = np.clip(D["gap"], 5, 50)
    Pr = E.assemble(D)
    assert len(Pr["ri"]) > 500
    spec = S0C.k2_setup(D)
    side = S0C.k2_side(len(Pr["ri"]), spec["seed_side"])
    sc = {k: np.full(len(Pr["ri"]), np.nan) for k in ("grid2", "mean2", "grid1")}
    for f in (0, 1, 2):
        tr = np.flatnonzero(folds != f)
        St = {"g": S0.grid_state(D["X"], tr), "m": S0.mean_state(M, tr, E.TOKD)}
        trp, tep = np.flatnonzero(Pr["fold_p"] != f), np.flatnonzero(Pr["fold_p"] == f)
        o = S0C.k2_fold(D["X"], M, St, E.sub_rows(trp), tep, Pr["ri"], Pr["rm"], spec, side, Pr["ep"], f)
        for k, v in o.items():
            sc[k][tep] = v
    a = {k: RP.auc_pooled(side, v) for k, v in sc.items()}
    assert S0C.check_k2(a["grid2"], a["mean2"], a["grid1"])["pass"], a


# ----------------------------------------------------------------------------- pairs and the change read
def test_pairs_use_gap_range_and_continuity_rule():
    D = synth(n_clips=8, frames=40)
    D["gap"][5] = np.nan                                   # inside clip 0: breaks every window that contains frame 5
    D["gap"][100] += 20.0                                  # a 20 m jump inside clip 2
    Pr = E.assemble(D)
    c = Pr["counts"]
    assert c["pairs_removed_by_the_continuity_rule"] > 0 and c["pairs_used"] == len(Pr["ri"])
    assert (np.abs(np.diff(np.stack([D["gap"][Pr["ri"] + q] for q in range(11)], 1), axis=1)) <= S0C.STEP_MAX_M).all()
    assert (D["bg"][Pr["rk"]] - D["bg"][Pr["ri"]] == 10).all() and (D["bg"][Pr["ri"]] - D["bg"][Pr["rm"]] == 5).all()
    assert Pr["Eb"].shape == (len(Pr["ri"]), 9)
    assert np.allclose(Pr["d"], D["gap"][Pr["rk"]] - D["gap"][Pr["ri"]])


def test_change_fold_runs_and_e1_style_permutation_gives_no_skill():
    D = synth(n_clips=36, frames=60, planted_cell=7)
    Pr = E.assemble(D)
    folds = Pr["folds_b"]
    M = S0.mean_token(D["X"])
    out = {}
    d_e1 = S0.permute_within_fold(Pr["d"], Pr["fold_p"], 1)
    for f in (0, 1):
        tr = np.flatnonzero(folds != f)
        St = {"g": S0.grid_state(D["X"], tr), "m": S0.mean_state(M, tr, E.TOKD)}
        info = E.change_fold(f, St, Pr, {"main": Pr["d"], "e1": d_e1}, out, folds)
        assert info["pairs_train_used"] <= E.PROBE_CAP and info["pairs_test"] > 0
    assert out["arm|main|H-g"].shape == (len(Pr["ri"]),)
    for a in E.ARMS:
        m = np.isfinite(out[f"arm|e1|{a}"])
        assert S0C.skill(d_e1[m], out[f"arm|e1|{a}"][m]) <= E.E1_BAR + 0.03     # one tiny synthetic fold pair: slack above the real bar
    sk = S0C.skill(Pr["d"][np.isfinite(out["arm|main|E"])], out["arm|main|E"][np.isfinite(out["arm|main|E"])])
    assert np.isfinite(sk)


def test_outcome_names_and_chunked_auc_match_unchunked():
    rng = np.random.default_rng(0)
    ep = rng.integers(0, 30, 3000); y = (rng.random(3000) < 0.4).astype(float); s = rng.random(3000) + 0.3 * y
    C = RP.boot_counts(30, 120, 0)
    assert np.allclose(E.boot_auc_chunked(ep, y, s, C, chunk=25), RP.boot_auc(ep, y, s, C))
    pos = {"diff": 0.1, "separated_pos": True, "ci95": [0.01, 0.2]}
    neg = {"diff": 0.1, "separated_pos": False, "ci95": [-0.01, 0.2]}
    assert E.ra_outcomes({"grid minus ego": pos, "mean minus ego": neg, "grid minus mean": pos}) == {"grid state": "S0-LEAD-KEPT", "mean state": "S0-LEAD-LOST", "pair": "GRID-ADDS"}
    assert E.ra_outcomes({"grid minus ego": neg, "mean minus ego": neg, "grid minus mean": neg})["pair"] == "GRID-EQUAL"


def test_block_swap_keeps_continuity_and_breaks_alignment():
    D = synth(n_clips=12, frames=30)
    orig = D["gap"].copy()
    E.block_swap_tracks(D, 0)
    assert not np.array_equal(D["gap"], orig)
    assert np.array_equal(D["gap_loaded"], orig)           # as-loaded labels stay for e4
    for c in range(12):
        assert np.abs(np.diff(D["gap"][D["epi"] == c])).max() < 5.0   # tracks stay smooth
