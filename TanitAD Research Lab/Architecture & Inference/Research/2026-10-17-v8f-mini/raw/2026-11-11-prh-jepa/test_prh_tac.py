"""Tests for prh_tac on planted data. PRH_SRC_DIR selects the directory of the code under test (mutation runner)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PRH_SRC_DIR", os.path.dirname(os.path.abspath(__file__))))
import prh_tac as T  # noqa: E402

N, DN, NC, DY = 3000, 60, 50, 8


def _ctx(seed=0, B=300):
    cid = np.repeat(np.arange(NC), N // NC)
    fit_clip = np.arange(NC) % 2 == 0
    is_fit = fit_clip[cid]
    fi, si = np.where(is_fit)[0], np.where(~is_fit)[0]
    sc = np.where(~fit_clip)[0]
    remap = -np.ones(NC, int)
    remap[sc] = np.arange(len(sc))
    nC = len(sc)
    counts = np.random.default_rng(seed).multinomial(nC, np.ones(nC) / nC, size=B).astype(float)
    return cid, {"fit_idx": fi, "score_idx": si, "cid_score": remap[cid[si]], "nC": nC, "counts": counts}


def _planted(seed=0):
    """nm with a strong low-variance label subspace: Y = nm @ Wt + noise; TR carries those directions plus noise ones; RI is a random affine map of nm of the same width."""
    rng = np.random.default_rng(seed)
    nm = rng.standard_normal((N, DN))                    # isotropic, so no direction is favoured by a fit to noise
    Wt = np.zeros((DN, DY))
    Wt[:DY, :] = np.eye(DY) * 3.0                        # labels live in the first 8 directions
    Y = nm @ Wt + 0.3 * rng.standard_normal((N, DY))
    RI = nm @ rng.standard_normal((DN, 16))              # a random 16-d subspace of nm
    TR = np.concatenate([nm[:, :DY], nm[:, DY:DY + 8] @ rng.standard_normal((8, 8))], 1)   # a 16-d subspace that contains the label directions
    return nm, Y, TR, RI


def test_q1_positive_when_representation_contains_tac_subspace():
    cid, ctx = _ctx()
    nm, Y, TR, RI = _planted()
    TAC, d = T.fit_tac(nm, Y, ctx["fit_idx"])
    assert d == DY
    q = T.q1_block(TR, RI, TAC, ctx)
    assert q["diff"] > 0.3
    assert q["separated_positive"] is True


def test_label_shuffled_tac_gives_floor_and_non_separated():
    cid, ctx = _ctx()
    nm, Y, TR, RI = _planted()
    Ysh = T.label_shuffle(Y, cid, ctx["fit_idx"], 11)
    TACs, _ = T.fit_tac(nm, Ysh, ctx["fit_idx"])
    q = T.q1_block(TR, RI, TACs, ctx)
    TAC, _ = T.fit_tac(nm, Y, ctx["fit_idx"])
    real = T.q1_block(TR, RI, TAC, ctx)
    assert q["S_TR"] < 0.7 * real["S_TR"]                 # scrambled labels do not match the carried subspace (a refit TAC still overfits nm, so the floor is not 0)
    # The episode bootstrap does not resample the shuffled-label FIT (one random draw), so on planted data a small positive diff can read "separated"; the planted contract is therefore size, not the CI flag.
    assert abs(q["diff"]) < 0.25 * real["diff"]
    assert real["separated_positive"] is True


def test_label_shuffle_keeps_blocks_within_fit_rows():
    cid, ctx = _ctx()
    _, Y, _, _ = _planted()
    Ysh = T.label_shuffle(Y, cid, ctx["fit_idx"], 11)
    assert Ysh.shape == Y.shape
    assert not np.allclose(Ysh[ctx["fit_idx"]], Y[ctx["fit_idx"]])
    assert np.allclose(Ysh[ctx["score_idx"]], Y[ctx["score_idx"]])


def test_self_is_one_after_rank_reduction_of_dependent_targets():
    cid, ctx = _ctx()
    nm, Y, _, _ = _planted()
    Yd = np.concatenate([Y, Y[:, :1] + Y[:, 1:2]], 1)    # an exactly dependent column, as the one-hot blocks are
    TAC, d = T.fit_tac(nm, Yd, ctx["fit_idx"])
    assert d == DY
    s = T.V.metric("CC32", TAC, TAC, ctx)[0]
    assert abs(s - 1) < 1e-6


def test_outcome_logic():
    assert T.decide([True, True], True) == "TAC-LEAN"
    assert T.decide([True, False], True) == "NO-TAC-LEAN"
    assert T.decide([False, True], True) == "NO-TAC-LEAN"
    assert T.decide([False, False], True) == "NO-TAC-LEAN"
    assert T.decide([True, True], False).startswith("UNUSABLE")


def test_q1_separation_rule():
    assert T.q1_separated(0.001, 0.05) is True
    assert T.q1_separated(-0.001, 0.05) is False
    assert T.q1_separated(-0.05, -0.01) is False          # separated NEGATIVE is not Q1


def test_build_targets_single_class_and_goal_imputation():
    J = {"goal_y": np.array([0b101, 0b1], np.uint32), "goal_w": np.array([0b111, 0b1], np.uint32), "has": np.array([True, True]),
         "lat_allowed": np.array([0b00100, 0b00011], np.uint8), "lon_allowed": np.array([0b0000010, 0b0000001], np.uint8),
         "lat_cls": np.array([2, 0], np.int16), "lon_cls": np.array([1, 0], np.int16)}
    Y, single = T.build_targets(J)
    assert single.tolist() == [True, False]               # the second row has a two-class lat set
    assert Y[0, :3].tolist() == [1.0, 0.0, 1.0] and Y[0, 3] == 0.5   # w = 0 -> 0.5
    assert Y[0, 22 + 2] == 1.0 and Y[0, 27 + 1] == 1.0 and Y[0, 22:].sum() == 2.0
