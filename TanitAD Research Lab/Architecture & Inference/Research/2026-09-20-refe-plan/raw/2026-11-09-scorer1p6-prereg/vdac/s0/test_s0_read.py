"""Synthetic tests of s0_read.py: footprint / distance on a hand-made raster, each control (incl. mutations), the outcome table."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_read as S  # noqa: E402


def road_mask(half_w=5.0):
    """Road band |y| < half_w (a multiple of 0.5 m so the boundary sits on a cell edge), all x."""
    yc = S.Y0 + (np.arange(S.NY) + 0.5) * S.RES
    return np.tile((np.abs(yc) < half_w)[None, :], (S.NX, 1))


def straight(y_off=0.0, speed=5.0):
    t = 0.5 * np.arange(1, 9)
    return np.stack([speed * t, np.full(8, y_off), np.zeros(8)], axis=-1)


def test_sdf_half_plane_is_half_cell_corrected():
    m = np.zeros((S.NX, S.NY), bool)
    m[80:, :] = True
    f = S.sdf_from_mask(m)
    assert f[80, 10] == pytest.approx(0.25) and f[79, 10] == pytest.approx(-0.25)
    assert f[90, 10] == pytest.approx(5.25)


def test_degenerate_masks():
    assert (S.sdf_from_mask(np.ones((S.NX, S.NY), bool)) == S.BIG_M).all()
    assert (S.sdf_from_mask(np.zeros((S.NX, S.NY), bool)) == -S.BIG_M).all()


def test_densify_41_poses_and_unwrapped_yaw():
    p = np.zeros((8, 3))
    p[:, 0] = np.arange(1, 9)
    p[:, 2] = [1.5, 2.0, 2.5, 3.0, -3.0, -2.5, -2.0, -1.5]          # crosses pi
    d = S.densify(p)
    assert d.shape == (41, 3) and np.allclose(d[0], 0) and np.allclose(d[5, :2], p[0, :2]) and np.allclose(d[40, :2], p[7, :2])
    assert np.allclose(np.cos(d[40, 2]), np.cos(p[7, 2])) and np.allclose(np.sin(d[5, 2]), np.sin(p[0, 2]))
    assert np.abs(np.diff(d[5:, 2])).max() < 0.2                        # no 2 pi jump


def test_footprint_164_points_and_geometry():
    c = S.corners(S.densify(straight()))
    assert c.shape == (41, 4, 2) and c.reshape(-1, 2).shape[0] == 164
    # t0 pose: centre at x = 1.461, corners +-2.588 / +-1.1485
    assert c[0, :, 0].max() == pytest.approx(1.461 + 2.588) and c[0, :, 0].min() == pytest.approx(1.461 - 2.588)
    assert c[0, :, 1].max() == pytest.approx(1.1485) and c[0, :, 1].min() == pytest.approx(-1.1485)


def test_dk_margin_on_road_band():
    f = S.sdf_from_mask(road_mask(5.0))
    d, nin, npts = S.dk_proposals(f, np.stack([straight(0.0), straight(4.5), straight(-4.5), straight(8.0)]))
    assert npts == 164 and (nin == 164).all()
    assert d[0] == pytest.approx(5.0 - 1.1485, abs=0.05)
    assert d[1] == pytest.approx(5.0 - 4.5 - 1.1485, abs=0.05) and d[1] < 0
    assert d[2] == pytest.approx(d[1], abs=1e-6)
    assert d[3] < -3.0


def test_points_outside_extent_excluded_and_none_inside_gives_inf():
    f = S.sdf_from_mask(road_mask())
    d, nin, _ = S.dk_proposals(f, straight(0.0)[None], x0=1000.0)       # the raster sits far away: no footprint point is inside it
    assert nin[0] == 0 and np.isinf(d[0]) and S.phi(d[0] / 1.0) == 1.0


def test_pick_matches_sel_common_and_first_index_on_ties():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(50, 64, 6)).astype(np.float32)
    L[3, :, :] = 0.0                                                   # a full tie -> index 0
    p = S.p6_probs(L)
    base, _ = S.pick_from_probs(p)
    assert base[3] == 0
    assert S.pick_reproduction(base, L)["ok"]


def _world(n=6, seed=1):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, 64, 6)).astype(np.float32)
    pdms = rng.uniform(size=(n, 64))
    dac = (rng.uniform(size=(n, 64)) > 0.2).astype(np.float32)
    p = S.p6_probs(L)
    base, _ = S.pick_from_probs(p)
    return L, p, base, pdms * 100, dac


def test_zero_control_all_drivable_exact():
    L, p, base, pdms, dac = _world()
    dk_free = np.full((6, 64), np.inf)
    z = S.zero_control(p, base, dk_free, dac, pdms)
    assert z["ok"] and all(v == 0 for v in z["n_changed"].values()) and all(v == 0.0 for v in z["delta_sum"].values())
    dk_free2 = np.full((6, 64), S.BIG_M)                              # finite big field: Phi = 1.0 exactly in float32
    assert S.zero_control(p, base, dk_free2, dac, pdms)["ok"]


def test_zero_control_mutation_turns_red():
    L, p, base, pdms, dac = _world()
    dk = np.full((6, 64), np.inf)
    dk[np.arange(6), base] = -3.0                                      # every P6 pick sits 3 m off-road
    assert not S.zero_control(p, base, dk, dac, pdms)["ok"]


def test_planted_offroad_field_moves_the_pick_for_every_rule_and_only_there():
    """Reachability: a multiplier that never changes a pick is a switch that is off."""
    L, p, base, pdms, dac = _world(n=8)
    dk = np.full((8, 64), np.inf)
    dk[:3, :][np.arange(3), base[:3]] = -5.0                          # sets 0-2: the pick is 5 m off-road
    picks = S.arm_picks(p, base, dk, dac)
    for arm in ("V0", "V25", "Phi-GT0.5", "Phi-GT1.0", "Phi-GT2.0"):
        assert (picks[arm][:3] != base[:3]).all(), arm
        assert (picks[arm][3:] == base[3:]).all(), arm


def test_veto_falls_back_to_p6_pick_when_all_dropped():
    L, p, base, pdms, dac = _world(n=2)
    pk = S.masked_pick(p, np.ones((2, 64), bool), base)
    assert (pk == base).all()


def test_odac_drops_dac_zero_picks():
    L, p, base, pdms, dac = _world(n=40, seed=3)
    pk = S.arm_picks(p, base, np.full((40, 64), np.inf), dac)["O-DAC"]
    ok_rows = ~(dac == 0).all(1)
    assert (dac[np.arange(40), pk][ok_rows] == 1).all()


def test_odac_dominance_control_and_its_mutation():
    assert S.odac_dominates({"O-DAC": 1.5, "V0": 0.4, "Phi-GT1.0": 0.5})["ok"]
    assert not S.odac_dominates({"O-DAC": 0.3, "V0": 0.4, "Phi-GT1.0": 0.1})["ok"]


def test_yflip_control_and_its_mutation():
    assert S.yflip_below({"V0": 0.30, "Phi-GT1.0": 0.35}, {"V0": 0.05, "Phi-GT1.0": 0.10})["ok"]
    assert not S.yflip_below({"V0": 0.30, "Phi-GT1.0": 0.35}, {"V0": 0.05, "Phi-GT1.0": 0.36})["ok"]


def test_yflip_changes_the_field_on_an_asymmetric_road():
    m = road_mask(5.0)
    m[:, 64 + 4:] = False                                              # road spans y in (-5, 2): asymmetric about y = 0
    d_t = S.dk_proposals(S.sdf_from_mask(m), straight(2.0)[None])[0][0]
    d_f = S.dk_proposals(S.sdf_from_mask(m[:, ::-1]), straight(2.0)[None])[0][0]
    assert d_t != pytest.approx(d_f)


@pytest.mark.parametrize("r,exp", [(0.28, "S0-NEAR"), (0.33, "S0-GO"), (0.2799999, "S0-STOP"), (0.3299999, "S0-NEAR"), (0.9, "S0-GO"),
                                   (0.0, "S0-STOP"), (-0.1, "S0-STOP"), (float("nan"), "INVALID")])
def test_outcome_table(r, exp):
    assert S.outcome(r) == exp


def test_bootstrap_ratio_block_identity_and_shape():
    log = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    boot = S.Boot(log, B=500, seed=1)
    g = np.linspace(0.0, 2.0, 15)
    blk, rd = S.ratio_block(boot, 0.5 * g, g)
    assert blk["R"] == pytest.approx(0.5)
    assert np.allclose(rd[np.isfinite(rd)], 0.5)
    assert blk["R_ci"][0] <= 0.5 <= blk["R_ci"][1]


def test_card_hash_constant_matches_file():
    if os.path.exists(S.CARD):
        assert S.sha256_file(S.CARD) == S.CARD_SHA
