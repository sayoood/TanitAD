"""test_anticheat_score.py -- known-value + deliberate-regression tests of the floor and of the verdict logic (anticheat_score.py).  CPU, numpy only.
Expectations are LITERALS (a recovered coefficient, a count, a status string), never an expression over the code under test."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import anticheat_score as S  # noqa: E402


def _train(n=600, n_ep=30, seed=0, resid=True):
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(0, 30, n)
    a0 = rng.normal(0, 1, n)
    om = rng.normal(0, 0.05, n)
    lim = rng.uniform(8, 40, n)
    X = np.stack([np.ones(n), v0, a0, om, v0 * om, v0 ** 2, a0 * v0, np.minimum(lim, 40), np.minimum(lim, 40) - v0, np.ones(n)], 1)
    t = np.array([0.5, 1.0, 1.5, 2.0])
    HP = np.stack([np.stack([v0 * tt, np.zeros(n)], 1) for tt in t], 1)                   # [n, 4, 2] constant-speed prior
    corr = np.stack([np.stack([0.4 * a0 * tt ** 2, 3.0 * om * v0 * tt], 1) for tt in t], 1)   # a LINEAR function of the features (a0 and v0*om)
    G = (HP + corr) if resid else (corr + 5.0)
    eid = np.repeat(np.arange(n_ep), n // n_ep)
    return {"X": X, "G": G.astype(np.float32), "HP": HP.astype(np.float32), "eid": eid}


def test_ridge_recovers_linear_target():
    tr = _train()
    y = tr["X"][:, 1:2]
    m = S.ridge_fit(tr["X"], y, 1e-3)
    p = S.ridge_pred(m, tr["X"])
    assert np.abs(p - y).max() < 1e-3


def test_fit_floor_picks_residual_when_target_is_prior_plus_linear():
    tr = _train(resid=True)
    fl = S.fit_floor(tr)
    assert fl["param"] == "RESIDUAL" and fl["n"] == 600 and fl["d"] == 10
    P = S.floor_predict(fl, tr["X"], tr["HP"])
    assert np.abs(P.astype(np.float64) - tr["G"].astype(np.float64)).max() < 0.05


def test_fit_floor_picks_direct_when_prior_is_irrelevant():
    tr = _train(resid=False)
    tr["HP"] = (np.random.default_rng(1).normal(0, 30, tr["HP"].shape)).astype(np.float32)        # a useless 'prior'
    fl = S.fit_floor(tr)
    assert fl["param"] == "DIRECT"


def test_folds_partition_episodes():
    eid = np.repeat(np.arange(40), 5)
    f = S.folds(eid)
    for e in range(40):
        assert len(set(f[eid == e].tolist())) == 1                   # an episode is never split across folds
    assert sorted(set(f.tolist())) == [0, 1, 2, 3, 4]


def test_known_value_controls_pass_and_shuffle_is_not_better_than_chance():
    kv = S.floor_known_value_controls(_train())
    assert kv["K_FL1_PASS"] is True and kv["K_FL2_PASS"] is True


def _cell(lo, hi):
    return {"delta": (lo + hi) / 2, "lo": lo, "hi": hi}


def _lib():
    return {"ade_m": True, "LON_speed_mae_mps": True, "LAT_heading_mae_deg": True, "TAC_traj_lat_correct": False, "TAC_traj_lon_correct": False}


def _table(better=True):
    err = _cell(-0.3, -0.1) if better else _cell(-0.3, 0.1)
    acc = _cell(0.1, 0.3) if better else _cell(-0.1, 0.3)
    return {sf: {"ade_m": err, "LON_speed_mae_mps": err, "LAT_heading_mae_deg": err, "TAC_traj_lat_correct": acc, "TAC_traj_lon_correct": acc} for sf in ("S2", "S6")}


def test_family_verdicts_pass_and_orientation():
    v = S.family_verdicts(_table(True), _lib())
    assert v["OVERALL"] == "PASS" and v["STRATEGIC"]["status"] == "N/A (n = 0)"
    assert all(v[f]["status"] == "PASS" for f in ("ADE", "LONGITUDINAL", "LATERAL", "TACTICAL"))


def test_family_verdicts_mutation_one_tac_row_not_separated_is_red():
    t = _table(True)
    t["S6"]["TAC_traj_lat_correct"] = _cell(-0.05, 0.2)                # ONE row not separated
    v = S.family_verdicts(t, _lib())
    assert v["TACTICAL"]["status"] == "FAIL (not separated)" and v["OVERALL"] == "FAIL: TACTICAL"
    assert v["ADE"]["status"] == "PASS"


def test_family_verdicts_separated_worse_and_flipped_orientation():
    t = _table(True)
    t["S2"]["ade_m"] = _cell(0.1, 0.3)                                 # error row, REAL separated WORSE
    v = S.family_verdicts(t, _lib())
    assert v["ADE"]["status"] == "FAIL (separated worse)"
    # MUTATION: treat the accuracy rows as error rows (the a0k0 'resummarise' mistake) -> a clear improvement reads as separated WORSE
    lib_bad = dict(_lib(), TAC_traj_lat_correct=True, TAC_traj_lon_correct=True)
    v2 = S.family_verdicts(_table(True), lib_bad)
    assert v2["TACTICAL"]["status"] == "FAIL (separated worse)"
