"""``taniteval.nav_compliance`` — pinned on synthetic arms whose reading is KNOWN.

The centrepiece is the trio of synthetic models that the verdict must tell
apart, because a metric that reads the same for all three is not a
measurement (the route-label bijection, D-REFAV1-ROUTE-LABEL-IS-THE-NAV):

  follower   turns the FED way when the turn is imminent, holds otherwise
             -> FOLLOWS_NAV: compliance drops under shuffle/flip, token-
                following under flip reads 1.0
  blind      never turns (a nav-blind straight planner)
             -> IGNORES_NAV with rate 0.0; every delta exactly 0.0
  gt_echo    reproduces the GT whatever the token (scene-coincident)
             -> IGNORES_NAV with rate 1.0; every delta exactly 0.0 — HIGH
                compliance and NO nav effect, the reading the old metric
                would have published as skill

Plus the deliberate-regression arm of the STRATA: a stale window (turn already
over) must be excluded, or a constant per-clip token would score a model for
not re-doing a turn it already made.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "taniteval"), str(_REPO / "stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import nav_compliance as nc  # noqa: E402

N_BOOT = 200


# --------------------------------------------------------------------------- #
# the synthetic corpus                                                         #
# --------------------------------------------------------------------------- #
def _corpus(seed: int = 0, n_ep: int = 12, per_ep: int = 10):
    """Windows over ``n_ep`` episodes. Even episodes carry a turn command with
    a mix of imminent / deferred / stale windows; odd episodes are follow
    windows whose GT wobbles like a road (the threshold reference)."""
    rng = np.random.default_rng(seed)
    eid, cmd, valid, ts, te, gt = [], [], [], [], [], []
    for e in range(n_ep):
        turn = e % 2 == 0
        c = (nc.NAV_LEFT if e % 4 == 0 else nc.NAV_RIGHT) if turn else nc.NAV_FOLLOW
        for w in range(per_ep):
            eid.append(f"ep{e:03d}")
            cmd.append(c)
            valid.append(True)
            if not turn:
                ts.append(np.nan); te.append(np.nan)
                gt.append(rng.normal(0.0, 0.05))            # road wobble
                continue
            # window w: the turn starts at 12 - 2w s and lasts 4 s
            start = 12.0 - 2.0 * w
            ts.append(start); te.append(start + 4.0)
            if start + 4.0 <= 0:            # stale: turn is over
                gt.append(rng.normal(0.0, 0.05))
            elif start <= 6.0:              # imminent: GT turns within 6 s
                gt.append(nc.SIGN[c] * (0.6 + 0.1 * rng.random()))
            else:                            # deferred: GT still straight
                gt.append(rng.normal(0.0, 0.05))
    return {"eid": np.array(eid), "cmd": np.array(cmd), "cmd_valid": np.array(valid),
            "turn_start_rel_s": np.array(ts), "turn_end_rel_s": np.array(te),
            "gt_dpsi": np.array(gt)}


def _arm(fed, model: str, gt, cmd, ts, rng):
    """One arm's readouts under a FED token, for the three synthetic models."""
    sgn_fed = np.where(fed == nc.NAV_LEFT, 1.0, np.where(fed == nc.NAV_RIGHT, -1.0, 0.0))
    imminent_like = np.isfinite(ts) & (ts <= 6.0) & (ts > -4.0)
    if model == "follower":
        dpsi = np.where(imminent_like, sgn_fed * 0.7, rng.normal(0.0, 0.03, fed.size))
    elif model == "blind":
        dpsi = rng.normal(0.0, 0.03, fed.size)
    elif model == "gt_echo":
        dpsi = np.asarray(gt, float).copy()
    elif model == "echo_template":                     # turns the fed way ALWAYS
        dpsi = sgn_fed * 0.7
    else:
        raise ValueError(model)
    return {"fed": fed, "plan_dpsi": dpsi, "plan_y": dpsi * 10.0,
            "anchor_dpsi": dpsi.copy(),
            "gstr_angle": dpsi * 0.8,          # goal points the same way
            "core_lat": nc.classify_lateral(dpsi, 0.3),
            "tac_lat": nc.classify_lateral(dpsi, 0.3)}


def _win(model: str, seed: int = 0):
    c = _corpus(seed)
    rng = np.random.default_rng(seed + 1)
    cmd = c["cmd"]
    fed_true = cmd.copy()
    perm = rng.permutation(cmd.size)
    fed_shuf = cmd[perm]
    fed_flip = nc.flip_nav(cmd)
    fed_zero = np.full(cmd.size, -1)
    arms = {
        "nav_true": _arm(fed_true, model, c["gt_dpsi"], cmd, c["turn_start_rel_s"], rng),
        "nav_shuffle": _arm(fed_shuf, model, c["gt_dpsi"], cmd, c["turn_start_rel_s"], rng),
        "nav_flip": _arm(fed_flip, model, c["gt_dpsi"], cmd, c["turn_start_rel_s"], rng),
        "nav_zero": _arm(np.zeros_like(cmd), model, c["gt_dpsi"], cmd,
                         c["turn_start_rel_s"], rng),
    }
    arms["nav_zero"]["fed"] = fed_zero
    return dict(c, arms=arms)


@pytest.fixture(scope="module")
def reports():
    return {m: nc.compliance_report(_win(m), n_boot=N_BOOT, seed=0)
            for m in ("follower", "blind", "gt_echo")}


# --------------------------------------------------------------------------- #
# geometry pinned to the programme's own emitter                               #
# --------------------------------------------------------------------------- #
def test_heading_change_matches_four_families_maneuver_kinematics():
    torch = pytest.importorskip("torch")
    from taniteval import four_families as ff
    rng = np.random.default_rng(3)
    P = np.cumsum(rng.normal([1.0, 0.0], [0.2, 0.4], size=(40, 8, 2)), axis=1)
    P[5] = 0.0                                   # never moved -> 0.0
    P[6, 4:] = P[6, 3:4]                         # stops: heading HELD
    mine = nc.heading_change(P, min_ds_m=0.05)
    dyaw, *_ = ff.maneuver_kinematics(torch.as_tensor(P).float(), dt=0.5)
    assert np.allclose(mine, dyaw.numpy(), atol=1e-5)


def test_nav_indices_are_pinned_to_refb():
    from tanitad.refs import refb
    assert tuple(refb.NAV_COMMANDS[:3]) == nc.NAV_NAMES
    assert refb.NAV_COMMANDS.index("follow") == nc.NAV_FOLLOW
    assert refb.NAV_COMMANDS.index("left") == nc.NAV_LEFT
    assert refb.NAV_COMMANDS.index("right") == nc.NAV_RIGHT


def test_classify_and_flip_are_nav_aligned():
    cls = nc.classify_lateral(np.array([0.5, -0.5, 0.0, np.nan]), 0.3)
    assert cls.tolist() == [nc.NAV_LEFT, nc.NAV_RIGHT, nc.STRAIGHT, -1]
    assert nc.flip_nav(np.array([0, 1, 2, 3])).tolist() == [0, 2, 1, 3]


def test_threshold_is_derived_from_follow_windows_only_and_fails_loud_on_empty():
    c = _corpus()
    thr = nc.derive_threshold(c["gt_dpsi"], c["cmd"] == nc.NAV_FOLLOW)
    assert 0.0 < thr["theta"] < 0.3, thr
    assert thr["n_ref"] == int((c["cmd"] == nc.NAV_FOLLOW).sum())
    with pytest.raises(ValueError):
        nc.derive_threshold(c["gt_dpsi"], np.zeros(c["cmd"].size, bool))


# --------------------------------------------------------------------------- #
# the strata                                                                   #
# --------------------------------------------------------------------------- #
def test_strata_partition_and_stale_windows_are_excluded():
    c = _corpus()
    thr = nc.derive_threshold(c["gt_dpsi"], c["cmd"] == nc.NAV_FOLLOW)
    gt_cls = nc.classify_lateral(c["gt_dpsi"], thr["theta"])
    s = nc.assign_strata(c["cmd"], c["cmd_valid"], gt_cls,
                         c["turn_start_rel_s"], c["turn_end_rel_s"], 6.0)
    assert set(s) <= set(nc.STRATA)
    # every turn window is in exactly one stratum, follow windows are 'follow'
    assert (s[c["cmd"] == nc.NAV_FOLLOW] == "follow").all()
    turn = c["cmd"] != nc.NAV_FOLLOW
    assert (s[turn] != "follow").all()
    stale = turn & (c["turn_end_rel_s"] <= 0)
    assert stale.any() and (s[stale] == "stale").all()
    assert (s == "imminent").sum() > 0 and (s == "deferred").sum() > 0


def test_DELIBERATE_REGRESSION_a_stale_window_never_scores():
    """Score a model that (correctly) does not re-turn on stale windows: if the
    strata leaked stale windows into 'imminent', its compliance would drop."""
    w = _win("follower")
    rep = nc.compliance_report(w, n_boot=N_BOOT)
    stale = rep["strata"]["n_windows"]["stale"]
    assert stale > 0
    assert rep["readouts"]["plan"]["nav_true"]["imminent"]["rate"] == 1.0


# --------------------------------------------------------------------------- #
# ⭐ the three models read DIFFERENTLY                                          #
# --------------------------------------------------------------------------- #
def test_follower_reads_FOLLOWS_NAV(reports):
    r = reports["follower"]
    assert r["known_value_controls"]["harness_ok"] is True
    assert r["verdict"]["nav_effect"] == "FOLLOWS_NAV", r["verdict"]
    plan = r["readouts"]["plan"]
    assert plan["nav_true"]["imminent"]["rate"] == 1.0
    assert plan["nav_true"]["deferred"]["rate"] == 1.0
    flip = r["controls"]["nav_flip"]["plan"]["imminent"]
    assert flip["separated"] and flip["delta"] > 0
    assert flip["token_following_rate"] == 1.0
    zero = r["controls"]["nav_zero"]["plan"]["imminent"]
    assert zero["separated"] and zero["delta"] > 0


def test_blind_reads_IGNORES_NAV_with_zero_deltas(reports):
    r = reports["blind"]
    assert r["verdict"]["nav_effect"] == "IGNORES_NAV", r["verdict"]
    plan = r["readouts"]["plan"]
    assert plan["nav_true"]["imminent"]["rate"] == 0.0
    assert plan["nav_true"]["deferred"]["rate"] == 1.0
    for a in ("nav_shuffle", "nav_flip", "nav_zero"):
        d = r["controls"][a]["plan"]["imminent"]
        assert d["delta"] == 0.0 and not d["separated"], (a, d)


def test_gt_echo_reads_high_compliance_but_IGNORES_NAV(reports):
    """The reading the route metric mistook for skill: perfect compliance under
    the true token, and NO nav effect."""
    r = reports["gt_echo"]
    assert r["readouts"]["plan"]["nav_true"]["imminent"]["rate"] == 1.0
    assert r["verdict"]["nav_effect"] == "IGNORES_NAV", r["verdict"]
    for a in ("nav_shuffle", "nav_flip", "nav_zero"):
        d = r["controls"][a]["plan"]["imminent"]
        assert d["delta"] == 0.0 and not d["separated"], (a, d)


def test_echo_template_is_indistinguishable_from_a_follower_HERE():
    """⚠️ The stated limit: a canned turn in the fed direction reads exactly
    like a follower on this metric. That is WHY the block is one row beside the
    four families and never a score — the template drives into the kerb and
    the LATERAL family sees it. Pinned so the limit stays documented."""
    r = nc.compliance_report(_win("echo_template"), n_boot=N_BOOT)
    assert r["verdict"]["nav_effect"] == "FOLLOWS_NAV"
    assert r["controls"]["nav_flip"]["plan"]["imminent"]["token_following_rate"] == 1.0
    # ...but it turns EARLY on deferred windows, which the hold rate exposes
    assert r["readouts"]["plan"]["nav_true"]["deferred"]["rate"] == 0.0
    assert any("cannot separate token-following from token-echo" in s
               for s in r["_falsifiability"])


# --------------------------------------------------------------------------- #
# controls, consistency, power                                                 #
# --------------------------------------------------------------------------- #
def test_known_value_controls_read_exact_values(reports):
    kv = reports["follower"]["known_value_controls"]
    assert kv["harness_ok"] is True
    for name, chk in kv["checks"].items():
        assert chk["pass"], (name, chk)


def test_consistency_crosstab_names_the_seam_failure(reports):
    r = reports["follower"]
    c = r["consistency"]["imminent"]
    assert c["both_ok"]["rate"] == 1.0
    assert c["goal_ok_plan_wrong__SEAM_FAILURE"]["rate"] == 0.0
    assert "SEAM_FAILURE" in c["_reads"]


def test_seam_failure_is_separated_from_nav_failure():
    """A model whose goal points the right way but whose plan drives straight
    must land in the SEAM_FAILURE cell, not read as 'ignores nav'."""
    w = _win("follower")
    for a in w["arms"].values():
        a["plan_dpsi"] = np.zeros_like(a["plan_dpsi"])   # plan never turns
        a["anchor_dpsi"] = np.zeros_like(a["anchor_dpsi"])
    r = nc.compliance_report(w, n_boot=N_BOOT)
    c = r["consistency"]["imminent"]
    assert c["goal_ok_plan_wrong__SEAM_FAILURE"]["rate"] == 1.0
    assert r["readouts"]["g_str"]["nav_true"]["imminent"]["rate"] == 1.0
    assert r["readouts"]["plan"]["nav_true"]["imminent"]["rate"] == 0.0


def test_underpowered_stratum_is_CANNOT_RULE_not_a_number():
    w = _win("follower")
    keep = np.isin(w["eid"], ["ep000", "ep001", "ep002"])   # 3 episodes only
    w2 = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in w.items()
          if k != "arms"}
    w2["arms"] = {a: {k: (v[keep] if isinstance(v, np.ndarray) else v)
                      for k, v in arm.items()} for a, arm in w["arms"].items()}
    r = nc.compliance_report(w2, n_boot=N_BOOT)
    assert r["verdict"]["nav_effect"] == "CANNOT_RULE", r["verdict"]
    assert r["readouts"]["plan"]["nav_true"]["imminent"]["powered"] is False


def test_estimator_is_the_paired_episode_cluster_bootstrap(reports):
    r = reports["follower"]
    d = r["controls"]["nav_flip"]["plan"]["imminent"]
    assert d["estimator"] == "paired_episode_cluster_bootstrap"
    assert "n_episodes" in d and d["n_episodes"] >= nc.MIN_N_EPISODES
    assert "overlapping_holdout_se" not in str(r)


def test_missing_readout_is_reported_absent_not_zero_filled():
    w = _win("follower")
    for a in w["arms"].values():
        a["gstr_angle"] = None
    r = nc.compliance_report(w, n_boot=N_BOOT)
    assert r["readouts"]["g_str"]["nav_true"]["status"] == "UNAVAILABLE"
    assert r["consistency"]["imminent"]["status"] == "UNAVAILABLE"
