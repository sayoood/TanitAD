"""Unit tests for w_a7_read.py.  CPU only.  Run: CUDA_VISIBLE_DEVICES=-1 TMP=F:/tanitad_scratch/w_a7_tmp python -m pytest -q test_w_a7.py

Pure-logic tests (A7, plans, bootstrap, outcome rule, controls evaluation on synthetic score rows) run anywhere.  Two mutation tests make
the checks go RED: A7 skipped in the with-A7 arm (the RED/identity logic must notice) and a different bootstrap seed (the interval must move).
`test_scorer_*` run the real tracked scorer on 3 rows and need the dev-box data; skipped when absent (W_A7_SCORER=0)."""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w_a7_read as W  # noqa: E402

sys.path[:0] = [W.REFE]
from planner import repair_last_heading  # noqa: E402


def traj(seed=0):
    rng = np.random.default_rng(seed)
    t = np.zeros((20, 3))
    t[:, 0] = np.cumsum(rng.uniform(0.5, 1.0, 20))
    t[:, 1] = np.cumsum(rng.uniform(-0.05, 0.05, 20))
    t[:, 2] = np.cumsum(rng.uniform(-0.02, 0.02, 20))
    t[19, 2] += 2.0           # corrupt last heading, as the A7 defect describes
    return t


def test_a7_is_the_programme_function_and_only_touches_index19_heading():
    t = traj()
    r = repair_last_heading(t)
    assert r[19, 2] == t[18, 2]
    assert np.array_equal(r[:19], t[:19]) and np.array_equal(r[19, :2], t[19, :2])
    assert r is not t and t[19, 2] != t[18, 2]


def test_plans_arms_and_identity():
    t = traj()
    p = W.make_plans(t, t + 0.01, t + 0.02, repair_last_heading)
    assert list(p) == W.ARMS
    assert p["R_A7"][19, 2] == p["R_noA7"][18, 2] and p["R_noA7"][19, 2] != p["R_noA7"][18, 2]
    assert abs(W.wrap(p["RED_R_h19_plus05"][19, 2] - p["R_noA7"][19, 2]) - 0.5) < 1e-12
    assert np.array_equal(p["IDN_R_pre"], p["IDN_R_A7"])            # control (iii) premise
    assert np.array_equal(p["R_noA7"][:19], p["R_A7"][:19])


def test_plans_raw_fallback_uses_traj():
    t = traj()
    p = W.make_plans(t, None, None, repair_last_heading)
    assert np.array_equal(p["R_noA7"], t) and np.array_equal(p["T_noA7"], t)


def test_mutation_a7_skipped_is_caught_by_identity_of_with_and_without():
    """MUTATION: with-A7 arm built with a no-op A7.  R_A7 then equals R_noA7 on every row, i.e. delta == 0 exactly, which the
    pipeline's plan-level check (R_A7 != R_noA7 when h19 != h18) exposes."""
    t = traj()
    good = W.make_plans(t, t, t, repair_last_heading)
    bad = W.make_plans(t, t, t, lambda x: np.array(x, copy=True))
    assert not np.array_equal(good["R_A7"], good["R_noA7"])
    assert np.array_equal(bad["R_A7"], bad["R_noA7"])               # RED: the arm the A7 read exists to measure vanished


def synth_rows(n_logs=6, per=5, delta=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows, sc = [], []
    for i in range(n_logs * per):
        lg, tok = f"log{i // per}", f"t{i}"
        base = float(rng.uniform(0.5, 1.0))
        arm = lambda v: {"nc": 1.0, "pdms": v}                      # noqa: E731
        d = delta
        a = {"R_noA7": arm(base), "R_A7": arm(base + d), "T_noA7": arm(base), "T_A7": arm(base + d),
             "RED_R_h19_plus05": arm(base - 0.05), "IDN_R_pre": arm(base), "IDN_R_A7": arm(base)}
        r = {"log": lg, "token": tok, "arms": a, "rerun": dict(a), "idn_plan_equal": True, "label_src": {"p5r": "p5r", "tc20s": "tc20s"},
             "tc20s_vs_banked_maxabs": 1e-15}
        rows.append({"log": lg, "token": tok, "traj": np.zeros((20, 3)), "tc20s_banked": np.zeros((20, 3))})
        sc.append(r)
    return rows, sc


def _analyze(tmp_path, rows, sc, banked=None):
    run = str(tmp_path)
    os.makedirs(run + "/labels", exist_ok=True)
    open(run + "/labels/labels_s0of1.jsonl", "w").write("x\n")
    with open(run + "/scores.jsonl", "w") as f:
        for r in sc:
            f.write(json.dumps(r) + "\n")
    if banked is None:
        banked = {(r["log"], r["token"]): r["arms"]["T_noA7"]["pdms"] for r in sc}
    return W.analyze(run, rows, False, banked)


def test_outcome_pass_when_delta_zero(tmp_path):
    rows, sc = synth_rows(delta=0.0)
    res = _analyze(tmp_path, rows, sc)
    assert res["controls_all_pass"] and res["outcome_row"] == "W-A7-PASS"
    assert res["delta_pp_point_lo_hi"]["P5-R"][0] == 0.0


def test_outcome_rebuild_when_big_negative(tmp_path):
    rows, sc = synth_rows(delta=-0.01)                               # -1.0 pp on every row
    res = _analyze(tmp_path, rows, sc)
    assert res["outcome_row"] == "W-A7-REBUILD"


def test_outcome_pass_when_separated_but_small(tmp_path):
    rows, sc = synth_rows(delta=-0.0005)                             # -0.05 pp, CI < 0 but point > -0.10
    res = _analyze(tmp_path, rows, sc)
    assert res["delta_pp_point_lo_hi"]["P5-R"][2] < 0 and res["outcome_row"] == "W-A7-PASS"


def test_outcome_rule_boundaries():
    assert W.outcome_row([-0.10, -0.2, -0.01], True) == "W-A7-REBUILD"
    assert W.outcome_row([-0.0999, -0.2, -0.01], True) == "W-A7-PASS"
    assert W.outcome_row([-0.5, -0.9, 0.0], True) == "W-A7-PASS"      # CI upper bound not < 0
    assert W.outcome_row([-0.5, -0.9, -0.1], False) == "INVALID"


@pytest.mark.parametrize("which", ["i", "ii", "iii", "iv"])
def test_each_control_failing_makes_invalid(tmp_path, which):
    rows, sc = synth_rows(delta=-0.01)
    banked = None
    if which == "i":
        banked = {(r["log"], r["token"]): r["arms"]["T_noA7"]["pdms"] + 1e-3 for r in sc}
    if which == "ii":
        for r in sc:
            r["arms"]["RED_R_h19_plus05"] = dict(r["arms"]["R_noA7"])
    if which == "iii":
        sc[0]["arms"]["IDN_R_A7"] = {"nc": 1.0, "pdms": 0.123}
    if which == "iv":
        sc[0]["rerun"]["R_noA7"] = {"nc": 1.0, "pdms": sc[0]["arms"]["R_noA7"]["pdms"] + 1e-12}
    res = _analyze(tmp_path, rows, sc, banked)
    assert not res["controls_all_pass"] and res["outcome_row"] == "INVALID"
    key = {"i": "i_reproduction_tf1_tc20s_tracked", "ii": "ii_red_arm", "iii": "iii_identity", "iv": "iv_determinism"}[which]
    assert res["controls"][key]["pass"] is False


def test_bootstrap_seed_mutation_moves_interval():
    rng = np.random.default_rng(3)
    v = rng.normal(-0.002, 0.003, 60)
    logs = [f"l{i // 6}" for i in range(60)]
    a = W.bootstrap_delta({"x": v}, logs)["x"]
    b = W.bootstrap_delta({"x": v}, logs)["x"]
    c = W.bootstrap_delta({"x": v}, logs, seed=W.SEED + 1)["x"]
    assert a == b                                                     # deterministic at the frozen seed
    assert (a[1], a[2]) != (c[1], c[2])                               # RED: another seed moves the CI
    assert W.SEED == 20260927 and W.B_DRAWS == 10000


def test_bootstrap_matches_tf1_formula():
    v = np.array([0.0, 0.01, -0.02, 0.0, 0.03, -0.01, 0.0, 0.02])
    logs = ["a", "a", "b", "b", "c", "c", "d", "d"]
    ul, inv = np.unique(logs, return_inverse=True)
    rng = np.random.default_rng(20260927)
    cnt = np.bincount(inv, minlength=4).astype(float)
    draws = rng.integers(0, 4, size=(10000, 4))
    e = np.bincount(inv, weights=v, minlength=4)[draws].sum(1) / cnt[draws].sum(1)
    ref = [100 * v.mean(), 100 * np.percentile(e, 2.5), 100 * np.percentile(e, 97.5)]
    assert np.allclose(W.bootstrap_delta({"x": v}, logs)["x"], ref, rtol=0, atol=1e-12)


def test_pins_resolve():
    assert W.check_pins() == {}


# ----------------------------------------------------------------- real scorer, 3 rows (slow, needs dev-box data)
SCORER = os.environ.get("W_A7_SCORER", "1") == "1" and os.path.isdir(W.DB_ROOT) and os.path.isdir(W.NAVSIM)


@pytest.mark.skipif(not SCORER, reason="needs the dev-box nuPlan DBs and navsim")
def test_scorer_a7_changes_a_tracked_score_and_identity_is_bitwise():
    sys.path[:0] = [W.NAVSIM, W.REFE, W.INSTR]
    import navtrain_scenarios as NS
    import navsim_dac as ND
    import navsim_pdm_targets as PT
    import transp_score as TS
    rows = W.load_population(3)
    by = {}
    for r in rows:
        by.setdefault(r["log"], []).append(r["token"])
    dbs = NS.index_dbs(db_root=W.DB_ROOT)
    seen = 0
    for lg, toks in by.items():
        for sc in NS.build_scenarios_for_log(dbs[lg], toks):
            r = next(x for x in rows if x["token"] == sc._initial_lidar_token)
            parts = PT.metric_cache_parts(sc)
            plans = W.make_plans(r["traj"], r["tc20s_banked"], r["tc20s_banked"], repair_last_heading)
            out = W.score_row(TS, ND, parts, plans)
            assert out["IDN_R_pre"] == out["IDN_R_A7"]                          # control (iii), real scorer
            assert out == W.score_row(TS, ND, PT.metric_cache_parts(sc), plans)  # control (iv), real scorer
            seen += 1
    assert seen == 3
