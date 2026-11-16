"""LON-PLAN arm S (SHRINK) and the SCORE-side DIR-ACC table: controls with deliberate-regression arms (AMEND_HWMPLAN_3 sections 4, 5, 8.2). CPU only.
C-SHRINK: S at alpha 0 equals HOLD on knots 1-10 and its 0-2 s contrast against `hold` through contrast_l1full_strat.py reads exactly 0.0 [0, 0]; S at alpha 1 equals A0 bit for bit; knots 11-30 equal A0's.
alpha: recomputed from the selection records, equals the frozen values; altering D moves it. TRAIN-match: S applied to the TRAIN lambda-0 plans has D equal to D(lambda*) of the record.
Run: python -m pytest -q -p no:cacheprovider --basetemp=<existing parent>/bt test_lonplan_shrink.py"""
import copy, json, os, subprocess, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lonplan_common as C
import numpy as np
import pytest
import hwm_plan as H
import lonplan_shrink as SH
import lonplan_diracc as DA
import lonplan_select as S

ROOT = "F:/TanitAD-scratch/actrsp/lonplan"
SEL = {k: f"{ROOT}/sel_s{k}/LAMBDA_SELECTION_R35_s{k}.json" for k in (0, 1)}
REAL = "F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win"
HAVE_SEL = all(os.path.exists(p) for p in SEL.values())
HAVE_REAL = os.path.exists(REAL + "/windows.npz") and os.path.exists(REAL + "/plans_hwm_oracle_s0.npz") and os.path.exists(HERE + "/reader_src")
FROZEN = {0: (0.07843263912195275, 0.09896859284406319), 1: (0.12367962338894242, 0.12045464645362154)}      # literals of the task, typed here independently of the module


def synth(n=40, n_ep=8, seed=3):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, 31, 2)); eid = np.array([f"e{i % n_ep}" for i in range(n)], dtype="U36")
    plan = A[:, :30] + 0.5 * r.normal(size=(n, 30, 2))
    return A, eid, plan


def knots_11_30_equal(s, a0):
    return bool(np.array_equal(s[:, 10:], a0[:, 10:]))


# ------------------------------------------------------------------------------------------------ alpha from the records
@pytest.mark.skipif(not HAVE_SEL, reason="selection records absent")
def test_alpha_recomputed_from_the_records_equals_the_frozen_values():
    for k in (0, 1):
        a_acc, a_yaw, info = SH.alpha_from_selection(SEL[k])
        assert abs(a_acc - FROZEN[k][0]) < 1e-12 and abs(a_yaw - FROZEN[k][1]) < 1e-12, (k, a_acc, a_yaw)
        assert info["lam_star"] == (300.0, 100.0)[k]
        SH.assert_expected(f"R35_s{k}", a_acc, a_yaw)


@pytest.mark.skipif(not HAVE_SEL, reason="selection records absent")
def test_alpha_regression_altered_D_gives_a_different_alpha_and_trips_the_assert(tmp_path):
    j = json.load(open(SEL[0], encoding="utf-8"))
    j2 = copy.deepcopy(j); j2["runs"][str(float(j["lam_star"]))]["D"] *= 1.5
    p = str(tmp_path / "sel_altered.json"); json.dump(j2, open(p, "w"))
    a_acc, a_yaw, _ = SH.alpha_from_selection(p)
    assert abs(a_acc - FROZEN[0][0]) > 0.01 and abs(a_yaw - FROZEN[0][1]) < 1e-12        # acc moved by 50 percent, yaw (D_yaw untouched) did not
    with pytest.raises(AssertionError):
        SH.assert_expected("R35_s0", a_acc, a_yaw)
    j3 = copy.deepcopy(j); j3["runs"][str(float(j["lam_star"]))]["D_yaw"] *= 0.5
    json.dump(j3, open(p, "w"))
    assert abs(SH.alpha_from_selection(p)[1] - FROZEN[0][1]) > 0.01


def test_alpha_refuses_a_record_without_a_selected_lambda(tmp_path):
    p = str(tmp_path / "sel_collapse.json")
    json.dump(dict(lam_star=None, label="LAMBDA-COLLAPSE", runs={}), open(p, "w"))
    with pytest.raises(ValueError):
        SH.alpha_from_selection(p)


# ------------------------------------------------------------------------------------------------ C-SHRINK on a synthetic stack
def test_c_shrink_synth_alpha_0_is_hold_alpha_1_is_a0_and_tail_is_a0():
    A, _, plan = synth()
    hold = H.hold_plan(A[:, 0], 30)
    s0 = SH.shrink_plan(plan, A, 0.0, 0.0)
    assert np.array_equal(s0[:, :10], hold[:, :10]) and knots_11_30_equal(s0, plan)
    s1 = SH.shrink_plan(plan, A, 1.0, 1.0)
    assert np.array_equal(s1, plan) and s1.tobytes() == plan.tobytes()
    sm = SH.shrink_plan(plan, A, 0.3, 0.6)
    assert knots_11_30_equal(sm, plan)
    np.testing.assert_allclose(sm[:, :10, 0], hold[:, :10, 0] + 0.3 * (plan[:, :10, 0] - hold[:, :10, 0]), atol=1e-15)
    np.testing.assert_allclose(sm[:, :10, 1], hold[:, :10, 1] + 0.6 * (plan[:, :10, 1] - hold[:, :10, 1]), atol=1e-15)
    assert not np.array_equal(sm[:, :10], plan[:, :10]) and not np.array_equal(sm[:, :10], hold[:, :10])


def test_c_shrink_synth_regression_arms_go_red():
    A, _, plan = synth()
    hold = H.hold_plan(A[:, 0], 30)
    assert not np.array_equal(SH.shrink_plan(plan, A, 0.5, 0.5)[:, :10], hold[:, :10])                       # alpha != 0 is not HOLD
    assert not np.array_equal(SH.shrink_plan(plan, A, 0.999999, 0.999999), plan)                              # alpha != 1 is not A0
    assert not knots_11_30_equal(SH.shrink_plan(plan, A, 0.3, 0.3, n_shrink=11), plan)                        # a shrink that reaches knot 11 breaks the tail check
    wrong_hold = lambda a, n: H.hold_plan(np.clip(np.asarray(a)[:, 0], -0.1, 0.1), n)                          # a prior-box-clipped HOLD is a different reference
    orig = SH.hold_of
    try:
        SH.hold_of = wrong_hold
        assert not np.array_equal(SH.shrink_plan(plan, A, 0.0, 0.0)[:, :10], hold[:, :10])
    finally:
        SH.hold_of = orig


def test_c_shrink_hold_is_the_reference_distance_D_uses():
    A, _, plan = synth()
    std = np.array([0.86, 0.095])
    mine = SH.distance_D(plan, A, std)
    hold = H.hold_plan(A[:, 0], 30)
    import plantest2_lib as Q
    ref = Q.hold_distance(plan[:, :10], hold[:, :10], std)
    assert mine == (float(ref[0]), float(ref[1]))
    assert SH.distance_D(SH.shrink_plan(plan, A, 0.0, 0.0), A, std) == (0.0, 0.0)


# ------------------------------------------------------------------------------------------------ real-format input
def _real():
    with np.load(REAL + "/windows.npz") as w:
        A = w["A"].astype(np.float64)
    with np.load(REAL + "/plans_hwm_oracle_s0.npz") as z:
        plan = z["plan_raw"]
    return A, plan


@pytest.mark.skipif(not HAVE_REAL, reason="C7 R35_s0 windows / plans absent")
def test_c_shrink_real_alpha_0_alpha_1_tail():
    A, plan = _real()
    hold = H.hold_plan(A[:, 0], 30)
    assert np.array_equal(SH.shrink_plan(plan, A, 0.0, 0.0)[:, :10], hold[:, :10])
    assert SH.shrink_plan(plan, A, 1.0, 1.0).tobytes() == plan.tobytes()
    s = SH.shrink_plan(plan, A, FROZEN[0][0], FROZEN[0][1])
    assert knots_11_30_equal(s, plan) and not np.array_equal(s[:, :10], plan[:, :10])
    assert not knots_11_30_equal(SH.shrink_plan(plan, A, FROZEN[0][0], FROZEN[0][1], n_shrink=11), plan)


def _contrast_0_2(tmp, a_plan, name):
    pa = str(tmp / f"{name}.npz"); np.savez(pa, plan_raw=a_plan)
    out = str(tmp / f"{name}.json")
    r = subprocess.run([sys.executable, os.path.join(HERE, "contrast_l1full_strat.py"), "--windows", REAL, "--a", pa, "--b", "hold", "--out", out, "--n-boot", "200"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [q for q in json.load(open(out))["rows"] if q["band"] == "0-2s" and q["a"] == "A" and q["b"] == "B"]       # A = S, B = hold
    fin = [q for q in rows if q["delta"] == q["delta"]]
    return rows, fin


def _all_zero(fin):
    return all(q["delta"] == 0.0 and q["lo"] == 0.0 and q["hi"] == 0.0 for q in fin)


@pytest.mark.skipif(not HAVE_REAL, reason="C7 R35_s0 windows / plans / reader absent")
def test_c_shrink_real_alpha_0_contrast_against_hold_reads_exactly_zero_and_alpha_half_does_not(tmp_path):
    A, plan = _real()
    rows, fin = _contrast_0_2(tmp_path, SH.shrink_plan(plan, A, 0.0, 0.0), "s_a0")
    print("C-SHRINK 0-2 s rows", len(rows), "finite", len(fin))
    assert len(fin) >= 10 and {"acc_mae", "yaw_mae"} <= {q["metric"] for q in fin} and _all_zero(fin), [q for q in fin if q["delta"] != 0.0 or q["lo"] != 0.0 or q["hi"] != 0.0][:3]
    rows2, fin2 = _contrast_0_2(tmp_path, SH.shrink_plan(plan, A, 0.5, 0.5), "s_a05")                # deliberate regression: a different alpha must read non-zero
    assert not _all_zero(fin2)


# ------------------------------------------------------------------------------------------------ TRAIN-match
def _train_match(k, al_acc, al_yaw):
    with np.load(f"{ROOT}/sel_s{k}/lam_0/windows.npz") as w:
        A = w["A"].astype(np.float64)
    with np.load(f"{ROOT}/sel_s{k}/lam_0/plans_hwm_oracle_s0.npz") as z:
        plan = z["plan_raw"].astype(np.float64)
    j = json.load(open(SEL[k], encoding="utf-8"))
    s = SH.shrink_plan(plan, A, al_acc, al_yaw)
    d_s = SH.distance_D(s, A, j["prior_std_raw"])
    r = j["runs"][str(float(j["lam_star"]))]
    d0 = SH.distance_D(plan, A, j["prior_std_raw"])
    return d_s, (r["D"], r["D_yaw"]), d0, (j["runs"]["0.0"]["D"], j["runs"]["0.0"]["D_yaw"])


@pytest.mark.skipif(not HAVE_SEL, reason="selection roots absent")
@pytest.mark.parametrize("k", [0, 1])
def test_train_match_S_distance_equals_D_at_lambda_star(k):
    a_acc, a_yaw, _ = SH.alpha_from_selection(SEL[k])
    d_s, d_star, d0, d0_json = _train_match(k, a_acc, a_yaw)
    diff = (abs(d_s[0] - d_star[0]), abs(d_s[1] - d_star[1]))
    print(f"TRAIN-MATCH s{k}: D(S) {d_s} vs D(lam*) {d_star}, abs diff {diff}; D(lam0) recomputed {d0} vs record {d0_json}")
    assert diff[0] < 1e-9 and diff[1] < 1e-9
    assert abs(d0[0] - d0_json[0]) < 1e-9 and abs(d0[1] - d0_json[1]) < 1e-9                          # the plans on disk are the ones the record measured
    d_bad, d_star2, _, _ = _train_match(k, a_yaw, a_acc)                                              # regression arm: swapped channels must break the match
    assert abs(d_bad[0] - d_star2[0]) > 1e-4


# ------------------------------------------------------------------------------------------------ the CLI path
@pytest.mark.skipif(not (HAVE_REAL and HAVE_SEL), reason="real inputs absent")
def test_build_writes_same_format_plus_stamp_and_refuses_overwrite(tmp_path):
    out = str(tmp_path / "S" / "plans_hwm_oracle_s0.npz")
    st = SH.build(REAL + "/plans_hwm_oracle_s0.npz", REAL + "/windows.npz", SEL[0], out)
    with np.load(REAL + "/plans_hwm_oracle_s0.npz") as a0, np.load(out) as s:
        assert set(s.files) == set(a0.files) | {"shrink_alpha"}
        for k in a0.files:
            if k != "plan_raw":
                assert np.array_equal(a0[k], s[k]), k
        assert s["plan_raw"].shape == a0["plan_raw"].shape and knots_11_30_equal(s["plan_raw"], a0["plan_raw"])
    stamp = json.load(open(str(tmp_path / "S" / "plans_hwm_oracle_s0_STAMP.json")))
    assert stamp["alpha_acc"] == st["alpha_acc"] and stamp["expected_alpha_asserted"] and len(stamp["sha256"]["out"]) == 64
    assert abs(stamp["D_score_in_train_prior_std"]["S"][0] - stamp["alpha_acc"] * stamp["D_score_in_train_prior_std"]["A0"][0]) < 1e-9
    with pytest.raises(FileExistsError):
        SH.build(REAL + "/plans_hwm_oracle_s0.npz", REAL + "/windows.npz", SEL[0], out)


# ------------------------------------------------------------------------------------------------ DIR-ACC table
def _write(tmp, A, eid, plans):
    wp = str(tmp / "windows.npz"); np.savez(wp, A=A, eid=eid)
    arms = []
    for name, p in plans.items():
        pp = str(tmp / f"{name}.npz"); np.savez(pp, plan_raw=p); arms.append((name, pp))
    return wp, arms


def test_diracc_table_plan_equals_logged_reads_one_hold_reads_na_shuffled_moves(tmp_path):
    A, eid, plan = synth(n=60)
    logged = A[:, :30].copy()
    hold = H.hold_plan(A[:, 0], 30); shuf = hold + (logged - hold)[np.random.default_rng(1).permutation(len(A))]      # the departure from HOLD of another window
    hold = H.hold_plan(A[:, 0], 30)
    wp, arms = _write(tmp_path, A, eid, dict(LOGGED=logged, HOLD=hold, NOISY=plan, SHUF=shuf))
    tab = DA.diracc_table(wp, arms, n_boot=200)
    r = tab["arms"]
    assert abs(r["LOGGED"]["r"] - 1.0) < 1e-12 and r["LOGGED"]["lo"] > 0.999
    assert r["HOLD"]["r"] != r["HOLD"]["r"] and "n/a" in r["HOLD"]["reason"]
    assert 0.3 < r["NOISY"]["r"] < 0.999 and r["NOISY"]["lo"] <= r["NOISY"]["r"] <= r["NOISY"]["hi"]
    assert abs(r["SHUF"]["r"]) < 0.5 and abs(r["SHUF"]["r"] - r["LOGGED"]["r"]) > 0.5                  # regression arm: breaking the pairing moves r
    assert tab["B"] == 200 and tab["seed"] == 0 and tab["estimator"] == "episode_cluster_bootstrap" and r["LOGGED"]["n_episodes"] == 8
    md = DA.markdown(tab)
    assert "LOGGED" in md and "n/a" in md


def test_diracc_table_uses_the_selection_implementation_and_is_deterministic(tmp_path):
    A, eid, plan = synth(n=60)
    wp, arms = _write(tmp_path, A, eid, dict(NOISY=plan))
    t1 = DA.diracc_table(wp, arms, n_boot=100)["arms"]["NOISY"]
    t2 = DA.diracc_table(wp, arms, n_boot=100)["arms"]["NOISY"]
    ref = S.dir_acc(plan.astype(np.float64), A, eid, 100, 0)
    assert (t1["r"], t1["lo"], t1["hi"]) == (t2["r"], t2["lo"], t2["hi"]) == (ref["r"], ref["lo"], ref["hi"])
    t3 = DA.diracc_table(wp, arms, n_boot=100, seed=1)["arms"]["NOISY"]
    assert t3["r"] == t1["r"] and (t3["lo"], t3["hi"]) != (t1["lo"], t1["hi"])                         # the seed moves the interval only


@pytest.mark.skipif(not HAVE_REAL, reason="C7 R35_s0 windows / plans absent")
def test_diracc_real_format_runs_on_a0_and_s(tmp_path):
    A, plan = _real()
    sp = str(tmp_path / "S.npz"); np.savez(sp, plan_raw=SH.shrink_plan(plan, A, *FROZEN[0]))
    tab = DA.diracc_table(REAL + "/windows.npz", [("A0", REAL + "/plans_hwm_oracle_s0.npz"), ("S", sp)], n_boot=100)
    a0, s = tab["arms"]["A0"], tab["arms"]["S"]
    print("DIR-ACC real", a0["r"], s["r"])
    assert a0["n_windows"] == 994 and a0["n_episodes"] == 71
    assert abs(a0["r"] - s["r"]) < 1e-9          # a positive per-channel scaling of x leaves Pearson r unchanged: S carries the direction of A0 by construction
