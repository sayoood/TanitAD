"""test_tab_tables.py -- pytest of make_tab_tables.py (card ADD84, pod table tool). Synthetic data only; CPU only.
  PYCAP_MB=3000 bash ops/pycap.sh -m pytest -q -p no:cacheprovider tabclose/test_tab_tables.py --basetemp tabclose/tmp_pt/<new name>
The module under test is taken from TABTT_TOOL (the mutation runner points it at a mutated copy) and defaults to make_tab_tables.py beside this file."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.environ.get("TABTT_TOOL", os.path.join(HERE, "make_tab_tables.py"))
os.environ.setdefault("SURROGATE_TOOL_DIR", os.path.join(HERE, "..", "shufclose"))
_spec = importlib.util.spec_from_file_location("make_tab_tables_under_test", TOOL)
M = importlib.util.module_from_spec(_spec)
sys.modules["make_tab_tables_under_test"] = M
_spec.loader.exec_module(M)
RUN = "I10RQ50cKF5P30Hb20"
SEEDS = (0, 1)


def _args(root, out):
    return dict(dump_dir=os.path.join(root, "dump"), tk_dir=os.path.join(root, "tk"), fit_dir="", rrc_run_dir=os.path.join(root, "runs"), rrc_run_name=RUN, out_dir=out, seeds=list(SEEDS),
                surrogate_tool_dir=os.path.join(HERE, "..", "shufclose"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("world"))
    w = M.synth(root)
    out = os.path.join(root, "out_clean")
    rec, failed = M.run(**_args(root, out))
    back = {(a, k): M.load_tab(os.path.join(out, "tab_%s_s%d.npz" % (a, k))) for a in M.TABLE_ARMS for k in SEEDS}
    return {"root": root, "w": w, "out": out, "rec": rec, "failed": failed, "back": back}


def _run_planted(world, tmp_path, tamper):
    out = str(tmp_path / "out_p")
    return M.run(tamper=tamper, **_args(world["root"], out))


# ------------------------------------------------------------------------------------------------ the clean case and the record
def test_clean_case_passes_every_control(world):
    r = world["rec"]
    assert world["failed"] == [] and r["pass"] is True
    for c in ("a0", "a1", "a4_u1", "a4_u4"):
        assert r[c]["PASS"] is True


def test_record_contents(world):
    r, out = world["rec"], world["out"]
    assert len(r["tables"]) == 16
    for nm, h in r["tables"].items():
        assert hashlib.sha256(open(os.path.join(out, nm), "rb").read()).hexdigest() == h
    on_disk = json.load(open(os.path.join(out, "TAB_TABLES_RECORD.json"), encoding="utf-8"))
    assert on_disk["pass"] is True and on_disk["tables"] == r["tables"]
    assert r["rng"] == {"%s_s%d" % (a, k): 84000 + 10 * i + k for a, i in (("W00", 0), ("W50", 1), ("W70", 2), ("W90", 3), ("S70", 4)) for k in SEEDS}
    assert all(set(v) >= {"m", "s", "rows", "valued"} for v in r["m_s"].values())
    assert "dump_s0" in r["inputs"] and "tk_s1" in r["inputs"] and "rrc_mid_log_s1" in r["inputs"]
    for nm, c in r["counts"].items():
        assert c["rows"] == len(world["w"]["frame"])


def test_table_file_format(world):
    for (a, k), (ci, fr, est) in world["back"].items():
        z = np.load(os.path.join(world["out"], "tab_%s_s%d.npz" % (a, k)))
        assert sorted(z.files) == ["clip_index", "est", "frame"] and z["clip_index"].dtype == np.int64 and z["frame"].dtype == np.int64 and z["est"].dtype == np.float64
        assert np.array_equal(ci, world["w"]["ci"]) and np.array_equal(fr, world["w"]["frame"])


# ------------------------------------------------------------------------------------------------ L100, TK, STK
def test_l100_is_the_label_slope_where_the_label_has_one(world):
    w = world["w"]
    want = np.where(w["lab_present"] & (w["lab_run"] >= 3), w["lab_closing"], np.nan)
    for k in SEEDS:
        assert np.array_equal(world["back"][("L100", k)][2], want, equal_nan=True)
    assert int((want == 0.0).sum()) > 0 and int(np.isfinite(want).sum()) < len(want)
    # a present frame with a run of 1 or 2 frames carries no value; a frame without a lead carries none
    assert not np.isfinite(want[w["lab_present"] & (w["lab_run"] < 3)]).any() and not np.isfinite(want[~w["lab_present"]]).any()


def test_tk_equals_the_input_table(world):
    for k in SEEDS:
        z = np.load(os.path.join(world["root"], "tk", "tokclose_T25_s%d_table.npz" % k))
        got = world["back"][("TK", k)]
        assert np.array_equal(got[0], z["clip_index"]) and np.array_equal(got[1], z["frame"]) and np.array_equal(got[2], z["est"], equal_nan=True)


def test_stk_is_the_map_of_tk(world):
    SU = M.load_surrogate_tool(os.path.join(HERE, "..", "shufclose"))
    for k in SEEDS:
        d = np.load(os.path.join(world["root"], "dump", "reader_s%d_score.npz" % k))
        ci, p = d["clip_index"], d["p"]
        ids = np.unique(ci)
        share = {int(c): (p[ci == c] >= 0.5).mean() for c in ids}
        order = sorted(ids.tolist(), key=lambda c: (share[c], c))
        img = {}
        n = len(order)
        for a in range(0, n - 3 if n % 2 else n, 2):
            img[order[a]], img[order[a + 1]] = order[a + 1], order[a]
        if n % 2:
            a, b, c = order[-3:]
            img[a], img[b], img[c] = b, c, a
        tk = world["back"][("TK", k)]
        look = {(int(a), int(b)): float(v) for a, b, v in zip(*tk)}
        want = np.array([look.get((img[int(c)], int(f)), np.nan) for c, f in zip(tk[0], tk[1])])
        assert np.array_equal(world["back"][("STK", k)][2], want, equal_nan=True)
        mp = np.load(os.path.join(world["out"], "map_STK_s%d.npz" % k))
        assert dict(zip(mp["map_clips"].tolist(), mp["map"].tolist())) == img
        assert not any(a == b for a, b in img.items())


# ------------------------------------------------------------------------------------------------ the W arms: formula, rng, planting
def _manual_arm(l100, ci, fr, a, k, r, smooth=False, w00=False):
    rng = np.random.default_rng(84000 + 10 * a + k)
    z = rng.standard_normal(len(l100))
    if smooth:
        e = np.empty_like(z)
        for i in range(len(z)):
            e[i] = z[i] if (i == 0 or ci[i] != ci[i - 1]) else 0.9 * e[i - 1] + np.sqrt(1 - 0.81) * z[i]
        z = e
    V = np.isfinite(l100)
    L, e = l100[V], z[V]
    m, s = L.mean(), L.std()
    if w00:
        e = (e - e.mean()) / (e - e.mean()).std()
        out = np.full(len(l100), np.nan)
        out[V] = m + 0.5 * s * e
        return out
    X = np.stack([np.ones(len(L)), L], 1)
    e = e - X @ np.linalg.solve(X.T @ X, X.T @ e)
    e = e / e.std()
    out = np.full(len(l100), np.nan)
    out[V] = m + r * r * (L - m) + s * np.sqrt(r * r * (1 - r * r)) * e
    return out


@pytest.mark.parametrize("arm,a,r,kw", [("W50", 1, 0.5, {}), ("W70", 2, 0.7, {}), ("W90", 3, 0.9, {}), ("S70", 4, 0.7, {"smooth": True}), ("W00", 0, 0.0, {"w00": True})])
def test_arm_formula_rng_and_planting(world, arm, a, r, kw):
    for k in SEEDS:
        ci, fr, l100 = world["back"][("L100", k)]
        want = _manual_arm(l100, ci, fr, a, k, r, **kw)
        assert np.allclose(world["back"][(arm, k)][2], want, rtol=0, atol=1e-9, equal_nan=True)


def test_exact_planting_of_r_sd_and_mean(world):
    for k in SEEDS:
        ci, fr, l100 = world["back"][("L100", k)]
        V = np.isfinite(l100)
        L = l100[V]
        for arm, r in (("W50", 0.5), ("W70", 0.7), ("W90", 0.9), ("S70", 0.7)):
            x = world["back"][(arm, k)][2][V]
            assert abs(np.corrcoef(x, L)[0, 1] - r) < 1e-9 and abs(x.std() - r * L.std()) < 1e-9 and abs(x.mean() - L.mean()) < 1e-9
        x = world["back"][("W00", k)][2][V]
        assert abs(x.std() - 0.5 * L.std()) < 1e-9 and abs(x.mean() - L.mean()) < 1e-9


def test_different_arms_and_seeds_are_different_draws(world):
    seen = set()
    for k in SEEDS:
        for arm in ("W00", "W50", "W70", "W90", "S70"):
            seen.add(world["back"][(arm, k)][2][np.isfinite(world["back"][("L100", k)][2])][:20].tobytes())
    assert len(seen) == 10


def test_w_arms_have_values_on_the_frames_of_l100_only(world):
    for k in SEEDS:
        V = np.isfinite(world["back"][("L100", k)][2])
        for arm in ("W00", "W50", "W70", "W90", "S70"):
            assert np.array_equal(np.isfinite(world["back"][(arm, k)][2]), V)


def test_autoregression_formula_and_restart():
    ci = np.array([0, 0, 0, 1, 1, 2])
    fr = np.array([0, 1, 2, 0, 1, 0])
    z = np.array([1.0, 2.0, -1.0, 0.5, 0.25, 3.0])
    e = M.ar_along_clips(z, ci, fr)
    w = np.sqrt(1 - 0.81)
    want = [1.0, 0.9 * 1.0 + w * 2.0, 0.9 * (0.9 + w * 2.0) + w * -1.0, 0.5, 0.9 * 0.5 + w * 0.25, 3.0]
    assert np.allclose(e, want, atol=1e-12)
    # rows given out of order are processed in frame order and returned in row order
    perm = np.array([2, 0, 1, 4, 3, 5])
    e2 = M.ar_along_clips(z[perm], ci[perm], fr[perm])
    assert np.allclose(e2, e[perm], atol=1e-12)


def test_plant_noise_exactness_and_refusals():
    rg = np.random.default_rng(1)
    L = rg.normal(size=500)
    e = M.plant_noise(rg.normal(size=500) + 0.7 * L + 2.0, L, uncorrelated=True)
    assert abs(e.mean()) < 1e-12 and abs(e.std() - 1) < 1e-12 and abs(np.corrcoef(e, L)[0, 1]) < 1e-12
    e0 = M.plant_noise(rg.normal(size=500) + 0.7 * L, L, uncorrelated=False)
    assert abs(e0.mean()) < 1e-12 and abs(e0.std() - 1) < 1e-12 and abs(np.corrcoef(e0, L)[0, 1]) > 0.1
    with pytest.raises(SystemExit):
        M.plant_noise(np.ones(10), np.ones(10), True)
    with pytest.raises(SystemExit):
        M.plant_noise(np.array([1.0, 2.0]), np.array([1.0, 2.0]), True)


# ------------------------------------------------------------------------------------------------ a1 on crafted tables
def _craft(world, k, arm, rho, sd_mult=1.0, shift=0.0, ar=0.0, r_t=None):
    """an arm table with chosen r against L (rho), sd (sd_mult x target), mean shift and noise autocorrelation (AR coefficient ar); other arms/L100 from the clean run."""
    ci, fr, l100 = world["back"][("L100", k)]
    V = np.isfinite(l100)
    L = l100[V]
    m, s = L.mean(), L.std()
    rng = np.random.default_rng(3)
    z = rng.standard_normal(len(l100))
    if ar:
        e = np.empty_like(z)
        for i in range(len(z)):
            e[i] = z[i] if (i == 0 or ci[i] != ci[i - 1]) else ar * e[i - 1] + np.sqrt(1 - ar * ar) * z[i]
        z = e
    n = M.plant_noise(z[V], L, True)
    r_t = r_t if r_t is not None else M.R_TARGET.get(arm, 0.0)
    sd_t = (0.5 if arm == "W00" else r_t) * s
    out = np.full(len(l100), np.nan)
    out[V] = m + shift + sd_t * sd_mult * (rho * (L - m) / s + np.sqrt(1 - rho * rho) * n)
    return out


def _a1_with(world, arm, est, k=0):
    tabs = dict(world["back"])
    ci, fr, _ = tabs[(arm, k)]
    tabs[(arm, k)] = (ci, fr, est)
    return M.check_a1(tabs, [k])["runs"]["%s_s%d" % (arm, k)]


def test_a1_clean_tables_pass(world):
    assert M.check_a1(world["back"], list(SEEDS))["PASS"] is True


@pytest.mark.parametrize("rho,ok", [(0.70, True), (0.685, True), (0.715, True), (0.675, False), (0.725, False)])
def test_a1_r_tolerance(world, rho, ok):
    est = _craft(world, 0, "W70", rho, ar=0.0)
    assert _a1_with(world, "W70", est)["ok"] is ok


@pytest.mark.parametrize("mult,ok", [(1.0, True), (0.96, True), (1.04, True), (0.93, False), (1.07, False)])
def test_a1_sd_band(world, mult, ok):
    est = _craft(world, 0, "W50", 0.5, sd_mult=mult)
    assert _a1_with(world, "W50", est)["ok"] is ok


@pytest.mark.parametrize("shift,ok", [(0.0, True), (0.045, True), (-0.045, True), (0.055, False), (-0.055, False)])
def test_a1_mean_tolerance(world, shift, ok):
    est = _craft(world, 0, "W90", 0.9, shift=shift)
    assert _a1_with(world, "W90", est)["ok"] is ok


@pytest.mark.parametrize("arm,ar,ok", [("S70", 0.9, True), ("S70", 0.8, False), ("S70", 0.99, False), ("S70", 0.0, False), ("W70", 0.0, True), ("W70", 0.2, False), ("W70", -0.2, False), ("W70", 0.9, False)])
def test_a1_autocorrelation_bands(world, arm, ar, ok):
    est = _craft(world, 0, arm, 0.7, ar=ar)
    assert _a1_with(world, arm, est)["ok"] is ok


@pytest.mark.parametrize("rho,ok", [(0.0, True), (0.02, True), (-0.02, True), (0.05, False), (-0.05, False)])
def test_a1_w00_r_band_is_plus_minus_003(world, rho, ok):
    est = _craft(world, 0, "W00", rho, r_t=0.0)
    assert _a1_with(world, "W00", est)["ok"] is ok


def test_a1_w00_target_sd_is_half_s(world):
    assert _a1_with(world, "W00", _craft(world, 0, "W00", 0.0, sd_mult=1.0))["ok"] is True
    assert _a1_with(world, "W00", _craft(world, 0, "W00", 0.0, sd_mult=1.2))["ok"] is False
    assert _a1_with(world, "W00", _craft(world, 0, "W00", 0.0, sd_mult=0.8))["ok"] is False


def test_a1_valued_frame_sets_must_equal_those_of_l100(world):
    est = world["back"][("W50", 0)][2].copy()
    est[int(np.nonzero(np.isfinite(est))[0][3])] = np.nan
    r = _a1_with(world, "W50", est)
    assert r["ok"] is False and r["valued_frames_equal_L100"] is False
    est = world["back"][("W50", 0)][2].copy()
    est[int(np.nonzero(~np.isfinite(est))[0][3])] = 0.1
    assert _a1_with(world, "W50", est)["ok"] is False


def test_a1_rows_must_be_the_rows_of_l100(world):
    tabs = dict(world["back"])
    ci, fr, v = tabs[("W70", 0)]
    tabs[("W70", 0)] = (ci, fr + 1, v)
    assert M.check_a1(tabs, [0])["PASS"] is False


# ------------------------------------------------------------------------------------------------ a0 on crafted logs
def _logs(world, **ch):
    z = {}
    for k in SEEDS:
        for tag, fn in (("rule", "rule_log_%s_s%d.npz"), ("mid", "rule_log_mid_%s_s%d.npz")):
            p = os.path.join(world["root"], "runs", "%s_s%d" % (RUN, k), "ticks", fn % (RUN, k))
            d = M.load_log(p)
            z[(k, tag)] = {a: np.array(b) for a, b in d.items()}
    return z


def _a0(world, logs):
    return M.check_a0({k: world["back"][("L100", k)] for k in SEEDS}, logs)


def test_a0_clean_logs_agree_on_all_looks_and_present_looks(world):
    r = _a0(world, _logs(world))
    assert r["PASS"] is True
    for d in r["runs"].values():
        assert d["share_agree_all_looks"] == 1.0 and d["share_agree_present_looks"] == 1.0 and d["log_nonzero_table_differs"] == 0 and d["log_zero_table_nonzero"] == 0
        assert d["looks"] > 100 and d["looks_present"] > 0 and d["looks_log_nonzero"] > 0 and d["looks_present"] < d["looks"]


def test_a0_shifted_label_fails_and_is_counted(world):
    lg = _logs(world)
    c = lg[(0, "rule")]["c_src"]
    c[c != 0] += 1e-3
    r = _a0(world, lg)
    d = r["runs"]["s0_rule_log"]
    assert r["PASS"] is False and d["ok"] is False and d["log_nonzero_table_differs"] == d["looks_log_nonzero"] and d["share_agree_all_looks"] < 0.999


def test_a0_difference_of_one_millionth_is_the_edge(world):
    lg = _logs(world)
    c = lg[(0, "mid")]["c_src"]
    c[c != 0] += 5e-7
    assert _a0(world, lg)["PASS"] is True
    c[c != 0] += 1e-6
    assert _a0(world, lg)["PASS"] is False


def test_a0_wrong_frame_key_fails(world):
    lg = _logs(world)
    lg[(1, "mid")]["j"] = lg[(1, "mid")]["j"] + 1
    assert _a0(world, lg)["PASS"] is False
    lg = _logs(world)
    lg[(1, "rule")]["ci"] = (lg[(1, "rule")]["ci"] + 1) % 21
    assert _a0(world, lg)["PASS"] is False


def test_a0_log_zero_where_table_has_a_value_fails(world):
    lg = _logs(world)
    c = lg[(0, "rule")]["c_src"]
    c[c != 0] = 0.0
    r = _a0(world, lg)
    assert r["PASS"] is False and r["runs"]["s0_rule_log"]["log_zero_table_nonzero"] > 0


def test_a0_share_threshold_is_999_per_thousand(world):
    lg = _logs(world)
    big = {a: np.tile(b, 10) for a, b in lg[(0, "rule")].items()}                 # 5390 looks: 5 disagreeing looks read 0.99907, 6 read 0.99889
    nz = np.nonzero(big["c_src"] != 0)[0]
    n = len(big["c_src"])
    assert 5390 <= n < 5400 and n * 0.001 < 6
    for bad, ok in ((5, True), (6, False)):
        c = big["c_src"].copy()
        c[nz[:bad]] += 1.0
        lg2 = dict(lg)
        lg2[(0, "rule")] = dict(big, c_src=c)
        r = _a0(world, lg2)["runs"]["s0_rule_log"]
        assert r["ok"] is ok and r["log_nonzero_table_differs"] == bad


def test_a0_present_only_share_is_reported_beside(world):
    lg = _logs(world)
    pr = lg[(0, "rule")]["present"]
    c = lg[(0, "rule")]["c_src"]
    hit = np.nonzero((c != 0) & ~pr)[0]
    c[hit] += 1.0                                     # break only looks without a reader lead
    d = _a0(world, lg)["runs"]["s0_rule_log"]
    assert d["share_agree_present_looks"] == 1.0 and d["share_agree_all_looks"] < 1.0 and d["log_nonzero_table_differs"] == len(hit)


def test_a0_missing_log_nan_log_and_vacuous_cases_fail(world):
    lg = _logs(world)
    lg[(1, "mid")] = None
    assert _a0(world, lg)["PASS"] is False
    lg = _logs(world)
    c = lg[(0, "rule")]["c_src"]
    c[c == 0] = np.nan
    assert _a0(world, lg)["PASS"] is False
    assert M.check_a0({k: world["back"][("L100", k)] for k in SEEDS}, {})["PASS"] is False
    lg = _logs(world)
    lg[(0, "rule")]["c_src"] = np.zeros_like(lg[(0, "rule")]["c_src"])           # a log without any nonzero value reads nothing
    lg[(0, "mid")]["c_src"] = np.zeros_like(lg[(0, "mid")]["c_src"])
    assert _a0(world, lg)["PASS"] is False


def test_find_rrc_logs_layouts(tmp_path):
    key = "%s_s0" % RUN
    d = tmp_path / key / "ticks"
    d.mkdir(parents=True)
    (d / ("rule_log_%s.npz" % key)).write_bytes(b"x")
    (d / ("rule_log_mid_%s.npz" % key)).write_bytes(b"x")
    a, b = M.find_rrc_logs(str(tmp_path), RUN, 0)
    assert a.endswith("rule_log_%s.npz" % key) and b.endswith("rule_log_mid_%s.npz" % key)
    assert M.find_rrc_logs(str(tmp_path), RUN, 1) == (None, None)


# ------------------------------------------------------------------------------------------------ a4 and refusals (full runs)
def test_a4_fixed_point_fails_u1(world, tmp_path):
    def t(kind, d):
        if kind[0] == "map":
            d["map"] = d["map"].copy()
            d["map"][0] = d["map_clips"][0]
    rec, failed = _run_planted(world, tmp_path, t)
    assert "a4_u1" in failed and rec["pass"] is False and "a0" not in failed and "a1" not in failed


def test_a4_wrong_value_fails_u1(world, tmp_path):
    def t(kind, tab):
        if kind[0] == "tab" and kind[1] == "STK" and kind[2] == 1:
            e = tab["est"].copy()
            e[int(np.nonzero(np.isfinite(e))[0][40])] += 0.001
            tab["est"] = e
    rec, failed = _run_planted(world, tmp_path, t)
    assert "a4_u1" in failed and rec["a4_u1"]["seeds"]["s1"]["values_not_equal_to_the_mapped_estimate"] == 1 and rec["a4_u1"]["seeds"]["s0"]["passed"] is True


def test_a4_surrogate_equal_to_estimate_fails_u4(world, tmp_path):
    def t(kind, tab):
        if kind[0] == "tab" and kind[1] == "STK":
            tab["est"] = world["back"][("TK", kind[2])][2].copy()
    rec, failed = _run_planted(world, tmp_path, t)
    assert "a4_u4" in failed and rec["a4_u4"]["r_surrogate"] > 0.15 and rec["a4_u4"]["H"] == 25


def test_a4_u4_uses_h_25_rows(world):
    rec = world["rec"]
    assert rec["a4_u4"]["H"] == 25 and rec["a4_u4"]["rows_scored"] > 1000 and abs(rec["a4_u4"]["r_estimate"]) > 0.3 and abs(rec["a4_u4"]["r_surrogate"]) < 0.15
    SU = M.load_surrogate_tool(os.path.join(HERE, "..", "shufclose"))
    CF = SU.load_fit("")
    want = sum(len(CF.build_rows(CF.load_dump(os.path.join(world["root"], "dump", "reader_s%d_score.npz" % k)), 25)["y"]) for k in SEEDS)
    assert rec["a4_u4"]["rows_scored"] == want
    want10 = sum(len(CF.build_rows(CF.load_dump(os.path.join(world["root"], "dump", "reader_s%d_score.npz" % k)), 10)["y"]) for k in SEEDS)
    assert want10 != want


def test_a0_reading_nothing_fails(world):
    ci, fr, ev = world["back"][("L100", 0)]
    empty = {0: (ci, fr, np.full(len(ev), np.nan))}
    lg = {(0, "rule"): {"ci": ci[:50], "j": fr[:50], "c_src": np.zeros(50), "present": np.ones(50, bool)}}
    assert M.check_a0(empty, lg)["PASS"] is False
    lg0 = {(0, "rule"): {"ci": ci[:0], "j": fr[:0], "c_src": np.zeros(0), "present": np.zeros(0, bool)}}
    assert M.check_a0(empty, lg0)["PASS"] is False


def test_run_min_of_the_fit_module_must_agree(world, tmp_path, monkeypatch):
    SU = M.load_surrogate_tool(os.path.join(HERE, "..", "shufclose"))
    real = SU.load_fit

    def odd(fit_dir=""):
        cf = real(fit_dir)
        cf.RUN_MIN = 4
        return cf
    SU.load_fit = odd
    monkeypatch.setattr(M, "load_surrogate_tool", lambda d="": SU)
    with pytest.raises(SystemExit) as e:
        M.run(**_args(world["root"], str(tmp_path / "o")))
    assert "RUN_MIN" in str(e.value)


def test_tables_are_written_when_a_control_fails(world, tmp_path):
    def t(kind, tab):
        if kind[0] == "tab" and kind[1] == "W70" and kind[2] == 0:
            tab["est"] = tab["est"] + 0.3
    rec, failed = _run_planted(world, tmp_path, t)
    assert failed == ["a1"] and rec["pass"] is False
    assert all(os.path.isfile(str(tmp_path / "out_p" / ("tab_%s_s%d.npz" % (a, k)))) for a in M.TABLE_ARMS for k in SEEDS)
    assert json.load(open(str(tmp_path / "out_p" / "TAB_TABLES_RECORD.json"), encoding="utf-8"))["pass"] is False


def test_existing_out_folder_is_refused(world):
    with pytest.raises(SystemExit) as e:
        M.run(**_args(world["root"], world["out"]))
    assert "REFUSED" in str(e.value)


def test_label_columns_that_differ_between_seeds_are_refused(world, tmp_path):
    d2 = str(tmp_path / "dump_bad")
    os.makedirs(d2)
    for k in SEEDS:
        z = {a: b for a, b in np.load(os.path.join(world["root"], "dump", "reader_s%d_score.npz" % k)).items()}
        if k == 1:
            z["lab_run"] = z["lab_run"].copy()
            z["lab_run"][7] += 1
        np.savez(os.path.join(d2, "reader_s%d_score.npz" % k), **z)
    a = _args(world["root"], str(tmp_path / "o"))
    a["dump_dir"] = d2
    with pytest.raises(SystemExit) as e:
        M.run(**a)
    assert "label column lab_run" in str(e.value)


def test_tk_rows_that_differ_from_the_dump_are_refused(world, tmp_path):
    t2 = str(tmp_path / "tk_bad")
    os.makedirs(t2)
    for k in SEEDS:
        z = {a: b for a, b in np.load(os.path.join(world["root"], "tk", "tokclose_T25_s%d_table.npz" % k)).items()}
        if k == 0:
            z["frame"] = z["frame"] + 1
        np.savez(os.path.join(t2, "tokclose_T25_s%d_table.npz" % k), **z)
    a = _args(world["root"], str(tmp_path / "o"))
    a["tk_dir"] = t2
    with pytest.raises(SystemExit) as e:
        M.run(**a)
    assert "not the rows of the dump" in str(e.value)


def test_missing_rule_log_fails_a0_but_writes_the_tables(world, tmp_path):
    a = _args(world["root"], str(tmp_path / "o"))
    a["rrc_run_dir"] = str(tmp_path / "no_runs")
    os.makedirs(a["rrc_run_dir"])
    rec, failed = M.run(**a)
    assert failed == ["a0"] and os.path.isfile(str(tmp_path / "o" / "tab_L100_s0.npz"))


def test_main_prints_one_line_per_table_the_controls_and_the_last_line(world, tmp_path, capsys, monkeypatch):
    out = str(tmp_path / "out_main")
    monkeypatch.setattr(sys, "argv", ["x", "--dump-dir", os.path.join(world["root"], "dump"), "--tk-dir", os.path.join(world["root"], "tk"), "--fit-dir", "", "--rrc-run-dir",
                                      os.path.join(world["root"], "runs"), "--out-dir", out, "--surrogate-tool-dir", os.path.join(HERE, "..", "shufclose")])
    with pytest.raises(SystemExit) as e:
        M.main()
    assert e.value.code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "ZZ TABTABLES PASS"
    tl = [x for x in lines if x.startswith("TABLE ")]
    assert len(tl) == 16
    for x in tl:
        t = x.split()
        assert t[2] == "rows" and t[4] == "valued" and t[6] == "sha256" and len(t[7]) == 64
    assert sum(1 for x in lines if x.startswith("PASS a0")) == 1 and sum(1 for x in lines if x.startswith("PASS a1 ")) == 10 and sum(1 for x in lines if x.startswith("PASS a4_u1")) == 2 and sum(1 for x in lines if x.startswith("PASS a4_u4")) == 1
    rec = json.load(open(os.path.join(out, "TAB_TABLES_RECORD.json"), encoding="utf-8"))
    for x in tl:
        assert rec["tables"][x.split()[1]] == x.split()[7]
