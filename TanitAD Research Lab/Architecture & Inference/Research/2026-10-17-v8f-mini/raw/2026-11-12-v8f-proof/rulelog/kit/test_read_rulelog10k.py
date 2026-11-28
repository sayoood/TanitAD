"""Tests of read_rulelog10k.py on tiny synthetic runs with rule logs, written here (portable to Linux). Child processes run in the folder named by the environment variable RULELOG_TEST_CWD (default: this
folder). The tool of card ADD135 and its test module are imported by path from the folder of that card (env RULELOG_LU_KIT, else the pod path, else the dev path); nothing here reads a real run.

The synthetic world (all motion along the x axis): window i is one clip of 31 label rows. The logged ego moves at 8 m/s (a_l(k) = 1.6 k); the model of a run moves at 10 m/s on the windows named in FAST of its
run (a_m(k) = 2 k) and at 8 m/s elsewhere (so those windows stay safe). The label gap G(k) gives the lead's position P(k) = a_l(k) + G(k). Four lead kinds, hand-derived (rule every 5 steps, cap = max(3, 1.2 v_m) = 12):
  dip   G = 15.2 until step 5, 14.2, 13.2, then 12.2: the lead moves at 8 m/s, slows to 3 m/s for steps 6 to 8, then moves at 8 again. First unsafe step 8, tick 5, i = 3, g0 = 13.2, S = 3.0,
        e_L(v) = 0.6 v - 1.8, e_X(v) = 4.8 - 0.6 v   (v = the lead speed the rule assumed: at 8 the whole loss is forecast, e_X = 0)
  c13   G = 13 constant (the lead moves at 8 m/s): first unsafe step 8, tick 5, i = 3, g0 = 11, S = 2.2, e_L(v) = 0.6 v - 4.8, e_X(v) = 7 - 0.6 v   (at 8: e_L = 0, the whole loss is execution)
  c11   G = 11: first unsafe step 3, tick 0, i = 3, g0 = 11, S = 2.2, the same e_L, e_X as c13
  c150  G = 15: first unsafe step 13, tick 10, i = 3, g0 = 11, S = 2.2, the same e_L, e_X as c13
  c7    G = 7: the logged driver has tau 0.875 s, so U is not defined there.
For the candidate GEc the rule's gap is the camera gap, label gap + 0.5 m: g0 = g_m(k_t) + 0.5, e_G = 0.5, e_X lower by 0.5.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.environ.get("RULELOG_LU_KIT"), "/workspace/v8proof/labelunsafe/kit", "F:/TanitAD-scratch/v8proof/labelunsafe/kit"):
    if _cand and os.path.isfile(os.path.join(_cand, "read_labelunsafe10k.py")):
        LU_KIT = _cand
        break
else:
    raise RuntimeError("the folder of the tool of card ADD135 is not found")
sys.path.insert(0, LU_KIT)
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import test_read_labelunsafe10k as TL  # noqa: E402
import read_rulelog10k as RL  # noqa: E402

LU = RL.bind_lu(TL.LU.__file__)
CWD = os.environ.get("RULELOG_TEST_CWD") or HERE

LEAD = ["dip", "c13", "dip", "c11", "c150", "dip", "c7", "c13", "dip", "c150", "c11", "dip"]
FAST = {("RLc", 0): {0, 1, 2, 3, 4, 5, 6, 7}, ("RLc", 1): {0, 1, 3, 5, 8, 9}, ("RLc", 2): {2, 4, 6, 11}, ("RLc", 3): {0, 2, 5, 8, 10, 11},
        ("GEc", 0): {1, 3}, ("GEc", 1): {0, 9}, ("GEc", 2): {2, 5, 7}, ("GEc", 3): set()}
LOGOPT = {("RLc", 0): {2: {"vL": 7.0}, 5: {"active": False}, 7: {"sat": 2}}, ("RLc", 1): {3: {"vL": 9.0}, 8: {"present": False}}, ("RLc", 2): {4: {"sat": 1}}, ("RLc", 3): {0: {"vL": 6.5}}}
TYPEF = {"dip": (8, 5, 3, 3.0, lambda v: 0.6 * v - 1.8, lambda v: 4.8 - 0.6 * v), "c13": (8, 5, 3, 2.2, lambda v: 0.6 * v - 4.8, lambda v: 7.0 - 0.6 * v),
         "c11": (3, 0, 3, 2.2, lambda v: 0.6 * v - 4.8, lambda v: 7.0 - 0.6 * v), "c150": (13, 10, 3, 2.2, lambda v: 0.6 * v - 4.8, lambda v: 7.0 - 0.6 * v)}
DELTA = {"RLc": 0.0, "GEc": 0.5}


def gk(kind):
    if kind == "dip":
        return [15.2] * 6 + [14.2, 13.2] + [12.2] * 23
    return [{"c13": 13.0, "c11": 11.0, "c150": 15.0, "c7": 7.0}[kind]] * 31


def tiny_cfg(**kw):
    c = dict(RL.CFG)
    c.update({"expect_files": 6, "expect_unsafe": [7, 6, 3, 6], "n_boot": 2000,
              "expected_rates": {"RLc": {"per_seed": [7 / 11, 6 / 11, 3 / 11, 6 / 11], "pooled": 22 / 44}, "GEc": {"pooled": 7 / 44}}})
    c.update(kw)
    return c


def log_rows(tag_arm, s, fast, opts, Gs, k_set, extra_row=False):
    """rows of the rule log at the steps k_set for every window: by construction from the track equations of the world"""
    cols = {k: [] for k in ("ci", "t0", "j", "k", "tick", "active", "present", "src_bad", "sat", "sat_exec", "g_src", "v0", "v_l", "v_L", "a_l", "a_m", "g0")}
    star = opts.get("*", {})

    def add(i, k, o):
        vm = 10.0 if i in fast else 8.0
        a_l, a_m = 1.6 * k, vm * 0.2 * k
        G = float(np.float32(Gs[i][k]))
        present = o.get("present", True)
        g_src = G + DELTA[tag_arm] if present else float("nan")
        g0 = (g_src + a_l - a_m + o.get("dg0", 0.0)) if present else float("nan")
        vals = dict(ci=i, t0=0, j=k, k=k, tick=k // 5, active=o.get("active", present), present=present, src_bad=False, sat=o.get("sat", 0), sat_exec=o.get("sat", 0), g_src=g_src, v0=vm, v_l=8.0,
                    v_L=o.get("vL", 8.0), a_l=a_l, a_m=a_m, g0=g0)
        for c in cols:
            cols[c].append(vals[c])
    for i in range(len(Gs)):
        o = dict(star, **opts.get(i, {}))
        for k in k_set:
            if k in o.get("skip_k", ()):
                continue
            add(i, k, o)
            if o.get("dup") and k == o["dup"]:
                add(i, k, o)
            if o.get("dup_diff") and k == o["dup_diff"]:
                add(i, k, dict(o, dg0=1.0))
    if extra_row:
        add(0, 0, {})
        cols["ci"][-1] = 999
    dt = {"ci": np.int64, "t0": np.int64, "j": np.int64, "k": np.int64, "tick": np.int64, "active": bool, "present": bool, "src_bad": bool, "sat": np.int64, "sat_exec": np.int64}
    return {c: np.array(v, dtype=dt.get(c, np.float64)) for c, v in cols.items()}


def build_rl(tmp, fast=FAST, logopt=LOGOPT, lead=LEAD, skip_done=(), no_mid=(), extra_dump=False, per_ep=2):
    tmp = str(tmp)
    os.makedirs(tmp, exist_ok=True)
    root = os.path.join(tmp, "t1")
    n = len(lead)
    names = ["clip%04d" % i for i in range(n)]
    Gs = [gk(t) for t in lead]
    np.savez(os.path.join(tmp, "join.npz"), clip=np.array(names), off=np.arange(n, dtype=np.int64) * 31, n=np.full(n, 31, np.int64), lead_valid=np.ones(31 * n, np.int8),
             lead_gap=np.concatenate([np.asarray(g, np.float32) for g in Gs]))
    with open(os.path.join(tmp, "grid.json"), "w", encoding="utf-8") as f:
        json.dump({"order": names, "windows": {c: [0] for c in names}}, f)
    t = np.arange(1, 31) * 0.2
    for arm, tag in RL.TAGS.items():
        for s in (0, 1, 2, 3):
            folder = LU.run_folder(root, tag, s)
            dd = LU.dump_folder(root, tag, s)
            os.makedirs(dd, exist_ok=True)
            os.makedirs(os.path.join(folder, "ticks"), exist_ok=True)
            if (arm, s) not in skip_done:
                with open(os.path.join(folder, "T1_DONE"), "w", encoding="utf-8") as f:
                    f.write("done\n")
            fs = fast[(arm, s)]
            for e in range(n // per_ep):
                ws_ = list(range(e * per_ep, (e + 1) * per_ep))
                g, cl = np.zeros((len(ws_), 30, 2)), np.zeros((len(ws_), 30, 2))
                for r, i in enumerate(ws_):
                    g[r, :, 0] = 8.0 * t
                    cl[r, :, 0] = (10.0 if i in fs else 8.0) * t
                np.savez(os.path.join(dd, "ep%03d.npz" % e), g=g, cl=cl, ws=np.zeros(len(ws_), np.int64), eid=np.array(["epi%05d" % e] * len(ws_)), clip_index=np.array(ws_, np.int64),
                         v0=np.full(len(ws_), 8.0))
            if extra_dump and (arm, s) == ("RLc", 0):
                np.savez(os.path.join(dd, "ep999.npz"), g=np.zeros((1, 30, 2)), cl=np.zeros((1, 30, 2)), ws=np.zeros(1, np.int64), eid=np.array(["epi99999"]), clip_index=np.zeros(1, np.int64), v0=np.zeros(1))
            o = logopt.get((arm, s), {})
            first, mid = log_rows(arm, s, fs, o, Gs, (0, 10, 20), extra_row=True), log_rows(arm, s, fs, o, Gs, (5, 15, 25))
            np.savez(os.path.join(folder, "ticks", "rule_log_%s_s%d.npz" % (tag, s)), rule_every=np.int64(5), **first)
            if (arm, s) not in no_mid:
                np.savez(os.path.join(folder, "ticks", "rule_log_mid_%s_s%d.npz" % (tag, s)), rule_every=np.int64(5), **mid)
    return {"root": root, "join": os.path.join(tmp, "join.npz"), "grid": os.path.join(tmp, "grid.json"), "tag": tmp}


def edit_log(p, arm, s, fn, mid=False):
    path = os.path.join(LU.run_folder(p["root"], RL.TAGS[arm], s), "ticks", ("rule_log_mid_%s_s%d.npz" if mid else "rule_log_%s_s%d.npz") % (RL.TAGS[arm], s))
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    fn(d)
    np.savez(path, **d)


def expected_rows(arm):
    """hand-derived rows of the unsafe windows of an arm over the four model seeds: (seed, window, kind, v assumed, on, sat)"""
    rows = []
    for s in range(4):
        for i in sorted(FAST[(arm, s)]):
            if LEAD[i] == "c7":
                continue
            o = LOGOPT.get((arm, s), {}).get(i, {})
            rows.append((s, i, LEAD[i], o.get("vL", 8.0), bool(o.get("active", o.get("present", True))), o.get("sat", 0)))
    return rows


@pytest.fixture(scope="module")
def E():
    return LU.bind_engine(LU.default_paths())


@pytest.fixture(scope="module")
def world(E, tmp_path_factory):
    p = build_rl(tmp_path_factory.mktemp("rl"))
    cfg = tiny_cfg()
    w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
    tabs, probs = RL.load_tables(E, cfg, w, p["root"])
    return w, p, cfg, tabs, probs


def pytest_text(n=7):
    return "test_read_rulelog10k.py .......\n\n%d passed in 3.21s\n" % n


# ---------------------------------------------------------------------------------------------- the tool it builds on
def test_the_tool_of_card_add135_hashes_as_recorded_and_is_the_one_imported():
    assert RL.lf_sha(LU.__file__)[:16] == RL.LU_SHA16 == RL.CFG["lu_sha16"]
    assert RL.LU is TL.LU is LU


def test_tick_of_the_last_rule_tick_before_the_first_unsafe_step():
    assert [RL.tick_of(k, 5) for k in (1, 2, 5, 6, 10, 11, 26, 30)] == [0, 0, 0, 5, 5, 10, 25, 25]
    assert RL.tick_of(8, 5) == 5 and RL.tick_of(3, 5) == 0 and RL.tick_of(13, 5) == 10 and RL.tick_of(7, 2) == 6


# ---------------------------------------------------------------------------------------------- the world and the split by construction
def test_world_is_the_one_derived_by_hand(world):
    w, p, cfg, tabs, probs = world
    assert w.complete and not w.problems and not any(probs.values())
    for s in range(4):
        S = w.S[("RLc", s)]
        want = {i: LEAD[i] for i in sorted(FAST[("RLc", s)]) if LEAD[i] != "c7"}
        assert np.flatnonzero(S["sel"]).tolist() == sorted(want)
        for i, kind in want.items():
            assert S["first"][i] == TYPEF[kind][0]
        assert not S["ud"][6]
    assert [int(w.S[("RLc", s)]["sel"].sum()) for s in range(4)] == cfg["expect_unsafe"]


@pytest.mark.parametrize("arm", ["RLc", "GEc"])
def test_split_equals_the_hand_numbers_window_by_window(world, arm):
    w, p, cfg, tabs, probs = world
    for s in range(4):
        T = tabs[(arm, s)]
        rows = [r for r in expected_rows(arm) if r[0] == s]
        ci = w.D[(arm, s)]["clip_index"][np.flatnonzero(w.S[(arm, s)]["sel"])]
        assert [r[1] for r in rows] == ci.tolist()
        d = RL.parts(T)
        for j, (_, i, kind, v, on, sat) in enumerate(rows):
            ks, kt, istep, S, eL, eX = TYPEF[kind]
            assert (T["ks"][j], T["kt"][j], T["i"][j]) == (ks, kt, istep)
            assert bool(T["on"][j]) == on and int(T["sat"][j]) == sat
            assert d["S"][j] == pytest.approx(S, abs=1e-5)
            if on:
                assert d["eG"][j] == pytest.approx(DELTA[arm], abs=1e-5)
                assert T["vL"][j] == v
                assert d["eL"][j] == pytest.approx(eL(v), abs=1e-5) and d["eX"][j] == pytest.approx(eX(v) - DELTA[arm], abs=1e-5)
                assert d["S"][j] == pytest.approx(d["eL"][j] + d["eX"][j] + d["eG"][j], abs=1e-6)


def test_a_lead_that_slows_after_the_tick_is_forecast_and_a_rollout_beyond_the_cap_is_execution(world):
    w, p, cfg, tabs, probs = world
    T = tabs[("RLc", 3)]                                   # windows 0 (dip, v 6.5), 2, 5, 8, 11 (dip, v 8), 10 (c11, v 8)
    d = RL.parts(T)
    dip8 = [j for j in range(len(T["on"])) if T["ks"][j] == 8 and T["vL"][j] == 8.0]
    assert len(dip8) == 4
    for j in dip8:
        assert d["eL"][j] == pytest.approx(3.0, abs=1e-5) and abs(d["eX"][j]) < 1e-5
    jc = [j for j in range(len(T["on"])) if T["ks"][j] == 3][0]
    assert abs(d["eL"][jc]) < 1e-5 and d["eX"][jc] == pytest.approx(2.2, abs=1e-5) and d["S"][jc] == pytest.approx(2.2, abs=1e-5)


def test_measures_on_the_world_with_known_answers(world):
    w, p, cfg, tabs, probs = world
    m, beside = RL.read_all(cfg, w, tabs)
    rows = expected_rows("RLc")
    assert len(rows) == 22
    on = [r for r in rows if r[4]]
    eLs = [TYPEF[r[2]][4](r[3]) for r in on]
    eXs = [TYPEF[r[2]][5](r[3]) for r in on]
    Ss = [TYPEF[r[2]][3] for r in on]
    assert m["R1"]["value"] == pytest.approx(20 / 22) and m["R1"]["n"] == 22
    assert m["R2"]["value"] == pytest.approx(2 / 22) and m["R2"]["n"] == 22
    assert m["R3"]["value"] == pytest.approx(sum(eLs) / sum(Ss), abs=1e-5) and m["R3"]["n"] == 20
    assert m["R4"]["value"] == pytest.approx(sum(1 for a, b in zip(eLs, eXs) if a > b) / 20) and m["R4"]["n"] == 20
    for k in ("R1", "R2", "R3", "R4"):
        lo, hi = m[k]["interval"]
        assert 0.0 <= lo <= m[k]["value"] <= hi <= 1.0 and set(m[k]) == {"value", "interval", "n"}
    s = beside["subject"]
    assert s["n_unsafe_windows_per_seed"] == [7, 6, 3, 6] and s["n_rule_on_windows"] == 20 and s["n_unsafe_windows"] == 22
    assert s["medians_rule_on"]["S"] == pytest.approx(float(np.median(Ss)), abs=1e-5) and s["medians_rule_on"]["i"] == 3.0
    assert s["medians_rule_on"]["e_L"] == pytest.approx(float(np.median(eLs)), abs=1e-5)
    assert s["medians_rule_on"]["e_X"] == pytest.approx(float(np.median(eXs)), abs=1e-5)
    nspeed = sum(1 for r in on if (r[2] == "dip") or (r[2] == "c11" and r[3] == 9.0))
    assert s["share_lead_mean_speed_more_than_0.5_below_v_L"] == pytest.approx(nspeed / 20)
    assert s["share_tick_at_step_0"] == pytest.approx(sum(1 for r in rows if r[2] == "c11") / 22)
    ps = s["per_seed_R1_to_R4"]
    assert ps["R1"] == pytest.approx([6 / 7, 5 / 6, 1.0, 1.0]) and ps["R2"] == pytest.approx([1 / 7, 0.0, 1 / 3, 0.0])
    g = beside["GEc"]
    assert g["R1_to_R4"]["R1"]["value"] == 1.0 and g["R1_to_R4"]["R1"]["n"] == 7 and g["n_unsafe_windows_per_seed"] == [2, 2, 3, 0]
    assert g["third_part_g0_minus_label_gap_at_the_tick"]["median"] == pytest.approx(0.5, abs=1e-5) and g["third_part_g0_minus_label_gap_at_the_tick"]["share_positive"] == 1.0
    gr = expected_rows("GEc")
    gl = [TYPEF[r[2]][4](r[3]) for r in gr]
    gs = [TYPEF[r[2]][3] for r in gr]
    assert g["R1_to_R4"]["R3"]["value"] == pytest.approx(sum(gl) / sum(gs), abs=1e-5)
    assert math.isnan(ps["R1"][3]) is False and math.isnan(g["per_seed_R1_to_R4"]["R1"][3])        # the seed with no unsafe window of GEc reads nan


def test_bootstrap_is_the_cluster_bootstrap_over_episodes_with_seed_137(world):
    w, p, cfg, tabs, probs = world
    m, _ = RL.read_all(cfg, w, tabs)
    n_ep = 6
    rows = expected_rows("RLc")
    on = [r for r in rows if r[4]]
    num3, den3, num1, den1 = np.zeros(n_ep), np.zeros(n_ep), np.zeros(n_ep), np.zeros(n_ep)
    for s, i, kind, v, o, sat in rows:
        den1[i // 2] += 1
        num1[i // 2] += 1 if o else 0
        if o:
            num3[i // 2] += TYPEF[kind][4](v)
            den3[i // 2] += TYPEF[kind][3]
    idx = np.random.default_rng(137).integers(0, n_ep, size=(2000, n_ep))
    for k, num, den in (("R3", num3, den3), ("R1", num1, den1)):
        vals = [num[d].sum() / den[d].sum() for d in idx if den[d].sum() > 0]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        assert m[k]["interval"] == pytest.approx([lo, hi], abs=1e-5)
    assert len(on) == 20


def test_const_gap_world_of_the_labelunsafe_tests_with_rule_logs_added(E, tmp_path):
    """the world of the tests of card ADD135 (the lead moves at the logged speed, a constant gap): with the rule forecasting that speed the forecast part is 0 and the whole loss is execution"""
    p = TL.build(tmp_path / "tl", TL.W1)
    t0 = np.arange(1, 31) * 0.2
    G = {c: b[0] for c, b in zip(range(12), TL.BASE)}
    for arm, tag in RL.TAGS.items():
        for s in range(4):
            dd = LU.dump_folder(p["root"], tag, s)
            cols = {k: [] for k in ("ci", "t0", "j", "k", "tick", "active", "present", "src_bad", "sat", "sat_exec", "g_src", "v0", "v_l", "v_L", "a_l", "a_m", "g0")}
            for f in sorted(os.listdir(dd)):
                with np.load(os.path.join(dd, f), allow_pickle=False) as z:
                    g, cl, ci = z["g"], z["cl"], z["clip_index"]
                for r, c in enumerate(ci.tolist()):
                    v = float(g[r, -1, 0] / 6.0)
                    for k in range(0, 30, 5):
                        a_l = 0.0 if k == 0 else float(g[r, k - 1, 0])
                        a_m = 0.0 if k == 0 else float(cl[r, k - 1, 0])
                        vals = dict(ci=c, t0=0, j=k, k=k, tick=k // 5, active=True, present=True, src_bad=False, sat=0, sat_exec=0, g_src=G[c], v0=v, v_l=v, v_L=v, a_l=a_l, a_m=a_m, g0=G[c] + a_l - a_m)
                        for kk in cols:
                            cols[kk].append(vals[kk])
            dt = {"ci": np.int64, "t0": np.int64, "j": np.int64, "k": np.int64, "tick": np.int64, "active": bool, "present": bool, "src_bad": bool, "sat": np.int64, "sat_exec": np.int64}
            arr = {c: np.array(v, dtype=dt.get(c, np.float64)) for c, v in cols.items()}
            half = np.isin(arr["k"], (0, 10, 20))
            d = os.path.join(LU.run_folder(p["root"], tag, s), "ticks")
            os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, "rule_log_%s_s%d.npz" % (tag, s)), **{c: a[half] for c, a in arr.items()})
            np.savez(os.path.join(d, "rule_log_mid_%s_s%d.npz" % (tag, s)), **{c: a[~half] for c, a in arr.items()})
    cfg = tiny_cfg(expect_unsafe=[5, 4, 4, 4], expected_rates={"RLc": {"per_seed": [5 / 11, 4 / 11, 4 / 11, 4 / 11], "pooled": 17 / 44}, "GEc": {"pooled": 6 / 44}})
    w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
    tabs, probs = RL.load_tables(E, cfg, w, p["root"])
    assert not any(probs.values())
    n = 0
    for key, T in tabs.items():
        d = RL.parts(T)
        assert np.all(np.abs(d["eL"]) < 1e-4) and np.allclose(d["eX"], d["S"], atol=1e-4)
        n += len(T["on"])
    assert n == 17 + 6
    m, _ = RL.read_all(cfg, w, tabs)
    assert m["R1"]["value"] == 1.0 and abs(m["R3"]["value"]) < 1e-4 and m["R3"]["interval"][1] < 1e-3 and m["R4"]["value"] == 0.0
    assert RL.derive_names(m) == [RL.N_EXEC, RL.N_RULE_ON, RL.N_NOBRAKE]
    res, _, _ = RL.run_controls(E, cfg, p["root"], {"join": p["join"], "grid": p["grid"]}, [], LU.__file__)
    assert [res[c][0] for c in ("w0", "w1", "w2", "w3", "w5")] == [True] * 5 and not res["w4"][0]


# ---------------------------------------------------------------------------------------------- the names: every one reached
def meas(r1, r2, r3):
    return {"R1": {"value": 0.0, "interval": list(r1), "n": 5}, "R2": {"value": 0.0, "interval": list(r2), "n": 5}, "R3": {"value": 0.0, "interval": list(r3), "n": 5}}


def test_names_by_rule_and_the_two_routines_agree():
    nan = float("nan")
    cases = [((0.9, 1.0), (0.5, 1.0), (0.5, 0.9), [RL.N_FORECAST, RL.N_RULE_ON, RL.N_BRAKE]), ((0.89, 1.0), (0.49, 1.0), (0.1, 0.5), [RL.N_EXEC, RL.N_RULE_OFF, RL.N_NOBRAKE]),
             ((0.95, 1.0), (0.0, 0.2), (0.3, 0.8), [RL.N_MIXED, RL.N_RULE_ON, RL.N_NOBRAKE]), ((0.0, 0.5), (0.7, 0.9), (0.0, 0.0), [RL.N_EXEC, RL.N_RULE_OFF, RL.N_BRAKE]),
             ((nan, nan), (nan, nan), (nan, nan), [RL.N_MIXED, RL.N_RULE_OFF, RL.N_NOBRAKE]), ((1.0, 1.0), (1.0, 1.0), (0.5, 0.5), [RL.N_FORECAST, RL.N_RULE_ON, RL.N_BRAKE]),
             ((0.9, 0.9), (0.4999, 0.6), (0.4999, 0.5001), [RL.N_MIXED, RL.N_RULE_ON, RL.N_NOBRAKE])]
    for r1, r2, r3, want in cases:
        assert RL.derive_names(meas(r1, r2, r3)) == want
    assert {c[3][0] for c in cases} == {RL.N_FORECAST, RL.N_EXEC, RL.N_MIXED}
    assert {c[3][1] for c in cases} == {RL.N_RULE_ON, RL.N_RULE_OFF} and {c[3][2] for c in cases} == {RL.N_BRAKE, RL.N_NOBRAKE}


def test_a_disagreement_of_the_two_routines_is_refused(monkeypatch):
    for fn in ("split_name_b", "rule_name_b", "brake_name_b"):
        with monkeypatch.context() as mp:
            mp.setattr(RL, fn, lambda m: "other")
            with pytest.raises(SystemExit):
                RL.derive_names(meas((0.95, 1.0), (0.6, 1.0), (0.6, 0.9)))


def _world_names(E, tmp, fast_rows, opts=None):
    fast = {("RLc", s): set(fast_rows) for s in range(4)}
    fast.update({("GEc", s): set() for s in range(4)})
    lo = {("RLc", s): dict(opts or {}) for s in range(4)}
    p = build_rl(tmp, fast=fast, logopt=lo)
    cfg = tiny_cfg()
    w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
    tabs, probs = RL.load_tables(E, cfg, w, p["root"])
    assert not any(probs.values())
    m, _ = RL.read_all(cfg, w, tabs)
    return m


def test_names_reached_through_worlds(E, tmp_path):
    m = _world_names(E, tmp_path / "a", {0, 2, 5, 8, 11})                             # dips only, v assumed 8: the lead's change of speed after the tick is the whole loss
    assert m["R3"]["interval"][0] >= 0.99 and m["R1"]["interval"] == [1.0, 1.0] and m["R2"]["interval"] == [0.0, 0.0]
    assert RL.derive_names(m) == [RL.N_FORECAST, RL.N_RULE_ON, RL.N_NOBRAKE]
    m = _world_names(E, tmp_path / "b", {1, 3, 4, 7, 9, 10}, {"*": {"sat": 1}})        # lead at constant speed, rollout beyond the cap, the braking limit binds in every window
    assert m["R3"]["interval"][1] <= 1e-3 and m["R2"]["interval"] == [1.0, 1.0]
    assert RL.derive_names(m) == [RL.N_EXEC, RL.N_RULE_ON, RL.N_BRAKE]
    m = _world_names(E, tmp_path / "c", {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11}, {"*": {"vL": 8.0}})     # dips and constant-speed windows together: the split is mixed
    assert RL.derive_names(m)[0] == RL.N_MIXED
    m = _world_names(E, tmp_path / "d", {0, 1, 2, 3}, {"*": {"present": False}})           # no lead present at any tick: the rule is off
    assert m["R1"]["value"] == 0.0 and RL.derive_names(m)[1] == RL.N_RULE_OFF and math.isnan(m["R3"]["value"])
    assert RL.derive_names(m)[0] == RL.N_MIXED


def test_the_main_world_names_follow_the_intervals(world):
    w, p, cfg, tabs, probs = world
    m, _ = RL.read_all(cfg, w, tabs)
    lo3, hi3 = m["R3"]["interval"]
    split = RL.N_FORECAST if lo3 >= 0.5 else RL.N_EXEC if hi3 <= 0.5 else RL.N_MIXED
    rule = RL.N_RULE_ON if m["R1"]["interval"][0] >= 0.9 else RL.N_RULE_OFF
    brake = RL.N_BRAKE if m["R2"]["interval"][0] >= 0.5 else RL.N_NOBRAKE
    assert RL.derive_names(m) == [split, rule, brake]


# ---------------------------------------------------------------------------------------------- controls, each with a planted fault
def run_ctl(E, p, cfg, tests=None, lu_path=None):
    return RL.run_controls(E, cfg, p["root"], {"join": p["join"], "grid": p["grid"]}, tests or [], lu_path or LU.__file__)[0]


def test_all_controls_pass_on_the_world(E, world, tmp_path):
    w, p, cfg, tabs, probs = world
    a, b = tmp_path / "dev.out", tmp_path / "pod.out"
    a.write_text(pytest_text(), encoding="utf-8")
    b.write_text(pytest_text(9), encoding="utf-8")
    res = run_ctl(E, p, cfg, [str(a), str(b)])
    assert {c: v[0] for c, v in res.items()} == {c: True for c in RL.CONTROLS}


def test_w0_each_fault_fails(E, world, tmp_path):
    w, p, cfg, tabs, probs = world
    assert RL.ctl_w0(E, cfg, w, LU.__file__)[0]
    ok, why = RL.ctl_w0(E, tiny_cfg(lu_sha16="0" * 16), w, LU.__file__)
    assert not ok and "hash" in why
    ok, why = RL.ctl_w0(E, tiny_cfg(expect_unsafe=[7, 6, 3, 7]), w, LU.__file__)
    assert not ok and "model seed 3" in why
    assert not RL.ctl_w0(E, tiny_cfg(expected_rates={"RLc": {"per_seed": [7 / 11, 6 / 11, 3 / 11, 6 / 11], "pooled": 0.4}, "GEc": {"pooled": 7 / 44}}), w, LU.__file__)[0]
    assert not RL.ctl_w0(E, tiny_cfg(expected_rates={"RLc": {"per_seed": [7 / 11, 6 / 11, 3 / 11, 6 / 11], "pooled": 22 / 44}, "GEc": {"pooled": 0.5}}), w, LU.__file__)[0]
    k = ("GEc", 1)
    saved = w.S[k]["first"].copy()
    w.S[k]["first"][np.flatnonzero(w.S[k]["sel"])[0]] += 1
    try:
        ok, why = RL.ctl_w0(E, cfg, w, LU.__file__)
        assert not ok and "first unsafe step" in why
    finally:
        w.S[k]["first"][:] = saved
    assert RL.ctl_w0(E, cfg, w, LU.__file__)[0]
    other = tmp_path / "other.py"
    other.write_text(open(LU.__file__, encoding="utf-8", newline="").read() + "\n# changed\n", encoding="utf-8")
    assert not RL.ctl_w0(E, cfg, w, str(other))[0]


def _fault_world(E, tmp, fn=None, arm="RLc", s=1, mid=False, **kw):
    p = build_rl(tmp, **kw)
    if fn:
        edit_log(p, arm, s, fn, mid=mid)
    cfg = tiny_cfg()
    w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
    tabs, probs = RL.load_tables(E, cfg, w, p["root"])
    return w, p, cfg, tabs, probs


def _row(d, ci, k):
    return int(np.flatnonzero((d["ci"] == ci) & (d["k"] == k))[0])


def test_w1_each_fault_fails(E, world, tmp_path):
    w, p, cfg, tabs, probs = world
    assert RL.ctl_w1(tabs, probs, cfg)[0]
    # RLc seed 1 window 0 (dip, first unsafe 8, tick 5): the row of the tick lies in the later-look log
    def shift_arc(d):
        d["a_m"][_row(d, 0, 5)] += 0.01
    _, _, _, tb, pr = _fault_world(E, tmp_path / "a", shift_arc, mid=True)
    ok, why = RL.ctl_w1(tb, pr, cfg)
    assert not ok and "arc" in why
    def shift_g0(d):
        d["g0"][_row(d, 0, 5)] += 0.01
    _, _, _, tb, pr = _fault_world(E, tmp_path / "b", shift_g0, mid=True)
    ok, why = RL.ctl_w1(tb, pr, cfg)
    assert not ok and "g0" in why
    def drop_tick(d):
        keep = np.ones(len(d["ci"]), bool)
        keep[_row(d, 0, 5)] = False
        for k in list(d):
            if np.ndim(d[k]) >= 1 and len(d[k]) == len(keep):
                d[k] = d[k][keep]
    _, _, _, tb, pr = _fault_world(E, tmp_path / "c", drop_tick, mid=True)
    ok, why = RL.ctl_w1(tb, pr, cfg)
    assert not ok and "missing" in why
    # a tick the rule rate gives (step 0) missing from the tick log
    def drop0(d):
        keep = ~((d["ci"] == 0) & (d["k"] == 0))
        for k in list(d):
            if np.ndim(d[k]) >= 1 and len(d[k]) == len(keep):
                d[k] = d[k][keep]
    _, _, _, tb, pr = _fault_world(E, tmp_path / "e", drop0, mid=False)
    assert not RL.ctl_w1(tb, pr, cfg)[0]
    # two rows of one tick that differ: ambiguous; two equal rows count as one
    opts = {("RLc", 1): {0: {"dup_diff": 5}}}
    _, _, _, tb, pr = _fault_world(E, tmp_path / "f", logopt={**LOGOPT, **opts})
    ok, why = RL.ctl_w1(tb, pr, cfg)
    assert not ok and "several differing rows" in why
    opts = {("RLc", 1): {0: {"dup": 5}}}
    _, _, _, tb, pr = _fault_world(E, tmp_path / "g", logopt={**LOGOPT, **opts})
    assert RL.ctl_w1(tb, pr, cfg)[0]
    # a unsafe window of the subject with no row at all (log of the seed emptied) fails
    def empty(d):
        for k in list(d):
            if np.ndim(d[k]) >= 1:
                d[k] = d[k][:0]
    _, _, _, tb, pr = _fault_world(E, tmp_path / "h", empty, mid=False)
    assert not RL.ctl_w1(tb, pr, cfg)[0]
    # rows of windows outside the grid are cut and do not matter (the world carries one such row in every tick log)
    assert RL.ctl_w1(tabs, probs, cfg)[0]


def test_w2_identity_fails_on_a_small_g0_error_that_w1_allows(E, world, tmp_path, monkeypatch):
    w, p, cfg, tabs, probs = world
    assert RL.ctl_w2(tabs, cfg)[0]
    opts = {("RLc", 1): {0: {"dg0": 5e-4}}}                                         # within the 1e-3 m of the join control, above the 1e-6 m of the identity
    _, _, _, tb, pr = _fault_world(E, tmp_path / "g0", logopt={**LOGOPT, **opts})
    assert RL.ctl_w1(tb, pr, cfg)[0]
    ok, why = RL.ctl_w2(tb, cfg)
    assert not ok and "sum of its parts" in why
    orig = RL.parts

    def bad(T, vL=None):
        d = orig(T, vL)
        d["eX"] = d["eX"] + 1e-3
        return d
    monkeypatch.setattr(RL, "parts", bad)
    assert not RL.ctl_w2(tabs, cfg)[0]
    monkeypatch.undo()
    assert RL.ctl_w2({k: {c: v[:0] for c, v in T.items()} for k, T in tabs.items()}, cfg)[0] is False        # nothing to read is not a pass


def test_w3_forecast_part_reads_the_forecast_alone(E, world, monkeypatch):
    w, p, cfg, tabs, probs = world
    assert RL.ctl_w3(tabs, cfg)[0]
    orig = RL.parts
    monkeypatch.setattr(RL, "parts", lambda T, vL=None: orig(T))                       # a split that ignores the lead speed it is given
    ok, why = RL.ctl_w3(tabs, cfg)
    assert not ok and "forecast part" in why and RL.ctl_w2(tabs, cfg)[0]
    monkeypatch.undo()

    def bad(T, vL=None):
        d = orig(T, vL)
        if vL is not None:
            d["eL"] = d["eL"] + 0.1 * (vL - T["vL"])                                   # a forecast part that rises by 0.1 per m/s of the assumed speed (and not 0.2 m per step)
        return d
    monkeypatch.setattr(RL, "parts", bad)
    assert not RL.ctl_w3(tabs, cfg)[0]
    monkeypatch.undo()
    assert RL.ctl_w3(tabs, cfg)[0]


def test_w4_tests_outputs(tmp_path):
    a, b = tmp_path / "dev.out", tmp_path / "pod.out"
    a.write_text(pytest_text(), encoding="utf-8")
    b.write_text(pytest_text(9), encoding="utf-8")
    assert RL.ctl_w4([str(a), str(b)])[0]
    for txt in ("test_read_rulelog10k.py ...F\n1 failed, 3 passed in 1s\n", "test_read_rulelog10k.py ....\n4 passed, 1 error in 1s\n", "\n", "5 passed in 1s\n", "test_read_rulelog10k.py\n0 passed in 1s\n",
                "test_read_rulelog10k.py ..\nFAILED x\n2 passed in 1s\n"):
        b.write_text(txt, encoding="utf-8")
        assert not RL.ctl_w4([str(a), str(b)])[0]
    b.write_text(pytest_text(), encoding="utf-8")
    assert not RL.ctl_w4([str(a)])[0] and not RL.ctl_w4([str(a), str(tmp_path / "missing.out")])[0]
    kit = tmp_path / "kit"
    kit.mkdir()
    c = kit / "pod.out"
    c.write_text(pytest_text(), encoding="utf-8")
    assert not RL.ctl_w4([str(a), str(c)])[0]


def test_w5_counts_and_each_fault_fails(E, tmp_path):
    cfg = tiny_cfg()
    for tag, kw, why in (("a", {"skip_done": [("GEc", 2)]}, "done marker"), ("b", {"no_mid": [("RLc", 1)]}, "rule log file"), ("c", {"extra_dump": True}, "dump file count")):
        p = build_rl(tmp_path / tag, **kw)
        w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
        ok, reason = RL.ctl_w5(cfg, w, p["root"])
        assert not ok and why in reason
    p = build_rl(tmp_path / "ok")
    w = LU.load_world(E, RL.cfg_lu(cfg), p["root"], p["join"], p["grid"])
    assert RL.ctl_w5(cfg, w, p["root"])[0]
    assert not RL.ctl_w5(tiny_cfg(expect_files=7), w, p["root"])[0]
    res = run_ctl(E, build_rl(tmp_path / "nomid", no_mid=[("RLc", 1)]), cfg)
    assert not res["w5"][0] and not res["w1"][0] and "could not be read" in res["w1"][1]


def test_controls_failure_text_holds_no_value(E, tmp_path):
    p = build_rl(tmp_path / "w")
    bad = tiny_cfg(expect_unsafe=[7, 6, 3, 8])
    res = run_ctl(E, p, bad)
    ok, why = res["w0"]
    assert not ok and "seed 3" in why
    assert not any(ch.isdigit() for ch in why.replace("seed 3", "").replace("ADD135", "").replace("RLc", ""))


# ---------------------------------------------------------------------------------------------- stages end to end (in process) and the refusals
def stage_args(p, out, tests, stage):
    return [stage, "--t1-root", p["root"], "--out-dir", str(out), "--join", p["join"], "--grid", p["grid"], "--lu-tool", LU.__file__] + sum([["--tests-out", t] for t in tests], [])


def test_stages_in_process_and_refusals(E, tmp_path, capsys):
    p = build_rl(tmp_path / "w")
    out = tmp_path / "out"
    cfg = tiny_cfg()
    dev, pod = tmp_path / "dev.out", tmp_path / "pod.out"
    dev.write_text(pytest_text(), encoding="utf-8")
    pod.write_text(pytest_text(), encoding="utf-8")
    tests = [str(dev), str(pod)]
    assert RL.main(stage_args(p, out, tests, "names"), cfg) == 3                      # no controls file yet
    assert RL.main(stage_args(p, out, tests, "controls"), cfg) == 0
    txt = capsys.readouterr().out
    lines = txt.strip().splitlines()
    assert [x for x in lines if x.startswith("ctl ")] == ["ctl w%d PASS" % i for i in range(6)]
    assert lines[-1] == "ZZ READ-RULELOG10K controls all_pass True"
    rec = json.load(open(out / RL.CONTROLS_NAME, encoding="utf-8"))
    assert rec["card"] == "PREREG_V8F_PROOF_ADD137" and rec["all_pass"] is True and all(rec["pass_w%d" % i] is True for i in range(6)) and "inputs" in rec and "reasons" in rec
    assert "measures" not in rec and "R3" not in json.dumps(rec)
    assert RL.main(stage_args(p, out, tests, "controls"), cfg) == 3                      # never overwritten
    pod.write_text(pytest_text(8), encoding="utf-8")                                      # an input that changed since the controls stage
    assert RL.main(stage_args(p, out, tests, "names"), cfg) == 3
    assert not (out / RL.RESULT_NAME).exists()
    pod.write_text(pytest_text(), encoding="utf-8")
    capsys.readouterr()
    assert RL.main(stage_args(p, out, tests, "names"), cfg) == 0
    txt = capsys.readouterr().out
    assert txt.strip().splitlines()[-1] == "ZZ READ-RULELOG10K names done"
    res = json.load(open(out / RL.RESULT_NAME, encoding="utf-8"))
    assert res["card"] == "PREREG_V8F_PROOF_ADD137" and "T1-kin/hb0.2" in res["tier"] and len(res["names"]) == 3 and "not driving performance" in res["tier"]
    assert set(res["measures"]) == {"R1", "R2", "R3", "R4"} and all(set(res["measures"][k]) == {"value", "interval", "n"} for k in res["measures"])
    assert "beside" in res and len(res["families"]) == 4 and res["families"][0].startswith("LONGITUDINAL") and all("n = 0 here" in x for x in res["families"][1:])
    assert res["measures"]["R1"]["value"] == pytest.approx(20 / 22)
    md = (out / RL.VERDICT_NAME).read_text(encoding="utf-8")
    assert "not driving performance" in md.lower() and "T1-kin/hb0.2" in md and res["names"][0] in md
    for bad in ("epi0000", "clip00"):
        assert bad not in md and bad not in json.dumps(res) and bad not in txt and bad not in json.dumps(rec)
    assert RL.main(stage_args(p, out, tests, "names"), cfg) == 3                         # outputs exist
    for word in ("voi" + "d", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"):
        assert word not in md.lower() and word not in json.dumps(res).lower()


def test_a_changed_rule_log_after_the_controls_stage_is_refused(E, tmp_path, capsys):
    p = build_rl(tmp_path / "w")
    out = tmp_path / "out"
    cfg = tiny_cfg()
    dev, pod = tmp_path / "dev.out", tmp_path / "pod.out"
    dev.write_text(pytest_text(), encoding="utf-8")
    pod.write_text(pytest_text(), encoding="utf-8")
    tests = [str(dev), str(pod)]
    assert RL.main(stage_args(p, out, tests, "controls"), cfg) == 0
    edit_log(p, "GEc", 0, lambda d: d["v_L"].__setitem__(0, 3.0))
    capsys.readouterr()
    assert RL.main(stage_args(p, out, tests, "names"), cfg) == 3
    assert "REFUSED" in capsys.readouterr().out and not (out / RL.RESULT_NAME).exists()


def test_controls_stage_with_a_failed_control_prints_no_value_and_names_is_refused(E, tmp_path, capsys):
    p = build_rl(tmp_path / "w")
    out = tmp_path / "out"
    dev = tmp_path / "dev.out"
    dev.write_text(pytest_text(), encoding="utf-8")
    cfg = tiny_cfg(expect_unsafe=[7, 6, 3, 8])
    assert RL.main(stage_args(p, out, [str(dev)], "controls"), cfg) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("ctl w0 FAIL") and lines[4].startswith("ctl w4 FAIL") and lines[1] == "ctl w1 PASS"
    assert lines[-1] == "ZZ READ-RULELOG10K controls all_pass False"
    rec = json.load(open(out / RL.CONTROLS_NAME, encoding="utf-8"))
    assert rec["all_pass"] is False and rec["pass_w0"] is False and rec["pass_w1"] is True
    assert RL.main(stage_args(p, out, [str(dev)], "names"), cfg) == 3


def run_child(args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "read_rulelog10k.py")] + args, cwd=CWD, env=env, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout


@pytest.mark.parametrize("variant", ["no_file", "w2_false", "all_pass_false", "string_true", "no_inputs", "wrong_card"])
def test_names_stage_is_refused_in_a_child_process(tmp_path, variant):
    out = tmp_path / "o"
    out.mkdir()
    rec = {"card": "PREREG_V8F_PROOF_ADD137", "all_pass": True, "inputs": {"files": {"x": {"path": str(tmp_path / "nothing"), "sha256": None}}, "run_dumps": {"a": {"path": "", "sha256": "", "n_files": 0}}}}
    for i in range(6):
        rec["pass_w%d" % i] = True
    if variant == "w2_false":
        rec["pass_w2"] = False
    elif variant == "all_pass_false":
        rec["all_pass"] = False
    elif variant == "string_true":
        rec["pass_w1"] = "true"
    elif variant == "no_inputs":
        del rec["inputs"]
    elif variant == "wrong_card":
        rec["card"] = "PREREG_V8F_PROOF_ADD135"
    if variant != "no_file":
        (out / RL.CONTROLS_NAME).write_text(json.dumps(rec), encoding="utf-8")
    rc, txt = run_child(["names", "--out-dir", str(out)])
    assert rc == 3 and "REFUSED" in txt and "names done" not in txt
    assert not (out / RL.RESULT_NAME).exists() and not (out / RL.VERDICT_NAME).exists()


def test_parser_has_the_documented_options():
    a = RL.build_parser().parse_args(["controls", "--tests-out", "a", "--tests-out", "b"])
    assert a.tests_out == ["a", "b"] and a.t1_root.endswith("integ10k/t1") and a.out_dir.endswith("rulelog/read_out") and a.lu_tool is None
    with pytest.raises(SystemExit):
        RL.build_parser().parse_args(["other"])
