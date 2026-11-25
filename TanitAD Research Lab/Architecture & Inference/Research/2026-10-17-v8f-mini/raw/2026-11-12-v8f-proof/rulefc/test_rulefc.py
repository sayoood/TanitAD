"""Tests of card ADD144 (RULEFC10K): dist_rule_fc.py, the wrapper rzgfhfc (pure parts and refusals) and the queue box2_rulefc_queue.py. Synthetic data only.

  RZGFH_DIR=<v8proof>/rhacc10k/kit_build DISTRULE_DIR=<v8proof>/rhacc10k/kit_build python -m pytest -q test_rulefc.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, ".."))
KB = os.path.join(V8, "rhacc10k", "kit_build")
os.environ.setdefault("RZGFH_DIR", KB)
os.environ.setdefault("DISTRULE_DIR", KB)
for p in (os.path.join(HERE, "build"), KB, os.path.join(HERE, "kit")):
    if p not in sys.path:
        sys.path.insert(0, p)

import dist_rule as R0  # noqa: E402
import dist_rule_fc as F  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FC = load("rzgfhfc_t", os.path.join(HERE, "build", "v8f_t1_eval_reader_rule_rzgfhfc.py"))
Q = load("rulefc_q_t", os.path.join(HERE, "kit", "box2_rulefc_queue.py"))


def batch(seed, B=64, N=30):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.0, 15.0, B)
    sp = np.clip(v[:, None] + np.cumsum(rng.normal(0, 0.3, (B, N)), 1), 0, None)
    hd = np.cumsum(rng.normal(0, 0.01, (B, N)), 1)
    x = np.cumsum(sp * 0.2 * np.cos(hd), 1)
    y = np.cumsum(sp * 0.2 * np.sin(hd), 1)
    plan = np.stack([x, y], -1).astype(np.float32)
    present = rng.random(B) < 0.7
    g = rng.uniform(2.0, 70.0, B)
    g0 = g + rng.normal(0, 0.5, B)
    vL = np.maximum(0.0, v + rng.normal(0, 2.0, B))
    return plan, v, present, g, g0, vL


# ---------------------------------------------------------------- r1: a_L = 0 (or None) is the rule of record bit for bit
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("m", [10, 5, 1])
def test_r1_identity(seed, m):
    plan, v0, pres, g, g0, vL = batch(seed)
    ref = R0.distance_rule(plan, v0, pres, g, g0, vL, m=m)
    for aL in (None, np.zeros(len(v0))):
        out = F.distance_rule(plan, v0, pres, g, g0, vL, m=m, a_L=aL)
        assert np.array_equal(out.plan, ref.plan)
        assert out.plan.dtype == ref.plan.dtype
        for k in ("active", "n_sat", "n_sat_exec", "n_lift", "changed"):
            assert np.array_equal(getattr(out, k), getattr(ref, k)), k
        assert np.array_equal(np.nan_to_num(out.a_new, nan=-7.0), np.nan_to_num(ref.a_new, nan=-7.0))


def test_r1_identity_kwargs():
    plan, v0, pres, g, g0, vL = batch(11)
    for kw in ({"tau": 0.8}, {"floor_lifts": False}, {"use_cap": False, "use_floor": False}):
        ref = R0.distance_rule(plan, v0, pres, g, g0, vL, m=10, **kw)
        out = F.distance_rule(plan, v0, pres, g, g0, vL, m=10, a_L=np.zeros(len(v0)), **kw)
        assert np.array_equal(out.plan, ref.plan)


def test_lead_disp_zero_same_arithmetic():
    v = np.random.default_rng(3).uniform(0, 30, 1000)
    for i in range(1, 31):
        assert np.array_equal(F.lead_disp(v, np.zeros_like(v), 0.2, i), v * 0.2 * i)
        assert np.array_equal(F.lead_disp(v, None, 0.2, i), v * 0.2 * i)


# ---------------------------------------------------------------- r2: the planted forecast
def analytic(v, a, t):
    if a >= 0:
        return v * t + 0.5 * a * t * t
    ts = v / -a
    return v * t + 0.5 * a * t * t if t <= ts else v * v / (2 * -a)


def test_r2_lead_disp_analytic_and_stop():
    v = np.array([10.0, 4.0, 0.0, 7.0])
    a = np.array([-2.0, -2.0, -2.0, 1.5])
    prev = None
    for i in range(1, 31):
        t = 0.2 * i
        s = F.lead_disp(v, a, 0.2, i)
        for b in range(4):
            assert abs(s[b] - analytic(v[b], a[b], t)) < 1e-9
        if prev is not None:
            assert (s[:3] >= prev[:3] - 1e-12).all()
        prev = s
    s_late = [F.lead_disp(v, a, 0.2, i)[1] for i in range(11, 31)]   # v = 4, a = -2 stops at 2.0 s = step 10
    assert max(s_late) - min(s_late) == 0.0 and abs(s_late[0] - 4.0) < 1e-12


def test_r2_cap_in_rule():
    """With tau = 0, no floor and a plan far ahead, the rule's arc is exactly the cap g0 + S_L(i) - D."""
    B, N = 3, 30
    plan = np.stack([np.arange(1, N + 1, dtype=np.float64)[None].repeat(B, 0) * 50.0, np.zeros((B, N))], -1).astype(np.float32)
    v0 = np.full(B, 10.0)
    pres = np.ones(B, bool)
    g = np.full(B, 30.0)
    vL = np.array([10.0, 4.0, 6.0])
    aL = np.array([-2.0, -2.0, 0.0])
    out = F.distance_rule(plan, v0, pres, g, g, vL, m=10, tau=0.0, use_floor=False, a_L=aL)
    for b in range(B):
        for i in range(1, N + 1):
            want = g[b] + analytic(vL[b], aL[b], 0.2 * i) - 3.0
            assert abs(out.a_new[b, i - 1] - want) < 1e-9


def test_nonfinite_aL_is_zero():
    plan, v0, pres, g, g0, vL = batch(5)
    aL = np.full(len(v0), np.nan)
    ref = R0.distance_rule(plan, v0, pres, g, g0, vL, m=10)
    out = F.distance_rule(plan, v0, pres, g, g0, vL, m=10, a_L=aL)
    assert np.array_equal(out.plan, ref.plan)


# ---------------------------------------------------------------- r3: the measured change of speed is causal
class FakeLabel:
    def __init__(self, rows):
        self.rows = rows        # (ci, j) -> (present, gap, closing)

    def read(self, ci, j):
        return self.rows.get((ci, j), (False, float("nan"), 0.0))


def test_lead_accel_values():
    lv = np.linspace(5.0, 8.0, 40)
    rows = {(0, 20): (True, 15.0, -1.0), (0, 15): (True, 16.0, 0.5), (1, 20): (True, 12.0, -0.3), (2, 3): (True, 10.0, 0.0)}
    lab = FakeLabel(rows)
    a, vn, vp, okn, okp = FC.lead_accel(lab, [lv, lv, lv], np.array([0, 1, 2]), np.array([20, 20, 3]))
    assert okn.tolist() == [True, True, True] and okp.tolist() == [True, False, False]
    assert abs(vn[0] - (lv[20] - 1.0)) < 1e-12 and abs(vp[0] - (lv[15] + 0.5)) < 1e-12
    assert abs(a[0] - ((lv[20] - 1.0) - (lv[15] + 0.5)) / 1.0) < 1e-12
    assert a[1] == 0.0 and a[2] == 0.0          # no lead at f - 5; f - 5 outside the clip


def test_lead_accel_clips_at_zero_speed():
    lv = np.full(40, 1.0)
    lab = FakeLabel({(0, 20): (True, 5.0, -3.0), (0, 15): (True, 6.0, -0.5)})
    a, vn, vp, _, _ = FC.lead_accel(lab, [lv], np.array([0]), np.array([20]))
    assert vn[0] == 0.0 and vp[0] == 0.5 and abs(a[0] + 0.5) < 1e-12


def test_r3_causal_future_label():
    lv = np.linspace(5.0, 8.0, 40)
    rows = {(0, 20): (True, 15.0, -1.0), (0, 15): (True, 16.0, 0.5)}
    a1 = FC.lead_accel(FakeLabel(dict(rows)), [lv], np.array([0]), np.array([20]))[0]
    rows2 = dict(rows)
    for x in range(21, 40):
        rows2[(0, x)] = (True, 1.0, -9.0)
    lv2 = lv.copy()
    lv2[21:] = 99.0
    a2 = FC.lead_accel(FakeLabel(rows2), [lv2], np.array([0]), np.array([20]))[0]
    assert a1[0] == a2[0]


# ---------------------------------------------------------------- the routed call
def setup_routed(zero=False):
    lv = np.linspace(5.0, 8.0, 60)
    rows = {}
    for x in range(60):
        rows[(0, x)] = (True, 20.0 - 0.1 * x, -0.2 - 0.01 * x)
    lab = FakeLabel(rows)
    for k in FC.STATE["log"]:
        FC.STATE["log"][k] = []
    FC.STATE["M"], FC.STATE["zero"] = 10, zero
    FC.STATE["ctx"] = (np.array([0]), np.array([30]), [lv])
    return lab, lv, rows


def test_routed_q_from_m_and_log():
    lab, lv, rows = setup_routed()
    rt = FC.routed(F, lab)
    plan, v0, _, g, g0, _ = batch(1, B=1)
    for m, q in ((10, 0), (7, 3), (1, 9)):
        f = 30 + q
        vL = np.array([max(0.0, lv[f] + rows[(0, f)][2])])
        out = rt(plan, v0, np.array([True]), g, g0, vL, m=m)
        aL = ((lv[f] + rows[(0, f)][2]) - (lv[f - 5] + rows[(0, f - 5)][2])) / 1.0
        ref = F.distance_rule(plan, v0, np.array([True]), g, g0, vL, m=m, a_L=np.array([aL]))
        assert np.array_equal(out.plan, ref.plan)
    assert np.concatenate(FC.STATE["log"]["q"]).tolist() == [0, 3, 9]
    assert np.concatenate(FC.STATE["log"]["f"]).tolist() == [30, 33, 39]
    FC.STATE["ctx"] = None


def test_routed_mismatch_stops():
    lab, lv, rows = setup_routed()
    rt = FC.routed(F, lab)
    plan, v0, _, g, g0, _ = batch(1, B=1)
    f_wrong = 31
    vL = np.array([max(0.0, lv[f_wrong] + rows[(0, f_wrong)][2])])
    with pytest.raises(RuntimeError, match="differs"):
        rt(plan, v0, np.array([True]), g, g0, vL, m=10)
    out = rt(plan, v0, np.array([False]), g, g0, vL, m=10)          # no lead for the caller: nothing to compare, inactive
    assert not out.active.any()
    FC.STATE["ctx"] = None


def test_routed_zero_equals_record():
    lab, lv, rows = setup_routed(zero=True)
    rt = FC.routed(F, lab)
    plan, v0, _, g, g0, _ = batch(2, B=1)
    vL = np.array([max(0.0, lv[30] + rows[(0, 30)][2])])
    out = rt(plan, v0, np.array([True]), g, g0, vL, m=10)
    ref = R0.distance_rule(plan, v0, np.array([True]), g, g0, vL, m=10)
    assert np.array_equal(out.plan, ref.plan)
    assert np.concatenate(FC.STATE["log"]["a_L"]).tolist() == [0.0]
    FC.STATE["ctx"] = None


def test_routed_outside_call_refuses():
    FC.STATE["ctx"] = None
    rt = FC.routed(F, FakeLabel({}))
    plan, v0, pres, g, g0, vL = batch(0, B=2)
    with pytest.raises(RuntimeError, match="outside"):
        rt(plan, v0, pres, g, g0, vL, m=10)


# ---------------------------------------------------------------- refusals of the wrapper
@pytest.mark.parametrize("argv,msg", [
    (["--lead-forecast", "cv", "--fc-zero", "--tree", "x", "run"], "go with"),
    (["--lead-forecast", "ca", "--tree", "x", "--form", "RLc", "run"], "needs --fc-log"),
    (["--lead-forecast", "ca", "--fc-log", "x.npz", "--tree", "x", "--form", "RR", "run"], "label forms"),
])
def test_refusals(argv, msg):
    with pytest.raises(SystemExit, match=msg):
        FC.main(argv)


# ---------------------------------------------------------------- the queue
def test_queue_rows_and_order():
    runs = Q.build_runs("D:/tanitad/rhacc10k")
    assert [r["arm"] for r in runs] == ["H_RLcFCz_s0", "H_RLcE1_s0", "H_RLcFC_s0", "H_RLcE1FC_s0", "H_RLcE1_s1", "H_RLcFC_s1", "H_RLcE1FC_s1",
                                        "H_RLcE1_s2", "H_RLcFC_s2", "H_RLcE1FC_s2"]
    for r in runs:
        a = r["argv"]
        assert a[a.index("--form") + 1] == "RLc"
        assert a[a.index("--ckpt") + 1].endswith("/inputs/RH_s%d/ckpt_last.pt" % r["m"])
        ev = a[a.index("--rule-every") + 1]
        assert ev == ("1" if "E1" in r["name"] else "5")
        if "FC" in r["name"]:
            assert a[1].endswith("rzgfhfc.py") and a[2:4] == ["--lead-forecast", "ca"]
            assert ("--fc-zero" in a) == (r["name"] == "H_RLcFCz")
        else:
            assert a[1].endswith("rzgfh.py") and "--lead-forecast" not in a


def normalise(argv, arm, ref_arm):
    out, skip = [], 0
    for i, x in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if x in ("--lead-forecast", "--fc-log"):
            skip = 1
            continue
        if x == "--fc-zero":
            continue
        out.append(x.replace("rzgfhfc.py", "rzgfh.py").replace(arm, ref_arm))
    return out


def test_queue_argv_equals_record_partner():
    """r4 argv: every run equals the argv of H_RLcK_s<m> (card ADD140, pulled) except the forecast flags, the wrapper, --rule-every and the names."""
    pulled = os.path.join(V8, "rhacc10k", "pull")
    runs = Q.build_runs("D:/tanitad/rhacc10k")
    n = 0
    for r in runs:
        p = os.path.join(pulled, "H_RLcK_s%d" % r["m"], "argv.json")
        if not os.path.isfile(p):
            continue
        ref = json.load(open(p, encoding="utf-8"))["argv"]
        got = normalise(r["argv"], r["arm"], "H_RLcK_s%d" % r["m"])
        i = got.index("--rule-every")
        got[i + 1] = "5"
        assert got == ref, r["arm"]
        n += 1
    assert n == len(runs)


def test_queue_dry(capsys):
    assert Q.main(["--dry", "--root", "D:/nonexistent_root_for_test"]) == 0
    o = capsys.readouterr().out
    assert "DRY: 10 runs listed" in o
