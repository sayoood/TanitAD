"""Tests of vdt_read.py (PREREG_VDAC_TRK.md B5).  Synthetic tests are fast; `real` tests use the banked navtest table and the TRK dump and TIE THE READER TO A BANKED NUMBER
(FILTER-ORACLE F_dac).  Run:  VDT_PK=<PK root export> pytest -q test_vdt_read.py        (VDT_READ_MODULE=<path> runs the same tests against a mutated copy: the mutation tests do this)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MOD = os.environ.get("VDT_READ_MODULE", str(HERE / "vdt_read.py"))
spec = importlib.util.spec_from_file_location("vdt_read_under_test", MOD)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

A_ROOT = "E:/Projects/TanitAD-artifacts/"
TABLE = A_ROOT + "scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
TRK = A_ROOT + "trkscore/work/trk_c1/trk_c1.npz"
ORACLE = A_ROOT + "vetodiag/FILTER_ORACLE.json"
real = pytest.mark.skipif(not (Path(TRK).exists() and Path(ORACLE).exists()), reason="banked navtest inputs not on this box")
PK = os.environ.get("VDT_PK") or V.DEFAULT_PK


def ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def bars_in(vt_pdms=ci(0.5, 0.3, 0.7), M=ci(-2, -3, -1), vt_sub=None, vtr_pdms=ci(0.4, 0.2, 0.6), shuf_pdms=ci(0.0, -0.1, 0.1), vtr=True):
    flat = ci(0.0, -0.1, 0.1)
    vt = {m: dict(flat) for m in V.JUDGED}
    vt["pdms"] = dict(vt_pdms)
    vt.update(vt_sub or {})
    r = None
    if vtr:
        r = {m: dict(flat) for m in V.JUDGED}
        r["pdms"] = dict(vtr_pdms)
    return {"vt": vt, "M": M, "vtr": r, "shuf": {"pdms": shuf_pdms}}


def outcome_of(c, **kw):
    return V.decide(V.eval_bars(c), **kw)["outcome"]


ALL_OK = dict(controls_failed=(), controls_pending=(), cal_null=False, ctrl_t=True, families=True)


# ------------------------------------------------------------------------------------------------------------------ synthetic: the rule
def synth(n=300, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 64, 6)).astype(np.float32) + 1.0
    return S


def test_a_all_drivable_field_changes_nothing_synthetic():
    S = synth(); a = V.load_pk(PK)[0](S); ks = a.argmax(1)
    for m in V.GRID:
        k = V.veto_pick(a, np.full(a.shape, 6.0), m, ks)
        assert (k == ks).all()


def test_c_always_veto_falls_back_to_c1_synthetic():
    S = synth(); a = V.load_pk(PK)[0](S); ks = a.argmax(1)
    for m in V.GRID:
        k = V.veto_pick(a, np.full(a.shape, -50.0), m, ks)
        assert (k == ks).all()


def test_veto_removes_only_below_margin_and_picks_best_survivor():
    a = np.array([[0.9, 0.8, 0.7, 0.1]]); ks = np.array([0])
    dk = np.array([[-0.2, 0.4, 2.0, 5.0]])
    assert V.veto_pick(a, dk, 0.0, ks)[0] == 1          # 0 vetoed, 1 survives at m=0
    assert V.veto_pick(a, dk, 0.5, ks)[0] == 2          # 1 (0.4 < 0.5) vetoed too
    assert V.veto_pick(a, dk, 0.4, ks)[0] == 1          # boundary: d_k == m survives (veto iff d_k < m)
    assert V.veto_pick(a, dk, 9.0, ks)[0] == 0          # all vetoed -> C1's pick


def test_calibration_tie_goes_to_smaller_m_and_cal_null():
    n = 40
    a = np.tile(np.array([0.9, 0.5] + [0.0] * 62), (n, 1)); ks = np.zeros(n, int)
    pdms = np.zeros((n, 64)); pdms[:, 0] = 0.2; pdms[:, 1] = 0.9
    tab = {m: np.ones((n, 64)) for m in V.M7}; tab["pdms"] = pdms; tab["dac"] = np.where(np.arange(64) == 0, 0.0, 1.0)[None].repeat(n, 0)
    dk = np.tile(np.concatenate([[-0.7], np.full(63, 5.0)]), (n, 1))       # proposal 0 sits at -0.7 m
    r = V.calibrate_head(a, dk, tab, ks)
    gains = {c["m"]: c["gain_pdms_x100"] for c in r["curve"]}
    assert gains[-1.0] == 0.0                                                # -0.7 >= -1.0: not vetoed, no change
    assert all(abs(gains[m] - 70.0) < 1e-9 for m in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0))
    assert r["m_star"] == -0.5                                               # seven-way tie on the gain: the SMALLEST m
    assert not r["cal_null"] and abs(r["gain_at_best_m"] - 70.0) < 1e-9
    r0 = V.calibrate_head(a, np.full(a.shape, 6.0), tab, ks)
    assert r0["cal_null"] and r0["m_star"] is None                          # a veto that never fires has gain exactly 0 -> CAL-NULL


def test_auc_basic():
    assert V.auc(np.array([3., 2., 1., 0.]), np.array([1, 1, 0, 0], bool)) == 1.0
    assert V.auc(np.array([1., 1., 1., 1.]), np.array([1, 1, 0, 0], bool)) == 0.5


# ------------------------------------------------------------------------------------------------------------------ (d) every outcome branch is reachable
def test_d_outcome_branches():
    assert outcome_of(bars_in(), **ALL_OK) == "ADOPT"
    assert outcome_of(bars_in(), **{**ALL_OK, "ctrl_t": None}) == "ADOPT-CANDIDATE"
    assert outcome_of(bars_in(), **{**ALL_OK, "families": None}) == "ADOPT-CANDIDATE"
    assert outcome_of(bars_in(), **{**ALL_OK, "families": False}) == "Unnamed"
    assert outcome_of(bars_in(), **{**ALL_OK, "ctrl_t": False}) == "INVALID"
    assert outcome_of(bars_in(), **{**ALL_OK, "controls_failed": ["CTRL-ID"]}) == "INVALID"
    assert outcome_of(bars_in(shuf_pdms=ci(0.4, 0.1, 0.7)), **ALL_OK) == "INVALID"                 # X: the shuffled head meets P1
    assert outcome_of(bars_in(), **{**ALL_OK, "cal_null": True}) == "CAL-NULL"
    assert outcome_of(bars_in(vt_pdms=ci(-0.5, -0.9, -0.1)), **ALL_OK) == "ADVERSE"               # VT CI-hi < 0
    assert outcome_of(bars_in(vt_sub={"ep": ci(-1.0, -1.5, -0.4)}), **ALL_OK) == "ADVERSE"        # G fails
    assert outcome_of(bars_in(vt_pdms=ci(0.05, -0.1, 0.2), M=ci(-0.1, -0.4, 0.3)), **ALL_OK) == "NULL"
    assert outcome_of(bars_in(vt_pdms=ci(0.1, -0.05, 0.3)), **ALL_OK) == "MECHANISM-SMALL"         # M holds, P1 fails
    assert outcome_of(bars_in(vt_pdms=ci(0.2, 0.05, 0.4)), **ALL_OK) == "MECHANISM-SMALL"         # M, P1 hold, P2 fails (0.2 < 0.2316)
    assert outcome_of(bars_in(vtr=False), **ALL_OK) == "SINGLE-SEED"                               # R fails
    assert outcome_of(bars_in(vtr_pdms=ci(-0.1, -0.3, 0.1)), **ALL_OK) == "SINGLE-SEED"
    assert outcome_of(bars_in(vt_pdms=ci(0.5, 0.3, 0.7), M=ci(0.0, -1.0, 0.5)), **ALL_OK) == "Unnamed"   # P1 holds, M fails: no committed outcome


def test_d_precedence():
    # INVALID outranks CAL-NULL and ADVERSE; CAL-NULL outranks ADVERSE
    bad = bars_in(vt_pdms=ci(-0.5, -0.9, -0.1))
    assert outcome_of(bad, **{**ALL_OK, "controls_failed": ["x"], "cal_null": True}) == "INVALID"
    assert outcome_of(bad, **{**ALL_OK, "cal_null": True}) == "CAL-NULL"
    # ADVERSE outranks NULL: P1 and M fail AND G fails
    c = bars_in(vt_pdms=ci(0.0, -0.2, 0.2), M=ci(0.0, -1, 1), vt_sub={"nc": ci(-1, -2, -0.5)})
    assert outcome_of(c, **ALL_OK) == "ADVERSE"


def test_d_bar_boundaries():
    b = V.eval_bars(bars_in(vt_pdms=ci(V.P2_BAR, 0.0, 1.0)))
    assert b["P1"] is False and b["P2"] is True                         # P1: CI-lo > 0 strictly; P2: point >= 0.2316 inclusive
    assert V.eval_bars(bars_in(vt_pdms=ci(0.23159, 0.1, 1.0)))["P2"] is False
    assert V.eval_bars(bars_in(M=ci(-1, -2, 0.0)))["M"] is False        # M: CI-hi < 0 strictly
    g = V.eval_bars(bars_in(vt_sub={"dac": ci(-1, -2, 0.0)}))
    assert g["G"] is True                                               # separated worse = CI-hi < 0 strictly (hi == 0 is not separated)
    assert V.eval_bars(bars_in(vt_sub={"dac": ci(-1, -2, -1e-9)}))["G"] is False
    assert V.eval_bars(bars_in(vtr_pdms=ci(0.0, -0.1, 0.1)))["R"] is False       # R: point > 0 strictly
    assert V.eval_bars(bars_in(shuf_pdms=ci(0.1, 0.0, 0.2)))["X_pass_P1"] is False   # shuffled arm lo == 0 does not meet P1


def test_families_check():
    ok = {arm: {f: {"c": {"point": 0.1, "lo": -0.1, "hi": 0.3}} for f in V.FAMILIES} for arm in ("VT", "VT-r")}
    assert V.check_families(ok)["ok"] is True and V.check_families(None)["ok"] is None
    bad = json.loads(json.dumps(ok)); bad["VT"]["LATERAL"]["c"] = {"point": -1, "lo": -2, "hi": -0.5}
    assert V.check_families(bad)["ok"] is False
    pos = json.loads(json.dumps(ok)); pos["VT"]["LATERAL"]["c"] = {"point": 1, "lo": 0.5, "hi": 2, "worse_if": "positive"}
    assert V.check_families(pos)["ok"] is False
    miss = json.loads(json.dumps(ok)); del miss["VT-r"]["STRATEGIC"]
    assert V.check_families(miss)["ok"] is False


# ------------------------------------------------------------------------------------------------------------------ real table fixtures
@pytest.fixture(scope="module")
def R():
    import time
    t0 = time.time()
    while V.commit_free_mb() < 6000:                      # the dev box's commit gate (as p5c_prop_score.wait_commit): wait, never load below it
        if time.time() - t0 > 1500:
            pytest.skip("free commit < 6000 MB for 25 min: not loading the 80 MB navtest table")
        time.sleep(15)
    agg32, Boot, _ = V.load_pk(PK)
    ix, logs, tab_all, shas, probs = V.load_table(TABLE)
    z = np.load(TRK, allow_pickle=False)
    tok = z["token"].astype(str)
    ti = np.array([ix[t] for t in tok])
    tab = {m: tab_all[m][ti] for m in V.M7}
    A = agg32(z["s_plan"]); ks = z["k_ship"].astype(int)
    return dict(tok=tok, logs=logs[ti], tab=tab, A=A, ks=ks, S=z["s_plan"], Boot=Boot, probs=probs, shas=shas, ix=ix, tab_all=tab_all)


@real
def test_real_table_sha_pins_and_ctrl_id(R):
    assert not R["probs"], R["probs"]
    n = len(R["tok"]); ar = np.arange(n)
    assert n == 12146 and len(set(R["logs"])) == 136
    assert (R["A"].argmax(1) == R["ks"]).all()
    assert abs(R["tab"]["pdms"][ar, R["ks"]].mean() * 100 - V.C1_PDMS) <= V.C1_PDMS_TOL


@real
def test_a_all_drivable_field_delta_exactly_zero_real(R):
    ar = np.arange(len(R["tok"]))
    for m in V.GRID:
        k = V.veto_pick(R["A"], np.full(R["A"].shape, 6.0), m, R["ks"])
        assert (k != R["ks"]).sum() == 0
        assert float((R["tab"]["pdms"][ar, k] - R["tab"]["pdms"][ar, R["ks"]]).sum()) == 0.0


@real
def test_c_always_veto_real(R):
    k = V.veto_pick(R["A"], np.full(R["A"].shape, -50.0), 0.0, R["ks"])
    assert (k == R["ks"]).all()


@real
def test_b_true_tracked_dac_field_reproduces_filter_oracle(R):
    """The reader, fed the TRUE tracked DAC as a +/-1 field at m = 0, equals vetodiag/FILTER_ORACLE.json F_dac to 1e-9 (a banked number: +1.532 [1.147, 1.974])."""
    F = json.load(open(ORACLE))["F_dac"]
    k = V.veto_pick(R["A"], V.dk_from_tracked_dac(R["tab"]["dac"]), 0.0, R["ks"])
    assert int((k != R["ks"]).sum()) == F["n_switched"] == 199
    bt = R["Boot"](R["logs"], n_boot=10000, seed=20260927)
    for m in ("pdms", "nc", "dac", "ttc", "ep", "c"):
        got = bt.ci(V._delta(R["tab"][m], k, R["ks"]))
        for q in ("point", "lo", "hi"):
            assert abs(got[q] - F[m][q]) < 1e-9, (m, q, got[q], F[m][q])
    d = V._delta(R["tab"]["pdms"], k, R["ks"])
    assert int((d > 0).sum()) == F["n_switched_pdms_better"] == 198 and int((d < 0).sum()) == F["n_switched_pdms_worse"] == 0
    assert abs(bt.ci(d)["point"] - 1.5322694578324927) < 1e-9


@real
def test_cli_end_to_end_calibrate_read_confirm(R, tmp_path):
    """The three sub-commands end to end on the real table with ORACLE fields standing in for the heads (no head exists): the arms VT = VT-r = O-DAC-T; VT-SHUF = a token-deranged field."""
    n = len(R["tok"]); rng = np.random.default_rng(1)
    orc = V.dk_from_tracked_dac(R["tab"]["dac"])
    dump = dict(token=R["tok"], s_plan=R["S"], k_ship=R["ks"], dk_HT_s0=orc, dk_HT_s1=orc.copy(), dk_HT_SHUF=orc[rng.permutation(n)])
    np.savez(tmp_path / "nt.npz", **dump)
    np.savez(tmp_path / "nt2.npz", **{k: v[:100] for k, v in dump.items()})
    # HC40 stand-in: calibrate on 2 synthetic labelled "logs" cut from the real arrays (labels = the real table rows), oracle fields
    for i, (lo, hi) in enumerate(((0, 600), (600, 1200))):
        np.savez(tmp_path / f"logA{i}.lab.npz", token=R["tok"][lo:hi], **{"tr_" + m: R["tab"][m][lo:hi] for m in V.M7})
    np.savez(tmp_path / "hc.npz", token=R["tok"][:1200], s_plan=R["S"][:1200], k_ship=R["ks"][:1200], dk_HT_s0=orc[:1200], dk_HT_s1=orc[:1200], dk_HT_SHUF=orc[:1200][::-1], dk_HT_s0_flip=-orc[:1200])
    (tmp_path / "iou.json").write_text(json.dumps({"HT_s0": 0.78, "HT_s1": 0.78, "HT_SHUF": 0.44}))
    assert V.main(["calibrate", "--pk", PK, "--dump", str(tmp_path / "hc.npz"), "--lab-glob", str(tmp_path / "*.lab.npz"), "--out", str(tmp_path / "cal.json"), "--iou-json", str(tmp_path / "iou.json")]) == 0
    cal = json.load(open(tmp_path / "cal.json"))
    assert cal["heads"]["HT_s0"]["m_star"] == -0.5 and cal["controls"]["C-FLIP"]["pass"] is True and cal["controls"]["C-SHUF"]["pass"] is True
    (tmp_path / "g0.json").write_text(json.dumps({"n_match": 777344, "n_total": 777344, "second_dump_bit_identical": True}))
    out = tmp_path / "out"
    assert V.main(["read", "--pk", PK, "--dump", str(tmp_path / "nt.npz"), "--table", TABLE, "--cal", str(tmp_path / "cal.json"), "--trk-npz", TRK, "--out-dir", str(out),
                   "--g0-json", str(tmp_path / "g0.json"), "--dump2", str(tmp_path / "nt2.npz"), "--no-mem-check"]) == 0
    rd = json.load(open(out / "READ_VDT.json"))
    assert rd["controls_failed"] == [] and rd["controls_pending"] == [], (rd["controls_failed"], rd["controls_pending"])
    assert abs(rd["arms"]["VT"]["delta_vs_C1"]["pdms"]["point"] - 1.5322694578324927) < 1e-9
    assert rd["arms"]["VT"]["delta_vs_C1"]["pdms"]["point"] == rd["arms"]["O-DAC-T"]["delta_vs_C1"]["pdms"]["point"]
    assert rd["bars"]["P1"] and rd["bars"]["P2"] and rd["bars"]["M"] and rd["bars"]["G"] and rd["bars"]["R"]
    assert rd["K_ledger"]["k_increment"] == 4
    assert rd["outcome"]["outcome"] in ("ADOPT-CANDIDATE", "INVALID")           # INVALID only if the deranged field happened to meet P1 (it must not)
    assert rd["outcome"]["outcome"] == "ADOPT-CANDIDATE" and rd["status"].startswith("PROVISIONAL")
    # confirm: the harness CSV = the table at VT's picks -> CTRL-T passes; families supplied -> ADOPT
    pk = np.load(out / "vdt_picks.npz")
    ar = np.arange(n)
    with open(tmp_path / "vt.csv", "w") as f:
        f.write("token,score,no_at_fault_collisions\n")
        for i, t in enumerate(R["tok"]):
            f.write(f"{t},{float(R['tab']['pdms'][i, pk['VT'][i]])!r},{float(R['tab']['nc'][i, pk['VT'][i]])!r}\n")
        f.write("average,0.9,0.9\n")
    fam = {arm: {fm: {"c": {"point": 0.1, "lo": -0.1, "hi": 0.3}} for fm in V.FAMILIES} for arm in ("VT", "VT-r")}
    (tmp_path / "fam.json").write_text(json.dumps(fam))
    assert V.main(["confirm", "--read", str(out / "READ_VDT.json"), "--table", TABLE, "--csv-vt", str(tmp_path / "vt.csv"), "--families", str(tmp_path / "fam.json"), "--out-dir", str(out)]) == 0
    vd = json.load(open(out / "VERDICT_VDT.json"))
    assert vd["controls"]["CTRL-T"]["pass"] is True and vd["outcome"]["outcome"] == "ADOPT" and vd["status"] == "FINAL"
    # a one-token CSV corruption (> 1e-6) fails CTRL-T -> INVALID
    txt = open(tmp_path / "vt.csv").read().split("\n"); a1 = txt[5].split(","); a1[1] = repr(float(a1[1]) + 1e-5); txt[5] = ",".join(a1)
    open(tmp_path / "vt_bad.csv", "w").write("\n".join(txt))
    assert V.main(["confirm", "--read", str(out / "READ_VDT.json"), "--table", TABLE, "--csv-vt", str(tmp_path / "vt_bad.csv"), "--families", str(tmp_path / "fam.json"), "--out-dir", str(tmp_path / "out_bad")]) == 0
    assert json.load(open(tmp_path / "out_bad" / "VERDICT_VDT.json"))["outcome"]["outcome"] == "INVALID"


# ------------------------------------------------------------------------------------------------------------------ (e) mutations: breaking one bar comparison must turn the suite RED
MUTATIONS = [
    ('P1 = vt["pdms"]["lo"] > 0', 'P1 = vt["pdms"]["lo"] >= 0'),                          # P1 strictness
    ("P2 = vt[\"pdms\"][\"point\"] >= P2_BAR", "P2 = vt[\"pdms\"][\"point\"] > P2_BAR"),       # P2 inclusivity
    ('Mb = c["M"]["hi"] < 0', 'Mb = c["M"]["hi"] <= 0'),                                  # M strictness
    ('adverse_vt = [k for k in JUDGED if vt[k]["hi"] < 0]', 'adverse_vt = [k for k in JUDGED if vt[k]["hi"] <= 0]'),   # G strictness
    ('out["X_pass_P1"] = None if shuf is None else bool(shuf["pdms"]["lo"] > 0)', 'out["X_pass_P1"] = None if shuf is None else bool(shuf["pdms"]["lo"] >= 0)'),
    ("surv = np.asarray(dk) >= m", "surv = np.asarray(dk) > m"),                           # the veto margin boundary
    ('if controls_failed or ctrl_t is False or b.get("X_pass_P1"):', 'if controls_failed or ctrl_t is False:'),   # drop INVALID-on-X
]


@pytest.mark.parametrize("old,new", MUTATIONS)
def test_e_mutation_turns_suite_red(old, new, tmp_path):
    if os.environ.get("VDT_READ_MODULE"):
        pytest.skip("inside a mutated run")
    src = Path(MOD).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"mutation anchor not unique/present: {old}"
    mut = tmp_path / "vdt_read_mut.py"
    mut.write_text(src.replace(old, new), encoding="utf-8")
    env = {**os.environ, "VDT_READ_MODULE": str(mut), "VDT_PK": PK, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(Path(__file__)), "-k", "not mutation and not real and not cli and not oracle and not table and not all_drivable_field_delta and not always_veto_real"],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0, "the mutation survived: the suite stayed GREEN\n" + r.stdout[-800:]
