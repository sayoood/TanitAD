"""Tests of read_rhk10k.py on SYNTHETIC fixtures written into pytest tmp folders only (no real data is read; the landed engines are not run: their result files are synthetic)."""
import hashlib
import io
import json
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_rhk10k as RK  # noqa: E402

B = RK.B
SEEDS = (0, 1, 2)
NFILES = 3
CELLS_HW = ("U", "C", "F")
FAM_CELLS = ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m", "LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps", "TACTICAL:ade_tac_m"]
QNEW = B.load_queue(RK.DEFAULT_QUEUE)
QPART = B.load_queue(RK.DEFAULT_PARTNER_QUEUE)
PAIR_TAGS = ["%s_minus_%s" % (x, y) for x, y in RK.PAIRS.values()]
R3_TAG = "%s_minus_%s" % RK.R3_PAIR


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


def hwcell(d=0.0, verdict="NOT-SEPARATED", lo=-0.01, hi=0.01):
    return {"D": d, "A_lo": lo, "A_hi": hi, "d_s": [d] * 3, "verdict": verdict}


def sep(d, verdict):
    c = hwcell(d, verdict, d - 0.01, d + 0.01)
    return c


def hw_pair():
    return {"cells": {c: hwcell() for c in CELLS_HW}}


def q3_pair(over=None):
    fam = {c: hwcell() for c in FAM_CELLS}
    fam.update(over or {})
    return {"families": fam}


def strat_pair(verds=("NOT-SEPARATED",) * 3):
    pos = [{"arc_m": arc, "mean": 0.0, "lo": -1.0, "hi": 1.0, "verdict": v, "n_per_seed": [5, 5, 5]} for arc, v in zip((30.0, 50.0, 80.0), verds)]
    return {"pos_30_50_80m_third_tick": pos, "reading": B.strat_reading(pos)}


def write_run(pull, arm, m, rows, rng, shared_cl=None):
    key = RK.key_of(arm, m)
    base = os.path.join(pull, key)
    wj(os.path.join(base, "argv.json"), {"arm": key, "argv": rows[key], "start": "t"})
    wj(os.path.join(base, "RUN_DONE.json"), {"arm": key, "rc": 0})
    os.makedirs(os.path.join(base, "ticks"), exist_ok=True)
    np.savez(os.path.join(base, "ticks", "rule_log_%s.npz" % key), ci=np.zeros(2))
    if arm in RK.P_ARMS:
        wj(os.path.join(base, "ticks", "preroll_log_%s.json" % key), {"x": 1})
        np.savez(os.path.join(base, "ticks", "preroll_offsets_%s.npz" % key), dpos=np.zeros(2))
    dd = os.path.join(base, "dump_%s_m10_causal_heads" % key)
    os.makedirs(dd)
    for i in range(NFILES):
        np.savez(os.path.join(dd, "ep%03d.npz" % i), cl=rng.normal(size=(4, 30, 2)), clip_index=np.full(4, i), ws=np.arange(4) + 10 * i, eid=np.full(4, i))


def build(tmp, mutate=None):
    tmp = str(tmp)
    pull, rd = os.path.join(tmp, "pull"), os.path.join(tmp, "read")
    inputs = {"inputs/%s/ckpt_last.pt" % k: {"sha256": v} for k, v in QNEW.CKPT_SHA.items()}
    wj(os.path.join(tmp, "grid_p30.json"), {"order": ["c%d" % i for i in range(NFILES)], "windows": {}})
    rows = {r["arm"]: r["argv"] for r in QNEW.build_runs()}
    rows.update({r["arm"]: r["argv"] for r in QPART.build_runs() if r["arm"].split("_s")[0] in RK.PARTNER_ARMS})
    rng = np.random.default_rng(0)
    for m in SEEDS:
        for arm in RK.ALL_ARMS:
            write_run(pull, arm, m, rows, rng)
    lines = []
    for m in SEEDS:
        for arm in RK.PARTNER_ARMS:
            base = os.path.join(pull, RK.key_of(arm, m))
            for root, _, fs in os.walk(base):
                for f in sorted(fs):
                    fp = os.path.join(root, f)
                    lines.append("%s *%s" % (hashlib.sha256(open(fp, "rb").read()).hexdigest(), os.path.relpath(fp, pull).replace("\\", "/")))
    with io.open(os.path.join(tmp, "SHA.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    hw = {"controls": {"ALL_PASS": True}, "counts": {"PASS": True, "n_windows_total": 2399, "n_lead_windows": 642},
          "levels": {arm: {"cells": {c: {"per_seed": [0.04] * 3} for c in CELLS_HW}} for arm in RK.FRESH_ARMS},
          "pairs": {t: hw_pair() for t in PAIR_TAGS}}
    q3 = {"controls": {"ALL_PASS": True}, "pairs": {t: q3_pair() for t in PAIR_TAGS}}
    st = {"ALL_PASS": True, "pairs": {nm: strat_pair() for nm in RK.PAIRS}}
    r3 = {"ALL_PASS": True, "windows": 1271, "levels": {}, "arms": list(RK.P_ARMS), "pair": {"headway": hw_pair()["cells"], "family": q3_pair()["families"]},
          "strategic": {"pairs": {"R3": strat_pair()}}}
    state = {"hw": hw, "q3": q3, "st": st, "r3": r3, "inputs": inputs}
    if mutate:
        mutate(state, pull, tmp)
    wj(os.path.join(rd, "hw", "RESULT_HEADWAY.json"), state["hw"])
    wj(os.path.join(rd, "q3", "RESULT_Q3.json"), state["q3"])
    wj(os.path.join(rd, "strat", "STRAT_READ.json"), state["st"])
    wj(os.path.join(rd, "r3", "R3_READ.json"), state["r3"])
    wj(os.path.join(tmp, "INPUTS.json"), state["inputs"])
    out = os.path.join(tmp, "out", "R.json")
    argv = ["--pull", pull, "--read-dir", rd, "--out", out, "--skip-engines", "--expect-files", str(NFILES), "--inputs", os.path.join(tmp, "INPUTS.json"),
            "--grid-p30", os.path.join(tmp, "grid_p30.json"), "--partners-sha", os.path.join(tmp, "SHA.txt")]
    return argv, out


def run(tmp, mutate=None):
    argv, out = build(tmp, mutate)
    rc = RK.main(argv)
    return rc, json.load(io.open(out, encoding="utf-8"))


def test_all_pass_names(tmp_path):
    rc, r = run(tmp_path)
    assert rc == 0 and r["controls_all_pass"] is True
    assert r["names"] == {"FRESH": "RHK10K-FRESH-EQUAL", "AGED": "RHK10K-AGED-EQUAL"}
    assert r["spread_arm"] == "U_RLcK" and r["spread_arm_r3"] == "U_RLcK_P30"
    assert set(r["pairs"]) == {"R1", "R2", "R3"} and set(r["pairs"]["R3"]["families"]) == set(RK.FAMS)
    assert "candidate of record" in r["consequence"]["card_sentence"] and r["consequence"]["applied"].startswith("not stated")
    for p in ("R1", "R2", "R3"):
        assert set(r["pairs"][p]["family_cells"]) == set(FAM_CELLS)       # per-family cells reported, never pooled


# ---- controls
def q1_missing_done(st, pull, tmp):
    os.remove(os.path.join(pull, "H_RQ50K_s1", "RUN_DONE.json"))


def q1_short_dump(st, pull, tmp):
    os.remove(os.path.join(pull, "U_RLcK_P30_s2", "dump_U_RLcK_P30_s2_m10_causal_heads", "ep002.npz"))


def q1_nonfinite(st, pull, tmp):
    p = os.path.join(pull, "H_RLcK_s0", "dump_H_RLcK_s0_m10_causal_heads", "ep000.npz")
    d = dict(np.load(p))
    d["cl"][0, 0, 0] = np.nan
    np.savez(p, **d)


def q1_no_preroll_log(st, pull, tmp):
    os.remove(os.path.join(pull, "H_RLcK_P30_s0", "ticks", "preroll_log_H_RLcK_P30_s0.json"))


def q2_hash(st, pull, tmp):
    st["inputs"]["inputs/RH_s1/ckpt_last.pt"]["sha256"] = "0" * 64


def q2_argv_new(st, pull, tmp):
    p = os.path.join(pull, "H_RLcK_s0", "argv.json")
    d = json.load(io.open(p, encoding="utf-8"))
    d["argv"] = d["argv"] + ["--acc-gain", "2"]
    wj(p, d)


def q2_argv_partner(st, pull, tmp):
    p = os.path.join(pull, "U_RQ50K_s1", "argv.json")
    d = json.load(io.open(p, encoding="utf-8"))
    d["argv"] = d["argv"] + ["--x", "1"]
    wj(p, d)


def q2_wrapper_pre(st, pull, tmp):
    p = os.path.join(pull, "U_RLcK_P30_s0", "argv.json")
    d = json.load(io.open(p, encoding="utf-8"))
    d["argv"][1] = d["argv"][1].replace("rzgfhq", "rzgfh")
    wj(p, d)


def q3_changed_partner(st, pull, tmp):
    p = os.path.join(pull, "U_RLcK_s2", "RUN_DONE.json")
    d = json.load(io.open(p, encoding="utf-8"))
    d["note"] = "changed"
    wj(p, d)


def q3_missing_partner_file(st, pull, tmp):
    os.remove(os.path.join(pull, "U_RQ50K_s0", "ticks", "rule_log_U_RQ50K_s0.npz"))


def q4_same_dumps(st, pull, tmp):
    import shutil
    for m in SEEDS:
        a, e = (os.path.join(pull, "%s_s%d" % (x, m), "dump_%s_s%d_m10_causal_heads" % (x, m)) for x in ("H_RLcK", "U_RLcK"))
        for f in os.listdir(a):
            shutil.copyfile(os.path.join(a, f), os.path.join(e, f))
        # the partner folder must not change its recorded hashes: restore is not needed, q3 fails too; the assertion below reads q4 only


def q4_windows_differ(st, pull, tmp):
    p = os.path.join(pull, "U_RLcK_P30_s1", "dump_U_RLcK_P30_s1_m10_causal_heads", "ep001.npz")
    d = dict(np.load(p))
    d["ws"] = d["ws"] + 1
    np.savez(p, **d)


def e_hw(st, pull, tmp):
    st["hw"]["controls"]["ALL_PASS"] = False


def e_hw_count(st, pull, tmp):
    st["hw"]["counts"]["n_windows_total"] = 2398


def e_q3(st, pull, tmp):
    st["q3"]["controls"]["ALL_PASS"] = False


def e_strat(st, pull, tmp):
    st["st"]["ALL_PASS"] = False


def e_r3(st, pull, tmp):
    st["r3"]["ALL_PASS"] = False


CASES = [(q1_missing_done, "q1_complete"), (q1_short_dump, "q1_complete"), (q1_nonfinite, "q1_complete"), (q1_no_preroll_log, "q1_complete"),
         (q2_hash, "q2_models"), (q2_argv_new, "q2_argv"), (q2_argv_partner, "q2_argv"), (q2_wrapper_pre, "q2_argv"),
         (q3_changed_partner, "q3_partners"), (q3_missing_partner_file, "q3_partners"), (q4_same_dumps, "q4_identity"), (q4_windows_differ, "q4_identity"),
         (e_hw, "e_reader_controls"), (e_hw_count, "e_reader_controls"), (e_q3, "e_reader_controls"), (e_strat, "e_reader_controls"), (e_r3, "e_reader_controls")]


@pytest.mark.parametrize("mut,ctrl", CASES, ids=[m.__name__ for m, _ in CASES])
def test_failed_control_withholds_names(tmp_path, mut, ctrl):
    rc, r = run(tmp_path, mut)
    assert rc == 2 and r["controls_all_pass"] is False
    assert r["controls"][ctrl]["PASS"] is False
    assert r["names"] is None and r["checks"] is None and r["pairs"] is None and r["levels"] is None and r["r3"] is None and r["consequence"] is None


# ---- names and mutation
def fam_over(pair_tag, d, verdict, cells=("LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps")):
    def f(st, pull, tmp):
        for c in cells:
            v = sep(d, verdict)
            v["d_s"] = [d] * 3
            if pair_tag == R3_TAG:
                st["r3"]["pair"]["family"][c] = v
            else:
                st["q3"]["pairs"][pair_tag]["families"][c] = v
    return f


def u_over(pair_tag, d, verdict):
    def f(st, pull, tmp):
        v = sep(d, verdict)
        v["d_s"] = [d, d * 1.2, d * 0.8]
        st["hw"]["pairs"][pair_tag]["cells"]["U"] = v
    return f


def test_mutation_flip_pair_sign_changes_fresh_name(tmp_path):
    t = PAIR_TAGS[0]
    _, rb = run(os.path.join(str(tmp_path), "b"), fam_over(t, -0.3, "BETTER"))
    _, rw = run(os.path.join(str(tmp_path), "w"), fam_over(t, +0.3, "WORSE"))
    assert rb["names"]["FRESH"] == "RHK10K-FRESH-BETTER" and rw["names"]["FRESH"] == "RHK10K-FRESH-WORSE"
    assert rw["checks"]["FRESH"]["R1_R2_families_WORSE"]["R1"] == ["LATERAL"]
    assert rw["consequence"]["applied"].startswith("the line candidate stays")


def test_fresh_rules(tmp_path):
    t2 = PAIR_TAGS[1]
    _, r1 = run(os.path.join(str(tmp_path), "a"), fam_over(t2, +0.3, "WORSE"))                # R2 worse alone is enough
    assert r1["names"]["FRESH"] == "RHK10K-FRESH-WORSE"
    _, r2 = run(os.path.join(str(tmp_path), "b"), u_over(PAIR_TAGS[0], 0.02, "WORSE"))        # U higher, no family separated: neither BETTER nor EQUAL
    assert r2["names"]["FRESH"] == "RHK10K-FRESH-MIXED"
    _, r3 = run(os.path.join(str(tmp_path), "c"), u_over(PAIR_TAGS[0], -0.02, "BETTER"))      # U lower alone is not EQUAL
    assert r3["names"]["FRESH"] == "RHK10K-FRESH-MIXED"
    one = lambda st, pull, tmp: (fam_over(PAIR_TAGS[0], -0.3, "BETTER")(st, pull, tmp), u_over(PAIR_TAGS[0], 0.02, "WORSE")(st, pull, tmp))  # noqa: E731
    _, r4 = run(os.path.join(str(tmp_path), "d"), one)                                        # a BETTER family with U higher
    assert r4["names"]["FRESH"] == "RHK10K-FRESH-MIXED"


def test_one_worse_cell_is_mixed_not_worse(tmp_path):
    _, r = run(tmp_path, fam_over(PAIR_TAGS[0], 0.3, "WORSE", cells=("LATERAL:heading_mae_deg",)))
    assert r["pairs"]["R1"]["families"]["LATERAL"] == "MIXED" and r["names"]["FRESH"] == "RHK10K-FRESH-MIXED"


def test_aged_rules(tmp_path):
    _, rb = run(os.path.join(str(tmp_path), "b"), fam_over(R3_TAG, -0.3, "BETTER", cells=("TACTICAL:ade_tac_m",)))
    _, rw = run(os.path.join(str(tmp_path), "w"), fam_over(R3_TAG, +0.3, "WORSE", cells=("TACTICAL:ade_tac_m",)))
    _, rl = run(os.path.join(str(tmp_path), "l"), fam_over(R3_TAG, -0.3, "BETTER"))             # LATERAL better only: not AGED-BETTER, not EQUAL, no WORSE
    assert rb["names"]["AGED"] == "RHK10K-AGED-BETTER" and rw["names"]["AGED"] == "RHK10K-AGED-WORSE" and rl["names"]["AGED"] == "RHK10K-AGED-MIXED"
    both = lambda st, pull, tmp: (fam_over(R3_TAG, -0.3, "BETTER", cells=("TACTICAL:ade_tac_m",))(st, pull, tmp),  # noqa: E731
                                  fam_over(R3_TAG, +0.3, "WORSE", cells=("LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"))(st, pull, tmp))
    _, rm = run(os.path.join(str(tmp_path), "m"), both)
    assert rm["names"]["AGED"] == "RHK10K-AGED-MIXED"


def test_consequence_rh_candidate(tmp_path):
    def f(st, pull, tmp):
        fam_over(R3_TAG, -0.3, "BETTER", cells=("TACTICAL:ade_tac_m",))(st, pull, tmp)
    _, r = run(tmp_path, f)
    assert r["names"] == {"FRESH": "RHK10K-FRESH-EQUAL", "AGED": "RHK10K-AGED-BETTER"}
    assert r["consequence"]["applied"].startswith("RH under K becomes the candidate of record")


# ---- R3 is a pair of two pre-rolled arms
def test_r3_pair_and_registration_are_two_prerolled_arms():
    assert RK.R3_PAIR == ("H_RLcK_P30", "U_RLcK_P30") and set(RK.R3_PAIR) == set(RK.P_ARMS)
    R = types.SimpleNamespace(ALL_RUNS={})
    RK.register_arms(R)
    assert len(R.ALL_RUNS) == 6
    for k, spec in R.ALL_RUNS.items():
        assert spec[2] == 30 and spec[5] == "plan" and spec[1] == "K", k


def test_r3_engine_called_with_both_prerolled_arms(tmp_path, monkeypatch):
    calls = {}

    class FakeR:
        ALL_RUNS = {}
        SEEDS = None
        key_of = None

        @staticmethod
        def engine(ns):
            return ("H", "Q3", "RH")

        @staticmethod
        def grid_keys(grid, names):
            return np.zeros((3, 2), np.int64)

        @staticmethod
        def read_grid(ns, E, gname, keys, runs, spread, pairs, dod, log):
            calls.update(runs=list(runs), spread=spread, pairs=dict(pairs))
            return {"controls": {"ALL_PASS": True}, "pairs": {"R3": {"headway": {}, "family": {}}}, "levels": {}}

    fake_n4 = types.SimpleNamespace(R=FakeR, controls_n4=lambda: {"n2": {"PASS": True}})
    monkeypatch.setitem(sys.modules, "read_longroll10k_n4", fake_n4)

    def fake_strat(a, runs, pairs, s3_arm, p_of, keys=None, log=print, waive=True):
        calls.update(s_runs=list(runs), s_pairs=dict(pairs), p_of=dict(p_of), waive=waive)
        return {"ALL_PASS": True, "pairs": {"R3": strat_pair()}}

    monkeypatch.setattr(RK, "strategic_engine_pre", fake_strat)
    grid = os.path.join(str(tmp_path), "g.json")
    wj(grid, {"order": []})
    join = os.path.join(str(tmp_path), "j.npz")
    np.savez(join, clip=np.array(["a", "b"]))
    a = RK.parse(["--pull", str(tmp_path), "--read-dir", str(tmp_path), "--out", str(tmp_path / "o.json"), "--grid-p30", grid, "--join", join])
    res = RK.r3_engine(a, log=lambda s: None)
    assert calls["runs"] == ["H_RLcK_P30", "U_RLcK_P30"] and calls["spread"] == "U_RLcK_P30" and calls["pairs"] == {"R3": ("H_RLcK_P30", "U_RLcK_P30")}
    assert calls["s_runs"] == ["H_RLcK_P30", "U_RLcK_P30"] and calls["p_of"] == {"H_RLcK_P30": 30, "U_RLcK_P30": 30} and calls["waive"] is True
    assert all(v[2] == 30 for v in FakeR.ALL_RUNS.values()) and res["ALL_PASS"] is True


def test_derived_strategic_engine_has_the_waivers():
    import inspect
    src = inspect.getsource(B.strategic_engine)
    assert src.count("s2[\"files\"])") == 1 and RK.strategic_engine_pre.__name__ == "strategic_engine_pre"
    assert "waive" in RK.strategic_engine_pre.__code__.co_varnames


def test_no_banned_words_in_files():
    banned = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lo" + "nger", "closed " + "loop"]
    for fn in ("read_rhk10k.py", "test_read_rhk10k.py"):
        t = io.open(os.path.join(HERE, fn), encoding="utf-8").read().lower()
        for w in banned:
            assert w not in t, (fn, w)
