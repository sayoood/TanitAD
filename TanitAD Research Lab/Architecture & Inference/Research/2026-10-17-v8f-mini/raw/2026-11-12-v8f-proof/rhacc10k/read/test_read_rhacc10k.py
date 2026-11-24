"""Tests of read_rhacc10k.py on SYNTHETIC fixtures written into pytest tmp folders only (no real data is read; the landed engines are not run: their result files are synthetic)."""
import copy
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_rhacc10k as RR  # noqa: E402

SEEDS = (0, 1, 2)
NFILES = 3
CELLS_HW = ("U", "C", "F")
FAM_CELLS = ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m", "LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps", "TACTICAL:ade_tac_m"]
QUEUE = RR.load_queue(RR.DEFAULT_QUEUE)


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


def hwcell(d=0.0, verdict="NOT-SEPARATED", lo=-0.01, hi=0.01):
    return {"D": d, "A_lo": lo, "A_hi": hi, "d_s": [d] * 3, "verdict": verdict}


def hw_pair(f=None, u=None):
    cells = {c: hwcell() for c in CELLS_HW}
    if f is not None:
        cells["F"] = f
    if u is not None:
        cells["U"] = u
    return {"cells": cells}


def q3_pair(over=None):
    fam = {c: hwcell() for c in FAM_CELLS}
    for k, v in (over or {}).items():
        fam[k] = v
    return {"families": fam}


def strat_pair(verds=("NOT-SEPARATED",) * 3):
    pos = [{"arc_m": arc, "mean": 0.0, "lo": -1.0, "hi": 1.0, "verdict": v, "n_per_seed": [5, 5, 5]} for arc, v in zip((30.0, 50.0, 80.0), verds)]
    return {"pos_30_50_80m_third_tick": pos, "reading": RR.strat_reading(pos)}


def build(tmp, mutate=None):
    """a complete synthetic pull, inputs, queue-conformant argv files and the four engine result files; returns the argument list"""
    tmp = str(tmp)
    pull, rd = os.path.join(tmp, "pull"), os.path.join(tmp, "read")
    inputs = {"inputs/%s/ckpt_last.pt" % k: {"sha256": v} for k, v in QUEUE.CKPT_SHA.items()}
    wj(os.path.join(tmp, "INPUTS.json"), inputs)
    wj(os.path.join(tmp, "grid_p30.json"), {"order": ["c%d" % i for i in range(NFILES)], "windows": {}})
    rows = {r["arm"]: r["argv"] for r in QUEUE.build_runs()}
    rng = np.random.default_rng(0)
    for m in SEEDS:
        for arm in RR.ARMS:
            key = RR.key_of(arm, m)
            base = os.path.join(pull, key)
            wj(os.path.join(base, "argv.json"), {"arm": key, "argv": rows[key], "start": "t"})
            wj(os.path.join(base, "RUN_DONE.json"), {"arm": key, "rc": 0, "dump_files": NFILES})
            os.makedirs(os.path.join(base, "ticks"), exist_ok=True)
            if arm not in ("H_R0K", "H_R0A"):
                np.savez(os.path.join(base, "ticks", "rule_log_%s.npz" % key), ci=np.zeros(2))
            dd = os.path.join(base, "dump_%s_m10_causal_heads" % key)
            os.makedirs(dd)
            for i in range(NFILES):
                np.savez(os.path.join(dd, "ep%03d.npz" % i), cl=rng.normal(size=(4, 30, 2)))
    hw = {"controls": {"ALL_PASS": True}, "counts": {"PASS": True, "n_windows_total": 2399, "n_lead_windows": 642},
          "levels": {arm: {"cells": {c: {"per_seed": [0.0391 if c == "U" else 0.0, 0.04, 0.04]} for c in CELLS_HW}} for arm in RR.ARMS},
          "pairs": {"%s_minus_%s" % (A, B): hw_pair() for A, B in RR.PAIRS.values()}}
    q3 = {"controls": {"ALL_PASS": True}, "pairs": {"%s_minus_%s" % (A, B): q3_pair() for A, B in RR.PAIRS.values()}}
    st = {"ALL_PASS": True, "pairs": {nm: strat_pair() for nm in RR.PAIRS}}
    c5 = {"ALL_PASS": True, "windows": 1271, "levels": {}, "pair": {"headway": hw_pair()["cells"], "family": q3_pair()["families"]}, "strategic": {"pairs": {"C5": strat_pair()}}}
    state = {"hw": hw, "q3": q3, "st": st, "c5": c5, "inputs": inputs}
    if mutate:
        mutate(state, pull, tmp)
    wj(os.path.join(rd, "hw", "RESULT_HEADWAY.json"), state["hw"])
    wj(os.path.join(rd, "q3", "RESULT_Q3.json"), state["q3"])
    wj(os.path.join(rd, "strat", "STRAT_READ.json"), state["st"])
    wj(os.path.join(rd, "c5", "C5_READ.json"), state["c5"])
    wj(os.path.join(tmp, "INPUTS.json"), state["inputs"])
    out = os.path.join(tmp, "out", "R.json")
    argv = ["--pull", pull, "--read-dir", rd, "--out", out, "--skip-engines", "--expect-files", str(NFILES), "--inputs", os.path.join(tmp, "INPUTS.json"),
            "--grid-p30", os.path.join(tmp, "grid_p30.json")]
    return argv, out


def run(tmp, mutate=None, extra=()):
    argv, out = build(tmp, mutate)
    rc = RR.main(argv + list(extra))
    return rc, json.load(io.open(out, encoding="utf-8"))


def test_all_pass_names(tmp_path):
    rc, r = run(tmp_path)
    assert rc == 0 and r["controls_all_pass"] is True
    n = r["names"]
    assert n["C1"] == "RHACC10K-C1-PASS"
    assert n["C2_LABEL"] == "RHACC10K-C2-LABEL-PASS" and n["C2_READER"] == "RHACC10K-C2-READER-PASS"
    assert n["C3"] == "RHACC10K-C3-PASS" and n["C5"] == "RHACC10K-C5-PASS"
    assert n["OVERALL"] == "RHACC10K-SCALE-CANDIDATE"
    assert r["spread_arm"] == "U_RLcK" and r["checks"]["OVERALL"]["failed_checks"] == []
    assert set(r["pairs"]["C1"]["families"]) == set(RR.FAMS)


def k1_missing_done(st, pull, tmp):
    os.remove(os.path.join(pull, "H_RLcA_s1", "RUN_DONE.json"))


def k1_short_dump(st, pull, tmp):
    os.remove(os.path.join(pull, "U_RQ50K_s2", "dump_U_RQ50K_s2_m10_causal_heads", "ep002.npz"))


def k1_nonfinite(st, pull, tmp):
    p = os.path.join(pull, "H_R0K_s0", "dump_H_R0K_s0_m10_causal_heads", "ep000.npz")
    cl = np.load(p)["cl"]
    cl[0, 0, 0] = np.nan
    np.savez(p, cl=cl)


def k2_hash(st, pull, tmp):
    st["inputs"]["inputs/RH_s1/ckpt_last.pt"]["sha256"] = "0" * 64


def k3_argv(st, pull, tmp):
    p = os.path.join(pull, "H_RLcA_s0", "argv.json")
    d = json.load(io.open(p, encoding="utf-8"))
    d["argv"] = d["argv"] + ["--acc-gain", "2"]
    wj(p, d)


def k3_wrapper_g(st, pull, tmp):
    for m in SEEDS:
        p = os.path.join(pull, "H_RLcA_P30_s%d" % m, "argv.json")
        d = json.load(io.open(p, encoding="utf-8"))
        d["argv"][1] = d["argv"][1].replace("rzgfhq", "rzgfh")
        wj(p, d)


def test_k3_g_run_with_parent_wrapper_fails(tmp_path):
    rc, r = run(tmp_path, k3_wrapper_g)
    c = r["controls"]["k3_argv"]
    assert rc == 2 and c["PASS"] is False and r["names"] is None
    assert c["per_run"]["H_RLcA_P30_s0"]["wrapper_ok"] is False and c["per_run"]["H_RLcA_s0"]["wrapper_ok"] is True


def test_k3_rows_g_h_use_pre_roll_wrapper(tmp_path):
    rc, r = run(tmp_path)
    for arm in ("H_RLcA_P30", "H_RLcA_P0"):
        assert r["controls"]["k3_argv"]["per_run"][arm + "_s0"]["wrapper_in_argv"].endswith("rzgfhq.py")
    assert r["controls"]["k3_argv"]["per_run"]["H_RQ50A_s0"]["wrapper_in_argv"].endswith("rzgfh.py")


def k4_env(st, pull, tmp):
    st["hw"]["levels"]["U_RLcK"]["cells"]["U"]["per_seed"][0] = 0.0391 + 0.02


def k5_wrapper(st, pull, tmp):
    wj(os.path.join(tmp, "other_wrapper.py"), {"x": 1})


def k6_same_sha(st, pull, tmp):
    st["inputs"]["inputs/UE_s0/ckpt_last.pt"]["sha256"] = st["inputs"]["inputs/RH_s0/ckpt_last.pt"]["sha256"]


def k6_same_dumps(st, pull, tmp):
    import shutil
    for m in SEEDS:
        a, e = (os.path.join(pull, "%s_s%d" % (x, m), "dump_%s_s%d_m10_causal_heads" % (x, m)) for x in ("H_R0K", "U_RLcK"))
        for f in os.listdir(a):
            shutil.copyfile(os.path.join(a, f), os.path.join(e, f))


def e_hw(st, pull, tmp):
    st["hw"]["controls"]["ALL_PASS"] = False


def e_q3(st, pull, tmp):
    st["q3"]["controls"]["ALL_PASS"] = False


def e_strat(st, pull, tmp):
    st["st"]["ALL_PASS"] = False


def e_c5(st, pull, tmp):
    st["c5"]["ALL_PASS"] = False


@pytest.mark.parametrize("mut,ctrl", [(k1_missing_done, "k1_complete"), (k1_short_dump, "k1_complete"), (k1_nonfinite, "k1_complete"), (k2_hash, "k2_models"), (k3_argv, "k3_argv"),
                                      (k4_env, "k4_environment"), (k6_same_sha, "k6_identity"), (k6_same_dumps, "k6_identity"), (e_hw, "e_reader_controls"), (e_q3, "e_reader_controls"),
                                      (e_strat, "e_reader_controls"), (e_c5, "e_reader_controls")])
def test_failed_control_withholds_names(tmp_path, mut, ctrl):
    rc, r = run(tmp_path, mut)
    assert rc == 2 and r["controls_all_pass"] is False
    assert r["controls"][ctrl]["PASS"] is False
    assert r["names"] is None and r["checks"] is None and r["pairs"] is None and r["levels"] is None and r["c5"] is None


def test_k5_wrong_wrapper_withholds(tmp_path):
    argv, out = build(tmp_path, k5_wrapper)
    rc = RR.main(argv + ["--wrapper", os.path.join(str(tmp_path), "other_wrapper.py")])
    r = json.load(io.open(out, encoding="utf-8"))
    assert rc == 2 and r["controls"]["k5_reader_input"]["PASS"] is False and r["names"] is None


def test_k5_finding_is_recorded_with_file_lines():
    f = RR.K5_FINDING
    assert f["reader_reads_model_latent"] is False
    assert any("rzgfh.py:188-192" in e for e in f["evidence"]) and any("rzgfh.py:73" in e for e in f["evidence"])


def test_k5_latent_reads_label_form_only(tmp_path, monkeypatch):
    monkeypatch.setitem(RR.K5_FINDING, "reader_reads_model_latent", True)
    rc, r = run(tmp_path)
    assert rc == 0 and r["controls"]["k5_reader_input"]["reader_input_is_eval_cache_tokens"] is False
    assert r["names"]["C2_READER"] == "RHACC10K-C2-READER-NOT-READ" and r["names"]["C2_LABEL"] == "RHACC10K-C2-LABEL-PASS"
    assert r["names"]["OVERALL"] == "RHACC10K-SCALE-CANDIDATE"


def sep(d, verdict):
    return hwcell(d, verdict, d - 0.01, d + 0.01)


def mut_c1_f_higher(st, pull, tmp):
    st["hw"]["pairs"]["H_R0A_minus_H_R0K"]["cells"]["F"] = sep(0.05, "WORSE")
    st["hw"]["pairs"]["H_R0A_minus_H_R0K"]["cells"]["F"]["d_s"] = [0.05, 0.04, 0.06]


def mut_c1_f_lower(st, pull, tmp):
    c = sep(-0.05, "BETTER")
    c["d_s"] = [-0.05, -0.04, -0.06]
    st["hw"]["pairs"]["H_R0A_minus_H_R0K"]["cells"]["F"] = c


def test_mutation_flip_sign_changes_named_outcome(tmp_path):
    rc1, r1 = run(os.path.join(str(tmp_path), "a"), mut_c1_f_lower)
    rc2, r2 = run(os.path.join(str(tmp_path), "b"), mut_c1_f_higher)
    assert rc1 == 0 and rc2 == 0
    assert r1["names"]["C1"] == "RHACC10K-C1-PASS"
    assert r2["names"]["C1"] == "RHACC10K-C1-FAIL"
    assert r1["names"]["OVERALL"] == "RHACC10K-SCALE-CANDIDATE" and r2["names"]["OVERALL"] == "RHACC10K-NOT-YET"
    assert r2["checks"]["OVERALL"]["failed_checks"][0] == "C1"


def worse_family(over):
    def f(st, pull, tmp):
        for k in ("H_RLcA_minus_U_RLcK",):
            for c in ("LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"):
                v = sep(0.3 * over, "WORSE" if over > 0 else "BETTER")
                v["d_s"] = [0.3 * over] * 3
                st["q3"]["pairs"][k]["families"][c] = v
    return f


def test_mutation_family_sign_c2(tmp_path):
    _, rw = run(os.path.join(str(tmp_path), "w"), worse_family(+1))
    _, rb = run(os.path.join(str(tmp_path), "b"), worse_family(-1))
    assert rw["names"]["C2_LABEL"] == "RHACC10K-C2-LABEL-FAIL" and rb["names"]["C2_LABEL"] == "RHACC10K-C2-LABEL-PASS"
    assert rw["checks"]["C2_LABEL"]["families_WORSE"] == ["LATERAL"]


def one_cell_worse(st, pull, tmp):
    v = sep(0.3, "WORSE")
    v["d_s"] = [0.3] * 3
    st["q3"]["pairs"]["H_RLcA_minus_U_RLcK"]["families"]["LATERAL:heading_mae_deg"] = v


def test_one_worse_cell_is_mixed_not_worse(tmp_path):
    _, r = run(tmp_path, one_cell_worse)
    assert r["pairs"]["C2_LABEL"]["families"]["LATERAL"] == "MIXED" and r["names"]["C2_LABEL"] == "RHACC10K-C2-LABEL-PASS"


def c3_better(st, pull, tmp):
    st["st"]["pairs"]["C2_LABEL"] = strat_pair(("BETTER",) * 3)


def c3_worse(st, pull, tmp):
    st["st"]["pairs"]["C2_LABEL"] = strat_pair(("WORSE",) * 3)


def c3_ns_but_c1_fails(st, pull, tmp):
    mut_c1_f_higher(st, pull, tmp)


def test_c3_rules(tmp_path):
    _, rb = run(os.path.join(str(tmp_path), "b"), c3_better)
    _, rw = run(os.path.join(str(tmp_path), "w"), c3_worse)
    _, rn = run(os.path.join(str(tmp_path), "n"), c3_ns_but_c1_fails)
    assert rb["names"]["C3"] == "RHACC10K-C3-PASS" and rw["names"]["C3"] == "RHACC10K-C3-FAIL"
    assert rn["names"]["C1"] == "RHACC10K-C1-FAIL" and rn["names"]["C3"] == "RHACC10K-C3-FAIL"


def c5_u_lower(st, pull, tmp):
    c = sep(-0.02, "BETTER")
    c["d_s"] = [-0.02, -0.03, -0.01]
    st["c5"]["pair"]["headway"]["U"] = c


def c5_u_higher(st, pull, tmp):
    c = sep(0.02, "WORSE")
    c["d_s"] = [0.02, 0.03, 0.01]
    st["c5"]["pair"]["headway"]["U"] = c


def c5_strategic_mixed(st, pull, tmp):
    st["c5"]["strategic"]["pairs"]["C5"] = strat_pair(("WORSE", "NOT-SEPARATED", "NOT-SEPARATED"))


def test_c5_rules_and_overall(tmp_path):
    _, rl = run(os.path.join(str(tmp_path), "l"), c5_u_lower)
    _, rh = run(os.path.join(str(tmp_path), "h"), c5_u_higher)
    _, rm = run(os.path.join(str(tmp_path), "m"), c5_strategic_mixed)
    assert rl["names"]["C5"] == "RHACC10K-C5-PASS" and rl["names"]["OVERALL"] == "RHACC10K-SCALE-CANDIDATE"
    assert rh["names"]["C5"] == "RHACC10K-C5-FAIL" and rh["names"]["OVERALL"] == "RHACC10K-NOT-YET" and rh["checks"]["OVERALL"]["failed_checks"] == ["C5"]
    assert rm["names"]["C5"] == "RHACC10K-C5-FAIL"


def test_no_banned_words_in_files():
    banned = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lo" + "nger", "closed " + "loop"]
    for fn in ("read_rhacc10k.py", "test_read_rhacc10k.py"):
        t = io.open(os.path.join(HERE, fn), encoding="utf-8").read().lower()
        for w in banned:
            assert w not in t, (fn, w)
