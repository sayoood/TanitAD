"""Tests of stopperc_s1_read.py on SYNTHETIC per-window arrays and planted banks (no dumps, no torch, no GPU, no box): every outcome row of the card's table is reachable, the floor is the MAX of the two
replicate gaps (tests/run_mutation_floor_s1.py replaces it with the NULL_rep gap alone and requires this file to go RED), a missing floor / bank / UNKNOWN reuse state refuses, STRATEGIC is never a pass.
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_stopperc_s1_read.py --basetemp <existing parent>/pt
The module under test is read from env STOPPERC_READ_SRC when set (the mutation runner), else stopperc_s1_read.py beside this file."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("STOPPERC_READ_SRC", os.path.join(HERE, "stopperc_s1_read.py"))
_spec = importlib.util.spec_from_file_location("stopperc_s1_read_under_test", SRC)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
SR = S.SR
CI = SR.load_ci()
NBOOT = 200
P_CELL = SR.CELL_P[1]
N_EP, PER_EP = 40, 20
N = N_EP * PER_EP
EID = np.repeat(np.arange(N_EP), PER_EP)
IDX = np.arange(N)
OK = np.ones(N, bool)
SP = (IDX % 10 == 0)
SP_I, OTH_I = np.where(SP)[0], np.where(~SP)[0]
rng = np.random.default_rng(11)
BASE = 2.0 + rng.normal(0, 0.3, N)
U_G, U_L = rng.normal(0, 0.3, N), rng.normal(0, 0.3, N)      # fixed score noise (the same windows in every arm)


def arm(d_sp=0.0, d_all=0.0, shift=0.0, g_sep=0.0, l_sep=0.0, fpr=0.0, hit=0.0):
    """BASE tactical error moved by d_sp on stop-point windows and d_all on the others (+ shift); goal / lon scores separate the stop windows by g_sep / l_sep; the 0.5-decision fires on `hit` of SP and `fpr` of the rest."""
    ade = BASE + shift + np.where(SP, d_sp, d_all)
    gp, lp = U_G + SP * g_sep, U_L + SP * l_sep
    dec = np.zeros(N, bool)
    for pool, share in ((SP_I, hit), (OTH_I, fpr)):                # evenly spread over the pool (and so over the episodes)
        k = int(round(share * len(pool)))
        if k:
            dec[pool[np.round(np.linspace(0, len(pool) - 1, k)).astype(int)]] = True
    return {"cells": {P_CELL: ade, "TAC-LON": np.full(N, 0.8)}, "goal_stop": dec, "lon_stop": dec.copy(), "goal_score": gp, "lon_score": lp,
            "cue": {"has": np.ones(N, bool), "pos": SP.copy(), "reg_stop": SP.copy(), "goal_p": gp, "lon_p": lp}}


NULL = arm()
REP = arm(shift=0.01, g_sep=0.05, l_sep=0.05, fpr=0.01)         # gap to NULL: P 0.01, AUROC small, fpr 0.01
REP2 = arm(shift=0.05, g_sep=0.20, l_sep=0.20, fpr=0.04)        # the LARGER gap: P 0.05, AUROC large, fpr 0.04


def floors():
    return S.compute_floors_s1(REP, REP2, NULL, EID, SP, OK)


def seeds(*specs):
    fl = floors()
    return [S.read_seed_s1(CI, NULL, arm(**x), fl, EID, SP, OK, NBOOT) for x in specs]


GOOD = dict(d_sp=-0.5, g_sep=0.8, l_sep=0.8)                      # P gain, G-AUROC gain, no fpr change, no NR cost


def test_floor_is_max_of_the_two_replicate_gaps():
    f = floors()
    assert f["cells"][("sp", P_CELL)] == pytest.approx(0.05)    # NOT the NULL_rep gap 0.01
    assert f["cells"][("all", P_CELL)] == pytest.approx(0.05)
    d = f["detail"]["sp|" + P_CELL]
    assert d["gap_rep"] == pytest.approx(0.01) and d["gap_rep2"] == pytest.approx(0.05) and d["floor"] == pytest.approx(0.05)
    assert f["fpr"] == pytest.approx(f["detail"]["fpr"]["gap_rep2"]) and f["fpr"] > f["detail"]["fpr"]["gap_rep"]
    for k in ("auroc_goal", "auroc_lon"):
        assert f[k] == pytest.approx(f["detail"][k]["gap_rep2"]) and f[k] > f["detail"][k]["gap_rep"]


def test_delta_between_the_two_floors_is_not_a_gain():
    """P delta -0.03: above the NULL_rep gap (0.01), below the NULL_rep2 gap (0.05).  Using only NULL_rep would call it a gain: the mutation check turns this RED."""
    (r,) = seeds(dict(d_sp=-0.03))
    P = r["P_cell"]
    assert P["separated"] and P["delta"] == pytest.approx(-0.03) and P["floor"] == pytest.approx(0.05)
    assert P["above_floor"] is False and P["gain"] is False and r["flags"]["P"] is False


def test_auroc_between_the_two_floors_is_not_a_gain():
    f = floors()
    mid_sep = 0.12                                              # AUROC delta lands between gap_rep and gap_rep2 (asserted below)
    (r,) = seeds(dict(g_sep=mid_sep, l_sep=mid_sep))
    g = r["G_AUROC_goal"]
    assert f["detail"]["auroc_goal"]["gap_rep"] < g["delta"] < f["detail"]["auroc_goal"]["gap_rep2"], (g["delta"], f["detail"]["auroc_goal"])
    assert g["separated"] and g["gain"] is False and r["flags"]["G_AUROC"] is False


def test_fpr_between_the_two_floors_is_not_a_cost():
    (r,) = seeds(dict(fpr=0.025))
    fp = r["G_FP"]
    assert fp["delta"] == pytest.approx(0.025, abs=2e-3) and fp["separated"]
    assert fp["floor"] == pytest.approx(floors()["fpr"]) and fp["cost"] is False and r["flags"]["G_FP_ok"] is True


def test_planted_gain_above_both_floors_is_a_gain_and_a_cost_is_a_cost():
    (g,) = seeds(GOOD)
    assert g["flags"]["P"] and g["G_AUROC_goal"]["gain"] and g["flags"]["G_AUROC"] and g["flags"]["G_FP_ok"] and g["flags"]["NR_ok"]
    (c,) = seeds(dict(fpr=0.30))
    assert c["G_FP"]["cost"] and c["flags"]["G_FP_cost"]
    (n,) = seeds(dict(d_all=+0.6))
    assert n["NR_cell"]["cost"] and n["flags"]["NR_cost"]


def outcome(s0, s1):
    return S.arm_outcome(seeds(s0, s1))


def test_outcome_adopt():
    o = outcome(GOOD, GOOD)
    assert o["label"] == "PERC-ADOPT" and o["holding"] == ["PERC-ADOPT"] and "stacked arm" in o["trigger"]


def test_outcome_auroc_only():
    g = dict(g_sep=0.8, l_sep=0.8)
    o = outcome(g, g)
    assert o["label"] == "PERC-AUROC-ONLY" and o["holding"] == ["PERC-AUROC-ONLY"] and "STOPORC and STOPDEC" in o["trigger"]


def test_outcome_p_other_path():
    g = dict(d_sp=-0.5)
    o = outcome(g, g)
    assert o["label"] == "PERC-P-OTHER-PATH" and "CONSORC" in o["trigger"]


def test_outcome_fp_costs_and_precedence_over_auroc_only():
    g = dict(g_sep=0.8, l_sep=0.8, fpr=0.30)                      # G-AUROC gain, P fails, G-FP cost: AUROC-ONLY and FP-COSTS both hold literally
    o = outcome(g, g)
    assert o["label"] == "PERC-FP-COSTS" and o["holding"] == ["PERC-FP-COSTS", "PERC-AUROC-ONLY"] and set(o["triggers_of_all_holding"]) == {"PERC-FP-COSTS", "PERC-AUROC-ONLY"}
    g2 = dict(GOOD, fpr=0.30)                                      # P + G-AUROC + fp cost: FP-COSTS alone (not ADOPT)
    o2 = outcome(g2, g2)
    assert o2["label"] == "PERC-FP-COSTS" and "PERC-ADOPT" not in o2["holding"]


def test_outcome_null():
    o = outcome({}, {})
    assert o["label"] == "PERC-NULL" and o["holding"] == ["PERC-NULL"] and "eliminated" in o["trigger"]


def test_outcome_regresses_whatever_else_reads():
    bad = dict(GOOD, d_all=+0.6)
    o = outcome(bad, GOOD)                                         # NR cost at ONE seed is enough
    assert o["label"] == "PERC-REGRESSES" and o["per_seed_flags"][0]["NR_cost"] and not o["per_seed_flags"][1]["NR_cost"] and "W is halved" in o["trigger"]


def test_outcome_seeds_disagree_on_a_mixture():
    g = dict(g_sep=0.8, l_sep=0.8)
    o = outcome(dict(g, d_sp=-0.5), g)                             # P at s0 only, G-AUROC at both: no predicate holds
    assert o["label"] == "SEEDS-DISAGREE" and o["holding"] == [] and "third seed" in o["trigger"]
    o2 = outcome(GOOD, {})                                         # everything at s0, nothing at s1
    assert o2["label"] == "SEEDS-DISAGREE"


def test_lon_auroc_cost_makes_g_auroc_fail():
    (r,) = seeds(dict(g_sep=0.8, l_sep=-0.8))                      # goal AUROC up, lon-STOP AUROC down below the floor: a cost
    assert r["G_AUROC_goal"]["gain"] and r["G_AUROC_lon"]["cost"] and r["flags"]["G_AUROC"] is False
    assert "cue|auroc_lon" in r["named_costs"] and "cue|auroc_goal" in r["named_gains"]


def test_exact_zero_control():
    fl = floors()
    r = S.read_seed_s1(CI, NULL, arm(), fl, EID, SP, OK, NBOOT)
    for k, c in r["cells"].items():
        assert c["status"] == "OK" and c["delta"] == 0.0 and not c["gain"] and not c["cost"], k
    for k in ("G_AUROC_goal", "G_AUROC_lon", "G_AUROC_goal_beside_first_dose_windows", "G_FP", "hit_beside"):
        assert r[k]["delta"] == 0.0 and not r[k]["gain"] and not r[k]["cost"], k
    assert not r["named_costs"] and not r["named_gains"] and r["flags"]["P"] is False


def test_missing_floor_is_never_a_gain_or_cost():
    fl = floors()
    fl["cells"] = {k: None for k in fl["cells"]}
    fl.update(fpr=None, hit=None, auroc_goal=None, auroc_lon=None, auroc_goal_sp=None)
    r = S.read_seed_s1(CI, NULL, arm(**GOOD, fpr=0.3, d_all=0.5), fl, EID, SP, OK, NBOOT)
    assert not r["named_costs"] and not r["named_gains"]
    assert S.floor_max(None, 0.1) is None and S.floor_max(0.1, None) is None and S.floor_max(0.03, 0.2) == 0.2 and S.floor_max(0.2, 0.03) == 0.2


def test_read_all_needs_both_replicates_and_reports_strategic_paused():
    base = {"NULL_s0": NULL, "NULL_s1": NULL, "NULL_rep": REP, "NULL_rep2": REP2, "A_s0": arm(**GOOD), "A_s1": arm(**GOOD)}
    for gone in ("NULL_rep", "NULL_rep2", "NULL_s0"):
        a = {k: v for k, v in base.items() if k != gone}
        r = S.read_all(CI, a, EID, SP, OK, NBOOT)
        assert r["outcome"] == "INVALID" and "control 4" in r["reason"] and gone in r["reason"]
    r = S.read_all(CI, base, EID, SP, OK, NBOOT)
    assert r["outcome"] == "PERC-ADOPT" and r["A"]["label"] == "PERC-ADOPT"
    one = {k: v for k, v in base.items() if not k.endswith("_s1") or k == "NULL_s1"}
    one.pop("NULL_s1")
    r1 = S.read_all(CI, one, EID, SP, OK, NBOOT)
    assert r1["outcome"].startswith("NOT-YET") and r1["A"]["seeds"]["s1"]["status"] == "NOT-YET"
    # the verdict text: tier, estimator, STRATEGIC paused and never a pass
    rep = {"outcome": r["outcome"], "estimator": S.ESTIMATOR.format(nb=NBOOT), "controls": {"problems": []}, "null_reuse": "EQUAL", "smoke": False, **r}
    md = S.render_verdict(rep)
    assert "T1-kin" in md and "STRATEGIC: **PAUSED**" in md and "paired_episode_cluster_bootstrap" in md and "Control 5" in md
    assert "STRATEGIC" in S.TIER or "STRATEGIC: **PAUSED**" in md
    low = md.lower()
    assert "strategic: pass" not in low and "strategic pass" not in low


def test_auroc_fast_equals_the_first_dose_auroc_incl_ties():
    r = np.random.default_rng(3)
    for _ in range(20):
        s = np.round(r.normal(size=300), 1)                         # many ties
        y = r.random(300) < 0.2
        assert S.auroc_fast(s, y) == pytest.approx(SR.auroc(s, y), abs=1e-12)
    assert np.isnan(S.auroc_fast(np.ones(4), np.zeros(4, bool)))
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    assert S.auroc_fast(np.where(y, .9, .1), y) == 1.0 and S.auroc_fast(np.full(8, .3), y) == 0.5 and S.auroc_fast(np.where(y, .1, .9), y) == 0.0


def test_judge_rule():
    assert S.judge(-0.2, True, 0.1, False)[0] is True and S.judge(0.2, True, 0.1, False)[1] is True
    assert S.judge(-0.2, False, 0.1, False) == (False, False, True) and S.judge(-0.05, True, 0.1, False) == (False, False, False)
    assert S.judge(0.2, True, 0.1, True)[0] is True and S.judge(-0.2, True, 0.1, True)[1] is True and S.judge(0.2, True, None, True) == (False, False, False)


# ---------------------------------------------------------------- banks and the reuse state (planted files)
def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def plant(pull, tag, kind="null", w="12.3457", commit=None, stop_flag=None, lever_w=None, ck=None, man_sha=None, declared="PASS", prefix="stopperc_"):
    d = os.path.join(pull, prefix + tag)
    os.makedirs(d, exist_ok=True)
    ck = ck if ck is not None else ("ckpt of " + tag).encode()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(ck)
    s = man_sha or sha_b(ck)
    json.dump({"tag": tag, "commit": commit or S.COMMIT_S1, "files": {"ckpt_last.pt": {"bytes": len(ck), "sha256": s}}}, open(os.path.join(d, "BANK_MANIFEST.json"), "w"))
    json.dump({"tag": tag, "evidence": {"ckpt_sha256": s}}, open(os.path.join(d, "RUN.done"), "w"))
    lv = {"goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0}
    argv = "--arm M2b --seed 0 --grid pool4x10"
    if kind == "A":
        lv.update(stop_aux="on" if stop_flag is None else stop_flag, w_stop_aux=float(w) if lever_w is None else lever_w)
        argv += f" --stop-aux on --w-stop-aux {w} --stop-aux-extr D:\\x.json"
    json.dump({"declared_vs_built": declared, "levers": lv, "live": {"terms": {"goal": 0.1, "lon": 0.2}, "dead_params": [], "groups_grad": {"T": 1.0}}}, open(os.path.join(d, "selfcheck.json"), "w"))
    open(os.path.join(d, "argv.txt"), "w").write(argv + "\n")
    open(os.path.join(d, "BANK.done"), "w").write("verified\n")
    return d


def plant_bal(pull, tag, ck=None):
    d = os.path.join(pull, "stopbal_" + tag)
    os.makedirs(d, exist_ok=True)
    ck = ck if ck is not None else ("bal ckpt of " + tag).encode()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(ck)
    s = sha_b(ck)
    json.dump({"tag": tag, "commit": S.BAL_COMMIT, "files": {"ckpt_last.pt": {"bytes": len(ck), "sha256": s}}}, open(os.path.join(d, "BANK_MANIFEST.json"), "w"))
    json.dump({"tag": tag, "evidence": {"ckpt_sha256": s}}, open(os.path.join(d, "RUN.done"), "w"))
    json.dump({"declared_vs_built": "PASS", "levers": {"goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0}}, open(os.path.join(d, "selfcheck.json"), "w"))
    open(os.path.join(d, "BANK.done"), "w").write("verified\n")
    return s


def test_selfcheck_and_bank_refusals(tmp_path):
    pull = str(tmp_path / "pulls")
    plant(pull, "NULL_s0")
    plant(pull, "A_s0", "A")
    assert S.bank_problems("NULL_s0", pull) == [] and S.bank_problems("A_s0", pull) == []
    assert any("missing" in x for x in S.bank_problems("A_s1", pull))                                              # absent bank
    plant(pull, "A_s1", "A", stop_flag="off")
    assert any("levers.stop_aux" in x for x in S.bank_problems("A_s1", pull))                                       # arm A built with the head off
    plant(pull, "A_s1", "A", lever_w=1.0)
    assert any("w_stop_aux" in x and "argv" in x for x in S.bank_problems("A_s1", pull))                            # W read-back differs from the argv
    plant(pull, "A_s1", "A", commit="715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf")
    assert any("commit" in x for x in S.bank_problems("A_s1", pull))
    plant(pull, "A_s1", "A", ck=b"x", man_sha="0" * 64)
    assert any("checkpoint changed" in x for x in S.bank_problems("A_s1", pull))
    plant(pull, "A_s1", "A", declared="FAIL")
    assert any("G-DVB" in x for x in S.bank_problems("A_s1", pull))
    d = plant(pull, "NULL_rep2")
    sc = json.load(open(d + "/selfcheck.json"))
    sc["levers"]["stop_aux"] = "off"
    json.dump(sc, open(d + "/selfcheck.json", "w"))
    assert any("a NULL arm carries stop-aux" in x for x in S.bank_problems("NULL_rep2", pull))


def write_reuse(state, **kw):
    os.makedirs(state, exist_ok=True)
    j = {"schema": "stopperc-null-reuse-v1", "status": "computed", "bal_pinned_sha256": S.BAL_NULL_S0_PIN, "null_reuse": "EQUAL", "tensor_equal": True}
    j.update(kw)
    json.dump(j, open(os.path.join(state, "NULL_REUSE.json"), "w"))


def test_reuse_state_refusals_and_modes(tmp_path):
    pull, state = str(tmp_path / "pulls"), str(tmp_path / "state")
    plant(pull, "NULL_s0")
    mode, pr, _ = S.reuse_state(state, pull)                                                                         # absent
    assert mode is None and pr and "absent" in pr[0]
    write_reuse(state, null_reuse="UNKNOWN", status="not-computed")
    mode, pr, _ = S.reuse_state(state, pull)
    assert mode is None and pr and "UNKNOWN" in pr[0]
    write_reuse(state, null_reuse="MAYBE")
    assert S.reuse_state(state, pull)[0] is None
    with open(os.path.join(state, "NULL_REUSE.json"), "w") as f:
        f.write("{not json")
    assert S.reuse_state(state, pull)[0] is None
    # EQUAL needs the reused first-dose banks whole and BAL-NULL_s0 at its pin (a planted pin: the real one is a 87 MB file)
    write_reuse(state)
    mode, pr, _ = S.reuse_state(state, pull)
    assert mode == "EQUAL" and any("reused first-dose bank" in x for x in pr)                                        # BAL banks absent
    for t in ("BAL-NULL_s1", "BAL-NULL_rep"):
        plant_bal(pull, t)
    pin = plant_bal(pull, "BAL-NULL_s0", ck=b"bal null s0")
    S.BAL_NULL_S0_PIN = pin
    write_reuse(state, bal_pinned_sha256=pin)
    mode, pr, _ = S.reuse_state(state, pull)
    assert (mode, pr) == ("EQUAL", [])
    assert S.stage_tags("EQUAL") == ["NULL_s0", "NULL_rep2", "A_s0", "A_s1"]
    m = S.arm_map("EQUAL", pull, "D1", "DB", "S1", "SB")
    assert m["NULL_s1"]["bank"].endswith("stopbal_BAL-NULL_s1") and m["NULL_s1"]["dump"].startswith("DB/") and m["NULL_s1"]["stopdec_tag"] == "BAL-NULL_s1"
    assert m["NULL_rep"]["bank"].endswith("stopbal_BAL-NULL_rep") and m["A_s0"]["dump"] == "D1/dump_A_s0_m10_causal_heads" and m["A_s0"]["bank"].endswith("stopperc_A_s0")
    write_reuse(state, bal_pinned_sha256="0" * 64)                                                                   # JSON pins another BAL-NULL_s0
    assert S.reuse_state(state, pull)[1]
    write_reuse(state, bal_pinned_sha256=pin, tensor_equal=False)                                                    # says EQUAL, tensors not equal
    assert S.reuse_state(state, pull)[1]
    write_reuse(state, bal_pinned_sha256=pin, ckpt_sha256="f" * 64)                                                 # computed on another NULL_s0 checkpoint
    assert any("another NULL_s0" in x for x in S.reuse_state(state, pull)[1])
    # DIFFERS: the stopperc banks are the NULL_s1 / NULL_rep arms
    write_reuse(state, bal_pinned_sha256=pin, null_reuse="DIFFERS", tensor_equal=False)
    mode, pr, _ = S.reuse_state(state, pull)
    assert (mode, pr) == ("DIFFERS", [])
    assert S.stage_tags("DIFFERS") == ["NULL_s0", "NULL_rep2", "A_s0", "A_s1", "NULL_s1", "NULL_rep"]
    m = S.arm_map("DIFFERS", pull, "D1", "DB", "S1", "SB")
    assert m["NULL_s1"]["bank"].endswith("stopperc_NULL_s1") and m["NULL_rep"]["dump"] == "D1/dump_NULL_rep_m10_causal_heads"
    S.BAL_NULL_S0_PIN = "5169ea9554bf74004f801b2d594358355363ba47fe07aa8cac052d1da9aa2fdf"


def test_main_refuses_without_a_clean_reuse_state(tmp_path):
    state = str(tmp_path / "state")
    with pytest.raises(SystemExit) as e:
        S.cmd_dumps([], state, str(tmp_path / "p"))
    assert e.value.code == 2
    out = tmp_path / "r.json"
    with pytest.raises(SystemExit) as e:
        S.main_read(["--out", str(out), "--state", state, "--pull", str(tmp_path / "p")])
    assert e.value.code == 2 and json.load(open(out))["outcome"] == "REFUSED"


def test_compare_dumps(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    for d in (a, b):
        np.savez(d / "ep0000.npz", g=x, n=np.array([1, 2]))
    assert S.compare_dumps(str(a), str(b))["equal"] is True
    np.savez(b / "ep0000.npz", g=x + 1e-9, n=np.array([1, 2]))
    r = S.compare_dumps(str(a), str(b))
    assert r["equal"] is False and r["differing"] == ["ep0000.npz:g"] and r["max_abs_diff"] == pytest.approx(1e-9)
    assert S.compare_dumps(str(a), str(tmp_path / "none"))["equal"] is False


def test_no_cr_bytes_and_commit_pin():
    for f in ("stopperc_s1_read.py", "make_tree_s1.py", "test_stopperc_s1_read.py"):
        assert b"\r" not in open(os.path.join(HERE, f), "rb").read(), f
    assert S.COMMIT_S1 == "3e264bbf3fa2f6f17bea448de09aa972faaf7725" and S.BAL_COMMIT == "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
