"""Tests of leadsup_read_n1.py (card PREREG_V8F_PROOF_ADD24 with the frozen note PREREG_V8F_PROOF_ADD24_N1). Fixtures of test_leadsup_read.py are reused by import.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_leadsup_read_n1.py --basetemp F:/TanitAD-scratch/v8proof/spatial/leadsup/read/tmp/n1/pt   (the parent of --basetemp must exist)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
sys.path.insert(0, HERE)
import leadsup_read_n1 as R  # noqa: E402
from test_leadsup_read import B, pack, probe_fit, windows_of, world  # noqa: E402,F401  (fixtures and helpers)

O, H = R.O, R.H


def banned_words():
    return ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "correc" + "tion", "with" + "drawn", "no lon" + "ger", "closed lo" + "op"]


# ------------------------------------------------------------------------------------------------ R3n is a real refit
def test_r3n_is_a_refit_without_the_closing_column_and_not_r3_with_a_zeroed_coefficient(world):
    ctx, T = world["ctx"], world["T"]
    Z15, _ = windows_of(world)
    out = R.window_outputs(T, Z15)
    PVc, PVn = R.n1_inputs(out["gap_m"], out["closing_ms"], ctx["V"], ctx["y"], ctx["sc"], ctx["fold"])
    X = PVc["inp"]["X"]
    g = out["gap_m"]
    zeroed_after = np.full(len(g), np.nan)
    for f in range(5):
        a_n, b_n, d_n = PVn["coefs"][f]
        assert a_n == 0.0                                                    # the column was zero in the fit
        tr = ctx["sc"] & np.isfinite(ctx["y"]) & (ctx["fold"] != f) & np.isfinite(g) & np.isfinite(X)
        A = np.c_[X[tr], np.ones(int(tr.sum()))]
        coef = np.linalg.lstsq(A, (ctx["y"] - g)[tr], rcond=None)[0]         # an explicit two-column fit on the same rows
        assert abs(coef[0] - b_n) < 1e-9 and abs(coef[1] - d_n) < 1e-9
        a_c, b_c, d_c = PVc["coefs"][f]
        te = np.isfinite(g) & (ctx["fold"] == f)
        zeroed_after[te] = g[te] + b_c * X[te] + d_c                        # R3 fitted WITH the closing column, coefficient zeroed afterwards
    m = ctx["sc"] & np.isfinite(zeroed_after) & np.isfinite(PVn["pred"]["R3"])
    assert np.max(np.abs(zeroed_after[m] - PVn["pred"]["R3"][m])) > 1e-6     # not the same predictor: a different b and d
    assert (PVc["coefs"][0][0] != 0.0)


# ------------------------------------------------------------------------------------------------ DELTA and k3_slot_n1
def test_informative_closing_output_gives_delta_above_zero_separated_and_k3_slot_n1_passes_and_its_regression_fails(world):
    ctx, T = world["ctx"], world["T"]
    Z15, _ = windows_of(world)
    out = R.window_outputs(T, Z15)
    Rd, PV = R.eval_read(ctx, out)
    d = Rd["M3"]["delta_n1"]
    assert d["skill"] > 0 and d["separated_pos"] and d["n"] > 300 and d["ci95"][0] > 0
    k = R.control_k3_slot_n1(out["gap_m"], out["closing_ms"], ctx)
    assert k["pass"] and k["delta_permuted"] <= R.BAR_K3 and len(k["info_only_delta_other_permutations"]) == 4
    reg = R.control_k3_slot_n1(out["gap_m"], out["closing_ms"], ctx, perm=np.arange(len(Z15)))      # deliberate regression: identity permutation
    assert reg["permutation_is_identity"] and reg["pass"] is False and reg["delta_permuted"] > R.BAR_K3
    kc, kf = R.control_k3_split(T, Z15, ctx)
    assert kc["pass"] and kf["gates_under_note_n1"] is False
    kc2, _ = R.control_k3_split(T, Z15, ctx, perm=np.arange(len(Z15)))
    assert kc2["pass"] is False                                              # k3_closing also has its deliberate regression


def test_noise_closing_output_with_informative_gap_and_strong_ego_term_separates_what_the_card_form_mixes(world):
    ctx, T = world["ctx"], world["T"]
    Z15, _ = windows_of(world)
    out = dict(R.window_outputs(T, Z15))
    rng = np.random.default_rng(7)
    out["closing_ms"] = rng.standard_normal(len(out["closing_ms"])) * 3.0           # pure noise
    ctx2 = dict(ctx)
    X = O.rule_inputs(out["gap_m"], out["closing_ms"], ctx["V"], R.K)["X"]
    ctx2["y"] = ctx["y"] + 12.0 * X                                                  # strong ego term in the target
    Rd, _ = R.eval_read(ctx2, out)
    assert Rd["M3"]["rules"]["R3"]["skill_over_copy_bc"]["skill"] > 0.5             # the card form reads the ego term as a slot skill
    assert Rd["M3"]["delta_n1"]["skill"] <= 0.02                                     # the note: the closing speed adds nothing
    assert not Rd["M3"]["delta_n1"]["separated_pos"]
    # and the same states permuted keep the ego term: the card-form k3 slot part fails while k3_slot_n1 (closing output permuted) reads at most the bar
    k = R.control_k3_slot_n1(out["gap_m"], out["closing_ms"], ctx2)
    assert k["delta_permuted"] <= R.BAR_K3


# ------------------------------------------------------------------------------------------------ gating
def cd(**kw):
    return {k: {"pass": v} for k, v in kw.items()}


def ctrl_all_pass(arms=("LS_s0", "LS_s1")):
    c = {"k1_label_path": {"pass": True}, "a2_pack_counts": {"pass": True}}
    for a in arms:
        c["a1_records|%s" % a] = {"pass": True}
        c["k2_head_forward_eval|%s" % a] = {"pass": True}
        c["k4_no_future_input|%s" % a] = {"pass": True}
        c["k3_closing|%s" % a] = {"pass": True}
        c["k3_slot_n1|%s" % a] = {"pass": True}
        c["k3_card_form_slot|%s" % a] = {"pass": True}
    return c


def test_a_failing_card_form_slot_does_not_block_the_n1_names():
    c = ctrl_all_pass()
    c["k3_card_form_slot|LS_s0"] = {"pass": False}
    g = R.gate_controls(c)
    assert g["closing_ok"] and g["slot_ok"] and g["card_form_failed"] == ["k3_card_form_slot|LS_s0"]


def test_a_failing_k3_slot_n1_blocks_the_slot_names_not_the_closing_names():
    c = ctrl_all_pass()
    c["k3_slot_n1|LS_s1"] = {"pass": False}
    g = R.gate_controls(c)
    assert g["closing_ok"] and not g["slot_ok"] and g["failed_slot_extra"] == ["k3_slot_n1|LS_s1"]
    pa, paired = six_arms(), six_pairs()
    o = R.name_outcomes_n1(pa, paired, 0.02, g)
    assert o["per_type"]["LS"]["closing"] == "SUP-READS-CLOSING" and o["per_type"]["LS"]["slot"][0].startswith("not named (control failed")
    assert o["gradient"]["LS against LSD"] == "SUP-GRADIENT-HELPS"


def test_a_failing_k3_closing_blocks_the_closing_names_the_gradient_question_and_the_slot_names():
    c = ctrl_all_pass()
    c["k3_closing|LS_s0"] = {"pass": False}
    g = R.gate_controls(c)
    assert not g["closing_ok"] and not g["slot_ok"]
    o = R.name_outcomes_n1(six_arms(), six_pairs(), 0.02, g)
    assert o["per_type"]["LS"]["closing"].startswith("not named (control failed")
    assert o["per_type"]["LS"]["slot"][0].startswith("not named (control failed")
    assert o["gradient"]["LS against LSD"].startswith("not named (control failed")
    for k in ("k1_label_path", "a2_pack_counts"):                            # the common controls block everything
        c2 = ctrl_all_pass()
        c2[k] = {"pass": False}
        g2 = R.gate_controls(c2)
        assert not g2["closing_ok"] and not g2["slot_ok"]
    c3 = ctrl_all_pass()
    c3["k4_no_future_input|LS_s0"] = {"pass": False}
    assert not R.gate_controls(c3)["closing_ok"]


# ------------------------------------------------------------------------------------------------ the names, on synthetic numbers
def m1(r2, lo, hi):
    return {"r2": r2, "r2_ci95": [lo, hi]}


def m3n(skill, lo, hi, banked=None):
    r = {"delta_n1": {"skill": skill, "ci95": [lo, hi], "separated_pos": bool(skill > 0 and lo > 0)}}
    r["banked"] = None if banked is None else {"R3": {"separated_pos": banked}}
    return r


def prow(diff, lo, hi):
    return {"diff": diff, "ci95": [lo, hi], "separated_pos": lo > 0, "separated_neg": hi < 0}


def six_arms():
    pa = {}
    for t, good in (("LS", True), ("LSH", True), ("LSD", False)):
        for s in ("s0", "s1"):
            pa["%s_%s" % (t, s)] = {"M1": m1(0.3 if good else 0.0, 0.1 if good else -0.1, 0.4 if good else 0.1),
                                    "M3": m3n(0.2, 0.05, 0.3, True) if good else m3n(0.01, -0.05, 0.06, True)}
    return pa


def six_pairs():
    return {("LS", "LSD", "s0"): prow(0.3, 0.1, 0.4), ("LS", "LSD", "s1"): prow(0.32, 0.1, 0.5), ("LSH", "LSD", "s0"): prow(0.01, -0.1, 0.1), ("LSH", "LSD", "s1"): prow(0.02, -0.1, 0.1)}


def test_slot_names_under_n1():
    assert R.slot_names_n1([m3n(0.2, 0.05, 0.3, True), m3n(0.1, 0.02, 0.2, True)]) == ["SUP-SLOT-IMAGINES-N1", "SUP-SLOT-BEATS-ROLLOUT"]
    assert R.slot_names_n1([m3n(0.2, 0.05, 0.3, True), m3n(0.1, 0.02, 0.2, False)]) == ["SUP-SLOT-IMAGINES-N1"]
    assert R.slot_names_n1([m3n(0.2, 0.05, 0.3, True), m3n(0.1, -0.02, 0.2, True)]) == ["SUP-SLOT-FLAT-N1"]       # one interval includes 0
    assert R.slot_names_n1([m3n(0.2, 0.05, 0.3), m3n(-0.1, -0.3, -0.02)]) == ["SUP-SLOT-FLAT-N1"]
    n = R.slot_names_n1([m3n(0.2, 0.05, 0.3), m3n(0.1, 0.02, 0.2)])
    assert n[0] == "SUP-SLOT-IMAGINES-N1" and "not evaluable" in n[1]
    assert R.slot_names_n1([m3n(0.2, 0.05, 0.3)])[0].startswith("not named")


def test_name_outcomes_n1_over_six_synthetic_arms():
    g = R.gate_controls(ctrl_all_pass(("LS_s0", "LS_s1", "LSH_s0", "LSH_s1", "LSD_s0", "LSD_s1")))
    o = R.name_outcomes_n1(six_arms(), six_pairs(), 0.02, g)
    assert o["per_type"]["LS"]["closing"] == "SUP-READS-CLOSING" and o["per_type"]["LSD"]["closing"] == "SUP-CLOSING-FLAT"
    assert o["per_type"]["LS"]["slot"] == ["SUP-SLOT-IMAGINES-N1", "SUP-SLOT-BEATS-ROLLOUT"] and o["per_type"]["LSD"]["slot"] == ["SUP-SLOT-FLAT-N1"]
    assert o["gradient"]["LS against LSD"] == "SUP-GRADIENT-HELPS" and o["gradient"]["LSH against LSD"] == "SUP-GRADIENT-EQUAL"
    pa = six_arms()
    pa.pop("LSD_s1")
    assert R.name_outcomes_n1(pa, six_pairs(), None, g)["per_type"]["LSD"]["closing"].startswith("not named")


# ------------------------------------------------------------------------------------------------ a whole run: files, sections, wording
def test_a_run_writes_the_n1_files_with_delta_the_card_form_section_and_the_n1_names(world, pack):
    d = world["dir"]
    out = d + "/out_n1"
    rc = R.run(["LS_s0", "LS_s1"], d, out, B, HERE, do_train=False, do_probe=True, probe_fit=probe_fit, banked_dir=d + "/none")
    res = json.load(open(out + "/RESULT_LEADSUP_T0_N1.json", encoding="utf-8"))
    assert rc == 0 and res["controls_pass"] is True and res["gate"]["closing_ok"] and res["gate"]["slot_ok"]
    assert os.path.exists(out + "/VERDICT_LEADSUP_T0_N1.md") and not os.path.exists(out + "/RESULT_LEADSUP_T0.json")
    for arm in ("LS_s0", "LS_s1"):
        dn = res["arms"][arm]["M3"]["delta_n1"]
        assert dn["skill"] > 0 and dn["separated_pos"] and set(dn) >= {"skill", "ci95", "n", "separated_pos"}
        assert "k3_slot_n1|%s" % arm in res["controls"] and "k3_closing|%s" % arm in res["controls"] and "k3_card_form_slot|%s" % arm in res["controls"]
        assert "k3_shuffled_states|%s" % arm not in res["controls"]
    o = res["outcomes"]
    assert o["per_type"]["LS"]["closing"] == "SUP-READS-CLOSING" and o["per_type"]["LS"]["slot"][0] == "SUP-SLOT-IMAGINES-N1"
    txt = open(out + "/VERDICT_LEADSUP_T0_N1.md", encoding="utf-8").read()
    assert txt.index("## Controls") < txt.index("## Card form (as written)") < txt.index("## Names under note N1") < txt.index("## Per arm")
    assert "M3 DELTA (note N1; names the slot):" in txt and "STRATEGIC: PAUSED (PI item 25)" in txt and "the card names no outcome for an arm type with a failed control" in txt.lower()
    assert "SUP-SLOT-IMAGINES-N1" in txt
    for w in banned_words():
        assert w not in txt.lower(), w
    src = open(HERE + "/leadsup_read_n1.py", encoding="utf-8").read().lower()
    for w in banned_words():
        assert w not in src, w
