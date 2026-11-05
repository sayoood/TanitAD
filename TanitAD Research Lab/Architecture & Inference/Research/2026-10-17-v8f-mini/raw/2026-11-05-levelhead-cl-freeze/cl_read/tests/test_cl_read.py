"""Rule, known-answer and end-to-end tests of code/cl_read.py (the CL read: PREREG_LEVELHEAD_CL.md sections 3-5).

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_cl_read.py --basetemp F:/TanitAD-scratch/motion_c/cl_read/pt/basetemp        (CPU only, ~1 minute; the basetemp parent must exist)
The reader under test is loaded BY FILE PATH (env CL_READ_PATH overrides, which is how the mutation arms in code/run_mutations_cl.py aim at a mutated copy).  No stack root is ever put on sys.path.
Layers:  (1) every rule on SYNTHETIC rows (P, G, F, LEAK-BORNE, D, the section-5 decision table, the family table);
         (2) the known answers on EXISTING arrays (ka-null: the arm IS its base; ka-ri: the RI rollout fed in the logged slot reproduces RI_C0_READ.json row for row);
         (3) END-TO-END on real per-window arrays with DESIGNED arms (a linear blend of the M1 array toward the P10 floor closes exactly alpha of the excess, so every branch of section 5 has a known answer);
         (4) the refusals: production with no CL rollouts must REFUSE (exit 3) and name its failed controls; pins.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = f"{PK}/code/cl_read.py"


def _load(path):
    spec = importlib.util.spec_from_file_location("cl_read_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load(os.environ.get("CL_READ_PATH", SRC))


@pytest.fixture(scope="module", autouse=True)
def heavy():
    C.init()


# ------------------------------------------------------------------------------------------------ helpers
def cell(better, share, xy=-0.03, yp=0.05):
    return {"better": better, "xy_delta": xy, "yp_delta": yp, "yp_separated_worse": False, "share_closed": share}


def rows(delta=-1.0, lo=-2.0, hi=-0.5, sep_worse=(), unsep=(), ade=None):
    """19 rows, all separated on the BETTER side (TAC rows: positive), except ``sep_worse`` (separated worse) and ``unsep`` (CI straddles 0); ``ade`` = (delta, lo, hi) overrides ADE6 as an UNSEPARATED row."""
    out = {}
    for lab in C.GD.LABELS:
        up = lab in C.GD.HIGHER_BETTER
        d = -delta if up else delta
        lo_, hi_ = (-hi, -lo) if up else (lo, hi)
        out[lab] = {"delta": d, "lo": lo_, "hi": hi_, "separated": True}
        if lab in unsep:
            out[lab] = {"delta": 0.001, "lo": -0.5, "hi": 0.5, "separated": False}
        if lab in sep_worse:
            out[lab] = {"delta": -d, "lo": -hi_, "hi": -lo_, "separated": True}
    if ade is not None:
        out["ADE6"] = {"delta": ade[0], "lo": ade[1], "hi": ade[2], "separated": False}
    return out


def xy_of(r):
    return C.SR.read_xy(C.GD, r)


def ra_of(r):
    return C.rr.read_a_block(C.SR, C.GD, r)


STRAT = ("S-POS@30", "S-POS@50", "S-POS@80")


# ------------------------------------------------------------------------------------------------ (1) rule tests on synthetic rows
def test_P_is_per_seed_not_the_mean():
    cells = {"s0": cell(True, 0.9), "s1": cell(True, 0.2)}
    p = C.rule_P(cells)
    assert p["PASS"] is False                                                   # the card: ">= 0.5 at both seeds"
    assert p["mean_form_reported_beside"]["PASS"] is True and abs(p["mean_form_reported_beside"]["mean_share_closed"] - 0.55) < 1e-12
    assert p["per_seed"]["s0"]["holds"] is True and p["per_seed"]["s1"]["holds"] is False


def test_P_edges():
    assert C.rule_P({"s0": cell(True, 0.5), "s1": cell(True, 0.5)})["PASS"] is True                 # the bar is inclusive
    assert C.rule_P({"s0": cell(True, 0.5), "s1": cell(True, 0.4999)})["PASS"] is False
    assert C.rule_P({"s0": cell(True, 0.9), "s1": cell(True, None)})["PASS"] is False               # no excess to close = no pass
    assert C.rule_P({"s0": cell(True, 0.9), "s1": cell(False, 0.9)})["PASS"] is False               # share without separation
    assert C.rule_P({"s0": cell(True, 0.9)})["PASS"] is False                                        # both seeds are required
    assert C.rule_P({"s0": cell(True, 0.9), "s1": cell(True, 0.9)})["n_seeds_separated_better"] == 2


def test_G_ade6_edge_and_noncrawl_rows():
    ok = rows(ade=(0.0604, -0.5, 0.7))
    over = rows(ade=(0.0605, -0.5, 0.7))
    assert C.rule_G({"s0": xy_of(ok), "s1": xy_of(ok)}, {"s0": ok, "s1": ok})["PASS"] is True
    assert C.rule_G({"s0": xy_of(ok), "s1": xy_of(over)})["PASS"] is False
    worse = rows(sep_worse=("speed MAE",))
    g = C.rule_G({"s0": xy_of(worse), "s1": xy_of(ok)})
    assert g["PASS"] is False and g["per_seed"]["s0"]["non_crawl_separated_worse"] == ["speed MAE"]
    crawl_worse = rows(sep_worse=("M1 crawl share",))                                                 # a crawl row is not a NON-crawl row
    assert C.rule_G({"s0": xy_of(crawl_worse), "s1": xy_of(ok)})["PASS"] is True
    both = C.rule_G({"s0": xy_of(ok), "s1": xy_of(ok)}, {"s0": ok, "s1": ok})
    assert both["ade6_ci_upper_end_within_guard_reported_beside"] == {"s0": False, "s1": False}      # the stricter reading is shown, and does not decide


def test_F_is_strict_and_per_seed():
    eff, rep = {"s0": -0.04, "s1": -0.05}, {"s0": 0.01, "s1": 0.02}
    assert C.rule_F(eff, rep)["PASS"] is True
    assert C.rule_F({"s0": -0.01, "s1": -0.05}, rep)["PASS"] is False                                 # |effect| == |floor| at one seed fails
    assert C.rule_F({"s0": -0.04, "s1": -0.019}, rep)["PASS"] is False
    f = C.rule_F(eff, rep, {"s0": -0.0, "s1": -0.0})
    assert f["PASS"] is True and f["logged_effect_form_reported_beside"]["PASS"] is False


def test_leak_borne_rule():
    L = lambda sl, sr: C.rule_leak({"s0": cell(True, sl), "s1": cell(True, 0.9)}, {"s0": cell(True, sr), "s1": cell(True, 0.9)})
    assert L(0.8, 0.3)["LEAK_BORNE"] is True                                    # 0.8 - 0.3 = 0.5 > 0.4
    assert L(0.8, 0.4)["LEAK_BORNE"] is False                                   # 0.8 - 0.4 = 0.4 is not MORE than half of 0.8
    assert L(0.8, 0.41)["LEAK_BORNE"] is False
    assert L(0.8, None)["LEAK_BORNE"] is True                                   # an undefined reindexed share counts 0
    assert L(None, 0.5)["LEAK_BORNE"] is False and L(0.0, 0.0)["LEAK_BORNE"] is False and L(-0.2, -0.9)["LEAK_BORNE"] is False   # nothing closed on the camera clock
    s1_only = C.rule_leak({"s0": cell(True, 0.5), "s1": cell(True, 1.0)}, {"s0": cell(True, 0.5), "s1": cell(True, 0.2)})
    assert s1_only["LEAK_BORNE"] is True and s1_only["per_seed"]["s1"]["leak_borne"] is True and s1_only["per_seed"]["s0"]["leak_borne"] is False   # either seed


def test_D_needs_read_a_at_both_seeds_and_P_and_G():
    good = ra_of(rows())
    assert good["DOMINANT"] is True
    assert C.rule_D({"s0": good, "s1": good}, True, True)["DOMINANT_COUNTED"] is True
    assert C.rule_D({"s0": good, "s1": good}, False, True)["DOMINANT_COUNTED"] is False                # D counts only with P
    assert C.rule_D({"s0": good, "s1": good}, True, False)["DOMINANT_COUNTED"] is False                # ... and G
    d1_bad = ra_of(rows(ade=(-0.2, -0.5, -0.1)))                                                       # ADE6 not separated: D1 fails
    assert d1_bad["D1"] is False
    assert C.rule_D({"s0": good, "s1": d1_bad}, True, True)["DOMINANT_COUNTED"] is False             # BOTH seeds
    d2_bad = ra_of(rows(sep_worse=("speed MAE",)))
    assert d2_bad["D2"] is False and C.rule_D({"s0": d2_bad, "s1": good}, True, True)["read_a_both_seeds"] is False


def test_D3_includes_the_strategic_family_and_reports_the_three_family_form_beside():
    no_strat = ra_of(rows(unsep=STRAT))
    assert no_strat["D3_by_family"]["STRATEGIC"] is False and no_strat["D3"] is False
    d = C.rule_D({"s0": no_strat, "s1": no_strat}, True, True)
    assert d["DOMINANT_COUNTED"] is False                                       # four families, STRATEGIC included, decide
    assert d["three_family_form_reported_beside"] is True                       # the three-family form is shown, and does not decide
    assert any("STRATEGIC" in t for t in d["by_seed"]["s0"]["failing_terms"])


def test_decision_table_section_5():
    D = C.decide
    r = D(True, True, True, 2, False)
    assert r["label"] == "D-HOLDS" and "signal the PI for scale" in r["next"] and r["counts_as_result"] is True
    r = D(True, True, False, 2, False)
    assert r["label"] == "P-HOLDS-G-OR-D-FAILS" and "CL-W2" in r["next"] and "G holds, D fails" in r["notes"]
    r = D(True, False, False, 2, False)
    assert r["label"] == "P-HOLDS-G-OR-D-FAILS" and "G fails" in r["notes"]
    r = D(False, True, False, 2, False)
    assert r["label"] == "P-FAILS-M1-SEPARATED-BETTER" and "CLR" in r["next"] and "RAPR-P01" in r["next"]
    r = D(False, True, False, 1, False)
    assert r["label"] == "P-FAILS-M1-SEPARATED-BETTER" and any("ONE seed" in n for n in r["notes"])
    r = D(False, True, False, 0, False)
    assert r["label"] == "P-FAILS-M1-NOT-MOVED" and "head read-out check" in r["next"] and "CRAWL-LEAD" in r["next"]
    r = D(False, False, False, 0, False)
    assert r["label"] == "P-FAILS-M1-NOT-MOVED" and any("G fails too" in n for n in r["notes"])


def test_leak_borne_overrides_the_label_whatever_else_it_reads():
    r = C.decide(True, True, True, 2, True)                                     # a DOMINANT-looking row that is camera-clock borne
    assert r["label"] == "LEAK-BORNE" and r["underlying"] == "D-HOLDS" and r["counts_as_result"] is False
    assert "camera-clock" in r["next"] and "never a result" in r["next"]
    r = C.decide(False, True, False, 0, True)
    assert r["label"] == "LEAK-BORNE" and r["underlying"] == "P-FAILS-M1-NOT-MOVED" and "head read-out check" in r["next"]       # next as for 'P fails'
    r = C.decide(False, True, False, 2, True)
    assert "CLR" in r["next"]


def test_family_table_counts_the_four_families():
    t = C.family_table(rows(sep_worse=("S-POS@50",), unsep=("TAC-LAT",)))
    assert set(t) == set(C.FOUR_FAMILIES) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
    assert t["STRATEGIC"]["n_rows"] == 3 and t["STRATEGIC"]["separated_worse"] == ["S-POS@50"] and len(t["STRATEGIC"]["separated_better"]) == 2
    assert t["TACTICAL"]["n_rows"] == 2 and t["TACTICAL"]["separated_better"] == ["TAC-LON"] and t["TACTICAL"]["separated_worse"] == []
    assert t["LONGITUDINAL"]["n_rows"] == 6 and t["LATERAL"]["n_rows"] == 6


# ------------------------------------------------------------------------------------------------ (2) known answers on existing arrays
@pytest.mark.parametrize("mode", ["ka-null", "ka-ri"])
def test_reader_known_answers(tmp_path, mode):
    assert C.main(["--mode", mode, "--raw", str(tmp_path)]) == 0
    res = json.load(open(tmp_path / f"KA_READER_{mode}.json", encoding="utf-8"))
    assert res["KA_PASS"] is True and all(res["ka_checks"].values())
    assert res["outcome"]["label"] == "P-FAILS-M1-NOT-MOVED"                   # the arm is its base (null) / the reindexed slots are null: nothing moved
    if mode == "ka-ri":
        assert len(res["ka_checks"]) >= 40                                      # every one of the 19 rows x 2 seeds is compared with the banked RI_C0_READ.json


# ------------------------------------------------------------------------------------------------ (3) end to end on real arrays with DESIGNED arms
def blend(z, label, alpha):
    """A copy of the array dict whose ``label`` row moves ``alpha`` of the way from the run's own value to the P10 floor's: its contrast against ``z`` closes exactly ``alpha`` of (z - P10)."""
    i = C.GD.LABELS.index(label)
    o = dict(z)
    o[f"a_{i}"] = z[f"a_{i}"] + alpha * (z[f"b_{i}"] - z[f"a_{i}"])
    return o


def dominant(z):
    """A composite run that beats the floor by a wide margin on EVERY row (TAC rows: higher is better)."""
    o = dict(z)
    for i, lab in enumerate(C.GD.LABELS):
        b = z[f"b_{i}"]
        o[f"a_{i}"] = b + 0.5 * (np.abs(b) + 1.0) if lab in C.GD.HIGHER_BETTER else b - 0.5 * (np.abs(b) + 1.0)
    return o


def worsen(z, label, by):
    i = C.GD.LABELS.index(label)
    o = dict(z)
    o[f"a_{i}"] = z[f"a_{i}"] + by
    return o


def designed(cl=0.0, ri=0.0, riclc=None, ri_edit=None):
    out = {}
    for s in C.SEEDS:
        z, _ = C.reused_arrays(s)
        z["CL"] = blend(z["C0"], C.M1, cl) if cl else z["C0"]
        z["CLC"], z["FZCL"] = z["C0C"], z["FZC0"]
        z["RICL"] = blend(z["RIC0"], C.M1, ri) if ri else z["RIC0"]
        if ri_edit:
            z["RICL"] = ri_edit(z["RICL"])
        z["RICLC"] = riclc(z["RIC0C"]) if riclc else z["RIC0C"]
        out[s] = z
    return out


@pytest.fixture(scope="module")
def reads():
    R = {}
    R["p_holds"] = C.read_all(designed(cl=0.8, ri=0.8), 2000, log=lambda *a: None)
    R["moved"] = C.read_all(designed(cl=0.3, ri=0.3), 2000, log=lambda *a: None)
    R["leak"] = C.read_all(designed(cl=0.8, ri=0.0), 2000, log=lambda *a: None)
    R["d_holds"] = C.read_all(designed(cl=0.8, ri=0.8, riclc=dominant), 2000, log=lambda *a: None)
    R["g_fails"] = C.read_all(designed(cl=0.8, ri=0.8, ri_edit=lambda z: worsen(z, "ADE6", 0.5)), 2000, log=lambda *a: None)
    return R


def test_designed_arm_closes_exactly_alpha(reads):
    for k in ("s0", "s1"):
        assert abs(reads["p_holds"]["seeds"][k]["m1_cell_reindexed"]["share_closed"] - 0.8) < 1e-9
        assert abs(reads["p_holds"]["seeds"][k]["m1_cell_logged"]["share_closed"] - 0.8) < 1e-9
        assert abs(reads["moved"]["seeds"][k]["m1_cell_reindexed"]["share_closed"] - 0.3) < 1e-9


def test_branch_P_holds_D_fails(reads):
    r = reads["p_holds"]
    assert r["rules"]["P"]["PASS"] is True and r["rules"]["G"]["PASS"] is True
    assert r["rules"]["LEAK_BORNE"]["LEAK_BORNE"] is False                      # the same share on both clocks
    assert r["rules"]["D"]["DOMINANT_COUNTED"] is False                         # the base composite is not DOMINANT, so neither is this arm's
    assert r["outcome"]["label"] == "P-HOLDS-G-OR-D-FAILS" and "CL-W2" in r["outcome"]["next"]


def test_branch_P_fails_M1_moved(reads):
    r = reads["moved"]
    assert r["rules"]["P"]["PASS"] is False and r["rules"]["P"]["n_seeds_separated_better"] == 2
    assert r["outcome"]["label"] == "P-FAILS-M1-SEPARATED-BETTER" and "CLR" in r["outcome"]["next"]


def test_branch_leak_borne_when_only_the_logged_clock_moves(reads):
    r = reads["leak"]
    assert r["rules"]["LEAK_BORNE"]["LEAK_BORNE"] is True
    assert r["outcome"]["label"] == "LEAK-BORNE" and r["outcome"]["underlying"] == "P-FAILS-M1-NOT-MOVED" and r["outcome"]["counts_as_result"] is False


def test_branch_D_holds_with_a_dominant_composite(reads):
    r = reads["d_holds"]
    assert r["rules"]["D"]["read_a_both_seeds"] is True and r["rules"]["P"]["PASS"] and r["rules"]["G"]["PASS"]
    assert r["outcome"]["label"] == "D-HOLDS" and "signal the PI for scale" in r["outcome"]["next"]
    for k in ("s0", "s1"):                                                       # four families, STRATEGIC included, each with a separated-better row
        ft = r["seeds"][k]["family_tables"]["RICLC_minus_P10"]
        assert all(len(ft[f]["separated_better"]) >= 1 for f in C.FOUR_FAMILIES) and all(ft[f]["separated_worse"] == [] for f in C.FOUR_FAMILIES)


def test_G_fails_when_ade6_worsens_beyond_the_guard(reads):
    r = reads["g_fails"]
    assert r["rules"]["P"]["PASS"] is True and r["rules"]["G"]["PASS"] is False
    assert r["outcome"]["label"] == "P-HOLDS-G-OR-D-FAILS" and "G fails" in r["outcome"]["notes"]


def test_nineteen_rows_per_contrast_and_results_md(reads):
    r = reads["d_holds"]
    for k in ("s0", "s1"):
        for c in ("RICL_minus_RIC0", "CL_minus_C0", "RICLC_minus_P10", "CLC_minus_P10", "FZCL_minus_FZC0", "C0REP_minus_C0", "RIC0_minus_P10"):
            assert len(r["seeds"][k][c]) == 19 and set(r["seeds"][k][c]) == set(C.GD.LABELS)
    r = dict(r, controls={"KA_NEW": {"PASS": True}}, tier="T1-kin", estimator="paired episode-cluster bootstrap B=2000")
    md = C.render_md(r)
    assert "**Outcome: D-HOLDS**" in md and "## 19 rows: RICL - RIC0" in md and "## Families" in md and "STRATEGIC" in md
    for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop"):
        assert w not in md.lower(), w


def test_floor_beside_is_the_replicate_pair(reads):
    f = reads["p_holds"]["rules"]["F"]
    for k in ("s0", "s1"):
        assert f["per_seed"][k]["abs_C0rep_minus_C0"] == abs(reads["p_holds"]["seeds"][k]["C0REP_minus_C0"][C.M1]["delta"])
        assert f["per_seed"][k]["abs_R_minus_C0"] == abs(reads["p_holds"]["seeds"][k]["RICL_minus_RIC0"][C.M1]["delta"])


# ------------------------------------------------------------------------------------------------ (4) refusals and pins
def test_production_without_cl_rollouts_refuses_and_names_the_controls(tmp_path):
    empty = tmp_path / "out"
    empty.mkdir()
    rc = C.main(["--mode", "production", "--out-root", str(empty), "--raw", str(tmp_path / "raw")])
    assert rc == 3
    res = json.load(open(tmp_path / "raw" / "CL_READ.json", encoding="utf-8"))
    assert "refused" in res and all(n in res["refused"] for n in ("rollout_RICL_s0", "rollout_FZCL_s1", "KA_NEW", "CL_PRECHECK"))
    assert not os.path.exists(tmp_path / "raw" / "arrays")                      # nothing was extracted


def test_card_rules_control_fails_on_a_changed_card(tmp_path, monkeypatch):
    empty = tmp_path / "out"
    empty.mkdir()
    ctl = C.run_controls(str(empty), str(tmp_path))
    assert ctl["card_rules"]["PASS"] is True and ctl["KA_NEW"]["PASS"] is False and ctl["CL_PRECHECK"]["PASS"] is False
    monkeypatch.setattr(C, "CARD_RULES_SHA", "0" * 64)                          # the committed rules were edited after the pin
    assert C.run_controls(str(empty), str(tmp_path))["card_rules"]["PASS"] is False


def test_pins_hold():
    assert C.card_rules_sha() == C.CARD_RULES_SHA
    assert C.sha256_file(C.RR) == C.RR_SHA
    for s in C.SEEDS:
        _, ctl = C.reused_arrays(s)
        assert all(v["PASS"] for v in ctl.values()) and len(ctl) == 6
    assert set(C.REUSED_SHA) == {f"{r}_s{s}" for r in C.REUSED for s in C.SEEDS}


def test_expected_flags_are_the_cards_scene_modes():
    assert C.expected_flags("CL") == "--smooth sg72 --scene logged"
    assert C.expected_flags("CLC") == "--smooth sg72 --scene logged --feasible kbox"
    assert C.expected_flags("FZCL") == "--smooth sg72 --scene hold_now"
    assert C.expected_flags("RICL") == "--smooth sg72 --scene reindexed"
    assert C.expected_flags("RICLC") == "--smooth sg72 --scene reindexed --feasible kbox"
