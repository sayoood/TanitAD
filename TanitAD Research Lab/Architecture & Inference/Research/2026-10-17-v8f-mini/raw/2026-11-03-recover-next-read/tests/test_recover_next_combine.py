"""Branch tests and mutation tests of code/recover_next_combine.py (the RECOVER-NEXT combiner: per-arm label / composite, attribution Q1, RAPR secondary, card section 4 next arm).

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_recover_next_combine.py     (CPU only, seconds)
The combiner under test is loaded BY FILE PATH (env RECOVER_NEXT_COMBINE_PATH aims the mutation tests at a mutated copy).  Inputs are SYNTHETIC per-arm JSONs with the reader's schema;
test_schema_coupling_with_the_real_reader_json proves the synthetic schema is the real one by running the combiner's accessors on the real RECOVER_READ.json.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = f"{PK}/code/recover_next_combine.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REAL_R = "E:/Projects/TanitAD-artifacts/dagland/recover_read/raw/RECOVER_READ.json"


def _load(path):
    spec = importlib.util.spec_from_file_location("recover_next_combine_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load(os.environ.get("RECOVER_NEXT_COMBINE_PATH", SRC))
M1 = C.M1


# ------------------------------------------------------------------------------------------------ synthetic per-arm reads (the reader's schema)
def guards(ok0=True, ok1=True, worse0=(), worse1=()):
    per = {"s0": {"ade6_delta": 0.01 if ok0 else 0.2, "ade6_within_guard": ok0, "non_crawl_separated_worse": list(worse0)},
           "s1": {"ade6_delta": 0.01 if ok1 else 0.2, "ade6_within_guard": ok1, "non_crawl_separated_worse": list(worse1)}}
    return {"per_seed": per, "PASS": bool(all(v["ade6_within_guard"] and not v["non_crawl_separated_worse"] for v in per.values()))}


def comp_seed(d1=True, d2_rows=(), d3=True):
    return {"DOMINANT": bool(d1 and not d2_rows and d3), "D1": d1, "D2": not d2_rows, "D3": d3, "D3_by_family": {}, "separated_worse_rows": list(d2_rows),
            "DOMINANT_three_families_strategic_paused": True, "failing_terms": []}


def composite(s0=None, s1=None, counts=False, status="NOT DOMINANT"):
    return {"status": status, "counts": counts, "attribution": "R-SPECIFIC", "by_seed": {"s0": s0 or comp_seed(), "s1": s1 or comp_seed()}}


def arm(name, shares=(0.3, 0.3), g=None, comp=None, label="HARMS", refused=None, r_arm="same"):
    return {"r_arm": name if r_arm == "same" else r_arm, "refused": refused, "label": label, "reader_sha256": "x",
            "primary": {"PASS": False, "share_closed": {"s0": shares[0], "s1": shares[1]}, "mean_share_closed": sum(0.0 if v is None else v for v in shares) / 2},
            "guards": g or guards(), "floor": {"PASS": True}, "frame_freeze": {"reading": "PARTIAL"}, "composite": comp or composite(), "controls_failed": []}


def rjson(label="HARMS", shares=(0.22, 0.132)):
    return {"label": label, "primary": {"PASS": False, "share_closed": {"s0": shares[0], "s1": shares[1]}, "mean_share_closed": sum(shares) / 2}}


M1_ALONE = composite(comp_seed(d2_rows=[M1]), comp_seed(d2_rows=[M1]))
NON_M1 = composite(comp_seed(d2_rows=["heading MAE"]), comp_seed(d2_rows=[M1]))
DOM_COUNTED = composite(comp_seed(), comp_seed(), counts=True, status="DOMINANT, COUNTED")


# ------------------------------------------------------------------------------------------------ attribution Q1
def test_q1_all_four_answers():
    bad, good = arm("RTL", g=guards(False, False)), arm("RTL", g=guards())
    badd, goodd = arm("RP01", g=guards(False, False)), arm("RP01", g=guards())
    assert C.attribution_q1(good, goodd, "HARMS")["answer"] == "both"
    assert C.attribution_q1(good, badd, "HARMS")["answer"] == "blend"            # only RTL's guards pass: the blend harmed
    assert C.attribution_q1(bad, goodd, "HARMS")["answer"] == "dose"             # only RP01's guards pass: the dose harmed
    assert C.attribution_q1(bad, badd, "HARMS")["answer"] == "neither"
    a = C.attribution_q1(good, badd, "HARMS")
    assert a["blend_harmed"] is True and a["dose_harmed"] is False


def test_q1_needs_both_seeds_and_names_the_non_crawl_row():
    one_seed = arm("RTL", g=guards(True, False))                                 # passes at s0, ADE6 guard fails at s1
    assert C.attribution_q1(one_seed, arm("RP01", g=guards()), "HARMS")["answer"] == "dose"
    worse = arm("RTL", g=guards(True, True, worse1=["heading MAE"]))             # a separated-worse non-crawl row at s1
    assert C.attribution_q1(worse, arm("RP01", g=guards()), "HARMS")["answer"] == "dose"
    a = C.attribution_q1(arm("RTL", shares=(0.4, 0.5)), arm("RP01", shares=(0.1, 0.2)), "HARMS")        # reported with both arms' M1 share closed
    assert a["RTL_m1_share_closed"] == {"s0": 0.4, "s1": 0.5} and a["RP01_m1_share_closed"] == {"s0": 0.1, "s1": 0.2}
    assert abs(a["RTL_m1_mean_share_closed"] - 0.45) < 1e-12


def test_q1_incomplete_and_not_applicable():
    good = arm("RTL")
    for missing in (None, arm("RP01", refused="a control failed"), arm("RP01", r_arm="RTL")):
        a = C.attribution_q1(good, missing, "HARMS")
        assert a["answer"] == "INCOMPLETE" and a["blend_harmed"] is True
    assert C.attribution_q1(None, None, "HARMS")["answer"] == "INCOMPLETE"
    na = C.attribution_q1(good, arm("RP01"), "NOT MOVED")
    assert na["answer"] == "NOT APPLICABLE" and "NOT MOVED" in na["note"]


# ------------------------------------------------------------------------------------------------ RAPR secondary
def test_rapr_secondary_is_strict_per_seed_and_reads_R_from_its_json():
    r = rjson(shares=(0.5, 0.4))                                                 # NOT the typed 0.220 / 0.132: the numbers come from the JSON
    both = C.rapr_secondary(arm("RAPR", shares=(0.51, 0.41)), r)
    assert both["exceeds_at_both"] is True and both["R_m1_share_closed"] == {"s0": 0.5, "s1": 0.4}
    one = C.rapr_secondary(arm("RAPR", shares=(0.51, 0.40)), r)                  # equal at s1 is NOT exceeding
    assert one["exceeds_at_both"] is False and one["exceeds_at_exactly_one"] is True and one["exceeds_per_seed"] == {"s0": True, "s1": False}
    none = C.rapr_secondary(arm("RAPR", shares=(0.5, 0.1)), r)
    assert none["exceeds_at_neither"] is True and none["exceeds_at_both"] is False
    und = C.rapr_secondary(arm("RAPR", shares=(None, 0.9)), rjson(shares=(0.0, 0.1)))     # an undefined share counts 0: 0 > 0 is False
    assert und["exceeds_per_seed"] == {"s0": False, "s1": True}
    assert C.rapr_secondary(None, r)["usable"] is False and C.rapr_secondary(arm("RAPR", refused="x"), r)["exceeds_at_both"] is None


# ------------------------------------------------------------------------------------------------ next arm (card section 4)
def arms_of(rapr=None, rtl=None, rp01=None):
    return {"RAPR": rapr, "RTL": rtl, "RP01": rp01}


def run_next(arms, shares_r=(0.22, 0.132)):
    r = rjson(shares=shares_r)
    return C.combine(arms, r)


def test_row1_any_counted_dominant():
    for who in ("RAPR", "RTL", "RP01"):
        arms = arms_of(**{who.lower(): arm(who, comp=DOM_COUNTED, label="RECOVERS")})
        n = run_next(arms)["next_arm"]
        assert n["selected_row"] == "row1" and who in n["row1_arms"] and "third seed pair" in n["selected"]
    uncounted = composite(comp_seed(), comp_seed(), counts=False, status="DOMINANT, RIG ARTEFACT")
    assert run_next(arms_of(rtl=arm("RTL", comp=uncounted)))["next_arm"]["selected_row"] is None        # DOMINANT but not counted is not row 1


def test_row2_m1_alone_and_closure_exceeds_at_both():
    n = run_next(arms_of(rapr=arm("RAPR", shares=(0.5, 0.5), comp=M1_ALONE)))["next_arm"]
    assert n["selected_row"] == "row2" and "--recover-p 0.5" in n["selected"] and n["matching_rows"] == ["row2"]
    assert n["facts"]["RAPRC_d2_profile"]["fails_D2_on_M1_alone"] is True
    d1_fails = composite(comp_seed(d1=False, d2_rows=[M1]), comp_seed(d2_rows=[M1]))                        # D1 also fails: not "M1 alone"
    assert run_next(arms_of(rapr=arm("RAPR", shares=(0.5, 0.5), comp=d1_fails)))["next_arm"]["selected_row"] is None


def test_row3_non_m1_row_names_the_setting_from_the_attribution():
    ok, bad = guards(), guards(False, False)
    cases = [(ok, bad, "target logged"), (bad, ok, "recover-p 0.1"), (ok, ok, "AND"), (bad, bad, "NO setting")]       # (RTL guards, RP01 guards, setting text)
    for gb, gd, frag in cases:
        arms = arms_of(rapr=arm("RAPR", shares=(0.5, 0.5), comp=NON_M1), rtl=arm("RTL", g=gb), rp01=arm("RP01", g=gd))
        n = run_next(arms)["next_arm"]
        assert n["selected_row"] == "row3" and frag in n["row3_setting"], (frag, n)
    pending = run_next(arms_of(rapr=arm("RAPR", shares=(0.5, 0.5), comp=NON_M1)))["next_arm"]
    assert pending["selected_row"] == "row3" and "pending" in pending["row3_setting"]


def test_row4_closure_does_not_exceed_at_either_seed():
    n = run_next(arms_of(rapr=arm("RAPR", shares=(0.1, 0.05), comp=M1_ALONE)))["next_arm"]
    assert n["selected_row"] == "row4" and "PI-decision" in n["selected"]


def test_table_order_decides_overlaps_and_lists_every_match():
    both = run_next(arms_of(rapr=arm("RAPR", shares=(0.1, 0.05), comp=NON_M1)))["next_arm"]                   # rows 3 and 4 both match; table order: row 3
    assert both["matching_rows"] == ["row3", "row4"] and both["selected_row"] == "row3" and "SELECTED (first of several" in both["status"]
    top = run_next(arms_of(rapr=arm("RAPR", shares=(0.1, 0.05), comp=NON_M1), rtl=arm("RTL", comp=DOM_COUNTED)))["next_arm"]
    assert top["matching_rows"][0] == "row1" and top["selected_row"] == "row1"


def test_uncovered_cases_are_reported_not_filled():
    mixed = run_next(arms_of(rapr=arm("RAPR", shares=(0.5, 0.05), comp=M1_ALONE)))["next_arm"]                 # exceeds at exactly one seed: row 2 needs both, row 4 needs neither
    assert mixed["selected_row"] is None and mixed["selected"] is None and mixed["status"].startswith("UNCOVERED") and "exactly one seed" in mixed["why_uncovered"]
    assert run_next(arms_of())["next_arm"]["selected_row"] is None and "no usable read" in run_next(arms_of())["next_arm"]["why_uncovered"]
    assert "rule_zero" in mixed


def test_per_arm_summary_and_wrong_arm_file_is_refused():
    out = run_next(arms_of(rapr=arm("RAPR", label="PARTIAL"), rtl=arm("RTL", r_arm="RP01")))
    assert out["per_arm"]["RAPR"]["usable"] and out["per_arm"]["RAPR"]["label"] == "PARTIAL" and out["per_arm"]["RAPR"]["composite_status"] == "NOT DOMINANT"
    assert out["per_arm"]["RTL"]["usable"] is False and out["per_arm"]["RTL"]["why"].startswith("WRONG ARM")
    assert out["per_arm"]["RP01"] == {"arm": "RP01", "usable": False, "why": "ABSENT"}


def test_schema_coupling_with_the_real_reader_json():
    """The real RECOVER_READ.json (the reader's output for R) carries every field the combiner reads; run the combiner's accessors on it as if it were an RTL read."""
    real = json.load(open(REAL_R, encoding="utf-8"))
    j = copy.deepcopy(real)
    j["r_arm"] = "RTL"
    s = C.arm_summary(j, "RTL")
    assert s["usable"] and s["label"] == real["label"] == "HARMS" and s["composite_status"] == real["composite"]["status"]
    assert C.guards_pass_both_seeds(j) is real["guards"]["PASS"] is False            # the RECOVER-FT HARMS read: R's guards fail
    assert C.d2_profile(j["composite"])["fails_D2_on_M1_alone"] is True              # RC fails D2 on M1 alone at both seeds (RECOVER-FT composite read)
    assert C.share(j, "s0") == real["primary"]["share_closed"]["s0"] and 0.21 < C.share(j, "s0") < 0.23 and 0.13 < C.share(j, "s1") < 0.14
    assert C.attribution_q1(j, copy.deepcopy(j) | {"r_arm": "RP01"}, real["label"])["answer"] == "neither"


def test_main_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rj = tmp_path / "r.json"
    rj.write_text(json.dumps(rjson()), encoding="utf-8")
    for a_ in (arm("RAPR", shares=(0.5, 0.5), comp=M1_ALONE), arm("RTL", g=guards()), arm("RP01", g=guards(False, False))):
        (raw / f"RECOVER_READ_{a_['r_arm']}.json").write_text(json.dumps(a_), encoding="utf-8")
    assert C.main(["--raw", str(raw), "--r-json", str(rj)]) == 0
    out = json.load(open(raw / "RECOVER_NEXT.json", encoding="utf-8"))
    assert out["attribution_q1"]["answer"] == "blend" and out["next_arm"]["selected_row"] == "row2" and out["rapr_secondary"]["exceeds_at_both"] is True
    assert set(out["inputs"]) == {"RAPR", "RTL", "RP01", "R"} and out["inputs"]["RTL"]["sha256"] and out["combiner_sha256"]
    (raw / "RECOVER_READ_RTL.json").unlink()                                       # any that exist: a missing arm is reported, not an error
    assert C.main(["--raw", str(raw), "--r-json", str(rj)]) == 0
    assert json.load(open(raw / "RECOVER_NEXT.json", encoding="utf-8"))["attribution_q1"]["answer"] == "INCOMPLETE"


# ------------------------------------------------------------------------------------------------ mutation tests: change a clause, a named test must go RED
MUTATIONS = {
    "Q1 requires BOTH seeds (all -> any)": ('return bool(all(per[sk]["ade6_within_guard"] and not per[sk]["non_crawl_separated_worse"] for sk in SEEDS))',
                                           'return bool(any(per[sk]["ade6_within_guard"] and not per[sk]["non_crawl_separated_worse"] for sk in SEEDS))', ["test_q1_needs_both_seeds_and_names_the_non_crawl_row"]),
    "Q1 blend / dose swapped": ('("blend" if gb else ("dose" if gd else "neither"))', '("dose" if gb else ("blend" if gd else "neither"))', ["test_q1_all_four_answers"]),
    "RAPR secondary strict (> -> >=)": ('ex = {sk: bool(share(rapr, sk) > share(r, sk)) for sk in SEEDS}', 'ex = {sk: bool(share(rapr, sk) >= share(r, sk)) for sk in SEEDS}', ["test_rapr_secondary_is_strict_per_seed_and_reads_R_from_its_json"]),
    "next arm: table order (first -> last match)": ('sel = matching[0] if matching else None', 'sel = matching[-1] if matching else None', ["test_table_order_decides_overlaps_and_lists_every_match"]),
    "row 2 needs the closure at both seeds": ('if prof and prof["fails_D2_on_M1_alone"] and secondary.get("exceeds_at_both"):', 'if prof and prof["fails_D2_on_M1_alone"]:', ["test_uncovered_cases_are_reported_not_filled"]),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_goes_red(name, tmp_path):
    old, new, nodes = MUTATIONS[name]
    src = open(SRC, encoding="utf-8").read()
    assert src.count(old) == 1, f"the clause to change is not present exactly once: {name}"
    mut = tmp_path / "recover_next_combine_mut.py"
    mut.write_text(src.replace(old, new, 1), encoding="utf-8", newline="\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = lambda: [PY, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + [f"{HERE}/test_recover_next_combine.py::{n}" for n in nodes]   # noqa: E731
    clean = subprocess.run(cmd(), env=dict(env, RECOVER_NEXT_COMBINE_PATH=SRC), capture_output=True, text=True)
    assert clean.returncode == 0, "the control (unmutated combiner) must be GREEN:\n" + clean.stdout[-600:]
    red = subprocess.run(cmd(), env=dict(env, RECOVER_NEXT_COMBINE_PATH=str(mut)), capture_output=True, text=True)
    assert red.returncode != 0, f"mutation '{name}' left {nodes} GREEN: the test cannot see this clause"
