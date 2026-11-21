"""test_r9_e3_verdict.py -- the refcv9 E3 final-read judge (REFCV9_DESIGN section 5, sealed), on SYNTHETIC JSONs.  CPU only, no model, no GPU.
No HOLD number of any refcv9 arm is read here: every value below is invented to exercise the judge.

(1) the BARS table is PINNED to the section-5 text + the seal decisions (a moved bar turns RED);  (2) the all-pass case;  (3) one RED arm per bar;
(4) the deliberate regressions: a moved bar, a dropped family, a DEV input, STRATEGIC marked PASS;  (5) the refusals;  (6) seeds, K-BASE, NOT-COMPUTABLE.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "r9_e3_verdict.py"
_spec = importlib.util.spec_from_file_location("r9_e3_verdict", SCRIPT)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

REPO = Path(__file__).resolve().parents[2]
PR = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-27-perception-redesign"

# ------------------------------------------------------------------------------------------------------------- (1) the pin: an INDEPENDENT copy of the sealed values
#: (bar, sub id) -> (kind, value, key).  Transcribed from REFCV9_DESIGN section 5.2 and SEAL_DECISIONS_PRE_DATA_2026-11-08.md (+ the PI amendment "Use 0.9 bar").
PINNED = {
    ("MAP-1", "rate_le"): ("le", 0.098, None), ("MAP-1", "sep_below_ref"): ("sep_better", None, "off_drivable"),
    ("MAP-2", "p_0_5_ge"): ("ge", 0.60, "nf_prec_lv_0_5"), ("MAP-2", "p_5_10_ge"): ("ge", 0.90, "nf_prec_5_10"), ("MAP-2", "p_10_20_ge"): ("ge", 0.90, "nf_prec_10_20"),
    ("MAP-2", "p_20_40_sep_above_ref"): ("sep_better", None, "nf_prec_20_40"),
    ("MAP-3", "lane_ge_1p5x_ref"): ("ge_ref", 1.5, "lane_tiou05_0_20"), ("MAP-3", "lane_sep_above_ref"): ("sep_better", None, "lane_tiou05_0_20"),
    ("MAP-3", "bnd_recall_ge"): ("ge", 0.50, "drv_bnd_rec03_0_20"),
    ("MAP-4", "drv_iou_not_sep_worse"): ("not_sep_worse", None, ("drv_iou_0_5", "drv_iou_5_10", "drv_iou_10_20", "drv_iou_20_40")),
    ("MAP-5", "unobs_flags_zero"): ("zero", 0, None),
    ("BOX-1", "ap_ge"): ("ge", 0.45, "ap2m_all"), ("BOX-1", "ap_sep_above_ref"): ("sep_better", None, "ap2m_all"),
    ("BOX-2", "ap_0_20_ge"): ("ge", 0.62, "ap2m_0_20"), ("BOX-2", "ap_0_20_sep"): ("sep_better", None, "ap2m_0_20"),
    ("BOX-2", "ap_20_40_ge"): ("ge", 0.32, "ap2m_20_40"), ("BOX-2", "ap_20_40_sep"): ("sep_better", None, "ap2m_20_40"),
    ("BOX-2", "ap_40_60_ge"): ("ge", 0.15, "ap2m_40_60"), ("BOX-2", "ap_40_60_sep"): ("sep_better", None, "ap2m_40_60"),
    ("BOX-3", "bpo_le"): ("le", 1.25, "boxes_per_object_noNMS"),
    ("BOX-4", "tp_class_acc_beside"): ("beside", 0.822, "tp_class_acc"),
    ("FAM-1", "no_row_separated_worse"): ("not_sep_worse", None, None),
    ("FAM-2", "off_drivable_sep_better"): ("sep_better", None, "off_drivable"), ("FAM-2", "edge_contact_sep_better"): ("sep_better", None, "edge_contact"),
    ("FAM-2", "lane_cross_sep_better"): ("sep_better", None, "lane_cross"),
    ("FAM-2", "distance_keeping_not_worse"): ("not_sep_worse", None, ("LON_dk_headway_min_m", "LON_dk_time_gap_min_s", "LON_dk_min_ttc_s")),
    ("W1", "fd_sep_lower"): ("sep_better", None, "FD"), ("W2", "md_not_sep_worse"): ("not_sep_worse", None, "MD"),
    ("W4", "sep_better_than_camblind"): ("sep_better", None, None), ("W5", "sep_better_than_egofloor"): ("sep_better", None, None),
}
PINNED_REQUIRED = {"MAP-1", "MAP-2", "MAP-3", "MAP-4", "MAP-5", "BOX-1", "BOX-2", "BOX-3", "FAM-1", "FAM-2", "W1", "W2", "W4", "W5"}     # BOX-4 is beside
PINNED_PLANNER = {"MAP-1", "FAM-1", "FAM-2", "W1", "W2", "W4", "W5"}                                                                  # read at seeds 0 AND 1
PINNED_SPLIT = {"MAP-1": "hold", "MAP-2": "hold", "MAP-3": "hold", "MAP-4": "hold", "MAP-5": "hold", "BOX-1": "hold", "BOX-2": "hold", "BOX-3": "hold", "BOX-4": "hold",
                "FAM-1": "all", "FAM-2": "hold", "W1": "all", "W2": "all", "W4": "all", "W5": "all"}


def table_view(bars=None):
    bars = V.BARS if bars is None else bars
    return {(b, s["id"]): (s["kind"], s.get("value"), s.get("key")) for b, spec in bars.items() for s in spec["subs"]}


def test_bars_table_is_pinned_to_section5_and_seal():
    assert table_view() == PINNED
    assert {b for b, s in V.BARS.items() if s["required"]} == PINNED_REQUIRED and not V.BARS["BOX-4"]["required"]
    assert {b for b, s in V.BARS.items() if s["planner"]} == PINNED_PLANNER
    assert {b: s["split"] for b, s in V.BARS.items()} == PINNED_SPLIT
    assert V.SCORED_EMA_STEP == 73280 and V.SEEDS == ("0", "1")
    # the PI amendment (2026-11-08, "Use 0.9 bar"): MAP-2 5-10 and 10-20 m are 0.90 -- not the 0.99 of the earlier seal line, not the provisional 0.85 of the design text
    m2 = {s["id"]: s["value"] for s in V.BARS["MAP-2"]["subs"] if "value" in s}
    assert m2["p_5_10_ge"] == 0.90 and m2["p_10_20_ge"] == 0.90 and m2["p_0_5_ge"] == 0.60
    assert 0.99 not in m2.values() and 0.85 not in m2.values()


@pytest.mark.parametrize("bar,sub,newval", [("MAP-2", "p_5_10_ge", 0.99), ("MAP-2", "p_10_20_ge", 0.85), ("MAP-1", "rate_le", 0.179), ("BOX-1", "ap_ge", 0.328), ("BOX-3", "bpo_le", 2.14),
                                            ("MAP-3", "bnd_recall_ge", 0.29), ("BOX-2", "ap_40_60_ge", 0.092)])
def test_a_moved_bar_turns_the_pin_red(bar, sub, newval):
    moved = copy.deepcopy(V.BARS)
    next(s for s in moved[bar]["subs"] if s["id"] == sub)["value"] = newval
    assert table_view(moved) != PINNED                                  # the pin test above would FAIL on this table


def test_a_moved_bar_changes_verdicts_because_the_table_is_the_only_source(tmp_path, monkeypatch):
    rc, v, _ = run(tmp_path, good())
    assert rc == 0
    monkeypatch.setitem(next(s for s in V.BARS["MAP-2"]["subs"] if s["id"] == "p_5_10_ge"), "value", 0.99)      # the earlier, superseded value: the good case (0.93) now misses
    (tmp_path / "x").mkdir()
    rc, v, _ = run(tmp_path / "x", good())
    assert rc == 1 and [f["id"] for f in v["failing_bars"]] == ["MAP-2"]


def _grep(path, *needles):
    txt = path.read_text(encoding="utf-8")
    missing = [n for n in needles if n not in txt]
    assert not missing, (str(path), missing)


@pytest.mark.skipif(not (PR / "REFCV9_DESIGN.md").is_file(), reason="the design doc is not in this checkout (tests run on a stack+taniteval export)")
def test_table_matches_design_section5_text():
    _grep(PR / "REFCV9_DESIGN.md", "≤ 9.8 %", "0-5 m ≥ 0.60", "≥ 1.5 × refcv8", "≥ 0.50", "≥ 0.45", "0-20 m ≥ 0.62; 20-40 m ≥ 0.32; 40-60 m ≥ 0.15", "≤ 1.25", "0.822")


@pytest.mark.skipif(not (PR / "e3" / "SEAL_DECISIONS_PRE_DATA_2026-11-08.md").is_file(), reason="seal decisions not in this checkout")
def test_table_matches_seal_decisions_and_pi_amendment():
    _grep(PR / "e3" / "SEAL_DECISIONS_PRE_DATA_2026-11-08.md", "Use 0.9 bar", "The binding value is ≥ 0.90 at 5-10 m and at 10-20 m", "SIDEWALK-ONLY")


# ------------------------------------------------------------------------------------------------------------- fixtures
EST = "paired_episode_cluster_bootstrap_stat"


def prec(delta, sep=True, better=None, est=EST):
    r = {"delta": delta, "lo": delta - 0.01 if sep else delta - 1.0, "hi": delta + 0.01 if sep else delta + 1.0, "separated": bool(sep), "estimator": est, "n_windows": 900, "n_episodes": 68}
    if better:
        r["better"] = better
    return r


def good_perception():
    boxes = {"ap2m_all": 0.50, "ap2m_0_20": 0.68, "ap2m_20_40": 0.36, "ap2m_40_60": 0.18}
    e9 = {"split": "hold", "tier": "T0 (label level)", "n_windows": 1578,
          "boxes": {**{k: {"point": v, "lo": v - 0.03, "hi": v + 0.03} for k, v in boxes.items()},
                    "boxes_per_object_noNMS": {"point": 1.10, "lo": 1.0, "hi": 1.2}, "tp_class_acc": {"point": 0.84, "lo": 0.82, "hi": 0.86}},
          "paired_vs_B": {"boxes": {k: prec(0.1) for k in boxes}},
          "nearfield": {"ci": {"nf_prec_lv_0_5": {"point": 0.70}, "nf_prec_5_10": {"point": 0.93}, "nf_prec_10_20": {"point": 0.92}, "lane_tiou05_0_20": {"point": 0.40},
                               "drv_bnd_rec03_0_20": {"point": 0.55}}, "unobs_flags": {"sidewalk": 0, "edge": 0}}}
    e8 = {"split": "hold", "tier": "T0 (label level)", "nearfield": {"ci": {"lane_tiou05_0_20": {"point": 0.20}}}}
    mp = {"split": "hold", "tier": "T0 (label level)", "rows": {"nf_prec_20_40": prec(0.05), "lane_tiou05_0_20": prec(0.2),
                                                                 **{k: prec(0.01, False) for k in V.MAP4_ROWS}}}
    return {"eval_r9": e9, "eval_r8": e8, "map_paired": mp}


def good_seed(s):
    st = {"split": "all", "tier": "T1 self-action open loop", "infer_seed": int(s)}
    fam = {**st, "rows": {"LONGITUDINAL": {"ade_m": prec(-0.2, True, "lower"), "speed_err": prec(0.0, False, "lower")}, "LATERAL": {"curv_err": prec(-0.01, False, "lower")},
                          "TACTICAL": {"decision_acc": prec(0.02, False, "higher")}}}
    floor = {**st, "rows": {"ade_m": prec(-0.5, True, "lower"), "LONGITUDINAL": prec(-0.3, True, "lower"), "LATERAL": prec(-0.2, True, "lower"), "TACTICAL": prec(0.1, True, "higher")}}
    return {"s6c_r9": {"tier": "T1 self-action open loop", "infer_seed": int(s), "split": "/x/DEV_HOLD_SPLIT.json", "HOLD": {"subset": "hold", "rate": 0.08, "n": 1578, "bootstrap": {"lo": 0.05, "hi": 0.11}},
                       "salt": s},
            "compliance": {**st, "split": "hold", "rows": {"off_drivable": prec(-0.09), "edge_contact": prec(-0.1), "lane_cross": prec(-0.05)}, "salt": s},
            "families": fam, "camblind": floor, "egofloor": copy.deepcopy(floor),
            "distkeep": {**st, "rows": {"LONGITUDINAL": {k: prec(0.5, False, "higher") for k in V.DK_ROWS}}, "salt": s},
            "w12": {**st, "rows": {"FD": prec(-0.03), "MD": prec(0.0, False)}, "salt": s}}


def good():
    return {"perc": good_perception(), "0": good_seed("0"), "1": good_seed("1"), "kbase": {"sha256_match": True, "forward_refit_reproduces": True}}


def both(D, kind, fn):
    for s in ("0", "1"):
        fn(D[s][kind])


def write(tmp, D, extra=None):
    inp = {"refcv9_label": "refcv9", "reference_label": "refcv8", "scored_ema_step": 73280, "perception": {}, "seeds": {"0": {}, "1": {}}}
    for k, d in D["perc"].items():
        (tmp / f"perc_{k}.json").write_text(json.dumps(d), encoding="utf-8")
        inp["perception"][k] = str(tmp / f"perc_{k}.json")
    for s in ("0", "1"):
        for k, d in D[s].items():
            (tmp / f"s{s}_{k}.json").write_text(json.dumps(d), encoding="utf-8")
            inp["seeds"][s][k] = str(tmp / f"s{s}_{k}.json")
    (tmp / "kbase.json").write_text(json.dumps(D["kbase"]), encoding="utf-8")
    inp["kbase"] = str(tmp / "kbase.json")
    inp.update(extra or {})
    p = tmp / "INPUTS.json"
    p.write_text(json.dumps(inp), encoding="utf-8")
    return p


def run(tmp, D, extra=None):
    out = tmp / "out"
    rc = V.main(["--inputs", str(write(tmp, D, extra)), "--out-dir", str(out)])
    if rc == 2:
        return rc, None, None
    return rc, json.loads((out / "E3_VERDICT.json").read_text(encoding="utf-8")), (out / "E3_VERDICT.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------- (2) the all-pass case
def test_all_pass(tmp_path):
    rc, v, md = run(tmp_path, good())
    assert rc == 0 and v["verdict"] == "PASS" and v["failing_bars"] == [] and v["not_computable_bars"] == [] and v["k_base"] == "TRUE"
    assert {b for b, x in v["bars"].items() if x["status"] == "PASS"} == PINNED_REQUIRED and v["bars"]["BOX-4"]["status"] == "BESIDE"
    assert "E3 final-read verdict: PASS" in md and "episode-cluster" in v["estimator"] and "overlapping_holdout_se" in v["estimator"]       # named only to exclude it
    # every number carries its provenance (path + sha256 + split + tier) and its estimator
    ev = v["bars"]["BOX-1"]["subs"][0]["per_seed"]["n/a (deterministic, read once on HOLD)"]["evidence"]
    assert ev["sha256"] and ev["split"] == "hold" and ev["tier"].startswith("T0") and ev["path"].endswith("perc_eval_r9.json")
    row = v["bars"]["BOX-1"]["subs"][1]["per_seed"]["n/a (deterministic, read once on HOLD)"]["row"]
    assert "episode_cluster" in row["estimator"]
    ev1 = v["bars"]["MAP-1"]["subs"][0]["per_seed"]["1"]["evidence"]
    assert ev1["seed"] == "1" and ev1["tier"].startswith("T1") and ev1["split"] == "hold"
    assert set(v["bars"]["W4"]["subs"][0]["per_seed"]) == {"0", "1"}


# ------------------------------------------------------------------------------------------------------------- (3) one RED arm per bar
def f_map1_rate(D): both(D, "s6c_r9", lambda d: d["HOLD"].__setitem__("rate", 0.12))
def f_map1_sep(D): both(D, "compliance", lambda d: d["rows"].__setitem__("off_drivable", prec(-0.01, False)))
def f_map2_05(D): D["perc"]["eval_r9"]["nearfield"]["ci"]["nf_prec_lv_0_5"]["point"] = 0.55
def f_map2_510(D): D["perc"]["eval_r9"]["nearfield"]["ci"]["nf_prec_5_10"]["point"] = 0.899
def f_map2_1020(D): D["perc"]["eval_r9"]["nearfield"]["ci"]["nf_prec_10_20"]["point"] = 0.7
def f_map2_2040(D): D["perc"]["map_paired"]["rows"]["nf_prec_20_40"] = prec(0.01, False)
def f_map3_ratio(D): D["perc"]["eval_r9"]["nearfield"]["ci"]["lane_tiou05_0_20"]["point"] = 0.29        # < 1.5 x 0.20 = 0.30
def f_map3_rec(D): D["perc"]["eval_r9"]["nearfield"]["ci"]["drv_bnd_rec03_0_20"]["point"] = 0.45
def f_map4(D): D["perc"]["map_paired"]["rows"]["drv_iou_10_20"] = prec(-0.05, True)
def f_map5(D): D["perc"]["eval_r9"]["nearfield"]["unobs_flags"]["edge"] = 1
def f_box1(D): D["perc"]["eval_r9"]["boxes"]["ap2m_all"]["point"] = 0.44
def f_box2(D): D["perc"]["eval_r9"]["paired_vs_B"]["boxes"]["ap2m_40_60"] = prec(0.001, False)
def f_box3(D): D["perc"]["eval_r9"]["boxes"]["boxes_per_object_noNMS"]["point"] = 1.26
def f_fam1(D): both(D, "families", lambda d: d["rows"]["LATERAL"].__setitem__("curv_err", prec(0.05, True, "lower")))
def f_fam2_comp(D): both(D, "compliance", lambda d: d["rows"].__setitem__("lane_cross", prec(-0.001, False)))
def f_fam2_dk(D): both(D, "distkeep", lambda d: d["rows"]["LONGITUDINAL"].__setitem__("LON_dk_min_ttc_s", prec(-2.0, True, "higher")))
def f_w1(D): both(D, "w12", lambda d: d["rows"].__setitem__("FD", prec(0.0, False)))
def f_w2(D): both(D, "w12", lambda d: d["rows"].__setitem__("MD", prec(0.2, True)))
def f_w4(D): both(D, "camblind", lambda d: d["rows"].__setitem__("TACTICAL", prec(0.01, False, "higher")))
def f_w5(D): both(D, "egofloor", lambda d: d["rows"].__setitem__("ade_m", prec(-0.01, False, "lower")))


@pytest.mark.parametrize("bar,flip", [("MAP-1", f_map1_rate), ("MAP-1+FAM-2", f_map1_sep), ("MAP-2", f_map2_05), ("MAP-2", f_map2_510), ("MAP-2", f_map2_1020), ("MAP-2", f_map2_2040),
                                      ("MAP-3", f_map3_ratio), ("MAP-3", f_map3_rec), ("MAP-4", f_map4), ("MAP-5", f_map5), ("BOX-1", f_box1), ("BOX-2", f_box2), ("BOX-3", f_box3),
                                      ("FAM-1", f_fam1), ("FAM-2", f_fam2_comp), ("FAM-2", f_fam2_dk), ("W1", f_w1), ("W2", f_w2), ("W4", f_w4), ("W5", f_w5)])
def test_each_bar_red_arm_names_that_bar_only(tmp_path, bar, flip):
    D = good()
    flip(D)
    rc, v, md = run(tmp_path, D)
    assert rc == 1 and v["verdict"] == "FAIL"
    assert [f["id"] for f in v["failing_bars"]] == bar.split("+"), v["failing_bars"]          # MAP-1's paired row IS FAM-2's off_drivable row
    assert v["failing_bars"][0]["next_lever"] and f"FAILED as written: {bar.split('+')[0]}" in md


def test_map2_boundary_is_inclusive_at_0_90(tmp_path):
    D = good()
    D["perc"]["eval_r9"]["nearfield"]["ci"]["nf_prec_5_10"]["point"] = 0.90
    D["perc"]["eval_r9"]["nearfield"]["ci"]["nf_prec_10_20"]["point"] = 0.90
    assert run(tmp_path, D)[0] == 0


def test_box4_beside_never_changes_the_verdict(tmp_path):
    D = good()
    D["perc"]["eval_r9"]["boxes"]["tp_class_acc"] = {"point": 0.70, "lo": 0.69, "hi": 0.71}
    rc, v, _ = run(tmp_path, D)
    assert rc == 0 and v["bars"]["BOX-4"]["status"] == "BESIDE" and "NOT met" in v["bars"]["BOX-4"]["subs"][0]["per_seed"]["n/a (deterministic, read once on HOLD)"]["detail"]


# ------------------------------------------------------------------------------------------------------------- (4) the deliberate regressions
def test_dropping_a_family_is_incomplete_never_a_pass(tmp_path):
    for fam in V.FAMILIES:
        D = good()
        both(D, "families", lambda d: d["rows"].pop(fam))
        (tmp_path / fam).mkdir()
        rc, v, _ = run(tmp_path / fam, D)
        assert rc == 1 and v["verdict"] == "INCOMPLETE" and "FAM-1" in v["not_computable_bars"], fam
        for s in ("0", "1"):
            assert v["four_families"][s][fam]["status"] == "NOT-COMPUTABLE" and v["four_families"][s][fam]["n_rows"] == 0 and v["four_families"][s][fam]["reason"]
            assert "STRATEGIC" in v["four_families"][s] and set(v["four_families"][s]) == set(V.FAMILIES) | {"STRATEGIC"}


def test_all_four_families_are_always_reported(tmp_path):
    rc, v, md = run(tmp_path, good())
    for s in ("0", "1"):
        assert set(v["four_families"][s]) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
        assert all(v["four_families"][s][f]["n_rows"] > 0 for f in V.FAMILIES)
    assert "distance_keeping_not_worse" in {s["id"] for s in v["bars"]["FAM-2"]["subs"]}


def assert_strategic_paused(v):
    for s in ("0", "1"):
        st = v["four_families"][s]["STRATEGIC"]
        assert st["status"] == "PAUSED" and "ITEM 25" in st["reason"]
    assert v["strategic"].startswith("PAUSED")


def test_strategic_is_paused_never_pass_never_omitted(tmp_path):
    D = good()
    both(D, "families", lambda d: d["rows"].__setitem__("STRATEGIC", {"x": prec(-9.0, True, "lower")}))      # a supplied strategic row can never turn the pause into a pass
    rc, v, md = run(tmp_path, D)
    assert_strategic_paused(v)
    assert "STRATEGIC: PAUSED" in md


def test_strategic_marked_pass_turns_the_check_red(tmp_path):
    rc, v, _ = run(tmp_path, good())
    bad = copy.deepcopy(v)
    bad["four_families"]["0"]["STRATEGIC"]["status"] = "PASS"
    with pytest.raises(AssertionError):
        assert_strategic_paused(bad)
    bad = copy.deepcopy(v)
    del bad["four_families"]["1"]["STRATEGIC"]
    with pytest.raises(KeyError):
        assert_strategic_paused(bad)


def test_strategic_pass_in_the_judge_would_be_caught(tmp_path, monkeypatch):
    """Mutate the JUDGE: a families_report that marks STRATEGIC PASS must make the pause check RED."""
    orig = V.families_report

    def mutated(fdoc):
        r = orig(fdoc)
        r["STRATEGIC"] = {"status": "PASS", "reason": "mutant", "rows": {}}
        return r

    monkeypatch.setattr(V, "families_report", mutated)
    rc, v, _ = run(tmp_path, good())
    with pytest.raises(AssertionError):
        assert_strategic_paused(v)


# ------------------------------------------------------------------------------------------------------------- (5) refusals: rc 2, no verdict
def dev(d): d["split"] = "dev"


def test_dev_input_is_refused_for_a_hold_bar(tmp_path):
    for kind in ("eval_r9", "eval_r8", "map_paired"):
        D = good()
        dev(D["perc"][kind])
        (tmp_path / kind).mkdir()
        assert run(tmp_path / kind, D)[0] == 2, kind
    D = good()
    D["0"]["s6c_r9"]["HOLD"]["subset"] = "dev"
    (tmp_path / "s6c").mkdir()
    assert run(tmp_path / "s6c", D)[0] == 2
    D = good()
    dev(D["1"]["compliance"])
    (tmp_path / "comp").mkdir()
    assert run(tmp_path / "comp", D)[0] == 2
    D = good()
    dev(D["0"]["families"])
    (tmp_path / "fam").mkdir()
    assert run(tmp_path / "fam", D)[0] == 2


def test_hold_only_s6c_dump_with_dev_beside_is_read_on_its_hold_block(tmp_path):
    D = good()
    D["0"]["s6c_r9"]["DEV_beside"] = {"subset": "dev", "rate": 0.99}                  # s6c_hold.py writes this beside HOLD; it must never be read
    rc, v, _ = run(tmp_path, D)
    assert rc == 0 and v["bars"]["MAP-1"]["subs"][0]["per_seed"]["0"]["point"] == 0.08


def test_unstamped_split_or_tier_is_refused(tmp_path):
    cases = [("perc", "eval_r9", "split"), ("perc", "eval_r9", "tier"), ("0", "families", "split"), ("1", "w12", "tier"), ("0", "s6c_r9", "tier")]
    for i, (a, k, field) in enumerate(cases):
        D = good()
        if field == "split" and k == "s6c_r9":
            del D[a][k]["HOLD"]
        D[a][k].pop(field, None)
        (tmp_path / str(i)).mkdir()
        assert run(tmp_path / str(i), D)[0] == 2, (a, k, field)


def test_wrong_tier_and_split_for_the_bar_are_refused(tmp_path):
    D = good()
    D["perc"]["eval_r9"]["tier"] = "T1 self-action open loop"           # perception bars are T0
    assert run(tmp_path, D)[0] == 2
    D = good()
    D["0"]["families"]["split"] = "hold"                                # FAM-1 / W-bars are read on all of eval139 (design 5.1)
    (tmp_path / "b").mkdir()
    assert run(tmp_path / "b", D)[0] == 2


def test_planner_input_must_carry_its_seed_slot(tmp_path):
    D = good()
    D["1"]["families"]["infer_seed"] = 0
    assert run(tmp_path, D)[0] == 2
    D = good()
    del D["0"]["s6c_r9"]["infer_seed"]
    (tmp_path / "b").mkdir()
    assert run(tmp_path / "b", D)[0] == 2


def test_one_file_for_both_seeds_is_refused(tmp_path):
    D = good()
    D["1"]["w12"] = copy.deepcopy(D["0"]["w12"])
    D["1"]["w12"]["infer_seed"] = 0                                     # same bytes in slot 1 => also the wrong seed; make the bytes identical AND stamped for slot 0
    assert run(tmp_path, D)[0] == 2


def test_overlapping_holdout_estimator_is_refused(tmp_path):
    for est in ("overlapping_holdout_se", "8-split jackknife", None):
        D = good()
        D["perc"]["eval_r9"]["paired_vs_B"]["boxes"]["ap2m_all"]["estimator"] = est
        if est is None:
            del D["perc"]["eval_r9"]["paired_vs_B"]["boxes"]["ap2m_all"]["estimator"]
        (tmp_path / str(est)[:4]).mkdir()
        assert run(tmp_path / str(est)[:4], D)[0] == 2, est


def test_an_input_cannot_override_a_bar(tmp_path):
    assert run(tmp_path, good(), {"bars": {"MAP-2": 0.1}})[0] == 2                          # INPUTS.json key set is closed
    (tmp_path / "t").mkdir()
    assert run(tmp_path / "t", good(), {"thresholds": {"BOX-1": 0.1}})[0] == 2
    D = good()                                                                              # a bar smuggled inside a producer doc is ignored: the table decides
    D["perc"]["eval_r9"]["bars"] = {"BOX-1": 0.0}
    D["perc"]["eval_r9"]["boxes"]["ap2m_all"]["bar"] = 0.0
    D["perc"]["eval_r9"]["boxes"]["ap2m_all"]["point"] = 0.40
    (tmp_path / "d").mkdir()
    rc, v, _ = run(tmp_path / "d", D)
    assert rc == 1 and [f["id"] for f in v["failing_bars"]] == ["BOX-1"]


def test_only_ema_73280_is_a_candidate(tmp_path):
    assert run(tmp_path, good(), {"scored_ema_step": 48852})[0] == 2


def test_both_seeds_required(tmp_path):
    D = good()
    p = write(tmp_path, D)
    j = json.loads(p.read_text(encoding="utf-8"))
    del j["seeds"]["1"]
    p.write_text(json.dumps(j), encoding="utf-8")
    assert V.main(["--inputs", str(p), "--out-dir", str(tmp_path / "o")]) == 2
    assert V.main(["--inputs", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")]) == 2


# ------------------------------------------------------------------------------------------------------------- (6) seeds, K-BASE, NOT-COMPUTABLE
def test_seed1_failing_alone_fails_the_planner_bar_and_names_the_seed(tmp_path):
    D = good()
    D["1"]["s6c_r9"]["HOLD"]["rate"] = 0.15
    rc, v, md = run(tmp_path, D)
    assert rc == 1 and [f["id"] for f in v["failing_bars"]] == ["MAP-1"]
    sub = v["bars"]["MAP-1"]["subs"][0]
    assert sub["per_seed"]["0"]["status"] == "PASS" and sub["per_seed"]["1"]["status"] == "FAIL" and sub["seeds_not_pass"] == ["1"]
    assert "| 1 |" in md.split("MAP-1")[1].split("\n")[0]


def test_perception_bars_are_read_once_not_per_seed(tmp_path):
    rc, v, _ = run(tmp_path, good())
    assert all(list(s["per_seed"]) == ["n/a (deterministic, read once on HOLD)"] for s in v["bars"]["BOX-1"]["subs"])
    assert all(set(s["per_seed"]) == {"0", "1"} for b in PINNED_PLANNER for s in v["bars"][b]["subs"])


def test_missing_producer_output_is_incomplete_not_pass(tmp_path):
    D = good()
    del D["perc"]["eval_r9"]["nearfield"]["unobs_flags"]                 # r9_e1_eval does not emit it yet (open producer item)
    rc, v, _ = run(tmp_path, D)
    assert rc == 1 and v["verdict"] == "INCOMPLETE" and v["not_computable_bars"] == ["MAP-5"]
    D = good()
    both(D, "distkeep", lambda d: d["rows"]["LONGITUDINAL"].pop("LON_dk_headway_min_m"))
    (tmp_path / "b").mkdir()
    rc, v, _ = run(tmp_path / "b", D)
    assert v["verdict"] == "INCOMPLETE" and v["not_computable_bars"] == ["FAM-2"]
    D = good()
    D["0"].pop("w12")                                                    # an absent file is NOT-COMPUTABLE (named), never dropped
    (tmp_path / "c").mkdir()
    p = write(tmp_path / "c", D)
    j = json.loads(p.read_text(encoding="utf-8"))
    assert "w12" not in j["seeds"]["0"]
    assert V.main(["--inputs", str(p), "--out-dir", str(tmp_path / "c" / "o")]) == 1
    v = json.loads((tmp_path / "c" / "o" / "E3_VERDICT.json").read_text(encoding="utf-8"))
    assert v["not_computable_bars"] == ["W1", "W2"]


def test_a_fail_outranks_not_computable(tmp_path):
    D = good()
    del D["perc"]["eval_r9"]["nearfield"]["unobs_flags"]
    f_box1(D)
    rc, v, _ = run(tmp_path, D)
    assert v["verdict"] == "FAIL" and [f["id"] for f in v["failing_bars"]] == ["BOX-1"] and v["not_computable_bars"] == ["MAP-5"]


def test_kbase_must_reproduce_or_the_bars_are_void(tmp_path):
    D = good()
    D["kbase"] = {"sha256_match": True, "forward_refit_reproduces": None}       # hash alone does not close it (seal decision)
    rc, v, _ = run(tmp_path, D)
    assert rc == 1 and v["verdict"] == "INCOMPLETE" and v["k_base"] == "OPEN"
    D["kbase"] = {"sha256_match": True, "forward_refit_reproduces": False}
    (tmp_path / "b").mkdir()
    rc, v, _ = run(tmp_path / "b", D)
    assert rc == 1 and v["verdict"] == "VOID" and v["k_base"] == "FALSE"


def test_per_window_rows_use_ci_paired_bootstrap(tmp_path):
    """The per-window form is scored by taniteval.ci (no new estimator): refcv9 uniformly worse by 1.0 on a `lower`-better row => separated worse."""
    eid = [e for e in range(12) for _ in range(5)]
    a = [2.0 + 0.01 * i for i in range(len(eid))]
    b = [x - 1.0 for x in a]
    r = V.row_read({"a": a, "b": b, "eid": eid, "better": "lower"})
    assert r["ok"] and r["separated_worse"] and not r["separated_better"] and r["src"].startswith("ci.paired") and "episode_cluster" in r["estimator"]
    r2 = V.row_read({"a": b, "b": a, "eid": eid, "better": "lower"})
    assert r2["ok"] and r2["separated_better"]
    assert V.row_read({"a": [1.0], "b": [1.0, 2.0], "eid": [0], "better": "lower"})["ok"] is False
    D = good()
    both(D, "w12", lambda d: d["rows"].__setitem__("FD", {"a": b, "b": a, "eid": eid}))              # direction comes from the TABLE (FD: lower is better)
    assert run(tmp_path, D)[0] == 0
