"""TanitAD's own REFe arms on ``PDMS_v1_navtest`` -- the board must SHOW them, from the raw artifacts, and say what they read.

⛔ WHY (2026-10-18; leaderboard currency): §0E.3 printed ``TanitAD — any arm | ⛔ NOT MEASURED — NO TanitAD ARM yet`` while FIVE
REFe states had been scored on the FULL navtest (12,146 tokens / 136 logs, NAVSIM v1.1): the deployed system (84.46), SFT-4 u400
(88.25), P2 (88.82) and its seed replicate P2-s1 (89.15), plus the superseded route-fix-OFF state (83.24). The position block carried
the same stale sentence ("No TanitAD arm is on this column either"), and its headline said "at any tier".

What these tests refuse to let rot:

1. every printed number of a REFe row is the raw artifact's -- written below as LITERALS for P2 (an expectation computed from the
   code under test measures determinism, not correctness) and re-read from the raw JSON, by code that shares nothing with
   ``sources.py``, for every other arm;
2. the DECLARED INPUTS are true of the model source: 4 cameras, a 7-D ego vector that includes yaw rate and steering angle, a route
   goal. The row must say "NOT vision-only"; the goal variant must follow the arm's OWN seam-report field ``goal_fix``;
3. a replicate verdict is printed from its artifact, not typed -- flip the artifact and the page flips (deliberate-regression arm);
4. an interval that belongs to a different arm, a tier that is not NAVSIM v1.1 open loop, or a goal_fix with no declared-input text
   is REFUSED rather than printed;
5. the old sentences return when no REFe arm is loaded (the fallback arm), so the generator is still honest on a checkout without them.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_TE = _HERE.parent
REPO = _TE.parent
if str(_TE) not in sys.path:
    sys.path.insert(0, str(_TE))

from taniteval.leaderboard import render as R          # noqa: E402
from taniteval.leaderboard import sources as S         # noqa: E402
from taniteval.leaderboard.build import load_model     # noqa: E402

PROTO = "PDMS_v1_navtest"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
RAW = PK + "/raw"
S2 = RAW + "/2026-10-05-sft-stage2"
CFG = S.load_config(REPO)
PAGE = REPO / CFG["target_md"]
SPEC = next(s for s in CFG["legacy_bench"] if s["id"] == "refe_navtest_arms")
pytestmark = pytest.mark.skipif(not (REPO / RAW / "p2_readout" / "p2_full" / "r2_result.json").exists(),
                                reason="the REFe navtest artifacts are not in this checkout")


def J(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return load_model(REPO, CFG)


@pytest.fixture(scope="module")
def section(model):
    return R.render_md(model, CFG)


def _v1(section: str) -> str:
    return section[section.index("### 0E.3"):section.index("### 0E.4")]


def _rows(v1: str) -> dict:
    """{short name: cells} for the REFe LEVEL table -- an independent parser (no import of the renderer's helpers)."""
    out = {}
    for ln in v1.splitlines():
        m = re.match(r"^\| \*\*TanitAD — (REFe[^*]*)\*\* — ", ln)
        if m and "ours (MEASURED)" not in ln:
            out[m.group(1)] = [c.strip() for c in re.split(r" \| ", ln.strip().strip("|").strip())]
    return out


def _f(s: str) -> float:
    return float(s.replace("−", "-").replace("+", "").strip("*` "))


def _expected_from_raw() -> dict:
    """{short name: (PDMS, lo, hi, {NC..EP})} read straight from the raw JSON -- shares no code with sources.py."""
    mci, mci1 = J(RAW + "/2026-10-17-paper-s720-recompute/marginal_ci.json")["rows"], \
        J("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-leaderboard-refe-navtest/raw/marginal_ci_p2s1.json")["rows"]
    parse = J(RAW + "/2026-10-04-navtest-final/routefix_full/3_parse.json")["arms"]
    h = lambda p: J(S2 + p)["harness_status"]["summary_x100_4dp"]       # noqa: E731
    sub = lambda d: {k: d[k] for k in ("NC", "DAC", "DDC", "TTC", "C", "EP")}   # noqa: E731
    return {
        "REFe model_final (route fix OFF)": (parse["R6_final_off"]["PDMS"], mci["model_final"]["lo"], mci["model_final"]["hi"], sub(parse["R6_final_off"])),
        "REFe deployed system (model_final + route fix)": (parse["R6_A1"]["PDMS"], mci["model_final_routefix_A9"]["lo"], mci["model_final_routefix_A9"]["hi"], sub(parse["R6_A1"])),
        "REFe SFT-4 u400": (h("/sft4_B_u400_navtest_full/result.json")["PDMS"], mci["sft4_B_u400"]["lo"], mci["sft4_B_u400"]["hi"], sub(h("/sft4_B_u400_navtest_full/result.json"))),
        "REFe P2 seed 0": (h("/p2_full/result.json")["PDMS"], mci["p2_s0"]["lo"], mci["p2_s0"]["hi"], sub(h("/p2_full/result.json"))),
        "REFe P2 seed 1 (replicate)": (h("/p2s1_full/result.json")["PDMS"], mci1["p2_s1"]["lo"], mci1["p2_s1"]["hi"], sub(h("/p2s1_full/result.json"))),
    }


def _errors(v1: str) -> list:
    """Every printed number of every REFe level row against the raw artifact. [] means all trace."""
    rows, exp, err = _rows(v1), _expected_from_raw(), []
    if set(rows) != set(exp):
        return [f"arms on the page {sorted(rows)} != arms in the artifacts {sorted(exp)}"]
    for name, (pdms, lo, hi, sub) in exp.items():
        c = rows[name]
        if abs(_f(c[3]) - pdms) > 5e-5:
            err.append((name, "PDMS", c[3], pdms))
        m = re.search(r"\[([\d.]+), ([\d.]+)\]", c[4])
        if not m or abs(float(m.group(1)) - lo) > 5e-3 or abs(float(m.group(2)) - hi) > 5e-3:
            err.append((name, "CI", c[4], (lo, hi)))
        if c[5] != "12,146":
            err.append((name, "n", c[5], 12146))
        for k, cell in zip(("NC", "DAC", "DDC", "TTC", "C", "EP"), c[6:12]):
            if abs(_f(cell) - sub[k]) > 5e-3:
                err.append((name, k, cell, sub[k]))
    return err


# ------------------------------------------------------------------------------------------------ 1. the numbers
def test_p2_row_is_the_raw_artifact_as_literals(section):
    """P2 seed 0, read by hand from result.json / marginal_ci.json / r2_result.json 2026-10-18 -- literals on purpose."""
    c = _rows(_v1(section))["REFe P2 seed 0"]
    assert c[3] == "88.8184" and c[4].startswith("[87.86, 89.77]") and c[5] == "12,146"
    assert c[6:12] == ["98.81", "96.87", "97.51", "95.32", "99.98", "82.34"]       # NC DAC DDC TTC C EP
    v1 = _v1(section)
    assert "| P2 seed 0 minus SFT-4 u400 (R2, the P6 GO/NO-GO) | +0.57 [+0.02, +1.18] | 5,449 / 3,519 / 3,178 | none | GO |" in v1
    assert "| P2 seed 1 minus SFT-4 u400 (RR1) | +0.90 [+0.36, +1.59] | 5,945 / 3,507 / 2,694 | TTC | NO-GO (adverse sub-score) |" in v1


def test_every_refe_row_traces_to_its_raw_artifact(section):
    assert _errors(_v1(section)) == []


def test_the_independent_trace_goes_red_on_a_swapped_value(section):
    """⛔ a check that cannot fail is not a check: swap two PDMS cells and the trace must report it."""
    v1 = _v1(section)
    a, b = "| 88.8184 |", "| 84.4561 |"
    assert a in v1 and b in v1
    bad = v1.replace(a, "| @@ |").replace(b, a).replace("| @@ |", b)
    assert _errors(bad), "the trace is BLIND to a swapped PDMS"


def test_five_arms_are_on_the_board_and_the_placeholder_is_gone(section):
    v1 = _v1(section)
    assert len(_rows(v1)) == 5
    assert "NO TanitAD ARM yet" not in v1 and "TanitAD — any arm" not in v1
    assert sum(1 for ln in v1.splitlines() if ln.startswith("| **TanitAD — REFe") and "ours (MEASURED)" in ln) == 5


def test_the_level_interval_estimator_and_the_seed_statement_are_printed_verbatim(section):
    v1 = _v1(section)
    mci = J(RAW + "/2026-10-17-paper-s720-recompute/marginal_ci.json")
    assert mci["estimator"] in v1 and mci["answers"] in v1
    for e in (J(S2 + "/sft4_B_u400_navtest_full/result.json")["estimator"],
              J(RAW + "/p2_readout/p2_full/r2_result.json")["estimator"],
              J(RAW + "/2026-10-16-p2s1-read/out/read/p2rep_result.json")["estimator"]):
        assert e in v1, f"paired estimator not printed in the artifact's own words: {e[:60]}"


def test_p2s1_has_a_level_interval_only_through_its_controlled_sibling():
    """MODEL_REGISTRY 14.2 recorded 'interval NOT COMPUTED' for P2-s1; the sibling file adds it behind four asserted controls."""
    mci = J(RAW + "/2026-10-17-paper-s720-recompute/marginal_ci.json")
    sib = J(SPEC["level_ci"]["extra"]["path"])
    assert "p2_s1" not in mci["rows"], "marginal_ci.json now has a P2-s1 row -- read it from there and retire the sibling"
    assert sib["estimator"] == mci["estimator"] and sib["answers"] == mci["answers"]
    c = sib["controls"]
    assert c["C1_tokens_md5"] == c["C1_pin"] == "3a8b3653fd70891e50a230a37eb76ae8" and (c["C1_n_tokens"], c["C1_n_logs"]) == (12146, 136)
    assert c["C3_p2_s0_recomputed_equals_banked"]["equal"] is True
    f = ("n", "point", "lo", "hi", "simple_mean")
    rec, bank = c["C3_p2_s0_recomputed_equals_banked"]["recomputed"], c["C3_p2_s0_recomputed_equals_banked"]["banked"]
    assert all(rec[k] == bank[k] == mci["rows"]["p2_s0"][k] for k in f)
    assert c["C4_p2_s1_point_equals_harness_summary"]["point"] == round(J(S2 + "/p2s1_full/result.json")["harness_status"]["summary_x100_4dp"]["PDMS"], 4)
    assert all(v["md5_crlf_normalised_equals_pin"] for v in c["C2_inputs"].values())


# ------------------------------------------------------------------------------------------------ 2. declared inputs
def test_declared_inputs_are_true_of_the_model_source():
    """The cells say 4 cameras / 7-D ego / a route goal. refe/model.py and SPEC_NAVTEST.md must still say the same."""
    src = (REPO / SPEC["model_src"]).read_text(encoding="utf-8")
    assert re.search(r"n_cameras: int = 4\b", src) and "cameras: tuple = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0')" in src
    assert re.search(r"img_h: int = 512\b", src) and re.search(r"img_w: int = 960\b", src)
    assert re.search(r"ego_dim: int = 7\s+# vx, vy, ax, ay, yaw_rate, tire_steering_angle, speed", src)
    assert "n_goal_points: int = 2" in src
    spec = (REPO / SPEC["spec_doc"]).read_text(encoding="utf-8")
    assert "DEPARTURE 1" in spec and "NAVSIM's `ego_status` carries velocity + acceleration only" in spec
    assert "DEPARTURE 2" in spec and "optimistic by construction" in spec
    parts = SPEC["declared_parts"]
    assert "4 cameras (CAM_F0/L0/R0/B0, 512x960)" in parts["cameras"]
    assert all(w in parts["ego"] for w in ("vx", "vy", "ax", "ay", "yaw rate", "steering angle", "speed", "NOT vision-only"))


def test_the_pdm_route_text_is_pinned_to_its_result_and_the_pi_ruling():
    a9 = (REPO / RAW / "2026-10-01-goal-trigger" / "RESULT_A9_GOAL_FIX.md").read_text(encoding="utf-8")
    assert "the route NAVSIM's own privileged planner uses" in a9 and "map's roadblock-graph topology" in a9
    goals = REPO / "Project Steering" / "GOALS_AND_CLAIMS.md"
    if goals.exists():                       # 2.8 MB; a checkout without it still runs the rest
        g = goals.read_text(encoding="utf-8")
        i = g.index("| **D-REFE-GOAL-2** |")
        assert 'adopt the route fix' in g[i:i + 1500] and "DEPLOYED 2026-10-04" in g[i:i + 1500]


def test_the_published_field_rows_cite_what_the_registry_says_about_the_model():
    """encoder · data in the published-field table is typed from MODEL_REGISTRY 14.1; pin the words it rests on."""
    reg = (REPO / CFG["registry"]).read_text(encoding="utf-8")
    fr = SPEC["field_row"]
    assert "frozen **DINOv3 ViT-L/16**" in reg and "+ LoRA" in reg and "DINOv3 ViT-L/16 + LoRA" in fr["encoder_data"]
    assert "NAVSIM **navtrain**" in reg and "navtrain" in fr["encoder_data"]
    assert SPEC["registry_anchor"] in reg, "the registry section this block cites has moved"


def test_every_row_declares_ego_status_and_marks_the_route_oracle(model, section):
    n = next(x for x in model["notes"] if isinstance(x, dict) and x.get("render") == "refe_navtest")
    for a in n["arms"]:
        assert "NOT vision-only" in a["declared"] and "route oracle" in a["declared"] and a["declared"].count("2 route goal points") == 1
    v1 = _v1(section)
    assert "Not vision-only" in v1 and "all 5/5 rows read ego status" in v1
    field = [ln for ln in v1.splitlines() if ln.startswith("| **TanitAD — REFe") and "ours (MEASURED)" in ln]
    assert field and all("yes — 7-D kinematics" in ln and "route oracle" in ln for ln in field)


def test_the_goal_variant_follows_the_seam_reports_own_field(model):
    n = {a["key"]: a for a in next(x for x in model["notes"] if isinstance(x, dict) and x.get("render") == "refe_navtest")["arms"]}
    off, dep = n["REFE_FINAL_OFF"], n["REFE_DEPLOYED"]
    assert off["goal_fix"] is None and "LOGGED drive" in off["declared"] and "pdm_route" not in off["declared"]
    for k in ("REFE_DEPLOYED", "REFE_SFT4_U400", "REFE_P2_S0", "REFE_P2_S1"):
        assert n[k]["goal_fix"] == "pdm_route" and "goal_fix=pdm_route" in n[k]["declared"]
    assert dep["rule"] == "navsim_v1" and n["REFE_SFT4_U400"]["rule"] == "navsim_v1_lane"


def _mutating_loader(monkeypatch, fn):
    real = S.load_json

    def patched(path):
        d = real(path)
        out = fn(Path(path).as_posix(), copy.deepcopy(d))
        return d if out is None else out

    monkeypatch.setattr(S, "load_json", patched)


def test_an_unknown_goal_fix_is_refused_not_borrowed(monkeypatch):
    def mut(p, d):
        if p.endswith("p2_readout/p2_full/r2_result.json"):
            d["gates"]["seam_report"]["goal_fix"] = "navgoal_straight"
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="no declared-input text"):
        S.legacy_summaries(REPO, CFG)


def test_a_missing_goal_fix_on_a_non_off_arm_is_refused(monkeypatch):
    def mut(p, d):
        if p.endswith("p2_readout/p2_full/r2_result.json"):
            del d["gates"]["seam_report"]["goal_fix"]
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="states no goal_fix"):
        S.legacy_summaries(REPO, CFG)


def test_another_arms_interval_is_refused(monkeypatch):
    def mut(p, d):
        if p.endswith("2026-10-17-paper-s720-recompute/marginal_ci.json"):
            d["rows"]["p2_s0"]["point"] = d["rows"]["sft4_B_u400"]["point"]
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="not this arm's score"):
        S.legacy_summaries(REPO, CFG)


def test_a_tier_that_is_not_navsim_v1_open_loop_is_refused(monkeypatch):
    def mut(p, d):
        if p.endswith("p2_full/result.json") and "tier" in d:
            d["tier"] = "NAVSIM v2 EPDMS = closed loop"
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="not NAVSIM v1.1 PDMS open loop"):
        S.legacy_summaries(REPO, CFG)


def test_two_estimators_for_one_interval_column_are_refused(monkeypatch):
    def mut(p, d):
        if p.endswith("marginal_ci_p2s1.json"):
            d["estimator"] = "a different bootstrap"
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="one estimator"):
        S.legacy_summaries(REPO, CFG)


def test_a_sibling_interval_whose_controls_do_not_read_is_refused(monkeypatch):
    def mut(p, d):
        if p.endswith("marginal_ci_p2s1.json"):
            d["controls"]["C3_p2_s0_recomputed_equals_banked"]["equal"] = False
            return d
    _mutating_loader(monkeypatch, mut)
    with pytest.raises(S.SourceError, match="control C3"):
        S.legacy_summaries(REPO, CFG)


# ------------------------------------------------------------------------------------------------ 3. the replicate verdict
def test_the_replicate_verdict_is_printed_from_its_artifact(section):
    v1 = _v1(section)
    rep = J(RAW + "/2026-10-16-p2s1-read/out/read/p2rep_result.json")
    assert rep["outcome"] == "NOT REPLICATED" and f"reads `{rep['outcome']}`" in v1
    assert "seed 1 minus u400 is +0.90 [+0.36, +1.59] but TTC is separated below 0" in v1
    assert "F = 0.33" in v1 and "below the smaller seed gain 0.57" in v1
    assert "does not rescue the lever claim" in v1


def test_flip_the_replicate_artifact_and_the_page_flips(monkeypatch):
    """Deliberate-regression arm: the verdict sentence must come from the artifact, not from the template."""
    def mut(p, d):
        if p.endswith("p2rep_result.json"):
            d["outcome"] = "REPLICATED"
            d["outcome_detail"]["F_lt_min_delta"] = False
            return d
    _mutating_loader(monkeypatch, mut)
    v1 = _v1(R.render_md(load_model(REPO, CFG), CFG))
    assert "reads `REPLICATED`" in v1 and "NOT below the smaller seed gain" in v1 and "reads `NOT REPLICATED`" not in v1


# ------------------------------------------------------------------------------------------------ 4. the position block
def test_the_position_block_no_longer_says_no_tanitad_arm_is_on_navtest(section):
    pos = section[section.index("5 October position"):section.index("| claim |")]
    assert "at any tier" not in pos
    head = pos.splitlines()[2]
    assert "internal T1 / closed-loop table" in head and "NAVSIM v1 navtest is a different protocol" in head
    line = next(l for l in pos.splitlines() if "navtest either" in l)
    assert "STOP 61.82" in line and "CV 20.65" in line
    assert "No TanitAD arm is on this column either" not in line
    assert "TanitAD REFe arms ARE on this column" in line and "5 REFe states" in line and "ONE training seed per arm" in line
    assert "P2 seed 0 (the P2 of record) reads 88.82 [87.86, 89.77]" in line and "`NOT REPLICATED`" in line
    assert "not a vision-only, not a command-free entry" in line
    nav = next(l for l in pos.splitlines() if "OFFICIAL column (navhard" in l)
    assert "No TanitAD ARM is on this column yet" in nav, "navhard still has NO TanitAD arm; this change must not touch that sentence"


def test_the_old_sentences_return_when_no_refe_arm_is_loaded(model):
    """Fallback arm: drop the REFe note and the generator says what it always said -- it never invents an arm."""
    m = dict(model)
    m["notes"] = [x for x in model["notes"] if not (isinstance(x, dict) and x.get("render") == "refe_navtest")]
    sec = R.render_md(m, CFG)
    pos = sec[sec.index("5 October position"):sec.index("| claim |")]
    assert "No TanitAD arm beats every trivial floor it is read against — at any tier, in either loop." in pos
    assert "No TanitAD arm is on this column either" in next(l for l in pos.splitlines() if "navtest either" in l)
    v1 = _v1(sec)
    assert "TanitAD — any arm" in v1 and "TanitAD IS ON THIS PROTOCOL" not in v1


def test_the_published_field_table_carries_ours_after_the_published_rows(section):
    v1 = _v1(section)
    i = v1.index("*The published field on this protocol:*")
    tail = [ln for ln in v1[i:].splitlines() if ln.startswith("|")]
    ours = [k for k, ln in enumerate(tail) if ln.startswith("| **TanitAD — REFe")]
    assert ours == list(range(len(tail) - 5, len(tail))), "our five rows must close the table, after every published row"
    assert not any("any arm" in ln for ln in tail)


# ------------------------------------------------------------------------------------------------ 5. the committed page
@pytest.mark.skipif(not PAGE.exists(), reason="LEADERBOARD.md not in this checkout")
def test_the_committed_page_shows_the_refe_rows():
    """The page that ships must carry what the generator reads. (Whole-page byte equality is deliberately NOT asserted: the page is
    legacy-unfingerprinted and lags the other inputs; see the landing note. This pins ONLY the rows this change is about.)"""
    page = PAGE.read_text(encoding="utf-8").replace("\r\n", "\n")
    v1 = page[page.index("### 0E.3"):page.index("### 0E.4")]
    assert _errors(v1) == []
    assert "NO TanitAD ARM yet" not in v1 and "reads `NOT REPLICATED`" in v1
    pos = page[page.index("5 October position"):page.index("| claim |")]
    assert "TanitAD REFe arms ARE on this column" in pos and "at any tier" not in pos
