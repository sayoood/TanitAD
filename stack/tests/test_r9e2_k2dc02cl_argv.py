"""K2DC02cl (PREREG_K2DC02CL_2026-11-22): K2DC02's argv with EXACTLY three token changes (build agent for the MM, 2026-11-22).

  --w-map-hires 25.0 -> 3.0                 (the INITIAL weight; map_dose_ctl.pin_args requires it to lie inside the bounds)
  --map-hires-w-bounds 0.1,25 -> 2.5,3.5    (the PI's clamp, "Yes, use the clamp", 2026-11-22)
  --out  .../r9e2-K2DC02 -> .../r9e2-K2DC02cl

Everything else, the seed included (0), is K2DC02's. There is NO trainer / model code change: both flags already exist. The file is derived from r9e2_K2DC02.argv.json by that
token replacement, not composed by r9_e2_make_argv.py (no composer registration), so this test is the only thing holding the diff: it compares BOTH directions.
PI-APPROVED, CONDITIONAL on the K2DC02s1 read being DOSE-VOID: r9e2_K2DC02cl.pi_cost_approval.json carries the PI's words, the condition and the 12.3 s/step cap, pinned here.

GATE ADMISSION (2026-11-22, FLIPPED): the tip 0fd186ef admitted --w-map-hires in {1, 0.511, 1.018, 25} for a banded refcv9-e2 argv (`_R9F_MAP_W_ADMITTED`) and refused 3.0; this hold's
launch_gate.py edit adds 3.0. `test_static_gate_adds_no_refusal_beyond_the_known_weight_admission` now requires NO weight reason, and the RED arm (3.0 removed) must bring the refusal back.
Mutants (each must turn a test RED): one extra changed token; the initial weight outside the bounds; the approval file removed.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.train import bev_warp_flag as BWF  # noqa: E402
from tanitad.train import map_dose_ctl as MDC  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
BASE = "/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02"
OUT = BASE + "cl"
APPROVAL = RUNS / "r9e2_K2DC02cl.pi_cost_approval.json"
WEIGHT_ADMISSION = "--w-map-hires must be ONE of"      # the gate's refusal text (launch_gate `admitted_values_when`), the only one allowed for this arm
# in ARGV TOKEN ORDER (--w-map-hires at 101, --out at 162, --map-hires-w-bounds at 230). MM 4a3d05 2026-11-22: the first version listed the bounds
# before --out; difflib reports opcodes in position order, so this test was RED the first time it ran (on Thor), never having run before.
EXPECTED = [("replace", ["25.0"], ["3.0"]), ("replace", [BASE], [OUT]), ("replace", ["0.1,25"], ["2.5,3.5"])]


def _doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def _opcodes(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(op, a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def _diff_from_k2dc02(argv):
    """The two-direction diff of `argv` against K2DC02's: the length difference AND the opcodes (an insertion, a deletion and a replacement all show)."""
    c = _doc("K2DC02")["argv"]
    return len(argv) - len(c), _opcodes(c, argv)


def test_K2DC02cl_is_K2DC02_except_exactly_three_tokens_both_directions():
    a, c = _doc("K2DC02cl")["argv"], _doc("K2DC02")["argv"]
    assert len(a) == len(c)
    assert _diff_from_k2dc02(a) == (0, EXPECTED)
    assert _opcodes(a, c) == [("replace", ["3.0"], ["25.0"]), ("replace", [OUT], [BASE]), ("replace", ["2.5,3.5"], ["0.1,25"])]      # the reverse direction
    assert a[a.index(OUT) - 1] == "--out" and a[a.index("--seed") + 1] == "0" and a.count("--seed") == 1                           # seed 0, once: NOT the replicate
    assert a[a.index("--w-map-hires") + 1] == "3.0" and a.count("--w-map-hires") == 1
    assert a[a.index("--map-hires-w-bounds") + 1] == "2.5,3.5" and a.count("--map-hires-w-bounds") == 1
    assert "25.0" not in a and "0.1,25" not in a                                                                                   # no stale K2DC02 value anywhere in the argv
    assert a.count("--r9-hist-supervise-geom") == 1                                                                                # T1v kept (same flags as K2DC02)
    k, m = _doc("K2DC02cl"), _doc("K2DC02")
    for key in ("schema", "seed", "steps", "base_sha256", "derived_from", "profile", "w_cost_ratio", "w_cost_source"):
        assert k[key] == m[key], key
    assert k["arm"] == "r9e2_K2DC02cl"


def test_every_place_in_the_document_that_carries_the_tokens_is_consistent():
    k = _doc("K2DC02cl")
    i = k["argv"].index("--map-hires-share-target")
    assert k["lever"]["w_map_hires"] == 3.0 and k["lever"]["dose_ctl"][-2:] == ["--map-hires-w-bounds", "2.5,3.5"]
    assert k["lever"]["dose_ctl"] == k["argv"][i:i + 8]
    assert k["lever"]["w_map_1"] == _doc("K2DC02")["lever"]["w_map_1"] == ["--r8-critic-drivable", "--w-r8-drivable", "0.06472"]      # the W-COST weight is K2DC02's (UNVERIFIED here)
    assert k["argv"][k["argv"].index("--w-r8-drivable") + 1] == "0.06472"
    text = json.dumps({x: y for x, y in k.items() if x not in ("what",)})
    assert '"0.1,25"' not in text and '"25.0"' not in text                                                                         # structured fields: no stale value
    assert "CONDITIONAL on the K2DC02s1 read being DOSE-VOID" in k["what"] and "token replacement" in k["what"]


def test_weight_is_inside_the_clamp_and_parse_bounds_accepts_it():
    a = _doc("K2DC02cl")["argv"]
    lo, hi = MDC.parse_bounds("2.5,3.5")
    assert (lo, hi) == (2.5, 3.5)
    w = float(a[a.index("--w-map-hires") + 1])
    assert lo <= w <= hi and w == 3.0
    ns = T.build_parser().parse_args(a)
    assert ns.seed == 0 and ns.w_map_hires == 3.0 and ns.map_hires_w_bounds == "2.5,3.5" and ns.map_hires_share_target == 0.02 and ns.r9_hist_supervise_geom
    assert ns.map_hires_share_every == 250 and ns.map_hires_share_group == "trunk"
    BWF.pin_args(ns)
    MDC.pin_args(ns)                                                                                                               # the trainer's own pin: initial weight inside the bounds
    assert LG.dose_controller_static_reasons(a)[0] == [] and len(LG.dc_reading_rows(a)) == 24


def test_static_gate_adds_no_refusal_beyond_the_known_weight_admission():
    d = _doc("K2DC02cl")
    prof = LG.PROFILES[d["profile"]]
    got = LG.profile_argv_rules(prof, d["argv"])
    ctl = LG.profile_argv_rules(prof, _doc("K2DC02")["argv"])
    new = [r for r in got[0] if r not in ctl[0]]
    assert not any(WEIGHT_ADMISSION in r or "--w-map-hires" in r for r in new), new                                                   # FLIPPED (2026-11-22): the gate now ADMITS 3.0 (`_R9F_MAP_W_ADMITTED`)
    assert [r for r in LG.required_on_reasons(prof, d["argv"])[0] if r.startswith("--w-map-hires")] == []
    assert LG.map_weight_reasons(prof, d["argv"])[0] == []
    assert got[1] == ctl[1] == []


def test_RED_without_3_0_in_the_admitted_set_the_weight_refusal_comes_back():
    d = _doc("K2DC02cl")
    prof = dict(LG.PROFILES[d["profile"]])
    prof["admitted_values_when"] = {"--r9-map-banded": {"--w-map-hires": tuple(f"{w:g}" for w in LG._R9F_MAP_W_ADMITTED if w != 3.0)}}   # the mutant: 3.0 removed
    got = [r for r in LG.required_on_reasons(prof, d["argv"])[0] if r.startswith("--w-map-hires")]
    assert len(got) == 1 and WEIGHT_ADMISSION in got[0], got
    ok = [r for r in LG.required_on_reasons(LG.PROFILES[d["profile"]], d["argv"])[0] if r.startswith("--w-map-hires")]
    assert ok == []                                                                                                                   # the control: the real profile admits it


def test_wcost_is_the_own_smoke_rule_not_the_replicate_exception():
    a, c = _doc("K2DC02cl")["argv"], _doc("K2DC02")["argv"]
    assert LG.wcost_out_name(a) == "r9e2-K2DC02cl" and "r9e2-K2DC02cl" not in LG.WCOST_REPLICATES      # not a declared replicate: held to its OWN smoke's R
    assert LG.wcost_replicate_diff(a, c) != []                                                         # and not one by the gate's definition (more than --seed / --out moved)
    assert LG.WCOST_REPLICATES["r9e2-K2DC02s1"]["parent_R"] == 1.5450                                  # the parent R the W-COST note quotes


def test_pi_cost_approval_carries_the_PIs_words_the_condition_and_the_cap():
    ap = json.loads(APPROVAL.read_text(encoding="utf-8"))
    assert "i approve it" in ap["pi_verbatim"] and "Yes, use the clamp" in ap["pi_verbatim"] and ap["pi_verbatim"].startswith("i approve it  [PI, chat, 2026-11-22")
    assert ap["go_verbatim"] == "i approve it / Yes, use the clamp"
    assert ap["condition"] == "K2DC02s1 read DOSE-VOID" and "ONLY IF the K2DC02s1 read is DOSE-VOID" in ap["status"]
    assert ap["max_s_per_step"] == 12.3 and ap["date"] == "2026-11-22" and ap["status"].startswith("PI-APPROVED, CONDITIONAL")
    assert '"max_s_per_step": 12.3,' in APPROVAL.read_text(encoding="utf-8")                           # the restart's NOAPPROVAL guard greps this exact text


def test_MUTANT_one_extra_changed_token_turns_the_diff_RED():
    a = list(_doc("K2DC02cl")["argv"])
    assert _diff_from_k2dc02(a) == (0, EXPECTED)                                                       # the control reads exactly as pinned
    i = a.index("--lr-schedule") + 1
    a[i] = "cosine" if a[i] != "cosine" else "wsd"
    assert _diff_from_k2dc02(a) != (0, EXPECTED)
    b = list(_doc("K2DC02cl")["argv"]) + ["--extra-flag"]
    assert _diff_from_k2dc02(b) != (0, EXPECTED)
    c = list(_doc("K2DC02cl")["argv"])
    c[c.index("--seed") + 1] = "1"                                                                     # the replicate seed is also a change
    assert _diff_from_k2dc02(c) != (0, EXPECTED)


def test_MUTANT_initial_weight_outside_the_bounds_is_refused_by_the_static_gate_and_the_trainer_pin():
    a = list(_doc("K2DC02cl")["argv"])
    a[a.index("--w-map-hires") + 1] = "25.0"                                                           # K2DC02's initial weight with the clamp bounds
    assert any("outside --map-hires-w-bounds" in r for r in LG.dose_controller_static_reasons(a)[0])
    with pytest.raises(SystemExit):
        MDC.pin_args(T.build_parser().parse_args(a))
