"""K2DC02nh (the PREREG_K2DC02_2026-11-18 section 4.3 candidate: K2 minus --r9-hist-supervise-geom) is COMPOSED by r9_e2_make_argv.py, never hand-written (MM 4a3d05, 2026-11-20).

K2DC02nh = K2DC02's argv with the ONE token ``--r9-hist-supervise-geom`` (a plain switch, no value) removed and ``--out`` naming the arm: NOTHING else. It is a CANDIDATE arm
(running it is a PI decision), so there is NO r9e2_K2DC02nh.pi_cost_approval.json and this file pins that absence. No scipy, no GPU.

What the removal changes in the BUILT model (traced and cited in the kit's HANDOFF.md): no parameter, no buffer, no state_dict entry (bev_warp.HistGeom is a plain-Python
holder), no loss term. The 10 cm map loss and the banded loss supervise ``lift_valid_now`` alone instead of ``lift_valid_now | hist_valid_geom``. No other argv flag depends on it.

Mutants (each must turn a test RED): the lever left in (T1_ARM_OF back to T1v), and the seed also moved.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import launch_gate as LG  # noqa: E402
import r9_e2_make_argv as MK  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.train import bev_warp_flag as BWF  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
BASE = "/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02"
OUT = BASE + "nh"
HSG = "--r9-hist-supervise-geom"
FILE = RUNS / "r9e2_K2DC02nh.argv.json"


def _doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def _regen(v) -> bytes:
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc(v)
    r = float(d["w_cost_ratio"])
    rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, v, MK.w_from_ratio(r), provisional=False,
                         w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % r)
    return MK.dump(MK.stamp_w_cost(rec, r, d["w_cost_source"])).encode("utf-8")


def _opcodes(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(op, a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def test_K2DC02nh_regenerates_byte_equal_from_the_composer():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2DC02nh")
    assert _regen("K2DC02nh") == FILE.read_bytes()
    assert d["arm"] == "r9e2_K2DC02nh" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0" and d["seed"] == 0
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()
    assert "t1v" not in d["lever"] and "t1h_bundle" not in d["lever"] and d["lever"]["t1"].startswith("NONE: --r9-hist-supervise-geom REMOVED")
    assert d["lever"]["followup_of"] == "r9e2_K2DC02" and "section 4.3" in d["what"]
    assert "running it is a PI decision (no approval recorded)" in d["what"] and "covers r9e2_K2DC02 ONLY" in d["lever"]["dose_ctl_provenance"]


def test_K2DC02nh_is_K2DC02_except_the_out_and_the_one_removed_token():
    a, c = _doc("K2DC02nh")["argv"], _doc("K2DC02")["argv"]
    assert len(a) == len(c) - 1
    assert _opcodes(c, a) == [("delete", [HSG], []), ("replace", [BASE], [OUT])]
    assert HSG not in a and a[a.index(OUT) - 1] == "--out"
    assert a.count("--seed") == 1 and a[a.index("--seed") + 1] == "0"
    k, m = _doc("K2DC02nh"), _doc("K2DC02")
    assert k["w_cost_ratio"] == m["w_cost_ratio"] and k["w_cost_source"] == m["w_cost_source"] and k["lever"]["dose_ctl"] == m["lever"]["dose_ctl"]
    assert "--r9-map-banded" in a and "--map-hires-share-target" in a and "--r9-plan-map-channels" in a   # every other lever stays


def test_K2DC02nh_parses_with_the_lever_off_and_the_static_gate_adds_no_refusal_beyond_the_control():
    d = _doc("K2DC02nh")
    ns = T.build_parser().parse_args(d["argv"])
    assert ns.r9_hist_supervise_geom is False and not BWF.geom_on(ns) and not BWF.is_on(ns)
    assert ns.r9_map_banded and ns.map_hires_share_target == 0.02 and ns.seed == 0
    BWF.pin_args(ns)                                                            # a no-op without a T1 flag
    prof = LG.PROFILES[d["profile"]]
    got, ctl = LG.profile_argv_rules(prof, d["argv"]), LG.profile_argv_rules(prof, _doc("K2DC02")["argv"])
    assert got == ctl
    assert LG.dose_controller_static_reasons(d["argv"])[0] == [] and len(LG.dc_reading_rows(d["argv"])) == 24


def test_no_other_parsed_flag_differs_from_K2DC02_so_none_is_a_dependent_of_the_removed_lever():
    pa = vars(T.build_parser().parse_args(_doc("K2DC02")["argv"]))
    pb = vars(T.build_parser().parse_args(_doc("K2DC02nh")["argv"]))
    assert sorted(k for k in pa if pa[k] != pb[k]) == ["out", "r9_hist_supervise_geom"]


def test_K2DC02nh_has_no_pi_cost_approval_file_because_running_it_is_a_PI_decision():
    assert not (RUNS / "r9e2_K2DC02nh.pi_cost_approval.json").exists()


def test_every_existing_variant_still_regenerates_byte_equal_and_the_follow_up_is_registered():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    k0 = json.loads(k0b.decode("utf-8"))
    assert MK.REPLICATE_OF["K2DC02nh"] == "K2DC02" and MK.T1_ARM_OF["K2DC02nh"] is None and MK.VARIANTS["K2DC02nh"] == "M1N20"
    assert MK.MAP_W_REDOSE == {"K2M1N20": 25.0, "K2DC": 25.0, "K2DC02": 25.0, "K2SG": 25.0}
    for v in sorted(MK.VARIANTS):
        f = RUNS / f"r9e2_{v}.argv.json"
        if not f.is_file():
            continue
        d = json.loads(f.read_bytes().decode("utf-8"))
        r = d.get("w_cost_ratio")
        if r is None:
            continue
        rec = MK.variant_doc(k0, k0b, v, MK.w_from_ratio(float(r)), provisional=False, w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % float(r))
        assert MK.dump(MK.stamp_w_cost(rec, float(r), d["w_cost_source"])).encode("utf-8") == f.read_bytes(), v


def test_MUTANT_lever_left_in_turns_the_byte_equal_and_the_diff_tests_RED(monkeypatch):
    monkeypatch.setitem(MK.T1_ARM_OF, "K2DC02nh", MK.T1V_ARM)
    assert _regen("K2DC02nh") != FILE.read_bytes()
    comp = MK.compose(json.loads((RUNS / "r9e2_K0.argv.json").read_bytes().decode("utf-8")), "K2DC02nh", 0.06472)
    assert HSG in comp and len(comp) == len(_doc("K2DC02nh")["argv"]) + 1       # the token-diff test's length assertion would fail


def test_MUTANT_the_seed_is_also_moved_turns_the_byte_equal_RED(monkeypatch):
    monkeypatch.setitem(MK.SEED_OF, "K2DC02nh", 1)
    with pytest.raises(AssertionError):
        assert _regen("K2DC02nh") == FILE.read_bytes()
