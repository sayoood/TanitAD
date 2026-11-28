"""K2DC02s1 (the PREREG_K2DC02_2026-11-18 section 4.4 replicate) is COMPOSED by r9_e2_make_argv.py, never hand-written (MM 4a3d05, 2026-11-20).

K2DC02s1 = K2DC02's argv with ``--seed`` 0 -> 1 and ``--out`` naming the arm: NOTHING else. PI-APPROVED 2026-11-20 ("Yes, train it"): the
r9e2_K2DC02s1.pi_cost_approval.json carries the PI's words and the 12.3 s/step cap, and this file pins them. No scipy, no GPU.

Mutants (each must turn a test RED): the seed left at 0, and the seed moved to 2.
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
OUT = BASE + "s1"
HSG = "--r9-hist-supervise-geom"
FILE = RUNS / "r9e2_K2DC02s1.argv.json"


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


def test_K2DC02s1_regenerates_byte_equal_from_the_composer():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2DC02s1")
    assert _regen("K2DC02s1") == FILE.read_bytes()
    assert d["arm"] == "r9e2_K2DC02s1" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0" and d["seed"] == 1
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()
    assert d["lever"]["seed"]["changed"] == ["--seed", "1"] and d["lever"]["followup_of"] == "r9e2_K2DC02"
    assert "running it is a PI decision (no approval recorded)" in d["what"] and "section 4.4" in d["what"]
    assert "covers r9e2_K2DC02 ONLY" in d["lever"]["dose_ctl_provenance"]


def test_K2DC02s1_is_K2DC02_except_the_out_and_the_seed():
    a, c = _doc("K2DC02s1")["argv"], _doc("K2DC02")["argv"]
    assert len(a) == len(c)
    assert _opcodes(c, a) == [("replace", ["0"], ["1"]), ("replace", [BASE], [OUT])]
    assert a[a.index(OUT) - 1] == "--out" and a[a.index("--seed") + 1] == "1" and a.count("--seed") == 1
    assert "--r8-seed" not in a                                                 # the dedicated generator keeps its default: the replicate moves --seed ONLY
    k, m = _doc("K2DC02s1"), _doc("K2DC02")
    assert k["w_cost_ratio"] == m["w_cost_ratio"] and k["w_cost_source"] == m["w_cost_source"] and k["lever"]["dose_ctl"] == m["lever"]["dose_ctl"]
    assert HSG in a                                                             # the replicate KEEPS the T1v lever (same flags)


def test_K2DC02s1_parses_seed_1_and_the_static_gate_adds_no_refusal_beyond_the_control():
    d = _doc("K2DC02s1")
    ns = T.build_parser().parse_args(d["argv"])
    assert ns.seed == 1 and ns.r8_seed == 20261004 and ns.r9_hist_supervise_geom and ns.map_hires_share_target == 0.02
    BWF.pin_args(ns)
    prof = LG.PROFILES[d["profile"]]
    got = LG.profile_argv_rules(prof, d["argv"])
    ctl = LG.profile_argv_rules(prof, _doc("K2DC02")["argv"])
    assert got == ctl                                                           # the SAME reasons as the K2DC02 control (the two host-only ones on a dev box) and no more
    assert LG.dose_controller_static_reasons(d["argv"])[0] == [] and len(LG.dc_reading_rows(d["argv"])) == 24


def test_K2DC02s1_pi_cost_approval_carries_the_PIs_words_and_the_cap():
    # MM 11-20: was "no approval file" until the PI approved the arm ("Yes, train it", 2026-11-20)
    ap = json.loads((RUNS / "r9e2_K2DC02s1.pi_cost_approval.json").read_text(encoding="utf-8"))
    assert ap["go_verbatim"] == "Yes, train it" and ap["pi_verbatim"].startswith("Yes, train it  [PI, chat, 2026-11-20")
    assert ap["max_s_per_step"] == 12.3 and ap["date"] == "2026-11-20" and ap["status"].startswith("PI-APPROVED")
    # the restart's NOAPPROVAL guard greps the file for this exact text
    assert '"max_s_per_step": 12.3,' in (RUNS / "r9e2_K2DC02s1.pi_cost_approval.json").read_text(encoding="utf-8")


def test_every_existing_variant_still_regenerates_byte_equal_and_the_follow_up_is_registered():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    k0 = json.loads(k0b.decode("utf-8"))
    assert MK.REPLICATE_OF["K2DC02s1"] == "K2DC02" and MK.SEED_OF == {"K2DC02s1": 1} and MK.VARIANTS["K2DC02s1"] == "M1N20"
    assert MK.MAP_W_REDOSE == {"K2M1N20": 25.0, "K2DC": 25.0, "K2DC02": 25.0, "K2SG": 25.0}     # the follow-ups read the table through base_of: it did not grow
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


def test_MUTANT_seed_left_at_0_turns_the_byte_equal_and_the_diff_tests_RED(monkeypatch):
    monkeypatch.setitem(MK.SEED_OF, "K2DC02s1", 0)
    assert _regen("K2DC02s1") != FILE.read_bytes()
    comp = MK.compose(json.loads((RUNS / "r9e2_K0.argv.json").read_bytes().decode("utf-8")), "K2DC02s1", 0.06472)
    assert comp[comp.index("--seed") + 1] == "0"                               # the token-diff test (seed 0 -> 1) would fail


def test_MUTANT_seed_2_turns_the_byte_equal_RED(monkeypatch):
    monkeypatch.setitem(MK.SEED_OF, "K2DC02s1", 2)
    with pytest.raises(AssertionError):
        assert _regen("K2DC02s1") == FILE.read_bytes()
