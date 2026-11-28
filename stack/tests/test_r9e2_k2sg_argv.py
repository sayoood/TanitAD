"""K2SG's argv is COMPOSED by r9_e2_make_argv.py, never hand-written (MM 4a3d05, 2026-11-19).

K2SG = K2M1N20's argv + ONE lever, ``--map-hires-trunk-grad-scale 0.0`` (the map->trunk stop-gradient, stack/tanitad/train/map_trunk_gs.py), appended after every other
token. Pre-registered in PREREG_K2SG_2026-11-19.md (02dede10). It is a CANDIDATE arm: running it is a PI decision, so there is NO r9e2_K2SG.pi_cost_approval.json and this file
pins that absence. No scipy, no GPU.

Mutants (each must turn a test RED): the scale 0.0 -> 0.5 and the lever dropped, both in the composer's ``TRUNK_GS``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import r9_e2_make_argv as MK  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.train import map_trunk_gs as GS  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
FILE = RUNS / "r9e2_K2SG.argv.json"
FLAG = "--map-hires-trunk-grad-scale"
OUT_K2SG, OUT_K2M1N20 = "/home/nvidia/r8ns/r9e2/runs/r9e2-K2SG", "/home/nvidia/r8ns/r9e2/runs/r9e2-K2M1N20"


def _doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def _regen() -> bytes:
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2SG")
    r = float(d["w_cost_ratio"])
    rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, "K2SG", MK.w_from_ratio(r), provisional=False,
                         w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % r)
    return MK.dump(MK.stamp_w_cost(rec, r, d["w_cost_source"])).encode("utf-8")


def _token_diff():
    a, c = _doc("K2SG")["argv"], _doc("K2M1N20")["argv"]
    assert len(a) == len(c) + 2
    assert a[:-2] != c                                   # the --out differs
    diff = [(x, y) for x, y in zip(a[:-2], c) if x != y]
    return a, diff


def test_K2SG_regenerates_byte_equal_from_the_composer():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = _doc("K2SG")
    assert _regen() == FILE.read_bytes()
    assert d["arm"] == "r9e2_K2SG" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0"
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()
    assert d["lever"]["trunk_grad_scale"] == [FLAG, "0.0"]
    prov = d["lever"]["trunk_grad_scale_provenance"]
    assert "PREREG_K2SG_2026-11-19.md" in prov and "candidate arm; running it is a PI decision (no approval recorded)" in prov
    assert "running it is a PI decision (no approval recorded)" in d["what"] and " K2SG: " in d["what"]


def test_K2SG_is_K2M1N20_except_the_out_and_the_two_appended_tokens():
    a, diff = _token_diff()
    assert diff == [(OUT_K2SG, OUT_K2M1N20)]
    assert a[a.index(OUT_K2SG) - 1] == "--out"
    assert a[-2:] == [FLAG, "0.0"] and a.count(FLAG) == 1                     # appended after every other token
    k, m = _doc("K2SG"), _doc("K2M1N20")
    assert k["w_cost_ratio"] == m["w_cost_ratio"] and k["w_cost_source"] == m["w_cost_source"]
    assert k["lever"]["w_map_hires"] == 25.0 and k["lever"]["map_arm"] == "M1N20"
    assert "--map-hires-share-target" not in a                                 # no share controller: K2SG is K2M1N20, not K2DC


def test_K2SG_parses_with_the_tip_trainer_and_pin_args_accepts_it():
    argv = _doc("K2SG")["argv"]
    ns = T.build_parser().parse_args(argv)
    assert ns.map_hires_trunk_grad_scale == 0.0
    assert ns.r9_map_banded and ns.map_hires == "on" and ns.w_map_hires == 25.0
    GS.pin_args(ns)                                                            # raises SystemExit if the banded head / map-hires / weight preconditions fail
    assert GS.is_on(ns) and GS.scale_for(ns) == 0.0
    ns_off = T.build_parser().parse_args(_doc("K2M1N20")["argv"])
    assert ns_off.map_hires_trunk_grad_scale == 1.0 and not GS.is_on(ns_off)


def test_K2SG_has_no_pi_cost_approval_file_because_running_it_is_a_PI_decision():
    assert not (RUNS / "r9e2_K2SG.pi_cost_approval.json").exists()


def test_every_other_variant_regenerates_byte_equal():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    k0 = json.loads(k0b.decode("utf-8"))
    for v in sorted(MK.VARIANTS):
        f = RUNS / f"r9e2_{v}.argv.json"
        if v == "K2SG" or not f.is_file():
            continue
        d = json.loads(f.read_bytes().decode("utf-8"))
        r = d.get("w_cost_ratio")
        if r is None:
            continue                                                           # a provisional file carries no ratio: the K2 test file covers it
        rec = MK.variant_doc(k0, k0b, v, MK.w_from_ratio(float(r)), provisional=False, w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % float(r))
        assert MK.dump(MK.stamp_w_cost(rec, float(r), d["w_cost_source"])).encode("utf-8") == f.read_bytes(), v


def test_MUTANT_scale_0_5_turns_the_byte_equal_and_the_diff_tests_RED(monkeypatch):
    monkeypatch.setitem(MK.TRUNK_GS, "K2SG", [FLAG, "0.5"])
    assert _regen() != FILE.read_bytes()                                       # the byte-equal test would fail
    with pytest.raises(AssertionError):
        assert _regen() == FILE.read_bytes()


def test_MUTANT_lever_dropped_turns_the_byte_equal_and_the_diff_tests_RED(monkeypatch):
    monkeypatch.setitem(MK.TRUNK_GS, "K2SG", [])
    assert _regen() != FILE.read_bytes()
    with pytest.raises(AssertionError):
        assert _regen() == FILE.read_bytes()
    comp = MK.compose(json.loads((RUNS / "r9e2_K0.argv.json").read_bytes().decode("utf-8")), "K2SG", 0.06472)
    assert FLAG not in comp and len(comp) == len(_doc("K2SG")["argv"]) - 2     # the token-diff test's length assertion would fail
