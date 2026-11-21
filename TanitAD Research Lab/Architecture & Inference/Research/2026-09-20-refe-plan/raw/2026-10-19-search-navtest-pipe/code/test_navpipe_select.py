"""Tests of the C2 FINAL pieces (MM_RULING_C2AMEND.md): the config the look is sent with (C8 or O4, == `c2_config`), the dose result bound like C1's, the fixture record (DQ7 (b)), the ONE-RUN marker, the re-score of
P2's picks through the identical route (DQ6 (b)), the outcome names and the safety miss in the headline (DQ5), and the repo-relative sibling packages (portability).  CPU only; planted data except where marked.

Every guard has a RED arm that must REFUSE with its NAMED reason (a later check must not be able to stand in for a removed one); mutation_check.py deletes each `@GUARD` line in turn and this suite must then fail.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
import navpipe_chain as CH  # noqa: E402
import navpipe_analyse as NA  # noqa: E402
import test_navpipe as T  # noqa: E402

raises = T.raises
SAFE = {"nc": 0.0, "dac": 0.17, "ttc": 0.25, "c": 0.0}
POS = {"pooled": {"point": 0.2, "lo": 0.05, "hi": 0.4}}
NEG = {"pooled": {"point": 0.0, "lo": -0.2, "hi": 0.2}}
O4_FILE, C8_FILE = NC.CONFIG_O4, NC.CONFIG_C8


def seal_for(tmp, config="C8", c1_doc=None, dose_kw=None, **mk):
    """a sealed planted world (C1 result, dose result bound to it, dose-chain double, fixture record) and the call that checks it with `config` (C8 / O4 / a path / None)"""
    c1 = T.C1(tmp, doc=c1_doc if c1_doc is not None else T.c1_doc())
    am, sha, _, pins = T.make_seal(tmp, c1=c1, dose_kw=dose_kw, **mk)
    path = {"C8": C8_FILE, "O4": O4_FILE}.get(config, config)
    return (lambda **over: T.check_c2_seal(am, sha, c1.path, c1.sha, **{"config": path, **over})), c1, am, sha


def refused(reason, tmp, config="C8", c1_doc=None, dose_kw=None, over=None, **mk):
    call, c1, am, sha = seal_for(tmp, config, c1_doc, dose_kw, **mk)
    e = raises("seal", lambda: call(**(over or {})))
    assert e.code == 5 and e.reason == reason, (e.reason, e.detail)
    return e


# ====================================================================================================== the config: exactly C8 or O4, and the one `c2_config` returns
UNLIC = lambda **kw: T.c1_doc(outcome="O3_NULL", **kw)         # noqa: E731
O1_DOSE = {"outcome": "O1_CONFIRMED"}
O2_DOSE = {"outcome": "O2_DIRECTIONAL"}

OK_CASES = [
    ("row1_c1_licensed_dose_not_C8", "C8", None, {}, 1, "C1"),
    ("row3_both_licensed_contrast_not_positive_C8", "C8", None, {**O1_DOSE, "contrast_O4_minus_C8": NEG}, 3, "C1"),
    ("row3_both_licensed_contrast_positive_O4", "O4", None, {**O2_DOSE, "contrast_O4_minus_C8": POS}, 3, "C1"),
    ("row2_c1_not_licensed_dose_O1_O4", "O4", UNLIC(), {**O1_DOSE}, 2, "DOSE"),
]


@pytest.mark.parametrize("cid,config,c1d,dose_kw,row,reader", OK_CASES, ids=[c[0] for c in OK_CASES])
def test_the_config_c2_config_returns_is_accepted_and_the_licensing_read_follows_the_row(tmp_path, cid, config, c1d, dose_kw, row, reader):
    call, c1, am, sha = seal_for(tmp_path, config, c1d, dose_kw)
    r = call()
    assert r["selection"]["config"] == config and r["selection"]["row"] == row and r["licence"]["licensing_read"] == reader
    assert r["pins"]["config_name"] == config and r["pins"]["config_sha256"] == (NC.CONFIG_O4_SHA256 if config == "O4" else NC.CONFIG_C8_SHA256)
    assert r["licence"]["c1_result_sha256"] == c1.sha and r["selection"]["dose_result_sha256"] and r["selection"]["dose_chain_sha256"] == r["pins"]["dose_chain_sha256"]
    if reader == "DOSE":                                                 # the dose-only path: the DOSE read is O1_CONFIRMED with safe terms, C1's is not licensed
        assert r["licence"]["outcome"] == "O1_CONFIRMED" and r["licence"]["licensing_result_sha256"] == r["selection"]["dose_result_sha256"]


REFUSE_CASES = [
    ("row1_but_O4_given", "O4", None, {}, "CONFIG_NOT_C2_CONFIG"),
    ("row3_contrast_not_positive_but_O4_given", "O4", None, {**O1_DOSE, "contrast_O4_minus_C8": NEG}, "CONFIG_NOT_C2_CONFIG"),
    ("row3_contrast_positive_but_C8_given", "C8", None, {**O2_DOSE, "contrast_O4_minus_C8": POS}, "CONFIG_NOT_C2_CONFIG"),
    ("row2_but_C8_given_needs_C1s_licence", "C8", UNLIC(), {**O1_DOSE}, "LICENCE_OUTCOME"),
    ("row4_dose_O2_is_not_enough_on_the_dose_only_path", "O4", UNLIC(), {**O2_DOSE}, "LICENCE_OUTCOME"),
    ("row4_dose_null", "O4", UNLIC(), {}, "LICENCE_OUTCOME"),
    ("dose_only_path_safety_miss", "O4", UNLIC(), {**O1_DOSE, "safety_points": {**SAFE, "ttc": -0.3}}, "LICENCE_SAFETY_MISS"),
    ("dose_only_path_inert", "O4", UNLIC(), {**O1_DOSE, "inert": {"inert": False, "frac_moved": 0.01}}, "LICENCE_INERT"),
    ("c1_inert_by_the_c2_definition_C8", "C8", T.c1_doc(inert={"inert": False, "frac_moved": 0.01}), {}, "LICENCE_INERT"),
]


@pytest.mark.parametrize("cid,config,c1d,dose_kw,reason", REFUSE_CASES, ids=[c[0] for c in REFUSE_CASES])
def test_a_config_c2_config_did_not_return_is_refused_by_name(tmp_path, cid, config, c1d, dose_kw, reason):
    refused(reason, tmp_path, config, c1d, dose_kw)


def test_moved_share_vs_frac_moved_disagreement_is_not_licensed_through_c2s_path(tmp_path):
    """MM_RULING_C2FINAL.md point 1: "moved share" is `inert.frac_moved` EVERYWHERE.  At Revision 1 the dose chain's `c2_config` and C2's own licence terms read the same quantity, so a read with moved_share >= 5 % and
    frac_moved < 5 % is NOT licensed: refused by name, never a config.  (Before Revision 1 the two disagreed on exactly this file and this test documented it; against the REAL file see the `real_file` tests below.)"""
    a = T.c1_doc(moved_share=0.34, inert={"inert": False, "frac_moved": 0.01})                      # moved_share 34 %, frac_moved 1 %: not licensed
    call, c1, am, sha = seal_for(tmp_path / "a", "C8", a)
    assert raises("seal", lambda: call()).reason == "LICENCE_INERT"
    b = T.c1_doc(moved_share=0.01, inert={"inert": False, "frac_moved": 0.165})                     # the other direction: licensed by frac_moved although moved_share is 1 %, so row 1 / C8
    call, c1, am, sha = seal_for(tmp_path / "b", "C8", b)
    r = call()
    assert (r["selection"]["config"], r["selection"]["row"]) == ("C8", 1)


def test_licence_row_disagreement_is_refused_by_name(tmp_path):
    """G-LICENCE-ROW-AGREES is the net under the two licences.  A dose chain that still reads `moved_share` (the pre-Revision-1 semantics, PLANTED: the real file is Revision 1 and cannot do this) calls C1 licensed (moved_share 0.34)
    while C2's terms say INERT (frac_moved 0.01): row 3 would then rely on the dose read to license, which is refused by name.  At Revision 1 the same world is a consistent row 2 and is accepted."""
    c1d = T.c1_doc(moved_share=0.34, inert={"inert": False, "frac_moved": 0.01})
    refused("LICENCE_ROW_DISAGREES", tmp_path / "legacy", "O4", c1d, {**O1_DOSE, "contrast_O4_minus_C8": POS}, dose_src=T.DOSE_STUB_LEGACY_SRC)
    # the other direction, same planted pre-Revision-1 chain: C2's terms license C1 (frac_moved 16.5 %) while the chain's c2_config calls it unlicensed (moved_share 1 %): row 4, no config (G-CONFIG-NONE; unreachable with the real Revision 1 file)
    refused("C2_CONFIG_NONE", tmp_path / "legacy_none", "C8", T.c1_doc(moved_share=0.01, inert={"inert": False, "frac_moved": 0.165}), None, dose_src=T.DOSE_STUB_LEGACY_SRC)
    call, c1, am, sha = seal_for(tmp_path / "r1", "O4", c1d, {**O1_DOSE, "contrast_O4_minus_C8": POS})
    r = call()
    assert r["selection"]["row"] == 2 and r["licence"]["licensing_read"] == "DOSE"


def test_a_config_that_is_neither_c8_nor_o4_is_refused_and_so_is_none(tmp_path):
    bad = tmp_path / "other.json"
    bad.write_bytes(open(C8_FILE, "rb").read() + b" ")
    refused("CONFIG_NOT_PINNED", tmp_path / "x", str(bad))
    refused("CONFIG_MISSING", tmp_path / "y", None)
    refused("CONFIG_MISSING", tmp_path / "z", str(tmp_path / "does_not_exist.json"))


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.1", True])
def test_an_unreadable_contrast_refuses_instead_of_silently_turning_O4_into_C8(tmp_path, bad):
    refused("CONFIG_CONTRAST_UNREADABLE", tmp_path, "C8", None, {"contrast_O4_minus_C8": {"pooled": {"point": 0.0, "lo": bad, "hi": 0.3}}})


@pytest.mark.parametrize("mut", ["no_contrast", "no_pooled"])
def test_a_missing_contrast_block_refuses(tmp_path, mut):
    refused("CONFIG_CONTRAST_UNREADABLE", tmp_path, "C8", None, None, dose_mut=(lambda d: d.pop("contrast_O4_minus_C8")) if mut == "no_contrast" else (lambda d: d["contrast_O4_minus_C8"].pop("pooled")))


@pytest.mark.parametrize("rec", [None, {"config": "O4", "row": 1}, {"config": "C8", "row": 3}, "C8"])
def test_the_dose_results_own_c2_config_record_must_agree(tmp_path, rec):
    refused("CONFIG_DOSE_RECORD_DIFFERS", tmp_path, "C8", None, {"c2_config": rec})


def test_the_pinned_constants_are_the_files_the_repo_holds():
    """C8 / O4 are pinned by sha256 in code AND in the amendment; the files in the sibling Step B package are those bytes (a different config file would be refused by name)"""
    assert NC.sha_lf(C8_FILE) == NC.CONFIG_C8_SHA256 and NC.sha_lf(O4_FILE) == NC.CONFIG_O4_SHA256
    c8, o4 = json.load(open(C8_FILE)), json.load(open(O4_FILE))
    assert (c8["m_pop"], c8["n_elite"], c8["widen"]) == (64, 8, 1.0) and (o4["m_pop"], o4["n_elite"], o4["widen"]) == (256, 32, 2.0)


# ====================================================================================================== the dose result, bound like C1's
def test_dose_result_arguments_are_required_in_every_case(tmp_path):
    for k, over in (("a", {"dose_result": None}), ("b", {"dose_result_sha256": None}), ("c", {"dose_result": None, "dose_result_sha256": None})):
        call, c1, am, sha = seal_for(tmp_path / k)
        assert raises("seal", lambda: call(**over)).reason == "DOSE_MISSING_ARGS"


def test_dose_result_file_sha_and_pin(tmp_path):
    call, c1, am, sha = seal_for(tmp_path / "a")
    assert raises("seal", lambda: call(dose_result=str(tmp_path / "nope.json"))).reason == "DOSE_RESULT_MISSING"
    assert raises("seal", lambda: call(dose_result_sha256="0" * 64)).reason == "DOSE_RESULT_SHA_MISMATCH"
    ev = T.EVID[os.path.abspath(am)]
    assert call(dose_result_sha256=ev.dose_sha.upper())["selection"]["dose_result_sha256"] == ev.dose_sha        # the typed sha is case-insensitive
    other = T.C1(tmp_path / "b")
    am2, sha2, _, _ = T.make_seal(tmp_path / "b", c1=other, extra_pins={"dose_result_sha256": "e" * 64})        # an amendment sealed AFTER the dose read pins its sha: a different (valid) file is refused
    assert raises("seal", lambda: T.check_c2_seal(am2, sha2, other.path, other.sha, config=C8_FILE)).reason == "DOSE_RESULT_PIN_MISMATCH"
    third = T.C1(tmp_path / "c")
    am3, sha3, _, _ = T.make_seal(tmp_path / "c", c1=third)                                                  # no dose pin in the amendment (sealed BEFORE the dose read exists): the typed sha is the only binding
    assert T.check_c2_seal(am3, sha3, third.path, third.sha, config=C8_FILE)["licence"]["outcome"] == "O1_CONFIRMED"


def test_a_matching_dose_pin_passes(tmp_path):
    c1 = T.C1(tmp_path)
    am, sha, _, pins = T.make_seal(tmp_path, c1=c1)
    ev = T.EVID[os.path.abspath(am)]
    text = open(am, encoding="utf-8").read().replace("<!-- PINS-END -->", f"dose_result_sha256 {ev.dose_sha}\n<!-- PINS-END -->")        # re-seal the same text with the dose pin added
    open(am, "w", encoding="utf-8", newline="\n").write(text)
    import hashlib
    sha2 = hashlib.sha256(text.encode()).hexdigest()
    open(os.path.join(str(tmp_path), "AMEND.sha256"), "w").write(sha2 + "  AMEND.md\n")
    assert T.check_c2_seal(am, sha2, c1.path, c1.sha, config=C8_FILE)["selection"]["dose_result_sha256"] == ev.dose_sha


def test_dose_result_unreadable_or_not_an_object(tmp_path):
    for k, raw, why in (("a", "{not json", "DOSE_RESULT_UNREADABLE"), ("b", "[1, 2]", "DOSE_RESULT_UNREADABLE")):
        call, c1, am, sha = seal_for(tmp_path / k)
        ev = T.EVID[os.path.abspath(am)]
        open(ev.dose_path, "w").write(raw)
        e = raises("seal", lambda: call(dose_result_sha256=NC.sha256_file(ev.dose_path)))
        assert e.reason == why


DOSE_NOT_DOSE = [({"read": "C1"}, "read"), ({"n_tokens": 400}, "tokens"), ({"n_logs": 23}, "logs"), ({"arm": "P2HEAD search, C8 config, no head substitution"}, "arm_not_O4"),
                 ({"arm": "O4 search"}, "arm_not_P2HEAD"), ({"read": None}, "read_none")]


@pytest.mark.parametrize("kw,cid", DOSE_NOT_DOSE, ids=[c[1] for c in DOSE_NOT_DOSE])
def test_a_result_that_is_not_the_dose_read_on_b_rest_is_refused(tmp_path, kw, cid):
    refused("DOSE_NOT_THE_DOSE_READ", tmp_path, "C8", None, kw)


def _c1_block(**kw):
    def mut(d):
        d["c1"].update(kw)
    return mut


BOUND = [("other_c1_result", _c1_block(result_sha256="0" * 64)), ("other_c1_outcome", _c1_block(outcome="O2_DIRECTIONAL")), ("c8_config_sha", _c1_block(config_c8_sha256="1" * 64)),
         ("o4_config_sha", _c1_block(config_o4_sha256="1" * 64)), ("no_bundle_sha", _c1_block(bundle_sha256=None)), ("empty_bundle_sha", _c1_block(bundle_sha256="")), ("no_c1_block", lambda d: d.pop("c1")),
         ("c1_outcome_field", lambda d: d.update(c1_outcome="TRADE"))]


@pytest.mark.parametrize("cid,mut", BOUND, ids=[b[0] for b in BOUND])
def test_a_dose_read_that_was_not_computed_against_this_c1_result_is_refused(tmp_path, cid, mut):
    refused("DOSE_NOT_BOUND_TO_C1", tmp_path, "C8", None, None, dose_mut=mut)


def test_dose_chain_state_must_exist_carry_c1s_prereg_and_every_dose_stage(tmp_path):
    refused("DOSE_CHAIN_STATE_MISSING", tmp_path / "a", dose_state=False)
    stages = ("SEAL_OK", "AMEND_SEAL_OK", "DATA_OK", "C1_DONE", "BUNDLE_OK", "SUBID_PASS", "REACH_PASS", "SEARCH_DONE", "SELFREP_PASS", "LABEL_DONE")
    refused("DOSE_PREREG_MISMATCH", tmp_path / "b", dose_state={"done": list(stages), "prereg_sha256": "0" * 64})
    refused("DOSE_PREREG_MISMATCH", tmp_path / "c", dose_state={"done": [], "prereg_sha256": None})
    for s in stages:
        e = refused("DOSE_INCOMPLETE", tmp_path / ("s_" + s), dose_state={"done": [x for x in stages if x != s], "prereg_sha256": T.PLANT_PREREG})
        assert s in e.detail


# ====================================================================================================== the dose-chain loader: by path, pinned, never re-implemented
def test_dose_chain_is_imported_by_path_under_its_pin(tmp_path):
    call, c1, am, sha = seal_for(tmp_path / "a")
    ev = T.EVID[os.path.abspath(am)]
    assert raises("seal", lambda: call(dose_chain=str(tmp_path / "nope.py"))).reason == "DOSE_CHAIN_MISSING"
    other = tmp_path / "dose_chain_other.py"
    other.write_text(open(ev.stub_path).read() + "\n# edited\n")
    assert raises("seal", lambda: call(dose_chain=str(other))).reason == "DOSE_CHAIN_PIN_MISMATCH"        # a different file than the amendment pinned
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    assert raises("seal", NC.load_dose_chain, str(broken), NC.sha_lf(str(broken))).reason == "DOSE_CHAIN_IMPORT_FAILED"
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1\n")
    assert raises("seal", NC.load_dose_chain, str(empty), NC.sha_lf(str(empty))).reason == "DOSE_CHAIN_NO_C2_CONFIG"
    notuple = tmp_path / "notuple.py"
    notuple.write_text("DOSE_STAGES = ['a']\ndef c2_config(a, b, c):\n    return {}\n")
    assert raises("seal", NC.load_dose_chain, str(notuple), NC.sha_lf(str(notuple))).reason == "DOSE_CHAIN_NO_C2_CONFIG"
    m = NC.load_dose_chain(ev.stub_path, NC.sha_lf(ev.stub_path))
    assert m is NC.load_dose_chain(ev.stub_path, NC.sha_lf(ev.stub_path))                                  # cached by (path, sha)


def test_the_amendment_pin_on_the_dose_chain_is_checked_in_the_seal_call(tmp_path):
    c1 = T.C1(tmp_path)
    am, sha, _, pins = T.make_seal(tmp_path, c1=c1, tamper_pin="dose_chain_sha256")
    assert raises("seal", lambda: T.check_c2_seal(am, sha, c1.path, c1.sha, config=C8_FILE)).reason == "DOSE_CHAIN_PIN_MISMATCH"


def test_a_c2_config_that_returns_garbage_is_refused(tmp_path):
    call, c1, am, sha = seal_for(tmp_path / "a")
    garbage = T.plant_dose_package(str(tmp_path / "garbage_pkg"), 'DOSE_STAGES = ("SEAL_OK", "AMEND_SEAL_OK", "DATA_OK", "C1_DONE", "BUNDLE_OK", "SUBID_PASS", "REACH_PASS", "SEARCH_DONE", "SELFREP_PASS", "LABEL_DONE")\ndef c2_config(a, b, c):\n    return 5\n')    # sealed at R1 (planted), so only the garbage can refuse
    c1b = T.C1(tmp_path / "b")
    am2, sha2, _, _ = T.make_seal(tmp_path / "b", c1=c1b, extra_pins={"dose_chain_sha256": NC.sha_lf(str(garbage))})
    e = raises("seal", lambda: T.check_c2_seal(am2, sha2, c1b.path, c1b.sha, config=C8_FILE, dose_chain=str(garbage)))
    assert e.reason == "CONFIG_C2CONFIG_UNREADABLE"


DOSE_CHAIN_R1_SHA256 = "b41ccab5a24fa67c4d07c3aa4dce1b60175c97ad9787f513e91491193dccc217"      # dose/code/dose_chain.py at REVISION 1 (AMENDMENT_DOSE_R1.md pins `code:dose_chain.py` to it): the pin follows Revision 1
DOSE_R1_AMENDMENT_SHA256 = "4769bb8e1961288feebae913c20aaf429de7ddc02f3d595876df0977558203ca"   # dose/AMENDMENT_DOSE_R1.md, equal to its .sha256
REAL_DOSE = NC.DOSE_CHAIN                  # the repo-relative file (C2_DOSE_CHAIN moves the PATH only); no dev-box fallback and no skip: a missing or changed real file is a RED test, never a silent pass
SAFE_LOW = {**SAFE, "ttc": -0.3}
OUTCOMES = ("O1_CONFIRMED", "O2_DIRECTIONAL", "O3_NULL", "TRADE", "NEGATIVE", "O4_INERT", None)


def real_and_double():
    """the REAL dose_chain.py imported by path under the Revision 1 sha pin (a changed file is a RED test, not a new baseline), and the test double exec'd from T.DOSE_STUB_SRC"""
    real = NC.load_dose_chain(REAL_DOSE, DOSE_CHAIN_R1_SHA256)
    ns = {}
    exec(compile(T.DOSE_STUB_SRC, "dose_stub", "exec"), ns)
    return real, ns


def read_of(o, sp, ms, fm):
    """a read with its descriptive `moved_share` and its `inert` block (`fm` == "absent" leaves the block out)"""
    d = {"outcome": o, "safety_points": sp, "moved_share": ms}
    if fm != "absent":
        d["inert"] = {"inert": False, "frac_moved": fm}
    return d


def test_the_real_dose_chain_is_the_revision_1_file_and_its_package_is_sealed_at_r1():
    assert os.path.isfile(REAL_DOSE), REAL_DOSE
    assert NC.sha_lf(REAL_DOSE) == DOSE_CHAIN_R1_SHA256
    assert NC.check_dose_package_r1(REAL_DOSE) == {"amendment_sha256": DOSE_R1_AMENDMENT_SHA256, "dose_chain_sha256": DOSE_CHAIN_R1_SHA256}
    pkg = os.path.dirname(os.path.dirname(REAL_DOSE))
    assert open(os.path.join(pkg, NC.DOSE_R1_SIDECAR), encoding="utf-8").read().split()[0] == DOSE_R1_AMENDMENT_SHA256


def test_the_test_double_agrees_with_the_real_dose_chain_over_the_whole_grid():
    """the grid the double was validated on: 7 outcomes x 3 safety x 3 moved values, squared, x 5 contrast values.  The moved value is carried in BOTH `moved_share` and `inert.frac_moved` (the two agree), so the grid is the old one read at Revision 1"""
    real, ns = real_and_double()
    reads = [read_of(o, sp, x, x) for o in OUTCOMES for sp in (SAFE, SAFE_LOW, None) for x in (0.3, 0.049, None)]
    n = 0
    for c1 in reads:
        for d in reads:
            for lo in (-1.0, 0.0, 0.3, None, float("nan")):
                a, b = real.c2_config(c1, d, lo), ns["c2_config"](c1, d, lo)
                assert (a["config"], a["row"]) == (b["config"], b["row"]), (c1, d, lo, a, b)
                n += 1
    assert n == len(reads) ** 2 * 5 == 19845
    assert tuple(real.DOSE_STAGES) == tuple(ns["DOSE_STAGES"])


def test_the_double_and_the_real_chain_agree_where_moved_share_and_frac_moved_DISAGREE_and_neither_licenses_on_moved_share():
    """the Revision 1 ruling on the REAL file: frac_moved < 5 % (or absent / NaN / None) is never a licence whatever `moved_share` says, and frac_moved >= 5 % is one whatever `moved_share` says"""
    real, ns = real_and_double()
    pairs = [(0.3, 0.049), (0.049, 0.3), (0.3, "absent"), (0.3, float("nan")), (None, 0.3), (0.3, 0.05), (0.0, 0.0499), (0.9, None)]
    reads = [read_of(o, sp, ms, fm) for o in ("O1_CONFIRMED", "O2_DIRECTIONAL", "O3_NULL", None) for sp in (SAFE, SAFE_LOW) for ms, fm in pairs]
    n = 0
    for r in reads:
        for strict in (False, True):
            assert real.licensed(r, strict) == ns["licensed"](r, strict), (r, strict)
    for c1 in reads:
        for d in reads:
            for lo in (-1.0, 0.3, None):
                a, b = real.c2_config(c1, d, lo), ns["c2_config"](c1, d, lo)
                assert (a["config"], a["row"]) == (b["config"], b["row"]), (c1, d, lo, a, b)
                n += 1
    assert n == len(reads) ** 2 * 3
    for ms, fm, want in [(0.34, 0.01, False), (0.34, "absent", False), (0.34, float("nan"), False), (0.34, None, False), (0.01, 0.165, True), (None, 0.05, True), (0.0, 0.3, True)]:
        assert real.licensed(read_of("O1_CONFIRMED", SAFE, ms, fm)) is want and real.licensed(read_of("O1_CONFIRMED", SAFE, ms, fm), strict=True) is want, (ms, fm)
    t = real.c2_config(read_of("O1_CONFIRMED", SAFE, 0.34, 0.01), read_of("O3_NULL", SAFE, 0.5, 0.3), 0.1)["moved_term"]
    assert t["quantity"] == "inert.frac_moved" and t["c1"] == 0.01 and t["dose"] == 0.3


def real_call(tmp, config, c1_doc=None, dose_kw=None):
    """a sealed planted world whose amendment pins the REAL Revision 1 dose_chain.py, and the check call that imports that real file"""
    c1 = T.C1(tmp, doc=c1_doc if c1_doc is not None else T.c1_doc())
    am, sha, _, _ = T.make_seal(tmp, c1=c1, dose_kw=dose_kw, extra_pins={"dose_chain_sha256": DOSE_CHAIN_R1_SHA256})
    return lambda **over: T.check_c2_seal(am, sha, c1.path, c1.sha, **{"config": {"C8": C8_FILE, "O4": O4_FILE}[config], "dose_chain": REAL_DOSE, **over})


def test_the_real_file_drives_the_seal_and_the_r1_amendment_is_banked_with_the_selection(tmp_path):
    r = real_call(tmp_path, "C8")()
    assert (r["selection"]["config"], r["selection"]["row"]) == ("C8", 1)
    assert r["selection"]["dose_chain_sha256"] == DOSE_CHAIN_R1_SHA256 and r["selection"]["dose_r1_amendment_sha256"] == DOSE_R1_AMENDMENT_SHA256


def test_a_read_licensed_by_frac_moved_alone_is_accepted_through_the_real_file_so_c2_config_was_handed_the_inert_block(tmp_path):
    """moved_share 1 %, frac_moved 16.5 %: licensed at Revision 1.  If `select_config` did not hand the real `c2_config` the `inert` block it would read NaN, call C1 not licensed and return row 4 (C2_CONFIG_NONE)"""
    r = real_call(tmp_path, "C8", T.c1_doc(moved_share=0.01, inert={"inert": False, "frac_moved": 0.165}))()
    assert (r["selection"]["config"], r["selection"]["row"]) == ("C8", 1)


REAL_PATH_CASES = [
    ("c1_moved_share_34pct_frac_moved_1pct", "C8", T.c1_doc(moved_share=0.34, inert={"inert": False, "frac_moved": 0.01}), None, "LICENCE_INERT"),
    ("dose_only_path_moved_share_50pct_frac_moved_1pct", "O4", UNLIC(), {**O1_DOSE, "moved_share": 0.5, "inert": {"inert": False, "frac_moved": 0.01}}, "LICENCE_INERT"),
    ("dose_moved_share_50pct_frac_moved_1pct_is_row_1_so_O4_is_refused", "O4", None, {**O1_DOSE, "moved_share": 0.5, "inert": {"inert": False, "frac_moved": 0.01}, "contrast_O4_minus_C8": POS}, "CONFIG_NOT_C2_CONFIG"),
]


@pytest.mark.parametrize("cid,config,c1d,dose_kw,reason", REAL_PATH_CASES, ids=[c[0] for c in REAL_PATH_CASES])
def test_a_read_with_moved_share_5pct_or_more_and_frac_moved_below_5pct_is_NOT_licensed_through_c2s_path_with_the_real_file(tmp_path, cid, config, c1d, dose_kw, reason):
    e = raises("seal", real_call(tmp_path, config, c1d, dose_kw))
    assert e.code == 5 and e.reason == reason, (e.reason, e.detail)


# ---- the Revision 1 seal of the dose package (MM_RULING_C2FINAL.md points 1 and 3): the package C2 imports must carry AMENDMENT_DOSE_R1.md + .sha256, matching, sealed, and pinning THIS file
def copy_real_package(dst, with_r1=True, with_original=True):
    """a byte-exact copy of the real dose package (`<dst>/code/dose_chain.py`, and the amendment texts that sit beside `code/`); returns the copied dose_chain.py"""
    src = os.path.dirname(os.path.dirname(REAL_DOSE))
    os.makedirs(os.path.join(str(dst), "code"), exist_ok=True)
    shutil.copyfile(REAL_DOSE, os.path.join(str(dst), "code", "dose_chain.py"))
    for n in (([NC.DOSE_R1_AMENDMENT, NC.DOSE_R1_SIDECAR] if with_r1 else []) + (["AMENDMENT_DOSE.md", "AMENDMENT_DOSE.sha256"] if with_original else [])):
        shutil.copyfile(os.path.join(src, n), os.path.join(str(dst), n))
    return os.path.join(str(dst), "code", "dose_chain.py")


def rewrite_r1(path, fn):
    """rewrite the copied AMENDMENT_DOSE_R1.md with `fn(text)` and re-bind its sidecar to the new text (so only the checked property differs)"""
    pkg = os.path.dirname(os.path.dirname(path))
    am = os.path.join(pkg, NC.DOSE_R1_AMENDMENT)
    text = fn(open(am, "rb").read().decode("utf-8"))
    open(am, "wb").write(text.encode("utf-8"))
    open(os.path.join(pkg, NC.DOSE_R1_SIDECAR), "w", newline="\n").write(hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest() + "  " + NC.DOSE_R1_AMENDMENT + "\n")


def _append(path, name, data):
    open(os.path.join(os.path.dirname(os.path.dirname(path)), name) if name else path, "ab").write(data)


R1_RED = [
    ("no_r1_files_only_the_original_seal", lambda p: [os.remove(os.path.join(os.path.dirname(os.path.dirname(p)), n)) for n in (NC.DOSE_R1_AMENDMENT, NC.DOSE_R1_SIDECAR)], "DOSE_R1_SEAL_MISSING"),
    ("no_amendment_files_at_all", lambda p: [os.remove(os.path.join(os.path.dirname(os.path.dirname(p)), n)) for n in (NC.DOSE_R1_AMENDMENT, NC.DOSE_R1_SIDECAR, "AMENDMENT_DOSE.md", "AMENDMENT_DOSE.sha256")], "DOSE_R1_SEAL_MISSING"),
    ("r1_text_without_its_sidecar", lambda p: os.remove(os.path.join(os.path.dirname(os.path.dirname(p)), NC.DOSE_R1_SIDECAR)), "DOSE_R1_SEAL_MISSING"),
    ("r1_sidecar_without_its_text", lambda p: os.remove(os.path.join(os.path.dirname(os.path.dirname(p)), NC.DOSE_R1_AMENDMENT)), "DOSE_R1_SEAL_MISSING"),
    ("r1_text_edited_one_byte_after_the_seal", lambda p: _append(p, NC.DOSE_R1_AMENDMENT, b" "), "DOSE_R1_SHA_MISMATCH"),
    ("r1_sidecar_names_another_text", lambda p: open(os.path.join(os.path.dirname(os.path.dirname(p)), NC.DOSE_R1_SIDECAR), "w").write("0" * 64 + "\n"), "DOSE_R1_SHA_MISMATCH"),
    ("r1_sidecar_empty", lambda p: open(os.path.join(os.path.dirname(os.path.dirname(p)), NC.DOSE_R1_SIDECAR), "w").write(""), "DOSE_R1_SHA_MISMATCH"),
    ("r1_text_still_a_draft", lambda p: rewrite_r1(p, lambda t: t.replace("Status: **SEALED**", "Status: **DRAFT**", 1)), "DOSE_R1_NOT_SEALED"),
    ("r1_text_has_no_pins_block", lambda p: rewrite_r1(p, lambda t: t.replace("<!-- PINS-BEGIN -->", "").replace("<!-- PINS-END -->", "")), "DOSE_R1_NO_PINS"),
    ("r1_text_does_not_pin_dose_chain", lambda p: rewrite_r1(p, lambda t: t.replace("code:dose_chain.py ", "code:not_dose_chain.py ", 1)), "DOSE_R1_CODE_PIN_MISMATCH"),
    ("r1_text_pins_another_dose_chain", lambda p: rewrite_r1(p, lambda t: t.replace("code:dose_chain.py " + DOSE_CHAIN_R1_SHA256, "code:dose_chain.py " + "1" * 64, 1)), "DOSE_R1_CODE_PIN_MISMATCH"),
    ("dose_chain_edited_one_byte", lambda p: _append(p, None, b"\n# edited\n"), "DOSE_R1_CODE_PIN_MISMATCH"),
]


def test_the_r1_package_check_passes_a_byte_exact_copy_of_the_real_package_and_a_crlf_checkout(tmp_path):
    """the control that makes the RED arms below mean something: the unedited copy passes, and so does a CRLF checkout of the same bytes (the checks fold CRLF to LF, as every other sha in this package)"""
    p = copy_real_package(tmp_path / "ok")
    assert NC.check_dose_package_r1(p) == {"amendment_sha256": DOSE_R1_AMENDMENT_SHA256, "dose_chain_sha256": DOSE_CHAIN_R1_SHA256}
    crlf = copy_real_package(tmp_path / "crlf")
    am = os.path.join(os.path.dirname(os.path.dirname(crlf)), NC.DOSE_R1_AMENDMENT)
    for f in (crlf, am):
        data = open(f, "rb").read()                                # read first: open(f, "wb") truncates
        open(f, "wb").write(data.replace(b"\n", b"\r\n"))
    assert NC.check_dose_package_r1(crlf)["dose_chain_sha256"] == DOSE_CHAIN_R1_SHA256


@pytest.mark.parametrize("cid,mut,reason", R1_RED, ids=[c[0] for c in R1_RED])
def test_a_dose_package_that_is_not_sealed_at_r1_refuses_by_name(tmp_path, cid, mut, reason):
    p = copy_real_package(tmp_path / "pkg")
    mut(p)
    e = raises("seal", NC.check_dose_package_r1, p)
    assert e.code == 5 and e.reason == reason, (e.reason, e.detail)


def test_c2_refuses_a_dose_package_without_the_r1_files_and_an_edited_dose_chain_byte_on_its_own_call_path(tmp_path):
    """through `check_c2_seal`: in each arm the amendment's `dose_chain_sha256` pin is filled from the file C2 is handed, so the pin layer passes and ONLY the Revision 1 seal can refuse; then the pin layer's own arm"""
    def attempt(name, pkg_mut, pin_of):
        d = tmp_path / name
        p = copy_real_package(d / "pkg")
        pkg_mut(p)
        c1 = T.C1(d / "w")
        am, sha, _, _ = T.make_seal(d / "w", c1=c1, extra_pins={"dose_chain_sha256": pin_of(p)})
        return raises("seal", lambda: T.check_c2_seal(am, sha, c1.path, c1.sha, config=C8_FILE, dose_chain=p))
    own = lambda p: NC.sha_lf(p)                                   # noqa: E731
    e = attempt("no_r1", lambda p: [os.remove(os.path.join(os.path.dirname(os.path.dirname(p)), n)) for n in (NC.DOSE_R1_AMENDMENT, NC.DOSE_R1_SIDECAR)], own)
    assert e.code == 5 and e.reason == "DOSE_R1_SEAL_MISSING", (e.reason, e.detail)
    e = attempt("edited_byte", lambda p: _append(p, None, b"\n# edited\n"), own)
    assert e.code == 5 and e.reason == "DOSE_R1_CODE_PIN_MISMATCH", (e.reason, e.detail)
    e = attempt("edited_byte_amendment_pins_the_real_r1_sha", lambda p: _append(p, None, b"\n# edited\n"), lambda p: DOSE_CHAIN_R1_SHA256)
    assert e.code == 5 and e.reason == "DOSE_CHAIN_PIN_MISMATCH", (e.reason, e.detail)
    d = tmp_path / "control"                                       # control: the unedited package, imported from the copy by path, passes the same call
    p = copy_real_package(d / "pkg")
    c1 = T.C1(d / "w")
    am, sha, _, _ = T.make_seal(d / "w", c1=c1, extra_pins={"dose_chain_sha256": DOSE_CHAIN_R1_SHA256})
    r = T.check_c2_seal(am, sha, c1.path, c1.sha, config=C8_FILE, dose_chain=p)
    assert (r["selection"]["config"], r["selection"]["row"]) == ("C8", 1) and r["selection"]["dose_r1_amendment_sha256"] == DOSE_R1_AMENDMENT_SHA256


# ====================================================================================================== the fixture record (DQ7 (b))
def test_the_fixture_record_must_be_pinned_identical_of_carve_b_and_carry_the_sealed_code(tmp_path):
    ok = seal_for(tmp_path / "ok")[0]()
    assert ok["fixture"]["verdict"] == "IDENTICAL"
    call, c1, am, sha = seal_for(tmp_path / "args")
    assert raises("seal", lambda: call(fixture_record=None)).reason == "FIXTURE_RECORD_MISSING_ARGS"
    assert raises("seal", lambda: call(fixture_record=str(tmp_path / "nope.json"))).reason == "FIXTURE_RECORD_MISSING"
    ev = T.EVID[os.path.abspath(am)]
    open(ev.fixture_path, "a").write(" ")                                                                # edited after the seal: the pin catches it
    assert raises("seal", lambda: call()).reason == "FIXTURE_RECORD_PIN_MISMATCH"


FIXTURE_RED = [("differs", {"verdict": "DIFFERS"}, "FIXTURE_NOT_IDENTICAL"), ("lowercase", {"verdict": "identical"}, "FIXTURE_NOT_IDENTICAL"), ("none", {"verdict": None}, "FIXTURE_NOT_IDENTICAL"),
               ("tokens", {"tokens_sha256": "0" * 64}, "FIXTURE_RECORD_TOKENS"), ("schema", {"schema": "x"}, "FIXTURE_RECORD_UNREADABLE"), ("mode", {"mode": "c2"}, "FIXTURE_RECORD_UNREADABLE"),
               ("code_not_a_dict", {"code_sha256": []}, "FIXTURE_CODE_NOT_THE_SEALED_CODE")]


@pytest.mark.parametrize("cid,fx,reason", FIXTURE_RED, ids=[c[0] for c in FIXTURE_RED])
def test_each_fixture_record_red_arm(tmp_path, cid, fx, reason):
    refused(reason, tmp_path, "C8", None, None, fixture=fx)


@pytest.mark.parametrize("f", list(NC.OPERATIVE))
def test_a_fixture_that_ran_other_code_than_the_sealed_pin_is_refused_for_every_operative_file(tmp_path, f):
    def mut(rec):
        rec["code_sha256"][f] = "0" * 64
    e = refused("FIXTURE_CODE_NOT_THE_SEALED_CODE", tmp_path, "C8", None, None, fixture_mut=mut)
    assert f in e.detail


def test_a_fixture_record_missing_a_code_entry_is_refused(tmp_path):
    refused("FIXTURE_CODE_NOT_THE_SEALED_CODE", tmp_path, "C8", None, None, fixture_mut=lambda r: r["code_sha256"].pop("navpipe_chain.py"))


def test_step_b_and_l1_code_must_be_what_the_fixture_ran(tmp_path):
    def stale(rec):
        rec["extra_code_sha256"]["stepb/stepB.py"] = "0" * 64
    e = refused("FIXTURE_EXTRA_CODE_CHANGED", tmp_path / "a", "C8", None, None, fixture_mut=stale)
    assert "stepb/stepB.py" in e.detail
    e = refused("FIXTURE_EXTRA_CODE_CHANGED", tmp_path / "b", "C8", None, None, fixture_mut=lambda r: r["extra_code_sha256"].pop("stepb/label_stepB.py"))      # not recorded = not checkable = refused
    assert "stepb/label_stepB.py" in e.detail
    e = refused("FIXTURE_EXTRA_CODE_CHANGED", tmp_path / "c", "C8", None, None, fixture_mut=lambda r: r["extra_code_sha256"].update({"l1/tts_cem.py": "0" * 64}))
    assert "l1/tts_cem.py" in e.detail
    e = refused("FIXTURE_EXTRA_CODE_CHANGED", tmp_path / "d", "C8", None, None, fixture_mut=lambda r: r["extra_code_sha256"].update({"l1/not_a_file.py": "0" * 64}))
    assert "l1/not_a_file.py" in e.detail


def test_an_unreadable_fixture_record_is_refused(tmp_path):
    call, c1, am, sha = seal_for(tmp_path)
    ev = T.EVID[os.path.abspath(am)]
    open(ev.fixture_path, "w").write("{broken")
    new = NC.sha256_file(ev.fixture_path)
    import re, hashlib
    text = re.sub(r"^fixture_record_sha256 .*$", f"fixture_record_sha256 {new}", open(am, encoding="utf-8").read(), flags=re.M)
    open(am, "w", encoding="utf-8", newline="\n").write(text)
    s2 = hashlib.sha256(text.encode()).hexdigest()
    open(os.path.join(os.path.dirname(am), "AMEND.sha256"), "w").write(s2 + "  AMEND.md\n")
    assert raises("seal", lambda: T.check_c2_seal(am, s2, c1.path, c1.sha, config=C8_FILE)).reason == "FIXTURE_RECORD_UNREADABLE"


# ---- the record the fixture chain WRITES
def _fixture_chain(tmp, ok=True):
    cfg = CH.Cfg(str(tmp / "fx"), mode="fixture")
    ch = CH.Chain(cfg, runner=lambda *a, **k: 0, out=io.StringIO())
    ch.token_doc = {"tokens": ["a", "b"], "token_log": {"a": "l", "b": "l"}}
    ch.plan = {"plan_sha256": "p" * 64}
    ch.pins = {"ckpt_sha256": "k" * 64, "config_sha256": "c" * 64}
    ch.code_at_start = NC.code_shas(cfg.code_dir)
    os.makedirs(cfg.work, exist_ok=True)
    return ch


def test_the_fixture_chain_writes_its_record_and_c2_accepts_exactly_that_file(tmp_path):
    ch = _fixture_chain(tmp_path)
    path = ch.write_fixture_record(True, {"x": 1})
    rec = json.load(open(path))
    assert rec["verdict"] == "IDENTICAL" and rec["schema"] == NC.FIXTURE_SCHEMA and rec["mode"] == "fixture" and rec["tokens_sha256"] == NC.tokens_sha256(["a", "b"])
    assert rec["code_sha256"] == NC.code_shas(HERE) and set(rec["extra_code_sha256"]) >= {f"stepb/{f}" for f in NC.FIXTURE_STEPB_FILES}
    assert "ZZNAVPIPE_FIXTURE_RECORD IDENTICAL" in ch.out.getvalue()
    pins = {f"code:{f}": s for f, s in NC.code_shas(HERE).items()}
    pins["fixture_record_sha256"] = NC.sha256_file(path)
    rec["tokens_sha256"] = NC.CARVE_B_SHA256                                           # the real chain writes carve B's list; this planted chain wrote a 2-token list, so compare with its own
    got = NC.check_fixture_record(path, pins, NC.tokens_sha256(["a", "b"]))
    assert got["verdict"] == "IDENTICAL"


def test_the_fixture_record_says_differs_when_the_run_differs_or_the_code_moved_under_it(tmp_path):
    ch = _fixture_chain(tmp_path / "a")
    assert json.load(open(ch.write_fixture_record(False, {})))["verdict"] == "DIFFERS"
    ch2 = _fixture_chain(tmp_path / "b")
    ch2.code_at_start = {**ch2.code_at_start, "navpipe_core.py": "0" * 64}              # a file edited while the fixture ran
    rec = json.load(open(ch2.write_fixture_record(True, {})))
    assert rec["verdict"] == "DIFFERS" and rec["detail"]["code_changed_during_the_run"] is True


def test_fixture_identity_writes_the_record_with_the_verdict_it_reached(tmp_path, monkeypatch):
    ch = _fixture_chain(tmp_path)
    banked = tmp_path / "banked.json"
    keys = ("endpoint_dPDMS_points", "per_seed_points", "sub_score_deltas_points", "families", "inert", "moved_share")
    doc = {k: {"v": i} for i, k in enumerate(keys)}
    json.dump({"arms": {"P2HEAD": doc}}, open(banked, "w"))
    monkeypatch.setattr(CH, "BANKED_RESULT", str(banked))
    monkeypatch.setattr(CH, "fixture_row_compare", lambda *a, **k: {"ok": True})
    assert ch.fixture_identity(dict(doc)) is True
    assert json.load(open(ch.cfg.d("FIXTURE_RECORD.json")))["verdict"] == "IDENTICAL"
    bad = dict(doc)
    bad["inert"] = {"v": 99}
    assert ch.fixture_identity(bad) is False
    assert json.load(open(ch.cfg.d("FIXTURE_RECORD.json")))["verdict"] == "DIFFERS"


# ====================================================================================================== the ONE-RUN marker
IDENT = {"amendment_sha256": "a" * 64, "config_sha256": NC.CONFIG_C8_SHA256, "c1_result_sha256": "c" * 64, "dose_result_sha256": "d" * 64, "work": NC.norm_work("W")}


def test_one_run_marker_lifecycle(tmp_path):
    m = str(tmp_path / "m" / "ONE_RUN.json")
    assert NC.one_run_check(m, IDENT) == "NEW"
    NC.one_run_start(m, IDENT, "C8")
    assert json.load(open(m))["status"] == "STARTED" and NC.one_run_check(m, IDENT) == "RESUME"
    raises("seal", NC.one_run_start, m, IDENT, "C8")                                      # exclusive: a second start finds the file
    assert raises("seal", NC.one_run_start, m, IDENT, "O4").reason == "C2_SECOND_RUN_REFUSED"
    NC.one_run_finish(m, "9" * 64)
    d = json.load(open(m))
    assert d["status"] == "ANALYSED" and d["result_navpipe_sha256"] == "9" * 64 and d["hypothesis"] == NC.HYPOTHESIS
    assert raises("seal", NC.one_run_check, m, IDENT).reason == "C2_ALREADY_RUN"           # consumed: the same run, again
    assert raises("seal", NC.one_run_check, m, {**IDENT, "config_sha256": NC.CONFIG_O4_SHA256}).reason == "C2_ALREADY_RUN"      # consumed: ANY config


@pytest.mark.parametrize("key", list(NC.ONE_RUN_IDENTITY))
def test_a_started_marker_names_one_run_and_refuses_any_other(tmp_path, key):
    m = str(tmp_path / "ONE_RUN.json")
    NC.one_run_start(m, IDENT, "C8")
    other = {**IDENT, key: "different"}
    e = raises("seal", NC.one_run_check, m, other)
    assert e.reason == "C2_SECOND_RUN_REFUSED" and key in e.detail


def test_an_unreadable_marker_is_not_an_absent_one(tmp_path):
    m = str(tmp_path / "ONE_RUN.json")
    open(m, "w").write("{broken")
    assert raises("seal", NC.one_run_check, m, IDENT).reason == "C2_MARKER_UNREADABLE"
    json.dump({"hypothesis": "H-OTHER", "status": "STARTED"}, open(m, "w"))
    assert raises("seal", NC.one_run_check, m, IDENT).reason == "C2_MARKER_UNREADABLE"
    json.dump([1], open(m, "w"))
    assert raises("seal", NC.one_run_check, m, IDENT).reason == "C2_MARKER_UNREADABLE"


def test_chain_writes_the_marker_before_any_data_and_consumes_it_at_the_analysis(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_ONE_RUN_NEW H-SEARCH-P2-1 C8" in out and out.index("ONE_RUN_NEW") < out.index("DATA_OK") < out.index("BUNDLE_OK")
    m = json.load(open(w.marker))
    res_sha = NC.sha256_file(os.path.join(w.tmp, "work", "final", "RESULT_NAVPIPE.json"))
    assert m["status"] == "ANALYSED" and m["config"] == "C8" and m["result_navpipe_sha256"] == res_sha and m["work"] == NC.norm_work(os.path.join(w.tmp, "work"))
    assert m["amendment_sha256"] == w.sha and m["c1_result_sha256"] == w.c1.sha and m["dose_result_sha256"] == w.ev.dose_sha and m["config_sha256"] == NC.CONFIG_C8_SHA256


def test_a_second_run_of_the_same_or_any_other_config_or_work_dir_is_refused_before_any_child(tmp_path):
    w = T.World(tmp_path)
    assert w.chain().run() == 0
    n = len(w.calls)
    again = w.chain()
    assert again.run() == 5 and len(w.calls) == n and "ZZNAVPIPE_STOP C2_ALREADY_RUN" in again.out.getvalue()
    cfg = w.cfg()
    cfg.work = os.path.join(w.tmp, "work2")                                                   # another work dir does not hide the look
    other = CH.Chain(cfg, runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w.free, harness_fn=w.harness_fn)
    assert other.run() == 5 and "C2_ALREADY_RUN" in other.out.getvalue() and not os.path.exists(os.path.join(w.tmp, "work2", "chunks"))


def test_an_interrupted_run_resumes_but_another_work_dir_or_amendment_does_not(tmp_path):
    w = T.World(tmp_path)
    part = w.chain(max_chunks=1)
    assert part.run() == 0 and "ZZNAVPIPE_PARTIAL" in part.out.getvalue() and json.load(open(w.marker))["status"] == "STARTED"
    res = w.chain()
    assert res.run() == 0, res.out.getvalue()
    assert "ZZNAVPIPE_ONE_RUN_RESUME" in res.out.getvalue() and json.load(open(w.marker))["status"] == "ANALYSED"
    w2 = T.World(tmp_path / "second")
    p2 = w2.chain(max_chunks=1)
    assert p2.run() == 0
    cfg = w2.cfg()
    cfg.work = os.path.join(w2.tmp, "elsewhere")
    foreign = CH.Chain(cfg, runner=w2.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w2.free, harness_fn=w2.harness_fn)
    assert foreign.run() == 5 and "ZZNAVPIPE_STOP C2_SECOND_RUN_REFUSED" in foreign.out.getvalue() and "work" in foreign.out.getvalue()


def test_a_dry_run_reads_the_marker_but_never_writes_it(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    ch.check_c2()
    assert ch.one_run == "NEW" and not os.path.exists(w.marker)
    assert w.chain().run() == 0
    consumed = w.chain()
    with pytest.raises(NC.Refusal) as e:
        consumed.check_c2()
    assert e.value.reason == "C2_ALREADY_RUN"


def test_the_chain_banks_the_config_choice_and_the_fixture_record_it_read(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0
    out = ch.out.getvalue()
    assert f"ZZNAVPIPE_CONFIG_OK C8 row 1 config {NC.CONFIG_C8_SHA256[:12]}" in out and "ZZNAVPIPE_FIXTURE_RECORD_OK IDENTICAL" in out and "licensing_read C1" in out
    assert out.index("CONFIG_OK") < out.index("DATA_OK")
    doc = json.load(open(os.path.join(w.tmp, "work", "licence_c1.json")))
    assert doc["selection"]["config"] == "C8" and doc["selection"]["row"] == 1 and doc["fixture"]["verdict"] == "IDENTICAL" and doc["licence"]["licensing_read"] == "C1"


def test_the_chain_refuses_a_config_that_c2_config_did_not_return_before_any_data(tmp_path):
    w = T.World(tmp_path)
    cfg = w.cfg(config=O4_FILE)                                                           # row 1 returns C8
    ch = CH.Chain(cfg, runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w.free, harness_fn=w.harness_fn)
    assert ch.run() == 5 and not w.calls and "ZZNAVPIPE_STOP CONFIG_NOT_C2_CONFIG" in ch.out.getvalue() and not os.path.exists(w.marker)
    cfg = w.cfg(config=None)
    ch = CH.Chain(cfg, runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w.free, harness_fn=w.harness_fn)
    assert ch.run() == 5 and "ZZNAVPIPE_STOP CONFIG_MISSING" in ch.out.getvalue() and not os.path.exists(w.marker)


def test_the_chain_refuses_a_fixture_record_that_does_not_read_identical_before_any_data(tmp_path):
    w = T.World(tmp_path)
    cfg = w.cfg(fixture_record=None)
    ch = CH.Chain(cfg, runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w.free, harness_fn=w.harness_fn)
    assert ch.run() == 5 and not w.calls and "ZZNAVPIPE_STOP FIXTURE_RECORD_MISSING_ARGS" in ch.out.getvalue() and not os.path.exists(os.path.join(w.tmp, "work", "chunks"))


def test_every_pin_that_names_a_file_is_checked_against_that_file_in_the_chain(tmp_path):
    w = T.World(tmp_path)
    open(w.bcsv, "a").write("\n")                                                         # the banked CSV is not the file the amendment pinned
    ch = w.chain()
    assert ch.run() == 5 and not w.calls and "ZZNAVPIPE_STOP SEAL_PIN_MISMATCH banked_csv_sha256" in ch.out.getvalue()


def test_file_pin_mismatches_names_each_wrong_or_missing_file(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"x\r\ny\r\n")
    pins = {"ckpt_sha256": NC.sha256_file(str(f)), "dose_chain_sha256": NC.sha_lf(str(f)), "head_s0_sha256": "0" * 64}
    assert NC.file_pin_mismatches(pins, {"ckpt_sha256": str(f), "dose_chain_sha256": str(f), "head_s0_sha256": None}) == []                 # raw vs LF hashing, None skipped
    assert NC.file_pin_mismatches(pins, {"head_s0_sha256": str(f), "ckpt_sha256": str(tmp_path / "nope")}) == ["head_s0_sha256", "ckpt_sha256"]
    assert NC.sha256_file(str(f)) != NC.sha_lf(str(f))                                    # the two hashings really differ on CRLF


def test_main_passes_the_new_flags_and_names_no_default_config(tmp_path, monkeypatch, capsys):
    seen = {}

    class Fake:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def emit(self, *a):
            pass

        def check_c2(self):
            pass

        def disk(self, n):
            pass

        check_s0bank = check_fixture = check_c2

    monkeypatch.setattr(CH, "Chain", Fake)
    rc = CH.main(["--mode", "navtest", "--work", str(tmp_path / "w"), "--amendment", "a.md", "--amendment-sha256", "0" * 64, "--c1-result", "r.json", "--c1-result-sha256", "1" * 64, "--config", "c.json",
                  "--dose-result", "d.json", "--dose-result-sha256", "2" * 64, "--fixture-record", "f.json", "--dry-run"])
    c = seen["cfg"]
    assert rc == 0 and (c.config, c.dose_result, c.dose_result_sha256, c.fixture_record) == ("c.json", "d.json", "2" * 64, "f.json") and c.one_run_marker == NC.ONE_RUN_MARKER
    CH.main(["--mode", "navtest", "--work", str(tmp_path / "w2"), "--dry-run"])
    assert seen["cfg"].config is None                                                      # no default: the chain then refuses CONFIG_MISSING, it does not pick C8
    CH.main(["--mode", "fixture", "--work", str(tmp_path / "w3"), "--dry-run"])
    assert seen["cfg"].config == CH.CONFIG and seen["cfg"].config == NC.CONFIG_C8         # the fixture and s0bank modes keep the C8 default


# ====================================================================================================== DQ6 (b): P2's picks are RE-SCORED through the identical route
def test_crosscheck_rescored_picks_core():
    toks = [f"t{i}" for i in range(5)]
    sub = np.linspace(0.2, 0.9, 30).reshape(5, 6)
    p = np.linspace(0.3, 0.8, 5)
    ok = NC.crosscheck_rescored_picks(toks, (sub, p), (sub.copy(), p.copy()))
    assert ok["ok"] and ok["n_bad"] == 0 and ok["max_abs_diff"] == 0.0
    off = NC.crosscheck_rescored_picks(toks, (sub, p), (sub, p + np.array([0, 0, 0, 0, 2e-9])))              # ONE token of five, 2e-9 off: not the planted ones, still caught
    assert not off["ok"] and off["n_bad"] == 1 and off["bad_examples"] == ["t4"]
    sub2 = sub.copy()
    sub2[2, 3] += 5e-9
    assert not NC.crosscheck_rescored_picks(toks, (sub2, p), (sub, p))["ok"]
    assert NC.crosscheck_rescored_picks(toks, (sub, p + 9e-10), (sub, p))["ok"]                            # inside the tolerance
    nan = p.copy()
    nan[1] = np.nan
    r = NC.crosscheck_rescored_picks(toks, (sub, nan), (sub, p))
    assert not r["ok"] and r["n_bad"] == 1                                                  # NaN compares False: it is a mismatch, not a pass
    assert not NC.crosscheck_rescored_picks([], (sub[:0], p[:0]), (sub[:0], p[:0]))["ok"]
    assert not NC.crosscheck_rescored_picks(toks, (sub[:4], p[:4]), (sub, p))["ok"]


def test_the_pick_plan_must_be_p2s_banked_seam_row_byte_for_byte():
    toks = sorted(T.planted_doc()["tokens"])[:6]
    raw, pick = T.planted_bundle(toks)
    banked = T.banked_for(toks)
    plan = NC.exec_to_poses(NC.repair_last_heading(raw[np.arange(6), pick]))
    rows = np.stack([banked["poses"][t] for t in toks])
    assert np.array_equal(plan, rows)
    assert NC.check_pick_plan_is_banked(plan, toks, banked)["ok"]
    bad = plan.copy()
    bad[3, 0, 0] += np.float32(1e-6)
    r = NC.check_pick_plan_is_banked(bad, toks, banked)
    assert not r["ok"] and r["n_bad"] == 1
    z = plan.copy()
    z[1, 2, 2] = np.float32(-0.0) if plan[1, 2, 2] == 0 else plan[1, 2, 2]
    r = NC.check_pick_plan_is_banked(plan, toks + ["unknown"], {**banked})
    assert not r["ok"] and r["n_absent_from_banked"] == 1
    assert not NC.check_pick_plan_is_banked(plan[:0], [], banked)["ok"]


def test_chain_rescoring_goes_through_the_same_harness_route_pick_first_then_the_seeds_then_the_control(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    for cid in (0, 1):
        got = [x for x in w.labels if x.startswith(f"refe_navpipe_c{cid:02d}_")]
        assert got == [f"refe_navpipe_c{cid:02d}_{k}" for k in ("pick", "s0", "s1", "s2", "ctl")], got      # five harness calls per chunk, the pick first: a wrong route stops before the seeds are spent
    assert "ZZNAVPIPE_PICK_RESCORE_OK 0 11 tokens equal the banked CSV" in ch.out.getvalue() or "ZZNAVPIPE_PICK_RESCORE_OK 0 9 tokens" in ch.out.getvalue() or "ZZNAVPIPE_PICK_RESCORE_OK 0" in ch.out.getvalue()
    sc = np.load(os.path.join(w.tmp, "work", "chunks", "c00", "scores.npz"), allow_pickle=False)
    assert sc["pick_sub"].shape == (len(sc["token"]), 6) and sc["pick_pdms"].shape == (len(sc["token"]),)
    ident = json.loads(str(sc["identity"]))
    assert ident["pick_crosscheck"]["ok"] and ident["pick_plan"]["ok"] and ident["passed"] and ident["control_ok"]


def test_the_analysis_baseline_is_the_rescored_pick_not_the_banked_csv(tmp_path):
    w = T.World(tmp_path, harness_bias=5e-10)                                              # the re-scored picks are 5e-10 off the banked rows: inside the 1e-9 tolerance, but NOT equal
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    fin = np.load(os.path.join(w.tmp, "work", "final", "bundle_merged.npz"), allow_pickle=False)
    toks = [str(t) for t in fin["token"]]
    banked_sub, banked_p = NC.read_harness_csv(w.bcsv, toks)
    assert not np.array_equal(fin["pdms"][:, 0], banked_p) and np.allclose(fin["pdms"][:, 0], banked_p, atol=1e-9)
    assert not np.array_equal(fin["sub"][:, 0, :], banked_sub)
    resc = np.load(os.path.join(w.tmp, "work", "final", "scores_merged.npz"), allow_pickle=False)
    assert np.array_equal(fin["pdms"][:, 0], resc["pick_pdms"]) and np.array_equal(fin["sub"][:, 0, :], resc["pick_sub"])


def test_a_rescore_that_misses_the_banked_csv_outside_the_planted_tokens_still_stops(tmp_path):
    w = T.World(tmp_path)
    toks = NC.chunk_tokens(w.doc, w.plan["chunks"][0])
    import hashlib
    order = sorted(range(len(toks)), key=lambda i: hashlib.sha1(toks[i].encode()).hexdigest())
    planted = {toks[i] for i in order[:2]}
    off = toks[order[-1]]
    assert off not in planted
    w2 = T.World(tmp_path / "b", harness_bias=1e-3, bias_tokens={off})
    ch = w2.chain(t1id_k=2)
    assert ch.run() == 7
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_PICK_RESCORE_MISMATCH 0 1 of" in out and "G_PICK_RESCORE_DIFFERS_FROM_BANKED" in out and off in out
    st = NC.State(os.path.join(w2.tmp, "work", "navpipe_state.json"), w2.plan["plan_sha256"])
    assert st.status(0) == "SEARCHED" and os.path.exists(os.path.join(w2.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))


def test_a_route_that_cannot_see_the_control_shift_stops_at_the_planted_identity(tmp_path):
    w = T.World(tmp_path, blind_control=True)
    ch = w.chain()
    assert ch.run() == 7 and "ZZNAVPIPE_STOP G_T1ID_NOT_PASSED" in ch.out.getvalue() and "PICK_RESCORE_MISMATCH" not in ch.out.getvalue()


class NoPickExec(T.World):
    def search(self, argv):
        rc = super().search(argv)
        d = self.arg(argv, "--out-dir")
        for f in os.listdir(d):
            if f.startswith("shard_P2HEAD_"):
                z = {k: v for k, v in np.load(os.path.join(d, f), allow_pickle=False).items() if k != "pick_exec"}
                np.savez(os.path.join(d, f), **z)
        return rc


class WrongPickExec(T.World):
    def search(self, argv):
        rc = super().search(argv)
        d = self.arg(argv, "--out-dir")
        for f in os.listdir(d):
            if f.startswith("shard_P2HEAD_"):
                z = {k: v for k, v in np.load(os.path.join(d, f), allow_pickle=False).items()}
                z["pick_exec"] = z["pick_exec"] + np.float32(0.01)
                np.savez(os.path.join(d, f), **z)
        return rc


def test_a_search_output_without_the_pick_plan_or_with_another_plan_is_refused(tmp_path):
    w = NoPickExec(tmp_path / "a")
    ch = w.chain()
    assert ch.run() == 7 and "ZZNAVPIPE_STOP PICK_EXEC_MISSING" in ch.out.getvalue()
    w = WrongPickExec(tmp_path / "b")
    ch = w.chain()
    assert ch.run() == 7 and "ZZNAVPIPE_PICK_PLAN_NOT_BANKED 0" in ch.out.getvalue() and "ZZNAVPIPE_STOP G_PICK_PLAN_NOT_BANKED" in ch.out.getvalue()
    assert not getattr(w, "labels", [])                                                      # refused before the harness was called: nothing was scored on the wrong plan


def test_scores_written_before_the_rescore_existed_are_not_accepted_in_c2_mode(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0
    scores = os.path.join(w.tmp, "work", "chunks", "c00", "scores.npz")
    merged = os.path.join(w.tmp, "work", "chunks", "c00", "search_P2HEAD.npz")
    assert ch.scores_valid(scores, merged)[0]
    z = {k: v for k, v in np.load(scores, allow_pickle=False).items()}
    old = os.path.join(w.tmp, "old_scores.npz")
    np.savez(old, **{k: v for k, v in z.items() if not k.startswith("pick_")})
    ok, why = ch.scores_valid(old, merged)
    assert not ok and "DQ6" in why
    bad = dict(z)
    bad["pick_pdms"] = np.full_like(bad["pick_pdms"], np.nan)
    np.savez(old, **bad)
    assert not ch.scores_valid(old, merged)[0]
    fx = CH.Chain(CH.Cfg(str(tmp_path / "fx"), mode="fixture"), runner=lambda *a, **k: 0, out=io.StringIO())
    np.savez(old, **{k: v for k, v in z.items() if not k.startswith("pick_")})
    assert fx.scores_valid(old, merged)[0]                                                  # the fixture's label route has no re-scored picks


# ====================================================================================================== DQ5: the outcome names, and a safety miss in the headline LINE whatever the outcome
def _terms(lo, point, rng, sp, bar=0.10, bars=-0.25):
    d = {"point": point, "lo": lo, "hi": point + 0.3}
    return d, NA.c2_terms(d, rng, sp, {"bar_dpdms_points": bar, "bar_safety_points": bars})


MISS = {"nc": 0.0, "dac": -0.4, "ttc": 0.0, "c": 0.0}


@pytest.mark.parametrize("lo,point,rng,sp,name,neg,miss", [
    (-0.1, 0.5, 0.1, SAFE, "C2_FAIL_LOWER_CI", False, {}),
    (-0.1, 0.5, 0.1, MISS, "C2_FAIL_LOWER_CI", False, {"dac": -0.4}),                       # the safety miss is reported under EVERY name
    (0.05, 0.05, 0.01, SAFE, "C2_FAIL_BAR", False, {}),
    (0.05, 0.05, 0.01, MISS, "C2_FAIL_BAR", False, {"dac": -0.4}),
    (0.05, 0.2, 0.3, SAFE, "C2_FAIL_SEED_RANGE", False, {}),
    (0.05, 0.2, 0.3, MISS, "C2_FAIL_SEED_RANGE", False, {"dac": -0.4}),
    (0.05, 0.3, 0.1, MISS, "C2_SAFETY_MISS", False, {"dac": -0.4}),
    (0.05, 0.3, 0.1, SAFE, "C2_PASS", False, {}),
    (0.05, 0.10, 0.05, {**SAFE, "ttc": -0.25}, "C2_PASS", False, {}),                        # met exactly at the bars (dPDMS bar and safety bar are >=)
])
def test_c2_outcome_names_follow_the_sealed_precedence_and_flag_every_safety_miss(lo, point, rng, sp, name, neg, miss):
    d, v = _terms(lo, point, rng, sp)
    o = NA.c2_outcome(d, v, sp)
    assert (o["outcome"], o["negative"], o["safety_miss"]) == (name, neg, miss)
    assert o["outcome"] in NA.C2_OUTCOMES


def test_negative_is_flagged_only_when_the_upper_ci_is_below_zero():
    d = {"point": -0.5, "lo": -0.9, "hi": -0.1}
    v = NA.c2_terms(d, 0.1, SAFE, {"bar_dpdms_points": 0.1, "bar_safety_points": -0.25})
    assert NA.c2_outcome(d, v, SAFE) == {"outcome": "C2_FAIL_LOWER_CI", "negative": True, "safety_miss": {}}
    d2 = {"point": -0.1, "lo": -0.5, "hi": 0.3}
    assert NA.c2_outcome(d2, NA.c2_terms(d2, 0.1, SAFE, {"bar_dpdms_points": 0.1, "bar_safety_points": -0.25}), SAFE)["negative"] is False


def _planted_result(tmp_path, sp, point=0.5):
    """run the REAL analysis stage (the child `navpipe_analyse.main`) on a planted one-column world whose paired deltas we control, return the RESULT dict"""
    n = 24
    rs = np.random.RandomState(3)
    toks = [f"{i:04x}" for i in range(n)]
    logs = [f"log{i % 6}" for i in range(n)]
    sub = np.stack([np.ones(n), np.full(n, 0.9), np.full(n, 0.5), np.ones(n), np.ones(n), np.ones(n)], 1)
    pdms = sub[:, 0] * sub[:, 1] * (5 * sub[:, 2] + 5 * sub[:, 3] + 2 * sub[:, 4]) / 12.0
    return toks, logs, sub, pdms


def test_the_chain_headline_names_the_outcome_and_carries_the_safety_miss(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    res = json.load(open(os.path.join(w.tmp, "work", "final", "RESULT_NAVPIPE.json")))
    out = ch.out.getvalue()
    done = [ln for ln in out.splitlines() if ln.startswith("ZZNAVPIPE_DONE ")]
    assert len(done) == 1 and res["outcome"] in NA.C2_OUTCOMES and done[0].startswith("ZZNAVPIPE_DONE " + res["outcome"])
    assert ("SAFETY_MISS" in done[0]) == bool(res["safety_miss"])                           # the headline LINE carries it, whatever the outcome
    assert set(res["safety_miss"]) <= set(NC.LICENCE_SAFETY_TERMS)


class SafetyMissWorld(T.World):
    """the planted search moves every output 6 m sideways: dDAC collapses far below -0.25 points"""

    def search(self, argv):
        rc = super().search(argv)
        d = self.arg(argv, "--out-dir")
        for f in os.listdir(d):
            if f.startswith("shard_P2HEAD_"):
                z = {k: v for k, v in np.load(os.path.join(d, f), allow_pickle=False).items()}
                z["exec"] = z["exec"].copy()
                z["exec"][..., 1] += np.float32(6.0)
                z["moved"] = np.ones_like(z["moved"])
                np.savez(os.path.join(d, f), **z)
        return rc


def test_a_safety_miss_is_in_the_headline_line_even_when_the_outcome_is_not_the_safety_name(tmp_path):
    w = SafetyMissWorld(tmp_path)
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    res = json.load(open(os.path.join(w.tmp, "work", "final", "RESULT_NAVPIPE.json")))
    done = [ln for ln in ch.out.getvalue().splitlines() if ln.startswith("ZZNAVPIPE_DONE ")][0]
    assert "dac" in res["safety_miss"] and res["outcome"] != "C2_PASS"
    assert done.startswith("ZZNAVPIPE_DONE " + res["outcome"]) and " SAFETY_MISS " in done and '"dac"' in done


# ====================================================================================================== portability: the sibling packages are REPO-RELATIVE (item 4)
SRC_FILES = ("navpipe_chain.py", "navpipe_analyse.py", "navpipe_launch.py", "navpipe_bridge.py", "make_navtest_bundle.py")


def _fresh(code):
    env = {k: v for k, v in os.environ.items() if k not in ("SPC_STEPB_CODE", "L1_CODE", "SPC_CODE", "C2_DOSE_CHAIN", "C2_CONFIG_C8", "C2_CONFIG_O4")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r); import navpipe_core as N; %s" % (HERE, code)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout.strip().splitlines()[-1]


SCRATCH = bool(os.environ.get("NPMUT_SCRATCH"))        # mutation_check.py runs a copy of this package in a temp dir: the siblings are then named by environment variables and are not at their repo-relative place


@pytest.mark.skipif(SCRATCH, reason="mutation_check.py scratch copy: the siblings are not repo-relative to it")
def test_the_sibling_packages_resolve_repo_relative_and_exist():
    pkg = os.path.dirname(HERE)
    raw, research = os.path.dirname(pkg), os.path.dirname(os.path.dirname(os.path.dirname(pkg)))
    got = json.loads(_fresh("import json; print(json.dumps([N.STEPB_CODE, N.L1_CODE, N.SPC_CODE, N.CONFIG_C8, N.CONFIG_O4, N.find_launcher(), N.DOSE_CHAIN]))"))
    want = [os.path.join(raw, "2026-10-19-l2-stepB", "code"), os.path.join(research, "2026-10-18-ep-rl-lit", "l1_stepA"), os.path.join(raw, "2026-10-19-search-p2-confirm", "code"),
            os.path.join(raw, "2026-10-19-l2-stepB", "raw", "stepB_config.json"), os.path.join(raw, "2026-10-19-l2-stepB", "raw", "stepB_config_O4.json"),
            os.path.join(research, "2026-10-18-ep-rl-lit", "l1_stepA", "run_2026-10-19", "launch_stepA_ctype.py"), os.path.join(raw, "2026-10-19-search-p2-confirm", "dose", "code", "dose_chain.py")]
    assert [os.path.normpath(x) for x in got] == [os.path.normpath(x) for x in want]
    for p, must in ((want[0], ("stepB.py", "stepB_chain.py", "label_stepB.py", "analyse_stepB.py", "g_reach_stepb.py", "l2_scorer.py")), (want[1], ("stepA.py", "tts_cem.py", "real_scorer.py", "launch_stepA.py")),
                    (want[2], ("spc_analyse.py", "spc_chain.py", "seal_spc.py")), (os.path.join(want[1], "run_2026-10-19"), ("launch_stepA_ctype.py", "make_stepA_bundle.py"))):
        assert os.path.isdir(p), p
        for f in must:
            assert os.path.isfile(os.path.join(p, f)), (p, f)
    assert os.path.isfile(want[3]) and os.path.isfile(want[4]) and os.path.isfile(want[5])
    assert NC.sha_lf(want[3]) == NC.CONFIG_C8_SHA256 and NC.sha_lf(want[4]) == NC.CONFIG_O4_SHA256


def test_the_environment_variables_still_override_the_defaults(tmp_path):
    env = {**os.environ, "SPC_STEPB_CODE": str(tmp_path / "sb"), "L1_CODE": str(tmp_path / "l1"), "SPC_CODE": str(tmp_path / "spc"), "C2_DOSE_CHAIN": str(tmp_path / "dc.py"), "C2_CONFIG_C8": str(tmp_path / "c8.json"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-I", "-c", "import sys, json; sys.path.insert(0, %r); import navpipe_core as N; print(json.dumps([N.STEPB_CODE, N.L1_CODE, N.SPC_CODE, N.DOSE_CHAIN, N.CONFIG_C8, N.L1_PATCHED]))" % HERE],
                       capture_output=True, text=True, env=env)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert [os.path.normpath(x) for x in got] == [os.path.normpath(str(tmp_path / x)) for x in ("sb", "l1", "spc", "dc.py", "c8.json")] + [os.path.normpath(str(tmp_path / "l1" / "run_2026-10-19"))]


def test_no_source_file_still_names_a_dev_box_scratch_package_as_a_default():
    for f in SRC_FILES + ("navpipe_core.py",):
        text = chr(10).join(ln for ln in open(os.path.join(HERE, f), encoding="utf-8").read().splitlines() if not ln.lstrip().startswith("#"))       # the layout may be DOCUMENTED in a comment; it may not be a default
        for bad in ("par18/l2stepb/code", "par18/l1run/code", "par18/spc/code"):
            assert bad not in text, (f, bad)


@pytest.mark.skipif(SCRATCH, reason="mutation_check.py scratch copy: the siblings are not repo-relative to it")
def test_the_patched_l1_copies_shadow_the_unpatched_ones_on_sys_path():
    code = "N.add_code_paths(); import sys; print(sys.path.index(N.L1_PATCHED) < sys.path.index(N.L1_CODE) < sys.path.index(N.STEPB_CODE))"
    assert _fresh(code) == "True"
    code = "N.add_code_paths(); import make_stepA_bundle as M, os; print(os.path.basename(os.path.dirname(M.__file__)))"
    assert _fresh(code) == "run_2026-10-19"                                                 # the bf16-patched builder, not l1_stepA's older one
