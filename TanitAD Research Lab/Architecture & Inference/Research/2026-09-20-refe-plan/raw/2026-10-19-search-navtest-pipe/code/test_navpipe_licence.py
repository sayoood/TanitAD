"""Tests of the C1 LICENCE that replaced the EP-ADMISSION record in the navtest (C2) seal (MM_RULING_C2_LICENCE.md, sealed 2026-10-20 02:56).  CPU only; planted data except where marked (the real banked carve-B arrays
and the real sibling files are read, never modified).

The licence is: C1's RESULT_SPC.json, sha256-pinned, reads O1_CONFIRMED or O2_DIRECTIONAL, AND each of dNC / dDAC / dTTC / dC POINT >= -0.25, AND the moved share >= 5 %.  Every term has a RED arm that must REFUSE
(class `seal`, exit 5, with the NAMED reason: a later check must not be able to stand in for a removed one), and so must a missing or edited file.  mutation_check.py deletes each `@GUARD:G-LICENCE-*` line in turn.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
import navpipe_chain as CH  # noqa: E402
import test_navpipe as T  # noqa: E402

raises = T.raises
SPC = "C:/Users/Admin/par18/spc"


def seal_with(tmp_path, doc=None, state=None, raw=None, extra_pins=None):
    c1 = T.C1(tmp_path, doc=doc, state=state, raw=raw)
    am, sha, _, pins = T.make_seal(tmp_path, c1=c1, extra_pins=extra_pins)
    return am, sha, c1


def check(tmp_path, **kw):
    am, sha, c1 = seal_with(tmp_path, **kw)
    return NC.check_c2_seal(am, sha, c1.path, c1.sha)


def refused(tmp_path, reason, **kw):
    am, sha, c1 = seal_with(tmp_path, **kw)
    e = raises("seal", NC.check_c2_seal, am, sha, c1.path, c1.sha)
    assert e.code == 5 and e.reason == reason, (e.reason, e.detail)
    return e


SAFE = {"nc": 0.0, "dac": 0.17, "ttc": 0.25, "c": 0.0}


# ====================================================================================================== the terms: pass at the boundary, refuse just beyond it
def test_o2_directional_with_every_term_exactly_at_its_bar_licenses(tmp_path):
    r = check(tmp_path, doc=T.c1_doc(outcome="O2_DIRECTIONAL", safety_points={k: -0.25 for k in SAFE}, inert={"inert": False, "frac_moved": 0.05}))
    assert r["licence"]["outcome"] == "O2_DIRECTIONAL" and r["licence"]["safety_points"] == {k: -0.25 for k in SAFE}


@pytest.mark.parametrize("outcome", ["TRADE", "O3_NULL", "NEGATIVE", "O4_INERT", "O1_PASS", "o1_confirmed", "", None])
def test_only_o1_and_o2_license(tmp_path, outcome):
    refused(tmp_path, "LICENCE_OUTCOME", doc=T.c1_doc(outcome=outcome))


@pytest.mark.parametrize("fam", list(SAFE))
@pytest.mark.parametrize("outcome", ["O1_CONFIRMED", "O2_DIRECTIONAL"])
def test_each_safety_point_below_minus_0_25_refuses_for_o1_and_o2_alike(tmp_path, fam, outcome):
    e = refused(tmp_path, "LICENCE_SAFETY_MISS", doc=T.c1_doc(outcome=outcome, safety_points={**SAFE, fam: -0.2501}))
    assert f'"{fam}"' in e.detail                                                              # the failing family is named


def test_two_safety_misses_are_both_named(tmp_path):
    e = refused(tmp_path, "LICENCE_SAFETY_MISS", doc=T.c1_doc(safety_points={**SAFE, "nc": -1.0, "ttc": -0.5}))
    assert '"nc"' in e.detail and '"ttc"' in e.detail


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.1", True, [0.1]])
@pytest.mark.parametrize("fam", ["nc", "c"])
def test_an_unreadable_safety_point_refuses_rather_than_comparing_false(tmp_path, fam, bad):
    refused(tmp_path, "LICENCE_SAFETY_UNREADABLE", doc=T.c1_doc(safety_points={**SAFE, fam: bad}))      # NaN < -0.25 is False: without this guard a NaN would pass the safety bar


@pytest.mark.parametrize("sp", [None, {}, {"nc": 0.0, "dac": 0.0, "ttc": 0.0}, "x"])
def test_a_missing_safety_block_or_family_refuses(tmp_path, sp):
    refused(tmp_path, "LICENCE_SAFETY_UNREADABLE", doc=T.c1_doc(safety_points=sp))


def test_moved_share_below_5_percent_refuses_and_exactly_5_percent_passes(tmp_path):
    refused(tmp_path, "LICENCE_INERT", doc=T.c1_doc(inert={"inert": False, "frac_moved": 0.0499}))
    check(tmp_path / "ok", doc=T.c1_doc(inert={"inert": False, "frac_moved": 0.05}))


def test_an_inert_flag_refuses_even_when_the_fraction_looks_fine(tmp_path):
    refused(tmp_path, "LICENCE_INERT", doc=T.c1_doc(inert={"inert": True, "frac_moved": 0.2}))
    refused(tmp_path / "missing_flag", "LICENCE_INERT", doc=T.c1_doc(inert={"frac_moved": 0.2}))


@pytest.mark.parametrize("inert", [None, {}, {"inert": False}, {"inert": False, "frac_moved": None}, {"inert": False, "frac_moved": float("nan")}, {"inert": False, "frac_moved": "0.2"}, "x"])
def test_an_unreadable_moved_share_refuses_rather_than_comparing_false(tmp_path, inert):
    refused(tmp_path, "LICENCE_MOVED_UNREADABLE", doc=T.c1_doc(inert=inert))                    # NaN < 0.05 is False: without this guard a NaN would pass the moved-share bar


# ====================================================================================================== it must BE the C1 read, not a read that merely looks like one
@pytest.mark.parametrize("kw", [{"read": "C2"}, {"read": None}, {"n_tokens": 400}, {"n_logs": 23}, {"arm": "L2 head seed 0"}, {"arm": None}], ids=lambda k: json.dumps(k))
def test_a_result_that_is_not_the_c1_read_refuses(tmp_path, kw):
    refused(tmp_path, "LICENCE_NOT_C1", doc=T.c1_doc(**kw))                                    # e.g. carve B (400 tokens) reads O1_CONFIRMED too


def test_a_result_with_no_identity_keys_refuses(tmp_path):
    d = T.c1_doc()
    for k in ("read", "n_tokens", "n_logs", "arm"):
        d.pop(k)
    refused(tmp_path, "LICENCE_NOT_C1", doc=d)


def test_the_chain_state_beside_the_result_must_exist(tmp_path):
    refused(tmp_path, "LICENCE_CHAIN_STATE_MISSING", state=False)
    c1 = T.C1(tmp_path / "bad", state=None)
    open(os.path.join(c1.dir, "chain_state.json"), "w").write("{not json")
    am, sha, _, _ = T.make_seal(tmp_path / "bad", c1=c1)
    assert raises("seal", NC.check_c2_seal, am, sha, c1.path, c1.sha).reason == "LICENCE_CHAIN_STATE_MISSING"


def test_a_result_from_another_prereg_refuses(tmp_path):
    refused(tmp_path, "LICENCE_PREREG_MISMATCH", state={"done": list(NC.C1_STAGES), "prereg_sha256": "0" * 64})
    refused(tmp_path / "none", "LICENCE_PREREG_MISMATCH", state={"done": list(NC.C1_STAGES)})
    refused(tmp_path / "list", "LICENCE_PREREG_MISMATCH", state=["done"])


@pytest.mark.parametrize("stage", list(NC.C1_STAGES))
def test_a_c1_chain_that_did_not_finish_each_stage_refuses(tmp_path, stage):
    done = [s for s in NC.C1_STAGES if s != stage]
    e = refused(tmp_path, "LICENCE_C1_INCOMPLETE", state={"done": done, "prereg_sha256": T.PLANT_PREREG})
    assert stage in e.detail


# ====================================================================================================== the sha pin, the missing file, the edited file
def test_a_missing_result_file_refuses(tmp_path):
    am, sha, c1 = seal_with(tmp_path)
    e = raises("seal", NC.check_c2_seal, am, sha, str(tmp_path / "nope" / "RESULT_SPC.json"), c1.sha)
    assert e.reason == "LICENCE_RESULT_MISSING" and e.code == 5


def test_a_wrong_typed_sha_refuses_and_an_uppercase_correct_one_passes(tmp_path):
    am, sha, c1 = seal_with(tmp_path)
    assert raises("seal", NC.check_c2_seal, am, sha, c1.path, "0" * 64).reason == "LICENCE_RESULT_SHA_MISMATCH"
    assert NC.check_c2_seal(am, sha, c1.path, c1.sha.upper())["licence"]["c1_result_sha256"] == c1.sha


def test_a_result_edited_after_its_sha_was_taken_refuses(tmp_path):
    am, sha, c1 = seal_with(tmp_path)
    d = json.load(open(c1.path))
    d["outcome"] = "O2_DIRECTIONAL"                                                            # a one-word edit of an otherwise valid licence
    json.dump(d, open(c1.path, "w"), indent=1)
    assert raises("seal", NC.check_c2_seal, am, sha, c1.path, c1.sha).reason == "LICENCE_RESULT_SHA_MISMATCH"


def test_a_missing_c1_argument_refuses_after_a_valid_amendment(tmp_path):
    am, sha, c1 = seal_with(tmp_path)
    assert raises("seal", NC.check_c2_seal, am, sha, None, None).reason == "LICENCE_MISSING_ARGS"
    assert raises("seal", NC.check_c2_seal, am, sha, c1.path, None).reason == "LICENCE_MISSING_ARGS"
    assert raises("seal", NC.check_c2_seal, am, sha, None, c1.sha).reason == "LICENCE_MISSING_ARGS"


def test_an_amendment_sealed_after_c1_may_pin_the_result_and_the_pin_binds(tmp_path):
    c1 = T.C1(tmp_path)
    am, sha, _, _ = T.make_seal(tmp_path, c1=c1, extra_pins={"c1_result_sha256": c1.sha})
    assert NC.check_c2_seal(am, sha, c1.path, c1.sha)["licence"]["c1_result_sha256"] == c1.sha
    other = T.C1(tmp_path / "other", doc=T.c1_doc(endpoint_dPDMS_points={"point": 0.9, "lo": 0.5, "hi": 1.3}))   # a different, equally valid C1-shaped file
    am2, sha2, _, _ = T.make_seal(tmp_path / "other", c1=other, extra_pins={"c1_result_sha256": c1.sha})
    assert raises("seal", NC.check_c2_seal, am2, sha2, other.path, other.sha).reason == "LICENCE_RESULT_PIN_MISMATCH"


@pytest.mark.parametrize("raw,why", [("{not json", "JSONDecodeError"), ("[1, 2]", "not a JSON object"), ("", "JSONDecodeError")])
def test_an_unreadable_or_non_object_result_refuses(tmp_path, raw, why):
    e = refused(tmp_path, "LICENCE_RESULT_UNREADABLE", raw=raw)
    assert why in e.detail


# ====================================================================================================== the amendment's own licence pins
def test_the_amendment_must_pin_the_licence_ruling_this_code_enforces(tmp_path):
    am, sha, c1, pins = T.make_seal(tmp_path, tamper_pin="licence_ruling_sha256")
    assert raises("seal", NC.check_c2_seal, am, sha, c1.path, c1.sha).reason == "SEAL_LICENCE_RULING_PIN"


def test_an_old_admission_pin_is_neither_required_nor_a_substitute_for_the_licence(tmp_path):
    am, sha, c1, pins = T.make_seal(tmp_path, extra_pins={"admission_record_sha256": "a" * 64})
    assert NC.check_c2_seal(am, sha, c1.path, c1.sha)["licence"]["outcome"] == "O1_CONFIRMED"              # an extra pin is harmless
    bad = T.C1(tmp_path / "bad", doc=T.c1_doc(outcome="TRADE"))
    am2, sha2, _, _ = T.make_seal(tmp_path / "bad", c1=bad, extra_pins={"admission_record_sha256": "a" * 64})
    assert raises("seal", NC.check_c2_seal, am2, sha2, bad.path, bad.sha).reason == "LICENCE_OUTCOME"      # an ADMIT record would not have helped; the licence is the only key


# ====================================================================================================== the constants ARE the ruling's, and the schema IS the writer's
def test_licence_constants_are_the_sealed_rulings():
    assert NC.LICENCE_SAFETY_MIN == -0.25 and NC.LICENCE_MOVED_MIN == 0.05
    assert NC.LICENCE_OUTCOMES == ("O1_CONFIRMED", "O2_DIRECTIONAL") and (NC.C1_N_TOKENS, NC.C1_N_LOGS) == (905, 24)
    import analyse_stepB as AB
    assert NC.LICENCE_SAFETY_TERMS == tuple(AB.SAFETY_TERMS)


@pytest.mark.skipif(not os.path.exists(f"{SPC}/MM_RULING_C2_LICENCE.md"), reason="the sealed ruling is absent")
def test_licence_ruling_sha_is_the_sealed_files():
    p = f"{SPC}/MM_RULING_C2_LICENCE.md"
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == NC.LICENCE_RULING_SHA256
    assert open(f"{SPC}/MM_RULING_C2_LICENCE.sha256").read().split()[0] == NC.LICENCE_RULING_SHA256
    txt = open(p, encoding="utf-8").read()
    assert ("-0.25" in txt or "−0.25" in txt) and "5 %" in txt and "O1_CONFIRMED or O2_DIRECTIONAL" in txt


@pytest.mark.skipif(not os.path.exists(f"{SPC}/code/spc_chain.py"), reason="the C1 chain source is absent")
def test_c1_stages_are_the_c1_chains_own_analysis_preconditions():
    src = open(f"{SPC}/code/spc_chain.py", encoding="utf-8").read()
    need = re.search(r"need = \[([^\]]*)\]", src).group(1)
    assert tuple(s.strip().strip('"\'') for s in need.split(",")) == NC.C1_STAGES
    assert 'self.state("SEAL_OK", prereg_sha256=want)' in src and 'c.d("RESULT_SPC.json")' in src          # the chain_state.json key and the file this guard reads


@pytest.mark.skipif(not os.path.exists(f"{SPC}/PREREG_SEARCH_P2_CONFIRM.md"), reason="the sealed C1 prereg is absent")
def test_the_planted_c1_prereg_is_the_real_sealed_one():
    assert hashlib.sha256(open(f"{SPC}/PREREG_SEARCH_P2_CONFIRM.md", "rb").read().replace(b"\r\n", b"\n")).hexdigest() == T.PLANT_PREREG


@T.needs_banked_b
def test_the_licence_reads_the_real_analyse_c1_schema(tmp_path):
    """REAL writer, REAL banked carve-B arrays (the C1 analysis UNCHANGED): carve B reads O1 with the same numeric terms, so the numeric terms accept it, and the identity term is what refuses it (400 tokens, not C1)"""
    import numpy as np
    import spc_analyse as SA
    b = {k: v for k, v in np.load(os.path.join(CH.SRC_BUNDLE, "bundle.npz"), allow_pickle=False).items()}
    so = {k: v for k, v in np.load(CH.BANKED_SEARCH, allow_pickle=False).items()}
    lb = {k: v for k, v in np.load(CH.BANKED_LABELS, allow_pickle=False).items()}
    res = SA.analyse_c1(b, so, lb, with_concordance=False)
    text = json.dumps(res, indent=1, default=lambda o: None if (isinstance(o, float) and not __import__("math").isfinite(o)) else (o.item() if hasattr(o, "item") else str(o)))
    real = json.loads(text)
    assert real["outcome"] == "O1_CONFIRMED" and real["n_tokens"] == 400 and set(real["safety_points"]) == set(NC.LICENCE_SAFETY_TERMS) and "frac_moved" in real["inert"]
    refused(tmp_path / "as_is", "LICENCE_NOT_C1", raw=text)                                    # the carve-B read as the writer wrote it: not C1
    r = check(tmp_path / "relabelled", raw=text.replace('"n_tokens": 400', '"n_tokens": 905', 1))
    assert r["licence"]["outcome"] == "O1_CONFIRMED" and r["licence"]["moved_share_inert_frac_moved"] == real["inert"]["frac_moved"] > 0.05
    assert r["licence"]["safety_points"] == {k: float(real["safety_points"][k]) for k in NC.LICENCE_SAFETY_TERMS}


# ====================================================================================================== at chain level: the licence is read FIRST, banked, and the EP-ADMISSION flag is gone
def test_chain_banks_the_licence_it_read_and_names_it_as_the_basis(tmp_path):
    w = T.World(tmp_path)
    ch = w.chain()
    assert ch.run() == 0, ch.out.getvalue()
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_LICENCE_OK O1_CONFIRMED" in out and "ZZNAVPIPE_LICENCE_BASIS" in out and "ZZNAVPIPE_ADMISSION" not in out
    assert out.index("ZZNAVPIPE_LICENCE_OK") < out.index("ZZNAVPIPE_DATA_OK")
    lic = json.load(open(os.path.join(w.tmp, "work", "licence_c1.json")))
    assert lic["amendment_sha256"] == w.sha and lic["licence"]["c1_result_sha256"] == w.c1.sha and lic["licence"]["bars"] == {"safety_min": -0.25, "moved_min": 0.05, "outcomes": ["O1_CONFIRMED", "O2_DIRECTIONAL"]}


@pytest.mark.parametrize("doc,reason", [(T.c1_doc(outcome="TRADE"), "LICENCE_OUTCOME"), (T.c1_doc(safety_points={**SAFE, "ttc": -0.3}), "LICENCE_SAFETY_MISS"),
                                        (T.c1_doc(inert={"inert": False, "frac_moved": 0.01}), "LICENCE_INERT"), (T.c1_doc(n_tokens=400), "LICENCE_NOT_C1")], ids=["trade", "safety", "inert", "not_c1"])
def test_chain_refuses_an_unlicensed_c1_before_opening_any_data_or_child(tmp_path, doc, reason):
    w = T.World(tmp_path)
    w.c1 = T.C1(tmp_path, doc=doc, name="c1_unlicensed")
    ch = w.chain()
    ch.cfg.token_file = str(tmp_path / "does_not_exist.json")                                   # if the data were opened first the stop would be a crash, not the licence's reason
    assert ch.run() == 5 and not w.calls
    out = ch.out.getvalue()
    assert f"ZZNAVPIPE_STOP {reason}" in out and "DATA_OK" not in out and "LICENCE_OK" not in out and not os.path.exists(os.path.join(w.tmp, "work", "chunks")) and not os.path.exists(os.path.join(w.tmp, "work", "licence_c1.json"))


def test_the_chain_without_a_c1_result_refuses_with_the_licence_reason(tmp_path):
    w = T.World(tmp_path)
    cfg = w.cfg()
    cfg.c1_result = cfg.c1_result_sha256 = None
    ch = CH.Chain(cfg, runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: w.free, harness_fn=w.harness_fn)
    assert ch.run() == 5 and not w.calls and "ZZNAVPIPE_STOP LICENCE_MISSING_ARGS" in ch.out.getvalue()


def test_the_command_line_takes_the_c1_flags_and_no_longer_takes_an_admission_record(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        CH.main(["--mode", "navtest", "--work", str(tmp_path / "w"), "--admission-record", "x.json", "--dry-run"])
    assert e.value.code == 2                                                                   # argparse: unrecognised arguments
    rc = CH.main(["--mode", "navtest", "--work", str(tmp_path / "w2"), "--amendment", "nope.md", "--amendment-sha256", "0" * 64, "--c1-result", "r.json", "--c1-result-sha256", "0" * 64, "--dry-run"])
    assert rc == 5 and "SEAL_AMENDMENT_MISSING" in capsys.readouterr().out
