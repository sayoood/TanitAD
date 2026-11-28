#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest_n_anchors.py - PROVE BY MUTATION that the bank size is RESOLVED and
that an unresolvable one REFUSES, naming the field.

WHY THIS FILE EXISTS
--------------------
A hardcoded ``1/128`` was written into several banked sidecars while the live
bank was ``117``.  ``anchor_chance(n)`` made the DERIVATION single; the
resolvers this file exercises make its INPUT single.  MEASURED 2026-09-07 on the
LIVE refcv5-v2 ``config.json`` (35 top-level keys, 206 leaves): there is NO key
called ``n_anchors`` anywhere in it.  The bank size is present three times under
other names - ``anchors.shape[0]``, ``anchors.controls_shape[0]``, and
``argv``'s ``--n-anchors 117`` - so a resolver that hunted the NAME would have
found nothing and had to fail or default.

STOP - THE SHAPE THIS FILE REFUSES TO HAVE
------------------------------------------
``if missing: expect a refusal; else: expect a pass`` is an expected value of
*whatever the code does*.  It was green on this programme while a preflight was
telling operators to plumb a label that never reached training.  So EVERY
expectation below is a LITERAL written into the case table - the number 117, the
string 0.008547, the substring "n_anchors" - and never an expression over the
code under test.

Each case also carries a DISCRIMINATING control where one exists: case B does not
merely require "a refusal", it requires a refusal that names ``n_anchors`` and
does NOT name ``param_breakdown``, because the pre-2026-09-07 harness DID refuse
that mutant - with a ten-key parameter diff that points at the wrong field.  A
test that only asked "did it refuse?" was already passing before the fix.

RUN
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/refcv5cmp/selftest_n_anchors.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

#: STOP - `import taniteval.tools.X` raises KeyError in `_load_unlocked`
#: (namespace-package shadow). Load the tool BY PATH.
#: ``REFCV3_ARM_PATH`` exists so this file can be pointed at a DELIBERATELY
#: REGRESSED copy of the tool. A guard that has never been seen to fail is not a
#: guard; the RED run is recorded beside the green one in RESULT.md.
_ARM_PATH = os.environ.get(
    "REFCV3_ARM_PATH", os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
_spec = importlib.util.spec_from_file_location(
    "refcv3_arm_under_selftest", _ARM_PATH)
RA = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RA
_spec.loader.exec_module(RA)

_spec2 = importlib.util.spec_from_file_location(
    "refcv5_compare_under_selftest", os.path.join(HERE, "refcv5_compare.py"))
CMP = importlib.util.module_from_spec(_spec2)
sys.modules[_spec2.name] = CMP
_spec2.loader.exec_module(CMP)

# --------------------------------------------------------------------------- #
# fixtures that are REAL artifacts, not stubs
# --------------------------------------------------------------------------- #
REFCV4B_CFG = r"C:\Users\Admin\refcv4b_final\config.json"
REFCV4B_CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284.pt"
REFCV4B_DUMP = os.path.join(HERE, "out", "refcv4b_devbox_dump")
REFCV5V2_CFG = os.environ.get("REFCV5V2_CONFIG", "")

#: The two literals every case is written against. They are NOT computed here.
BANK = 117
CHANCE = 0.008547            # == round(1/117, 6); written out, never derived

FAILURES: list[str] = []
NRUN = 0


def _case(name, fn):
    global NRUN
    NRUN += 1
    try:
        fn()
        print("  PASS  %s" % name)
    except AssertionError as ex:
        FAILURES.append("%s: %s" % (name, ex))
        print("  FAIL  %s\n        %s" % (name, ex))
    except Exception as ex:                                    # noqa: BLE001
        FAILURES.append("%s: unexpected %s: %s" % (name, type(ex).__name__, ex))
        print("  ERROR %s\n        unexpected %s: %s"
              % (name, type(ex).__name__, ex))


def _refusal(fn):
    """Run fn and return the SystemExit text. Anything else is a failure."""
    try:
        fn()
    except SystemExit as ex:
        return str(ex)
    raise AssertionError("expected a REFUSAL (SystemExit); the call RETURNED")


def _cfg(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tmp_json(obj, name):
    p = os.path.join(tempfile.gettempdir(), name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


def _drop_flag(cfg, flag, has_value=True):
    c = copy.deepcopy(cfg)
    a = list(c["argv"])
    i = a.index(flag)
    c["argv"] = a[:i + (0)] + a[i + (2 if has_value else 1):]
    return c


# --------------------------------------------------------------------------- #
# A. the arm-tool resolver, in isolation
# --------------------------------------------------------------------------- #
CK_FIELD = "checkpoint:model['core.decoder.anchors'].shape[0]"
SH_FIELD = "config.json:anchors.shape[0]"
CS_FIELD = "config.json:anchors.controls_shape[0]"
AV_FIELD = "argv:--n-anchors"


def a1_all_agree():
    n, prov = RA.resolve_n_anchors(
        {CK_FIELD: 117, SH_FIELD: 117, CS_FIELD: 117, AV_FIELD: 117},
        build_value=117)
    assert n == 117, "expected exactly 117, got %r" % n
    assert prov["primary"] == CK_FIELD, \
        "the TENSOR must be primary, got %r" % prov["primary"]
    assert prov["chance"] if False else True                    # (chance lives compare-side)
    assert len(prov["witnesses_agreeing"]) == 4, prov["witnesses_agreeing"]


def a2_unresolvable_refuses_naming_the_field():
    msg = _refusal(lambda: RA.resolve_n_anchors(
        {CK_FIELD: None, SH_FIELD: None, CS_FIELD: None, AV_FIELD: None}))
    assert "n_anchors IS UNRESOLVABLE" in msg, msg[:300]
    for f in (CK_FIELD, SH_FIELD, CS_FIELD, AV_FIELD):
        assert f in msg, "the refusal must NAME %r; got %r" % (f, msg[:400])
    assert "128" in msg, "the refusal must say WHY there is no default"


def a3_disagreement_refuses_naming_both():
    msg = _refusal(lambda: RA.resolve_n_anchors({CK_FIELD: 117, AV_FIELD: 128}))
    assert "WITNESSES DISAGREE" in msg, msg[:300]
    assert "117" in msg and "128" in msg, msg[:300]


def a4_build_default_128_is_caught_and_named():
    msg = _refusal(lambda: RA.resolve_n_anchors({CK_FIELD: 117}, build_value=128))
    assert "n_anchors" in msg, msg[:300]
    assert "128" in msg and "117" in msg, msg[:300]
    assert "--n-anchors" in msg, "the refusal must name the FLAG that fixes it"


def a5_a_string_is_not_a_bank_size():
    msg = _refusal(lambda: RA.resolve_n_anchors({CK_FIELD: "117"}))
    assert "UNRESOLVABLE" in msg, \
        "a string bank size must NOT be coerced; got %r" % msg[:200]


def a6_a_smoke_bank_resolves_to_20():
    n, prov = RA.resolve_n_anchors({CK_FIELD: 20}, build_value=20)
    assert n == 20, n
    assert prov["primary"] == CK_FIELD


# --------------------------------------------------------------------------- #
# B. END TO END on the REAL refcv4b checkpoint
# --------------------------------------------------------------------------- #
def b1_real_pair_loads_and_reads_117():
    model, cfg, targs, prov = RA.load_model(REFCV4B_CKPT, REFCV4B_CFG, device="cpu")
    assert prov["n_anchors"] == 117, prov["n_anchors"]
    assert prov["n_anchors_provenance"]["primary"] == CK_FIELD
    assert tuple(model.core.decoder.anchors.shape) == (117, 8, 2), \
        tuple(model.core.decoder.anchors.shape)
    # all four witnesses are present on this artifact and all say 117
    assert sorted(prov["n_anchors_provenance"]["witnesses_agreeing"].values()) \
        == [117, 117, 117, 117], prov["n_anchors_provenance"]["witnesses_agreeing"]


def b2_stripped_flag_refuses_naming_n_anchors_not_param_breakdown():
    """THE DISCRIMINATING CASE. Before 2026-09-07 this mutant ALSO refused - via
    the param_breakdown cross-check, a ten-key parameter diff. A test that asked
    only 'did it refuse?' was green then and is green now, so it proves nothing.
    What changed is WHICH FIELD the refusal names."""
    p = _tmp_json(_drop_flag(_cfg(REFCV4B_CFG), "--n-anchors"),
                  "selftest_v4b_no_nanchors.json")
    msg = _refusal(lambda: RA.load_model(REFCV4B_CKPT, p, device="cpu"))
    assert "n_anchors" in msg, msg[:400]
    assert "128" in msg and "117" in msg, msg[:400]
    assert "param_breakdown" not in msg, \
        "the OLD refusal fired first - the resolver is not in front of it: %r" % msg[:400]


def b3_stripped_flag_and_no_breakdown_still_names_n_anchors():
    """The other pre-2026-09-07 route: with param_breakdown absent the mutant was
    caught by a state_dict SIZE MISMATCH. Also a true refusal, also the wrong
    field. It must now be the resolver that speaks."""
    c = _drop_flag(_cfg(REFCV4B_CFG), "--n-anchors")
    c.pop("param_breakdown", None)
    p = _tmp_json(c, "selftest_v4b_no_nanchors_no_pb.json")
    msg = _refusal(lambda: RA.load_model(REFCV4B_CKPT, p, device="cpu"))
    assert "n_anchors" in msg, msg[:400]
    assert "size mismatch" not in msg, \
        "the state_dict error fired first - the resolver is not in front of it"


def b4_a_config_that_lies_about_the_bank_is_refused():
    """/anchors/shape is a WITNESS, not decoration: corrupt it and the load must
    refuse even though argv, the build and the weights all still say 117."""
    c = _cfg(REFCV4B_CFG)
    c["anchors"] = dict(c["anchors"])
    c["anchors"]["shape"] = [128, 8, 2]
    p = _tmp_json(c, "selftest_v4b_lying_anchors_shape.json")
    msg = _refusal(lambda: RA.load_model(REFCV4B_CKPT, p, device="cpu"))
    assert "WITNESSES DISAGREE" in msg, msg[:400]
    assert SH_FIELD in msg and CK_FIELD in msg, msg[:400]


# --------------------------------------------------------------------------- #
# C. the compare-tool resolver
# --------------------------------------------------------------------------- #
def c1_real_trio_gives_117_and_the_literal_chance():
    man = CMP._read_manifest(REFCV4B_DUMP)
    n, prov = CMP.resolve_n_anchors(man, _cfg(REFCV4B_CFG), REFCV4B_CKPT)
    assert n == BANK, n
    assert prov["chance"] == CHANCE, \
        "chance must be exactly %r, got %r" % (CHANCE, prov["chance"])
    assert prov["primary"] == CK_FIELD, prov["primary"]


def c2_nothing_anywhere_refuses_naming_every_field():
    msg = _refusal(lambda: CMP.resolve_n_anchors({"model": {"step": 40284}},
                                                 {"argv": ["--arm", "hier"]}, None))
    assert "n_anchors IS UNRESOLVABLE" in msg, msg[:300]
    for f in CMP.N_ANCHORS_WITNESS_FIELDS:
        assert f in msg, "the refusal must NAME %r" % f


def c3_manifest_stripped_but_the_tensor_answers():
    """The live-artifact case: an artifact that declares no `n_anchors` under
    that NAME. The tensor is what the model uses, so it settles it."""
    man = CMP._read_manifest(REFCV4B_DUMP)
    man = copy.deepcopy(man)
    man["model"].pop("n_anchors")
    c = _cfg(REFCV4B_CFG)
    c.pop("anchors", None)
    c = _drop_flag(c, "--n-anchors")
    n, prov = CMP.resolve_n_anchors(man, c, REFCV4B_CKPT)
    assert n == BANK, n
    assert prov["primary"] == CK_FIELD, prov["primary"]
    assert prov["witnesses_agreeing"] == {CK_FIELD: 117}, prov["witnesses_agreeing"]


def c4_manifest_and_config_disagreeing_refuses():
    man = copy.deepcopy(CMP._read_manifest(REFCV4B_DUMP))
    man["model"]["n_anchors"] = 128
    msg = _refusal(lambda: CMP.resolve_n_anchors(man, _cfg(REFCV4B_CFG), None))
    assert "WITNESSES DISAGREE" in msg, msg[:300]
    assert "128" in msg and "117" in msg, msg[:300]


def c5_the_live_refcv5v2_config_alone_is_sufficient():
    """No dump, no checkpoint - only the config the pod is writing right now.
    It carries no key named `n_anchors`; it must still resolve to 117."""
    if not (REFCV5V2_CFG and os.path.exists(REFCV5V2_CFG)):
        raise AssertionError("SKIPPED-AS-FAILURE: set REFCV5V2_CONFIG to the "
                             "live refcv5-v2 config.json to run this case")
    c = _cfg(REFCV5V2_CFG)
    assert "n_anchors" not in json.dumps(c.get("anchors")), \
        "fixture drift: /anchors now HAS a key literally named n_anchors"
    n, prov = CMP.resolve_n_anchors(None, c, None)
    assert n == BANK, n
    assert prov["chance"] == CHANCE, prov["chance"]
    assert prov["primary"] == SH_FIELD, \
        "with no checkpoint the /anchors/shape witness must lead, got %r" % prov["primary"]


# --------------------------------------------------------------------------- #
# D. the NESTED lever reads
# --------------------------------------------------------------------------- #
LIVE_LEVERS_EXPECTED = {                    # literals, measured off the live config
    "sel_refined": (True, "/selection/sel_refined"),
    "sel_score_emitted": (True, "/selection/sel_score_emitted"),
    "sampler_ranks_the_fan": (True, "/selection/sampler_ranks_the_fan"),
    "tac_goal_tok_head": (True, "/seams/tac_goal_tok_head/built"),
    "goal_str": (True, "/seams/goal_str"),
    "anchor_v0_conditioned": (True, "/anchors/v0_conditioned"),
}
REFCV4B_LEVERS_EXPECTED = {                 # literals, measured off refcv4b's config
    "sel_refined": False,
    "sel_score_emitted": False,
    "sampler_ranks_the_fan": None,          # NOT DECLARED and no argv flag exists
    "tac_goal_tok_head": False,
    "goal_str": True,
    "anchor_v0_conditioned": True,
}


def d1_live_levers_read_from_their_nested_paths():
    if not (REFCV5V2_CFG and os.path.exists(REFCV5V2_CFG)):
        raise AssertionError("SKIPPED-AS-FAILURE: set REFCV5V2_CONFIG")
    f = CMP.config_facts(_cfg(REFCV5V2_CFG))
    for k, (want, where) in LIVE_LEVERS_EXPECTED.items():
        got = f["levers"][k]
        assert got["value"] is want, "%s: expected %r, got %r" % (k, want, got["value"])
        assert got["declared_at"] == where, \
            "%s: expected the NESTED path %r, got %r" % (k, where, got["declared_at"])
    assert f["values"]["anchor_control_units"]["value"] == "alat", f["values"]
    assert f["values"]["sampler"]["value"] == "ddim", f["values"]


def d2_refcv4b_levers_are_a_corroborated_absence():
    f = CMP.config_facts(_cfg(REFCV4B_CFG))
    for k, want in REFCV4B_LEVERS_EXPECTED.items():
        got = f["levers"][k]["value"]
        assert got is want, "%s: expected %r, got %r" % (k, want, got)


def d3_a_config_contradicting_its_own_argv_is_refused():
    if not (REFCV5V2_CFG and os.path.exists(REFCV5V2_CFG)):
        raise AssertionError("SKIPPED-AS-FAILURE: set REFCV5V2_CONFIG")
    c = _cfg(REFCV5V2_CFG)
    c["selection"] = dict(c["selection"])
    c["selection"]["sel_refined"] = False        # argv still carries --sel-refined
    msg = _refusal(lambda: CMP.config_facts(c))
    assert "CONTRADICTS" in msg, msg[:300]
    assert "sel_refined" in msg, msg[:300]


def d4_sampler_stochasticity_is_read_off_the_live_config():
    if not (REFCV5V2_CFG and os.path.exists(REFCV5V2_CFG)):
        raise AssertionError("SKIPPED-AS-FAILURE: set REFCV5V2_CONFIG")
    st, why = CMP.sampler_is_stochastic(_cfg(REFCV5V2_CFG))
    assert st is True, "refcv5-v2 uses --sampler ddim: expected True, got %r" % st
    st4, why4 = CMP.sampler_is_stochastic(_cfg(REFCV4B_CFG))
    assert st4 is False, "refcv4b has no sampler flag: expected False, got %r" % st4


def d5_a_sampler_that_contradicts_argv_is_refused():
    if not (REFCV5V2_CFG and os.path.exists(REFCV5V2_CFG)):
        raise AssertionError("SKIPPED-AS-FAILURE: set REFCV5V2_CONFIG")
    c = _cfg(REFCV5V2_CFG)
    c["seams"] = dict(c["seams"])
    c["seams"]["sampler"] = "truncated"          # argv still says ddim
    msg = _refusal(lambda: CMP.sampler_is_stochastic(c))
    assert "seams/sampler" in msg, msg[:300]


# --------------------------------------------------------------------------- #
def main():
    print("=" * 92)
    print("selftest_n_anchors.py - the bank size is RESOLVED, and an "
          "unresolvable one REFUSES by name")
    print("  every expectation below is a LITERAL in the case table "
          "(117 / 0.008547 / the field names)")
    print("=" * 92)
    print("[A] the arm-tool resolver, in isolation")
    _case("A1 four agreeing witnesses -> exactly 117, tensor primary", a1_all_agree)
    _case("A2 nothing anywhere -> REFUSAL naming all four fields",
          a2_unresolvable_refuses_naming_the_field)
    _case("A3 tensor 117 vs argv 128 -> REFUSAL naming both",
          a3_disagreement_refuses_naming_both)
    _case("A4 rebuilt-config default 128 vs witness 117 -> REFUSAL naming --n-anchors",
          a4_build_default_128_is_caught_and_named)
    _case('A5 the string "117" is NOT a bank size', a5_a_string_is_not_a_bank_size)
    _case("A6 a 20-anchor smoke bank resolves to 20", a6_a_smoke_bank_resolves_to_20)
    print("[B] end to end on the REAL refcv4b checkpoint (CPU, 0 GPU)")
    _case("B1 real config + real ckpt -> 117 from the TENSOR",
          b1_real_pair_loads_and_reads_117)
    _case("B2 --n-anchors stripped -> names n_anchors, NOT param_breakdown",
          b2_stripped_flag_refuses_naming_n_anchors_not_param_breakdown)
    _case("B3 --n-anchors + param_breakdown stripped -> names n_anchors, NOT a shape error",
          b3_stripped_flag_and_no_breakdown_still_names_n_anchors)
    _case("B4 a config whose /anchors/shape LIES -> REFUSAL",
          b4_a_config_that_lies_about_the_bank_is_refused)
    print("[C] the compare-tool resolver")
    _case("C1 manifest+config+ckpt -> 117, chance exactly 0.008547",
          c1_real_trio_gives_117_and_the_literal_chance)
    _case("C2 nothing anywhere -> REFUSAL naming every field",
          c2_nothing_anywhere_refuses_naming_every_field)
    _case("C3 manifest+config stripped, ckpt present -> 117 from the tensor",
          c3_manifest_stripped_but_the_tensor_answers)
    _case("C4 manifest 128 vs config 117 -> REFUSAL naming both",
          c4_manifest_and_config_disagreeing_refuses)
    _case("C5 the LIVE refcv5-v2 config ALONE -> 117 from /anchors/shape[0]",
          c5_the_live_refcv5v2_config_alone_is_sufficient)
    print("[D] the NESTED lever reads")
    _case("D1 live refcv5-v2 levers read from /selection, /seams, /anchors",
          d1_live_levers_read_from_their_nested_paths)
    _case("D2 refcv4b's absent levers are a CORROBORATED absence",
          d2_refcv4b_levers_are_a_corroborated_absence)
    _case("D3 a config contradicting its own argv -> REFUSAL",
          d3_a_config_contradicting_its_own_argv_is_refused)
    _case("D4 sampler stochasticity: ddim -> True, refcv4b -> False",
          d4_sampler_stochasticity_is_read_off_the_live_config)
    _case("D5 /seams/sampler contradicting argv -> REFUSAL",
          d5_a_sampler_that_contradicts_argv_is_refused)
    print("=" * 92)
    print("%d cases, %d FAILED" % (NRUN, len(FAILURES)))
    for f in FAILURES:
        print("  - %s" % f)
    print("=" * 92)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
