"""Pin 5 on the REAL stage 2/3/4 pulls: the identity of the default read with the read of record, and the seed-0-only read of C0C against the record's seed 0. The reads are made by `pin5/run_real.sh`
(the read of record's arguments, `audit/read_record_args.sh`); this module reads their outputs and decides. A missing output is a SKIP that names the script, never a pass."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.normpath(os.path.join(HERE, "..", "..")).replace("\\", "/")
RECORD = f"{T}/stage4/score/DOMINANCE_T2.json"
PIN4_READ = f"{T}/stage4/score_q_identity/DOMINANCE_T2.json"
IDENT = f"{T}/pin5/score_identity/DOMINANCE_T2.json"
SEED0 = f"{T}/pin5/score_seed0/DOMINANCE_T2.json"
PIN5_SHA = hashlib.sha256(open(f"{T}/pin5/read_t2_dominance_pin5.py", "rb").read()).hexdigest()
PIN4_SHA = "7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9"
spec = importlib.util.spec_from_file_location("compare_dominance_reads", f"{T}/code/compare_dominance_reads.py")
CMP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CMP)


def need(path):
    if not os.path.isfile(path):
        pytest.skip(f"{path} is absent: run pin5/run_real.sh first (a skip is not a pass)")
    return json.load(open(path, encoding="utf-8"))


def test_default_read_of_the_real_pulls_is_identical_to_the_read_of_record():
    new, rec = need(IDENT), need(RECORD)
    c = CMP.compare(RECORD, IDENT)
    assert c["identical"], {k: c[k] for k in ("differences", "md_differences", "card_sha256_now")}
    assert c["decision_equal"] and c["gates_equal"] and c["per_seed_rows_equal"] and c["r_scn_equal"] and c["seed_gap_equal"]
    assert c["n_rows_per_seed"] == {"s0": 19, "s1": 19} and new["outcome"] == rec["outcome"] == "NOT-DOMINANT"
    assert new["script_sha256"] == PIN5_SHA != rec["script_sha256"]
    assert sorted(set(new) - set(rec)) == ["argv", "arm", "arm_stage", "floor"] and new["arm"] == "C0C" and new["arm_stage"] == 4         # no key of pin 5 in a default read
    assert "arm_seeds" not in new and "n_seeds" not in new and sorted(new["by_seed"]) == ["s0", "s1"] and new["decision"] == rec["decision"]
    md_new, md_rec = (open(p.replace(".json", ".md"), encoding="utf-8").read() for p in (IDENT, RECORD))
    assert [x for x in md_new.splitlines() if not x.startswith("Arm read:") and "script sha256" not in x] == [x for x in md_rec.splitlines() if not x.startswith("Arm read:") and "script sha256" not in x]


def test_default_read_equals_the_fourth_pins_read_of_the_same_pulls():
    new, p4 = need(IDENT), need(PIN4_READ)
    assert p4["script_sha256"] == PIN4_SHA and new["script_sha256"] == PIN5_SHA
    skip = ("script_sha256", "argv", "card_sha256_now")
    assert {k: v for k, v in new.items() if k not in skip} == {k: v for k, v in p4.items() if k not in skip}
    assert [x for x in open(IDENT.replace(".json", ".md"), encoding="utf-8").read().splitlines() if not x.startswith("Arm read:") and "script sha256" not in x] == \
        [x for x in open(PIN4_READ.replace(".json", ".md"), encoding="utf-8").read().splitlines() if not x.startswith("Arm read:") and "script sha256" not in x]


def test_seed0_only_read_of_c0c_has_the_records_seed0_rows_and_the_word_seed0_only():
    s0, rec = need(SEED0), need(RECORD)
    assert s0["outcome"] == s0["decision"]["outcome"] == "SEED0-ONLY" and s0["script_sha256"] == PIN5_SHA
    assert s0["arm"] == "C0C" and s0["arm_stage"] == 4 and s0["arm_seeds"] == [0] and s0["n_seeds"] == 1 and s0["counted_dominant"] is False and s0["seed_statement"] == "seed 0 only, not a counted DOMINANT"
    assert "--arm-seeds" in s0["argv"] and s0["argv"][s0["argv"].index("--arm-seeds") + 1] == "0"
    assert sorted(s0["by_seed"]) == ["s0"] and len(s0["by_seed"]["s0"]["rows"]) == 19 and sorted(rec["by_seed"]) == ["s0", "s1"]
    assert s0["by_seed"]["s0"] == rec["by_seed"]["s0"], "the seed-0 rows (and read_a, classes, family table) differ from the read of record"
    assert s0["decision"]["per_seed"] == {"s0": rec["decision"]["per_seed"]["s0"]} and s0["decision"]["failing_terms"] == {"s0": rec["decision"]["failing_terms"]["s0"]}
    assert s0["decision"]["n_seeds"] == 1 and s0["decision"]["counted_dominant"] is False and s0["decision"]["statement"] == "seed 0 only, not a counted DOMINANT"
    assert s0["decision"]["seed0_block"] in ("DOMINANT", "NOT-DOMINANT", "UNDERPOWERED") and "DOMINANT-T2" not in json.dumps(s0)
    assert s0["three_family_form_beside"] == {"s0": rec["three_family_form_beside"]["s0"]}
    assert s0["seed_gap_beside"]["computable"] is False and "needs both seeds" in s0["seed_gap_beside"]["reason"]
    assert s0["r_scn"] == rec["r_scn"] and s0["card_sha_ok"] is True and s0["estimator"] == rec["estimator"] and s0["tier"] == rec["tier"]
    for g in ("G1", "G4"):
        assert s0["gates"][g] == rec["gates"][g]
    assert {k: v for k, v in s0["gates"]["G2"]["parts"].items() if k != "e"} == {k: v for k, v in rec["gates"]["G2"]["parts"].items() if k != "e"} and s0["gates"]["G2"]["PASS"]
    assert s0["gates"]["G3"]["PASS"] and s0["gates"]["G3"]["pairing"] == {"s0": "PAIRED"}
    assert s0["gates"]["G3"]["n_starts"] == {k: v for k, v in rec["gates"]["G3"]["n_starts"].items() if k != "C0C_s1"} and "C0C_s1" not in json.dumps(s0["gates"]["G3"])
    e0, er = s0["gates"]["G2"]["parts"]["e"], rec["gates"]["G2"]["parts"]["e"]
    assert {sc: {k: v for k, v in d.items() if not k.startswith("C0C_s1@")} for sc, d in er["per_scene"].items()} == e0["per_scene"]               # the record's table without its seed-1 column
    md = open(SEED0.replace(".json", ".md"), encoding="utf-8").read()
    assert "# DOMINANCE_T2: SEED0-ONLY" in md and "seed 0 only, not a counted DOMINANT" in md and "DOMINANT-T2" not in md and "## Per-family table, s1" not in md and "Not computable:" in md


def test_the_fourth_pin_is_untouched():
    assert hashlib.sha256(open(f"{T}/code/read_t2_dominance.py", "rb").read()).hexdigest() == PIN4_SHA
