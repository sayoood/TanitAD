"""refcv9 E1-REC2 -- the own-record rows for the B3 / B3' / B1 / M3 arms, their argvs, and the Thor-side bind script.

* `E1-BOX-OWN-RECORD`: `E1_BOX_OWN_FLAGS` = B3's four `--r9-box-*` levers (E1-REC3: the training-only aux heads are NOT in it, see
  test_launch_gate_e1rig_e1rec3.py); the item closes iff the record's schema is
  `/2-b3`, its STAMPED flag set EQUALS the argv's `--r9-box-*` set, RESULT is PASS, the must-fail arms ran and failed, and it is bound to THAT
  argv's sha256 (the M-GATE mechanism). Each of those has a RED arm. C0 / C1 / the M arms are untouched.
* `E1-MAP-OWN-RECORD`: `--r9-map-own-enc` (M3) joins the banded row; a banded record made WITHOUT the own encoder FAILS an M3 argv (the record
  must stamp `own_enc` and its BUILT branch config must hold it), and the converse.
* the harness side (`map_hires_overfit.py --banded --launch-argv`; E1-REC3: `g_box_overfit.py` no longer refuses `--r9-box-depth-aux`, it stamps it);
* `r9e1_B3` / `r9e1_B3p` (committed) and the M3 template (`--map-arm M3 --w-map-hires W`, no committed file: W is the M2 winner's);
* `r9e1_bind.sh`: any `r9e1_<ARM>.argv.json` is an arm, a KIND that does not match the argv's flags is refused, `box-b3` is the /2-b3 record.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import r9_e1_make_argv as MK  # noqa: E402
import test_launch_gate_e1rig_own_record as OR  # noqa: E402

G = pytest.importorskip("g_box_overfit")

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
BD = RIG["map_overfit_banded"]
COMMIT = "cd" * 20


def _argv(name):
    return json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


C0, C1, M1, B3, B3P = (_argv(n) for n in ("C0", "C1", "M1", "B3", "B3p"))
M3 = MK.map_arm_argv(C0, "M3", w=0.511)                      # the M3 template, with M2a's weight as a stand-in for "the M2 winner's"
BOX_FLAGS = ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms")
OK_ARMS = {"main": {"verdict": "PASS"}, "memory_zeros": {"verdict": "failed as required"},
           "presence_w0": {"verdict": "failed as required"}}


def _write(tmp_path, rec, name="rec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def _box_record(argv, **over):
    """The record's shape as `g_box_overfit.main` writes it: top-level `tool` / `RESULT` / `arms[*].verdict`; the schema + `b3.flags` come
    from the harness's OWN `b3_record_fields` (so a drift in the harness's stamp is read here), the argv hash from its own function."""
    rec = {"tool": "g_box_overfit.py", "binding": True, "commit": COMMIT, "RESULT": "PASS", "arms": copy.deepcopy(OK_ARMS),
           "launch_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8")).hexdigest()}
    rec.update(G.b3_record_fields(argv))
    rec.update(over)
    return rec


def _box_closes(tmp_path, rec, argv):
    return LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", argv, {"box_overfit_record": _write(tmp_path, rec)}, COMMIT, LG.argv_sha256(argv))


# ============================================================================================== the box row ==
def test_the_box_row_is_filled_and_pinned_to_the_harness():
    assert LG.E1_BOX_OWN_FLAGS == BOX_FLAGS + ("--r9-box-layers", "--r9-box-s8-horizon")   # E1-REC3: B3's four (the aux heads are LG.E1_BOX_AUX_FLAGS) + C5-S8's valued depth flag + C5-S8B's s8 horizon band
    row = LG.E1_OWN_RECORDS["E1-BOX-OWN-RECORD"]
    assert row["flags"] == LG.E1_BOX_OWN_FLAGS and row["judge"] == "judge_box_own_record" and row["schema"] == G.B3_SCHEMA
    assert tuple(G.B3_FLAGS) == BOX_FLAGS + ("--r9-box-layers", "--r9-box-s8-horizon") and tuple(row["must_fail_arms"]) == tuple(G.MUST_FAIL) == LG.E1_BOX_MUST_FAIL
    assert LG.E1_BOX_OWN_SCHEMA == G.B3_SCHEMA == "tanitad.g_box_overfit_record/2-b3"
    # the verdict text a failing must-fail arm carries is the harness's own (arm_verdict), not a second spelling
    fail = {k: {"pass": False} for k in ("1", "2", "3")}
    assert G.arm_verdict("memory_zeros", fail) == G.arm_verdict("presence_w0", fail) == LG.E1_BOX_ARM_FAILED
    assert G.arm_verdict("memory_zeros", {k: {"pass": True} for k in ("1", "2", "3")}).startswith("VOID")
    # every flag the item watches (`--r9-*` minus the rig and map namespaces) is one the row covers
    assert {f for f, _ in LG.flag_pairs(B3P) if f.startswith("--r9-box-")} <= set(row["flags"])


@pytest.mark.parametrize("name,argv,flags", [("B3", B3, BOX_FLAGS[:3]), ("B3p", B3P, BOX_FLAGS)])
def test_GREEN_a_b3_record_made_for_exactly_this_argv_closes_the_item(tmp_path, name, argv, flags):
    rec = _box_record(argv)
    assert rec["schema"] == G.B3_SCHEMA and rec["b3"]["flags"] == list(flags)
    assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, argv)] == ["OPEN ITEM E1-BOX-OWN-RECORD"]            # open without the record
    assert _box_closes(tmp_path, rec, argv) == (True, [])
    opts = {"box_overfit_record": _write(tmp_path, rec)}
    got = LG.own_records_closed(RIG, argv, opts, COMMIT, LG.argv_sha256(argv))
    assert list(got) == ["E1-BOX-OWN-RECORD"] and got["E1-BOX-OWN-RECORD"] == LG.sha256_file(opts["box_overfit_record"])
    assert LG.open_item_reasons(RIG, argv, closed=got) == []
    assert any(it["id"] == "E1-BOX-OWN-RECORD" for it in RIG["open_items"])                                       # never deleted
    assert LG.own_record_row(RIG, argv, "G-BOX-OVERFIT")["item"] == "E1-BOX-OWN-RECORD"


def test_RED_arms_each_box_record_defect_fails_with_its_reason(tmp_path):
    good = _box_record(B3)

    def why(rec, argv=B3):
        ok, reasons = _box_closes(tmp_path, rec, argv)
        assert not ok, "the defective record closed the item"
        return " | ".join(reasons)

    assert "schema" in why(dict(good, schema="tanitad.g_box_overfit_record/1"))
    assert "schema" in why({k: v for k, v in good.items() if k != "schema"})
    assert "bound to argv" in why(dict(good, launch_argv_sha256="0" * 64))                                         # unbound argv
    assert "bound to argv" in why(good, argv=LG.set_flag(B3, "--seed", ["7"]))                                     # another arm's argv
    assert "RESULT" in why(dict(good, RESULT="FAIL"))
    assert "RESULT" in why(dict(good, RESULT="VOID"))
    assert "RESULT" in why(dict(good, RESULT="INCOMPLETE"))
    assert "stamps no flag set" in why({k: v for k, v in good.items() if k != "b3"})
    assert "stamps no flag set" in why(dict(good, b3={"flags": []}))
    assert "tool" in why(dict(good, tool="map_hires_overfit.py"))
    for arm in ("memory_zeros", "presence_w0"):                                                                    # a must-fail arm that did not run / passed
        arms = copy.deepcopy(good["arms"])
        del arms[arm]
        assert arm in why(dict(good, arms=arms))
        arms = copy.deepcopy(good["arms"])
        arms[arm]["verdict"] = "VOID (a must-fail arm passed its criteria)"
        assert arm in why(dict(good, arms=arms))
    assert "no /2-b3 record" in " ".join(LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", B3, {"box_overfit_record": str(tmp_path / "nope.json")},
                                                               COMMIT, LG.argv_sha256(B3))[1]).replace("tanitad.g_box_overfit_record", "")


def test_RED_a_flag_set_that_is_not_the_argvs_does_not_close(tmp_path):
    """A record covers EXACTLY the `--r9-box-*` set it was built with: B3's record never closes B3', nor the reverse, nor a B1 argv."""
    rec3, rec3p = _box_record(B3), _box_record(B3P)
    # right sha, wrong stamp: the B3 stamp offered to B3' (the stamp is the harness's own, the sha is B3' -- forged binding)
    forged = dict(rec3, launch_argv_sha256=hashlib.sha256(json.dumps(B3P, separators=(",", ":")).encode()).hexdigest())
    ok, why = _box_closes(tmp_path, forged, B3P)
    assert not ok and "was made for" in " ".join(why) and "--r9-box-inloss-nms" in " ".join(why)
    forged = dict(rec3p, launch_argv_sha256=hashlib.sha256(json.dumps(B3, separators=(",", ":")).encode()).hexdigest())
    ok, why = _box_closes(tmp_path, forged, B3)
    assert not ok and "was made for" in " ".join(why)
    # a stamp with an EXTRA / MISSING flag, or a duplicate
    for stamp in (list(BOX_FLAGS[:2]), list(BOX_FLAGS), BOX_FLAGS[:3] + BOX_FLAGS[:1]):
        assert not _box_closes(tmp_path, dict(rec3, b3={"flags": list(stamp)}), B3)[0]
    # E1-REC3: an argv that adds the aux flag `--r9-box-depth-aux` to B3's levers is not covered by a B3 stamp WITHOUT the aux stamp ...
    b1 = LG.set_flag(LG.set_flag(B3, "--r9-box-depth-aux", []), "--w-box-depth", ["0.1"])
    forged = dict(rec3, launch_argv_sha256=hashlib.sha256(json.dumps(b1, separators=(",", ":")).encode()).hexdigest())
    ok, why = _box_closes(tmp_path, forged, b1)
    assert not ok and "--r9-box-depth-aux" in " ".join(why) and "aux_heads_not_certified" in " ".join(why)
    assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, b1)] == ["OPEN ITEM E1-BOX-OWN-RECORD"]            # B3's levers still open the item
    # ... and IS covered by the harness's own record for the combination (the b3 stamp compares B3's flags only; the aux stamp equals the aux set)
    assert _box_closes(tmp_path, _box_record(b1), b1) == (True, [])
    # any other `--r9-*` flag keeps the item open even with a passing B3 record
    other = LG.set_flag(B3, "--r9-dn-groups", ["3"])
    forged = dict(rec3, launch_argv_sha256=hashlib.sha256(json.dumps(other, separators=(",", ":")).encode()).hexdigest())
    ok, why = _box_closes(tmp_path, forged, other)
    assert not ok and "not covered" in " ".join(why)


def test_the_map_record_never_closes_the_box_item_and_the_box_record_never_the_map_item(tmp_path):
    assert not LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", B3, {"box_overfit_record": _write(tmp_path, OR._record(M1))}, COMMIT, LG.argv_sha256(B3))[0]
    both = LG.set_flag(B3, "--r9-map-banded", [])
    assert not LG.own_record_closes(RIG, "E1-MAP-OWN-RECORD", both, {"map_overfit_record": _write(tmp_path, _box_record(both))}, COMMIT,
                                    LG.argv_sha256(both))[0]


def test_C0_C1_and_the_map_arms_are_untouched_by_the_box_row(tmp_path):
    for a in (C0, C1, M1, M3):
        assert LG.own_record_row(RIG, a, "G-BOX-OVERFIT") is None
        assert LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", a, {"box_overfit_record": "/nonexistent"}, COMMIT, LG.argv_sha256(a))[0] is False
        assert not any("E1-BOX-OWN-RECORD" in x for x in LG.open_item_reasons(RIG, a))
    for a in (C0, C1):
        assert LG.open_item_reasons(RIG, a) == [] and LG.own_records_closed(RIG, a, {"box_overfit_record": "/nonexistent"}, COMMIT, LG.argv_sha256(a)) == {}
        assert LG.required_checks_of(RIG, a) == list(dict.fromkeys([*RIG["required_checks"], "G-MAP-OVERFIT"]))
    for name, p in LG.PROFILES.items():
        if name not in ("refcv9-e1rig", "refcv9-e2", "refcv9-full", "refcv9-full-d1"):   # the E1 rig and C12-FULL's three refcv9 profiles carry E1's lever-keyed rows by design
            assert LG.open_item_reasons(p) == LG.open_item_reasons(p, closed=("E1-BOX-OWN-RECORD", "E1-MAP-OWN-RECORD"))


def test_the_real_g_box_overfit_stamp_is_what_the_judge_reads():
    """The stamp path / schema the judge reads is the one `main()` writes (a source scan with a RED arm: drop the call, the scan fails)."""
    src = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    call = "rec.update(b3_record_fields(launch_argv))"
    assert src.count(call) == 1 and call not in src.replace(call, "pass")
    assert '"b3": {"flags": on' in src and LG.E1_OWN_RECORDS["E1-BOX-OWN-RECORD"]["stamp"] == ("b3", "flags")
    assert 'rec["RESULT"] = (' in src                                          # RESULT is a top-level key of the box record (not under `verdict`)


# ============================================================================================ the M3 map row ==
def _m3_record(argv, **over):
    """A /2-banded record as `map_hires_overfit.py --banded` writes it for an own-encoder build: the BUILT config and parameter breakdown are the
    real branch's (real-size, CPU), the stamp is the harness's."""
    torch = pytest.importorskip("torch")
    from tanitad.models import map_head_hires as H
    cfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, near_lift_x_m=10.0, near_refine_blocks=1, banded=True, own_enc=True)
    br = H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))
    rec = OR._record(argv, own_enc=True, own_enc_source="launch_argv", branch_config=cfg.as_dict(), branch_params=br.param_breakdown())
    rec.update(over)
    return rec


def test_the_banded_row_covers_both_map_flags_and_the_gate_literals_equal_the_harness_and_the_flag_module():
    assert LG.E1_OWN_RECORDS["E1-MAP-OWN-RECORD"]["flags"] == ("--r9-map-banded", "--r9-map-own-enc")
    import map_hires_overfit as O
    from tanitad.train import map_own_enc_flag as MOE
    assert BD["own_enc"] == {"flag": O.OWN_ENC_FLAG, "d_map_enc": MOE.D_MAP_ENC} and O.OWN_ENC_FLAG == MOE.FLAG == MK.OWN_ENC_FLAG
    assert O.BANDED_FLAG == "--r9-map-banded" == MK.BANDED_FLAG


def test_GREEN_an_M3_record_with_the_own_encoder_closes_the_item(tmp_path):
    rec = _m3_record(M3)
    assert rec["branch_config"]["own_enc"] is True and rec["branch_config"]["d_map_enc"] == 128 and rec["branch_params"]["map_encoder"] == 1_624_832
    assert OR._closes(tmp_path, rec, M3) == (True, [])
    assert OR._judge(tmp_path, rec, M3) == []
    got = LG.own_records_closed(RIG, M3, {"map_overfit_record": _write(tmp_path, rec)}, COMMIT, LG.argv_sha256(M3))
    assert list(got) == ["E1-MAP-OWN-RECORD"] and LG.open_item_reasons(RIG, M3, closed=got) == []
    assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, M3)] == ["OPEN ITEM E1-MAP-OWN-RECORD"]


def test_RED_arms_an_M3_argv_needs_a_record_made_WITH_the_own_encoder(tmp_path):
    def why(rec, argv=M3):
        reasons = OR._judge(tmp_path, rec, argv)
        assert reasons, "the defective record passed the banded judge"
        assert not OR._closes(tmp_path, rec, argv)[0]
        return " | ".join(reasons)

    good = _m3_record(M3)
    # the M3 argv offered a plain M1-style banded record (no own_enc stamp, no own encoder in the built config) -> FAIL
    assert "own_enc" in why(OR._record(M3))
    assert "own_enc" in why({k: v for k, v in good.items() if k != "own_enc"})                        # not stamped
    assert "own_enc" in why(dict(good, own_enc=False))
    bc = {k: v for k, v in good["branch_config"].items() if k not in ("own_enc", "d_map_enc", "lift_width")}
    assert "own_enc" in why(dict(good, branch_config=bc))                                              # stamped, but the BUILT config holds none
    assert "128" in why(dict(good, branch_config=dict(good["branch_config"], d_map_enc=64)))
    assert "own-encoder parameters" in why(dict(good, branch_params={k: v for k, v in good["branch_params"].items() if k != "map_encoder"}))
    assert "own-encoder parameters" in why(dict(good, branch_params=dict(good["branch_params"], map_encoder=0)))
    # the converse: an argv WITHOUT the own encoder offered a record that built it
    assert "does not carry" in why(_m3_record(M1), argv=M1)
    # own_enc without banded in the argv
    lone = [t for t in M3 if t != "--r9-map-banded"]
    assert "without --r9-map-banded" in why(_m3_record(lone), argv=lone)
    # the binding still holds: M3's record offered to M2a's argv
    assert "bound to argv" in why(_m3_record(M3), argv=LG.set_flag(M3, "--seed", ["3"]))
    # an M1 record is unchanged by the new clause (M1 builds no own encoder)
    assert OR._judge(tmp_path, OR._record(M1), M1) == []


# ========================================================================================== the harness side ==
def test_the_banded_harness_builds_the_own_encoder_from_the_launch_argv_it_is_bound_to(tmp_path):
    pytest.importorskip("torch")
    import map_hires_overfit as O

    def f(argv, name):
        p = tmp_path / name
        p.write_text(json.dumps({"argv": argv}), encoding="utf-8")
        return p

    p3, p1 = f(M3, "m3.json"), f(M1, "m1.json")
    f3 = O.launch_argv_facts(p3, LG.argv_sha256(M3))
    assert f3["sha256"] == LG.argv_sha256(M3) and f3["own_enc"] and f3["banded"]                       # the harness's hash IS the gate's
    f1 = O.launch_argv_facts(p1, LG.argv_sha256(M1))
    assert not f1["own_enc"] and f1["banded"]
    assert O.resolve_own_enc(False, f3) == (True, "launch_argv") and O.resolve_own_enc(False, f1) == (False, "off")
    assert O.resolve_own_enc(True, None) == (True, "--own-enc") and O.resolve_own_enc(False, None) == (False, "off")
    assert O.resolve_own_enc(True, f3) == (True, "launch_argv")
    with pytest.raises(SystemExit, match="built from one argv and bound to another"):
        O.launch_argv_facts(p3, LG.argv_sha256(M1))                                                    # file != the sha it is bound to
    with pytest.raises(SystemExit, match="does not carry"):
        O.resolve_own_enc(True, f1)                                                                    # --own-enc but the argv is M1's
    with pytest.raises(SystemExit, match="without --r9-map-banded"):
        O.resolve_own_enc(False, dict(f3, banded=False))
    # the record carries the stamp and the main() wiring is in place (RED: each wiring line, removed, breaks the scan)
    src = inspect.getsource(O.main_banded)
    for needle in ('facts = launch_argv_facts(a.launch_argv, a.launch_argv_sha256)', 'own_enc, own_enc_source = resolve_own_enc(bool(a.own_enc), facts)',
                   '**({"own_enc": True} if own_enc else {})', '"own_enc": own_enc, "own_enc_source": own_enc_source'):
        assert src.count(needle) == 1 and needle not in src.replace(needle, "pass"), needle
    assert src.index("resolve_own_enc(") < src.index("TT.TimmResNetTrunk(")                          # resolved BEFORE the (slow) trunk build


def test_the_banded_harness_options_are_banded_only():
    pytest.importorskip("torch")
    import map_hires_overfit as O
    with pytest.raises(SystemExit, match="--banded options"):
        O.main(["--spec", "x", "--v2-cache", "x", "--gt-root", "x", "--extrinsics", "x", "--out", "x", "--own-enc"])


# =========================================================================================== the argv emitter ==
def _diff(a, b):
    da, db = dict(LG.flag_pairs(a)), dict(LG.flag_pairs(b))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


@pytest.mark.parametrize("name,argv,lever", [("B3", B3, BOX_FLAGS[:3]), ("B3p", B3P, BOX_FLAGS)])
def test_box_arm_argvs_differ_from_C0_in_exactly_the_lever_flags(name, argv, lever):
    assert _diff(C0, argv) == set(lever) | {"--out"}
    assert [f for f, _ in LG.flag_pairs(C0)] == [f for f, _ in LG.flag_pairs(argv) if f not in lever]            # same flags, same order
    d = dict(LG.flag_pairs(argv))
    assert all(d[f] == [] for f in lever) and d["--seed"] == ["0"] and d["--out"] == [f"/home/nvidia/r8ns/r9e1/runs/r9e1-{name}"]
    assert d["--slot-presence-loss"] == ["focal"] and d["--slot-query-select"] == ["learned_ref"] and d["--agent-cls-weight"] == ["b1"]
    raw = (RUNS / f"r9e1_{name}.argv.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    rec = json.loads(raw)
    assert rec["arm"] == f"r9e1_{name}" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0" and rec["lever"] == {"flags": list(lever)}
    assert rec["base_sha256"] == hashlib.sha256((RUNS / "r9e1_C0.argv.json").read_bytes()).hexdigest()
    assert _diff(B3, B3P) == {"--r9-box-inloss-nms", "--out"}                                                    # B3' = B3 + ONE lever


def test_box_arm_emitter_reproduces_the_committed_files_and_refuses_bad_bases():
    for name in ("B3", "B3p"):
        assert MK.box_arm_argv(C0, name) == _argv(name)
        MK.check_box_arm_pins(_argv(name))                                         # the trainer's parser, the rig pin AND box_b3_flag.pin_args accept
    with pytest.raises(SystemExit):
        MK.box_arm_argv(C0, "B9")
    with pytest.raises(SystemExit):
        MK.box_arm_argv(B3, "B3p")                                                 # derive from C0 only
    with pytest.raises(SystemExit):
        MK.box_arm_argv([t for t in C0 if t != "--slot-query-select" and t != "learned_ref"], "B3")
    assert set(MK.BOX_ARMS) == {"B3", "B3p"}
    # the emitter's pin check is a real check: B3 with the focal presence loss removed is refused by box_b3_flag.pin_args
    bad = [t for t in B3 if t not in ("--slot-presence-loss", "focal")]
    with pytest.raises(SystemExit):
        MK.check_box_arm_pins(bad)


def test_M3_template_needs_the_M2_winners_weight_and_differs_from_C0_in_exactly_the_lever_flags(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="needs --w-map-hires"):
        MK.map_arm_argv(C0, "M3")                                                  # the weight is REQUIRED (no committed r9e1_M3.argv.json yet)
    with pytest.raises(SystemExit, match="must be > 0"):
        MK.map_arm_argv(C0, "M3", w=0.0)
    for arm in ("M1", "M2a", "M2b"):
        with pytest.raises(SystemExit, match="M3 only"):
            MK.map_arm_argv(C0, arm, w=0.5)                                        # the fixed-w arms refuse a weight
    assert MK.MAP_ARMS == {"M1": 1.0, "M2a": 0.511, "M2b": 1.018} and MK.OWN_ENC_ARMS == ("M3",)
    assert _diff(C0, M3) == {"--r9-map-banded", "--r9-map-own-enc", "--map-hires-near-lift-m", "--w-map-hires", "--out"}
    d = dict(LG.flag_pairs(M3))
    assert d["--r9-map-own-enc"] == [] and d["--w-map-hires"] == ["0.511"] and d["--map-hires-near-lift-m"] == ["10"]
    assert d["--out"] == ["/home/nvidia/r8ns/r9e1/runs/r9e1-M3"] and d["--seed"] == ["0"]
    toks = [f for f, _ in LG.flag_pairs(M3)]
    assert toks.index("--r9-map-own-enc") == toks.index("--r9-map-banded") + 1                                  # right after the banded flag
    assert [t for t in M3 if t != "--r9-map-own-enc"] == MK.map_arm_argv(C0, "M2a", out="/home/nvidia/r8ns/r9e1/runs/r9e1-M3")   # = M2a's argv + the lever
    MK.check_map_arm_pins(M3)                                                  # banded pin + map_own_enc_flag.pin_args accept it
    # the CLI: M3 needs --w-map-hires, --w-map-hires is M3's, and it writes the record with its lever
    c0f = RUNS / "r9e1_C0.argv.json"
    out = tmp_path / "r9e1_M3.argv.json"
    monkeypatch.setenv("MSYS_NO_PATHCONV", "1")
    with pytest.raises(SystemExit):
        MK.main(["--base", str(c0f), "--map-arm", "M3", "--write", str(out)])
    with pytest.raises(SystemExit):
        MK.main(["--base", str(c0f), "--map-arm", "M2a", "--w-map-hires", "0.5", "--write", str(out)])
    assert MK.main(["--base", str(c0f), "--map-arm", "M3", "--w-map-hires", "0.511", "--write", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["argv"] == M3 and rec["arm"] == "r9e1_M3" and rec["lever"]["own_enc_flag"] == "--r9-map-own-enc" and rec["lever"]["w_map_hires"] == 0.511
    assert not (RUNS / "r9e1_M3.argv.json").exists()                               # no committed M3 file until the winner is read
    # a pin the own-encoder flag owns: it needs --r9-map-banded, so an argv with it alone is refused by the emitter's pin check
    lone = [t for t in M3 if t != "--r9-map-banded"]
    with pytest.raises(SystemExit):
        MK.check_map_arm_pins(lone)


def test_the_M_arm_files_and_the_C0_C1_files_are_unchanged_by_the_emitter():
    for name in ("M1", "M2a", "M2b"):
        assert MK.map_arm_argv(C0, name) == _argv(name)
    assert MK.build is not None and MK.map_arm_argv(C0, "M1") == M1


# ============================================================================================= the bind script ==
BASH = shutil.which("bash")
BIND = ROOT / "ops" / "r9e1_bind.sh"
CODE_TREE = ROOT.parent
HASH = "4e36c721d0e1c8befe3371b73f92e4a455ff4732"


def _bash_ok():
    if not BASH:
        return False
    try:
        r = subprocess.run([BASH, "-c", "echo $BASH_VERSION"], capture_output=True, text=True, encoding="utf-8", timeout=60)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


needs_bash = pytest.mark.skipif(not _bash_ok(), reason="no working bash (the bind script is a Thor-side bash tool)")


def _bind(tmp_path, *args, code=CODE_TREE, inputs=None):
    env = dict(os.environ, PYTHON=sys.executable, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [BASH, BIND.as_posix(), "--dry-run", *args, "--commit", HASH, "--root", (tmp_path / "root").as_posix(), "--code", Path(code).as_posix()]
    if inputs:
        cmd += ["--inputs", Path(inputs).as_posix()]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    return r.stdout.replace("\r", ""), tmp_path / "root"


@needs_bash
@pytest.mark.parametrize("kind,arm", [("box", "M1"), ("box", "M2a"), ("box", "M2b"), ("box", "C0"), ("map", "B3"), ("map", "B3p"),
                                      ("box-b3", "B3"), ("box-b3", "B3p"), ("map", "C1")])
def test_dry_run_accepts_every_arm_whose_argv_exists_and_matches_the_kind(tmp_path, kind, arm):
    out, root = _bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    sha = LG.argv_sha256(_argv(arm))
    assert f"sha256 {sha}" in out and f"argv sha {sha[:16]}" in out
    assert (root / f"ovf_{arm}_{HASH[:12]}" / f"{kind}.args").is_file()


@needs_bash
@pytest.mark.parametrize("kind,arm,why", [
    ("map", "M1", "use map-banded"), ("map", "M2b", "use map-banded"), ("box", "B3", "use box-b3"), ("box", "B3p", "use box-b3"),
    ("map-banded", "C0", "WITHOUT --r9-map-banded"), ("map-banded", "B3", "WITHOUT --r9-map-banded"),
    ("box-b3", "C0", "WITHOUT any --r9-box-"), ("box-b3", "M1", "WITHOUT any --r9-box-"),
    ("box", "nosuch", "no rig argv"), ("box-b3", "B3p/../C0", "letters, digits"), ("mapp", "C0", "kind must be")])
def test_dry_run_refuses_a_kind_that_does_not_match_the_argv_before_touching_anything(tmp_path, kind, arm, why):
    out, root = _bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and why in out and "DRYOK" not in out, out
    assert not root.exists() or not list(root.iterdir()), "a refused call must not create the arm's directory"


@needs_bash
def test_dry_run_of_the_banded_kind_needs_the_registered_A18_spec_but_reads_the_kind_first(tmp_path):
    out, _ = _bind(tmp_path, "map-banded", "M1", inputs=tmp_path)                          # no gmo_spec_A18.json there: refused, but ABOUT THE SPEC
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "gmo_spec_A18.json" in out and "WITHOUT" not in out
    spec = next((p for p in CODE_TREE.rglob("gmo_spec_A18.json")), None) if (CODE_TREE / "TanitAD Research Lab").exists() else None
    if spec is not None:
        out, _ = _bind(tmp_path / "x", "map-banded", "M1", inputs=spec.parent)
        assert "ZZR9E1BIND-DRYOKZZ" in out


def _snippets():
    s = BIND.read_text(encoding="utf-8")
    return re.findall(r'(?:"\$PY" -c|pyval) \'(.*?)\' "\$CODE"', s, re.S)


def test_every_python_snippet_in_the_bind_script_compiles():
    """MEASURED on the tip: the map-banded assertion snippet held a LITERAL newline inside a string (a SyntaxError the first real run would hit AFTER
    its GPU hours). Every `\"$PY\" -c '...'` program is compiled here, and none may contain an unterminated string."""
    sn = _snippets()
    assert len(sn) >= 3
    for i, code in enumerate(sn):
        compile(code, f"r9e1_bind.sh snippet {i}", "exec")
    assert 'print("\n"' not in BIND.read_text(encoding="utf-8")


def test_the_box_b3_and_map_banded_assertion_snippets_run_and_refuse_a_missing_or_foreign_record(tmp_path):
    sn = {("box-b3" if "box-b3" in c else "map-banded" if "map-banded" in c else "other"): c for c in _snippets()}
    assert {"box-b3", "map-banded"} <= set(sn)
    argv_file = RUNS / "r9e1_B3.argv.json"
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps(_box_record(C0)), encoding="utf-8")                      # a record bound to C0's argv, offered for B3
    for kind, argvf in (("box-b3", argv_file), ("map-banded", RUNS / "r9e1_M1.argv.json")):
        for rec in (tmp_path / "missing.json", foreign):
            r = subprocess.run([sys.executable, "-I", "-c", sn[kind], str(CODE_TREE), str(rec), str(argvf), HASH], capture_output=True, text=True, encoding="utf-8",
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=300)
            assert r.returncode == 0 and "REFUSED:" in r.stdout and "ASSERT-OK" not in r.stdout, (kind, rec.name, r.stdout, r.stderr[-400:])


@needs_bash
def test_the_box_kind_renders_exactly_the_direct_template_for_any_arm(tmp_path):
    """The relaxed whitelist must not change a byte of the `box` args: render the template directly with the script's own substitutions."""
    for arm in ("M1", "M2a", "M2b"):
        out, root = _bind(tmp_path / arm, "box", arm)
        assert "DRYOK" in out
        got = (root / f"ovf_{arm}_{HASH[:12]}" / "box.args").read_text(encoding="utf-8")
        bd = (root / f"ovf_{arm}_{HASH[:12]}").as_posix()
        tpl = (ROOT / "scripts" / "r9e1_bind_box.args.tpl").read_text(encoding="utf-8")
        want = [ln for ln in (tpl.replace("\r", "").split("\n")) if ln.strip() and not ln.startswith("#") and not ln.startswith("DATA-ROOT")]
        want = [ln.replace("{ARGV_SHA}", LG.argv_sha256(_argv(arm))).replace("{ARGV}", (RUNS / f"r9e1_{arm}.argv.json").as_posix())
                .replace("{OUT}", bd + "/run_box").replace("{COMMIT}", HASH) for ln in want]
        assert [ln for ln in got.replace("\r", "").split("\n") if ln] == want


def test_the_box_b3_template_differs_from_the_box_template_only_in_arms_and_candidate_text():
    a = (ROOT / "scripts" / "r9e1_bind_box.args.tpl").read_text(encoding="utf-8").replace("\r", "")
    b = (ROOT / "scripts" / "r9e1_bind_box-b3.args.tpl").read_text(encoding="utf-8").replace("\r", "")
    ta = [ln for ln in a.split("\n") if ln and not ln.startswith("#")]
    tb = [ln for ln in b.split("\n") if ln and not ln.startswith("#")]
    assert len(ta) == len(tb)
    diff = [(x, y) for x, y in zip(ta, tb) if x != y]
    assert [x for x, _ in diff][1:] == ["main"] and [y for _, y in diff][1:] == ["main,memory_zeros,presence_w0"]
    assert "box-b3" in diff[0][1] and "{ARGV_SHA}" in diff[0][1] and "{COMMIT}" in diff[0][1]
    assert tb[tb.index("--arms") + 1] == "main,memory_zeros,presence_w0"                          # the /2-b3 record needs RESULT PASS => all three arms
    m = (ROOT / "scripts" / "r9e1_bind_map-banded.args.tpl").read_text(encoding="utf-8")
    assert "--launch-argv\n{ARGV}\n" in m.replace("\r", "") and "--launch-argv-sha256\n{ARGV_SHA}\n" in m.replace("\r", "")
