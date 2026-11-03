"""refcv9 M-GATE -- the banded map head's OWN overfit record, bound to the gate (`E1-MAP-OWN-RECORD`), and the M1 / M2a / M2b argvs.

* the TABLE `E1_OWN_RECORDS` {open-item id: (flags, record schema, check, judge)}: under `refcv9-e1rig`, `--r9-map-banded` in the argv
  makes G-MAP-OVERFIT demand a `/2-banded` record bound to THAT argv's sha256, re-judged against the INHERITED literals; the item is
  closed ONLY when the gate reads such a record and it passes (never deleted); every other `--r9-map-*` flag keeps it open;
* C0 / C1 are untouched: no table row applies, their open-item reasons, required checks and judge route are the tip's;
* the box lever (B3) closes `E1-BOX-OWN-RECORD` by adding ONE row -- shown here with a synthetic record;
* `r9e1_M1 / M2a / M2b` differ from C0 in exactly the lever's flags (+ the run directory).
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"


def _argv(name):
    return json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


C0, C1, M1, M2A, M2B = (_argv(n) for n in ("C0", "C1", "M1", "M2a", "M2b"))
COMMIT = "ab" * 20
MO, BD = RIG["map_overfit"], RIG["map_overfit_banded"]


# ============================================================================================ the argvs ==
@pytest.mark.parametrize("name,w", [("M1", "1.0"), ("M2a", "0.511"), ("M2b", "1.018")])
def test_map_arm_argvs_differ_from_C0_in_exactly_the_lever_flags(name, w):
    arm = _argv(name)
    pa, pb = LG.flag_pairs(C0), LG.flag_pairs(arm)
    assert [f for f, _ in pa if f != "--r9-map-banded"] == [f for f, _ in pb if f != "--r9-map-banded"]     # same flags, same order
    da, db = dict(pa), dict(pb)
    diff = {f for f in set(da) | set(db) if da.get(f) != db.get(f)}
    assert diff == {"--r9-map-banded", "--map-hires-near-lift-m", "--w-map-hires", "--out"} - ({"--w-map-hires"} if w == "1.0" else set())
    assert db["--r9-map-banded"] == [] and db["--map-hires-near-lift-m"] == ["10"] and db["--w-map-hires"] == [w]
    assert db["--seed"] == ["0"] and db["--out"] == [f"/home/nvidia/r8ns/r9e1/runs/r9e1-{name}"]
    assert db["--map-hires-near-refine-blocks"] == da["--map-hires-near-refine-blocks"] == ["1"]
    raw = (RUNS / f"r9e1_{name}.argv.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    rec = json.loads(raw)
    assert rec["arm"] == f"r9e1_{name}" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0"
    assert rec["base_sha256"] == hashlib.sha256((RUNS / "r9e1_C0.argv.json").read_bytes()).hexdigest()


def test_map_arm_emitter_reproduces_the_committed_files_and_refuses_a_non_C0_base(tmp_path):
    import r9_e1_make_argv as MK
    for name in ("M1", "M2a", "M2b"):
        assert MK.map_arm_argv(C0, name) == _argv(name)
    with pytest.raises(SystemExit):
        MK.map_arm_argv(C0, "M3")
    with pytest.raises(SystemExit):
        MK.map_arm_argv(MK.map_arm_argv(C0, "M1"), "M2a")            # already banded: derive from C0 only
    assert MK.MAP_ARMS == {"M1": 1.0, "M2a": 0.511, "M2b": 1.018}


def test_the_trainers_own_pins_accept_the_map_arm_argvs():
    import r9_e1_make_argv as MK
    for name in ("M1", "M2a", "M2b"):
        MK.check_map_arm_pins(_argv(name))


def test_C0_and_C1_are_unchanged_by_the_binding():
    assert RIG["map_overfit"] == LG.PROFILES["refcv8"]["map_overfit"]                 # the INHERITED literals, untouched
    for a in (C0, C1):
        assert LG.own_record_row(RIG, a, "G-MAP-OVERFIT") is None and LG.own_record_row(RIG, a, "G-BOX-OVERFIT") is None
        assert LG.open_item_reasons(RIG, a) == [] and LG.open_item_reasons(RIG, a, closed=("E1-MAP-OWN-RECORD",)) == []
        assert LG.own_records_closed(RIG, a, {"map_overfit_record": "/nonexistent"}, COMMIT, LG.argv_sha256(a)) == {}
        assert LG.required_checks_of(RIG, a) == list(dict.fromkeys([*RIG["required_checks"], "G-MAP-OVERFIT"]))
    for name, p in LG.PROFILES.items():                                               # no other profile carries the banded literals
        assert ("map_overfit_banded" in p) == (name == "refcv9-e1rig")
        if name != "refcv9-e1rig":
            assert LG.open_item_reasons(p) == LG.open_item_reasons(p, closed=("E1-MAP-OWN-RECORD",))


# ============================================================================== a synthetic /2-banded record ==
def _cell(iou, n=5000, ns=20000):
    return {"gate": {"iou": iou, "n_pos": n, "n_scored": ns}, "gated": True}


def _cells(f):
    return {ch: {bn: _cell(f(ch, bn)) for bn in ("near", "mid", "far")} for ch in BD["channel_class"]}


def _record(argv, **over):
    terms = {f"{ch}_{bn}": 2.0 for ch in BD["channel_class"] for bn in ("near", "mid", "far")}
    thin = BD["must_fail_all"]["s8_zeros"]
    rec = {
        "schema": BD["schema"], "tool": BD["tool"], "launch_commit": COMMIT, "launch_argv_sha256": LG.argv_sha256(argv),
        "spec_sha256": MO["spec_sha256"], "frameset_md5": MO["frameset_md5"],
        "frames_sha12": [f for f, _ in MO["frames"]], "raw_frames": [r for _, r in MO["frames"]],
        "extent": dict(MO["extent"]), "optimiser": {"steps": 3000, "lr": 1e-3, "batch": 4, "seed": 0,
                                                   "lr_decay": {"kind": "cosine_to_zero", "start_step": 2700}},
        "near_lift_m": 10.0, "near_refine_blocks": 1, "branch_config": {"banded": True},
        "bands": [{"name": n, "gate_rows": r} for n, r in BD["gate_rows"].items()],
        "results": {
            "healthy": {"step0": {"loss": dict(terms)}, "final": {"cells": _cells(lambda c, b: 0.95),
                                                                    "loss": {k: 0.5 for k in terms}},
                        "loss_finite_every_step": True},
            "lane_w0": {"final": {"cells": _cells(lambda c, b: 0.0 if c == "lane" else 0.95)}},
            "s8_zeros": {"final": {"cells": _cells(lambda c, b: 0.0 if c in thin else 0.9)}}},
        "verdict": {"band": MO["band"], "G_MAP_OVERFIT_BANDED": "PASS", "controls_reproduced": True,
                    "time_guard": {"time_1ms_on_every_clip": True},
                    "controls": {"C1_constant_drv": {"reproduced": True,
                                                     "cells": _cells(lambda c, b: 0.25 if c == "drv" else 0.0)},
                                 "C2_gt_as_logits": {"reproduced": True, "cells": _cells(lambda c, b: 1.0)}}}}
    rec.update(over)
    return rec


def _write(tmp_path, rec, name="rec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def _judge(tmp_path, rec, argv=M1):
    return LG.judge_map_overfit_banded(RIG, _write(tmp_path, rec), COMMIT, argv_sha=LG.argv_sha256(argv), argv=argv)[0]


def _closes(tmp_path, rec, argv=M1, item="E1-MAP-OWN-RECORD", opt="map_overfit_record"):
    return LG.own_record_closes(RIG, item, argv, {opt: _write(tmp_path, rec)}, COMMIT, LG.argv_sha256(argv))


def test_GREEN_a_banded_record_bound_to_the_M1_argv_passes_and_closes_the_item(tmp_path):
    assert _judge(tmp_path, _record(M1)) == []
    ok, why = _closes(tmp_path, _record(M1))
    assert ok and why == []
    p = _write(tmp_path, _record(M1))
    assert LG.own_records_closed(RIG, M1, {"map_overfit_record": p}, COMMIT, LG.argv_sha256(M1)) == {
        "E1-MAP-OWN-RECORD": LG.sha256_file(p)}
    assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, M1)] == ["OPEN ITEM E1-MAP-OWN-RECORD"]     # open without the record
    assert LG.open_item_reasons(RIG, M1, closed=("E1-MAP-OWN-RECORD",)) == []
    assert any(it["id"] == "E1-MAP-OWN-RECORD" for it in RIG["open_items"])                              # never deleted
    for a in (M2A, M2B):                                                                                 # each arm needs ITS OWN argv's record
        assert _judge(tmp_path, _record(a), argv=a) == []
        assert any("bound to argv" in r for r in _judge(tmp_path, _record(M1), argv=a))


def test_RED_arms_each_fails_with_its_reason(tmp_path):
    r1 = LG.judge_map_overfit(RIG, _write(tmp_path, _record(M1)), COMMIT, argv_sha=LG.argv_sha256(M1))[0]
    assert any("schema" in r for r in r1)                                                    # M1 argv + a record that is not /2-banded
    legacy = _record(M1, schema=MO["schema"])
    assert any("schema" in r for r in _judge(tmp_path, legacy)) and not _closes(tmp_path, legacy)[0]   # M1 argv + a /1-schema record
    other = _record(C0)                                                                      # bound to ANOTHER argv
    assert any("bound to argv" in r for r in _judge(tmp_path, other)) and not _closes(tmp_path, other)[0]
    bad = _record(M1)
    bad["results"]["lane_w0"]["final"]["cells"] = _cells(lambda c, b: 0.95)                  # R1 did not FAIL lane
    assert any("lane_w0" in r for r in _judge(tmp_path, bad)) and not _closes(tmp_path, bad)[0]
    bad = _record(M1)
    bad["results"]["s8_zeros"]["final"]["cells"] = _cells(lambda c, b: 0.9 if c == "arrow" else 0.0)    # R2: one thin channel passes
    assert any("s8_zeros" in r for r in _judge(tmp_path, bad))
    bad = _record(M1)
    bad["results"].pop("s8_zeros")                                                           # an arm that did not run: no A19 excusal
    assert any("not run" in r for r in _judge(tmp_path, bad))
    bad = _record(M1)
    bad["results"]["healthy"]["final"]["cells"]["lane"]["mid"]["gate"]["iou"] = 0.4         # a gated cell under the inherited bar 0.5
    assert any("lane" in r and "0.5" in r for r in _judge(tmp_path, bad))
    bad = _record(M1)
    bad["results"]["healthy"]["final"]["cells"]["arrow"]["mid"]["gate"]["n_pos"] = 100       # 6.25 m^2 < the 10 m^2 inherited floor
    assert any("INCONCLUSIVE" in r for r in _judge(tmp_path, bad))
    bad = _record(M1)
    bad["results"]["healthy"]["final"]["cells"]["drv"]["near"]["gate"]["iou"] = 0.8          # drv carries the AREA bar 0.85
    assert any("drv" in r and "0.85" in r for r in _judge(tmp_path, bad))
    far = _record(M1)
    far["results"]["healthy"]["final"]["cells"]["lane"]["far"]["gate"]["iou"] = 0.0           # far is informative, not gated
    assert _judge(tmp_path, far) == []
    for k, v in (("spec_sha256", "0" * 64), ("near_lift_m", 20.0), ("raw_frames", [0] * 16), ("tool", "other.py")):
        assert _judge(tmp_path, _record(M1, **{k: v})), k
    assert _judge(tmp_path, _record(M1, branch_config={"banded": False}))
    nog = _record(M1)
    nog["verdict"]["time_guard"] = {"time_1ms_on_every_clip": False}
    assert any("1 ms" in r for r in _judge(tmp_path, nog))
    noc = _record(M1)
    noc["verdict"]["controls"]["C2_gt_as_logits"]["cells"] = _cells(lambda c, b: 0.7)
    assert any("control C2" in r for r in _judge(tmp_path, noc))
    assert any("no /2-banded" in r for r in LG.judge_map_overfit_banded(RIG, str(tmp_path / "absent.json"), COMMIT,
                                                                       argv_sha=LG.argv_sha256(M1), argv=M1)[0])


def test_every_other_r9_map_flag_keeps_the_item_open_even_with_a_passing_record(tmp_path):
    rec = _record(M1)
    for extra, val in (("--r9-map-future-lever", []), ("--w-map-fp", ["0.5"])):
        a = LG.set_flag(M1, extra, val)
        ok, why = LG.own_record_closes(RIG, "E1-MAP-OWN-RECORD", a, {"map_overfit_record": _write(tmp_path, _record(a))},
                                       COMMIT, LG.argv_sha256(a))
        assert not ok and any("not covered" in w for w in why), (extra, why)
    # the box lever's flags never close the MAP item, and the map record never closes the BOX item
    assert not LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", LG.set_flag(M1, "--r9-box-depth-aux", []),
                                    {"box_overfit_record": _write(tmp_path, rec)}, COMMIT, LG.argv_sha256(M1))[0]


def test_the_job_dispatches_the_banded_argv_to_the_banded_judge_and_C0_to_the_inherited_one(tmp_path):
    def ctx(argv, rec):
        return LG.Ctx(profile="refcv9-e1rig", tree=str(tmp_path), commit=COMMIT, argv=list(argv), out_dir=str(tmp_path), path_map=[],
                      tree_sha256="0" * 64, argv_sha256=LG.argv_sha256(argv),
                      options={"map_overfit_record": _write(tmp_path, rec), "map_overfit_closure": None})
    ev = LG._job_map_overfit(ctx(M1, _record(M1)))["G-MAP-OVERFIT"]
    assert ev["details"]["own_record"]["item"] == "E1-MAP-OWN-RECORD" and ev["details"]["schema"] == BD["schema"]
    assert not any("g_map_overfit_record/1" in r for r in ev["reasons"])              # only the closure (not given here) is missing
    ev = LG._job_map_overfit(ctx(C0, _record(C0)))["G-MAP-OVERFIT"]                   # C0: the INHERITED judge reads the record
    assert "own_record" not in ev["details"] and any("g_map_overfit_record/1" in r for r in ev["reasons"])


# ===================================================================== the box row: ONE row, no new mechanism ==
def test_the_box_lever_closes_its_item_by_adding_one_row_with_no_new_mechanism(tmp_path, monkeypatch):
    row = dict(LG.E1_OWN_RECORDS["E1-BOX-OWN-RECORD"])
    assert row["flags"] == LG.E1_BOX_OWN_FLAGS and row["judge"] == "judge_box_own_record"   # E1-REC2 filled the named constants (was: empty)
    row = dict(row, judge=None)                                                       # this test shows the GENERIC reader: a row with no judge of its own
    box = LG.set_flag(C0, "--r9-xlever-synthetic", [])               # E1-REC3: not `--r9-box-*` (an unclassified box flag opens E1-BOX-UNKNOWN-LEVER too)
    sha = LG.argv_sha256(box)
    ids = lambda: [x.split(":")[0] for x in LG.open_item_reasons(RIG, box)]
    assert ids() == ["OPEN ITEM E1-BOX-OWN-RECORD"]
    synth = {"schema": "synthetic/2-b3", "launch_argv_sha256": sha,
             "verdict": {"RESULT": "PASS", "regression_arms": {"memory_zeros": {"ran": True, "failed_as_required": True}}}}
    monkeypatch.setitem(LG.E1_OWN_RECORDS, "E1-BOX-OWN-RECORD",
                        dict(row, flags=("--r9-xlever-synthetic",), schema="synthetic/2-b3", must_fail_arms=("memory_zeros",)))
    opts = lambda rec: {"box_overfit_record": _write(tmp_path, rec)}
    assert LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", box, opts(synth), COMMIT, sha) == (True, [])
    assert LG.own_records_closed(RIG, box, opts(synth), COMMIT, sha).keys() == {"E1-BOX-OWN-RECORD"}
    assert LG.open_item_reasons(RIG, box, closed=("E1-BOX-OWN-RECORD",)) == []
    for bad in (dict(synth, schema="x/2"), dict(synth, launch_argv_sha256="0" * 64),
                dict(synth, verdict={"RESULT": "FAIL", "regression_arms": synth["verdict"]["regression_arms"]}),
                dict(synth, verdict={"RESULT": "PASS", "regression_arms": {"memory_zeros": {"ran": True, "failed_as_required": False}}}),
                dict(synth, verdict={"RESULT": "PASS", "regression_arms": {}})):
        assert not LG.own_record_closes(RIG, "E1-BOX-OWN-RECORD", box, opts(bad), COMMIT, sha)[0]
    # the check that demands the lever's own record is the row's own (G-BOX-OVERFIT), not G-MAP-OVERFIT
    assert LG.own_record_row(RIG, box, "G-BOX-OVERFIT")["item"] == "E1-BOX-OWN-RECORD" and LG.own_record_row(RIG, box, "G-MAP-OVERFIT") is None


def test_the_gate_literals_equal_the_harness_literals():
    sys.path.insert(0, str(ROOT / "scripts"))
    import map_hires_overfit as O
    assert BD["channel_class"] == O.BANDED_CHANNEL_CLASS and BD["schema"] == O.BANDED_SCHEMA
    assert tuple(BD["gated_bands"]) == O.BANDED_GATED_BANDS
    assert {k: tuple(v) for k, v in BD["must_fail_any"].items()} == {"lane_w0": O.BANDED_MUST_FAIL["lane_w0"]}
    assert {k: tuple(v) for k, v in BD["must_fail_all"].items()} == {"s8_zeros": O.BANDED_MUST_FAIL["s8_zeros"]}
    assert BD["tool"] == "map_hires_overfit.py --banded" and BD["fine_cell_m"] == 0.10
    from tanitad.models import map_banded as MB
    geom = MB.band_geometry(100.0, 30.0)
    assert BD["gate_rows"] == {g.name: O.banded_gate_rows(g, 20.0) for g in geom} and BD["cell_m"] == {g.name: g.cell for g in geom}
    assert float(BD["near_lift_m"]) == MB.BAND_START_M[1]
