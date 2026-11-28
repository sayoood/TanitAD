"""The ARM argument of `code/read_t2_dominance.py` (AGENT Q, 2026-11-07): another arm read against P10 in C0C's place. SYNTHETIC pulls only: no stage-2/3/4 rollout is opened. CPU only.

The world is `test_t2v8f_dominance_rscn.World` (6 scenes, the drivers' stage layouts, the REAL exclusions builder) plus a FIFTH stage that holds the arm `EGAIN` (T2 x {EGAIN_s0, EGAIN_s1} deciding, RR x the same beside, over
all 6 scenes, with its own STAGE5_RESULT.json, rollouts dir and EXCLUSIONS5.json). Stages 2, 3 and 4 refuse the scenes the rscn world refuses (1 and 2, 5, and 1, 5, 2); stage 5 refuses scene 3 ONLY, so the union is
{1, 2, 3, 5} and the read keeps scenes 0 and 4. Every RED arm asserts a NAMED problem, so deleting one guard turns exactly its arm RED (`code/run_mutations_dom_q.py` does that). The module name `RD` is swapped by
that runner for a mutated copy; helper methods of the rscn world that read `RD` are re-written here so the swap reaches them."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
import pytest  # noqa: E402

import read_t2_dominance as RD  # noqa: E402
import test_t2v8f_dominance as T  # noqa: E402
import test_t2v8f_dominance_g2 as G2  # noqa: E402
import test_t2v8f_dominance_rscn as R  # noqa: E402

ARM = "EGAIN"
EX5 = (3,)
IDX5 = (0, 1, 2, 3, 4, 5)


def ns(**kw):
    return types.SimpleNamespace(**kw)


class World5(R.World):
    """The rscn world plus stage 5 with the arm. `arm` = the arm's name in the stage-5 cells; `ex5` = the scenes stage 5 refuses (rule R-SCN, through the REAL builder)."""

    def __init__(self, tmp, monkeypatch, ex5=EX5, arm=ARM, ex=None):
        monkeypatch.setitem(R.STAGE_IDX, 5, IDX5)
        monkeypatch.setitem(R.PASS_CELLS, 5, {"deciding": [("T2", f"{arm}_s0"), ("T2", f"{arm}_s1")], "beside": [("RR", f"{arm}_s0"), ("RR", f"{arm}_s1")]})
        for i in ex5:
            monkeypatch.setitem(R.REASON, i, R.REASON.get(i, "ACTOR_REFUSED"))
            monkeypatch.setitem(R.NTRACKS, i, R.NTRACKS.get(i, 6))
        super().__init__(tmp, ex={**(ex or {}), 5: tuple(ex5)})
        self.arm = arm
        self.src[5] = [e for e in self.allsrc if e["idx"] in IDX5]
        self.build_stage(5)
        for i in IDX5:
            if i not in self.ex[5]:
                for s in (0, 1):
                    (self.cell(5, i, "T2", f"{arm}_s{s}") / "actor_attach.json").write_text(json.dumps({"verdict": "ACCEPTED"}))

    def rscn(self, arm=None, arm_stage=5):
        return RD.rscn_check(self.roots, self.results, R.SIDS, dict(self.exclusions), R.EXP, arm_name=arm or self.arm, arm_stage=arm_stage)

    def g3(self, exclusions="all", arm=None, arm_stage=5):
        return RD.g3_check(self.roots, self.results, R.SIDS, R.EXP, exclusions=dict(self.exclusions) if exclusions == "all" else exclusions, arm_name=arm or self.arm, arm_stage=arm_stage)


def world5(tmp_path, monkeypatch, name="w", **kw):
    d = tmp_path / name
    d.mkdir()
    return World5(d, monkeypatch, **kw)


def world4(tmp_path, name="w4", **kw):
    d = tmp_path / name
    d.mkdir()
    return R.World(d, **kw)


# ============================================================================================================================ the layouts
def test_stage_cells_of_the_arm_and_of_the_other_stages():
    assert RD.arm_cells("C0C") == RD.STAGE_CELLS[4]                                                          # the default arm IS stage 4's layout
    assert RD.stage_cells(4) == RD.STAGE_CELLS[4] and RD.stage_cells(2) == RD.STAGE_CELLS[2] and RD.stage_cells(3) == RD.STAGE_CELLS[3]
    assert RD.stage_cells(5, "EGAIN", 5) == {"deciding": frozenset({"T2|EGAIN_s0", "T2|EGAIN_s1"}), "beside": frozenset({"RR|EGAIN_s0", "RR|EGAIN_s1"})}
    assert RD.stage_cells(4, "EGAIN", 5) == RD.STAGE_CELLS[4]                                                # stage 4 stays the C0C stage when the arm is read from stage 5
    assert RD.stage_cells(2, "EGAIN", 5) == RD.STAGE_CELLS[2]
    assert RD.STAGE_LABEL[5] == "stage5" and RD.STAGE_LABEL == {2: "stage2", 3: "stage3", 4: "stage4"}
    assert RD.manifest_key(4) == "manifest4_sha256" and RD.manifest_key(5) == "manifest5_sha256"


# ============================================================================================================================ the arm's own stage, read: result, root, exclusions
def test_stage5_arm_is_read_with_its_own_result_root_and_exclusions(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    assert (w.cell(5, 1, "T2", "EGAIN_s0") / "DONE").exists() and (w.cell(5, 3, "T2", "EGAIN_s0") / "FAIL").exists()      # scene 1 is DONE in stage 5 (refused in 2 and 4), scene 3 is refused in stage 5
    assert w.exc(5)["stage"] == "stage5" and w.exc(5)["verdict"] == "ATTRIBUTED", w.exc(5)["problems"]
    gate, recs = w.g3()
    assert gate["PASS"] and recs and gate["pairing"] == {"s0": "PAIRED", "s1": "PAIRED"}, gate["why"]
    assert sorted(recs) == ["EGAIN_s0", "EGAIN_s1", "P10"], sorted(recs)                                      # the arm's cells are named by the arm
    assert gate["n_starts"] == {"P10": 4, "EGAIN_s0": 4, "EGAIN_s1": 4}, gate["n_starts"]                      # 11 starts - (scene 1: 1, scene 2: 2, scene 3: 3, scene 5: 1)
    for arm, rs_ in recs.items():
        assert sorted({r["scene"] for r in rs_}) == [0, 4], arm                                                # the union {1, 2, 3, 5} is gone from BOTH the arm and P10
    u = gate["r_scn"]
    assert sorted(u["per_stage"]) == ["2", "3", "4", "5"]
    assert {k: v["stages"] for k, v in u["union"].items()} == {"bbbbbbbb": {"2": "ACTOR_REFUSED", "4": "ACTOR_REFUSED"}, "cccccccc": {"4": "ACTOR_REFUSED"}, "dddddddd": {"5": "ACTOR_REFUSED"},
                                                               "ffffffff": {"3": "NO_DYNAMIC_LAYER", "4": "NO_DYNAMIC_LAYER"}}
    assert u["per_stage"]["5"]["excluded"] == {"dddddddd": {"scene": "dddddddd-3", "idx": 3, "reason": "ACTOR_REFUSED", "cells_failed": {"deciding": 2, "beside": 2}, "cells_total": 4, "starts_lost": 3}}
    assert u["per_stage"]["5"]["deciding_done"] == 10 == u["per_stage"]["5"]["deciding_required"] and u["per_stage"]["5"]["beside_required"] == 10
    assert u["starts"]["lost_launch"] == 6 and u["starts"]["lost_guard"] == 1 and u["starts"]["read_on"] == {"launch": 2, "guard": 2, "total": 4}, u["starts"]
    assert u["scenes"] == {"of": 6, "excluded": 4, "read_on": 2}


def test_a_scene_excluded_only_in_stage5_leaves_p10_too(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    for tc in ("T2", "RR"):
        assert (w.cell(2, 3, tc, "P10") / "DONE").exists(), tc                                                 # scene 3 is DONE for P10 in stage 2 ... and only stage 5 refused it
    for st in (2, 3, 4):
        assert 3 not in w.ex[st]
    gate, recs = w.g3()
    assert gate["PASS"], gate["why"]
    assert 3 not in {r["scene"] for r in recs["P10"]} and 3 not in {r["scene"] for r in recs["EGAIN_s0"]}, "scene 3 is still paired"
    L, S = RD.STATE["L"], RD.STATE["S"]
    with_3 = sorted(recs["P10"] + [{"scene": 3, "f0": 22}], key=lambda r: (r["scene"], r["f0"]))
    with pytest.raises(ValueError):
        S.pair_check(L.pair_keys(recs["EGAIN_s0"]), L.pair_keys(with_3))                                       # what an unremoved scene would do to the pairing
    assert RD.rscn_pairs_on_excluded({"P10": with_3}, {1, 2, 3, 5}) == ["RSCN_PAIRS_ON_EXCLUDED P10: records on excluded scenes [3]"]


def test_scenes_excluded_in_other_stages_still_leave_the_arm(tmp_path, monkeypatch):
    """Scene 1 (refused in stages 2 and 4) and scene 2 (stage 4 only) are DONE in stage 5; rule 4 takes them out of the ARM."""
    w = world5(tmp_path, monkeypatch)
    assert (w.cell(5, 1, "T2", "EGAIN_s1") / "DONE").exists() and (w.cell(5, 2, "T2", "EGAIN_s1") / "DONE").exists()
    gate, recs = w.g3()
    assert gate["PASS"] and all(not ({1, 2} & {r["scene"] for r in rs_}) for rs_ in recs.values())
    assert w.exclusions[4] and w.exc(4)["n_excluded"] == 3                                                     # stage 4 is still read for its exclusions when the arm is read from stage 5


def test_stage4_stays_the_c0c_stage_when_the_arm_is_read_from_stage5(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    hit = [p for p in w.rscn()["problems"]]
    assert hit == [], hit
    assert w.rscn()["per_stage"]["4"]["cells_per_scene"] == {"deciding": 2, "beside": 2} and w.rscn()["per_stage"]["2"]["cells_per_scene"] == {"deciding": 10, "beside": 4}
    as_default = w.rscn(arm="C0C", arm_stage=4)                                                                # the same world read as the default (arm at stage 4): stage 5 is not read
    assert "5" not in as_default["per_stage"] and as_default["problems"] == []


# ============================================================================================================================ RED arms of the arm's stage
def test_red_stage5_layout_and_conditions_are_the_arms(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    w.edit_result(5, lambda r: r["steps"]["rollouts"]["state"]["deciding"]["failed"].pop())                    # only part of the refused scene's stage-5 cells failed
    hits = [p for p in w.rscn()["problems"] if "RSCN_PARTIAL_SCENE stage 5 deciding" in p]
    assert hits, w.rscn()["problems"]
    gate, recs = w.g3()
    assert not gate["PASS"] and recs is None and any("RSCN_PARTIAL_SCENE stage 5 deciding" in x for x in gate["why"]), gate["why"]
    w2 = world5(tmp_path, monkeypatch, name="w2", arm="C0C")                                                   # the stage-5 pull holds cells named C0C_*, the read is asked for EGAIN
    hits = [p for p in w2.rscn(arm="EGAIN")["problems"] if "RSCN_INCLUDED_CELL_ABSENT stage 5 scene aaaaaaaa T2|EGAIN_s0" in p]
    assert hits, w2.rscn(arm="EGAIN")["problems"][:3]
    w3 = world5(tmp_path, monkeypatch, name="w3")
    done5 = w3.cell(5, 4, "RR", "EGAIN_s1") / "DONE"
    done5.write_text("rc=0 records=1/2 executor=base\n")                                                         # a beside cell of an included scene says 1 of 2, the file holds 2
    assert any("RSCN_INCLUDED_CELL_NOT_DONE stage 5 scene eeeeeeee RR|EGAIN_s1" in p for p in w3.rscn()["problems"]), w3.rscn()["problems"][:3]
    w4 = world5(tmp_path, monkeypatch, name="w4")
    w4.edit_exc(5, lambda e: e.update(stage="stage4"))                                                         # the stage-5 exclusions handed over are another stage's
    assert any("RSCN_OTHER_STAGE stage 5" in p for p in w4.rscn()["problems"]), w4.rscn()["problems"][:3]
    w5 = world5(tmp_path, monkeypatch, name="w5")
    w5.edit_result(5, lambda r: r.update(note="the stage-5 result changed after the exclusions were built"), rebind=False)
    assert any("RSCN_OTHER_RESULT stage 5: the exclusions were built from another stage result file" in p for p in w5.rscn()["problems"]), w5.rscn()["problems"][:3]


def test_red_flags_for_stage5_are_all_or_none(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    part = {s: p for s, p in w.exclusions.items() if s != 5}                                                  # the arm's exclusions missing
    gate, recs = w.g3(exclusions=part)
    assert not gate["PASS"] and recs is None and any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [5]" in x and "--arm-exclusions" in x for x in gate["why"]), gate["why"]
    only5 = {5: w.exclusions[5]}
    gate, recs = w.g3(exclusions=only5)
    assert not gate["PASS"] and any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [2, 3, 4]" in x for x in gate["why"]), gate["why"]
    w4 = world4(tmp_path)
    part4 = {s: p for s, p in w4.exclusions.items() if s != 4}                                                # the default arm: the previous message, with no mention of the arm flag
    g4, _ = RD.g3_check(w4.roots, w4.results, R.SIDS, R.EXP, exclusions=part4)
    assert any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [4]; rule R-SCN needs --exclusions2, --exclusions3 and --exclusions4 together" in x for x in g4["why"]), g4["why"]


# ============================================================================================================================ the name of the arm: refused, named
def test_wrong_arm_name_refuses_with_a_named_error():
    base = dict(root4="r4")
    cases = [
        (ns(arm="P10"), "ARM_NAME_RESERVED"),
        (ns(arm="zero"), "ARM_NAME_RESERVED"),
        (ns(arm="C0", arm_stage=5, arm_root="r5"), "ARM_NAME_RESERVED"),
        (ns(arm="EGAIN_s0", arm_stage=5, arm_root="r5"), "ARM_NAME_SEED_SUFFIX"),
        (ns(arm="LC_s1", arm_stage=5, arm_root="r5"), "ARM_NAME_SEED_SUFFIX"),
        (ns(arm="", arm_stage=5, arm_root="r5"), "ARM_NAME_FORM"),
        (ns(arm="E GAIN", arm_stage=5, arm_root="r5"), "ARM_NAME_FORM"),
        (ns(arm="a|b", arm_stage=5, arm_root="r5"), "ARM_NAME_FORM"),
        (ns(arm="../x", arm_stage=5, arm_root="r5"), "ARM_NAME_FORM"),
        (ns(arm=7, arm_stage=5, arm_root="r5"), "ARM_NAME_FORM"),
        (ns(arm="EGAIN", arm_stage=3), "ARM_STAGE_BELOW_4"),
        (ns(arm="C0C", arm_stage=2), "ARM_STAGE_BELOW_4"),
        (ns(arm="EGAIN"), "ARM_STAGE4_IS_C0C"),                                                                # stage 4 holds C0C: another arm lives in a later stage
        (ns(arm="EGAIN", arm_stage=4), "ARM_STAGE4_IS_C0C"),
        (ns(arm="C0C", arm_stage=4, arm_root="r5"), "ARM_STAGE4_EXTRA_INPUTS"),
        (ns(arm="C0C", arm_stage=4, arm_exclusions="e5.json"), "ARM_STAGE4_EXTRA_INPUTS"),
        (ns(arm="EGAIN", arm_stage=5), "ARM_ROOT_REQUIRED"),
        (ns(arm="EGAIN", arm_stage=7, arm_result="x.json"), "ARM_ROOT_REQUIRED"),
    ]
    for a, code in cases:
        with pytest.raises(SystemExit) as e:
            RD.arm_spec(a)
        assert str(e.value).startswith("REFUSED: ") and code in str(e.value), (a, str(e.value))
    for ok in (ns(), ns(arm=None, arm_stage=None), ns(arm="C0C", arm_stage=4), ns(arm="EGAIN", arm_stage=5, arm_root="r5"), ns(arm="LC", arm_stage=6, arm_root="r6"), ns(arm="E-GAIN_v2", arm_stage=5, arm_root="r5")):
        sp = RD.arm_spec(ok)
        assert sp["arm"] == (getattr(ok, "arm", None) or "C0C") and sp["stage"] == (getattr(ok, "arm_stage", None) or 4), sp
    assert RD.arm_spec(ns())["arm"] == "C0C" and RD.arm_spec(ns())["stage"] == 4 and RD.arm_spec(ns(arm_stage=5, arm_root="r"))["stage"] == 5


def wire(w, tmp_path, monkeypatch):
    """-> (go, seen). `go(extra_argv, flags=(2, 3, 4), out='out')` runs `main()` on the world with the heavy gates and rows replaced by recorders; `seen` records what reached them."""
    seen = {"skip": None, "roots": None, "arm_name": "unset", "arrays_kw": "unset", "rec_scenes": None, "g2a_overlays": "unset", "g2b_logs": "unset", "g2e_cells": None, "excluded": "unset", "g4_calls": 0}
    monkeypatch.setattr(RD, "EXPECT", R.EXP)

    def g4(*a, **k):
        seen["g4_calls"] += 1
        return {"PASS": True, "why": []}
    monkeypatch.setattr(RD, "g4_check", g4)
    monkeypatch.setattr(RD, "g1_check", lambda *a, **k: {"PASS": True, "why": []})

    def g2a(*a, **k):
        seen["g2a_overlays"] = a[3] if len(a) > 3 else k.get("overlays")
        seen["g2a_stages"] = sorted(a[1])
        return {"PASS": True, "why": []}

    def g2b(t2v8f_root=None, logs=None):
        seen["g2b_logs"] = logs
        return {"PASS": True, "why": []}
    real_g2e = RD.g2e_check

    def g2e(cells, excluded=None):
        seen["g2e_cells"] = [(label, arm) for label, _, arm in cells]
        seen["excluded"] = excluded
        return real_g2e(cells, excluded=excluded)

    def scenes_from(recs, srcs, roots, skip=()):
        seen["skip"], seen["roots"] = set(skip), sorted(roots)
        return {}

    def arrays_for_seed(rc, rp, scenes, s, **kw):
        seen["arm_name"], seen["arrays_kw"] = kw.get("arm_name", "C0C"), kw                                      # the keyword arguments exactly as run() passed them
        seen["rec_scenes"] = (sorted({r["scene"] for r in rc}), sorted({r["scene"] for r in rp}))
        return T.synth(T.dom_effects(), seed=s + 1)
    real_blocks = RD.read_blocks
    monkeypatch.setattr(RD, "g2a_check", g2a)
    monkeypatch.setattr(RD, "g2b_check", g2b)
    monkeypatch.setattr(RD, "g2e_check", g2e)
    monkeypatch.setattr(RD, "scenes_from", scenes_from)
    monkeypatch.setattr(RD, "arrays_for_seed", arrays_for_seed)
    monkeypatch.setattr(RD, "read_blocks", lambda arrays, n_boot=200: real_blocks(arrays, n_boot=200))
    s2, s3, g1 = tmp_path / "SCENE_SOURCES.json", tmp_path / "SCENE_SOURCES3.json", tmp_path / "g1.json"
    s2.write_text(json.dumps(w.src[2]))
    s3.write_text(json.dumps(w.src[3]))
    g1.write_text("{}")

    def go(extra=(), flags=(2, 3, 4), out="out"):
        argv = ["--root2", w.roots[2], "--root3", w.roots[3], "--root4", w.roots[4], "--sources2", str(s2), "--sources3", str(s3), "--g1", str(g1), "--out-dir", str(tmp_path / out)]
        for s in flags:
            argv += [f"--exclusions{s}", w.exclusions[s]]
        argv += list(extra)
        for k in ("skip", "roots", "rec_scenes", "g2e_cells"):
            seen[k] = None
        seen["arm_name"] = seen["arrays_kw"] = "unset"
        assert RD.main(argv) == 0
        return argv, tmp_path / out

    return go, seen


def arm5_args(w, tmp_path, arm=ARM, result=True):
    a = ["--arm", arm, "--arm-stage", "5", "--arm-root", w.roots[5], "--arm-exclusions", w.exclusions[5], "--arm-overlay", "OV5.tar::pkg/MANIFEST5.json", "--arm-gate-log", str(tmp_path / "pytest_stage5.log")]
    return a + (["--arm-result", w.results[5]] if result else [])


# ============================================================================================================================ main(), end to end
def test_main_reads_a_stage5_arm_names_it_and_records_the_argv(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    go, seen = wire(w, tmp_path, monkeypatch)
    argv, out = go(arm5_args(w, tmp_path))
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    md = open(out / "DOMINANCE_T2.md", encoding="utf-8").read()
    assert j["outcome"] == "DOMINANT-T2" and all(g["PASS"] for g in j["gates"].values()), {k: g.get("why") for k, g in j["gates"].items()}
    assert j["arm"] == "EGAIN" and j["arm_stage"] == 5 and j["argv"] == argv and j["floor"] == "P10 (stages 2 and 3)"
    assert "Arm read: **EGAIN** (stage 5, cells EGAIN_s0 and EGAIN_s1) against the floor P10 (stages 2 and 3). argv: `--root2 " in md and "--arm EGAIN --arm-stage 5 --arm-root " in md
    assert seen["arm_name"] == "EGAIN" and seen["arrays_kw"] == {"arm_name": "EGAIN"} and seen["rec_scenes"] == ([0, 4], [0, 4]) and seen["skip"] == {1, 2, 3, 5}                       # the arm's rows, the union in the scene loader
    assert seen["roots"] == [2, 3, 4, 5]                                                                       # the scene loader reads the arm's stage too
    assert seen["g2e_cells"] == [("stage2", "P10"), ("stage3", "P10"), ("stage5", "EGAIN_s0"), ("stage5", "EGAIN_s1")], seen["g2e_cells"]
    assert seen["g2a_stages"] == [2, 3, 4, 5] and seen["g2a_overlays"] == {**RD.DEFAULT_OVERLAYS, 5: "OV5.tar::pkg/MANIFEST5.json"}, seen["g2a_overlays"]
    assert seen["g2b_logs"] == {**RD.DEFAULT_GATE_LOGS, 5: str(tmp_path / "pytest_stage5.log")}, seen["g2b_logs"]
    assert set(seen["excluded"]) == {"bbbbbbbb", "cccccccc", "dddddddd", "ffffffff"}
    pe = j["gates"]["G2"]["parts"]["e"]["per_scene"]
    assert pe["dddddddd"]["EGAIN_s0@stage5"].startswith("EXCLUDED (R-SCN): ACTOR_REFUSED (stage 5)") and pe["aaaaaaaa"]["EGAIN_s1@stage5"] == "ACCEPTED" and pe["eeeeeeee"]["P10@stage3"] == "ACCEPTED"
    assert sorted(j["r_scn"]["per_stage"]) == ["2", "3", "4", "5"] and j["r_scn"]["scenes"] == {"of": 6, "excluded": 4, "read_on": 2} and j["r_scn"]["starts"]["lost_total"] == 7
    assert "excluded in the union of stages 2, 3, 4 and 5" in md and "Starts read on: 4 of 11" in md and "| 5 | dddddddd | 3 | ACTOR_REFUSED | 2 + 2 of 4 | 3 |" in md
    assert "Rule R-SCN applied: this read is on the 2 refusal-surviving scenes of the 62" in md
    argv2, out2 = go(arm5_args(w, tmp_path, result=False), out="out2")                                         # the result defaults to STAGE5_RESULT.json beside the rollouts dir
    j2 = json.load(open(out2 / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j2["outcome"] == "DOMINANT-T2" and {**j2, "argv": None} == {**j, "argv": None}


def test_arm_c0c_given_explicitly_equals_the_default(tmp_path, monkeypatch):
    w = world4(tmp_path)
    go, seen = wire(w, tmp_path, monkeypatch)
    argv0, out0 = go([], out="out0")
    assert seen["arm_name"] == "C0C" and seen["arrays_kw"] == {}                                                # the default arm: the previous pin's call to the row builder, no new keyword
    rec0 = dict(seen)
    argv1, out1 = go(["--arm", "C0C", "--arm-stage", "4"], out="out1")
    j0, j1 = (json.load(open(o / "DOMINANCE_T2.json", encoding="utf-8")) for o in (out0, out1))
    assert j0["outcome"] == "DOMINANT-T2" and j0["arm"] == j1["arm"] == "C0C" and j0["arm_stage"] == j1["arm_stage"] == 4
    assert j0["argv"] == argv0 and j1["argv"] == argv1 and argv0 != argv1                                      # the argv is recorded as given
    assert {**j0, "argv": None} == {**j1, "argv": None}, "the explicit default arm reads differently from the default"
    m0, m1 = (open(o / "DOMINANCE_T2.md", encoding="utf-8").read().splitlines() for o in (out0, out1))
    assert [x for x in m0 if not x.startswith("Arm read:")] == [x for x in m1 if not x.startswith("Arm read:")]
    assert seen["rec_scenes"] == rec0["rec_scenes"] == ([0, 3, 4], [0, 3, 4]) and seen["g2e_cells"] == rec0["g2e_cells"] == [("stage2", "P10"), ("stage3", "P10"), ("stage4", "C0C_s0"), ("stage4", "C0C_s1")]
    assert seen["g2a_overlays"] is None and seen["g2b_logs"] is None and seen["roots"] == [2, 3, 4] and seen["arrays_kw"] == {}   # the default arm: the previous pin's calls
    g_a = RD.g3_check(w.roots, w.results, R.SIDS, R.EXP, exclusions=dict(w.exclusions))
    g_b = RD.g3_check(w.roots, w.results, R.SIDS, R.EXP, exclusions=dict(w.exclusions), arm_name="C0C", arm_stage=4)
    assert g_a[0] == g_b[0] and g_a[1] == g_b[1] and sorted(g_a[1]) == ["C0C_s0", "C0C_s1", "P10"]


def test_main_refuses_a_wrong_arm_before_reading_and_a_mistyped_arm_as_a_gate(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    go, seen = wire(w, tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        go(["--arm", "P10"], out="out_bad")
    assert "ARM_NAME_RESERVED" in str(e.value) and not (tmp_path / "out_bad").exists() and seen["skip"] is None             # refused before anything was read or written
    assert seen["g4_calls"] == 0, "a gate ran before the arm was validated"
    with pytest.raises(SystemExit) as e:
        go(["--arm", "EGAIN"], out="out_bad2")                                                                 # stage 4 holds C0C
    assert "ARM_STAGE4_IS_C0C" in str(e.value) and not (tmp_path / "out_bad2").exists() and seen["g4_calls"] == 0
    argv, out = go(arm5_args(w, tmp_path, arm="EGAINN"), out="out_typo")                                      # a mistyped name: the pull holds EGAIN_s0 / EGAIN_s1 only
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j["outcome"] == "GATE-FAIL" and j["arm"] == "EGAINN" and seen["skip"] is None, j["outcome"]         # no row was read
    why = j["gates"]["ARM"]["why"]
    assert not j["gates"]["ARM"]["PASS"] and any("ARM_ABSENT_IN_PULL EGAINN_s0" in x and "EGAIN_s0" in x and "EGAIN_s1" in x for x in why) and any("ARM_ABSENT_IN_PULL EGAINN_s1" in x for x in why), why
    md = open(out / "DOMINANCE_T2.md", encoding="utf-8").read()
    assert "* ARM: FAIL ARM_ABSENT_IN_PULL EGAINN_s0" in md
    argv, out = go(arm5_args(w, tmp_path), out="out_ok")
    assert "ARM" not in json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))["gates"]                  # the gate appears only when it fails


def test_main_stage5_flags_are_all_or_none(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    go, seen = wire(w, tmp_path, monkeypatch)
    extra = [x for i, x in enumerate(arm5_args(w, tmp_path)) if x not in ("--arm-exclusions", w.exclusions[5])]
    argv, out = go(extra, out="out_noexc")                                                                     # --exclusions2/3/4 given, the arm's own exclusions not
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j["outcome"] == "GATE-FAIL" and any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [5]" in x for x in j["gates"]["G3"]["why"]), j["gates"]["G3"]["why"]
    assert seen["skip"] is None
    argv, out = go(arm5_args(w, tmp_path), flags=(), out="out_only5")                                          # only the arm's exclusions
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j["outcome"] == "GATE-FAIL" and any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [2, 3, 4]" in x for x in j["gates"]["G3"]["why"]), j["gates"]["G3"]["why"]


# ============================================================================================================================ G2 for the arm's stage
def test_g2_full_reads_the_arms_stage_and_names_a_missing_overlay_and_log(tmp_path, monkeypatch):
    w = world5(tmp_path, monkeypatch)
    a = ns(root2=w.roots[2], root3=w.roots[3], root4=w.roots[4], result1=RD.DEFAULT_RESULT1, manifest1=RD.DEFAULT_MANIFEST1, arm="EGAIN", arm_stage=5, arm_root=w.roots[5])
    results = {s: w.results[s] for s in (2, 3, 4, 5)}
    out = RD.g2_full(a, results)
    assert any(x.startswith("(a) G2A_OVERLAY_MANIFEST_MISSING stage 5") for x in out["why"]), out["why"][:6]    # the arm's stage joins (a): no --arm-overlay is a named failure
    assert any(x.startswith("(b) G2B_LOG_MISSING stage 5") for x in out["why"]), out["why"][:6]                # ... and (b)
    assert "5" in {str(k) for k in out["parts"]["c"]["stages"]} and set(out["parts"]["e"]["per_scene"]["aaaaaaaa"]) == {"P10@stage2", "EGAIN_s0@stage5", "EGAIN_s1@stage5"}
    ov5, log5 = tmp_path / "MANIFEST5.json", tmp_path / "pytest_stage5.log"
    ov5.write_text(json.dumps({"package": "arm_overlay", "stage1_manifest_sha256": "0" * 64, "files": []}))
    log5.write_text("5 passed in 1.0s\n")
    given = RD.g2_full(ns(**{**vars(a), "arm_overlay": str(ov5), "arm_gate_log": str(log5)}), results)                                       # an overlay and a green log GIVEN for the arm's stage are the ones read
    on_hand = hashlib.sha256(ov5.read_bytes()).hexdigest()[:12]
    assert any(x.startswith("(a) G2A_OVERLAY_DIFFERS stage 5") and f"on hand {on_hand}" in x for x in given["why"]), given["why"][:6]
    assert not any("G2A_OVERLAY_MANIFEST_MISSING stage 5" in x or "G2B_LOG_MISSING stage 5" in x for x in given["why"]) and given["parts"]["b"]["logs"][5]["n_passed"] == 5
    b = ns(root2=w.roots[2], root3=w.roots[3], root4=w.roots[4], result1=RD.DEFAULT_RESULT1, manifest1=RD.DEFAULT_MANIFEST1)                 # the default arm: no stage 5 anywhere in the answer
    out4 = RD.g2_full(b, {s: w.results[s] for s in (2, 3, 4)})
    assert not any("stage 5" in x for x in out4["why"]) and 5 not in out4["parts"]["c"]["stages"]


def test_g2a_reads_the_arms_overlay_manifest_key_of_its_stage(tmp_path):
    res, ov = G2.stage_results(tmp_path)
    sha1 = json.load(open(RD.DEFAULT_RESULT1, encoding="utf-8"))["manifest_sha256"]

    def with5(name, files, recorded=None):
        p5 = tmp_path / f"MANIFEST5_{name}.json"
        p5.write_text(json.dumps({"package": "arm_overlay", "stage1_manifest_sha256": sha1, "files": files}))
        rp = tmp_path / f"STAGE5_RESULT_{name}.json"
        rp.write_text(json.dumps({"manifest_sha256": sha1, "steps": {"preflight": {"verdict": "PASS", "info": {"pkg_files_ok": 471, "pkg_files": 471}}},
                                  "manifest5_sha256": recorded or hashlib.sha256(p5.read_bytes()).hexdigest()}))
        return {**res, 5: str(rp)}, {**ov, 5: str(p5)}
    r5, o5 = with5("ok", [{"path": "code/t2v8f_egain.py", "size": 1, "sha256": "0" * 64}])
    out = RD.g2a_check(RD.DEFAULT_RESULT1, r5, RD.DEFAULT_MANIFEST1, o5)
    assert out["PASS"], out["why"]
    assert out["stages"][5]["overlay_manifest_sha256"] == hashlib.sha256(open(o5[5], "rb").read()).hexdigest()
    r5, o5 = with5("other", [{"path": "code/t2v8f_egain.py", "size": 1, "sha256": "0" * 64}], recorded="e" * 64)         # the result recorded another overlay manifest
    assert any("G2A_OVERLAY_DIFFERS stage 5" in x for x in RD.g2a_check(RD.DEFAULT_RESULT1, r5, RD.DEFAULT_MANIFEST1, o5)["why"])
    r5, o5 = with5("replace", [{"path": "code/t2v8f_run.py", "size": 1, "sha256": "0" * 64}])                     # an overlay replacing a rollout-code file
    assert any("G2A_CODE_REPLACED stage 5" in x for x in RD.g2a_check(RD.DEFAULT_RESULT1, r5, RD.DEFAULT_MANIFEST1, o5)["why"])
    r5, o5 = with5("missing", [])
    o5.pop(5)
    assert any("G2A_OVERLAY_MANIFEST_MISSING stage 5" in x for x in RD.g2a_check(RD.DEFAULT_RESULT1, r5, RD.DEFAULT_MANIFEST1, o5)["why"])
