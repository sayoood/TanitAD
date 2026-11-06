"""Rule R-SCN in `code/read_t2_dominance.py` (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb), AGENT P, 2026-11-06. SYNTHETIC pulls only: no stage-2/3/4 rollout is opened. CPU only.

The world: 6 scenes (4 launch, 2 guard), the stage layouts of the DRIVERS (stages 2 and 3: 10 deciding + 4 beside cells per scene; stage 4: 2 deciding + 2 beside), written the way the drivers write them
(`DONE` / `FAIL` marker text `rc=.. records=a/N`, `records.jsonl`, `actor_attach.json`), the stage results with the driver's own keys, and the EXCLUSIONS*.json produced by the REAL builder
(`scene_exclusions.build`) from synthetic cell logs. Scene 1 is refused in stages 2 and 4, scene 5 in stages 3 and 4, scene 2 in stage 4 ONLY.
Every RED arm asserts the NAMED problem, so deleting one guard turns exactly its arm RED (`code/run_mutations_dom_p.py` does that). The module name `RD` is swapped by that runner for a mutated copy."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
import pytest  # noqa: E402

import read_t2_dominance as RD  # noqa: E402
import scene_exclusions as X  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402

SIDS = {0: "aaaaaaaa-0", 1: "bbbbbbbb-1", 2: "cccccccc-2", 3: "dddddddd-3", 4: "eeeeeeee-4", 5: "ffffffff-5"}
S8 = {i: v[:8] for i, v in SIDS.items()}
EXP = {"scenes_launch": 4, "scenes_guard": 2, "starts_launch": 8, "starts_guard": 3}
STARTS = {0: (20, 30), 1: (25,), 2: (40, 50), 3: (22, 33, 44), 4: (21, 31), 5: (26,)}
STAGE_IDX = {2: (0, 1, 2, 3), 3: (4, 5), 4: (0, 1, 2, 3, 4, 5)}
PASS_CELLS = {2: D.PASSES, 3: D.PASSES, 4: D4.PASSES4}
REASON = {1: "ACTOR_REFUSED", 2: "ACTOR_REFUSED", 5: "NO_DYNAMIC_LAYER"}
NTRACKS = {1: 5, 2: 7}
EX = {2: (1,), 3: (5,), 4: (1, 5, 2)}                                   # scene 2 is refused in stage 4 only
KEYERR = ["Traceback (most recent call last):", '  File "actor_map.py", line 77, in layer_track_ranges',
          """KeyError: "no timestamps_us_ranges for layer 'dynamic_rigids'; probed: []" """.strip()]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def refusal(n):
    return X.REFUSED + json.dumps({"n_tracks_json": n, "n_layer_tracks": 3, "n_mapped": 3, "falsifier_pass_frames": 0, "verdict": "REFUSED"})


class World:
    """A synthetic set of three stage pulls with results, cell logs, scene sources and the exclusions the REAL builder derives from them."""

    def __init__(self, tmp, ex=None, stray_tb_stage=None):
        tmp.mkdir(parents=True, exist_ok=True)
        self.tmp, self.ex = tmp, {**EX, **(ex or {})}
        self.roots, self.results, self.exclusions, self.pull, self.logs, self.src = {}, {}, {}, {}, {}, {}
        allsrc = []
        for i, s in SIDS.items():
            tr = tmp / f"tracks_{S8[i]}.json"
            tr.write_text(json.dumps({"c0": {"tracks_data": {"tracks_id": [str(k) for k in range(NTRACKS.get(i, 9))]}}}))
            allsrc.append({"idx": i, "scene": s, "rel": "extracted/sequence_tracks.json", "source": str(tr), "sha256": sha(tr)})
        self.allsrc = allsrc
        for st in (2, 3, 4):
            self.src[st] = [e for e in allsrc if e["idx"] in STAGE_IDX[st]]
            self.build_stage(st, stray_tb_stage == st)

    def cell(self, st, i, tc, arm):
        return self.pull[st] / "rollouts" / S8[i] / "objects" / tc / arm

    def build_stage(self, st, stray_tb=False):
        pull = self.tmp / f"p{st}"
        self.pull[st], self.roots[st] = pull, str(pull / "rollouts")
        failed, done, lines = {"deciding": [], "beside": []}, {"deciding": 0, "beside": 0}, {"deciding": ["an earlier run", "Traceback (most recent call last):", "OSError: old"], "beside": ["an earlier run", "x", "y"]}
        for p, cells in PASS_CELLS[st].items():
            for i in STAGE_IDX[st]:
                for tc, arm in cells:
                    d = self.cell(st, i, tc, arm)
                    d.mkdir(parents=True)
                    n = len(STARTS[i])
                    if i in self.ex[st]:
                        (d / "FAIL").write_text(f"rc=1 records=0/{n} executor=base\n")
                        failed[p].append(f"{S8[i]}|{tc}|{arm}")
                        lines[p] += [refusal(NTRACKS[i])] if REASON[i] == "ACTOR_REFUSED" else KEYERR
                    else:
                        (d / "records.jsonl").write_text("".join(json.dumps({"scene": i, "f0": f}) + "\n" for f in STARTS[i]))
                        (d / "DONE").write_text(f"rc=0 records={n}/{n} executor=base\n")
                        done[p] += 1
                        lines[p].append("12:00:00.000 INFO: ok")
                        if tc == "T2" and arm in ("P10", "C0C_s0", "C0C_s1"):
                            (d / "actor_attach.json").write_text(json.dumps({"verdict": "ACCEPTED"}))
        if stray_tb:
            lines["deciding"] += ["Traceback (most recent call last):", "ValueError: stray"]
        per_scene = sum(len(c) for c in PASS_CELLS[st].values())
        skipped = 2 if st == 2 else 0                                                   # some DONE cells were skipped as DONE from an earlier run: they still count
        res = {"stage": f"stage{st}", "manifest_sha256": "m" * 64, "n_scenes_manifest": len(STAGE_IDX[st]),
               "gates": {"preflight": "PASS", "gate": "PASS", "rollouts": "INCOMPLETE" if self.ex[st] else "PASS"}, "overall": "INCOMPLETE" if self.ex[st] else "COMPLETE",
               "steps": {"rollouts": {"verdict": "INCOMPLETE" if self.ex[st] else "PASS", "n_cells_planned": per_scene * len(STAGE_IDX[st]), "passes_skipped_disk": [], "scenes_skipped": {},
                                      "state": {p: {"done": done[p] - (skipped if p == "deciding" else 0), "skipped_done": skipped if p == "deciding" else 0, "failed": failed[p]} for p in failed}}}}
        self.results[st] = str(pull / f"STAGE{st}_RESULT.json")
        self.put_result(st, res)
        logs = {}
        for p in lines:
            f = pull / f"rollouts_{p}.log"
            f.write_bytes(("\n".join(lines[p]) + "\n").encode())
            if failed[p]:
                logs[p] = (str(f), 4)
        self.logs[st] = logs
        self.rebuild_exclusions(st)

    def put_result(self, st, res):
        open(self.results[st], "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))

    def result(self, st):
        return json.load(open(self.results[st], encoding="utf-8"))

    def edit_result(self, st, f, rebind=True):
        r = self.result(st)
        f(r)
        self.put_result(st, r)
        if rebind:
            self.rebind(st)

    def rebuild_exclusions(self, st):
        exc = X.build(self.result(st), str(self.pull[st]), self.logs[st], self.src[st])
        exc["inputs"] = {"result_sha256": sha(self.results[st]), "sources_sha256": "0" * 64}
        self.exclusions[st] = str(self.pull[st] / f"EXCLUSIONS{st}.json")
        open(self.exclusions[st], "w", encoding="utf-8", newline="\n").write(json.dumps(exc, indent=1))
        return exc

    def exc(self, st):
        return json.load(open(self.exclusions[st], encoding="utf-8"))

    def edit_exc(self, st, f):
        e = self.exc(st)
        f(e)
        open(self.exclusions[st], "w", encoding="utf-8", newline="\n").write(json.dumps(e, indent=1))

    def rebind(self, st):
        self.edit_exc(st, lambda e: e.setdefault("inputs", {}).update(result_sha256=sha(self.results[st])))

    def rscn(self):
        return RD.rscn_check(self.roots, self.results, SIDS, dict(self.exclusions), EXP)

    def g3(self, exclusions="all"):
        return RD.g3_check(self.roots, self.results, SIDS, EXP, exclusions=dict(self.exclusions) if exclusions == "all" else exclusions)


def sub(tmp_path, name, **kw):
    d = tmp_path / name
    d.mkdir()
    return World(d, **kw)


def problems(w, needle):
    rs = w.rscn()
    return [p for p in rs["problems"] if needle in p], rs


def red(w, needle):
    """the R-SCN read of this world must name `needle`, and G3 must fail with it."""
    hits, rs = problems(w, needle)
    assert hits, (needle, rs["problems"])
    gate, recs = w.g3()
    assert not gate["PASS"] and recs is None and any(needle in x for x in gate["why"]), gate["why"]


# ============================================================================================================================ the layouts, pinned to the drivers
def test_stage_cells_are_the_drivers_cells():
    for st in (2, 3, 4):
        for p in ("deciding", "beside"):
            assert RD.STAGE_CELLS[st][p] == frozenset(f"{tc}|{a}" for tc, a in PASS_CELLS[st][p]), (st, p)
    assert [len(RD.STAGE_CELLS[s]["deciding"]) + len(RD.STAGE_CELLS[s]["beside"]) for s in (2, 3, 4)] == [14, 14, 4]
    assert RD.STAGE_LABEL == {2: "stage2", 3: "stage3", 4: "stage4"}
    assert set(RD.RSCN_REASONS) == set(X.REASONS)


# ============================================================================================================================ item 2: scene_exclusions.py on the stage-3 and stage-4 layouts
def test_builder_on_stage3_and_stage4_layouts(tmp_path):
    w = World(tmp_path)
    e3, e4 = w.exc(3), w.exc(4)
    assert e3["verdict"] == "ATTRIBUTED" and e3["stage"] == "stage3" and {k: v["reason"] for k, v in e3["excluded"].items()} == {"ffffffff": "NO_DYNAMIC_LAYER"}, e3["problems"]
    assert e3["excluded"]["ffffffff"]["n_failed_cells"] == 14 == e3["excluded"]["ffffffff"]["n_cell_dirs"] and e3["excluded"]["ffffffff"]["cell_states"] == ["FAIL0"]
    assert e3["logs"]["deciding"]["n_events"] == 10 and e3["logs"]["beside"]["n_events"] == 4
    assert e4["verdict"] == "ATTRIBUTED" and e4["stage"] == "stage4", e4["problems"]
    assert {k: v["reason"] for k, v in e4["excluded"].items()} == {"bbbbbbbb": "ACTOR_REFUSED", "cccccccc": "ACTOR_REFUSED", "ffffffff": "NO_DYNAMIC_LAYER"}
    assert all(v["n_failed_cells"] == 4 == v["n_cell_dirs"] and v["cell_states"] == ["FAIL0"] for v in e4["excluded"].values())          # 4 cells per scene in stage 4
    assert e4["logs"]["deciding"]["n_events"] == 6 and e4["logs"]["beside"]["n_events"] == 6 and e4["excluded"]["cccccccc"]["tracks_file_count"] == 7
    assert e4["manifest_sha256"] == "m" * 64 and e4["problems"] == []


def test_builder_red_arms_on_stage3_and_stage4_layouts(tmp_path):
    w = World(tmp_path)
    d = w.cell(4, 2, "T2", "C0C_s1")                                            # stage 4: one cell of a refused scene is DONE
    os.remove(d / "FAIL")
    (d / "DONE").write_text("rc=0 records=2/2 executor=base\n")
    e = w.rebuild_exclusions(4)
    assert e["verdict"] == "UNATTRIBUTED" and any("not every cell is FAIL" in p for p in e["problems"]), e["problems"]
    os.remove(d / "DONE")
    (d / "FAIL").write_text("rc=1 records=0/2 executor=base\n")
    assert w.rebuild_exclusions(4)["verdict"] == "ATTRIBUTED"
    w2 = sub(tmp_path, "stray", stray_tb_stage=3)
    e = w2.exc(3)                                                               # stage 3: a stray traceback in the deciding log
    assert e["verdict"] == "UNATTRIBUTED" and any("terminal events" in p for p in e["problems"]), e["problems"]
    e = X.build(w.result(4), str(w.pull[4]), w.logs[4], w.src[3])               # stage 4 built with the guard sources only: the refused launch scenes' tracks files are not in them
    assert e["verdict"] == "UNATTRIBUTED" and any("no sequence_tracks.json" in p for p in e["problems"]), e["problems"]
    e = X.build(w.result(4), str(w.pull[4]), w.logs[4], w.allsrc)               # the stage-4 builder takes the stage-2 + stage-3 sources concatenated (62 scenes in production)
    assert e["verdict"] == "ATTRIBUTED" and all(v["idx"] is not None for v in e["excluded"].values())
    e = X.build(w.result(4), str(w.pull[4]), w.logs[4], w.src[2])               # the launch sources only: the builder accepts a NO_DYNAMIC_LAYER scene with no index; the READ does not (RSCN_IDX_MISMATCH)
    assert e["excluded"]["ffffffff"]["idx"] is None
    w.exclusions[4] = str(tmp_path / "EXC4_launch_sources.json")
    e["inputs"] = {"result_sha256": sha(w.results[4]), "sources_sha256": "0" * 64}
    open(w.exclusions[4], "w", encoding="utf-8", newline="\n").write(json.dumps(e))
    red(w, "RSCN_IDX_MISMATCH stage 4 scene ffffffff")


# ============================================================================================================================ G3 under R-SCN: the positive read
def test_g3_positive_union_removed_from_both_arms(tmp_path):
    w = World(tmp_path)
    assert (w.cell(2, 2, "T2", "P10") / "DONE").exists() and (w.cell(2, 2, "T2", "P10") / "records.jsonl").exists()            # scene 2 is DONE in stage 2 ...
    assert (w.cell(4, 2, "T2", "C0C_s0") / "FAIL").exists()                                                                       # ... and refused in stage 4 only
    gate, recs = w.g3()
    assert gate["PASS"] and recs and gate["pairing"] == {"s0": "PAIRED", "s1": "PAIRED"}, gate["why"]
    assert gate["n_starts"] == {"P10": 7, "C0C_s0": 7, "C0C_s1": 7}, gate["n_starts"]                                               # 11 starts - (1 + 2 + 1) lost
    for arm, rs_ in recs.items():
        assert sorted({r["scene"] for r in rs_}) == [0, 3, 4], arm                                                                  # the union {1, 2, 5} is gone from BOTH arms
    u = gate["r_scn"]
    assert {k: v["stages"] for k, v in u["union"].items()} == {"bbbbbbbb": {"2": "ACTOR_REFUSED", "4": "ACTOR_REFUSED"}, "cccccccc": {"4": "ACTOR_REFUSED"}, "ffffffff": {"3": "NO_DYNAMIC_LAYER", "4": "NO_DYNAMIC_LAYER"}}
    assert u["starts"]["lost_launch"] == 3 and u["starts"]["lost_guard"] == 1 and u["starts"]["read_on"] == {"launch": 5, "guard": 2, "total": 7}, u["starts"]
    assert u["scenes"] == {"of": 6, "excluded": 3, "read_on": 3}
    assert u["per_stage"]["2"]["excluded"]["bbbbbbbb"] == {"scene": "bbbbbbbb-1", "idx": 1, "reason": "ACTOR_REFUSED", "cells_failed": {"deciding": 10, "beside": 4}, "cells_total": 14, "starts_lost": 1}
    assert u["per_stage"]["4"]["excluded"]["cccccccc"]["cells_failed"] == {"deciding": 2, "beside": 2} and u["per_stage"]["4"]["excluded"]["cccccccc"]["starts_lost"] == 2
    assert u["per_stage"]["2"]["deciding_done"] == 30 == u["per_stage"]["2"]["deciding_required"] and u["per_stage"]["4"]["beside_required"] == 6


def test_stage2_only_exclusion_is_removed_from_c0c(tmp_path):
    w = World(tmp_path, ex={4: (5,)})                                           # scene 1 is refused in stage 2 and DONE in stage 4: rule 4 still takes it out of C0C
    assert (w.cell(4, 1, "T2", "C0C_s0") / "DONE").exists()
    gate, recs = w.g3()
    assert gate["PASS"], gate["why"]
    assert all(1 not in {r["scene"] for r in rs_} for rs_ in recs.values()) and gate["n_starts"] == {"P10": 9, "C0C_s0": 9, "C0C_s1": 9}, gate["n_starts"]          # scenes 0, 2, 3, 4: 2 + 2 + 3 + 2 starts
    assert sorted({r["scene"] for r in recs["C0C_s0"]}) == sorted({r["scene"] for r in recs["P10"]}) == [0, 2, 3, 4]


def test_stage4_only_exclusion_is_removed_from_p10(tmp_path):
    """Scene 2 is DONE in stage 2 and refused in stage 4 only. P10 must lose it; left in, P10 would pair on a scene C0C does not have (the arm of `pair_check` shown below)."""
    w = World(tmp_path)
    gate, recs = w.g3()
    assert gate["PASS"] and 2 not in {r["scene"] for r in recs["P10"]} and 2 not in {r["scene"] for r in recs["C0C_s0"]}, gate["why"]
    L, S = RD.STATE["L"], RD.STATE["S"]
    with_2 = [{"scene": 2, "f0": 40}] + [r for r in recs["P10"]]
    with pytest.raises(ValueError):
        S.pair_check(L.pair_keys(recs["C0C_s0"]), L.pair_keys(sorted(with_2, key=lambda r: (r["scene"], r["f0"]))))                 # what an unremoved scene would do to the pairing
    assert RD.rscn_pairs_on_excluded({"P10": with_2}, {2}) and not RD.rscn_pairs_on_excluded({"P10": recs["P10"]}, {2})


def test_clean_world_reads_the_same_with_and_without_the_rule(tmp_path):
    w = World(tmp_path, ex={2: (), 3: (), 4: ()})
    assert all(w.result(s)["overall"] == "COMPLETE" and w.exc(s)["n_excluded"] == 0 for s in (2, 3, 4))
    g_with, r_with = w.g3()
    g_without, r_without = w.g3(exclusions=None)
    assert g_with["PASS"] and g_without["PASS"] and r_with == r_without and g_with["n_starts"] == g_without["n_starts"] == {"P10": 11, "C0C_s0": 11, "C0C_s1": 11}
    assert "r_scn" in g_with and "r_scn" not in g_without and g_with["r_scn"]["union"] == {} and g_with["r_scn"]["starts"]["lost_total"] == 0


def test_without_the_flags_the_read_is_the_previous_pin(tmp_path):
    w = World(tmp_path)
    gate, recs = w.g3(exclusions=None)
    assert not gate["PASS"] and recs is None and "r_scn" not in gate
    assert any("G3_OVERALL_NOT_COMPLETE stage 2: 'INCOMPLETE'" in x for x in gate["why"]) and any("G3_CELL_NOT_DONE" in x and "FAIL marker" in x for x in gate["why"]), gate["why"][:4]


# ============================================================================================================================ the six RED arms of the brief, each with its named problem
def test_red_a_failed_cell_outside_the_excluded_scenes(tmp_path):
    w = World(tmp_path)
    w.edit_result(2, lambda r: r["steps"]["rollouts"]["state"]["deciding"]["failed"].append("aaaaaaaa|T2|P10"))
    red(w, "RSCN_FAILED_OUTSIDE stage 2 deciding")
    w2 = sub(tmp_path, "w2")
    w2.edit_result(4, lambda r: r["steps"]["rollouts"]["state"]["beside"]["failed"].append("dddddddd|RR|C0C_s1"))
    red(w2, "RSCN_FAILED_OUTSIDE stage 4 beside")


def test_red_b_partial_scene_or_a_done_cell_in_an_excluded_scene(tmp_path):
    w = World(tmp_path)
    w.edit_result(2, lambda r: r["steps"]["rollouts"]["state"]["deciding"]["failed"].pop())                          # only part of the refused scene's cells failed (result)
    red(w, "RSCN_PARTIAL_SCENE stage 2 deciding")
    w2 = sub(tmp_path, "w2")
    w2.edit_result(4, lambda r: r["steps"]["rollouts"]["state"]["beside"]["failed"].pop(0))
    red(w2, "RSCN_PARTIAL_SCENE stage 4 beside")
    w3 = sub(tmp_path, "w3")
    d = w3.cell(4, 2, "T2", "C0C_s1")                                                                            # one DONE cell in an excluded scene (disk), the result and the exclusions unchanged
    os.remove(d / "FAIL")
    (d / "records.jsonl").write_text('{"scene": 2, "f0": 40}\n{"scene": 2, "f0": 50}\n')
    (d / "DONE").write_text("rc=0 records=2/2 executor=base\n")
    red(w3, "RSCN_EXCLUDED_SCENE_HAS_DONE_CELL stage 4 scene cccccccc")
    w4 = sub(tmp_path, "w4")
    shutil.rmtree(w4.cell(3, 5, "RR", "BLIND"))                                                                  # an excluded scene whose cell directory is absent
    red(w4, "RSCN_EXCLUDED_CELLS_ABSENT stage 3 scene ffffffff")


def test_red_c_unattributed_file(tmp_path):
    w = World(tmp_path, stray_tb_stage=3)                                                                        # the REAL builder says UNATTRIBUTED (a stray traceback in the log)
    assert w.exc(3)["verdict"] == "UNATTRIBUTED"
    red(w, "RSCN_NOT_ATTRIBUTED stage 3")
    w2 = sub(tmp_path, "w2")
    w2.edit_exc(4, lambda e: e.update(problems=["a problem was listed"]))                                        # a verdict ATTRIBUTED with a problem still listed is refused too
    red(w2, "RSCN_NOT_ATTRIBUTED stage 4")


def test_red_d_exclusions_of_another_stage(tmp_path):
    w = World(tmp_path)
    w.exclusions[2], w.exclusions[3] = w.exclusions[3], w.exclusions[2]                                          # stage 3's file given as --exclusions2: same manifest sha, other stage
    assert w.exc(2)["manifest_sha256"] == w.result(2)["manifest_sha256"]
    red(w, "RSCN_OTHER_STAGE stage 2")
    w2 = sub(tmp_path, "w2")
    w2.edit_exc(4, lambda e: e.update(manifest_sha256="z" * 64))                                                 # built from a result of another package
    red(w2, "RSCN_OTHER_RESULT stage 4: the exclusions were built from another stage result (manifest_sha256")
    w3 = sub(tmp_path, "w3")
    w3.edit_result(3, lambda r: r.update(note="the result file changed after the exclusions were built"), rebind=False)
    red(w3, "RSCN_OTHER_RESULT stage 3: the exclusions were built from another stage result file (inputs.result_sha256")
    w4 = sub(tmp_path, "w4")
    w4.edit_exc(2, lambda e: e.pop("inputs"))
    red(w4, "RSCN_NO_RESULT_BINDING stage 2")
    w5 = sub(tmp_path, "w5")
    w5.edit_result(3, lambda r: r.update(stage="stage2"))                                                        # the result handed over is another stage's
    red(w5, "RSCN_OTHER_STAGE stage 3")


def test_red_e_a_pairing_that_still_sits_on_an_excluded_scene(tmp_path):
    w = World(tmp_path)
    f = w.cell(2, 0, "T2", "P10") / "records.jsonl"                                                              # a P10 record of an INCLUDED scene's cell that is labelled with excluded scene 1
    f.write_text(f.read_text() + json.dumps({"scene": 1, "f0": 25}) + "\n")
    (w.cell(2, 0, "T2", "P10") / "DONE").write_text("rc=0 records=3/3 executor=base\n")
    gate, recs = w.g3()
    assert not gate["PASS"] and recs is None and any("RSCN_PAIRS_ON_EXCLUDED P10: records on excluded scenes [1]" in x for x in gate["why"]), gate["why"]


def test_red_f_scene_excluded_in_stage4_only_left_in_p10(tmp_path):
    """The post-condition on its own, then the whole read: P10 records that still hold stage 4's refused scene 2 are named, and so is C0C holding stage 2's scene 1."""
    out = RD.rscn_pairs_on_excluded({"P10": [{"scene": 0, "f0": 20}, {"scene": 2, "f0": 40}], "C0C_s0": [{"scene": 0, "f0": 20}]}, {1, 2, 5})
    assert out == ["RSCN_PAIRS_ON_EXCLUDED P10: records on excluded scenes [2]"], out
    assert RD.rscn_pairs_on_excluded({"P10": [{"scene": 0, "f0": 20}], "C0C_s0": [{"scene": 1, "f0": 25}]}, {1, 2, 5}) == ["RSCN_PAIRS_ON_EXCLUDED C0C_s0: records on excluded scenes [1]"]
    w = World(tmp_path)
    f = w.cell(2, 3, "T2", "P10") / "records.jsonl"
    f.write_text(f.read_text() + json.dumps({"scene": 2, "f0": 40}) + "\n")
    (w.cell(2, 3, "T2", "P10") / "DONE").write_text("rc=0 records=4/4 executor=base\n")
    gate, _ = w.g3()
    assert any("RSCN_PAIRS_ON_EXCLUDED P10: records on excluded scenes [2]" in x for x in gate["why"]), gate["why"]


# ============================================================================================================================ the other conditions of G3 under R-SCN
def test_red_flags_all_three_or_none(tmp_path):
    w = World(tmp_path)
    for drop in (2, 3, 4):
        part = {s: p for s, p in w.exclusions.items() if s != drop}
        gate, recs = w.g3(exclusions=part)
        assert not gate["PASS"] and recs is None and any(f"RSCN_FLAGS_PARTIAL: no exclusions file for stage [{drop}]" in x for x in gate["why"]), gate["why"]


def test_red_included_cells_must_all_be_done(tmp_path):
    w = World(tmp_path)
    os.remove(w.cell(2, 0, "T2", "ZERO") / "DONE")                                                               # a cell the dominance rows never use, in an included scene
    red(w, "RSCN_INCLUDED_CELL_NOT_DONE stage 2 scene aaaaaaaa T2|ZERO")
    w2 = sub(tmp_path, "w2")
    shutil.rmtree(w2.cell(3, 4, "RR", "DAGOACC_s1"))                                                             # a beside cell of an included scene is absent
    red(w2, "RSCN_INCLUDED_CELL_ABSENT stage 3 scene eeeeeeee RR|DAGOACC_s1")
    w3 = sub(tmp_path, "w3")
    w3.edit_result(3, lambda r: r["steps"]["rollouts"]["state"]["beside"].update(done=r["steps"]["rollouts"]["state"]["beside"]["done"] - 1))
    red(w3, "RSCN_INCLUDED_NOT_DONE stage 3 beside")
    w4 = sub(tmp_path, "w4")
    w4.edit_result(4, lambda r: r["steps"]["rollouts"]["state"]["deciding"].update(done=r["steps"]["rollouts"]["state"]["deciding"]["done"] - 2))
    red(w4, "RSCN_INCLUDED_NOT_DONE stage 4 deciding")
    w5 = sub(tmp_path, "w5")
    (w5.cell(2, 3, "RR", "P10") / "DONE").write_text("rc=0 records=1/3 executor=base\n")                         # DONE says 1 of 3, the file holds 3
    red(w5, "RSCN_INCLUDED_CELL_NOT_DONE stage 2 scene dddddddd RR|P10: DONE says records=1/3, file has 3")


def test_red_overall_incomplete_only_under_the_conditions(tmp_path):
    w = World(tmp_path)
    assert w.result(2)["overall"] == w.result(4)["overall"] == "INCOMPLETE" and w.g3()[0]["PASS"]                  # INCOMPLETE is accepted with exactly the excluded scenes failed
    w1 = sub(tmp_path, "w1")
    w1.edit_result(2, lambda r: r.update(overall="COMPLETE"))
    red(w1, "RSCN_OVERALL stage 2: overall COMPLETE but excluded scenes")
    w2 = sub(tmp_path, "w2")
    w2.edit_result(3, lambda r: r.update(overall="GATE_FAILED"))
    red(w2, "G3_OVERALL_NOT_COMPLETE stage 3: 'GATE_FAILED'")
    w3 = World(tmp_path / "w3", ex={2: (), 3: (), 4: ()})
    w3.edit_result(3, lambda r: r.update(overall="INCOMPLETE"))                                                  # INCOMPLETE and nothing excluded: not a refusal, something else did not run
    red(w3, "RSCN_OVERALL stage 3: overall INCOMPLETE with no excluded scene")
    w4 = sub(tmp_path, "w4")
    w4.edit_result(4, lambda r: r["steps"]["rollouts"].update(passes_skipped_disk=[{"pass": "beside", "reason": "DISK_LOW"}]))
    red(w4, "RSCN_PASS_SKIPPED stage 4")
    w5 = sub(tmp_path, "w5")
    w5.edit_result(2, lambda r: r["steps"]["rollouts"].update(scenes_skipped={"dddddddd": "failed its check"}))
    red(w5, "RSCN_SCENES_SKIPPED stage 2")
    w6 = sub(tmp_path, "w6")
    w6.edit_result(3, lambda r: r["steps"]["rollouts"].update(n_cells_planned=27))
    red(w6, "RSCN_CELLS_PLANNED stage 3: 27 cells planned, 28 expected")
    w7 = sub(tmp_path, "w7")
    w7.edit_result(4, lambda r: r.update(n_scenes_manifest=5))
    red(w7, "RSCN_SCENE_COUNT stage 4: the result ran 5 scenes, 6 expected")


def test_red_exclusion_entries_are_checked(tmp_path):
    w = World(tmp_path)
    w.edit_exc(2, lambda e: e["excluded"]["bbbbbbbb"].update(idx=2))
    red(w, "RSCN_IDX_MISMATCH stage 2 scene bbbbbbbb")
    w2 = sub(tmp_path, "w2")
    w2.edit_exc(2, lambda e: e["excluded"].update({"eeeeeeee": dict(e["excluded"]["bbbbbbbb"], idx=4, scene="eeeeeeee-4")}))             # a guard scene named in stage 2
    red(w2, "RSCN_SCENE_NOT_IN_STAGE stage 2: excluded scene eeeeeeee")
    w3 = sub(tmp_path, "w3")
    w3.edit_exc(3, lambda e: e["excluded"]["ffffffff"].update(reason="SOMETHING_ELSE"))
    red(w3, "RSCN_REASON stage 3 scene ffffffff")
    w4 = sub(tmp_path, "w4")
    w4.edit_exc(4, lambda e: e["excluded"]["cccccccc"].update(problems=["x"]))
    red(w4, "RSCN_SCENE_UNATTRIBUTED stage 4 scene cccccccc")
    w5 = sub(tmp_path, "w5")
    (w5.cell(4, 1, "RR", "C0C_s1") / "FAIL").write_text("rc=1 records=0/9 executor=base\n")                      # the cells of one refused scene disagree on how many starts it had
    red(w5, "RSCN_STARTS_UNKNOWN stage 4 scene bbbbbbbb")
    w6 = sub(tmp_path, "w6")
    (w6.cell(2, 2, "T2", "P10") / "DONE").write_text("rc=0 records=3/3 executor=base\n")                          # scene 2 (refused in stage 4): its cells across the stages disagree on the number of starts
    red(w6, "RSCN_STARTS_UNKNOWN scene cccccccc")


def test_start_counts_fall_by_the_starts_of_the_excluded_scenes(tmp_path):
    w = World(tmp_path)
    f = w.cell(4, 3, "T2", "C0C_s1") / "records.jsonl"
    f.write_text("".join(f.read_text().splitlines(True)[:-1]))
    (w.cell(4, 3, "T2", "C0C_s1") / "DONE").write_text("rc=0 records=2/2 executor=base\n")                       # one start fewer in C0C_s1 than the read expects (7)
    gate, recs = w.g3()
    assert not gate["PASS"] and recs is None and any("G3_START_COUNT C0C_s1: 6 != 7" in x for x in gate["why"]), gate["why"]
    assert any("G3_PAIRING seed 1" in x for x in gate["why"])


# ============================================================================================================================ item 3: G2(e)
def test_g2e_tabulates_excluded_scenes_and_still_needs_the_file_for_included_ones(tmp_path):
    w = World(tmp_path)
    labels = RD.rscn_labels(w.rscn())
    assert labels == {"bbbbbbbb": "ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)", "cccccccc": "ACTOR_REFUSED (stage 4)", "ffffffff": "NO_DYNAMIC_LAYER (stage 3), NO_DYNAMIC_LAYER (stage 4)"}
    cells = [("stage2", w.roots[2], "P10"), ("stage3", w.roots[3], "P10"), ("stage4", w.roots[4], "C0C_s0"), ("stage4", w.roots[4], "C0C_s1")]
    out = RD.g2e_check(cells, excluded=labels)
    assert out["PASS"], out["why"]
    ps = out["per_scene"]
    assert ps["bbbbbbbb"] == {"P10@stage2": "EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)", "C0C_s0@stage4": "EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)",
                              "C0C_s1@stage4": "EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)"}, ps["bbbbbbbb"]
    assert ps["cccccccc"]["P10@stage2"].startswith("EXCLUDED (R-SCN): ACTOR_REFUSED (stage 4); verdict on file ACCEPTED")      # excluded in stage 4 only: its stage-2 file exists and is shown, the scene is still out
    assert {k: v for k, v in ps["aaaaaaaa"].items()} == {"P10@stage2": "ACCEPTED", "C0C_s0@stage4": "ACCEPTED", "C0C_s1@stage4": "ACCEPTED"}
    assert out["non_accepted"] == []
    os.remove(w.cell(2, 0, "T2", "P10") / "actor_attach.json")                                                  # an INCLUDED scene without the file is still a failure
    bad = RD.g2e_check(cells, excluded=labels)
    assert not bad["PASS"] and any("G2E_ACTOR_ATTACH_MISSING stage2 aaaaaaaa P10" in x for x in bad["why"]), bad["why"]
    w2 = sub(tmp_path, "w2")
    plain = RD.g2e_check([("stage4", w2.roots[4], "C0C_s0")])                                                    # without the rule an excluded scene's missing file IS a failure (the previous pin's reading)
    assert not plain["PASS"] and any("G2E_ACTOR_ATTACH_MISSING stage4 bbbbbbbb C0C_s0" in x for x in plain["why"]), plain["why"]


# ============================================================================================================================ item 4 and the wiring of the flags: main() end to end on the synthetic world
def _wire(w, tmp_path, monkeypatch, flags=(2, 3, 4)):
    import test_t2v8f_dominance as T
    seen = {"skip": None, "recs": None, "excluded": "unset"}
    monkeypatch.setattr(RD, "EXPECT", EXP)
    monkeypatch.setattr(RD, "g4_check", lambda *a, **k: {"PASS": True, "why": []})
    monkeypatch.setattr(RD, "g1_check", lambda *a, **k: {"PASS": True, "why": []})
    monkeypatch.setattr(RD, "g2a_check", lambda *a, **k: {"PASS": True, "why": []})
    monkeypatch.setattr(RD, "g2b_check", lambda *a, **k: {"PASS": True, "why": []})
    real_g2e = RD.g2e_check

    def g2e(cells, excluded=None):
        seen["excluded"] = excluded
        return real_g2e(cells, excluded=excluded)
    monkeypatch.setattr(RD, "g2e_check", g2e)

    def scenes_from(recs, srcs, roots, skip=()):
        seen["skip"] = set(skip)
        return {}

    def arrays_for_seed(rc, rp, scenes, s):
        seen["recs"] = (sorted({r["scene"] for r in rc}), sorted({r["scene"] for r in rp}))
        return T.synth(T.dom_effects(), seed=s + 1)
    real_blocks = RD.read_blocks
    monkeypatch.setattr(RD, "scenes_from", scenes_from)
    monkeypatch.setattr(RD, "arrays_for_seed", arrays_for_seed)
    monkeypatch.setattr(RD, "read_blocks", lambda arrays, n_boot=200: real_blocks(arrays, n_boot=200))
    s2, s3 = tmp_path / "SCENE_SOURCES.json", tmp_path / "SCENE_SOURCES3.json"
    s2.write_text(json.dumps(w.src[2]))
    s3.write_text(json.dumps(w.src[3]))
    g1 = tmp_path / "g1.json"
    g1.write_text("{}")
    out = tmp_path / "out"
    argv = ["--root2", w.roots[2], "--root3", w.roots[3], "--root4", w.roots[4], "--sources2", str(s2), "--sources3", str(s3), "--g1", str(g1), "--out-dir", str(out)]
    for s in flags:
        argv += [f"--exclusions{s}", w.exclusions[s]]
    return argv, out, seen


def test_main_with_the_three_flags_reads_the_refusal_surviving_scenes_and_discloses(tmp_path, monkeypatch):
    w = sub(tmp_path, "w")
    argv, out, seen = _wire(w, tmp_path, monkeypatch)
    assert RD.main(argv) == 0
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    md = open(out / "DOMINANCE_T2.md", encoding="utf-8").read()
    assert j["outcome"] == "DOMINANT-T2" and all(g["PASS"] for g in j["gates"].values()), {k: g.get("why") for k, g in j["gates"].items()}
    assert seen["skip"] == {1, 2, 5} and seen["recs"] == ([0, 3, 4], [0, 3, 4])                                    # the union reached the scene loader and both arms
    assert set(seen["excluded"]) == {"bbbbbbbb", "cccccccc", "ffffffff"}                                         # ... and G2(e)
    assert j["gates"]["G2"]["parts"]["e"]["per_scene"]["bbbbbbbb"]["P10@stage2"].startswith("EXCLUDED (R-SCN): ACTOR_REFUSED")
    r = j["r_scn"]
    assert r["rule"] == RD.RSCN_RULE and r["scenes"] == {"of": 6, "excluded": 3, "read_on": 3} and r["starts"]["lost_total"] == 4
    assert r["per_stage"]["2"]["excluded"]["bbbbbbbb"]["cells_failed"] == {"deciding": 10, "beside": 4} and r["union"]["cccccccc"]["stages"] == {"4": "ACTOR_REFUSED"}
    assert "## Excluded scenes (rule R-SCN)" in md and "| 2 | bbbbbbbb | 1 | ACTOR_REFUSED | 10 + 4 of 14 | 1 |" in md and "| 4 | cccccccc | 2 | ACTOR_REFUSED | 2 + 2 of 4 | 2 |" in md
    assert "| 3 | ffffffff | 5 | NO_DYNAMIC_LAYER | 10 + 4 of 14 | 1 |" in md and "| cccccccc | 2 | 4: ACTOR_REFUSED | 2 |" in md
    assert "Scenes read on: 3 of 6" in md and "Starts read on: 7 of 11" in md and "starts lost 4 (launch 3, guard 1)" in md and "Rule R-SCN applied: this read is on the 3 refusal-surviving scenes of the 62" in md
    assert "EXCLUDED (R-SCN): ACTOR_REFUSED (stage 2), ACTOR_REFUSED (stage 4)" in md                             # G2(e) in the md


def test_main_with_two_flags_or_none_is_gate_fail(tmp_path, monkeypatch):
    w = sub(tmp_path, "w")
    argv, out, seen = _wire(w, tmp_path, monkeypatch, flags=(2, 4))
    RD.main(argv)
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j["outcome"] == "GATE-FAIL" and any("RSCN_FLAGS_PARTIAL: no exclusions file for stage [3]" in x for x in j["gates"]["G3"]["why"]), j["gates"]["G3"]["why"]
    assert seen["skip"] is None                                                                                 # no row was read
    argv, out, seen = _wire(w, tmp_path, monkeypatch, flags=())
    RD.main(argv)
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    md = open(out / "DOMINANCE_T2.md", encoding="utf-8").read()
    assert j["outcome"] == "GATE-FAIL" and "r_scn" not in j and "R-SCN" not in md.split("## Gates")[0] and "## Excluded scenes" not in md        # no flags: the previous reading, INCOMPLETE stages refuse
    assert any("G3_OVERALL_NOT_COMPLETE stage 2: 'INCOMPLETE'" in x for x in j["gates"]["G3"]["why"])


def test_main_clean_world_without_flags_reads(tmp_path, monkeypatch):
    w = sub(tmp_path, "w", ex={2: (), 3: (), 4: ()})
    argv, out, seen = _wire(w, tmp_path, monkeypatch, flags=())
    RD.main(argv)
    j = json.load(open(out / "DOMINANCE_T2.json", encoding="utf-8"))
    assert j["outcome"] == "DOMINANT-T2" and "r_scn" not in j and seen["recs"] == ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]) and seen["skip"] == set()
