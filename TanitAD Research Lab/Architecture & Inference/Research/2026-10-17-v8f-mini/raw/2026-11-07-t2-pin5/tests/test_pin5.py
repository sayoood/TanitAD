"""The FIFTH pin of the T2 dominance read (`pin5/read_t2_dominance_pin5.py`): `--arm-seeds {0,1 | 0}`. SYNTHETIC worlds only (the frozen helper modules of `tests/`: the rscn world, the stage-5 arm world,
the synthetic arrays): no stage rollout is opened. CPU only. The real-data identity and the real seed-0 read are `test_pin5_real.py`.

`RD` = the pin-5 module under test (the mutation runner swaps this global and the helper modules' `RD`); `RD4` = the fourth pin, loaded from `code/read_t2_dominance.py` (sha256 checked) as the reference of the identity.
The helper modules import `read_t2_dominance` by name: it is bound to the pin-5 module here, so they build their worlds and call their checks through it."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
T2V8F = os.path.normpath(os.path.join(HERE, "..", "..")).replace("\\", "/")
PIN4_SHA = "7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9"
sys.path.insert(0, f"{T2V8F}/code")
sys.path.insert(0, f"{T2V8F}/tests")
import pytest  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RD = _load(f"{T2V8F}/pin5/read_t2_dominance_pin5.py", "read_t2_dominance")
sys.modules["read_t2_dominance"] = RD
assert hashlib.sha256(open(f"{T2V8F}/code/read_t2_dominance.py", "rb").read()).hexdigest() == PIN4_SHA, "the fourth pin changed"
RD4 = _load(f"{T2V8F}/code/read_t2_dominance.py", "read_t2_dominance_pin4")
import test_t2v8f_dominance as T  # noqa: E402
import test_t2v8f_dominance_q as TQ  # noqa: E402
import test_t2v8f_dominance_rscn as R  # noqa: E402

ARM = TQ.ARM
JSON_NAME, MD_NAME = "DOMINANCE_T2.json", "DOMINANCE_T2.md"


def ns(**kw):
    return types.SimpleNamespace(**kw)


# ============================================================================================================================ worlds
class World5S0(TQ.World5):
    """The stage-5 arm world of a `--seeds 0` driver: T2 x {ARM_s0} deciding, RR x {ARM_s0} beside, over all 6 scenes (stage 5 refuses scene 3 only); no seed-1 cell exists anywhere in that stage."""

    def __init__(self, tmp, monkeypatch, ex5=TQ.EX5, arm=ARM, ex=None):
        monkeypatch.setitem(R.STAGE_IDX, 5, TQ.IDX5)
        monkeypatch.setitem(R.PASS_CELLS, 5, {"deciding": [("T2", f"{arm}_s0")], "beside": [("RR", f"{arm}_s0")]})
        for i in ex5:
            monkeypatch.setitem(R.REASON, i, R.REASON.get(i, "ACTOR_REFUSED"))
            monkeypatch.setitem(R.NTRACKS, i, R.NTRACKS.get(i, 6))
        R.World.__init__(self, tmp, ex={**(ex or {}), 5: tuple(ex5)})
        self.arm = arm
        self.src[5] = [e for e in self.allsrc if e["idx"] in TQ.IDX5]
        self.build_stage(5)
        for i in TQ.IDX5:
            if i not in self.ex[5]:
                (self.cell(5, i, "T2", f"{arm}_s0") / "actor_attach.json").write_text(json.dumps({"verdict": "ACCEPTED"}))


def world5s0(tmp_path, monkeypatch, name="w5s0"):
    d = tmp_path / name
    d.mkdir()
    return World5S0(d, monkeypatch)


def poison_seed1(w, st=4, arm="C0C_s1"):
    """Mark every record of the arm's seed-1 cells: a read that loads them is visible to the recorder of `wire`."""
    n = 0
    for i in R.STAGE_IDX[st]:
        if i in w.ex[st]:
            continue
        p = w.cell(st, i, "T2", arm) / "records.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        p.write_text("".join(json.dumps({**r, "poison": True}) + "\n" for r in rows))
        n += len(rows)
    assert n > 0
    return n


def wire(mod, w, tmp_path, mp):
    """-> (go, seen). `go(extra_argv, flags, out)` runs `mod.main()` on the world with the heavy gates and rows replaced by recorders; `seen` records what reached them (including the seed and the poison of every `arrays_for_seed`
    call and the arm of every `load_cells` call)."""
    seen = {"as_calls": [], "load_arms": [], "g2e_cells": None, "skip": None, "g4_calls": 0}
    mp.setattr(mod, "EXPECT", R.EXP)

    def g4(*a, **k):
        seen["g4_calls"] += 1
        return {"PASS": True, "why": []}
    mp.setattr(mod, "g4_check", g4)
    mp.setattr(mod, "g1_check", lambda *a, **k: {"PASS": True, "why": []})
    mp.setattr(mod, "g2a_check", lambda *a, **k: {"PASS": True, "why": []})
    mp.setattr(mod, "g2b_check", lambda t2v8f_root=None, logs=None: {"PASS": True, "why": []})
    real_g2e, real_lc, real_blocks = mod.g2e_check, mod.load_cells, mod.read_blocks

    def g2e(cells, excluded=None):
        seen["g2e_cells"] = [(label, arm) for label, _, arm in cells]
        return real_g2e(cells, excluded=excluded)

    def load_cells(root, arm, *a, **k):
        seen["load_arms"].append(arm)
        return real_lc(root, arm, *a, **k)

    def scenes_from(recs, srcs, roots, skip=()):
        seen["skip"] = set(skip)
        return {}

    def arrays_for_seed(rc, rp, scenes, s, **kw):
        seen["as_calls"].append((s, any("poison" in r for r in rc), any("poison" in r for r in rp)))
        return T.synth(T.dom_effects(), seed=s + 1)
    mp.setattr(mod, "g2e_check", g2e)
    mp.setattr(mod, "load_cells", load_cells)
    mp.setattr(mod, "scenes_from", scenes_from)
    mp.setattr(mod, "arrays_for_seed", arrays_for_seed)
    mp.setattr(mod, "read_blocks", lambda arrays, n_boot=200: real_blocks(arrays, n_boot=200))
    s2, s3, g1 = tmp_path / "SCENE_SOURCES.json", tmp_path / "SCENE_SOURCES3.json", tmp_path / "g1.json"
    s2.write_text(json.dumps(w.src[2]))
    s3.write_text(json.dumps(w.src[3]))
    g1.write_text("{}")

    def go(extra=(), flags=(2, 3, 4), out="out"):
        argv = ["--root2", w.roots[2], "--root3", w.roots[3], "--root4", w.roots[4], "--sources2", str(s2), "--sources3", str(s3), "--g1", str(g1), "--out-dir", str(tmp_path / out)]
        for s in flags:
            argv += [f"--exclusions{s}", w.exclusions[s]]
        argv += list(extra)
        for k in ("as_calls", "load_arms"):
            seen[k] = []
        seen["g2e_cells"] = seen["skip"] = None
        assert mod.main(argv) == 0
        return argv, tmp_path / out

    return go, seen


def read(out):
    j = json.load(open(out / JSON_NAME, encoding="utf-8"))
    md = open(out / MD_NAME, encoding="utf-8").read()
    return j, md


def md_without_script_and_arm_lines(md):
    return [x for x in md.splitlines() if not x.startswith("Arm read:") and "script sha256" not in x]


# ============================================================================================================================ the flag, the layout, the outcome (pure)
def test_arm_seeds_forms_default_and_named_refusals():
    assert RD.SEEDS == (0, 1) and RD.arm_spec(ns())["seeds"] == (0, 1) and RD.arm_spec(ns(arm_seeds=None))["seeds"] == (0, 1)
    assert RD.arm_spec(ns(arm_seeds="0,1"))["seeds"] == (0, 1) and RD.arm_spec(ns(arm_seeds="0"))["seeds"] == (0,)
    assert RD.arm_spec(ns(arm_seeds=(0, 1)))["seeds"] == (0, 1) and RD.arm_spec(ns(arm_seeds=[0]))["seeds"] == (0,)
    assert RD.arm_spec(ns(arm="EGAIN", arm_stage=5, arm_root="r", arm_seeds="0"))["seeds"] == (0,)
    for bad in ("1", "1,0", "0,0", "", "0, 1", "01", "0,1,2", " 0", "all", (1,), [0, 1, 2], (True,), 0, 1.0, (0, 0), ()):
        with pytest.raises(SystemExit) as e:
            RD.arm_spec(ns(arm_seeds=bad))
        assert str(e.value).startswith("REFUSED: ") and "ARM_SEEDS_FORM" in str(e.value), (bad, str(e.value))
    for bad in ("1", "0,0"):                                                                                 # the refusal is not hidden behind the others: it is listed beside them
        with pytest.raises(SystemExit) as e:
            RD.arm_spec(ns(arm="P10", arm_seeds=bad))
        assert "ARM_SEEDS_FORM" in str(e.value) and "ARM_NAME_RESERVED" in str(e.value)
    assert RD.parse_arm_seeds("0,1") == (0, 1) and RD.parse_arm_seeds("0") == (0,) and RD.parse_arm_seeds("1") is None
    assert RD.OUTCOMES[:4] == ("DOMINANT-T2", "NOT-DOMINANT", "UNDERPOWERED", "GATE-FAIL") and RD.OUTCOMES[4] == RD.SEED0_ONLY == "SEED0-ONLY"


def test_the_layout_follows_the_seeds_beyond_stage4_and_stays_two_seed_at_stage4():
    assert RD.arm_cells("C0C") == RD.STAGE_CELLS[4] == RD.arm_cells("C0C", (0, 1))
    assert RD.arm_cells("LC", (0,)) == {"deciding": frozenset({"T2|LC_s0"}), "beside": frozenset({"RR|LC_s0"})}
    assert RD.stage_cells(5, "LC", 5) == RD.arm_cells("LC") and RD.stage_cells(5, "LC", 5, (0, 1)) == RD.arm_cells("LC")
    assert RD.stage_cells(5, "LC", 5, (0,)) == RD.arm_cells("LC", (0,))                                       # a stage beyond 4 ran the seeds that are read
    assert RD.stage_cells(4, "C0C", 4, (0,)) == RD.STAGE_CELLS[4]                                              # stage 4 ran both seeds whatever is read
    assert RD.stage_cells(4, "LC", 5, (0,)) == RD.STAGE_CELLS[4] and RD.stage_cells(2, "LC", 5, (0,)) == RD.STAGE_CELLS[2]
    assert RD.stage_ran_seeds(4, (0,)) == (0, 1) and RD.stage_ran_seeds(5, (0,)) == (0,) and RD.stage_ran_seeds(5) == (0, 1)


def test_decide_seed0_only_never_names_dominant_t2_and_the_default_is_unchanged():
    b0 = RD.read_blocks({0: T.synth(T.dom_effects(), seed=1)}, n_boot=200)[0]
    b1 = RD.read_blocks({1: T.synth(T.dom_effects(), seed=2)}, n_boot=200)[1]
    assert b0["read_a"]["DOMINANT"] and b1["read_a"]["DOMINANT"], "the control: both seed blocks of this synthetic world are DOMINANT"
    d2 = RD.decide(True, {0: b0, 1: b1})
    assert d2["outcome"] == "DOMINANT-T2" and sorted(d2["per_seed"]) == ["s0", "s1"] and sorted(d2) == ["failing_terms", "outcome", "per_seed", "reading_underpowered_first"]       # the default: the fourth pin's keys
    assert RD.decide(True, {0: b0, 1: b1}, (0, 1)) == d2
    d1 = RD.decide(True, {0: b0}, (0,))
    assert d1["outcome"] == "SEED0-ONLY" and d1["outcome"] != "DOMINANT-T2", d1["outcome"]
    assert d1["n_seeds"] == 1 and d1["counted_dominant"] is False and d1["statement"] == "seed 0 only, not a counted DOMINANT" and d1["seed0_block"] == "DOMINANT"
    assert sorted(d1["per_seed"]) == ["s0"] and d1["per_seed"]["s0"] == d2["per_seed"]["s0"] and d1["failing_terms"] == {"s0": d2["failing_terms"]["s0"]}
    null = RD.read_blocks({0: T.synth(None, seed=1)}, n_boot=200)[0]
    dn = RD.decide(True, {0: null}, (0,))
    assert dn["outcome"] == "SEED0-ONLY" and dn["seed0_block"] in ("NOT-DOMINANT", "UNDERPOWERED") and dn["counted_dominant"] is False, dn
    assert RD.decide(False, None, (0,))["outcome"] == "GATE-FAIL" and RD.decide(True, None, (0,))["outcome"] == "GATE-FAIL"                    # a failed gate is not relabelled
    with pytest.raises(ValueError, match="need both seeds"):
        RD.decide(True, {0: b0})                                                                              # the default still needs both seeds
    with pytest.raises(ValueError, match="need seeds"):
        RD.decide(True, {0: b0, 1: b1}, (0,))                                                                 # a seed-0-only read holds seed 0 alone
    with pytest.raises(ValueError):
        RD.decide(True, {1: b1}, (1,))
    assert RD.seed_gap_table({0: b0}) == {"computable": False, "reason": RD.SEED_GAP_NOT_COMPUTABLE}
    assert "needs both seeds" in RD.SEED_GAP_NOT_COMPUTABLE and RD.seed_gap_table({0: b0, 1: b1})[RD.STATE["rr"].GD.LABELS[0]]["gap"] >= 0


# ============================================================================================================================ main(), end to end, on synthetic worlds
def test_main_arm_seeds_0_on_the_c0c_stage4_world_reads_seed0_only_and_never_loads_seed1(tmp_path, monkeypatch):
    w = TQ.world4(tmp_path)
    poison_seed1(w)
    go, seen = wire(RD, w, tmp_path, monkeypatch)
    argv2, out2 = go([], out="both")                                                                          # the control: the default reads BOTH seeds and the probe sees the poisoned seed-1 records
    assert seen["as_calls"] == [(0, False, False), (1, True, False)], seen["as_calls"]
    assert "C0C_s1" in seen["load_arms"] and seen["g2e_cells"][-2:] == [("stage4", "C0C_s0"), ("stage4", "C0C_s1")]
    j2, md2 = read(out2)
    assert j2["outcome"] == "DOMINANT-T2" and sorted(j2["by_seed"]) == ["s0", "s1"]
    argv0, out0 = go(["--arm-seeds", "0"], out="seed0")
    j0, md0 = read(out0)
    assert seen["as_calls"] == [(0, False, False)], seen["as_calls"]                                          # only seed 0 reached the row builder, and no poisoned record with it
    assert not any(a.endswith("_s1") for a in seen["load_arms"]) and set(seen["load_arms"]) == {"P10", "ZERO", "C0C_s0"}, seen["load_arms"]                 # ZERO: G2 (d) pairs it with P10
    assert seen["g2e_cells"] == [("stage2", "P10"), ("stage3", "P10"), ("stage4", "C0C_s0")]
    assert j0["outcome"] == "SEED0-ONLY" and j0["decision"]["outcome"] == "SEED0-ONLY" and j0["arm"] == "C0C" and j0["arm_stage"] == 4 and j0["argv"] == argv0
    assert j0["arm_seeds"] == [0] and j0["n_seeds"] == 1 and j0["counted_dominant"] is False and j0["seed_statement"] == "seed 0 only, not a counted DOMINANT"
    assert j0["decision"]["n_seeds"] == 1 and j0["decision"]["counted_dominant"] is False and j0["decision"]["seed0_block"] == "DOMINANT" and sorted(j0["decision"]["per_seed"]) == ["s0"]
    assert sorted(j0["by_seed"]) == ["s0"] and sorted(j0["three_family_form_beside"]) == ["s0"] and j0["seed_gap_beside"] == {"computable": False, "reason": RD.SEED_GAP_NOT_COMPUTABLE}
    assert "DOMINANT-T2" not in json.dumps(j0) and "DOMINANT-T2" not in md0
    assert "# DOMINANCE_T2: SEED0-ONLY" in md0 and "Outcome: **SEED0-ONLY**. seed 0 only, not a counted DOMINANT: the seed-0 block alone reads DOMINANT, which is not an outcome." in md0
    assert "Seed 0 only: seed 0 only, not a counted DOMINANT." in md0 and "cells C0C_s0 only (seed 0 only)" in md0 and "Not computable: seed 0 only: the seed gap" in md0 and "## Per-family table, s1" not in md0
    assert j0["gates"]["G3"]["pairing"] == {"s0": "PAIRED"} and sorted(j0["gates"]["G3"]["n_starts"]) == ["C0C_s0", "P10"]
    # seed 0 of the seed-0-only read IS seed 0 of the two-seed read: rows, read_a, classes, family table, r_scn, the gates that do not name a seed-1 cell
    assert j0["by_seed"]["s0"] == j2["by_seed"]["s0"] and j0["three_family_form_beside"]["s0"] == j2["three_family_form_beside"]["s0"] and j0["r_scn"] == j2["r_scn"]
    assert j0["decision"]["per_seed"]["s0"] == j2["decision"]["per_seed"]["s0"] and len(j0["by_seed"]["s0"]["rows"]) == 19
    for g in ("G1", "G4"):
        assert j0["gates"][g] == j2["gates"][g]
    assert {k: v for k, v in j0["gates"]["G2"]["parts"].items() if k != "e"} == {k: v for k, v in j2["gates"]["G2"]["parts"].items() if k != "e"}
    assert all(g["PASS"] for g in j0["gates"].values())
    # the explicit '0,1' equals the default, field for field except the argv
    argv3, out3 = go(["--arm-seeds", "0,1"], out="both_explicit")
    j3, md3 = read(out3)
    assert {**j3, "argv": None} == {**j2, "argv": None} and [x for x in md3.splitlines() if not x.startswith("Arm read:")] == [x for x in md2.splitlines() if not x.startswith("Arm read:")]


def test_main_stage5_arm_seed0_only_is_read_on_a_seed0_stage_and_refused_by_name_on_a_two_seed_stage(tmp_path, monkeypatch):
    w0 = world5s0(tmp_path, monkeypatch)
    go, seen = wire(RD, w0, tmp_path, monkeypatch)
    argv, out = go(TQ.arm5_args(w0, tmp_path) + ["--arm-seeds", "0"], out="s0_world")
    j, md = read(out)
    assert j["outcome"] == "SEED0-ONLY" and all(g["PASS"] for g in j["gates"].values()), {k: g.get("why") for k, g in j["gates"].items()}
    assert j["arm"] == ARM and j["arm_stage"] == 5 and j["arm_seeds"] == [0] and sorted(j["by_seed"]) == ["s0"]
    assert j["r_scn"]["per_stage"]["5"]["cells_per_scene"] == {"deciding": 1, "beside": 1} and j["r_scn"]["per_stage"]["4"]["cells_per_scene"] == {"deciding": 2, "beside": 2}
    assert j["r_scn"]["scenes"] == {"of": 6, "excluded": 4, "read_on": 2} and seen["skip"] == {1, 2, 3, 5}
    assert seen["g2e_cells"] == [("stage2", "P10"), ("stage3", "P10"), ("stage5", f"{ARM}_s0")] and seen["as_calls"] == [(0, False, False)]
    assert set(seen["load_arms"]) == {"P10", "ZERO", f"{ARM}_s0"} and "cells EGAIN_s0 only (seed 0 only)" in md and "DOMINANT-T2" not in md
    # the converse, both ways: a seed-0 stage read as two seeds, and a two-seed stage read as seed 0 only. Both are gate failures NAMED, no row is read
    argv, out = go(TQ.arm5_args(w0, tmp_path), out="s0_world_default")
    j, _ = read(out)
    assert j["outcome"] == "GATE-FAIL" and seen["skip"] is None and seen["as_calls"] == []
    why = " ".join(j["gates"]["ARM"]["why"] + j["gates"]["G3"]["why"])
    assert "ARM_ABSENT_IN_PULL EGAIN_s1" in why and "RSCN_" in why, why
    w2 = TQ.world5(tmp_path, monkeypatch, name="w5_two_seeds")
    go2, seen2 = wire(RD, w2, tmp_path, monkeypatch)
    argv, out = go2(TQ.arm5_args(w2, tmp_path) + ["--arm-seeds", "0"], out="two_seed_world_seed0")
    j, _ = read(out)
    assert j["outcome"] == "GATE-FAIL" and j["arm_seeds"] == [0] and seen2["skip"] is None and seen2["as_calls"] == []
    assert "RSCN_CELLS_PLANNED stage 5" in " ".join(j["gates"]["G3"]["why"]), j["gates"]["G3"]["why"]


def test_main_refuses_a_wrong_arm_seeds_before_anything_is_read(tmp_path, monkeypatch):
    w = TQ.world4(tmp_path)
    go, seen = wire(RD, w, tmp_path, monkeypatch)
    for bad in ("1", "0,0", "1,0", ""):
        with pytest.raises(SystemExit) as e:
            go(["--arm-seeds", bad], out="out_bad")
        assert "ARM_SEEDS_FORM" in str(e.value) and not (tmp_path / "out_bad").exists() and seen["g4_calls"] == 0, (bad, str(e.value))


def test_default_read_is_the_fourth_pins_read_field_for_field_on_synthetic_worlds(tmp_path, monkeypatch):
    """The pin-5 module and the fourth pin read the SAME worlds with the SAME arguments: the JSON (script sha256 and argv aside), the md (the script-sha line and the arm line aside) and the files' key sets are equal."""
    w = TQ.world4(tmp_path)
    go5, seen5 = wire(RD, w, tmp_path, monkeypatch)
    go4, seen4 = wire(RD4, w, tmp_path, monkeypatch)
    for extra in ([], ["--arm", "C0C", "--arm-stage", "4"], ["--arm-seeds", "0,1"]):
        _, o5 = go5(extra, out="p5")
        _, o4 = go4([x for x in extra if x not in ("--arm-seeds", "0,1")], out="p4")
        j5, m5 = read(o5)
        j4, m4 = read(o4)
        assert j5["script_sha256"] == hashlib.sha256(open(RD.__file__, "rb").read()).hexdigest() and j4["script_sha256"] == PIN4_SHA and j5["script_sha256"] != j4["script_sha256"]
        assert {**j5, "script_sha256": None, "argv": None} == {**j4, "script_sha256": None, "argv": None} and sorted(j5) == sorted(j4)
        assert md_without_script_and_arm_lines(m5) == md_without_script_and_arm_lines(m4)
        a5 = [x for x in m5.splitlines() if x.startswith("Arm read:")][0].replace(str(o5), "<OUT>")
        a4 = [x for x in m4.splitlines() if x.startswith("Arm read:")][0].replace(str(o4), "<OUT>")
        assert a5.split(" argv: ")[0] == a4.split(" argv: ")[0]
        assert seen5["as_calls"] == seen4["as_calls"] == [(0, False, False), (1, False, False)] and seen5["g2e_cells"] == seen4["g2e_cells"] and seen5["load_arms"] == seen4["load_arms"]
        assert "arm_seeds" not in j5 and "n_seeds" not in j5 and j5["outcome"] == "DOMINANT-T2"
    w5 = TQ.world5(tmp_path, monkeypatch, name="w5")
    go5b, _ = wire(RD, w5, tmp_path, monkeypatch)
    go4b, _ = wire(RD4, w5, tmp_path, monkeypatch)
    _, o5 = go5b(TQ.arm5_args(w5, tmp_path), out="q5")
    _, o4 = go4b(TQ.arm5_args(w5, tmp_path), out="q4")
    j5, m5 = read(o5)
    j4, m4 = read(o4)
    assert {**j5, "script_sha256": None, "argv": None} == {**j4, "script_sha256": None, "argv": None} and md_without_script_and_arm_lines(m5) == md_without_script_and_arm_lines(m4) and j5["outcome"] == "DOMINANT-T2"


def test_the_fourth_pin_is_untouched_and_the_fifth_is_a_separate_file():
    assert hashlib.sha256(open(f"{T2V8F}/code/read_t2_dominance.py", "rb").read()).hexdigest() == PIN4_SHA
    assert open(f"{T2V8F}/code/read_t2_dominance.py.sha256.txt", encoding="utf-8").read().splitlines()[0] == PIN4_SHA
    assert os.path.abspath(RD.__file__).replace("\\", "/") == f"{T2V8F}/pin5/read_t2_dominance_pin5.py" and RD.HERE == f"{T2V8F}/code" and RD.T2V8F == T2V8F
    assert b"\r" not in open(RD.__file__, "rb").read()
