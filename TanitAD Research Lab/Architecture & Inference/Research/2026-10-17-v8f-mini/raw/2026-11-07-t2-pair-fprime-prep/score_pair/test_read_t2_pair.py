"""Tests of `read_t2_pair.py` (PREREG_T2_ATTRIB's arm-against-arm read). CPU only. Two groups:

  * FAST, no data: the pin is refused on a sha mismatch, the arm options are refused as the pinned read refuses them, `merge_rscn` and `sources_of` on synthetic dictionaries.
  * DATA, on the stage 2/3/4 pulls of the read of record (one gated context for the module): the identity (C0C - P10 reproduces the read of record's 19 rows at both seeds EXACTLY), the pinned
    `arrays_for_seed` equals `pair_arrays`, antisymmetry, the null pair, the union of exclusions leaving every arm, additivity of a pair of two seeded arms, the seed floor equals the card's seed gap,
    and the deliberate-regression arms (each must make the identity check go RED, or the pairing guard refuse).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -p no:cacheprovider test_read_t2_pair.py        (the DATA group reads ~100 MB of rollouts; it needs the pulls on C:)"""
from __future__ import annotations

import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import read_t2_pair as P  # noqa: E402

D = P.D
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
RECORD = f"{T}/stage4/score/DOMINANCE_T2.json"
PULLS = "C:/Users/Admin/t2v8f_pulls"
HAVE_DATA = os.path.isdir(f"{PULLS}/stage4/pull_box2/pull4/rollouts") and os.path.isfile(RECORD)
data = pytest.mark.skipif(not HAVE_DATA, reason="the stage 2/3/4 pulls of the read of record are not on this machine")


def record_argv(a_arm="C0C", b_arm="P10", extra=(), out="unused"):
    """The read of record's own inputs (audit/read_record_args.sh, literal E:/ and C:/ paths) plus the pair options."""
    return ["--root2", f"{T}/stage2/pull_box2_run2/pull2/rollouts", "--root3", f"{PULLS}/stage3/pull_box2/pull3/rollouts", "--root4", f"{PULLS}/stage4/pull_box2/pull4/rollouts",
            "--sources2", f"{T}/stage2/SCENE_SOURCES.json", "--sources3", f"{T}/stage3/SCENE_SOURCES3.json", "--g1", f"{T}/C0C_ID_RESULT.json",
            "--result1", f"{T}/stage1/pull_box2/pull/STAGE1_RESULT.json", "--manifest1", f"{T}/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json",
            "--result2", f"{T}/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json", "--result3", f"{PULLS}/stage3/pull_box2/pull3/STAGE3_RESULT.json", "--result4", f"{PULLS}/stage4/pull_box2/pull4/STAGE4_RESULT.json",
            "--exclusions2", f"{T}/stage2/EXCLUSIONS2.json", "--exclusions3", f"{T}/stage3/EXCLUSIONS3.json", "--exclusions4", f"{T}/stage4/EXCLUSIONS4.json",
            "--out-dir", out, "--a-arm", a_arm, "--b-arm", b_arm, *extra]


def parsed(a_arm="C0C", b_arm="P10", extra=()):
    a = P.make_parser().parse_args(record_argv(a_arm, b_arm, extra))
    a.argv = []
    return a


# ============================================================================================================================ FAST: no data
def test_the_pinned_read_is_the_one_loaded_and_a_wrong_sha_is_refused():
    assert os.path.abspath(D.__file__).replace("\\", "/") == P.PIN4 and P.sha256_file(P.PIN4) == P.PIN4_SHA == "7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9"
    with pytest.raises(SystemExit, match="REFUSED: the pinned dominance read"):
        P.load_pin(P.PIN4, "0" * 64, name="read_t2_dominance_wrong_sha")
    with pytest.raises(SystemExit, match="REFUSED: the pinned dominance read"):
        P.load_pin(os.path.abspath(__file__), P.PIN4_SHA, name="read_t2_dominance_other_file")


def test_the_main_function_is_not_run_on_import_and_the_functions_are_the_pinned_ones():
    assert D.__name__ == "read_t2_dominance_pin4" and callable(D.read_blocks) and callable(D.rscn_check) and callable(D.g3_check) and callable(D.arrays_for_seed)
    assert D.__name__ != "__main__" and callable(D.main)                       # `main()` is behind `__name__ == "__main__"` (line 1014): importing it ran no read


def test_the_frozen_card_is_intact():
    assert P.attrib_frozen_ok()                                                 # a dated log entry may have been appended below the freeze; the frozen prefix must still hold


def ns(**kw):
    base = {f"{s}_{k}": None for s in P.SIDES for k in P.STAGE_FLAGS}
    base.update({"a_arm": "C0C", "b_arm": "P10"})
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_arm_options_are_refused_as_the_pinned_read_refuses_them():
    assert P.arm_spec_of(ns(), "a")["arm"] == "C0C" and P.arm_spec_of(ns(), "a")["stage"] == 4 and P.arm_spec_of(ns(), "b")["floor"] is True
    with pytest.raises(SystemExit, match="FLOOR_EXTRA_INPUTS"):
        P.arm_spec_of(ns(b_root="x"), "b")
    with pytest.raises(SystemExit, match="FLOOR_EXTRA_INPUTS"):
        P.arm_spec_of(ns(b_stage=3), "b")
    with pytest.raises(SystemExit, match="ARM_STAGE4_IS_C0C"):
        P.arm_spec_of(ns(a_arm="MC"), "a")
    with pytest.raises(SystemExit, match="ARM_ROOT_REQUIRED"):
        P.arm_spec_of(ns(a_arm="MC", a_stage=5), "a")
    with pytest.raises(SystemExit, match="ARM_NAME_SEED_SUFFIX"):
        P.arm_spec_of(ns(a_arm="MC_s0", a_stage=5, a_root="r"), "a")
    with pytest.raises(SystemExit, match="ARM_NAME_RESERVED"):
        P.arm_spec_of(ns(a_arm="ZERO", a_stage=5, a_root="r"), "a")
    with pytest.raises(SystemExit, match="ARM_STAGE4_EXTRA_INPUTS"):
        P.arm_spec_of(ns(a_arm="C0C", a_root="r"), "a")
    sp = P.arm_spec_of(ns(a_arm="MC", a_stage=5, a_root="r5", a_result="res5", a_exclusions="ex5", a_overlay="ov5", a_gate_log="log5"), "a")
    assert sp == {"arm": "MC", "stage": 5, "root": "r5", "result": "res5", "exclusions": "ex5", "overlay": "ov5", "gate_log": "log5", "floor": False}


def test_sources_one_per_distinct_arm_two_arms_may_share_a_stage_number():
    a = types.SimpleNamespace(root2="r2", root3="r3", root4="r4", result2=None, result3=None, result4=None, exclusions2="e2", exclusions3="e3", exclusions4="e4")
    mc = P.arm_spec_of(ns(a_arm="MC", a_stage=5, a_root="/p/stage5_MC/pull5_MC/rollouts", a_exclusions="exMC"), "a")
    bpc = P.arm_spec_of(ns(b_arm="BPC", b_stage=5, b_root="/p/stage5_BPC/pull5_BPC/rollouts", b_exclusions="exBPC"), "b")
    srcs = P.sources_of(a, [bpc, mc])                       # BPC - MC
    assert [s["spec"]["arm"] for s in srcs] == ["BPC", "MC"]
    assert srcs[0]["roots"][5].endswith("pull5_BPC/rollouts") and srcs[1]["roots"][5].endswith("pull5_MC/rollouts")
    assert srcs[0]["flags"][5] == "exBPC" and srcs[1]["flags"][5] == "exMC" and srcs[0]["roots"][4] == "r4"
    assert srcs[1]["results"][5] == "/p/stage5_MC/pull5_MC/STAGE5_RESULT.json"          # the pinned default: beside the root
    c0c = P.arm_spec_of(ns(), "a")
    floor = P.arm_spec_of(ns(), "b")
    assert [s["spec"]["arm"] for s in P.sources_of(a, [mc, c0c])] == ["MC", "C0C"]      # MC - C0C: two sources, stage 4 is C0C's
    assert [s["spec"]["arm"] for s in P.sources_of(a, [floor, floor])] == ["C0C"]       # P10 - P10: the pinned default source loads the floor
    assert len(P.sources_of(a, [mc, mc])) == 1                                          # MC - MC: one source


def rs_part(excluded, problems=()):
    """a synthetic pinned `rscn_check` result: excluded = {scene8: (idx, {stage: reason}, starts_lost)}."""
    n_l, n_g = D.EXPECT["scenes_launch"], D.EXPECT["scenes_guard"]
    union = {sc: {"scene": sc + "-full", "idx": idx, "stages": dict(st), "starts_lost": lost} for sc, (idx, st, lost) in excluded.items()}
    ll = sum(u["starts_lost"] for u in union.values() if u["idx"] < n_l)
    lg = sum(u["starts_lost"] for u in union.values() if u["idx"] >= n_l)
    return {"applied": True, "rule": "R", "problems": list(problems), "per_stage": {"2": {"s": 2}, "3": {"s": 3}, "4": {"s": 4}}, "union": union,
            "starts": {"expected_before": {"launch": 81, "guard": 40, "total": 121}, "lost_launch": ll, "lost_guard": lg, "lost_total": ll + lg,
                       "read_on": {"launch": 81 - ll, "guard": 40 - lg, "total": 121 - ll - lg}}, "scenes": {"of": n_l + n_g, "excluded": len(union), "read_on": n_l + n_g - len(union)}}


def test_merge_rscn_one_part_is_the_pinned_object_itself():
    rs = rs_part({"aaaaaaaa": (3, {"2": "ACTOR_REFUSED"}, 2)})
    assert P.merge_rscn([("C0C", 4, rs)]) is rs


def test_merge_rscn_unions_the_exclusions_of_every_stage_read():
    p1 = rs_part({"aaaaaaaa": (3, {"2": "ACTOR_REFUSED", "5": "NO_DYNAMIC_LAYER"}, 2), "bbbbbbbb": (50, {"3": "ACTOR_REFUSED"}, 2)})
    p2 = rs_part({"aaaaaaaa": (3, {"2": "ACTOR_REFUSED"}, 2), "bbbbbbbb": (50, {"3": "ACTOR_REFUSED"}, 2), "cccccccc": (10, {"5": "ACTOR_REFUSED"}, 3)})
    p1["per_stage"]["5"], p2["per_stage"]["5"] = {"s": "MC"}, {"s": "BPC"}
    m = P.merge_rscn([("MC", 5, p1), ("BPC", 5, p2)])
    assert sorted(m["union"]) == ["aaaaaaaa", "bbbbbbbb", "cccccccc"]                    # a scene excluded in ANY part leaves every arm
    assert m["union"]["aaaaaaaa"]["stages"] == {"2": "ACTOR_REFUSED", "5:MC": "NO_DYNAMIC_LAYER"}
    assert m["union"]["cccccccc"]["stages"] == {"5:BPC": "ACTOR_REFUSED"}
    assert sorted(m["per_stage"]) == ["2", "3", "4", "5:BPC", "5:MC"]
    assert m["starts"]["lost_launch"] == 2 + 3 and m["starts"]["lost_guard"] == 2 and m["starts"]["lost_total"] == 7 and m["starts"]["read_on"] == {"launch": 76, "guard": 38, "total": 114}
    assert m["scenes"] == {"of": 62, "excluded": 3, "read_on": 59} and m["problems"] == []
    assert D.rscn_labels(m)["aaaaaaaa"] == "ACTOR_REFUSED (stage 2), NO_DYNAMIC_LAYER (stage 5:MC)"


def test_merge_rscn_keeps_a_problem_of_either_part_and_names_disagreeing_starts():
    p1 = rs_part({"aaaaaaaa": (3, {"2": "ACTOR_REFUSED"}, 2)}, problems=["RSCN_X one"])
    p2 = rs_part({"aaaaaaaa": (3, {"2": "ACTOR_REFUSED"}, 4)}, problems=["RSCN_X one", "RSCN_Y two"])
    m = P.merge_rscn([("MC", 5, p1), ("BPC", 5, p2)])
    assert m["problems"][:2] == ["RSCN_X one", "RSCN_Y two"] and any(x.startswith("RSCN_STARTS_DISAGREE") for x in m["problems"])


def test_build_context_glue_for_two_stage5_arms_with_the_pinned_functions_stubbed(tmp_path, monkeypatch):
    """BPC - MC: both arms in stage 5 with different pulls. The pinned gates are stubbed (their own tests are the pinned read's); what is checked is THIS file's glue: one source per arm, the R-SCN union over
    BOTH stage-5 reads handed to each G3 call, each G3 call reading its own stage-5 root, the records of every arm kept, the floor read through both sources and compared, the scene logs loaded off the union."""
    n_l = D.EXPECT["scenes_launch"]
    for n in (2, 3):
        (tmp_path / f"sources{n}.json").write_text(json.dumps([{"rel": "extracted/sequence_tracks.json", "idx": i, "scene": f"{i:08d}-scene", "source": f"src{i}"} for i in (range(0, n_l) if n == 2 else range(n_l, 62))]))
    a = types.SimpleNamespace(root2="r2", root3="r3", root4="r4", sources2=str(tmp_path / "sources2.json"), sources3=str(tmp_path / "sources3.json"), g1="g1", result1=None, result2=None, result3=None,
                              result4=None, manifest1=None, exclusions2="e2", exclusions3="e3", exclusions4="e4", **{f"{s}_{k}": None for s in P.SIDES for k in P.STAGE_FLAGS})
    a.a_arm, a.a_stage, a.a_root, a.a_exclusions, a.a_overlay, a.a_gate_log = "BPC", 5, "/p/BPC/rollouts", "exBPC", "ovBPC", "logBPC"
    a.b_arm, a.b_stage, a.b_root, a.b_exclusions, a.b_overlay, a.b_gate_log = "MC", 5, "/p/MC/rollouts", "exMC", "ovMC", "logMC"
    L = types.SimpleNamespace(pair_keys=lambda recs: [(int(r["scene"]), int(r["f0"])) for r in recs])
    monkeypatch.setattr(D, "STATE", {"L": L, "S": None, "rr": None})
    seen = {"g3": [], "g2": [], "scenes": None}
    parts = {"BPC": rs_part({"00000003": (3, {"2": "ACTOR_REFUSED"}, 2), "00000010": (10, {"5": "ACTOR_REFUSED"}, 3)}),
             "MC": rs_part({"00000003": (3, {"2": "ACTOR_REFUSED"}, 2), "00000050": (50, {"5": "NO_DYNAMIC_LAYER"}, 2)})}
    monkeypatch.setattr(D, "init", lambda *x, **k: None)
    monkeypatch.setattr(D, "g4_check", lambda *x, **k: {"PASS": True, "why": []})
    monkeypatch.setattr(D, "g1_check", lambda *x, **k: {"PASS": True, "why": []})
    monkeypatch.setattr(D, "card_frozen_ok", lambda *x, **k: True)
    monkeypatch.setattr(D, "arm_absent", lambda *x, **k: [])
    monkeypatch.setattr(D, "rscn_check", lambda roots, results, sids, flags, expect, arm_name, arm_stage: {**parts[arm_name], "per_stage": {**parts[arm_name]["per_stage"], str(arm_stage): {"s": arm_name}}})
    monkeypatch.setattr(D, "g2_full", lambda ns, results, excluded=None: seen["g2"].append((ns.arm, ns.arm_root, ns.arm_overlay, ns.arm_gate_log, sorted(results), excluded)) or {"PASS": True, "why": []})

    def g3(roots, results, sids, expect, arm_name, arm_stage, exclusions=None, rscn=None):
        seen["g3"].append((arm_name, roots[arm_stage], rscn))
        fl = [{"scene": 1, "f0": 0}, {"scene": 2, "f0": 7}]
        return {"PASS": True, "why": []}, {f"{arm_name}_s0": [{"scene": 1, "f0": 0, "x": arm_name}], f"{arm_name}_s1": [{"scene": 1, "f0": 0, "x": arm_name}], "P10": fl}
    monkeypatch.setattr(D, "g3_check", g3)
    monkeypatch.setattr(D, "scenes_from", lambda recs, srcs, roots, skip=(): seen.update(scenes=(list(roots), sorted(skip), srcs)) or {})
    ctx = P.build_context(a)
    assert ctx["ok"] and sorted(ctx["gates"]) == ["ATTRIB", "G1", "G2[BPC@5]", "G2[MC@5]", "G3[BPC@5]", "G3[MC@5]", "G4"]
    assert [g[0] for g in seen["g3"]] == ["BPC", "MC"] and [g[1] for g in seen["g3"]] == ["/p/BPC/rollouts", "/p/MC/rollouts"]
    assert seen["g3"][0][2] is seen["g3"][1][2] is ctx["rscn"]                                       # BOTH G3 calls read the ONE merged union
    assert sorted(ctx["rscn"]["union"]) == ["00000003", "00000010", "00000050"] and ctx["rscn"]["starts"]["lost_total"] == 7
    assert [g[:4] for g in seen["g2"]] == [("BPC", "/p/BPC/rollouts", "ovBPC", "logBPC"), ("MC", "/p/MC/rollouts", "ovMC", "logMC")] and seen["g2"][0][5]["00000010"] == "ACTOR_REFUSED (stage 5:BPC)"
    assert sorted(ctx["recs"]) == ["BPC_s0", "BPC_s1", "MC_s0", "MC_s1", "P10"]
    assert seen["scenes"][0] == [2, 3, 4, "5:BPC", "5:MC"] and seen["scenes"][1] == [3, 10, 50]       # the roots in the pinned order; the scenes of the union skipped
    assert P.recs_of(ctx["recs"], "BPC", 1)[0]["x"] == "BPC" and P.recs_of(ctx["recs"], "MC", 0)[0]["x"] == "MC" and P.recs_of(ctx["recs"], "P10", 1) is ctx["recs"]["P10"]
    seen["g3"].clear()
    parts_floor = P.build_context(types.SimpleNamespace(**{**vars(a), "a_arm": "MC", "b_arm": "P10", "a_stage": 5, "a_root": "/p/MC/rollouts", "a_exclusions": "exMC", "a_overlay": "ovMC", "a_gate_log": "logMC",
                                                          "b_stage": None, "b_root": None, "b_exclusions": None, "b_overlay": None, "b_gate_log": None}))
    assert parts_floor["ok"] and sorted(parts_floor["gates"]) == ["ATTRIB", "G1", "G2", "G3", "G4"] and [g[0] for g in seen["g3"]] == ["MC"]


def test_a_floor_that_differs_between_two_sources_fails_the_pair(tmp_path, monkeypatch):
    n_l = D.EXPECT["scenes_launch"]
    (tmp_path / "s.json").write_text(json.dumps([{"rel": "extracted/sequence_tracks.json", "idx": i, "scene": f"{i:08d}-x", "source": "s"} for i in range(62)]))
    a = types.SimpleNamespace(root2="r2", root3="r3", root4="r4", sources2=str(tmp_path / "s.json"), sources3=None, g1="g1", result1=None, result2=None, result3=None, result4=None, manifest1=None,
                              exclusions2="e2", exclusions3="e3", exclusions4="e4", **{f"{s}_{k}": None for s in P.SIDES for k in P.STAGE_FLAGS})
    a.a_arm, a.a_stage, a.a_root, a.a_exclusions, a.a_overlay, a.a_gate_log = "MC", 5, "/p/MC/rollouts", "exMC", "ov", "lg"
    a.b_arm = "C0C"
    L = types.SimpleNamespace(pair_keys=lambda recs: [(int(r["scene"]), int(r["f0"])) for r in recs])
    monkeypatch.setattr(D, "STATE", {"L": L, "S": None, "rr": None})
    for name, f in (("init", lambda *x, **k: None), ("g4_check", lambda *x, **k: {"PASS": True, "why": []}), ("g1_check", lambda *x, **k: {"PASS": True, "why": []}), ("card_frozen_ok", lambda *x, **k: True),
                    ("arm_absent", lambda *x, **k: []), ("g2_full", lambda *x, **k: {"PASS": True, "why": []}), ("scenes_from", lambda *x, **k: {}),
                    ("rscn_check", lambda roots, results, sids, flags, expect, arm_name, arm_stage: rs_part({}))):
        monkeypatch.setattr(D, name, f)
    monkeypatch.setattr(D, "g3_check", lambda roots, results, sids, expect, arm_name, arm_stage, exclusions=None, rscn=None: (
        {"PASS": True, "why": []}, {f"{arm_name}_s0": [{"scene": 1, "f0": 0}], f"{arm_name}_s1": [{"scene": 1, "f0": 0}], "P10": [{"scene": 1, "f0": 0 if arm_name == "MC" else 1}]}))
    ctx = P.build_context(a)
    assert not ctx["ok"] and "FLOOR_RECORDS_DIFFER" in ctx["gates"]["FLOOR"]["why"][0] and n_l == 42


# ============================================================================================================================ DATA
@pytest.fixture(scope="module")
def world():
    a = parsed()
    ctx = P.build_context(a)
    return types.SimpleNamespace(a=a, ctx=ctx, record=json.load(open(RECORD, encoding="utf-8")), cache={})


def blocks(w, A, B):
    """cached read_pair of the module's one gated context."""
    k = (A, B)
    if k not in w.cache:
        w.cache[k] = P.read_pair(w.ctx, A, B)
    return w.cache[k]


def rows_equal_record(w, bl):
    rec = w.record["by_seed"]
    return all(P.rt(bl[s]["rows"]) == rec[f"s{s}"]["rows"] for s in P.SEEDS)


@data
def test_the_gates_of_the_read_of_record_pass_through_the_pair_context(world):
    c = world.ctx
    assert c["ok"], {k: g["why"][:2] for k, g in c["gates"].items() if not g["PASS"]}
    assert sorted(c["gates"]) == ["ATTRIB", "G1", "G2", "G3", "G4"] and all(g["PASS"] for g in c["gates"].values())
    assert P.rt(c["rscn"]["scenes"]) == world.record["r_scn"]["scenes"] and P.rt(c["rscn"]["starts"]) == world.record["r_scn"]["starts"]
    assert [s["spec"]["arm"] for s in c["sources"]] == ["C0C"]


@data
def test_identity_c0c_minus_p10_is_the_read_of_record_exactly(world):
    res = P.run_pair(world.a, world.ctx)
    ident = P.identity_report(res, RECORD)
    assert ident["identical"], ident["why"][:5]
    assert ident["max_abs_diff"] == 0.0 and ident["rows_compared"] == 38 and ident["scenes_and_starts_equal"]
    assert res["pair"] == "C0C - P10" and res["outcome"] == "READ" and len(res["by_seed"]["s0"]["rows"]) == 19 == len(res["by_seed"]["s1"]["rows"])
    # the per-arm block of C0C against P10 is the pair itself; its seed floor is the card's seed gap
    assert all(res["arm_vs_P10"]["C0C"][f"s{s}"][lab] == {k: res["by_seed"][f"s{s}"]["rows"][lab][k] for k in P.FIELDS} for s in P.SEEDS for lab in res["by_seed"]["s0"]["rows"])
    gap = world.record["seed_gap_beside"]
    assert all(P.rt(res["seed_floor"]["a"][lab]["gap"]) == gap[lab]["gap"] for lab in gap) and res["seed_floor"]["b"] is None
    world.cache[("res", "C0C", "P10")] = res


@data
def test_every_record_is_off_the_excluded_scenes_for_both_arms(world):
    c = world.ctx
    union = {u["idx"] for u in c["rscn"]["union"].values()}
    assert len(union) == 14 and c["rscn"]["starts"]["read_on"]["total"] == 93
    for k, recs in c["recs"].items():
        assert len(recs) == 93 and not ({int(r["scene"]) for r in recs} & union), k
    assert set(c["scenes"]) and not (set(c["scenes"]) & union)                    # the scene logs of an excluded scene were never loaded


@data
def test_pair_arrays_equals_the_pinned_arrays_for_seed_bit_for_bit(world):
    c = world.ctx
    for s in P.SEEDS:
        z_pinned = D.arrays_for_seed(c["recs"][f"C0C_s{s}"], c["recs"]["P10"], c["scenes"], s)
        z_pair = P.pair_arrays(c["recs"][f"C0C_s{s}"], c["recs"]["P10"], c["scenes"], P.label_of("C0C", s), P.label_of("P10", s))
        assert sorted(z_pinned) == sorted(z_pair)
        for k in z_pinned:
            assert np.array_equal(np.asarray(z_pinned[k]), np.asarray(z_pair[k]), equal_nan=True) if np.asarray(z_pinned[k]).dtype.kind != "U" else list(z_pinned[k]) == list(z_pair[k]), (s, k)


@data
def test_the_label_handed_to_the_row_builder_moves_no_row(world):
    c = world.ctx
    z1 = P.pair_arrays(c["recs"]["C0C_s0"], c["recs"]["P10"], c["scenes"], "C0C_s0|T2|objects", "P10|T2|objects")
    z2 = P.pair_arrays(c["recs"]["C0C_s0"], c["recs"]["P10"], c["scenes"], "SOMETHING_ELSE|T2|objects", "A_THIRD_NAME|T2|objects")
    for k in z1:
        if np.asarray(z1[k]).dtype.kind != "U":
            assert np.array_equal(z1[k], z2[k], equal_nan=True), k


@data
def test_antisymmetry_p10_minus_c0c_is_minus_c0c_minus_p10(world):
    fwd, rev = blocks(world, "C0C", "P10"), blocks(world, "P10", "C0C")
    worst = 0.0
    for s in P.SEEDS:
        for lab, f in fwd[s]["rows"].items():
            r = rev[s]["rows"][lab]
            assert r["delta"] == -f["delta"], (s, lab)                                       # the point is exactly negated
            assert r["separated"] == f["separated"], (s, lab)
            worst = max(worst, abs(r["lo"] + f["hi"]), abs(r["hi"] + f["lo"]))               # the interval is mirrored: the SAME resamples (seed 0), negated
            assert fwd[s]["counts"][lab] == rev[s]["counts"][lab]
    assert worst <= 1e-12, worst
    json.dump({"antisymmetry_max_abs_interval_mirror_deviation": worst, "point_exactly_negated": True, "intervals": "same resamples (seed 0, B 2000) with the sign flipped"},
              open(f"{HERE}/logs/antisymmetry.json", "w", encoding="utf-8", newline="\n"), indent=1)


@data
def test_a_pair_of_the_same_arm_is_exactly_zero_with_a_zero_interval(world):
    for arm in ("C0C", "P10"):
        bl = blocks(world, arm, arm)
        for s in P.SEEDS:
            for lab, r in bl[s]["rows"].items():
                assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["separated"] is False, (arm, s, lab, r)


@data
def test_two_seeded_arms_pair_through_pair_arrays_and_add_up(world):
    """C0C_s0 - C0C_s1 takes the generic path (neither side is the floor). Where every window is finite the point is additive: (s0 - s1) = (s0 - P10) - (s1 - P10)."""
    c = world.ctx
    z = P.pair_arrays(c["recs"]["C0C_s0"], c["recs"]["C0C_s1"], c["scenes"], P.label_of("C0C", 0), P.label_of("C0C", 1))
    bl = D.read_blocks({0: z})[0]
    cnt = P.window_counts(z)
    s0, s1 = blocks(world, "C0C", "P10")[0]["rows"], blocks(world, "C0C", "P10")[1]["rows"]
    n_all = len(c["recs"]["C0C_s0"])
    full = [lab for lab, v in cnt.items() if v["n_windows"] == n_all]
    assert {"ADE6", "heading MAE", "curvature MAE"} <= set(full)
    for lab in full:
        assert abs(bl["rows"][lab]["delta"] - (s0[lab]["delta"] - s1[lab]["delta"])) <= 1e-12, lab
    assert all(r["delta"] != 0.0 for lab, r in bl["rows"].items() if lab in ("ADE6",))


@data
def test_the_seed_floor_is_the_cards_seed_gap(world):
    fl = P.seed_floor(blocks(world, "C0C", "P10"))
    gap = world.record["seed_gap_beside"]
    assert sorted(fl) == sorted(gap) and all(P.rt(fl[lab]["gap"]) == gap[lab]["gap"] for lab in gap)
    assert all(fl[lab]["gap"] == abs(fl[lab]["delta_s1"] - fl[lab]["delta_s0"]) for lab in fl)


@data
def test_a_mispaired_set_of_starts_is_refused_by_the_pairing_guard(world):
    c = world.ctx
    recs = dict(c["recs"])
    recs["P10"] = list(recs["P10"][1:]) + list(recs["P10"][:1])                               # P10's starts rolled by one: same set, other order
    with pytest.raises(ValueError, match="PAIRING GUARD"):
        P.read_pair({**c, "recs": recs}, "C0C", "P10")


# ------------------------------------------------------------------------------------------------ the deliberate-regression arms: each must go RED
def unpaired_decompose(orig):
    """MUTANT: the paired bootstrap replaced by an UNPAIRED one -- A's scenes and B's scenes are resampled independently. The point is unchanged; the interval is not the paired one."""
    def dec(a, b, eid, cls, pool="masked", n_boot=2000, seed=0):
        r = orig(a, b, eid, cls, pool, n_boot, seed)
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        ep = np.asarray([str(x) for x in eid])
        uniq = np.unique(ep[ok] if pool == "masked" else ep)
        sa, sb, na, nb = (np.array([x[ep[ok] == u].sum() for u in uniq]) for x in (a[ok], b[ok], np.ones(ok.sum()), np.ones(ok.sum())))
        rng = np.random.default_rng(seed)
        ia, ib = rng.integers(0, len(uniq), (n_boot, len(uniq))), rng.integers(0, len(uniq), (n_boot, len(uniq)))
        with np.errstate(all="ignore"):
            d = sa[ia].sum(1) / na[ia].sum(1) - sb[ib].sum(1) / nb[ib].sum(1)
        r = dict(r)
        r["lo"], r["hi"] = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
        return r
    return dec


def red(w, ctx=None, hooks=None, blocks_hook=None):
    """run the pair C0C - P10 through the mutated pipeline and compare with the read of record -> identity_report."""
    res = P.run_pair(w.a, ctx or w.ctx, hooks)
    return P.identity_report(res, RECORD), res


@data
def test_mutant_unpaired_resampling_goes_red(world, monkeypatch):
    GD = D.STATE["rr"].GD
    monkeypatch.setattr(GD, "decompose", unpaired_decompose(GD.decompose))
    ident, res = red(world)
    assert res["outcome"] == "READ" and not ident["identical"] and any(" lo: " in w_ or " hi: " in w_ for w_ in ident["why"]), ident["why"][:3]
    assert ident["max_abs_diff"] > 0.0
    json.dump({"mutant": "unpaired resampling", "identical": ident["identical"], "max_abs_diff": ident["max_abs_diff"], "first": ident["why"][:3]}, open(f"{HERE}/logs/red_unpaired.json", "w", encoding="utf-8", newline="\n"), indent=1)


@data
def test_mutant_sign_flip_goes_red(world):
    def swap(s, z):
        for i in range(19):
            z[f"a_{i}"], z[f"b_{i}"] = z[f"b_{i}"], z[f"a_{i}"]
    ident, res = red(world, hooks={"arrays": swap})
    assert not ident["identical"] and any("delta" in w_ for w_ in ident["why"])


@data
def test_mutant_seeds_swapped_goes_red(world):
    c = world.ctx
    recs = dict(c["recs"])
    recs["C0C_s0"], recs["C0C_s1"] = c["recs"]["C0C_s1"], c["recs"]["C0C_s0"]
    ident, _ = red(world, ctx={**c, "recs": recs})
    assert not ident["identical"]


@data
def test_mutant_dropped_exclusion_goes_red_by_a_gate_refusal(world):
    def drop_one(rs):
        sc = sorted(rs["union"])[0]
        u = rs["union"].pop(sc)
        rs["scenes"]["excluded"] -= 1
        rs["scenes"]["read_on"] += 1
        rs["starts"]["lost_launch" if u["idx"] < D.EXPECT["scenes_launch"] else "lost_guard"] -= u["starts_lost"]
    a = parsed()
    ctx = P.build_context(a, hooks={"rscn": drop_one})
    assert not ctx["ok"] and not ctx["gates"]["G3"]["PASS"]
    res = P.run_pair(a, ctx)
    ident = P.identity_report(res, RECORD)
    assert res["outcome"] == "GATE-FAIL" and "by_seed" not in res and not ident["identical"] and ident["why"][0].startswith("IDENTITY_NO_ROWS")
    json.dump({"mutant": "dropped exclusion (one excluded scene removed from the union before G3)", "outcome": res["outcome"], "g3_first_problems": ctx["gates"]["G3"]["why"][:3]},
              open(f"{HERE}/logs/red_dropped_exclusion.json", "w", encoding="utf-8", newline="\n"), indent=1)


@data
def test_mutant_exclusion_applied_to_one_arm_only_is_refused_by_the_pairing_guard(world):
    c = world.ctx
    # put back, into C0C only, a record that the union removed from both arms: a start of a scene P10 does not hold
    r0 = dict(c["recs"]["C0C_s0"][0])
    r0["scene"] = int(r0["scene"]) + 1000
    recs = dict(c["recs"])
    recs["C0C_s0"] = sorted(list(c["recs"]["C0C_s0"]) + [r0], key=lambda r: (int(r["scene"]), int(r["f0"])))
    with pytest.raises(ValueError, match="PAIRING GUARD"):
        P.read_pair({**c, "recs": recs}, "C0C", "P10")


@data
def test_the_control_of_the_red_arms_is_green(world):
    """the unmutated pipeline passes the same check the mutants fail (the monkeypatch of the unpaired mutant is undone)."""
    ident, res = red(world)
    assert ident["identical"] and res["outcome"] == "READ"
