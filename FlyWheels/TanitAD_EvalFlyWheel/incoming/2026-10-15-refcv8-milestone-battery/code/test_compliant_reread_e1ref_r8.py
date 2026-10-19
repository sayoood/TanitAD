"""Pins for D-R8-E1-REF in compliant_reread_r8.py (PI 2026-10-19: BAR-R8-E1 / E3 are scored against ha0_ext_pose; the ha0_ext read stays beside).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_compliant_reread_e1ref_r8.py

EVERY expectation is a LITERAL (from the banked JSON of raw/a0k0_headroom/A0K0_HEADROOM.json and raw/compliant_reread/trees_pre_e1ref/), never an
expression over the code under test.  Three groups:
  1. pure functions on synthetic data: the output schema, which cell each bar reads, the pose-vs-steer identity of the reference (analytic), the control aggregator;
  2. the DELIBERATE REGRESSION (control c): swapping the pose prior back to ha0_ext turns the known-value control RED -- at function level on the banked
     numbers, and on the real mutation-arm artifact raw/compliant_reread/mutation/step30000ema_unc.json;
  3. the real artifacts: (a) the uncapped compliant S2 `os - ha0_ext_pose` of EMA-30k equals A0K0_HEADROOM.json exactly; (b) every ha0_ext cell of every tree equals
     the pre-change JSON byte for byte; every re-read tree carries the new schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compliant_reread_r8 as CR  # noqa: E402
import pair_read_r8 as PR  # noqa: E402

PKG = HERE.parent
TREES = PKG / "raw" / "compliant_reread" / "trees"
PRE = PKG / "raw" / "compliant_reread" / "trees_pre_e1ref"
MUT = PKG / "raw" / "compliant_reread" / "mutation" / "step30000ema_unc.json"
A0K0 = PKG / "raw" / "a0k0_headroom" / "A0K0_HEADROOM.json"

#: LITERALS (MEASURED, raw/a0k0_headroom/A0K0_HEADROOM.json, step30000ema_unc, S2, compliant set, uncapped os - ha0_ext_pose, ADE 0-2 s), PI brief 2026-10-19
LIT_POSE = {0: (-0.0136, -0.0286, 0.0018), 1: (-0.0127, -0.0272, 0.0024)}
#: LITERALS (MEASURED, pre-change trees_pre_e1ref/step30000ema_unc.json == kdraw_e1_r8.LIT_K1 == e1_decomp): the REGISTERED read, os - ha0_ext
LIT_EXT = {0: (-0.0085, -0.0229, 0.0065), 1: (-0.0076, -0.0216, 0.0073)}
N_WIN, N_EP = 4159, 138
#: LITERAL (MEASURED 2026-09-26, measure_prior_vs_echo.json B_ade_0_2s): ha0_ext_pose - echo, S2 all windows
LIT_R0926 = {"delta": 0.0055, "lo": 0.0036, "hi": 0.0076, "n_windows": 4754, "n_episodes": 139}


def _cell(t, n=N_WIN, e=N_EP):
    return {"delta": t[0], "lo": t[1], "hi": t[2], "n_windows": n, "n_episodes": e, "separated": bool(t[2] < 0 or t[1] > 0)}


# ====================================================================================================== 1. pure functions
def test_known_value_check_is_exact_and_never_passes_when_it_could_not_run():
    lit = _cell(LIT_POSE[0])
    assert CR.known_value_check(_cell(LIT_POSE[0]), lit)["exact"] is True
    assert CR.known_value_check(_cell((-0.0136, -0.0286, 0.0019)), lit)["exact"] is False          # one 4th-decimal off
    assert CR.known_value_check(None, lit)["exact"] is False and CR.known_value_check(_cell(LIT_POSE[0]), None)["exact"] is False
    assert CR.known_value_check({"status": "REFUSED"}, lit)["exact"] is False


def _mk_reads(pose_pass: bool, ext_pass: bool):
    """reads[(seed, surf)] as read_masked emits them; the pose cell passes/fails independently of the ha0_ext cell."""
    good = {"delta": -0.1, "lo": -0.2, "hi": -0.05, "n_windows": 10, "n_episodes": 3}
    bad = {"delta": 0.1, "lo": -0.05, "hi": 0.2, "n_windows": 10, "n_episodes": 3}
    nr = {"delta": 0.0, "lo": -0.1, "hi": 0.1, "n_windows": 10, "n_episodes": 3}
    reads = {}
    for sd, ade in ((0, 1.00), (1, 1.01)):
        for surf in ("S2", "S6"):
            rows = {m[1]: {"os_minus_ha0_ext": nr, "os_minus_refcv7_50400": nr} for m in PR.METRICS}
            rows["ade_m"] = {"os_minus_ha0_ext": good if ext_pass else bad, "os_minus_ha0_ext_pose": good if pose_pass else bad, "os_minus_refcv7_50400": nr}
            reads[(sd, surf)] = {"means": {"os": {"ade_m": ade}}, "rows": rows}
    return reads


def test_bars_of_schema_reference_labels_and_which_cell_each_bar_reads():
    b = CR.bars_of(PR, _mk_reads(pose_pass=True, ext_pass=False))
    assert list(b)[:5] == ["inference_floor_ADE_m", "BAR-R8-E1", "BAR-R8-E3", "BAR-R8-E1_ha0_ext", "BAR-R8-E3_ha0_ext"] and "BAR-R8-NR" in b
    # the bar reads the POSE cell: pose passes, ha0_ext fails -> E1/E3 pass-class, the beside rows FAILED
    for k in ("BAR-R8-E1", "BAR-R8-E3"):
        assert b[k]["reference"] == "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)" and b[k]["reference_cell"] == "os_minus_ha0_ext_pose"
        assert b[k]["verdict"].startswith("PASS"), b[k]["verdict"]
        assert b[k + "_ha0_ext"]["verdict"] == "FAILED" and b[k + "_ha0_ext"]["reference_cell"] == "os_minus_ha0_ext"
        assert "PRE-RULING" in b[k + "_ha0_ext"]["reference"]
    c = CR.bars_of(PR, _mk_reads(pose_pass=False, ext_pass=True))                  # and the converse: the bar does NOT follow the echo
    assert c["BAR-R8-E1"]["verdict"] == "FAILED" and c["BAR-R8-E1_ha0_ext"]["verdict"].startswith("PASS")


def test_bars_of_never_falls_back_to_the_echo_when_the_pose_cell_is_absent():
    r = _mk_reads(pose_pass=True, ext_pass=True)
    for v in r.values():
        del v["rows"]["ade_m"]["os_minus_ha0_ext_pose"]
    b = CR.bars_of(PR, r)
    assert b["BAR-R8-E1"]["verdict"].startswith("NOT EVALUABLE") and b["BAR-R8-E3"]["verdict"].startswith("NOT EVALUABLE")
    assert b["BAR-R8-E1_ha0_ext"]["verdict"].startswith("PASS")


def test_read_masked_adds_the_pose_cell_on_ade_only_and_leaves_every_other_cell_untouched():
    rng = np.random.default_rng(0)
    n = 60
    eid = np.repeat(np.arange(6), 10)
    mk = np.ones(n, bool)
    mk[::7] = False
    base = {k: {m[1]: rng.normal(1.0, 0.3, n) for m in PR.METRICS} for k in ("os", "ha0_ext", "ref")}
    pose = {m[1]: rng.normal(1.0, 0.3, n) for m in PR.METRICS}
    r0 = CR.read_masked(PR, base, eid, mk)
    r1 = CR.read_masked(PR, {**base, "ha0_ext_pose": pose}, eid, mk)
    assert "os_minus_ha0_ext_pose" not in r0["rows"]["ade_m"]
    assert set(k for k, v in r1["rows"].items() if "os_minus_ha0_ext_pose" in v) == {"ade_m"}
    for k in r0["rows"]:                                    # adding the arm moves NO existing cell (byte for byte)
        for which in ("os_minus_ha0_ext", "os_minus_refcv7_50400"):
            assert json.dumps(r0["rows"][k][which]) == json.dumps(r1["rows"][k][which]), (k, which)
    want = PR.paired_cell(base["os"]["ade_m"][mk], pose["ade_m"][mk], eid[mk], CR.N_BOOT)
    assert json.dumps(r1["rows"]["ade_m"]["os_minus_ha0_ext_pose"]) == json.dumps(want)


def test_the_compliance_row_lists_the_new_echo_arm():
    assert "echo_ha0_ext_pose" in CR.ARMS and "echo_ha0_ext" in CR.ARMS                  # the old arm is never dropped
    src = (HERE / "compliant_reread_r8.py").read_text(encoding="utf-8")
    assert 'arms["echo_ha0_ext_pose"] = A["ha0_ext_pose"]' in src


def _poses(n, v, dyaw, a=0.0):
    """[n, 8, 4] (x, y, yaw, v) windows: constant yaw increment `dyaw` rad/step, speed v + a*t."""
    t = np.arange(8)
    yaw = dyaw * t
    vv = v + a * 0.1 * t
    P = np.zeros((n, 8, 4), np.float32)
    P[:, :, 2] = yaw
    P[:, :, 3] = vv
    return P


def test_the_pose_reference_reads_yaw_rate_not_the_steer_channel_analytic():
    """STRAIGHT poses with a NONZERO steer curvature in ego_t0: the pose prior must go straight (y == 0, x = v t); the echo rebuilt from steer must curve.
    A CURVED pose track with ZERO steer must curve with kappa0 = omega0 / v0 exactly (closed form)."""
    import a0k0_headroom_r8 as AK
    hz = [5, 10, 15, 20]
    n = 3
    win = _poses(n, 10.0, 0.0)
    ego_t0 = np.zeros((n, 4), np.float32)
    ego_t0[:, 0] = 10.0
    ego_t0[:, 3] = 0.05                                                      # steer-channel curvature (inadmissible)
    dec = {"win": win, "pose_last": win[:, -1], "ego_t0": ego_t0, "ws": np.arange(n)}
    HX_dummy = np.zeros((n, len(hz), 2), np.float32)
    REF, V = AK.references(dec, HX_dummy, hz)
    HP = REF["HP"]
    assert np.allclose(HP[..., 1], 0.0, atol=1e-6)                           # no lateral motion: kappa0 from the POSE (0), not from the steer (0.05)
    assert np.allclose(HP[0, :, 0], [10.0 * 0.1 * h for h in hz], atol=1e-4)  # forward Euler at 0.1 s, a0 = 0
    # curved pose track, zero steer: omega0 = 0.02 rad / 0.1 s = 0.2 rad/s, v0 = 10 -> kappa0 = 0.02
    win = _poses(n, 10.0, 0.02)
    ego_t0 = np.zeros((n, 4), np.float32)
    ego_t0[:, 0] = 10.0
    dec = {"win": win, "pose_last": win[:, -1], "ego_t0": ego_t0, "ws": np.arange(n)}
    HP = AK.references(dec, HX_dummy, hz)[0]["HP"]
    x = y = yaw = 0.0
    want = []
    for k in range(1, max(hz) + 1):                                          # rollout_unicycle's update order: position on the start-of-step state, then heading
        x += 10.0 * np.cos(yaw) * 0.1
        y += 10.0 * np.sin(yaw) * 0.1
        yaw += 10.0 * 0.02 * 0.1
        if k in hz:
            want.append((x, y))
    assert np.allclose(HP[0], np.array(want), atol=1e-4) and (HP[..., 1] > 0).all()


def _synthetic_res(tree: str, a0k0_exact: bool, r0926_exact: bool = True, ident: bool = True, cons: bool = True):
    res = {"seeds": {}}
    for s in ("0", "1"):
        res["seeds"][s] = {"pose_reference": {"S2": {"PASS": cons}, "S6": {"PASS": cons}}, "surfaces": {}}
        for surf in ("S2", "S6"):
            c4 = {"A0K0_all_window_uncapped": {"exact": a0k0_exact}, "A0K0_compliant_uncapped": {"exact": a0k0_exact}}
            if surf == "S2":
                c4["R0926_S2_all_ha0_ext_pose_minus_ha0_ext"] = {"exact": r0926_exact}
            res["seeds"][s]["surfaces"][surf] = {"C4_pose_known_values": c4}
    k = np.stack([np.arange(5), np.arange(5)])
    hp = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)
    hp1 = hp if ident else hp + 1.0
    hbs = {0: {"S2": (k, hp), "S6": (k, hp)}, 1: {"S2": (k, hp1), "S6": (k, hp1)}}
    return res, hbs


@pytest.mark.parametrize("kw,want", [({}, True), ({"a0k0_exact": False}, False), ({"r0926_exact": False}, False), ({"ident": False}, False), ({"cons": False}, False)])
def test_c4_aggregator_pass_and_every_way_to_fail(kw, want):
    args = {"a0k0_exact": True, **kw}
    res, hbs = _synthetic_res("step30000ema_unc", **args)
    assert CR.c4_pose_control("step30000ema_unc", res, hbs, {"errors": []})["PASS"] is want


def test_c4_known_value_is_not_applicable_off_the_30k_pair_and_an_unreadable_artifact_fails_it():
    res, hbs = _synthetic_res("step40000ema_unc", a0k0_exact=True)
    for s in ("0", "1"):
        for surf in ("S2", "S6"):
            for k in ("A0K0_all_window_uncapped", "A0K0_compliant_uncapped"):
                res["seeds"][s]["surfaces"][surf]["C4_pose_known_values"].pop(k)
    ok = CR.c4_pose_control("step40000ema_unc", res, hbs, {"errors": []})
    assert ok["PASS"] is True and ok["A0K0_known_value"]["applicable"] is False
    assert CR.c4_pose_control("step40000ema_unc", res, hbs, {"errors": ["A0K0_HEADROOM.json unreadable: x"]})["PASS"] is False
    res30, hbs30 = _synthetic_res("step30000ema_unc", a0k0_exact=True)
    for s in ("0", "1"):                                                       # a 30k tree whose A0K0 cells were never checked must FAIL, not pass vacuously
        for surf in ("S2", "S6"):
            res30["seeds"][s]["surfaces"][surf]["C4_pose_known_values"].pop("A0K0_compliant_uncapped")
    assert CR.c4_pose_control("step30000ema_unc", res30, hbs30, {"errors": []})["PASS"] is False


# ====================================================================================================== 2. DELIBERATE REGRESSION (control c)
@pytest.mark.parametrize("seed", [0, 1])
def test_mutation_swapping_the_pose_prior_back_to_ha0_ext_turns_the_known_value_control_red(seed):
    """The ha0_ext cell (what a pose arm swapped back to ha0_ext would produce) against the A0K0 literal: RED.  The pose literal against itself: green."""
    swapped = _cell(LIT_EXT[seed])
    lit = _cell(LIT_POSE[seed])
    r = CR.known_value_check(swapped, lit)
    assert r["exact"] is False and "delta" in r["diff"] and "lo" in r["diff"] and "hi" in r["diff"]
    assert CR.known_value_check(_cell(LIT_POSE[seed]), lit)["exact"] is True


def test_the_mutated_builder_returns_the_echo_unchanged():
    P = {"A": {"ha0_ext": np.random.default_rng(1).normal(size=(5, 4, 2)).astype(np.float32)}}
    hp, meta = CR.build_pose_reference_MUTATED_TO_HA0_EXT({}, {"panels": {"S2": P}}, "S2")
    assert np.array_equal(hp, P["A"]["ha0_ext"]) and hp is not P["A"]["ha0_ext"] and meta["MUTATED"] is True


def test_attach_pose_reference_uses_the_injected_builder_and_refuses_a_shape_mismatch():
    A = {"ha0_ext": np.zeros((4, 3, 2), np.float32)}
    ctx = {"tree": "t", "seed": 0, "panels": {"S2": {"A": A}, "S6": {"A": dict(A)}}}
    meta = CR.attach_pose_reference({}, ctx, builder=lambda tools, c, surf: (np.ones((4, 3, 2), np.float32), {"PASS": True}))
    assert set(meta) == {"S2", "S6"} and float(ctx["panels"]["S2"]["A"]["ha0_ext_pose"].min()) == 1.0
    with pytest.raises(SystemExit):
        CR.attach_pose_reference({}, ctx, builder=lambda tools, c, surf: (np.ones((4, 2, 2), np.float32), {}))


def test_the_mutation_arm_artifact_is_red_and_never_lands_in_trees():
    if not MUT.exists():
        pytest.skip("mutation-arm artifact not produced yet (read --tree step30000ema_unc --mutate-pose-to-ha0-ext)")
    j = json.loads(MUT.read_text(encoding="utf-8"))
    assert j["ruling"]["MUTATION_ARM_NOT_A_RESULT"] is True and "NOT A RESULT" in j["MUTATION_ARM"]
    c4 = j["controls"]["C4_pose_reference_valid"]
    assert c4["PASS"] is False and c4["A0K0_known_value"]["all_exact"] is False and j["controls"]["ALL_PASS"] is False
    assert j["bars"]["status"] == "WITHHELD"
    # the swapped arm reads exactly the ha0_ext literal where the A0K0 control expected the pose literal
    d = c4["A0K0_known_value"]["detail"]
    got = d["A0K0_compliant_uncapped|s0|S2"]["got"]
    assert (got["delta"], got["lo"], got["hi"]) == LIT_EXT[0] and d["A0K0_compliant_uncapped|s0|S2"]["banked"]["delta"] == LIT_POSE[0][0]
    assert d["A0K0_compliant_uncapped|s1|S2"]["exact"] is False
    # the R0926 control is RED too: ha0_ext_pose swapped for ha0_ext makes (pose - echo) identically zero
    assert c4["R0926_known_value_S2_all_ha0_ext_pose_minus_ha0_ext"]["exact_per_seed"] == {"s0": False, "s1": False}


# ====================================================================================================== 3. the real artifacts
def _trees_pre():
    return sorted(p.stem for p in PRE.glob("step*.json")) if PRE.exists() else []


@pytest.fixture(scope="module")
def ema30k():
    p = TREES / "step30000ema_unc.json"
    if not p.exists():
        pytest.skip("trees/step30000ema_unc.json absent")
    return json.loads(p.read_text(encoding="utf-8"))


def test_a0k0_banked_json_carries_the_literals():
    if not A0K0.exists():
        pytest.skip("A0K0_HEADROOM.json absent")
    j = json.loads(A0K0.read_text(encoding="utf-8"))
    for s in (0, 1):
        c = j["trees"]["step30000ema_unc"]["seeds"][str(s)]["surfaces"]["S2"]["sets"]["compliant"]["refs"]["os_minus_ha0_ext_pose"]["ade_m"]
        assert (c["delta"], c["lo"], c["hi"], c["n_windows"], c["n_episodes"]) == (*LIT_POSE[s], N_WIN, N_EP)


@pytest.mark.parametrize("seed", [0, 1])
def test_control_a_the_reread_uncapped_compliant_s2_pose_cell_equals_a0k0_exactly(ema30k, seed):
    assert "E1_E3_reference" in ema30k["bars"], "trees/step30000ema_unc.json has not been re-read under D-R8-E1-REF"
    for variant in ("uncapped", "capped"):                                       # capped == uncapped on the compliant set (0 binding windows)
        c = ema30k["seeds"][str(seed)]["surfaces"]["S2"]["compliant"][variant]["rows"]["ade_m"]["os_minus_ha0_ext_pose"]
        assert (c["delta"], c["lo"], c["hi"], c["n_windows"], c["n_episodes"]) == (*LIT_POSE[seed], N_WIN, N_EP), variant
    e = ema30k["seeds"][str(seed)]["surfaces"]["S2"]["compliant"]["uncapped"]["rows"]["ade_m"]["os_minus_ha0_ext"]
    assert (e["delta"], e["lo"], e["hi"]) == LIT_EXT[seed]                      # the registered read, beside, unchanged


def test_the_reread_tree_carries_the_new_schema_and_its_controls_pass(ema30k):
    b = ema30k["bars"]
    assert b["status"].startswith("WRITTEN") and ema30k["controls"]["ALL_PASS"] is True and ema30k["controls"]["C4_pose_reference_valid"]["PASS"] is True
    assert b["E1_E3_reference"] == "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)"
    for kind in ("compliant", "all_window"):
        for v in ("capped", "uncapped"):
            x = b[kind][v]
            assert x["BAR-R8-E1"]["reference"] == "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)" == x["BAR-R8-E3"]["reference"]
            assert "PRE-RULING" in x["BAR-R8-E1_ha0_ext"]["reference"] and "PRE-RULING" in x["BAR-R8-E3_ha0_ext"]["reference"]
            assert x["BAR-R8-E1_ha0_ext"]["reference_cell"] == "os_minus_ha0_ext" and x["BAR-R8-E1"]["reference_cell"] == "os_minus_ha0_ext_pose"
    sd = ema30k["seeds"]["0"]["compliance"]["grids"]
    for g in ("S2_panel_grid", "S6_panel_grid"):
        arms = sd[g]["arms"]
        assert {"refcv8_capped", "refcv8_uncapped", "refcv7_50400", "echo_ha0_ext", "echo_ha0_ext_pose"} <= set(arms)
        assert isinstance(arms["echo_ha0_ext_pose"]["compliant_set"], dict) and arms["echo_ha0_ext_pose"]["all_windows"]["n_windows"] > 0
    c4 = ema30k["controls"]["C4_pose_reference_valid"]
    assert c4["R0926_known_value_S2_all_ha0_ext_pose_minus_ha0_ext"]["exact_per_seed"] == {"s0": True, "s1": True}
    assert c4["A0K0_known_value"]["all_exact"] is True and c4["A0K0_known_value"]["n_cells_checked"] == 8


def _walk(old, new, path, ignore, out):
    """Every leaf of `old` must be present and identical in `new` (extra keys in `new` are the new schema).  Floats compared as their JSON text."""
    if path in ignore:
        return
    if isinstance(old, dict):
        if not isinstance(new, dict):
            out.append((path, "type"))
            return
        for k, v in old.items():
            if k not in new:
                out.append((path + "/" + k, "missing in the re-read"))
            else:
                _walk(v, new[k], path + "/" + k, ignore, out)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            out.append((path, "list shape"))
            return
        for i, (a, b) in enumerate(zip(old, new)):
            _walk(a, b, f"{path}[{i}]", ignore, out)
    elif json.dumps(old) != json.dumps(new):
        out.append((path, f"{old!r} != {new!r}"))


def pre_post_diffs(old: dict, new: dict) -> list:
    """Control (b).  Ignored by design: provenance stamps, and the bars' E1/E3 blocks whose MEANING changed (old BAR-R8-E1 is compared with new BAR-R8-E1_ha0_ext minus the two
    label fields).  Everything else in the pre-change JSON -- every ha0_ext cell, every refcv7 cell, the masks, the compliance arms, C1/C1v/C2/C3 -- must be reproduced exactly."""
    ignore = {"/generated", "/wall_s", "/script_sha256", "/ruling"}
    for kind in ("compliant", "all_window"):
        for v in ("capped", "uncapped"):
            for e in ("BAR-R8-E1", "BAR-R8-E3"):
                ignore.add(f"/bars/{kind}/{v}/{e}")
    out = []
    _walk(old, new, "", ignore, out)
    if isinstance(old["bars"].get("compliant"), dict):
        for kind in ("compliant", "all_window"):
            for v in ("capped", "uncapped"):
                for e in ("BAR-R8-E1", "BAR-R8-E3"):
                    o = old["bars"][kind][v][e]
                    n = {k: x for k, x in new["bars"][kind][v][e + "_ha0_ext"].items() if k not in ("reference", "reference_cell")}
                    if json.dumps(o) != json.dumps(n):
                        out.append((f"/bars/{kind}/{v}/{e}", "old bar != new <bar>_ha0_ext"))
    return out


def test_pre_post_comparator_detects_an_altered_ha0_ext_cell_and_a_dropped_arm():
    old = json.loads((PRE / "step30000ema_unc.json").read_text(encoding="utf-8")) if (PRE / "step30000ema_unc.json").exists() else None
    if old is None:
        pytest.skip("no pre-change tree")
    new = json.loads(json.dumps(old))                                                       # identical copy + the new bars keys == a perfect re-read
    for kind in ("compliant", "all_window"):
        for v in ("capped", "uncapped"):
            for e in ("BAR-R8-E1", "BAR-R8-E3"):
                new["bars"][kind][v][e + "_ha0_ext"] = {**old["bars"][kind][v][e], "reference": "x", "reference_cell": "y"}
                new["bars"][kind][v][e] = {"verdict": "changed meaning"}
    assert pre_post_diffs(old, new) == []
    bad = json.loads(json.dumps(new))
    bad["seeds"]["0"]["surfaces"]["S2"]["compliant"]["capped"]["rows"]["ade_m"]["os_minus_ha0_ext"]["lo"] += 1e-9               # a mutation: one ha0_ext cell moved
    assert pre_post_diffs(old, bad)
    bad2 = json.loads(json.dumps(new))
    del bad2["seeds"]["1"]["compliance"]["grids"]["S2_panel_grid"]["arms"]["echo_ha0_ext"]
    assert pre_post_diffs(old, bad2)


def test_control_b_every_ha0_ext_cell_of_every_banked_tree_is_reproduced_byte_for_byte():
    pre = _trees_pre()
    if not pre:
        pytest.skip("trees_pre_e1ref absent")
    assert len(pre) == 10, pre                                                              # 5k, 10.5k, 15k, 20k, 30k x (raw, EMA)
    n_cells = 0
    for t in pre:
        new_p = TREES / f"{t}.json"
        assert new_p.exists(), t
        new = json.loads(new_p.read_text(encoding="utf-8"))
        assert "E1_E3_reference" in new["bars"], f"{t} has not been re-read under D-R8-E1-REF"
        old = json.loads((PRE / f"{t}.json").read_text(encoding="utf-8"))
        d = pre_post_diffs(old, new)
        assert d == [], (t, d[:5])
        for s in ("0", "1"):
            for surf in ("S2", "S6"):
                for kind in ("compliant", "all_window"):
                    for v in ("capped", "uncapped"):
                        o = old["seeds"][s]["surfaces"][surf][kind]
                        if isinstance(o, str):
                            continue
                        for key, row in o[v]["rows"].items():
                            n_cells += 1
                            assert json.dumps(row["os_minus_ha0_ext"]) == json.dumps(new["seeds"][s]["surfaces"][surf][kind][v]["rows"][key]["os_minus_ha0_ext"]), (t, s, surf, kind, v, key)
    assert n_cells >= 10 * 2 * 2 * 2 * 2 * 14 - 1, n_cells


def test_every_reread_tree_has_a_passing_pose_control_and_the_pose_cells():
    for t in _trees_pre():
        new = json.loads((TREES / f"{t}.json").read_text(encoding="utf-8"))
        assert new["controls"]["ALL_PASS"] is True, t
        assert new["controls"]["C4_pose_reference_valid"]["PASS"] is True, t
        for s in ("0", "1"):
            assert "os_minus_ha0_ext_pose" in new["seeds"][s]["surfaces"]["S2"]["compliant"]["capped"]["rows"]["ade_m"], (t, s)
            assert "os_minus_ha0_ext_pose" in new["seeds"][s]["surfaces"]["S6"]["compliant"]["capped"]["rows"]["ade_m"], (t, s)
