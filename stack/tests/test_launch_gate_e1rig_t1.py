"""refcv9 E1-REC4 -- the gate rows, the harness stamps, the argvs, the G-LIVE dead-group rule and the bind kinds for T1, the TEMPORAL BEV WARP
(`--r9-bev-warp`, `--r9-bev-warp-lags`, `--r9-bev-warp-shuffle`), so T1 can join E1 chain 2.

THE MM'S DESIGN DECISION (pinned, 207ebd): T1 changes BOTH heads' INPUT (the fused BEV), not their structure; at gate 0 the certified head IS the base's.
The overfit harnesses replay a head on FROZEN frames with no staged history, so a T1 flag is NOT built there and the record STAMPS
`bev_warp_not_certified` == the argv's T1 set. A T1 arm binds the BASE arm's inherited records (map /1, or /2-banded for a banded base; box /1) to ITS OWN argv.
refcv9-e1rig ADMITS T1; refcv9-e2 / -full / -full-d1 REFUSE it. Liveness is G-LIVE's: `_map_hires.bev_warp.gate` and `.fuse.weight` read ACCUMULATED over
the smoke (the fuse is exactly 0.0 at step 0 by construction) and `bw_has` on a majority of windows.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
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
import test_launch_gate as TG  # noqa: E402
import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402
import test_launch_gate_e1rig_own_record as OR  # noqa: E402

G = pytest.importorskip("g_box_overfit")
O = pytest.importorskip("map_hires_overfit")
torch = pytest.importorskip("torch")

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
COMMIT = "ef" * 20
FULL_FAMILY = ("refcv9-e2", "refcv9-full", "refcv9-full-d1")
T1_STAMP_KEYS = ("bev_warp_not_certified", "bev_warp_not_certified_why")


def _argv(name):
    return json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


C0, C1, T1, T1S, B3, B1, M1 = (_argv(n) for n in ("C0", "C1", "T1", "T1s", "B3", "B1", "M1"))
T1_ONE = ["--r9-bev-warp"]
T1_BOTH = ["--r9-bev-warp", "--r9-bev-warp-shuffle"]


def _write(tmp_path, rec, name="rec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def _diff(a, b):
    da, db = dict(LG.flag_pairs(a)), dict(LG.flag_pairs(b))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


def _with_t1(argv, *flags):
    for f in flags:
        argv = LG.set_flag(argv, f, ["0.5,1.0"] if f.endswith("-lags") else [])
    return argv


# ============================================================================ the flag set is ONE fact ==
def test_the_T1_flag_set_and_stamp_keys_are_pinned_across_gate_harnesses_emitter_and_trainer():
    from tanitad.train import bev_warp_flag as BWF
    assert LG.E1_T1_FLAGS == ("--r9-bev-warp", "--r9-bev-warp-lags", "--r9-bev-warp-shuffle") == tuple(G.T1_FLAGS) == tuple(O.T1_FLAGS)
    assert (BWF.FLAG, BWF.LAGS_FLAG, BWF.SHUF_FLAG) == LG.E1_T1_FLAGS
    assert (LG.E1_T1_STAMP, LG.E1_T1_STAMP_WHY) == (G.T1_STAMP, G.T1_STAMP_WHY) == (O.T1_STAMP, O.T1_STAMP_WHY) == T1_STAMP_KEYS
    assert G.T1_WHY == O.T1_WHY and "\n" not in G.T1_WHY and "not built" in G.T1_WHY
    assert set(MK.T1_ARMS["T1"] + MK.T1_ARMS["T1s"]) <= set(LG.E1_T1_FLAGS) and MK.T1_ANCHOR == "--map-hires-near-refine-blocks"
    T = pytest.importorskip("refc_v3_train")
    assert set(LG.E1_T1_FLAGS) <= set(T.build_parser()._option_string_actions)
    assert LG.BEV_WARP_LIVE["flag"] == BWF.FLAG and LG.BEV_WARP_LIVE["batch_key"] == "bw_has" and "bw_has" in BWF.BATCH_KEYS


def test_the_G_LIVE_parameter_names_are_the_real_parameters_of_a_built_fuse(monkeypatch):
    import test_bev_warp as TB
    on, _cfg, _args = TB._rig(monkeypatch, on=True)
    names = {n for n, p in on.named_parameters() if p.requires_grad}
    assert LG.BEV_WARP_LIVE["gate"] in names and LG.BEV_WARP_LIVE["fuse"] in names
    assert {n for n in names if n.startswith("_map_hires.bev_warp.")} == {LG.BEV_WARP_LIVE["gate"], LG.BEV_WARP_LIVE["fuse"]}


# ======================================================================== the profile decision, per profile ==
def test_the_T1_arms_open_no_own_record_item_in_the_e1rig_profile_and_the_exception_is_exact():
    for argv in (T1, T1S, _with_t1(C0, "--r9-bev-warp", "--r9-bev-warp-lags", "--r9-bev-warp-shuffle")):
        assert LG.open_item_reasons(RIG, argv) == []
        assert LG.own_record_row(RIG, argv, "G-MAP-OVERFIT") is None and LG.own_record_row(RIG, argv, "G-BOX-OVERFIT") is None
    for f in LG.E1_T1_FLAGS:                                                   # each flag alone
        assert LG.open_item_reasons(RIG, LG.set_flag(C0, f, [])) == [], f
    for typo in ("--r9-bev-warp-v2", "--r9-bev-warps", "--r9-bev-warp-lagz"):   # the exception is EXACT: a sibling flag is still an unclassified lever
        ids = [x.split(":")[0] for x in LG.open_item_reasons(RIG, LG.set_flag(T1, typo, []))]
        assert ids == ["OPEN ITEM E1-BOX-OWN-RECORD"], (typo, ids)
    assert LG.open_item_reasons(RIG, C0) == LG.open_item_reasons(RIG, C1) == []


def test_the_e1rig_static_argv_rules_accept_T1_exactly_as_they_accept_C0(tmp_path):
    init = tmp_path / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped refcv8 EMA-48,852")
    prof = dict(RIG, init_pin=dict(RIG["init_pin"], init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    pmap = [("/home/nvidia/r8ns/ckpt", str(tmp_path))]
    assert LG.profile_argv_rules(prof, C0, {}, pmap) == ([], [])
    for argv in (T1, T1S):
        assert LG.profile_argv_rules(prof, argv, {}, pmap) == ([], [])
        assert LG.forbidden_lever_reasons(prof, argv)[0] == LG.forbidden_lever_reasons(prof, C0)[0] == []
        assert LG.required_flag_reasons(prof, argv) == LG.required_flag_reasons(prof, C0)


@pytest.mark.parametrize("name", FULL_FAMILY)
def test_RED_the_full_family_REFUSES_every_T1_flag_by_name(name):
    p = LG.PROFILES[name]
    for argv in (T1, T1S, LG.set_flag(C0, "--r9-bev-warp-lags", ["0.5"])):
        reasons, det = LG.forbidden_lever_reasons(p, argv)
        hit = [f for f in det["argv_hits"] if f in LG.E1_T1_FLAGS]
        assert hit and f"{hit[0]} is passed, but it is T1, the temporal BEV warp" in " | ".join(reasons), reasons       # one reason per entry: the first hit is named
    for f in LG.E1_T1_FLAGS:                                                    # every flag alone is refused, not only the pair
        argv = LG.set_flag(C0, f, [])
        assert any(f"{f} is passed, but it is T1" in r for r in LG.forbidden_lever_reasons(p, argv)[0]), f
    # classified, so no longer UNCLASSIFIED: the T1 flags are not a hit of that item, a sibling flag is
    it = next(i for i in p["open_items"] if i["id"] == "R9F-UNCLASSIFIED-LEVER")
    assert not set(LG.open_item_hits(it, T1S) or []) & set(LG.E1_T1_FLAGS)
    assert "--r9-bev-warp-v2" in LG.open_item_hits(it, LG.set_flag(T1, "--r9-bev-warp-v2", []))
    # control: the SAME refusal list on C0 holds nothing about T1
    assert not any("temporal BEV warp" in r for r in LG.forbidden_lever_reasons(p, C0)[0])


def test_every_profile_outside_e1rig_and_the_full_family_is_untouched_by_the_T1_flags():
    for name, p in LG.PROFILES.items():
        if name in ("refcv9-e1rig", *FULL_FAMILY):
            continue
        assert LG.open_item_reasons(p, T1) == LG.open_item_reasons(p, C0), name
        assert LG.forbidden_lever_reasons(p, T1)[0] == LG.forbidden_lever_reasons(p, C0)[0], name


# ============================================================================== the four judges read the stamp ==
def _map1(argv, cw="ef" * 32, **over):
    rec = TG._gmo_main_only(COMMIT, LG.argv_sha256(argv), cw)
    rec.update(O.t1_stamp_fields({"argv": argv}))
    rec.update(over)
    return rec


def _j_map1(tmp_path, rec, argv):
    return LG.judge_map_overfit(RIG, _write(tmp_path, rec), COMMIT, argv_sha=LG.argv_sha256(argv), class_weights_sha256="ef" * 32, argv=argv)


def _j_banded(tmp_path, rec, argv):
    return LG.judge_map_overfit_banded(RIG, _write(tmp_path, rec), COMMIT, argv_sha=LG.argv_sha256(argv), argv=argv)


def _box1(argv, **over):
    rec = TG._box_main_only(LG.argv_sha256(argv))
    rec.update(G.b3_record_fields(argv))
    rec.update(over)
    return rec


def _j_box1(tmp_path, rec, argv):
    return LG.judge_box_overfit(RIG, _write(tmp_path, rec), COMMIT, argv_sha=LG.argv_sha256(argv), argv=argv)


def _j_boxb3(tmp_path, rec, argv):
    ok, why = R2._box_closes(tmp_path, rec, argv)
    return why, {}


M1T1 = _with_t1(M1, "--r9-bev-warp")                                        # a BANDED base + T1 (the /2-banded record)
B3T1 = _with_t1(B3, "--r9-bev-warp")                                        # an own-head base + T1 (the /2-b3 record)
JUDGES = {  # name -> (record builder, judge, a T1 argv for that judge, the same judge's non-T1 argv)
    "map1": (lambda a, **o: _map1(a, **o), _j_map1, T1, C0),
    "banded": (lambda a, **o: OR._record(a, **{**O.t1_stamp_fields({"argv": a}), **o}), _j_banded, M1T1, M1),
    "box1": (lambda a, **o: _box1(a, **o), _j_box1, T1, C0),
    "boxb3": (lambda a, **o: R2._box_record(a, **{**G.b3_record_fields(a), **o}), _j_boxb3, B3T1, B3)}


@pytest.mark.parametrize("jn", sorted(JUDGES))
def test_GREEN_the_harness_stamped_record_for_a_T1_argv_passes_each_judge(jn, tmp_path):
    mk, judge, t1_argv, _plain = JUDGES[jn]
    rec = mk(t1_argv)
    assert rec[G.T1_STAMP] == ["--r9-bev-warp"] and rec[G.T1_STAMP_WHY]
    reasons, det = judge(tmp_path, rec, t1_argv)
    assert reasons == [], reasons
    if jn != "boxb3":
        assert det["bev_warp_not_certified"] == {"argv_t1_set": ["--r9-bev-warp"], "record_stamp": ["--r9-bev-warp"]}


@pytest.mark.parametrize("jn", sorted(JUDGES))
def test_RED_arms_the_T1_stamp_is_required_and_must_equal_the_argvs_T1_set(jn, tmp_path):
    mk, judge, t1_argv, plain = JUDGES[jn]
    good = mk(t1_argv)

    def why(rec, argv=t1_argv):
        reasons, _ = judge(tmp_path, rec, argv)
        assert reasons, "the defective record passed"
        return " | ".join(reasons)

    no_stamp = {k: v for k, v in good.items() if k not in T1_STAMP_KEYS}
    assert "stamps no `bev_warp_not_certified`" in why(no_stamp)                                        # a T1 argv + a record WITHOUT the stamp
    assert "T1 set is ['--r9-bev-warp']" in why(dict(good, bev_warp_not_certified=["--r9-bev-warp-shuffle"]))     # stamp != argv set
    assert "T1 set is ['--r9-bev-warp']" in why(dict(good, bev_warp_not_certified=list(LG.E1_T1_FLAGS)))         # extra flags
    assert "T1 set is ['--r9-bev-warp']" in why(dict(good, bev_warp_not_certified=[]))                           # empty stamp on a T1 argv
    assert "T1 set is" in why(dict(good, bev_warp_not_certified=["--r9-bev-warp", "--r9-bev-warp"]))             # duplicated
    assert "not a list" in why(dict(good, bev_warp_not_certified="--r9-bev-warp"))
    assert T1_STAMP_KEYS[1] in why({k: v for k, v in good.items() if k != T1_STAMP_KEYS[1]})                     # no one-line reason
    assert T1_STAMP_KEYS[1] in why(dict(good, bev_warp_not_certified_why="  "))
    # a stamp on an argv with NO T1 flag: the record was made for another argv
    forged = mk(plain, bev_warp_not_certified=["--r9-bev-warp"], bev_warp_not_certified_why="x")
    assert "T1 set is []" in why(forged, plain)
    # control: every defect above was the stamp's -- the good record passes, and the plain arm's record is identical to what it was
    assert judge(tmp_path, good, t1_argv)[0] == []
    assert judge(tmp_path, mk(plain), plain)[0] == [] and judge(tmp_path, {k: v for k, v in mk(plain).items() if k not in T1_STAMP_KEYS}, plain)[0] == []


def test_RED_the_T1_record_offered_to_another_T1_arm_fails_on_the_binding_and_on_the_stamp(tmp_path):
    good = _box1(T1)
    reasons, _ = _j_box1(tmp_path, good, T1S)                                                       # T1's record for T1s: wrong sha AND wrong set
    assert any("bound to argv" in r for r in reasons) and any("T1 set is ['--r9-bev-warp', '--r9-bev-warp-shuffle']" in r for r in reasons)
    forged = dict(good, launch_argv_sha256=LG.argv_sha256(T1S))                                      # right sha, wrong stamp
    reasons, _ = _j_box1(tmp_path, forged, T1S)
    assert [r for r in reasons if "bound to argv" in r] == [] and any("T1 set is ['--r9-bev-warp', '--r9-bev-warp-shuffle']" in r for r in reasons)
    both = _box1(T1S)
    assert both[G.T1_STAMP] == ["--r9-bev-warp", "--r9-bev-warp-shuffle"] and _j_box1(tmp_path, both, T1S)[0] == []


def test_C0_C1_the_M_and_B_arms_records_and_verdicts_are_identical(tmp_path):
    """No stamp, no T1 flag: every judge's verdict and details are exactly what they were (no new key in `det`)."""
    for argv in (C0, C1, B1, B3, _argv("B3p"), _argv("B2")):
        aux = {k: v for k, v in G.b3_record_fields(argv).items() if k.startswith("aux_heads")}         # the aux arms' own (E1-REC3) stamp: unchanged
        r, d = _j_box1(tmp_path, dict(TG._box_main_only(LG.argv_sha256(argv)), **aux), argv)
        assert r == [] and "bev_warp_not_certified" not in d, argv[-1]
        r, d = _j_map1(tmp_path, TG._gmo_main_only(COMMIT, LG.argv_sha256(argv), "ef" * 32), argv)
        assert r == [] and "bev_warp_not_certified" not in d
    for argv in (M1, _argv("M2a"), _argv("M2b")):
        r, d = _j_banded(tmp_path, OR._record(argv), argv)
        assert r == [] and "bev_warp_not_certified" not in d
    # judges without an argv (older callers): the clause is skipped, never an exception
    assert LG.judge_map_overfit(RIG, _write(tmp_path, TG._gmo_main_only(COMMIT, LG.argv_sha256(T1), "ef" * 32)), COMMIT, argv_sha=LG.argv_sha256(T1),
                                class_weights_sha256="ef" * 32)[0] == []
    for fn in (LG.judge_map_overfit, LG.judge_map_overfit_banded, LG.judge_box_overfit, LG.judge_box_own_record):
        assert "bev_warp_stamp_reasons(tag, rec, argv)" in inspect.getsource(fn), fn.__name__


def test_the_jobs_state_what_the_pass_does_not_certify(tmp_path, monkeypatch):
    monkeypatch.setattr(LG, "judge_closure", lambda *a, **k: ([], {}))                    # the closure is the bind run's own artifact; not under test here

    cwd = tmp_path / "cw"
    cwd.mkdir()
    (cwd / "map_hires_class_weights_train_100x30.json").write_bytes(b"stand-in class weights")
    cw = hashlib.sha256(b"stand-in class weights").hexdigest()

    def ev(job, key, argv, rec):
        c = LG.Ctx(profile="refcv9-e1rig", tree=str(tmp_path), commit=COMMIT, argv=list(argv), out_dir=str(tmp_path),
                   path_map=[("/home/nvidia/data/refcv7", str(cwd))],
                   tree_sha256="0" * 64, argv_sha256=LG.argv_sha256(argv),
                   options={"box_overfit_record": _write(tmp_path, rec), "map_overfit_record": _write(tmp_path, rec, "m.json"),
                            "box_overfit_closure": None, "map_overfit_closure": None})
        return job(c)[key]
    for job, key, mk in ((LG._job_box_overfit, "G-BOX-OVERFIT", _box1), (LG._job_map_overfit, "G-MAP-OVERFIT", lambda a, **o: _map1(a, cw=cw, **o))):
        e = ev(job, key, T1S, mk(T1S))
        assert e["status"] == "PASS" and "does NOT certify the temporal BEV warp ['--r9-bev-warp', '--r9-bev-warp-shuffle']" in " ".join(e["reasons"]), e["reasons"]
        e = ev(job, key, T1, {k: v for k, v in mk(T1).items() if k not in T1_STAMP_KEYS})        # RED: T1 argv, unstamped
        assert e["status"] == "FAIL" and any("bev_warp_not_certified" in r for r in e["reasons"])
        e = ev(job, key, C0, mk(C0))                                                        # C0: the PASS text is exactly what it was
        assert e["status"] == "PASS" and "temporal BEV warp" not in " ".join(e["reasons"])


# ================================================================================================ the harnesses ==
def test_the_box_harness_stamps_the_T1_set_and_keeps_the_other_records_byte_identical():
    assert G.b3_record_fields(T1) == {G.T1_STAMP: ["--r9-bev-warp"], G.T1_STAMP_WHY: G.T1_WHY}
    assert G.b3_record_fields(T1S)[G.T1_STAMP] == ["--r9-bev-warp", "--r9-bev-warp-shuffle"]
    assert G.b3_record_fields(_with_t1(C0, "--r9-bev-warp-shuffle", "--r9-bev-warp-lags", "--r9-bev-warp"))[G.T1_STAMP] == list(G.T1_FLAGS)     # T1_FLAGS order
    assert G.b3_record_fields(C0) == {} and G.b3_record_fields(["--w-box3d", "1.0", "--trunk-compile"]) == {}
    r = G.b3_record_fields(_with_t1(LG.set_flag(LG.set_flag(B3, "--r9-box-bev2-aux", []), "--w-box-bev2", ["0.1"]), "--r9-bev-warp"))
    assert r["schema"] == G.B3_SCHEMA and r[G.AUX_STAMP] == ["--r9-box-bev2-aux"] and r[G.T1_STAMP] == ["--r9-bev-warp"]            # three independent stamps
    assert G.b3_record_fields(["--r9-bev-warp-lags=0.5,1.0"])[G.T1_STAMP] == ["--r9-bev-warp-lags"]                                   # the `=` form counts


def test_the_box_harness_does_not_build_T1_the_build_argv_drops_the_flags_and_the_lags_value():
    T = pytest.importorskip("refc_v3_train")
    base, _ = G.harness_build_argv(C0)
    for argv, flags in ((T1, ["--r9-bev-warp"]), (T1S, ["--r9-bev-warp", "--r9-bev-warp-shuffle"])):
        build, rec = G.harness_build_argv(argv)
        assert not set(G.T1_FLAGS) & set(build) and rec["bev_warp_flags_not_built"] == flags
        d = dict(LG.flag_pairs(build)), dict(LG.flag_pairs(base))
        assert {f for f in set(d[0]) | set(d[1]) if d[0].get(f) != d[1].get(f)} == {"--out"}            # EXACTLY C0's build argv (no neighbour swallowed)
        a = T.build_parser().parse_args(build)                                                         # T1 is OFF in what the harness builds from
        assert a.r9_bev_warp is False and a.r9_bev_warp_shuffle is False and a.r9_bev_warp_lags == "0.5,1.0"
    with_lags = _with_t1(T1, "--r9-bev-warp-lags")
    build, rec = G.harness_build_argv(with_lags)
    assert build == G.harness_build_argv(T1)[0] and rec["bev_warp_flags_not_built"] == ["--r9-bev-warp", "--r9-bev-warp-lags"]      # the lags VALUE goes too
    assert G.strip_t1(["--a", "1", "--r9-bev-warp-lags", "0.5", "1.0", "--b", "2"]) == (["--a", "1", "--b", "2"], ["--r9-bev-warp-lags"])
    assert G.strip_t1(["--r9-bev-warp-lags=0.5,1.0", "x"]) == (["x"], ["--r9-bev-warp-lags"])
    assert G.strip_t1(C0)[0] == C0 and G.strip_t1(C0)[1] == []
    for argv in (C0, B3):                                                                              # every other launch argv: the record keys are what they were
        assert "bev_warp_flags_not_built" not in G.harness_build_argv(argv)[1]
    src = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    assert src.count("build_argv, tc_rec = harness_build_argv(launch_argv)") == 1 and src.count("TrainerAdapter(tr, L, build_argv,") == 1
    assert src.count("build, t1 = strip_t1(build)") == 1 and src.count("t1 = strip_t1(launch_argv)[1]") == 1


def test_the_map_harness_stamps_the_T1_set_in_both_modes_and_refuses_what_it_must(tmp_path):
    def f(argv, name):
        p = tmp_path / name
        p.write_text(json.dumps({"argv": argv}), encoding="utf-8")
        return p
    pc, pt, ps, pm = f(C0, "c0.json"), f(T1, "t1.json"), f(T1S, "t1s.json"), f(M1T1, "m1t1.json")
    assert O.t1_record_fields(None) == {} and O.t1_record_fields(pc, LG.argv_sha256(C0)) == {}                       # OFF: byte-for-byte what it was
    assert O.t1_record_fields(pt, LG.argv_sha256(T1)) == {O.T1_STAMP: ["--r9-bev-warp"], O.T1_STAMP_WHY: O.T1_WHY}
    assert O.t1_record_fields(ps, LG.argv_sha256(T1S))[O.T1_STAMP] == ["--r9-bev-warp", "--r9-bev-warp-shuffle"]
    assert O.t1_flags_of(["--x", "--r9-bev-warp-lags=1.0"]) == ["--r9-bev-warp-lags"] and O.t1_stamp_fields(None) == {}
    with pytest.raises(SystemExit, match="built from one argv and bound to another"):
        O.t1_record_fields(pt, LG.argv_sha256(T1S))                                                                  # file != the sha it is bound to
    with pytest.raises(SystemExit, match="BANDED head"):
        O.t1_record_fields(pm, LG.argv_sha256(M1T1))                                                                 # a banded argv needs --banded
    assert O.t1_stamp_fields(O.launch_argv_facts(pm, LG.argv_sha256(M1T1)))[O.T1_STAMP] == ["--r9-bev-warp"]       # ... and stamps there
    try:                                                                                                             # --launch-argv alone is no longer a banded-only refusal
        O.main(["--spec", "x", "--v2-cache", "x", "--gt-root", "x", "--extrinsics", "x", "--out", "x", "--launch-argv", str(pt)])
    except SystemExit as e:
        assert "banded option" not in str(e)
    except Exception:
        pass
    with pytest.raises(SystemExit, match="--banded options"):
        O.main(["--spec", "x", "--v2-cache", "x", "--gt-root", "x", "--extrinsics", "x", "--out", "x", "--own-enc"])
    # the wiring: each line, removed, breaks the scan (the stamp is written BEFORE the record is)
    for fn, needle in ((O.main, "rec.update(t1_record_fields(a.launch_argv, a.launch_argv_sha256))"), (O.main_banded, "rec.update(t1_stamp_fields(facts))")):
        src = inspect.getsource(fn)
        assert src.count(needle) == 1 and needle not in src.replace(needle, "pass") and src.index(needle) < src.index(".write_text(")


# ================================================================================================== the argvs ==
@pytest.mark.parametrize("arm,flags", [("T1", ["--r9-bev-warp"]), ("T1s", ["--r9-bev-warp", "--r9-bev-warp-shuffle"])])
def test_T1_arm_argvs_differ_from_C0_in_exactly_the_lever_flags(arm, flags):
    argv = {"T1": T1, "T1s": T1S}[arm]
    assert _diff(C0, argv) == set(flags) | {"--out"}
    assert [f for f, _ in LG.flag_pairs(C0)] == [f for f, _ in LG.flag_pairs(argv) if f not in flags]            # same flags, same order
    d = dict(LG.flag_pairs(argv))
    assert all(d[f] == [] for f in flags) and "--r9-bev-warp-lags" not in d                                        # the lags are T1's default 0.5 / 1.0 s
    assert d["--seed"] == ["0"] and d["--out"] == [f"/home/nvidia/r8ns/r9e1/runs/r9e1-{arm}"] and d["--workers"] == ["6"] and d["--map-hires"] == ["on"]
    i = argv.index("--r9-bev-warp")
    assert argv[i - 2:i] == ["--map-hires-near-refine-blocks", "1"] and argv[i + len(flags)] == "--bev-source"          # right after the last 10 cm map flag
    assert not any(f.startswith(("--r9-box-", "--r9-map-")) for f in d)                                             # no map / box lever
    raw = (RUNS / f"r9e1_{arm}.argv.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    rec = json.loads(raw)
    assert rec["arm"] == f"r9e1_{arm}" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0" and rec["steps"] == 4000
    assert rec["lever"]["flags"] == flags and rec["lever"]["lags_s"] == [0.5, 1.0] and rec["lever"]["shuffle_control"] == (arm == "T1s")
    assert rec["base_sha256"] == hashlib.sha256((RUNS / "r9e1_C0.argv.json").read_bytes()).hexdigest()
    assert LG.argv_sha256(argv) not in (LG.argv_sha256(C0), LG.argv_sha256(C1), LG.argv_sha256(B3)) and LG.argv_sha256(T1) != LG.argv_sha256(T1S)


def test_T1_emitter_reproduces_the_committed_files_and_its_pins_are_real():
    for arm in ("T1", "T1s"):
        argv = {"T1": T1, "T1s": T1S}[arm]
        assert MK.t1_arm_argv(C0, arm) == argv
        MK.check_t1_arm_pins(argv, arm)                              # the trainer's parser, the rig pin AND T1's own pin_args accept
    with pytest.raises(SystemExit, match="already carries"):
        MK.t1_arm_argv(T1, "T1s")                                    # derive from C0 only
    with pytest.raises(SystemExit, match="not in"):
        MK.t1_arm_argv(C0, "T2")
    # RED: each pin is a real check
    lone_lags = LG.set_flag(C0, "--r9-bev-warp-lags", ["1.0"])                       # a lags flag with no --r9-bev-warp: a dead flag
    with pytest.raises(SystemExit, match="did not parse as on"):
        MK.check_t1_arm_pins(lone_lags, "T1")
    with pytest.raises(SystemExit, match="shuffle control parsed as"):
        MK.check_t1_arm_pins(T1S, "T1")                                              # the wrong arm name for the argv
    with pytest.raises(SystemExit, match="shuffle control parsed as"):
        MK.check_t1_arm_pins(T1, "T1s")
    with pytest.raises(SystemExit, match="not T1's default"):
        MK.check_t1_arm_pins(LG.set_flag(T1, "--r9-bev-warp-lags", ["0.5,2.0"]), "T1")
    with pytest.raises(SystemExit):                                                  # T1 needs --map-hires on (its own pin_args)
        MK.check_t1_arm_pins(LG.set_flag(T1, "--map-hires", ["off"]), "T1")
    with pytest.raises(SystemExit):                                                  # ... and the perception rig
        MK.check_t1_arm_pins([t for t in T1 if t != "--r9-perc-rig"], "T1")
    with pytest.raises(SystemExit, match="map / box lever"):
        MK.check_t1_arm_pins(_with_t1(B3, "--r9-bev-warp"), "T1")                    # a T1 arm differs from C0 in its T1 flags alone
    with pytest.raises(SystemExit, match="not a T1 arm"):
        MK.check_t1_arm_pins(B3, "B3")
    with pytest.raises(SystemExit):                                                  # --t1-arm excludes --map-arm / --box-arm
        MK.main(["--base", str(RUNS / "r9e1_C0.argv.json"), "--t1-arm", "T1", "--box-arm", "B3", "--write", "x"])


def test_the_other_argv_files_are_unchanged_by_the_emitter():
    for name in ("B3", "B3p", "B1", "B2", "B2c"):
        assert MK.box_arm_argv(C0, name) == _argv(name)
    for name in ("M1", "M2a", "M2b"):
        assert MK.map_arm_argv(C0, name) == _argv(name)


# ============================================================================================ G-LIVE (T1) ==
def _bw(*rows):
    rec = {}
    LG.bev_warp_observe(rec, {"bw_has": torch.tensor(rows, dtype=torch.bool)})
    return rec["bw_has"]


def _g(gate=(3, 3), fuse=(3, 2)):
    mk = lambda st: {"numel": 4, "steps": st[0], "nz_steps": st[1], "nonfinite": False}
    return {LG.BEV_WARP_LIVE["gate"]: mk(gate), LG.BEV_WARP_LIVE["fuse"]: mk(fuse), "_map_hires.lift.proj.weight": mk((3, 3))}


HAS = _bw(*([[True, True]] * 8 + [[True, False]] * 2))                                # lag 1: 100 %, lag 2: 80 %
LIVE = lambda argv=T1, grads=None, bw=HAS: LG.judge_bev_warp_live(argv, _g() if grads is None else grads, bw, expect_steps=3)


def test_GREEN_G_LIVE_T1_accepts_a_healthy_smoke_and_is_INERT_for_every_other_argv():
    r, d = LIVE()
    assert r == [] and d["gate"]["nz_steps"] == 3 and d["fuse"]["nz_steps"] == 2 and d["bw_has_per_lag_frac"] == [1.0, 0.8]
    assert LG.judge_bev_warp_live(C0, {}, None) == ([], {}) and LG.judge_bev_warp_live(B3, {}, None) == ([], {}) and LG.judge_bev_warp_live(M1, {}, None) == ([], {})
    assert LG.judge_bev_warp_live(LG.set_flag(C0, "--r9-bev-warp-lags", ["0.5"]), {}, None) == ([], {})            # the lags flag alone is not the switch


def test_RED_G_LIVE_a_fuse_that_stays_at_0_0_over_the_whole_smoke_FAILS_by_name():
    r, _ = LIVE(grads=_g(fuse=(3, 0)))
    assert len(r) == 1 and "`_map_hires.bev_warp.fuse.weight` received ZERO gradient on all 3 optimiser steps" in r[0] and "ReZero" in r[0]
    r, _ = LIVE(grads=_g(gate=(3, 0)))
    assert len(r) == 1 and "`_map_hires.bev_warp.gate` received ZERO gradient" in r[0]
    # accumulated, not step 0: ONE nonzero step anywhere in the smoke is enough for the fuse, and a 1-step smoke cannot judge it at all
    assert LIVE(grads=_g(fuse=(3, 1)))[0] == []
    r, _ = LIVE(grads=_g(gate=(1, 1), fuse=(1, 0)))
    assert len(r) == 1 and "fewer than 2 steps cannot judge it" in r[0]
    # an untracked parameter (not built / not in the optimiser) is a failure, never a skip
    g = _g()
    del g[LG.BEV_WARP_LIVE["fuse"]]
    r, _ = LIVE(grads=g)
    assert len(r) == 1 and "`_map_hires.bev_warp.fuse.weight` is not among the trainable parameters" in r[0]
    assert len(LIVE(grads={})[0]) == 2


def test_RED_G_LIVE_a_smoke_with_bw_has_False_everywhere_FAILS_and_a_minority_FAILS():
    r, d = LIVE(bw=_bw(*([[False, False]] * 10)))
    assert len(r) == 1 and "`bw_has` is True on only [0.0, 0.0] of 10 smoke windows" in r[0] and d["bw_has_per_lag_frac"] == [0.0, 0.0]
    r, _ = LIVE(bw=_bw(*([[True, True]] * 4 + [[True, False]] * 6)))                       # lag 2 on 40 %: not a majority
    assert len(r) == 1 and "lag column(s) [1]" in r[0]
    assert LIVE(bw=_bw(*([[True, True]] * 5 + [[False, False]] * 5)))[0] != []              # exactly 50 % is not a MAJORITY
    assert LIVE(bw=_bw(*([[True, True]] * 6 + [[False, False]] * 4)))[0] == []
    r, _ = LIVE(bw=None)
    assert len(r) == 1 and "carried no `bw_has`" in r[0]
    bad = {}
    LG.bev_warp_observe(bad, {"bw_has": torch.ones(4, dtype=torch.bool)})                    # a malformed key is recorded, judged, never raised into the step
    assert "error" in bad["bw_has"] and "could not be read" in LIVE(bw=bad["bw_has"])[0][0]
    ign = {}
    LG.bev_warp_observe(ign, {"frames": 1})
    LG.bev_warp_observe(ign, None)
    assert ign == {}                                                                          # a batch without the key is ignored (every non-T1 launch)


def _smoke_on_the_real_model(monkeypatch, *, lr, history=True, steps=4):
    """What `run_smoke` records, on the REAL tiny rig model: `bev_warp_observe` per training call and `step_wrap`'s nz_steps count per parameter."""
    import refc_v3_train as T
    import test_bev_warp as TB
    on, cfg, args = TB._rig(monkeypatch, on=True)
    batch, _ = TB._batch_t1(cfg, on, args)
    if not history:
        batch = dict(batch, bw_has=torch.zeros_like(batch["bw_has"]))
    names = [LG.BEV_WARP_LIVE["gate"], LG.BEV_WARP_LIVE["fuse"]]
    params = dict(on.named_parameters())
    rec = {"grads": {n: {"numel": params[n].numel(), "steps": 0, "nz_steps": 0, "nonfinite": False} for n in names}}
    opt = torch.optim.SGD([params[n] for n in names], lr=lr)
    for _ in range(steps):
        on.zero_grad(set_to_none=True)
        on.train()
        torch.manual_seed(5)
        T.compute_losses_v3(on, batch, "cpu", log_metrics=False)["loss"].backward()
        LG.bev_warp_observe(rec, batch)
        for n in names:
            r, g = rec["grads"][n], params[n].grad
            r["steps"] += 1
            r["nz_steps"] += int(g is not None and bool(torch.count_nonzero(g) > 0))
        opt.step()
    return rec


def test_the_real_model_smoke_passes_when_the_gate_moves_and_the_RED_arms_fail_for_their_own_reason(monkeypatch):
    ok = _smoke_on_the_real_model(monkeypatch, lr=0.05)
    r, d = LG.judge_bev_warp_live(T1, ok["grads"], ok["bw_has"], expect_steps=4)
    assert r == [], r
    assert d["gate"]["nz_steps"] == 4 and d["fuse"]["nz_steps"] == 3 and d["bw_has_all_lags_frac"] == 1.0         # fuse: exactly 0.0 at step 0, > 0 after the gate moved
    frozen = _smoke_on_the_real_model(monkeypatch, lr=0.0)                                                        # nothing ever moves: the ReZero fuse never wakes
    r, d = LG.judge_bev_warp_live(T1, frozen["grads"], frozen["bw_has"], expect_steps=4)
    assert d["fuse"]["nz_steps"] == 0 and d["gate"]["nz_steps"] == 4 and len(r) == 1 and "fuse.weight` received ZERO gradient" in r[0]
    nohist = _smoke_on_the_real_model(monkeypatch, lr=0.05, history=False)                                       # no window carries history
    r, d = LG.judge_bev_warp_live(T1, nohist["grads"], nohist["bw_has"], expect_steps=4)
    assert d["gate"]["nz_steps"] == 0 and d["fuse"]["nz_steps"] == 0 and len(r) == 3
    assert any("gate` received ZERO" in x for x in r) and any("fuse.weight` received ZERO" in x for x in r) and any("`bw_has` is True on only [0.0, 0.0]" in x for x in r)


def test_the_smoke_wiring_is_in_place_each_line_removed_breaks_the_scan():
    src = inspect.getsource(LG.run_smoke)
    assert src.count("bev_warp_observe(rec, batch)") == 1 and src.index("bev_warp_observe(rec, batch)") < src.index("out = _orig(model, batch, device, *a, **k)")
    js = inspect.getsource(LG.job_smoke)
    needle = 'judge_bev_warp_live(ctx.argv, run1.get("grads") or {}, run1.get("bw_has"), expect_steps=steps)'
    assert js.count(needle) == 1 and "if has_flag(ctx.argv, BEV_WARP_LIVE[\"flag\"]):" in js and "reasons += btr" in js
    # T1 adds no `--w-*` weight: G-LIVE's registry rules are untouched and C0 / T1 declare the same terms
    T = pytest.importorskip("refc_v3_train")
    t0, p0 = LG.declared_terms(T, T.build_parser().parse_args(C0))
    t1, p1 = LG.declared_terms(T, T.build_parser().parse_args(T1))
    assert p0 == p1 == [] and [t["id"] for t in t0] == [t["id"] for t in t1]


# ============================================================================================ the bind script ==
def _args_of(root, arm, kind):
    return (root / f"ovf_{arm}_{R2.HASH[:12]}" / f"{kind}.args").read_text(encoding="utf-8").replace("\r", "").split("\n")


@R2.needs_bash
@pytest.mark.parametrize("kind,arm", [("map", "T1"), ("box", "T1"), ("map", "T1s"), ("box", "T1s")])
def test_dry_run_accepts_T1_with_the_inherited_map_and_box_kinds(tmp_path, kind, arm):
    out, root = R2._bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    sha = LG.argv_sha256(_argv(arm))
    assert f"sha256 {sha}" in out and f"argv sha {sha[:16]}" in out
    flags = "--r9-bev-warp" if arm == "T1" else "--r9-bev-warp --r9-bev-warp-shuffle"
    assert f"temporal BEV warp flag(s) {flags}: not built by the overfit harness" in out and "bev_warp_not_certified" in out
    toks = [t for t in _args_of(root, arm, kind) if t]
    argv_file = (RUNS / f"r9e1_{arm}.argv.json").as_posix()
    if kind == "map":                                   # the /1 map harness learns the argv ONLY for a T1 argv: the last two tokens
        assert toks[-2:] == ["--launch-argv", argv_file] and toks.count("--launch-argv") == 1
    else:                                               # the box template already hands the harness the launch argv
        assert toks[toks.index("--launch-argv") + 1] == argv_file


@R2.needs_bash
def test_the_other_arms_map_and_box_args_are_byte_identical_to_the_direct_template_render(tmp_path):
    """No T1 flag => the script appends NOTHING (C0 / C1 / B / M map and box args are what they were)."""
    for kind, tpl, arms in (("map", "refcv8_bind_map.args.tpl", ("C0", "C1", "B1", "B3")), ("box", "r9e1_bind_box.args.tpl", ("C0", "B2", "M1"))):
        for arm in arms:
            out, root = R2._bind(tmp_path / f"{kind}{arm}", kind, arm)
            assert "DRYOK" in out and "temporal BEV warp" not in out, out
            bd = (root / f"ovf_{arm}_{R2.HASH[:12]}").as_posix()
            want = [ln for ln in (ROOT / "scripts" / tpl).read_text(encoding="utf-8").replace("\r", "").split("\n")
                    if ln.strip() and not ln.startswith("#") and not ln.startswith("DATA-ROOT")]
            want = [ln.replace("{ARGV_SHA}", LG.argv_sha256(_argv(arm))).replace("{ARGV}", (RUNS / f"r9e1_{arm}.argv.json").as_posix())
                    .replace("{OUT}", bd + f"/run_{kind}").replace("{COMMIT}", R2.HASH).replace("{CODE}", R2.CODE_TREE.as_posix()) for ln in want]
            got = [t for t in _args_of(root, arm, kind) if t]
            assert got == want or (kind == "map" and "--launch-argv" not in got and len(got) == len(want)), (kind, arm)


@R2.needs_bash
@pytest.mark.parametrize("kind,arm,why", [("map-banded", "T1", "WITHOUT --r9-map-banded"), ("box-b3", "T1", "WITHOUT any --r9-box-"),
                                          ("box-b3", "T1s", "WITHOUT any --r9-box-")])
def test_the_banded_and_b3_kinds_still_refuse_a_T1_only_argv(tmp_path, kind, arm, why):
    out, root = R2._bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and why in out and "DRYOK" not in out, out
    assert not root.exists() or not list(root.iterdir())


def test_every_python_snippet_in_the_bind_script_still_compiles_and_the_new_ones_are_found():
    sn = R2._snippets()
    for i, code in enumerate(sn):
        compile(code, f"r9e1_bind.sh snippet {i}", "exec")
    assert any("E1_T1_FLAGS" in c for c in sn) and any("T1JUDGE-OK" in c for c in sn)


def _t1_snippet_run(tmp_path, rec, arm, kind):
    import os
    code = next(c for c in R2._snippets() if "T1JUDGE-OK" in c)
    r = subprocess.run([sys.executable, "-I", "-c", code, str(R2.CODE_TREE), str(_write(tmp_path, rec, f"{kind}_{arm}.json")),
                        str(RUNS / f"r9e1_{arm}.argv.json"), R2.HASH, kind], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=300)
    assert r.returncode == 0, r.stderr[-600:]
    return r.stdout


def test_the_T1_assertion_snippet_passes_a_stamped_record_and_refuses_an_unstamped_foreign_or_misstamped_one(tmp_path):
    for kind, mk in (("box", _box1), ("map", lambda a, **o: _map1(a, **o))):
        good = _t1_snippet_run(tmp_path, mk(T1S), "T1s", kind)
        assert "T1JUDGE-OK" in good and "REFUSED" not in good, (kind, good)
        for name, rec in (("unstamped", {k: v for k, v in mk(T1S).items() if k not in T1_STAMP_KEYS}), ("foreign", mk(T1)),
                          ("other_stamp", mk(T1S, bev_warp_not_certified=["--r9-bev-warp"]))):
            out = _t1_snippet_run(tmp_path, rec, "T1s", kind)
            assert "REFUSED:" in out and "T1JUDGE-OK" not in out, (kind, name, out)
    # the snippet reaches the GATE'S judge / stamp function (not a copy of it)
    assert "bev_warp_stamp_reasons" in next(c for c in R2._snippets() if "T1JUDGE-OK" in c) and "L.judge_box_overfit" in next(c for c in R2._snippets() if "T1JUDGE-OK" in c)
