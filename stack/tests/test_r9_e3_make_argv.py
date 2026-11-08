"""R9-E3-ARGV (MM 4a3d05, 2026-11-07) -- the E3 full-run argv and its D1 early-branch argv, COMPOSED from the E2 GO arm's argv doc and admitted by the launch gate's
``refcv9-full`` / ``refcv9-full-d1`` profiles, and nothing broader.

  composer   stack/scripts/r9_e3_make_argv.py   (E2 doc -> E3 doc [profile refcv9-full] + D1 doc [profile refcv9-full-d1]; byte-reproducible)
  sources    every committed r9e2_<V>.argv.json (K0, K2, K2M1N20, K2M3, K2M3N20) + a synthetic "hold L" K2 (``--w-r8-drivable`` 0.06843 + ``w_cost_ratio`` / ``w_cost_source``).
             NO test pins K2's weight: it is read from the SOURCE, so the file passes with either content of ``r9e2_K2.argv.json``.

STATIC GATE CHECKS CALLED (no GPU, no data): ``profile_argv_rules`` (required flags, forbidden values, forbidden levers [the branch flags outside the D1 profile], nav-tau rule, required_values /
required_values_when [T1h bundle, near lift, W-MAP-2], amendment acks, the init pin by md5, the WSD-ROW rule against the profile's probe literal ``_R9F_PROBE_FULL`` / ``_R9F_PROBE_D1`` and ``--save-at``
naming 30000, ``--w-map-hires``, and ``wsd_branch_reasons``: the 4-flag delta read against the SOURCE run's ``config.json``), ``lr_schedule_reasons``, ``wsd_branch_reasons``, ``open_item_reasons``,
``lever_attach_reasons`` (against the trainer source), ``smoke_argv`` / ``wsd_smoke_rewrite`` (the G-LIVE argv rewrite), ``declared_terms`` and ``passthrough_reasons`` (importorskip torch).
NOT CALLED (need a built model, the data, a smoke run or the Thor host -- the ones the module docstring of the HANDOFF lists): G-LIVE itself (``job_smoke``, ``wcost_reasons`` -- it needs the smoke's
W-COST line --, ``full_profile_live_reasons``'s share alarm and its measured lr rows, ``_save_at_check``), G-DVB's ``judge_dvb`` / ``box_required_reasons`` / ``hires_static_reasons`` (a built model),
G-CLOCK (``p10_clock_check``: the clock sidecar), G-EVAL, G-CKPT, G-SUITE.

THE TRAINER'S OWN PARSER and ``lr_schedule`` are called on the composed argvs (``parse_args``, ``check_args``, ``assert_sched_matches_args``, and for D1 the REAL ``branch_preflight`` against a
``config.json`` holding the E3 argv + ``check_branch_checkpoint``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
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
import r9_e2_make_argv as MK2  # noqa: E402
import r9_e3_make_argv as MK  # noqa: E402
from tanitad.train import lr_schedule as LRS  # noqa: E402
from test_launch_gate_refcv9_full import TRAINER, env  # noqa: E402,F401  (env: the module fixture -- the pinned-init profiles + the nav-tau record)

RUNS = ROOT / "ops" / "runs.d"
COMMITTED = ("K0", "K2", "K2M1N20", "K2M3", "K2M3N20")
SOURCES = COMMITTED + ("K2HOLDL",)
BRANCH = LG._R9F_BRANCH
OPEN_ITEMS = {"K0": [], "K2": [], "K2HOLDL": [], "K2M1N20": ["E1-MAP-OWN-RECORD"], "K2M3": ["E1-MAP-OWN-RECORD"], "K2M3N20": ["E1-MAP-OWN-RECORD"]}
CRITIC_SOURCES = [v for v in SOURCES if v != "K0"]
E3_OUT = "/home/nvidia/r8ns/r9e3/runs/r9e3-%s"
HOLD_L_W, HOLD_L_R = "0.06843", 1.4613


# ================================================================================================ fixtures ==
def _holdl_k2() -> dict:
    """What hold L does to K2's file: the weight 0.1 -> 0.06843 (0.1 / R) and the W-COST stamp at the top level (``stamp_w_cost``)."""
    d = json.loads((RUNS / "r9e2_K2.argv.json").read_text(encoding="utf-8"))
    d["argv"] = LG.set_flag(d["argv"], "--w-r8-drivable", [HOLD_L_W])
    d["lever"]["w_map_1"] = ["--r8-critic-drivable", "--w-r8-drivable", HOLD_L_W]
    d["lever"]["w_r8_drivable_status"] = "MEASURED (0.1 / R, the explore smoke's W-COST ratio)"
    return MK2.stamp_w_cost(d, HOLD_L_R, "F:/synthetic/k2_explore_smoke.log")


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("e3src")
    out = {v: RUNS / f"r9e2_{v}.argv.json" for v in COMMITTED}
    out["K2HOLDL"] = d / "r9e2_K2.argv.json"
    MK2.write(out["K2HOLDL"], _holdl_k2())
    return out


_CACHE: dict = {}


def built(v, sources, out=None, **kw):
    """(path, bytes, E2 doc, E3 doc, D1 doc) for source ``v`` (memoised for the default arguments)."""
    key = (v, out, tuple(sorted(kw.items())))
    if key not in _CACHE:
        p = sources[v]
        b = p.read_bytes()
        doc = json.loads(b.decode("utf-8"))
        e3, d1 = MK.build(doc, b, p, out or E3_OUT % v, **kw)
        _CACHE[key] = (p, b, doc, e3, d1)
    return _CACHE[key]


def items(prof, argv):
    return sorted(m.group(1) for m in (re.match(r"OPEN ITEM (\S+):", r) for r in LG.open_item_reasons(prof, argv)) if m)


def rules(env, key, argv, pmap=None):
    return LG.profile_argv_rules(env[key], argv, env["opts"], pmap if pmap is not None else env["pmap"])


def stage(env, v, e3_argv):
    """The D1 gate row reads the SOURCE run's ``config.json`` (``argv_effective``) beside ``ckpt_full_30000.pt``: stage them and map E3's ``--out`` onto the staged directory."""
    src = env["tmp"] / f"e3run_{v}"
    src.mkdir(exist_ok=True)
    (src / "config.json").write_text(json.dumps({"argv_effective": list(e3_argv)}), encoding="utf-8")
    (src / "ckpt_full_30000.pt").write_bytes(b"stand-in for the stable-phase full-state checkpoint")
    return list(env["pmap"]) + [(LG.flag_values(e3_argv, "--out")[0], str(src))]


def groups(argv):
    return LG.flag_pairs(argv)


def drop_save_at(argv):
    return LG.set_flag(argv, "--save-at", None)


# ========================================================================= the composer: E3 = E2 + the rows ==
@pytest.mark.parametrize("v", SOURCES)
def test_E3_is_the_E2_argv_with_only_the_schedule_rows_save_at_and_out_changed_and_every_other_token_is_the_sources_in_order(v, sources):
    _p, _b, doc, e3, _d1 = built(v, sources)
    a, b = groups(e3["argv"]), groups(doc["argv"])
    assert a[-1] == ("--save-at", ["30000"]) and "--save-at" not in dict(b)
    assert [f for f, _ in a[:-1]] == [f for f, _ in b]                                   # the same flags in the same order (+ --save-at LAST)
    want = {"--out", "--steps", "--wsd-decay-steps", "--grad-share-at"} | ({"--warmup"} if dict(b)["--warmup"] != ["1000"] else set())
    changed = {f for (f, va), (_, vb) in zip(a[:-1], b) if va != vb}
    assert changed == want, changed
    assert [g for g in a[:-1] if g[0] not in want] == [g for g in b if g[0] not in want]   # every other group is byte-identical (and so is its order)
    d = dict(a)
    assert d["--steps"] == ["73280"] and d["--warmup"] == ["1000"] and d["--wsd-decay-steps"] == ["12458"] and d["--lr-schedule"] == ["wsd"]
    assert d["--out"] == [E3_OUT % v] and d["--init-from"] == dict(b)["--init-from"] == [MK2.INIT_FROM]
    assert e3["argv"].count("--out") == 1 and [t for t in e3["argv"] if t.startswith("--") and "=" in t] == []


@pytest.mark.parametrize("v", SOURCES)
def test_E3_takes_the_designs_12_gradient_share_readings_and_every_one_is_due_in_E3_and_in_D1(v, sources):
    # REFCV9_DESIGN 4.5 costs 12 readings; the trainer's own parser refuses a step past --steps or off the --grad-share-every grid, and D1 (36,000 steps) must keep E3's value (the gate's 4-flag delta)
    GS = pytest.importorskip("tanitad.train.grad_share")
    _p, _b, _doc, e3, d1 = built(v, sources)
    for doc_, steps in ((e3, 73280), (d1, 36000)):
        spec, every = dict(groups(doc_["argv"]))["--grad-share-at"], dict(groups(doc_["argv"]))["--grad-share-every"]
        assert spec == [MK.GRAD_SHARE_AT] and len(GS.parse_at(spec[0], int(every[0]), steps)) == 12


@pytest.mark.parametrize("v", SOURCES)
def test_D1_is_the_E3_argv_with_exactly_the_gates_four_flag_delta_and_no_save_at(v, sources):
    _p, _b, _doc, e3, d1 = built(v, sources)
    a, b = groups(d1["argv"]), groups(e3["argv"])
    assert [f for f, _ in a] == [f for f, _ in b if f != "--save-at"] + ["--wsd-branch-from", "--wsd-branch-decay"]
    delta = {"--out", "--steps"}
    assert {f for (f, va), (g, vb) in zip(a, [x for x in b if x[0] != "--save-at"]) if va != vb} == delta
    assert [g for g in a[:-2] if g[0] not in delta] == [g for g in b if g[0] not in delta | {"--save-at"}]
    d = dict(a)
    assert d["--out"] == [(E3_OUT % v) + "-d1"] and d["--steps"] == ["36000"] and d["--wsd-branch-decay"] == ["6000"]
    assert d["--wsd-branch-from"] == [(E3_OUT % v) + "/ckpt_full_30000.pt"]
    assert set(d) ^ set(dict(b)) == {"--save-at", "--wsd-branch-from", "--wsd-branch-decay"}
    assert delta | {"--wsd-branch-from", "--wsd-branch-decay"} == set(BRANCH["delta"])             # the gate's 4-flag delta, by name
    assert dict(a)["--lr-schedule"] == ["wsd"] and dict(a)["--wsd-decay-steps"] == ["12458"] and dict(a)["--warmup"] == ["1000"]


def test_the_composers_constants_are_the_gates_own_literals():
    F, D = LG.PROFILES["refcv9-full"], LG.PROFILES["refcv9-full-d1"]
    assert {f: v for f, v in MK.SCHEDULE_ROWS} == {f: F["required_values"][f] for f in ("--steps", "--warmup", "--wsd-decay-steps")}
    assert D["required_values"]["--steps"] == str(MK.D1_STEPS) == str(BRANCH["steps"]) and D["required_values"]["--wsd-branch-decay"] == str(MK.D1_DECAY)
    assert (MK.SRC_STEP, MK.D1_DECAY, MK.D1_STEPS) == (BRANCH["src_step"], BRANCH["decay"], BRANCH["steps"]) == (30000, 6000, 36000)
    assert LRS.FULL_CKPT_FMT % MK.SRC_STEP == BRANCH["src_ckpt"] and F["lr_schedule_rule"]["save_at_includes"] == MK.SRC_STEP
    assert tuple(MK.BRANCH_FLAGS) == tuple(LG._R9F_BRANCH_FLAGS) and (MK.E3_PROFILE, MK.D1_PROFILE) == (F["name"], D["name"])
    assert LG._wsd_probe_expected(MK.E3_WARMUP, MK.E3_STEPS, MK.E3_DECAY) == LG._R9F_PROBE_FULL
    assert LG._wsd_probe_expected(MK.E3_WARMUP, MK.D1_STEPS, MK.D1_DECAY) == LG._R9F_PROBE_D1
    assert MK.SRC_STEP in range(MK.E3_WARMUP, MK.E3_STEPS - MK.E3_DECAY)                          # 30,000 is inside E3's stable phase [1000, 60822)


@pytest.mark.parametrize("v", SOURCES)
def test_the_docs_keep_the_E2_layout_and_record_their_provenance(v, sources):
    p, b, doc, e3, d1 = built(v, sources)
    assert list(e3)[:5] == ["schema", "arm", "seed", "steps", "argv"] and e3["schema"] == d1["schema"] == "r9e3-argv-v1"
    assert (e3["arm"], e3["steps"], e3["profile"], e3["derived_from"]) == ("r9e3_" + doc["arm"].split("_", 1)[1], 73280, "refcv9-full", doc["arm"])
    assert (d1["arm"], d1["steps"], d1["profile"], d1["derived_from"]) == (e3["arm"] + "_d1", 36000, "refcv9-full-d1", e3["arm"])
    assert e3["base_sha256"] == hashlib.sha256(b).hexdigest() and e3["e2_source"] == {"path": str(p).replace("\\", "/"), "sha256": hashlib.sha256(b).hexdigest()}
    assert d1["base_sha256"] == hashlib.sha256(MK.dump(e3).encode("utf-8")).hexdigest() and d1["e2_source"] == e3["e2_source"]
    assert e3["lever"] == d1["lever"] == doc["lever"] if "lever" in doc else "lever" not in e3 and "lever" not in d1
    assert json.loads(MK.dump(e3)) == e3 and json.loads(MK.dump(d1)) == d1 and MK.dump(e3).endswith("}\n") and "\r" not in MK.dump(e3) + MK.dump(d1)
    assert e3["seed"] == d1["seed"] == doc["seed"] and len(e3["set_lines"]) == 5 and len(d1["set_lines"]) == 5


@pytest.mark.parametrize("v", SOURCES)
def test_the_W_COST_stamp_the_lever_block_and_the_drivable_weight_are_carried_through_whatever_the_sources_content(v, sources):
    p, _b, doc, e3, d1 = built(v, sources)
    for k in MK.W_COST_KEYS:
        assert (k in e3) == (k in doc) == (k in d1), k
        if k in doc:
            assert e3[k] == d1[k] == doc[k]
    src_w = LG.flag_values(doc["argv"], "--w-r8-drivable")                                       # read from the SOURCE: never pinned here
    for rec in (e3, d1):
        assert LG.flag_values(rec["argv"], "--w-r8-drivable") == src_w and LG.flag_values(rec["argv"], "--r8-critic-drivable") == LG.flag_values(doc["argv"], "--r8-critic-drivable")
    assert (src_w is None) == (v == "K0")
    if v == "K2HOLDL":
        assert src_w == [HOLD_L_W] and e3["w_cost_ratio"] == HOLD_L_R and e3["w_cost_source"] == "F:/synthetic/k2_explore_smoke.log" and "w_cost" not in e3


def test_main_notes_a_provisional_or_unstamped_weight_and_stays_quiet_for_a_measured_one(sources, tmp_path, capsys):
    for v, loud in (("K2", "w_cost_ratio" not in json.loads(sources["K2"].read_text(encoding="utf-8"))), ("K2HOLDL", False), ("K0", False)):
        assert MK.main(["--e2", str(sources[v]), "--out", E3_OUT % v, "--write-e3", str(tmp_path / f"{v}.e3.json")]) == 0
        o = capsys.readouterr().out
        assert ("NOTE:" in o and "NOT a measured W-COST weight" in o) == loud, (v, o)
    prov = json.loads(sources["K2"].read_text(encoding="utf-8"))
    prov.pop("w_cost_ratio", None), prov.pop("w_cost_source", None)
    prov["w_cost"] = "provisional"
    pp = tmp_path / "r9e2_K2.argv.json"
    MK2.write(pp, prov)
    assert MK.main(["--e2", str(pp), "--out", E3_OUT % "K2", "--write-e3", str(tmp_path / "prov.e3.json")]) == 0
    assert "w_cost: provisional" in capsys.readouterr().out and json.loads((tmp_path / "prov.e3.json").read_text(encoding="utf-8"))["w_cost"] == "provisional"


# ===================================================================== the launch gate admits both, statically ==
@pytest.mark.parametrize("v", SOURCES)
def test_the_E3_argv_passes_every_static_gate_check_of_profile_refcv9_full(env, v, sources):
    _p, _b, _doc, e3, _d1 = built(v, sources)
    a = e3["argv"]
    r, p = rules(env, "FULL", a)
    assert r == [] and p == [], r
    lr, det = LG.lr_schedule_reasons(env["FULL"], a)
    assert lr == [] and det["literal"] == det["want"] == LG._R9F_PROBE_FULL                      # the profile's probe literal IS this argv's closed form
    assert LG.map_weight_reasons(env["FULL"], a)[0] == []
    assert items(env["FULL"], a) == OPEN_ITEMS[v]                                                  # K0 / K2: none; banded: E1-MAP-OWN-RECORD alone
    assert LG.lever_attach_reasons(env["FULL"], a, TRAINER)[0] == []                               # every declared lever's attach hop is in the trainer
    assert LG.wsd_branch_reasons(env["FULL"], a)[0] == []                                          # inert outside refcv9-full-d1 (no rule)
    assert LG.flag_values(a, "--save-at") == ["30000"] and not any(LG.has_flag(a, f) for f in LG._R9F_BRANCH_FLAGS)


@pytest.mark.parametrize("v", SOURCES)
def test_the_D1_argv_passes_every_static_gate_check_of_profile_refcv9_full_d1_incl_the_delta_against_its_source_run(env, v, sources):
    _p, _b, _doc, e3, d1 = built(v, sources)
    a, pm = d1["argv"], stage(env, v, e3["argv"])
    r, p = rules(env, "D1", a, pm)
    assert r == [] and p == [], r
    lr, det = LG.lr_schedule_reasons(env["D1"], a)
    assert lr == [] and det["literal"] == det["want"] == LG._R9F_PROBE_D1
    wr, wd = LG.wsd_branch_reasons(env["D1"], a, pm)
    assert wr == [] and set(wd["delta"]) == set(BRANCH["delta"]) | set(BRANCH["exempt"]), wd       # the 4 delta flags + the exempt --save-at, nothing else
    assert items(env["D1"], a) == OPEN_ITEMS[v] and LG.lever_attach_reasons(env["D1"], a, TRAINER)[0] == []
    assert LG.map_weight_reasons(env["D1"], a)[0] == []


@pytest.mark.parametrize("v", ("K0", "K2"))
def test_the_G_LIVE_argv_rewrite_of_both_argvs_survives_the_trainers_schedule_pins(v, sources):
    T = pytest.importorskip("refc_v3_train")
    pytest.importorskip("torch")
    _p, _b, _doc, e3, d1 = built(v, sources)
    with pytest.raises(SystemExit):                                                                # the defect the rewrite fixes: the launch argv at --steps 100
        LRS.check_args(T.build_parser().parse_args(LG.set_flag(e3["argv"], "--steps", ["100"])))
    for a in (e3["argv"], d1["argv"]):
        run, _dep = LG.smoke_argv(a, 100, 50)
        LRS.check_args(T.build_parser().parse_args(run))
        assert not any(LG.has_flag(run, f) for f in LG._R9F_BRANCH_FLAGS) and LG.flag_values(run, "--steps") == ["100"]


# ======================================================================================== RED arms (by name) ==
@pytest.mark.parametrize("v", ("K0", "K2"))
def test_RED_E3_without_save_at_or_without_30000_is_refused_naming_it(env, v, sources):
    _p, _b, _doc, e3, _d1 = built(v, sources)
    r, _ = rules(env, "FULL", drop_save_at(e3["argv"]))
    assert any(x.startswith("profile refcv9-full requires --save-at") for x in r) and any(x.startswith("WSD-ROW: --save-at <absent> does not name 30000") for x in r), r
    for bad in ("500", "500,1000", "29999,30001", "60000"):
        r, _ = rules(env, "FULL", LG.set_flag(e3["argv"], "--save-at", [bad]))
        assert len(r) == 1 and r[0].startswith("WSD-ROW: --save-at %s does not name 30000" % bad), (bad, r)
    r, _ = rules(env, "FULL", LG.set_flag(e3["argv"], "--save-at", ["1000,30000,60000"]))          # control: more steps is fine, 30000 present
    assert r == []


@pytest.mark.parametrize("v", ("K0", "K2"))
@pytest.mark.parametrize("flag,bad", [("--wsd-decay-steps", "12457"), ("--wsd-decay-steps", "12459"), ("--wsd-decay-steps", "1000"), ("--steps", "73281"),
                                      ("--steps", "6000"), ("--warmup", "999"), ("--warmup", "2000")])
def test_RED_E3_with_a_wrong_schedule_row_is_refused_by_the_required_value_and_the_wsd_row(env, v, sources, flag, bad):
    _p, _b, _doc, e3, _d1 = built(v, sources)
    r, _ = rules(env, "FULL", LG.set_flag(e3["argv"], flag, [bad]))
    assert any(x.startswith("%s %s is REQUIRED for a refcv9-full launch" % (flag, dict(MK.SCHEDULE_ROWS)[flag])) and ("['%s']" % bad) in x for x in r), r
    assert any(x.startswith("WSD-ROW:") for x in r), r                                               # the probe literal is the closed form of (1000, 73280, 12458): any other row moves it


@pytest.mark.parametrize("v", ("K0", "K2"))
@pytest.mark.parametrize("flag,val", [("--lr", "2e-4"), ("--seed", "1"), ("--batch", "20"), ("--ema-decay", "0.999"), ("--warmup", "999"), ("--trunk-name", "resnet50.a1_in1k")])
def test_RED_a_D1_argv_with_a_fifth_differing_flag_is_refused_by_the_gate_and_by_the_trainers_own_preflight(env, v, sources, flag, val):
    _p, _b, _doc, e3, d1 = built(v, sources)
    pm = stage(env, v, e3["argv"])
    bad = LG.set_flag(d1["argv"], flag, [val])
    r, _ = rules(env, "D1", bad, pm)
    assert any(x.startswith("D1-BRANCH: the branch argv differs from its source's in ['%s']" % flag) for x in r), r
    r, _ = LG.wsd_branch_reasons(env["D1"], bad, pm)
    assert len(r) == 1 and "4-flag delta" in r[0], r
    assert LG.wsd_branch_reasons(env["D1"], d1["argv"], pm)[0] == []                                  # control: the composed D1 passes the same row


@pytest.mark.parametrize("v", ("K0", "K2"))
@pytest.mark.parametrize("steps", ["36001", "35999", "30000", "73280", "6000"])
def test_RED_a_D1_argv_whose_steps_is_not_30000_plus_6000_is_refused(env, v, sources, steps):
    _p, _b, _doc, e3, d1 = built(v, sources)
    pm = stage(env, v, e3["argv"])
    r, _ = rules(env, "D1", LG.set_flag(d1["argv"], "--steps", [steps]), pm)
    assert any(x.startswith("--steps 36000 is REQUIRED for a refcv9-full-d1 launch") for x in r), r
    assert any(x.startswith("D1-BRANCH: --steps %s / --wsd-branch-decay 6000 are not the branch's 36000 = 30000 + 6000" % steps) for x in r), r


@pytest.mark.parametrize("v", ("K0", "K2"))
def test_RED_a_D1_with_another_decay_another_source_checkpoint_or_the_same_out_is_refused(env, v, sources):
    _p, _b, _doc, e3, d1 = built(v, sources)
    pm = stage(env, v, e3["argv"])
    r, _ = rules(env, "D1", LG.set_flag(LG.set_flag(d1["argv"], "--wsd-branch-decay", ["5000"]), "--steps", ["35000"]), pm)
    assert any("D1-BRANCH: --steps 35000 / --wsd-branch-decay 5000" in x for x in r) and any(x.startswith("--wsd-branch-decay 6000 is REQUIRED") for x in r), r
    r, _ = rules(env, "D1", LG.set_flag(d1["argv"], "--wsd-branch-from", [(E3_OUT % v) + "/ckpt_full_60000.pt"]), pm)
    assert any("is not the stable-phase checkpoint ckpt_full_30000.pt" in x for x in r), r
    r, _ = rules(env, "D1", LG.set_flag(d1["argv"], "--out", [E3_OUT % v]), pm)
    assert any("delta flag(s) ['--out'] are identical to the source's -- not a new run" in x for x in r), r
    (env["tmp"] / f"e3run_{v}" / "config.json").unlink()
    r, _ = rules(env, "D1", d1["argv"], pm)
    assert any("no config.json beside the source checkpoint" in x for x in r), r                    # unbindable -> refused, never skipped
    stage(env, v, e3["argv"])


@pytest.mark.parametrize("v", ("K0", "K2"))
def test_RED_each_argv_is_refused_under_the_other_profile(env, v, sources):
    _p, _b, _doc, e3, d1 = built(v, sources)
    r, _ = rules(env, "D1", e3["argv"], stage(env, v, e3["argv"]))
    assert any(x.startswith("profile refcv9-full-d1 requires --wsd-branch-from") for x in r) and any(x.startswith("D1-BRANCH: --wsd-branch-from is not passed") for x in r), r
    r, _ = rules(env, "FULL", d1["argv"])
    assert any("--wsd-branch-from is passed" in x and "refcv9-full-d1" in x for x in r) and any(x.startswith("--steps 73280 is REQUIRED") for x in r), r
    for prof in ("E2",):                                                                             # the E2 profile refuses both (its rows are 6000 / 1000)
        assert rules(env, prof, e3["argv"])[0] and rules(env, prof, d1["argv"], stage(env, v, e3["argv"]))[0]


def test_RED_the_init_pin_and_a_dropped_lever_flag_still_bite_on_the_composed_argvs(env, sources):
    _p, _b, _doc, e3, d1 = built("K2", sources)
    (env["tmp"] / "other_ema.pt").write_bytes(b"different")
    r, _ = rules(env, "FULL", LG.set_flag(e3["argv"], "--init-from", ["/home/nvidia/r8ns/ckpt/other_ema.pt"]))
    assert any("the profile pins" in x for x in r), r                                                  # the init pin is bound by content on E3 too
    for missing in ("--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only"):
        a = e3["argv"]
        i = a.index(missing)
        a = a[:i] + a[i + 1 + (1 if missing == "--r9-bev-warp-fuse-init" else 0):]
        r, _ = rules(env, "FULL", a)
        assert any(x.startswith(missing) and "is REQUIRED" in x for x in r), (missing, r)
    r, _ = rules(env, "FULL", e3["argv"] + ["--r9-bev-warp-shuffle"])
    assert any(x.startswith("--r9-bev-warp-shuffle is passed") for x in r), r


# ===================================== a regressed COMPOSER is caught by the gate (the composer's own constants) ==
def _gate_refuses(env, v, sources, monkeypatch, patches, which, **kw):
    for name, val in patches.items():
        monkeypatch.setattr(MK, name, val)
    p = sources[v]
    b = p.read_bytes()
    e3 = MK.e3_doc(json.loads(b.decode("utf-8")), b, p, E3_OUT % v, **kw)
    a = e3["argv"] if which == "E3" else MK.d1_doc(e3)["argv"]
    return rules(env, "FULL" if which == "E3" else "D1", a, stage(env, v, e3["argv"]))[0]


def test_RED_a_composer_with_a_wrong_decay_or_steps_or_save_at_step_or_branch_row_is_refused_by_the_gate(env, sources, monkeypatch):
    rows = lambda **ch: tuple((f, ch.get(f, v)) for f, v in MK.SCHEDULE_ROWS)   # noqa: E731
    assert _gate_refuses(env, "K2", sources, monkeypatch, {}, "E3") == [] and _gate_refuses(env, "K2", sources, monkeypatch, {}, "D1") == []        # control
    r = _gate_refuses(env, "K2", sources, monkeypatch, {"SCHEDULE_ROWS": rows(**{"--wsd-decay-steps": "12457"})}, "E3")
    assert any(x.startswith("--wsd-decay-steps 12458 is REQUIRED") for x in r) and any(x.startswith("WSD-ROW:") for x in r), r
    monkeypatch.undo()
    r = _gate_refuses(env, "K2", sources, monkeypatch, {"SCHEDULE_ROWS": rows(**{"--steps": "73281"})}, "E3")
    assert any(x.startswith("--steps 73280 is REQUIRED") for x in r), r
    monkeypatch.undo()
    r = _gate_refuses(env, "K2", sources, monkeypatch, {"SRC_STEP": 29999}, "E3", save_at="29999")                                                 # the branch source step moved
    assert any(x.startswith("WSD-ROW: --save-at 29999 does not name 30000") for x in r), r
    monkeypatch.undo()
    r = _gate_refuses(env, "K2", sources, monkeypatch, {"D1_STEPS": 36001}, "D1")
    assert any(x.startswith("--steps 36000 is REQUIRED for a refcv9-full-d1") for x in r) and any(x.startswith("D1-BRANCH: --steps 36001") for x in r), r
    monkeypatch.undo()
    r = _gate_refuses(env, "K2", sources, monkeypatch, {"D1_DECAY": 5000}, "D1")
    assert any(x.startswith("--wsd-branch-decay 6000 is REQUIRED") for x in r) and any("D1-BRANCH: --steps 36000 / --wsd-branch-decay 5000" in x for x in r), r
    monkeypatch.undo()
    assert _gate_refuses(env, "K2", sources, monkeypatch, {}, "D1") == []                                                                            # undone: green again


# ================================================================ the composer's own refusals (SystemExit) ==
def _doc(v="K2"):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def _compose(doc, out=E3_OUT % "K2", **kw):
    b = MK2.dump(doc).encode("utf-8")
    return MK.build(doc, b, "x.json", out, **kw)


def test_the_composer_refuses_a_source_that_is_not_an_E2_argv_doc():
    d = _doc()
    for chg, msg in ((dict(profile="refcv9-full"), "profile refcv9-e2"), (dict(schema="r9e3-argv-v1"), "r9e2-argv-v1"), (dict(arm="K2"), "not r9e2_<variant>"), (dict(argv=[]), "no argv"),
                     (dict(argv=["--arm", "hier", 3]), "no argv")):
        with pytest.raises(SystemExit, match=msg):
            _compose(dict(d, **chg))
    with pytest.raises(SystemExit, match="profile refcv9-e2"):
        _compose(json.loads(MK2.dump(_compose(d)[0])))                                             # an E3 doc is not an E2 doc
    for f, val in (("--save-at", ["30000"]), ("--wsd-branch-from", ["/x/ckpt_full_30000.pt"]), ("--wsd-branch-decay", ["6000"])):
        with pytest.raises(SystemExit, match="already carries"):
            _compose(dict(d, argv=LG.set_flag(d["argv"], f, val)))
    with pytest.raises(SystemExit, match="--flag=value"):
        _compose(dict(d, argv=[t if t != "--lr" else "--lr=1e-4" for t in d["argv"] if t != "1e-4"]))
    with pytest.raises(SystemExit, match="not E2's 6000 / 1000"):
        _compose(dict(d, argv=LG.set_flag(d["argv"], "--steps", ["7000"])))
    with pytest.raises(SystemExit, match="not E2's 6000 / 1000"):
        _compose(dict(d, argv=LG.set_flag(d["argv"], "--wsd-decay-steps", ["999"])))
    with pytest.raises(SystemExit, match=r"\['cosine'\]|not \['wsd'\]"):
        _compose(dict(d, argv=LG.set_flag(d["argv"], "--lr-schedule", ["cosine"])))
    for f in ("--out", "--init-from", "--warmup", "--steps"):
        with pytest.raises(SystemExit, match="exactly once"):
            _compose(dict(d, argv=LG.set_flag(d["argv"], f, None)))
        with pytest.raises(SystemExit, match="exactly once"):
            _compose(dict(d, argv=d["argv"] + [f, "1"]))
    with pytest.raises(SystemExit, match="bare|starts with"):
        _compose(dict(d, argv=["x"] + d["argv"]))


@pytest.mark.parametrize("out,msg", [("C:/Program Files/Git/home/nvidia/r9e3", "not an absolute POSIX path"), ("r9e3/runs/x", "not an absolute POSIX path"),
                                     ("/home/nvidia/r9e3/", "clean absolute POSIX run directory"), ("/home/nvidia/r9 e3", "clean absolute POSIX run directory"),
                                     ("/home/nvidia//r9e3", "clean absolute POSIX run directory"), ("/home/nvidia/../r9e3", "clean absolute POSIX run directory"),
                                     ("/home/nvidia/r8ns/r9e2/runs/r9e2-K2", "E2 source's own run directory")])
def test_the_composer_refuses_a_bad_out(out, msg):
    with pytest.raises(SystemExit, match=msg):
        _compose(_doc(), out=out)


def test_the_composer_refuses_a_bad_save_at_and_a_bad_d1_out():
    for spec, msg in (("500", "does not name 30000"), ("", "does not name 30000"), ("30000,30000", "duplicate"), ("30000,73281", "outside"), ("0,30000", "outside"), ("30000,x", "not a comma list")):
        with pytest.raises(SystemExit, match=msg):
            _compose(_doc(), save_at=spec)
    with pytest.raises(SystemExit, match="does not name 30000"):
        MK.compose_e3(_doc(), E3_OUT % "K2", "500")                                                    # E3 alone refuses it too (not only D1)
    e3, d1 = _compose(_doc(), save_at="60000, 30000,1000,")                                          # canonical: sorted, blanks dropped
    assert LG.flag_values(e3["argv"], "--save-at") == ["1000,30000,60000"] and LG.has_flag(d1["argv"], "--save-at") is False
    assert _compose(_doc(), save_at="30000")[0]["argv"] == _compose(_doc())[0]["argv"]
    with pytest.raises(SystemExit, match="is the E3 run directory"):
        _compose(_doc(), d1_out=E3_OUT % "K2")
    with pytest.raises(SystemExit, match="clean absolute POSIX"):
        _compose(_doc(), d1_out="/home/nvidia/r9e3-d1/")
    assert LG.flag_values(_compose(_doc(), d1_out="/home/nvidia/r8ns/r9e3/runs/mine")[1]["argv"], "--out") == ["/home/nvidia/r8ns/r9e3/runs/mine"]
    e3 = _compose(_doc())[0]["argv"]
    for chg, msg in ((lambda a: LG.set_flag(a, "--steps", ["6000"]), "not 73280"), (lambda a: drop_save_at(a), "exactly once"), (lambda a: LG.set_flag(a, "--save-at", ["500"]), "does not name 30000"),
                     (lambda a: a + ["--wsd-branch-from", "/x"], "already carries")):
        with pytest.raises(SystemExit, match=msg):
            MK.compose_d1(chg(e3))


# ====================================================================== byte-equal regeneration; E2 untouched ==
@pytest.mark.parametrize("v", ("K0", "K2", "K2HOLDL"))
def test_composing_twice_gives_the_same_bytes_and_the_E2_source_is_untouched(v, sources, tmp_path):
    p = sources[v]
    before = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)
    outs = []
    for i in (1, 2):
        e3p, d1p = tmp_path / f"e3_{i}.argv.json", tmp_path / f"d1_{i}.argv.json"
        assert MK.main(["--e2", str(p), "--out", E3_OUT % v, "--write-e3", str(e3p), "--write-d1", str(d1p)]) == 0
        outs.append((e3p.read_bytes(), d1p.read_bytes()))
    assert outs[0] == outs[1] and (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size) == before
    _p, _b, _doc_, e3, d1 = built(v, sources)
    assert outs[0] == (MK.dump(e3).encode("utf-8"), MK.dump(d1).encode("utf-8")) and b"\r" not in outs[0][0] + outs[0][1]
    assert MK.dump(json.loads(outs[0][0].decode("utf-8"))).encode("utf-8") == outs[0][0]           # the file is exactly dump(its own parse)
    assert MK.build(*built(v, sources)[2:3], p.read_bytes(), p, E3_OUT % v)[0] == e3                # in-memory: build is a pure function of its inputs


def test_check_mode_is_a_byte_equality_gate_and_never_writes(sources, tmp_path, capsys):
    p = sources["K2"]
    e3p, d1p = tmp_path / "e3.json", tmp_path / "d1.json"
    args = ["--e2", str(p), "--out", E3_OUT % "K2", "--write-e3", str(e3p), "--write-d1", str(d1p)]
    assert MK.main(args) == 0
    capsys.readouterr()
    assert MK.main(args + ["--check"]) == 0 and capsys.readouterr().out.count("regenerates BYTE-EQUAL") == 2
    good = e3p.read_bytes()
    e3p.write_bytes(good + b" ")
    with pytest.raises(SystemExit, match="does NOT regenerate byte-equal"):
        MK.main(args + ["--check"])
    assert e3p.read_bytes() == good + b" "                                                           # --check wrote nothing
    e3p.write_bytes(good)
    with pytest.raises(SystemExit, match="does NOT regenerate byte-equal"):
        MK.main(["--e2", str(p), "--out", E3_OUT % "K2" + "x", "--write-e3", str(e3p), "--check"])    # another --out is another file
    d1p.unlink()
    with pytest.raises(SystemExit, match="absent"):
        MK.main(args + ["--check"])


def test_the_cli_refuses_to_overwrite_the_source_and_needs_a_target(sources, tmp_path):
    p = tmp_path / "src" / "r9e2_K2.argv.json"                                                       # a COPY: a regressed guard must never clobber a committed file
    p.parent.mkdir()
    p.write_bytes(sources["K2"].read_bytes())
    base = ["--e2", str(p), "--out", E3_OUT % "K2"]
    before = p.read_bytes()
    for tgt in (["--write-e3", str(p)], ["--write-d1", str(p)]):
        with pytest.raises(SystemExit, match="refusing to write over it"):
            MK.main(base + tgt)
    with pytest.raises(SystemExit, match="same file"):
        MK.main(base + ["--write-e3", str(tmp_path / "x.json"), "--write-d1", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        MK.main(base)                                                                                  # argparse error (exit 2): no --write-*
    assert p.read_bytes() == before == sources["K2"].read_bytes() and not list(tmp_path.glob("*.json"))
    assert MK.main(base + ["--write-d1", str(tmp_path / "only_d1.json")]) == 0 and (tmp_path / "only_d1.json").is_file()


def test_the_script_runs_as_a_CLI_and_writes_the_same_bytes(sources, tmp_path):
    p = sources["K2"]
    e3p, d1p = tmp_path / "cli_e3.json", tmp_path / "cli_d1.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "r9_e3_make_argv.py"), "--e2", str(p), "--out", E3_OUT % "K2", "--write-e3", str(e3p), "--write-d1", str(d1p)],
                         capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, MSYS_NO_PATHCONV="1", CUDA_VISIBLE_DEVICES="-1"), timeout=120)
    assert res.returncode == 0, res.stderr
    _p, _b, _doc_, e3, d1 = built("K2", sources)
    assert (e3p.read_bytes(), d1p.read_bytes()) == (MK.dump(e3).encode("utf-8"), MK.dump(d1).encode("utf-8"))
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "r9_e3_make_argv.py"), "--e2", str(p), "--out", "C:/Program Files/Git/home/nvidia/x", "--write-e3", str(tmp_path / "bad.json")],
                         capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), timeout=120)
    assert bad.returncode != 0 and "absolute POSIX path" in bad.stderr and not (tmp_path / "bad.json").exists()


# ============================== the trainer's own parser and lr_schedule accept both (importorskip torch / timm) ==
@pytest.fixture(scope="module")
def trainer():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    return pytest.importorskip("refc_v3_train")


@pytest.mark.parametrize("v", SOURCES)
def test_the_trainer_parses_both_argvs_and_lr_schedule_check_args_raises_nothing(trainer, v, sources):
    _p, _b, doc, e3, d1 = built(v, sources)
    parser = trainer.build_parser()
    a3, a1 = parser.parse_args(e3["argv"]), parser.parse_args(d1["argv"])
    LRS.check_args(a3)
    LRS.check_args(a1)
    assert (a3.steps, a3.warmup, a3.lr_schedule, a3.wsd_decay_steps, a3.save_at, a3.wsd_branch_from, a3.wsd_branch_decay) == (73280, 1000, "wsd", 12458, "30000", None, 0)
    assert (a1.steps, a1.warmup, a1.lr_schedule, a1.wsd_decay_steps, a1.save_at, a1.wsd_branch_from, a1.wsd_branch_decay) == (
        36000, 1000, "wsd", 12458, None, (E3_OUT % v) + "/ckpt_full_30000.pt", 6000)
    assert LRS.parse_save_at(a3.save_at, a3.steps) == frozenset({30000}) and a3.init_from == a1.init_from == MK2.INIT_FROM and a3.out == E3_OUT % v and a1.out == (E3_OUT % v) + "-d1"
    src_w = LG.flag_values(doc["argv"], "--w-r8-drivable")
    assert (a3.w_r8_drivable if src_w else None) == (float(src_w[0]) if src_w else None) == (a1.w_r8_drivable if src_w else None)        # the weight is the SOURCE's, whatever it is
    assert LRS.assert_sched_matches_args(LRS.make_sched(a3), a3) == {str(k): x for k, x in LG._R9F_PROBE_FULL.items()}                 # the BUILT schedule IS the profile's literal
    assert LRS.assert_sched_matches_args(LRS.make_sched(a1), a1) == {str(k): x for k, x in LG._R9F_PROBE_D1.items()}


@pytest.mark.parametrize("v", ("K0", "K2"))
def test_the_trainers_real_branch_preflight_accepts_D1_against_its_E3_source_and_refuses_the_regressions(trainer, v, sources, tmp_path):
    _p, _b, _doc, e3, d1 = built(v, sources)
    parser = trainer.build_parser()
    run = tmp_path / "e3run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"argv_effective": e3["argv"]}), encoding="utf-8")
    ck = run / "ckpt_full_30000.pt"
    ck.write_bytes(b"x")

    def pre(argv, step=30000):
        a = parser.parse_args(argv)
        a.wsd_branch_from = str(ck)                                                                    # the Thor path -> the staged local one
        rec = LRS.branch_preflight(a, parser)
        LRS.check_branch_checkpoint(a, rec["src_ns"], step, keys=LRS.BRANCH_REQUIRED_KEYS)
        return rec

    assert pre(d1["argv"])["src_ns"].steps == 73280                                                     # control: the composed D1 passes the trainer's own checks
    with pytest.raises(SystemExit, match="differs from the source run's"):
        pre(LG.set_flag(d1["argv"], "--lr", ["2e-4"]))                                                  # a 5th differing flag
    with pytest.raises(SystemExit, match=r"--steps 36001 != checkpoint step 30000 \+ --wsd-branch-decay 6000 = 36000"):
        pre(LG.set_flag(d1["argv"], "--steps", ["36001"]))                                              # steps != 30000 + 6000
    with pytest.raises(SystemExit, match="never written"):
        pre(d1["argv"] + ["--save-at", "30000"])                                                        # why the composer drops --save-at from D1
    with pytest.raises(SystemExit, match="DECAY phase"):
        pre(d1["argv"], step=65000)                                                                     # a checkpoint inside E3's decay phase (>= 60,822) cannot be branched
    with pytest.raises(SystemExit, match="WARMUP phase"):
        pre(d1["argv"], step=500)
    stable = [s for s in (1000, 30000, 60821) if LRS.phase_of(s, 1000, 73280, 12458) == "stable"]
    assert stable == [1000, 30000, 60821] and LRS.phase_of(60822, 1000, 73280, 12458) == "decay"


@pytest.mark.parametrize("v", SOURCES)
def test_G_LIVE_coverage_and_the_decoder_passthrough_hold_for_both_argvs(trainer, env, v, sources):
    _p, _b, _doc, e3, d1 = built(v, sources)
    for key, rec in (("FULL", e3), ("D1", d1)):
        args = trainer.build_parser().parse_args(rec["argv"])
        terms, problems = LG.declared_terms(trainer, args)
        assert problems == [], problems
        ids = [t["id"] for t in terms]
        assert ("w_r8_drivable" in ids) == (v != "K0") and "w_map_hires" in ids and "w_box3d" in ids and "perc_rig" not in ids
        try:
            pr, _det = LG.passthrough_reasons(env[key], rec["argv"], args, trainer)
        except ImportError as e:
            pytest.skip("passthrough_reasons needs tanitad.refs.refc: %s" % e)
        assert pr == [], pr


@pytest.mark.parametrize("v", [x for x in SOURCES if x not in ("K0", "K2M1N20", "K2M3", "K2M3N20")])
def test_the_C8_planner_wire_pins_accept_the_composed_K2_argvs(trainer, v, sources):
    from tanitad.train import plan_wires_flag as PW
    _p, _b, _doc, e3, d1 = built(v, sources)
    for rec in (e3, d1):
        args = trainer.build_parser().parse_args(rec["argv"])
        assert PW.box_on(args) and PW.map_on(args) is False and args.r8_critic_drivable
        PW.pin_args(args)
