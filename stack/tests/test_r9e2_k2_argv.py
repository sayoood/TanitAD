"""K2-ARGV (MM 4a3d05, 2026-11-07, PI: "you can generate the code so K2 can launch") -- K2's argv for every E1 map outcome, COMPOSED from K0 and admitted by the full-family
launch gate (refcv9-e2 / refcv9-full / refcv9-full-d1), and nothing broader.

  composer   stack/scripts/r9_e2_make_argv.py  (K0 regenerates BYTE-EQUAL; K2 / K2M1N20 / K2M3 / K2M3N20 = K0 + T1h bundle + W-BOX + W-MAP-1 [+ banded head + W-MAP-2])
  gate       launch_gate.py  full-family `required_values_when` = the FLIP's T1h bundle rules + the near-lift rules (banded -> 10, near-m=20 -> 20) + the W-MAP-2 co-requirement
  files      stack/ops/runs.d/r9e2_K2*.argv.json  (written BY the composer; this file regenerates them and compares bytes)

Every guard has a deliberate-regression arm; the mutation list (each run, each RED) is in HANDOFF.md. The banded variants' BUILD-level checks (trainer parse + bev_warp_flag.pin_args:
`--r9-bev-warp-supervise-hist` with `--r9-map-banded`) are PENDING the other agent's hold: they are REPORTED here (skip with the reason), never asserted red.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import r9_e1_make_argv as E1  # noqa: E402
import r9_e2_make_argv as MK  # noqa: E402
from test_launch_gate_refcv9_full import K0, R8_LAUNCH, _d1, _full_argv, _rules, env  # noqa: E402,F401  (env: the module fixture)

RUNS = ROOT / "ops" / "runs.d"
VARIANTS = ("K2", "K2M1N20", "K2M3", "K2M3N20")
FAMILY = ("E2", "FULL", "D1")
T1H = ["--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only"]
T1V = ["--r9-hist-supervise-geom"]
#: the variants that compose T1v INSTEAD of the T1h bundle (REFCV9_E1_T1V_PREREG.md committed outcome 0, 2026-11-09); K2 / K2M3 keep T1h
T1V_VARIANTS = ("K2M1N20", "K2M3N20")
T1H_VARIANTS = ("K2", "K2M3")
WIRES = ["--r9-plan-box-tokens", "--r8-critic-drivable", "--w-r8-drivable"]
#: variant -> (flags ADDED to K0, flags whose VALUE changed, the open items the gate raises)
EXPECT = {
    "K2": (T1H + WIRES, {"--out"}, []),
    "K2M1N20": (T1V + WIRES + ["--r9-map-banded", "--r9-map-banded-near-m", "--r9-plan-map-channels"], {"--out", "--w-map-hires"}, ["E1-MAP-OWN-RECORD"]),   # --w-map-hires 25: PREREG_K2_REDOSE
    "K2M3": (T1H + WIRES + ["--r9-map-banded", "--r9-map-own-enc", "--r9-plan-map-channels"], {"--out", "--map-hires-near-lift-m"}, ["E1-MAP-OWN-RECORD"]),
    "K2M3N20": (T1V + WIRES + ["--r9-map-banded", "--r9-map-own-enc", "--r9-map-banded-near-m", "--r9-plan-map-channels"], {"--out"}, ["E1-MAP-OWN-RECORD"]),
}


#: MEASURED W-COST (MM 2026-11-07/08): the gate's own G-LIVE smoke of each provisional argv at 6d0e629f printed "refcv9 W-COST at step 0: ... = R"; K2M3 stays the placeholder
#: K2M1N20 / K2M3N20: measured WITH T1v at 926820a981ae (2026-11-09; the T1h readings were the same two numbers, 1.545 / 1.5425, and were not carried over: each was re-measured on its own composition)
MEASURED_WCOST = {"K2": (1.4613, "0.06843", "6d0e629f37a2_K2_glive/logs/job_smoke.log"),
                  "K2M1N20": (1.545, "0.06472", "926820a981ae_K2M1N20_wcost/logs/job_smoke.log"),
                  "K2M3N20": (1.5425, "0.06483", "926820a981ae_K2M3N20_wcost/logs/job_smoke.log")}

def doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def av(v):
    return list(doc(v)["argv"])


def drop(argv, flag):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def items(prof, argv):
    return sorted(m.group(1) for m in (re.match(r"OPEN ITEM (\S+):", r) for r in LG.open_item_reasons(prof, argv)) if m)


def argv_for(prof_key, argv, env):
    return _full_argv(argv) if prof_key == "FULL" else (_d1(env, argv=_full_argv(argv))[0] if prof_key == "D1" else list(argv))


# ============================================================================================= the composer ==
def test_K0_regenerates_byte_equal_from_refcv8s_launch_argv():
    base = json.loads((RUNS / "refcv8-r101-s0.argv.json").read_text(encoding="utf-8"))
    got = MK.dump(MK.regen_k0(base)).encode("utf-8")
    assert got == (RUNS / "r9e2_K0.argv.json").read_bytes()
    assert doc("K0")["base_sha256"] == MK.base_sha256(base["argv"]) == "4758a324945bbd0fadf3e83495d8832d3cc5c582c5387ceb9a3aa77d60b5d66b"


@pytest.mark.parametrize("v", VARIANTS)
def test_each_committed_variant_regenerates_byte_equal_from_K0_by_the_composer(v):
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    d = doc(v)
    w = float(LG.flag_values(d["argv"], "--w-r8-drivable")[0])
    if "w_cost_ratio" in d:      # K2-WCOST MEASURED (MM 2026-11-07): written by main() --w-cost-ratio R --w-cost-source <the smoke log>
        r = float(d["w_cost_ratio"])
        assert w == MK.w_from_ratio(r)
        rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, v, w, provisional=False,
                             w_source="0.1 / %g (--w-cost-ratio: the K2 explore smoke's W-COST line)" % r)
        got = MK.dump(MK.stamp_w_cost(rec, r, d["w_cost_source"])).encode("utf-8")
    else:
        rec = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, v, w, provisional=True,
                             w_source="operator-supplied PROVISIONAL placeholder (no W-COST reading exists yet)")
        if d.get("w_cost") == "provisional":      # main() without --w-cost-ratio stamps it (the T1v files); the older K2M3 file predates the stamp
            MK.stamp_w_cost(rec, None, None)
        got = MK.dump(rec).encode("utf-8")
    assert got == (RUNS / f"r9e2_{v}.argv.json").read_bytes()
    assert d["arm"] == f"r9e2_{v}" and d["profile"] == "refcv9-e2" and d["steps"] == 6000 and d["derived_from"] == "r9e2_K0"
    assert d["base_sha256"] == hashlib.sha256(k0b).hexdigest()


@pytest.mark.parametrize("v", VARIANTS)
def test_the_variants_delta_vs_K0_is_exactly_the_stated_flag_set_and_every_other_token_is_K0s(v):
    added, changed, _ = EXPECT[v]
    a, b = dict(LG.flag_pairs(av(v))), dict(LG.flag_pairs(K0))
    assert set(a) - set(b) == set(added) and not (set(b) - set(a)), (v, sorted(set(a) ^ set(b)))
    assert {f for f in set(a) & set(b) if a[f] != b[f]} == changed
    assert a["--out"] == [f"/home/nvidia/r8ns/r9e2/runs/r9e2-{v}"]
    assert a["--w-map-hires"] == (["25.0"] if v == "K2M1N20" else ["1.0"]) and a["--bev-source"] == ["map_hires_pool"] and a["--steps"] == ["6000"] and a["--lr-schedule"] == ["wsd"]
    assert (a.get("--r9-bev-warp-fuse-init") == ["identity"]) == (v in T1H_VARIANTS) and a["--map-hires-near-lift-m"] == (["10"] if v == "K2M3" else ["20"])
    assert ("--r9-map-banded-near-m" in a) == (v in ("K2M1N20", "K2M3N20")) and a.get("--r9-map-banded-near-m", ["20"]) == ["20"]
    assert ("--r9-map-own-enc" in a) == (v in ("K2M3", "K2M3N20")) and ("--r9-plan-map-channels" in a) == (v != "K2")


@pytest.mark.parametrize("v,arm", [("K2M1N20", "M1N20"), ("K2M3", "M3"), ("K2M3N20", "M3N20")])
def test_the_banded_variants_carry_exactly_the_map_flags_r9_e1_make_argv_builds(v, arm):
    e1 = E1.map_arm_argv(K0, arm, "/home/nvidia/r8ns/r9e2/runs/r9e2-" + v, w=1.0 if arm in E1.OWN_ENC_ARMS else None)
    mine, theirs = dict(LG.flag_pairs(av(v))), dict(LG.flag_pairs(e1))
    for f in set(theirs) - set(dict(LG.flag_pairs(K0))) | {"--map-hires-near-lift-m", "--w-map-hires"}:
        if (v, f) == ("K2M1N20", "--w-map-hires"):      # re-dosed to 25 (PREREG_K2_REDOSE_2026-11-10); E1's M1N20 builder still says 1.0
            assert mine[f] == ["25.0"] and theirs[f] == ["1.0"]
            continue
        assert mine[f] == theirs[f], (v, f)


@pytest.mark.parametrize("v", T1V_VARIANTS)
def test_T1v_variants_carry_the_one_T1v_flag_and_NONE_of_the_four_T1h_flags(v):
    toks = av(v)
    assert toks.count("--r9-hist-supervise-geom") == 1
    for f in T1H + ["--r9-bev-warp-lags", "--r9-bev-warp-shuffle"]:
        assert f not in toks, (v, f)
    assert [t for t in E1.T1_ARMS["T1v"]] == T1V and MK.T1_ARM_OF[v] == "T1v"
    d = doc(v)
    assert "t1h_bundle" not in d["lever"] and d["lever"]["t1v"] == T1V and "T1v" in d["what"] and "W-COST: MEASURED with T1v" in d["what"] and "PROVISIONAL" not in d["what"]
    assert "w_cost" not in d and "w_cost_ratio" in d
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    prov = MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, v, 0.1, provisional=True, w_source="x")      # the opposite value: an unmeasured composition still says so
    assert "W-COST: PROVISIONAL" in prov["what"] and prov["lever"]["w_r8_drivable_status"].startswith("PROVISIONAL")


@pytest.mark.parametrize("v", T1H_VARIANTS)
def test_T1h_variants_are_unchanged_by_the_T1v_switch(v):
    assert "--r9-hist-supervise-geom" not in av(v) and all(f in av(v) for f in T1H) and MK.T1_ARM_OF[v] == "T1h"


def test_the_T1h_bundle_is_the_E1_arms_bundle_and_W_MAP_2_is_not_in_K2():
    assert [t for t in E1.T1_ARMS["T1h"] if t.startswith("--")] == T1H
    assert "--r9-plan-map-channels" not in av("K2") and "--r9-map-banded" not in av("K2")
    assert doc("K2")["lever"]["w_map_2"].startswith("NOT AVAILABLE without the banded head")


def test_W_MAP_1_weight_is_a_measured_quantity_and_the_files_say_PROVISIONAL():
    assert MK.w_from_ratio(0.5) == 0.2 and MK.w_from_ratio(2.0) == 0.05 and MK.w_from_ratio(0.123456) == float("%.4g" % (0.1 / 0.123456))
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(SystemExit):
            MK.w_from_ratio(bad)
    for v in VARIANTS:
        d = doc(v)
        if v in MEASURED_WCOST:
            r, w_text, src = MEASURED_WCOST[v]
            assert d["w_cost_ratio"] == r and d["w_cost_source"].endswith(src)
            assert d["lever"]["w_r8_drivable_status"].startswith("MEASURED")
            assert LG.flag_values(d["argv"], "--w-r8-drivable") == [repr(MK.w_from_ratio(r))] == [w_text]
            continue
        assert d["lever"]["w_r8_drivable_status"].startswith("PROVISIONAL") and "W-COST" in d["lever"]["w_r8_drivable_status"]
        assert LG.flag_values(d["argv"], "--w-r8-drivable") == ["0.1"]


def test_the_composer_refuses_a_nonpositive_weight_a_lever_carrying_K0_an_unknown_variant_and_a_windows_out():
    k0 = json.loads((RUNS / "r9e2_K0.argv.json").read_text(encoding="utf-8"))
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(SystemExit, match="finite and > 0"):
            MK.compose(k0, "K2", bad)
    with pytest.raises(SystemExit, match="not in"):
        MK.compose(k0, "K2X", 0.1)
    dirty = dict(k0, argv=list(k0["argv"]) + ["--r9-bev-warp"])
    with pytest.raises(SystemExit, match="already carries lever"):
        MK.compose(dirty, "K2", 0.1)
    with pytest.raises(SystemExit, match="not an absolute POSIX path"):
        MK.compose(k0, "K2", 0.1, out="C:/Program Files/Git/home/nvidia/x")
    with pytest.raises(SystemExit, match="--wp-index on"):
        MK.compose(dict(k0, argv=list(k0["argv"]) + ["--wp-index", "on"]), "K2", 0.1)


# ===================================================================== the gate admits each variant statically ==
@pytest.mark.parametrize("prof", FAMILY)
@pytest.mark.parametrize("v", VARIANTS)
def test_the_full_family_admits_each_variant_statically_and_the_open_items_are_the_expected_ones(env, prof, v):
    a = argv_for(prof, av(v), env)
    r, p = _rules(env, prof, a)
    assert r == [] and p == [], (prof, v, r)
    want = sorted(EXPECT[v][2])
    # K2: none; K2M3 and the T1v variants: E1-MAP-OWN-RECORD alone. T1v is ADMITTED by the T1v FLIP (launch_gate `_R9F_T1V_ADMITTED`, MM 2026-11-09, H-R9-E1-T1V), so R9F-UNCLASSIFIED-LEVER is gone.
    assert items(env[prof], a) == want


def test_K0_is_still_admitted_with_no_open_item(env):
    assert _rules(env, "E2", K0) == ([], []) and items(env["E2"], K0) == []


# ================================================================================================= RED arms ==
@pytest.mark.parametrize("prof", FAMILY)
@pytest.mark.parametrize("v", VARIANTS)
@pytest.mark.parametrize("missing", ["--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only"])
def test_RED_a_variant_missing_one_T1h_flag_is_refused_naming_it(env, prof, v, missing):
    if v in T1V_VARIANTS:
        pytest.skip("T1v variants carry no T1h flag: see test_T1v_variants_carry_the_one_T1v_flag_and_NONE_of_the_four_T1h_flags")
    a = argv_for(prof, drop(av(v), missing), env)
    r, _ = _rules(env, prof, a)
    assert any(x.startswith(missing) and "is REQUIRED" in x and "--r9-bev-warp is passed" in x for x in r), r


@pytest.mark.parametrize("prof", FAMILY)
@pytest.mark.parametrize("v", ["K2M1N20", "K2M3N20"])
def test_RED_a_near_20_variant_with_near_lift_10_is_refused(env, prof, v):
    a = argv_for(prof, LG.set_flag(av(v), "--map-hires-near-lift-m", ["10"]), env)
    r, _ = _rules(env, prof, a)
    assert any(x.startswith("--map-hires-near-lift-m 20 is REQUIRED") and "--r9-map-banded-near-m=20 is passed" in x for x in r), r


@pytest.mark.parametrize("prof", FAMILY)
def test_RED_the_near_10_banded_variant_with_near_lift_20_is_refused(env, prof):
    a = argv_for(prof, LG.set_flag(av("K2M3"), "--map-hires-near-lift-m", ["20"]), env)
    r, _ = _rules(env, prof, a)
    assert any(x.startswith("--map-hires-near-lift-m 10 is REQUIRED") and "--r9-map-banded is passed" in x for x in r), r


@pytest.mark.parametrize("prof", FAMILY)
def test_RED_W_MAP_2_without_the_banded_head_is_refused_by_the_GATE(env, prof):
    a = argv_for(prof, av("K2") + ["--r9-plan-map-channels"], env)
    r, _ = _rules(env, prof, a)
    assert any(x.startswith("--r9-map-banded (a switch: no value) is REQUIRED") and "--r9-plan-map-channels is passed" in x for x in r), r


def test_RED_W_MAP_2_without_the_banded_head_is_also_refused_by_the_TRAINER_pin_and_the_composer_never_writes_it():
    from tanitad.train import plan_wires_flag as PW
    ns = dict(r9_plan_map_channels=True, r9_map_banded=False, bev_source="map_hires_pool")
    with pytest.raises(SystemExit, match="needs --r9-map-banded"):
        PW.pin_args(SimpleNamespace(**ns))
    PW.pin_args(SimpleNamespace(**dict(ns, r9_map_banded=True)))                         # control: the banded head satisfies it
    assert "--r9-plan-map-channels" not in av("K2") and "--r9-plan-map-channels" not in MK.compose(json.loads((RUNS / "r9e2_K0.argv.json").read_text(encoding="utf-8")), "K2", 0.1)


def test_RED_W_BOX_pin_refuses_wp_index_on_and_a_missing_box3d_weight():
    from tanitad.train import plan_wires_flag as PW
    ok = dict(r9_plan_box_tokens=True, agents="head", w_box3d=1.0, wp_index="off")
    PW.pin_args(SimpleNamespace(**ok))
    for chg, msg in ((dict(wp_index="on"), "wp-index on"), (dict(w_box3d=0.0), "w-box3d"), (dict(agents="off"), "agents head")):
        with pytest.raises(SystemExit, match=msg):
            PW.pin_args(SimpleNamespace(**dict(ok, **chg)))


def test_RED_a_variant_with_an_unclassified_or_refused_lever_is_not_silently_admitted(env):
    r, _ = _rules(env, "E2", av("K2") + ["--r9-bev-warp-shuffle"])
    assert any("--r9-bev-warp-shuffle is passed" in x for x in r), r
    assert "R9F-UNCLASSIFIED-LEVER" in items(env["E2"], av("K2") + ["--r9-alloc-force-schedule", "linear:1.0:0.0:0:3000"])
    r, _ = _rules(env, "E2", av("K2") + ["--r9-box-depth-aux", "--w-box-depth", "0.1"])
    assert r, "an aux head is still refused in the full family"


# ================================================================== profiles: pinned fingerprints, others unchanged ==
def _fp(n):
    return hashlib.sha256(repr(sorted((k, repr(v)) for k, v in LG.PROFILES[n].items())).encode()).hexdigest()[:16]


def test_fingerprints_the_full_family_moved_and_every_other_profile_is_unchanged():
    # OLD (tree e0cc3261): refcv9-e2 08c84658d9e98ceb, refcv9-full 551545dd6c9bbacc, refcv9-full-d1 9fbb9f8b7c41908d -- required_values_when gained the near-lift rules + the W-MAP-2 co-requirement
    # PIN6 (MM 2026-11-07, 5f47df80: the env-failure pin's sha literal sits in these profiles): refcv9-warm 09dbfa04d5a3ee4b -> 7099d75fa2f40240, refcv9-e1rig 18c28c61f19f5544 -> 0bb9664939426521, refcv8-mapft 783207af0d9e2062 -> f024877e5326f4ff, refcv9-e2 20297780a454a6c0 -> 2ee6fd2473fc296d, refcv9-full 8c9dcff2252b11ef -> 0af11fbfd62e6ed1, refcv9-full-d1 eae27696e70883cd -> b48f8af8c7fe34d4
    assert {n: _fp(n) for n in ("refcv8", "refcv9-warm", "refcv9-e1rig", "refcv8-nt", "refcv8-mapft", "v8f")} == {
        "refcv8": "4841b2a6f0758b95", "refcv9-warm": "7099d75fa2f40240", "refcv9-e1rig": "6fa39c86092dffec",     # refcv9 E1 arm T1v: E1_T1_FLAGS gained --r9-hist-supervise-geom (was 0bb9664939426521 after pin6, 18c28c61f19f5544 before it)
        "refcv8-nt": "a24d2c1a71042711", "refcv8-mapft": "f024877e5326f4ff", "v8f": "52860dd82f3a79af"}
    assert {n: _fp(n) for n in ("refcv9-e2", "refcv9-full", "refcv9-full-d1")} == {
        "refcv9-e2": "1a60c7095be5a175", "refcv9-full": "01b824851c5825d7", "refcv9-full-d1": "349a16c05861ef6d"}   # T1v FLIP: was 2ee6fd2473fc296d / 0af11fbfd62e6ed1 / b48f8af8c7fe34d4   # K2M1N20 RE-DOSE (2026-11-10, `_R9F_MAP_W_ADMITTED` admitted_values_when): moved acc27e87d94fe9e2 / 95a02abfad8aab76 / ad790da13c9c84b8 -> 1a60c7095be5a175 / 01b824851c5825d7 / 349a16c05861ef6d


def test_the_full_family_required_values_when_is_one_dict_holding_all_three_rule_sets_and_e1rig_shares_the_near_lift_constant():
    for n in ("refcv9-e2", "refcv9-full", "refcv9-full-d1"):
        rvw, why = LG.PROFILES[n]["required_values_when"], LG.PROFILES[n]["required_values_when_why"]
        assert set(rvw) == set(LG._R9F_T1H_ADMITTED) | {"--r9-map-banded", "--r9-map-banded-near-m=20", "--r9-plan-map-channels"}, n
        assert rvw["--r9-map-banded"] == {"--map-hires-near-lift-m": "10"} and rvw["--r9-map-banded-near-m=20"] == {"--map-hires-near-lift-m": "20"}
        assert rvw["--r9-plan-map-channels"] == {"--r9-map-banded": ()} and rvw["--r9-bev-warp"]["--r9-bev-warp-fuse-init"] == "identity"
        assert list(rvw)[-3:] == ["--r9-map-banded", "--r9-map-banded-near-m=20", "--r9-plan-map-channels"]          # near-20 AFTER banded: it overrides it
        assert set(why) == set(rvw)
    rig = LG.PROFILES["refcv9-e1rig"]["required_values_when"]
    assert rig == LG._NEAR_LIFT_WHEN and list(rig) == ["--r9-map-banded", "--r9-map-banded-near-m=20"]


# ================================================= build level (torch): K2 parses; the banded variants are PENDING ==
def _parse(v):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    T = pytest.importorskip("refc_v3_train")
    return T.build_parser().parse_args(av(v))


@pytest.mark.parametrize("v", VARIANTS)
def test_the_trainer_parses_each_variant_and_the_C8_wire_pins_accept_it(v):
    from tanitad.train import plan_wires_flag as PW
    args = _parse(v)
    assert PW.box_on(args) and PW.map_on(args) == (v != "K2") and args.r8_critic_drivable \
           and args.w_r8_drivable == float(LG.flag_values(doc(v)["argv"], "--w-r8-drivable")[0]) == (float(MEASURED_WCOST[v][1]) if v in MEASURED_WCOST else 0.1)
    PW.pin_args(args)


@pytest.mark.parametrize("v", VARIANTS)
def test_BUILD_LEVEL_bev_warp_pin_K2_passes_and_the_banded_variants_are_PENDING_the_supervise_hist_hold(v):
    from tanitad.train import bev_warp_flag as BWF
    args = _parse(v)
    if v in T1V_VARIANTS:       # T1v + --r9-map-banded [+ --r9-map-banded-near-m 20] [+ --r9-map-own-enc]: bev_warp_flag._pin_geom / pin_args accept it; T1h's flags are off
        assert BWF.geom_on(args) is True and BWF.is_on(args) is False
        BWF.pin_args(args)
        return
    try:
        BWF.pin_args(args)
    except SystemExit as e:
        if v != "K2" and "with --r9-map-banded" in str(e):
            pytest.skip("PENDING hold_r9_k2_bandhist: bev_warp_flag.pin_args still refuses --r9-bev-warp-supervise-hist with --r9-map-banded (bev_warp_flag.py:178-179)")
        raise
