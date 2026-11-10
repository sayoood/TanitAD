"""THE FLIP (MM 2026-11-07, H-R9-E1-T1H KEPT) -- the crowned T1h BUNDLE is ADMITTED in the full family (refcv9-e2 / refcv9-full / refcv9-full-d1), and nothing more.

  admitted : --r9-bev-warp --r9-bev-warp-supervise-hist --r9-bev-warp-fuse-init identity --r9-bev-warp-fill-only   (ONE unit: T1 alone FAILED its bar, H-R9-E1-T1)
  refused  : T1 alone; each T1h flag without the warp; fuse-init rezero; --r9-bev-warp-shuffle (T1hs's content control); --r9-bev-warp-lags (uncrowned value)
  unchanged: refcv9-warm and refcv9-e1rig (fingerprints pinned), the zero-grad admitted-dead rows (the gate and the fuse are TRAINED, never admitted dead)
Every guard has a deliberate-regression arm (the mutation list is in HANDOFF.md; each was run and went RED).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
from test_launch_gate_refcv9_full import K0, TRAINER, _d1, _full_argv, _rules, env  # noqa: E402,F401  (env: the module fixture)

WARP, SUP, FUSE, FILL = "--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only"
SHUF, LAGS = "--r9-bev-warp-shuffle", "--r9-bev-warp-lags"
BUNDLE = [WARP, SUP, FUSE, "identity", FILL]
FAMILY = ("refcv9-e2", "refcv9-full", "refcv9-full-d1")
KEY = {"refcv9-e2": "E2", "refcv9-full": "FULL", "refcv9-full-d1": "D1"}
RUNS = ROOT / "ops" / "runs.d"


def _argv_of(env, name):
    if name == "refcv9-e2":
        return list(K0), None
    if name == "refcv9-full":
        return _full_argv(), None
    return None, True                                   # D1: built by `_d1` from a main argv


def _rules_for(env, name, extra=(), drop=()):
    """Static argv rules for a family member whose argv = its base argv + `extra` (flags)."""
    base, d1 = _argv_of(env, name)
    if d1:
        main = _full_argv() + list(extra)
        a, _ = _d1(env, argv=main)
        return _rules(env, "D1", a)
    return _rules(env, KEY[name], base + list(extra))


def _bw(reasons):
    return [r for r in reasons if "bev-warp" in r]


# ====================================================================== 1. the bundle is ADMITTED, in every family member ==
@pytest.mark.parametrize("name", FAMILY)
def test_the_T1h_bundle_is_admitted_under_each_full_family_profile_in_any_order(env, name):
    assert _rules_for(env, name) == ([], [])                                           # control: the profile's own argv passes
    for extra in (BUNDLE, [FILL, FUSE, "identity", SUP, WARP], [FUSE, "identity", WARP, FILL, SUP]):
        refusals, pending = _rules_for(env, name, extra)
        assert refusals == [] and pending == [], (name, extra, refusals)
    p = env[KEY[name]]
    a = _full_argv() + BUNDLE
    assert LG.forbidden_lever_reasons(p, a)[0] == [] and LG.open_item_reasons(p, a) == []         # not forbidden; no own-record item; not an unclassified lever


# ====================================================================== 2. everything short of the bundle is REFUSED ==
@pytest.mark.parametrize("name", FAMILY)
def test_RED_T1_alone_is_refused_naming_the_three_missing_bundle_flags(env, name):
    r, _ = _rules_for(env, name, [WARP])
    b = _bw(r)
    assert len(b) == 3, b
    assert any(x.startswith(f"{SUP} (a switch: no value) is REQUIRED") and f"{WARP} is passed" in x and "T1 alone FAILED its bar" in x for x in b)
    assert any(x.startswith(f"{FUSE} identity is REQUIRED") and "['identity']" not in x for x in b)
    assert any(x.startswith(f"{FILL} (a switch: no value) is REQUIRED") for x in b)


@pytest.mark.parametrize("name", FAMILY)
def test_RED_each_T1h_flag_without_the_warp_is_refused_and_a_partial_bundle_names_what_is_missing(env, name):
    for flag, vals in ((SUP, []), (FUSE, ["identity"]), (FILL, [])):
        r, _ = _rules_for(env, name, [flag, *vals])
        assert any(x.startswith(f"{WARP} (a switch: no value) is REQUIRED") and f"{flag} is passed" in x for x in r), (flag, r)
    for drop, left in ((SUP, [WARP, FUSE, "identity", FILL]), (FILL, [WARP, SUP, FUSE, "identity"]), (FUSE, [WARP, SUP, FILL])):
        r, _ = _rules_for(env, name, left)
        assert any(x.startswith(drop) and "is REQUIRED" in x for x in _bw(r)), (drop, r)


@pytest.mark.parametrize("name", FAMILY)
def test_RED_fuse_init_must_be_identity_rezero_and_a_bare_flag_are_refused(env, name):
    for val in (["rezero"], []):
        r, _ = _rules_for(env, name, [WARP, SUP, FUSE, *val, FILL])
        assert any(x.startswith(f"{FUSE} identity is REQUIRED") and f"the argv has {val}" in x for x in r), (val, r)


@pytest.mark.parametrize("name", FAMILY)
def test_RED_the_shuffle_control_and_the_lags_value_stay_refused_each_with_its_own_reason(env, name):
    r, _ = _rules_for(env, name, BUNDLE + [SHUF])
    assert len([x for x in r if x.startswith(f"{SHUF} is passed")]) == 1 and "T1hs's CONTENT control" in [x for x in r if x.startswith(SHUF)][0] and len(r) == 1, r
    for lags in (["0.5,1.0"], ["1.5,2.0"], []):                                        # even the default value: it needs no flag
        r, _ = _rules_for(env, name, BUNDLE + [LAGS, *lags])
        assert len(r) == 1 and r[0].startswith(f"{LAGS} is passed") and "DEFAULT lags 0.5,1.0 s" in r[0], r
    r, _ = _rules_for(env, name, [SHUF])                                                # shuffle alone: forbidden AND the warp is missing
    assert any(x.startswith(f"{SHUF} is passed") for x in r)


# ====================================================================== 3. classification, fail-closed, and the crowned argv files ==
def test_the_six_flags_are_classified_by_explicit_literals_and_a_seventh_hits_the_unclassified_item(env):
    # refcv9 T1v (--r9-hist-supervise-geom) is the SEVENTH member of E1_T1_FLAGS and deliberately in NEITHER tuple (fail closed: R9F-UNCLASSIFIED-LEVER in the full family; test_r9_e1_t1v_gate.py)
    assert set(LG._R9F_T1H_ADMITTED) | set(LG._R9F_T1_REFUSED) == set(LG.E1_T1_FLAGS) - {"--r9-hist-supervise-geom"} == set(LG._E1_T1_FLAGS_PRE_T1V) and not set(LG._R9F_T1H_ADMITTED) & set(LG._R9F_T1_REFUSED)
    assert LG._R9F_T1H_ADMITTED == (WARP, SUP, FUSE, FILL) and LG._R9F_T1_REFUSED == (LAGS, SHUF)
    for name in FAMILY:
        it = next(i for i in LG.PROFILES[name]["open_items"] if i["id"] == "R9F-UNCLASSIFIED-LEVER")
        assert not set(LG.open_item_hits(it, K0 + BUNDLE + [SHUF, LAGS]) or []) & set(LG.E1_T1_FLAGS)
        assert "--r9-bev-warp-seventh" in LG.open_item_hits(it, K0 + [WARP, "--r9-bev-warp-seventh"])      # a new T1 flag is NOT silently admitted


def test_the_bundle_is_exactly_the_crowned_T1h_argv_T1_alone_and_T1hs_are_not_admitted_argvs():
    def t1(arm):
        a = json.loads((RUNS / f"r9e1_{arm}.argv.json").read_text(encoding="utf-8"))["argv"]
        return {f: v for f, v in LG.flag_pairs(a) if f in LG.E1_T1_FLAGS}
    assert t1("T1h") == {WARP: [], SUP: [], FUSE: ["identity"], FILL: []} and set(t1("T1h")) == set(LG._R9F_T1H_ADMITTED)      # the crowned arm's T1 flags ARE the bundle
    assert set(t1("T1")) == {WARP} and set(t1("T1hs")) == set(LG._R9F_T1H_ADMITTED) | {SHUF}                                # T1 alone / T1hs carry what the family refuses
    assert LG._R9F_T1H_REQUIRED[WARP] == {SUP: (), FUSE: "identity", FILL: ()}


def test_refcv9_warm_and_e1rig_are_unchanged_and_e1rig_still_admits_T1_alone_and_the_shuffle_arm():
    fp = lambda n: hashlib.sha256(repr(sorted((k, repr(v)) for k, v in LG.PROFILES[n].items())).encode()).hexdigest()[:16]   # noqa: E731
    # PIN6 (MM 2026-11-07, 5f47df80: the env-failure pin's sha literal sits in these profiles): refcv9-warm 09dbfa04d5a3ee4b -> 7099d75fa2f40240, refcv9-e1rig 18c28c61f19f5544 -> 0bb9664939426521
    assert fp("refcv9-warm") == "7099d75fa2f40240" and fp("refcv9-e1rig") == "6fa39c86092dffec"        # the values test_launch_gate_refcv9_full pins (R8); e1rig 0bb9664939426521 (pin6; 18c28c61f19f5544 before it) -> 6fa39c86092dffec only because E1_T1_FLAGS gained --r9-hist-supervise-geom (T1v)
    RIG = LG.PROFILES["refcv9-e1rig"]
    assert WARP not in (RIG.get("required_values_when") or {})
    for arm in ("T1", "T1s", "T1h", "T1hs"):
        a = json.loads((RUNS / f"r9e1_{arm}.argv.json").read_text(encoding="utf-8"))["argv"]
        assert LG.forbidden_lever_reasons(RIG, a)[0] == [] and _bw(LG.required_on_reasons(RIG, a)[0]) == [], arm


def test_the_family_fingerprints_moved_to_the_pinned_values():
    fp = lambda n: hashlib.sha256(repr(sorted((k, repr(v)) for k, v in LG.PROFILES[n].items())).encode()).hexdigest()[:16]   # noqa: E731
    # K2-ARGV (MM 2026-11-07): moved AGAIN, 08c84658d9e98ceb / 551545dd6c9bbacc / 9fbb9f8b7c41908d -> the values below -- the full family's `required_values_when` gained the near-lift rules
    # (`_NEAR_LIFT_WHEN`: banded -> 10, near-m=20 -> 20, the same constant refcv9-e1rig uses) and the W-MAP-2 co-requirement (`_R9F_PLAN_MAP_REQUIRED`); pinned also in test_r9e2_k2_argv.py
    # PIN6 (MM 2026-11-07, 5f47df80: the env-failure pin's sha literal sits in these profiles): refcv9-e2 20297780a454a6c0 -> 2ee6fd2473fc296d, refcv9-full 8c9dcff2252b11ef -> 0af11fbfd62e6ed1, refcv9-full-d1 eae27696e70883cd -> b48f8af8c7fe34d4
    # T1v FLIP (MM 2026-11-09, H-R9-E1-T1V): `_R9F_T1V_ADMITTED` in the R9F-UNCLASSIFIED-LEVER exemption + the T1v attach needles: refcv9-e2 2ee6fd2473fc296d -> acc27e87d94fe9e2, refcv9-full 0af11fbfd62e6ed1 -> 95a02abfad8aab76, refcv9-full-d1 b48f8af8c7fe34d4 -> ad790da13c9c84b8
    assert {n: fp(n) for n in FAMILY} == {"refcv9-e2": "1a60c7095be5a175", "refcv9-full": "01b824851c5825d7", "refcv9-full-d1": "349a16c05861ef6d"}   # was (before the T1v flip) 2ee6fd2473fc296d / 0af11fbfd62e6ed1 / b48f8af8c7fe34d4; before that 08c84658d9e98ceb / 551545dd6c9bbacc / 9fbb9f8b7c41908d (THE FLIP); before it 725ca7f0edac2d09 / ebcf1301de38ea4d / 97eec4e0d9d7eec6   # K2M1N20 RE-DOSE (2026-11-10, `_R9F_MAP_W_ADMITTED` admitted_values_when): moved acc27e87d94fe9e2 / 95a02abfad8aab76 / ad790da13c9c84b8 -> 1a60c7095be5a175 / 01b824851c5825d7 / 349a16c05861ef6d


# ====================================================================== 4. liveness: the full family reaches the SAME G-LIVE rule ==
def _ancestors(fn_src: str, callee: str) -> list[list[str]]:
    """For each call to `callee` in `fn_src`: the tests of its enclosing `if`s (source text), innermost first."""
    tree = ast.parse(fn_src)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == callee:
            anc, p = [], parents.get(n)
            while p is not None:
                if isinstance(p, ast.If):
                    anc.append(ast.unparse(p.test))
                p = parents.get(p)
            out.append(anc)
    return out


def _reach_ok(job_src: str, run_src: str) -> bool:
    """The smoke judges BEV_WARP_LIVE on the ARGV alone (not on the profile) and observes `bw_has` on every training loss call."""
    j, r = _ancestors(job_src, "judge_bev_warp_live"), _ancestors(run_src, "bev_warp_observe")
    return (len(j) == 1 and j[0] == ["has_flag(ctx.argv, BEV_WARP_LIVE['flag'])", "'G-LIVE' in evs"]
            and len(r) == 1 and r[0] == ["model.training"])


def test_G_LIVE_judges_the_bev_warp_for_a_full_family_argv_through_the_same_rule_as_e1rig():
    job, run = inspect.getsource(LG.job_smoke), inspect.getsource(LG.run_smoke)
    assert _reach_ok(job, run)                                                                      # profile-independent: no ctx.prof / profile-name condition on the path
    assert "ctx.prof" not in job.split("judge_bev_warp_live")[0].rsplit("if has_flag(ctx.argv, BEV_WARP_LIVE", 1)[1]
    assert LG.BEV_WARP_LIVE["flag"] == WARP
    mk = lambda st: {"numel": 4, "steps": st[0], "nz_steps": st[1], "nonfinite": False}             # noqa: E731
    g = lambda gate=(3, 3), fuse=(3, 2): {LG.BEV_WARP_LIVE["gate"]: mk(gate), LG.BEV_WARP_LIVE["fuse"]: mk(fuse)}   # noqa: E731
    import torch
    LG.bev_warp_observe(bw_rec := {}, {"bw_has": torch.tensor([[True, True]] * 8 + [[True, False]] * 2)})
    argv = _full_argv() + BUNDLE
    r, d = LG.judge_bev_warp_live(argv, g(), bw_rec["bw_has"], expect_steps=3)
    assert r == [] and d["on"] and d["gate"]["nz_steps"] == 3                                      # healthy smoke: GREEN
    r, _ = LG.judge_bev_warp_live(argv, g(fuse=(3, 0)), bw_rec["bw_has"], expect_steps=3)
    assert len(r) == 1 and "fuse" in r[0] and "ZERO gradient" in r[0], r                           # a dead fuse FAILS by name
    assert LG.judge_bev_warp_live(_full_argv(), {}, None) == ([], {})                               # a full argv WITHOUT the warp: inert


def test_RED_G_LIVE_not_reached_in_the_full_family_is_detected_by_the_structural_pin():
    job, run = inspect.getsource(LG.job_smoke), inspect.getsource(LG.run_smoke)
    gated = job.replace("if has_flag(ctx.argv, BEV_WARP_LIVE[\"flag\"]):", "if has_flag(ctx.argv, BEV_WARP_LIVE[\"flag\"]) and ctx.prof[\"name\"] == \"refcv9-e1rig\":")
    assert gated != job and not _reach_ok(gated, run)                                               # profile-gated: RED
    assert not _reach_ok(job.replace("btr, btd = judge_bev_warp_live(", "btr, btd = (lambda *a, **k: ([], {}))("), run)   # call removed: RED
    assert not _reach_ok(job, run.replace("bev_warp_observe(rec, batch)", "pass"))                  # observer removed: RED


def test_the_zero_grad_gate_admits_NO_dead_row_for_the_warp_gate_or_fuse_in_the_family():
    for name in FAMILY:
        rows = LG.PROFILES[name]["live_dead_admitted"]
        assert rows and not [r for r in rows if "bev_warp" in str(r[0]) or "bev-warp" in str(r[0])], name           # the gate and the fuse are TRAINED parameters
        assert LG.BEV_WARP_LIVE["gate"].startswith("_map_hires.bev_warp.") and LG.BEV_WARP_LIVE["fuse"].startswith("_map_hires.bev_warp.")


# ====================================================================== 5. the attach needles ==
@pytest.mark.parametrize("name", FAMILY)
def test_the_attach_needles_hold_for_the_bundle_and_each_hop_deleted_in_a_COPY_of_the_trainer_goes_red(name):
    p = LG.PROFILES[name]
    a = K0 + BUNDLE
    r, d = LG.lever_attach_reasons(p, a, TRAINER)
    assert r == [] and d["checked"] == {WARP: True, SUP: True}, (r, d)                              # fuse-init / fill-only are applied inside `attach` (covered by the warp needle)
    for flag, needle in ((WARP, "_bw.attach(model, args)"), (SUP, "_bw.supervised_lift_valid(model, pout, lv, sel)")):
        assert TRAINER.count(needle) == 1, needle
        r, d = LG.lever_attach_reasons(p, a, TRAINER.replace(needle, "pass"))
        assert len(r) == 1 and flag in r[0] and d["checked"][flag] is False, (flag, r)
    assert LG.lever_attach_reasons(p, K0, TRAINER.replace("_bw.attach(model, args)", "pass"))[0] == []        # no warp flag: inert


# ====================================================================== 6. no other profile moved ==
def test_every_non_family_profile_has_no_required_values_when_for_the_warp_and_the_reason_text_is_true():
    for name, p in LG.PROFILES.items():
        if name in FAMILY:
            continue
        assert WARP not in (p.get("required_values_when") or {}), name
    for name in FAMILY:
        forb = [e for e in LG.PROFILES[name]["forbidden_levers"] if any(f.startswith("--r9-bev-warp") for f in e[0])]
        assert [e[0] for e in forb] == [(SHUF,), (LAGS,)], name
        assert all("not crowned" not in e[1] and "until E1 has crowned" not in e[1] for e in forb)
