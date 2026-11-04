"""refcv9 E1-REC3 -- the gate rows, the harness stamp, the argvs and the bind kinds for the TRAINING-ONLY box aux heads
(B1 depth `--r9-box-depth-aux`, B2 perspective `--r9-box-persp-aux`, B2c BEV2 control `--r9-box-bev2-aux`).

THE MM'S DESIGN DECISION (pinned, 207ebd): the box overfit record certifies the INFERENCE box head in isolation on frozen frames. A training-only
aux head is never called at inference and its effect (shaping the trunk) cannot show in a frozen-frame overfit. Its liveness is G-LIVE's (its loss
key finite in the smoke, every new parameter with a gradient) and the post-run zero-gradient gate's. So an aux-only arm binds the INHERITED /1 box
record to its OWN argv, and that record must STAMP `aux_heads_not_certified` == the argv's aux set.

* G-LIVE: one LIVE_WEIGHT_RULES term per aux weight (`w_box_depth` / `w_box_persp` / `w_box_bev2`) -- required, finite, every step when the weight
  is > 0, INERT when 0; the trainer's REFC_WEIGHT_GATES entries are what makes this exhaustive (the land blocker test is
  `test_launch_gate_e1rig.py::test_G_LIVE_declares_the_rig_terms_instead_of_the_planner_objective`).
* the gate: `E1_BOX_OWN_FLAGS` = B3's four; `E1_BOX_AUX_FLAGS` = the three aux flags; an aux flag opens NO own-record item; a `--r9-box-*` flag in
  neither tuple opens `E1-BOX-UNKNOWN-LEVER`; the /1 judge reads the stamp (RED arms below).
* the harness: an aux flag in the launch argv is NOT built (build argv drops flag + weight), the record stamps `aux_heads_not_certified` + a reason.
* `r9e1_B1` / `r9e1_B2` / `r9e1_B2c`: C0 + the aux flag + its weight pair; `r9e1_bind.sh`: kind `box` takes an aux-only argv, refuses an own-head one.
"""
from __future__ import annotations

import hashlib
import inspect
import json
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

G = pytest.importorskip("g_box_overfit")

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
COMMIT = "ab" * 20


def _argv(name):
    return json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


C0, B1, B2, B2C, B3, B3P = (_argv(n) for n in ("C0", "B1", "B2", "B2c", "B3", "B3p"))
AUX = {"B1": ("--r9-box-depth-aux", "--w-box-depth", "w_box_depth", "box_depth"),
       "B2": ("--r9-box-persp-aux", "--w-box-persp", "w_box_persp", "box_persp"),
       "B2c": ("--r9-box-bev2-aux", "--w-box-bev2", "w_box_bev2", "box_bev2")}
ARGV = {"B1": B1, "B2": B2, "B2c": B2C}
PARTS = {"B1": (), "B2": ("heat", "depth", "size", "yaw", "ctr"), "B2c": ("heat", "pos", "size", "yaw", "ctr")}


def _write(tmp_path, rec, name="rec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def _args(argv):
    T = pytest.importorskip("refc_v3_train")
    return T, T.build_parser().parse_args(argv)


def _box1(argv, **over):
    """A /1 box record that PASSES the inherited judge for `argv` (the A19 MAIN-only form `closure_run` banks) + the harness's OWN stamp for it."""
    rec = TG._box_main_only(LG.argv_sha256(argv))
    rec.update(G.b3_record_fields(argv))                          # {} for C0; the aux stamp for B1 / B2 / B2c (the harness's own function)
    rec.update(over)
    return rec


def _judge(tmp_path, rec, argv):
    return LG.judge_box_overfit(RIG, _write(tmp_path, rec), COMMIT, argv_sha=LG.argv_sha256(argv), argv=argv)


# ================================================================================ the tuples are ONE fact ==
def test_the_two_flag_classes_are_pinned_across_gate_harness_emitter_and_trainer():
    assert LG.E1_BOX_OWN_FLAGS == ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms", "--r9-box-layers", "--r9-box-s8-horizon", "--r9-box-perframe-queries") == tuple(G.B3_FLAGS)
    assert LG.E1_BOX_AUX_FLAGS == ("--r9-box-depth-aux", "--r9-box-persp-aux", "--r9-box-bev2-aux") == tuple(G.AUX_FLAGS)
    assert not set(LG.E1_BOX_OWN_FLAGS) & set(LG.E1_BOX_AUX_FLAGS)
    assert (LG.E1_BOX_AUX_STAMP, LG.E1_BOX_AUX_STAMP_WHY) == (G.AUX_STAMP, G.AUX_STAMP_WHY)
    assert G.AUX_WEIGHT_FLAGS == {f: w for f, w, _d, _k in AUX.values()}
    assert {a: tuple(v[:2]) for a, v in MK.AUX_ARMS.items()} == {a: tuple(v[:2]) for a, v in AUX.items()}
    # the trainer's parser knows every flag and weight (a flag the trainer does not parse would be a dead pin)
    T = pytest.importorskip("refc_v3_train")
    opts = set(T.build_parser()._option_string_actions)
    assert set(LG.E1_BOX_AUX_FLAGS) | set(G.AUX_WEIGHT_FLAGS.values()) <= opts
    # the flag modules' own destinations
    from tanitad.train import box_depth_aux_flag as BDA
    from tanitad.train import persp3d_aux_flag as P3
    assert ("--" + BDA.DEST.replace("_", "-"), "--" + BDA.W_DEST.replace("_", "-")) == AUX["B1"][:2] and BDA.W_DEST == AUX["B1"][2]
    assert (P3.FLAG_P, P3.W_FLAG_P, P3.W_DEST_P, P3.LOSS_KEY_P) == (*AUX["B2"],)
    assert (P3.FLAG_B, P3.W_FLAG_B, P3.W_DEST_B, P3.LOSS_KEY_B) == (*AUX["B2c"],)
    assert 'extra["box_depth"] = loss' in Path(BDA.__file__).read_text(encoding="utf-8")        # C3 writes its key as a literal


# ================================================================================================= G-LIVE ==
def _step(keys, loss, nonfinite=()):
    return {"keys": list(keys), "nonfinite": list(nonfinite), "vals": {k: (loss if k == "loss" else 0.5) for k in keys}}


def _live(terms, steps, args, *, n=100):
    row = {"step": n, **{k: round(steps[-1]["vals"][k], 5) for k in steps[-1]["keys"] if k in steps[-1]["vals"]}}
    grads = {"n_trainable_params": 5, "nonfinite_params": [], "dead_groups": []}
    return LG.judge_live(terms, [], steps, expect_steps=n, summary={"done": True, "step": n}, train_rows=[row], grads=grads,
                         prior=None, exit_msg=None, absent_terms=LG.rig_absent_terms(args))


def _healthy(arm):
    T, args = _args(ARGV[arm])
    terms, problems = LG.declared_terms(T, args)
    assert problems == [], problems
    keys = sorted({k for t in terms for k in t["keys"]})
    return terms, keys, args


@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_G_LIVE_declares_the_aux_term_with_its_exact_keys_and_a_healthy_smoke_passes(arm):
    flag, wflag, dest, key = AUX[arm]
    terms, keys, args = _healthy(arm)
    mine = [t for t in terms if t["id"] == dest]
    assert len(mine) == 1 and mine[0]["min_frac"] == 1.0
    assert mine[0]["keys"] == [key] + [f"{key}_{p}" for p in PARTS[arm]]
    assert {t["id"] for t in terms} & {a[2] for a in AUX.values()} == {dest}          # ONLY the arm's own aux term
    reasons, det = _live(terms, [_step(keys, 10.0 - 0.01 * i) for i in range(100)], args)
    assert reasons == [] and det["absent_terms"][0]["emitted"] == {}, reasons


@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_RED_a_smoke_step_that_drops_the_aux_loss_key_fails_G_LIVE_by_name(arm):
    flag, wflag, dest, key = AUX[arm]
    terms, keys, args = _healthy(arm)
    reasons, _ = _live(terms, [_step([k for k in keys if k != key], 10.0 - 0.01 * i) for i in range(100)], args)
    assert any(f"declared term `{key}`" in r and "ABSENT from all 100" in r for r in reasons), reasons
    for part in PARTS[arm]:                                                           # a dropped PART is a failure too
        reasons, _ = _live(terms, [_step([k for k in keys if k != f"{key}_{part}"], 10.0 - 0.01 * i) for i in range(100)], args)
        assert any(f"declared term `{key}_{part}`" in r for r in reasons), (part, reasons)
    # present on only half the steps
    steps = [_step(keys if i % 2 else [k for k in keys if k != key], 10.0 - 0.01 * i) for i in range(100)]
    assert any(f"declared term `{key}` present on only 50/100" in r for r in _live(terms, steps, args)[0])
    # control: the SAME smoke with the key present is clean
    assert _live(terms, [_step(keys, 10.0 - 0.01 * i) for i in range(100)], args)[0] == []


@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_RED_a_non_finite_aux_loss_fails_G_LIVE(arm):
    flag, wflag, dest, key = AUX[arm]
    terms, keys, args = _healthy(arm)
    steps = [_step(keys, 10.0 - 0.01 * i, nonfinite=([key] if i == 7 else [])) for i in range(100)]
    reasons, _ = _live(terms, steps, args)
    assert any(f"declared term `{key}` was non-finite on 1 step(s)" in r for r in reasons), reasons


def test_the_aux_terms_are_INERT_at_weight_zero_and_for_every_other_arm():
    T = pytest.importorskip("refc_v3_train")
    for name, argv in (("C0", C0), ("B3", B3), ("B3p", B3P)):
        terms, problems = LG.declared_terms(T, T.build_parser().parse_args(argv))
        assert problems == [] and not {t["id"] for t in terms} & {a[2] for a in AUX.values()}, name
    for arm, argv in ARGV.items():
        args = T.build_parser().parse_args(argv)
        setattr(args, AUX[arm][2], 0.0)                                              # weight 0 (the trainer's pin refuses it with the flag; the gate's rule is INERT)
        assert AUX[arm][2] not in [t["id"] for t in LG.declared_terms(T, args)[0]], arm


def test_RED_the_land_blocker_G_LIVE_has_no_rule_for_a_registered_weight_when_a_rule_is_missing(monkeypatch):
    """The trainer registers the three aux weights (REFC_WEIGHT_GATES); G-LIVE is EXHAUSTIVE over that registry. Deleting any one rule must turn the
    coverage problem on (the MM's addendum: this was the single failing test of tree6)."""
    T, args = _args(B1)
    assert {a[2] for a in AUX.values()} <= set(T.REFC_WEIGHT_GATES) and {a[2] for a in AUX.values()} <= set(LG.LIVE_WEIGHT_RULES)
    assert LG.declared_terms(T, args)[1] == []
    for _flag, _wf, dest, _key in AUX.values():
        rules = dict(LG.LIVE_WEIGHT_RULES)
        rules.pop(dest)
        monkeypatch.setattr(LG, "LIVE_WEIGHT_RULES", rules)
        assert f"the trainer registers weight `{dest}` and G-LIVE has no rule for it" in LG.declared_terms(T, args)[1]
        monkeypatch.undo()


# ============================================================================ the open items and the rows ==
def test_an_aux_only_argv_opens_no_own_record_item_and_B3_is_unchanged():
    for arm, argv in ARGV.items():
        assert LG.open_item_reasons(RIG, argv) == [], arm
        assert LG.own_record_row(RIG, argv, "G-BOX-OVERFIT") is None, arm
        assert LG.own_records_closed(RIG, argv, {"box_overfit_record": "/nonexistent"}, COMMIT, LG.argv_sha256(argv)) == {}
    for argv in (B3, B3P):
        assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, argv)] == ["OPEN ITEM E1-BOX-OWN-RECORD"]
        assert LG.own_record_row(RIG, argv, "G-BOX-OVERFIT")["item"] == "E1-BOX-OWN-RECORD"
    both = LG.set_flag(LG.set_flag(B3, AUX["B1"][0], []), AUX["B1"][1], ["0.1"])      # B3 + B1: the OWN item stays open, ONLY for B3's levers
    assert [x.split(":")[0] for x in LG.open_item_reasons(RIG, both)] == ["OPEN ITEM E1-BOX-OWN-RECORD"]
    assert "[triggered by ['--r9-box-quality', '--r9-box-dn', '--r9-box-class-sqrt']]" in LG.open_item_reasons(RIG, both)[0]
    assert LG.open_item_reasons(RIG, C0) == [] and LG.open_item_reasons(RIG, _argv("C1")) == []


@pytest.mark.parametrize("flag", ["--r9-box-foo", "--r9-box-depth-aux-v2", "--r9-box-persp"])
def test_RED_an_unknown_box_lever_opens_an_item_and_is_never_silently_inherited(flag):
    a = LG.set_flag(B1, flag, [])
    ids = [x.split(":")[0] for x in LG.open_item_reasons(RIG, a)]
    assert "OPEN ITEM E1-BOX-UNKNOWN-LEVER" in ids and "OPEN ITEM E1-BOX-OWN-RECORD" in ids, (flag, ids)
    why = next(x for x in LG.open_item_reasons(RIG, a) if "E1-BOX-UNKNOWN-LEVER" in x)
    assert "unknown box lever: classify it" in why and f"[triggered by ['{flag}']]" in why
    # no record can close it: there is no table row (a reviewed edit that classifies the flag is the only way)
    assert "E1-BOX-UNKNOWN-LEVER" not in LG.E1_OWN_RECORDS
    ok, why2 = LG.own_record_closes(RIG, "E1-BOX-UNKNOWN-LEVER", a, {"box_overfit_record": "/nonexistent"}, COMMIT, LG.argv_sha256(a))
    assert not ok and "no table row" in " ".join(why2)
    # the gate's token would be INCOMPLETE: finalize turns any open item into INCOMPLETE (pinned in test_launch_gate_e1rig)
    assert any(it["id"] == "E1-BOX-UNKNOWN-LEVER" for it in RIG["open_items"])
    # control: the SAME argv without the unknown flag opens nothing
    assert LG.open_item_reasons(RIG, B1) == []


def test_every_other_profile_is_untouched_by_the_new_item():
    for name, p in LG.PROFILES.items():
        if name not in ("refcv9-e1rig", "refcv9-e2", "refcv9-full", "refcv9-full-d1"):   # the E1 rig and C12-FULL's three refcv9 profiles carry E1's lever-keyed rows by design
            assert LG.open_item_reasons(p) == LG.open_item_reasons(p, ["--r9-box-foo"]) == LG.open_item_reasons(p, [])
            assert all(it.get("id") != "E1-BOX-UNKNOWN-LEVER" for it in (p.get("open_items") or ()))


@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_the_profile_does_not_refuse_the_aux_arms_static_argv_rules(arm, tmp_path):
    """required / forbidden values and levers: the aux flags + weights are accepted (the same refusal / pending lists as C0's)."""
    init = tmp_path / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped refcv8 EMA-48,852")
    prof = dict(RIG, init_pin=dict(RIG["init_pin"], init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    pmap = [("/home/nvidia/r8ns/ckpt", str(tmp_path))]
    base = LG.profile_argv_rules(prof, C0, {}, pmap)
    got = LG.profile_argv_rules(prof, ARGV[arm], {}, pmap)
    assert base == ([], []) and got == ([], []), got
    assert LG.forbidden_lever_reasons(prof, ARGV[arm])[0] == LG.forbidden_lever_reasons(prof, C0)[0] == []
    assert LG.required_flag_reasons(prof, ARGV[arm]) == LG.required_flag_reasons(prof, C0)
    # the arm's aux flag and its weight are in the argv, and no sibling aux flag is
    d = dict(LG.flag_pairs(ARGV[arm]))
    assert d[AUX[arm][0]] == [] and d[AUX[arm][1]] == ["0.1"]
    assert sum(1 for a in AUX.values() if a[0] in d) == 1


# ====================================================================== the /1 judge reads the aux stamp ==
@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_GREEN_the_harness_stamped_record_for_an_aux_only_argv_passes_the_inherited_judge(arm, tmp_path):
    argv = ARGV[arm]
    rec = _box1(argv)
    assert rec[G.AUX_STAMP] == [AUX[arm][0]] and rec[G.AUX_STAMP_WHY] and "schema" not in rec and "b3" not in rec      # still the /1 shape
    reasons, det = _judge(tmp_path, rec, argv)
    assert reasons == [] and det["aux_heads_not_certified"] == {"argv_aux_set": [AUX[arm][0]], "record_stamp": [AUX[arm][0]]}


def test_RED_arms_the_aux_stamp_is_required_and_must_equal_the_argvs_aux_set(tmp_path):
    good = _box1(B1)

    def why(rec, argv=B1):
        reasons, _ = _judge(tmp_path, rec, argv)
        assert reasons, "the defective record passed"
        return " | ".join(reasons)

    no_stamp = {k: v for k, v in good.items() if k not in (G.AUX_STAMP, G.AUX_STAMP_WHY)}
    assert "stamps no `aux_heads_not_certified`" in why(no_stamp)                                                 # B1 argv + a record WITHOUT the stamp
    assert "stamps no `aux_heads_not_certified`" in why(TG._box_main_only(LG.argv_sha256(B1)))                    # a plain /1 record (C0's shape)
    assert "aux set is ['--r9-box-depth-aux']" in why(dict(good, **{G.AUX_STAMP: ["--r9-box-persp-aux"]}))        # stamp != argv aux set (another arm's)
    assert "aux set is ['--r9-box-depth-aux']" in why(dict(good, **{G.AUX_STAMP: list(G.AUX_FLAGS)}))             # extra heads
    assert "aux set is ['--r9-box-depth-aux']" in why(dict(good, **{G.AUX_STAMP: []}))                            # an empty stamp on an aux argv
    assert "aux set is" in why(dict(good, **{G.AUX_STAMP: ["--r9-box-depth-aux", "--r9-box-depth-aux"]}))         # duplicated
    assert "not a list" in why(dict(good, **{G.AUX_STAMP: "--r9-box-depth-aux"}))
    assert G.AUX_STAMP_WHY in why({k: v for k, v in good.items() if k != G.AUX_STAMP_WHY})                        # no one-line reason
    assert G.AUX_STAMP_WHY in why(dict(good, **{G.AUX_STAMP_WHY: "  "}))
    # the stamp offered to ANOTHER aux argv (B1's record for B2): wrong sha AND wrong set
    reasons, _ = _judge(tmp_path, good, B2)
    assert any("bound to argv" in r for r in reasons) and any("aux set is ['--r9-box-persp-aux']" in r for r in reasons)
    forged = dict(good, launch_argv_sha256=LG.argv_sha256(B2))                                                    # right sha, wrong stamp
    reasons, _ = _judge(tmp_path, forged, B2)
    assert [r for r in reasons if "bound to argv" in r] == [] and any("aux set is ['--r9-box-persp-aux']" in r for r in reasons)
    # a stamp on an argv with NO aux flag (C0 / B3): the record was made for another argv
    for argv in (C0, B3):
        reasons, _ = _judge(tmp_path, _box1(argv, **{G.AUX_STAMP: ["--r9-box-depth-aux"], G.AUX_STAMP_WHY: "x"}), argv)
        assert any("aux set is []" in r for r in reasons), (argv is C0, reasons)
    # control: every defect above was the stamp's -- the good record passes
    assert _judge(tmp_path, good, B1)[0] == []


def test_C0_C1_B3_and_the_M_arms_records_and_verdicts_are_identical(tmp_path):
    """No stamp, no aux flag: the /1 judge's verdict and its details are exactly what they were (no new key in `det`)."""
    for argv in (C0, _argv("C1"), _argv("M1"), _argv("M2a"), B3, B3P):
        reasons, det = _judge(tmp_path, TG._box_main_only(LG.argv_sha256(argv)), argv)
        assert reasons == [] and "aux_heads_not_certified" not in det, argv[-1]
        reasons, det = _judge(tmp_path, TG._gbo_record(LG.argv_sha256(argv)), argv)                                 # the full 3-arm record too
        assert reasons == [] and "aux_heads_not_certified" not in det
    # judge without an argv (older callers): the clause is skipped, never an exception
    assert LG.judge_box_overfit(RIG, _write(tmp_path, TG._box_main_only(LG.argv_sha256(B1))), COMMIT, argv_sha=LG.argv_sha256(B1))[0] == []


def test_the_job_states_what_the_pass_does_not_certify(tmp_path, monkeypatch):
    monkeypatch.setattr(LG, "judge_closure", lambda *a, **k: ([], {}))                    # the closure is the bind run's own artifact; not under test here

    def ev(argv, rec):
        c = LG.Ctx(profile="refcv9-e1rig", tree=str(tmp_path), commit=COMMIT, argv=list(argv), out_dir=str(tmp_path), path_map=[],
                   tree_sha256="0" * 64, argv_sha256=LG.argv_sha256(argv),
                   options={"box_overfit_record": _write(tmp_path, rec), "box_overfit_closure": None})
        return LG._job_box_overfit(c)["G-BOX-OVERFIT"]
    e = ev(B2, _box1(B2))
    assert e["status"] == "PASS" and "does NOT certify the training-only aux head(s) ['--r9-box-persp-aux']" in " ".join(e["reasons"])
    e = ev(B2, TG._box_main_only(LG.argv_sha256(B2)))                                      # RED: B2 argv, record without the stamp
    assert e["status"] == "FAIL" and any("aux_heads_not_certified" in r for r in e["reasons"])
    e = ev(C0, TG._box_main_only(LG.argv_sha256(C0)))                                      # C0: the PASS text is exactly what it was
    assert e["status"] == "PASS" and "does NOT certify" not in " ".join(e["reasons"])


def test_a_B3_plus_aux_argv_needs_the_b3_record_AND_the_aux_stamp(tmp_path):
    both = LG.set_flag(LG.set_flag(B3, AUX["B2"][0], []), AUX["B2"][1], ["0.1"])
    rec = R2._box_record(both)
    assert rec["b3"]["flags"] == list(R2.BOX_FLAGS[:3]) and rec[G.AUX_STAMP] == ["--r9-box-persp-aux"]
    assert R2._box_closes(tmp_path, rec, both) == (True, [])
    no_stamp = {k: v for k, v in rec.items() if k not in (G.AUX_STAMP, G.AUX_STAMP_WHY)}
    ok, why = R2._box_closes(tmp_path, no_stamp, both)
    assert not ok and "aux_heads_not_certified" in " ".join(why)
    ok, why = R2._box_closes(tmp_path, dict(rec, **{G.AUX_STAMP: ["--r9-box-bev2-aux"]}), both)
    assert not ok and "aux set is ['--r9-box-persp-aux']" in " ".join(why)
    # B3 alone is unchanged: the record needs no aux stamp
    assert R2._box_closes(tmp_path, R2._box_record(B3), B3) == (True, [])


# ============================================================================================ the harness ==
def test_the_harness_stamps_the_aux_set_and_keeps_the_other_records_byte_identical():
    for arm, argv in ARGV.items():
        assert G.b3_record_fields(argv) == {G.AUX_STAMP: [AUX[arm][0]], G.AUX_STAMP_WHY: G.AUX_WHY}
    assert G.b3_record_fields(C0) == {} and G.b3_record_fields(["--w-box3d", "1.0", "--trunk-compile"]) == {}
    r = G.b3_record_fields(LG.set_flag(LG.set_flag(B3, "--r9-box-bev2-aux", []), "--w-box-bev2", ["0.1"]))
    assert r["schema"] == G.B3_SCHEMA and r["b3"]["flags"] == list(G.B3_FLAGS[:3]) and r[G.AUX_STAMP] == ["--r9-box-bev2-aux"]
    for typo in ("--r9-box-typo", "--r9-box-depth-aux-v2", "--r9-box-persp"):
        with pytest.raises(SystemExit, match="unknown --r9-box-"):
            G.b3_record_fields(["--r9-box-depth-aux", typo])
    assert "\n" not in G.AUX_WHY and "training-only" in G.AUX_WHY                    # one line, and it says what the record is NOT


def test_the_harness_does_not_build_an_aux_head_the_build_argv_drops_the_flag_and_its_weight():
    T = pytest.importorskip("refc_v3_train")
    for arm, argv in ARGV.items():
        build, rec = G.harness_build_argv(argv)
        assert not set(G.AUX_FLAGS) & set(build) and not set(G.AUX_WEIGHT_FLAGS.values()) & set(build), arm
        assert rec["aux_flags_not_built"] == [AUX[arm][0]]
        want = [t for t in argv if t != "--trunk-compile"]                           # the harness's own (declared) drop
        i = want.index(AUX[arm][0])
        assert want[i:i + 3] == [AUX[arm][0], AUX[arm][1], "0.1"]
        del want[i:i + 3]
        assert build == want                                                           # EXACTLY the three tokens, no neighbour swallowed
        assert (build[i - 2:i + 2] == ["--slot-query-select", "learned_ref", "--equalize-bottom-rows", "43"])
        a = T.build_parser().parse_args(build)                                         # the head is OFF in what the harness builds from
        assert a.r9_box_depth_aux is False and a.r9_box_persp_aux is False and a.r9_box_bev2_aux is False
        assert a.w_box_depth == a.w_box_persp == a.w_box_bev2 == 0.0
    # every other launch argv: the build argv and the record keys are what they were
    for argv in (C0, B3):
        build, rec = G.harness_build_argv(argv)                                        # (C0's / B3's argv carries --trunk-compile: the harness drops it)
        assert build == [t for t in argv if t != "--trunk-compile"] and rec["trunk_compile_in_launch_argv"] is True
        assert set(rec) == {"trunk_compile", "trunk_compile_in_launch_argv", "trunk_compile_why"}
    # the record is bound to the LAUNCH argv (aux flags included, `load_launch_argv`), the adapter builds from the stripped one
    src = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    assert src.count("build_argv, tc_rec = harness_build_argv(launch_argv)") == 1 and src.count("TrainerAdapter(tr, L, build_argv,") == 1
    assert "TrainerAdapter(tr, L, launch_argv," not in src


def test_RED_the_aux_flag_is_not_built_by_the_harness_if_the_adapter_got_the_launch_argv():
    """MUTATION: hand the adapter the LAUNCH argv and the source scan above goes red -- and the launch argv, parsed as on, is a head the harness
    would have to replay (it has no replay), so the strip is the line that keeps the certificate honest."""
    T = pytest.importorskip("refc_v3_train")
    a = T.build_parser().parse_args(B1)
    assert a.r9_box_depth_aux is True and a.w_box_depth == 0.1
    src = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    mutated = src.replace("TrainerAdapter(tr, L, build_argv,", "TrainerAdapter(tr, L, launch_argv,")
    assert mutated != src and mutated.count("TrainerAdapter(tr, L, build_argv,") == 0


# ============================================================================================== the argvs ==
def _diff(a, b):
    da, db = dict(LG.flag_pairs(a)), dict(LG.flag_pairs(b))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


@pytest.mark.parametrize("arm", ["B1", "B2", "B2c"])
def test_aux_arm_argvs_differ_from_C0_in_exactly_the_lever_flags(arm):
    flag, wflag, dest, key = AUX[arm]
    argv = ARGV[arm]
    assert _diff(C0, argv) == {flag, wflag, "--out"}
    assert [f for f, _ in LG.flag_pairs(C0)] == [f for f, _ in LG.flag_pairs(argv) if f not in (flag, wflag)]        # same flags, same order
    d = dict(LG.flag_pairs(argv))
    assert d[flag] == [] and d[wflag] == ["0.1"] and d["--seed"] == ["0"] and d["--out"] == [f"/home/nvidia/r8ns/r9e1/runs/r9e1-{arm}"]
    i = argv.index(flag)
    assert argv[i - 2:i + 3] == ["--slot-query-select", "learned_ref", flag, wflag, "0.1"]                         # right after the last slot flag
    assert d["--workers"] == ["6"] and d["--w-box3d"] == C0[C0.index("--w-box3d") + 1:C0.index("--w-box3d") + 2]
    assert not any(f.startswith("--r9-box-") and f != flag for f in d)                                             # NO own-head lever and no sibling aux
    raw = (RUNS / f"r9e1_{arm}.argv.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    rec = json.loads(raw)
    assert rec["arm"] == f"r9e1_{arm}" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0" and rec["steps"] == 4000
    assert rec["lever"]["flags"] == [flag] and rec["lever"]["weight_flag"] == wflag and rec["lever"]["weight"] == 0.1
    assert "HYPOTHESIS" in rec["lever"]["weight_source"] and "not certified" in rec["lever"]["aux_head"]
    assert rec["base_sha256"] == hashlib.sha256((RUNS / "r9e1_C0.argv.json").read_bytes()).hexdigest()
    assert LG.argv_sha256(argv) not in (LG.argv_sha256(C0), LG.argv_sha256(B3))


def test_aux_arm_emitter_reproduces_the_committed_files_and_its_pins_are_real():
    for arm in ARGV:
        assert MK.box_arm_argv(C0, arm) == ARGV[arm]
        MK.check_aux_arm_pins(ARGV[arm], arm)                          # the trainer's parser, the rig pin AND the head's own pin_args accept
    assert set(MK.BOX_ARMS) == {"B3", "B3p"} and set(MK.AUX_ARMS) == {"B1", "B2", "B2c"}
    with pytest.raises(SystemExit):
        MK.box_arm_argv(ARGV["B1"], "B2")                              # derive from C0 only
    # RED: each pin is a real check
    i = B1.index("--w-box-depth")
    no_weight = B1[:i] + B1[i + 2:]                                    # the flag with no weight (default 0.0): a head that would sit untrained
    assert "--r9-box-depth-aux" in no_weight and "--w-box-depth" not in no_weight and len(no_weight) == len(B1) - 2
    with pytest.raises(SystemExit, match=r"--w-box-depth is 0"):
        MK.check_aux_arm_pins(no_weight, "B1")
    orphan = LG.set_flag(C0, "--w-box-persp", ["0.1"])                 # a weight with no flag: a dead flag
    with pytest.raises(SystemExit, match=r"aux heads parsed as on are \[\]"):
        MK.check_aux_arm_pins(orphan, "B2")
    both = LG.set_flag(LG.set_flag(B2, "--r9-box-bev2-aux", []), "--w-box-bev2", ["0.1"])
    with pytest.raises(SystemExit, match="aux heads parsed as on"):    # B2 and B2c both on (C4's own pin says MUTUALLY EXCLUSIVE)
        MK.check_aux_arm_pins(both, "B2")
    with pytest.raises(SystemExit, match="aux heads parsed as on"):    # the wrong arm name for the argv
        MK.check_aux_arm_pins(B2, "B1")
    own = LG.set_flag(B1, "--r9-box-quality", [])
    with pytest.raises(SystemExit, match="own-head"):                  # an aux arm differs from C0 in its aux lever alone
        MK.check_aux_arm_pins(own, "B1")
    with pytest.raises(SystemExit, match="not an aux-head arm"):
        MK.check_aux_arm_pins(B3, "B3")                                # not an aux arm at all


def test_the_M_arm_and_B3_files_are_unchanged_by_the_emitter():
    for name in ("B3", "B3p"):
        assert MK.box_arm_argv(C0, name) == _argv(name)
    for name in ("M1", "M2a", "M2b"):
        assert MK.map_arm_argv(C0, name) == _argv(name)


# ======================================================================================== the bind script ==
@R2.needs_bash
@pytest.mark.parametrize("kind,arm", [("box", "B1"), ("box", "B2"), ("box", "B2c"), ("map", "B1"), ("map", "B2"), ("map", "B2c"),
                                      ("box", "C0"), ("box", "M1"), ("box-b3", "B3"), ("box-b3", "B3p"), ("map", "B3")])
def test_dry_run_accepts_the_aux_arms_with_the_inherited_box_and_map_kinds(tmp_path, kind, arm):
    out, root = R2._bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    sha = LG.argv_sha256(_argv(arm))
    assert f"sha256 {sha}" in out and f"argv sha {sha[:16]}" in out
    assert (root / f"ovf_{arm}_{R2.HASH[:12]}" / f"{kind}.args").is_file()
    if kind == "box" and arm in ARGV:
        assert f"training-only aux head(s) {AUX[arm][0]}" in out                         # the script says what the record will not certify


@R2.needs_bash
@pytest.mark.parametrize("kind,arm,why", [
    ("box", "B3", "own-head --r9-box-* lever (--r9-box-quality --r9-box-dn --r9-box-class-sqrt): use box-b3"),
    ("box", "B3p", "use box-b3"),
    ("box-b3", "B1", "WITHOUT any --r9-box-* own-head lever"), ("box-b3", "B2", "WITHOUT any --r9-box-* own-head lever"),
    ("box-b3", "B2c", "aux-only arm is kind box"), ("box-b3", "C0", "WITHOUT any --r9-box-"), ("map-banded", "B1", "WITHOUT --r9-map-banded")])
def test_dry_run_refuses_the_wrong_box_kind_for_an_argv_before_touching_anything(tmp_path, kind, arm, why):
    out, root = R2._bind(tmp_path, kind, arm)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and why in out and "DRYOK" not in out, out
    assert not root.exists() or not list(root.iterdir()), "a refused call must not create the arm's directory"


@R2.needs_bash
@pytest.mark.parametrize("kind", ["box", "box-b3"])
def test_RED_dry_run_refuses_an_unknown_box_lever_in_either_box_kind(tmp_path, kind):
    code = tmp_path / "code"
    (code / "stack" / "scripts").mkdir(parents=True)
    (code / "stack" / "ops" / "runs.d").mkdir(parents=True)
    (code / "stack" / "scripts" / "launch_gate.py").write_bytes((ROOT / "scripts" / "launch_gate.py").read_bytes())
    unknown = LG.set_flag(B1, "--r9-box-foo", [])
    (code / "stack" / "ops" / "runs.d" / "r9e1_X.argv.json").write_text(json.dumps({"argv": unknown}), encoding="utf-8")
    out, root = R2._bind(tmp_path / "run", kind, "X", code=code)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "unknown box lever (--r9-box-foo): classify it in launch_gate.py" in out and "DRYOK" not in out, out
    assert not root.exists() or not list(root.iterdir())


@R2.needs_bash
def test_the_box_kind_renders_exactly_the_direct_template_for_the_aux_arms_too(tmp_path):
    """The relaxed refusal must not change a byte of the `box` args: render the template directly with the script's own substitutions."""
    for arm in ARGV:
        out, root = R2._bind(tmp_path / arm, "box", arm)
        assert "DRYOK" in out
        got = (root / f"ovf_{arm}_{R2.HASH[:12]}" / "box.args").read_text(encoding="utf-8")
        bd = (root / f"ovf_{arm}_{R2.HASH[:12]}").as_posix()
        tpl = (ROOT / "scripts" / "r9e1_bind_box.args.tpl").read_text(encoding="utf-8")
        want = [ln for ln in (tpl.replace("\r", "").split("\n")) if ln.strip() and not ln.startswith("#") and not ln.startswith("DATA-ROOT")]
        want = [ln.replace("{ARGV_SHA}", LG.argv_sha256(_argv(arm))).replace("{ARGV}", (RUNS / f"r9e1_{arm}.argv.json").as_posix())
                .replace("{OUT}", bd + "/run_box").replace("{COMMIT}", R2.HASH) for ln in want]
        assert [ln for ln in got.replace("\r", "").split("\n") if ln] == want


def test_every_python_snippet_in_the_bind_script_still_compiles_and_the_new_ones_are_found():
    sn = R2._snippets()
    assert len(sn) >= 5
    for i, code in enumerate(sn):
        compile(code, f"r9e1_bind.sh snippet {i}", "exec")
    assert any("E1_BOX_AUX_FLAGS" in c for c in sn) and any("AUXJUDGE-OK" in c for c in sn)


def test_the_aux_assertion_snippet_passes_a_stamped_record_and_refuses_a_missing_foreign_or_unstamped_one(tmp_path):
    code = next(c for c in R2._snippets() if "AUXJUDGE-OK" in c)
    argv_file = RUNS / "r9e1_B2.argv.json"

    def run(rec):
        r = subprocess.run([sys.executable, "-I", "-c", code, str(R2.CODE_TREE), str(rec), str(argv_file), R2.HASH], capture_output=True, text=True, encoding="utf-8",
                           env=dict(__import__("os").environ, PYTHONDONTWRITEBYTECODE="1"), timeout=300)
        assert r.returncode == 0, r.stderr[-600:]
        return r.stdout
    good = Path(_write(tmp_path, _box1(B2), "good.json"))
    assert "AUXJUDGE-OK" in run(good) and "REFUSED" not in run(good)
    for name, rec in (("unstamped", TG._box_main_only(LG.argv_sha256(B2))), ("foreign", _box1(B1)),
                      ("other_stamp", _box1(B2, **{G.AUX_STAMP: ["--r9-box-bev2-aux"]}))):
        out = run(Path(_write(tmp_path, rec, name + ".json")))
        assert "REFUSED:" in out and "AUXJUDGE-OK" not in out, (name, out)
    # the snippet reaches the GATE'S judge (not a copy of it): the stamp clause is called from `judge_box_overfit` and from the own-record judge
    assert "box_aux_stamp_reasons(tag, rec, argv)" in inspect.getsource(LG.judge_box_overfit)
    assert "box_aux_stamp_reasons(tag, rec, argv)" in inspect.getsource(LG.judge_box_own_record)
