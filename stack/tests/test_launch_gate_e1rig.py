"""refcv9 C12-E1 -- the launch-gate profile ``refcv9-e1rig`` for the E1 perception-only rig (``--r9-perc-rig``, refcv9 C7;
REFCV9_DESIGN 4.3). The gate could not mint a PASS for ANY E1 arm: G-LIVE demanded the planner objective (`traj` / `cls`) the rig
never builds, `PROFILES["refcv8"]` required planner flags `refcv9_perc_rig.pin_args` refuses, and G-EVAL built the FULL model.

Every expectation is a LITERAL; every guard has a DELIBERATE-REGRESSION arm that must make the gate FAIL with its named reason:

  R1  a rig argv carrying `--w-tac-v6 1.0`                                  -> forbidden lever (argv rules)
  R2  a rig that builds `core.decoder`                                      -> G-DVB (`r9_perc_rig` lever) + G-LIVE's absent-rule
  R3  `--init-from` a different file                                        -> md5 refusal (map_only_init_reasons' own reader)
  R4  `--steps 3000`                                                        -> required value
  R5  the rig argv under profile `refcv8` (and the whole refcv8 family)     -> refused, naming `refcv9-e1rig`
  R6  a smoke whose `compute_losses_perc_rig` drops `box3d` from `extra`    -> G-LIVE FAIL (declared term absent)
  R7  an eval loader that builds the full model                             -> G-EVAL FAIL (`rig_eval_build_reasons`)

OFF is inert: every switch is keyed on `args.r9_perc_rig` / the profile's own keys, so the existing profiles' verdicts are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RIG = LG.PROFILES["refcv9-e1rig"]
R8 = LG.PROFILES["refcv8"]
RUNS = ROOT / "ops" / "runs.d"
C0 = json.loads((RUNS / "r9e1_C0.argv.json").read_text(encoding="utf-8"))["argv"]
C1 = json.loads((RUNS / "r9e1_C1.argv.json").read_text(encoding="utf-8"))["argv"]
R8_LAUNCH = json.loads((RUNS / "refcv8-r101-s0.argv.json").read_text(encoding="utf-8"))["argv"]
THOR_INIT = "/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt"
SHIPPED_EMA_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"
MASKS = "/home/nvidia/r8ns/r9e1/launch_tree/stack/tanitad/configs/refcv8_join_label_defects_train4572.json"


def _rig_module():
    from tanitad.refs import refcv9_perc_rig as rig
    return rig


@pytest.fixture(scope="module")
def conformant(tmp_path_factory):
    """C0 / C1 (the committed launch argvs, which carry `--join-defect-masks` since MM decision (a), 2026-11-02) + a stand-in
    for the shipped EMA model, whose md5 the profile is told to expect."""
    tmp = tmp_path_factory.mktemp("e1rig")
    init = tmp / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped refcv8 EMA-48,852")
    prof = dict(RIG, init_pin=dict(RIG["init_pin"], init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    arms = {"C0": list(C0), "C1": list(C1)}                  # as committed: no flag is injected any more
    return arms, prof, [("/home/nvidia/r8ns/ckpt", str(tmp))], init


def _rules(prof, argv, pmap):
    return LG.profile_argv_rules(prof, argv, {}, pmap)


# ============================================================================================== the profile ==
def test_profile_literals():
    assert RIG["name"] == "refcv9-e1rig" and RIG["required_checks"] == R8["required_checks"]
    assert len(RIG["required_checks"]) == 10 and "G-BOX-OVERFIT" in RIG["required_checks"]
    for f in ("--r9-perc-rig", "--init-from", "--seed", "--steps", "--warmup", "--ema-decay", "--lr", "--out"):
        assert f in RIG["required_flags"], f
    assert len(RIG["required_flags"]) == len(set(RIG["required_flags"]))
    rv = RIG["required_values"]
    assert (rv["--steps"], rv["--warmup"], rv["--ema-decay"], rv["--lr"], rv["--batch"]) == ("4000", "200", "0.999", "1e-4", "16")
    assert (rv["--w-map"], rv["--w-map-hires"], rv["--map-hires-decision-rule"]) == ("0", "1.0", "prior_corrected")
    assert "--seed" not in rv                                   # C0 = 0, C1 = 1: the replicate floor
    assert RIG["live_dead_admitted"] == () and RIG["required_on"] == ()
    assert RIG["map_hires"]["ga_keys_pool"] == () and RIG["map_hires"]["ga_keys"] == R8["map_hires"]["ga_keys"]
    assert "ga_keys_zero" not in RIG["map_hires"]               # the map branch TRAINS: all four parts > 0
    assert RIG["init_pin"]["init_from_md5"] == SHIPPED_EMA_MD5 == LG._MAPFT_INIT_MD5
    assert "map_only" not in RIG                                # no freeze semantics: nothing is frozen in the rig
    assert RIG["only_values"] == {"--residual-prior": ("off",)} and RIG["forbidden_values"] == {}
    assert RIG["required_positive"] == ("--w-map-hires", "--w-agent", "--w-box3d")
    assert RIG["box_overfit"] == R8["box_overfit"] and RIG["map_overfit"] == R8["map_overfit"]   # the INHERITED records


def test_required_values_partition_refcv8s_table_and_name_every_removal():
    """refcv8's required_values are KEPT (equal), overridden (the E1 recipe) or REMOVED -- and every removal is named in
    required_values_why. A value refcv8 adds tomorrow falls in none and must be decided here."""
    kept = {f for f in R8["required_values"] if f in RIG["required_values"] and RIG["required_values"][f] == R8["required_values"][f]}
    recipe = {"--steps", "--warmup", "--ema-decay"}
    removed = set(R8["required_values"]) - kept - recipe
    assert removed == set(LG._E1_REMOVED)
    assert len(kept) == 22 and len(removed) == 24 and len(R8["required_values"]) == 49
    for f in removed:
        assert f not in RIG["required_values"] and RIG["required_values_why"][f].startswith("REMOVED for the rig"), f
    for f in ("--residual-prior", "--w-tac-v6", "--w-r8-cons", "--r8-v9-labels", "--nav-compliance-tau-rad"):
        assert f in removed, f
    assert set(RIG["required_values"]) == kept | recipe | {"--lr", "--agents"}


def test_forbidden_levers_import_the_planner_dests_and_each_is_a_real_flag():
    rig = _rig_module()
    names = [e[0] for e in RIG["forbidden_levers"]]
    for d in rig.PLANNER_FLAG_DESTS + rig.PLANNER_WEIGHT_DESTS:
        assert names.count(("--" + d.replace("_", "-"),)) == 1, d           # ONE entry per dest, by flag name
    assert LG._E1_PLANNER_FLAGS == tuple("--" + d.replace("_", "-") for d in rig.PLANNER_FLAG_DESTS)
    assert LG._E1_MODULE.PLANNER_FLAG_DESTS == rig.PLANNER_FLAG_DESTS and LG._E1_MODULE.PLANNER_WEIGHT_DESTS == rig.PLANNER_WEIGHT_DESTS
    assert ("--r9-perc-rig",) not in names and ("--r9-train-only",) in names        # the profile's own flag is not refused
    # the refcv8 entries are all inherited (minus the one that names this profile's flag)
    assert [e for e in R8["forbidden_levers"] if e[0] != ("--r9-perc-rig",)] == list(RIG["forbidden_levers"][:len(R8["forbidden_levers"]) - 1])
    # a refused name that is not a trainer flag is a DEAD guard (built, tested, unreachable): every one is a parser option
    T = pytest.importorskip("refc_v3_train")
    opts = {o for a in T.build_parser()._actions for o in a.option_strings}
    dead = [f for f in LG._E1_PLANNER_FLAGS + LG._E1_PLANNER_WEIGHT_FLAGS + LG._E1_R9_PLANNER_FLAGS if f not in opts]
    assert dead == []
    assert "--r9-perc-rig" in opts and "--r9-train-only" in opts


# ========================================================================= the argv rules (R1, R3, R4, R5) ==
@pytest.mark.parametrize("name", ["C0", "C1"])
def test_conformant_rig_argvs_have_no_refusal_no_pi_pending_and_no_open_item(conformant, name):
    arms, prof, pmap, _ = conformant
    refusals, pending = _rules(prof, arms[name], pmap)
    assert refusals == [] and pending == [], refusals
    assert LG.open_item_reasons(prof, arms[name]) == []                  # the replicate floor rides the inherited records
    assert LG.flag_values(arms[name], "--seed") == [name[1]]             # C0 = seed 0, C1 = seed 1: the only difference


def test_C0_and_C1_differ_only_in_seed_and_out():
    pa, pb = dict(LG.flag_pairs(C0)), dict(LG.flag_pairs(C1))
    assert {f for f in set(pa) | set(pb) if pa.get(f) != pb.get(f)} == {"--seed", "--out"}


def test_R1_a_planner_flag_is_refused_whatever_its_value(conformant):
    arms, prof, pmap, _ = conformant
    r, _p = _rules(prof, LG.set_flag(arms["C0"], "--w-tac-v6", ["1.0"]), pmap)
    assert any(x.startswith("--w-tac-v6 is passed, but it is a planner / tactical LOSS WEIGHT") for x in r), r
    rig = _rig_module()
    for d in rig.PLANNER_FLAG_DESTS + rig.PLANNER_WEIGHT_DESTS:
        flag = "--" + d.replace("_", "-")
        r, _p = _rules(prof, LG.set_flag(arms["C0"], flag, ["1.0"] if d in rig.PLANNER_WEIGHT_DESTS else []), pmap)
        assert any(x.startswith(f"{flag} is passed") for x in r), (flag, r)
    assert _rules(prof, arms["C0"], pmap)[0] == []                        # control: the conformant argv has none of them


@pytest.mark.parametrize("flag,vals,needle", [
    ("--r8-n-alloc", ["32"], "a refcv8 tactical / planner seam flag"),
    ("--w-r8-sat", ["0.1"], "a planner / tactical LOSS WEIGHT"),
    ("--r9-train-only", ["planner,tactical,map"], "refcv9-warm"),
    ("--r9-progress-head", [], "planner-side refcv9 flag"),
    ("--r9-alloc-force-schedule", ["linear:1.0:0.0:0:2000"], "planner-side refcv9 flag"),
    ("--residual-prior", ["ha0_ext_pose"], "the only admitted value(s): ['off']"),
    ("--residual-prior", ["ha0_ext"], "the only admitted value(s): ['off']"),
    ("--residual-prior", ["cv_yawrate"], "the only admitted value(s): ['off']"),
    ("--residual-prior", [], "the only admitted value(s): ['off']"),
    ("--steps", ["3000"], "--steps 4000 is REQUIRED"),
    ("--steps", None, "requires --steps"),
    ("--warmup", ["1000"], "--warmup 200 is REQUIRED"),
    ("--ema-decay", ["0.9998"], "--ema-decay 0.999 is REQUIRED"),
    ("--lr", ["5e-5"], "--lr 1e-4 is REQUIRED"),
    ("--lr", None, "requires --lr"),
    ("--seed", None, "requires --seed"),
    ("--r9-perc-rig", None, "requires --r9-perc-rig"),
    ("--init-from", None, "requires --init-from"),
    ("--batch", ["8"], "--batch 16 is REQUIRED"),
    ("--w-map", ["0.5"], "--w-map 0 is REQUIRED"),
    ("--w-map-hires", ["0"], "--w-map-hires"),
    ("--w-box3d", ["0"], "--w-box3d must be passed with a value > 0"),
    ("--w-agent", None, "requires --w-agent"),
    ("--agents", ["off"], "--agents head is REQUIRED"),
    ("--map-hires", ["off"], "--map-hires on is REQUIRED"),
    ("--v2-cache", ["/home/nvidia/data/refcv6-b1-416x1024-train"], "--v2-cache"),
    ("--join-defect-masks", None, "requires --join-defect-masks")])
def test_each_rule_refuses(conformant, flag, vals, needle):
    arms, prof, pmap, _ = conformant
    r, _p = _rules(prof, LG.set_flag(arms["C0"], flag, vals), pmap)
    assert any(needle in x for x in r), (flag, vals, r)


def test_residual_prior_off_is_admitted(conformant):
    arms, prof, pmap, _ = conformant
    assert _rules(prof, LG.set_flag(arms["C0"], "--residual-prior", ["off"]), pmap)[0] == []
    # the allow-list is inert on every other profile (a profile without `only_values`)
    assert LG.forbidden_value_reasons(R8, ["--residual-prior", "ha0_ext_pose"]) == []
    assert LG.forbidden_value_reasons(R8, ["--residual-prior", "off"]) != []          # refcv8's deny-list, unchanged


def test_R3_init_from_a_different_file_is_refused_by_the_md5_reader(conformant, tmp_path):
    arms, prof, pmap, _ = conformant
    other = tmp_path / "ckpt_ema_48852.pt"
    other.write_bytes(b"a different checkpoint")
    r, det = LG.map_only_init_reasons(prof, arms["C0"], [("/home/nvidia/r8ns/ckpt", str(tmp_path))])
    assert len(r) == 1 and r[0].startswith("--r9-perc-rig:") and "the profile pins" in r[0]
    assert r[0].endswith("the rig would not warm-start from the shipped refcv8 EMA-48,852") and det["md5"] != det["md5_pinned"]
    r2, _p = _rules(prof, arms["C0"], [("/home/nvidia/r8ns/ckpt", str(tmp_path))])
    assert any("has md5" in x for x in r2), r2
    r3, _d = LG.map_only_init_reasons(RIG, arms["C0"], pmap)                      # the REAL literal against the stand-in: refused
    assert len(r3) == 1 and SHIPPED_EMA_MD5 in r3[0]
    # one md5 reader: refcv9-warm's text is unchanged (its default note)
    rw = LG.map_only_init_reasons(LG.PROFILES["refcv9-warm"], ["--init-from", THOR_INIT], pmap)[0]
    assert rw[0].startswith("--r9-train-only:") and rw[0].endswith("the frozen planner would not be the shipped one")


def test_R4_a_missing_init_from_file_cannot_be_bound():
    r, _d = LG.map_only_init_reasons(RIG, C0, [])
    assert len(r) == 1 and "is not readable on this host" in r[0] and r[0].startswith("--r9-perc-rig:")


@pytest.mark.parametrize("name", ["refcv8", "refcv8-nt", "refcv8-mapft", "v8f", "v8f_steer", "refcv9-warm"])
def test_R5_the_rig_is_not_launchable_under_the_refcv8_family(name):
    prof = LG.PROFILES[name]
    r = LG.forbidden_lever_reasons(prof, ["--arm", "flat", "--r9-perc-rig"])[0]
    assert len(r) == 1 and "--r9-perc-rig" in r[0] and "refcv9-e1rig" in r[0], r
    assert LG.forbidden_lever_reasons(prof, ["--arm", "flat"])[0] == []            # control: absent flag, no refusal
    assert LG.forbidden_lever_reasons(RIG, ["--arm", "flat", "--r9-perc-rig"])[0] == []   # the rig profile admits it


def test_R5_the_real_rig_argv_under_refcv8_is_refused_by_the_gate():
    r, _p = LG.profile_argv_rules(R8, C0, {}, [])
    assert any("--r9-perc-rig is passed, but it is the refcv9 E1 perception-only rig" in x for x in r), r[:3]


# ============================================================================================ open items ==
def test_conditional_open_items_key_on_the_lever_flags_and_never_pass(conformant):
    arms, prof, pmap, _ = conformant
    ids = lambda argv: [x.split(":")[0] for x in LG.open_item_reasons(prof, argv)]
    c0 = arms["C0"]
    assert ids(c0) == []
    assert ids(LG.set_flag(c0, "--r9-map-banded", [])) == ["OPEN ITEM E1-MAP-OWN-RECORD"]
    assert ids(LG.set_flag(c0, "--w-map-fp", ["0.5"])) == ["OPEN ITEM E1-MAP-OWN-RECORD"]
    assert ids(LG.set_flag(c0, "--r9-box-depth-aux", [])) == ["OPEN ITEM E1-BOX-OWN-RECORD"]
    assert ids(LG.set_flag(LG.set_flag(c0, "--r9-map-banded", []), "--r9-dn-groups", ["3"])) == [
        "OPEN ITEM E1-MAP-OWN-RECORD", "OPEN ITEM E1-BOX-OWN-RECORD"]
    assert "[triggered by ['--r9-map-banded']]" in LG.open_item_reasons(prof, LG.set_flag(c0, "--r9-map-banded", []))[0]
    # no argv to read => fail closed: both apply
    assert len(LG.open_item_reasons(prof)) == 2
    # finalize reads the argv and turns PASS / PI-DECISION into INCOMPLETE on any open item
    src = inspect.getsource(LG.finalize)
    assert "open_item_reasons(ctx.prof, ctx.argv)" in src and 'verdict in ("PASS", "PI-DECISION")' in src
    # every OTHER profile's items are unconditional and read as before (the argv changes nothing)
    for name, p in LG.PROFILES.items():
        if name != "refcv9-e1rig":
            assert LG.open_item_reasons(p) == LG.open_item_reasons(p, ["--r9-map-banded"]) == LG.open_item_reasons(p, [])


# ============================================================================== G-LIVE (R2 live, R6) ==
def _rig_args():
    T = pytest.importorskip("refc_v3_train")
    return T, T.build_parser().parse_args(C0)


def _step(keys, loss):
    return {"keys": list(keys), "nonfinite": [], "vals": {k: (loss if k == "loss" else 0.5) for k in keys}}


def _live(terms, keys, *, n=100, absent=None):
    steps = [_step(keys, 10.0 - 0.01 * i) for i in range(n)]
    row = {"step": n, **{k: (round(steps[-1]["vals"][k], 5)) for k in keys}}
    grads = {"n_trainable_params": 5, "nonfinite_params": [], "dead_groups": []}
    return LG.judge_live(terms, [], steps, expect_steps=n, summary={"done": True, "step": n}, train_rows=[row], grads=grads,
                         prior=None, exit_msg=None, absent_terms=absent)


def test_G_LIVE_declares_the_rig_terms_instead_of_the_planner_objective():
    T, args = _rig_args()
    terms, problems = LG.declared_terms(T, args)
    ids = [t["id"] for t in terms]
    assert problems == [] and ids[0] == "perc_rig" and "planner" not in ids
    assert terms[0]["keys"] == ["loss"] and {"w_agent", "w_box3d", "w_map_hires"} <= set(ids)
    assert not {"w_tac_v6", "w_u0", "w_tac_goal", "w_r7_wta", "w_r7_scorer"} & set(ids) and not any(i.startswith("w_r8_") for i in ids)
    keys = [k for t in terms for k in t["keys"]]
    assert {"loss", "box3d", "map_hires", "agent_presence", "agent_centre"} <= set(keys)
    # OFF: the planner objective, exactly as before (refcv8's own argv)
    off = T.build_parser().parse_args(R8_LAUNCH)
    assert [t["id"] for t in LG.declared_terms(T, off)[0]][0] == "planner"
    assert LG.declared_terms(T, off)[0][0]["keys"] == list(LG.LIVE_ALWAYS[0]) and LG.rig_absent_terms(off) == []


def test_G_LIVE_a_healthy_rig_smoke_passes_and_the_planner_smoke_is_unchanged():
    T, args = _rig_args()
    terms, _ = LG.declared_terms(T, args)
    keys = sorted({k for t in terms for k in t["keys"]})
    reasons, det = _live(terms, keys, absent=LG.rig_absent_terms(args))
    assert reasons == [] and det["absent_terms"][0]["emitted"] == {}
    # OFF: judge_live without the rule has no `absent_terms` row and the same verdict
    assert "absent_terms" not in _live(terms, keys)[1]


def test_R6_a_smoke_whose_perc_rig_loss_drops_box3d_fails_G_LIVE():
    T, args = _rig_args()
    terms, _ = LG.declared_terms(T, args)
    keys = sorted({k for t in terms for k in t["keys"]})
    reasons, _d = _live(terms, [k for k in keys if k != "box3d"], absent=LG.rig_absent_terms(args))
    assert any("declared term `box3d`" in r and "ABSENT from all 100" in r for r in reasons), reasons
    reasons, _d = _live(terms, [k for k in keys if k != "map_hires"], absent=LG.rig_absent_terms(args))
    assert any("declared term `map_hires`" in r for r in reasons), reasons
    reasons, _d = _live(terms, [k for k in keys if k != "loss"], absent=LG.rig_absent_terms(args))
    assert any("declared term `loss`" in r for r in reasons), reasons


@pytest.mark.parametrize("planner_key", ["traj", "cls", "tac_v6", "r8_cons", "r8_alloc_l1", "r8", "cascade", "goal_tac", "lat",
                                         "lon", "law", "sel_v3", "u0", "tac_goal", "r7_wta", "lat_tac", "lon_tac"])
def test_R2_live_a_rig_smoke_that_emits_a_planner_term_fails_G_LIVE(planner_key):
    T, args = _rig_args()
    terms, _ = LG.declared_terms(T, args)
    keys = sorted({k for t in terms for k in t["keys"]})
    reasons, det = _live(terms, keys + [planner_key], absent=LG.rig_absent_terms(args))
    assert any(r.startswith("the rig emitted a planner term: the planner was built") and planner_key in r for r in reasons), reasons
    assert det["absent_terms"][0]["emitted"] == {planner_key: 100}
    # control: the same smoke WITHOUT the absent-rule (the OFF path) does not read it
    assert not any("the rig emitted" in r for r in _live(terms, keys + [planner_key])[0])


def test_job_smoke_hands_the_absent_rule_to_judge_live():
    """G-LIVE's verdict is `judge_live` called from `job_smoke`; the rule is built, tested and UNREACHABLE unless that call site passes
    it (MUTATION: dropping `absent_terms=absent` from the call turned nothing red until this pin)."""
    src = inspect.getsource(LG.job_smoke)
    assert "absent = rig_absent_terms(args)" in src and "admitted_dead=admitted, absent_terms=absent)" in src
    assert "terms, problems = declared_terms(T, args)" in src


def test_perception_keys_never_match_the_absent_rule():
    T, args = _rig_args()
    perception = ["loss", "agent", "agent_presence", "agent_centre", "agent_n_matched", "box3d", "box3d_n_z", "map", "map_hires",
                  "map_hires_n_labelled", "n_map_cells", "map_iou_drivable", "map_hires_lc_drivable_0_20", "map_fp"]
    reasons, det = _live(LG.declared_terms(T, args)[0], perception, absent=LG.rig_absent_terms(args))
    assert det["absent_terms"][0]["emitted"] == {}, det["absent_terms"]


def test_the_trainer_dispatches_the_rig_loss_through_compute_losses_v3_which_the_smoke_wraps():
    """The smoke's observer wraps `T.compute_losses_v3`; the rig's loss reaches it through that dispatch (refc_v3_train.py:
    `if getattr(model.core, '_perc_rig'...): return compute_losses_perc_rig(...)`), so the observed keys ARE the rig's."""
    T = pytest.importorskip("refc_v3_train")
    src = inspect.getsource(T.compute_losses_v3)
    assert "_perc_rig" in src and "return compute_losses_perc_rig(" in src
    assert "pat.set(T, \"compute_losses_v3\", losses_wrap)" in inspect.getsource(LG.run_smoke)


# ================================================================================ G-DVB (R2) and G-EVAL (R7) ==
_ORIG_MODEL: dict = {}


def _tiny(monkeypatch, perc_rig):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import test_map_only_ft as R
    from tanitad.refs import refc_v3 as v3
    orig = _ORIG_MODEL.setdefault("cls", v3.RefCV3Model)           # the ORIGINAL class, captured once (a patch is undone per test)
    monkeypatch.setattr(v3, "RefCV3Model", lambda cfg: orig(cfg, perc_rig=perc_rig))
    return R._build()[0]


def test_R2_G_DVB_calls_the_rig_lever_and_a_rig_that_builds_the_planner_fails(monkeypatch):
    T, args = _rig_args()
    from tanitad.train import declared_vs_built as D
    assert D.REGISTRY["r9_perc_rig"].kind == "built"                          # A's lever (registry entry 299), registered at import
    assert "fn(model, args, parser" in inspect.getsource(LG.judge_dvb)        # the gate calls `check`, which runs the registry
    parser = T.build_parser()
    full = _tiny(monkeypatch, perc_rig=False)                                  # a "rig" that built core.decoder
    assert hasattr(full.core, "decoder")
    reasons, det = LG.judge_dvb(D, full, args, parser, C0, "")
    assert reasons and any("r9_perc_rig" in str(m) for m in det["mismatches"]), det["mismatches"][:3]
    assert any(m["lever"] == "r9_perc_rig" and m["declared"] is True and m["built"] is False for m in det["mismatches"]
               if isinstance(m, dict)), det["mismatches"][:3]
    assert any(m["lever"] == "r9_perc_rig" and m["declared"] == "planner / tactical NOT built" and m["built"] == "built"
               for m in det["mismatches"] if isinstance(m, dict))
    rig = _tiny(monkeypatch, perc_rig=True)                                    # control: the real rig reads clean on that lever
    _r, det2 = LG.judge_dvb(D, rig, args, parser, C0, "")
    assert not any("r9_perc_rig" in str(m) for m in det2["mismatches"])


def test_R2_G_DVB_via_the_gate_the_planner_attribute_paths_are_the_modules_own():
    rig = _rig_module()
    assert LG._e1_rig_module().PLANNER_ATTR_PATHS == rig.PLANNER_ATTR_PATHS and "core.decoder" in rig.PLANNER_ATTR_PATHS


def test_R7_an_eval_loader_that_builds_the_full_model_fails_G_EVAL(monkeypatch):
    T, args = _rig_args()
    rig_m = _tiny(monkeypatch, perc_rig=True)
    full_m = _tiny(monkeypatch, perc_rig=False)
    r, det = LG.rig_eval_build_reasons(args, rig_m, full_m)
    assert len(r) == 1 and "the eval loader built the FULL model" in r[0] and "core.decoder" in r[0], r
    assert det["trainer_rig"] is True and det["loader_rig"] is False and "core.decoder" in det["loader_planner_paths"]
    # a loader that builds the rig reads clean
    r2, det2 = LG.rig_eval_build_reasons(args, rig_m, _tiny(monkeypatch, perc_rig=True))
    assert r2 == [] and det2["loader_rig"] is True and det2["loader_planner_paths"] == []
    # a trainer that did not build the rig for a rig launch is a reason too
    r3, _d = LG.rig_eval_build_reasons(args, full_m, full_m)
    assert any("TRAINER's model is not the rig" in x for x in r3)
    # OFF: inert for every other launch
    off = T.build_parser().parse_args(R8_LAUNCH)
    assert LG.rig_eval_build_reasons(off, full_m, rig_m) == ([], {"on": False})


def test_R7_eval_identity_reads_the_rig_build_and_hints_on_a_strict_load_failure():
    src = inspect.getsource(LG.eval_identity)
    assert "rig_eval_build_reasons(args_t, model_t, model_l)" in src and "reasons += rbr" in src
    assert 'getattr(args_t, "r9_perc_rig", False)' in src and "G-EVAL under the rig" in src
    assert "the eval loader's STRICT load failed: {str(e)[:600]}{hint}" in src      # OFF: hint == "" -> the old string


# ============================================================================== the eval loader (G-EVAL) ==
def test_G_EVAL_the_eval_loader_builds_the_rig_from_the_run_record():
    """`refcv7_loader._build_model` replays train()'s build: the rig behind `--r9-perc-rig` (pin first), the original line
    otherwise. The snippet is executed with stubs: flag ON -> `RefCV3Model(cfg, perc_rig=True)` after `pin_args`; OFF -> the
    original call, bit for bit (no kwarg)."""
    from tanitad.eval import refcv7_loader as L
    src = inspect.getsource(L._build_model)
    start = src.index("    _r9rig_on = bool(getattr(args, \"r9_perc_rig\", False))")
    end = src.index(".to(device)", start) + len(".to(device)")
    snippet = textwrap.dedent(src[start:end])
    assert snippet.count("\n") == 3 and "tr._r9rig_mod.pin_args(args)" in snippet and "perc_rig=True" in snippet

    def run(args):
        log = []

        class V3:
            @staticmethod
            def RefCV3Model(cfg, **kw):
                log.append(("build", dict(kw)))
                return types.SimpleNamespace(to=lambda d: ("model", d))

        tr = types.SimpleNamespace(_r9rig_mod=types.SimpleNamespace(pin_args=lambda a: log.append(("pin",))))
        ns = {"args": args, "tr": tr, "v3": V3, "cfg": object(), "device": "cpu"}
        exec(snippet.replace("model = ", "out = ", 1), ns)
        return log, ns["out"]

    assert run(types.SimpleNamespace(r9_perc_rig=True)) == ([("pin",), ("build", {"perc_rig": True})], ("model", "cpu"))
    assert run(types.SimpleNamespace(r9_perc_rig=False)) == ([("build", {})], ("model", "cpu"))
    assert run(types.SimpleNamespace()) == ([("build", {})], ("model", "cpu"))          # a pre-C7 args namespace: unchanged
    # MUTATION: the pre-edit line (the loader that builds the FULL model for every launch) carries neither the pin nor the kwarg
    assert "else v3.RefCV3Model(cfg)).to(device)" in src and src.count("v3.RefCV3Model(") == 2


# ====================================================================================== bits of the profile ==
def test_the_profile_is_built_by_subtraction_and_registered_only_with_the_rig_module():
    assert LG._E1_MODULE is not None and "refcv9-e1rig" in LG.PROFILES
    assert RIG["trainer"] == R8["trainer"] and RIG["eval_loader"] == R8["eval_loader"] and RIG["stages"] == R8["stages"]
    assert RIG["suite_env_failures"] == LG.PROFILES["refcv9-warm"]["suite_env_failures"]
    assert RIG["forbidden_prefixes"] == tuple(R8["forbidden_prefixes"]) + ("--r8-", "--w-r8-")
    # refcv8's own table is untouched by the rig profile (the dict is copied, not shared)
    assert R8["required_values"]["--steps"] == "48852" and R8["live_dead_admitted"] != ()
    assert R8["forbidden_levers"][-1][0] == ("--r9-perc-rig",)



# ============================================================================================================ #
# phase B (MM decisions of 2026-11-02): (a) the join-defect masks are in the argvs, (d) the cost rule reads STEADY steps #
# ============================================================================================================ #
import r9_e1_make_argv as M  # noqa: E402  (scripts/ is on sys.path)

MASKS_MD5 = "b2591cb2ec4f4089831124e4738a961c"
MASKS_FILE = ROOT / "tanitad" / "configs" / "refcv8_join_label_defects_train4572.json"


def test_a_C0_and_C1_carry_refcv8s_join_defect_masks_inside_the_launch_tree():
    want = "/home/nvidia/r8ns/r9e1/launch_tree/stack/tanitad/configs/refcv8_join_label_defects_train4572.json"
    assert MASKS == want == M.LAUNCH_TREE + "/" + M.JOIN_MASKS_REL
    for name, a in (("C0", C0), ("C1", C1)):
        assert LG.flag_values(a, "--join-defect-masks") == [want] and a.count("--join-defect-masks") == 1, name
    # the file the launch tree ships IS refcv8's file (md5 pinned in the gate's own why-line and in the builder)
    assert hashlib.md5(MASKS_FILE.read_bytes()).hexdigest() == MASKS_MD5 == M.JOIN_MASKS_MD5
    assert MASKS_MD5 in R8["required_flags_why"]["--join-defect-masks"] and "--join-defect-masks" in RIG["required_flags"]
    assert "--join-defect-masks" in RIG["model_input_flags"]           # G-HYG / G-DVB / G-EVAL OPEN it for real (not a sentinel)


def test_a_RED_the_argv_without_the_mask_is_refused_and_so_is_a_tampered_or_foreign_file(conformant, tmp_path):
    arms, prof, pmap, _ = conformant
    ok, _p = _rules(prof, arms["C0"], pmap)
    assert ok == []                                                       # control: the committed argv has no refusal
    bare = LG.set_flag(arms["C0"], "--join-defect-masks", None)
    assert "--join-defect-masks" not in bare
    refused, _p = _rules(prof, bare, pmap)
    assert any("--join-defect-masks" in r for r in refused), refused      # the profile REQUIRES it
    # the builder: refuses a tampered copy of the file, a base argv naming another file, a launch tree that is not on Thor
    bad = tmp_path / "masks.json"
    bad.write_bytes(MASKS_FILE.read_bytes() + b" ")
    with pytest.raises(SystemExit):
        M.check_join_masks([], repo_file=bad)
    assert M.check_join_masks([], repo_file=MASKS_FILE) == MASKS_MD5
    with pytest.raises(SystemExit):
        M.check_join_masks(["--join-defect-masks", "/x/refcv8_join_label_defects.json"], repo_file=MASKS_FILE)
    for tree in ("C:/Program Files/Git/home/nvidia/r8ns/r9e1/launch_tree", "relative/launch_tree"):
        with pytest.raises(SystemExit):
            M.build(["--w-agent", "1.0"], seed=0, out="/o", launch_tree=tree)
    built = M.build(["--w-agent", "1.0"], seed=0, out="/o", launch_tree="/home/nvidia/r8ns/x/launch_tree/")
    assert LG.flag_values(built, "--join-defect-masks") == ["/home/nvidia/r8ns/x/launch_tree/" + M.JOIN_MASKS_REL]


def _steady(n, base=4.7, spike_at=None, spike=195.0):
    t = [0.0]
    for j in range(n - 1):
        t.append(t[-1] + (base + spike if j == spike_at else base))
    return t


def test_d_step_cost_reads_the_steady_step_not_the_eval_and_ckpt_row():
    # S1's measured rig: steady 4.70 s/step; its 50-step smoke carried an eval + a ckpt save at step 25 and read 9.753 s/step
    t = _steady(60, spike_at=24)                                           # step 25's work sits in interval 24
    polluted = LG.step_cost(t, 10)                                         # deliberate regression: no exclusion -> a false PI
    assert polluted["s_per_step_mean"] > 8.0 > 4.7
    fixed = LG.step_cost(t, 10, nontrain_rows=[25])
    assert fixed["s_per_step_mean"] == 4.7 and fixed["s_per_step_max"] == 4.7
    assert fixed["nontrain_intervals_excluded"] == 3 and fixed["nontrain_rows"] == [25] and "reading_rows" not in fixed
    for j in (23, 25):                                                     # the work can sit on either side of the timestamp
        assert LG.step_cost(_steady(60, spike_at=j), 10, nontrain_rows=[25])["s_per_step_mean"] == 4.7
    # control: a slow interval NOT on a listed row is real cost and is kept
    assert LG.step_cost(_steady(60, spike_at=40), 10, nontrain_rows=[25])["s_per_step_mean"] > 8.0
    # without rows the function is the tip's: same keys, no extras
    assert set(LG.step_cost(t, 10)) == {"n_intervals", "n_used", "warmup_skipped", "s_per_step_mean", "s_per_step_median",
                                        "s_per_step_max"}


def test_d_nontrain_rows_literals_follow_the_trainers_own_conditions():
    nt = LG.nontrain_rows
    assert nt(C0) == list(range(500, 4000, 500)) and nt(C1) == nt(C0)       # eval + ckpt every 500 (7 rows; the final step is out)
    assert nt(LG.smoke_argv(C0, 100, 50)[0]) == []                          # the gate's 100-step smoke: only the final step works
    assert nt(["--steps", "50", "--eval-every", "25", "--eval-cache", "/x", "--save-every", "25"]) == [25]
    assert nt(["--steps", "50", "--eval-every", "10", "--save-every", "0"]) == []          # eval needs --eval-cache
    assert nt(["--steps", "1200"]) == [500, 1000]                           # --save-every defaults to 500
    assert nt(["--steps", "6000", "--save-every", "0"]) == [5000]           # MILESTONES
    assert nt(["--steps", "100", "--save-every", "0", "--snapshot-steps", "0,30,100"]) == [30]
    assert nt(["--steps", "50", "--eval-every", "20", "--eval-cache", "/x", "--save-every", "30"]) == [20, 30, 40]
    assert nt([]) == [] and nt(["--steps", "x"]) == []


def test_d_the_pinned_conditions_are_the_trainers_current_source():
    """The rule copies three trainer conditions; a trainer edit that changes one must turn this RED, not silently skew the cost."""
    T = pytest.importorskip("refc_v3_train")
    assert tuple(T.MILESTONES) == LG.TRAINER_MILESTONES
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    for needle in ("if step % args.save_every == 0 or step == args.steps:",
                   "if eval_dl is not None and (step % args.eval_every == 0",
                   "if step in MILESTONES or step in _nt_snaps:",
                   'ap.add_argument("--save-every", type=int, default=500)',
                   'ap.add_argument("--eval-every", type=int, default=0,'):
        assert needle in src, needle


def test_d_judge_new2_live_does_not_send_a_steady_rig_step_to_the_PI_because_of_an_eval_row():
    p = RIG
    h = p["map_hires"]
    ref = float(h["cost_ref_s_per_step"])
    assert abs(float(h["cost_max_ratio"]) * ref - 8.0) < 1e-9               # the > 8.0 s/step PI-DECISION bar
    argv = ["--steps", "60", "--eval-every", "25", "--eval-cache", "/x", "--save-every", "25"]
    run = {"fwd_train_calls": 60, "fmap_hits": 60, "hires_first": ([], {}), "step_times": _steady(60, spike_at=24),
           "cuda_max_mem_gb": 15.84}
    grads = {"groups": [{"group": "core.perception.map_hires_head", "numel": 5, "params": 1, "nz_params": 1}]}
    _r, needs_pi, det = LG.judge_new2_live(p, run, {}, argv, grads, cuda=True, approval=None)
    assert not needs_pi and det["cost"]["s_per_step_mean"] == 4.7 and det["cost"]["verdict"] == "within +25 %"
    assert det["cost"]["nontrain_rows"] == [25, 50]                          # steps 25 and 50 of the 60-step smoke
    # RED arm: the same measurement with the schedule hidden from the rule (the pre-fix behaviour) IS a PI-DECISION
    _r, needs_pi0, det0 = LG.judge_new2_live(p, run, {}, ["--steps", "60"], grads, cuda=True, approval=None)
    assert needs_pi0 and det0["cost"]["s_per_step_mean"] > 8.0



# ============================================================================================================ #
# phase B, decision (b): G-CLOCK stays BINDING on the rig -- it times the labels the rig reads, against the sidecar #
# ============================================================================================================ #
G0, DT, NS = 0.05, 0.1006666, 3


def _true(r):
    return G0 + (r + NS - 1) * DT


def _recs(shift=0.0, n=60, cid="c1", noise=0.004):
    return [(cid, r, _true(r) + shift + (noise if r % 2 else -noise)) for r in range(n)]


def _windows(recs, table=None, ns=None):
    sid_of = {"c1": 1, "c2": 2}.get
    return list(LG.rig_join_clock_windows(recs, sid_of, {1: (G0, DT)} if table is None else table, {1: NS, 2: NS} if ns is None else ns))


def _judge(recs, **kw):
    w = _windows(recs, **kw)
    n_clips = len({x[0] for x in w}) or 1
    return LG.judge_clock_windows(w, 0.05, n_clips=n_clips, cap=0.01)


def test_b_the_rig_clock_passes_an_aligned_join_and_FAILS_every_shifted_clock():
    ok = _judge(_recs())
    assert ok["n_windows"] == 60 and ok["n_violations"] == 0 and ok["legacy_control_violations"] == 60
    assert LG.rig_clock_reasons("agent join", ok, 0.05, 0.01) == []
    # deliberate-regression arms: a join whose registered time is a SHIFTED clock must FAIL, with its named reason
    for shift in (-2 * DT + 0.0058, -DT, +DT, 0.2, -0.2):      # the first is the offset MEASURED on the refcv8 join (-0.1955 s)
        r = _judge(_recs(shift=shift))
        reasons = LG.rig_clock_reasons("agent join", r, 0.05, 0.01)
        assert r["n_violations"] == 60 and any("more than 0.05 s" in x for x in reasons), (shift, reasons)
    assert _judge(_recs(shift=0.04))["n_violations"] == 0              # inside the tolerance: no false alarm
    assert _judge(_recs(shift=0.06))["n_violations"] == 60             # outside it: the bar is the committed 0.05 s


def test_b_unverified_clips_zero_windows_and_a_blind_control_are_refused():
    # a train clip the sidecar does not hold is UNVERIFIED (admitted only under the cap)
    w = _windows(_recs() + _recs(cid="c2"), table={1: (G0, DT)})
    r = LG.judge_clock_windows(w, 0.05, n_clips=2, cap=0.01)
    assert r["n_unverified_clips"] == 1 and r["cap_ok"] is False
    assert any("have no measured clock" in x for x in LG.rig_clock_reasons("3-D join", r, 0.05, 0.01))
    # an eval-only clip of the shared train+eval join is not a train window; r < 0 (the pre-trim rows) are not labelled rows
    assert _windows([("c2", 5, 1.0)], ns={1: NS}) == [] and _windows([("c1", -1, 0.0), ("c1", -2, 0.0)]) == []
    empty = LG.judge_clock_windows([], 0.05, n_clips=1, cap=0.01)
    assert any("ZERO labelled windows" in x for x in LG.rig_clock_reasons("agent join", empty, 0.05, 0.01))
    # control: an instrument that reads 0 violations for the historical clock cannot see the defect it exists for
    blind = dict(_judge(_recs()), legacy_control_violations=0)
    assert any("control" in x for x in LG.rig_clock_reasons("agent join", blind, 0.05, 0.01))


def test_b_join_records_are_streamed_from_both_formats_and_a_line_without_a_time_is_an_error(tmp_path):
    import lzma
    lines = ['{"clip_id":"a","frame":2,"frame_idx":0,"t_s":0.1234,"agents":[{"cx":1.0,"cy":2.0}]}',
             '{"clip_id": "b", "frame_idx": 3, "t_s": -0.0327, "agents": []}',
             '{"frame_idx": 4, "agents": [], "t_s": 9.5, "clip_id": "c"}']                    # key order the head regex does not know
    p = tmp_path / "j.jsonl.xz"
    with lzma.open(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    assert list(LG.iter_join_records(p)) == [("a", 0, 0.1234), ("b", 3, -0.0327), ("c", 4, 9.5)]
    assert list(LG.iter_join_records(p, limit=2)) == [("a", 0, 0.1234), ("b", 3, -0.0327)]
    q = tmp_path / "j2.jsonl"
    q.write_text('{"clip_id":"a","agents":[]}\n', encoding="utf-8")
    with pytest.raises(LG.GateError):
        list(LG.iter_join_records(q))


class _Frames:
    def __init__(self, i):
        self.frame_idx = [i]


class _Store:
    def __init__(self, drop_offset=False, missing=()):
        self.drop_offset, self.missing = drop_offset, set(missing)

    def frames_for_windows(self, cid, rows, *, n_stack):
        if cid in self.missing:
            raise FileNotFoundError(cid)
        return _Frames(rows[0] if self.drop_offset else rows[0] + n_stack - 1)


def test_b_the_map_gt_leg_checks_the_raw_frame_key_and_catches_a_dropped_offset():
    probes = [("c%d" % (i % 5), 10 + i, 3) for i in range(40)]
    assert LG.rig_map_key_reasons(_Store(), probes)[0] == []
    bad, det = LG.rig_map_key_reasons(_Store(drop_offset=True), probes)            # the +n_stack-1 dropped: labelled 2 rows LATE
    assert det["n_wrong_raw_frame"] == 40 and any("raw frame other than" in x for x in bad)
    r, d = LG.rig_map_key_reasons(_Store(), probes[:5])                            # too few answered probes: the leg read nothing
    assert any("the map leg read nothing" in x for x in r)
    r2, d2 = LG.rig_map_key_reasons(_Store(missing={"c0"}), probes)                # a clip without a GT file is counted, not failed
    assert r2 == [] and d2["n_no_gt_file"] == 8 and d2["n_ok"] == 32


def test_b_rig_clock_check_end_to_end_on_a_synthetic_corpus_and_the_job_is_keyed_on_the_rig(tmp_path, monkeypatch):
    import lzma
    from types import SimpleNamespace
    from tanitad.data import v2_dataset as v2d
    sid = v2d.stable_episode_id("clipA")
    side = tmp_path / "side.jsonl"
    side.write_text(json.dumps({"sid": sid, "grid_start_s": G0, "dt_s": DT}) + "\n", encoding="utf-8")
    monkeypatch.setattr(v2d, "load_or_build_manifest",
                        lambda *a, **k: {"clip_id": ["clipA"], "episode_uid": [sid], "n_stack": [NS]})

    def join(shift, name):
        p = tmp_path / name
        with lzma.open(p, "wt", encoding="utf-8") as fh:
            for r in range(80):
                fh.write(json.dumps({"clip_id": "clipA", "frame": r + 2, "frame_idx": r, "t_s": round(_true(r) + shift, 4),
                                     "agents": []}) + "\n")
        return p
    ctx = SimpleNamespace(options={})
    args = lambda j: SimpleNamespace(clip_clock_sidecar=str(side), v2_cache="x", agent_join=str(j), join3d=str(j), map_gt_root=None)
    ok, det = LG.rig_clock_check(ctx, None, args(join(0.0, "ok.jsonl.xz")), 0.05, 0.01)
    assert ok == [] and det["agent join"]["n_violations"] == 0 and det["3-D join"]["n_windows"] == 80
    assert det["map_gt"] == {"skipped": "no --map-gt-root in argv"}
    # RED arm: the shifted-clock join (the offset MEASURED on the refcv8 join) FAILS both sources
    bad, det2 = LG.rig_clock_check(ctx, None, args(join(-0.1955, "bad.jsonl.xz")), 0.05, 0.01)
    assert len(bad) == 2 and all("more than 0.05 s" in x for x in bad), bad
    assert det2["agent join"]["signed_err_s_quantiles"][0.5] == pytest.approx(-0.1955, abs=1e-4)
    # no sidecar: the rig's labels cannot be timed
    nos = LG.rig_clock_check(ctx, None, SimpleNamespace(clip_clock_sidecar=None, v2_cache="x"), 0.05, 0.01)[0]
    assert len(nos) == 1 and "both needed" in nos[0]
    # the job is keyed on the rig (inert when off): source-level pins of the branch
    src = inspect.getsource(LG.job_clock)
    assert 'rig_on = bool(getattr(args, "r9_perc_rig", False))' in src and "rig_clock_check(ctx, T, args, tol, cap)" in src
    assert 'for split in (() if rig_on else ("train", "eval")):' in src and "elif not ref:" in src
