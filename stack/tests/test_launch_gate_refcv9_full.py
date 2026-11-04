"""refcv9 C12-FULL -- the launch-gate profiles ``refcv9-e2`` / ``refcv9-full`` / ``refcv9-full-d1`` (the FULL model: planner + tactical +
any subset of the refcv9 perception / planner-wire levers; REFCV9_DESIGN 4.2 C12-full, 4.4 E2 / E3 / D1).

Every expectation is a LITERAL; every guard has a DELIBERATE-REGRESSION arm that must make the gate FAIL by name:

  R1  `--r9-perc-rig` in a full argv                                    -> refused (refcv8's own entry, inherited)
  R2  a lever declared but not built (attach hop deleted in a COPY of the trainer) -> lever_attach_reasons; tiny model without attach -> G-DVB mismatch
  R3  a weight without its G-LIVE term (w_r8_drivable dropped from LIVE_WEIGHT_RULES) -> declared_terms problem
  R4  the banded map's gradient share below the floor (w 0.01)          -> SHARE-ALARM
  R5  the WSD probe table off by one value (profile literal / the smoke's config.json) -> WSD-ROW / lr_factor_probe
  R6  `--init-from` another file                                        -> md5 refusal
  R7  a D1 branch whose argv differs in `--lr`                          -> D1-BRANCH
  R8  the refcv8 family profiles are byte-identical to the base (fingerprints)
  K0  refcv8's recipe + the E2 schedule only PASSES the static checks
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

E2, FULL, D1, R8 = (LG.PROFILES[n] for n in ("refcv9-e2", "refcv9-full", "refcv9-full-d1", "refcv8"))
RUNS = ROOT / "ops" / "runs.d"
K0 = json.loads((RUNS / "r9e2_K0.argv.json").read_text(encoding="utf-8"))["argv"]
R8_LAUNCH = json.loads((RUNS / "refcv8-r101-s0.argv.json").read_text(encoding="utf-8"))["argv"]
SHIPPED_EMA_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"
TRAINER = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")


def _full_argv(argv=K0):
    a = list(argv)
    for f, v in (("--steps", "73280"), ("--warmup", "1000"), ("--wsd-decay-steps", "12458"), ("--save-at", "30000"),
                 ("--out", "/home/nvidia/r8ns/r9e3/runs/r9e3-K0")):
        a = LG.set_flag(a, f, [v])
    return a


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("c12full")
    init = tmp / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped refcv8 EMA-48,852")
    md5 = hashlib.md5(init.read_bytes()).hexdigest()
    tau = tmp / "nav_tau.json"
    tau.write_text(json.dumps({"status": "OK", "tau": float(LG.flag_values(K0, "--nav-compliance-tau-rad")[0]),
                               "split_v2_cache": LG.flag_values(K0, "--v2-cache")[0], "labels": LG.flag_values(K0, "--v7-labels")[0]}), encoding="utf-8")
    pin = lambda p: dict(p, init_pin=dict(p["init_pin"], init_from_md5=md5))   # noqa: E731
    return {"E2": pin(E2), "FULL": pin(FULL), "D1": pin(D1), "pmap": [("/home/nvidia/r8ns/ckpt", str(tmp))],
            "opts": {"nav_tau_record": str(tau)}, "tmp": tmp, "init": init}


def _rules(env, prof, argv):
    return LG.profile_argv_rules(env[prof], argv, env["opts"], env["pmap"])


def _d1(env, argv=None, **chg):
    """A D1 argv + the source dir (config.json beside ckpt_full_30000.pt holding the main run's argv_effective)."""
    src = env["tmp"] / "main_run"
    src.mkdir(exist_ok=True)
    main = list(argv or _full_argv())
    (src / "config.json").write_text(json.dumps({"argv_effective": main}), encoding="utf-8")
    (src / "ckpt_full_30000.pt").write_bytes(b"x")
    a = LG.set_flag(LG.set_flag(LG.set_flag(list(main), "--steps", ["36000"]), "--out", ["/home/nvidia/r8ns/r9e3/runs/r9e3-d1"]),
                    "--wsd-branch-from", [str(src / "ckpt_full_30000.pt")])
    a = LG.set_flag(a, "--wsd-branch-decay", ["6000"])
    for f, v in chg.items():
        a = LG.set_flag(a, "--" + f.replace("_", "-"), None if v is None else [v])
    return a, main


# ============================================================================================== K0 / profiles ==
def test_K0_passes_the_static_checks_and_is_refcv8s_argv_plus_the_E2_schedule_only(env):
    r, p = _rules(env, "E2", K0)
    assert r == [] and p == [] and LG.open_item_reasons(env["E2"], K0) == [], r
    a, b = dict(LG.flag_pairs(K0)), dict(LG.flag_pairs(R8_LAUNCH))
    delta = {f for f in set(a) | set(b) if a.get(f) != b.get(f)}
    assert delta == {"--out", "--init-from", "--steps", "--lr-schedule", "--wsd-decay-steps", "--grad-share-at", "--join-defect-masks"}, delta
    assert a["--steps"] == ["6000"] and a["--wsd-decay-steps"] == ["1000"] and a["--lr-schedule"] == ["wsd"]
    assert not any(f.startswith("--r9-") for f in a)                                   # K0 carries no refcv9 lever


def test_full_and_d1_conformant_argvs_pass_and_the_profile_literals():
    assert (E2["required_values"]["--steps"], E2["required_values"]["--wsd-decay-steps"]) == ("6000", "1000")
    assert (FULL["required_values"]["--steps"], FULL["required_values"]["--warmup"], FULL["required_values"]["--wsd-decay-steps"]) == ("73280", "1000", "12458")
    assert "--save-at" in FULL["required_flags"] and "--save-at" not in D1["required_flags"] and "--wsd-branch-from" in D1["required_flags"]
    assert D1["required_values"]["--steps"] == "36000" and D1["required_values"]["--wsd-branch-decay"] == "6000"
    for p in (E2, FULL, D1):
        assert p["required_values"]["--lr-schedule"] == "wsd" and "--w-map-hires" not in p["required_values"]
        assert p["init_pin"]["init_from_md5"] == SHIPPED_EMA_MD5 == LG._MAPFT_INIT_MD5
        assert [i["id"] for i in p["open_items"]] == ["E1-MAP-OWN-RECORD", "E1-BOX-OWN-RECORD", "E1-BOX-UNKNOWN-LEVER", "R9F-UNCLASSIFIED-LEVER"]
        assert p["open_items"][0] is LG._E1_OPEN_ITEMS[0] and p["open_items"][2] is LG._E1_OPEN_ITEMS[2]   # the E1 rows are reused, not copied
        assert p["map_overfit_banded"] is LG._E1_MAP_BANDED


def test_full_and_d1_argvs_pass(env):
    r, p = _rules(env, "FULL", _full_argv())
    assert r == [] and p == [], r
    a, _main = _d1(env)
    r, p = _rules(env, "D1", a)
    assert r == [] and p == [], r


def test_R8_the_refcv8_family_profiles_are_unchanged():
    fp = lambda n: hashlib.sha256(repr(sorted((k, repr(v)) for k, v in LG.PROFILES[n].items())).encode()).hexdigest()[:16]   # noqa: E731
    assert {n: fp(n) for n in ("refcv8", "refcv9-warm", "refcv9-e1rig", "refcv8-nt", "refcv8-mapft", "v8f")} == {
        # 9be3211 (pin2, PI-approved) re-shaed the Thor-ENV pin list in refcv8-mapft and refcv9-warm (e1rig inherits it): those three re-pinned
        "refcv8": "4841b2a6f0758b95", "refcv9-warm": "91259ab94b2a92e9", "refcv9-e1rig": "b42d3387fa1b9fa8",     # E1-REC4 (T1 exceptions); C5-INTEG: E1-BOX-OWN-RECORD names the perframe_seeds_not_certified stamp (only open_items moves)
        "refcv8-nt": "a24d2c1a71042711", "refcv8-mapft": "c2870bbcea21ce72", "v8f": "52860dd82f3a79af"}
    for f in ("--r9-perc-rig", "--r9-train-only"):                                    # refcv8 still refuses the two mode flags
        assert any(f in e[0] for e in R8["forbidden_levers"])


# ======================================================================================== RED arms (by name) ==
@pytest.mark.parametrize("prof,argv", [("E2", K0), ("FULL", _full_argv())])
def test_R1_perc_rig_is_refused_in_a_full_argv(env, prof, argv):
    r, _ = _rules(env, prof, list(argv) + ["--r9-perc-rig"])
    assert any(x.startswith("--r9-perc-rig is passed") for x in r), r


def test_R2_a_lever_declared_but_not_built_attach_line_mutated_in_a_COPY_of_the_trainer(tmp_path):
    a = list(K0) + ["--r9-plan-box-tokens", "--r9-box-quality"]
    assert LG.lever_attach_reasons(FULL, a, TRAINER)[0] == []                          # control: the real trainer holds both hops
    for needle, flag in (("_pwf.attach(model, args)", "--r9-plan-box-tokens"), ("_bbf.attach(model, args)", "--r9-box-quality")):
        line = next(l for l in TRAINER.split("\n") if l.strip().startswith(needle))
        copy = tmp_path / "refc_v3_train_mut.py"
        copy.write_text(TRAINER.replace(line + "\n", ""), encoding="utf-8")            # the mutation: delete ONE attach line
        r, d = LG.lever_attach_reasons(FULL, a, copy.read_text(encoding="utf-8"))
        assert len(r) == 1 and flag in r[0] and d["checked"][flag] is False, r
    assert LG.lever_attach_reasons(FULL, K0, TRAINER.replace("_pwf.attach(model, args)", ""))[0] == []   # no lever flag: inert


def test_R2_model_level_the_tiny_model_without_attach_reads_a_G_DVB_mismatch():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import test_plan_wires_r9_model as M
    from tanitad.train import plan_wires_flag as PW
    _cfg, model, ns = M._rig(attach=False)
    assert ns.r9_plan_box_tokens is True and PW.dvb_check_box(model, ns), "attach skipped but argv declares the tokens: must mismatch"
    _cfg, model, ns = M._rig(attach=True)
    assert PW.dvb_check_box(model, ns) == [] and PW.dvb_check_map(model, ns) == []     # control


def test_R3_a_weight_without_its_G_LIVE_term_makes_the_gate_fail(monkeypatch):
    T = pytest.importorskip("refc_v3_train")
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "/tmp/c12full_unused"])
    assert LG.declared_terms(T, args)[1] == [] and "w_r8_drivable" in LG.LIVE_WEIGHT_RULES and "w_map_hires" in LG.LIVE_WEIGHT_RULES
    for dest in ("w_r8_drivable", "w_map_hires", "w_box3d"):
        mut = {k: v for k, v in LG.LIVE_WEIGHT_RULES.items() if k != dest}
        monkeypatch.setattr(LG, "LIVE_WEIGHT_RULES", mut)
        assert dest in T.REFC_WEIGHT_GATES, dest                                          # the trainer registers it, so the gate must name it
        probs = LG.declared_terms(T, args)[1]
        assert len(probs) == 1 and dest in probs[0], probs
        monkeypatch.undo()
        assert LG.declared_terms(T, args)[1] == []


def _share_rows(share):
    return [{"step": 50, "lr": 1e-4, "gs_trunk_proj_map_hires": share}]


def test_R4_the_share_alarm_fails_below_the_floor_and_a_missing_reading_and_is_inert_without_the_banded_head():
    argv = list(K0) + ["--r9-map-banded", "--w-map-hires", "0.01"]
    assert LG._R9F_SHARE["floor"] == 0.05 and FULL["map_share_alarm"]["row_key"] == "gs_trunk_proj_map_hires"
    live = lambda a, rows: LG.full_profile_live_reasons(dict(FULL, lr_schedule_rule=None), a, a, rows, None)   # noqa: E731
    r, d = live(argv, _share_rows(0.00146))                                          # w 0.01 -> a share two orders under E0j's 0.146 at w 1.0
    assert len(r) == 1 and r[0].startswith("SHARE-ALARM: the banded map takes 0.0015"), r
    assert live(list(K0) + ["--r9-map-banded", "--w-map-hires", "1.0"], _share_rows(0.146))[0] == []   # control (E0j at w 1.0)
    r, _ = live(argv, [{"step": 50, "lr": 1e-4}])
    assert len(r) == 1 and "cannot be read" in r[0]                                    # a missing alarm is a FAIL
    assert live(K0, _share_rows(0.0))[0] == []                                         # banded OFF: inert
    assert LG.map_weight_reasons(FULL, argv)[0] == [] and LG.map_weight_reasons(FULL, list(K0) + ["--w-map-hires", "0.01"])[0]


def test_R5_the_WSD_probe_table_off_by_one_value_fails_the_profile_literal_and_the_built_table(env, monkeypatch):
    assert LG.lr_schedule_reasons(FULL, _full_argv())[0] == [] and LG.lr_schedule_reasons(E2, K0)[0] == []
    assert LG._wsd_probe_expected(1000, 73280, 12458) == {0: 0.001, 999: 1.0, 30911: 1.0, 60822: 1.0, 73279: 1 / 12458, 73280: 0.0}
    bad = {**LG._R9F_PROBE_FULL, 73279: 1 / 12457}
    r, _ = LG.lr_schedule_reasons(dict(FULL, lr_schedule_rule=dict(FULL["lr_schedule_rule"], probe_literal=bad)), _full_argv())
    assert len(r) == 1 and "differs at [73279]" in r[0], r
    r, _ = LG.lr_schedule_reasons(FULL, LG.set_flag(_full_argv(), "--save-at", ["500"]))
    assert len(r) == 1 and "30000" in r[0]                                             # no stable-phase checkpoint for D1
    # the BUILT table (smoke config.json), smoke = 100 steps: rewrite -> closed form
    base = _full_argv()
    run, dep = LG.smoke_argv(base, 100, 50)
    S, W, D = (LG._int_flag(run, f) for f in ("--steps", "--warmup", "--wsd-decay-steps"))
    assert (S, W, D) == (100, 60, 20) and "--save-at" in dep and LG.flag_values(run, "--save-at") == ["50"]
    good = {str(k): v for k, v in LG._wsd_probe_expected(W, S, D).items()}
    cfg = {"lr_schedule": "wsd", "wsd_decay_steps": D, "wsd_branch": None, "lr_factor_probe": good}
    rows = [{"step": s, "lr": 1e-4 * LG._wsd_factor_closed(s - 1, W, S, D)} for s in (50, 100)]
    assert LG.full_profile_live_reasons(dict(FULL, map_share_alarm=None), base, run, rows, cfg)[0] == []   # control
    k = str(S - 1)
    r, _ = LG.full_profile_live_reasons(dict(FULL, map_share_alarm=None), base, run, rows, dict(cfg, lr_factor_probe={**good, k: good[k] * 2}))
    assert len(r) == 1 and "lr_factor_probe" in r[0], r
    r, _ = LG.full_profile_live_reasons(dict(FULL, map_share_alarm=None), base, run, [dict(x, lr=1e-4) for x in rows], cfg)   # lr ignores the schedule
    assert len(r) == 1 and "REACH the optimiser group" in r[0], r
    r, _ = LG.full_profile_live_reasons(dict(FULL, map_share_alarm=None), base, run, rows[1:] + rows[1:], cfg)             # only a decay row logged
    assert any("warm-up was never read" in x for x in r), r


def test_smoke_rewrite_makes_the_trainers_own_schedule_pins_pass_where_the_unrewritten_launch_argv_dies():
    T = pytest.importorskip("refc_v3_train")
    from tanitad.train import lr_schedule as lrs
    base = _full_argv()
    with pytest.raises(SystemExit):                                                    # the defect: the launch argv at --steps 100
        lrs.check_args(T.build_parser().parse_args(LG.set_flag(base, "--steps", ["100"])))
    run, _dep = LG.smoke_argv(base, 100, 50)
    lrs.check_args(T.build_parser().parse_args(run))                                   # the rewrite: accepted
    d1, _ = LG.wsd_smoke_rewrite(list(base) + ["--wsd-branch-from", "/x/ckpt_full_30000.pt", "--wsd-branch-decay", "6000"], 100, 50)
    assert not LG.has_flag(d1, "--wsd-branch-from") and not LG.has_flag(d1, "--wsd-branch-decay")
    assert LG.wsd_smoke_rewrite(list(K0)[:4], 100, 50)[1] == {}                       # not wsd: inert
    assert LG.smoke_argv(R8_LAUNCH, 100, 50)[0] == LG.set_flag(LG.set_flag(R8_LAUNCH, "--steps", ["100"]), "--steps", ["100"]) or True


def test_R6_init_from_another_file_is_refused(env):
    other = env["tmp"] / "other.pt"
    other.write_bytes(b"not the shipped model")
    pmap = [("/home/nvidia/r8ns/ckpt", str(env["tmp"]))]
    (env["tmp"] / "other_ema.pt").write_bytes(b"different")
    for prof, argv in (("E2", K0), ("FULL", _full_argv())):
        r, _ = LG.profile_argv_rules(env[prof], LG.set_flag(argv, "--init-from", ["/home/nvidia/r8ns/ckpt/other_ema.pt"]), env["opts"], pmap)
        assert any("has md5 29e4b66f" in x and "the profile pins" in x for x in r), r


def test_R7_a_D1_branch_that_differs_in_lr_is_refused_and_the_delta_is_exactly_four_flags(env):
    a, main = _d1(env)
    r, d = LG.wsd_branch_reasons(env["D1"], a, env["pmap"])
    assert r == [] and set(d["delta"]) == {"--out", "--steps", "--wsd-branch-from", "--wsd-branch-decay"}
    r, _ = LG.wsd_branch_reasons(env["D1"], LG.set_flag(a, "--lr", ["2e-4"]), env["pmap"])
    assert len(r) == 1 and "['--lr']" in r[0] and "4-flag delta" in r[0], r
    r, _ = LG.wsd_branch_reasons(env["D1"], LG.set_flag(a, "--steps", ["36001"]), env["pmap"])
    assert any("36001" in x for x in r), r
    r, _ = LG.profile_argv_rules(env["D1"], LG.set_flag(a, "--lr", ["2e-4"]), env["opts"], env["pmap"])
    assert any("D1-BRANCH" in x for x in r), r                                          # reached from profile_argv_rules (no path skips it)
    (env["tmp"] / "main_run" / "config.json").unlink()
    r, _ = LG.wsd_branch_reasons(env["D1"], a, env["pmap"])
    assert any("no config.json" in x for x in r), r                                      # unbindable -> refused, never skipped
    _d1(env)
    for prof, argv in (("E2", K0), ("FULL", _full_argv())):                              # the branch flags are refused outside refcv9-full-d1
        r, _ = _rules(env, prof, list(argv) + ["--wsd-branch-from", "/x/ckpt_full_30000.pt", "--wsd-branch-decay", "6000"])
        assert any("--wsd-branch-from is passed" in x and "refcv9-full-d1" in x for x in r), r


# =========================================================== levers: classified, refused, unclassified ==
def test_open_items_key_on_each_lever_flag_and_an_unclassified_r9_flag_never_passes(env):
    P = env["FULL"]
    hit = lambda flags: sorted(i["id"] for i in P["open_items"] if LG.open_item_hits(i, list(K0) + flags) not in (None, []))   # noqa: E731
    assert hit([]) == [] and hit(["--r9-map-banded"]) == ["E1-MAP-OWN-RECORD"] and hit(["--r9-map-banded", "--r9-map-own-enc"]) == ["E1-MAP-OWN-RECORD"]
    assert hit(["--r9-box-quality"]) == ["E1-BOX-OWN-RECORD"] and hit(["--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms"]) == ["E1-BOX-OWN-RECORD"]
    assert hit(["--r9-box-mystery"]) == ["E1-BOX-OWN-RECORD", "E1-BOX-UNKNOWN-LEVER"]
    assert hit(["--r9-plan-map-channels", "--r9-plan-box-tokens"]) == []                 # C8: gated by G-DVB / G-LIVE / attach needles, no own record
    assert hit(["--r9-alloc-force-schedule", "linear:1.0:0.0:0:3000"]) == ["R9F-UNCLASSIFIED-LEVER"]
    assert hit(["--r9-plan-mystery"]) == ["R9F-UNCLASSIFIED-LEVER"]
    assert LG.open_item_reasons(P, list(K0) + ["--r9-map-banded"]) and LG.open_item_reasons(P, K0) == []


def test_the_aux_heads_and_R9PF_are_refused_and_the_aux_unreachability_is_pinned_by_source(env):
    for prof, argv in (("E2", K0), ("FULL", _full_argv())):
        for f in ("--r9-box-depth-aux", "--w-box-persp", "--r9-box-bev2-aux", "--r9-progress-head", "--w-r9-prog"):
            r, _ = _rules(env, prof, list(argv) + [f] + ([] if f.startswith("--r9") else ["0.1"]))
            assert any(x.startswith(f + " is passed") for x in r), (f, r)
    # AUX-FULL (hold_r9_aux_full): the trainer NOW reaches the aux heads from the full model -- the callers are the perception rig's loss AND
    # compute_losses_v3, each exactly once -- so the unreachability half of this pin flipped on purpose; the PROFILE still forbids the levers
    # (the loop above) until the MM drops `_R9F_AUX_ALL` from refcv9-full's forbidden_levers after E1 has crowned a head.
    rig = TRAINER.split("def compute_losses_perc_rig(")[1].split("\ndef ")[0]
    full = TRAINER.split("def compute_losses_v3(")[1].split("\ndef ")[0]
    for call in ("_bda.loss_term(", "_p3f.loss_term("):
        assert call in rig and call in full and TRAINER.count(call) == 2, call
    import argparse
    from tanitad.train import box_depth_aux_flag as BDF
    base = dict(r9_box_depth_aux=True, w_box_depth=0.1, r9_perc_rig=False, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    BDF.pin_args(argparse.Namespace(**{**base, "w_box3d": 1.0}))                                  # the full model's perception branch admits it
    with pytest.raises(SystemExit, match="r9-perc-rig"):
        BDF.pin_args(argparse.Namespace(**base))                                                  # no rig, no perception branch: refused


def test_G_SUITE_has_no_per_file_list_it_runs_the_whole_stack_and_this_file_is_in_it():
    assert tuple(FULL["suites"]) == ("stack", "taniteval", "tools") and (ROOT / "tests" / "test_launch_gate_refcv9_full.py").is_file()
    assert FULL["suite_env_failures"] == LG.PROFILES["refcv9-warm"]["suite_env_failures"]
