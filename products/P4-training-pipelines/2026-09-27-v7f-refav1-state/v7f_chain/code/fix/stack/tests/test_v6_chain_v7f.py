"""The v6 chain's v7F PROFILE (PI R1-R6, 2026-09-27) -- EXECUTED, with LITERAL expectations.

`scripts/v6_chain.py --profile v7f` plans S-W -> S-T ONLY and emits the PI's binding flags:
  R1 max speed is an INPUT and a CAP (S-T)      R2 nav into tactical AND operative (both stages)
  R3 ALL tactical labels train layer_tac (S-T)  R4 tactical conditions the plan (S-T; mode = PI D1)
  R5 option B: ONE combined 6 s plan            R6 strategic OFF (no --tac-goal-cond, no S-S/S-J)
Every expectation below is a LITERAL (a flag, a value, a path) written from the PI's R1-R6 and the
launch gate's rehearsed argv (`tests/test_launch_gate_v7f.py::ST_ARGV`), never an expression over
the chain under test. The gate cross-check reads the argv through `launch_gate` -- an independently
authored rule set -- and the regression arms EDIT a planned step and must go RED.

⭐ The dry ladder (`test_the_v7f_DRY_LADDER_...`) runs the REAL trainer on the tiny geometry with the
REAL data inputs: the canonical v7.2 train blob (md5 0ff902130ce76886b8a925eceed9e3a5, 4,572
records) as --nav-labels AND --s2-labels, and a max-speed sidecar built over that same blob by the
REAL builder (`build_refcv6_speed_max_window.build`), so the R1 md5 pairing is really checked.

FACTS the executed ladder pins (MEASURED 2026-09-27, TrainingFlyWheel v7f_chain stream) -- each is a
DEFECT OF THE MERGE, not of the chain; when one is fixed its assertion flips IN THE SAME CHANGE:
  * X3: after S-T trains `nav.layer_proj.tactical` (zero-init, layer_tac) the tactical uplink probe
    reaches the SHARED nav embedding (`nav.embed`, `nav.arg_proj` -- grouped predictor_op, i.e.
    BELOW layer_tac): `assert_isolation` reads tactical_to_below = 3 at the end of S-T. Frozen in
    S-T (no real update), but the S-T gate's X3_isolation probe is recorded pass=False.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

import v6_chain as C                                              # noqa: E402
import launch_gate as LG                                          # noqa: E402

# the Thor paths the launch gate's rehearsed argv uses (test_launch_gate_v7f.py::ST_ARGV)
ROOT = "/home/nvidia/experiments"
NAV = "/home/nvidia/data/v72/nav.jsonl.gz"
S2 = "/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz"
SIDE = "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"
THOR = dict(root=ROOT, workdir="/home/nvidia/TanitAD/stack",
            train_cache="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
            val_cache="/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
            nav_labels=NAV, s2_labels=S2, speed_max_sidecar=SIDE)
#: the canonical v7.2 TRAIN blob (intrain_eval.V72 pin), as a literal
V72_TRAIN_MD5 = "0ff902130ce76886b8a925eceed9e3a5"
V72_TRAIN_N = 4572
V72_REL = ("TanitAD Research Lab/Data Engineering/Implementation/incoming/"
           "2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz")


def _v7f(**kw) -> C.ChainConfig:
    return C.v7f_config(**{**THOR, **kw})


def _sw_record(tmp_path: Path, sw_argv) -> Path:
    """What the trainer records as config.json['args'] for exactly this S-W line: its REAL parser on
    the emitted argv (the `_run_config` rule: vars(a) minus `_ew_*`, tuples as lists)."""
    import argparse
    import train_v6_staged as T
    ap = T.build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    a = ap.parse_args(list(sw_argv))
    p = tmp_path / "sw_record" / "config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"args": {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in vars(a).items()
                                      if not k.startswith("_ew_")}}), encoding="utf-8")
    return p


def _argvs(tmp_path: Path, **kw):
    cfg = _v7f(**kw)
    plan = C.build_plan(cfg)
    sw = C.trainer_argv(C.step_by_key(plan, "S-W"), cfg, plan)
    cfg.geometry_from = str(_sw_record(tmp_path, sw))
    st = C.trainer_argv(C.step_by_key(plan, "S-T"), cfg, plan)
    return cfg, plan, sw, st


def _val(argv, flag):
    return LG.flag_values(argv, flag)          # the GATE's reader, not the chain's


# ============================================================================
# 1. the ladder and its literal constants
# ============================================================================
def test_v7f_ladder_is_S_W_then_S_T_and_nothing_above():
    plan = C.build_plan(_v7f())
    assert [s.key for s in plan] == ["S-W", "S-T"]
    assert [s.out for s in plan] == [f"{ROOT}/v7f-SW-30k", f"{ROOT}/v7f-ST-10k"]
    assert C.assert_plan(plan)["ok"]
    st = C.step_by_key(plan, "S-T")
    assert (st.init_from_key, st.prev_gate_key, st.selector, st.w_select, st.tac_goal_cond) == \
        ("S-W", "S-W", "none", 0.0, False)


def test_v7f_literal_constants():
    assert C.PROFILES == ("v6", "v7f")
    assert C.V7F_TAC_OP_COND_DEFAULT == "detached"
    assert C.ChainConfig().tac_op_cond == "detached" and C.ChainConfig().profile == "v6"
    assert C.V7F_TAC_OP_COND_MODES == ("detached", "e2e")
    assert C.V7F_SW_GEOMETRY == ("--in-channels", "3", "--enc-dim", "768", "--enc-depth", "12",
                                 "--enc-heads", "12", "--frame-h", "256", "--frame-w", "640",
                                 "--horizons", "1")
    assert C.V7F_REFUSED_STAGES == ("S-S", "S-J")
    assert set(C.V7F_REQUIREMENTS) == {"R1", "R2", "R3", "R4", "R5", "R6"}
    c = C.v7f_config()
    assert (c.profile, c.n_candidates, c.tac_goal_cond, c.sw_dir) == ("v7f", 1, False, "v7f-SW-30k")


def test_the_CLI_applies_the_v7f_defaults(capsys):
    """`--profile v7f` alone: fan 1 (even in the dry ladder, whose v6 default is 3), the port off,
    the v7f S-W directory -- never the live v6F run's."""
    a = C.build_parser().parse_args(["plan", "--profile", "v7f", "--dry", "--nav-labels", "n",
                                     "--s2-labels", "s", "--speed-max-sidecar-v6", "v"])
    c = C._cfg_from_args(a)
    assert (c.profile, c.n_candidates, c.tac_goal_cond, c.sw_dir, c.tac_op_cond) == \
        ("v7f", 1, False, "v7f-SW-30k", "detached")
    assert (c.nav_labels, c.s2_labels, c.speed_max_sidecar) == ("n", "s", "v")
    a = C.build_parser().parse_args(["plan", "--profile", "v7f", "--tac-op-cond", "e2e",
                                     "--sw-steps", "20000", "--nav-labels", "n",
                                     "--s2-labels", "s", "--speed-max-sidecar-v6", "v"])
    c = C._cfg_from_args(a)
    assert (c.tac_op_cond, c.sw_dir) == ("e2e", "v7f-SW-20k")


# ============================================================================
# 2. the emitted lines, flag by flag (LITERAL)
# ============================================================================
SW_PRESENT = {
    "--stage": ["S-W"], "--out": [f"{ROOT}/v7f-SW-30k"], "--steps": ["30000"], "--batch": ["8"],
    "--lr": ["0.0001"], "--n-candidates": ["1"], "--nav-cond": [], "--nav-labels": [NAV],
    "--strategic-off": [], "--goal-multilabel": [], "--newest-frame-only": [],
    "--in-channels": ["3"], "--enc-dim": ["768"], "--enc-depth": ["12"], "--enc-heads": ["12"],
    "--frame-h": ["256"], "--frame-w": ["640"], "--horizons": ["1"], "--require-parity": [],
    "--save-every": ["250"]}
SW_ABSENT = ("--tac-goal-cond", "--max-speed-input-v6", "--plan-vmax-cap", "--speed-max-sidecar-v6",
             "--w-tac-label-all", "--s2-labels", "--tac-op-cond", "--proposals", "--selector",
             "--init-from", "--prev-gate", "--dump-seam-plan", "--plan-wta-eps", "--w-select",
             "--max-horizon", "--selector-tau-m", "--selector-mlp-hidden", "--lambda-plan",
             "--w-s2-goal", "--w-s1-multi", "--goal-factored", "--i-know-this-arm-predates-nav",
             "--i-know-this-is-the-control-arm", "--allow-inconclusive-gate")
ST_PRESENT = {
    "--stage": ["S-T"], "--out": [f"{ROOT}/v7f-ST-10k"], "--steps": ["10000"], "--batch": ["8"],
    "--lr": ["0.0001"], "--n-candidates": ["1"], "--proposals": ["query"], "--selector": ["none"],
    "--nav-cond": [], "--nav-labels": [NAV], "--strategic-off": [], "--max-speed-input-v6": [],
    "--plan-vmax-cap": [], "--speed-max-sidecar-v6": [SIDE], "--w-tac-label-all": ["1.0"],
    "--goal-multilabel": [], "--s2-labels": [S2], "--tac-op-cond": ["detached"], "--w-t1": ["1.0"],
    "--init-from": [f"{ROOT}/v7f-SW-30k/ckpt.pt"],
    "--prev-gate": [f"{ROOT}/v7f-SW-30k/stage_gate.json"], "--max-horizon": ["60"],
    # CARRIED from S-W's own record (E1), never typed for S-T
    "--newest-frame-only": [], "--in-channels": ["3"], "--enc-dim": ["768"], "--enc-depth": ["12"],
    "--enc-heads": ["12"], "--frame-h": ["256"], "--frame-w": ["640"], "--horizons": ["1"],
    # option B: the epsilon-WTA flag is NOT emitted; S-W's recorded default rides the carry
    "--plan-wta-eps": ["0.0"],
    "--dump-seam-plan": [f"{ROOT}/v7f-ST-10k/seam"], "--require-parity": [], "--save-every": ["250"]}
ST_ABSENT = ("--tac-goal-cond", "--w-select", "--goal-factored", "--w-s2-goal", "--w-s1-multi",
             "--i-know-this-arm-predates-nav", "--i-know-this-is-the-control-arm",
             "--allow-any-labels", "--lambda-plan", "--allow-inconclusive-gate")


def test_the_S_W_line_flag_by_flag(tmp_path):
    _c, _p, sw, _st = _argvs(tmp_path)
    for f, v in SW_PRESENT.items():
        assert _val(sw, f) == v, (f, _val(sw, f))
    for f in SW_ABSENT:
        assert not LG.has_flag(sw, f), f


def test_the_S_T_line_flag_by_flag(tmp_path):
    _c, _p, _sw, st = _argvs(tmp_path)
    for f, v in ST_PRESENT.items():
        assert _val(st, f) == v, (f, _val(st, f))
    for f in ST_ABSENT:
        assert not LG.has_flag(st, f), f


def test_the_launch_line_carries_both_pythonpath_roots_and_the_seam_dir(tmp_path):
    cfg, plan, _sw, _st = _argvs(tmp_path)
    line = C.launch_line(C.step_by_key(plan, "S-T"), cfg, plan)
    assert "PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval" in line
    assert line.startswith(f"mkdir -p {ROOT}/v7f-ST-10k/seam && ")
    assert "< /dev/null &" in line


def test_tac_op_cond_is_the_declared_D1_value(tmp_path):
    _c, _p, _sw, st = _argvs(tmp_path, tac_op_cond="e2e")
    assert _val(st, "--tac-op-cond") == ["e2e"]
    _r, pending = LG.profile_argv_rules(LG.PROFILES["v7f"], st)
    assert len(pending) == 1 and pending[0].startswith("--tac-op-cond e2e: PI decision D1")


# ============================================================================
# 3. the GATE's rules on the emitted lines (an independently authored rule set)
# ============================================================================
def test_the_launch_gate_v7f_rules_PASS_both_lines_and_D1_stays_open(tmp_path):
    _c, plan, sw, st = _argvs(tmp_path)
    prof = LG.PROFILES["v7f"]
    assert LG.profile_argv_rules(prof, sw) == ([], [])
    refusals, pending = LG.profile_argv_rules(prof, st)
    assert refusals == []
    assert len(pending) == 1 and pending[0].startswith("--tac-op-cond detached: PI decision D1")
    for k, av in (("S-W", sw), ("S-T", st)):
        rep = C.v7f_argv_rules(C.step_by_key(plan, k), av)
        assert rep["ok"] and rep["chain_refusals"] == [] and rep["gate_refusals"] == []


def test_the_emitted_R_flags_equal_the_gates_rehearsed_argv(tmp_path):
    """The R1-R6 flag VALUES of the gate's ST_ARGV (written out here as literals, paths aside)."""
    _c, _p, _sw, st = _argvs(tmp_path)
    gate_st = {"--n-candidates": ["1"], "--proposals": ["query"], "--selector": ["none"],
               "--strategic-off": [], "--max-speed-input-v6": [], "--plan-vmax-cap": [],
               "--w-tac-label-all": ["1.0"], "--goal-multilabel": [], "--tac-op-cond": ["detached"],
               "--nav-cond": [], "--newest-frame-only": [], "--horizons": ["1"],
               "--in-channels": ["3"], "--enc-dim": ["768"], "--enc-depth": ["12"],
               "--enc-heads": ["12"], "--frame-h": ["256"], "--frame-w": ["640"],
               "--require-parity": [], "--save-every": ["250"], "--batch": ["8"],
               "--lr": ["0.0001"], "--steps": ["10000"], "--stage": ["S-T"],
               "--nav-labels": [NAV], "--s2-labels": [S2], "--speed-max-sidecar-v6": [SIDE]}
    for f, v in gate_st.items():
        assert _val(st, f) == v, f


# ============================================================================
# 4. refusals -- every one names its requirement
# ============================================================================
@pytest.mark.parametrize("field,flag", [("nav_labels", "--nav-labels"),
                                        ("s2_labels", "--s2-labels"),
                                        ("speed_max_sidecar", "--speed-max-sidecar-v6")])
def test_each_v7f_data_path_is_REQUIRED(field, flag):
    for bad in (None, "", "  "):
        with pytest.raises(SystemExit, match=re.escape(flag) + " is REQUIRED"):
            C.build_plan(_v7f(**{field: bad}))


@pytest.mark.parametrize("field", ["nav_labels", "s2_labels", "speed_max_sidecar"])
def test_a_tilde_in_a_v7f_data_path_is_REFUSED(field):
    with pytest.raises(SystemExit, match="contain '~'"):
        C.build_plan(_v7f(**{field: "~/data/x.jsonl"}))


@pytest.mark.parametrize("kw,needle", [
    ({"tac_goal_cond": True}, "R6"),
    ({"n_candidates": 8}, "R5"),
    ({"n_candidates": 3}, "R5"),
    ({"st_arms": ("goal", "mlp")}, "SEL-1"),
    ({"st_winner": "goal"}, "R6"),
    ({"tac_op_cond": "off"}, "R4"),
    ({"tac_op_cond": "sometimes"}, "R4"),
    ({"sw_dir": "v6F-SW-30k"}, "R2"),
])
def test_the_v7f_config_refusals(kw, needle):
    with pytest.raises(SystemExit, match=needle):
        C.build_plan(_v7f(**kw))


def test_flipping_ONLY_the_profile_is_refused_by_name_never_a_mixed_ladder():
    """ChainConfig(profile='v7f') keeps the v6 defaults (port ON, fan 8, the live v6F dir, no data):
    every one is named in ONE refusal."""
    with pytest.raises(SystemExit) as e:
        C.build_plan(C.ChainConfig(profile="v7f"))
    msg = str(e.value)
    for needle in ("R6", "R5", "R2", "--nav-labels is REQUIRED", "--s2-labels is REQUIRED",
                   "--speed-max-sidecar-v6 is REQUIRED", "v6F-SW-30k"):
        assert needle in msg, needle


def test_an_unknown_profile_is_refused():
    with pytest.raises(SystemExit, match="unknown --profile"):
        C.build_plan(C.ChainConfig(profile="v8"))


V7F_CLI = ["--profile", "v7f", "--root", ROOT, "--nav-labels", NAV, "--s2-labels", S2,
           "--speed-max-sidecar-v6", SIDE]


@pytest.mark.parametrize("argv", [
    ["commands", "--step", "S-S"], ["commands", "--step", "S-J"], ["verify", "--step", "S-S"],
    ["run", "--dry", "--stop-after", "S-J"]])
def test_S_S_and_S_J_are_REFUSED_BY_NAME_under_v7f(argv):
    with pytest.raises(SystemExit, match="REFUSED under --profile v7f .* R6"):
        C.main(argv + V7F_CLI)


@pytest.mark.parametrize("extra", [["--nav-labels", "x"], ["--s2-labels", "x"],
                                   ["--speed-max-sidecar-v6", "x"], ["--tac-op-cond", "e2e"]])
def test_a_v7f_only_flag_on_a_v6_ladder_is_REFUSED_not_ignored(extra):
    with pytest.raises(SystemExit, match="v7f-profile inputs"):
        C.main(["plan"] + extra)


# ============================================================================
# 5. REGRESSION ARMS -- a planned step edited after build_plan must not be emitted
# ============================================================================
def _edited(tmp_path, key, **changes):
    cfg, plan, _sw, _st = _argvs(tmp_path)
    plan = tuple(dataclasses.replace(s, **changes) if s.key == key else s for s in plan)
    return C.step_by_key(plan, key), cfg, plan


def _drop(extra, *flags_with_arity):
    out, skip = [], 0
    arity = dict(flags_with_arity)
    for tok in extra:
        if skip:
            skip -= 1
            continue
        if tok in arity:
            skip = arity[tok]
            continue
        out.append(tok)
    return tuple(out)


def test_ARM_an_S_T_step_emitting_tac_goal_cond_is_CAUGHT(tmp_path):
    st, cfg, plan = _edited(tmp_path, "S-T", tac_goal_cond=True)
    with pytest.raises(SystemExit, match="--tac-goal-cond is passed, but it is REFUSED"):
        C.trainer_argv(st, cfg, plan)
    with pytest.raises(SystemExit, match="--tac-goal-cond"):
        C.launch_line(st, cfg, plan)


@pytest.mark.parametrize("key,drop,needle", [
    ("S-T", (("--plan-vmax-cap", 0),), "--plan-vmax-cap is REQUIRED ON"),
    ("S-T", (("--max-speed-input-v6", 0),), "--max-speed-input-v6 is REQUIRED ON"),
    ("S-T", (("--speed-max-sidecar-v6", 1),), "--speed-max-sidecar-v6 is REQUIRED ON"),
    # ⚠️ S-W's RECORD holds tac_op_cond 'off' (the trainer default, a geometry dest), so an S-T
    # step that forgets R4 gets the CARRIED `--tac-op-cond off` -- and the gate refuses THAT
    ("S-T", (("--tac-op-cond", 1),), "--tac-op-cond off is REFUSED"),
    ("S-T", (("--w-tac-label-all", 1),), "--w-tac-label-all must be passed with a value > 0"),
    ("S-T", (("--s2-labels", 1),), "--s2-labels is REQUIRED on a v7f S-T line"),
    ("S-T", (("--nav-labels", 1),), "--nav-labels is REQUIRED on a v7f S-T line"),
    ("S-W", (("--nav-cond", 0),), "--nav-cond is REQUIRED ON"),
    ("S-W", (("--strategic-off", 0),), "--strategic-off is REQUIRED ON"),
    ("S-W", (("--nav-labels", 1),), "--nav-labels is REQUIRED on a v7f S-W line"),
])
def test_ARM_a_step_missing_an_R_flag_is_CAUGHT(tmp_path, key, drop, needle):
    """(`--nav-cond`, `--strategic-off`, `--goal-multilabel`, `--proposals` are GEOMETRY dests: an
    S-T step that forgets one gets S-W's recorded value back through the E1 carry -- pinned by
    `test_S_T_gets_its_geometry_levers_back_from_S_W_s_record` -- so their arms live on S-W.)"""
    cfg, plan, _sw, _st = _argvs(tmp_path)
    s = C.step_by_key(plan, key)
    plan = tuple(dataclasses.replace(x, extra=_drop(x.extra, *drop)) if x.key == key else x
                 for x in plan)
    with pytest.raises(SystemExit, match=re.escape(needle)):
        C.trainer_argv(C.step_by_key(plan, key), cfg, plan)
    assert s.extra != C.step_by_key(plan, key).extra          # the arm really edited the step


def test_S_T_gets_its_geometry_levers_back_from_S_W_s_record(tmp_path):
    """The E1 carry, on the v7f seam: S-W RECORDED nav_cond / strategic_off / goal_multilabel True and
    proposals 'query', so an S-T step that forgot them still emits them -- the ladder cannot drift
    from what S-W trained."""
    cfg, plan, _sw, _st = _argvs(tmp_path)
    drop = (("--nav-cond", 0), ("--strategic-off", 0), ("--goal-multilabel", 0), ("--proposals", 1))
    plan = tuple(dataclasses.replace(x, extra=_drop(x.extra, *drop)) if x.key == "S-T" else x
                 for x in plan)
    av = C.trainer_argv(C.step_by_key(plan, "S-T"), cfg, plan)
    for f in ("--nav-cond", "--strategic-off", "--goal-multilabel"):
        assert LG.has_flag(av, f), f
    assert _val(av, "--proposals") == ["query"]


@pytest.mark.parametrize("extra,needle", [
    (("--selector", "goal"), "--selector goal is REFUSED"),
    (("--proposals", "diffusion"), "R5: S-T must pass --proposals query"),
    (("--tac-op-cond", "off"), "--tac-op-cond off is REFUSED"),
    (("--goal-factored",), "--goal-factored is passed"),
])
def test_ARM_an_S_T_step_with_a_forbidden_value_is_CAUGHT(tmp_path, extra, needle):
    st, cfg, plan = _edited(tmp_path, "S-T")
    plan = tuple(dataclasses.replace(x, extra=x.extra + extra) if x.key == "S-T" else x
                 for x in plan)
    with pytest.raises(SystemExit, match=re.escape(needle)):
        C.trainer_argv(C.step_by_key(plan, "S-T"), cfg, plan)


def test_ARM_a_fan_widened_after_planning_is_CAUGHT(tmp_path):
    cfg, plan, _sw, _st = _argvs(tmp_path)
    cfg.n_candidates = 8                                   # bypasses assert_v7f_config
    with pytest.raises(SystemExit, match=re.escape("R5: --n-candidates must be 1")):
        C.trainer_argv(C.step_by_key(plan, "S-W"), cfg, plan)


def test_ARM_SOURCE_without_the_emission_guard_the_tac_goal_cond_arm_ESCAPES(tmp_path, monkeypatch):
    """Mutation, not inspection: delete the ONE guard call from the chain's SOURCE and the edited
    step's --tac-goal-cond reaches the emitted line -- so the guard is the barrier, not an accident."""
    src_path = STACK / "scripts" / "v6_chain.py"
    src = src_path.read_text(encoding="utf-8")
    guard = "        assert_v7f_argv(step, argv)\n    return argv"
    assert src.count(guard) == 1
    mod = types.ModuleType("v6_chain_mutant")
    mod.__file__ = str(src_path)
    monkeypatch.setitem(sys.modules, "v6_chain_mutant", mod)
    exec(compile(src.replace(guard, "        pass\n    return argv"), str(src_path), "exec"),
         mod.__dict__)
    cfg = mod.v7f_config(**THOR)
    plan = mod.build_plan(cfg)
    cfg.geometry_from = str(_sw_record(tmp_path, mod.trainer_argv(plan[0], cfg, plan)))
    plan = tuple(dataclasses.replace(s, tac_goal_cond=True) if s.key == "S-T" else s for s in plan)
    av = mod.trainer_argv(mod.step_by_key(plan, "S-T"), cfg, plan)
    assert "--tac-goal-cond" in av                          # the arm escaped the mutant


# ============================================================================
# 6. the default profile is untouched (the full byte-identity proof is the RESULT's 22-invocation
#    harness, original vs patched file; these pin the seams it rests on)
# ============================================================================
V6_CONFIG_KEYS = ["root", "sw_dir", "train_cache", "val_cache", "workdir", "python", "batch",
                  "v2_lru", "lr", "sj_lr", "sw_steps", "st_steps", "ss_steps", "sj_steps",
                  "st_arms", "st_winner", "tac_goal_cond", "w_select", "n_candidates",
                  "selector_tau_m", "selector_mlp_hidden", "plan_wta_eps", "s_per_step", "tiny",
                  "dry", "dry_steps", "extra_common", "geometry_from", "dump_seam_plan",
                  "save_every"]


def test_the_v6_plan_view_is_the_pre_profile_key_list(capsys):
    assert list(C.config_view(C.ChainConfig())) == V6_CONFIG_KEYS
    assert C.main(["plan"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["config"]) == V6_CONFIG_KEYS and "v7f" not in out
    assert [s["key"] for s in out["steps"]] == ["S-W", "S-T", "S-S", "S-J"]


def test_the_v6_S_T_line_carries_nothing_from_v7f(tmp_path):
    c = C.ChainConfig(root=ROOT)
    geo = tmp_path / "config.json"
    geo.write_text(json.dumps({"args": C.parse_argv_geometry([])}))
    c.geometry_from = str(geo)
    plan = C.build_plan(c)
    av = C.trainer_argv(C.step_by_key(plan, "S-T"), c, plan)
    assert _val(av, "--n-candidates") == ["8"] and LG.has_flag(av, "--tac-goal-cond")
    assert _val(av, "--plan-wta-eps") == ["0.05"] and _val(av, "--w-t1") == ["1.0"]
    for f in ("--nav-cond", "--nav-labels", "--strategic-off", "--max-speed-input-v6",
              "--plan-vmax-cap", "--speed-max-sidecar-v6", "--s2-labels", "--w-tac-label-all",
              "--goal-multilabel"):
        assert not LG.has_flag(av, f), f
    assert _val(av, "--tac-op-cond") in (None, ["off"])      # at most the carried default


# ============================================================================
# 7. next / manifests under v7f
# ============================================================================
def test_next_under_v7f_names_S_W_with_the_gate_verdict(tmp_path, capsys):
    rc = C.main(["next", "--profile", "v7f", "--root", str(tmp_path).replace("\\", "/"),
                 "--nav-labels", NAV, "--s2-labels", S2, "--speed-max-sidecar-v6", SIDE])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["next"] == "S-W" and out["may_launch"] is True
    assert out["report"]["v7f_rules"]["ok"] is True
    assert "--nav-cond" in out["command"] and "--tac-goal-cond" not in out["command"]


def test_manifests_under_v7f_include_S_W_because_it_is_a_NEW_run(tmp_path, capsys):
    _c, _p, sw, _st = _argvs(tmp_path)
    dest = tmp_path / "runs.d"
    assert C.main(["manifests", *V7F_CLI, "--dest", str(dest),
                   "--geometry-from", str(tmp_path / "sw_record" / "config.json")]) == 0
    assert sorted(p.name for p in dest.glob("*.env")) == ["v7f-ST-10k.env", "v7f-SW-30k.env"]
    for p in dest.glob("*.env"):
        cmd = C.train_cmd_of(p.read_text(encoding="utf-8"))
        assert C.TRAINER in cmd and "v6_chain" not in cmd and "--nav-cond" in cmd


# ============================================================================
# 8. ⭐ THE v7f DRY LADDER, EXECUTED -- the real trainer, the real data inputs
# ============================================================================
def _canonical_blob() -> Path | None:
    import os
    for p in (os.environ.get("TANITAD_V72_TRAIN_BLOB"), str(STACK.parent / V72_REL)):
        if p and Path(p).is_file() and \
                hashlib.md5(Path(p).read_bytes()).hexdigest() == V72_TRAIN_MD5:
            return Path(p)
    return None


@pytest.fixture(scope="module")
def dry_inputs(tmp_path_factory):
    blob = _canonical_blob()
    if blob is None:
        pytest.skip(f"the canonical v7.2 train blob (md5 {V72_TRAIN_MD5}) is not at "
                    f"<repo>/{V72_REL} and TANITAD_V72_TRAIN_BLOB is unset")
    d = tmp_path_factory.mktemp("v7f_dry_inputs")
    (d / "labels").mkdir()
    local = d / "labels" / blob.name
    shutil.copyfile(blob, local)
    import build_refcv6_speed_max_window as B
    side = d / "speed_max_v72_train.jsonl"
    B.build(str(local), str(side), "train", omit_clip_id=True)
    meta = json.loads((d / "speed_max_v72_train.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["source_md5"] == V72_TRAIN_MD5 and meta["n_clips"] == V72_TRAIN_N
    return local, side


def _dry(root: Path, blob: Path, side: Path) -> C.ChainConfig:
    return C.v7f_config(root=str(root).replace("\\", "/"), dry=True, tiny=True, dry_steps=1,
                        nav_labels=str(blob), s2_labels=str(blob), speed_max_sidecar=str(side))


def test_the_v7f_DRY_LADDER_executes_S_W_then_S_T_on_the_real_inputs(tmp_path, dry_inputs):
    blob, side = dry_inputs
    res = C.run_chain(_dry(tmp_path / "ladder", blob, side), echo=lambda *a, **k: None)
    assert [r["step"] for r in res["steps"]] == ["S-W", "S-T"]
    assert [r["returncode"] for r in res["steps"]] == [0, 0], [
        (r["step"], r["stderr_tail"], r["stdout_tail"]) for r in res["steps"]]
    assert [r["gate_verdict"] for r in res["steps"]] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert all(r["gate_is_dry_run"] for r in res["steps"])
    sw_out, st_out = (Path(r["out"]) for r in res["steps"])
    assert (sw_out.name, st_out.name) == ("v7f-SW-30k", "v7f-ST-10k")
    swc = json.loads((sw_out / "config.json").read_text())["v6_config"]
    stc = json.loads((st_out / "config.json").read_text())["v6_config"]
    assert {k: swc[k] for k in ("nav_cond", "strategic_off", "tac_goal_cond", "n_candidates",
                                "tac_op_cond", "max_speed_input_v6", "plan_vmax_cap")} == \
        {"nav_cond": True, "strategic_off": True, "tac_goal_cond": False, "n_candidates": 1,
         "tac_op_cond": "off", "max_speed_input_v6": False, "plan_vmax_cap": False}
    assert {k: stc[k] for k in ("nav_cond", "strategic_off", "tac_goal_cond", "n_candidates",
                                "proposals", "selector", "tac_op_cond", "max_speed_input_v6",
                                "plan_vmax_cap", "goal_multilabel", "goal_factored")} == \
        {"nav_cond": True, "strategic_off": True, "tac_goal_cond": False, "n_candidates": 1,
         "proposals": "query", "selector": "none", "tac_op_cond": "detached",
         "max_speed_input_v6": True, "plan_vmax_cap": True, "goal_multilabel": True,
         "goal_factored": False}
    dr = json.loads((st_out / "dry_run.json").read_text())
    # the S-W -> S-T seam: loaded for real, nav carried, the two S-T ports INTRODUCED
    assert dr["init"]["exercised"] is True
    assert dr["init"]["missing_keys"] == [] and dr["init"]["unexpected_keys"] == []
    assert sorted(dr["init"]["introduced_keys"]) == ["tac_op_port.bias", "tac_op_port.weight",
                                                     "vmax_tac.bias", "vmax_tac.weight"]
    assert dr["precondition"]["exercised"] is True
    assert dr["precondition"]["override"] == "allow-inconclusive-gate"
    # R3: the REAL v7.2 blob was loaded and its label policy ran
    assert dr["s2_labels"]["exercised"] is True
    assert (dr["s2_labels"]["n_records"], dr["s2_labels"]["schema_version"]) == \
        (V72_TRAIN_N, "s2-geom-v7")
    assert dr["tac_label_all"]["exercised"] is True
    assert dr["tac_label_all"]["gdvb_v6"] == "PASS (refuse_on_mismatch_v6)"
    # R1: the sidecar was READ against md5(--nav-labels), and the cap never let a plan exceed
    assert dr["r1_r4"]["exercised"] is True
    assert dr["r1_r4"]["source_md5"] == V72_TRAIN_MD5 and dr["r1_r4"]["n_valid"] == V72_TRAIN_N
    assert dr["r1_r4"]["vmax_cap_smoke"]["n_plan_over_limit"] == 0
    # X3 -- S-W clean; S-T: the MEASURED merge defect (see the module docstring). Flip when fixed.
    sw_iso = json.loads((sw_out / "dry_run.json").read_text())["isolation"]
    assert sw_iso["pass"] is True
    assert dr["isolation"]["pass"] is False
    assert dr["isolation"]["violations"] == {
        "tactical_to_below": ["nav.embed.weight", "nav.arg_proj.weight", "nav.arg_proj.bias"]}


def test_the_dry_ladder_REFUSES_a_sidecar_built_over_a_different_label_blob(tmp_path, dry_inputs):
    """The discriminating control for R1's pairing: the same sidecar rows, but a meta whose
    source_md5 is the v8.0 blob's (fa89ea55..., the banked refcv6 sidecar's) -- S-W runs (it reads
    no sidecar), S-T must REFUSE at the trainer's reader."""
    blob, side = dry_inputs
    bad = tmp_path / "side_v8" / side.name
    bad.parent.mkdir()
    shutil.copyfile(side, bad)
    meta = json.loads(Path(str(side) + ".meta.json").read_text(encoding="utf-8"))
    meta["source_md5"] = "fa89ea55dfce68403eb30300e57852ab"
    Path(str(bad) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = C.run_chain(_dry(tmp_path / "ladder", blob, bad), echo=lambda *a, **k: None)
    assert [r["step"] for r in res["steps"]] == ["S-W", "S-T"]
    assert res["steps"][0]["returncode"] == 0 and res["steps"][1]["returncode"] != 0
    tail = " ".join(res["steps"][1]["stderr_tail"] + res["steps"][1]["stdout_tail"])
    assert "md5" in tail, tail
