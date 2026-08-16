"""The v6 STAGE CHAIN, executed — not read.

`scripts/v6_chain.py` wires S-W -> S-T -> S-S -> S-J. Six ladder-edge defects
were found in the two days before it was written and **every one was found by
running a transition, none by reading code**, so this file's centrepiece is an
END-TO-END EXECUTION of the whole ladder on a tiny CPU stack
(:func:`test_the_WHOLE_LADDER_executes_end_to_end_on_cpu`) with real
`--init-from` loads and real gate files, plus one test per refusal.

⛔ WHAT THE EXECUTED LADDER ALREADY CAUGHT, on its FIRST run (MEASURED
2026-08-16): S-W emitted no `--n-candidates`, so it took the trainer's default
8 while the rest of the ladder ran the tiny fan of 3, and `--init-from` died
with ``size mismatch for cand_queries.weight: [8, 256] vs [3, 256]``. ⚠️ That
failure does NOT go through `load_stage_init`'s adjudication —
``load_state_dict(strict=False)`` tolerates missing/unexpected KEYS but still
RAISES on a shape mismatch, so `STAGE_MAY_INTRODUCE` never sees it. Pinned by
:func:`test_the_fan_size_is_a_ladder_wide_constant`.

⛔ AND WHAT THE CHAIN IS *NOT* ALLOWED TO SCHEDULE. E-WC2 fired **REFUSED** on
2026-08-16 against a threshold `V6F_PLANNER_DESIGN.md` §5.2 pre-registered with
both outcomes committed in advance (σ/ADE 9.9915 [7.4492, 13.5119] vs a refusal
line of 3.0 — the interval's LOWER bound is 2.48x the threshold). So the default
S-T runs `--selector none`, and a selector arm is refused at LAUNCH, not merely
left out of the default plan.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v6_chain as C                                              # noqa: E402
from train_v6_staged import (STAGE_PRECONDITION,                  # noqa: E402
                             assert_stage_precondition)


# ============================================================================
# helpers
# ============================================================================
def cfg(root, **kw) -> C.ChainConfig:
    c = C.ChainConfig(root=str(root).replace("\\", "/"), dry=True, tiny=True,
                      dry_steps=1)
    for k, v in kw.items():
        setattr(c, k, v)
    if kw.get("st_arms"):
        c.n_candidates = 3
    return c


def write_gate(step, *, stage, verdict, dry=True, **extra):
    Path(step.out).mkdir(parents=True, exist_ok=True)
    g = {"stage": stage, "verdict": verdict,
         "pass": {"PASS": True, "FAIL": False, "INCONCLUSIVE": None}[verdict],
         "missing_required": [], "inconclusive_required": [],
         "failed_required": ["P3"] if verdict == "FAIL" else []}
    if dry:
        g["_dry_run"] = True
    g.update(extra)
    Path(step.gate).write_text(json.dumps(g))
    return g


def write_admission(c, sigma):
    p = Path(C.admission_path(c))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"sigma_2s_m": sigma}))
    return p


def argv_of(step, c, plan):
    return C.trainer_argv(step, c, plan)


# ============================================================================
# 1. THE PLAN — SEL-1 is REFUSED, so the default ladder has NO selector
# ============================================================================
def test_the_default_ladder_runs_S_T_WITHOUT_a_selector(tmp_path):
    """⛔ E-WC2 fired REFUSED. Scheduling a `goal` arm by default would be
    running an experiment the programme has already declined."""
    plan = C.build_plan(cfg(tmp_path))
    assert [s.key for s in plan] == ["S-W", "S-T", "S-S", "S-J"]
    st = C.step_by_key(plan, "S-T")
    assert st.selector == "none" and st.w_select == 0.0
    assert not any(s.arm for s in plan)
    assert "--selector" not in argv_of(st, cfg(tmp_path), plan)


def test_SEL1_admission_carries_the_fired_pre_registration_verbatim():
    a = C.SEL1_ADMISSION
    assert a["verdict"] == "REFUSED"
    # the CI's LOWER bound is still 2.48x the pre-registered refusal line
    assert a["sigma_over_ade_ci"][0] > a["refused_at_sigma_over_ade"]
    assert round(a["sigma_over_ade_ci"][0] / a["refused_at_sigma_over_ade"],
                 2) == 2.48
    # ⭐ the reading is "wrong SURFACE", not "unpredictable goal": a 0-param
    # kinematic rule beats the learned ridge...
    assert a["kinematic_floor"]["sigma_2s_m"] < a["sigma_2s_m"]
    # ...and is STILL not funded. Both halves must be stated.
    assert a["kinematic_floor"]["sigma_over_ade"] > a["funded_at_sigma_over_ade"]
    assert a["tier"] == "T0-DIAGNOSTIC"


def test_NO_scaled_6s_threshold_is_ever_emitted():
    """§5.3's REDERIVE row fired: σ(6 s)/σ(2 s) = 3.7481 > 3, so the ratio form
    does not transfer and a scaled 6 s number would be fabricated."""
    a = C.SEL1_ADMISSION
    assert a["sigma_6s_over_2s"] > 3.0
    assert a["ratio_form_transfers"] is False
    assert a["threshold_6s_m"] is None
    assert C.SW_LATENT_ADMISSION["horizon_s"] == 2.0
    assert "6 s" in C.SW_LATENT_ADMISSION["_read"]


@pytest.mark.parametrize("sigma,verdict", [
    (0.50, "FUNDED"), (0.7999, "FUNDED"), (0.80, "FUNDED"),
    (0.8001, "INCONCLUSIVE"), (1.41, "INCONCLUSIVE"),
    (1.4101, "REFUSED"), (4.7104, "REFUSED"),
])
def test_the_sw_latent_thresholds_are_pre_registered_and_exact(sigma, verdict):
    """PRE-REGISTERED before the dump is taken, so the decision cannot be made
    after seeing the number. 0.80 m = FUNDED, > 1.41 m = REFUSED stands."""
    r = C.adjudicate_sw_admission(sigma)
    assert r["verdict"] == verdict
    assert r["admits_a_selector_arm"] is (verdict == "FUNDED")
    # REF-C's own measured σ(2 s) sits far outside the funded band
    assert C.adjudicate_sw_admission(
        C.SEL1_ADMISSION["sigma_2s_m"])["verdict"] == "REFUSED"


def test_a_selector_arm_is_REFUSED_at_launch_not_merely_unscheduled(tmp_path):
    """⛔ "we left it out of the default plan" is a preference. A fired
    pre-registration must refuse the launch itself."""
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    plan = C.build_plan(c)
    with pytest.raises(SystemExit, match="SEL-1 is REFUSED"):
        C.assert_selector_admissible(C.step_by_key(plan, "S-T:goal"), c)
    # the refusal must teach the mechanism, not just say no
    try:
        C.assert_selector_admissible(C.step_by_key(plan, "S-T:goal"), c)
    except SystemExit as e:
        msg = str(e)
    assert "WRONG SURFACE" in msg and "ANCHOR_GOAL" in msg
    assert "T0-DIAGNOSTIC" in msg or "REF-C" in msg


def test_an_INCONCLUSIVE_sw_surface_leaves_the_refusal_standing(tmp_path):
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    write_admission(c, 1.20)                      # inside (0.80, 1.41]
    plan = C.build_plan(c)
    assert C.read_sw_admission(c)["verdict"] == "INCONCLUSIVE"
    with pytest.raises(SystemExit, match="SEL-1 is REFUSED"):
        C.assert_selector_admissible(C.step_by_key(plan, "S-T:goal"), c)


def test_a_FUNDED_sw_surface_reopens_the_arms(tmp_path):
    """The arms stay REACHABLE — implemented, dry-run-verified, and revivable
    on a better latent surface. They are gated, not deleted."""
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    write_admission(c, 0.75)
    plan = C.build_plan(c)
    r = C.assert_selector_admissible(C.step_by_key(plan, "S-T:goal"), c)
    assert r["ok"] and r["admission"]["verdict"] == "FUNDED"


def test_a_missing_probe_is_not_an_admission(tmp_path):
    c = cfg(tmp_path, st_arms=("goal",))
    adm = C.read_sw_admission(c)
    assert adm["present"] is False and adm["verdict"] is None
    assert "NOT an admission" in adm["_read"]


# ============================================================================
# 2. ⛔ PER-STAGE --out. The trap must be unbuildable, not merely caught.
# ============================================================================
def test_every_step_gets_its_own_out_directory(tmp_path):
    for arms in ((), ("goal", "mlp")):
        plan = C.build_plan(cfg(tmp_path, st_arms=arms, st_winner="goal"))
        outs = [s.out for s in plan]
        assert len(set(outs)) == len(outs), outs
        assert C.assert_plan(plan)["ok"]


def test_a_shared_out_directory_is_REFUSED_at_plan_time(tmp_path):
    from dataclasses import replace
    plan = list(C.build_plan(cfg(tmp_path, st_arms=("goal", "mlp"),
                                 st_winner="goal")))
    plan[2] = replace(plan[2], out=plan[1].out)
    with pytest.raises(SystemExit, match="share --out"):
        C.assert_plan(tuple(plan))


def test_the_out_dir_refusal_names_the_measured_tensor_counts(tmp_path):
    """The refusal has to explain WHY the accidental barrier is worthless:
    it holds only because S-W 240 / S-T 80 / S-S 54 / S-J 374 happen to differ."""
    from dataclasses import replace
    plan = list(C.build_plan(cfg(tmp_path, st_arms=("goal", "mlp"),
                                 st_winner="goal")))
    plan[2] = replace(plan[2], out=plan[1].out)
    try:
        C.assert_plan(tuple(plan))
    except SystemExit as e:
        msg = str(e)
    for n in ("240", "80", "54", "374"):
        assert n in msg


def test_launching_into_another_stages_directory_is_REFUSED(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    ss = C.step_by_key(plan, "S-S")
    Path(ss.out).mkdir(parents=True, exist_ok=True)
    import torch
    from train_v6_staged import _save_ckpt

    class _Stub:
        def state_dict(self):
            return {"x": torch.zeros(1)}
    _save_ckpt(Path(ss.ckpt), stack=_Stub(), opt=_Stub(), step=10000,
               cfg_json={"stage": "S-T"})
    with pytest.raises(SystemExit, match="written by stage 'S-T'"):
        C.assert_out_dir_free(ss)


# ============================================================================
# 3. --init-from AND --prev-gate on every stage after S-W
# ============================================================================
def test_every_stage_after_S_W_carries_both_the_weights_and_the_certificate(
        tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    for s in plan:
        av = argv_of(s, c, plan)
        if STAGE_PRECONDITION[s.stage] is None:
            assert "--init-from" not in av and "--prev-gate" not in av
            continue
        assert "--init-from" in av, s.key
        assert "--prev-gate" in av, s.key
        prev = C.step_by_key(plan, s.prev_gate_key)
        assert av[av.index("--prev-gate") + 1] == prev.gate
        assert av[av.index("--init-from") + 1].startswith(prev.out)


def test_the_fan_size_is_a_ladder_wide_constant(tmp_path):
    """⛔ MEASURED by the dry ladder's FIRST execution: S-W emitted no
    `--n-candidates`, took the default 8 against a ladder running 3, and
    `--init-from` died on ``size mismatch for cand_queries.weight``.
    ⚠️ A shape mismatch RAISES inside `load_state_dict(strict=False)`, so the
    `STAGE_MAY_INTRODUCE` adjudication never sees it."""
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    seen = set()
    for s in plan:
        av = argv_of(s, c, plan)
        assert "--n-candidates" in av, s.key
        seen.add(av[av.index("--n-candidates") + 1])
    assert len(seen) == 1, seen


def test_S_S_carries_the_selector_geometry_forward_without_lying_about_w_select(
        tmp_path):
    """⛔ MEASURED 2026-08-16 — every available S-S command was wrong:
    `--selector none` is fatal at `load_stage_init` (4 unexpected cand_score.*
    keys); `--selector goal` was refused by a stage-blind preflight; and
    `--w-select 1.0` only got through by advertising a weight that
    `for_stage('S-S')` zeroes."""
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    plan = C.build_plan(c)
    av = argv_of(C.step_by_key(plan, "S-S"), c, plan)
    assert av[av.index("--selector") + 1] == "goal"     # geometry carried
    assert "--w-select" not in av                       # and no lie about it


def test_the_geometry_carry_forward_is_REFUSED_when_it_breaks(tmp_path):
    from dataclasses import replace
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="mlp")
    plan = list(C.build_plan(c))
    ss = C.step_by_key(tuple(plan), "S-S")
    plan[plan.index(ss)] = replace(ss, selector="goal")
    Path(C.step_by_key(tuple(plan), "S-T:mlp").out).mkdir(parents=True,
                                                          exist_ok=True)
    Path(C.step_by_key(tuple(plan), "S-T:mlp").out + "/config.json").write_text(
        json.dumps({"args": {"selector": "mlp"}}))
    with pytest.raises(SystemExit, match="carry the winning arm forward"):
        C.assert_geometry_carry(plan[plan.index(
            [p for p in plan if p.key == "S-S"][0])], tuple(plan))


def test_S_T_introducing_the_selector_is_NOT_a_geometry_break(tmp_path):
    """S-T is where the scorer is BUILT (`STAGE_MAY_INTRODUCE['S-T']`), so an
    ancestor without one is the design, not a mismatch."""
    c = cfg(tmp_path, st_arms=("goal",), st_winner="goal")
    plan = C.build_plan(c)
    sw = C.step_by_key(plan, "S-W")
    Path(sw.out).mkdir(parents=True, exist_ok=True)
    (Path(sw.out) / "config.json").write_text(
        json.dumps({"args": {"selector": "none"}}))
    r = C.assert_geometry_carry(C.step_by_key(plan, "S-T:goal"), plan)
    assert r["ok"] and r["introduces"] == "cand_score."


# ============================================================================
# 4. ⛔ THE GATE. FAIL has no override; INCONCLUSIVE is not a pass.
# ============================================================================
def test_a_PASS_gate_advances(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-W"), stage="S-W", verdict="PASS")
    r = C.assert_may_launch(C.step_by_key(plan, "S-T"), plan, c, dry=True)
    assert r["precondition"]["prev_verdict"] == "PASS"


def test_an_INCONCLUSIVE_gate_REFUSES(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-W"), stage="S-W", verdict="INCONCLUSIVE",
               inconclusive_required=["P1"])
    with pytest.raises(SystemExit, match="INCONCLUSIVE IS NOT A PASS"):
        C.assert_may_launch(C.step_by_key(plan, "S-T"), plan, c, dry=True)


def test_an_INCONCLUSIVE_override_needs_a_RECORDED_reason(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-W"), stage="S-W", verdict="INCONCLUSIVE")
    with pytest.raises(SystemExit):
        C.assert_may_launch(C.step_by_key(plan, "S-T"), plan, c,
                            allow_inconclusive=True, off_reason="  ", dry=True)
    r = C.assert_may_launch(C.step_by_key(plan, "S-T"), plan, c,
                            allow_inconclusive=True, off_reason="because X",
                            dry=True)
    assert r["precondition"]["off_reason"] == "because X"


def test_a_FAIL_gate_has_NO_override_at_all(tmp_path):
    """X5: a failed stage never propagates upward. Every flag set, still refused."""
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-W"), stage="S-W", verdict="FAIL")
    with pytest.raises(SystemExit, match="no override for a FAIL"):
        C.assert_may_launch(
            C.step_by_key(plan, "S-T"), plan, c, allow_inconclusive=True,
            off_reason="I would very much like to proceed", dry=True,
            unpaired_arm_reason="and this too")


def test_a_missing_gate_file_REFUSES(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    with pytest.raises(SystemExit, match="does not exist"):
        C.assert_may_launch(C.step_by_key(plan, "S-T"), plan, c, dry=True)


def test_a_DRY_RUN_gate_cannot_license_a_REAL_launch(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-W"), stage="S-W", verdict="PASS",
               dry=True)
    with pytest.raises(SystemExit, match="written by a --dry-run"):
        assert_stage_precondition("S-T", C.step_by_key(plan, "S-W").gate,
                                  dry_run=False)
    # ...and the SAME file is fine for a dry ladder
    assert assert_stage_precondition(
        "S-T", C.step_by_key(plan, "S-W").gate, dry_run=True)["ok"]


def test_there_is_exactly_ONE_gate_adjudicator(tmp_path):
    """The chain must IMPORT `assert_stage_precondition`, never re-implement
    it: two copies of a gate rule is how a ladder ends up with two opinions
    about what a pass is."""
    src = (Path(C.__file__)).read_text(encoding="utf-8")
    assert "from train_v6_staged import assert_stage_precondition" in src
    for forbidden in ('"pass") is False', "gate.get('pass')", 'gate["pass"]'):
        assert forbidden not in src, forbidden


# ============================================================================
# 5. THE ARM PAIR (once a surface ever admits one)
# ============================================================================
def test_the_arm_pair_does_not_apply_to_the_default_ladder(tmp_path):
    c = cfg(tmp_path)
    plan = C.build_plan(c)
    r = C.assert_arm_pair(C.step_by_key(plan, "S-S"), plan)
    assert r["applies"] is False and "SEL-1 is REFUSED" in r["reason"]


def test_S_S_is_REFUSED_while_the_capacity_control_has_no_gate(tmp_path):
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-T:goal"), stage="S-T", verdict="PASS")
    with pytest.raises(SystemExit, match="ARM PAIR"):
        C.assert_arm_pair(C.step_by_key(plan, "S-S"), plan)


def test_an_unpaired_arm_is_possible_only_as_a_RECORDED_decision(tmp_path):
    c = cfg(tmp_path, st_arms=("goal", "mlp"), st_winner="goal")
    plan = C.build_plan(c)
    write_gate(C.step_by_key(plan, "S-T:goal"), stage="S-T", verdict="PASS")
    r = C.assert_arm_pair(C.step_by_key(plan, "S-S"), plan,
                          unpaired_arm_reason="PI: no second GPU")
    assert r["ok"] and r["reason"] == "PI: no second GPU"
    assert "NOT attributable" in r["_read"]


def test_S_S_refuses_to_guess_which_arm_it_continues(tmp_path):
    c = cfg(tmp_path, st_arms=("goal", "mlp"))          # no --st-winner
    plan = C.build_plan(c)
    assert C.step_by_key(plan, "S-S").needs_st_winner
    with pytest.raises(SystemExit, match="no --st-winner was declared"):
        C.assert_may_launch(C.step_by_key(plan, "S-S"), plan, c, dry=True)


# ============================================================================
# 6. THE SUPERVISOR SEAM — the manifest is sourced ONCE
# ============================================================================
def test_a_manifest_never_supervises_the_CHAIN(tmp_path):
    """⚠️ `supervise_run.sh` replays the TRAIN_CMD it captured at STARTUP. A
    supervised chain would replay stage 1 after a mid-ladder crash."""
    c = cfg(tmp_path, dry=False)
    plan = C.build_plan(c)
    for s in plan:
        if s.key == "S-W":
            continue
        text = C.manifest_text(s, c, plan)
        cmd = C.train_cmd_of(text)
        assert C.TRAINER in cmd
        assert "v6_chain" not in cmd
        assert "v6_chain" in text          # but the WARNING is in the comments
        for key in ("RUN_ID=", "OUT=", "WORKDIR=", "TRAIN_CMD=", "TRAIN_MATCH="):
            assert key in text, (s.key, key)
        assert s.run_id in text
        assert "PYTHONPATH" in text and "OMP_NUM_THREADS" in text


def test_the_manifest_says_how_to_change_a_supervised_run(tmp_path):
    c = cfg(tmp_path, dry=False)
    plan = C.build_plan(c)
    text = C.manifest_text(C.step_by_key(plan, "S-T"), c, plan)
    assert "SOURCED ONCE" in text
    assert "kill the SUPERVISOR" in text
    assert "verify" in text.lower()


def test_the_running_process_probe_cannot_match_itself(tmp_path):
    """⛔ Not `pgrep -f`, not `ps | grep`: both put the searched token into the
    searching process's own command line. MEASURED three times, most recently
    as a monitor that reported `Traceback CUDA out of memory` for a healthy run
    three minutes in. The emitted token must be DISJOINT from the searched one."""
    plan = C.build_plan(cfg(tmp_path))
    probe = C.verify_probe(C.step_by_key(plan, "S-T"))
    assert "pgrep -f" not in probe
    assert "ps -ef" not in probe and "ps aux" not in probe
    assert "train_v6_staged.py" not in probe        # never appears literally
    assert "/proc/" in probe and "ZZ" in probe


# ============================================================================
# 7. THOR — the constants are MEASURED, and they invert the A40 instinct
# ============================================================================
def test_thor_constants_are_the_measured_ones():
    assert C.THOR_S_PER_STEP == 27.18          # marginal, steps 6300->6400
    assert C.A40_S_PER_STEP == 20.46
    # ⚠️ The banked headline ratio is 1.329, computed on the UNROUNDED rates;
    # 27.18/20.46 rounds to 1.328. Asserting the published 3rd decimal here
    # would pin a number these two constants cannot reproduce — a small
    # instance of exactly the "quote the source, not the derived digit" rule.
    assert round(C.THOR_S_PER_STEP / C.A40_S_PER_STEP, 2) == 1.33
    assert C.THOR_BATCH == 8 and C.A40_BATCH == 16
    assert "max_memory_allocated" in C.THOR_MEMORY_PROBE


def test_the_ladder_defaults_to_thors_batch_not_the_A40_instinct(tmp_path):
    c = cfg(tmp_path, dry=False)
    plan = C.build_plan(c)
    av = argv_of(C.step_by_key(plan, "S-T"), c, plan)
    assert av[av.index("--batch") + 1] == str(C.THOR_BATCH)


def test_the_wallclock_estimate_is_arithmetic_on_the_measured_rate(tmp_path):
    c = cfg(tmp_path, dry=False)
    plan = C.build_plan(c)
    st = C.step_by_key(plan, "S-T")
    w = C.wallclock(st, c)
    assert w["s_per_step"] == C.THOR_S_PER_STEP
    assert w["days"] == round(st.steps * C.THOR_S_PER_STEP / 86400, 2)


# ============================================================================
# 8. ⭐ THE WHOLE LADDER, EXECUTED
# ============================================================================
def test_the_WHOLE_LADDER_executes_end_to_end_on_cpu(tmp_path):
    """S-W -> S-T -> S-S -> S-J as four real subprocesses on a tiny CPU stack,
    each one adjudicated by the chain and each one really loading its
    predecessor. This is the test the last six ladder defects would have
    failed."""
    c = cfg(tmp_path)
    res = C.run_chain(c, echo=lambda *a, **k: None)
    keys = [r["step"] for r in res["steps"]]
    assert keys == ["S-W", "S-T", "S-S", "S-J"]
    assert all(r["returncode"] == 0 for r in res["steps"]), [
        (r["step"], r["stderr_tail"]) for r in res["steps"]
        if r["returncode"]]
    # every gate is a REAL gate, and it is INCONCLUSIVE because no battery ran
    assert all(r["gate_verdict"] == "INCONCLUSIVE" for r in res["steps"])
    assert all(r["gate_is_dry_run"] for r in res["steps"])
    # every stage after S-W really loaded its predecessor
    for r in res["steps"][1:]:
        dr = json.loads((Path(r["out"]) / "dry_run.json").read_text())
        assert dr["init"]["exercised"] is True
        assert dr["init"]["missing_keys"] == []
        assert dr["init"]["unexpected_keys"] == []
        assert dr["precondition"]["exercised"] is True
        assert dr["isolation"]["pass"] is True


def test_the_dry_ladder_advances_ONLY_through_the_recorded_override(tmp_path):
    """No fabricated PASS anywhere: the dry gates are genuinely INCONCLUSIVE
    and the ladder moves through `--allow-inconclusive-gate` with its reason
    stamped into every run's config."""
    c = cfg(tmp_path)
    res = C.run_chain(c, echo=lambda *a, **k: None)
    for r in res["steps"][1:]:
        assert r["precondition"]["override"] == "allow-inconclusive-gate"
        assert r["precondition"]["off_reason"] == C.DRY_OFF_REASON
        assert "CPU DRY LADDER" in r["precondition"]["off_reason"]


def test_the_dry_ancestor_is_never_named_ckpt_pt(tmp_path):
    """A synthetic checkpoint that looks real is worse than none: `dry_ckpt.pt`
    can never be found by `--resume auto`."""
    c = cfg(tmp_path)
    res = C.run_chain(c, stop_after="S-T", echo=lambda *a, **k: None)
    for r in res["steps"]:
        assert Path(r["dry_ckpt"]).name == "dry_ckpt.pt"
        assert not (Path(r["out"]) / "ckpt.pt").exists()
