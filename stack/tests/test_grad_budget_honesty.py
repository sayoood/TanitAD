"""⛔ THE DECLARED TRAINABLE BUDGET MUST BE THE TRAINABLE BUDGET.

WHAT THIS FILE EXISTS TO PREVENT (MEASURED 2026-09-06,
``…/Research/2026-09-06-v7f-budget/``). ``apply_stage_freeze`` set
``requires_grad`` from the GROUP MAP alone, so a subtree that no loss can reach
was counted as trained purely because its NAME fell in a trained group. The
count ships in every run's ``config.json``:

  * **v7-tiny at its own 30 k config**: 42 of 138 optimizer tensors, **5,305,667
    params = 52.2 %** of the declared budget, received no gradient.
  * **v7f at its own pre-registered launch line**: **94,717,187 = 40.3 %** of a
    234.9 M declared budget — of which **86,138,112 was the O5 EMA TEACHER**,
    un-frozen by the group map against ``_EmaCopy``'s own docstring.

⭐ EVERY POSITIVE ASSERTION HERE HAS A MUTATION BESIDE IT — the
``guards-need-mutation-not-inspection`` rule. A test that reads the same on a
declared and an undeclared model has measured nothing, so ``test_MUTATION_*``
REMOVES the declaration and requires the verdict to FLIP.

⛔ AND THE SCOPE TEST IS NOT OPTIONAL. ``trained_horizons = (1,)`` is a fact
about the **v6 ladder**, not about ``OperativePredictor``: ``refa_train.py:213``
and ``finetune_traj.py:248`` both sum their loss over EVERY horizon. Declaring
those heads dead in the PREDICTOR's constructor would silently freeze REF-A's
multi-horizon objective. ``test_scope_*`` pins that the declaration is made by
``V6Stack`` and by nothing lower.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.config import (EncoderConfig, PredictorConfig,  # noqa: E402
                            ReadoutConfig)
from tanitad.models._gradreach import (GRAD_UNREACHABLE_FLAG,  # noqa: E402
                                       declare_grad_unreachable,
                                       grad_unreachable_prefixes,
                                       in_declared_subtree)
from tanitad.models.predictor import OperativePredictor  # noqa: E402
from tanitad.models.v6 import (V6Config, V6Stack,  # noqa: E402
                               apply_stage_freeze, _EmaCopy)


def tiny_cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def _stack(**kw):
    torch.manual_seed(0)
    return V6Stack(tiny_cfg(**kw))


def _trainable_names(stack):
    return {n for n, p in stack.named_parameters() if p.requires_grad}


# ===========================================================================
# 1. the group map no longer overstates - and the OLD number is reproduced
# ===========================================================================
def test_freeze_reports_BOTH_the_new_count_and_the_old_rules_count():
    """A count that merely went DOWN is not auditable. The audit must say what
    the group map alone WOULD have claimed and what was withheld from it."""
    s = _stack()
    rep = apply_stage_freeze(s, "S-W")
    assert rep["n_grad_unreachable"] > 0, (
        "nothing was withheld at S-W on a stack whose predictor carries "
        "horizons (1, 2) and an out_proj -- the declaration is not firing")
    assert (rep["n_trainable_by_group_map_alone"]
            == rep["n_trainable"] + rep["n_grad_unreachable"])
    # and the withheld subtrees each carry a REASON, because the audit ships
    # in config.json and an unexplained dead subtree is the defect.
    assert rep["grad_unreachable"]
    for name, why in rep["grad_unreachable"].items():
        assert len(why) > 20, (name, why)


def test_the_declared_budget_equals_what_a_gradient_can_reach():
    """The end-to-end statement, on a REAL backward: at S-W with the v7-tiny
    weights the ONLY residue must be the two STARVED objectives (O1's
    step_readout_op, O3's masked_cells) -- never a structurally dead tensor."""
    import dataclasses
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import (V6LossWeights, grad_reach_census,
                                 synthetic_train_batch, v6_loss_step)
    s = _stack()
    apply_stage_freeze(s, "S-W")
    trainable = [p for p in s.parameters() if p.requires_grad]
    w = V6LossWeights(o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                      o2_nearfield=0.0, o3_masked=0.0, o5_rollout=1.0,
                      o6_sigreg=0.1)
    s.zero_grad(set_to_none=True)
    b = synthetic_train_batch(s, batch=2, k=10, seed=0)
    b["gt_wp"] = torch.randn(2, 10, 2,
                             generator=torch.Generator().manual_seed(0))
    out = v6_loss_step(s, b, stage="S-W", weights=w, o1_k=10, o5_k=1,
                       o5_form="l1",
                       generator=torch.Generator().manual_seed(0),
                       sigreg_generator=torch.Generator().manual_seed(0))
    out["loss"].backward()
    c = grad_reach_census(s, trainable)
    for n in c["unreached_tensors"]:
        assert n.startswith(("step_readout_op.", "masked_cells.")), (
            f"{n} is unreached and is NOT a starved-objective module -- a "
            f"structurally dead tensor is back in the optimizer")
    assert c["trainable_numel"] == sum(int(p.numel()) for p in trainable)
    assert (c["effective_trainable_numel"]
            == c["trainable_numel"] - c["unreached_numel"])

    # MUTATION: switch the starving objectives ON and the residue must vanish
    # ENTIRELY. If anything survives, something else is dead and unnamed.
    s2 = _stack()
    apply_stage_freeze(s2, "S-W")
    tr2 = [p for p in s2.parameters() if p.requires_grad]
    s2.zero_grad(set_to_none=True)
    b2 = synthetic_train_batch(s2, batch=2, k=10, seed=0)
    b2["gt_wp"] = torch.randn(2, 10, 2,
                              generator=torch.Generator().manual_seed(0))
    o2 = v6_loss_step(s2, b2, stage="S-W",
                      weights=dataclasses.replace(
                          w, o1_ctrl=1.0, o1_fact=1.0, o1_scene=1.0,
                          o3_masked=1.0),
                      o1_k=10, o5_k=1, o5_form="l1",
                      generator=torch.Generator().manual_seed(0),
                      sigreg_generator=torch.Generator().manual_seed(0))
    o2["loss"].backward()
    c2 = grad_reach_census(s2, tr2)
    assert c2["n_unreached"] == 0, (
        f"with O1 and O3 ON, {c2['n_unreached']} optimizer tensors STILL get "
        f"no gradient: {sorted(c2['unreached_modules'])}")


# ===========================================================================
# 2. the EMA teacher - the 86 M term, with its mutation
# ===========================================================================
def test_ema_teacher_survives_apply_stage_freeze():
    """`_EmaCopy`'s docstring promises 'excluded from every optimiser'. Before
    2026-09-06 `apply_stage_freeze` broke that promise, because `ema_o5_enc.`
    maps to group `aux` and S-W trains `aux`."""
    s = _stack()
    s.ema_o5_enc = _EmaCopy(s.encoder, 0.996)
    apply_stage_freeze(s, "S-W")
    live = [n for n, p in s.named_parameters()
            if p.requires_grad and n.startswith("ema_o5_enc.")]
    assert not live, f"the EMA teacher is TRAINABLE after the freeze: {live}"
    assert any(n.startswith("ema_o5_enc.") for n, _ in s.named_parameters()), (
        "the teacher has no parameters at all -- the test is vacuous")


def test_MUTATION_removing_the_declaration_REVIVES_the_86M_defect():
    """⭐ THE FAILURE BRANCH, REACHED. Same stack, same stage; only the flag
    moves. If the teacher reads frozen in BOTH arms this file is inspecting,
    not measuring."""
    s = _stack()
    s.ema_o5_enc = _EmaCopy(s.encoder, 0.996)
    # reintroduce the pre-fix state exactly: the flag gone, the constructor's
    # own freeze still in place (that is what `apply_stage_freeze` used to undo)
    delattr(s.ema_o5_enc, GRAD_UNREACHABLE_FLAG)
    rep = apply_stage_freeze(s, "S-W")
    revived = [n for n, p in s.named_parameters()
               if p.requires_grad and n.startswith("ema_o5_enc.")]
    assert revived, (
        "removing the declaration did NOT un-freeze the teacher, so "
        "apply_stage_freeze is not the mechanism this fix targets")
    assert rep["n_trainable"] > 0


def test_ema_teacher_is_declared_by_the_CONSTRUCTOR_not_by_a_caller():
    """`reassert_frozen_external` is the cautionary precedent: a helper the
    caller must remember to call, and NOTHING outside its own test file calls
    it. The declaration must therefore ride the constructor."""
    c = _EmaCopy(nn.Linear(4, 4), 0.9)
    assert getattr(c, GRAD_UNREACHABLE_FLAG, None)
    assert all(not p.requires_grad for p in c.parameters())


# ===========================================================================
# 3. SCOPE - the near-miss this file was written after
# ===========================================================================
def test_scope_v6_freezes_the_untrained_horizon_heads():
    s = _stack()                       # horizons (1, 2); only 1 is trained
    apply_stage_freeze(s, "S-W")
    named = dict(s.named_parameters())
    assert named["predictor_op.heads.1.weight"].requires_grad is True
    assert named["predictor_op.heads.2.weight"].requires_grad is False


def test_scope_a_BARE_predictor_keeps_every_horizon_trainable():
    """⛔ REF-A AND finetune_traj TRAIN EVERY HORIZON (`for k in horizons:
    loss_pred += ...`). If the declaration lived in `OperativePredictor`'s
    constructor it would silently freeze their multi-horizon objective -- a
    true fact quoted outside its scope, the `df`/`step_s` family."""
    p = OperativePredictor(
        PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                        horizons=(1, 2, 4), action_dim=2), state_dim=16)
    for k in (1, 2, 4):
        assert p.heads[str(k)].weight.requires_grad is True, (
            f"head {k} is frozen in a BARE OperativePredictor -- REF-A's "
            f"multi-horizon loss would train nothing")
    # ⛔ AND `out_proj` TOO, THOUGH IT IS DEAD IN EVERY CONSUMER -- BECAUSE A
    # LIVE PI GATE READS `requires_grad`. MEASURED 2026-09-06: declaring it in
    # this constructor makes `train_flagship_v4.py:1636`'s not-frozen gate
    # REFUSE EVERY LAUNCH ("Sayed's hard requirement is that the encoder AND
    # predictor train jointly", trunk_tensors_frozen: 4), and broke three
    # `test_v5_trainer_v2_val.py` gates besides. The gate's INTENT is satisfied
    # -- a tensor no gradient reaches was never training -- but its PREDICATE is
    # this defect in miniature, and correcting a PI hard requirement is not a
    # v7f-budget deliverable. ⇒ `V6Stack` declares it; foreign consumers are
    # byte-identical. This assertion is what stops that being quietly undone.
    assert p.out_proj.weight.requires_grad is True, (
        "out_proj is frozen in a BARE OperativePredictor -- this re-breaks "
        "train_flagship_v4.py's not-frozen gate on every launch")


# ===========================================================================
# 4. NON-DESTRUCTIVE - the property the whole design rests on
# ===========================================================================
def test_state_dict_is_UNCHANGED_so_every_banked_checkpoint_still_loads():
    """`requires_grad` is not serialised. This is the entire reason the fix can
    be shipped without a PI decision: ~30 banked arms carry `heads.2`/`heads.4`
    and every arm carries `out_proj`."""
    s = _stack()
    keys_before = list(s.state_dict().keys())
    apply_stage_freeze(s, "S-W")
    keys_after = list(s.state_dict().keys())
    assert keys_before == keys_after
    for name in ("predictor_op.out_proj.weight", "predictor_op.out_proj.bias",
                 "predictor_op.heads.2.weight"):
        assert name in keys_after, f"{name} left the state_dict -- DESTRUCTIVE"
    s2 = _stack()
    apply_stage_freeze(s2, "S-W")
    got = s2.load_state_dict(s.state_dict(), strict=True)
    assert not got.missing_keys and not got.unexpected_keys


def test_total_param_count_is_unchanged_by_the_freeze():
    s = _stack()
    n = sum(p.numel() for p in s.parameters())
    apply_stage_freeze(s, "S-W")
    assert sum(p.numel() for p in s.parameters()) == n


# ===========================================================================
# 5. the primitive itself
# ===========================================================================
def test_declaration_REFUSES_an_empty_reason():
    with pytest.raises(ValueError, match="REASON"):
        declare_grad_unreachable(nn.Linear(2, 2), "   ")


def test_prefix_matching_is_on_SEGMENTS_not_characters():
    """`heads.4` must not swallow `heads.40`."""
    pres = {"heads.4": "why"}
    assert in_declared_subtree("heads.4.weight", pres) == "heads.4"
    assert in_declared_subtree("heads.4", pres) == "heads.4"
    assert in_declared_subtree("heads.40.weight", pres) is None
    assert in_declared_subtree("heads.41", pres) is None


def test_prefixes_are_discovered_by_walking_named_modules():
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    assert grad_unreachable_prefixes(m) == {}
    declare_grad_unreachable(m[1], "dead by construction, for the test")
    assert set(grad_unreachable_prefixes(m)) == {"1"}


# ===========================================================================
# 6. the EVAL HAZARD banner — untrained is one thing, untrained AND READ is
#    another, and only the second produces a retracted number
# ===========================================================================
def test_the_hazard_banner_fires_for_step_readout_op_and_names_its_consumers(
        capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import GRADREACH_EVAL_HAZARDS, report_grad_reach
    census = {"n_unreached": 6, "n_in_optimizer": 138, "unreached_numel": 2107395,
              "trainable_numel": 10169731,
              "effective_trainable_numel": 10169731 - 2107395,
              "unreached_frac_of_trainable": 2107395 / 10169731,
              "unreached_modules": {"step_readout_op.net.1":
                                    {"tensors": 2, "numel": 2097664,
                                     "group": "predictor_op"}}}
    report_grad_reach(census, refuse=False)
    out = capsys.readouterr().out
    assert "HAZARD step_readout_op" in out
    assert "roll_consistency" in out and "v6_probe_trunk" in out
    # the EFFECTIVE budget is a first-class line, not something to divide out
    assert "EFFECTIVE trainable budget" in out

    # MUTATION: a census with NO hazardous module must not print the banner,
    # or the banner is a constant rather than a test of the census.
    clean = dict(census, unreached_modules={"masked_cells.inp":
                                            {"tensors": 2, "numel": 16640,
                                             "group": "aux"}})
    report_grad_reach(clean, refuse=False)
    out2 = capsys.readouterr().out
    assert "HAZARD step_readout_op" not in out2
    assert "HAZARD masked_cells" in out2       # the other entry still fires
    assert set(GRADREACH_EVAL_HAZARDS) == {"step_readout_op", "masked_cells"}


def test_prefix_match_in_the_hazard_table_is_on_segments(capsys):
    """`step_readout_op` must not fire for a module merely BEGINNING with it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import report_grad_reach
    census = {"n_unreached": 2, "n_in_optimizer": 10, "unreached_numel": 8,
              "trainable_numel": 100, "effective_trainable_numel": 92,
              "unreached_frac_of_trainable": 0.08,
              "unreached_modules": {"step_readout_op_v2.net":
                                    {"tensors": 2, "numel": 8, "group": "x"}}}
    report_grad_reach(census, refuse=False)
    assert "HAZARD" not in capsys.readouterr().out


# ===========================================================================
# 7. --refuse-unreached + --allow-unreached: the pair, with both mutations
# ===========================================================================
def _census_for(mods):
    numel = sum(e["numel"] for e in mods.values())
    return {"n_unreached": 2 * len(mods), "n_in_optimizer": 138,
            "unreached_numel": numel, "trainable_numel": 10169731,
            "effective_trainable_numel": 10169731 - numel,
            "unreached_frac_of_trainable": numel / 10169731,
            "unreached_modules": mods}


def test_refuse_ACCEPTS_only_the_modules_the_run_named(capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import report_grad_reach
    c = _census_for({
        "step_readout_op.net.1": {"tensors": 2, "numel": 2097664,
                                  "group": "predictor_op"},
        "masked_cells.inp": {"tensors": 2, "numel": 16640, "group": "aux"}})
    # v7f's own case: BOTH starved by design and named on the record -> passes
    report_grad_reach(c, refuse=True,
                      allow=("step_readout_op", "masked_cells"))
    assert "ACKNOWLEDGED" in capsys.readouterr().out

    # MUTATION A: name only one -> the other must still refuse. If this passed,
    # `allow` would be a switch that disables the guard rather than scoping it.
    with pytest.raises(SystemExit) as e:
        report_grad_reach(c, refuse=True, allow=("step_readout_op",))
    msg = str(e.value)
    assert "masked_cells.inp" in msg
    # ...and the ACKNOWLEDGED one must NOT be in the refusal list, or the
    # message would send the operator to fix something they already decided.
    assert "step_readout_op.net.1" not in msg

    # MUTATION B: a NEW dead subtree nobody decided about must refuse even
    # though the two known ones are allowed -- this is the case the pair exists
    # for, and it is the one a bare --refuse-unreached could never reach
    # (nobody would keep a flag that refuses on every honest launch).
    c2 = _census_for({
        "step_readout_op.net.1": {"tensors": 2, "numel": 2097664,
                                  "group": "predictor_op"},
        "brand_new_head.fc": {"tensors": 2, "numel": 999, "group": "aux"}})
    with pytest.raises(SystemExit) as e2:
        report_grad_reach(c2, refuse=True,
                          allow=("step_readout_op", "masked_cells"))
    assert "brand_new_head.fc" in str(e2.value)


def test_allow_matches_on_SEGMENTS_so_a_lookalike_still_refuses():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import report_grad_reach
    c = _census_for({"step_readout_op_v2.net": {"tensors": 2, "numel": 8,
                                                "group": "x"}})
    with pytest.raises(SystemExit):
        report_grad_reach(c, refuse=True, allow=("step_readout_op",))


def test_the_flag_pair_exists_in_the_parser_and_defaults_are_inert():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v6_staged import build_parser
    a = build_parser().parse_args(["--stage", "S-W", "--out", "x"])
    assert a.refuse_unreached is False and a.allow_unreached == []
    b = build_parser().parse_args(
        ["--stage", "S-W", "--out", "x", "--refuse-unreached",
         "--allow-unreached", "step_readout_op", "masked_cells"])
    assert b.refuse_unreached is True
    assert b.allow_unreached == ["step_readout_op", "masked_cells"]
