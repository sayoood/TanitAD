"""THE DECLARED-FREEZE PREFLIGHT -- pinned by MUTATION, and pinned as WIRED.

WHY THIS FILE EXISTS, and why it is not just another copy of
`test_v6_frozen_external.py`. That guard was fully built, pinned in both
directions, and quantified its own trap at 86,580,480 parameters -- and it was
called by NOTHING outside its own test file. So the trap it was written for
happened anyway, on NATIVE modules instead of foreign ones: the O5 EMA teacher
was un-frozen by the group map and sat in the optimizer for every
`--o5-target ema` arm from 2026-08 until 2026-09-06.

MEASURED 2026-09-06 (`.../2026-09-06-ema-teacher-forensics/`) on the two banked
EMA arms: the teacher was IN the optimizer's parameter list but AdamW created
state for NONE of its 43 tensors, so it never took a step. The defect was
therefore a BUDGET defect, not an algorithmic one -- and the reason it stayed
invisible is that nothing ever asserted the declaration held.

=> two kinds of test here, and the second is the one that was missing:

  * MUTATION tests -- reintroduce each defect and prove the guard RAISES.
    An assertion that passes on the healthy AND the broken model has measured
    nothing (`guards-need-mutation-not-inspection`).
  * A WIRING test -- prove the trainer actually CALLS it on both launch paths.
    This is the check whose absence let a finished guard sit dead for months.

NO GPU, NO CORPUS, NO CHECKPOINT -- synthetic tiny stack, same wiring.
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    FrozenExternalViolation, GradUnreachableViolation, V6Config, V6Stack,
    _EmaCopy, apply_stage_freeze, assert_declared_freezes_hold,
    declare_frozen_external, grad_unreachable_prefixes,
    stage_trainable_groups)


def tiny_cfg() -> V6Config:
    return V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)


def build(with_teacher: bool = True) -> V6Stack:
    torch.manual_seed(0)
    st = V6Stack(tiny_cfg())
    if with_teacher:
        # exactly what `build_stack_from_args` does under --o5-target ema
        st.ema_o5_enc = _EmaCopy(st.encoder, 0.996)
        st.ema_o5_ro = _EmaCopy(st.readout, 0.996)
    return st


def freeze_by_group_map_alone(st: V6Stack, stage: str) -> None:
    """THE PRE-2026-09-06 RULE, verbatim: requires_grad from the GROUP MAP
    alone, ignoring every declaration. This IS the defect -- it is what trained
    every banked `--o5-target ema` arm."""
    groups = set(stage_trainable_groups(stage))
    for name, p in st.named_parameters():
        p.requires_grad_(st.group_of(name) in groups)


# ---------------------------------------------------------------------------
# the HEALTHY model -- the guard must be SILENT, or it gets deleted
# ---------------------------------------------------------------------------
def test_preflight_PASSES_the_healthy_stack_with_an_ema_teacher():
    st = build()
    apply_stage_freeze(st, "S-W")
    rep = assert_declared_freezes_hold(st, "S-W")
    assert rep["n_leaked_grad_unreachable"] == 0
    # the teacher IS declared, and the declaration IS what holds it frozen
    assert "ema_o5_enc" in rep["grad_unreachable_subtrees"]
    assert not any(p.requires_grad
                   for p in st.ema_o5_enc.parameters())


@pytest.mark.parametrize("stage", ["S-W", "S-T", "S-S", "S-J"])
def test_preflight_PASSES_every_stage(stage):
    st = build()
    apply_stage_freeze(st, stage)
    assert_declared_freezes_hold(st, stage)          # must not raise


# ---------------------------------------------------------------------------
# MUTATION A' -- reintroduce THE defect. This is the direction that FAILED.
# ---------------------------------------------------------------------------
def test_MUTATION_group_map_alone_freeze_RAISES_and_names_the_teacher():
    st = build()
    freeze_by_group_map_alone(st, "S-W")
    # the defect is real in this state: the teacher is trainable again
    assert any(p.requires_grad for p in st.ema_o5_enc.parameters()), (
        "the mutation did not reintroduce the defect -- this test would be "
        "vacuous, which is the failure mode it exists to prevent")
    with pytest.raises(GradUnreachableViolation) as e:
        assert_declared_freezes_hold(st, "S-W")
    msg = str(e.value)
    assert "ema_o5_enc" in msg
    assert "GROUP MAP" in msg


def test_MUTATION_is_not_vacuous_the_same_stack_PASSES_under_the_real_freeze():
    """The control for the test above: same stack, same stage, correct freeze
    -> no raise. Without this, a guard that raised unconditionally would pass
    the mutation test."""
    st = build()
    apply_stage_freeze(st, "S-W")
    assert_declared_freezes_hold(st, "S-W")


# ---------------------------------------------------------------------------
# MUTATION A -- the FOREIGN backbone (E-XENC-1's trap), still latent today
# ---------------------------------------------------------------------------
def test_MUTATION_frozen_external_under_a_trained_group_RAISES():
    st = build()
    declare_frozen_external(st.encoder, "DINOv3 stand-in")
    apply_stage_freeze(st, "S-W")        # does NOT honour the external flag
    with pytest.raises(FrozenExternalViolation):
        assert_declared_freezes_hold(st, "S-W")


# ---------------------------------------------------------------------------
# MUTATION B -- the OTHER-DIRECTION lie: a guard satisfied by freezing all
# ---------------------------------------------------------------------------
def test_MUTATION_whole_model_frozen_RAISES_direction_B():
    st = build()
    apply_stage_freeze(st, "S-W")
    for p in st.parameters():
        p.requires_grad_(False)
    with pytest.raises(FrozenExternalViolation) as e:
        assert_declared_freezes_hold(st, "S-W")
    assert "ZERO trainable native parameters" in str(e.value)


# ---------------------------------------------------------------------------
# the ARM check -- "A GATE ROW CARRIES ITS ARM", enforced before step 1
# ---------------------------------------------------------------------------
def test_expect_n_trainable_mismatch_REFUSES_and_a_match_does_not():
    st = build()
    apply_stage_freeze(st, "S-W")
    n = sum(int(p.numel()) for p in st.parameters() if p.requires_grad)
    assert_declared_freezes_hold(st, "S-W", expect_n_trainable=n)   # exact: ok
    with pytest.raises(FrozenExternalViolation) as e:
        assert_declared_freezes_hold(st, "S-W", expect_n_trainable=n + 1)
    assert "not the arm it claims" in str(e.value)


# ---------------------------------------------------------------------------
# THE WIRING TEST -- the check whose absence left the sibling guard dead
# ---------------------------------------------------------------------------
def test_the_TRAINER_ACTUALLY_CALLS_the_preflight_on_BOTH_launch_paths():
    """`assert_frozen_external` was correct, tested, and called by nothing.
    A guard's correctness and a guard's WIRING are different claims; this
    asserts the second one."""
    import train_v6_staged as T

    assert hasattr(T, "_declared_freeze_preflight")
    for fn_name in ("dry_run", "train"):
        src = inspect.getsource(getattr(T, fn_name))
        assert "_declared_freeze_preflight(" in src, (
            "%s() does not call the declared-freeze preflight -- a guard "
            "nobody calls is a comment" % fn_name)
        # and it must run AFTER the freeze it is checking
        assert src.index("apply_stage_freeze(") < src.index(
            "_declared_freeze_preflight("), (
            "%s() runs the preflight BEFORE apply_stage_freeze, so it would "
            "check requires_grad the freeze has not set yet" % fn_name)


def test_the_preflight_audit_reaches_config_json():
    """The run's own artifact must record that the declarations were CHECKED,
    not merely made."""
    import train_v6_staged as T
    src = inspect.getsource(T._run_config)
    assert "declared_freeze_preflight" in src
