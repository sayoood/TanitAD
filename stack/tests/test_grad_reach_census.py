"""The PER-PARAMETER gradient-reach census, and the fact it rests on.

⛔ WHAT THIS FILE EXISTS TO PREVENT (MEASURED 2026-09-06, seven v7-tiny 30k
checkpoints, the at-init set IDENTICAL on all seven): **25 of 39 one-dimensional
`.weight` tensors sat BIT-EXACTLY at 1.0 after 30,000 AdamW steps.** Twenty were
frozen by `apply_stage_freeze` (S-W does not train `layer_tac` / `layer_str` /
`planner`) and that is by design. The other **five were in the optimizer the
whole time** -- `step_readout_op.net.0` (group `predictor_op`) and the four
`masked_cells` norms (group `aux`), both groups TRAINABLE at S-W. Their only
objectives (O1, O3) were weighted 0.0, and `v6_loss_step` GUARDS those terms,
so the modules never entered the autograd graph.

⚠️ THE EXISTING GUARD CANNOT SEE THIS. `test_v6_ladder_edges._grad_census` is a
GROUP roll-up asserted with `any(census[g]["grad"] for g in trainable_here)` --
one reached parameter anywhere in a group satisfies it -- and it runs at DEFAULT
`V6LossWeights()`, never at a run's own zero-weighted set. This file is
PER-PARAMETER and runs at the weights under test.

⭐ EVERY POSITIVE ASSERTION HERE HAS A MUTATION BESIDE IT. A census that reads
the same on a reached and an unreached model has measured nothing -- which is
exactly how the AST-census lesson (`guards-need-mutation-not-inspection`) was
learned. `test_MUTATION_*` switches the objective ON and requires the verdict to
FLIP.
"""
import dataclasses
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack, apply_stage_freeze  # noqa: E402
from train_v6_staged import (  # noqa: E402
    V6LossWeights, grad_reach_census, report_grad_reach,
    synthetic_train_batch, v6_loss_step)

O1_K = 10


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


def _stack(stage="S-W", seed=0):
    torch.manual_seed(seed)
    s = V6Stack(tiny_cfg())
    apply_stage_freeze(s, stage)
    return s


def _census(stack, weights, stage="S-W", seed=0):
    """A REAL backward at ``weights``, then the census."""
    trainable = [p for p in stack.parameters() if p.requires_grad]
    stack.zero_grad(set_to_none=True)
    b = synthetic_train_batch(stack, batch=2, k=max(O1_K, 2), seed=seed)
    b["gt_wp"] = torch.randn(2, O1_K, 2,
                             generator=torch.Generator().manual_seed(seed))
    out = v6_loss_step(stack, b, stage=stage, weights=weights, o1_k=O1_K,
                       o5_k=1, o5_form="l1",
                       generator=torch.Generator().manual_seed(seed),
                       sigreg_generator=torch.Generator().manual_seed(seed))
    out["loss"].backward()
    return grad_reach_census(stack, trainable), trainable


def _weights_off():
    """The v7-tiny recipe: O5 + O6 only, O1/O2/O3 at exactly 0.0."""
    return V6LossWeights(o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                         o2_nearfield=0.0, o3_masked=0.0,
                         o5_rollout=1.0, o6_sigreg=0.1)


# ===========================================================================
# 1. the census SEES the dead modules the group roll-up cannot
# ===========================================================================
def test_census_names_the_unreached_modules_at_the_runs_own_weights():
    s = _stack()
    census, trainable = _census(s, _weights_off())
    assert census["n_in_optimizer"] == len(trainable)
    assert census["n_unreached"] > 0, (
        "the O5+O6 recipe left every optimizer parameter reached -- if that is "
        "genuinely true now, this whole file's premise has changed and the "
        "v7-tiny finding needs re-deriving, not this assertion relaxing")
    mods = census["unreached_modules"]
    assert any(m.startswith("step_readout_op") for m in mods), (
        f"step_readout_op is reached at O1=0, which contradicts the measured "
        f"v7-tiny checkpoints. unreached={sorted(mods)}")
    # every unreached tensor really is IN the optimizer -- the census must not
    # be silently re-reporting the stage freeze.
    opt_ids = {id(p) for p in trainable}
    named = dict(s.named_parameters())
    for n in census["unreached_tensors"]:
        assert id(named[n]) in opt_ids
        assert named[n].requires_grad is True


def test_MUTATION_the_verdict_FLIPS_when_the_objective_is_switched_on():
    """⭐ THE FAILURE BRANCH, REACHED. Same stack, same stage, same batch --
    only the loss weights move. If `step_readout_op` reads unreached in BOTH
    arms the census is inspecting, not measuring."""
    off = _stack()
    on = _stack()
    c_off, _ = _census(off, _weights_off())
    c_on, _ = _census(on, dataclasses.replace(
        _weights_off(), o1_ctrl=1.0, o1_fact=1.0, o1_scene=1.0, o3_masked=1.0))

    off_dead = {m for m in c_off["unreached_modules"]
                if m.startswith(("step_readout_op", "masked_cells"))}
    on_dead = {m for m in c_on["unreached_modules"]
               if m.startswith(("step_readout_op", "masked_cells"))}
    assert off_dead, "baseline arm has nothing to flip -- the test is vacuous"
    assert not on_dead, (
        f"switching O1/O3 on did NOT revive {sorted(on_dead)}; the census "
        f"cannot distinguish a reached module from an unreached one")
    assert c_on["n_reached"] > c_off["n_reached"]
    assert c_on["n_unreached"] < c_off["n_unreached"]


def test_census_does_not_confuse_a_FROZEN_param_with_an_unreached_one():
    """`layer_tac` is frozen at S-W: it must land in `n_not_in_optimizer`,
    never in `unreached_*`, or the two mechanisms become one number."""
    s = _stack()
    census, _ = _census(s, _weights_off())
    assert census["n_not_in_optimizer"] > 0
    for n in census["unreached_tensors"]:
        assert s.group_of(n) not in ("layer_tac", "layer_str", "planner"), (
            f"{n} is in a stage-FROZEN group but was reported as an unreached "
            f"OPTIMIZER parameter")


# ===========================================================================
# 2. the torch fact the census rests on -- pinned, with its mutation
# ===========================================================================
def test_adamw_SKIPS_none_grad_params_entirely_including_weight_decay():
    """⛔ THE INFERENCE THIS REFUTES: "a parameter that stayed bit-exactly at
    1.0 through 30,000 steps of a DECOUPLED-weight-decay optimiser cannot have
    been in the optimizer." It can. `AdamW._init_group` appends a parameter
    only `if p.grad is not None`, so a None-grad parameter is never stepped and
    never decayed.

    Both arms live in the SAME optimizer, so the only difference is whether
    `.grad` is None or an allocated zero -- which is precisely what
    `zero_grad(set_to_none=True)` decides.
    """
    lr, wd, n = 1e-4, 0.05, 500           # the v7-tiny runs' own lr / wd
    none_p = nn.LayerNorm(8)
    zero_p = nn.LayerNorm(8)
    opt = torch.optim.AdamW(list(none_p.parameters()) + list(zero_p.parameters()),
                            lr=lr, weight_decay=wd)
    w0 = none_p.weight.detach().clone()
    for _ in range(n):
        none_p.weight.grad = None
        zero_p.weight.grad = torch.zeros_like(zero_p.weight)
        opt.step()

    assert torch.equal(none_p.weight.detach(), w0), (
        "a None-grad parameter MOVED inside AdamW -- the census's central "
        "assumption is broken by this torch version")
    assert not torch.equal(zero_p.weight.detach(), w0), (
        "MUTATION FAILED: an allocated-ZERO-grad parameter in the SAME "
        "optimizer did not decay, so this test cannot tell the two apart")
    # and the movement is decoupled weight decay and nothing else. Replicate
    # in the SAME dtype: a float64 closed form drifts from a float32 chain of
    # multiplies by ~1e-5 over hundreds of steps, which is accumulation, not a
    # second effect.
    sim = torch.ones(1, dtype=torch.float32)
    f = torch.tensor(1.0 - lr * wd, dtype=torch.float32)
    for _ in range(n):
        sim = sim * f
    assert float(zero_p.weight.detach()[0]) == pytest.approx(
        float(sim[0]), abs=1e-7)


# ===========================================================================
# 3. the flag: records by default, refuses only when asked
# ===========================================================================
def test_report_records_by_default_and_REFUSES_only_when_asked():
    s = _stack()
    census, _ = _census(s, _weights_off())
    assert census["n_unreached"] > 0
    report_grad_reach(census, refuse=False)          # must not raise
    with pytest.raises(SystemExit) as e:
        report_grad_reach(census, refuse=True)
    assert "refuse-unreached" in str(e.value)
    # MUTATION: a clean census must NOT refuse even with the flag on, or the
    # flag is a constant rather than a test of the model.
    clean = dict(census, n_unreached=0, unreached_modules={},
                 unreached_tensors=[])
    report_grad_reach(clean, refuse=True)
