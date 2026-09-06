"""BEFORE vs AFTER at v7f's OWN pre-registered launch line, in ONE process.

The BEFORE arm is produced by REMOVING the grad-unreachable declarations from
the built stack before `apply_stage_freeze` -- the exact pre-2026-09-06 state,
reproduced rather than remembered. That is the same mutation
`test_MUTATION_removing_the_declaration_REVIVES_the_86M_defect` uses, so the
two numbers in this table are produced by one code path and differ in one
thing.

ASCII-only output. CPU only, zero GPU.
"""
import dataclasses
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.models._gradreach import (GRAD_UNREACHABLE_FLAG,       # noqa: E402
                                       grad_unreachable_prefixes)
from tanitad.models.v6 import apply_stage_freeze                    # noqa: E402
from train_v6_staged import (build_parser, build_stack_from_args,   # noqa: E402
                             grad_reach_census, synthetic_train_batch,
                             v6_loss_step, _weights_from_args,
                             O14_PIX_H, O14_PIX_W)
from v7f_budget_census import V7F_ARGV                              # noqa: E402


def run(argv, *, legacy: bool, mutate=None):
    a = build_parser().parse_args(argv)
    torch.manual_seed(int(a.seed))
    stack = build_stack_from_args(a)
    if legacy:
        for name in list(grad_unreachable_prefixes(stack)):
            mod = stack.get_submodule(name) if name else stack
            delattr(mod, GRAD_UNREACHABLE_FLAG)
        assert not grad_unreachable_prefixes(stack)
    rep = apply_stage_freeze(stack, a.stage)
    trainable = [p for p in stack.parameters() if p.requires_grad]
    n_train = sum(int(p.numel()) for p in trainable)
    w = _weights_from_args(a).for_stage(a.stage)
    if mutate:
        w = dataclasses.replace(w, **mutate)
    stack.zero_grad(set_to_none=True)
    b = synthetic_train_batch(stack, batch=2, k=10, seed=0)
    b["gt_wp"] = torch.randn(2, 10, 2,
                             generator=torch.Generator().manual_seed(0))
    if float(getattr(a, "w_o14", 0.0)) > 0:
        b["o14_tgt"] = torch.randn(2, O14_PIX_H * O14_PIX_W,
                                   generator=torch.Generator().manual_seed(0))
    out = v6_loss_step(stack, b, stage=a.stage, weights=w, o1_k=10, o5_k=2,
                       o5_form=a.o5_form,
                       generator=torch.Generator().manual_seed(0),
                       sigreg_generator=torch.Generator().manual_seed(0))
    out["loss"].backward()
    c = grad_reach_census(stack, trainable)
    c["_freeze"] = {k: rep[k] for k in
                    ("n_trainable", "n_grad_unreachable",
                     "n_trainable_by_group_map_alone", "grad_unreachable")}
    c["_total_numel"] = int(sum(p.numel() for p in stack.parameters()))
    return c


def line(tag, c):
    roll = {}
    for m, e in c["unreached_modules"].items():
        roll[m.split(".")[0]] = roll.get(m.split(".")[0], 0) + e["numel"]
    print("%-26s declared=%-11d unreached=%-10d (%5.1f %%)  EFFECTIVE=%-11d "
          "(%5.1f %%)  tensors %d/%d"
          % (tag, c["trainable_numel"], c["unreached_numel"],
             100.0 * c["unreached_frac_of_trainable"],
             c["effective_trainable_numel"],
             100.0 * (1 - c["unreached_frac_of_trainable"]),
             c["n_unreached"], c["n_in_optimizer"]))
    for m, v in sorted(roll.items(), key=lambda kv: -kv[1]):
        print("      %-20s %10d" % (m, v))


if __name__ == "__main__":
    before = run(V7F_ARGV, legacy=True)
    after = run(V7F_ARGV, legacy=False)
    after_on = run(V7F_ARGV, legacy=False,
                   mutate=dict(o1_ctrl=1.0, o1_fact=1.0, o1_scene=1.0,
                               o3_masked=1.0))
    print("")
    print("=" * 78)
    print("v7f, PREREG_V7F.md Sec.9 launch line, production geometry")
    print("total params %d (unchanged across all three arms: %s)"
          % (before["_total_numel"],
             before["_total_numel"] == after["_total_numel"] ==
             after_on["_total_numel"]))
    print("=" * 78)
    line("BEFORE (group map alone)", before)
    line("AFTER  (declaration on)", after)
    line("AFTER + O1/O3 ON (ctrl)", after_on)
    print("")
    print("CONTROL: the EFFECTIVE budget is IDENTICAL before and after -> %s"
          % (before["effective_trainable_numel"]
             == after["effective_trainable_numel"]))
    print("CONTROL: withheld = declared_before - declared_after = %d"
          % (before["trainable_numel"] - after["trainable_numel"]))
    print("CONTROL: freeze audit's own n_grad_unreachable              = %d"
          % after["_freeze"]["n_grad_unreachable"])
    (HERE / "v7f_budget_before_after.json").write_text(json.dumps(
        {"before_legacy": before, "after": after, "after_o1_o3_on": after_on},
        indent=1))
    print("[bank] wrote v7f_budget_before_after.json")
