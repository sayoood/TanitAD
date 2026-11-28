"""The gradient-reach census at v7f's OWN pre-registered launch line.

Builds the stack through the TRAINER'S OWN `build_parser` + `build_stack_from_args`
at PREREG_V7F.md Sec.9's flags, applies the launch line's stage freeze, runs a REAL
backward at the launch line's OWN loss weights, and reads the census.

The launch line's not-yet-existing flags (--enc-init-from, --w-ldad, --ldad-*,
--trunk-lr-scale, --trunk-lr-warmup-steps, --exclude-eval-clips) are DROPPED: none
of them changes the parameter geometry (they are a weight init, a loss term with no
parameters, an LR schedule and a corpus filter).

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

from tanitad.models.v6 import apply_stage_freeze                     # noqa: E402
from train_v6_staged import (build_parser, build_stack_from_args,    # noqa: E402
                             grad_reach_census, synthetic_train_batch,
                             v6_loss_step, _weights_from_args,
                             O14_PIX_H, O14_PIX_W)

# PREREG_V7F.md Sec.9, verbatim minus the flags that do not exist yet.
V7F_ARGV = [
    "--stage", "S-W", "--out", str(HERE / "_v7f_census_out"),
    "--newest-frame-only", "--in-channels", "3",
    "--enc-dim", "768", "--enc-depth", "12", "--enc-heads", "12",
    "--patch", "16", "--frame-h", "256", "--frame-w", "640",
    "--projection", "cylindrical", "--frame-hfov", "120",
    "--o5-form", "l1", "--w-o5", "1.0", "--w-o6", "0.1",
    "--sigreg-subspaces", "32", "--sigreg-slices", "512",
    "--spectrum-accum", "4096",
    "--cond-param", "omega_accel_v",
    "--w-o14", "1.0", "--o14-mode", "fut", "--o14-k", "4",
    "--o5-target", "ema", "--ema-decay", "0.996",
    "--o5-k", "60", "--bptt-truncate", "4",
    "--w-o1-ctrl", "0", "--w-o1-fact", "0", "--w-o1-scene", "0",
    "--w-o2", "0", "--w-o3", "0",
    "--w-o7-distill", "0", "--w-o8-pixel", "0", "--w-o9-ema", "0",
    "--w-o10-psg", "0", "--w-o11-cf", "0", "--w-o13-ego", "0",
    "--batch", "8", "--lr", "1e-4", "--clip", "1.0", "--seed", "0",
    "--steps", "1000", "--save-every", "1000", "--log-every", "50",
    "--param-budget", "300000000",
]


def census_at(argv, *, mutate=None, tag=""):
    a = build_parser().parse_args(argv)
    torch.manual_seed(int(a.seed))
    stack = build_stack_from_args(a)
    rep = apply_stage_freeze(stack, a.stage)
    trainable = [p for p in stack.parameters() if p.requires_grad]
    n_train = sum(int(p.numel()) for p in trainable)
    assert n_train == rep["n_trainable"], (n_train, rep["n_trainable"])

    w = _weights_from_args(a).for_stage(a.stage)
    if mutate:
        w = dataclasses.replace(w, **mutate)

    o1_k = 10
    stack.zero_grad(set_to_none=True)
    b = synthetic_train_batch(stack, batch=2, k=max(o1_k, 2), seed=0)
    b["gt_wp"] = torch.randn(2, o1_k, 2,
                             generator=torch.Generator().manual_seed(0))
    if float(getattr(a, "w_o14", 0.0)) > 0:
        # the trainer computes this at the batch-build site (:7086) from
        # `future_frames`, which `synthetic_train_batch` does not carry. The
        # census reads GRADIENT REACH, not loss value: any correctly-SHAPED
        # target puts `o14_head` in the graph identically, so a random one is
        # sufficient and is stated rather than disguised as the real target.
        b["o14_tgt"] = torch.randn(b["frames"].shape[0],
                                   O14_PIX_H * O14_PIX_W,
                                   generator=torch.Generator().manual_seed(0))
    out = v6_loss_step(stack, b, stage=a.stage, weights=w, o1_k=o1_k,
                       o5_k=2, o5_form=a.o5_form,
                       generator=torch.Generator().manual_seed(0),
                       sigreg_generator=torch.Generator().manual_seed(0))
    out["loss"].backward()
    c = grad_reach_census(stack, trainable)
    c["_trainable_numel"] = n_train
    c["_total_numel"] = int(sum(p.numel() for p in stack.parameters()))
    c["_per_group_trainable"] = {g: v["trainable"]
                                 for g, v in rep["per_group"].items()}
    c["_d_op"] = int(stack.cfg.d_op)
    c["_horizons"] = list(stack.cfg.predictor.horizons)
    print("")
    print("=" * 74)
    print("v7f PRODUCTION GEOMETRY %s" % tag)
    print("=" * 74)
    print("  d_op                     : %d" % c["_d_op"])
    print("  predictor horizons       : %s" % (c["_horizons"],))
    print("  TOTAL params             : %d (%.2f M)"
          % (c["_total_numel"], c["_total_numel"] / 1e6))
    print("  DECLARED trainable       : %d (%.2f M)" % (n_train, n_train / 1e6))
    print("  optimizer tensors        : %d  (reached %d / UNREACHED %d)"
          % (c["n_in_optimizer"], c["n_reached"], c["n_unreached"]))
    print("  UNREACHED params         : %d (%.2f M) = %.1f %% of declared"
          % (c["unreached_numel"], c["unreached_numel"] / 1e6,
             100.0 * c["unreached_numel"] / n_train))
    print("  EFFECTIVE trainable      : %d (%.2f M) = %.1f %% of declared"
          % (n_train - c["unreached_numel"],
             (n_train - c["unreached_numel"]) / 1e6,
             100.0 * (n_train - c["unreached_numel"]) / n_train))
    roll = {}
    for m, e in c["unreached_modules"].items():
        roll[m.split(".")[0]] = roll.get(m.split(".")[0], 0) + e["numel"]
    print("  UNREACHED, by top-level module:")
    for m, v in sorted(roll.items(), key=lambda kv: -kv[1]):
        print("    %-22s %10d params (%.2f M)" % (m, v, v / 1e6))
    return c


if __name__ == "__main__":
    off = census_at(V7F_ARGV, tag="- the PREREG launch line AS WRITTEN")
    on = census_at(V7F_ARGV,
                   mutate=dict(o1_ctrl=1.0, o1_fact=1.0, o1_scene=1.0,
                               o3_masked=1.0),
                   tag="- MUTATION CONTROL: O1/O3 switched ON")
    h1 = census_at(V7F_ARGV + ["--horizons", "1"],
                   tag="- --horizons 1 (dead heads never allocated)")
    (HERE / "v7f_budget_census.json").write_text(json.dumps(
        {"prereg_as_written": off, "o1_o3_on": on, "horizons_1": h1},
        indent=1))
    print("")
    print("[bank] wrote v7f_budget_census.json")
