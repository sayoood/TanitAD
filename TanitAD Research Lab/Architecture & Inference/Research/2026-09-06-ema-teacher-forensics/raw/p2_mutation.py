"""P2 MUTATION PROOF for `assert_declared_freezes_hold`.

A guard that passes on the healthy AND the broken model has measured nothing
(`guards-need-mutation-not-inspection`). So: run it on the two REAL launch lines
(must PASS), then reintroduce each defect and prove it RAISES the right
exception naming the right subtree.

ASCII-only output. CPU only, zero GPU.
"""
import json, sys
from pathlib import Path

import torch

CLONE = Path(r"C:\Users\Admin\tanitad-emaforensics")
sys.path.insert(0, str(CLONE / "stack"))
sys.path.insert(0, str(CLONE / "stack" / "scripts"))

from tanitad.models.v6 import (                                   # noqa: E402
    apply_stage_freeze, assert_declared_freezes_hold, declare_frozen_external,
    stage_trainable_groups, FrozenExternalViolation, GradUnreachableViolation)
from train_v6_staged import build_parser, build_stack_from_args   # noqa: E402

ASSETS = Path(r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901")

# PREREG_V7F.md Sec.9, verbatim minus the flags that do not exist yet
# (identical list to the predecessor's `v7f_budget_census.py`).
V7F_ARGV = [
    "--stage", "S-W", "--out", str(CLONE / "_p2_out"),
    "--newest-frame-only", "--in-channels", "3",
    "--enc-dim", "768", "--enc-depth", "12", "--enc-heads", "12",
    "--patch", "16", "--frame-h", "256", "--frame-w", "640",
    "--projection", "cylindrical", "--frame-hfov", "120",
    "--o5-form", "l1", "--w-o5", "1.0", "--w-o6", "0.1",
    "--sigreg-subspaces", "32", "--sigreg-slices", "512",
    "--spectrum-accum", "4096", "--cond-param", "omega_accel_v",
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


def build_v7f():
    a = build_parser().parse_args(V7F_ARGV)
    torch.manual_seed(int(a.seed))
    return build_stack_from_args(a), a


def build_tiny():
    ck = torch.load(ASSETS / "v7tiny_emao14_30k" / "ckpt.pt",
                    map_location="cpu", weights_only=False)
    rec = ck["config"]["args"]
    rec = rec if isinstance(rec, dict) else vars(rec)
    a = build_parser().parse_args(["--stage", "S-W", "--out", str(CLONE / "_p2_out")])
    for k, v in rec.items():
        if hasattr(a, k):
            setattr(a, k, v)
    torch.manual_seed(int(getattr(a, "seed", 0)))
    return build_stack_from_args(a), a


def freeze_by_group_map_alone(stack, stage):
    """The PRE-2026-09-06 rule: requires_grad from the GROUP MAP ONLY.
    This IS the defect -- it is what trained every banked --o5-target ema arm."""
    groups = set(stage_trainable_groups(stage))
    for name, p in stack.named_parameters():
        p.requires_grad_(stack.group_of(name) in groups)


def run(label, fn):
    try:
        rep = fn()
        return {"case": label, "outcome": "PASSED", "raised": None,
                "detail": rep if isinstance(rep, str) else None}
    except (FrozenExternalViolation, GradUnreachableViolation) as e:
        return {"case": label, "outcome": "RAISED",
                "raised": type(e).__name__, "message": str(e)[:420]}


OUT = []

for geom, builder in (("v7-tiny (emao14_30k own args)", build_tiny),
                      ("v7f (PREREG_V7F Sec.9)", build_v7f)):

    # ---- HEALTHY: the real launch line, current code. MUST PASS. ----
    stack, a = builder()
    apply_stage_freeze(stack, a.stage)
    r = run("%s | HEALTHY (current apply_stage_freeze)" % geom,
            lambda: assert_declared_freezes_hold(stack, a.stage) and "ok")
    n_train = sum(int(p.numel()) for p in stack.parameters() if p.requires_grad)
    r["n_trainable"] = n_train
    OUT.append(r)

    # ---- MUTATION A': reintroduce THE defect (group map alone). MUST RAISE. ----
    stack2, a2 = builder()
    freeze_by_group_map_alone(stack2, a2.stage)
    n_bad = sum(int(p.numel()) for p in stack2.parameters() if p.requires_grad)
    r = run("%s | MUTATION A' = pre-2026-09-06 group-map-alone freeze" % geom,
            lambda: assert_declared_freezes_hold(stack2, a2.stage) and "ok")
    r["n_trainable_under_mutation"] = n_bad
    r["n_trainable_healthy"] = n_train
    r["overstatement_params"] = n_bad - n_train
    OUT.append(r)

    # ---- MUTATION A: a FOREIGN backbone under a trained group. MUST RAISE. ----
    stack3, a3 = builder()
    declare_frozen_external(stack3.encoder,
                            "test mutation: pretend this is a DINOv3 backbone")
    apply_stage_freeze(stack3, a3.stage)   # does NOT honour the external flag
    OUT.append(run("%s | MUTATION A = frozen-external encoder un-frozen by S-W"
                   % geom,
                   lambda: assert_declared_freezes_hold(stack3, a3.stage) and "ok"))

    # ---- MUTATION B: nothing trains. MUST RAISE. ----
    stack4, a4 = builder()
    apply_stage_freeze(stack4, a4.stage)
    for p in stack4.parameters():
        p.requires_grad_(False)
    OUT.append(run("%s | MUTATION B = whole model frozen (the other-direction lie)"
                   % geom,
                   lambda: assert_declared_freezes_hold(stack4, a4.stage) and "ok"))

    # ---- expect_n_trainable mismatch: the arm-substitution check. MUST RAISE. ----
    stack5, a5 = builder()
    apply_stage_freeze(stack5, a5.stage)
    OUT.append(run("%s | MUTATION C = expect_n_trainable off by one" % geom,
                   lambda: assert_declared_freezes_hold(
                       stack5, a5.stage, expect_n_trainable=n_train + 1) and "ok"))

print(json.dumps(OUT, indent=1, default=str))
