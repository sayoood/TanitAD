"""D0 at NAME level: rebuild the emao14 arm, reproduce the PRE-FIX freeze
(group map alone, exactly what trained the banked checkpoint), map every
optimizer slot index to its parameter NAME, and read which slots the optimizer
actually created state for.

torch AdamW `_init_group` appends a parameter only `if p.grad is not None`, so a
slot WITHOUT state was never stepped -- weight decay included.

ASCII-only output. CPU only, zero GPU.
"""
import argparse, json, sys
from pathlib import Path

import torch

CLONE = Path(r"C:\Users\Admin\tanitad-emaforensics")
sys.path.insert(0, str(CLONE / "stack"))
sys.path.insert(0, str(CLONE / "stack" / "scripts"))

from tanitad.models.v6 import stage_trainable_groups          # noqa: E402
from train_v6_staged import build_parser, build_stack_from_args  # noqa: E402

ASSETS = Path(r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901")
OUT = {}

for arm in ["v7tiny_emao14_30k", "v7tiny_emao14_30k_tauramp"]:
    ck = torch.load(ASSETS / arm / "ckpt.pt", map_location="cpu", weights_only=False)
    rec_args = ck["config"]["args"]
    rec_args = rec_args if isinstance(rec_args, dict) else vars(rec_args)

    # Reproduce the arm EXACTLY from its own recorded args namespace.
    a = build_parser().parse_args(["--stage", "S-W", "--out", str(CLONE / "_d0_out")])
    unknown = []
    for k, v in rec_args.items():
        if hasattr(a, k):
            setattr(a, k, v)
        else:
            unknown.append(k)
    torch.manual_seed(int(getattr(a, "seed", 0)))
    stack = build_stack_from_args(a)

    # --- reproduce the PRE-FIX freeze: GROUP MAP ALONE (no gradreach) ---
    groups = set(stage_trainable_groups(a.stage))
    for name, p in stack.named_parameters():
        p.requires_grad_(stack.group_of(name) in groups)

    names_trainable = [n for n, p in stack.named_parameters() if p.requires_grad]
    n_trainable = sum(int(p.numel()) for p in stack.parameters() if p.requires_grad)

    # --- the checkpoint's optimizer ---
    opt = ck["opt"]
    pg = opt["param_groups"]
    st = opt["state"]
    idxs = [i for g in pg for i in g["params"]]

    row = {
        "arm": arm,
        "rebuilt_n_trainable_params_GROUP_MAP_ALONE": n_trainable,
        "ckpt_recorded_freeze_n_trainable": ck["config"]["freeze"]["n_trainable"],
        "MATCH_rebuild_reproduces_the_run": (
            n_trainable == ck["config"]["freeze"]["n_trainable"]),
        "n_trainable_tensors_rebuilt": len(names_trainable),
        "n_optimizer_slots_in_ckpt": len(idxs),
        "MATCH_slot_count": len(names_trainable) == len(idxs),
        "unknown_recorded_arg_keys": unknown[:10],
    }

    if row["MATCH_slot_count"]:
        stepped, skipped = [], []
        shape_ok = True
        params = [p for p in stack.parameters() if p.requires_grad]
        for i, (nm, p) in enumerate(zip(names_trainable, params)):
            has = i in st
            if has:
                ea = st[i].get("exp_avg")
                if ea is not None and tuple(ea.shape) != tuple(p.shape):
                    shape_ok = False
                stepped.append(nm)
            else:
                skipped.append(nm)
        row["shape_alignment_ok"] = shape_ok      # control: the mapping is real
        row["n_stepped"] = len(stepped)
        row["n_NEVER_STEPPED"] = len(skipped)

        def bucket(ns):
            b = {}
            for n in ns:
                top = n.split(".")[0]
                b[top] = b.get(top, 0) + 1
            return dict(sorted(b.items()))

        row["stepped_by_module"] = bucket(stepped)
        row["NEVER_STEPPED_by_module"] = bucket(skipped)
        nm2p = dict(stack.named_parameters())
        row["NEVER_STEPPED_numel_by_module"] = {}
        for n in skipped:
            top = n.split(".")[0]
            row["NEVER_STEPPED_numel_by_module"][top] = (
                row["NEVER_STEPPED_numel_by_module"].get(top, 0)
                + int(nm2p[n].numel()))
        row["NEVER_STEPPED_numel_by_module"] = dict(
            sorted(row["NEVER_STEPPED_numel_by_module"].items(),
                   key=lambda kv: -kv[1]))
        # THE HEADLINE CELL: was ANY ema teacher tensor stepped?
        ema_stepped = [n for n in stepped if n.startswith("ema_o5_")]
        ema_skipped = [n for n in skipped if n.startswith("ema_o5_")]
        row["EMA_TEACHER_tensors_STEPPED"] = len(ema_stepped)
        row["EMA_TEACHER_tensors_NEVER_STEPPED"] = len(ema_skipped)
        row["EMA_TEACHER_examples_stepped"] = ema_stepped[:5]
        # control that must read non-zero: the encoder student MUST be stepped
        row["CONTROL_encoder_student_tensors_STEPPED"] = len(
            [n for n in stepped if n.startswith("encoder.")])
        row["CONTROL_encoder_student_tensors_SKIPPED"] = len(
            [n for n in skipped if n.startswith("encoder.")])
    OUT[arm] = row
    del ck, stack

print(json.dumps(OUT, indent=1, default=str))
