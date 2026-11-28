#!/usr/bin/env python3
"""FIELDP7 reachability probe: does --w-field reach the proposal-side parameters of P7's trainer, through the REAL train.run, to .backward()?  CPU, real rows, P7's model_final.

One invocation = one real `train.run(a)` on the given argv, stopped at the FIRST optimiser step (accum 1, batch 1, one micro-batch).  It records, per parameter group, the gradient the
trainer's own loss assembly produced (captured at torch.nn.utils.clip_grad_norm_, i.e. BEFORE clipping, exactly what the optimiser would see), then raises Stop.  Nothing is trained.

Memory only: `model.Block.forward` is wrapped in activation checkpointing (same maths, the LoRA gradient is still computed; a ViT-L graph at 512x960x4 cameras is ~15 GB otherwise).

    python fieldp7_reach.py --refe <refe dir> --args "<argv>" --out <json> [--grads <pt>]
Compare runs with fieldp7_reach_compare.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GROUPS = ("traj_head", "dec", "queries", "scene_proj", "reg_compress", "registers", "cmd_embed", "ego_enc", "pos3d_mlp", "backbone", "score_q_mlp", "score_dec", "score_head")


class Stop(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--args", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grads", default=None, help="save the per-parameter pre-clip gradients (float32 CPU tensors) here")
    a_ = ap.parse_args()
    refe = Path(a_.refe).resolve()
    sys.path.insert(0, str(refe))
    import torch
    import torch.utils.checkpoint as ckpt
    import model as M
    import train as T

    orig_fwd = M.Block.forward

    def ck_fwd(self, x, rope=None, rope_offset: int = 0):
        return ckpt.checkpoint(orig_fwd, self, x, rope, rope_offset, use_reentrant=False)
    M.Block.forward = ck_fwd

    cap = {"model": None, "snap": None, "field_calls": [], "train_file": T.__file__}
    Orig = T.REFe

    class Rec(Orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            cap["model"] = self
    T.REFe = Rec

    ofl = T.field_loss

    def rec_field(traj, sdf, **kw):
        out = ofl(traj, sdf, **kw)
        v = out[0] if isinstance(out, tuple) else out
        cap["field_calls"].append({"value": float(v.detach()), "margin": kw.get("margin"), "interp": kw.get("interp"), "n_valid": int(kw["valid"].sum()) if kw.get("valid") is not None else None})
        return out
    T.field_loss = rec_field

    ocl = torch.nn.utils.clip_grad_norm_

    def snap_clip(params, *a, **k):
        if cap["snap"] is None:
            by = {}
            m = cap["model"]
            for n, p in m.named_parameters():
                by[n] = None if p.grad is None else p.grad.detach().float().clone()
            cap["snap"] = by
        return ocl(params, *a, **k)
    torch.nn.utils.clip_grad_norm_ = snap_clip

    class StopAdam(torch.optim.AdamW):
        def step(self, *a, **k):
            raise Stop()
    Adam0 = torch.optim.AdamW
    torch.optim.AdamW = StopAdam

    ap2 = T.build_parser()
    ns = ap2.parse_args(a_.args.split())
    rc = None
    try:
        rc = T.run(ns)
    except Stop:
        rc = "STOPPED_AT_FIRST_STEP"
    finally:
        torch.optim.AdamW = Adam0
        torch.nn.utils.clip_grad_norm_ = ocl
        T.field_loss = ofl
        T.REFe = Orig
        M.Block.forward = orig_fwd
    if rc != "STOPPED_AT_FIRST_STEP" or cap["snap"] is None:
        raise SystemExit(f"fieldp7_reach: the trainer did not reach its first optimiser step (rc={rc!r}); nothing was measured")
    m = cap["model"]
    rows = {}
    for g in GROUPS:
        names = [n for n, p in m.named_parameters() if n.split(".")[0] == g]
        trk = [n for n in names if dict(m.named_parameters())[n].requires_grad]
        gr = [cap["snap"][n] for n in trk]
        nn_none = sum(1 for x in gr if x is None)
        nz = sum(1 for x in gr if x is not None and bool((x != 0).any()))
        absum = float(sum(float(x.double().abs().sum()) for x in gr if x is not None))
        mx = float(max([float(x.abs().max()) for x in gr if x is not None] or [0.0]))
        rows[g] = {"n_params": len(names), "n_trainable": len(trk), "n_grad_none": nn_none, "n_grad_nonzero": nz, "abs_sum": absum, "abs_max": mx}
    h = hashlib.sha256()
    for n in sorted(cap["snap"]):
        h.update(n.encode())
        if cap["snap"][n] is not None:
            h.update(cap["snap"][n].numpy().tobytes())
    res = {"argv_sha256": hashlib.sha256("\x00".join(a_.args.split()).encode()).hexdigest(), "train_py": cap["train_file"], "w_field": ns.w_field, "field_labels": ns.field_labels,
           "field_loss_calls": cap["field_calls"], "groups": rows, "grad_sha256_all": h.hexdigest(), "n_params_with_grad": sum(1 for v in cap["snap"].values() if v is not None)}
    Path(a_.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a_.grads:
        torch.save({k: v for k, v in cap["snap"].items()}, a_.grads)
    print("FIELDP7_REACH", json.dumps({"w_field": ns.w_field, "calls": len(cap["field_calls"]), "grad_sha256": res["grad_sha256_all"][:16]}))
    for g, r in rows.items():
        print(f"  {g:13s} trainable {r['n_trainable']:4d} none {r['n_grad_none']:4d} nonzero {r['n_grad_nonzero']:4d} abs_sum {r['abs_sum']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
