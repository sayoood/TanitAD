#!/usr/bin/env python3
"""Launch gate for P2, the joint fine-tune (eval/PREREG_P2.md). Prints ZZP2GATE PASS|FAIL and writes <out>/gate.json.

  G1 warm start = what the planner loads: the warm-started trainer model and planner.py's load of the same checkpoint
     give bit-identical trajectories, proposal scores and labelled-set scores (fp32, eval) on real rows;
  G2 the mechanism, as an analytic pair on the SAME real batch, score loss ONLY:
     un-detached (--scorer-sees-trunk) -> a finite NON-ZERO gradient on the backbone LoRA;
     detached (the old default)        -> EXACTLY 0.0 on the LoRA (the deliberate-regression arm, built in);
     both -> exactly 0.0 on traj_head (the paper's detach of the candidate trajectories is kept);
  G3 PDM sets only: every set the pdm_only bank serves carries PDM targets, a training row without one gets cand_m = 0,
     and the MUTATION (the same bank without pdm_only) must show sets without PDM targets (else the check is inert);
  G4 the REAL trainer (train.py, the P2 flags) runs 2 optimiser steps, writes finite traj and score losses, saves
     model_final.pt; it reloads through the planner's loader; every LoRA tensor that moved, moved; the frozen trunk is
     bit-identical to the warm start.
    python p2_gate.py --init-from <model_final.pt> --onpolicy <small sets dir> --pdm-labels <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io  # noqa: E402
import load_dinov3 as LD  # noqa: E402
import train as T  # noqa: E402
from model import BACKBONES, REFe, REFeConfig  # noqa: E402


def key(r):
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))


def is_lora(n):
    return n.startswith("backbone") and (".A" in n or ".B" in n)


def build(backbone, init_from, detach, dev):
    cfg = REFeConfig.for_backbone(backbone)
    cfg.detach_scorer_context = detach
    m = REFe(cfg)
    loaded, missing, unused = LD.map_into_backbone(m.backbone, LD.load_state(
        os.path.join(LD.BACKBONE_ROOT, BACKBONES[backbone]["weights"])))
    assert loaded and not missing and not unused
    T.warm_start(m, init_from)
    return m.to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--targets", default="/workspace/data/refe_navtrain/train_grow")
    ap.add_argument("--images", default="/workspace/data/navtrain_pixels")
    ap.add_argument("--calib", default="/workspace/data/refe_navtrain/train_grow/calib_table.json")
    ap.add_argument("--onpolicy", required=True)
    ap.add_argument("--pdm-labels", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-trainer", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    g = {}
    cfg0 = REFeConfig.for_backbone(a.backbone)
    M, H = cfg0.n_proposals, cfg0.horizon_steps
    op = T.OnPolicyBank(a.onpolicy, M, H, pdm_labels=a.pdm_labels, pdm_only=True)
    op_mut = T.OnPolicyBank(a.onpolicy, M, H, pdm_labels=a.pdm_labels)               # the mutation: no pdm_only
    ds = T.TargetBank(a.targets, a.images, cfg0, calib=a.calib, onpolicy=op)
    ok_rows = [i for i, r in enumerate(ds.rows) if key(r) in op.by][:2]
    no_rows = [i for i, r in enumerate(ds.rows) if key(r) in op_mut.by and key(r) not in op.by][:1]
    batch = next(iter(DataLoader(Subset(ds, ok_rows + no_rows), batch_size=len(ok_rows) + len(no_rows))))
    img, ego, goal, _tgt, cxy, ctg, cm, _ci, cal = batch
    cmr = cm.sum(1)
    g["G3_pdm_only"] = {
        "served_sets": len(op.by), "served_without_pdm": sum(1 for k in op.by if k not in op.pdm_keys),
        "dropped_non_pdm": op.n_dropped_non_pdm, "mutation_served_without_pdm": sum(1 for k in op_mut.by if k not in op_mut.pdm_keys),
        "batch_pdm_rows_masked_in": [float(x) for x in cmr[:len(ok_rows)]], "batch_non_pdm_rows_mask": [float(x) for x in cmr[len(ok_rows):]]}
    q = g["G3_pdm_only"]
    q["ok"] = (q["served_sets"] > 0 and q["served_without_pdm"] == 0 and q["mutation_served_without_pdm"] > 0
               and len(no_rows) == 1 and all(x > 0 for x in q["batch_pdm_rows_masked_in"])
               and all(x == 0 for x in q["batch_non_pdm_rows_mask"]))
    img, ego, goal, cxy, ctg, cm = (x.to(dev) for x in (img, ego, goal, cxy, ctg, cm))
    calv = cal if cal.numel() else None
    # G1: warm start == the planner's load
    pl = REFe(ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.init_from)).to(dev)
    ckpt_io.load_for_inference(pl, a.init_from, map_location=dev, backbone=a.backbone)
    mu = build(a.backbone, a.init_from, detach=False, dev=dev)
    pl.eval(); mu.eval()
    with torch.no_grad():
        o_p = pl(img, ego, goal, calib=calv, score_extra=cxy)
        o_u = mu(img, ego, goal, calib=calv, score_extra=cxy)
    d1 = [float((x.float() - y.float()).abs().max()) for x, y in zip(o_p, o_u)]
    g["G1_warm_start_equals_planner_load"] = {"max_abs_traj_score_extra": d1, "ok": all(v == 0.0 for v in d1)}
    del pl
    torch.cuda.empty_cache()
    # G2: the analytic pair, score loss only (BCE on the labelled sets, masked as train.py masks)
    res = {}
    for nm, detach in (("undetached", False), ("detached", True)):
        m = mu if not detach else build(a.backbone, a.init_from, detach=True, dev=dev)
        m.eval()
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _t, _s, sx = m(img, ego, goal, calib=calv, score_extra=cxy)
        per = F.binary_cross_entropy_with_logits(sx.float(), ctg[..., :6], reduction="none").mean(-1)
        (per * cm).sum().backward()
        lora = [p.grad for n, p in m.named_parameters() if is_lora(n) and p.requires_grad]
        th = [p.grad for n, p in m.named_parameters() if n.startswith("traj_head") and p.requires_grad]
        res[nm] = {"lora_tensors": len(lora),
                   "lora_grad_abs_sum": float(sum(float(x.abs().sum()) for x in lora if x is not None)),
                   "lora_grad_finite": all(bool(torch.isfinite(x).all()) for x in lora if x is not None),
                   "traj_head_grad_abs_sum": float(sum(float(x.abs().sum()) for x in th if x is not None))}
        if detach:
            del m
    u, d = res["undetached"], res["detached"]
    g["G2_score_loss_reaches_lora_only_when_undetached"] = {
        **res, "ok": (u["lora_tensors"] > 0 and u["lora_grad_abs_sum"] > 0 and u["lora_grad_finite"]
                      and d["lora_grad_abs_sum"] == 0.0 and u["traj_head_grad_abs_sum"] == 0.0
                      and d["traj_head_grad_abs_sum"] == 0.0)}
    del mu, ds, op, op_mut                   # free the banks before the trainer loads its own (the pod's RAM is 50 GB)
    torch.cuda.empty_cache()
    # G4: the real trainer, 2 optimiser steps, the P2 flags
    if not a.skip_trainer:
        run_dir = os.path.join(a.out, "trainer_smoke")
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--backbone", a.backbone, "--targets", a.targets,
               "--scorer-targets", a.targets, "--images", a.images, "--calib", a.calib, "--init-from", a.init_from,
               "--scorer-mode", "onpolicy", "--onpolicy-targets", a.onpolicy, "--pdm-labels", a.pdm_labels, "--pdm-only",
               "--scorer-sees-trunk", "--batch", "2", "--accum", "2", "--steps", "2", "--lr", "5e-5", "--amp", "bf16",
               "--tf32", "--workers", "1", "--log-every", "1", "--out", run_dir]
        rc = subprocess.call(cmd, stdout=open(os.path.join(a.out, "trainer_smoke.log"), "w"), stderr=subprocess.STDOUT)
        rows = []
        mp = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(mp):
            for ln in open(mp, encoding="utf-8"):
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        steps = [r for r in rows if "traj_L1" in r or "l_traj" in r or r.get("event") == "step"]
        fin = os.path.join(run_dir, "model_final.pt")
        g4 = {"rc": rc, "metric_rows": len(rows), "step_rows": len(steps), "model_final": os.path.exists(fin)}
        if os.path.exists(fin):
            m2 = REFe(ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), fin))
            ckpt_io.load_for_inference(m2, fin, map_location="cpu", backbone=a.backbone)
            m0 = REFe(REFeConfig.for_backbone(a.backbone))
            ckpt_io.load_for_inference(m0, a.init_from, map_location="cpu", backbone=a.backbone)
            s2, s0 = m2.state_dict(), m0.state_dict()
            tr = {n for n, p in m0.named_parameters() if p.requires_grad}
            lora_moved = sum(1 for n in s0 if is_lora(n) and not torch.equal(s0[n], s2[n]))
            frozen_moved = sum(1 for n in s0 if n not in tr and n.startswith("backbone") and not torch.equal(s0[n], s2[n]))
            g4.update({"lora_tensors_moved": lora_moved, "frozen_backbone_tensors_moved": frozen_moved,
                       "finite_losses": all(math.isfinite(float(v)) for r in steps for k, v in r.items()
                                            if k in ("traj_L1", "score") and isinstance(v, (int, float)))})
        g4["ok"] = (rc == 0 and g4["model_final"] and g4.get("lora_tensors_moved", 0) > 0
                    and g4.get("frozen_backbone_tensors_moved", 1) == 0 and g4.get("finite_losses", False) and g4["step_rows"] > 0)
        g["G4_real_trainer_smoke"] = g4
    verdict = "PASS" if all(v.get("ok") for v in g.values()) else "FAIL"
    json.dump({"verdict": verdict, "gates": g}, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print(json.dumps(g), flush=True)
    print("ZZP2GATE", verdict, flush=True)
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
