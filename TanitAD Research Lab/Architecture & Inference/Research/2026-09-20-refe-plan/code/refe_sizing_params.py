"""MEASURED parameter accounting for REFe's unspecified modules.

TWO INDEPENDENT ROUTES, and they must agree:
  (A) INSTANTIATE refe/model.py and count real tensors  -> the artifact
  (B) an ANALYTIC formula derived from torch.nn module definitions, written here WITHOUT
      reading the model's reported totals -> the independently authored reference

⛔ A cross-check that re-runs the producer's own derivation measures determinism, not
correctness (CLAUDE.md). (B) is authored from nn.Linear/nn.LayerNorm/nn.MultiheadAttention
parameter shapes, so it is a genuinely separate derivation.
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")

import torch
from refe.model import REFe, REFeConfig


# ---------------------------------------------------------------- (B) analytic
def ln(d):                       return 2 * d
def lin(a, b, bias=True):        return a * b + (b if bias else 0)
def mha(d, ctx=None):
    """nn.MultiheadAttention: fused in_proj when kdim==vdim==embed_dim, else split."""
    if ctx is None or ctx == d:
        return 3 * d * d + 3 * d + lin(d, d)          # in_proj_weight+bias, out_proj
    return d * d + d * ctx + d * ctx + 3 * d + lin(d, d)

def crossblock(d, ctx, mlp_ratio=4.0):
    hid = int(d * mlp_ratio)
    return 3 * ln(d) + mha(d) + mha(d, ctx) + lin(d, hid) + lin(hid, d)

def regcompress(d, mlp_ratio=4.0, blocks=1):
    hid = int(d * mlp_ratio)
    return blocks * (3 * ln(d) + mha(d) + lin(d, hid) + lin(hid, d))

def vit_block(d, mlp_ratio, rank):
    hid = int(d * mlp_ratio)
    lora = 2 * (rank * d + d * rank)                  # Q and V each get A[r,d] + B[d,r]
    frozen_attn = d * d * 3 + lin(d, d)               # q.base, k, v.base (no bias) + proj
    return 2 * ln(d) + frozen_attn + lora + lin(d, hid) + lin(hid, d) + 2 * d

def analytic(cfg):
    d, dd = cfg.width, cfg.dec_width
    P = cfg.n_cameras * (cfg.img_h // cfg.patch) * (cfg.img_w // cfg.patch)
    R = cfg.n_cameras * cfg.n_registers
    sdd = cfg.dec_depth if cfg.score_dec_depth is None else cfg.score_dec_depth
    trunk = (lin(3 * cfg.patch * cfg.patch, d)        # Conv2d(3,d,p,p) == d*(3*p*p)+d
             + d + 4 * d                              # cls_token + 4 DINOv3 reg_tokens
             + cfg.depth * vit_block(d, cfg.mlp_ratio, cfg.lora_rank) + ln(d))
    lora = cfg.depth * 2 * (2 * cfg.lora_rank * d)
    parts = dict(
        registers=R * d,
        pos3d=P * d,
        reg_compress=regcompress(d, getattr(cfg, "_rc_mlp", 4.0), getattr(cfg, "_rc_blocks", 1)),
        scene_proj=lin(d, dd),
        ego_enc=lin(cfg.ego_dim + 2 * cfg.n_goal_points, dd) + lin(dd, dd),
        queries=cfg.n_proposals * dd,
        dec=cfg.dec_depth * crossblock(dd, dd),
        traj_head=lin(dd, dd) + lin(dd, cfg.horizon_steps * cfg.traj_dim),
        score_dec=sdd * crossblock(dd, dd),
        score_head=lin(dd, cfg.n_score_components),
    )
    head_trainable = sum(parts.values())
    return dict(parts=parts, trunk_total=trunk, lora=lora,
                total=trunk + head_trainable,
                trainable=lora + head_trainable,
                frozen_trunk=trunk - lora)


# ---------------------------------------------------------------- (A) measured
def measured(cfg):
    m = REFe(cfg)
    per = {}
    for name, mod in [("backbone", m.backbone), ("reg_compress", m.reg_compress),
                      ("scene_proj", m.scene_proj), ("ego_enc", m.ego_enc),
                      ("dec", m.dec), ("traj_head", m.traj_head),
                      ("score_dec", m.score_dec), ("score_head", m.score_head)]:
        per[name] = dict(total=sum(p.numel() for p in mod.parameters()),
                         trainable=sum(p.numel() for p in mod.parameters() if p.requires_grad))
    per["registers"] = dict(total=m.registers.numel(), trainable=m.registers.numel())
    per["pos3d"] = dict(total=m.pos3d.numel(), trainable=m.pos3d.numel())
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    del m
    return dict(per=per, total=tot, trainable=tr)


if __name__ == "__main__":
    cfg = REFeConfig.for_backbone("vitl16")
    a = analytic(cfg)
    print("=== ANALYTIC (route B, independently authored) ===")
    for k, v in a["parts"].items():
        print(f"  {k:14s} {v:>12,}")
    print(f"  {'LoRA(trunk)':14s} {a['lora']:>12,}")
    print(f"  {'frozen trunk':14s} {a['frozen_trunk']:>12,}")
    print(f"  TOTAL {a['total']:,}   TRAINABLE {a['trainable']:,} "
          f"({100*a['trainable']/a['total']:.2f} %)")

    print("\n=== MEASURED (route A, real instantiation) ===")
    m = measured(cfg)
    for k, v in m["per"].items():
        print(f"  {k:14s} total {v['total']:>12,}  trainable {v['trainable']:>12,}")
    print(f"  TOTAL {m['total']:,}   TRAINABLE {m['trainable']:,} "
          f"({100*m['trainable']/m['total']:.2f} %)")

    print("\n=== AGREEMENT CONTROL ===")
    ok = True
    for k in ("reg_compress", "scene_proj", "ego_enc", "dec", "traj_head",
              "score_dec", "score_head", "registers", "pos3d"):
        exp, got = a["parts"][k], m["per"][k]["trainable"]
        flag = "OK " if exp == got else "MISMATCH"
        if exp != got: ok = False
        print(f"  {flag} {k:14s} analytic {exp:>12,}  measured {got:>12,}")
    for k, exp, got in (("TOTAL", a["total"], m["total"]),
                        ("TRAINABLE", a["trainable"], m["trainable"])):
        flag = "OK " if exp == got else "MISMATCH"
        if exp != got: ok = False
        print(f"  {flag} {k:14s} analytic {exp:>12,}  measured {got:>12,}")
    print("VERDICT:", "ROUTES AGREE" if ok else "ROUTES DISAGREE -- do not quote either")
