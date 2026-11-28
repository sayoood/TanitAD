"""Is bf16 autocast numerically safe for REFe? FP32 vs bf16 on the SAME weights and inputs.

⚠️ Table A12: "Training precision FP32". `train.py --amp bf16` is 3.5x faster on the A40 (MEASURED,
refe/bench_speed.py) and is therefore a declared departure -- admissible only if it does not change
what the model learns. This compares one training step's forward outputs and gradients:

  A  traj / score relative error (bf16 vs FP32)             must be < 5 %
  B  loss relative difference                                 must be < 2 %
  C  cosine of the trainable-parameter gradient (all params) must be > 0.99
  D  CONTROL: the same cosine on a DIFFERENT batch            must be < 0.95 -- proves C discriminates
     (a gradient metric that agrees with everything is not evidence)

With --weights, the REAL pretrained DINOv3 trunk is loaded (realistic activation magnitudes -- a
random trunk can hide an overflow). Prints AMP_NUMERICS_OK or AMP_NUMERICS_FAILED.

  python diag_amp.py --backbone vitl16 --weights /workspace/data/backbones/dinov3-vitl16 --batch 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import REFe, REFeConfig, wta_loss  # noqa: E402


def one_step(net, batch, amp: bool):
    img, ego, goal, tgt, s_tgt, calib = batch
    net.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        traj, score = net(img, ego, goal, calib=calib)
    traj, score = traj.float(), score.float()
    l_traj, _ = wta_loss(traj, tgt)
    loss = l_traj + 0.1 * F.binary_cross_entropy_with_logits(score, s_tgt)
    loss.backward()
    g = torch.cat([p.grad.detach().float().flatten() for p in net.parameters()
                   if p.requires_grad and p.grad is not None])
    return traj.detach(), score.detach(), float(loss), g


def make_batch(cfg, net, B, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, generator=g).cuda()
    ego = (torch.randn(B, cfg.ego_dim, generator=g) * 3).cuda()
    goal = (torch.randn(B, 2 * cfg.n_goal_points, generator=g) * 20).cuda()
    tgt = (torch.randn(B, cfg.horizon_steps, cfg.traj_dim, generator=g) * 5).cuda()
    s_tgt = torch.rand(B, cfg.n_proposals, cfg.n_score_components, generator=g).cuda()
    calib = torch.tensor([list(map(list, net.baked_calib())) for _ in range(B)], dtype=torch.float64)
    return img, ego, goal, tgt, s_tgt, calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--compile", action="store_true",
                    help="compare bf16 + COMPILED backbone against eager FP32 (the pod's run config)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False          # the FP32 reference is strict FP32
    torch.backends.cudnn.allow_tf32 = False
    cfg = REFeConfig.for_backbone(a.backbone)
    net = REFe(cfg)
    if a.weights:
        import load_dinov3 as LD
        loaded, missing, unused = LD.map_into_backbone(net.backbone, LD.load_state(a.weights))
        print(f"  trunk: {len(loaded)} tensors loaded (missing {len(missing)}, unused {len(unused)})")
    net = net.cuda().train()
    ref = None
    if a.compile:
        import copy
        ref = copy.deepcopy(net)                 # eager FP32 reference with IDENTICAL weights
        net.backbone.compile()
    b1 = make_batch(cfg, net, a.batch, 1)
    b2 = make_batch(cfg, net, a.batch, 2)
    fp = ref if ref is not None else net
    t32, s32, l32, g32 = one_step(fp, b1, amp=False)
    t16, s16, l16, g16 = one_step(net, b1, amp=True)
    _, _, _, g32b = one_step(fp, b2, amp=False)
    if a.compile:
        print("  (bf16 arm runs the COMPILED backbone; the FP32 reference is an eager deep copy)")
    rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-12))
    cos = lambda x, y: float(F.cosine_similarity(x, y, dim=0))
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name}] {'PASS' if cond else 'FAIL'} -- {detail}")

    rt, rs = rel(t16, t32), rel(s16, s32)
    check("A.outputs", rt < 0.05 and rs < 0.05, f"traj rel err {100*rt:.2f} %, score rel err {100*rs:.2f} %")
    rl = abs(l16 - l32) / max(abs(l32), 1e-12)
    check("B.loss", rl < 0.02, f"loss FP32 {l32:.5f} vs bf16 {l16:.5f} ({100*rl:.2f} %)")
    c = cos(g16, g32)
    check("C.grad_cosine", c > 0.99, f"cos(grad bf16, grad FP32) = {c:.5f} over {g32.numel():,} values")
    cd = cos(g32b, g32)
    check("D.control_discriminates", cd < 0.95, f"cos(grad batch2, grad batch1) = {cd:.5f} (must be LOW)")
    print("AMP_NUMERICS_OK" if ok else "AMP_NUMERICS_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
