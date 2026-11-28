"""PROTOTYPE of the auxiliary drivable-area head (design reference; refe/model.py and refe/train.py are NOT edited).

`DrivableAuxHead` reads REFe's `visual_ctx` ([B, N_cam*P, dec_width] = scene_proj(visual), the SAME tensor the scoring
decoder cross-attends to) with a grid of learnable BEV queries and predicts, per 2 m cell, a drivable-mask logit and a
signed distance (m).  Inputs are IMAGE-DERIVED TOKENS ONLY: no ego, goal, nav, or map input (admissibility: SPEC section 6).
`aux_drivable_loss` supervises it from the offline label (int8 SDF at 0.5 m, see proto_label.py).
Run the self-test:  PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe head_proto.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REFE = Path(__file__).resolve().parents[2] / "refe"
sys.path.insert(0, str(REFE))
import model as REFE_MODEL  # noqa: E402  (refe/model.py: CrossBlock, REFe, REFeConfig)

# the label geometry (must equal proto_label.py): rear-axle frame, x forward / y left
X0, X1, Y0, Y1 = -8.0, 72.0, -32.0, 32.0
FINE_RES, SDF_UNIT_M = 0.5, 0.1
NX_F, NY_F = 160, 128


class DrivableAuxHead(nn.Module):
    def __init__(self, ctx_dim: int = 256, width: int = 256, heads: int = 8, depth: int = 2,
                 cell: float = 2.0, sdf_scale: float = 6.0, n_freq: int = 4):
        super().__init__()
        self.cell, self.sdf_scale = cell, sdf_scale
        self.nx, self.ny = int(round((X1 - X0) / cell)), int(round((Y1 - Y0) / cell))
        xs = X0 + (torch.arange(self.nx) + 0.5) * cell
        ys = Y0 + (torch.arange(self.ny) + 0.5) * cell
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        ux, uy = (gx - X0) / (X1 - X0), (gy - Y0) / (Y1 - Y0)                   # [0, 1]
        k = 2.0 ** torch.arange(n_freq) * math.pi
        feats = [ux.reshape(-1, 1), uy.reshape(-1, 1)]
        for u in (ux, uy):
            feats += [torch.sin(u.reshape(-1, 1) * k), torch.cos(u.reshape(-1, 1) * k)]
        self.register_buffer("pos_feat", torch.cat(feats, -1), persistent=False)      # [nx*ny, 2 + 4*n_freq]
        self.pos_mlp = nn.Linear(self.pos_feat.shape[-1], width)
        self.q = nn.Parameter(torch.zeros(1, self.nx * self.ny, width))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.blocks = nn.ModuleList([REFE_MODEL.CrossBlock(width, heads, ctx_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 2)

    def forward(self, ctx):
        B = ctx.shape[0]
        q = self.q.expand(B, -1, -1) + self.pos_mlp(self.pos_feat)[None]
        for blk in self.blocks:
            q = blk(q, ctx)
        o = self.out(self.norm(q))
        logit = o[..., 0].reshape(B, self.nx, self.ny)
        sdf = self.sdf_scale * torch.tanh(o[..., 1]).reshape(B, self.nx, self.ny)   # metres, bounded
        return logit, sdf


def coarse_targets(label_i8: torch.Tensor, cell: float = 2.0, sdf_scale: float = 6.0):
    """[B, 160, 128] int8 SDF (0.1 m units, + inside) -> (mask fraction [B, nx, ny], block-mean clipped SDF in m)."""
    b = int(round(cell / FINE_RES))
    s = label_i8.float() * SDF_UNIT_M
    B = s.shape[0]
    s = s.view(B, NX_F // b, b, NY_F // b, b)
    return (s > 0).float().mean((2, 4)), s.clamp(-sdf_scale, sdf_scale).mean((2, 4))


def aux_drivable_loss(logit, sdf_pred, label_i8, valid=None, sdf_scale: float = 6.0, cell: float = 2.0, w_sdf: float = 1.0):
    """BCE(soft mask fraction) + w_sdf * Huber(SDF / sdf_scale).  `valid` [B] bool: frames with no label contribute exactly 0."""
    frac, sdf_t = coarse_targets(label_i8, cell, sdf_scale)
    bce = F.binary_cross_entropy_with_logits(logit, frac, reduction="none").mean((1, 2))
    hub = F.smooth_l1_loss(sdf_pred / sdf_scale, sdf_t / sdf_scale, beta=0.1, reduction="none").mean((1, 2))
    per = bce + w_sdf * hub
    if valid is None:
        valid = torch.ones_like(per, dtype=torch.bool)
    v = valid.float()
    denom = v.sum().clamp(min=1.0)
    return (per * v).sum() / denom, (bce * v).sum() / denom, (hub * v).sum() / denom


def derangement(keys: list, seed: int) -> list:
    """The LABEL-SHUFFLED control arm: a permutation of the label keys with ZERO fixed points, so the multiset of labels is
    unchanged and only the image<->label alignment is destroyed."""
    import random
    n = len(keys)
    rng = random.Random(seed)
    for _ in range(1000):
        p = list(range(n)); rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return [keys[i] for i in p]
    raise RuntimeError("no derangement found")


# ------------------------------------------------------------------ self-test
def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return bool(ok)


def selftest() -> int:
    torch.manual_seed(0)
    bad = 0
    # 1. size and FLOPs at the real REFe dims (ctx = 4 cams x 1920 tokens x 256)
    from torch.utils.flop_counter import FlopCounterMode
    head = DrivableAuxHead()
    n_par = sum(p.numel() for p in head.parameters())
    ctx = torch.randn(1, 7680, 256, requires_grad=True)
    with FlopCounterMode(display=False) as fc:
        lg, sd = head(ctx)
    fwd = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc2:
        lg, sd = head(ctx)
        (lg.sum() + sd.sum()).backward()
    fb = fc2.get_total_flops()
    trunk_fwd = 4 * 764.98e9 * 2           # refe_flops.py (analytic, 4 cameras): 764.98 GMAC / camera
    # FlopCounterMode counts the matmuls but NOT the fused attention kernel on CPU (MEASURED: 8.7 GFLOP counted vs 32.2 analytic), so the
    # attention term is added analytically: per block self-attn 4*nq^2*d + cross-attn 4*nq*nk*d, nq = 1280 queries, nk = 7680 keys, d = 256.
    nq, nk, d, depth = head.nx * head.ny, 7680, 256, len(head.blocks)
    attn = depth * (4 * nq * nq * d + 4 * nq * nk * d)
    fwd_all, fb_all = fwd + attn, fb + 3 * attn
    print(f"  head params {n_par:,}  ({n_par / 18_991_426 * 100:.1f} % of REFe's 18,991,426 trainable)")
    print(f"  head fwd: {fwd / 1e9:.1f} GFLOP counted (matmuls) + {attn / 1e9:.1f} GFLOP attention (analytic) = {fwd_all / 1e9:.1f} GFLOP; "
          f"fwd+bwd {fb_all / 1e9:.1f} GFLOP per sample")
    print(f"  trunk fwd (refe_flops.py analytic, 4 cameras) {trunk_fwd / 1e12:.2f} TFLOP -> head fwd = {fwd_all / trunk_fwd * 100:.2f} % of it; "
          f"head fwd+bwd vs trunk fwd + dgrad (2x fwd) = {fb_all / (2 * trunk_fwd) * 100:.2f} %")
    out_shape = tuple(lg.shape)
    bad += not _check("output grid", out_shape == (1, 40, 32), str(out_shape))

    # 2. known-answer loss identities
    L = torch.full((2, NX_F, NY_F), 127, dtype=torch.int8)                         # everything inside, margin >> clip
    z = torch.zeros(2, 40, 32)
    l, bce, hub = aux_drivable_loss(z, torch.full((2, 40, 32), 6.0), L)
    bad += not _check("all-inside, logit 0: BCE == ln 2 exactly", abs(float(bce) - math.log(2)) < 1e-6, f"{float(bce):.8f}")
    bad += not _check("all-inside, sdf = +clip: Huber == 0.0 exactly", float(hub) == 0.0, f"{float(hub)!r}")
    Lh = L.clone(); Lh[:, :, 64:] = -127                                           # half drivable, half not (cells align to 2 m)
    fr, st = coarse_targets(Lh)
    bad += not _check("half-half label: mask fraction 1 on one side, 0 on the other", float(fr[0, 0, 0]) == 1.0 and float(fr[0, 0, -1]) == 0.0)
    l0, _, _ = aux_drivable_loss(z, z, L, valid=torch.tensor([False, False]))
    bad += not _check("no valid label in the batch: loss is exactly 0.0 (control reads 0)", float(l0) == 0.0)

    # 3. gradient plumbing on the REAL REFe (tiny backbone/image so it runs on CPU): the aux loss must reach the trunk
    #    through visual_ctx, and a head wired to ctx.detach() must NOT (the mutation arm)
    cfg = REFE_MODEL.REFeConfig.for_backbone("vits16", img_h=64, img_w=128)
    net = REFE_MODEL.REFe(cfg)
    cap = {}
    def hook(mod, inp, outp):                                   # scene_proj is applied twice; keep the visual-token call
        if outp.shape[1] > 100:
            cap["ctx"] = outp
    h = net.scene_proj.register_forward_hook(hook)
    img = torch.rand(1, 4, 3, 64, 128)
    ego = torch.zeros(1, cfg.ego_dim)
    goal = torch.zeros(1, 2 * cfg.n_goal_points)
    lab = torch.randint(-127, 127, (1, NX_F, NY_F), dtype=torch.int8)

    def groups(model):
        g = {"lora": [], "scene_proj": list(model.scene_proj.parameters()), "pos3d_mlp": list(model.pos3d_mlp.parameters()),
             "reg_compress": list(model.reg_compress.parameters()), "traj_head": list(model.traj_head.parameters())}
        for n_, p_ in model.named_parameters():
            if n_.startswith("backbone") and (".A" in n_ or ".B" in n_):
                g["lora"].append(p_)
        return g

    def run(detach_ctx: bool, w_aux: float):
        net.zero_grad(set_to_none=True)
        hd = DrivableAuxHead(ctx_dim=cfg.dec_width)
        net(img, ego, goal)
        c = cap["ctx"].detach() if detach_ctx else cap["ctx"]
        lg_, sd_ = hd(c)
        loss, _, _ = aux_drivable_loss(lg_, sd_, lab)
        (w_aux * loss).backward()
        gs = {k: sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in v) for k, v in groups(net).items()}
        gh = sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in hd.parameters())
        return gs, gh
    gs, gh = run(False, 1.0)
    bad += not _check("aux loss -> NON-ZERO gradient on the head", gh > 0, f"{gh:.3e}")
    for k in ("lora", "scene_proj", "pos3d_mlp"):
        bad += not _check(f"aux loss -> NON-ZERO gradient on trunk group {k}", gs[k] > 0, f"{gs[k]:.3e}")
    for k in ("traj_head", "reg_compress"):                       # these are NOT on the visual_ctx path: control must be exactly 0.0
        bad += not _check(f"CONTROL: aux-only gradient on {k} reads exactly 0.0 (not on the visual_ctx path)", gs[k] == 0.0, f"{gs[k]!r}")
    gs_d, gh_d = run(True, 1.0)
    bad += not _check("MUTATION (head fed ctx.detach()): head still learns", gh_d > 0, f"{gh_d:.3e}")
    for k in ("lora", "scene_proj", "pos3d_mlp"):
        bad += not _check(f"MUTATION (head detached): trunk group {k} gradient is exactly 0.0", gs_d[k] == 0.0, f"{gs_d[k]!r}")
    gs_z, gh_z = run(False, 0.0)
    bad += not _check("w_aux = 0: every gradient exactly 0.0 (head and trunk)", gh_z == 0.0 and all(v == 0.0 for v in gs_z.values()))
    h.remove()

    # 4. the label-shuffled control keeps the label multiset and has no fixed point
    keys = [f"k{i}" for i in range(50)]
    sh = derangement(keys, 7)
    bad += not _check("derangement: 0 fixed points", all(a != b for a, b in zip(keys, sh)))
    bad += not _check("derangement: same multiset of labels", sorted(keys) == sorted(sh))
    print("SELFTEST", "FAILED" if bad else "PASSED", f"({bad} failing)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
