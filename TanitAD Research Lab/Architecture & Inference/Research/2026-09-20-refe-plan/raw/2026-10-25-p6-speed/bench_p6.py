#!/usr/bin/env python3
"""bench_p6.py -- P6 micro-batch timing + equivalence harness (speed study, 2026-10-25).

Builds the REAL REFe through refe/model.py exactly as refe/train.py:run() does for P6's argv
(--backbone vitl16 --nav-mode command --scorer-sees-trunk --amp bf16 --tf32 --compile, trunk loaded by
load_dinov3.map_into_backbone, optionally the step-2603 LoRA/heads via ckpt_io.load_partial) and drives
the SAME micro-batch sequence as train.py's online-labeller path (D4 deferred backward, train.py:1612-1770):

  H2D(k) -> F(k) under bf16 autocast -> float() -> wta_loss -> n_cand sync -> props(k).double().cpu() + sha
         -> collect(k-1) -> BCE(k-1) + the no_grad diagnostics (same syncs) -> (tot/accum).backward() -> ...
  every `accum` micro-batches: clip_grad_norm_(1.0) -> AdamW.step -> zero_grad(set_to_none)

Labels come from a fixed synthetic table (the labeller is a CPU pool, measured separately), so this times
the GPU + host critical path. Levers are switches, each OFF by default (= today's P6):
  --compile {none,backbone,parts}  --compile-mode  --sdpa  --order {current,early}  --img {f32,u8}
  --fused-qkv  --n2-once  --batch/--accum  --fused-adam
Modes:  default = timing (CUDA events + wall), --phases = per-phase event split, --profile F = kernel table,
        --equiv F = one optimiser window on a FIXED batch -> losses, accumulated grads, param delta;
        --compare A B [C ...] = equivalence table (cosine, max rel diff) of B.. against A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", default="/root/tad/speed/refe")
    ap.add_argument("--weights", default="/workspace/data/backbones/dinov3-vitl16")
    ap.add_argument("--ckpt", default="", help="load LoRA+heads from a P6 ckpt_last.pt (ckpt_io.load_partial)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=64)
    ap.add_argument("--n-micro", type=int, default=16, help="timed micro-batches per rep")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile", default="backbone", choices=["none", "backbone", "parts"])
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--sdpa", default="auto", choices=["auto", "flash", "efficient", "math", "cudnn"])
    ap.add_argument("--order", default="current", choices=["current", "early"])
    ap.add_argument("--img", default="f32", choices=["f32", "u8"])
    ap.add_argument("--fused-qkv", action="store_true")
    ap.add_argument("--n2-once", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--rigs", type=int, default=4, help="distinct rigs per micro-batch (pool of 30)")
    # ---- RECIPE-CHANGING levers (PI decision; measured for the decision, never a free win)
    ap.add_argument("--lora-from", type=int, default=0, help="RECIPE: blocks < K run under no_grad (their LoRA frozen, no backward below K)")
    ap.add_argument("--res", default="", help="RECIPE: HxW input resolution (default 512x960)")
    ap.add_argument("--bf16-resid", action="store_true", help="RECIPE (precision): keep the trunk residual stream in bf16")
    ap.add_argument("--perturb-img", type=float, default=0.0, help="CALIBRATION: images *= (1+eps) -> an ULP-level forward perturbation (the floor for kernel changes)")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--profile", default="")
    ap.add_argument("--equiv", default="")
    ap.add_argument("--equiv-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", nargs="+", default=None)
    return ap.parse_args()


# ----------------------------------------------------------------------------- equivalence table
def compare(paths):
    ref = torch.load(paths[0], map_location="cpu", weights_only=False)
    out = []
    for p in paths[1:]:
        d = torch.load(p, map_location="cpu", weights_only=False)
        row = {"vs": os.path.basename(p), "ref": os.path.basename(paths[0])}
        for key in ("grad", "delta"):
            a, b = ref[key].double(), d[key].double()
            cos = float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))
            den = a.abs().max().clamp(min=1e-30)
            row[f"{key}_cos"] = cos
            row[f"{key}_maxabs_over_max"] = float((a - b).abs().max() / den)     # max |diff| / max |ref|
            row[f"{key}_rel_l2"] = float((a - b).norm() / a.norm().clamp(min=1e-30))
            row[f"{key}_bit_identical"] = bool(torch.equal(ref[key], d[key]))
        row["loss_ref"] = ref["losses"]
        row["loss"] = d["losses"]
        row["loss_max_absdiff"] = max(abs(x - y) for x, y in zip(sum(ref["losses"], []), sum(d["losses"], [])))
        row["gnorm_ref"], row["gnorm"] = ref["gnorm"], d["gnorm"]
        if "wta_idx" in ref and "wta_idx" in d:
            fa, fb = sum(ref["wta_idx"], []), sum(d["wta_idx"], [])
            row["wta_flips"] = sum(int(x != y) for x, y in zip(fa, fb))
            row["wta_n"] = len(fa)
        if "names" in ref:
            grp, off = {}, 0
            for n, k in ref["names"]:
                key = n.split(".")[0] if not n.startswith("backbone") else "backbone.lora"
                a_, b_ = ref["grad"][off:off + k].double(), d["grad"][off:off + k].double()
                s_ = grp.setdefault(key, [0.0, 0.0]); s_[0] += float(((a_ - b_) ** 2).sum()); s_[1] += float((a_ ** 2).sum())
                off += k
            row["grad_rel_l2_by_group"] = {k: (v[0] / max(v[1], 1e-30)) ** 0.5 for k, v in grp.items()}
        out.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("loss_ref", "loss")}))
    return out


# ----------------------------------------------------------------------------- levers (monkeypatches)
def patch_fused_qkv(M):
    """One GEMM for the three frozen base projections (q.base, k, v.base concatenated), LoRA added after.
    Parameters and state_dict keys are untouched."""
    def fwd(self, x, rope=None, rope_offset: int = 0):
        B, N, C = x.shape
        assert self.q.base.bias is None and self.k.bias is None and self.v.base.bias is None
        W = torch.cat([self.q.base.weight, self.k.weight, self.v.base.weight], 0)
        q, k, v = F.linear(x, W).split(C, -1)
        q = q + F.linear(F.linear(x, self.q.A), self.q.B) * self.q.scale
        v = v + F.linear(F.linear(x, self.v.A), self.v.B) * self.v.scale
        q = q.reshape(B, N, self.h, self.dh).transpose(1, 2)
        k = k.reshape(B, N, self.h, self.dh).transpose(1, 2)
        v = v.reshape(B, N, self.h, self.dh).transpose(1, 2)
        if rope is not None:
            cos, sin = rope
            q = M.apply_rope(q, cos, sin, rope_offset)
            k = M.apply_rope(k, cos, sin, rope_offset)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))
    M.Attention.forward = fwd


def patch_lora_from(M, K):
    """RECIPE-CHANGING: run blocks[:K] without autograd -> no backward (and no stored activations) below block K."""
    def fwd(self, img):
        B = img.shape[0]
        img = (img - self.img_mean) / self.img_std
        pe = self.patch_embed(img)
        gh, gw = pe.shape[-2], pe.shape[-1]
        x = pe.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.reg_token.expand(B, -1, -1), x], dim=1)
        rope = self._rope(gh, gw, x.device, x.dtype)
        off = 1 + self.N_DINOV3_REG
        with torch.no_grad():
            for b in self.blocks[:K]:
                x = b(x, rope, off)
        for b in self.blocks[K:]:
            x = b(x, rope, off)
        x = self.norm(x)
        return x[:, off:]
    M.VitS16.forward = fwd


def patch_bf16_resid(M):
    """RECIPE-CHANGING (precision): cls/register tokens and LayerScale cast to bf16 so the residual stream is bf16
    (today `torch.cat` promotes the bf16 patch tokens to fp32 and every block adds in fp32)."""
    orig = M.VitS16.forward
    def vfwd(self, img):
        B = img.shape[0]
        img = (img - self.img_mean) / self.img_std
        pe = self.patch_embed(img)
        gh, gw = pe.shape[-2], pe.shape[-1]
        x = pe.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1).to(x.dtype), self.reg_token.expand(B, -1, -1).to(x.dtype), x], dim=1)
        rope = self._rope(gh, gw, x.device, x.dtype)
        off = 1 + self.N_DINOV3_REG
        for b in self.blocks:
            x = b(x, rope, off)
        x = self.norm(x)
        return x[:, off:]
    def bfwd(self, x, rope=None, rope_offset: int = 0):
        x = x + self.gamma_1.to(x.dtype) * self.attn(self.n1(x), rope, rope_offset).to(x.dtype)
        return x + self.gamma_2.to(x.dtype) * self.mlp(self.n2(x)).to(x.dtype)
    M.VitS16.forward = vfwd
    M.Block.forward = bfwd


def patch_n2_once(M):
    """RegisterCompress computes LayerNorm n2(visual) twice on the same tensor; compute it once."""
    def fwd(self, reg, visual):
        h = self.n1(reg)
        nv = self.n2(visual)
        reg = reg + self.cross(h, nv, nv, need_weights=False)[0]
        return reg + self.mlp(self.n3(reg))
    M.RegisterCompress.forward = fwd


# ----------------------------------------------------------------------------- inputs
def make_rig_pool(model, n=30, seed=1234):
    base = np.array(model.baked_calib(), dtype=np.float64)          # [4, 16]
    g = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        r = base.copy()
        if i:
            r[:, 0:2] *= 1.0 + 0.01 * g.standard_normal((4, 1))     # fx, fy
            r[:, 9:12] += 0.02 * g.standard_normal((4, 3))          # translation
        pool.append(r)
    return pool


def make_sample_pool(n, cfg, rigs, seed):
    """n deterministic samples. Images are k/255 values (what uint8 JPEG pixels / 255 give), kept as uint8
    so the f32 and u8 paths see the SAME pixels."""
    g = torch.Generator().manual_seed(seed)
    S = []
    for i in range(n):
        img8 = torch.randint(0, 256, (cfg.n_cameras, 3, cfg.img_h, cfg.img_w), generator=g, dtype=torch.uint8)
        ego = torch.randn(cfg.ego_dim, generator=g)
        cmd = torch.zeros(cfg.n_commands); cmd[int(torch.randint(0, cfg.n_commands, (1,), generator=g))] = 1.0
        t = torch.linspace(0.2, 4.0, cfg.horizon_steps)
        v = 3.0 + 10.0 * float(torch.rand(1, generator=g))
        tgt = torch.stack([v * t, 0.3 * torch.randn(cfg.horizon_steps, generator=g).cumsum(0) * 0.1,
                           0.02 * torch.randn(cfg.horizon_steps, generator=g).cumsum(0)], -1)
        cal = torch.tensor(rigs[int(torch.randint(0, len(rigs), (1,), generator=g))], dtype=torch.float64)
        lab = torch.rand(cfg.n_proposals, cfg.n_score_components, generator=g)
        S.append((img8, ego, cmd, tgt, cal, lab))
    return S


def collate(samples, img_mode):
    img8 = torch.stack([s[0] for s in samples])
    if img_mode == "f32":
        # train.py:832: cv2 uint8 -> astype(float32) / 255.0 in the worker; the f32 tensor is what is pinned
        img = torch.from_numpy(img8.numpy().astype(np.float32) / 255.0)
    else:
        img = img8
    return (img.pin_memory(), torch.stack([s[1] for s in samples]).pin_memory(),
            torch.stack([s[2] for s in samples]).pin_memory(), torch.stack([s[3] for s in samples]).pin_memory(),
            torch.stack([s[4] for s in samples]), torch.stack([s[5] for s in samples]))


def _ensure_safetensors():
    """System python3 on the pod has no `safetensors`: a minimal reader of the documented format (8-byte LE header
    length, JSON header, raw little-endian tensors) stands in, so nothing is installed into the pod's interpreter."""
    try:
        import safetensors.torch  # noqa: F401
        return
    except ImportError:
        pass
    import struct
    import types
    DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}

    def load_file(path, device="cpu"):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            buf = f.read()
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            out[k] = torch.frombuffer(bytearray(buf[s:e]), dtype=DT[v["dtype"]]).reshape(v["shape"]).clone()
        return out
    pkg = types.ModuleType("safetensors"); sub = types.ModuleType("safetensors.torch")
    sub.load_file = load_file; pkg.torch = sub
    sys.modules["safetensors"] = pkg; sys.modules["safetensors.torch"] = sub
    print("  (safetensors not installed: using bench_p6's minimal reader)")


# ----------------------------------------------------------------------------- main
def main():
    a = parse()
    if a.compare:
        compare(a.compare); return
    sys.path.insert(0, a.refe)
    _ensure_safetensors()
    import model as M
    import load_dinov3 as LD
    import ckpt_io
    if a.fused_qkv:
        patch_fused_qkv(M)
    if a.n2_once:
        patch_n2_once(M)
    if a.lora_from:
        patch_lora_from(M, a.lora_from)
    if a.bf16_resid:
        patch_bf16_resid(M)
    dev = "cuda"
    if not a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = M.REFeConfig.for_backbone("vitl16")
    cfg.nav_mode = "command"
    cfg.detach_scorer_context = False          # --scorer-sees-trunk
    if a.res:
        cfg.img_h, cfg.img_w = (int(v) for v in a.res.lower().split("x"))
    torch.manual_seed(a.seed)
    model = M.REFe(cfg)
    if a.weights != "none":
        sd = LD.load_state(a.weights)
        loaded, missing, unused = LD.map_into_backbone(model.backbone, sd)
        assert loaded and not missing and not unused, (len(loaded), missing[:3], unused[:3])
    if a.ckpt:
        st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        n = ckpt_io.load_partial(model, st["model_partial"], (st.get("meta") or {}).get("frozen_sha256"))
        print(f"  loaded {n} tensors from {a.ckpt}")
    model = model.to(dev)
    model.train()
    ck = {} if a.compile_mode == "default" else {"mode": a.compile_mode}
    if a.compile == "backbone":
        model.backbone.compile(**ck)
    elif a.compile == "parts":
        model.backbone.compile(**ck)
        for m in [model.reg_compress, model.pos3d_mlp, model.scene_proj, model.ego_enc, model.traj_head,
                  model.score_q_mlp, model.score_head, *model.dec, *model.score_dec]:
            m.compile()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01, fused=bool(a.fused_adam))
    if a.ckpt and "opt" in st:
        opt.load_state_dict(st["opt"])           # the run's own AdamW moments: the equivalence step is a REAL step-2604 update
        print(f"  optimiser state loaded (lr {opt.param_groups[0]['lr']:.4g})")
    sdpa_ctx = None
    if a.sdpa != "auto":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_ctx = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                    "math": SDPBackend.MATH, "cudnn": SDPBackend.CUDNN_ATTENTION}[a.sdpa]
    t255 = torch.full((), 255.0, device=dev)          # a CUDA 0-dim divisor: TRUE division (a CPU scalar takes x*(1/255))
    rigs = make_rig_pool(model)
    cov_norm, score_w = float(cfg.n_proposals), 0.1
    equiv = bool(a.equiv)
    accum = a.equiv_samples // a.batch if equiv else a.accum
    n_total = accum if equiv else a.warmup + a.reps * a.n_micro
    pool = make_sample_pool(a.equiv_samples if equiv else 8 * a.batch, cfg, rigs, a.seed + 7)

    def batch_for(k):
        if equiv:
            return collate(pool[k * a.batch:(k + 1) * a.batch], a.img)
        j = (k * a.batch) % len(pool)
        return collate([pool[(j + i) % len(pool)] for i in range(a.batch)], a.img)

    batches = [batch_for(k) for k in range(min(n_total, 8 if not equiv else n_total))]
    ev_log = []          # per micro: dict of events
    losses = []
    wta_idx = []                     # the WTA winner per sample: a flip means the trajectory gradient went to ANOTHER proposal
    props_host = torch.empty((a.batch, cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim), dtype=torch.float64).pin_memory()

    def forward(k, E):
        img, ego, cmd, tgt, cal, lab = batches[k % len(batches)]
        if a.phases:
            E["h2d0"] = torch.cuda.Event(enable_timing=True); E["h2d0"].record()
        img = img.to(dev, non_blocking=True)
        if a.img == "u8":
            img = img.float() / t255
        if a.perturb_img:
            img = img * (1.0 + a.perturb_img)
        ego, cmd, tgt = (x.to(dev, non_blocking=True) for x in (ego, cmd, tgt))
        if a.phases:
            E["f0"] = torch.cuda.Event(enable_timing=True); E["f0"].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            traj, score = model(img, ego, cmd, calib=cal)
        traj, score = traj.float(), score.float()
        l_traj, idx, l_pos, l_yaw = M.wta_loss(traj, tgt, yaw_wrap=False, parts=True)
        if a.phases:
            E["f1"] = torch.cuda.Event(enable_timing=True); E["f1"].record()
        return {"k": k, "B": int(img.shape[0]), "score": score, "l_traj": l_traj, "traj": traj, "lab": lab, "E": E, "idx": idx}

    def host_submit(P, early_copy=None):
        # train.py: n_cand = float(cand_m.sum()) (a sync), props .double().cpu().numpy(), sha per sample (submit + re-hash)
        if early_copy is not None:
            early_copy.synchronize()
            props_np = props_host.numpy()[:P["B"]].copy()
        else:
            _ = float(P["traj"].new_zeros(()).sum())
            props_np = P["traj"].detach().double().cpu().numpy()
        P["shas"] = [hashlib.sha256(np.ascontiguousarray(props_np[j], dtype=np.float64).tobytes()).hexdigest()
                     for j in range(P["B"])]
        P["props"] = props_np

    def backward(P):
        shas = [hashlib.sha256(np.ascontiguousarray(P["props"][j], dtype=np.float64).tobytes()).hexdigest()
                for j in range(P["B"])]
        assert shas == P["shas"]
        lab_t = P["lab"].to(dev)
        msk = torch.ones(P["B"], device=dev)
        bce = F.binary_cross_entropy_with_logits(P["score"], lab_t, reduction="none")
        per = bce.mean(-1)
        denom = max(cov_norm * float(P["B"]), 1.0)
        l_score = (per * msk[:, None]).sum() / denom * score_w
        with torch.no_grad():                                  # the same diagnostics (and syncs) as train.py:1736-1750
            _bce_comp = [float(v) for v in ((bce * msk[:, None, None]).sum((0, 1)) / denom)]
            has = msk > 0
            if bool(has.any()):
                sc = torch.sigmoid(P["score"][has])
                pa = sc.mean(-1); ta = lab_t[has].mean(-1)
                pick = ta.gather(1, pa.argmax(1, keepdim=True)).squeeze(1)
                _ = (float(pick.sum()), float(ta.mean(1).sum()), float(ta.max(1).values.sum()), int(has.sum()))
        E = P["E"]
        if a.phases:
            E["b0"] = torch.cuda.Event(enable_timing=True); E["b0"].record()
        tot = P["l_traj"] + l_score
        (tot / accum).backward()
        if a.phases:
            E["b1"] = torch.cuda.Event(enable_timing=True); E["b1"].record()
        if equiv:
            losses.append([float(P["l_traj"]), float(l_score)])
            wta_idx.append(P["idx"].tolist())

    # backbone output hook -> trunk/heads split in both passes
    if a.phases:
        cur = {}

        @torch.compiler.disable                  # the compiled backbone would otherwise trace (and recompile) the hook
        def fhook(_m, _i, out):
            e = torch.cuda.Event(enable_timing=True); e.record(); cur["E"]["trunk_end"] = e
            E = cur["E"]
            def ghook(g):
                e2 = torch.cuda.Event(enable_timing=True); e2.record(); E["bheads_end"] = e2
            out.register_hook(ghook)
        model.backbone.register_forward_hook(fhook)

    prof = None
    step_t = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ctx = torch.nn.attention.sdpa_kernel(sdpa_ctx) if sdpa_ctx is not None else None
    if ctx is not None:
        ctx.__enter__()
    pend = None
    wall = []
    t_rep = None
    micro = 0
    for k in range(n_total):
        if not equiv and k >= a.warmup and (k - a.warmup) % a.n_micro == 0:
            torch.cuda.synchronize(); t_rep = time.perf_counter()
        if a.profile and k == a.warmup:
            from torch.profiler import profile, ProfilerActivity
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof.__enter__()
        E = {}
        if a.phases:
            cur["E"] = E                     # the backbone hook writes trunk_end / bheads_end into THIS micro's dict
        Pk = forward(k, E)
        if a.order == "early":
            # copy props(k) to pinned host memory NOW (enqueued before B(k-1)), launch B(k-1), then wait on the copy
            props_host[:Pk["B"]].copy_(Pk["traj"].detach().double(), non_blocking=True)
            cp = torch.cuda.Event(); cp.record()
            done = [pend] if pend is not None else []
            for P in done:
                backward(P)
            host_submit(Pk, early_copy=cp)
            pend = Pk
            if (micro + 1) % accum == 0:
                backward(pend); pend = None
        else:
            host_submit(Pk)
            done = [pend] if pend is not None else []
            pend = Pk
            if (micro + 1) % accum == 0:
                done.append(pend); pend = None
            for P in done:
                backward(P)
        ev_log.append(Pk["E"])
        micro += 1
        if micro % accum == 0:
            if equiv:
                g = torch.cat([p.grad.detach().float().flatten() for p in params])
                before = torch.cat([p.detach().float().flatten() for p in params]).clone()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if equiv:
                after = torch.cat([p.detach().float().flatten() for p in params])
                names = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
                torch.save({"grad": g.cpu(), "delta": (after - before).cpu(), "losses": losses, "gnorm": float(gn),
                            "wta_idx": wta_idx, "names": names, "args": vars(a)}, a.equiv)
                print(f"  EQUIV dump -> {a.equiv}  gnorm {float(gn):.6g}  losses {losses}")
        if a.profile and k == a.warmup + 1:
            torch.cuda.synchronize(); prof.__exit__(None, None, None)
            ka = prof.key_averages()
            tab = ""
            for key in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
                try:
                    tab = ka.table(sort_by=key, row_limit=60); break
                except Exception:  # noqa: BLE001 -- the sort-key name moved across torch versions
                    continue
            # kernel-class totals (self device time), so the residual after GEMM + attention is a number, not a guess
            cls = {}
            for e in ka:
                t_us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0
                if not t_us:
                    continue
                n = e.key.lower()
                c = ("attention" if ("flash" in n or "fmha" in n or "attention" in n) else
                     "gemm" if any(x in n for x in ("gemm", "sm80_xmma", "cutlass", "ampere_bf16", "ampere_s16816", "mm_", "triton_tem")) else
                     "triton_fused" if n.startswith("triton_") else
                     "copy/cast" if any(x in n for x in ("copy", "memcpy", "cast", "to_copy")) else
                     "reduce/norm" if any(x in n for x in ("norm", "reduce", "softmax")) else "other")
                if e.device_type is not None and "cuda" in str(e.device_type).lower():
                    cls[c] = cls.get(c, 0.0) + t_us / 1e3
            tab += chr(10) * 2 + "KERNEL CLASSES (self device ms over the 2 profiled micro-batches): " + json.dumps(cls)
            open(a.profile, "w").write(tab)
            print(f"  profile -> {a.profile}")
        if not equiv and k >= a.warmup and (k - a.warmup) % a.n_micro == a.n_micro - 1:
            torch.cuda.synchronize(); wall.append((time.perf_counter() - t_rep) / a.n_micro)
    if ctx is not None:
        ctx.__exit__(None, None, None)
    torch.cuda.synchronize()
    if equiv:
        return
    # optimiser step cost (once per accum micro-batches): time 5 clip+step on live grads
    for p in params:
        p.grad = torch.zeros_like(p)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    torch.cuda.synchronize(); t_opt = (time.perf_counter() - t0) / 5
    res = {"tag": a.tag, "torch": torch.__version__, "batch": a.batch, "compile": a.compile, "compile_mode": a.compile_mode,
           "sdpa": a.sdpa, "order": a.order, "img": a.img, "fused_qkv": a.fused_qkv, "n2_once": a.n2_once,
           "fused_adam": a.fused_adam, "lora_from": a.lora_from, "res": a.res, "bf16_resid": a.bf16_resid, "s_per_micro_reps": wall, "s_per_micro": float(np.median(wall)),
           "s_per_sample": float(np.median(wall)) / a.batch, "opt_step_s": t_opt,
           "s_per_step_256": float(np.median(wall)) / a.batch * 256 + t_opt,
           "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    if a.phases:
        keys = [("h2d", "h2d0", "f0"), ("f_trunk", "f0", "trunk_end"), ("f_heads_loss", "trunk_end", "f1"),
                ("b_heads", "b0", "bheads_end"), ("b_trunk", "bheads_end", "b1")]
        agg = {n: [] for n, _, _ in keys}
        # backward of micro k-1 is recorded into k-1's own dict (b0, bheads_end, b1)
        for E in ev_log[a.warmup:-1]:
            for n, s, e in keys:
                if s in E and e in E:
                    agg[n].append(E[s].elapsed_time(E[e]) / 1e3)
        res["phases_s"] = {n: float(np.median(v)) if v else None for n, v in agg.items()}
        # true GPU idle between F(k) end and B(k-1) start: f1(k) -> b0(k-1)
        gaps = []
        for E0, E1 in zip(ev_log[a.warmup:-1], ev_log[a.warmup + 1:]):
            if "b0" in E0 and "f1" in E1:
                gaps.append(E1["f1"].elapsed_time(E0["b0"]) / 1e3)
        res["phases_s"]["gap_F(k)end_to_B(k-1)start"] = float(np.median(gaps)) if gaps else None
    print(json.dumps(res))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
