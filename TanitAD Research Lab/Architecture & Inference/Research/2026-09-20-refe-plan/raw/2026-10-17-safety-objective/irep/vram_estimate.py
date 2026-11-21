#!/usr/bin/env python3
"""vram_estimate.py: how much GPU memory does ONE forward of the navtest planner (ViT-L, 4 cameras, 512x960, fp32, no_grad) need at eval batch B?   CPU only, nothing allocated.

    "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe" -I -B -X utf8 vram_estimate.py [--batches 1,8,32] > logs/vram_estimate.log

METHOD.  The REAL `REFe` forward (C:/fp/PK/refe/model.py, md5-pinned below: the file the free-point worker runs) is executed on the META device: every tensor has its real shape, dtype and
lifetime but no storage, so the whole forward runs in seconds on any machine.  A `TorchDispatchMode` counts the bytes of every op output, de-duplicated by storage (a view is not a new
allocation) and released when the last Python reference to the storage dies, which is when the CUDA caching allocator would reuse the block.  The peak of that live-byte count is the
peak the allocator must hold, up to rounding and fragmentation.  Two CONTROLS run first and must read their analytic values EXACTLY, or the script exits 2 without printing a result:
    C1  a hand-computable chain of matmuls (peak = three live tensors)           C2  the tracker must see a freed tensor leave (live bytes after a `del` = before - nbytes)
THE ATTENTION BACKEND IS A SWITCH, AND BOTH SETTINGS ARE PRINTED.  On the meta device `F.scaled_dot_product_attention` runs PyTorch's MATH decomposition (it materialises the [images, 16, N, N]
score matrix and a softmax of it: the first run of this script peaked inside `_safe_softmax`).  On CUDA, fp32 with head_dim 64 and no mask is eligible for the MEMORY-EFFICIENT kernel, which
allocates only its output.  So: `sdpa=efficient` replaces the call by an allocation of exactly the efficient kernel's output (a [B, N, H, dh] block seen as [B, H, N, dh]); `sdpa=math` leaves the real
decomposition in.  Which one the dev-box GPU takes is NOT measured here (no GPU is touched); the PLAN argues it from the package's own training-memory measurement.
WHAT IT DOES NOT COVER (stated, not hidden): the CUDA context (about 0.3-0.5 GB, ESTIMATED), allocator rounding / fragmentation (ESTIMATED +15 %), and cuBLAS / cuDNN workspaces.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import weakref

sys.dont_write_bytecode = True
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_leaves  # noqa: E402

REFE_DIR = "C:/fp/PK/refe"
MODEL_SHA16 = "e6011cd83c846dfb"                       # model.py, the hash fp_ship.MODEL_HASHES pins on the pod
GB = 1e9


class Tracker(TorchDispatchMode):
    """live bytes of op outputs, storage-deduplicated"""

    def __init__(self):
        super().__init__()
        self.live = 0
        self.peak = 0
        self.peak_op = None
        self.peak_live = []                           # the biggest live storages (bytes) at the moment of the peak
        self.st = {}                                  # storage id -> [nbytes, live tensor objects holding it]
        self.by_op = {}

    def _release(self, sid):
        rec = self.st.get(sid)
        if rec is None:
            return
        rec[1] -= 1
        if rec[1] <= 0:
            self.live -= rec[0]
            del self.st[sid]

    def register(self, t, op="input"):
        if not isinstance(t, torch.Tensor) or t.device.type != "meta":
            return
        sid = t.untyped_storage()._cdata
        rec = self.st.get(sid)
        if rec is None:
            nb = t.untyped_storage().nbytes()
            self.st[sid] = [nb, 1]
            self.live += nb
            if self.live > self.peak:
                self.peak, self.peak_op = self.live, op
                self.peak_live = sorted((r[0] for r in self.st.values()), reverse=True)[:12]
        else:
            rec[1] += 1
        weakref.finalize(t, self._release, sid)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            self.register(t, str(func))
        return out


def controls() -> dict:
    n = 1024                                          # 1024 x 1024 fp32 = 4 MiB
    nb = n * n * 4
    tr = Tracker()
    x = torch.empty(n, n, device="meta")
    tr.register(x)
    with tr:
        y = x @ x                                      # live: x, y
        z = y @ y                                      # live: x, y, z  <- the peak: 3 tensors
        del y                                          # live: x, z
        after_del = tr.live
        w = z + 1                                      # live: x, z, w (3 again, not more)
    c1 = tr.peak == 3 * nb
    c2 = after_del == 2 * nb
    return {"C1_peak_is_three_tensors": c1, "C1_peak_bytes": tr.peak, "C1_expected": 3 * nb, "C2_freed_tensor_leaves": c2, "C2_live_after_del": after_del, "C2_expected": 2 * nb,
            "ok": bool(c1 and c2), "_keep": (x, z, w)}


def fake_efficient_sdpa(q, k, v, *a, **kw):
    """what the memory-efficient forward kernel allocates: its output, laid out [B, N, H, dh] and viewed [B, H, N, dh] (no score matrix, no logsumexp at inference)"""
    return torch.empty(q.shape[0], q.shape[2], q.shape[1], q.shape[3], device=q.device, dtype=q.dtype).transpose(1, 2)


def one_batch(B: int, sdpa: str, rigs: str = "1") -> dict:
    sys.path.insert(0, REFE_DIR)
    from model import REFe, REFeConfig
    real_sdpa = torch.nn.functional.scaled_dot_product_attention
    if sdpa == "efficient":
        torch.nn.functional.scaled_dot_product_attention = fake_efficient_sdpa
    try:
        return _one_batch(B, sdpa, rigs, REFe, REFeConfig)
    finally:
        torch.nn.functional.scaled_dot_product_attention = real_sdpa


def _one_batch(B: int, sdpa: str, rigs: str, REFe, REFeConfig) -> dict:
    cfg = REFeConfig.for_backbone("vitl16")
    cfg.n_score_components = 7                          # the u400 checkpoint's config (ckpt_io.config_for_checkpoint)
    with torch.device("meta"):
        model = REFe(cfg).eval()
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers())
    img = torch.empty(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, device="meta")          # the device copy of the images: alive for the whole call, as the argument of model(...)
    ego = torch.empty(B, cfg.ego_dim, device="meta")
    goal = torch.empty(B, 2 * cfg.n_goal_points, device="meta")
    calib = torch.zeros(B, cfg.n_cameras, 16, dtype=torch.float64)                       # a real CPU tensor: `_pos3d` reads it with .tolist()
    calib[:, :, 0] = calib[:, :, 1] = 1545.0
    calib[:, :, 2], calib[:, :, 3], calib[:, :, 12] = 960.0, 560.0, 1.0
    if rigs == "B":                                       # a shuffled I-rep batch mixes logs: `_pos3d` then builds one frustum + one pos3d_mlp row-block per DISTINCT rig in the batch
        calib[:, :, 4] = torch.arange(B, dtype=torch.float64).view(B, 1) * 1e-3      # k1 differs per sample => B distinct rigs (the worst case)
    tr = Tracker()
    for t in (img, ego, goal):
        tr.register(t, "input")
    with torch.no_grad(), tr:
        traj, score = model(img, ego, goal, calib=calib)
    n_img = B * cfg.n_cameras
    n_tok = (cfg.img_h // cfg.patch) * (cfg.img_w // cfg.patch) + 1 + 4
    mlp_hidden = n_img * n_tok * int(cfg.width * cfg.mlp_ratio) * 4
    stream = n_img * n_tok * cfg.width * 4
    # the analytic bound to read the measured peak against: weights + the images + (the residual stream, its LayerNorm, the MLP hidden, the GELU output) alive together in one block
    analytic = wbytes + img.numel() * 4 + 2 * stream + 2 * mlp_hidden
    return {"B": B, "sdpa": sdpa, "rigs": rigs, "images": n_img, "tokens_per_image": n_tok, "weights_GB": wbytes / GB, "activation_peak_GB": tr.peak / GB, "peak_op": tr.peak_op,
            "model_peak_GB": (tr.peak + wbytes) / GB, "stream_GB": stream / GB, "mlp_hidden_GB": mlp_hidden / GB, "analytic_bound_GB": analytic / GB,
            "estimated_nvidia_smi_GB": (tr.peak + wbytes) * 1.15 / GB + 0.4, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "out_shapes": [tuple(traj.shape), tuple(score.shape)], "peak_live_GB": [round(b / GB, 3) for b in tr.peak_live]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32")
    a = ap.parse_args()
    m16 = hashlib.sha256(open(f"{REFE_DIR}/model.py", "rb").read()).hexdigest()[:16]
    print(f"torch {torch.__version__}; model.py sha256[:16] {m16} (pinned {MODEL_SHA16}): {'OK' if m16 == MODEL_SHA16 else 'MISMATCH'}")
    if m16 != MODEL_SHA16:
        return 2
    c = controls()
    print({k: v for k, v in c.items() if k != "_keep"})
    if not c["ok"]:
        print("ZZVRAM_CONTROL_FAILED: the tracker does not read its analytic controls; no result is printed")
        return 2
    rows = [one_batch(int(b), sd, rg) for sd in ("efficient", "math") for rg in ("1", "B") for b in a.batches.split(",")]
    print(f"{'sdpa':>9s} {'rigs':>4s} {'B':>3s} {'images':>6s} {'weights':>8s} {'act.peak':>9s} {'model.peak':>10s} {'analytic':>9s} {'~nvidia-smi':>11s}   (GB, 1e9 B)   peak op")
    for r in rows:
        print(f"{r['sdpa']:>9s} {r['rigs']:>4s} {r['B']:3d} {r['images']:6d} {r['weights_GB']:8.2f} {r['activation_peak_GB']:9.2f} {r['model_peak_GB']:10.2f} {r['analytic_bound_GB']:9.2f} "
              f"{r['estimated_nvidia_smi_GB']:11.2f}   {r['peak_op']}")
    for r in [r for r in rows if r['sdpa'] == 'efficient' and r['rigs'] == '1']:
        print(f"B={r['B']}: the 12 biggest live tensors at the peak (GB): {r['peak_live_GB']}  (sum of all live at the peak {r['activation_peak_GB']:.2f}, + weights {r['weights_GB']:.2f})")
        print(f"B={r['B']}: params {r['params_M']:.2f} M, {r['tokens_per_image']} tokens/image, residual stream {r['stream_GB']:.2f} GB, MLP hidden {r['mlp_hidden_GB']:.2f} GB, outputs {r['out_shapes']}")
    print("ZZVRAM_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
