#!/usr/bin/env python3
"""P4 - the BEV TRANSFORMER head on FROZEN refcv5-v2 trunk tokens, supervised by LiDAR BEV.

Pre-registration: `PREREG_BEVHEAD_FROZEN_TRUNK.md` (`E-BEVHEAD-FROZEN-1`), written before
any arm ran. This script trains ONE arm per invocation; every trained arm is the same
command with exactly one differing lever token:

  --arm main       tokens=s32, seed 0
  --arm main_s1    tokens=s32, seed 1                 (training-variance floor)
  --arm shuffled   tokens=s32, targets permuted       (zero image<->target information)
  --arm pixel      tokens=pix64                       (raw-pixel floor)
  (--arm s16       tokens=s16 [352,16,40], frozen   -- pre-registered lever L1, post-hoc row)
  (--arm s32ft     cached s16 -> UNFROZEN last ResNet stage (51,562,368 params, BN stats
                   frozen) -> head                  -- pre-registered lever L2, post-hoc row)

⛔ VISION-ONLY INPUT: trunk tokens or pixels of the D-015 frame stack. LiDAR is the LABEL.

Outputs (per arm, local): `runs/<arm>/{config.json, log.jsonl, best.pt, test_probs.npy,
val_probs.npy}`; the eval (`p4_eval.py`) reads them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

#: MEASURED 2026-09-13 on the RTX 4060 (torch 2.11+cu128, Windows): flash attention is
#: UNAVAILABLE; for [32, 6, 1920, 32] bf16 fwd+bwd the math kernel pages at 11.56 GB,
#: memory-efficient runs 0.204 s, cuDNN 0.035 s; the full head step 0.278 -> 0.220 s.
#: Kernel choice changes speed, not the function (loss identical at 4 dp in the probe).
ATTN_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bev_gt_loader as L  # noqa: E402

WORK = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913")
TOK_DIR = Path(os.environ.get("BEVHEAD_TOK_DIR", str(WORK / "tokens")))
GT_DIR = WORK / "bev_gt"
N_RNG, N_AZ = 48, 40


# ---------------------------------------------------------------------------
# positional encodings (fixed, 0 parameters)
# ---------------------------------------------------------------------------
def sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """[h*w, d] 2-D sine-cosine PE: first half encodes the row, second half the column."""
    assert d % 4 == 0
    def enc(n, dd):
        pos = torch.arange(n, dtype=torch.float32)[:, None]
        k = torch.arange(dd // 2, dtype=torch.float32)[None, :]
        ang = pos / (10000.0 ** (2 * k / dd))
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)       # [n, dd]
    er = enc(h, d // 2)[:, None, :].expand(h, w, d // 2)
    ec = enc(w, d // 2)[None, :, :].expand(h, w, d // 2)
    return torch.cat([er, ec], dim=-1).reshape(h * w, d)


class BEVCrossAttnHead(nn.Module):
    """Variant B of 09-11 `p4_head_sizing.py` (`B_xattn_polar48x40_d192_L3`), made TRAINABLE:
    random-init queries + fixed PEs on both sides (the sizing class had zero queries and no
    PE -- every query identical, every token unaddressed)."""

    def __init__(self, d_in: int, grid_hw=(8, 20), bev_hw=(N_RNG, N_AZ), d=192, n_layers=3,
                 n_heads=6, d_ff=768, dropout=0.1):
        super().__init__()
        self.bev_hw = bev_hw
        self.q = nn.Parameter(torch.randn(bev_hw[0] * bev_hw[1], d) * 0.02)
        self.register_buffer("q_pe", sincos_2d(*bev_hw, d), persistent=False)
        self.kv_proj = nn.Linear(d_in, d)
        self.register_buffer("kv_pe", sincos_2d(*grid_hw, d), persistent=False)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:          # tok [B, d_in, gh, gw]
        b = tok.shape[0]
        kv = self.kv_proj(tok.flatten(2).transpose(1, 2)) + self.kv_pe[None]
        x = (self.q + self.q_pe)[None].expand(b, -1, -1)
        mm = getattr(self, "memory_mask", None)
        for lyr in self.layers:
            x = lyr(x, kv, memory_mask=mm) if mm is not None else lyr(x, kv)
        return self.out(self.norm(x)).reshape(b, *self.bev_hw)       # logits [B, 48, 40]


def geo_memory_mask(grid_hw, bev_hw=(N_RNG, N_AZ), tol_deg: float = 9.0) -> torch.Tensor:
    """Lever L3: [n_queries, n_tokens] bool, True = BLOCKED. A polar query may attend only to
    image-token COLUMNS whose azimuth band lies within `tol_deg` of where its cell centre
    projects through the NOMINAL rig camera (median per-clip front-wide extrinsics of the
    B1 EVAL join), at z = 1.0 m.

    ⭐ Exact because the corpus is CYLINDRICAL: image column is linear in camera azimuth
    (`rig_projection.project_cam_to_frame`: col = (W-1)/2 + f*phi), so a token column is an
    azimuth band of exactly 120 deg / grid_w. Rows are NOT restricted (range is what the head
    must infer). A query whose cell does not project into the frame keeps ALL tokens (it is
    never scored; a fully blocked row would make the softmax NaN)."""
    import p2_build_corpus as B
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera
    exts = [B.extrinsics(c, "camera_front_wide_120fov") for c in B.join_clips()]
    med = {k: float(np.median([e[k] for e in exts])) for k in exts[0]}
    frame = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")
    cam = RigCamera.from_extrinsics(FrontWideExtrinsics(**med), frame)
    nr, na = bev_hw
    r = (np.arange(nr) + 0.5) * (60.0 / nr)
    az = np.radians(60.0 - (np.arange(na) + 0.5) * (120.0 / na))
    X = r[:, None] * np.cos(az)[None, :]
    Y = r[:, None] * np.sin(az)[None, :]
    P = torch.as_tensor(np.stack([X, Y, np.ones_like(X)], -1).reshape(-1, 3), dtype=torch.float64)
    col, _row, valid = cam.project(P)
    gh, gw = grid_hw
    px_per_col = 640.0 / gw
    tok_centre = (np.arange(gw) + 0.5) * px_per_col - 0.5            # pixel centre of each token column
    deg_per_px = 120.0 / 640.0
    d = np.abs(col.numpy()[:, None] - tok_centre[None, :]) * deg_per_px  # [Q, gw] degrees
    allow_col = d <= tol_deg + 0.5 * (120.0 / gw)                      # band half-width + tolerance
    allow_col[~valid.numpy()] = True
    allow = np.repeat(allow_col[:, None, :], gh, axis=1).reshape(nr * na, gh * gw)
    return torch.from_numpy(~allow)


class FinetuneLastStage(nn.Module):
    """Lever L2: the refcv5-v2 encoder's LAST ResNet stage (`stages[3]`, 6 BasicBlocks,
    352 -> 704 ch, stride 2), initialised from `ckpt_40284.pt` (STRICT) and UNFROZEN, on the
    cached stride-16 map, followed by the same head. BatchNorm running statistics stay FROZEN
    (eval mode) -- batch 32 over ~87 clips is too small a population to re-estimate them;
    the BN affine parameters train. This changes the trunk: it answers "can the last stage
    be ADAPTED to expose occupancy", not "does the deployed trunk carry it"."""

    CKPT = "C:/Users/Admin/hf-refcv5v2/ckpt_40284.pt"

    def __init__(self):
        super().__init__()
        from tanitad.refs import refc
        cfg = refc.CNNEncoderConfig(in_channels=9, image_size=256, image_width=640,
                                    base_width=88, blocks=(3, 6, 16, 6))
        enc = refc.ResNetEncoder(cfg)
        ck = torch.load(self.CKPT, map_location="cpu", weights_only=False, mmap=True)
        pre = "core.encoder.stages.3."
        sd = {k[len(pre):]: v for k, v in ck["model"].items() if k.startswith(pre)}
        self.stage = enc.stages[3]
        self.stage.load_state_dict(sd, strict=True)
        self.head = BEVCrossAttnHead(d_in=704, grid_hw=(8, 20))
        del ck, enc

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.stage.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, s16: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage(s16))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_panel_data(rule_primary: str = "B"):
    """Rows x targets for every stacked row of every OK clip. Returns a dict of arrays."""
    idx = np.load(TOK_DIR / "index.npz")
    sha = [str(s) for s in idx["clip_sha12"]]
    man = {}
    for line in (GT_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        man[r["clip_sha12"]] = r
    ok = {s for s, r in man.items() if r.get("ok")}
    import p2_build_corpus as B
    N = len(idx["clip_ordinal"])
    occ = np.zeros((N, N_RNG, N_AZ), dtype=bool)
    mB = np.zeros((N, N_RNG, N_AZ), dtype=bool)
    mC = np.zeros((N, N_RNG, N_AZ), dtype=bool)
    row_ok = np.zeros(N, dtype=bool)
    clip_of_row = idx["clip_ordinal"].astype(np.int32)
    raw_frame = idx["raw_frame"].astype(np.int32)
    for ci, s in enumerate(sha):
        rows = np.nonzero(clip_of_row == ci)[0]
        if s not in ok:
            continue
        cb = L.load_clip(GT_DIR / man[s]["artifact"], grid="polar48", verify=True,
                         builder_specs=B.SPECS, zband=B.ZB, rule="B")
        cc = L.load_clip(GT_DIR / man[s]["artifact"], grid="polar48", verify=False, rule="C")
        fr = raw_frame[rows]
        if fr.max() >= cb.occ.shape[0]:
            raise SystemExit(f"row->frame out of range for clip {s}")
        occ[rows] = cb.occ[fr]
        mB[rows] = cb.scored_mask()[fr]
        mC[rows] = cc.scored_mask()[fr]
        row_ok[rows] = cb.label_valid[fr] & mB[rows].reshape(len(rows), -1).any(1)
    # content-blind clip-disjoint split: sorted sha12 position k
    order = sorted(range(len(sha)), key=lambda i: sha[i])
    split_of_clip = np.full(len(sha), -1, dtype=np.int8)    # 0 train, 1 val, 2 test, -1 excluded
    for k, ci in enumerate(order):
        if sha[ci] not in ok:
            continue
        split_of_clip[ci] = 2 if k % 4 == 0 else (1 if k % 8 == 1 else 0)
    split = split_of_clip[clip_of_row]
    return {"occ": occ, "mB": mB, "mC": mC, "row_ok": row_ok, "clip_of_row": clip_of_row,
            "raw_frame": raw_frame, "split": split, "sha": sha, "ok": ok,
            "excluded_clips": [s for s in sha if s not in ok]}


class TokenSource:
    """Batches of VISION tokens from the local memmaps."""

    def __init__(self, kind: str, tok_dir: Path | None = None):
        self.kind = kind
        TOK = TOK_DIR if tok_dir is None else Path(tok_dir)
        if kind == "s32":
            self.mm = np.load(TOK / "tokens_s32_fp16.npy", mmap_mode="r")
            self.d_in, self.grid = 704, (8, 20)
        elif kind == "pix64":
            self.mm = np.load(TOK / "pix64_u8.npy", mmap_mode="r")
            self.d_in, self.grid = 9 * 8 * 8, (8, 20)
        elif kind == "s16":
            self.mm = np.load(TOK / "tokens_s16_fp16.npy", mmap_mode="r")
            self.d_in, self.grid = 352, (16, 40)
        else:
            raise ValueError(kind)

    def get(self, rows: np.ndarray, dev) -> torch.Tensor:
        srt = np.sort(rows)
        inv = np.argsort(np.argsort(rows))
        a = np.asarray(self.mm[srt])[inv]
        t = torch.from_numpy(a).to(dev, non_blocking=True)
        if self.kind == "pix64":
            x = t.float().div_(255.0)                                    # [B, 9, 64, 160]
            x = F.unfold(x, kernel_size=8, stride=8)                     # [B, 576, 160]
            return x.reshape(x.shape[0], 576, 8, 20)
        return t.float()


TOK_DIR_EXTRA = Path(os.environ.get("BEVHEAD_TOK_DIR_EXTRA", str(WORK / "tokens_extra")))
GT_DIR_EXTRA = Path(os.environ.get("BEVHEAD_GT_DIR_EXTRA", str(WORK / "bev_gt_extra")))


class ConcatTokenSource:
    """Rows < n0 come from the panel's token cache, rows >= n0 from the extra-clip cache."""

    def __init__(self, a: "TokenSource", b: "TokenSource", n0: int):
        self.a, self.b, self.n0 = a, b, n0
        self.d_in, self.grid, self.kind = a.d_in, a.grid, a.kind

    def get(self, rows: np.ndarray, dev) -> torch.Tensor:
        rows = np.asarray(rows)
        lo = rows < self.n0
        out = None
        for mask, src, off in ((lo, self.a, 0), (~lo, self.b, self.n0)):
            if mask.any():
                t = src.get(rows[mask] - off, dev)
                if out is None:
                    out = torch.empty((len(rows), *t.shape[1:]), dtype=t.dtype, device=t.device)
                out[torch.from_numpy(np.nonzero(mask)[0]).to(t.device)] = t
        return out


def load_extra_rows():
    """LEVER L4: targets for every stacked row of the extra TRAINING clips (tokens_extra/index.npz)."""
    import p2_build_corpus as B
    idx = np.load(TOK_DIR_EXTRA / "index.npz")
    sha = [str(x) for x in idx["clip_sha12"]]
    man = {}
    for line in (GT_DIR_EXTRA / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        man[r["clip_sha12"]] = r
    N = len(idx["clip_ordinal"])
    occ = np.zeros((N, N_RNG, N_AZ), dtype=bool)
    mB = np.zeros_like(occ)
    mC = np.zeros_like(occ)
    ok = np.zeros(N, dtype=bool)
    cor = idx["clip_ordinal"].astype(np.int32)
    rf = idx["raw_frame"].astype(np.int32)
    for ci, s12 in enumerate(sha):
        if not man.get(s12, {}).get("ok"):
            continue
        rows = np.nonzero(cor == ci)[0]
        cb = L.load_clip(GT_DIR_EXTRA / man[s12]["artifact"], grid="polar48", verify=True,
                         builder_specs=B.SPECS, zband=B.ZB, rule="B")
        cc = L.load_clip(GT_DIR_EXTRA / man[s12]["artifact"], grid="polar48", verify=False, rule="C")
        fr = rf[rows]
        occ[rows] = cb.occ[fr]
        mB[rows] = cb.scored_mask()[fr]
        mC[rows] = cc.scored_mask()[fr]
        ok[rows] = cb.label_valid[fr] & mB[rows].reshape(len(rows), -1).any(1)
    return {"occ": occ, "mB": mB, "mC": mC, "row_ok": ok, "n_clips": len(sha),
            "n_clips_ok": sum(1 for x in sha if man.get(x, {}).get("ok"))}


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------
def predict(model, src, rows, dev, bs=128) -> np.ndarray:
    model.eval()
    out = np.zeros((len(rows), N_RNG, N_AZ), dtype=np.float16)
    with torch.no_grad(), sdpa_kernel(ATTN_BACKENDS), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in range(0, len(rows), bs):
            r = rows[a:a + bs]
            out[a:a + len(r)] = torch.sigmoid(model(src.get(r, dev)).float()).cpu().numpy()
    model.train()
    return out


def ap_of(probs: np.ndarray, occ: np.ndarray, mask: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = occ[mask]
    s = probs[mask].astype(np.float32)
    return float(average_precision_score(y, s)) if y.any() and (~y).any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["main", "main_s1", "shuffled", "pixel", "s16",
                                                     "s32ft", "smoke"])
    ap.add_argument("--lr-trunk", type=float, default=1e-4, help="s32ft only: last-stage lr")
    ap.add_argument("--geo-mask", action="store_true",
                    help="lever L3: azimuth-aligned cross-attention mask (run dir suffix _geo)")
    ap.add_argument("--geo-tol-deg", type=float, default=9.0)
    ap.add_argument("--tag", default="", help="appended to the run dir name (e.g. rb = rebuilt frames)")
    ap.add_argument("--extra-train", action="store_true",
                    help="lever L4: add the P6b extra TRAINING clips to the train split (run dir suffix _data)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the arm's seed; the run dir becomes <arm>_s<seed> (a lever REPLICATE)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=str(WORK / "runs"))
    args = ap.parse_args()

    arm = args.arm
    seed = 1 if arm == "main_s1" else 0
    run_name = arm
    if args.seed is not None and args.seed != seed:
        seed = args.seed
        run_name = f"{arm}_s{seed}"
    kind = {"main": "s32", "main_s1": "s32", "shuffled": "s32", "pixel": "pix64",
            "s16": "s16", "s32ft": "s16", "smoke": "s32"}[arm]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda")
    if args.extra_train:
        run_name = arm + "_data" + ("" if args.seed is None or args.seed == 0 else f"_s{args.seed}")
    if args.geo_mask:
        run_name = run_name + "_geo"
    if args.tag:
        run_name = run_name + "_" + args.tag
    out = Path(args.out) / run_name
    out.mkdir(parents=True, exist_ok=True)

    D = load_panel_data()
    tr = np.nonzero(D["row_ok"] & (D["split"] == 0))[0]
    va = np.nonzero(D["row_ok"] & (D["split"] == 1))[0]
    te = np.nonzero(D["row_ok"] & (D["split"] == 2))[0]
    n0 = len(D["occ"])
    n_extra_rows = 0
    if args.extra_train:
        Xr = load_extra_rows()
        D["occ"] = np.concatenate([D["occ"], Xr["occ"]])
        D["mB"] = np.concatenate([D["mB"], Xr["mB"]])
        D["mC"] = np.concatenate([D["mC"], Xr["mC"]])
        extra = n0 + np.nonzero(Xr["row_ok"])[0]
        n_extra_rows = int(len(extra))
        tr = np.concatenate([tr, extra])
        D["clip_of_row"] = np.concatenate([D["clip_of_row"], np.full(len(Xr["occ"]), -1, np.int32)])
    occ_t = torch.from_numpy(D["occ"])
    msk_t = torch.from_numpy(D["mB"])

    # the SHUFFLED arm: a fixed DERANGEMENT of (target, mask) over train rows. Masks travel
    # with their targets, so the scored-cell statistics are unchanged; only the image<->
    # target correspondence is destroyed.
    tgt_row = np.arange(len(D["occ"]))
    if arm == "shuffled":
        perm = rng.permutation(len(tr))
        fixed = np.nonzero(perm == np.arange(len(tr)))[0]
        for f in fixed:                                   # break every fixed point
            j = (f + 1) % len(tr)
            perm[f], perm[j] = perm[j], perm[f]
        assert not np.any(perm == np.arange(len(tr))), "not a derangement"
        tgt_row[tr] = tr[perm]

    src = TokenSource(kind)
    if args.extra_train:
        src = ConcatTokenSource(src, TokenSource(kind, TOK_DIR_EXTRA), n0)
    if arm == "s32ft":
        model = FinetuneLastStage().to(dev)
        # ⛔ modules START in training mode; without this call the stage's BatchNorm would use
        # batch statistics and UPDATE its running stats until the first predict() re-entered
        # train() at step 500 -- the "BN frozen" claim would have been false for 500 steps.
        model.train()
        assert all(not m.training for m in model.stage.modules() if isinstance(m, nn.BatchNorm2d))
        n_params = sum(p.numel() for p in model.head.parameters())
        n_trunk_trainable = sum(p.numel() for p in model.stage.parameters())
        opt = torch.optim.AdamW([{"params": list(model.stage.parameters()), "lr": args.lr_trunk},
                                 {"params": list(model.head.parameters()), "lr": args.lr}],
                                weight_decay=args.wd, fused=True)
        opt.param_groups[0]["base"] = args.lr_trunk
        opt.param_groups[1]["base"] = args.lr
    else:
        model = BEVCrossAttnHead(d_in=src.d_in, grid_hw=src.grid).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        n_trunk_trainable = 0
        if args.geo_mask:
            model.register_buffer("memory_mask", geo_memory_mask(src.grid, tol_deg=args.geo_tol_deg).to(dev),
                                  persistent=False)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True)
        for g in opt.param_groups:
            g["base"] = args.lr

    def lr_at(step, base=None):
        base = args.lr if base is None else base
        if step < args.warmup:
            return base * (step + 1) / args.warmup
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    cfg = {"arm": arm, "run_name": run_name, "seed": seed, "tokens": kind, "tok_dir": str(TOK_DIR),
           "geo_mask": bool(args.geo_mask), "geo_tol_deg": args.geo_tol_deg if args.geo_mask else None, "d_in": src.d_in, "grid": list(src.grid),
           "head_params": n_params, "trunk_params_trainable": n_trunk_trainable,
           "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)),
           "n_test_rows": int(len(te)),
           "n_train_clips": int(len(np.unique(D["clip_of_row"][tr][D["clip_of_row"][tr] >= 0]))),
           "extra_train": bool(args.extra_train), "n_extra_train_rows": n_extra_rows,
           "n_val_clips": int(len(np.unique(D["clip_of_row"][va]))),
           "n_test_clips": int(len(np.unique(D["clip_of_row"][te]))),
           "excluded_clips_sha12": D["excluded_clips"], "argv": sys.argv[1:],
           "train_marginal_B": float(D["occ"][tr][D["mB"][tr]].mean()),
           "evidence_class": "MEASURED (ours, dev-box RTX 4060)"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(json.dumps(cfg), flush=True)

    logf = open(out / "log.jsonl", "w", encoding="utf-8")
    best = (-1.0, -1)
    t0 = time.time()
    order = rng.permutation(tr)
    pos = 0
    for step in range(args.steps):
        if pos + args.batch > len(order):
            order = rng.permutation(tr)
            pos = 0
        rows = order[pos:pos + args.batch]
        pos += args.batch
        x = src.get(rows, dev)
        y = occ_t[tgt_row[rows]].to(dev, non_blocking=True).float()
        m = msk_t[tgt_row[rows]].to(dev, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, g["base"])
        with sdpa_kernel(ATTN_BACKENDS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float()[m], y[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            rec = {"step": step, "loss": round(float(loss.detach()), 5), "lr": lr_at(step),
                   "s_per_step": round((time.time() - t0) / (step + 1), 4)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            pv = predict(model, src, va, dev)
            vap = ap_of(pv, D["occ"][va], D["mB"][va])
            rec = {"step": step + 1, "val_ap_B": round(vap, 5), "loss": round(float(loss.detach()), 5),
                   "elapsed_s": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[p4] {arm} {rec}", flush=True)
            if vap > best[0]:
                best = (vap, step + 1)
                torch.save(model.state_dict(), out / "best.pt")
    logf.close()

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    pv = predict(model, src, va, dev)
    pt = predict(model, src, te, dev)
    np.save(out / "val_probs.npy", pv)
    np.save(out / "test_probs.npy", pt)
    np.savez(out / "rows.npz", train=tr, val=va, test=te, tgt_row_train=tgt_row[tr])
    cfg.update({"best_val_ap_B": best[0], "best_step": best[1],
                "wall_s": round(time.time() - t0, 1),
                "test_ap_B_quick": ap_of(pt, D["occ"][te], D["mB"][te])})
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"[p4] {arm} DONE best val AP {best[0]:.4f} @ {best[1]}  wall {cfg['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
