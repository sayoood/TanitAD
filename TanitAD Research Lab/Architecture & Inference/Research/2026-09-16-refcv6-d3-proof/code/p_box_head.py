#!/usr/bin/env python3
"""RUNG 2 -- the BOX head on FROZEN refcv5-v2 trunk tokens, supervised by the AGENT JOIN.

    python p_box_head.py --arm main --tokens s32 --seed 0 --tag main_s32

⛔ VISION-ONLY INPUT: cached frozen tokens (or raw pixels), never the join. The trunk is
NOT touched -- the tokens are a memmap on disk. `BEVHEAD_TOK_DIR` selects the token dir,
the same env var `p4_bev_head.py` uses, and `TokenSource` is that file's class, imported
rather than re-written.

TARGET. `b1eval_agents.jsonl.xz` (md5 3ddb42ecbd3926066795a94587af2aed), per-frame EGO
frame +x fwd / +y left, `occ == 0` = the centre is inside the 120 deg front field. Scored
at 0-30 m ahead, |y| <= 20 m, vehicle classes only.

HEAD. A centre HEATMAP (15 x 20 cells, 2 m) + `--n-slots` DETR slots over the same tokens,
plus one lead-presence logit. Hungarian matching; focal BCE on the heatmap.

ARMS (exactly one lever per invocation):
  main      the tokens, as they are
  shuffled  targets PERMUTED across rows at TRAINING time, scored against the TRUE targets
            -- zero image<->target information
  pixel     `--tokens pix64`
  mirror    the token grid FLIPPED left-right at train AND test (the
            `R-2026-09-08-wpa-mirror` class) -- it MUST LOSE

SPLIT: clip-disjoint and CONTENT-BLIND -- `p4_bev_head.load_panel_data`'s own rule, sorted
sha12 position k: test if k % 4 == 0, val if k % 8 == 1, else train. ⛔ Clips, never frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
P4 = Path(os.environ.get(
    "D3_P4_DIR",
    r"C:\Users\Admin\tanitad-snap-20260915\TanitAD Research Lab\Architecture & Inference"
    r"\Research\2026-09-13-bev-lidar-corpus-and-head\code"))
sys.path.insert(0, str(P4))
sys.path.insert(0, str(HERE))

TOK_DIR = Path(os.environ.get("BEVHEAD_TOK_DIR",
                              r"C:\Users\Admin\tanitad-caches\bevhead-20260913\tokens"))
JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"
VEHICLE = {"automobile", "heavy_truck", "trailer", "bus", "other_vehicle"}
X_MAX, Y_ABS, CELL = 30.0, 20.0, 2.0
#: 15 x 20 = 300 heatmap queries. ⛔ 1 m cells (30 x 40 = 1200 queries) put the
#: decoder's SELF-attention at 24 x 6 x 1200^2 per layer and paged the 8 GB card;
#: the heatmap is an AUXILIARY dense target -- every AP number comes from the SLOTS,
#: whose centres are continuous and unaffected by the cell size.
GX, GY = int(X_MAX / CELL), int(2 * Y_ABS / CELL)          # 15 x 20
LEAD_LAT_M, MAX_BOXES = 2.0, 24


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Identical to `p4_bev_head.sincos_2d` (imported there; duplicated only if absent)."""
    import p4_bev_head as P
    return P.sincos_2d(h, w, d)


# --------------------------------------------------------------------------- #
# targets                                                                      #
# --------------------------------------------------------------------------- #
def build_targets(idx) -> dict:
    """Per stacked row: up to MAX_BOXES (x, y, l, w, yaw) in the 0-30 m front box."""
    sha = [str(s) for s in idx["clip_sha12"]]
    by_sha = {s: i for i, s in enumerate(sha)}
    cor = idx["clip_ordinal"].astype(np.int32)
    raw = idx["raw_frame"].astype(np.int32)
    key = {}
    for r in range(len(cor)):
        key[(int(cor[r]), int(raw[r]))] = r
    N = len(cor)
    boxes = np.zeros((N, MAX_BOXES, 5), dtype=np.float32)
    nbox = np.zeros(N, dtype=np.int16)
    has_row = np.zeros(N, dtype=bool)
    n_lines = 0
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ci = by_sha.get(sha12(d["clip_id"]))
            if ci is None:
                continue
            r = key.get((ci, int(d["frame"])))
            if r is None:
                continue
            n_lines += 1
            has_row[r] = True
            keep = []
            for a in d["agents"]:
                if str(a.get("cls", "")) not in VEHICLE:
                    continue
                if int(a.get("occ", 1)) != 0:
                    continue
                cx, cy = float(a["cx"]), float(a["cy"])
                if not (0.0 < cx <= X_MAX and abs(cy) <= Y_ABS):
                    continue
                keep.append((cx, cy, float(a["l"]), float(a["w"]), float(a["yaw"])))
            keep.sort(key=lambda t: t[0])
            for j, b in enumerate(keep[:MAX_BOXES]):
                boxes[r, j] = b
            nbox[r] = min(len(keep), MAX_BOXES)
    # lead presence: a vehicle in the ego corridor
    lead = np.zeros(N, dtype=np.float32)
    lead_rng = np.full(N, np.nan, dtype=np.float32)
    for r in range(N):
        for j in range(int(nbox[r])):
            x, y = boxes[r, j, 0], boxes[r, j, 1]
            if abs(y) <= LEAD_LAT_M:
                lead[r] = 1.0
                lead_rng[r] = x if not np.isfinite(lead_rng[r]) else min(lead_rng[r], x)
                break
    order = sorted(range(len(sha)), key=lambda i: sha[i])
    split_of_clip = np.full(len(sha), 0, dtype=np.int8)
    for k, ci in enumerate(order):
        split_of_clip[ci] = 2 if k % 4 == 0 else (1 if k % 8 == 1 else 0)
    return {"boxes": boxes, "nbox": nbox, "has_row": has_row, "lead": lead,
            "lead_range_m": lead_rng, "clip_of_row": cor, "raw_frame": raw,
            "split": split_of_clip[cor], "sha": sha, "n_join_lines": n_lines}


def heatmap_targets(boxes, nbox, sigma_cells: float = 1.0):
    """CenterNet-style Gaussian splat + the SUB-CELL offset target and its mask.

    ⭐ The offsets exist because the grid is 2 m: without them a peak's centre carries up
    to 1.41 m of quantisation error, which alone would sink AP@1m and dent AP@2m.
    """
    n = boxes.shape[0]
    H = np.zeros((n, GX, GY), dtype=np.float32)
    OFF = np.zeros((n, GX, GY, 2), dtype=np.float32)
    OM = np.zeros((n, GX, GY), dtype=np.float32)
    gx = np.arange(GX)[:, None]
    gy = np.arange(GY)[None, :]
    for r in range(n):
        for j in range(int(nbox[r])):
            fx = boxes[r, j, 0] / CELL
            fy = (boxes[r, j, 1] + Y_ABS) / CELL
            cx, cy = fx - 0.5, fy - 0.5
            g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma_cells ** 2))
            np.maximum(H[r], g.astype(np.float32), out=H[r])
            ix, iy = int(fx), int(fy)
            if 0 <= ix < GX and 0 <= iy < GY:
                OFF[r, ix, iy, 0] = (fx - ix - 0.5) * CELL
                OFF[r, ix, iy, 1] = (fy - iy - 0.5) * CELL
                OM[r, ix, iy] = 1.0
    return H, OFF, OM


# --------------------------------------------------------------------------- #
# model                                                                        #
# --------------------------------------------------------------------------- #
class BoxHead(nn.Module):
    def __init__(self, d_in: int, grid_hw, d=192, n_layers=3, n_heads=6, d_ff=768,
                 n_slots=20, dropout=0.1):
        super().__init__()
        self.n_slots = n_slots
        self.kv_proj = nn.Linear(d_in, d)
        self.register_buffer("kv_pe", sincos_2d(*grid_hw, d), persistent=False)
        self.q_map = nn.Parameter(torch.randn(GX * GY, d) * 0.02)
        self.register_buffer("q_map_pe", sincos_2d(GX, GY, d), persistent=False)
        self.q_slot = nn.Parameter(torch.randn(n_slots + 1, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d_ff,
                                       dropout=dropout, batch_first=True,
                                       norm_first=True)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out_map = nn.Linear(d, 3)      # logit + sub-cell (dx, dy)
        self.out_box = nn.Linear(d, 6)          # x, y, l, w, sin, cos
        self.out_obj = nn.Linear(d, 1)
        self.out_lead = nn.Linear(d, 1)

    def forward(self, tok):
        b = tok.shape[0]
        kv = self.kv_proj(tok.flatten(2).transpose(1, 2)) + self.kv_pe[None]
        qm = (self.q_map + self.q_map_pe)[None].expand(b, -1, -1)
        qs = self.q_slot[None].expand(b, -1, -1)
        x = torch.cat([qm, qs], dim=1)
        for lyr in self.layers:
            x = lyr(x, kv)
        x = self.norm(x)
        nm = GX * GY
        m = self.out_map(x[:, :nm]).reshape(b, GX, GY, 3)
        hm, off = m[..., 0], torch.tanh(m[..., 1:]) * (CELL / 2.0)
        sl = x[:, nm:nm + self.n_slots]
        return (hm, off, self.out_box(sl), self.out_obj(sl).squeeze(-1),
                self.out_lead(x[:, -1]).squeeze(-1))


def decode_boxes(raw):
    """[B, S, 6] -> metric (x, y, l, w, yaw)."""
    x = torch.sigmoid(raw[..., 0]) * X_MAX
    y = (torch.sigmoid(raw[..., 1]) * 2 - 1) * Y_ABS
    l = F.softplus(raw[..., 2]) + 0.5
    w = F.softplus(raw[..., 3]) + 0.5
    yaw = torch.atan2(raw[..., 4], raw[..., 5])
    return torch.stack([x, y, l, w, yaw], dim=-1)


def hungarian_loss(pred, obj, tb, tn):
    from scipy.optimize import linear_sum_assignment
    B, S, _ = pred.shape
    l_box = pred.new_zeros(())
    tgt_obj = torch.zeros_like(obj)
    n_m = 0
    for i in range(B):
        k = int(tn[i])
        if k == 0:
            continue
        p = pred[i, :, :2]
        t = tb[i, :k, :2]
        cost = (torch.cdist(p, t, p=1) / 10.0
                - torch.sigmoid(obj[i])[:, None]).detach().cpu().numpy()
        ri, ci = linear_sum_assignment(cost)
        tgt_obj[i, ri] = 1.0
        l_box = l_box + F.l1_loss(pred[i, ri, :4], tb[i, ci, :4], reduction="sum")
        n_m += len(ri)
    l_box = l_box / max(n_m, 1)
    l_obj = F.binary_cross_entropy_with_logits(obj, tgt_obj)
    return l_box, l_obj, n_m


def focal(hm, t, alpha=2.0, beta=4.0):
    p = torch.sigmoid(hm).clamp(1e-4, 1 - 1e-4)
    pos = (t >= 0.999).float()
    lp = -((1 - p) ** alpha) * torch.log(p) * pos
    ln = -((1 - t) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (lp.sum() + ln.sum()) / max(pos.sum().item(), 1.0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("main", "shuffled", "pixel", "mirror"))
    ap.add_argument("--tokens", default="s32", choices=("s32", "s16", "pix64"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-slots", type=int, default=20)
    ap.add_argument("--out", default=r"C:\Users\Admin\d3_out\box")
    a = ap.parse_args()
    if a.arm == "pixel":
        a.tokens = "pix64"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)

    import p4_bev_head as P4M
    idx = np.load(TOK_DIR / "index.npz")
    T = build_targets(idx)
    src = P4M.TokenSource(a.tokens, tok_dir=TOK_DIR)
    rows_ok = np.nonzero(T["has_row"])[0]
    tr = rows_ok[T["split"][rows_ok] == 0]
    va = rows_ok[T["split"][rows_ok] == 1]
    te = rows_ok[T["split"][rows_ok] == 2]
    print(f"[data] rows {len(rows_ok)}/{len(T['has_row'])} · train {len(tr)} "
          f"val {len(va)} test {len(te)} · clips {len(set(T['clip_of_row'][rows_ok]))}"
          f" · mean boxes {T['nbox'][rows_ok].mean():.2f} · lead prevalence "
          f"{T['lead'][rows_ok].mean():.4f}", flush=True)

    HM, OFF, OM = heatmap_targets(T["boxes"], T["nbox"])
    # -- the SHUFFLED arm: targets permuted ACROSS ROWS, train only ------------
    perm = np.arange(len(T["boxes"]))
    if a.arm == "shuffled":
        rng = np.random.default_rng(1234 + a.seed)
        p = tr.copy()
        rng.shuffle(p)
        perm[tr] = p

    model = BoxHead(src.d_in, src.grid, n_slots=a.n_slots).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    nstep = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(nstep, 1))
    bx = torch.from_numpy(T["boxes"])
    nb = torch.from_numpy(T["nbox"].astype(np.int64))
    ld = torch.from_numpy(T["lead"])
    hm_t = torch.from_numpy(HM)
    off_t = torch.from_numpy(OFF)
    om_t = torch.from_numpy(OM)

    def batch(rows, train):
        tk = src.get(rows, dev)
        if a.arm == "mirror":
            tk = torch.flip(tk, dims=[-1])
        src_rows = perm[rows] if train else rows
        return (tk, hm_t[src_rows].to(dev), bx[src_rows].to(dev),
                nb[src_rows].to(dev), ld[src_rows].to(dev),
                off_t[src_rows].to(dev), om_t[src_rows].to(dev))

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    log = []
    step = 0
    for ep in range(a.epochs):
        model.train()
        order = tr.copy()
        rng.shuffle(order)
        acc = []
        for i in range(0, len(order) - a.batch + 1, a.batch):
            rows = order[i:i + a.batch]
            tk, th, tb, tn, tl, toff, tom = batch(rows, True)
            hm, off, rb, ob, lo = model(tk)
            pb = decode_boxes(rb)
            l_box, l_obj, _ = hungarian_loss(pb, ob, tb, tn)
            # sub-cell offset L1, on the POSITIVE cells only
            l_off = ((off - toff).abs().sum(-1) * tom).sum() / tom.sum().clamp(min=1.0)
            loss = (focal(hm, th) + 2.0 * l_box + l_obj + l_off
                    + F.binary_cross_entropy_with_logits(lo, tl))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            acc.append(float(loss))
        log.append({"epoch": ep, "loss": float(np.mean(acc)), "s": round(time.time() - t0, 1)})
        print(f"  ep{ep} loss {np.mean(acc):.4f}  {time.time() - t0:.0f}s", flush=True)

    # -- inference on the TEST clips ---------------------------------------- #
    model.eval()
    PB, OB, LO, HMP, OFP = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            rows = te[i:i + 64]
            tk = src.get(rows, dev)
            if a.arm == "mirror":
                tk = torch.flip(tk, dims=[-1])
            hm, off, rb, ob, lo = model(tk)
            PB.append(decode_boxes(rb).cpu().numpy())
            OB.append(torch.sigmoid(ob).cpu().numpy())
            LO.append(torch.sigmoid(lo).cpu().numpy())
            HMP.append(torch.sigmoid(hm).cpu().numpy().astype(np.float16))
            OFP.append(off.cpu().numpy().astype(np.float16))
    np.savez_compressed(out / "test_pred.npz", rows=te,
                        boxes=np.concatenate(PB), score=np.concatenate(OB),
                        lead=np.concatenate(LO),
                        heatmap=np.concatenate(HMP), offsets=np.concatenate(OFP),
                        gt_boxes=T["boxes"][te], gt_n=T["nbox"][te],
                        gt_lead=T["lead"][te], gt_lead_range=T["lead_range_m"][te],
                        clip_of_row=T["clip_of_row"][te], raw_frame=T["raw_frame"][te])
    cfg = {"arm": a.arm, "tokens": a.tokens, "tag": a.tag, "seed": a.seed,
           "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "n_slots": a.n_slots,
           "d_in": src.d_in, "grid": list(src.grid), "tok_dir": str(TOK_DIR),
           "join": JOIN, "n_train": int(len(tr)), "n_val": int(len(va)),
           "n_test": int(len(te)), "n_join_lines": int(T["n_join_lines"]),
           "lead_prevalence_all": float(T["lead"][rows_ok].mean()),
           "lead_prevalence_test": float(T["lead"][te].mean()),
           "mean_boxes_test": float(T["nbox"][te].mean()),
           "grid_cells": [GX, GY], "cell_m": CELL, "x_max_m": X_MAX,
           "y_abs_m": Y_ABS, "vehicle_classes": sorted(VEHICLE),
           "split_rule": "sorted sha12 position k: test k%4==0, val k%8==1, else train",
           "wall_s": round(time.time() - t0, 1), "log": log,
           "device": dev}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"[done] {a.tag} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
