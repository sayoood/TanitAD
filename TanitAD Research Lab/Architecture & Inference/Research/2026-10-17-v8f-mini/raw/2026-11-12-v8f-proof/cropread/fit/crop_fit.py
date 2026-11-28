#!/usr/bin/env python
"""crop_fit.py -- gap head of card PREREG_V8F_PROOF_ADD99 (CROPREAD), section 5. One fit per call. Tier T0.

  python crop_fit.py --arm D|B --head H1|H5 --seed K [--permute-labels] [--root /workspace/v8proof/cropread] [--device cuda]

Reads <root>/tok/<arm>/<clip>.npy (float16 [n_frames, 2, 65, 1024]; window W32 then W16; token 0 the class token, 1..64 the 8x8 grid row-major),
<root>/lists/train_final.txt (`<clip> <role>`, role fit or sel), <root>/lists/eval141.txt, the train join (labels). Writes
<root>/fits/<arm>_<head>_s<K>[_perm]/head.pt and FIT_RECORD.json. Refuses when a clip of the evaluation list is in the train list.
The fit reads fit and sel clips only; the evaluation tokens and labels are never opened here. torch is imported inside the functions that need it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import queue
import time

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD99"
N_WIN, N_TOK_WIN, WIDTH = 2, 65, 1024
N_TOK = N_WIN * N_TOK_WIN          # 130 tokens per frame
D_MODEL, N_HEADS, TOKEN_DROP = 256, 4, 0.1
HIST = 5                           # frames f-4..f of head H5
GAP_SCALE_M, GAP_MIN_M, GAP_MAX_M = 60.0, 0.0, 60.0
BETA_M = 1.0
CARD_HPARAMS = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 40, "warmup_epochs": 1, "beta_m": 1.0, "token_drop": 0.1, "d_model": 256, "n_heads": 4}
PERM_SEED_BASE = 9900
ARMS, HEADS = ("D", "B"), ("H1", "H5")
DEFAULT_ROOT = "/workspace/v8proof/cropread"
DEFAULT_JOIN_TRAIN = "/workspace/v8proof/data/inputs/join_train.npz"


# ============================================================================================ small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_name(arm, head, seed, perm=False):
    return "%s_%s_s%d%s" % (arm, head, int(seed), "_perm" if perm else "")


def dump_name(arm, head, perm=False):
    return "%s_%s%s" % (arm, head, "_perm" if perm else "")


def read_ids(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_train_list(path):
    """-> list of (clip, role) in file order; refuses a role other than fit or sel and a repeated clip."""
    rows, seen = [], set()
    for ln in read_ids(path):
        p = ln.split()
        if len(p) != 2 or p[1] not in ("fit", "sel"):
            raise SystemExit("REFUSED: train list line is not '<clip> fit|sel': %r" % ln)
        if p[0] in seen:
            raise SystemExit("REFUSED: clip %s appears twice in the train list" % p[0])
        seen.add(p[0])
        rows.append((p[0], p[1]))
    return rows


def refuse_overlap(train_rows, eval_ids):
    """control c4 in the tool: an evaluation clip in the train list stops the fit."""
    both = sorted(set(c for c, _ in train_rows) & set(eval_ids))
    if both:
        raise SystemExit("REFUSED: %d evaluation clip(s) are in the train list, first: %s" % (len(both), both[0]))


def valid_gap(gap, valid):
    g = np.asarray(gap, np.float64)
    return (np.asarray(valid) == 1) & np.isfinite(g) & (g >= GAP_MIN_M) & (g <= GAP_MAX_M)


def load_join_labels(path, clips):
    """-> {clip: (gap float64 [n], ok bool [n])}; frame i of clip c is row off[c]+i. Refuses a clip that is not in the join."""
    J = np.load(path, allow_pickle=True)
    names = [str(c) for c in J["clip"]]
    idx = {c: i for i, c in enumerate(names)}
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    gap, val = np.asarray(J["lead_gap"], np.float64), np.asarray(J["lead_valid"])
    ok = valid_gap(gap, val)
    out = {}
    for c in clips:
        if c not in idx:
            raise SystemExit("REFUSED: clip %s is not in the train join %s" % (c, path))
        i = idx[c]
        out[c] = (gap[off[i]:off[i] + n[i]].copy(), ok[off[i]:off[i] + n[i]].copy())
    return out


class TokenStore:
    """memory-maps <root>/tok/<arm>/<clip>.npy; one map per clip, opened on first use."""

    def __init__(self, root, arm):
        self.dir = os.path.join(root, "tok", arm)
        self.maps = {}

    def path(self, clip):
        return os.path.join(self.dir, clip + ".npy")

    def get(self, clip):
        a = self.maps.get(clip)
        if a is None:
            a = np.load(self.path(clip), mmap_mode="r")
            if a.ndim != 4 or a.shape[1:] != (N_WIN, N_TOK_WIN, WIDTH) or a.dtype != np.float16:
                raise SystemExit("REFUSED: tokens of %s have shape %s dtype %s, expected [n, 2, 65, 1024] float16" % (clip, list(a.shape), a.dtype))
            self.maps[clip] = a
        return a


def build_samples(clips, labels, store, head):
    """samples of a role: frames with a valid lead gap (H5 also needs frames f-4..f in the clip). -> clip index [N], frame [N], label gap [N] float32."""
    sc, sf, sy = [], [], []
    lo = HIST - 1 if head == "H5" else 0
    for k, c in enumerate(clips):
        g, ok = labels[c]
        n = store.get(c).shape[0]
        if n != len(g):
            raise SystemExit("REFUSED: clip %s has %d token frames and %d join frames" % (c, n, len(g)))
        fr = np.flatnonzero(ok)
        fr = fr[fr >= lo]
        sc.append(np.full(len(fr), k, np.int64))
        sf.append(fr.astype(np.int64))
        sy.append(g[fr].astype(np.float32))
    return np.concatenate(sc), np.concatenate(sf), np.concatenate(sy)


def gather(store, clips, sc, sf, idx, head):
    nf = HIST if head == "H5" else 1
    out = np.empty((len(idx), nf, N_TOK, WIDTH), np.float16)
    for k, i in enumerate(idx):
        a = store.get(clips[sc[i]])
        f = int(sf[i])
        out[k] = np.asarray(a[f - nf + 1:f + 1]).reshape(nf, N_TOK, WIDTH)
    return out


def prefetch(batches, make, depth=3):
    """yields make(b) for b in batches, computed by one worker thread (numpy copies release the GIL)."""
    q, end = queue.Queue(depth), object()

    def work():
        try:
            for b in batches:
                q.put(make(b))
        except BaseException as e:      # noqa: BLE001
            q.put(e)
        q.put(end)
    t = threading.Thread(target=work, daemon=True)
    t.start()
    while True:
        x = q.get()
        if x is end:
            return
        if isinstance(x, BaseException):
            raise x
        yield x


# ============================================================================================ the model (card section 5)
def make_model(head):
    import torch
    nn = torch.nn

    class CropHead(nn.Module):
        """layer norm, linear to 256, learned position vector per token, one learned query reading the tokens through one cross-attention layer (4 heads), token dropout 0.1 in training;
        H1: MLP 256-256-1 on z of the frame; H5: MLP 1280-256-1 on z of the frames f-4..f; gap = 60 * sigmoid. The activation of the MLP (GELU) is the fork's choice: the card names none."""

        def __init__(self):
            super().__init__()
            self.head = head
            self.ln = nn.LayerNorm(WIDTH)
            self.proj = nn.Linear(WIDTH, D_MODEL)
            self.pos = nn.Parameter(torch.randn(N_TOK, D_MODEL) * 0.02)
            self.query = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
            self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(D_MODEL * (HIST if head == "H5" else 1), 256), nn.GELU(), nn.Linear(256, 1))

        def embed(self, x):
            """x [N, 130, 1024] float -> z [N, 256]"""
            h = self.proj(self.ln(x)) + self.pos
            mask = None
            if self.training and TOKEN_DROP > 0:
                keep = torch.rand(x.shape[0], N_TOK, device=x.device) >= TOKEN_DROP
                keep[:, 0] |= ~keep.any(1)
                mask = ~keep
            z, _ = self.attn(self.query.expand(x.shape[0], -1, -1), h, h, key_padding_mask=mask, need_weights=False)
            return z[:, 0]

        def from_z(self, z):
            """H1: z [N, 256]; H5: z [N, 5*256] (frames f-4..f, oldest first) -> gap in metres [N]"""
            return GAP_SCALE_M * torch.sigmoid(self.mlp(z)[:, 0])

        def forward(self, x):
            """H1: x [N, 1, 130, 1024]; H5: x [N, 5, 130, 1024]"""
            n, nf = x.shape[0], x.shape[1]
            z = self.embed(x.reshape(n * nf, N_TOK, WIDTH)).reshape(n, nf * D_MODEL)
            return self.from_z(z)
    return CropHead()


def lr_at(step, steps_per_epoch, epochs, lr, warmup_epochs=1):
    warm = steps_per_epoch * warmup_epochs
    total = steps_per_epoch * epochs
    if step < warm:
        return lr * (step + 1) / warm
    return 0.5 * lr * (1.0 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


# ============================================================================================ the fit
def run_fit(a, log=print):
    import torch
    t0 = time.time()
    root = a.root
    arm, head, seed = a.arm, a.head, int(a.seed)
    if arm not in ARMS or head not in HEADS:
        raise SystemExit("REFUSED: arm %s head %s" % (arm, head))
    lists_dir = os.path.join(root, "lists")
    train_path, eval_path = os.path.join(lists_dir, "train_final.txt"), os.path.join(lists_dir, "eval141.txt")
    rows, eval_ids = read_train_list(train_path), read_ids(eval_path)
    refuse_overlap(rows, eval_ids)
    out_dir = os.path.join(root, "fits", fit_name(arm, head, seed, a.permute_labels))
    if os.path.exists(os.path.join(out_dir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: %s exists (a finished fit)" % out_dir)
    fit_ids = [c for c, r in rows if r == "fit"]
    sel_ids = [c for c, r in rows if r == "sel"]
    if not fit_ids or not sel_ids:
        raise SystemExit("REFUSED: the train list has %d fit and %d sel clips" % (len(fit_ids), len(sel_ids)))
    labels = load_join_labels(a.join_train, fit_ids + sel_ids)
    store = TokenStore(root, arm)
    fc, ff, fy = build_samples(fit_ids, labels, store, head)
    sc, sf, sy = build_samples(sel_ids, labels, store, head)
    if a.permute_labels:                                    # control c5: the label gaps are permuted across the train frames (fit and sel separately)
        fy = fy[np.random.default_rng(PERM_SEED_BASE + seed).permutation(len(fy))]
        sy = sy[np.random.default_rng(PERM_SEED_BASE + 1000 + seed).permutation(len(sy))]
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(head).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    N = len(fy)
    spe = int(math.ceil(N / a.batch))
    loss_fn = torch.nn.SmoothL1Loss(beta=BETA_M, reduction="none")

    def to_dev(pair):
        x, y = pair
        return torch.from_numpy(x).to(dev).float(), torch.from_numpy(y).to(dev)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        batches = [np.arange(i, min(i + a.batch, len(sy))) for i in range(0, len(sy), a.batch)]
        with torch.no_grad():
            for x, y in prefetch(batches, lambda b: (gather(store, sel_ids, sc, sf, b, head), sy[b])):
                x, y = to_dev((x, y))
                tot += float(loss_fn(model(x), y).sum())
                n += len(y)
        return tot / n

    best, best_state, hist = None, None, []
    step = 0
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(N)
        batches = [order[i:i + a.batch] for i in range(0, N, a.batch)]
        tl, tn = 0.0, 0
        for x, y in prefetch(batches, lambda b: (gather(store, fit_ids, fc, ff, b, head), fy[b])):
            x, y = to_dev((x, y))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, spe, a.epochs, a.lr, a.warmup_epochs)
            loss = loss_fn(model(x), y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss.detach()) * len(y)
            tn += len(y)
            step += 1
        sl = evaluate()
        hist.append({"epoch": ep, "train_loss": tl / tn, "sel_loss": sl})
        if best is None or sl < best:
            best, best_ep = sl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log("epoch %d train %.4f sel %.4f%s" % (ep, tl / tn, sl, " *" if best_ep == ep else ""))
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": best_state, "head": head, "arm": arm, "seed": seed, "epoch": best_ep, "n_tok": N_TOK, "width": WIDTH, "card": CARD}, os.path.join(out_dir, "head.pt"))
    hp = {"lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "epochs": a.epochs, "warmup_epochs": a.warmup_epochs, "beta_m": BETA_M, "token_drop": TOKEN_DROP, "d_model": D_MODEL, "n_heads": N_HEADS}
    rec = {"card": CARD, "tier": "T0", "arm": arm, "head": head, "seed": seed, "permute_labels": bool(a.permute_labels),
           "argv": list(sys.argv), "tool_sha256": sha256_file(os.path.abspath(__file__)), "train_list_sha256": sha256_file(train_path), "eval_list_sha256": sha256_file(eval_path),
           "clips_read": {"fit": fit_ids, "sel": sel_ids}, "n_samples": {"fit": int(N), "sel": int(len(sy))}, "hparams": hp, "hparams_equal_card": bool(hp == CARD_HPARAMS),
           "history": hist, "kept_epoch": int(best_ep), "kept_sel_loss": float(best), "device": str(dev), "torch": torch.__version__,
           "head_pt_sha256": sha256_file(os.path.join(out_dir, "head.pt")), "seconds": float(time.time() - t0), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "FIT_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    log("ZZ CROP-FIT %s kept epoch %d sel %.4f" % (fit_name(arm, head, seed, a.permute_labels), best_ep, best))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--join-train", default=DEFAULT_JOIN_TRAIN)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=CARD_HPARAMS["epochs"])
    ap.add_argument("--batch", type=int, default=CARD_HPARAMS["batch"])
    ap.add_argument("--lr", type=float, default=CARD_HPARAMS["lr"])
    ap.add_argument("--weight-decay", type=float, default=CARD_HPARAMS["weight_decay"])
    ap.add_argument("--warmup-epochs", type=int, default=CARD_HPARAMS["warmup_epochs"])
    a = ap.parse_args(argv)
    return run_fit(a)


if __name__ == "__main__":
    sys.exit(main())
