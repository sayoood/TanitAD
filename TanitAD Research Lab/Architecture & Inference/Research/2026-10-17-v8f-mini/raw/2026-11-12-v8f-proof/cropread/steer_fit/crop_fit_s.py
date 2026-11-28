#!/usr/bin/env python
"""crop_fit_s.py -- gap head on arm S of card PREREG_V8F_PROOF_ADD100 (STEERREAD), section 3. One fit per call. Tier T0.

  python crop_fit_s.py --head H5|H5d --seed K [--permute-labels] [--run1 R1] [--s-root S] [--join-train NPZ] [--device cuda]

Form S_H5  = head H5 of card ADD99 (crop_fit.make_model("H5"), tokens only) on <s-root>/tok/S.
Form S_H5d = the same with the steering direction of each of the five frames (azimuth / 16, elevation / 16, from <s-root>/steer/<clip>.npy) appended to that frame's vector
             (256 + 2 per frame, 5 frames, MLP 1290-256-1; per frame, oldest frame first).
Fit as in card ADD99 section 5 with note N1 (40 epochs, batch 256, AdamW 1e-3, weight decay 0.05, one warm-up epoch then cosine, the epoch with the lowest loss on the 50 selection
clips kept), the same fit and selection clips (<run1>/lists/train_final.txt), the same refusals as crop_fit.py (an evaluation clip in the train list, a finished fit, a frame count that
differs from the join). Also refuses unless <s-root>/ctl/S8.json carries pass true and <s-root>/tok/S/MANIFEST.json exists. Writes <s-root>/fits/S_<head>_s<K>[_perm]/head.pt, FIT_RECORD.json.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import steer_common as SC  # noqa: E402

CFT = SC.CFT
DIM_FRAME = CFT.D_MODEL + 2           # a frame's vector with the direction appended


def make_model_s(head):
    """S_H5: head H5 of card ADD99 as it is. S_H5d: the same reader with the MLP input 5 * (256 + 2) = 1290; forward(x, d)."""
    import torch
    nn = torch.nn
    if head == "H5":
        return CFT.make_model("H5")
    if head != "H5d":
        raise SystemExit("REFUSED: head %r" % head)
    base = CFT.make_model("H5")
    base.mlp = nn.Identity()

    class SteeredHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = "H5d"
            self.base = base
            self.mlp = nn.Sequential(nn.Linear(CFT.HIST * DIM_FRAME, 256), nn.GELU(), nn.Linear(256, 1))

        def embed(self, x):
            return self.base.embed(x)

        def from_zd(self, zd):
            """zd [N, 1290] -> gap in metres [N]"""
            return CFT.GAP_SCALE_M * torch.sigmoid(self.mlp(zd)[:, 0])

        def forward(self, x, d):
            """x [N, 5, 130, 1024] float, d [N, 5, 2] degrees (azimuth, elevation of the five frames)"""
            n, nf = x.shape[0], x.shape[1]
            z = self.embed(x.reshape(n * nf, CFT.N_TOK, CFT.WIDTH)).reshape(n, nf, CFT.D_MODEL)
            return self.from_zd(torch.cat([z, d / SC.DIR_SCALE], 2).reshape(n, nf * DIM_FRAME))
    return SteeredHead()


def call(model, x, d):
    return model(x, d) if getattr(model, "head", "") == "H5d" else model(x)


def gather_s(store, steer, clips, sc, sf, idx):
    """-> (tokens float16 [n, 5, 130, 1024], directions float32 [n, 5, 2]) of frames f-4..f of sample i."""
    x = CFT.gather(store, clips, sc, sf, idx, "H5")
    d = np.empty((len(idx), CFT.HIST, 2), np.float32)
    for k, i in enumerate(idx):
        c = clips[sc[i]]
        s = steer.get(c, store.get(c).shape[0])
        f = int(sf[i])
        d[k] = s[f - CFT.HIST + 1:f + 1]
    return x, d


def run_fit_s(a, log=print):
    import torch
    t0 = time.time()
    head, seed = a.head, int(a.seed)
    if head not in SC.SHEADS:
        raise SystemExit("REFUSED: head %s" % head)
    rows, eval_ids = SC.read_lists(a.run1)                       # refuses an evaluation clip in the train list
    s8p = os.path.join(a.s_root, "ctl", "S8.json")
    if not (os.path.isfile(s8p) and SC.read_json(s8p).get("pass") is True):
        raise SystemExit("REFUSED: %s is missing or does not carry pass true" % s8p)
    man = os.path.join(a.s_root, "tok", "S", "MANIFEST.json")
    if not os.path.isfile(man):
        raise SystemExit("REFUSED: no %s" % man)
    out_dir = os.path.join(a.s_root, "fits", CFT.fit_name("S", head, seed, a.permute_labels))
    if os.path.exists(os.path.join(out_dir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: %s exists (a finished fit)" % out_dir)
    fit_ids = [c for c, r in rows if r == "fit"]
    sel_ids = [c for c, r in rows if r == "sel"]
    if not fit_ids or not sel_ids:
        raise SystemExit("REFUSED: the train list has %d fit and %d sel clips" % (len(fit_ids), len(sel_ids)))
    labels = CFT.load_join_labels(a.join_train, fit_ids + sel_ids)
    store, steer = CFT.TokenStore(a.s_root, "S"), SC.SteerStore(a.s_root)
    fc, ff, fy = CFT.build_samples(fit_ids, labels, store, "H5")
    sc, sf, sy = CFT.build_samples(sel_ids, labels, store, "H5")
    if a.permute_labels:                                          # control s6: label gaps permuted within the fit frames and within the selection frames (as control c5)
        fy = fy[np.random.default_rng(CFT.PERM_SEED_BASE + seed).permutation(len(fy))]
        sy = sy[np.random.default_rng(CFT.PERM_SEED_BASE + 1000 + seed).permutation(len(sy))]
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model_s(head).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    N = len(fy)
    spe = int(math.ceil(N / a.batch))
    loss_fn = torch.nn.SmoothL1Loss(beta=CFT.BETA_M, reduction="none")

    def to_dev(t):
        x, d, y = t
        return torch.from_numpy(x).to(dev).float(), torch.from_numpy(d).to(dev), torch.from_numpy(y).to(dev)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        batches = [np.arange(i, min(i + a.batch, len(sy))) for i in range(0, len(sy), a.batch)]
        with torch.no_grad():
            for t in CFT.prefetch(batches, lambda b: gather_s(store, steer, sel_ids, sc, sf, b) + (sy[b],)):
                x, d, y = to_dev(t)
                tot += float(loss_fn(call(model, x, d), y).sum())
                n += len(y)
        return tot / n

    best, best_ep, hist, step = None, -1, [], 0
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(N)
        batches = [order[i:i + a.batch] for i in range(0, N, a.batch)]
        tl, tn = 0.0, 0
        for t in CFT.prefetch(batches, lambda b: gather_s(store, steer, fit_ids, fc, ff, b) + (fy[b],)):
            x, d, y = to_dev(t)
            for g in opt.param_groups:
                g["lr"] = CFT.lr_at(step, spe, a.epochs, a.lr, a.warmup_epochs)
            loss = loss_fn(call(model, x, d), y).mean()
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
    pt = os.path.join(out_dir, "head.pt")
    torch.save({"state_dict": best_state, "head": head, "arm": "S", "seed": seed, "epoch": best_ep, "n_tok": CFT.N_TOK, "width": CFT.WIDTH, "card": SC.CARD}, pt)
    hp = {"lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "epochs": a.epochs, "warmup_epochs": a.warmup_epochs, "beta_m": CFT.BETA_M, "token_drop": CFT.TOKEN_DROP,
          "d_model": CFT.D_MODEL, "n_heads": CFT.N_HEADS}
    lists = SC.paths(a.run1)
    rec = {"card": SC.CARD, "tier": "T0", "arm": "S", "head": head, "seed": seed, "permute_labels": bool(a.permute_labels), "argv": list(sys.argv),
           "tool_sha256": SC.sha256_file(os.path.abspath(__file__)), "train_list_sha256": SC.sha256_file(lists["train_list"]), "eval_list_sha256": SC.sha256_file(lists["eval_list"]),
           "clips_read": {"fit": fit_ids, "sel": sel_ids}, "n_samples": {"fit": int(N), "sel": int(len(sy))}, "hparams": hp, "hparams_equal_card": bool(hp == CFT.CARD_HPARAMS),
           "steer_sha256": {c: SC.sha256_file(steer.path(c)) for c in fit_ids + sel_ids}, "s8_sha256": SC.sha256_file(s8p), "tok_s_manifest_sha256": SC.sha256_file(man),
           "history": hist, "kept_epoch": int(best_ep), "kept_sel_loss": float(best), "device": str(dev), "torch": torch.__version__,
           "head_pt_sha256": SC.sha256_file(pt), "seconds": float(time.time() - t0), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    SC.write_json(os.path.join(out_dir, "FIT_RECORD.json"), rec)
    log("ZZ CROP-FIT-S %s kept epoch %d sel %.4f" % (CFT.fit_name("S", head, seed, a.permute_labels), best_ep, best))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--head", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--run1", default=SC.RUN1)
    ap.add_argument("--s-root", default=SC.S_ROOT)
    ap.add_argument("--join-train", default=CFT.DEFAULT_JOIN_TRAIN)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=CFT.CARD_HPARAMS["epochs"])
    ap.add_argument("--batch", type=int, default=CFT.CARD_HPARAMS["batch"])
    ap.add_argument("--lr", type=float, default=CFT.CARD_HPARAMS["lr"])
    ap.add_argument("--weight-decay", type=float, default=CFT.CARD_HPARAMS["weight_decay"])
    ap.add_argument("--warmup-epochs", type=int, default=CFT.CARD_HPARAMS["warmup_epochs"])
    return run_fit_s(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
