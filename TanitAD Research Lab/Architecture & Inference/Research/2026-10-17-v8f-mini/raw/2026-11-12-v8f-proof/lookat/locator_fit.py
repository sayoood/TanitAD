#!/usr/bin/env python
"""locator_fit.py -- locator head of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0), section 2. One fit per call. Tier T0.

  python locator_fit.py --seed K --root R --box-dir B --out-root O [--permute-targets] [--device cuda]      fit, then predict
  python locator_fit.py --seed K --root R --box-dir B --out-root O [--permute-targets] --predict            predict only (a finished fit)
  python locator_fit.py --identity --root R --out-root O --join-train J                                      control k5, no fit

Imports crop_fit.py (tool of card ADD99) from --kit-fit for TokenStore, the token reader (CropHead, H5), gather, prefetch and lr_at. The head is CropHead of H5 with the last layer 256 -> 2 and no sigmoid.
Targets t_az = clip(az, -32, 32) / 16, t_el = clip(el + 2, -16, 16) / 8; loss smooth L1 beta 0.25, mean over both values and the batch; read-out az = 16 a, el = -2 + 8 e.
Samples: frames with has_lead and finite ctr_az, ctr_el of the box file and a full five-frame history (f >= 4), as crop_fit.build_samples does for H5.
Writes <O>/fits/LOC_s<K>[_perm]/{FIT_RECORD.json,head.pt} and <O>/pred/LOC_s<K>[_perm].npz (+ _meta.json); --identity writes <O>/IDENTITY_K5.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

CARD, TIER = "PREREG_V8F_PROOF_ADD107", "T0"
BETA = 0.25
AZ_SCALE, AZ_CLIP, EL_SCALE, EL_CLIP, EL_OFF = 16.0, 32.0, 8.0, 16.0, -2.0
DEFAULT_KIT_FIT = "/workspace/v8proof/cropread/kit/fit"
DEFAULT_JOIN_TRAIN = "/workspace/v8proof/data/inputs/join_train.npz"
PERM_SEED_BASE = 9900
CFT = None


def set_cft(kit_dir):
    """imports crop_fit.py from kit_dir (once) and returns it."""
    global CFT
    p = os.path.join(kit_dir, "crop_fit.py")
    if not os.path.isfile(p):
        raise SystemExit("REFUSED: no crop_fit.py in %s" % kit_dir)
    spec = importlib.util.spec_from_file_location("crop_fit_add99", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CFT = mod
    CFT.__sha256_path__ = p
    return mod


def loc_hparams():
    """settings of card section 2: those of card ADD99 with the smooth-L1 beta of the locator (0.25)."""
    return dict(CFT.CARD_HPARAMS, beta_m=BETA)


def fit_name(seed, perm=False):
    return "LOC_s%d%s" % (int(seed), "_perm" if perm else "")


# ============================================================================================ targets
def targets_of(az, el):
    az, el = np.asarray(az, np.float64), np.asarray(el, np.float64)
    return np.stack([np.clip(az, -AZ_CLIP, AZ_CLIP) / AZ_SCALE, np.clip(el - EL_OFF, -EL_CLIP, EL_CLIP) / EL_SCALE], axis=-1).astype(np.float32)


def readout(a, e):
    return AZ_SCALE * np.asarray(a, np.float64), EL_OFF + EL_SCALE * np.asarray(e, np.float64)


def load_box(box_dir, clip, n):
    """-> (has bool[n], az[n], el[n]) of the box centre in degrees (0 where no centre), as window_encode.load_box_dirs; refuses a frame count other than n."""
    p = os.path.join(box_dir, clip + ".npz")
    if not os.path.exists(p):
        raise SystemExit("REFUSED: no box file for %s in %s" % (clip, box_dir))
    z = np.load(p)
    has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
    if len(has) != n:
        raise SystemExit("REFUSED: box file of %s has %d frames, the token file %d" % (clip, len(has), n))
    return has, np.where(has, z["ctr_az"], 0.0).astype(np.float64), np.where(has, z["ctr_el"], 0.0).astype(np.float64)


def build_samples(clips, store, mode, box_dir=None, labels=None):
    """samples of a role. mode 'box': frames with a box centre (targets [N, 2]); mode 'gap': frames with a valid join gap (labels {clip: (gap, ok)}, targets [N]; used by the identity control only).
    Both need frames f-4..f in the clip (head H5). -> clip index [N], frame [N], target [N, 2] or [N] float32."""
    lo = CFT.HIST - 1
    sc, sf, sy = [], [], []
    for k, c in enumerate(clips):
        n = store.get(c).shape[0]
        if mode == "box":
            has, az, el = load_box(box_dir, c, n)
            ok, y = has, targets_of(az, el)
        else:
            g, ok = labels[c]
            if n != len(g):
                raise SystemExit("REFUSED: clip %s has %d token frames and %d join frames" % (c, n, len(g)))
            y = np.asarray(g, np.float32)
        fr = np.flatnonzero(ok)
        fr = fr[fr >= lo]
        sc.append(np.full(len(fr), k, np.int64))
        sf.append(fr.astype(np.int64))
        sy.append(y[fr])
    shape = (0, 2) if mode == "box" else (0,)
    return np.concatenate(sc), np.concatenate(sf), (np.concatenate(sy) if sy else np.zeros(shape, np.float32))


# ============================================================================================ the model
def make_loc_model():
    """CropHead of H5 with the last layer of the MLP giving two values and no sigmoid."""
    import torch
    m = CFT.make_model("H5")
    m.mlp[2] = torch.nn.Linear(256, 2)
    m.from_z = lambda z, _m=m: _m.mlp(z)
    return m


def predict_clip(model, tokens, dev, chunk=64):
    """tokens [n, 2, 65, 1024] float16 -> (az [n], el [n]) in degrees, NaN for frames 0 to 3."""
    import torch
    n = tokens.shape[0]
    az, el = np.full(n, np.nan), np.full(n, np.nan)
    if n < CFT.HIST:
        return az, el
    model.eval()
    with torch.no_grad():
        Z = []
        for i in range(0, n, chunk):
            x = torch.from_numpy(np.array(tokens[i:i + chunk]).reshape(-1, CFT.N_TOK, CFT.WIDTH)).to(dev).float()
            Z.append(model.embed(x))
        Z = torch.cat(Z)
        idx = torch.arange(CFT.HIST - 1, n, device=dev)
        win = torch.stack([Z[idx - (CFT.HIST - 1 - k)] for k in range(CFT.HIST)], 1).reshape(len(idx), -1)
        out = model.from_z(win).double().cpu().numpy()
    az[CFT.HIST - 1:], el[CFT.HIST - 1:] = readout(out[:, 0], out[:, 1])
    return az, el


def write_pred(model, store, eval_ids, dev, path):
    cid, fr, az, el = [], [], [], []
    for k, c in enumerate(eval_ids):
        a, e = predict_clip(model, store.get(c), dev)
        cid.append(np.full(len(a), k, np.int64))
        fr.append(np.arange(len(a), dtype=np.int64))
        az.append(a)
        el.append(e)
    az, el = np.concatenate(az), np.concatenate(el)
    with open(path, "wb") as fh:
        np.savez(fh, clip_index=np.concatenate(cid), frame=np.concatenate(fr), az_pred=az, el_pred=el, has_out=np.isfinite(az) & np.isfinite(el), clips=np.asarray(eval_ids))
    return int(np.isfinite(az).sum())


def pick_device(a):
    import torch
    return torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")


def lists_of(root):
    ld = os.path.join(root, "lists")
    tp, ep = os.path.join(ld, "train_final.txt"), os.path.join(ld, "eval141.txt")
    rows, ev = CFT.read_train_list(tp), CFT.read_ids(ep)
    CFT.refuse_overlap(rows, ev)
    return tp, ep, [c for c, r in rows if r == "fit"], [c for c, r in rows if r == "sel"], ev


# ============================================================================================ the fit
def run_fit(a, log=print):
    import torch
    t0 = time.time()
    seed = int(a.seed)
    tp, ep, fit_ids, sel_ids, ev = lists_of(a.root)
    out_dir = os.path.join(a.out_root, "fits", fit_name(seed, a.permute_targets))
    pred_dir = os.path.join(a.out_root, "pred")
    ppath = os.path.join(pred_dir, fit_name(seed, a.permute_targets) + ".npz")
    if os.path.exists(os.path.join(out_dir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: %s exists (a finished fit)" % out_dir)
    if os.path.exists(ppath):
        raise SystemExit("REFUSED: %s exists" % ppath)
    if not fit_ids or not sel_ids:
        raise SystemExit("REFUSED: the train list has %d fit and %d sel clips" % (len(fit_ids), len(sel_ids)))
    store = CFT.TokenStore(a.root, "D")
    fc, ff, fy = build_samples(fit_ids, store, "box", box_dir=a.box_dir)
    sc, sf, sy = build_samples(sel_ids, store, "box", box_dir=a.box_dir)
    if a.permute_targets:                                   # deliberate regression: the (az, el) pairs are permuted across the train frames (fit and sel separately), the seeds as crop_fit.py
        fy = fy[np.random.default_rng(PERM_SEED_BASE + seed).permutation(len(fy))]
        sy = sy[np.random.default_rng(PERM_SEED_BASE + 1000 + seed).permutation(len(sy))]
    dev = pick_device(a)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_loc_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    N = len(fy)
    spe = int(math.ceil(N / a.batch))
    loss_fn = torch.nn.SmoothL1Loss(beta=BETA, reduction="none")

    def to_dev(pair):
        x, y = pair
        return torch.from_numpy(x).to(dev).float(), torch.from_numpy(y).to(dev)

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        batches = [np.arange(i, min(i + a.batch, len(sy))) for i in range(0, len(sy), a.batch)]
        with torch.no_grad():
            for x, y in CFT.prefetch(batches, lambda b: (CFT.gather(store, sel_ids, sc, sf, b, "H5"), sy[b])):
                x, y = to_dev((x, y))
                tot += float(loss_fn(model(x), y).sum())
                n += y.numel()                              # elements: the mean is over both values and the samples
        return tot / n

    best, best_ep, best_state, hist, step = None, -1, None, [], 0
    for ep_i in range(a.epochs):
        model.train()
        order = rng.permutation(N)
        batches = [order[i:i + a.batch] for i in range(0, N, a.batch)]
        tl, tn = 0.0, 0
        for x, y in CFT.prefetch(batches, lambda b: (CFT.gather(store, fit_ids, fc, ff, b, "H5"), fy[b])):
            x, y = to_dev((x, y))
            for g in opt.param_groups:
                g["lr"] = CFT.lr_at(step, spe, a.epochs, a.lr, a.warmup_epochs)
            loss = loss_fn(model(x), y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss.detach()) * len(y)
            tn += len(y)
            step += 1
        sl = evaluate()
        hist.append({"epoch": ep_i, "train_loss": tl / tn, "sel_loss": sl})
        if best is None or sl < best:
            best, best_ep = sl, ep_i
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log("epoch %d train %.4f sel %.4f%s" % (ep_i, tl / tn, sl, " *" if best_ep == ep_i else ""))
    os.makedirs(out_dir, exist_ok=True)
    hpt = os.path.join(out_dir, "head.pt")
    torch.save({"state_dict": best_state, "head": "H5", "arm": "D", "seed": seed, "epoch": best_ep, "n_tok": CFT.N_TOK, "width": CFT.WIDTH, "card": CARD, "out_dim": 2}, hpt)
    hp = {"lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "epochs": a.epochs, "warmup_epochs": a.warmup_epochs, "beta_m": BETA, "token_drop": CFT.TOKEN_DROP, "d_model": CFT.D_MODEL, "n_heads": CFT.N_HEADS}
    rec = {"card": CARD, "tier": TIER, "seed": seed, "permute_targets": bool(a.permute_targets), "argv": list(sys.argv), "tool_sha256": CFT.sha256_file(os.path.abspath(__file__)),
           "crop_fit_sha256": CFT.sha256_file(CFT.__sha256_path__), "train_list_sha256": CFT.sha256_file(tp), "eval_list_sha256": CFT.sha256_file(ep),
           "clips_read": {"fit": fit_ids, "sel": sel_ids}, "n_samples": {"fit": int(N), "sel": int(len(sy))}, "hparams": hp, "hparams_equal_card": bool(hp == loc_hparams()),
           "history": hist, "kept_epoch": int(best_ep), "kept_sel_loss": float(best), "device": str(dev), "torch": torch.__version__, "head_pt_sha256": CFT.sha256_file(hpt),
           "seconds": float(time.time() - t0), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "FIT_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    log("ZZ LOCATOR-FIT %s kept epoch %d sel %.4f" % (fit_name(seed, a.permute_targets), best_ep, best))
    if not a.no_predict:
        run_predict(a, log)
    return 0


def run_predict(a, log=print):
    import torch
    seed = int(a.seed)
    name = fit_name(seed, a.permute_targets)
    fdir = os.path.join(a.out_root, "fits", name)
    if not os.path.isfile(os.path.join(fdir, "FIT_RECORD.json")):
        raise SystemExit("REFUSED: no finished fit in " + fdir)
    pdir = os.path.join(a.out_root, "pred")
    ppath = os.path.join(pdir, name + ".npz")
    if os.path.exists(ppath):
        raise SystemExit("REFUSED: %s exists" % ppath)
    _, _, _, _, ev = lists_of(a.root)
    dev = pick_device(a)
    ck = torch.load(os.path.join(fdir, "head.pt"), map_location="cpu")
    if ck.get("card") != CARD or ck.get("out_dim") != 2:
        raise SystemExit("REFUSED: head.pt is not a locator head of this card")
    model = make_loc_model()
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    os.makedirs(pdir, exist_ok=True)
    n_out = write_pred(model, CFT.TokenStore(a.root, "D"), ev, dev, ppath)
    meta = {"card": CARD, "fit": name, "head_pt_sha256": CFT.sha256_file(os.path.join(fdir, "head.pt")), "pred_sha256": CFT.sha256_file(ppath), "n_eval_clips": len(ev), "n_rows_with_output": n_out}
    with open(os.path.join(pdir, name + "_meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log("ZZ LOCATOR-PRED %s rows with output %d" % (name, n_out))
    return 0


# ============================================================================================ control k5
def run_identity(a, log=print):
    """the ORIGINAL head of card ADD99 (gap output, sigmoid) through this tool's token path and sample builder: the selection loss as crop_fit.py's sel_loss (smooth L1, beta 1.0 m, sum over samples / n)."""
    import torch
    out = os.path.join(a.out_root, "IDENTITY_K5.json")
    if os.path.exists(out):
        raise SystemExit("REFUSED: %s exists" % out)
    fdir = os.path.join(a.root, "fits", "D_H5_s0")
    rec = json.load(open(os.path.join(fdir, "FIT_RECORD.json"), encoding="utf-8"))
    ck = torch.load(os.path.join(fdir, "head.pt"), map_location="cpu")
    _, _, _, sel_ids, _ = lists_of(a.root)
    dev = pick_device(a)
    model = CFT.make_model("H5")
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    store = CFT.TokenStore(a.root, "D")
    labels = CFT.load_join_labels(a.join_train, sel_ids)
    sc, sf, sy = build_samples(sel_ids, store, "gap", labels=labels)
    loss_fn = torch.nn.SmoothL1Loss(beta=CFT.BETA_M, reduction="none")
    tot, n = 0.0, 0
    batches = [np.arange(i, min(i + CFT.CARD_HPARAMS["batch"], len(sy))) for i in range(0, len(sy), CFT.CARD_HPARAMS["batch"])]
    with torch.no_grad():
        for x, y in CFT.prefetch(batches, lambda b: (CFT.gather(store, sel_ids, sc, sf, b, "H5"), sy[b])):
            tot += float(loss_fn(model(torch.from_numpy(x).to(dev).float()), torch.from_numpy(y).to(dev)).sum())
            n += len(y)
    got, want = tot / n, float(rec["kept_sel_loss"])
    res = {"card": CARD, "control": "k5", "sel_loss_recomputed": got, "kept_sel_loss_record": want, "abs_diff": abs(got - want), "tolerance": a.identity_tol, "n_sel_samples": int(n),
           "n_sel_samples_record": rec.get("n_samples", {}).get("sel"), "kept_epoch_record": rec.get("kept_epoch"), "kept_epoch_head_file": ck.get("epoch"),
           "head_pt_sha256": CFT.sha256_file(os.path.join(fdir, "head.pt")), "head_pt_sha256_record": rec.get("head_pt_sha256"), "device": str(dev), "torch": torch.__version__,
           "tool_sha256": CFT.sha256_file(os.path.abspath(__file__)), "crop_fit_sha256": CFT.sha256_file(CFT.__sha256_path__)}
    res["PASS"] = bool(res["abs_diff"] < a.identity_tol and n == res["n_sel_samples_record"] and res["head_pt_sha256"] == res["head_pt_sha256_record"] and res["kept_epoch_record"] == res["kept_epoch_head_file"])
    os.makedirs(a.out_root, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    log("k5 recomputed %.8f record %.8f diff %.3g n %d -> %s" % (got, want, res["abs_diff"], n, "PASS" if res["PASS"] else "FAIL"))
    log("ZZ LOCATOR-IDENTITY pass %s" % res["PASS"])
    return 0 if res["PASS"] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", required=True)
    ap.add_argument("--box-dir", default="")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--kit-fit", default=DEFAULT_KIT_FIT)
    ap.add_argument("--permute-targets", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--join-train", default=DEFAULT_JOIN_TRAIN)
    ap.add_argument("--identity-tol", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    a, rest = ap.parse_known_args(argv)
    set_cft(a.kit_fit)
    H = CFT.CARD_HPARAMS
    ap2 = argparse.ArgumentParser(allow_abbrev=False)
    ap2.add_argument("--epochs", type=int, default=H["epochs"])
    ap2.add_argument("--batch", type=int, default=H["batch"])
    ap2.add_argument("--lr", type=float, default=H["lr"])
    ap2.add_argument("--weight-decay", type=float, default=H["weight_decay"])
    ap2.add_argument("--warmup-epochs", type=int, default=H["warmup_epochs"])
    b = ap2.parse_args(rest)
    for k, v in vars(b).items():
        setattr(a, k, v)
    if a.identity:
        return run_identity(a)
    if not a.box_dir:
        raise SystemExit("REFUSED: --box-dir is required")
    return run_predict(a) if a.predict else run_fit(a)


if __name__ == "__main__":
    sys.exit(main())
