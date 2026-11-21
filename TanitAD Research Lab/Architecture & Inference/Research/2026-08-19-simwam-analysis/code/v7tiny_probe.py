"""Does the trunk's latent linearly encode EGO STATE? — properly powered.

⛔ WHY THE PREVIOUS ATTEMPT WAS VOID. `v7tiny_dyn.py` panel A regressed d(ego)
on the RAW 2048-dim latent using ~700 training rows. With n << d the validation
split correctly selects maximal regularisation -- the ridge collapses to the
constant predictor and every representation, INCLUDING the constant control,
reads +0.0000. That is UNDERPOWERED BY CONSTRUCTION and says nothing about the
latent. Three earlier versions of that panel were also wrong (raw-energy
normalisation; lambda chosen on the point estimate; lambda chosen on the test
set). This file fixes the power problem rather than the selector:

  * PCA to k=128 components, FIT ON THE TRAINING CLIPS ONLY and applied to the
    held-out clips. n/d goes from ~0.34 to ~25.
  * every frame of every clip (up to 200), not 120.
  * lambda chosen on a validation split of the FIT clips; test used only to score.

WHAT IS ASKED. The standard linear-probe question, which the programme already
has a reference point for -- REF-A's frozen-DINO trunk reads ego speed at
R^2 0.61, and D-A5 attributed REF-A's failure to that frozen encoder:

    speed      poses[:, 3]
    yaw        poses[:, 2]
    yaw_rate   d(yaw)/dt
    d_ego      the one-tick displacement (the panel that was void)

R^2 is reported OVER THE TRAINING-MEAN PREDICTOR, so a representation carrying
nothing scores 0.0000 -- and the CONSTANT control is carried in every panel to
make that visible on sight.

⚠️ `pixel` (raw pooled frames, same PCA treatment) is the floor in every panel.
A latent that does not beat raw pixels has added nothing.

⚠️ Episode-disjoint throughout; episode-cluster bootstrap of the POOLED
statistic.

TIER: T0-DIAGNOSTIC. This is a representation diagnostic, never a driving claim.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects"
                            r"\TanitAD\stack")))
HELD = SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
DT = 0.1


def pooled_frames(path, n):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
        np.int64)
    rows = []
    for i in range(min(n, len(off) - 1)):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("L")
        rows.append(np.asarray(im.resize((40, 16), Image.BOX),
                               dtype=np.float64).ravel() / 255.0)
    px = np.stack(rows)
    if float(np.abs(px).mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path.name} all-zero frames")
    return px


def dinov3_encode(paths, max_frames, dev):
    """Frozen DINOv3 ViT-L/16 patch-token mean per frame, on the SAME clips.

    ⭐ THE APPLES-TO-APPLES REFERENCE. v6F's trained trunk reads ego speed at
    R^2 +0.0025 here. The programme's standing claim (D-A5) is that REF-A failed
    BECAUSE of its frozen DINO encoder, quoting speed R^2 0.61 -- but that was a
    different corpus and protocol, so it cannot be compared directly. Running a
    frozen encoder through THIS probe on THESE clips is what makes the
    comparison admissible.

    ⛔ bf16, NEVER fp16: MEASURED 2026-08-19, a DINOv3 ViT-L/16 forward in fp16
    returns ALL-NaN SILENTLY and a full extraction printed DONE on garbage.
    Verified by content below, not by existence.
    """
    import truststore
    truststore.inject_into_ssl()
    from transformers import AutoImageProcessor, DINOv3ViTModel
    M = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    # ⛔ local_files_only: the repo is GATED, and the weights are already in the
    # local HF cache. Reading from cache means no token is read, copied, or
    # passed as an argument anywhere -- which is the required handling.
    proc = AutoImageProcessor.from_pretrained(M, local_files_only=True)
    model = DINOv3ViTModel.from_pretrained(
        M, dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    H, W = 256, 640
    n_patch = (H // 16) * (W // 16)
    out_all = []
    for n, p in enumerate(paths, 1):
        d = torch.load(p, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
            np.int64)
        m = min(max_frames, len(off) - 1)
        rows = []
        for s in range(0, m, 8):
            ims = [Image.open(io.BytesIO(raw[off[j]:off[j + 1]])).convert("RGB")
                   .resize((W, H)) for j in range(s, min(s + 8, m))]
            inp = proc(images=ims, return_tensors="pt", do_resize=False)
            inp = {k: (v.to(dev, torch.bfloat16) if v.dtype.is_floating_point
                       else v.to(dev)) for k, v in inp.items()}
            with torch.no_grad():
                o = model(**inp).last_hidden_state[:, -n_patch:]
            rows.append(o.float().mean(1).cpu().numpy())
        F = np.concatenate(rows).astype(np.float64)
        if not np.isfinite(F).all() or float(np.abs(F).mean()) == 0.0:
            raise SystemExit(f"[FATAL] DINOv3 produced non-finite/zero features "
                             f"on {p.name} -- the fp16 NaN failure mode")
        out_all.append(F)
        print(f"    [dino {n}/{len(paths)}] {p.name[:10]} {len(F)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out_all


def encode_tokens_meanpooled(world, path, dev, max_frames, n_stack=3):
    """v6's OWN encoder patch tokens, mean-pooled per frame.

    THE POINT. `z_op` is the operative latent AFTER the readout bottleneck
    (16x40 tokens pooled to a 4x4 grid). DINOv3's column is mean-pooled patch
    tokens. Comparing those two and concluding "the encoder is bad" confuses the
    ENCODER with the READOUT. This gives v6's encoder EXACTLY the DINOv3
    treatment, so the three columns separate the two:

        enc tokens HIGH, z_op LOW  ->  the READOUT destroys it
        enc tokens LOW              ->  the ENCODER never had it

    Memory note: tokens are pooled per batch immediately. Holding
    [N, W, n_tok, d] float32 is the 25.6 GB trap that thrashed a box for 2.5 h.
    """
    import io as _io
    from PIL import Image as _Image
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
    n = min(max_frames, len(off) - 1)
    imgs = []
    for i in range(n):
        im = _Image.open(_io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    out = []
    with torch.no_grad():
        for s in range(0, n, 16):
            chunk = []
            for i in range(s, min(s + 16, n)):
                idx = [max(i - j, 0) for j in range(n_stack - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)
            _z, tok = world.stack.encode_window(x, return_tokens=True)
            out.append(tok[:, 0].float().mean(1).cpu().numpy())   # pool NOW
    F = np.concatenate(out).astype(np.float64)
    if not np.isfinite(F).all():
        raise SystemExit(f"[FATAL] non-finite encoder tokens on {path.name}")
    return F


def pca_fit(X, k):
    mu = X.mean(0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:min(k, Vt.shape[0])].T


def probe(feats, targets, k_pca, seed=0):
    """-> (r2, ci_lo, ci_hi, lam, n_rows, d). Episode-disjoint; PCA and lambda
    both fit on TRAINING clips only."""
    n = len(feats)
    half = n // 2
    fit, te = list(range(half)), list(range(half, n))
    nv = max(1, len(fit) // 4)
    inner, val = fit[:-nv], fit[-nv:]

    Xin = np.concatenate([feats[i] for i in inner])
    if k_pca and Xin.shape[1] > k_pca:
        mu_x, P = pca_fit(Xin, k_pca)
        red = lambda A: (A - mu_x) @ P
    else:
        red = lambda A: A
    Xin = red(Xin)
    Yin = np.concatenate([targets[i] for i in inner])
    xm, ym = Xin.mean(0, keepdims=True), Yin.mean(0, keepdims=True)
    Xc, Yc = Xin - xm, Yin - ym
    G, C = Xc.T @ Xc, Xc.T @ Yc

    Ws, vs = {}, {}
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), C)
        Ws[lam] = W
        vs[lam] = sum(float((((targets[i])
                              - ((red(feats[i]) - xm) @ W + ym)) ** 2).sum())
                      for i in val)
    lam = min(vs, key=vs.get)
    W = Ws[lam]
    errs, tots = [], []
    for i in te:
        p = (red(feats[i]) - xm) @ W + ym
        errs.append(float(((targets[i] - p) ** 2).sum()))
        tots.append(float(((targets[i] - ym) ** 2).sum()))
    errs, tots = np.array(errs), np.array(tots)
    r2 = 1.0 - errs.sum() / tots.sum()
    rng = np.random.default_rng(seed)
    bs = np.empty(4000)
    for b in range(4000):
        j = rng.integers(0, len(errs), len(errs))
        bs[b] = 1.0 - errs[j].sum() / tots[j].sum()
    return (float(r2), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)), lam, int(Xin.shape[0]),
            int(Xin.shape[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description="ego linear probe, powered")
    ap.add_argument("--arm", default="fixed")
    ap.add_argument("--v6f", action="store_true")
    ap.add_argument("--v6f-ckpt", default=str(SP / "ckpt/v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--v6f-config", default=str(SP / "sp2/v6F_config.json"))
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames-per-clip", type=int, default=200)
    ap.add_argument("--k-pca", type=int, default=128)
    ap.add_argument("--enc-tokens", action="store_true",
                    help="add v6's own PRE-READOUT encoder tokens")
    ap.add_argument("--dinov3", action="store_true",
                    help="add a FROZEN DINOv3 reference column")
    ap.add_argument("--out", default=str(SP / "v7tiny_probe.json"))
    a = ap.parse_args()

    import v7tiny_g2 as G
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.v6f:
        import e_pred_probe as E
        world, step = E.load_world(Path(a.v6f_ckpt), Path(a.v6f_config), dev)
        name = f"v6F@{step}"
    else:
        world, step = G.load_arm(a.arm, dev)
        name = f"v7tiny-{a.arm}@{step}"

    paths = sorted(HELD.glob("*.v2ep.pt"))[:a.clips]
    Z, PX, PO = [], [], []
    for i, p in enumerate(paths, 1):
        z, _act, _s = G.encode_clip(world, p, dev, a.frames_per_clip)
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = min(len(z), len(d["poses"]))
        Z.append(z.numpy()[:m].astype(np.float64))
        PO.append(d["poses"].numpy()[:m].astype(np.float64))
        PX.append(pooled_frames(p, m)[:m])
        print(f"    [{i}/{len(paths)}] {p.name[:10]} {m}", flush=True)

    TARGETS = {
        "speed":    [p[:, 3:4] for p in PO],
        "yaw":      [p[:, 2:3] for p in PO],
        "yaw_rate": [np.concatenate([np.diff(p[:, 2:3], axis=0) / DT,
                                     np.zeros((1, 1))]) for p in PO],
        "d_ego":    [np.concatenate([np.diff(p, axis=0),
                                     np.zeros((1, 4))]) for p in PO],
    }
    FEATS = {"latent": Z, "pixel (floor)": PX,
             "constant (control)": [np.ones((len(z), 1)) for z in Z]}
    if a.enc_tokens:
        ET = [encode_tokens_meanpooled(world, p, dev, a.frames_per_clip)
              for p in paths]
        FEATS = {"z_op (post-readout)": Z, "enc tokens (pre-readout)": ET,
                 "pixel (floor)": PX,
                 "constant (control)": [np.ones((len(z), 1)) for z in Z]}
    if a.dinov3:
        DN = dinov3_encode(paths, a.frames_per_clip, dev)
        FEATS = dict(FEATS)
        FEATS["frozen DINOv3"] = DN
        FEATS = {k: FEATS[k] for k in
                 [x for x in ("z_op (post-readout)", "latent",
                              "enc tokens (pre-readout)", "frozen DINOv3",
                              "pixel (floor)", "constant (control)")
                  if x in FEATS]}

    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "model": name, "parity": False,
           "n_clips": len(Z), "k_pca": a.k_pca,
           "reference": "REF-A frozen-DINO trunk reads ego speed at R^2 0.61 "
                        "(D-A5 attributed REF-A's failure to that encoder)",
           "probes": {}}
    print(f"\n  LINEAR PROBE ({name}) — PCA k={a.k_pca} fit on TRAIN clips only,"
          f" lambda on VAL, {len(Z)} clips episode-disjoint")
    print(f"  R^2 is OVER the training-mean predictor: a representation "
          f"carrying nothing reads 0.0000\n")
    print(f"  {'target':<10}{'features':<20}{'R2':>9}  {'CI95':<22}"
          f"{'lam':>7}{'n':>7}{'d':>6}")
    print("  " + "-" * 82)
    for tname, Y in TARGETS.items():
        for fname, X in FEATS.items():
            Xa = [x[:len(y)] for x, y in zip(X, Y)]
            Ya = [y[:len(x)] for x, y in zip(Xa, Y)]
            r2, lo, hi, lam, nrow, d = probe(Xa, Ya, a.k_pca)
            res["probes"].setdefault(tname, {})[fname] = {
                "r2": round(r2, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "lambda": lam, "n_train_rows": nrow, "d_features": d}
            print(f"  {tname:<10}{fname:<20}{r2:>+9.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]{lam:>7g}{nrow:>7d}{d:>6d}")
        print()

    sp = res["probes"]["speed"]
    # the trunk column is named differently depending on which columns were
    # requested; pick whichever is present rather than KeyError-ing after the
    # whole (expensive) table has already been computed and printed.
    lat = next(sp[k] for k in ("latent", "z_op (post-readout)") if k in sp)
    px = sp["pixel (floor)"]
    res["verdict"] = (
        f"THE LATENT ENCODES EGO SPEED (R^2 {lat['r2']:+.4f}, CI {lat['ci95']}) "
        f"vs raw-pixel floor {px['r2']:+.4f}. The information IS in the trunk, "
        f"so a predictor scoring EM<0 on the full latent is failing a target "
        f"that carries signal -- the prediction TARGET is the problem, not the "
        f"encoder."
        if lat["ci95"][0] > 0.20 else
        f"THE LATENT DOES NOT LINEARLY ENCODE EGO SPEED (R^2 {lat['r2']:+.4f}, "
        f"CI {lat['ci95']}; pixel floor {px['r2']:+.4f}). With the power problem "
        f"fixed (n={lat['n_train_rows']} rows on d={lat['d_features']}), this is "
        f"now a real negative and points at the ENCODER.")
    print(f"  VERDICT: {res['verdict']}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
