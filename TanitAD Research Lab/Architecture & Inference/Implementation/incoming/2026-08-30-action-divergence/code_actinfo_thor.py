"""MM-E12 — IS THE ACTION CHANNEL REDUNDANT GIVEN THE SCENE?

⭐ THE QUESTION, and it may undercut MM-E11 before that arm finishes. MM-E10 measured
the predictor to be ACTION-DEAF. Two mechanisms explain that, and they demand
OPPOSITE fixes:

  (A) WIRING/OBJECTIVE — the model *could* use actions but was never asked to
      (O1 at weight 0). ⇒ MM-E11 (turn O1 on) is the right lever.
  (B) REDUNDANCY — the action carries almost no information the SCENE does not
      already carry, so ignoring it is CORRECT inference, not a defect.
      ⇒ ⛔ O1 CANNOT HELP: it would ask the model to respond to something it can
      already infer from the input it has. The lever would be the ACTION
      REPRESENTATION, not the loss.

**Why (B) is a live hypothesis and not a stretch:** the programme's own record says
our action channel IS REALISED MOTION, correlating r = 0.9988 with the pose change
it accompanies. An action that is a near-deterministic function of the transition is
close to a relabelling of the target.

THE TEST. Fit a ridge predicting the action at the last context step from the
LATENT WINDOW ALONE (no action input). Cross-fitted, held-out R².

  high R^2  => the scene ALREADY DETERMINES the action  => (B), MM-E11 undercut
  low  R^2  => the action carries independent information the model is ignoring
               => (A), MM-E11 is aimed correctly

CONTROLS — each must read its known value or the panel is VOID:
  C0 CONSTANT      : predict from a constant column   -> R^2 must be ~0.000
  C1 SHUFFLED-Z    : rows of z shuffled against a     -> R^2 must collapse to ~C0.
                     Without it, a high R^2 could be leakage or overfit, not
                     determination. ⚠️ Shuffle by ROLL, never permutation.
  n and d PRINTED  : n << d is underpowered BY CONSTRUCTION, not a negative.
"""
import glob
import io
import json
import os
import sys

import numpy as np
import torch

ST = "/home/nvidia/TanitAD/stack"
sys.path.insert(0, ST)

VAL = os.environ.get("AI_CORPUS",
                     "/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl")
ARMS = os.environ.get("AI_ARMS", "postrain30k,emao14_30k").split(",")
OUT = os.environ.get("AI_OUT", "/home/nvidia/staging/actinfo.json")
N_CLIPS = int(os.environ.get("AI_NCLIPS", "40"))
F_MAX = 80
N_STACK = 3
K_FOLDS = 5
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]


def frames_of(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    return d, raw, off, len(lens)


def encode_clip(world, path, dev, max_frames):
    from PIL import Image
    d, raw, off, n = frames_of(path)
    n = min(n, max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    Z, B = [], 16
    with torch.no_grad():
        for s in range(0, n, B):
            chunk = []
            for i in range(s, min(s + B, n)):
                idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)
            Z.append(world.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z), d["actions"].float()[:n]


def ridge_r2(X, Y, folds=K_FOLDS):
    """Cross-fitted held-out R^2. Lambda chosen on a FIT-SIDE inner split only —
    never on the scored fold (the 2026-08-22 lesson: tuning on the scored split
    manufactures a result)."""
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cuts = np.array_split(idx, folds)
    preds = np.zeros_like(Y)
    for f in range(folds):
        te = cuts[f]
        tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
        inner = max(1, int(0.8 * len(tr)))
        tr_a, tr_b = tr[:inner], tr[inner:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xva, Xte = (X[tr_a] - mu) / sd, (X[tr_b] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        best, best_l = None, None
        for lam in LAMBDAS:
            A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
            W = np.linalg.solve(A, Xtr.T @ (Y[tr_a] - ym))
            e = float(((Xva @ W + ym - Y[tr_b]) ** 2).mean())
            if best is None or e < best:
                best, best_l = e, lam
        A = X[tr]; A = (A - mu) / sd
        M = A.T @ A + best_l * np.eye(A.shape[1])
        W = np.linalg.solve(M, A.T @ (Y[tr] - ym))
        preds[te] = Xte @ W + ym
    ss_res = ((Y - preds) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot)


def main():
    import tanitad
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clips = sorted(glob.glob(os.path.join(VAL, "*.v2ep.pt")))[:N_CLIPS]
    res = {"_evidence_class": "MEASURED (ours; Thor)", "eval_tier": "T0-DIAGNOSTIC",
           "hypothesis": "MM-E12 (does the scene determine the action?)",
           "tanitad_imported_from": tanitad.__file__,
           "n_clips": len(clips), "arms": {}}
    for arm in ARMS:
        p = f"/home/nvidia/v7tiny/{arm}/ckpt.pt"
        if not os.path.exists(p):
            print(f"  {arm}: NO CKPT", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        world, _g, _s = load_trunk_auto(ck, dev, ckpt_path=p)
        W = int(world.window)
        Zs, As, Cs = [], [], []
        for ci, c in enumerate(clips):
            z, act = encode_clip(world, c, dev, F_MAX)
            n = min(len(z) - W, len(act) - W)
            for i in range(0, n, max(1, n // 3)):
                Zs.append(z[i + W - 1].numpy())          # last context latent
                As.append(act[i + W - 1].numpy())        # the action at that step
                Cs.append(ci)                            # CLIP id, for the control
        X = np.asarray(Zs, dtype=np.float64)
        Y = np.asarray(As, dtype=np.float64)
        C = np.asarray(Cs)
        n, d = X.shape
        # PCA to keep n >> d — n << d is underpowered by construction
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # ⛔ d must stay well under the EFFECTIVE n, which for a clip-level
        # control is the number of CLIPS, not the number of rows.
        k = min(24, len(np.unique(C)) // 4)
        Xp = Xc @ Vt[:k].T
        r2 = ridge_r2(Xp, Y)
        r2_const = ridge_r2(np.ones((n, 1)), Y)
        # ⛔ C1 MUST SHUFFLE AT CLIP LEVEL. A row-level roll does NOT break the
        # association because ~8 rows come from the SAME clip and are strongly
        # autocorrelated — measured: a row-roll retained 79% of the apparent
        # signal, i.e. the control could not fail. Permuting whole clips is the
        # only shuffle that destroys scene<->action correspondence here.
        rngc = np.random.default_rng(0)
        uniq = np.unique(C)
        perm = uniq.copy(); rngc.shuffle(perm)
        remap = {int(a): int(b) for a, b in zip(uniq, perm)}
        order = np.argsort([remap[int(c)] * 10_000 + k for k, c in enumerate(C)])
        r2_shuf = ridge_r2(Xp, Y[order])
        res["arms"][arm] = {
            "n": int(n), "d_latent": int(d), "d_pca": int(k),
            "R2_action_from_scene": [round(float(v), 5) for v in r2],
            "C0_constant": [round(float(v), 5) for v in r2_const],
            "C1_clip_shuffled": [round(float(v), 5) for v in r2_shuf],
        }
        print(f"[{arm}] n={n} d_pca={k}", flush=True)
        print(f"   R2(action | scene) = {[round(float(v),4) for v in r2]}", flush=True)
        print(f"   C0 constant        = {[round(float(v),4) for v in r2_const]}", flush=True)
        print(f"   C1 shuffled-z      = {[round(float(v),4) for v in r2_shuf]}", flush=True)
        del world
        torch.cuda.empty_cache()
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("WROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
