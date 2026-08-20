"""E-TRUNK-2 — does the v6 trunk CONTAIN decodable environment information?

⛔ THIS IS NOT E-TRUNK-1'S QUESTION, AND CONFLATING THEM IS THE ERROR THIS FILE
EXISTS TO END. E-TRUNK-1 asks whether a predictor beats persistence at
predicting the FUTURE FIELD — dynamics predictability, dominated by the scale
and variance structure of the representation. A CONSTANT representation is
perfectly predictable and carries nothing. The PI's question is prior and
different: **is the environment information THERE, decodably?**

⭐ THE DESIGN, and the fourth arm is the one that makes it conclusive.
A decodability number alone is uninterpretable, so every arm is scored against
a floor, a leak control and a reference, on the SAME frames and folds:

    v6 cells   16 x 128 = 2048     the DEPLOYED operative latent (= d_op)
    v6 tokens  640 x 768           the v6 encoder BEFORE the readout
    dino tokens 640 x d            encoder reference at MATCHED granularity
    dino pooled -> 16 x 128        ⭐ is the POOLING itself fatal, regardless
                                      of encoder? This is the discriminating cell
    C-EGO      speed, yaw, accel   car-following statistics with NO perception
    C-PIXEL    downsampled frame   trivial appearance floor
    C-MEAN     constant            absolute floor (R^2 = 0 by construction)

Reading:
  v6tok ~ dinotok AND both collapse pooled -> the READOUT is the defect
  v6tok << dinotok                         -> the v6 ENCODER is weak
  v6cell ~ v6tok, both ~ C-EGO             -> never encoded at all

⚠️ WHY C-EGO IS NOT OPTIONAL. `lead_gap_m` is derived from obstacle.offline
cuboids, so it is a genuine environment target — but ego speed PREDICTS lead gap
through plain car-following. An arm that merely matches C-EGO has demonstrated
nothing about perception. Same family as the sitclf leak: ask what an input
could reconstruct the target from without doing the task.

PROTOCOL (binding):
  * folds are EPISODE-DISJOINT, never window-disjoint. A window and its
    near-duplicate neighbour may not straddle train/test — the REF-A I-JEPA
    lesson, where ~80 % of val sat inside train and made the number unusable.
  * ridge lambda is chosen by an INNER episode-disjoint split on the TRAIN fold
    only. Choosing it on the test fold is the same leak wearing a hyperparameter.
  * d >> n on the token arms (491,520 vs 5,617), so a primal ridge is degenerate.
    The DUAL (Gram) form is used, which is EXACT and identical to primal ridge,
    with kernel-centering so the train-mean subtraction is not skipped.
  * intervals are the paired episode-cluster bootstrap over per-episode scores.

TIER: T0-DIAGNOSTIC. Decodability is a representation property, never driving
performance, and no number here may be quoted as a capability claim.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))

CACHE = SP / os.environ.get("TRUNK_CACHE", "sp2/cache_tok20000_s4/latents.pt")
TARGETS = SP / "sp2/e_trunk2_targets.jsonl"
EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"

REGRESSION = ("lead_gap_m", "nearest_any_m", "n_agents_log", "occluded_frac")
BINARY = ("left_occupied", "right_occupied", "vru_ahead")
#: ⛔ EXCLUDED ON PURPOSE, not silently: 95.8 % positive, so a constant
#: predictor scores 0.958 and the target cannot separate arms.
DEGENERATE = ("lead_present",)

#: ⚠️ WIDENED 2026-08-20. The first pass used 1e-2..1e5 and 68 folds
#: selected the MAXIMUM, i.e. the inner CV wanted more shrinkage than the
#: grid allowed. Saturating at the top is not fatal for a NULL (max
#: shrinkage = the constant predictor = R^2 -> 0, which is the null
#: itself) but a truncated grid is not a defensible selection, so the
#: range now spans 12 decades and edge-hits are reported in the JSON.
LAMBDAS = tuple(10.0 ** k for k in range(-4, 9))
N_FOLDS = 5
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# feature arms
# --------------------------------------------------------------------------- #
def load_v6(kind: str):
    """-> (keys, X) with X float32 [n, d]. `kind` in {'cells','tokens'}."""
    obj = torch.load(CACHE, map_location="cpu", weights_only=False)
    keys, vecs = [], []
    for r in obj["rows"]:
        v = r["cells"] if kind == "cells" else r["tokens"]
        if v is None:
            continue
        keys.append((r["clip_id"], int(r["frame_idx"])))
        vecs.append(v.reshape(-1).to(torch.float16))
    X = torch.stack(vecs).numpy()
    del vecs, obj
    return keys, X


def pool_tokens_like_readout(X, grid=(16, 40), out=(4, 4), d=768):
    """Reproduce the v6 readout's PARAMETER-FREE 40x average pool.

    ⭐ This is what makes the `dino pooled` arm meaningful: the same spatial
    destruction applied to a different encoder. `readout.py` pools a 16x40 token
    grid to 4x4 cells with no parameters, then projects 768 -> 128. The pool is
    the lossy part and it is the part reproduced here; the projection is a
    learned linear map, which a LINEAR PROBE can absorb, so omitting it does not
    advantage either arm.
    """
    n = X.shape[0]
    t = torch.from_numpy(X).reshape(n, grid[0], grid[1], d).permute(0, 3, 1, 2)
    p = torch.nn.functional.adaptive_avg_pool2d(t.float(), out)
    return p.reshape(n, -1).numpy().astype(np.float16)


def ego_features(keys):
    """C-EGO — speed, yaw rate, accel at the frame. NO perception whatsoever."""
    cache: dict[str, torch.Tensor] = {}
    rows = []
    for cid, f in keys:
        if cid not in cache:
            p = EPS / f"{cid}.v2ep.pt"
            cache[cid] = (torch.load(p, map_location="cpu",
                                     weights_only=False)["poses"]
                          if p.exists() else None)
        po = cache[cid]
        if po is None:
            rows.append([np.nan] * 5)
            continue
        T = po.shape[0]
        i = min(max(f, 1), T - 2)
        v = float(po[i, 3]); h = float(po[i, 2])
        dv = float(po[i + 1, 3] - po[i - 1, 3]) / 0.2
        dh = float(np.arctan2(np.sin(po[i + 1, 2] - po[i - 1, 2]),
                              np.cos(po[i + 1, 2] - po[i - 1, 2]))) / 0.2
        rows.append([v, dv, dh, v * v, abs(dh) * v])
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# dual (kernel) ridge with episode-disjoint folds
# --------------------------------------------------------------------------- #
def gram(X, chunk=64 * 768):
    """G = X X^T, accumulated in float64 over feature chunks (X may be 5.5 GB)."""
    n, d = X.shape
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(0, d, chunk):
        b = X[:, i:i + chunk].astype(np.float32)
        G += (b @ b.T).astype(np.float64)
    return G


def center_gram(G, tr):
    """Kernel-centering on the TRAIN mean — the dual of subtracting mu from X.

    Skipping this is not cosmetic: an uncentred dual ridge penalises the mean
    component and is NOT equivalent to the primal ridge it claims to be.
    """
    n = G.shape[0]
    Gtr = G[np.ix_(tr, tr)]
    row = G[:, tr].mean(axis=1, keepdims=True)          # X mu^T
    tot = Gtr.mean()                                     # mu^T mu
    return G - row - row.T + tot


def dual_ridge_oof(G, y, ep, folds, lambdas=LAMBDAS):
    """-> out-of-fold predictions, and the lambda chosen per fold.

    lambda is selected on an INNER episode-disjoint split of the TRAIN fold.
    """
    n = len(y)
    pred = np.full(n, np.nan)
    chosen = []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        # inner split: hold out ~1/4 of the TRAIN episodes to pick lambda
        tr_eps = sorted({ep[i] for i in tr})
        cut = max(1, len(tr_eps) // 4)
        inner_te_eps = set(tr_eps[:cut])
        itr = np.array([i for i in tr if ep[i] not in inner_te_eps])
        ite = np.array([i for i in tr if ep[i] in inner_te_eps])
        Gc = center_gram(G, itr)
        ym = y[itr].mean()
        best, best_lam = -np.inf, lambdas[0]
        A = Gc[np.ix_(itr, itr)]
        for lam in lambdas:
            try:
                alpha = np.linalg.solve(A + lam * np.eye(len(itr)), y[itr] - ym)
            except np.linalg.LinAlgError:
                continue
            p = Gc[np.ix_(ite, itr)] @ alpha + ym
            r2 = 1.0 - ((p - y[ite]) ** 2).sum() / ((y[ite] - y[ite].mean()) ** 2).sum()
            if r2 > best:
                best, best_lam = r2, lam
        chosen.append(best_lam)
        # refit on the FULL train fold with the chosen lambda
        Gc = center_gram(G, tr)
        ym = y[tr].mean()
        alpha = np.linalg.solve(Gc[np.ix_(tr, tr)] + best_lam * np.eye(len(tr)),
                                y[tr] - ym)
        pred[te] = Gc[np.ix_(te, tr)] @ alpha + ym
    return pred, chosen


def episode_folds(ep, n_folds=N_FOLDS, seed=0):
    """⛔ EPISODE-disjoint. Window-disjoint folds leak near-duplicate frames."""
    eps = sorted(set(ep))
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    assign = {e: i % n_folds for i, e in enumerate(eps)}
    idx = collections.defaultdict(list)
    for i, e in enumerate(ep):
        idx[assign[e]].append(i)
    return [np.asarray(idx[k]) for k in range(n_folds)]


def r2(pred, y):
    return float(1.0 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def auc(pred, y):
    """Rank AUC; ties averaged."""
    order = np.argsort(pred)
    ranks = np.empty(len(pred), dtype=np.float64)
    ranks[order] = np.arange(1, len(pred) + 1)
    pos, neg = y == 1, y == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def episode_cluster_bootstrap(pred, y, ep, binary, n_boot=N_BOOT, seed=0):
    """Resample EPISODES with replacement and recompute the POOLED statistic.

    ⛔ NOT the mean of per-episode statistics. MEASURED 2026-08-20: averaging
    per-episode R^2 gave C-EGO/lead_gap_m a mean of -182 against a pooled +0.334,
    because a per-episode R^2 divides by THAT EPISODE'S variance and an episode
    with a near-constant lead gap explodes. The doctrine's estimator is the
    episode-CLUSTER bootstrap over the pooled quantity, which is what this is.
    """
    by = collections.defaultdict(list)
    for i, e in enumerate(ep):
        by[e].append(i)
    eps = list(by)
    idx_of = {e: np.asarray(v) for e, v in by.items()}
    stat = auc if binary else r2
    point = stat(pred, y)
    g = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = g.integers(0, len(eps), len(eps))
        sel = np.concatenate([idx_of[eps[j]] for j in pick])
        v = stat(pred[sel], y[sel])
        if np.isfinite(v):
            boots.append(v)
    if not boots:
        return float(point), float("nan"), float("nan")
    b = np.asarray(boots)
    return (float(point), float(np.quantile(b, 0.025)),
            float(np.quantile(b, 0.975)))


# --------------------------------------------------------------------------- #
# export — features go to DISK as memmaps, never held whole in RAM
#
# ⛔ WHY. The token array is 5.5 GB and this box has ~15 GB free while the
# E-TRUNK-1 arm runs. Holding it alongside a Gram accumulation is exactly the
# dense-tensor swap that cost 2.5 h on 2026-08-20. The Gram is accumulated by
# reading feature CHUNKS off a memmap, so peak RAM stays ~1 GB.
# --------------------------------------------------------------------------- #
FEAT = SP / "sp2/e_trunk2_feat"


def export_features():
    FEAT.mkdir(parents=True, exist_ok=True)
    obj = torch.load(CACHE, map_location="cpu", weights_only=False)
    rows = [r for r in obj["rows"] if r.get("tokens") is not None]
    n = len(rows)
    keys = [(r["clip_id"], int(r["frame_idx"])) for r in rows]
    (FEAT / "keys.json").write_text(json.dumps(keys), encoding="utf-8")

    d_tok = rows[0]["tokens"].reshape(-1).numel()
    d_cell = rows[0]["cells"].reshape(-1).numel()
    mt = np.lib.format.open_memmap(FEAT / "v6_tokens.npy", mode="w+",
                                   dtype=np.float16, shape=(n, d_tok))
    mc = np.lib.format.open_memmap(FEAT / "v6_cells.npy", mode="w+",
                                   dtype=np.float32, shape=(n, d_cell))
    for i, r in enumerate(rows):
        mt[i] = r["tokens"].reshape(-1).to(torch.float16).numpy()
        mc[i] = r["cells"].reshape(-1).float().numpy()
    mt.flush(); mc.flush()
    del obj, rows, mt, mc
    print(f"exported {n} frames: v6_tokens {d_tok}, v6_cells {d_cell} -> {FEAT}")
    return keys


def export_pooled(grid=(16, 40), out=(4, 4), d=768):
    """v6 tokens pooled EXACTLY as `readout.py` pools them (parameter-free)."""
    keys = json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))
    src = np.load(FEAT / "v6_tokens.npy", mmap_mode="r")
    n = src.shape[0]
    dst = np.lib.format.open_memmap(FEAT / "v6_tokens_pooled.npy", mode="w+",
                                    dtype=np.float32,
                                    shape=(n, out[0] * out[1] * d))
    for i in range(0, n, 256):
        b = np.asarray(src[i:i + 256], dtype=np.float32)
        t = torch.from_numpy(b).reshape(-1, grid[0], grid[1], d).permute(0, 3, 1, 2)
        p = torch.nn.functional.adaptive_avg_pool2d(t, out)
        dst[i:i + 256] = p.reshape(p.shape[0], -1).numpy()
    dst.flush()
    print(f"pooled -> {out[0]}x{out[1]}x{d} = {dst.shape[1]} dims")


def export_pixel(side=(16, 40)):
    """C-PIXEL — the decoded frame, greyscaled and downsampled to the TOKEN grid.

    ⭐ Matched to the token grid on purpose: if raw pixels at the same spatial
    resolution decode the target as well as the trunk does, the trunk has added
    nothing over trivial appearance.
    """
    import io
    from PIL import Image
    keys = json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))
    dst = np.lib.format.open_memmap(FEAT / "c_pixel.npy", mode="w+",
                                    dtype=np.float32,
                                    shape=(len(keys), side[0] * side[1]))
    cur, buf, lens = None, None, None
    miss = 0
    for i, (cid, f) in enumerate(keys):
        if cid != cur:
            p = EPS / f"{cid}.v2ep.pt"
            o = torch.load(p, map_location="cpu", weights_only=False)
            buf, lens, cur = o["jpeg_buf"].numpy(), o["jpeg_len"].tolist(), cid
            offs = np.concatenate([[0], np.cumsum(lens)])
        if f >= len(lens):
            dst[i] = 0.0; miss += 1; continue
        raw = buf[offs[f]:offs[f] + lens[f]].tobytes()
        im = Image.open(io.BytesIO(raw)).convert("L").resize(
            (side[1], side[0]), Image.BILINEAR)
        dst[i] = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    dst.flush()
    print(f"C-PIXEL {side} -> {dst.shape[1]} dims (missing frames: {miss})")


def export_dino(pool_out=(4, 4)):
    """⭐ THE DECIDING CONTROL (the PI's suggestion). Same frames, same folds.

    DINOv3 ViT-L/16 on a 256x640 frame yields a 16x40 patch grid — the SAME grid
    v6 produces — so the readout's parameter-free 40x pool applies identically
    and `dino_pooled` is a true like-for-like of `v6_tokens_pooled`.

    ⚠️ Row order is VERIFIED, not assumed: meta's clip/frame sequence was checked
    equal to keys.json as an ordered list before this was written.
    """
    md = json.loads((SP / "dinov3_fields/meta.json").read_text(encoding="utf-8"))
    keys = [tuple(k) for k in
            json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))]
    order = [(c, f) for c, v in md["clips"].items() for f in v["frames"]]
    if order != keys:
        raise RuntimeError("DINOv3 row order != probe key order — refusing to "
                           "build a silently misaligned arm")
    n = len(keys)
    d_grid, d_emb = md["clips"][order[0][0]]["shape"][1:]
    tok = np.lib.format.open_memmap(FEAT / "dino_tokens.npy", mode="w+",
                                    dtype=np.float16, shape=(n, d_grid * d_emb))
    pol = np.lib.format.open_memmap(FEAT / "dino_pooled.npy", mode="w+",
                                    dtype=np.float32,
                                    shape=(n, pool_out[0] * pool_out[1] * d_emb))
    i = 0
    for cid in md["clips"]:
        a = np.load(SP / f"dinov3_fields/{cid}.npy")          # [k, 640, 1024]
        k = a.shape[0]
        tok[i:i + k] = a.reshape(k, -1).astype(np.float16)
        t = torch.from_numpy(a.astype(np.float32)).reshape(k, 16, 40, d_emb)
        t = t.permute(0, 3, 1, 2)
        q = torch.nn.functional.adaptive_avg_pool2d(t, pool_out)
        pol[i:i + k] = q.reshape(k, -1).numpy()
        i += k
    tok.flush(); pol.flush()
    print(f"dino_tokens {tok.shape}  dino_pooled {pol.shape}  (rows {i})")


def gram_memmap(path, chunk_cols=49152):
    X = np.load(path, mmap_mode="r")
    n, d = X.shape
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(0, d, chunk_cols):
        b = np.asarray(X[:, i:i + chunk_cols], dtype=np.float32)
        G += (b @ b.T).astype(np.float64)
    return G


ARMS = (("C-MEAN", None),
        ("C-EGO", "c_ego.npy"),
        ("C-PIXEL", "c_pixel.npy"),
        ("v6_cells", "v6_cells.npy"),
        ("v6_tokens_pooled", "v6_tokens_pooled.npy"),
        ("v6_tokens", "v6_tokens.npy"),
        ("dino_pooled", "dino_pooled.npy"),
        ("dino_tokens", "dino_tokens.npy"))


def main() -> None:
    if not (FEAT / "keys.json").exists():
        export_features()
        export_pooled()
        export_pixel()
        keys = [tuple(k) for k in
                json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))]
        np.save(FEAT / "c_ego.npy", ego_features(keys))
    keys = [tuple(k) for k in
            json.loads((FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]

    tgt = {}
    for line in TARGETS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tgt[(r["clip_id"], int(r["frame_idx"]))] = r

    folds = episode_folds(ep)
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060 / CPU ridge)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "decodability of ENVIRONMENT properties, NOT dynamics "
                       "predictability (that is E-TRUNK-1)",
           "run_stamp": json.loads((CACHE.parent / "sp1_meta.json")
                                   .read_text(encoding="utf-8")).get("run_stamp"),
           "n_frames": len(keys), "n_episodes": len(set(ep)),
           "n_folds": N_FOLDS, "excluded_targets": {
               t: "degenerate base rate" for t in DEGENERATE},
           "arms": {}}

    for arm, fn in ARMS:
        if fn is None:                     # C-MEAN: R2 = 0 / AUC = 0.5 exactly
            out["arms"][arm] = {"dims": 0, "note":
                                "constant predictor; R2=0 and AUC=0.5 BY "
                                "CONSTRUCTION — the absolute floor"}
            print("=== " + arm + " === (floor by construction)")
            continue
        p = FEAT / fn
        if not p.exists():
            print(f"  [skip] {arm}: {fn} absent")
            continue
        print(f"\n=== {arm} ===", flush=True)
        gp = FEAT / f"gram_{arm}.npy"
        if gp.exists():
            G = np.load(gp)
            print(f"  (gram cached {G.shape})", flush=True)
        else:
            G = gram_memmap(p)
            np.save(gp, G)
        out["arms"][arm] = {"dims": int(np.load(p, mmap_mode="r").shape[1]),
                            "targets": {}}
        for name in REGRESSION + BINARY:
            y_all = np.array([tgt.get(k, {}).get(name, np.nan) for k in keys],
                             dtype=np.float64)
            ok = np.isfinite(y_all)
            if ok.sum() < 500:
                out["arms"][arm]["targets"][name] = {"skipped": "n<500",
                                                     "n": int(ok.sum())}
                continue
            idx = np.nonzero(ok)[0]
            remap = {v: i for i, v in enumerate(idx)}
            sub_folds = [np.array([remap[i] for i in f if i in remap])
                         for f in folds]
            Gs = G[np.ix_(idx, idx)]
            ys, eps = y_all[idx], [ep[i] for i in idx]
            pred, lams = dual_ridge_oof(Gs, ys, eps, sub_folds)
            binary = name in BINARY
            score, lo, hi = episode_cluster_bootstrap(pred, ys, eps, binary)
            out["arms"][arm]["targets"][name] = {
                "metric": "AUC" if binary else "R2",
                "estimator": "episode-cluster bootstrap of the pooled statistic",
                "point": round(score, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "n": int(ok.sum()), "n_episodes": len(set(eps)),
                "lambdas": lams,
                "lambda_at_grid_edge": sum(l in (LAMBDAS[0], LAMBDAS[-1])
                                           for l in lams)}
            print(f"  {name:<16} {'AUC' if binary else 'R2 '} "
                  f"{score:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"n={int(ok.sum())}", flush=True)
        del G

    dest = SP / "e_trunk2_probe.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
