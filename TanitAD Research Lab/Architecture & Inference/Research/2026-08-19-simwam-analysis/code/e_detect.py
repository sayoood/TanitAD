"""E-DETECT-1 — can a perception head read VEHICLES off a frozen trunk?

The PI's design (2026-08-21): "why are we not designing and training a
perception/prediction head, it extracts the bounding boxes of vehicles based on
the frozen trunk and predicts their states based on the predicted latent space
states, we can easily supervise it based on the gt object data of the av
dataset."

⭐ WHY THIS INSTRUMENT DIFFERS FROM THE NINE NULLS. Every probe so far collapsed
the scene to ONE SCALAR (`lead_gap_m`, `n_agents_log`, ...) and read it with a
ridge. This decodes a SPATIAL FIELD — a 15 x 8 BEV occupancy grid in metres —
with a structured head that has to say WHERE. A representation can carry "there
are cars over there" in a form no scalar ridge can express.

ARMS. Every arm is a token sequence (N, d) into ONE shared head:

  v6_tokens     640 x 768   the v6 encoder field
  dino_tokens   640 x 1024  encoder reference at matched granularity
  v6_cells       16 x 128   ⭐ THE DEPLOYED LATENT — what pooling costs spatially
  dino_pooled    16 x 1024  reference through v6's own 40x pool
  pixel         640 x 768   ⛔ THE FLOOR: raw 16x16x3 patches, same grid

  prior          -          ⛔⛔ CLOSED FORM: per-fold train-mean occupancy. No
                            features at all. A head that cannot beat this has
                            learned only where cars usually are.
  <arm>_shuf    as arm      ⛔ features permuted ACROSS FRAMES in the eval fold.
                            Must fall to `prior`, or the head is reading
                            something frame-independent.

PROTOCOL, unchanged from E-TRUNK-2: the same 5,617 keys, the same
EPISODE-DISJOINT folds, the same episode-cluster bootstrap of the POOLED
statistic. Only the readout changes.

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_trunk2_probe as T   # noqa: E402  (folds, unchanged)
import e_detect_prep as P    # noqa: E402  (grid geometry + banks)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ANCHOR = False
N_BOOT = 400
EPOCHS = 30
BATCH = 64
D_MODEL = 192
SEED = 0

#: name -> (file, n_tokens, dim, grid)
ARMS: dict[str, tuple[str, int, int, tuple[int, int]]] = {
    "v6_tokens":   ("e_trunk2_feat/v6_tokens.npy",   640, 768,  (16, 40)),
    "dino_tokens": ("e_trunk2_feat/dino_tokens.npy", 640, 1024, (16, 40)),
    "v6_cells":    ("e_trunk2_feat/v6_cells.npy",     16, 128,  (1, 16)),
    # ⭐ v6 tokens through the 40x pool but WITHOUT the learned 768->128
    # projection. Isolates the POOL from the PROJECTION: if this matches
    # `v6_cells`, the projection costs nothing and the pool is the whole story.
    "v6_tokens_pooled": ("e_trunk2_feat/v6_tokens_pooled.npy", 16, 768, (1, 16)),
    "dino_pooled": ("e_trunk2_feat/dino_pooled.npy",  16, 1024, (1, 16)),
    "pixel":       ("detect/pixels.npy",             640, 768,  (16, 40)),
    # ⭐ raw patches through v6's OWN 4x4 pool. THE MECHANISM CONTROL: v6's only
    # spatial objective (O3, MaskedCellPredictor) operates on 16 cells and NEVER
    # on the 640 tokens. If content sits where the pressure is, v6_cells should
    # beat THIS while v6_tokens merely matches `pixel`.
    "pixel_pooled": ("detect/pixels_pooled.npy",      16, 768,  (1, 16)),
    # ⛔ instrument-validity control; see e_detect_oracle.py. NOT a trunk.
    "oracle":      ("detect/oracle.npy",             640, 64,   (16, 40)),
    # ⭐⭐ THE GEOMETRY CEILING. The oracle through v6's OWN parameter-free
    # 40x pool (16x40 -> 4x4). PERFECT perception delivered through the DEPLOYED
    # readout, so whatever this scores is the BEST any encoder could reach
    # via that readout.
    # MEASURED 2026-08-21 -> AP 0.2414 [0.2208, 0.2620], against `prior` 0.1242.
    # ⛔ SO THE CEILING DOES **NOT** EXPLAIN v6's NULL, and the hedge above was
    # the right way to write it: the readout permits 0.2414 and `v6_cells`
    # reaches 0.0888 — 37% of what its own readout allows. The pooling cost is
    # real (-34.3% on the oracle, -24.8% on DINOv3) and separately worth fixing,
    # but the content is already absent BEFORE the pool: `v6_tokens` at full 640
    # resolution (0.0923) is indistinguishable from raw pixels (0.0912).
    # Retracted as an explanation in RETRACTION_LOG C130.
    "oracle_pooled": ("detect/oracle_pooled.npy",     16, 64,   (1, 16)),
}


class OccHead(nn.Module):
    """Cross-attention BEV decoder. IDENTICAL across arms bar the input proj.

    120 learned BEV-cell queries attend into the token field. This is the
    smallest head that can express "there is a vehicle at (x, y)" while being
    forced to fetch the evidence from somewhere in the field.

    ⭐ ANCHORED MODE (`prior_logit` given). The head then predicts a RESIDUAL
    over the closed-form prior and its final layer is ZERO-INITIALISED, so at
    step 0 it IS the prior exactly. This is the repo's own zero-init discipline
    (H19 / `gstr_film`), and it matters here for a measured reason: unanchored,
    `v6_cells` scored AP 0.0888 against the prior's 0.1242 — i.e. training on
    features moved it BELOW what no features at all achieve. A comparison whose
    baseline the arms can fall beneath answers "did it overfit?" and not "does
    the representation carry vehicles?". Anchored, the floor is the starting
    point and any lift is attributable to the features.

    ⚠️ DECLARED, NOT SILENT: the anchor was added AFTER seeing arms fall below
    the floor. Both variants are run and both are reported; the unanchored run
    is the pre-registered one.
    """

    def __init__(self, d_in: int, n_tok: int, n_cell: int = P.N_CELL,
                 d: int = D_MODEL, layers: int = 2, heads: int = 4,
                 prior_logit: torch.Tensor | None = None):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, n_cell, d) * 0.02)
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True)
             for _ in range(layers)])
        self.ln_kv = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                           nn.Linear(d * 2, d)) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.anchored = prior_logit is not None
        if self.anchored:
            self.register_buffer("prior_logit", prior_logit.clone())
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.inp(x) + self.pos
        q = self.q.expand(x.shape[0], -1, -1)
        for at, lq, lkv, ff in zip(self.attn, self.ln_q, self.ln_kv, self.ff):
            q = q + at(lq(q), lkv(kv), lkv(kv), need_weights=False)[0]
            q = q + ff(q)
        z = self.out(q).squeeze(-1)                  # (B, n_cell) logits
        return z + self.prior_logit if self.anchored else z


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP by the recall-increment definition. y in {0,1}, s any real score."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="stable")
    ys = y[o].astype(np.float64)
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with ties averaged — O(n log n), threshold-free.

    ⚠️ The tie handling is FULLY VECTORISED on purpose. A per-element Python
    loop here is not merely slow, it is pathological for exactly the arm that
    matters most: `prior` emits only 120 distinct scores over 674,040 cells, so
    the tied-run structure is extreme. Ranks are averaged within each run via a
    cumulative sum over run boundaries — no Python-level iteration."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    n = len(s)
    o = np.argsort(s, kind="stable")
    ss = s[o]
    bnd = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
    lo, hi = bnd[:-1], bnd[1:]
    csum = np.r_[0.0, np.cumsum(np.arange(1, n + 1, dtype=np.float64))]
    avg = (csum[hi] - csum[lo]) / (hi - lo)
    ranks = np.empty(n, dtype=np.float64)
    ranks[o] = np.repeat(avg, hi - lo)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cell_centres() -> np.ndarray:
    xs = (np.arange(P.X_BINS) + 0.5) * (P.X_MAX / P.X_BINS)
    ys = (np.arange(P.Y_BINS) + 0.5) * (2 * P.Y_HALF / P.Y_BINS) - P.Y_HALF
    return np.stack([np.repeat(xs, P.Y_BINS), np.tile(ys, P.X_BINS)], 1)


def topk_loc_err(pred: np.ndarray, occ: np.ndarray) -> float:
    """Mean metres from each of the top-k predicted cells to the NEAREST true
    occupied cell, k = that frame's true count. A direct "how far off, in
    metres" reading derived from the same out-of-fold predictions."""
    C = cell_centres()
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    errs = []
    for i in range(len(occ)):
        k = int(occ[i].sum())
        if k == 0:
            continue
        top = np.argpartition(-pred[i], k - 1)[:k]
        true = np.nonzero(occ[i])[0]
        errs.append(D[np.ix_(top, true)].min(1).mean())
    return float(np.mean(errs)) if errs else float("nan")


def cluster_boot(pred: np.ndarray, occ: np.ndarray,
                 rows_by_ep: dict[str, np.ndarray],
                 rng: np.random.Generator) -> dict:
    """Episode-cluster bootstrap of the POOLED AP/AUC.

    ⚠️ The statistic is recomputed on the pooled resampled cells — NEVER a mean
    of per-episode APs, which is the mean-of-split-means error the registry
    warns about (it biases the point estimate, not merely the interval)."""
    names = list(rows_by_ep)
    pt_ap = average_precision(occ.ravel(), pred.ravel())
    pt_auc = auroc(occ.ravel(), pred.ravel())
    aps, aucs = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(names), len(names), replace=True)
        idx = np.concatenate([rows_by_ep[names[j]] for j in pick])
        y2, s2 = occ[idx].ravel(), pred[idx].ravel()
        aps.append(average_precision(y2, s2))
        aucs.append(auroc(y2, s2))
    return {
        "ap": round(pt_ap, 4),
        "ap_ci95": [round(float(np.nanpercentile(aps, 2.5)), 4),
                    round(float(np.nanpercentile(aps, 97.5)), 4)],
        "auc": round(pt_auc, 4),
        "auc_ci95": [round(float(np.nanpercentile(aucs, 2.5)), 4),
                     round(float(np.nanpercentile(aucs, 97.5)), 4)],
    }


def load_arm(name: str) -> np.ndarray:
    f, n, d, _ = ARMS[name]
    a = np.load(SP / "sp2" / f, mmap_mode="r")
    return a.reshape(len(a), n, d) if a.ndim == 2 else a


def run_fold(X: np.ndarray, occ: np.ndarray, tr: np.ndarray, te: np.ndarray,
             n_tok: int, d_in: int, pos_w: float, *, shuffle_eval: bool = False,
             rng: np.random.Generator | None = None,
             anchor: bool = False) -> tuple[np.ndarray, float]:
    torch.manual_seed(SEED)
    pl = None
    if anchor:
        # ⚠️ TRAIN-FOLD ONLY. An anchor built from all rows would leak the eval
        # fold's marginal into the model that is scored on it.
        rate = np.clip(occ[tr].mean(0), 1e-4, 1 - 1e-4)
        pl = torch.from_numpy(np.log(rate / (1 - rate)).astype(np.float32))
    net = OccHead(d_in, n_tok, prior_logit=pl).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    steps = EPOCHS * max(1, len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                total_steps=steps, pct_start=0.2)
    # per-dim standardisation from the TRAIN fold only. Subsampled: the mean of
    # 1024 x n_tok vectors is far inside the noise floor of a per-dim mean.
    g = np.random.default_rng(SEED)
    sub = np.sort(g.choice(tr, min(1024, len(tr)), replace=False))
    ref = np.asarray(X[sub], dtype=np.float32)
    mu = torch.from_numpy(ref.mean((0, 1))).to(DEV)
    sd = torch.from_numpy(ref.std((0, 1)) + 1e-5).to(DEV)
    del ref
    yb = torch.from_numpy(occ.astype(np.float32)).to(DEV)
    pw = torch.tensor(pos_w, device=DEV)

    def batch(ix_sorted: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(X[ix_sorted], dtype=np.float32)).to(DEV)
        return (x - mu) / sd

    gg = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = tr[torch.randperm(len(tr), generator=gg).numpy()]
        net.train()
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            ix = np.sort(perm[i:i + BATCH])
            loss = F.binary_cross_entropy_with_logits(
                net(batch(ix)), yb[ix], pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    net.eval()

    def predict(rows: np.ndarray) -> np.ndarray:
        o = np.zeros((len(rows), P.N_CELL), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                ix = rows[i:i + 128]
                order = np.argsort(ix)
                blk = net(batch(ix[order])).float().cpu().numpy()
                un = np.zeros_like(blk)
                un[order] = blk
                o[i:i + len(ix)] = un
        return o

    # ⭐ TRAIN-FOLD AP — the diagnostic that says WHICH failure a null is.
    # test ~= prior AND train >> test  -> the head OVERFITS: it can express the
    #   task but 130 clips do not support generalisation (instrument-limited).
    # test ~= prior AND train ~= test  -> nothing to fit: the features do not
    #   carry it (representation-limited).
    # ⚠️ Diagnostic only. It is a mean over folds on a TRAIN subsample, NOT the
    # pooled episode-cluster statistic, and must never be quoted as a result.
    tsub = np.sort(g.choice(tr, min(1200, len(tr)), replace=False))
    train_ap = average_precision(occ[tsub].ravel(), predict(tsub).ravel())

    # ⚠️ under shuffle_eval the FEATURES come from a permutation of the eval
    # rows while the TARGETS stay with their own row, so any surviving score is
    # frame-independent structure.
    src = te[rng.permutation(len(te))] if shuffle_eval else te
    return predict(src), train_ap


def score_arm(arm: str, X: np.ndarray, occ: np.ndarray, folds, rows_by_ep,
              pos_w: float, *, shuffle: bool = False, anchor: bool = False
              ) -> tuple[dict, np.ndarray]:
    _, n_tok, d_in, _ = ARMS[arm]
    t0 = time.time()
    pred = np.zeros_like(occ, dtype=np.float32)
    srng = np.random.default_rng(SEED + 17)
    tr_aps = []
    for k, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        pred[te], tap = run_fold(X, occ, tr, te, n_tok, d_in, pos_w,
                                 shuffle_eval=shuffle, rng=srng, anchor=anchor)
        tr_aps.append(tap)
        print(f"    fold {k + 1}/{len(folds)} train_ap {tap:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    rec = {"n_tok": n_tok, "d": d_in,
              "anchored": bool(anchor),
           "params": sum(p.numel() for p in OccHead(d_in, n_tok).parameters()),
           **cluster_boot(pred, occ, rows_by_ep, rng),
           "topk_loc_err_m": round(topk_loc_err(pred, occ), 3),
           "train_ap_mean": round(float(np.mean(tr_aps)), 4),
           "train_ap_folds": [round(float(x), 4) for x in tr_aps],
           "train_ap_note": "DIAGNOSTIC ONLY — fold-mean on a train subsample, "
                            "not the pooled cluster-bootstrap statistic",
           "train_s": round(time.time() - t0, 1)}
    return rec, pred


def main() -> None:
    t_all = time.time()
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    occ = np.load(P.OUT / "occ.npy")
    folds = T.episode_folds(ep)
    tmp: dict[str, list[int]] = {}
    for i, e in enumerate(ep):
        tmp.setdefault(e, []).append(i)
    rows_by_ep = {k: np.array(v) for k, v in tmp.items()}
    base = float(occ.mean())
    pos_w = (1 - base) / base

    argv = sys.argv[1:]
    global ANCHOR
    ANCHOR = "--anchor" in argv
    want = [a for a in argv if not a.startswith("-")] or (["prior"] + list(ARMS))
    res_path = SP / ("e_detect_anchored.json" if ANCHOR else "e_detect.json")
    if res_path.exists() and "--fresh" not in argv:
        out = json.loads(res_path.read_text(encoding="utf-8"))
    else:
        out = {
            "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
            "eval_tier": "T0-DIAGNOSTIC",
            "question": "can a spatial head localise VEHICLES in metres from a "
                        "frozen trunk?",
            "prereg": "TanitAD Research Lab/Architecture & Inference/Research/"
                      "2026-08-19-simwam-analysis/PREREG_E_DETECT_1.md",
            "n_rows": len(keys), "n_episodes": len(rows_by_ep),
            "grid": json.loads(
                (P.OUT / "occ_stats.json").read_text())["grid"],
            "base_rate": round(base, 6),
            "occupied_cells_per_frame": round(float(occ.sum(1).mean()), 4),
            "head": {"d_model": D_MODEL, "layers": 2, "heads": 4,
                     "epochs": EPOCHS, "batch": BATCH, "lr": 3e-4,
                     "pos_weight": round(pos_w, 3)},
            "protocol": "same 5,617 keys / episode-disjoint 5-fold / "
                        "episode-cluster bootstrap of the POOLED statistic "
                        "(never a mean of per-episode scores)",
            "arms": {}}

    def bank(name: str, rec: dict) -> None:
        out["arms"][name] = rec
        out["_wall_s"] = round(time.time() - t_all, 1)
        res_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        tap = rec.get("train_ap_mean")
        print(f"  {name:<16} AP {rec['ap']:.4f} {rec['ap_ci95']}  "
              f"AUC {rec['auc']:.4f}  loc_err {rec['topk_loc_err_m']} m"
              + (f"  train_ap {tap:.4f}" if tap is not None else ""),
              flush=True)

    if "prior" in want:
        pr = np.zeros_like(occ, dtype=np.float32)
        for k, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            pr[te] = occ[tr].mean(0)[None, :]
        bank("prior", {"n_tok": 0, "d": 0, "params": 0,
                       **cluster_boot(pr, occ, rows_by_ep,
                                      np.random.default_rng(SEED)),
                       "topk_loc_err_m": round(topk_loc_err(pr, occ), 3),
                       "note": "closed form; NO features; the floor"})

    for arm in want:
        base_arm = arm[:-5] if arm.endswith("_shuf") else arm
        if base_arm not in ARMS:
            continue
        f = ARMS[base_arm][0]
        if not (SP / "sp2" / f).exists():
            print(f"  [skip] {arm}: {f} absent", flush=True)
            continue
        print(f"  == {arm} ==", flush=True)
        X = load_arm(base_arm)
        rec, pred = score_arm(base_arm, X, occ, folds, rows_by_ep, pos_w,
                              shuffle=arm.endswith("_shuf"), anchor=ANCHOR)
        if arm.endswith("_shuf"):
            rec["note"] = ("features permuted across frames in the eval fold; "
                           "must fall to `prior`")
        else:
            np.save(SP / f"e_detect_pred_{arm}.npy", pred)
        bank(arm, rec)
        del X, pred
        gc.collect()

    print(f"\n-> {res_path}   wall {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
