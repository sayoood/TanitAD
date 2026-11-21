"""E-DETECT-1B — BOUNDING BOXES off a frozen trunk (the PI's actual ask).

⛔ WHY THIS EXISTS ALONGSIDE THE GRID. The PI asked for a head that "extracts the
bounding boxes of vehicles" and "predicts their states". The grid variant
(`e_detect.py`) answers a strictly narrower question — IS object information
present and localisable — and it gives up three things the box head restores:

  1. **Extent and heading.** A box is (cx, cy, l, w, yaw). The grid measured
     centre-occupancy only, so "can the trunk tell a bus from a hatchback, or
     which way it points" was never asked.
  2. **Continuous metres.** The grid quantises to 4 m, which puts a hard floor
     under any localisation error it can report. Box centres are regressed.
  3. ⭐ **IDENTITY — and this is the one that matters most.** Forecasting object
     STATES at t+k needs per-object correspondence. A grid has none, so the
     grid variant CANNOT express the second half of the PI's design at all.
     Slots can be tracked; cells cannot.

The grid keeps two advantages worth naming, because they are why it ran first:
no matcher (a bad matcher makes a good representation look bad), and a
CLOSED-FORM floor. Here the floor has to be trained — see `const`.

ARMS: identical to `e_detect.py`, plus

  const   ⛔⛔ THE FLOOR. The same head with its input projection FORCED TO
          ZERO, so only the K learned queries carry information. It emits one
          fixed set of boxes for every frame — "where vehicles usually are",
          the box analogue of the grid's mean-occupancy prior. MEASURED there:
          the prior alone reaches 2.5x the base rate, so this floor is not a
          formality. An arm that does not clearly beat `const` has shown
          nothing.

K = 40 slots: MEASURED to cover 100.00% of in-grid instances and 100.00% of
frames. ⚠️ The feasibility pass said "K=16 covers 100%"; that was a
6,000-frame sample, and on the real 5,617 probe rows K=16 covers 96.13%.

PROTOCOL: the same 5,617 keys, the same EPISODE-DISJOINT folds, the same
episode-cluster bootstrap of the POOLED statistic.

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
from scipy.optimize import linear_sum_assignment

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_detect as E        # noqa: E402  (ARMS, load_arm, folds, AP)
import e_detect_prep as P   # noqa: E402
import e_trunk2_probe as T  # noqa: E402

DEV = E.DEV
K_SLOTS = 40
EPOCHS = 30
BATCH = 64
D_MODEL = 192
SEED = 0
N_BOOT = 400
#: matching / reporting thresholds in metres (centre distance)
TAUS = (1.0, 2.0, 4.0)
#: loss weights. Position dominates on purpose: extent and yaw are reported as
#: SECONDARY families and must not be able to buy a better headline.
W_POS, W_EXT, W_YAW = 5.0, 1.0, 1.0
#: Hungarian cost weights (DETR-style: class term + L1 position term)
C_CLS, C_POS = 1.0, 5.0


def build_boxes() -> tuple[np.ndarray, np.ndarray]:
    """-> (boxes [N, K, 5], valid [N, K]).  5 = (cx, cy, l, w, yaw)."""
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    want = {k: i for i, k in enumerate(keys)}
    box = np.zeros((len(keys), K_SLOTS, 5), dtype=np.float32)
    val = np.zeros((len(keys), K_SLOTS), dtype=bool)
    dropped = 0
    for line in P.JOIN.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        i = want.get((r["clip_id"], int(r["frame_idx"])))
        if i is None:
            continue
        veh = [a for a in r.get("agents", ())
               if a.get("cls") in P.VEHICLE
               and P.cell_of(float(a["cx"]), float(a["cy"])) is not None]
        # ⚠️ nearest-first, so a truncation at K drops the FARTHEST vehicles —
        # the ones a monocular camera can least resolve anyway. Silent
        # truncation of the nearest would bias the task toward easy frames.
        veh.sort(key=lambda a: a["cx"] ** 2 + a["cy"] ** 2)
        dropped += max(0, len(veh) - K_SLOTS)
        for j, a in enumerate(veh[:K_SLOTS]):
            box[i, j] = (a["cx"], a["cy"], a.get("l", 4.24), a.get("w", 1.92),
                         a.get("yaw", 0.0))
            val[i, j] = True
    if dropped:
        print(f"  [warn] {dropped} instances beyond K={K_SLOTS} dropped "
              "(farthest-first)")
    return box, val


class BoxHead(nn.Module):
    """DETR-style set head. IDENTICAL across arms bar the input projection.

    `zero_input=True` is the `const` floor: the projection output is zeroed, so
    the cross-attention can only return a constant and the K queries alone
    decide. It is a TRAINED constant-set baseline, not an untrained one.
    """

    def __init__(self, d_in: int, n_tok: int, k: int = K_SLOTS,
                 d: int = D_MODEL, layers: int = 2, heads: int = 4,
                 zero_input: bool = False):
        super().__init__()
        self.zero_input = zero_input
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads,
                                                         batch_first=True)
                                   for _ in range(layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_kv = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                           nn.Linear(d * 2, d)) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.obj = nn.Linear(d, 1)
        #: (cx_norm, cy_norm, log l, log w, sin yaw, cos yaw)
        self.geo = nn.Linear(d, 6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.inp(x)
        if self.zero_input:
            kv = kv * 0.0
        kv = kv + self.pos
        q = self.q.expand(x.shape[0], -1, -1)
        for at, lq, lkv, ff in zip(self.attn, self.ln_q, self.ln_kv, self.ff):
            q = q + at(lq(q), lkv(kv), lkv(kv), need_weights=False)[0]
            q = q + ff(q)
        q = self.norm(q)
        g = self.geo(q)
        cx = torch.sigmoid(g[..., 0]) * P.X_MAX
        cy = torch.tanh(g[..., 1]) * P.Y_HALF
        return self.obj(q).squeeze(-1), torch.stack(
            [cx, cy, g[..., 2], g[..., 3], g[..., 4], g[..., 5]], -1)


def match(pl: torch.Tensor, pg: torch.Tensor, tb: torch.Tensor,
          tv: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hungarian assignment per frame. Cost = class term + L1 centre term."""
    out = []
    prob = torch.sigmoid(pl).detach().cpu().numpy()
    ctr = pg[..., :2].detach().cpu().numpy()
    tbn = tb.cpu().numpy()
    tvn = tv.cpu().numpy()
    for b in range(len(prob)):
        m = int(tvn[b].sum())
        if m == 0:
            out.append((np.zeros(0, int), np.zeros(0, int)))
            continue
        d = np.abs(ctr[b][:, None, :] - tbn[b, :m][None, :, :2]).sum(-1)
        cost = C_CLS * (-prob[b][:, None]) + C_POS * d / 10.0
        r, c = linear_sum_assignment(cost)
        out.append((r, c))
    return out


def loss_fn(pl, pg, tb, tv, pairs):
    """⚠️ FULLY VECTORISED over the batch on purpose.

    The obvious implementation loops over the 64 frames and issues ~4 small CUDA
    kernels each. MEASURED: that costs 12.7 s/epoch on the CHEAPEST arm, which
    extrapolates to ~5 h per token arm — the loss, not the model, would have set
    the experiment's cost. Flattening the matched pairs into one index set makes
    it a handful of kernels per batch regardless of batch size."""
    dev = pl.device
    bi = np.concatenate([np.full(len(r), b, np.int64)
                         for b, (r, _) in enumerate(pairs)])         if pairs else np.zeros(0, np.int64)
    ri = np.concatenate([r for r, _ in pairs]) if pairs else np.zeros(0, int)
    ci = np.concatenate([c for _, c in pairs]) if pairs else np.zeros(0, int)
    tgt = torch.zeros_like(pl)
    cls = F.binary_cross_entropy_with_logits
    if len(bi) == 0:
        return cls(pl, tgt)
    B_ = torch.from_numpy(bi).to(dev)
    R_ = torch.from_numpy(np.asarray(ri, np.int64)).to(dev)
    C_ = torch.from_numpy(np.asarray(ci, np.int64)).to(dev)
    tgt[B_, R_] = 1.0
    pm = pg[B_, R_]                       # [n, 6]
    tm = tb[B_, C_]                       # [n, 5]
    n = len(bi)
    lp = F.l1_loss(pm[:, :2], tm[:, :2], reduction="sum")
    lg = F.l1_loss(pm[:, 2:4], torch.log(tm[:, 2:4].clamp_min(0.1)),
                   reduction="sum")
    cosd = ((pm[:, 4] * torch.sin(tm[:, 4]) + pm[:, 5] * torch.cos(tm[:, 4]))
            / torch.linalg.vector_norm(pm[:, 4:6], dim=-1).clamp_min(1e-6))
    ly = (1 - cosd).sum()
    return (cls(pl, tgt) + W_POS * lp / n / 10.0 + W_EXT * lg / n
            + W_YAW * ly / n)


def evaluate(pl: np.ndarray, pg: np.ndarray, tb: np.ndarray, tv: np.ndarray,
             tau: float) -> dict:
    """Greedy score-ordered matching at a centre-distance threshold.

    Returns per-detection (score, tp) plus per-TP geometry errors, so the
    bootstrap can re-pool them without redoing the matching."""
    sc, tp, ce, ee, ye = [], [], [], [], []
    ngt = np.zeros(len(pl), dtype=np.int64)
    for b in range(len(pl)):
        m = int(tv[b].sum())
        ngt[b] = m
        order = np.argsort(-pl[b])
        used = np.zeros(m, dtype=bool)
        for i in order:
            sc.append(pl[b, i])
            if m == 0:
                tp.append(0.0)
                continue
            d = np.linalg.norm(tb[b, :m, :2] - pg[b, i, :2], axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tau:
                used[j] = True
                tp.append(1.0)
                ce.append(d[j])
                ee.append(np.abs(np.exp(pg[b, i, 2:4]) - tb[b, j, 2:4]).mean())
                yp = np.arctan2(pg[b, i, 4], pg[b, i, 5])
                ye.append(np.abs(np.arctan2(np.sin(yp - tb[b, j, 4]),
                                            np.cos(yp - tb[b, j, 4]))))
            else:
                tp.append(0.0)
    return {"score": np.array(sc, np.float32), "tp": np.array(tp, np.float32),
            "ngt": ngt, "ctr_err": np.array(ce, np.float32),
            "ext_err": np.array(ee, np.float32),
            "yaw_err": np.array(ye, np.float32),
            "per_frame": len(pl[0])}


def ap_from(score: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    t = tp[o]
    ctp = np.cumsum(t)
    prec = ctp / np.arange(1, len(t) + 1)
    return float((prec * t).sum() / n_gt)


def boot(ev: dict, rows_by_ep, rng) -> dict:
    """Episode-cluster bootstrap. AP is recomputed on the POOLED resample; the
    geometry errors are pooled TP means, never means-of-episode-means."""
    names = list(rows_by_ep)
    kpf = ev["per_frame"]
    out = {}
    pt_ap = ap_from(ev["score"], ev["tp"], int(ev["ngt"].sum()))
    reps = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(names), len(names), replace=True)
        rows = np.concatenate([rows_by_ep[names[j]] for j in pick])
        det = (rows[:, None] * kpf + np.arange(kpf)[None, :]).ravel()
        reps.append(ap_from(ev["score"][det], ev["tp"][det],
                            int(ev["ngt"][rows].sum())))
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    out["ap"] = round(pt_ap, 4)
    out["ap_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    for k, arr in (("ctr_err_m", ev["ctr_err"]), ("ext_err_m", ev["ext_err"]),
                   ("yaw_err_deg", np.degrees(ev["yaw_err"]))):
        out[k] = round(float(arr.mean()), 4) if len(arr) else None
    out["n_tp"] = int(ev["tp"].sum())
    out["n_gt"] = int(ev["ngt"].sum())
    out["recall"] = round(float(ev["tp"].sum() / max(ev["ngt"].sum(), 1)), 4)
    return out


def run_fold(X, tb, tv, tr, te, n_tok, d_in, zero_input):
    torch.manual_seed(SEED)
    net = BoxHead(d_in, n_tok, zero_input=zero_input).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    steps = EPOCHS * max(1, len(tr) // BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                              total_steps=steps, pct_start=0.2)
    g = np.random.default_rng(SEED)
    sub = np.sort(g.choice(tr, min(1024, len(tr)), replace=False))
    ref = np.asarray(X[sub], dtype=np.float32)
    mu = torch.from_numpy(ref.mean((0, 1))).to(DEV)
    sd = torch.from_numpy(ref.std((0, 1)) + 1e-5).to(DEV)
    del ref
    TB = torch.from_numpy(tb).to(DEV)
    TV = torch.from_numpy(tv).to(DEV)

    def batch(ix):
        x = torch.from_numpy(np.asarray(X[ix], dtype=np.float32)).to(DEV)
        return (x - mu) / sd

    gg = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = tr[torch.randperm(len(tr), generator=gg).numpy()]
        net.train()
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            ix = np.sort(perm[i:i + BATCH])
            pl, pg = net(batch(ix))
            pairs = match(pl, pg, TB[ix], TV[ix])
            loss = loss_fn(pl, pg, TB[ix], TV[ix], pairs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    net.eval()
    OL = np.zeros((len(te), K_SLOTS), np.float32)
    OG = np.zeros((len(te), K_SLOTS, 6), np.float32)
    with torch.no_grad():
        for i in range(0, len(te), 128):
            ix = te[i:i + 128]
            o = np.argsort(ix)
            a, b = net(batch(ix[o]))
            ua = np.zeros_like(a.cpu().numpy())
            ub = np.zeros_like(b.cpu().numpy())
            ua[o] = a.float().cpu().numpy()
            ub[o] = b.float().cpu().numpy()
            OL[i:i + len(ix)] = ua
            OG[i:i + len(ix)] = ub
    return OL, OG


def main() -> None:
    t0all = time.time()
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    folds = T.episode_folds(ep)
    tmp: dict[str, list[int]] = {}
    for i, e in enumerate(ep):
        tmp.setdefault(e, []).append(i)
    rows_by_ep = {k: np.array(v) for k, v in tmp.items()}

    bp = P.OUT / "boxes.npy"
    if bp.exists():
        tb = np.load(bp)
        tv = np.load(P.OUT / "boxes_valid.npy")
    else:
        tb, tv = build_boxes()
        np.save(bp, tb)
        np.save(P.OUT / "boxes_valid.npy", tv)
    print(f"boxes: K={K_SLOTS}, {int(tv.sum()):,} instances over {len(tb):,} "
          f"frames ({tv.sum(1).mean():.2f}/frame)")

    argv = sys.argv[1:]
    want = [a for a in argv if not a.startswith("-")] or \
        ["const", "v6_cells", "dino_pooled", "oracle_pooled", "oracle",
         "pixel", "v6_tokens", "dino_tokens"]
    res = SP / "e_detect_box.json"
    out = (json.loads(res.read_text(encoding="utf-8"))
           if res.exists() and "--fresh" not in argv else {
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
        "eval_tier": "T0-DIAGNOSTIC",
        "question": "can a BOX head recover vehicle STATES (centre, extent, "
                    "heading) from a frozen trunk?",
        "k_slots": K_SLOTS,
        "k_coverage": "K=40 covers 100.00% of in-grid instances and frames "
                      "(MEASURED on the 5,617 probe rows)",
        "taus_m": list(TAUS),
        "protocol": "same 5,617 keys / episode-disjoint 5-fold / "
                    "episode-cluster bootstrap of the POOLED statistic",
        "arms": {}})

    for arm in want:
        base = "v6_tokens" if arm == "const" else arm
        if base not in E.ARMS:
            continue
        f, n_tok, d_in, _ = E.ARMS[base]
        if not (SP / "sp2" / f).exists():
            print(f"  [skip] {arm}: {f} absent", flush=True)
            continue
        print(f"  == {arm} ==", flush=True)
        X = E.load_arm(base)
        t0 = time.time()
        OL = np.zeros((len(tb), K_SLOTS), np.float32)
        OG = np.zeros((len(tb), K_SLOTS, 6), np.float32)
        for k, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            OL[te], OG[te] = run_fold(X, tb, tv, tr, te, n_tok, d_in,
                                      arm == "const")
            print(f"    fold {k + 1}/{len(folds)} ({time.time() - t0:.0f}s)",
                  flush=True)
        rec = {"n_tok": n_tok, "d": d_in, "zero_input": arm == "const",
               "train_s": round(time.time() - t0, 1), "by_tau": {}}
        for tau in TAUS:
            ev = evaluate(OL, OG, tb, tv, tau)
            rec["by_tau"][f"{tau:g}m"] = boot(ev, rows_by_ep,
                                              np.random.default_rng(SEED))
        a2 = rec["by_tau"]["2m"]
        print(f"  {arm:<16} AP@2m {a2['ap']:.4f} {a2['ap_ci95']}  "
              f"recall {a2['recall']:.3f}  ctr {a2['ctr_err_m']} m  "
              f"ext {a2['ext_err_m']} m  yaw {a2['yaw_err_deg']} deg",
              flush=True)
        out["arms"][arm] = rec
        np.save(SP / f"e_detect_box_pred_{arm}.npy",
                np.concatenate([OL[..., None], OG], -1))
        res.write_text(json.dumps(out, indent=1), encoding="utf-8")
        del X
        gc.collect()
    print(f"\n-> {res}   wall {time.time() - t0all:.0f}s")


if __name__ == "__main__":
    main()
