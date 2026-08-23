"""H2 classifier — STEP 3 (pod2, GPU): train the attention head on FROZEN v1 encoder features.

This trains a HEAD, not a backbone: the encoder never sees a gradient (its features arrive
pre-computed from `h2c_features.py`). Six arms, all pre-registered in `PRE_REGISTRATION.md §4`:

    head_img_ego  PRIMARY  8-step window of 2048-d encoder states  + ego (v, a_pre)
    head_img               encoder states only
    head_ego               ego only, same head capacity
    heur_ego               NOT learned — a 2-threshold (v, a_pre) rule family swept on a grid
    random@rate / always / never    computed in `h2c_eval.py`, they need no training

and, per `PRE_REGISTRATION.md §1.1` (C12), the two CONJUNCTS of the composite target are trained
as their own models UNCONDITIONALLY:

    T_off   areq_off >= tau*      — an agent the encoder CANNOT see requires braking
    T_seen  areq_seen < tau*      — nothing the encoder CAN see requires braking

Model selection (epochs, pos_weight) is by 5-fold GROUPED cross-validation inside TRAIN, grouped by
CHUNK. The held-out CONFIRM side is never read here — this script does not even compute a metric on
it; it only writes scores for `h2c_eval.py`.

usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 h2c_train.py \
     --feats /workspace/h2clf/feats --bundle /workspace/h2clf/bundle --out /workspace/h2clf/run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

TAU = 0.5
W = 8                      # 0.8 s of context — the world model's own predictor window
SPEED_SCALE = 10.0         # v1's hard contract (MODEL_REGISTRY 1.2); reused so the head's ego
                           # channel is on the same scale the trunk was trained with
ACC_SCALE = 5.0


# --------------------------------------------------------------------------------- the head
class SensorNeedHead(nn.Module):
    """Attention over an 8-step window of frozen states -> per-camera independent Bernoulli.

    Never a softmax over mixed axes (`H2_SUBSTRATE §C.1`: the 5-way maneuver-softmax defect).
    `n_out` logits, each its own sigmoid.
    """

    def __init__(self, d_img: int = 0, d_ego: int = 0, d: int = 256, n_layers: int = 2,
                 n_heads: int = 4, n_out: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_img or d_ego
        self.d_img, self.d_ego = d_img, d_ego
        self.in_proj = nn.Linear(d_img + d_ego, d)
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=dropout,
                                       batch_first=True, norm_first=True, activation="gelu"),
            num_layers=n_layers)
        self.pool_q = nn.Linear(d, 1)                       # attention pooling over time
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d, n_out))

    def forward(self, x):                                    # x [B, W, d_img+d_ego]
        h = self.in_proj(x) + self.pos
        h = self.blocks(h)
        a = torch.softmax(self.pool_q(h), dim=1)             # [B, W, 1]
        return self.out(self.norm((a * h).sum(1)))


# --------------------------------------------------------------------------------- data
def load_split(feats_dir, meta, side):
    """Assemble windowed inputs + targets for one side of the pre-registered split."""
    IMG, EGO, Y, CLIP, EXTRA = [], [], [], [], []
    for m in meta:
        if m["side"] != side:
            continue
        p = os.path.join(feats_dir, f"clip_{m['k']:04d}.npz")
        if not os.path.exists(p):
            continue                                          # dropped by the alignment floor
        z = np.load(p)
        F = z["feats"].astype(np.float32)                     # [T, 2048]
        j = z["j"].astype(np.int64)                           # episode frame index per label row
        idx = np.clip(j[:, None] - np.arange(W - 1, -1, -1)[None, :], 0, F.shape[0] - 1)
        IMG.append(F[idx])                                    # [n, W, 2048]
        v = z["ego_v"].astype(np.float32)
        ap = z["alon_pre"].astype(np.float32)
        # ego history over the same window, taken from the LABEL grid (10 Hz, same clock as j)
        r = np.arange(len(j))
        ridx = np.clip(r[:, None] - np.arange(W - 1, -1, -1)[None, :], 0, len(j) - 1)
        EGO.append(np.stack([v[ridx] / SPEED_SCALE, ap[ridx] / ACC_SCALE], -1))   # [n, W, 2]
        oL = z["areq_off_L_res"].astype(np.float32)
        oR = z["areq_off_R_res"].astype(np.float32)
        se = z["areq_seen_res"].astype(np.float32)
        tL = (oL >= TAU) & (se < TAU)
        tR = (oR >= TAU) & (se < TAU)
        Y.append(np.stack([tL, tR], -1).astype(np.float32))
        CLIP.append(np.full(len(j), m["k"], np.int64))
        EXTRA.append(np.stack([
            np.maximum(oL, oR) >= TAU,                        # T_off   (conjunct 1)
            se < TAU,                                         # T_seen  (conjunct 2)
            (z["ego_v"] >= 3.0) & (z["alon_fut_min"] <= -2.0),   # R2 (behavioural response)
            z["junction"], z["lane_change"],
            (z["areq_off_Lr_res"] >= TAU) & (se < TAU),        # residual-scope L
            (z["areq_off_Rr_res"] >= TAU) & (se < TAU),        # residual-scope R
            np.full(len(j), m["encoder_seen"]),
        ], -1).astype(np.float32))
    return (np.concatenate(IMG), np.concatenate(EGO), np.concatenate(Y),
            np.concatenate(CLIP), np.concatenate(EXTRA),
            [m for m in meta if m["side"] == side
             and os.path.exists(os.path.join(feats_dir, f"clip_{m['k']:04d}.npz"))])


def make_x(img, ego, arm, mu, sd, dev):
    parts = []
    if arm in ("head_img_ego", "head_img"):
        parts.append((torch.from_numpy(img).to(dev) - mu) / sd)
    if arm in ("head_img_ego", "head_ego"):
        parts.append(torch.from_numpy(ego).to(dev))
    return torch.cat(parts, -1)


def ap_score(y, s):
    """Average precision = sum_k (R_k - R_{k-1}) * P_k over the realised PR points.

    Stated exactly because a metric NAME is not a metric DEFINITION (C1): this is the
    step-interpolated AP (`sklearn.average_precision_score`'s definition), NOT the trapezoidal
    area under a smoothed PR curve, and NOT AUROC."""
    o = np.argsort(-s, kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / max(yt.sum(), 1e-12)
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))


def train_one(Xtr, Ytr, Xva, cfg, dev, seed=0, epochs=40, log=None):
    torch.manual_seed(seed)
    head = SensorNeedHead(d_img=cfg["d_img"], d_ego=cfg["d_ego"], d=cfg.get("d", 256),
                          dropout=cfg.get("dropout", 0.1), n_out=Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    pw = torch.tensor(cfg["pos_weight"], device=dev, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    bs = cfg["batch"]
    curves = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for b in range(0, n, bs):
            sl = perm[b:b + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(head(Xtr[sl]), Ytr[sl])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(sl)
        head.eval()
        with torch.no_grad():
            sv = torch.cat([torch.sigmoid(head(Xva[b:b + 4096]))
                            for b in range(0, Xva.shape[0], 4096)]).cpu().numpy()
        curves.append(sv)
        if log is not None:
            log.append({"epoch": ep, "train_loss": tot / n})
    return head, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--plan", default="primary", choices=["primary", "c12fix"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    meta = json.load(open(os.path.join(args.bundle, "h2c_meta.json")))

    t0 = time.time()
    tr = load_split(args.feats, meta, "TRAIN")
    ho = load_split(args.feats, meta, "HELDOUT")
    IMGtr, EGOtr, Ytr, CLtr, EXtr, Mtr = tr
    IMGho, EGOho, Yho, CLho, EXho, Mho = ho
    print(f"[train] TRAIN {IMGtr.shape} {int(Ytr.max(1).sum())} pos / "
          f"{len(set(CLtr.tolist()))} clips | HELDOUT {IMGho.shape} "
          f"{int(Yho.max(1).sum())} pos / {len(set(CLho.tolist()))} clips "
          f"({time.time()-t0:.0f}s)", flush=True)

    # feature standardisation — TRAIN statistics only, never the held-out side
    mu = torch.from_numpy(IMGtr.reshape(-1, IMGtr.shape[-1]).mean(0)).to(dev)
    sd = torch.from_numpy(IMGtr.reshape(-1, IMGtr.shape[-1]).std(0) + 1e-5).to(dev)

    TARGETS = {
        "trigger": (Ytr, Yho, ["cam_left", "cam_right"]),
        "T_off": (EXtr[:, :1], EXho[:, :1], ["T_off"]),
        "T_seen": (EXtr[:, 1:2], EXho[:, 1:2], ["T_seen"]),
        # AMENDMENT (see H2_CLASSIFIER.md): `T_seen` is a ~97 %-POSITIVE target, so training it
        # with BCE + pos_weight (a rare-positive recipe) up-weights the MAJORITY class and the
        # resulting head says nothing about the rare, informative side. `NOT_T_seen` — "an agent
        # the encoder CAN see requires braking" — is the same question posed as a rare-positive
        # target, and at ~3.3 % it is 5x better powered than the composite. Diagnostic only; it
        # is NOT an arm in the primary comparison.
        "NOT_T_seen": (1.0 - EXtr[:, 1:2], 1.0 - EXho[:, 1:2], ["NOT_T_seen"]),
    }
    ARMS = {"head_img_ego": dict(d_img=2048, d_ego=2), "head_img": dict(d_img=2048, d_ego=0),
            "head_ego": dict(d_img=0, d_ego=2)}
    PLANS = {
        "primary": ([("trigger", a) for a in ARMS]
                    + [("T_off", "head_img_ego"), ("T_seen", "head_img_ego")]),
        # the corrected C12 diagnostic, plus its ego-only control so the same
        # image-vs-ego question is answered on the well-powered target
        "c12fix": [("NOT_T_seen", "head_img_ego"), ("NOT_T_seen", "head_img"),
                   ("NOT_T_seen", "head_ego")],
    }
    PLAN = PLANS[args.plan]
    # Model selection grid. Only capacity + class weight are searched, and ONLY on TRAIN CV —
    # 169 training positives against a ~2 M-parameter head is an overfitting regime, so the
    # CV is what stops it, not a held-out peek.
    GRID = [{"pos_weight": pw, "d": d_, "dropout": 0.2, "lr": 3e-4, "batch": 512}
            for pw in (20.0, 100.0) for d_ in (128, 256)]

    chunks = sorted({m["chunk"] for m in Mtr})
    fold_of = {c: i % args.folds for i, c in enumerate(chunks)}
    clip_fold = {m["k"]: fold_of[m["chunk"]] for m in Mtr}
    foldid = np.array([clip_fold[k] for k in CLtr])
    print(f"[train] grouped CV folds by chunk: "
          f"{ {f: sorted(c for c in chunks if fold_of[c] == f) for f in range(args.folds)} }",
          flush=True)

    OOF, HOS, SEL, LOGS = {}, {}, {}, {}
    for tgt, arm in PLAN:
        key = f"{arm}|{tgt}"
        Yt, Yh, names = TARGETS[tgt]
        cfg0 = dict(ARMS[arm])
        Xtr_all = make_x(IMGtr, EGOtr, arm, mu, sd, dev)
        Xho_all = make_x(IMGho, EGOho, arm, mu, sd, dev)
        Ytr_t = torch.from_numpy(Yt).to(dev)
        best = None
        for gi_, g in enumerate(GRID):
            cfg = {**cfg0, **g}
            oof = np.zeros_like(Yt)
            per_epoch = np.zeros((args.epochs, *Yt.shape), dtype=np.float32)
            for f in range(args.folds):
                m_tr = foldid != f
                m_va = ~m_tr
                _, curves = train_one(Xtr_all[m_tr], Ytr_t[m_tr], Xtr_all[m_va], cfg, dev,
                                      epochs=args.epochs)
                for e, sv in enumerate(curves):
                    per_epoch[e][m_va] = sv
            aps = [np.mean([ap_score(Yt[:, c], per_epoch[e][:, c]) for c in range(Yt.shape[1])])
                   for e in range(args.epochs)]
            e_star = int(np.argmax(aps))
            oof = per_epoch[e_star]
            cand = {"cfg": cfg, "epoch": e_star + 1, "cv_ap": float(aps[e_star]),
                    "cv_ap_curve": [round(float(a), 5) for a in aps]}
            print(f"[cv] {key} grid{gi_} pw={g['pos_weight']} -> best epoch {e_star+1} "
                  f"CV-AP {aps[e_star]:.4f}", flush=True)
            if best is None or cand["cv_ap"] > best["cv_ap"]:
                best, OOF[key] = cand, oof
        # final model: retrained on ALL of TRAIN at the CV-selected config/epoch
        log = []
        head, curves = train_one(Xtr_all, Ytr_t, Xho_all, best["cfg"], dev,
                                 epochs=best["epoch"], log=log)
        HOS[key] = curves[-1]
        SEL[key] = {**best, "target_names": names, "n_params": sum(p.numel()
                                                                   for p in head.parameters())}
        LOGS[key] = log
        torch.save({"head": head.state_dict(), "cfg": best["cfg"], "arm": arm, "target": tgt,
                    "W": W, "target_names": names,
                    "feat_mu": mu.cpu(), "feat_sd": sd.cpu()},
                   os.path.join(args.out, f"head_{arm}_{tgt}.pt"))
        del Xtr_all, Xho_all
        torch.cuda.empty_cache()
        print(f"[final] {key} epoch={best['epoch']} params={SEL[key]['n_params']:,}", flush=True)

    np.savez_compressed(os.path.join(args.out, "scores_heldout.npz"),
                        clip=CLho, Y=Yho, EX=EXho, ego_v=EGOho[:, -1, 0] * SPEED_SCALE,
                        alon_pre=EGOho[:, -1, 1] * ACC_SCALE,
                        **{f"s__{k.replace('|','__')}": v for k, v in HOS.items()})
    np.savez_compressed(os.path.join(args.out, "scores_oof_train.npz"),
                        clip=CLtr, Y=Ytr, EX=EXtr, ego_v=EGOtr[:, -1, 0] * SPEED_SCALE,
                        alon_pre=EGOtr[:, -1, 1] * ACC_SCALE,
                        **{f"s__{k.replace('|','__')}": v for k, v in OOF.items()})
    json.dump({"selection": SEL, "folds": {str(f): sorted(c for c in chunks if fold_of[c] == f)
                                           for f in range(args.folds)},
               "n_train_windows": int(IMGtr.shape[0]), "n_heldout_windows": int(IMGho.shape[0]),
               "window": W, "epochs_max": args.epochs, "grid": GRID,
               "train_logs": LOGS, "wallclock_s": round(time.time() - t0, 1)},
              open(os.path.join(args.out, "train_summary.json"), "w"), indent=2)
    print(f"[train] done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
