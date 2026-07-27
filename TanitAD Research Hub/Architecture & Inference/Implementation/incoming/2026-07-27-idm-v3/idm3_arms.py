"""IDM v3 — the arm ladder: REPAIRED LABELS + GEOMETRY CONDITIONING + its controls.

Every arm trains on the same 68 train episodes and scores the SAME 4,195 val
windows, so every contrast is paired. Labels are the v3 REPAIRED set
(`idm3_labels.heading_repair`, v_min = 0.5 m/s) unless the arm name says LEG
(legacy).

  --- label layer -------------------------------------------------------
  R0      v1 RECIPE (k=4, d256, no winsor, no ctx) on REPAIRED labels
  R0LEG   the same recipe on the LEGACY (broken) labels — isolates the repair
  V2R     the shipped v2 recipe (k=8, d128, winsor, clip-ctx) on REPAIRED labels

  --- geometry layer, WITHOUT clip-context ------------------------------
  Because the clip-context token is already a 4096-number description of the
  clip, geometry stacked on top of it has nothing left to explain. The
  informative contrast is therefore geometry vs context vs nothing, all off R0:
  G1n     R0 + 6-feature GEOMETRY token
  G1h     R0 + camera-height-ONLY token (1 feature) — isolates the height
  Ccorpn  R0 + 2-dim CORPUS one-hot          <- CONTROL (knows the dataset)
  Crign   R0 + 3-dim RIG one-hot             <- CONTROL (knows the rig)
  Cshufn  R0 + geometry token, values PERMUTED across clips  <- NEGATIVE CONTROL
  Sctxn   R0 + clip-context token (the v2 lever, for scale reference)

  --- geometry layer, ON TOP of clip-context ----------------------------
  G1      V2R + geometry token
  Ccorp   V2R + corpus one-hot               <- CONTROL
  Cshuf   V2R + shuffled geometry            <- NEGATIVE CONTROL

  --- the physics arm ---------------------------------------------------
  G2      V2R but the speed channel regresses the CAMERA-INDEPENDENT quantity
          PHI = v / (f_eff * cam_h) and multiplies the known geometry back in
          (eq. (1) of idm3_geom.py).
          >> PRE-REGISTERED PREDICTION: G2 FAILS. `idm3_geomtest.py` measured
          the closed-form version of exactly this correction and it made speed
          MAE significantly WORSE (2.960 -> 3.236, CI-separated), with the
          oracle per-clip scale factor tracking cam_h at r = -0.466 — the
          OPPOSITE of the ground-plane sign. G2 is run anyway because the
          learned version can re-weight the feature, and because a
          pre-registered prediction is only worth something if it is scored.

  --- the worst channel -------------------------------------------------
  Dacc    V2R + long_accel as a 21-bin CLASSIFICATION head (quantile bins from
          train), decoded by softmax expectation. Tests whether R2 = -0.240 is
          continuous regression collapsing to the mean of a heavy-tailed,
          near-zero-mean target. The bins are over the LONGITUDINAL axis ALONE —
          our 5-way manoeuvre softmax mixed lateral and longitudinal and
          produced "0 of 881 accelerate"; the axes stay separable here.

Estimator everywhere: taniteval.ci.(paired_)episode_cluster_bootstrap, unit =
episode, B = 2000. `overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L            # noqa: E402
import idm_head as ih           # noqa: E402
import idm3_geom as GEO         # noqa: E402
from idm3_labels import heading_repair, yaw_rate_from   # noqa: E402

DEV = "cuda"
KBUILD = 8
HOR = ih.DEFAULT_HORIZONS
WINSOR = {"speed": 60.0, "yaw_rate": 1.5, "steer": 1.0, "long_accel": 12.0}
V_MIN = 0.5
N_ACC_BINS = 21


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
class IDMHeadV3(nn.Module):
    """v2 head + an optional SIDE-INFO token (geometry / one-hot / shuffled) and
    an optional binned longitudinal-acceleration classifier."""

    def __init__(self, state_dim=2048, d_model=256, depth=3, n_heads=4,
                 window=9, n_scalars=4, horizons=HOR, use_ctx=False,
                 side_dim=0, acc_bins=0):
        super().__init__()
        self.window, self.center = window, window // 2
        self.horizons = tuple(horizons)
        self.use_ctx, self.side_dim, self.acc_bins = use_ctx, side_dim, acc_bins
        self.n_extra = int(use_ctx) + int(side_dim > 0)
        self.in_proj = nn.Linear(state_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window + self.n_extra, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if use_ctx:
            self.ctx_proj = nn.Sequential(nn.Linear(2 * state_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        if side_dim > 0:
            self.side_proj = nn.Sequential(nn.Linear(side_dim, d_model),
                                           nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.scalar_head = nn.Linear(d_model, n_scalars)
        self.traj_head = nn.Linear(d_model, 2 * len(self.horizons))
        if acc_bins:
            self.acc_head = nn.Linear(d_model, acc_bins)

    def forward(self, z, ctx=None, side=None):
        b = z.shape[0]
        x = self.in_proj(z)
        pre = []
        if self.use_ctx:
            pre.append(self.ctx_proj(ctx).unsqueeze(1))
        if self.side_dim > 0:
            pre.append(self.side_proj(side).unsqueeze(1))
        if pre:
            x = torch.cat(pre + [x], 1)
        x = x + self.pos[:, :x.shape[1]]
        x = self.blocks(x)
        h = self.norm(x[:, self.n_extra + self.center])
        out = {"scalars": self.scalar_head(h),
               "traj": self.traj_head(h).reshape(b, len(self.horizons), 2)}
        if self.acc_bins:
            out["acc_logits"] = self.acc_head(h)
        return out


class Std:
    def __init__(self, S):
        self.c, self.s = S.mean(0), S.std(0).clamp_min(1e-6)

    def norm(self, x):
        return (x - self.c.to(x)) / self.s.to(x)


def acc_bin_grid(y: np.ndarray, nb: int, spacing: str):
    """Bin edges + centres for the longitudinal-acceleration head.

    `quantile` puts equal MASS in every bin, which is automatically fine near
    zero for a zero-concentrated target. `symexp` is DreamerV3's spacing
    (Hafner et al. arXiv:2301.04104): uniform in symlog space, i.e. dense near
    zero and exponentially coarse in the tails. Both are run because the
    literature warns that UNIFORM bins over a heavy-tailed near-zero-mean
    target put all the mass in 1-2 bins, letting the head satisfy
    cross-entropy while remaining a mean-predictor — the pathology re-expressed
    in a new coordinate system rather than fixed.
    """
    if spacing == "symexp":
        lo, hi = np.percentile(y, 0.05), np.percentile(y, 99.95)
        m = max(abs(lo), abs(hi))
        u = np.linspace(-np.log1p(m), np.log1p(m), nb + 1)
        edges = np.sign(u) * (np.expm1(np.abs(u)))
    else:
        edges = np.quantile(y, np.linspace(0, 1, nb + 1))
    edges = np.unique(edges)
    if edges.size < nb + 1:                      # degenerate ties -> pad
        edges = np.linspace(edges[0], edges[-1], nb + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    e = edges.copy()
    e[0], e[-1] = -1e9, 1e9
    return centres.astype(np.float64), e.astype(np.float64)


def hl_gauss_targets(y: np.ndarray, edges: np.ndarray, sigma_ratio=0.75):
    """HL-Gauss soft targets: a Gaussian centred on the scalar target,
    INTEGRATED over each bin (Imani & White 2018; Farebrother et al. 2024).
    sigma = sigma_ratio * (mean bin width)."""
    fin = edges[1:-1]                       # the finite interior edges
    lo = np.concatenate([[fin[0] - 10 * (fin[1] - fin[0])], fin])
    hi = np.concatenate([fin, [fin[-1] + 10 * (fin[-1] - fin[-2])]])
    sigma = float(sigma_ratio * np.mean(np.diff(fin)))
    s = max(sigma, 1e-6) * float(np.sqrt(2.0))
    yt = torch.from_numpy(y.astype(np.float64)).unsqueeze(1)
    lo_t = torch.from_numpy(lo.astype(np.float64)).unsqueeze(0)
    hi_t = torch.from_numpy(hi.astype(np.float64)).unsqueeze(0)
    # erf via torch so the pod needs no scipy
    p = 0.5 * (torch.erf((hi_t - yt) / s) - torch.erf((lo_t - yt) / s))
    p = p.clamp_min(0.0)
    p = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    return p.float()


def winsorise(S):
    out, frac = S.clone(), {}
    for j, n in enumerate(L.SCALARS):
        frac[n] = float((S[:, j].abs() > WINSOR[n]).float().mean())
        out[:, j] = S[:, j].clamp(-WINSOR[n], WINSOR[n])
    return out, frac


# --------------------------------------------------------------------------- #
# substrate: repaired labels + side info                                       #
# --------------------------------------------------------------------------- #
def repair_labels(setd, v_min=V_MIN):
    """Replace the yaw_rate column with the standstill-repaired derivation.
    PhysicalAI (quaternion heading) is untouched by construction."""
    S = setd["S"].clone()
    n_changed = 0
    for tag in np.unique(setd["eid"]):
        m = setd["eid"] == tag
        po = L.load_ep(tag)["poses"].float()
        yaw = heading_repair(po, v_min)[0] if tag.startswith("cm_") else po[:, 2]
        t = setd["tcen"][torch.from_numpy(m)]
        yr = yaw_rate_from(yaw, t)
        n_changed += int((yr - S[torch.from_numpy(m), 1]).abs().gt(1e-9).sum())
        S[torch.from_numpy(m), 1] = yr
    return S, n_changed


def episode_ctx(setd):
    D = setd["Z"].shape[-1]
    ctx = torch.zeros(setd["Z"].shape[0], 2 * D)
    for tag in np.unique(setd["eid"]):
        m = torch.tensor(setd["eid"] == tag)
        z = L.load_ep(tag)["z"].float()
        ctx[m] = torch.cat([z.mean(0), z.std(0)])
    return ctx


def side_info(setd, kind, norm=None, seed=7):
    """Per-window side-information matrix for the conditioning token."""
    tab = GEO.load_table()
    tags = sorted(set(setd["eid"]))
    if kind == "shuffle":
        rng = np.random.default_rng(seed)
        # permute WITHIN corpus, so the marginal distribution is identical and
        # only the clip<->geometry PAIRING is destroyed
        mp = {}
        for dom in ("pai_", "cm_"):
            sub = [t for t in tags if t.startswith(dom)]
            per = rng.permutation(len(sub))
            for i, t in enumerate(sub):
                mp[t] = sub[per[i]]
        per_tag = {t: GEO.feature_vector(GEO.geom_for_tag(mp[t], tab)) for t in tags}
    elif kind == "geom":
        per_tag = {t: GEO.feature_vector(GEO.geom_for_tag(t, tab)) for t in tags}
    elif kind == "camh":
        per_tag = {t: np.array([GEO.geom_for_tag(t, tab)["cam_h_m"]]) for t in tags}
    elif kind == "rig":
        per_tag = {t: GEO.rig_onehot(GEO.geom_for_tag(t, tab)) for t in tags}
    elif kind == "corpus":
        per_tag = {t: GEO.corpus_onehot(GEO.geom_for_tag(t, tab)) for t in tags}
    else:
        raise ValueError(kind)
    X = np.stack([per_tag[t] for t in setd["eid"]]).astype(np.float32)
    if norm is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-8] = 1.0                       # constant features stay 0
        norm = (mu, sd)
    X = (X - norm[0]) / norm[1]
    return torch.from_numpy(X.astype(np.float32)), norm


def metric_gain_of(setd):
    tab = GEO.load_table()
    g = {t: GEO.geom_for_tag(t, tab) for t in set(setd["eid"])}
    return torch.tensor([g[t]["f_eff_px"] * g[t]["cam_h_m"] for t in setd["eid"]],
                        dtype=torch.float32)


# --------------------------------------------------------------------------- #
def train_arm(cfg, tr, va, seed=0, epochs=50, batch=256, lr=3e-4, wd=0.01):
    torch.manual_seed(seed)
    k = cfg["k"]
    sl = slice(KBUILD - k, KBUILD + k + 1)
    Ztr = tr["Z"][:, sl].to(DEV).float()
    Zva = va["Z"][:, sl].to(DEV).float()
    Str = (tr["S_leg"] if cfg.get("legacy_labels") else tr["S"]).clone()
    meta = {}
    if cfg["winsor"]:
        Str, meta["winsor_frac"] = winsorise(Str)
    Ttr = Str.clone()
    if cfg.get("phys_scale"):                       # eq. (1): regress v/(f*h)
        Ttr[:, 0] = Str[:, 0] / tr["mgain"]
    std = Std(Ttr)
    meta["std_center"] = [float(x) for x in std.c]
    meta["std_scale"] = [float(x) for x in std.s]

    Sd, TJ = Ttr.to(DEV), tr["Traj"].to(DEV)
    ctr = tr["ctx"].to(DEV) if cfg["ctx"] else None
    cva = va["ctx"].to(DEV) if cfg["ctx"] else None
    side_kind = cfg.get("side")
    if side_kind:
        str_, nrm = side_info(tr, side_kind)
        sva_, _ = side_info(va, side_kind, norm=nrm)
        str_, sva_ = str_.to(DEV), sva_.to(DEV)
        meta["side_dim"] = int(str_.shape[1])
        meta["side_kind"] = side_kind
    else:
        str_ = sva_ = None

    acc_bins, acc_soft = 0, None
    if cfg.get("acc_cls"):
        nb = int(cfg.get("acc_bins", N_ACC_BINS))
        y = Str[:, 3].numpy().astype(np.float64)
        centres, edges = acc_bin_grid(y, nb, cfg.get("acc_spacing", "quantile"))
        if cfg.get("acc_loss", "hard") == "hlgauss":
            # HL-Gauss (Imani & White 2018; Farebrother et al. arXiv:2403.03950).
            # A hard/two-hot categorical target is measured in that paper to
            # UNDERPERFORM MSE — the categorical parameterisation is NOT the
            # active ingredient, the GAUSSIAN LABEL SMOOTHING is. sigma is set
            # to 0.75 bin widths, their swept optimum, which is reported to be
            # independent of bin count.
            acc_soft = hl_gauss_targets(y, edges, sigma_ratio=0.75).to(DEV)
            meta["acc_loss"] = "hlgauss(sigma/bin=0.75)"
        else:
            meta["acc_loss"] = "hard_ce"
        ytr = torch.clamp(torch.bucketize(Str[:, 3],
                                          torch.tensor(edges[1:-1],
                                                       dtype=torch.float32)),
                          0, nb - 1).to(DEV)
        # FIDELITY CHECK (validate in both directions): decode the TARGET's own
        # soft/one-hot representation back through the bin centres. That is the
        # ceiling the discretisation itself imposes — if it is well below 1.0,
        # the head cannot beat it no matter how well it predicts, and any
        # "classification failed" verdict would really be a binning failure.
        if acc_soft is not None:
            dec = (acc_soft.cpu().numpy() @ centres)
        else:
            hard = np.clip(np.searchsorted(edges[1:-1], y), 0, nb - 1)
            dec = centres[hard]
        meta["acc_discretisation_ceiling_r2"] = float(
            1 - ((dec - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        centres = torch.tensor(centres, dtype=torch.float32).to(DEV)
        acc_bins = nb
        meta["acc_bins"] = nb
        meta["acc_spacing"] = cfg.get("acc_spacing", "quantile")

    head = IDMHeadV3(state_dim=Ztr.shape[-1], window=2 * k + 1,
                     d_model=cfg.get("dmodel", 256), use_ctx=cfg["ctx"],
                     side_dim=(int(str_.shape[1]) if str_ is not None else 0),
                     acc_bins=acc_bins).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Ztr.shape[0]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    w = torch.tensor(cfg["chan_w"], device=DEV, dtype=torch.float32)
    t0 = time.time()
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            out = head(Ztr[ix], None if ctr is None else ctr[ix],
                       None if str_ is None else str_[ix])
            ps, ts = std.norm(out["scalars"]), std.norm(Sd[ix])
            hl = F.huber_loss(ps, ts, delta=1.0, reduction="none").mean(0)
            loss = (hl * w).sum() / w.sum().clamp_min(1e-6)
            loss = loss + F.smooth_l1_loss(out["traj"] / 10.0, TJ[ix] / 10.0, beta=1.0)
            if acc_bins:
                if acc_soft is not None:        # HL-Gauss: soft-target CE
                    loss = loss - (acc_soft[ix] *
                                   out["acc_logits"].log_softmax(-1)).sum(-1).mean()
                else:
                    loss = loss + F.cross_entropy(out["acc_logits"], ytr[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    meta["train_s"] = time.time() - t0
    meta["params"] = sum(p.numel() for p in head.parameters())
    P = predict(head, Zva, cva, sva_, cfg,
                va["mgain"].to(DEV) if cfg.get("phys_scale") else None,
                centres if acc_bins else None)
    sd_out = {k_: v.cpu() for k_, v in head.state_dict().items()}
    del head
    torch.cuda.empty_cache()
    return P, meta, sd_out


@torch.no_grad()
def predict(head, Z, ctx, side, cfg, mgain=None, centres=None, batch=1024):
    head.eval()
    S, T, A = [], [], []
    for i in range(0, Z.shape[0], batch):
        o = head(Z[i:i + batch],
                 None if ctx is None else ctx[i:i + batch],
                 None if side is None else side[i:i + batch])
        s = o["scalars"].clone()
        if mgain is not None:
            s[:, 0] = s[:, 0] * mgain[i:i + batch]
        if centres is not None:
            A.append((o["acc_logits"].softmax(-1) @ centres).cpu())
        S.append(s.cpu())
        T.append(o["traj"].cpu())
    out = {"S": torch.cat(S).numpy().astype(np.float64),
           "Traj": torch.cat(T).numpy().astype(np.float64)}
    if A:
        out["acc_cls"] = torch.cat(A).numpy().astype(np.float64)
    return out


def eval_preds(P, va, Sgt):
    G = Sgt.numpy().astype(np.float64)
    Akin = va["Akin"].numpy().astype(np.float64)
    dom = va["dom"]
    out = {"channels": {}}
    chans = [(nm, P["S"][:, j], G[:, j]) for j, nm in enumerate(L.SCALARS)]
    chans.append(("long_accel_vs_kinematic", P["S"][:, 3], Akin))
    if "acc_cls" in P:
        chans.append(("long_accel_binned", P["acc_cls"], G[:, 3]))
        chans.append(("long_accel_binned_vs_kinematic", P["acc_cls"], Akin))
    doms = [d for d in ("pai", "cm") if (dom == d).sum() > 0]
    for nm, p, g in chans:
        m = L.chan_metrics(p, g)
        m["per_domain"] = {d: L.chan_metrics(p[dom == d], g[dom == d])
                           for d in doms}
        for d in ("pai", "cm"):                 # keep the schema stable
            m["per_domain"].setdefault(d, {"r2": float("nan"), "n": 0})
        out["channels"][nm] = m
    gt, pr = va["Traj"].numpy().astype(np.float64), P["Traj"]
    de = np.linalg.norm(pr - gt, axis=-1)
    lon, lat = np.abs(pr[..., 0] - gt[..., 0]), np.abs(pr[..., 1] - gt[..., 1])
    out["traj"] = {"ade_2s": float(de.mean()),
                   "de_per_horizon": [float(x) for x in de.mean(0)],
                   "lon_mae_2s": float(lon[:, -1].mean()),
                   "lat_mae_2s": float(lat[:, -1].mean()),
                   "lat_p90_at_2s": float(np.percentile(lat[:, -1], 90))}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[
        "R0LEG", "R0", "V2R",
        "Sctxn", "G1n", "G1h", "Ccorpn", "Crign", "Cshufn",
        "G1", "Ccorp", "Cshuf", "G2", "Dacc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="/workspace/idm3/out/arms_v3.json")
    ap.add_argument("--save-ckpt", default="")
    ap.add_argument("--corpus", default="both", choices=("both", "pai", "cm"),
                    help="'pai' = the DECISIVE within-corpus geometry test: "
                         "train AND evaluate on PhysicalAI only, where corpus "
                         "identity is constant and camera height still varies "
                         "1.245-1.607 m. Any gain there cannot be corpus "
                         "memorisation, because there is only one corpus.")
    a = ap.parse_args()

    tr_tags, va_tags = L.split_tags()
    if a.corpus != "both":
        tr_tags = [t for t in tr_tags if t.startswith(a.corpus + "_")]
        va_tags = [t for t in va_tags if t.startswith(a.corpus + "_")]
        log(f"CORPUS-RESTRICTED to {a.corpus}: {len(tr_tags)} train / "
            f"{len(va_tags)} val episodes")
    tr = L.build_set(tr_tags, k=KBUILD, stride=1, want_seq=True)
    va = L.build_set(va_tags, k=KBUILD, stride=2, want_seq=True)
    for s in (tr, va):
        s["Akin"] = (s["Vseq"][:, KBUILD + 1] - s["Vseq"][:, KBUILD - 1]) / (2 * ih.DT)
        s["S_leg"] = s["S"].clone()
        s["S"], nch = repair_labels(s)
        s["mgain"] = metric_gain_of(s)
        log(f"repaired {nch} yaw labels of {s['S'].shape[0]} windows")
    tr["ctx"], va["ctx"] = episode_ctx(tr), episode_ctx(va)
    log(f"train {tuple(tr['Z'].shape)} / val {tuple(va['Z'].shape)}")

    base = dict(k=4, winsor=False, ctx=False, chan_w=[1, 1, 1, 1], dmodel=256)
    v2r = dict(base, k=8, winsor=True, ctx=True, dmodel=128)
    ARMS = {
        "R0LEG": dict(base, legacy_labels=True),
        "R0": dict(base),
        "V2R": dict(v2r),
        # geometry WITHOUT clip-context (the informative contrast)
        "Sctxn": dict(base, ctx=True),
        "G1n": dict(base, side="geom"),
        "G1h": dict(base, side="camh"),
        "Ccorpn": dict(base, side="corpus"),
        "Crign": dict(base, side="rig"),
        "Cshufn": dict(base, side="shuffle"),
        # geometry ON TOP of clip-context
        "G1": dict(v2r, side="geom"),
        "Ccorp": dict(v2r, side="corpus"),
        "Cshuf": dict(v2r, side="shuffle"),
        # physics parametrisation (pre-registered to FAIL)
        "G2": dict(v2r, phys_scale=True),
        # the worst channel, as classification
        "Dacc": dict(v2r, acc_cls=True),
        "Dacc0": dict(base, acc_cls=True),
        # the sibling latent-action stream's RANK-1 upgrade to this arm:
        # Farebrother et al. measure that a HARD two-hot categorical target
        # UNDERPERFORMS MSE — the win comes from the Gaussian label smoothing,
        # not from being categorical. So the hard-CE arm above is the CONTROL
        # and these are the treatment.
        "DaccHL": dict(v2r, acc_cls=True, acc_loss="hlgauss", acc_bins=101),
        "DaccHLsx": dict(v2r, acc_cls=True, acc_loss="hlgauss", acc_bins=101,
                         acc_spacing="symexp"),
        "DaccHL0": dict(base, acc_cls=True, acc_loss="hlgauss", acc_bins=101),
        # --- decoupled rotation vs translation ---------------------------
        # PUBLISHED: monocular rotation is recoverable independently of
        # translation and of scale (Nister 2004 five-point; TartanVO's
        # up-to-scale loss; Lee & Civera Rotation-Only BA). A shared 4-channel
        # head therefore lets an ill-posed SPEED target contaminate the
        # rotation gradient for no benefit. v2 already measured that the four
        # channels compete: the MAD standardiser re-weights them by up to 14x
        # and swings yaw and speed in OPPOSITE directions.
        # Hrot supervises ONLY rotation (yaw, steer); Htra ONLY translation
        # (speed, long_accel). No new architecture — just the loss weights.
        "Hrot": dict(v2r, chan_w=[0, 1, 1, 0]),
        "Htra": dict(v2r, chan_w=[1, 0, 0, 1]),
        "HrotG": dict(v2r, chan_w=[0, 1, 1, 0], side="geom"),
        "HtraG": dict(v2r, chan_w=[1, 0, 0, 1], side="geom"),
    }
    res = {"split": {"train_eps": tr_tags, "val_eps": va_tags,
                     "n_train_windows": int(tr["S"].shape[0]),
                     "n_val_windows": int(va["S"].shape[0])},
           "v_min": V_MIN, "arms": {}}
    store = {}
    for arm in a.arms:
        cfg = ARMS[arm]
        res["arms"][arm] = {"cfg": {k_: v for k_, v in cfg.items()}, "seeds": {}}
        Ps, sds = [], None
        Sgt = tr["S_leg"] if cfg.get("legacy_labels") else None
        Vgt = va["S_leg"] if cfg.get("legacy_labels") else va["S"]
        for sd in a.seeds:
            P, meta, sdict = train_arm(cfg, tr, va, seed=sd, epochs=a.epochs)
            ev = eval_preds(P, va, Vgt)
            ev["meta"] = meta
            res["arms"][arm]["seeds"][str(sd)] = ev
            Ps.append(P)
            if sds is None:
                sds = sdict
            c = ev["channels"]
            log("%-7s s%d speed R2 %+.4f MAE %.3f (pai %+.3f cm %+.3f) | yaw "
                "%+.4f (pai %+.3f cm %+.3f) | steer %+.4f | accel %+.4f | "
                "ade %.3f [%.0fs]" % (
                    arm, sd, c["speed"]["r2"], c["speed"]["mae"],
                    c["speed"]["per_domain"]["pai"]["r2"],
                    c["speed"]["per_domain"]["cm"]["r2"], c["yaw_rate"]["r2"],
                    c["yaw_rate"]["per_domain"]["pai"]["r2"],
                    c["yaw_rate"]["per_domain"]["cm"]["r2"], c["steer"]["r2"],
                    c["long_accel"]["r2"], ev["traj"]["ade_2s"],
                    meta["train_s"]))
            if "long_accel_binned" in c:
                log("        %s binned accel R2 %+.4f (kin %+.4f) pai %+.4f cm %+.4f"
                    % (" " * 4, c["long_accel_binned"]["r2"],
                       c["long_accel_binned_vs_kinematic"]["r2"],
                       c["long_accel_binned"]["per_domain"]["pai"]["r2"],
                       c["long_accel_binned"]["per_domain"]["cm"]["r2"]))
        entry = {"S": np.mean([p["S"] for p in Ps], 0),
                 "Traj": np.mean([p["Traj"] for p in Ps], 0)}
        if "acc_cls" in Ps[0]:
            entry["acc_cls"] = np.mean([p["acc_cls"] for p in Ps], 0)
        store[arm] = entry
        if a.save_ckpt and arm == a.save_ckpt:
            torch.save({"state_dict": sds, "cfg": cfg,
                        "geom_features": list(GEO.GEOM_FEATURES)},
                       f"/workspace/idm3/out/idm_head_v3_{arm}.pt")

    tag = a.out.replace(".json", "")
    np.save(f"{tag}_preds.npy", store, allow_pickle=True)
    np.save("/workspace/idm3/out/val_gt_v3.npy",
            {"S": va["S"].numpy(), "S_leg": va["S_leg"].numpy(),
             "Traj": va["Traj"].numpy(), "Akin": va["Akin"].numpy(),
             "eid": va["eid"], "dom": va["dom"]}, allow_pickle=True)
    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
