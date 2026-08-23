"""IDM-v2 -- the candidate ladder, exactly as pre-registered in
PRE_REGISTRATION_IDMV2.md.  Every arm trains on the same 68 train episodes and
scores the SAME 4,195 val windows, so every contrast is paired.

  A0  idm_head_v1.pt (the PERSISTED artifact)          -- the deployed "before"
  B0  v1 RECIPE retrained here                         -- substrate control
  B1  + ROBUST TARGETS   (physical winsorise + median/MAD standardiser)
  B2  + LOG-SPEED        (regress log v, exponentiate)
  B3  + CLIP CONTEXT     (mean/std of z over the whole clip; offline labeler)
  B4  + DERIVED TARGETS  (speed SEQUENCE -> long_accel by differentiation;
                          steer from bicycle geometry; both leave the loss)
  B5  + 17-FRAME WINDOW
  S2..S5  each single change applied to B0 ALONE (de-confounds the ladder)
  P1  B0 + 1 % of yaw labels replaced by +-8 rad/s  -- POSITIVE CONTROL
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                      # noqa: E402
import idm_head as ih                     # noqa: E402

DEV = "cuda"
KBUILD = 8
HOR = ih.DEFAULT_HORIZONS
WINSOR = {"speed": 60.0, "yaw_rate": 1.5, "steer": 1.0, "long_accel": 12.0}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def savgol_taps(width, order=2):
    h = width // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    P = np.linalg.pinv(np.vander(t, order + 1, increasing=True))
    return P[0], P[1]


# --------------------------------------------------------------------------- #
class IDMHeadV2(nn.Module):
    def __init__(self, state_dim=2048, d_model=256, depth=3, n_heads=4,
                 window=9, n_scalars=4, horizons=HOR, use_ctx=False,
                 seq_speed=False):
        super().__init__()
        self.window, self.center = window, window // 2
        self.horizons = tuple(horizons)
        self.use_ctx, self.seq_speed = use_ctx, seq_speed
        self.in_proj = nn.Linear(state_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window + int(use_ctx), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if use_ctx:
            self.ctx_proj = nn.Sequential(nn.Linear(2 * state_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.scalar_head = nn.Linear(d_model, n_scalars)
        self.traj_head = nn.Linear(d_model, 2 * len(self.horizons))
        if seq_speed:
            self.seq_head = nn.Linear(d_model, 1)
        self.register_buffer("dtap", torch.tensor(savgol_taps(window)[1] / ih.DT,
                                                  dtype=torch.float32))

    def forward(self, z, ctx=None):
        b, w, _ = z.shape
        x = self.in_proj(z)
        if self.use_ctx:
            x = torch.cat([self.ctx_proj(ctx).unsqueeze(1), x], 1)
        x = x + self.pos[:, :x.shape[1]]
        x = self.blocks(x)
        off = int(self.use_ctx)
        h = self.norm(x[:, off + self.center])
        out = {"scalars": self.scalar_head(h),
               "traj": self.traj_head(h).reshape(b, len(self.horizons), 2)}
        if self.seq_speed:
            out["vseq_t"] = self.seq_head(self.norm(x[:, off:])).squeeze(-1)
        return out


class Std:
    def __init__(self, S, kind="std"):
        if kind == "std":
            self.c, self.s = S.mean(0), S.std(0).clamp_min(1e-6)
        else:
            self.c = S.median(0).values
            self.s = (1.4826 * (S - self.c).abs().median(0).values).clamp_min(1e-6)
        self.kind = kind

    def norm(self, x):
        return (x - self.c.to(x)) / self.s.to(x)


def winsorise(S):
    out, frac = S.clone(), {}
    for j, n in enumerate(L.SCALARS):
        frac[n] = float((S[:, j].abs() > WINSOR[n]).float().mean())
        out[:, j] = S[:, j].clamp(-WINSOR[n], WINSOR[n])
    return out, frac


# speed <-> transformed-speed
def fwd_v(v, logv):
    return torch.log(v.clamp_min(0.5)) if logv else v


def inv_v(u, logv):
    return torch.exp(u.clamp(-2.0, 5.0)) if logv else u


# --------------------------------------------------------------------------- #
def episode_ctx(setd):
    D = setd["Z"].shape[-1]
    ctx = torch.zeros(setd["Z"].shape[0], 2 * D)
    for tag in np.unique(setd["eid"]):
        m = torch.tensor(setd["eid"] == tag)
        z = L.load_ep(tag)["z"].float()
        ctx[m] = torch.cat([z.mean(0), z.std(0)])
    return ctx


def train_arm(cfg, tr, va, seed=0, epochs=50, batch=256, lr=3e-4, wd=0.01):
    torch.manual_seed(seed)
    k, logv = cfg["k"], cfg["logspeed"]
    sl = slice(KBUILD - k, KBUILD + k + 1)
    Ztr = tr["Z"][:, sl].to(DEV).float()
    Zva = va["Z"][:, sl].to(DEV).float()
    Str = tr["S"].clone()
    meta = {}
    if cfg.get("inject_yaw_outliers"):
        g = torch.Generator().manual_seed(1234 + seed)
        n0 = Str.shape[0]
        idx = torch.randperm(n0, generator=g)[:int(0.01 * n0)]
        Str[idx, 1] = 8.0 * torch.where(torch.rand(idx.numel(), generator=g) > .5,
                                        1.0, -1.0)
        meta["injected_outliers"] = int(idx.numel())
    if cfg["winsor"]:
        Str, meta["winsor_frac"] = winsorise(Str)
    Ttr = Str.clone()
    Ttr[:, 0] = fwd_v(Str[:, 0], logv)                 # target space
    # NOTE: with winsor=True the mean/std standardiser is refit on the CLEANED
    # labels, which is the whole point -- comma yaw std 0.429 -> 0.109.  The MAD
    # standardiser is a SEPARATE knob because it also re-weights the loss
    # across channels by up to 14x (measured), which is not a label fix.
    std = Std(Ttr, "robust" if cfg["madstd"] else "std")
    meta["std_kind"] = std.kind
    meta["std_center"] = [float(x) for x in std.c]
    meta["std_scale"] = [float(x) for x in std.s]

    Sd, TJ = Ttr.to(DEV), tr["Traj"].to(DEV)
    ctr = tr["ctx"].to(DEV) if cfg["ctx"] else None
    cva = va["ctx"].to(DEV) if cfg["ctx"] else None
    Vt = fwd_v(tr["Vseq"][:, sl].to(DEV), logv) if cfg["seq"] else None

    head = IDMHeadV2(state_dim=Ztr.shape[-1], window=2 * k + 1,
                     d_model=cfg.get("dmodel", 256),
                     use_ctx=cfg["ctx"], seq_speed=cfg["seq"]).to(DEV)
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
            out = head(Ztr[ix], None if ctr is None else ctr[ix])
            ps, ts = std.norm(out["scalars"]), std.norm(Sd[ix])
            hl = F.huber_loss(ps, ts, delta=1.0, reduction="none").mean(0)
            loss = (hl * w).sum() / w.sum().clamp_min(1e-6)
            loss = loss + F.smooth_l1_loss(out["traj"] / 10.0, TJ[ix] / 10.0, beta=1.0)
            if cfg["seq"]:
                sc = 1.0 if logv else 10.0
                loss = loss + F.smooth_l1_loss(out["vseq_t"] / sc, Vt[ix] / sc,
                                               beta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sch.step()
    meta["train_s"] = time.time() - t0
    meta["params"] = sum(p.numel() for p in head.parameters())
    P = predict(head, Zva, cva, cfg)
    del head
    torch.cuda.empty_cache()
    return P, meta


@torch.no_grad()
def predict(head, Z, ctx, cfg, batch=1024):
    head.eval()
    logv, seq = cfg["logspeed"], cfg["seq"]
    S, T = [], []
    for i in range(0, Z.shape[0], batch):
        zb = Z[i:i + batch]
        o = head(zb) if ctx is None else head(zb, ctx[i:i + batch])
        s = o["scalars"].clone()
        s[:, 0] = inv_v(s[:, 0], logv)
        if seq:
            v = inv_v(o["vseq_t"], logv)                     # [B,W] m/s
            s[:, 0] = v[:, head.center]
            s[:, 3] = v @ head.dtap                          # d/dt  -> m/s^2
        S.append(s.cpu()); T.append(o["traj"].cpu())
    return {"S": torch.cat(S).numpy().astype(np.float64),
            "Traj": torch.cat(T).numpy().astype(np.float64)}


def fit_steer_coefs(kap, steer, dom):
    return {d: tuple(float(x) for x in np.linalg.lstsq(
        np.stack([kap[dom == d], np.ones(int((dom == d).sum()))], 1),
        steer[dom == d], rcond=None)[0]) for d in np.unique(dom)}


def derive_steer(P, dom, coefs):
    kap = P["S"][:, 1] / np.clip(P["S"][:, 0], 3.0, None)
    out = np.zeros_like(kap)
    for d, (a, b) in coefs.items():
        m = dom == d
        out[m] = a * kap[m] + b
    return out


def eval_preds(P, va, steer_der):
    G = va["S"].numpy().astype(np.float64)
    Akin = va["Akin"].numpy().astype(np.float64)
    out = {"channels": {}}
    chans = [(nm, P["S"][:, j], G[:, j]) for j, nm in enumerate(L.SCALARS)]
    chans.append(("steer_derived", steer_der, G[:, 2]))
    chans.append(("long_accel_vs_kinematic", P["S"][:, 3], Akin))
    for nm, p, g in chans:
        m = L.chan_metrics(p, g)
        m["per_domain"] = {d: L.chan_metrics(p[va["dom"] == d], g[va["dom"] == d])
                           for d in ("pai", "cm")}
        out["channels"][nm] = m
    gt, pr = va["Traj"].numpy().astype(np.float64), P["Traj"]
    de = np.linalg.norm(pr - gt, axis=-1)
    lon, lat = np.abs(pr[..., 0] - gt[..., 0]), np.abs(pr[..., 1] - gt[..., 1])
    out["traj"] = {"ade_2s": float(de.mean()),
                   "de_per_horizon": [float(x) for x in de.mean(0)],
                   "lon_mae_per_horizon": [float(x) for x in lon.mean(0)],
                   "lat_mae_per_horizon": [float(x) for x in lat.mean(0)],
                   "lat_p90_at_2s": float(np.percentile(lat[:, -1], 90)),
                   "lon_share_sq_err": float((lon ** 2).sum() /
                                             max((lon ** 2 + lat ** 2).sum(), 1e-12))}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["A0", "B0", "B1", "B2", "B3", "B4", "B5",
                             "S2", "S3", "S4", "S5", "S6", "S7", "P1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="/root/idm2/out/v2_results.json")
    a = ap.parse_args()

    tr_tags, va_tags = L.split_tags()
    tr = L.build_set(tr_tags, k=KBUILD, stride=1, want_seq=True)
    va = L.build_set(va_tags, k=KBUILD, stride=2, want_seq=True)
    for s in (tr, va):
        s["Akin"] = (s["Vseq"][:, KBUILD + 1] - s["Vseq"][:, KBUILD - 1]) / (2 * ih.DT)
    log(f"train {tuple(tr['Z'].shape)} / val {tuple(va['Z'].shape)} "
        f"({len(tr_tags)}/{len(va_tags)} eps)")
    tr["ctx"], va["ctx"] = episode_ctx(tr), episode_ctx(va)

    base = dict(k=4, winsor=False, madstd=False, ctx=False, seq=False,
                logspeed=False, chan_w=[1, 1, 1, 1])
    derived_w = [0, 1, 0, 0]           # speed/accel from seq, steer from geometry
    ARMS = {
        "B0": dict(base),                                       # v1 recipe
        "B1": dict(base, winsor=True),                          # + label winsorise
        "B2": dict(base, winsor=True, logspeed=True),           # + log-speed
        "B3": dict(base, winsor=True, logspeed=True, ctx=True),  # + clip context
        "B4": dict(base, winsor=True, logspeed=True, ctx=True, seq=True,
                   chan_w=derived_w),                            # + derived targets
        "B5": dict(base, k=8, winsor=True, logspeed=True, ctx=True, seq=True,
                   chan_w=derived_w),                            # + 17-frame window
        # single changes off B0, to de-confound the ladder
        "S2": dict(base, logspeed=True),
        "S3": dict(base, ctx=True),
        "S4": dict(base, seq=True, chan_w=derived_w),
        "S5": dict(base, k=8),
        "S6": dict(base, madstd=True),          # MAD standardiser ALONE
        "S7": dict(base, winsor=True, madstd=True),   # the first B1 (both) -- kept
        "P1": dict(base, inject_yaw_outliers=True),
        # ---- the assembled candidate: ONLY the single changes that MEASURED
        # a win off B0 (winsorise + clip context + derived targets).  log-speed
        # is deliberately EXCLUDED -- arm B2 measured it as a regression.
        "V2": dict(base, winsor=True, ctx=True, seq=True, chan_w=derived_w),
        "V2s": dict(base, winsor=True, ctx=True, seq=True, chan_w=derived_w,
                    dmodel=128),
        "V2w": dict(base, k=8, winsor=True, ctx=True, seq=True,
                    chan_w=derived_w, dmodel=128),
        # ---- the candidate that the SINGLE-CHANGE evidence actually supports:
        # winsorise (yaw) + clip context (speed).  No log-speed (B2 regressed),
        # no derived targets (B4 regressed).  Capacity variants because the
        # measured capacity curve says smaller is better.
        "V3": dict(base, winsor=True, ctx=True),
        "V3s": dict(base, winsor=True, ctx=True, dmodel=128),
        "V3w": dict(base, k=8, winsor=True, ctx=True, dmodel=128),
        # extra seeds of the winner, under distinct keys so they merge cleanly
        "V3wB": dict(base, k=8, winsor=True, ctx=True, dmodel=128),
        "V3wC": dict(base, k=8, winsor=True, ctx=True, dmodel=128),
        "V3sB": dict(base, winsor=True, ctx=True, dmodel=128),
    }
    res = {"split": {"train_eps": tr_tags, "val_eps": va_tags,
                     "n_train_windows": int(tr["S"].shape[0]),
                     "n_val_windows": int(va["S"].shape[0])}, "arms": {}}
    Gtr = tr["S"].numpy().astype(np.float64)
    coefs = fit_steer_coefs(Gtr[:, 1] / np.clip(Gtr[:, 0], 3.0, None),
                            Gtr[:, 2], tr["dom"])
    res["steer_geometry_coefs_from_train"] = {k: list(v) for k, v in coefs.items()}
    store = {}

    if "A0" in a.arms:
        d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
        h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
        h.load_state_dict(d["state_dict"]); h.eval()
        h.center = h.window // 2
        Zva = va["Z"][:, KBUILD - 4:KBUILD + 5].to(DEV).float()
        P = predict(h, Zva, None, dict(logspeed=False, seq=False))
        ev = eval_preds(P, va, derive_steer(P, va["dom"], coefs))
        ev["meta"] = {"artifact": "/root/idmval/idm_head_v1.pt",
                      "weights_md5": "fa4462f0b898b036be729c790278b823",
                      "params": 2899724,
                      "note": "trained on pod3 tr_a/tr_b/cm -- has seen NONE of "
                              "these 104 episodes"}
        res["arms"]["A0"] = {"cfg": {}, "seeds": {"persisted": ev}}
        store["A0"] = P
        del Zva; torch.cuda.empty_cache()
        c = ev["channels"]
        log(f"A0 speed R2 {c['speed']['r2']:+.4f} MAE {c['speed']['mae']:.3f} | "
            f"yaw R2 {c['yaw_rate']['r2']:+.4f} nMedAE {c['yaw_rate']['nmedae']:.3f} | "
            f"steer {c['steer']['r2']:+.4f} | accel {c['long_accel']['r2']:+.4f} | "
            f"ade {ev['traj']['ade_2s']:.3f}")

    for arm in [x for x in a.arms if x != "A0"]:
        cfg = ARMS[arm]
        seeds = a.seeds if (arm.startswith("B") or arm.endswith(("B", "C")))             else a.seeds[:2]
        res["arms"][arm] = {"cfg": dict(cfg), "seeds": {}}
        Ps = []
        for sd in seeds:
            P, meta = train_arm(cfg, tr, va, seed=sd, epochs=a.epochs)
            ev = eval_preds(P, va, derive_steer(P, va["dom"], coefs))
            ev["meta"] = meta
            res["arms"][arm]["seeds"][str(sd)] = ev
            Ps.append(P)
            c = ev["channels"]
            log(f"{arm} s{sd} speed R2 {c['speed']['r2']:+.4f} MAE "
                f"{c['speed']['mae']:.3f} | yaw R2 {c['yaw_rate']['r2']:+.4f} "
                f"nMedAE {c['yaw_rate']['nmedae']:.3f} | steer "
                f"{c['steer']['r2']:+.4f}/{c['steer_derived']['r2']:+.4f}der | "
                f"accel {c['long_accel']['r2']:+.4f} "
                f"(kin {c['long_accel_vs_kinematic']['r2']:+.4f}) | ade "
                f"{ev['traj']['ade_2s']:.3f} [{meta['train_s']:.0f}s]")
        store[arm] = {"S": np.mean([p["S"] for p in Ps], 0),
                      "Traj": np.mean([p["Traj"] for p in Ps], 0),
                      "S_seeds": np.stack([p["S"] for p in Ps])}

    tagout = a.out.replace(".json", "")
    np.save(f"{tagout}_preds.npy", store, allow_pickle=True)
    np.save("/root/idm2/out/val_gt.npy",
            {"S": va["S"].numpy(), "Traj": va["Traj"].numpy(),
             "Akin": va["Akin"].numpy(), "eid": va["eid"], "dom": va["dom"]},
            allow_pickle=True)
    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
