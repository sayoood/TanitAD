"""E2a — RECOVERY-LOCALIZE. Decompose recovery_ratio = 0.0074 into the four
candidate stages and name the dominant one. ZERO training, zero renderer.

PRE-REGISTERED (written before the run; see E1a_E2a_RESULTS.md §PRE-REGISTRATION):
  S1 PERCEPTION       does the ENCODER's latent move at all under the warp?
                      (relative L2 / cosine of pooled and of the conv fmap the
                      decoder cross-attends, vs offset magnitude, expressed in the
                      scale-free unit "how many frames of real driving is this?")
  S2 REPRESENTABILITY is the lateral offset LINEARLY DECODABLE from that latent?
                      Ridge probe latent -> true (dlat, dpsi), EPISODE-DISJOINT
                      held-out R^2, with a label-shuffled control. THIS IS THE CRUX:
                      if the offset is not in the representation, no recovery
                      objective could ever have worked.
  S3 TRUNCATION       present in the latent but lost by the 2-step diffusion decoder
                      / anchor set? (a) denoise-step sweep {0,2,4,8,16};
                      (b) stage-gain ladder fmap -> anchor posterior -> selected traj;
                      (c) anchor-vocabulary geometric coverage (zero model calls).
  S4 CONDITIONING     present but swamped by ego / nav conditioning? ablate v0
                      (zero / randomised) and vary nav_cmd; also the direct
                      "1 m of image warp vs 1 m/s of v0" plan-sensitivity ratio.

  COMMITTED OUTCOMES:
   BLIND-PERCEPTION  S2 held-out R^2 < ~0.3 on the fmap => the offset is NOT in the
     representation; every recovery objective in the arc was arguing about the wrong
     layer; next action is the aux lane-relative pose head (E2b), and E1b defers.
   PERCEIVABLE       S2 R^2 >= ~0.6 => the information IS there and the planner
     ignores it => the fault is conditioning/objective; E1b becomes the lever.
   UNREPRESENTABLE   S3c: < ~50 % of offset states have a returning anchor => refit
     the anchor vocabulary first.
   TRUNCATION-BOUND  S3a: recovery_ratio rises >= 5x from 2 -> 16 denoise steps =>
     a FREE inference-time improvement exists.
   NULL              all four flat => the decomposition is wrong; record the null.

ATTRIBUTION (stated so the numbers cannot be quoted without their construction):
  rho  = sqrt(max(heldout R^2 of the linear offset probe, 0))  -- the fraction of the
         offset an ORACLE LINEAR readout of this latent could recover: the ceiling
         any downstream objective could reach from this representation.
  R    = the realized recovery_ratio (2 denoise steps, deployed decode path).
  loss_representation = 1 - rho          (ideal 1.0 -> what the latent supports)
  loss_downstream     = rho - R          (what the latent supports -> what is used)
  the downstream loss is then sub-attributed by the MEASURED recoveries:
  truncation   = (R@16steps - R@2steps) / (rho - R)
  conditioning = (R@ego-ablated - R@2steps) / (rho - R)
  anchors      = capped by the MEASURED anchor-coverage fraction
  residual     = the remainder.

recovery_ratio itself is recovery_probe.py VERBATIM: response along the demand
direction at the 0.5 s lookahead / demand magnitude, episode-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import perturb  # noqa: E402  (byte-copy of the instrument's warp; self-checked)
import taniteval_ci as _ci  # noqa: E402  (vendored; pod3's taniteval has no ci.py)
from refb_labels import waypoint_targets  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,  # noqa: E402
                               refc_xl_config)

W = 8
LOOKAHEAD = 0                      # index into horizons (5,10,15,20) -> 0.5 s
PRIMARY = (1.0, 0.0)               # the headline perturbation (the 0.0074)
# BOTH SIGNS are required or the probe cannot learn a SIGNED readout of the offset
# (recovery_probe.py's grid was all-positive; that grid can only be used for the
# ratio, never for a representability probe).
GRID = [(-1.75, 0.0), (-1.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (1.0, 0.0), (1.75, 0.0),
        (0.0, -5.0), (0.0, 5.0), (1.0, 3.0), (-1.0, -3.0)]
DENOISE_STEPS = [0, 2, 4, 8, 16]
_REFC_PRESETS = {"base": refc_config, "small": refc_small_config, "xl": refc_xl_config}
DT = 0.1
CORRIDOR = 1.75


def _apply(cfg, d):
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


def load_refc(ckpt, preset, device):
    cfg = _REFC_PRESETS[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply(cfg, json.loads(cj.read_text()).get("cfg", {}))
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(ck.get("step", -1)), cfg


def _boot(x, eid):
    return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid, n_boot=2000)


@torch.no_grad()
def encode(model, frames):
    """The deployed encode path, exposing BOTH latents the decoder consumes:
    fmap [B,F,g,g] (cross-attended) and pooled [B,F] (aux heads / strategic GRU;
    == fmap.mean((2,3)) by construction)."""
    b, w = frames.shape[:2]
    if model.cfg.hierarchy:
        fmap_all, pooled_all = model.encoder(frames.reshape(b * w, *frames.shape[2:]))
        pooled_seq = pooled_all.reshape(b, w, -1)
        pooled = pooled_seq[:, -1]
        fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
        ctx = model.strategic(pooled_seq)
    else:
        fmap, pooled = model.encoder(frames[:, -1])
        ctx = None
    if model.cfg.graft_imagination:
        fmap, _ = model.imagination(fmap)
    return fmap, pooled, ctx


@torch.no_grad()
def decode(model, fmap, pooled, ctx, v0, nav_cmd, steps):
    """The deployed decode path from cached latents (so the denoise-step sweep and
    every conditioning ablation cost NO encoder forward)."""
    b = pooled.shape[0]
    dev = pooled.device
    nav = torch.nn.functional.one_hot(nav_cmd, 4).to(pooled.dtype)
    v = (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
    m = model.measurement(torch.cat([v, nav], dim=-1))
    man = model.maneuver_head(pooled)
    return model.decoder(fmap, m, ctx=ctx, maneuver_logits=man,
                         target_latent=None, steps=steps)


def ridge_fit_eval(X, y, ep, folds=4, lams=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4),
                   device="cuda", shuffle_control=False, seed=0):
    """EPISODE-DISJOINT ridge probe. X [n,d] float32, y [n,k], ep [n] episode ids.

    Returns held-out R^2 per target (pooled over folds), the held-out regression
    SLOPE of truth on prediction, and the lambda chosen. `shuffle_control` permutes
    y within each episode -> R^2 must collapse to ~0 or the probe is leaking.
    """
    n, d = X.shape
    ep = np.asarray(ep)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    if shuffle_control:
        y = y.copy()
        for u in uniq:
            m = np.flatnonzero(ep == u)
            y[m] = y[rng.permutation(m)]
    fold_of = {u: i % folds for i, u in enumerate(rng.permutation(uniq))}
    fid = np.array([fold_of[e] for e in ep])
    pred = np.full_like(y, np.nan)
    chosen = []
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    for f in range(folds):
        tr = torch.as_tensor(np.flatnonzero(fid != f), device=device)
        te = torch.as_tensor(np.flatnonzero(fid == f), device=device)
        if len(tr) < 10 or len(te) < 5:
            continue
        Xtr, ytr = Xt[tr], yt[tr]
        mu = Xtr.mean(0, keepdim=True)
        ymu = ytr.mean(0, keepdim=True)
        Xc, yc = Xtr - mu, ytr - ymu
        # inner episode-disjoint split for lambda selection
        itr_ep = np.unique(ep[fid != f])
        icut = set(itr_ep[::4].tolist())
        sel = np.array([e not in icut for e in ep[fid != f]])
        sel_t = torch.as_tensor(np.flatnonzero(sel), device=device)
        val_t = torch.as_tensor(np.flatnonzero(~sel), device=device)
        best, best_l = None, lams[0]
        if len(val_t) >= 5:
            A = Xc[sel_t].T @ Xc[sel_t]
            Bv = Xc[sel_t].T @ yc[sel_t]
            eye = torch.eye(d, device=device, dtype=A.dtype)
            for lam in lams:
                Wt = torch.linalg.solve(A + lam * eye, Bv)
                pv = Xc[val_t] @ Wt
                err = float(((pv - yc[val_t]) ** 2).mean())
                if best is None or err < best:
                    best, best_l = err, lam
        A = Xc.T @ Xc
        Bv = Xc.T @ yc
        eye = torch.eye(d, device=device, dtype=A.dtype)
        Wt = torch.linalg.solve(A + best_l * eye, Bv)
        pred[np.flatnonzero(fid == f)] = ((Xt[te] - mu) @ Wt + ymu).cpu().numpy()
        chosen.append(best_l)
        del A, Bv, Wt, eye
        torch.cuda.empty_cache()
    ok = ~np.isnan(pred[:, 0])
    out = {"n": int(ok.sum()), "d": int(d), "folds": folds,
           "lambdas_chosen": chosen, "shuffle_control": bool(shuffle_control),
           "_split": "EPISODE-DISJOINT k-fold (episodes never straddle a fold)"}
    for j in range(y.shape[1]):
        yt_, yp_ = y[ok, j], pred[ok, j]
        ss_res = float(((yt_ - yp_) ** 2).sum())
        ss_tot = float(((yt_ - yt_.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        s = np.polyfit(yp_, yt_, 1) if yp_.std() > 1e-9 else [0.0, 0.0]
        out[f"target{j}"] = {
            "r2_heldout": round(r2, 4),
            "slope_true_on_pred": round(float(s[0]), 4),
            "corr": round(float(np.corrcoef(yt_, yp_)[0, 1]), 4)
            if yp_.std() > 1e-9 else 0.0,
            "rmse": round(float(np.sqrt(((yt_ - yp_) ** 2).mean())), 4),
            "target_std": round(float(yt_.std()), 4)}
    return out


def ridge_dual_fit_eval(X, y, ep, folds=4, lams=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5),
                        device="cuda", shuffle_control=False, seed=0):
    """EPISODE-DISJOINT ridge probe in the DUAL (kernel) form, for d >> n — used
    for the FULL conv fmap (64 tokens x F), i.e. the exact tensor the decoder
    cross-attends. Mathematically identical to primal ridge, O(n^2) not O(d^2)."""
    n, d = X.shape
    ep = np.asarray(ep)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    if shuffle_control:
        y = y.copy()
        for u in uniq:
            m = np.flatnonzero(ep == u)
            y[m] = y[rng.permutation(m)]
    fold_of = {u: i % folds for i, u in enumerate(rng.permutation(uniq))}
    fid = np.array([fold_of[e] for e in ep])
    pred = np.full_like(y, np.nan)
    chosen = []
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    for f in range(folds):
        tr = np.flatnonzero(fid != f)
        te = np.flatnonzero(fid == f)
        if len(tr) < 10 or len(te) < 5:
            continue
        tr_t = torch.as_tensor(tr, device=device)
        te_t = torch.as_tensor(te, device=device)
        mu = Xt[tr_t].mean(0, keepdim=True)
        ymu = yt[tr_t].mean(0, keepdim=True)
        Xtr = Xt[tr_t] - mu
        Xte = Xt[te_t] - mu
        Ktr = (Xtr @ Xtr.T).double()
        Kte = (Xte @ Xtr.T).double()
        ytr = (yt[tr_t] - ymu).double()
        # inner EPISODE-DISJOINT split for lambda
        inner_ep = np.unique(ep[tr])
        hold = set(inner_ep[::4].tolist())
        vmask = np.array([e in hold for e in ep[tr]])
        best, best_l = None, lams[0]
        if vmask.sum() >= 5 and (~vmask).sum() >= 10:
            a = torch.as_tensor(np.flatnonzero(~vmask), device=device)
            v = torch.as_tensor(np.flatnonzero(vmask), device=device)
            Kaa = Ktr[a][:, a]
            Kva = Ktr[v][:, a]
            eye = torch.eye(len(a), device=device, dtype=Kaa.dtype)
            for lam in lams:
                al = torch.linalg.solve(Kaa + lam * eye, ytr[a])
                err = float(((Kva @ al - ytr[v]) ** 2).mean())
                if best is None or err < best:
                    best, best_l = err, lam
            del Kaa, Kva, eye
        eye = torch.eye(len(tr), device=device, dtype=Ktr.dtype)
        al = torch.linalg.solve(Ktr + best_l * eye, ytr)
        pred[te] = (Kte @ al + ymu.double()).float().cpu().numpy()
        chosen.append(best_l)
        del Ktr, Kte, Xtr, Xte, al, eye
        torch.cuda.empty_cache()
    del Xt, yt
    torch.cuda.empty_cache()
    ok = ~np.isnan(pred[:, 0])
    out = {"n": int(ok.sum()), "d": int(d), "folds": folds, "form": "dual/kernel",
           "lambdas_chosen": chosen, "shuffle_control": bool(shuffle_control),
           "_split": "EPISODE-DISJOINT k-fold (episodes never straddle a fold)"}
    for j in range(y.shape[1]):
        yt_, yp_ = y[ok, j], pred[ok, j]
        r2 = 1.0 - float(((yt_ - yp_) ** 2).sum()) / max(
            float(((yt_ - yt_.mean()) ** 2).sum()), 1e-12)
        s = np.polyfit(yp_, yt_, 1) if yp_.std() > 1e-9 else [0.0, 0.0]
        out[f"target{j}"] = {
            "r2_heldout": round(r2, 4),
            "slope_true_on_pred": round(float(s[0]), 4),
            "corr": round(float(np.corrcoef(yt_, yp_)[0, 1]), 4)
            if yp_.std() > 1e-9 else 0.0,
            "rmse": round(float(np.sqrt(((yt_ - yp_) ** 2).mean())), 4),
            "target_std": round(float(yt_.std()), 4)}
    return out


def anchor_coverage(model, poses_last, dlat, corridor=CORRIDOR):
    """S3c — ZERO model calls. From an offset pose (+dlat left), does ANY of the
    N anchors bring the ego back inside the corridor by 2 s? An anchor is an
    ego-frame trajectory [S,2]; executed from the offset pose its lateral position
    relative to the recorded path at horizon s is (dlat + anchor_y[s]).
    Returns per-window: n_returning_anchors, has_returning_anchor."""
    A = model.decoder.anchors.detach().cpu()          # [N,S,2] ego-frame
    ay = A[..., 1]                                     # [N,S] lateral
    d = torch.as_tensor(dlat, dtype=ay.dtype)[:, None, None]   # [B,1,1]
    lat_end = (d + ay[None])[:, :, -1].abs()           # [B,N] |lat| at 2 s
    ret = lat_end < corridor
    return ret.sum(1).numpy(), ret.any(1).float().numpy()


@torch.no_grad()
def run(model, episodes, device, horizons, stride, batch, max_windows, log):
    rows = {"eid": [], "epi": []}
    acc = {}                       # accumulators keyed by name -> list of tensors

    def add(k, v):
        acc.setdefault(k, []).append(v.detach().cpu() if torch.is_tensor(v) else v)

    max_h = max(horizons)
    n_done = 0
    t_start = time.time()
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = poses.shape[0]
        starts = list(range(0, T - W - max_h, stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            frames = torch.stack([fr[t0:t0 + W] for t0 in ch]).to(device)
            last = torch.tensor([t0 + W - 1 for t0 in ch])
            pose_last = poses[last]
            fut = torch.stack([poses[l + 1:l + 1 + max_h] for l in last])
            v0 = pose_last[:, 3].to(device)
            nav0 = torch.zeros(b, dtype=torch.long, device=device)

            fmap0, pooled0, ctx0 = encode(model, frames)
            base_tgt = waypoint_targets(pose_last, fut, horizons)[:, LOOKAHEAD]
            dec0 = {s: decode(model, fmap0, pooled0, ctx0, v0, nav0, s)
                    for s in DENOISE_STEPS}
            base_plan = {s: dec0[s]["traj"][:, LOOKAHEAD].cpu() for s in DENOISE_STEPS}
            base_post = torch.softmax(dec0[2]["anchor_logits"], -1)
            # a reference scale for "how much does this latent move per unit of REAL
            # scene change": one frame of real driving (window shifted by 1 frame)
            fr1 = torch.stack([fr[t0 + 1:t0 + 1 + W] for t0 in ch]).to(device)
            fmap1, pooled1, _ = encode(model, fr1)
            add("ref_dfmap", (fmap1 - fmap0).flatten(1).norm(dim=1)
                / fmap0.flatten(1).norm(dim=1))
            add("ref_dpooled", (pooled1 - pooled0).norm(dim=1) / pooled0.norm(dim=1))
            del fr1, fmap1, pooled1

            # ---- S4 reference: sensitivity to a 1 m/s change in v0 (image fixed)
            dv = decode(model, fmap0, pooled0, ctx0, v0 + 1.0, nav0, 2)
            add("dtraj_per_1mps_v0",
                (dv["traj"][:, LOOKAHEAD].cpu() - base_plan[2]).norm(dim=-1))
            dnav = decode(model, fmap0, pooled0, ctx0, v0,
                          torch.ones_like(nav0), 2)      # nav follow -> left
            add("dtraj_per_navswitch",
                (dnav["traj"][:, LOOKAHEAD].cpu() - base_plan[2]).norm(dim=-1))

            # ---- probe feature rows (base included as delta = 0) ---------------
            for (dl, dyaw) in [(0.0, 0.0)] + GRID:
                dlat = torch.full((b,), float(dl))
                dpsi = torch.full((b,), float(np.radians(dyaw)))
                if dl == 0.0 and dyaw == 0.0:
                    fmapP, pooledP, ctxP = fmap0, pooled0, ctx0
                else:
                    warped = perturb.warp_windows(frames.cpu(), dlat, dpsi).to(device)
                    fmapP, pooledP, ctxP = encode(model, warped)
                    del warped
                tag = f"{dl}_{dyaw}"
                # S1 PERCEPTION
                add(f"dfmap_rel::{tag}", (fmapP - fmap0).flatten(1).norm(dim=1)
                    / fmap0.flatten(1).norm(dim=1))
                add(f"dpooled_rel::{tag}", (pooledP - pooled0).norm(dim=1)
                    / pooled0.norm(dim=1))
                add(f"pooled_cos::{tag}", torch.nn.functional.cosine_similarity(
                    pooledP, pooled0, dim=1))
                # S2 features (pooled + the fmap's column/row marginals; a lateral
                # offset is a HORIZONTAL image shift so the column marginal is its
                # natural carrier, and pooled == the fmap's global spatial mean)
                add(f"feat_pooled::{tag}", pooledP)
                add(f"feat_colmean::{tag}", fmapP.mean(2).flatten(1))
                add(f"feat_rowmean::{tag}", fmapP.mean(3).flatten(1))
                # the FULL conv fmap the decoder cross-attends (64 tokens x F),
                # probed in the dual form because d >> n
                add(f"feat_full::{tag}", fmapP.flatten(1))
                # S3 stage ladder + the recovery_ratio at every denoise step
                rec_tgt = perturb.recovery_targets(pose_last, fut, horizons,
                                                   dlat, dpsi, waypoint_targets
                                                   )[:, LOOKAHEAD]
                demand = rec_tgt - base_tgt
                dnorm = demand.norm(dim=-1).clamp_min(1e-6)
                dhat = demand / dnorm[:, None]
                add(f"demand::{tag}", dnorm)
                for s in DENOISE_STEPS:
                    dp = decode(model, fmapP, pooledP, ctxP, v0, nav0, s)
                    resp = dp["traj"][:, LOOKAHEAD].cpu() - base_plan[s]
                    add(f"resp::{tag}::s{s}", (resp * dhat).sum(-1))
                    if s == 2:
                        post = torch.softmax(dp["anchor_logits"], -1)
                        add(f"dpost_tv::{tag}",
                            0.5 * (post - base_post).abs().sum(-1))
                        add(f"dsel::{tag}",
                            (dp["sel_idx"] != dec0[2]["sel_idx"]).float())
                        add(f"dtraj_full::{tag}",
                            (dp["traj"].cpu() - dec0[2]["traj"].cpu()
                             ).flatten(1).norm(dim=1)
                            / dec0[2]["traj"].cpu().flatten(1).norm(dim=1))
                        # S4 CONDITIONING: same warped image, ego channel ablated
                        for nm, vv in (("v0zero", torch.zeros_like(v0)),
                                       ("v0rand", v0[torch.randperm(b, device=device)])):
                            da = decode(model, fmapP, pooledP, ctxP, vv, nav0, 2)
                            db = decode(model, fmap0, pooled0, ctx0, vv, nav0, 2)
                            r = (da["traj"][:, LOOKAHEAD].cpu()
                                 - db["traj"][:, LOOKAHEAD].cpu())
                            add(f"resp_{nm}::{tag}", (r * dhat).sum(-1))
                if not (dl == 0.0 and dyaw == 0.0):
                    del fmapP, pooledP
                # S3c anchor coverage (zero model calls)
                nret, has = anchor_coverage(model, pose_last, dlat.numpy())
                add(f"anchor_nret::{tag}", torch.as_tensor(nret, dtype=torch.float32))
                add(f"anchor_has::{tag}", torch.as_tensor(has))
            rows["eid"] += [str(ei)] * b
            rows["epi"].append(torch.full((b,), ei))
            n_done += b
            del fmap0, pooled0, dec0
            torch.cuda.empty_cache()
            if max_windows and n_done >= max_windows:
                break
        if log and ei % 5 == 0:
            print(f"[e2a] ep {ei + 1}/{len(episodes)} n={n_done} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        if max_windows and n_done >= max_windows:
            break
    out = {k: torch.cat(v) for k, v in acc.items()}
    out["eid"] = rows["eid"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--val-files", default="")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # geometry self-checks BEFORE any number is produced
    chk = perturb.validate_identity(waypoint_targets)
    assert chk["ok"], f"perturb geometry identity check FAILED: {chk}"

    model, step, cfg = load_refc(args.refc_ckpt, args.refc_preset, device)
    horizons = cfg.trajectory.horizons
    if args.val_files:
        eps = [Path(args.val_dir) / f for f in args.val_files.split(",")]
    else:
        eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    print(f"[e2a] REF-C {args.refc_preset} step {step} | anchors "
          f"{tuple(model.decoder.anchors.shape)} | {len(episodes)} eps | "
          f"horizons {horizons} | dev {device}", flush=True)

    with strict_numerics():
        r = run(model, episodes, device, horizons, args.stride, args.batch,
                args.max_windows or None, log=True)
    eid = r["eid"]
    n = len(eid)
    print(f"[e2a] collected n={n} windows / {len(set(eid))} episodes", flush=True)

    res = {"_experiment": "E2a RECOVERY-LOCALIZE",
           "_estimator": "episode_cluster_bootstrap (taniteval/ci.py, vendored), "
                         "B=2000, over val EPISODES. NEVER overlapping_holdout_se.",
           "refc_ckpt": args.refc_ckpt, "refc_step": step,
           "n_anchors": int(model.decoder.anchors.shape[0]),
           "val_dir": args.val_dir, "n_windows": n, "n_episodes": len(set(eid)),
           "stride": args.stride, "grid": [list(g) for g in GRID],
           "denoise_steps_swept": DENOISE_STEPS,
           "geometry_selfcheck": chk}

    # ---- headline recovery_ratio (recovery_probe.py convention) ---------------
    rr = {}
    for (dl, dyaw) in GRID:
        tag = f"{dl}_{dyaw}"
        dem = r[f"demand::{tag}"].numpy()
        blk = {}
        for s in DENOISE_STEPS:
            resp = r[f"resp::{tag}::s{s}"].numpy()
            blk[f"steps{s}"] = {
                "recovery_ratio": _boot(resp / np.maximum(dem, 1e-6), eid),
                "recovery_ratio_pooled": round(float(resp.sum()
                                                     / max(dem.sum(), 1e-6)), 5),
                "response_m": _boot(resp, eid)}
        blk["demand_m"] = _boot(dem, eid)
        for nm in ("v0zero", "v0rand"):
            resp = r[f"resp_{nm}::{tag}"].numpy()
            blk[f"steps2_{nm}"] = {
                "recovery_ratio": _boot(resp / np.maximum(dem, 1e-6), eid),
                "recovery_ratio_pooled": round(float(resp.sum()
                                                     / max(dem.sum(), 1e-6)), 5)}
        rr[tag] = blk
    res["S0_recovery_ratio"] = rr
    prim = f"{PRIMARY[0]}_{PRIMARY[1]}"
    p2 = rr[prim]["steps2"]["recovery_ratio"]
    print(f"[e2a] CANARY recovery_ratio @{PRIMARY} steps2 = {p2['mean']:.4f} "
          f"[{p2['lo']:.4f},{p2['hi']:.4f}]  (published 0.0074 [0.0036, 0.0115])",
          flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- S1 PERCEPTION --------------------------------------------------------
    s1 = {"_ref_scale": {
        "dfmap_rel_per_1_real_frame": _boot(r["ref_dfmap"].numpy(), eid),
        "dpooled_rel_per_1_real_frame": _boot(r["ref_dpooled"].numpy(), eid),
        "_meaning": "relative latent move caused by advancing the REAL window by one "
                    "frame (0.1 s of driving) — the natural unit for 'did the warp "
                    "move the latent at all'."}}
    for (dl, dyaw) in GRID:
        tag = f"{dl}_{dyaw}"
        df = r[f"dfmap_rel::{tag}"].numpy()
        s1[tag] = {
            "dfmap_rel_l2": _boot(df, eid),
            "dpooled_rel_l2": _boot(r[f"dpooled_rel::{tag}"].numpy(), eid),
            "pooled_cosine": _boot(r[f"pooled_cos::{tag}"].numpy(), eid),
            "dfmap_in_units_of_one_real_frame": round(
                float(df.mean() / max(r["ref_dfmap"].numpy().mean(), 1e-9)), 4)}
    res["S1_perception"] = s1

    # ---- S2 REPRESENTABILITY (the crux) --------------------------------------
    tags = [(0.0, 0.0)] + GRID
    y = np.concatenate([np.tile(np.array([[dl, dyaw]], dtype=np.float32), (n, 1))
                        for (dl, dyaw) in tags])
    ep_rep = np.concatenate([np.asarray(eid) for _ in tags])
    s2 = {"_targets": ["dlat_m", "dyaw_deg"],
          "_n_rows": int(len(y)),
          "_note": "rows = (window x perturbation); each window appears at EVERY "
                   "perturbation, so window identity carries ZERO information about "
                   "the target — the probe can only succeed by reading the warp."}
    for fname in ("feat_pooled", "feat_colmean", "feat_rowmean", "feat_full"):
        X = np.concatenate([r[f"{fname}::{dl}_{dyaw}"].numpy()
                            for (dl, dyaw) in tags]).astype(np.float32)
        fit = ridge_dual_fit_eval if fname == "feat_full" else ridge_fit_eval
        s2[fname] = fit(X, y.copy(), ep_rep, device=device)
        s2[fname + "_SHUFFLED_CONTROL"] = fit(
            X, y.copy(), ep_rep, device=device, shuffle_control=True)
        print(f"[e2a] S2 {fname:14s} d={X.shape[1]:5d} "
              f"R2(dlat)={s2[fname]['target0']['r2_heldout']:+.4f} "
              f"R2(dyaw)={s2[fname]['target1']['r2_heldout']:+.4f} | shuffled "
              f"{s2[fname + '_SHUFFLED_CONTROL']['target0']['r2_heldout']:+.4f}",
              flush=True)
        del X
        torch.cuda.empty_cache()
    res["S2_representability"] = s2
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- S3 TRUNCATION --------------------------------------------------------
    s3 = {"a_denoise_sweep": {
        f"steps{s}": rr[prim][f"steps{s}"]["recovery_ratio"] for s in DENOISE_STEPS},
        "a_ratio_16_over_2": round(
            float(rr[prim]["steps16"]["recovery_ratio"]["mean"]
                  / max(abs(rr[prim]["steps2"]["recovery_ratio"]["mean"]), 1e-9)), 3)}
    ladder = {}
    for (dl, dyaw) in GRID:
        tag = f"{dl}_{dyaw}"
        ladder[tag] = {
            "rel_shift_fmap": _boot(r[f"dfmap_rel::{tag}"].numpy(), eid),
            "anchor_posterior_TV": _boot(r[f"dpost_tv::{tag}"].numpy(), eid),
            "frac_selected_anchor_changed": _boot(r[f"dsel::{tag}"].numpy(), eid),
            "rel_shift_selected_traj": _boot(r[f"dtraj_full::{tag}"].numpy(), eid)}
    s3["b_stage_ladder"] = ladder
    s3["b_note"] = ("relative movement at each stage of the cascade: encoder fmap -> "
                    "anchor posterior (TV distance) -> selected trajectory. A large "
                    "fmap shift with a small posterior/traj shift localises the loss "
                    "in the DECODER, not the encoder.")
    cov = {}
    for (dl, dyaw) in GRID:
        tag = f"{dl}_{dyaw}"
        cov[tag] = {
            "frac_windows_with_returning_anchor": _boot(
                r[f"anchor_has::{tag}"].numpy(), eid),
            "mean_n_returning_anchors_of_N": _boot(
                r[f"anchor_nret::{tag}"].numpy(), eid)}
    s3["c_anchor_coverage"] = cov
    s3["c_note"] = (f"ZERO model calls. From a +dlat offset pose, an anchor 'returns' "
                    f"if |dlat + anchor_y(2s)| < {CORRIDOR} m. N="
                    f"{int(model.decoder.anchors.shape[0])} anchors.")
    res["S3_truncation"] = s3

    # ---- S4 CONDITIONING ------------------------------------------------------
    dw = r[f"dtraj_full::{prim}"].numpy()
    s4 = {
        "plan_move_per_1m_image_warp_m": _boot(
            np.abs(r[f"resp::{prim}::s2"].numpy()), eid),
        "plan_move_per_1mps_v0_change_m": _boot(
            r["dtraj_per_1mps_v0"].numpy(), eid),
        "plan_move_per_nav_switch_follow_to_left_m": _boot(
            r["dtraj_per_navswitch"].numpy(), eid),
        "recovery_ratio_v0_ablated": rr[prim]["steps2_v0zero"]["recovery_ratio"],
        "recovery_ratio_v0_randomised": rr[prim]["steps2_v0rand"]["recovery_ratio"],
        "_note": "if a 1 m displacement of the WORLD moves the plan far less than a "
                 "1 m/s change of the EGO SPEED CHANNEL, the plan is ego-status "
                 "dominated (the published shortcut, arXiv:2312.03031).",
        "_rel_traj_shift_under_1m_warp": _boot(dw, eid)}
    s4["ego_dominance_ratio_v0_over_image"] = round(
        float(s4["plan_move_per_1mps_v0_change_m"]["mean"]
              / max(s4["plan_move_per_1m_image_warp_m"]["mean"], 1e-9)), 2)
    res["S4_conditioning"] = s4

    # ---- ATTRIBUTION ----------------------------------------------------------
    best_feat = max(("feat_pooled", "feat_colmean", "feat_rowmean", "feat_full"),
                    key=lambda f: s2[f]["target0"]["r2_heldout"])
    r2 = s2[best_feat]["target0"]["r2_heldout"]
    rho = float(np.sqrt(max(r2, 0.0)))
    R = float(rr[prim]["steps2"]["recovery_ratio"]["mean"])
    R16 = float(rr[prim]["steps16"]["recovery_ratio"]["mean"])
    Rego = float(rr[prim]["steps2_v0zero"]["recovery_ratio"]["mean"])
    gap = rho - R
    res["ATTRIBUTION"] = {
        "_construction": (
            "rho = sqrt(max(episode-disjoint held-out R^2 of the linear dlat probe on "
            "the best latent, 0)) = the fraction of the offset an ORACLE LINEAR readout "
            "of this representation could recover; R = the realized recovery_ratio at "
            "the deployed 2 denoise steps. loss_representation = 1 - rho; "
            "loss_downstream = rho - R; the downstream loss is sub-attributed by the "
            "MEASURED recoveries from more denoise steps (truncation) and from ego "
            "ablation (conditioning). Quoting any share without this construction is "
            "not permitted."),
        "best_latent_for_probe": best_feat,
        "rho_representation_ceiling": round(rho, 4),
        "R_realized_recovery_ratio_steps2": round(R, 5),
        "loss_representation_frac_of_ideal": round(1.0 - rho, 4),
        "loss_downstream_frac_of_ideal": round(gap, 4),
        "share_of_total_loss_representation": round((1.0 - rho) / max(1.0 - R, 1e-9), 4),
        "share_of_total_loss_downstream": round(gap / max(1.0 - R, 1e-9), 4),
        "downstream_subattribution": {
            "truncation_frac_of_downstream_gap": round((R16 - R) / max(gap, 1e-9), 4),
            "conditioning_frac_of_downstream_gap": round((Rego - R) / max(gap, 1e-9), 4),
            "anchor_coverage_frac_windows_with_returning_anchor":
                cov[prim]["frac_windows_with_returning_anchor"]["mean"],
        }}
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[e2a] ATTRIBUTION rho={rho:.4f} R={R:.5f} "
          f"loss_repr_share={res['ATTRIBUTION']['share_of_total_loss_representation']:.4f} "
          f"loss_down_share={res['ATTRIBUTION']['share_of_total_loss_downstream']:.4f}",
          flush=True)
    print(f"[e2a] wrote {args.out}", flush=True)
    print("E2A_DONE", flush=True)


if __name__ == "__main__":
    main()
