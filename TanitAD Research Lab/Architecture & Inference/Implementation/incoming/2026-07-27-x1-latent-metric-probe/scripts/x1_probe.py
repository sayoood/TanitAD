#!/usr/bin/env python3
"""X1 stage 2 (+X1b) — the frozen-latent probe ceiling, on CPU.

Pre-registered falsifier (MANIFOLD_MISMATCH_RESEARCH.md §6, X1):

    "REFUTES the 'head is the locus' hypothesis if the 4x-width MLP does not beat
     the trained invdyn head's 1.0129 m by >= 30 %."
    "If a fresh probe reaches <~ 0.2 m at k=1, the information IS there and
     invdyn's 1.0 m is an optimisation/multi-task-interference failure => C1/C3
     are worth a run.  If all three probes plateau near invdyn's ~1.0 m, the
     information is NOT there => C3 is futile without new inputs, and M2 stands."

X1b (same cache, +0 GPU):
    "Split the probe fit and score by per-clip cy (rig A ~ 543 / rig B ~ 755).
     Fit within-rig and cross-rig.  REFUTES the two-rig-confound hypothesis if
     within-rig probe error is not >= 20 % better than pooled."

⚠️ The cached episodes carry NO clip_id and NO intrinsics (``ToyEpisode`` =
frames/actions/poses/episode_id/maneuvers), so ``cy`` is not directly readable.
The rig is instead split on a MEASURED image statistic — the per-episode mean
row-intensity profile, whose vertical structure is exactly what a ~215 px
principal-point difference moves — clustered into 2 groups by 1-D k-means on the
profile's first principal component.  A RANDOM-SPLIT CONTROL with the same group
sizes is fitted alongside, so "two fits beat one fit" cannot be mistaken for a
rig effect.

WHAT WOULD MAKE THIS PROBE REPORT "NO METRIC INFORMATION" SPURIOUSLY — stated in
advance, each with the check that rules it out:
  R1 wrong preprocessing / broken latents  -> ruled out by stage 1's FIDELITY
     check (the trained imagined-pair decoder must stay small on these latents).
  R2 probe under-fitting                   -> ruled out by reporting TRAIN error
     alongside test error, and by the 4x-width arm; if train error is also ~1 m
     the ceiling is informational, not optimisational.
  R3 target/units error                    -> ruled out by the dt self-check
     (poses' v vs realised displacement) and by the true-v0 integrator baseline,
     which must land at a sane metre value.
  R4 too few samples                       -> ruled out by the sample-size curve
     (25/50/100 % of train pairs).
  R5 episode-split bad luck                -> ruled out by episode-cluster
     bootstrap intervals and by reporting the split.

Usage:  python x1_probe.py --latents <dir>/x1_latents.pt \
            --heads <dir>/x1_grounding_heads.pt --out <artifacts>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci                                    # noqa: E402
from tanitad.models.metric_dynamics import (                       # noqa: E402
    MetricInverseDynamics, relative_ego_pose, wrap_angle)

POSE_SCALE = 10.0        # v1 config.json
DT = 0.1                 # 10 Hz corpus; verified by the dt self-check below
KS = (1, 2, 4, 20)
# horizons the TRAINED heads own, so each k is anchored to the right head
HEAD_FOR_K = {1: "op", 2: "op", 4: "op", 8: "tac", 16: "tac", 20: "str"}


# --------------------------------------------------------------------------- #
# data                                                                         #
# --------------------------------------------------------------------------- #
def build_pairs(states, poses, k, stride, ep_ids, offsets):
    """(a_idx, b_idx, target[3], eid, ep) for every valid anchor at horizon k."""
    a_idx, b_idx, tgt, eid, epi = [], [], [], [], []
    for e, (st, po) in enumerate(zip(states, poses)):
        T = min(int(st.shape[0]), int(po.shape[0]))
        a = torch.arange(0, T - k, stride)
        if a.numel() == 0:
            continue
        t = relative_ego_pose(po[a], po[a + k])                    # [n,3]
        a_idx.append(a + offsets[e])
        b_idx.append(a + k + offsets[e])
        tgt.append(t)
        eid += [str(ep_ids[e])] * a.numel()
        epi.append(torch.full((a.numel(),), e))
    return (torch.cat(a_idx), torch.cat(b_idx), torch.cat(tgt), eid,
            torch.cat(epi))


def de_m(pred, tgt):
    """The program's metre displacement error: ||Δxy_pred − Δxy_true||₂."""
    return (pred[..., :2] - tgt[..., :2]).norm(dim=-1)


# --------------------------------------------------------------------------- #
# probes                                                                       #
# --------------------------------------------------------------------------- #
def ridge_fit(Z, a_idx, b_idx, Y, lams, chunk=4096, pair=True):
    """Closed-form ridge on [z_a, z_b, 1] (or [z_a, 1] when pair=False).

    Accumulates X'X / X'Y in chunks so the design matrix is never materialised.
    Returns {lam: W [D, out]}.  float64 throughout — a 4097-wide normal equation
    in float32 is numerically marginal and this is a ceiling claim.
    """
    S = Z.shape[1]
    D = (2 * S if pair else S) + 1
    XtX = torch.zeros(D, D, dtype=torch.float64)
    XtY = torch.zeros(D, Y.shape[1], dtype=torch.float64)
    for i in range(0, a_idx.numel(), chunk):
        za = Z[a_idx[i:i + chunk]].double()
        x = (torch.cat([za, Z[b_idx[i:i + chunk]].double()], 1) if pair else za)
        x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
        XtX += x.T @ x
        XtY += x.T @ Y[i:i + chunk].double()
    eye = torch.eye(D, dtype=torch.float64)
    eye[-1, -1] = 0.0                       # never penalise the bias
    return {lam: torch.linalg.solve(XtX + lam * eye, XtY) for lam in lams}


def ridge_pred(Z, a_idx, b_idx, W, chunk=8192, pair=True):
    out = []
    for i in range(0, a_idx.numel(), chunk):
        za = Z[a_idx[i:i + chunk]].double()
        x = (torch.cat([za, Z[b_idx[i:i + chunk]].double()], 1) if pair else za)
        x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
        out.append((x @ W).float())
    return torch.cat(out)


def train_mlp(Z, a_tr, b_tr, Y_tr, a_va, b_va, Y_va, S, hidden, device,
              epochs=60, batch=512, lr=1e-3, seed=0, log=None):
    """Fresh ``MetricInverseDynamics`` (the EXACT trained architecture at
    ``hidden``), fit on real pairs with ``grounding_losses`` term (a)'s loss."""
    torch.manual_seed(seed)
    net = MetricInverseDynamics(S, hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (a_tr.numel() // batch + 1)))
    best, best_sd, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(a_tr.numel())
        for i in range(0, a_tr.numel(), batch):
            sel = perm[i:i + batch]
            za = Z[a_tr[sel]].to(device).float()
            zb = Z[b_tr[sel]].to(device).float()
            y = Y_tr[sel].to(device)
            p = net(za, zb)
            loss = ((p[:, :2] - y[:, :2]) / POSE_SCALE).pow(2).mean() \
                + wrap_angle(p[:, 2] - y[:, 2]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        va = float(mlp_de(net, Z, a_va, b_va, Y_va, device).mean())
        if log is not None and (ep % 10 == 0 or ep == epochs - 1):
            log.append({"epoch": ep, "val_de_m": round(va, 4)})
        if va < best - 1e-4:
            best, bad = va, 0
            best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    if best_sd is not None:
        net.load_state_dict(best_sd)
    return net.eval(), best


@torch.no_grad()
def mlp_de(net, Z, a, b, Y, device, chunk=8192):
    out = []
    for i in range(0, a.numel(), chunk):
        p = net(Z[a[i:i + chunk]].to(device).float(),
                Z[b[i:i + chunk]].to(device).float()).cpu()
        out.append(de_m(p, Y[i:i + chunk]))
    return torch.cat(out)


@torch.no_grad()
def head_de(head, Z, a, b, Y, device, chunk=8192):
    return mlp_de(head, Z, a, b, Y, device, chunk)


# --------------------------------------------------------------------------- #
# X1b — rig split from the row-intensity profile                               #
# --------------------------------------------------------------------------- #
def rig_split(profiles, seed=0):
    """2-cluster split of episodes on the row-profile's first PC (1-D k-means)."""
    P = profiles.double()
    P = P - P.mean(0, keepdim=True)
    u, s, v = torch.linalg.svd(P, full_matrices=False)
    pc = (P @ v[0]).numpy()
    c = np.array([pc.min(), pc.max()], dtype=np.float64)
    for _ in range(100):
        lab = (np.abs(pc[:, None] - c[None, :])).argmin(1)
        nc = np.array([pc[lab == j].mean() if (lab == j).any() else c[j]
                       for j in range(2)])
        if np.allclose(nc, c):
            break
        c = nc
    sep = float(abs(c[1] - c[0]) / (pc.std() + 1e-12))
    return lab, {"pc1_var_explained": float((s[0] ** 2 / (s ** 2).sum())),
                 "cluster_centres": [float(x) for x in c],
                 "separation_in_sd": round(sep, 3),
                 "sizes": [int((lab == 0).sum()), int((lab == 1).sum())]}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "artifacts"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride-train", type=int, default=2)
    ap.add_argument("--stride-eval", type=int, default=4)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-x1b", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    d = torch.load(a.latents, map_location="cpu", weights_only=False)
    states, poses, ep_ids = d["states"], d["poses"], d["episode_id"]
    S = int(d["state_dim"])
    offsets, off = [], 0
    for st in states:
        offsets.append(off)
        off += int(st.shape[0])
    Z = torch.cat([s.float() for s in states])                     # [Ntot, S]

    # ---- R3: dt self-check — realised displacement vs logged speed --------- #
    dsp, spd = [], []
    for po in poses:
        dsp.append((po[1:, :2] - po[:-1, :2]).norm(dim=-1))
        spd.append(po[:-1, 3])
    dsp, spd = torch.cat(dsp), torch.cat(spd)
    sel = spd > 1.0
    dt_hat = float((dsp[sel] / spd[sel]).median())

    heads_sd = torch.load(a.heads, map_location="cpu", weights_only=False)
    dev = a.device
    trained = {}
    for lvl in ("op", "tac", "str"):
        h = MetricInverseDynamics(S, hidden=512)
        h.load_state_dict({k.split(".", 1)[1]: v
                           for k, v in heads_sd["invdyn"].items()
                           if k.startswith(lvl + ".")})
        trained[lvl] = h.to(dev).eval()

    # ---- episode-disjoint split ------------------------------------------- #
    n_ep = len(states)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(n_ep, generator=g)
    n_tr = int(round(a.train_frac * n_ep))
    tr_eps = set(order[:n_tr].tolist())
    te_eps = set(order[n_tr:].tolist())

    prof = d.get("row_profile")
    rig_lab, rig_meta = (rig_split(prof) if (prof is not None and not a.skip_x1b)
                         else (None, None))
    rng = np.random.default_rng(a.seed)
    rand_lab = rng.permutation(
        np.array([0] * int((rig_lab == 0).sum()) + [1] * int((rig_lab == 1).sum()))
    ) if rig_lab is not None else None

    results, x1b = {}, {}
    for k in KS:
        ai, bi, Y, eid, epi = build_pairs(states, poses, k, 1, ep_ids, offsets)
        m_tr = torch.tensor([int(e) in tr_eps for e in epi.tolist()])
        m_te = ~m_tr
        sub_tr = torch.nonzero(m_tr).squeeze(1)[:: a.stride_train]
        sub_te = torch.nonzero(m_te).squeeze(1)[:: a.stride_eval]
        a_tr, b_tr, Y_tr = ai[sub_tr], bi[sub_tr], Y[sub_tr]
        a_te, b_te, Y_te = ai[sub_te], bi[sub_te], Y[sub_te]
        eid_te = [eid[i] for i in sub_te.tolist()]
        # inner validation split of TRAIN, episode-disjoint, for model selection
        etr = sorted(tr_eps)
        inner = set(etr[: max(1, len(etr) // 5)])
        mv = torch.tensor([int(e) in inner for e in epi[sub_tr].tolist()])
        a_va, b_va, Y_va = a_tr[mv], b_tr[mv], Y_tr[mv]
        a_ft, b_ft, Y_ft = a_tr[~mv], b_tr[~mv], Y_tr[~mv]

        r = {"k": k, "seconds": round(k * DT, 2),
             "n_train_pairs": int(a_tr.numel()), "n_test_pairs": int(a_te.numel()),
             "n_train_eps": len(tr_eps), "n_test_eps": len(te_eps)}

        # --- baselines ----------------------------------------------------- #
        mean_dp = Y_ft.mean(0, keepdim=True)
        r["baseline_const_mean_de_m"] = round(
            float(de_m(mean_dp.expand_as(Y_te), Y_te).mean()), 4)
        v_true = torch.cat([po[:, 3] for po in poses])[a_te]
        v0_pred = torch.zeros_like(Y_te)
        v0_pred[:, 0] = v_true * k * DT
        per_v0 = de_m(v0_pred, Y_te)
        r["baseline_true_v0_integrator_de_m"] = round(float(per_v0.mean()), 4)

        # --- ridge probes -------------------------------------------------- #
        lams = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        Ws = ridge_fit(Z, a_ft, b_ft, Y_ft, lams)
        val = {lam: float(de_m(ridge_pred(Z, a_va, b_va, W), Y_va).mean())
               for lam, W in Ws.items()}
        best_lam = min(val, key=val.get)
        per_ridge = de_m(ridge_pred(Z, a_te, b_te, Ws[best_lam]), Y_te)
        r["ridge"] = {"lambda": best_lam, "val_curve": {str(x): round(y, 4)
                                                        for x, y in val.items()},
                      "test_de_m": round(float(per_ridge.mean()), 4),
                      "train_de_m": round(float(de_m(
                          ridge_pred(Z, a_ft, b_ft, Ws[best_lam]), Y_ft).mean()), 4)}
        # single-slot control (z_a only): how much needs the SECOND latent?
        Ws1 = ridge_fit(Z, a_ft, b_ft, Y_ft, [best_lam], pair=False)
        r["ridge_single_slot_de_m"] = round(float(de_m(
            ridge_pred(Z, a_te, b_te, Ws1[best_lam], pair=False), Y_te).mean()), 4)
        # shuffled-pair control (must FAIL)
        pg = torch.Generator().manual_seed(a.seed + 7)
        shuf = b_te[torch.randperm(b_te.numel(), generator=pg)]
        r["ridge_shuffled_pair_de_m"] = round(float(de_m(
            ridge_pred(Z, a_te, shuf, Ws[best_lam]), Y_te).mean()), 4)

        # --- speed probe: does the latent PERCEIVE speed at all? ------------ #
        Vall = torch.cat([po[:, 3:4] for po in poses])
        Wv = ridge_fit(Z, a_ft, a_ft, Vall[a_ft], [best_lam], pair=False)
        vhat = ridge_pred(Z, a_te, a_te, Wv[best_lam], pair=False).squeeze(1)
        vt = Vall[a_te].squeeze(1)
        ss = float(((vt - vhat) ** 2).sum() / ((vt - vt.mean()) ** 2).sum())
        sp_pred = torch.zeros_like(Y_te)
        sp_pred[:, 0] = vhat * k * DT
        r["speed_probe"] = {
            "r2": round(1.0 - ss, 4),
            "integrated_de_m": round(float(de_m(sp_pred, Y_te).mean()), 4)}

        # --- MLP probes ---------------------------------------------------- #
        for name, hid in (("mlp_invdyn_h512", 512), ("mlp_4x_h2048", 2048)):
            log = []
            net, va = train_mlp(Z, a_ft, b_ft, Y_ft, a_va, b_va, Y_va, S, hid,
                                dev, epochs=a.epochs, seed=a.seed, log=log)
            per = mlp_de(net, Z, a_te, b_te, Y_te, dev)
            r[name] = {"hidden": hid, "val_de_m": round(va, 4),
                       "test_de_m": round(float(per.mean()), 4),
                       "train_de_m": round(float(mlp_de(
                           net, Z, a_ft, b_ft, Y_ft, dev).mean()), 4),
                       "curve": log}
            r[name + "_per_window"] = per
            del net

        # --- the TRAINED head, on the SAME test pairs (the paired anchor) --- #
        lvl = HEAD_FOR_K[k]
        per_head = head_de(trained[lvl], Z, a_te, b_te, Y_te, dev)
        r["trained_invdyn"] = {"level": lvl,
                               "test_de_m": round(float(per_head.mean()), 4)}

        # --- paired episode-cluster bootstrap, best probe vs trained head --- #
        cands = {"ridge": per_ridge,
                 "mlp_invdyn_h512": r["mlp_invdyn_h512_per_window"],
                 "mlp_4x_h2048": r["mlp_4x_h2048_per_window"]}
        best_name = min(cands, key=lambda n: float(cands[n].mean()))
        r["best_probe"] = best_name
        r["paired_bootstrap_probe_minus_head"] = _ci.paired_episode_cluster_bootstrap(
            cands[best_name].numpy(), per_head.numpy(), eid_te, n_boot=2000, seed=0)
        r["paired_bootstrap_4x_minus_head"] = _ci.paired_episode_cluster_bootstrap(
            cands["mlp_4x_h2048"].numpy(), per_head.numpy(), eid_te,
            n_boot=2000, seed=0)
        r["head_ci"] = _ci.episode_cluster_bootstrap(per_head.numpy(), eid_te,
                                                     n_boot=2000, seed=0)
        r["best_probe_ci"] = _ci.episode_cluster_bootstrap(
            cands[best_name].numpy(), eid_te, n_boot=2000, seed=0)

        # --- R4: sample-size curve on the ridge ---------------------------- #
        curve = {}
        for frac in (0.25, 0.5, 1.0):
            n = max(64, int(frac * a_ft.numel()))
            Wf = ridge_fit(Z, a_ft[:n], b_ft[:n], Y_ft[:n], [best_lam])
            curve[str(frac)] = round(float(de_m(
                ridge_pred(Z, a_te, b_te, Wf[best_lam]), Y_te).mean()), 4)
        r["ridge_sample_size_curve"] = curve

        # --- pre-registered verdict ---------------------------------------- #
        head_m = r["trained_invdyn"]["test_de_m"]
        four_x = r["mlp_4x_h2048"]["test_de_m"]
        r["prereg"] = {
            "falsifier": ("REFUTES 'head is the locus' if the 4x-width MLP does "
                          "not beat the trained invdyn head by >= 30 %"),
            "head_de_m": head_m, "mlp_4x_de_m": four_x,
            "improvement_pct": round(100.0 * (head_m - four_x) / head_m, 2),
            "beats_head_by_30pct": bool(four_x <= 0.70 * head_m),
            "reaches_0p2m_at_k1": bool(k == 1 and four_x <= 0.20),
        }
        for key in list(r):
            if key.endswith("_per_window"):
                del r[key]
        results[str(k)] = r
        print(f"[k={k}] ridge {r['ridge']['test_de_m']}  "
              f"mlp512 {r['mlp_invdyn_h512']['test_de_m']}  "
              f"mlp4x {four_x}  trained-head {head_m}  "
              f"const {r['baseline_const_mean_de_m']}  "
              f"true-v0 {r['baseline_true_v0_integrator_de_m']}  "
              f"speedR2 {r['speed_probe']['r2']}", flush=True)

        # ---------------- X1b: rig-conditioned fit ------------------------- #
        if rig_lab is not None:
            x1b[str(k)] = {}
            for tag, lab in (("rig_proxy", rig_lab), ("random_control", rand_lab)):
                pooled, within = per_ridge.clone(), torch.zeros_like(per_ridge)
                ep_of_te = epi[sub_te]
                for grp in (0, 1):
                    gm = torch.tensor([lab[int(e)] == grp
                                       for e in ep_of_te.tolist()])
                    gf = torch.tensor([lab[int(e)] == grp
                                       for e in epi[sub_tr][~mv].tolist()])
                    if gm.sum() == 0 or gf.sum() < 128:
                        within[gm] = pooled[gm]
                        continue
                    Wg = ridge_fit(Z, a_ft[gf], b_ft[gf], Y_ft[gf], [best_lam])
                    within[gm] = de_m(ridge_pred(Z, a_te[gm], b_te[gm],
                                                 Wg[best_lam]), Y_te[gm])
                x1b[str(k)][tag] = {
                    "pooled_de_m": round(float(pooled.mean()), 4),
                    "within_group_de_m": round(float(within.mean()), 4),
                    "improvement_pct": round(
                        100.0 * (float(pooled.mean()) - float(within.mean()))
                        / float(pooled.mean()), 2),
                    "paired_bootstrap": _ci.paired_episode_cluster_bootstrap(
                        within.numpy(), pooled.numpy(), eid_te, n_boot=2000,
                        seed=0)}
            imp = x1b[str(k)]["rig_proxy"]["improvement_pct"]
            ctl = x1b[str(k)]["random_control"]["improvement_pct"]
            x1b[str(k)]["prereg"] = {
                "falsifier": ("REFUTES the two-rig confound if within-rig probe "
                              "error is not >= 20 % better than pooled"),
                "rig_improvement_pct": imp,
                "random_control_improvement_pct": ctl,
                "excess_over_random_pct": round(imp - ctl, 2),
                "fires": bool(imp >= 20.0)}
            print(f"        X1b k={k}: rig {imp:+.2f} %  random-control "
                  f"{ctl:+.2f} %", flush=True)

    out = {
        "experiment": "X1 — frozen-latent metric probe ceiling (+X1b rig split)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "dev box", "device": dev, "torch": torch.__version__,
        "latents": a.latents, "ckpt_step": d.get("ckpt_step"),
        "corpus": {"cache_dir": d.get("cache_dir"), "n_episodes": n_ep,
                   "PARITY": "NON-PARITY (bb543bdf7836) — within-run contrasts only"},
        "split": {"train_frac": a.train_frac, "seed": a.seed,
                  "n_train_eps": len(tr_eps), "n_test_eps": len(te_eps),
                  "episode_disjoint": True},
        "dt_self_check": {"assumed_dt_s": DT, "median_disp_over_speed_s":
                          round(dt_hat, 4),
                          "PASS": bool(abs(dt_hat - DT) < 0.02)},
        "pose_scale": POSE_SCALE,
        "estimator": ("paired episode-cluster bootstrap, B=2000, seed 0, unit = "
                      "episode cluster (taniteval/ci.py). NEVER overlapping_holdout_se."),
        "results_by_k": results,
        "x1b_rig": x1b, "x1b_rig_split_meta": rig_meta,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    fp = os.path.join(a.out, "x1_probe.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", fp)


if __name__ == "__main__":
    main()
