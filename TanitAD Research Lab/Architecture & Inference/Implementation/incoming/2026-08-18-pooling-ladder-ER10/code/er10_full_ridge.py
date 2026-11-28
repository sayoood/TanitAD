"""E-R1-0 SUPPLEMENT — THE SAME LADDER WITH **NO RANDOM PROJECTION AT ALL**.

⛔ WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL FOR A *NEGATIVE* VERDICT.
§7.1 mandates a fixed 2 048-dim random projection so the four arms are
dimension-matched — without it the 1:1 arm's 491 520 features would win for a
reason that has nothing to do with pooling. That control is correct, and it has
a cost that runs the OTHER way: **a 2 048-dim projection of a 491 520-dim space
keeps a smaller fraction of its arm than a 2 048-dim projection of a
12 288-dim space does.** ⇒ the RP HANDICAPS the fine arms, so "the ladder is
flat" could in principle be the projection throwing the signal away rather than
the tokens not carrying it. A DROP verdict must not rest on that.

⭐ THE FIX IS EXACT, NOT ANOTHER APPROXIMATION. With n_train ~1 300 windows the
ridge has a **DUAL (kernel) form** that is ALGEBRAICALLY IDENTICAL to the primal
and costs O(n²D + n³) instead of O(D³):

    w = Zc'(Zc Zc' + αI)^{-1} yc      ⇒   pred = Zc_ev Zc' (K + αI)^{-1} yc + ȳ

so ALL 491 520 features can be fitted exactly. This answers the question the RP
cannot: *is there ANY linear signal in the un-pooled tokens at all?*

⚠️ WHAT IT IS AND IS NOT. Across arms this row is **DIMENSION-CONFOUNDED BY
CONSTRUCTION** and must NEVER be read as a pooling ladder — that is exactly what
§7.1 forbids. It is a **PER-ARM CEILING**: the best a linear readout could do
with everything that arm has. Its job is to make a ZERO unambiguous.

⛔ EQUIVALENCE IS PROVED, NOT ASSERTED. `--gate-primal` fits the SAME target on
the SAME windows through `pc6_linear_readout.ridge_fit(..., intercept_col=-1)`
on the small arm and asserts the two prediction vectors agree to 1e-6.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er10_pool_ladder import (POOL_ARMS, fit_one, paired_delta_r2c,  # noqa: E402
                              pool_tokens)
import ll1_ladder as LL                                          # noqa: E402
from pc6_linear_readout import ridge_fit                         # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,             # noqa: E402
                          paired_episode_cluster_bootstrap)


def dual_predict(Ztr, ytr, Zev, alpha):
    """Exact ridge with an UNPENALISED intercept, in the dual. Torch, GPU-able.

    ``Ztr``/``Zev`` are [n, D] float32 tensors WITHOUT a bias column.
    """
    mu = Ztr.mean(0, keepdim=True)
    yb = float(ytr.mean())
    A = Ztr - mu
    B = Zev - mu
    K = A @ A.T                                            # [n_tr, n_tr]
    n = K.shape[0]
    rhs = (ytr - yb).unsqueeze(1)
    dual = torch.linalg.solve(K + alpha * torch.eye(n, device=K.device,
                                                    dtype=K.dtype), rhs)
    return (B @ (A.T @ dual)).squeeze(1) + yb


def zscore_gpu(X, tr_idx, device, chunk=256):
    """[n, D] fp16 host tensor -> fp32 GPU tensor z-scored on the TRAIN rows."""
    n, D = X.shape
    s = torch.zeros(D, dtype=torch.float64)
    s2 = torch.zeros(D, dtype=torch.float64)
    for i in range(0, len(tr_idx), chunk):
        blk = X[tr_idx[i:i + chunk]].to(torch.float64)
        s += blk.sum(0)
        s2 += (blk * blk).sum(0)
    m = len(tr_idx)
    mu = s / m
    sd = torch.sqrt(torch.clamp(s2 / m - mu * mu, min=0.0))
    sd[sd < 1e-12] = 1.0
    mu32, sd32 = mu.to(torch.float32).to(device), sd.to(torch.float32).to(device)
    out = torch.empty((n, D), dtype=torch.float32, device=device)
    for i in range(0, n, chunk):
        out[i:i + chunk] = (X[i:i + chunk].to(device).float() - mu32) / sd32
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arms", nargs="+", default=["p40", "p1"])
    ap.add_argument("--targets", nargs="+",
                    default=["ego_v0", "ego_yawrate", "ego_curv", "lead_gap",
                             "lead_closing", "lead_inv_ttc", "n_agents_all"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--ridge-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--sd-ratio-floor", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate-primal", default="p40",
                    help="arm on which the DUAL is asserted equal to the "
                         "PRIMAL pc6 solve (small enough to fit densely)")
    a = ap.parse_args(argv)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    th, tw = int(meta["token_grid"][0]), int(meta["token_grid"][1])
    d_model = int(rows[0]["tokens"].shape[-1])
    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])
    ego = LL.load_ego(Path(a.episodes_dir), Path(a.join_file),
                      {r["clip_id"] for r in rows})
    align = LL.bind_pose_grid(rows, ego)
    poff = align["pose_index_offset"]

    idx_tr = [i for i, r in enumerate(rows) if r["clip_id"] in tr_c]
    idx_ev = [i for i, r in enumerate(rows) if r["clip_id"] in ev_c]
    keep = idx_tr + idx_ev
    sub = [rows[i] for i in keep]
    pos_tr = np.arange(len(idx_tr))
    pos_ev = np.arange(len(idx_tr), len(keep))
    ctr_all = np.array([sub[i]["clip_id"] for i in pos_tr])
    cev_all = np.array([sub[i]["clip_id"] for i in pos_ev])
    eev_all = np.array([sub[i]["episode_uid"] for i in pos_ev])
    v0_all = np.array([float(r["v0"]) for r in sub])
    tvals = {t: (lambda z: (np.array(z[0], float), np.array(z[1])))(
        tuple(zip(*[LL.target_of(r, ego, t, poff) for r in sub])))
        for t in a.targets}

    res = {"_evidence_class": "MEASURED (ours; EXACT full-feature dual ridge on "
                              "a frozen banked checkpoint)",
           "eval_tier": "T0-DIAGNOSTIC",
           "experiment": "E-R1-0 SUPPLEMENT — no random projection",
           "arm_label": a.label, "run_stamp": meta.get("run_stamp"),
           "step": meta.get("step"), "token_grid": [th, tw],
           "d_model": d_model, "alphas": a.alphas, "n_boot": a.n_boot,
           "solve": "DUAL (kernel) ridge, unpenalised intercept — "
                    "ALGEBRAICALLY IDENTICAL to the primal, gated below",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "⛔ cross-arm reading": "DIMENSION-CONFOUNDED BY CONSTRUCTION — this "
                                   "is a PER-ARM CEILING, never a pooling "
                                   "ladder. §7.1's ladder is the RP table.",
           "arms": {}, "preds": {}}

    preds: dict = {}
    for arm in a.arms:
        ta = time.time()
        kernel = POOL_ARMS[arm]
        n_units = (th // kernel[0]) * (tw // kernel[1])
        D = n_units * d_model
        Xh = torch.empty((len(sub), D), dtype=torch.float16)
        for s0 in range(0, len(sub), 48):
            tk = torch.stack([r["tokens"].float()
                              for r in sub[s0:s0 + 48]]).to(dev).half()
            Xh[s0:s0 + tk.shape[0]] = pool_tokens(tk, kernel, th, tw).cpu()
        Z = zscore_gpu(Xh, pos_tr, dev)
        del Xh
        arec = {"pool_kernel": list(kernel), "n_units": n_units,
                "n_raw_features": D, "projected": False, "targets": {}}
        for tname in a.targets:
            y, ok = tvals[tname]
            mtr, mev = ok[pos_tr], ok[pos_ev]
            ytr = torch.from_numpy(y[pos_tr][mtr]).float().to(dev)
            yev_np = y[pos_ev][mev]
            Ztr = Z[torch.from_numpy(pos_tr[mtr]).to(dev)]
            Zev = Z[torch.from_numpy(pos_ev[mev]).to(dev)]
            ctr, cev, eev = ctr_all[mtr], cev_all[mev], eev_all[mev]
            v0ev = None if tname == "ego_v0" else v0_all[pos_ev][mev]
            # alpha on an EPISODE-DISJOINT inner split of the PROBE-TRAIN clips
            rng = np.random.default_rng(a.ridge_seed)
            clips = np.array(sorted(set(ctr.tolist())))
            rng.shuffle(clips)
            inner = set(clips[:max(1, int(round(len(clips) * a.inner_frac)))]
                        .tolist())
            m_in = torch.from_numpy(np.array([c in inner for c in ctr])).to(dev)
            best, best_mae, tried = None, np.inf, {}
            for al in a.alphas:
                p = dual_predict(Ztr[~m_in], ytr[~m_in], Ztr[m_in], al)
                mae = float((p - ytr[m_in]).abs().mean())
                tried[f"{al:g}"] = round(mae, 6)
                if mae < best_mae:
                    best, best_mae = al, mae
            pred = dual_predict(Ztr, ytr, Zev, best).cpu().numpy().astype(
                np.float64)
            preds.setdefault(tname, {})[arm] = (pred, yev_np, eev, v0ev)

            const_v = float(np.median(y[pos_tr][mtr]))
            epm = LL.loo_epmean(yev_np, cev, const_v)
            e_arm, e_con, e_ep = (np.abs(pred - yev_np),
                                  np.abs(const_v - yev_np),
                                  np.abs(epm - yev_np))
            k1 = paired_episode_cluster_bootstrap(e_arm, e_con, eev,
                                                  n_boot=a.n_boot)
            arm_ci = episode_cluster_bootstrap(e_arm, eev, n_boot=a.n_boot)
            psd, gsd = float(pred.std()), float(yev_np.std())
            r = LL.corr(pred, yev_np)
            rec = {"unit": LL.UNITS[tname], "rung": LL.RUNG[tname],
                   "n_train": int(mtr.sum()), "n_eval": int(mev.sum()),
                   "n_eval_clusters": int(len(np.unique(eev))),
                   "alpha_chosen": best, "alpha_inner_mae": tried,
                   "alpha_at_grid_edge": bool(best in (a.alphas[0],
                                                       a.alphas[-1])),
                   "err": round(float(e_arm.mean()), 4),
                   "err_lo": arm_ci["lo"], "err_hi": arm_ci["hi"],
                   "c_const_err": round(float(e_con.mean()), 4),
                   "c_epmean_err": round(float(e_ep.mean()), 4),
                   "K1_delta": k1["delta"], "K1_lo": k1["lo"],
                   "K1_hi": k1["hi"], "K1_separated": k1["separated"],
                   "K1_PASSES": bool(k1["separated"] and k1["delta"] < 0),
                   "corr": round(r, 4), "r2_ceiling": round(float(r * r), 5),
                   "pred_sd": round(psd, 4), "gt_sd": round(gsd, 4),
                   "pred_sd_over_gt_sd": round(psd / max(gsd, 1e-12), 4),
                   "K1_DEGENERATE": bool(k1["separated"] and k1["delta"] < 0
                                         and psd / max(gsd, 1e-12)
                                         < a.sd_ratio_floor)}
            if v0ev is not None:
                pr = LL.partial_corr(pred, yev_np, v0ev)
                rec["corr_partial_v0"] = round(pr, 4)
                rec["r2_ceiling_partial_v0"] = round(float(pr * pr), 5)
            arec["targets"][tname] = rec
            print("  %-5s %-14s D=%7d n=%4d r2c=%.5f rpv0=%+.3f K1=%+8.4f %-4s "
                  "psd/gsd=%.3f alpha=%g%s"
                  % (arm, tname, D, int(mev.sum()), rec["r2_ceiling"],
                     rec.get("corr_partial_v0", float("nan")),
                     rec["K1_delta"], "PASS" if rec["K1_PASSES"] else "fail",
                     rec["pred_sd_over_gt_sd"], best,
                     "  ⛔EDGE" if rec["alpha_at_grid_edge"] else ""),
                  flush=True)

            # ---- ⛔ the DUAL == PRIMAL gate, on the small arm ---------------
            if arm == a.gate_primal and "dual_primal_gate" not in res:
                Ztr_n = Ztr.cpu().numpy().astype(np.float64)
                Zev_n = Zev.cpu().numpy().astype(np.float64)
                ones = np.ones((Ztr_n.shape[0], 1))
                w = ridge_fit(np.concatenate([Ztr_n, ones], 1),
                              ytr.cpu().numpy().astype(np.float64), best,
                              intercept_col=-1)
                pp = np.concatenate([Zev_n,
                                     np.ones((Zev_n.shape[0], 1))], 1) @ w
                md = float(np.abs(pp - pred).max())
                res["dual_primal_gate"] = {
                    "arm": arm, "target": tname, "alpha": best,
                    "n_features": int(Ztr_n.shape[1]),
                    "max_abs_pred_diff": md,
                    "rel_to_pred_sd": float(md / max(pred.std(), 1e-12)),
                    "PASSED": bool(md / max(pred.std(), 1e-12) < 1e-4),
                    "what": "pc6_linear_readout.ridge_fit(intercept_col=-1) "
                            "PRIMAL vs this file's DUAL, same alpha/windows"}
                print(f"  [gate] dual==primal max|Δpred| {md:.3e} "
                      f"({res['dual_primal_gate']['rel_to_pred_sd']:.2e} of "
                      f"pred sd) -> "
                      f"{'PASS' if res['dual_primal_gate']['PASSED'] else 'FAIL'}",
                      flush=True)
        arec["wall_s"] = round(time.time() - ta, 1)
        res["arms"][arm] = arec
        del Z
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if "p40" in a.arms:
        res["deltas_vs_p40"] = {}
        for tname, byarm in preds.items():
            pb, yv, ev, z = byarm["p40"]
            row = {}
            for arm, (pa, _, _, _) in byarm.items():
                if arm == "p40":
                    continue
                row[arm] = {
                    "delta_r2_ceiling": paired_delta_r2c(pa, pb, yv, ev,
                                                         a.n_boot),
                    "delta_r2_ceiling_partial_v0": (
                        paired_delta_r2c(pa, pb, yv, ev, a.n_boot, z=z)
                        if z is not None else None),
                    "delta_mae": paired_episode_cluster_bootstrap(
                        np.abs(pa - yv), np.abs(pb - yv), ev, n_boot=a.n_boot)}
                d = row[arm]["delta_r2_ceiling"]
                print("  Δ%-4s-p40 %-14s r2c=%+.5f [%+.5f, %+.5f] sep=%s"
                      % (arm, tname, d["delta"], d["lo"], d["hi"],
                         d["separated"]), flush=True)
            res["deltas_vs_p40"][tname] = row

    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
    print(f"[full] wrote {a.out}  {res['wall_s']} s", flush=True)
    if not res.get("dual_primal_gate", {}).get("PASSED", False):
        raise SystemExit("[full] ⛔ DUAL==PRIMAL GATE DID NOT PASS — the solve "
                         "is not the program's ridge and nothing here is "
                         "readable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
