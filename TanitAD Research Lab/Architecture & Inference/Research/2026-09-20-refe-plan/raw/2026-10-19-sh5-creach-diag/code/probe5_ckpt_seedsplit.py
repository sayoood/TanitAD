"""PROBE 5 (Research Lab, 2026-10-20, CPU only): why does the cos16_k6o8 lever read the plant on seed 0 (W60 F 0.846) and not on seed 1 (F 0.562)?  Read from the SAVED CHECKPOINTS.

Question (written before it ran): for each W60 checkpoint (and the grid-3 cos16 / base checkpoints), on the SAME first 24 fold-0 tokens the attention diagnostic used and on 48 TRAIN tokens:
  (1) does the checkpoint reproduce the logged final attention numbers (beta_eff per head, plant mass) -- a check that the checkpoint IS the evaluated learner (tolerance: |d beta| <= 0.5,
      the CPU run differs from the GPU run by bf16-vs-fp32 arithmetic);
  (2) the per-learner TRAIN fit the sealed log cannot give (it logs a POOLED last-batch mean over learners): BCE, Huber, point AUC on train tokens vs fold-0 tokens;
  (3) the SOURCE of the read, per learner: the same three ablations (zero_clear / shuffle_clear / no_plant) on train tokens, so a train-set read of the plant is distinguished from a
      visual-ctx fit.  A learner that fits train via the plant must lose its train fit under zero_clear; one that fits via the visual ctx must not.
No training, no GPU, no navtest.  usage: CUDA_VISIBLE_DEVICES=-1 PY probe5_ckpt_seedsplit.py --out ../raw/probe5_ckpt_seedsplit.json [--n-train 48]"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre as DP                 # noqa: E402
import diag_attn as DA                # noqa: E402
L, B, TE = DP.L, DP.B, DP.TE
DD = DP.DD
W = "C:/Users/Admin/par18/sh5diag/work"
COS16 = {"mode": "cos", "tau0": 16.0, "mult": 10.0}
CKPTS = [  # (file, logged-json, run, tag, lever, K, seed)
    ("diag4/ckpt/W60_s0__BEST_60ep.pt", "diag4/diag_W60_s0.json", "W60_s0", "BEST_60ep", COS16, 6, 0),
    ("diag4/ckpt/W60_s1__BEST_60ep.pt", "diag4/diag_W60_s1.json", "W60_s1", "BEST_60ep", COS16, 6, 1),
    ("diag4/ckpt/W60_s0__base_k8o8_60ep.pt", "diag4/diag_W60_s0.json", "W60_s0", "base_k8o8_60ep", None, 8, 0),
    ("diag4/ckpt/W60_s1__base_k8o8_60ep.pt", "diag4/diag_W60_s1.json", "W60_s1", "base_k8o8_60ep", None, 8, 1),
    ("diag3/ckpt/S12_s0__cos16_k8o8.pt", "diag3/diag_S12_s0.json", "S12_s0", "cos16_k8o8", COS16, 8, 0),
    ("diag3/ckpt/S12_s1__cos16_k8o8.pt", "diag3/diag_S12_s1.json", "S12_s1", "cos16_k8o8", COS16, 8, 1),
]


def auc(z, y):
    z, y = np.asarray(z, np.float64).ravel(), np.asarray(y).ravel().astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = np.argsort(np.argsort(z, kind="mergesort"), kind="mergesort") + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-attn", type=int, default=24)
    ap.add_argument("--only", default=None, help="comma list of ckpt indices")
    x = ap.parse_args(argv)
    torch.set_num_threads(8)
    t0 = time.time()
    allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
    keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
    D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
    ctx_rows = [row_of[str(t)] for t in D["token"]]
    prov = TE.NpyCtx(f"{DD}/ctx_local/ctx.npy", ctx_rows, "cpu")
    out = {"question": __doc__.split("\n\n")[0], "n_train_tokens": x.n_train, "n_attn_tokens": x.n_attn, "arith": "CPU fp32 (autocast off)", "ckpts": []}
    idx = range(len(CKPTS)) if not x.only else [int(i) for i in x.only.split(",")]
    for i in idx:
        f, lj, run, tag, lever, K, seed = CKPTS[i]
        rc = TE.RunCfg(seed=seed, cycles=3, inner=4, epochs_total=12, device="cpu", autocast=False)
        tr = DP.DiagTrainer(D, [TE.Spec("plant", 0, seed)], rc, prov, log=lambda m: None)
        ln = tr.learners[0]
        cfg = B.SH5Cfg(d_ctx=prov.Dc, fourier_k=K)
        ln.model = B.SH5Branch(cfg)
        DA.apply_lever(ln.model, lever)
        sd = torch.load(f"{W}/{f}", map_location="cpu")
        miss = ln.model.load_state_dict(sd, strict=True)
        ln.model.eval()
        ln.plant_oct, ln.ablate = 8, None
        # the attention tokens: the first n_attn fold-0 tokens in predict()'s order (rng(seed + 424242)), shard partition preserves order
        order = np.random.default_rng(seed + 424242).permutation(D["T"])
        f0_toks = [int(t) for t in order if D["tok_fold"][t] == 0][:x.n_attn]
        tr_toks = [int(t) for t in np.random.default_rng(5).permutation(D["T"]) if D["tok_fold"][t] != 0][:x.n_train]
        toks = f0_toks + tr_toks
        shard = prov.load(toks)
        ctx_pos = {t: j for j, t in enumerate(toks)}
        acc, nacc = None, 0
        rec = {"ckpt": f, "run": run, "tag": tag, "K": K, "lever": lever, "seed": seed, "taus": DA.taus(ln.model)}
        for abl in (None, "zero_clear", "shuffle_clear", "no_plant"):
            ln.ablate = abl
            part = {}
            for name, tt in (("fold0", f0_toks), ("train", tr_toks)):
                zs, ys, rs, ts, ss = [], [], [], [], []
                for t in tt:
                    pts, s2t, ctx, _, _, sets = tr._batch(ln, [t], ctx_pos, shard, False)
                    ctx = ctx.float()                       # the plant was written into the bf16 ctx exactly as on the GPU; the arithmetic is then fp32
                    with torch.no_grad():
                        kv = ln.model.kv(ctx)
                        logit, reg = ln.model(pts, kv, s2t)
                    if abl is None and name == "fold0":
                        st = DA.attn_token_stats(ln.model, pts, ctx, min(K, 8), tr._ac)
                        acc, nacc = DA.acc_add(acc, st), nacc + 1
                    zs.append(logit.float().numpy()); rs.append(reg.float().numpy())
                    ys.append(ln.y[sets]); ts.append(ln.tgt[sets]); ss.append(sets)
                z, y, r_, tg, se = np.concatenate(zs), np.concatenate(ys), np.concatenate(rs), np.concatenate(ts), np.concatenate(ss)
                lab = D["lab"][..., L.DAC_COL][se]
                FL, _ = L.refs()
                st = FL.pair_stats(z, lab == 0.0, lab == 1.0, D["code"][se], names=("ALLQ", "Q4", "Q123"))
                pa = {k: (float(n.sum() / d.sum()) if d.sum() > 0 else None) for k, (n, d) in st.items()}
                lt = B.loss_terms(torch.as_tensor(z), torch.as_tensor(r_), torch.as_tensor(y), torch.as_tensor(tg), ln.w_reg)
                part[name] = {"n_sets": int(len(z)), "bce": float(lt["bce"]), "huber": float(lt.get("reg", torch.tensor(float("nan")))), "auc_point": auc(z, y), "auc_all_strata": pa["ALLQ"], "auc_F": pa["Q4"],
                              "r2_reg": TE.r2_reg(r_.astype(np.float32), tg)}
            rec["full" if abl is None else f"ablate_{abl}"] = part
        rep = DA.acc_report(acc, nacc)
        logged = json.load(open(f"{W}/{lj}"))["curve"][tag][-1]["attn"]
        rec["attn_ckpt"] = rep
        rec["attn_logged_beta"] = [a["beta_eff"] for a in logged]
        rec["attn_reproduces"] = bool(all(abs(p - q) <= 0.5 for a, b in zip(rep, logged) for p, q in zip(a["beta_eff"], b["beta_eff"])))
        out["ckpts"].append(rec)
        print(f"ZZP5 {run} {tag}: train bce {rec['full']['train']['bce']:.4f} F {rec['full']['train']['auc_F']:.4f} ALL {rec['full']['train']['auc_all_strata']:.4f} | f0 bce {rec['full']['fold0']['bce']:.4f} F {rec['full']['fold0']['auc_F']:.4f}"
              f" | train zc F {rec['ablate_zero_clear']['train']['auc_F']:.4f} np F {rec['ablate_no_plant']['train']['auc_F']:.4f} | attn_reproduces {rec['attn_reproduces']} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(x.out, "w"), indent=1)
    out["elapsed_s"] = time.time() - t0
    json.dump(out, open(x.out, "w"), indent=1)
    print("ZZP5_DONE", flush=True)


if __name__ == "__main__":
    main()
