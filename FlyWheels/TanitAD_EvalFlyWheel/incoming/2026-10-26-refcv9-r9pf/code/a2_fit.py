"""a2_fit.py -- PREREG_REFCV9_INMODEL_PA PHASE A2: ONE stage-2 fit of the R9-PF head (one arm x one fit seed) on the FRESH calibration rows.  CPU only.
NEW file (EvalFlyWheel 2026-10-26).  NO eval139 window is read or written here.

The head IS refcv9_progress.R9ProgressHead (the in-model module, loaded from the copy of the patched tree), trained in float64 with its BatchNorm in
EVAL mode on statistics set from the training rows (population mean / var of [x9 ; z_tac]: the module's own eval formula (x - m) / sqrt(v + 1e-5)
IS the fitted standardisation).  Objective = refcv9_progress.r9_losses' formula with weights 1.0 / 1.0 (design):
    L = mean_{rows, 8 slots} |s_P + dS[r*] - s_H|  +  CE(regime logits, r*)       (the profile TEACHER-FORCED on r*)
s_P = arc_at_slots(ha0_ext_pose prior8), s_H = arc_at_slots(human G8), r* = gt_regime (K-LABSRC-R9 checked in A1).
Selection = H2cn's (l1_fit_score.h2n_fit): folds_of(eid, fit_seed) 5-fold EPISODE CV, wd in MLP_WD x epochs <= MLP_MAX_EP, AdamW lr 1e-3, batch 256,
torch.manual_seed(fit_seed) before the build + a generator seeded fit_seed for the batch order; the CV score is the SAME objective, pooled out-of-fold
(sum |err| / (8 n) + sum CE / n over all held-out rows) -- the composite analogue of h2n_fit's pooled held-out L1 sum; refit on all rows.
Arms: R9PF (true z, true r*) | R9PF-CONSTZ (z := the calibration-split mean, at fit AND eval) | R9PF-SHUF (r* permuted across calibration EPISODES:
a seeded episode derangement, rows position-matched mod donor length, before the fit).  R9PF-TL shares R9PF's head (a2_freeze.py).

    python -X utf8 a2_fit.py --arm R9PF --fit-seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402

EPS_BN = 1e-5


def derange_episodes(eid: np.ndarray, seed: int) -> np.ndarray:
    """row index of the donor label for every row: episodes in a seeded random cyclic order, episode p[i] receives episode p[i+1]'s rows
    (position-matched, index mod donor length).  No episode keeps its own labels (a cycle over >= 2 episodes)."""
    u = np.unique(eid)
    p = np.random.default_rng(20261026 + int(seed)).permutation(len(u))
    pos = {e: np.where(eid == e)[0] for e in u}
    out = np.empty(len(eid), np.int64)
    for i in range(len(u)):
        e, d = u[p[i]], u[p[(i + 1) % len(u)]]
        r, dr = pos[e], pos[d]
        out[r] = dr[np.arange(len(r)) % len(dr)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("R9PF", "R9PF-CONSTZ", "R9PF-SHUF"))
    ap.add_argument("--fit-seed", type=int, required=True, choices=RC.FIT_SEEDS)
    ap.add_argument("--max-ep", type=int, default=None, help="SMOKE ONLY (writes nothing to heads/)")
    a = ap.parse_args()
    t0 = time.time()
    prereg = RC.check_prereg()
    RC.setup_paths()
    import torch
    import l1_fit_score as F
    R = RC.r9p()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    smoke = a.max_ep is not None
    tag = f"{a.arm}_fs{a.fit_seed}"
    head_path = RC.HEADS / f"{tag}.pt"
    if not smoke and head_path.exists():
        raise SystemExit(f"[a2] REFUSING: {head_path} exists (a frozen head is written ONCE)")
    cache = RC.SCRATCH / "calib_rows.npz"
    want = RC.hex64((RC.SCRATCH / "calib_rows.sha256").read_text(encoding="utf-8"))
    got = RC.sha256_file(cache)
    if len(want) != 64 or got != want:
        raise SystemExit(f"[a2] calib cache sha {got} != recorded {want}")
    d = np.load(cache, allow_pickle=False)
    n = len(d["v0"])
    Z = d["z"].astype(np.float64)
    constz = None
    if a.arm == "R9PF-CONSTZ":
        constz = d["z"].astype(np.float64).mean(0).astype(np.float32)          # the calibration-split mean (float32: the eval input)
        Z = np.broadcast_to(constz.astype(np.float64), Z.shape).copy()
    X9 = d["x9"].astype(np.float64)
    eid = np.asarray(d["eid"]).astype(str)
    r_true = d["r_star"].astype(np.int64)
    perm_rec = None
    r_fit = r_true.copy()
    if a.arm == "R9PF-SHUF":
        donor = derange_episodes(eid, a.fit_seed)
        r_fit = r_true[donor]
        perm_rec = {"kind": "seeded episode derangement (rng 20261026 + fit_seed), rows position-matched mod donor length",
                    "episodes_keeping_own_labels": int(sum(bool(np.all(eid[donor[eid == e]] == e)) for e in np.unique(eid))),
                    "frac_rows_label_unchanged": float((r_fit == r_true).mean()), "label_counts_after": np.bincount(r_fit, minlength=3).tolist(),
                    "label_counts_before": np.bincount(r_true, minlength=3).tolist()}
    with torch.no_grad():
        sP = R.arc_at_slots(torch.from_numpy(d["prior8"].astype(np.float32))).to(torch.float64)
        sH = R.arc_at_slots(torch.from_numpy(d["G8"].astype(np.float64)))
    IN = np.concatenate([X9, Z], 1)                                       # the module's own order: cat([x, z])
    xt, zt = torch.from_numpy(X9), torch.from_numpy(Z)
    rt = torch.from_numpy(r_fit)

    def build(seed, mean, var):
        torch.manual_seed(seed)
        h = R.R9ProgressHead(512, 8, 128).double()
        with torch.no_grad():
            h.norm.running_mean.copy_(torch.from_numpy(mean))
            h.norm.running_var.copy_(torch.from_numpy(var))
            h.norm.num_batches_tracked.fill_(1)
        h.eval()                                                          # BN on the set statistics; nothing else is mode-dependent
        return h

    def obj_terms(h, idx):
        o = h(zt[idx], xt[idx])
        r = rt[idx]
        ds = o["profiles"][torch.arange(len(idx)), r]
        err = (sP[idx] + ds - sH[idx]).abs()
        ce = torch.nn.functional.cross_entropy(o["regime_logits"], r, reduction="none")
        return err, ce

    def train(idx_tr, wd, epochs, seed, idx_val=None):
        mean, var = IN[idx_tr].mean(0), IN[idx_tr].var(0)
        h = build(seed, mean, var)
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=wd)
        itr = torch.from_numpy(np.asarray(idx_tr, np.int64))
        curve = []
        for _ep in range(epochs):
            perm = itr[torch.randperm(len(itr), generator=g)]
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                err, ce = obj_terms(h, b)
                loss = err.mean() + ce.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if idx_val is not None:
                with torch.no_grad():
                    err, ce = obj_terms(h, torch.from_numpy(np.asarray(idx_val, np.int64)))
                    curve.append((float(err.sum()), float(ce.sum())))
        return h, curve, mean, var

    # ---- K-LOSS: this objective == refcv9_progress.r9_losses (the in-model loss) on true labels, on a fixed batch
    hk = build(a.fit_seed, IN.mean(0), IN.var(0))
    b = torch.arange(min(512, n))
    with torch.no_grad():
        o = hk(zt[b], xt[b])
        err, ce = obj_terms(hk, b) if a.arm != "R9PF-SHUF" else (None, None)
        if a.arm == "R9PF-SHUF":                                             # compare with TRUE labels for the identity
            ds = o["profiles"][torch.arange(len(b)), torch.from_numpy(r_true)[b]]
            err = (sP[b] + ds - sH[b]).abs()
            ce = torch.nn.functional.cross_entropy(o["regime_logits"], torch.from_numpy(r_true)[b], reduction="none")
        mine = float(err.mean() + ce.mean())
        cfg = R.R9Config(enable=True, w_prog=1.0, w_regime=1.0, calib_stage=True)
        ref, _tele = R.r9_losses(cfg, {"r9_profiles": o["profiles"], "r9_regime_logits": o["regime_logits"], "r9_s_prior": sP[b]},
                                 torch.from_numpy(d["G8"][: len(b)]), torch.ones(len(b), 8, dtype=torch.bool), torch.from_numpy(d["v0"][: len(b)]), RC.HZ8)
    k_loss = {"mine": mine, "r9_losses": float(ref), "abs_diff": abs(mine - float(ref)), "PASS": abs(mine - float(ref)) <= 1e-10 * max(1.0, abs(mine))}
    # ---- CV (H2cn's grid and folds)
    max_ep = F.MLP_MAX_EP if not smoke else int(a.max_ep)
    f = F.folds_of(eid, a.fit_seed)
    tot = {}
    for wd in F.MLP_WD:
        sa, sc = np.zeros(max_ep), np.zeros(max_ep)
        for j in range(F.NFOLD):
            tr, te = np.where(f != j)[0], np.where(f == j)[0]
            _h, curve, _m, _v = train(tr, wd, max_ep, a.fit_seed, te)
            c = np.asarray(curve)
            sa += c[:, 0]
            sc += c[:, 1]
        tot[wd] = {"cv_obj": (sa / (8 * n) + sc / n), "cv_l1_mean": sa / (8 * n), "cv_ce_mean": sc / n}
        print(f"[a2] {tag} wd={wd} best ep {int(np.argmin(tot[wd]['cv_obj'])) + 1} obj {float(tot[wd]['cv_obj'].min()):.5f} ({time.time() - t0:.0f} s)", flush=True)
    best = min(((wd, int(np.argmin(v["cv_obj"])) + 1, float(v["cv_obj"].min())) for wd, v in tot.items()), key=lambda x: x[2])
    wd, ep = best[0], best[1]
    h, _c, mean, var = train(np.arange(n), wd, ep, a.fit_seed)
    with torch.no_grad():
        o = h(zt, xt)
        lg64, pf64 = o["regime_logits"].numpy(), o["profiles"].numpy()
        err, ce = obj_terms(h, torch.arange(n))
    r_hat = lg64.argmax(1)
    fit = {"arm": a.arm, "fit_seed": a.fit_seed, "wd": float(wd), "epochs": int(ep), "cv_obj": best[2],
           "cv_l1_mean_at_best": float(tot[wd]["cv_l1_mean"][ep - 1]), "cv_ce_mean_at_best": float(tot[wd]["cv_ce_mean"][ep - 1]),
           "cv_obj_by_wd": {str(k): {"best_epoch": int(np.argmin(v["cv_obj"])) + 1, "min": float(v["cv_obj"].min())} for k, v in tot.items()},
           "grid": {"wd": list(F.MLP_WD), "max_epochs": max_ep, "nfold": F.NFOLD, "folds": "l1_fit_score.folds_of(eid, fit_seed)"},
           "n_rows": int(n), "n_episodes": int(len(np.unique(eid))), "n_params": int(R.count_params(h)),
           "train_obj_final": float(err.mean() + ce.mean()), "train_l1_mean": float(err.mean()), "train_ce_mean": float(ce.mean()),
           "train_regime_acc_vs_fit_target": float((r_hat == r_fit).mean()), "train_regime_acc_vs_true_target": float((r_hat == r_true).mean()),
           "train_decision_counts": np.bincount(r_hat, minlength=3).tolist(), "K-LOSS": k_loss, "label_permutation": perm_rec,
           "torch": str(torch.__version__), "threads": torch.get_num_threads(), "wall_s": round(time.time() - t0, 1), "smoke": smoke}
    print(f"[a2] {tag} CHOSE wd={wd} ep={ep} cv={best[2]:.5f} acc_true={fit['train_regime_acc_vs_true_target']:.3f} K-LOSS={k_loss['PASS']}", flush=True)
    if smoke:
        print(json.dumps(fit, default=str)[:1500])
        return 0
    sd32 = {k: v.detach().to(torch.float32) if v.is_floating_point() else v.detach().clone() for k, v in h.state_dict().items()}
    prov = {"prereg_sha256": prereg["sha256"], "calib_rows": {"file": str(cache), "sha256": got}, "a2_fit_sha256": RC.sha256_file(__file__),
            "r9pf_common_sha256": RC.sha256_file(HERE / "r9pf_common.py"), "refcv9_progress_sha256": RC._R9["sha256"],
            "l1_fit_score_sha256": RC.sha256_file(Path(F.__file__)), "fresh_bank_sha256": RC.FRESH_BANK_SHA}
    payload = RC.head_payload(a.arm, a.fit_seed, "regime", sd32, fit, constz, prov)
    b1 = RC.ser(payload)
    if b1 != RC.ser(payload):
        raise SystemExit("[a2] non-deterministic serialisation")
    RC.SCRATCH.joinpath("fits").mkdir(parents=True, exist_ok=True)
    np.savez(RC.SCRATCH / "fits" / f"{tag}_fitrows_f64.npz", logits=lg64, profiles=pf64, mean=mean, var=var)
    tmp = RC.HEADS / f".{tag}.tmp"
    tmp.write_bytes(b1)
    tmp.replace(head_path)
    RC.jdump({**fit, "head_file": str(head_path), "head_sha256": RC.sha256_file(head_path), "provenance": prov},
             RC.SCRATCH / "fits" / f"{tag}_FIT.json")
    print(f"[a2] WROTE {head_path} sha256 {RC.sha256_file(head_path)}  ZZA2FIT_{tag}_OKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
