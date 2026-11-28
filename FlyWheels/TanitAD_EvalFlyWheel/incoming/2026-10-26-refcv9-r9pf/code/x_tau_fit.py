"""x_tau_fit.py -- the prereg s1 EXPLORATORY 'decel tau tuned on the CALIBRATION split, reported beside' (deviation D7: not run in A2; run here
after the bars, as the RULE-ZERO next lever, because the attribution put the loss on the DECISION, not the profiles).  NO eval139 row is read.

Rule (fixed here, before any tau is applied to eval): out-of-fold regime logits on the calibration rows (the head's own selected wd and epoch
count, the same folds_of(eid, fit_seed) 5-fold EPISODE split and training loop as a2_fit.py); tau = the grid value (0.05..0.95 step 0.01)
maximising Youden's J for DECEL on GT-decel vs GT-cruise rows, J = P(decide decel | r* decel) - P(decide decel | r* cruise), with
decide(logits, tau) = refcv9_progress.decide (decel iff p_decel >= tau, else argmax of the rest); ties -> the LARGER tau (fewer decels).
    python -X utf8 x_tau_fit.py --fit-seed K      -> F:/refcv9_r9pf_scratch/fits/R9PF_fs{K}_oof.npz + _tau.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-seed", type=int, required=True, choices=RC.FIT_SEEDS)
    a = ap.parse_args()
    t0 = time.time()
    RC.check_prereg()
    RC.setup_paths()
    import torch
    import l1_fit_score as F
    R = RC.r9p()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    tag = f"R9PF_fs{a.fit_seed}"
    _h, blob, hsha = RC.load_head(RC.HEADS / f"{tag}.pt")
    wd, ep = float(blob["fit"]["wd"]), int(blob["fit"]["epochs"])
    cache = RC.SCRATCH / "calib_rows.npz"
    if RC.sha256_file(cache) != RC.hex64((RC.SCRATCH / "calib_rows.sha256").read_text(encoding="utf-8")):
        raise SystemExit("[tau] calib cache sha mismatch")
    d = np.load(cache, allow_pickle=False)
    n = len(d["v0"])
    Z, X9 = d["z"].astype(np.float64), d["x9"].astype(np.float64)
    eid = np.asarray(d["eid"]).astype(str)
    r = d["r_star"].astype(np.int64)
    with torch.no_grad():
        sP = R.arc_at_slots(torch.from_numpy(d["prior8"].astype(np.float32))).to(torch.float64)
        sH = R.arc_at_slots(torch.from_numpy(d["G8"].astype(np.float64)))
    IN = np.concatenate([X9, Z], 1)
    xt, zt, rt = torch.from_numpy(X9), torch.from_numpy(Z), torch.from_numpy(r)

    def train(idx_tr, seed):                                   # a2_fit.train, verbatim in effect (no validation curve)
        mean, var = IN[idx_tr].mean(0), IN[idx_tr].var(0)
        torch.manual_seed(seed)
        h = R.R9ProgressHead(512, 8, 128).double()
        with torch.no_grad():
            h.norm.running_mean.copy_(torch.from_numpy(mean))
            h.norm.running_var.copy_(torch.from_numpy(var))
            h.norm.num_batches_tracked.fill_(1)
        h.eval()
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=wd)
        itr = torch.from_numpy(np.asarray(idx_tr, np.int64))
        for _ in range(ep):
            perm = itr[torch.randperm(len(itr), generator=g)]
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                o = h(zt[b], xt[b])
                ds = o["profiles"][torch.arange(len(b)), rt[b]]
                loss = (sP[b] + ds - sH[b]).abs().mean() + torch.nn.functional.cross_entropy(o["regime_logits"], rt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return h

    f = F.folds_of(eid, a.fit_seed)
    oof = np.full((n, 3), np.nan)
    for j in range(F.NFOLD):
        tr, te = np.where(f != j)[0], np.where(f == j)[0]
        h = train(tr, a.fit_seed)
        with torch.no_grad():
            oof[te] = h(zt[te], xt[te])["regime_logits"].numpy()
        print(f"[tau] {tag} fold {j} ({time.time() - t0:.0f} s)", flush=True)
    assert np.isfinite(oof).all()
    lg = torch.from_numpy(oof)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curve = []
    for tau in grid:
        dec = R.decide(lg, float(tau)).numpy()
        rec = float((dec[r == 0] == 0).mean())
        far = float((dec[r == 1] == 0).mean())
        curve.append({"tau": float(tau), "decel_recall": rec, "decel_false_on_cruise": far, "J": rec - far})
    J = np.array([c["J"] for c in curve])
    best = int(np.flatnonzero(J == J.max()).max())
    am = oof.argmax(1)
    res = {"label": "EXPLORATORY (prereg s1 'reported beside'); tau fixed on the calibration split, never on eval", "head": tag, "head_sha256": hsha,
           "wd": wd, "epochs": ep, "n_rows": int(n), "n_episodes": int(len(np.unique(eid))), "tau": float(grid[best]), "J_at_tau": float(J[best]),
           "argmax_oof": {"decel_recall": float((am[r == 0] == 0).mean()), "decel_false_on_cruise": float((am[r == 1] == 0).mean()),
                          "accuracy": float((am == r).mean())},
           "curve": curve, "rule": "max Youden J (decel recall - decel false rate on GT cruise), OOF, ties -> larger tau",
           "wall_s": round(time.time() - t0, 1)}
    np.savez(RC.SCRATCH / "fits" / f"{tag}_oof.npz", oof_logits=oof, r_star=r)
    RC.jdump(res, RC.SCRATCH / "fits" / f"{tag}_tau.json")
    print(f"[tau] {tag} tau={res['tau']} J={res['J_at_tau']:.4f} argmax-OOF {res['argmax_oof']}  ZZTAUOKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
