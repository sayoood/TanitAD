"""P4 — the NEXT LEVER after the closing-rate null.

⭐ THE QUESTION THE NULL LEFT BEHIND. `lead_closing_mps` did not decode from
`_last_state` on ANY arm. Two very different things produce that:

  (A) the TRUNK genuinely does not represent relative motion  -> upstream fix,
      expensive: objective/architecture.
  (B) `_last_state` DISCARDS it. It collapses a 4-frame window into ONE tensor
      (`field[:,-1] + motion_in(field[:,-1]-field[:,-2])`), so a rate could be
      present ACROSS the field sequence and absent from the collapsed state.
      -> cheap fix, planner-side: give the cost a temporal pair.

⛔ These are NOT the same defect and the first is ~100x the work of the second.
This probe separates them, and it is nearly free: the fields are already banked.

ARMS (same targets, same splits, same estimator, same controls as probe.py):
  field_t          the collapsed state          -- reproduces the null (control)
  field_diff       field[t] - field[t-1]        -- the explicit temporal DIFFERENCE
  field_pair       [field[t], field[t]-field[t-1]]  -- state AND difference
  pix_diff         pixel[t] - pixel[t-1]        -- the RAW-INPUT FLOOR for motion
  constant                                      -- must read EXACTLY 0
  shuffle_within_clip                           -- clip identity control

⚠️ NOTE ON WHAT IS BANKED. The bank stores `_last_state(field)` per row, i.e.
the collapsed state -- so `field[t]-field[t-1]` here is a difference OF collapsed
states across the 0.2 s grid, not the intra-window field sequence. That is the
right object anyway: it is exactly what a cost function could be handed without
retraining anything.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import (LAMBDAS, boot_ci, paired_boot_delta, pick_lambda,  # noqa
                   r2_skill, ridge_fit, stage_a_channel_pca, stage_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="/home/nvidia/percprobe/bank")
    ap.add_argument("--out", default="/home/nvidia/percprobe/raw/motion_probe.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--final-k", type=int, default=128)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.bank, "*.npz")))
    clips = [os.path.basename(p)[:-4] for p in files]
    data = [np.load(p, allow_pickle=True) for p in files]
    print("clips: %d" % len(clips), flush=True)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(clips))
    fit_clips = set(np.array(clips)[perm[:int(round(0.6 * len(clips)))]])
    fit_mask = [c in fit_clips for c in clips]

    n_per = np.array([len(z["gap_unc"]) for z in data])
    clip_ids = np.concatenate([[c] * n for c, n in zip(clips, n_per)])
    fit_rows = np.array([c in fit_clips for c in clip_ids])
    score_rows = ~fit_rows

    # ---- targets (identical definitions to probe.py) ----------------------
    gap = np.concatenate([z["gap_unc"].astype(np.float64) for z in data])
    closing = np.full(len(gap), np.nan)
    prev_ok = np.zeros(len(gap), bool)     # row i has a valid i-1 partner
    pos = 0
    for z in data:
        n = len(z["gap_unc"])
        g = z["gap_unc"].astype(np.float64)
        rw = z["rows"].astype(np.int64)
        tk = np.asarray(z["trk"]).astype(str)
        same = (tk[1:] == tk[:-1]) & (tk[1:] != "") & (rw[1:] - rw[:-1] == 1)
        d = np.full(n, np.nan)
        d[1:] = np.where(same, (g[1:] - g[:-1]) / 0.2, np.nan)
        closing[pos:pos + n] = d
        adj = np.zeros(n, bool)
        adj[1:] = (rw[1:] - rw[:-1] == 1)
        prev_ok[pos:pos + n] = adj
        pos += n

    # ---- build the three feature banks ------------------------------------
    def chan_reduce(key):
        mu, P = stage_a_channel_pca(data, fit_mask, key, np.random.RandomState(args.seed + 17))
        out = []
        for z in data:
            a = z[key].astype(np.float32)
            out.append(((a - mu) @ P).reshape(len(a), -1))
        return np.concatenate(out, 0)

    F = chan_reduce("field")
    PIXR = np.concatenate([z["pix"].reshape(len(z["pix"]), -1).astype(np.float32) / 255.0
                           for z in data], 0)

    def prev_of(X):
        """X[i-1] within the same clip on adjacent rows; NaN-guarded by prev_ok."""
        out = np.zeros_like(X)
        pos = 0
        for z in data:
            n = len(z["gap_unc"])
            blk = X[pos:pos + n]
            sh = np.vstack([blk[:1], blk[:-1]])
            out[pos:pos + n] = sh
            pos += n
        return out

    Fd = F - prev_of(F)
    Pd = PIXR - prev_of(PIXR)

    arms_raw = {
        "field_t": F,
        "field_diff": Fd,
        "field_pair": np.concatenate([F, Fd], 1),
        "pix_diff": Pd,
    }
    Zs = {}
    for k, X in arms_raw.items():
        Z, kk = stage_b(X, fit_rows, args.final_k)
        Zs[k] = Z
        print("arm %-12s raw_d=%-7d -> d=%d" % (k, X.shape[1], kk), flush=True)
    del arms_raw, F, Fd, PIXR, Pd

    results = {"_meta": {"seed": args.seed, "n_clips": len(clips),
                         "final_k": args.final_k, "n_boot": args.n_boot,
                         "function_class": "ridge (LINEAR)",
                         "estimator": "episode-cluster bootstrap over SCORED clips",
                         "note": "field_diff is a difference of COLLAPSED states "
                                 "on the 0.2 s grid, which is exactly what a cost "
                                 "could be handed without retraining"}}

    targets = {
        "lead_closing_mps_cap30": (closing, np.isfinite(closing) & prev_ok
                                   & np.isfinite(gap) & (gap <= 30.0)),
        "lead_gap_m_cap30": (gap, np.isfinite(gap) & (gap <= 30.0) & prev_ok),
    }

    for tname, (y, ok) in targets.items():
        y = np.asarray(y, np.float64)
        results[tname] = {}
        fit = fit_rows & ok
        sco = score_rows & ok
        if fit.sum() < 50 or sco.sum() < 50:
            print("SKIP %s" % tname, flush=True)
            continue
        fm = float(y[fit].mean())
        results[tname]["constant"] = {
            "r2": r2_skill(y[sco], np.full(int(sco.sum()), fm), fm),
            "d": 0, "n_score": int(sco.sum())}
        print("\n=== %s ===" % tname, flush=True)
        print("  %-22s R2=%+.6f  (control)" % ("constant", results[tname]["constant"]["r2"]),
              flush=True)
        preds = {}
        for a in ("pix_diff", "field_t", "field_diff", "field_pair"):
            Z = Zs[a]
            lam = pick_lambda(Z[fit], y[fit], clip_ids[fit], seed=args.seed)
            w = ridge_fit(Z[fit], y[fit] - fm, lam)
            pr = Z[sco] @ w + fm
            preds[a] = pr
            r2 = r2_skill(y[sco], pr, fm)
            lo, hi = boot_ci(y[sco], pr, fm, clip_ids[sco], n_boot=args.n_boot,
                             seed=args.seed)
            results[tname][a] = {"r2": r2, "ci": [lo, hi], "lam": float(lam),
                                 "d": int(Z.shape[1]), "n_score": int(sco.sum()),
                                 "n_clips_score": int(len(np.unique(clip_ids[sco])))}
            print("  %-22s R2=%+.4f  CI[%+.4f,%+.4f]  n=%d d=%d clusters=%d"
                  % (a, r2, lo, hi, sco.sum(), Z.shape[1],
                     len(np.unique(clip_ids[sco]))), flush=True)
        # within-clip shuffle control on the strongest arm
        ysh = y.copy()
        rsh = np.random.RandomState(args.seed + 99)
        for c in clips:
            m = (clip_ids == c) & ok
            if m.sum() > 1:
                v = ysh[m]
                ysh[m] = v[rsh.permutation(len(v))]
        fms = float(ysh[fit].mean())
        lam = pick_lambda(Zs["field_pair"][fit], ysh[fit], clip_ids[fit], seed=args.seed)
        w = ridge_fit(Zs["field_pair"][fit], ysh[fit] - fms, lam)
        prs = Zs["field_pair"][sco] @ w + fms
        results[tname]["shuffle_within_clip"] = {
            "r2": r2_skill(ysh[sco], prs, fms), "d": int(Zs["field_pair"].shape[1]),
            "n_score": int(sco.sum())}
        print("  %-22s R2=%+.4f  (clip-identity control)"
              % ("shuffle_within_clip", results[tname]["shuffle_within_clip"]["r2"]),
              flush=True)

        results[tname]["_paired"] = {}
        for a, b in (("field_diff", "field_t"), ("field_pair", "field_t"),
                     ("field_diff", "pix_diff"), ("field_pair", "pix_diff")):
            d = paired_boot_delta(y[sco], preds[a], preds[b], fm, clip_ids[sco],
                                  n_boot=args.n_boot, seed=args.seed)
            results[tname]["_paired"]["%s_minus_%s" % (a, b)] = d
            print("  PAIRED %-28s d=%+.4f CI[%+.4f,%+.4f] %s"
                  % ("%s - %s" % (a, b), d["delta"], d["ci"][0], d["ci"][1],
                     "EXCLUDES 0" if d["excludes_zero"] else "spans 0"), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
