"""E-DEC-54 — THE EGO CENSUS'S NULL DISTRIBUTION, MEASURED RATHER THAN ANECDOTED.

⭐ WHY. The 8-arm census (E-DEC-53) reported one cell above threshold —
`postrain10k`'s predicted Δyaw at t 3.00 — and I argued it was barely above the
noise because ANOTHER cell, `rdw8s30k`'s yaw-rate at **−2.64**, is physically
meaningless (a latent cannot anti-carry its own yaw-rate) and therefore must be
noise. **That argument rests on ONE anecdotal cell.** An anecdote about the null is
still an anecdote.

⛔ THIS REPLACES IT WITH A MEASUREMENT. Run the identical panel — same clips, same
real targets, same RFF+ridge, same clip-disjoint λ, same within-clip Pearson r,
same time-shuffled control — but with the LATENT REPLACED BY GAUSSIAN NOISE of the
same shape. A random latent provably carries nothing about the ego, so **every t it
produces is a draw from the null.** Repeat over seeds and read the tail.

⭐ THE OUTPUT IS THE NUMBER TOMORROW'S READ NEEDS: the |t| that the panel reaches
by chance. `PREREG_POSTRAIN30K_EGO_REPLICATION.md` set its REPLICATED threshold at
t > 2.6 from the anecdote; this either supports that or corrects it, and it is
being run BEFORE `postrain30k` is scored so the threshold cannot be tuned to the
result.

⚠️ NO GPU AND NO MODEL. Only the targets are real (speed and yaw from `poses`), so
this needs neither a checkpoint nor a forward pass — which is also why there is no
excuse for having asserted the null instead of measuring it.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "egonull.json")))
N_CLIPS, F, K, W = 20, 100, 4, 6
D_Z = int(os.environ.get("SPD_DZ", "2048"))
SEEDS = int(os.environ.get("SPD_SEEDS", "6"))
TN = ("speed_t (LEVEL)", "yawrate_t (LEVEL)", "dv_4tick (CHANGE)",
      "dyaw_4tick (CHANGE)")


def wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from rangeprobe_rff import rff_fold, within_clip_r

    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    T = {k: [] for k in TN}
    n_ok = 0
    for c in clips:
        d = torch.load(c, map_location="cpu", weights_only=False)
        yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
        v = np.asarray(d["poses"], dtype=np.float64)[:, 3].ravel()
        n = min(len(v), len(yaw), F)
        tt = {k: [] for k in TN}
        for i in range(0, n - W - K):
            j = i + W - 1
            if j + K >= n:
                break
            tt["speed_t (LEVEL)"].append([v[j]])
            tt["yawrate_t (LEVEL)"].append([wrap(yaw[j + 1] - yaw[j])])
            tt["dv_4tick (CHANGE)"].append([v[j + K] - v[j]])
            tt["dyaw_4tick (CHANGE)"].append([wrap(yaw[j + K] - yaw[j])])
        if len(tt["speed_t (LEVEL)"]) < 25:
            continue
        n_ok += 1
        for k in TN:
            T[k].append(np.asarray(tt[k], dtype=np.float64))

    print(f"\n  E-DEC-54 — THE EGO CENSUS NULL, MEASURED")
    print(f"  {n_ok} clips · real targets · RANDOM latent d={D_Z} · {SEEDS} seeds")
    print(f"  every t below is a draw from the null: the latent carries NOTHING\n")
    print(f"  {'seed':>5}" + "".join(f"{k.split(' ')[0]:>12}" for k in TN))
    print("  " + "-" * 55)
    allt = []
    rep = {"_evidence_class": "MEASURED (ours; dev-box, CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "n_clips": n_ok, "d_z": D_Z,
           "seeds": SEEDS, "per_seed": {}}
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        Z = [rng.standard_normal((len(y), D_Z)) for y in T[TN[0]]]
        row = []
        for k in TN:
            Y = T[k]
            Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
            tr, sh = [], []
            for i in range(len(Z)):
                idx = [q for q in range(len(Z)) if q != i]
                for Yv, sink in ((Y, tr), (Ysh, sh)):
                    pred, _ = rff_fold([Z[q] for q in idx],
                                       [Yv[q] for q in idx], Z[i])
                    sink.append(within_clip_r(pred, Yv[i].ravel()))
            dd = np.array(tr) - np.array(sh)
            t = float(dd.mean()) / max(
                float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            row.append(t)
            allt.append(t)
        rep["per_seed"][str(s)] = {k: round(x, 2) for k, x in zip(TN, row)}
        print(f"  {s:>5}" + "".join(f"{x:>12.2f}" for x in row), flush=True)

    a = np.abs(np.array(allt))
    rep["n_null_draws"] = len(allt)
    rep["abs_t"] = {"max": round(float(a.max()), 2),
                    "p95": round(float(np.percentile(a, 95)), 2),
                    "p90": round(float(np.percentile(a, 90)), 2),
                    "median": round(float(np.median(a)), 2)}
    print(f"\n  {len(allt)} null draws · |t| median {np.median(a):.2f} · "
          f"p90 {np.percentile(a, 90):.2f} · p95 {np.percentile(a, 95):.2f} · "
          f"max {a.max():.2f}")
    thr = float(np.percentile(a, 95))
    rep["recommended_threshold"] = round(thr, 2)
    rep["prereg_threshold_was"] = 2.6
    if thr <= 2.6:
        rep["verdict"] = (f"the pre-registered t > 2.6 threshold is CONSERVATIVE — "
                          f"the measured null p95 is {thr:.2f}. Keep 2.6; it does "
                          f"not need loosening and loosening it after seeing the "
                          f"data would be exactly the defect this file exists to "
                          f"prevent.")
    else:
        rep["verdict"] = (f"⛔ the pre-registered t > 2.6 threshold is TOO LOOSE — "
                          f"the measured null p95 is {thr:.2f}. postrain10k's 3.00 "
                          f"is inside the null and E-DEC-53 must be reported as "
                          f"ENTIRELY NULL.")
    print(f"  => {rep['verdict']}\n")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
