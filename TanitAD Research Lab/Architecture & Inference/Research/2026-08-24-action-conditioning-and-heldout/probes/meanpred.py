"""E-DEC-20b — DID THE PREDICTOR DIE, OR DID IT BECOME A CONSTANT-MOTION MODEL?

E-DEC-20 measured `splitfrz10k`'s mean-centred cos at 0.0007 (z 0.40) against
`splitfrz`'s 0.1872 (z 7.99) and I called the predictor "dead". But the weight
norms say the opposite happened: `predictor_op.heads.1.weight` GREW 2.676x
(0.5537 -> 1.4821) over those 8,000 steps. A predictor that unlearned would
shrink toward its 1e-3 init (the C139 signature); one that grew is doing MORE,
not less.

There is an obvious alternative that the headline statistic CANNOT see. Our
predictor metric is MEAN-CENTRED on purpose (C137 retired divergence-over-
movement). A model that learns the SYSTEMATIC component of latent motion --
essentially "the scene flows this way as the car moves" -- and nothing
window-specific will read:

    * a LARGE predicted delta,
    * a HIGH raw (uncentred) cos,
    * and a mean-centred cos of ~ZERO.

That is not a dead predictor. It is a constant-velocity model, and it would be a
completely different finding: "the frozen field admits only the average motion"
rather than "training destroys the predictor".

THE CONTROL THAT DECIDES IT (and it must read a known value):
`nrmse_meanonly` predicts every window with the DATASET-MEAN delta. If an arm's
`nrmse` equals it, the arm has learned exactly the mean and nothing else. If
`nrmse` is clearly below it, the arm carries window-specific information that the
centred cos is simply blind to. `nrmse_zero` = 1.0 by construction (predict no
motion at all) and is printed as the second anchor.

Reported per arm, h=1 only, on held-out val clips.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
OUT = SP / "meanpred.json"
# ⭐ SELECTION ONLY — the computation below is untouched. The list was
# hardcoded, so the pre-registered crossed-cell read (postrain30k vs
# postrain30k_freeze) could not have run at all; found by preflighting the
# instrument while the arm trains rather than after it finishes, which is
# the E-DEC-11 lesson (an analysis-time failure destroys a completed run).
ARMS = os.environ.get(
    "SPD_ARMS", "splitfrz,splitfrz10k,rdw8p30k,o5k4").split(",")
OUT = Path(os.environ.get("SPD_OUT", str(OUT)))   # the file imports Path, not pathlib
F = 100


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    clips = sorted(VAL.glob("*.v2ep.pt"))[:10]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-20b  ·  arms {present}  ·  {len(clips)} held-out val clips\n", flush=True)
    print(f"  {'arm':<13}{'step':>7}{'cos_raw':>9}{'cos_ctr':>9}{'z':>7}"
          f"{'meanfrac':>10}{'nrmse':>8}{'nrmse_mean':>12}{'verdict':>26}")
    print("  " + "-" * 101, flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "method": "h=1 head, one step. cos_raw is UNCENTRED; cos_ctr is mean-centred "
                     "with a 200-draw permutation null. nrmse_meanonly is the "
                     "CONSTANT-DELTA control and nrmse_zero = 1.0 by construction.",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        D, T = [], []
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                for i in range(0, max(1, len(zt) - W - 2), 5):
                    k = i + W
                    if k >= len(zt):
                        break
                    win = zt[i:i + W][None].to(dev)
                    if win.shape[1] != W:
                        break
                    aa = act[i:i + W][None].to(dev)
                    if aa.shape[1] != W:
                        break
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    zh = w.predictor(win, torch.cat([aa, vv], -1))[1]
                    zh = zh.reshape(1, -1)[:, :zt.shape[1]]
                    base = zt[i + W - 1].to(dev)
                    D.append((zh.reshape(-1) - base).cpu().numpy())
                    T.append((zt[k].to(dev) - base).cpu().numpy())
        d = np.stack(D).astype(np.float64)
        t = np.stack(T).astype(np.float64)

        cos_raw = float((d * t).sum() / max(np.linalg.norm(d) * np.linalg.norm(t), 1e-30))
        dc, tc = d - d.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
        den = max(float(np.linalg.norm(dc) * np.linalg.norm(tc)), 1e-30)
        cos_ctr = float((dc * tc).sum()) / den
        null = [float((dc * tc[rng.permutation(len(tc))]).sum()) / den for _ in range(200)]
        z_ = (cos_ctr - float(np.mean(null))) / max(float(np.std(null)), 1e-12)

        meanfrac = float(np.linalg.norm(d.mean(0)) /
                         max(np.sqrt((d ** 2).sum() / len(d)), 1e-30))
        tnorm = max(float(np.linalg.norm(t)), 1e-30)
        nrmse = float(np.linalg.norm(d - t)) / tnorm
        nrmse_mean = float(np.linalg.norm(t.mean(0, keepdims=True) - t)) / tnorm

        # a verdict that names the mechanism, not just the number
        if nrmse < nrmse_mean * 0.97:
            v = "BEATS the mean predictor"
        elif nrmse <= nrmse_mean * 1.03:
            v = "IS the mean predictor"
        else:
            v = "WORSE than the mean"
        rep["arms"][arm] = {"step": int(st), "cos_raw": round(cos_raw, 4),
                            "cos_centred": round(cos_ctr, 4), "z_centred": round(z_, 2),
                            "mean_fraction_of_prediction": round(meanfrac, 4),
                            "nrmse": round(nrmse, 4),
                            "nrmse_meanonly_control": round(nrmse_mean, 4),
                            "nrmse_zero_control": 1.0, "n_windows": len(d),
                            "verdict": v}
        print(f"  {arm:<13}{st:>7}{cos_raw:>9.4f}{cos_ctr:>9.4f}{z_:>7.2f}"
              f"{meanfrac:>10.4f}{nrmse:>8.4f}{nrmse_mean:>12.4f}{v:>26}")
        del w
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
