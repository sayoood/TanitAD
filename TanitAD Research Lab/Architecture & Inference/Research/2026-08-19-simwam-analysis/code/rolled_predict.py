"""H-PROOF-7: the FIRST HONEST multi-step prediction measurement in the programme.

⛔ WHY EVERY PREVIOUS ONE WAS WRONG (C139). `rollout_transitions` builds the
rollout from `t[1]` -- the h=1 head ONLY -- applied autoregressively. Every arm
ran `--o5-k 1`, so k_roll = 1 and the h=2 / h=4 HEADS NEVER RECEIVED A GRADIENT:
their weight norms are 0.02612 / 0.02614 in every arm at 2k AND at 30k, the
untouched 1e-3 init. Querying `predictor(...)[2]` therefore measured an untrained
head and I read the result as a property of the model.

⇒ THE CORRECT READ-OUT ROLLS. Feed the prediction back in and step the h=1 head
k times, exactly as O5 does during training:

    z_hat(1) = P(window)[1]
    z_hat(j) = P(window shifted, z_hat(j-1) appended)[1]

and compare z_hat(j) against the TRUE z at t+j.

Statistic: MEAN-CENTRED cos vs a 200-draw PERMUTATION NULL (C137 retired
divergence-over-movement). Reported per rolled step with its z.

Arms: `o5k4` / `o5k8` trained with k_roll = 4 / 8 -- the fix -- against `rdw8`
(k_roll = 1), which is the control that should still fail beyond step 1.
⚠️ `rdw8` failing at j>=2 is EXPECTED and is what makes the comparison readable;
if it were to succeed, the rollout would be doing something O5 never trained.
"""
from __future__ import annotations

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
OUT = SP / "h_proof7_rolled.json"
ARMS = ["rdw8", "o5k4", "o5k8"]
KMAX = 6
F = 100


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    clips = sorted(VAL.glob("*.v2ep.pt"))[:8]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    missing = [a for a in ARMS if a not in present]
    print(f"\n  H-PROOF-7 ROLLED multi-step · arms {present}"
          + (f"  MISSING {missing}" if missing else "") + "\n", flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "method": "autoregressive roll of the h=1 head; mean-centred cos vs a 200-draw "
                     "permutation null at each rolled step",
           "arms_missing": missing, "arms": {}}
    print(f"  {'arm':<8}{'o5_k':>6}" + "".join(f"{'j=' + str(j):>16}" for j in range(1, KMAX + 1)))
    print("  " + "-" * (14 + 16 * KMAX), flush=True)

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        D = {j: [] for j in range(1, KMAX + 1)}
        T = {j: [] for j in range(1, KMAX + 1)}
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                for i in range(0, max(1, len(zt) - W - KMAX - 1), 5):
                    win = zt[i:i + W][None].to(dev).clone()
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    for j in range(1, KMAX + 1):
                        a0 = i + j - 1
                        aa = act[a0:a0 + W][None].to(dev)
                        if aa.shape[1] != W:
                            break
                        zh = w.predictor(win, torch.cat([aa, vv], -1))[1]  # h=1 head ONLY
                        zh = zh.reshape(1, -1)[:, :zt.shape[1]]
                        base = zt[i + W - 1].to(dev)                        # last OBSERVED z
                        k = i + W - 1 + j
                        if k >= len(zt):
                            break
                        D[j].append((zh.reshape(-1) - base).cpu().numpy())
                        T[j].append((zt[k].to(dev) - base).cpu().numpy())
                        # roll: shift the window and append the PREDICTION
                        win = torch.cat([win[:, 1:], zh[None]], dim=1)
        row = {"step": int(st), "by_rolled_step": {}}
        cells = []
        for j in range(1, KMAX + 1):
            if len(D[j]) < 20:
                cells.append(f"{'-':>16}")
                continue
            d = np.stack(D[j]).astype(np.float64)
            t = np.stack(T[j]).astype(np.float64)
            d -= d.mean(0, keepdims=True)
            t -= t.mean(0, keepdims=True)
            den = max(float(np.linalg.norm(d) * np.linalg.norm(t)), 1e-30)
            cos = float((d * t).sum()) / den
            null = [float((d * t[rng.permutation(len(t))]).sum()) / den for _ in range(200)]
            z_ = (cos - float(np.mean(null))) / max(float(np.std(null)), 1e-12)
            row["by_rolled_step"][str(j)] = {"cos": round(cos, 4), "z": round(z_, 2),
                                             "n": len(d)}
            cells.append(f"{cos:>8.4f}/{z_:>6.1f}")
        rep["arms"][arm] = row
        k_used = "?"
        print(f"  {arm:<8}{k_used:>6}" + "".join(cells), flush=True)
        del w
        torch.cuda.empty_cache()

    def deepest(a):
        r = rep["arms"].get(a, {}).get("by_rolled_step", {})
        got = [int(j) for j, v in r.items() if v["z"] > 2.0]
        return max(got) if got else 0
    rep["deepest_significant_rolled_step"] = {a: deepest(a) for a in present}
    base = deepest(present[0]) if present else 0
    better = [a for a in present[1:] if deepest(a) > base]
    rep["verdict"] = (
        f"deepest rolled step with z>2: {rep['deepest_significant_rolled_step']}. "
        + (f"⭐ {better} PREDICT DEEPER than the k_roll=1 control — the C139 fix works and "
           f"multi-step prediction was a TRAINING-CONFIG problem, not a model limit."
           if better else
           "No arm predicts deeper than the k_roll=1 control ⇒ deepening the rollout did NOT "
           "buy multi-step competence, and the limit is real rather than configural."))
    print(f"\n  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
