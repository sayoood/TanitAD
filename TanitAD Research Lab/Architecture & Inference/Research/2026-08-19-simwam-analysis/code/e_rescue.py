"""E-RESCUE — can the residual-scale defect be fixed WITHOUT retraining v6?

⛔ THE PI'S QUESTION, exactly: "is the bug in the training code of v6 and that's
why it must restart? or can we fix it without retraining?"

THE BUG is an INITIALISATION defect: `OperativePredictor`'s residual delta heads
were default-initialised, so at step 0 they emit a delta O(1) per dim while the
latent moves 0.000892 per tick. Initialisation only sets where a run STARTS, so
patching the code does nothing to an existing checkpoint — strictly, the benefit
needs a restart (~9 days at 26.5 s/step).

⭐ BUT THERE MAY BE A RESCUE, AND IT IS CHEAP TO TEST. The heads HAVE learned
something (`o5_step1` fell 0.32 -> 0.137 over 26k steps). If the learned delta
has roughly the right DIRECTION but the wrong MAGNITUDE, then scaling the head
weights by a single scalar alpha shrinks the output proportionally:

      delta_scaled = alpha * delta          (scale W and b together)

and some alpha may beat the hold baseline on the EXISTING weights. That is
checkpoint surgery, not a retrain.

⚠️⚠️ THE TICK TRAP, and why this file encodes its own latents. The predictor's
tick is `dt = 0.1 s` = ONE frame. The banked `cache_tok20000_s4` is STRIDE 4, so
consecutive banked rows are 0.4 s apart and a 6-row window would span 2.4 s
instead of 0.6 s — the predictor would be fed a history it never trained on, and
every number would be meaningless. So this encodes STRIDE-1 latents directly
through the frozen trunk.

⚠️ The encoder takes NINE channels — `n_stack: 3`, three frames stacked as
channels (`weight [768, 9, 16, 16]`). A 3-channel input is refused, correctly.

WHAT IS REPORTED, per alpha:
    mae_pred   mean|zhat_{t+1} - z_{t+1}|     the predictor
    mae_hold   mean|z_t      - z_{t+1}|       ⛔ THE BASELINE
    ratio      mae_pred / mae_hold            <1 BEATS hold; >1 worse

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
N_STACK = 3
ALPHAS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0)


def frames_of(cid: str):
    d = torch.load(EPS / f"{cid}.v2ep.pt", map_location="cpu",
                   weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
        np.int64)
    n = len(off) - 1
    return d, raw, off, n


def encode_clip(stack, cid: str, dev, max_frames: int):
    """STRIDE-1 latents for one clip, through the frozen trunk.

    ⚠️ Input is the 3-FRAME CHANNEL STACK the encoder was built for, so frame i
    is encoded from frames (i-2, i-1, i), clamped at the clip start."""
    d, raw, off, n = frames_of(cid)
    n = min(n, max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    Z, B = [], 16
    with torch.no_grad():
        for s in range(0, n, B):
            chunk = []
            for i in range(s, min(s + B, n)):
                idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)          # [b,1,9,H,W]
            Z.append(stack.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z), d["actions"].float()[:n], d["poses"].float()[:n, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description="E-RESCUE")
    ap.add_argument("--ckpt", default=str(SP / "ckpt/v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--config-json", default=str(SP / "sp2/v6F_config.json"))
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames-per-clip", type=int, default=140)
    ap.add_argument("--out", default=str(SP / "e_rescue.json"))
    a = ap.parse_args()

    import e_pred_probe as E
    from tanitad.models.flagship_v15 import SPEED_SCALE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, step = E.load_world(Path(a.ckpt), Path(a.config_json), dev)
    stack = world.stack
    W = int(getattr(world, "window", 6))
    print(f"  trunk FROZEN @ step {step} · window {W} · tick dt=0.1s "
          f"(STRIDE-1 latents, not the stride-4 cache)", flush=True)

    cids = sorted(p.name.split(".")[0] for p in EPS.glob("*.v2ep.pt"))[:a.clips]
    Zs, As, Vs = [], [], []
    t0 = time.time()
    for n, cid in enumerate(cids, 1):
        z, act, spd = encode_clip(stack, cid, dev, a.frames_per_clip)
        Zs.append(z); As.append(act); Vs.append(spd)
        print(f"    [{n}/{len(cids)}] {cid[:8]} {len(z)} frames "
              f"({time.time() - t0:.0f}s)", flush=True)

    # windows of W consecutive STRIDE-1 latents; target = the next tick
    win_z, win_a, win_v, tgt = [], [], [], []
    for z, act, spd in zip(Zs, As, Vs):
        for i in range(len(z) - W):
            win_z.append(z[i:i + W]); win_a.append(act[i:i + W])
            win_v.append(spd[i]); tgt.append(z[i + W])
    Zw = torch.stack(win_z); Aw = torch.stack(win_a)
    Vw = torch.stack(win_v); T = torch.stack(tgt)
    print(f"  {len(Zw):,} windows · latent mean|z| {float(Zw[:, -1].abs().mean()):.6f}",
          flush=True)

    mae_hold = float((Zw[:, -1] - T).abs().mean())
    print(f"  ⛔ HOLD baseline  mae {mae_hold:.6f}\n", flush=True)

    W1 = stack.predictor_op.heads["1"].weight.data.clone()
    B1 = stack.predictor_op.heads["1"].bias.data.clone()
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "can the residual-scale defect be fixed WITHOUT "
                       "retraining, by rescaling the delta heads?",
           "ckpt": Path(a.ckpt).name, "step": int(step),
           "n_windows": int(len(Zw)), "tick_s": 0.1,
           "mae_hold": round(mae_hold, 8), "alphas": {}}

    for al in ALPHAS:
        stack.predictor_op.heads["1"].weight.data = W1 * al
        stack.predictor_op.heads["1"].bias.data = B1 * al
        errs = []
        with torch.no_grad():
            for s in range(0, len(Zw), 64):
                zs = Zw[s:s + 64].to(dev)
                a2 = Aw[s:s + 64].to(dev)
                v = (Vw[s:s + 64].to(dev) / SPEED_SCALE)[:, None, None] \
                    .expand(-1, W, -1)
                zh = stack.predictor_op(zs, torch.cat([a2, v], -1))[1]
                errs.append((zh.cpu() - T[s:s + 64]).abs().mean(-1))
        mae = float(torch.cat(errs).mean())
        ratio = mae / mae_hold
        out["alphas"][f"{al:g}"] = {"mae_pred": round(mae, 8),
                                    "ratio_vs_hold": round(ratio, 4)}
        verdict = ("BEATS hold" if ratio < 0.98 else
                   "== hold" if ratio < 1.02 else "worse than hold")
        print(f"  alpha {al:<7g} mae {mae:.6f}  ratio {ratio:>9.2f}x  {verdict}",
              flush=True)
    stack.predictor_op.heads["1"].weight.data = W1
    stack.predictor_op.heads["1"].bias.data = B1

    best = min(out["alphas"], key=lambda k: out["alphas"][k]["ratio_vs_hold"])
    out["best_alpha"] = best
    out["best_ratio"] = out["alphas"][best]["ratio_vs_hold"]
    out["verdict"] = (
        "RESCUABLE: a scalar rescale of the existing heads beats the hold "
        "baseline, so no retrain is needed for this defect"
        if out["best_ratio"] < 0.98 else
        "NOT RESCUABLE by rescaling alone: even the best alpha fails to beat "
        "hold, so the learned delta direction is not usable and a retrain from "
        "an identity start is required")
    print(f"\n  best alpha {best} -> {out['best_ratio']}x\n  {out['verdict']}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
