#!/usr/bin/env python3
"""D — WHICH UNIT DID THE TRAINED CHECKPOINT ACTUALLY LEARN?

The units question, put to the weights instead of to the source. For each window
feed the SAME banked checkpoint the SAME recorded action tensor twice:

  COMMAND  a2 = (a, steer)              — what `refav1_loader._kin_actions` emits
  GEOMETRY a2 = (a, tan(steer)/2.9)     — the same motion, spelled as curvature

and score the model's feature prediction against the RECORDED FUTURE features.

Prediction under the PI ruling: COMMAND wins on essentially every window, because
that is the tensor the loss was computed on. If GEOMETRY won, the model would have
learned to read the channel as a curvature and the ruling's premise would fail.

CPU only. `--n` windows (7 per episode in the banked selection).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval")

_spec = importlib.util.spec_from_file_location(
    "refav1_arm_units", r"C:\Users\Admin\tanitad-wt\taniteval\tools\refav1_arm.py")
ARM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ARM)

from tanitad.models.kinematic import STEER_WHEELBASE_M          # noqa: E402

DUMP = Path(r"C:\Users\Admin\refav1_eval_slice\t1_dump")
EPS = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
FP8 = Path(r"C:\Users\Admin\refav1_eval_slice\fp8")
CKPT = Path(r"C:\Users\Admin\refav1_eval_slice\ckpt\ckpt.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kwm", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, cfg, prov = ARM.load_model(str(CKPT), device="cpu")
    print(f"[units] ckpt step={prov.get('step')} a_dim={cfg.a_dim} "
          f"target_space={cfg.target_space} W={cfg.op_window}", flush=True)
    man = json.load(open(DUMP / "manifest.json", encoding="utf-8"))
    names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
    W, K = int(cfg.op_window), int(a.kwm)

    rows = []
    for fi, f in enumerate(sorted(glob.glob(str(DUMP / "ep*.npz")))):
        if len(rows) >= a.n:
            break
        nm = names[fi]
        z = np.load(f)
        feats = torch.load(FP8 / f"{nm}.pt", map_location="cpu", weights_only=True)
        o = torch.load(EPS / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        v = o["poses"][:, 3].float()
        st = o["actions"][:, 0].float()
        for i in range(z["g"].shape[0]):
            if len(rows) >= a.n:
                break
            t = int(z["ws"][i])
            if t - W + 1 < 0 or t + 1 + K > feats.shape[0]:
                continue
            fw = feats[t - W + 1:t + 1].float()[None]
            fu = feats[t + 1:t + 1 + K].float()[None]
            idx = torch.arange(t, t + K)
            fnx = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
            a_ch = (v[fnx] - v[idx * 2]) / 0.2
            cmd = torch.stack([a_ch, st[idx * 2]], dim=-1)[None]          # COMMAND
            geo = cmd.clone()
            geo[..., 1] = torch.tan(cmd[..., 1]) / STEER_WHEELBASE_M      # GEOMETRY
            v0 = torch.tensor([float(v[2 * t])])
            with torch.no_grad():
                lc = float(model(fw, cmd, future_feats=fu,
                                 v0=v0)["loss_feat_op"])
                lg = float(model(fw, geo, future_feats=fu,
                                 v0=v0)["loss_feat_op"])
            rows.append({"clip": nm, "t": t, "loss_command": lc,
                         "loss_geometry": lg, "command_wins": int(lc < lg),
                         "kappa_absmax": float(torch.tan(cmd[..., 1]).abs().max()
                                               / STEER_WHEELBASE_M)})
            print(f"  {nm[:8]} t={t:4d} cmd={lc:.6f} geo={lg:.6f} "
                  f"{'CMD' if lc < lg else 'GEO'}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    turn = df[df.kappa_absmax >= 0.01]
    print(f"\n[units] n={len(df)}  COMMAND wins {int(df.command_wins.sum())}/{len(df)}"
          f"   mean loss: command={df.loss_command.mean():.6f} "
          f"geometry={df.loss_geometry.mean():.6f} "
          f"(geometry is {df.loss_geometry.mean()/df.loss_command.mean():.4f}x)")
    if len(turn):
        print(f"[units] on the {len(turn)} TURNING windows (|kappa|>=0.01): "
              f"COMMAND wins {int(turn.command_wins.sum())}/{len(turn)}, "
              f"ratio {turn.loss_geometry.mean()/turn.loss_command.mean():.4f}x")
    print("[units] wrote", a.out)


if __name__ == "__main__":
    main()
