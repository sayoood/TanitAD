"""THE STILL-FRAME CONTROL — the same encoder, fed an input with the motion REMOVED.

WHY (coordinator's mid-flight correction, 2026-08-03)
----------------------------------------------------
The latent-bottleneck stream measured that a single 32x32 grayscale STILL FRAME reads ego `speed` at
**93 % of the 800 ms learned latent** and **1.75x the best motion-only arm** — appearance dominates
motion. Since the situation labels are pure functions of the EGO POSE TRACK, a "vision-only"
situation classifier may therefore be riding an **appearance shortcut** (road type, lane markings,
scene furniture that co-vary with the manoeuvre) rather than perceiving the situation. That is the
indirect form of the PI's binding leak test: *does an input at inference contain something the label
was derived from?* — here by the route appearance -> speed -> ego-derived label.

WHAT THIS BUILDS. The identical substrate as `…/2026-08-03-sitclf-matched-capacity/
build_substrate.py` — same encoder (v1 `flagship4b-speedjerk-30k` @ 29999, STRICT-loaded, frozen),
same clips in the same order, same labels — with ONE change: the 9-channel input is the LAST RGB
frame REPLICATED THREE TIMES, so the D-015 3-frame stack carries **zero** inter-frame motion while
keeping the exact tensor shape, dtype and preprocessing the encoder expects.

  frames_u8[t] = [rgb(t-2) | rgb(t-1) | rgb(t)]   ->   [rgb(t) | rgb(t) | rgb(t)]

⭐ WHY THIS IS THE RIGHT CONTROL AND `win=1` IS NOT QUITE. A `win=1` arm already shows that ONE
latent matches the 8-latent window — but that latent is itself computed from a 3-frame stack, so it
still contains 200 ms of motion. This removes the last of it. If the still-frame arm recovers most
of the real arm's lift, then **appearance, not motion, is what the situation classifier is reading**,
and the temporal story is not what carries the signal.

⚠️ Rows, labels, clip ids and folds are IDENTICAL to the real substrate by construction, so the
still-frame arm and the real arm can be compared with the PAIRED episode-cluster bootstrap.

usage:
  python build_stillframe_substrate.py --out <npz> [--limit 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data.situations import (HZ, anticipation_target,        # noqa: E402
                                     detect_intersection,
                                     detect_lane_change,
                                     detect_roundabout, kinematics)

SITS = ("lane_change", "roundabout", "intersection")
EGO_SCALE = np.array([10.0, 2.0, 0.5], dtype=np.float32)
LEAD_S = 3.0
V1_CKPT = r"C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt"
CACHES = ((r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74",
           "train"),
          (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
           "val"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_trunk(device):
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel

    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    w = WorldModel(flagship4b_config())
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    ro = {k[len("readout."):]: v for k, v in sd.items() if k.startswith("readout.")}
    w.encoder.load_state_dict(enc, strict=True)
    w.readout.load_state_dict(ro, strict=True)
    w = w.to(device).eval()
    for p in w.parameters():
        p.requires_grad_(False)
    meta = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)),
            "encoder_params": int(sum(v.numel() for v in enc.values())),
            "readout_params": int(sum(v.numel() for v in ro.values())),
            "state_dim": int(w.state_dim), "strict_load": True, "frozen": True}
    log(f"trunk STRICT-loaded: {json.dumps(meta)}")
    return w, meta


def labels_for(P):
    K = kinematics(np.asarray(P, dtype=np.float64))
    T = int(K["T"])
    ev = {"lane_change": detect_lane_change(K),
          "roundabout": detect_roundabout(K, bracket=True),
          "intersection": detect_intersection(K, cross=None)[0]}
    y = np.zeros((T, 3), bool)
    v = np.zeros((T, 3), bool)
    for i, s in enumerate(SITS):
        y[:, i], v[:, i] = anticipation_target(T, ev[s], lead_s=LEAD_S)
    E = np.stack([K["v"], K["alon_pre"], K["omega_pre"]], 1).astype(np.float32) / EGO_SCALE
    return y, v, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stillframe_substrate.npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    files = []
    for root, tag in CACHES:
        fs = sorted(glob.glob(os.path.join(root, "ep_*.pt")))
        files += [(f, tag) for f in fs]
        log(f"{tag}: {len(fs)} episodes in {root}")
    if a.limit:
        files = files[:a.limit]

    world, tmeta = load_trunk(device)
    F, Y, V, E, CC, TAG, TT = [], [], [], [], [], [], []
    checked = False
    t0 = time.time()
    for k, (f, tag) in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        fr = d["frames_u8"]
        P = d["poses"]
        P = P.numpy() if torch.is_tensor(P) else np.asarray(P)
        if int(fr.shape[0]) != int(P.shape[0]):
            raise SystemExit(f"C-FID: {f} frames {fr.shape[0]} != poses {P.shape[0]}")
        if int(fr.shape[1]) != 9:
            raise SystemExit(f"expected a 9-channel 3-frame stack, got {tuple(fr.shape)}")
        y, v, e = labels_for(P)
        feats = np.empty((fr.shape[0], world.state_dim), np.float16)
        with torch.no_grad():
            for b in range(0, fr.shape[0], a.batch):
                raw = fr[b:b + a.batch].to(device, non_blocking=True).float().div_(255.0)
                # ⭐ THE ONE CHANGE: replicate the LAST rgb frame across all three slots
                x = raw[:, 6:9].repeat(1, 3, 1, 1)
                if not checked:
                    # both sides derived from the SAME device tensor. An earlier version
                    # rebuilt the comparison on the CPU and compared with torch.equal; the
                    # two differed by 5.96e-08 — float32 rounding of /255 on GPU vs CPU, not
                    # a defect. Comparing within one device makes exact equality correct again.
                    assert torch.equal(x[:, 0:3], x[:, 3:6]), "slots 0 and 1 differ"
                    assert torch.equal(x[:, 3:6], x[:, 6:9]), "slots 1 and 2 differ"
                    assert torch.equal(x[:, 6:9], raw[:, 6:9]), "last frame must be untouched"
                    moved = float((raw[:, 0:3] - raw[:, 6:9]).abs().mean())
                    assert moved > 1e-4, ("the source stack carries no motion either — "
                                          f"mean |frame(t-2) - frame(t)| = {moved}")
                    log(f"  motion-removal VERIFIED: three slots identical, last frame exact; "
                        f"the REAL stack's mean |rgb(t-2) - rgb(t)| was {moved:.5f}")
                    checked = True
                feats[b:b + a.batch] = world.encode(x).to(torch.float16).cpu().numpy()
        F.append(feats); Y.append(y); V.append(v); E.append(e)
        CC.append(np.full(len(y), k, np.int32))
        TAG.append(np.full(len(y), 0 if tag == "train" else 1, np.int8))
        TT.append(np.arange(len(y), dtype=np.int32))
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            log(f"  {k+1}/{len(files)} clips ({el:.0f}s, {el/(k+1):.2f}s/clip)")

    F = np.concatenate(F); Y = np.concatenate(Y); V = np.concatenate(V)
    E = np.concatenate(E); CC = np.concatenate(CC); TAG = np.concatenate(TAG)
    TT = np.concatenate(TT)
    log(f"still-frame substrate: {F.shape[0]:,} x {F.shape[1]}, {len(np.unique(CC))} clips")
    np.savez(a.out, F=F, Y=Y.astype(np.uint8), V=V.astype(np.uint8), E=E,
             clip_cluster=CC, cache_tag=TAG, t=TT, situations=np.array(SITS))
    meta = {"_what": "STILL-FRAME control substrate: last RGB frame replicated 3x, motion removed",
            "trunk": tmeta, "caches": [c[0] for c in CACHES], "n_clips": len(files),
            "n_frames": int(F.shape[0]), "state_dim": int(F.shape[1]),
            "lead_s": LEAD_S, "hz": HZ, "motion_removed": True,
            "identical_to_real_substrate_except": "the 9-channel input carries no inter-frame motion",
            "wallclock_s": round(time.time() - t0, 1)}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
