"""E-SEED-1 stage 1 — build the frame bank + labels from the LOCAL val epcache.

⛔ CONTENT ASSERTIONS, not exit codes. The B1 epcache trap (CLAUDE.md): a decode
that raises into a pre-allocated memmap leaves a full-size file of ZEROS and the
job can still exit 0. Every bank here is checked for non-zero content and its
mean is printed.

Frame contract (MEASURED, stack/tanitad/data/physicalai.py:144 CORPUS_META):
  frames_u8 [T, 9, 256, 256] uint8 -- 3 stacked frames as channels; the NEWEST
  frame is the LAST 3 channels (the [:, -3:] slice O7Distill takes).
  actions [T, 2] = (steer_road_rad, accel_mps2)
  poses   [T, 4] = (x_east_m, y_north_m, yaw_rad, v_mps)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EPDIR = Path(r"C:\Users\Admin\tanitad-data\physicalai\_epcache"
             r"\physicalai-val-bb543bdf7836")
OUT = Path(__file__).resolve().parent / "eseed1"
OUT.mkdir(exist_ok=True)

N_EP = 90          # episodes 0..89
STRIDE = 6         # pair start stride
DT = 0.1           # 10 Hz (CORPUS_META hz)


def main() -> int:
    eps = sorted(EPDIR.glob("ep_*.pt"))[:N_EP]
    if len(eps) < N_EP:
        print(f"REFUSE: only {len(eps)} episodes, wanted {N_EP}")
        return 2

    rows = []          # (ep_idx, t)
    frames = []        # uint8 [3,256,256] for t and t+1 -- stored as pairs
    lab = []           # per row: v, yaw_rate, steer, accel, dv
    for i, p in enumerate(eps):
        d = torch.load(p, map_location="cpu", weights_only=False)
        f = d["frames_u8"]                       # [T,9,256,256] uint8
        act = d["actions"].numpy()               # [T,2]
        pos = d["poses"].numpy()                 # [T,4]
        T = f.shape[0]
        # yaw rate from unwrapped yaw
        yaw = np.unwrap(pos[:, 2].astype(np.float64))
        yr = np.gradient(yaw, DT).astype(np.float32)
        v = pos[:, 3].astype(np.float32)
        ts = list(range(0, T - 1, STRIDE))
        for t in ts:
            frames.append(f[t, 6:9].numpy())      # newest frame at t
            frames.append(f[t + 1, 6:9].numpy())  # newest frame at t+1
            rows.append((i, t))
            lab.append((v[t], yr[t], act[t, 0], act[t, 1], v[t + 1] - v[t]))
        del d, f
        if (i + 1) % 15 == 0:
            print(f"  ..{i+1}/{N_EP} eps, {len(rows)} pairs", flush=True)

    X = np.stack(frames).astype(np.uint8)         # [2n, 3, 256, 256]
    L = np.asarray(lab, dtype=np.float32)         # [n, 5]
    R = np.asarray(rows, dtype=np.int32)          # [n, 2]

    # ---- CONTENT ASSERTIONS (never trust the exit code) --------------------
    assert X.shape[0] == 2 * L.shape[0], (X.shape, L.shape)
    fm = float(X.mean())
    nz = float((X != 0).mean())
    print(f"[bank] frames {X.shape} mean={fm:.3f} nonzero_frac={nz:.4f}")
    if fm < 1.0 or nz < 0.5:
        print("⛔ REFUSE: frame bank looks zero-filled (the memmap trap)")
        return 3
    for j, nm in enumerate(("v_mps", "yaw_rate", "steer", "accel", "dv")):
        c = L[:, j]
        print(f"[bank] {nm:9s} n={len(c)} mean={c.mean():+.4f} "
              f"std={c.std():.4f} finite={np.isfinite(c).all()}")
        if not np.isfinite(c).all() or c.std() <= 0:
            print(f"⛔ REFUSE: label {nm} degenerate")
            return 4

    np.save(OUT / "frames_u8.npy", X)
    np.save(OUT / "labels.npy", L)
    np.save(OUT / "rows.npy", R)
    meta = dict(n_pairs=int(L.shape[0]), n_frames=int(X.shape[0]),
                n_episodes=N_EP, stride=STRIDE, dt=DT,
                epdir=str(EPDIR), frame_mean=fm, nonzero_frac=nz,
                label_cols=["v_mps", "yaw_rate", "steer", "accel", "dv"],
                frame_slice="channels 6:9 (newest frame)",
                evidence_class="MEASURED (ours; local val epcache)")
    (OUT / "bank_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bank] wrote {OUT}  ({X.nbytes/2**30:.2f} GiB frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
