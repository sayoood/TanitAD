"""extract_states.py -- level-1 world-model states of ONE checkpoint on a fixed clip set (stage-1 reads of PREREG_V8F_PROOF_ADD12; note N7 section 4).

  python extract_states.py --tree /workspace/v8proof/tree --ckpt <ckpt_last.pt> --cache /workspace/v8proof/data/cache_pod --split train --lo 600 --hi 1800 --out <states.npz>

CPU only, one thread, ONE clip in memory at a time, no GPU. Clips = positions lo .. hi-1 of the ascending sha12 order of the split manifest (the order of extract_probe_set.py, so
row i of this file is row i of the probe set). Per clip: pool4x10 [T, 40, 1024] fp8 -> float32 -> the model's own adapter and position term (V8FMini._encode) -> WMAux.z1 -> float16 [T, D]
(D = 128 for a mean state at the default width, 640 for the grid state and for the wide mean control). The state of a frame depends on that frame only (no temporal context enters z1).
Writes <out> with Z (float16 [N, D]), clip_idx (int32 [N], index into the clip list), t (int16 [N]), clips; and <out>.json with the checkpoint sha256, its cfg state mode, D and the counts.
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(a.tree, "stack"))
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("grid", "pool4x10") != "pool4x10":
        raise SystemExit(f"REFUSED: checkpoint grid {ck.get('grid')!r}, this tool reads pool4x10")
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    if model.wm is None:
        raise SystemExit("REFUSED: the checkpoint has no world-model auxiliary, so no level-1 state")
    man = D.read_manifest(a.cache, a.split)
    order = sorted([c for c, _ in man], key=D.sha12)
    clips = order[a.lo:a.hi]
    Zs, Ci, Ts = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for j, c in enumerate(clips):
            d = torch.load(os.path.join(a.cache, a.split, "clips", c + ".pt"), map_location="cpu")
            x = d["pool4x10"].float()[None]                                 # [1, T, 40, 1024]
            adapt = model._encode({"feats": x})                             # [1, T, 40, d]
            z = model.wm.z1(adapt)[0]                                       # [T, D]
            Zs.append(z.to(torch.float16).numpy())
            Ci.append(np.full(z.shape[0], j, np.int32))
            Ts.append(np.arange(z.shape[0], dtype=np.int16))
            del d, x, adapt, z
            if (j + 1) % 200 == 0:
                print(f"[states {time.strftime('%H:%M:%S', time.gmtime())}] clips {j + 1}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
    Z = np.concatenate(Zs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, Z=Z, clip_idx=np.concatenate(Ci), t=np.concatenate(Ts), clips=np.array(clips))
    meta = {"ckpt": a.ckpt, "ckpt_sha256": sha_file(a.ckpt), "step": ck.get("step"), "wm_state": getattr(cfg, "wm_state", "mean"), "wm": cfg.wm, "D": int(Z.shape[1]),
            "n_frames": int(Z.shape[0]), "n_clips": len(clips), "split": a.split, "lo": a.lo, "hi": a.hi, "n_manifest": len(man), "out_sha256": sha_file(a.out),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
