"""pc_states.py -- stage 0f of PREREG_V8F_PROOF_ADD12 (note N9): untrained principal-component states at full token width, bases from clips that are in no read.

  python pc_states.py bases  --cache <cache> --split train --lo 1800 --hi 2400 --out <bases.npz>
  python pc_states.py states --cache <cache> --split train --lo 600 --hi 1800 --bases <bases.npz> --out-prefix <dir>/PC_train      (writes PC_train_G.npz, PC_train_M.npz + .json records)
  python pc_states.py states --cache <cache> --split eval  --lo 0   --hi 141  --bases <bases.npz> --out-prefix <dir>/PC_eval

CPU only, one thread, ONE clip in memory at a time, no GPU, no label. Clips = positions lo .. hi-1 of the ascending sha12 order of the split manifest.
bases:  per cell c of the 4x10 grid: mean mu_c [1024] and the top 16 principal directions B_c [1024, 16] of the centred cell tokens; for the mean token (mean over the 40 cells):
        mean mu [1024] and the top 128 principal directions Bm [1024, 128]. Covariances are accumulated in float64 (sum of x and of x^T x), eigendecomposed once.
states: grid state [T, 640] = concat over cells of (x_c - mu_c) @ B_c (cell-major: cell 0 first); mean state [T, 128] = (mean_c x_c - mu) @ Bm; float16.
        Layout of extract_states.py: Z, clip_idx, t, clips; record <file>.json with wm_state "grid" / "mean", D, counts, sha256 of the bases file and of the output.
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
NCELL, DTOK, KCELL, KMEAN = 40, 1024, 16, 128


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clip_list(cache, split, lo, hi):
    sys.path.insert(0, "/workspace/v8proof/tree/stack")
    from tanitad.data import v8f_mini_data as D
    man = D.read_manifest(cache, split)
    order = sorted([c for c, _ in man], key=D.sha12)
    return order[lo:hi], len(man)


def load_tokens(cache, split, c):
    d = torch.load(os.path.join(cache, split, "clips", c + ".pt"), map_location="cpu")
    x = d["pool4x10"].float()                                           # [T, 40, 1024]
    if x.shape[1] != NCELL or x.shape[2] != DTOK:
        raise SystemExit(f"REFUSED: {c}: token shape {tuple(x.shape)}")
    return x.double()


def top_dirs(S1, S2, n, k):
    mu = S1 / n
    C = S2 / n - np.outer(mu, mu)
    w, V = np.linalg.eigh(C)                                            # ascending
    idx = np.argsort(w)[::-1][:k]
    return mu.astype(np.float32), V[:, idx].astype(np.float32), w[idx].astype(np.float64), float(w.sum())


def cmd_bases(a):
    clips, n_man = clip_list(a.cache, a.split, a.lo, a.hi)
    S1 = np.zeros((NCELL, DTOK)); S2 = np.zeros((NCELL, DTOK, DTOK)); M1 = np.zeros(DTOK); M2 = np.zeros((DTOK, DTOK)); n = 0
    t0 = time.time()
    for j, c in enumerate(clips):
        x = load_tokens(a.cache, a.split, c)
        n += x.shape[0]
        S1 += x.sum(0).numpy()
        S2 += torch.einsum("tci,tcj->cij", x, x).numpy()
        m = x.mean(1)
        M1 += m.sum(0).numpy(); M2 += (m.T @ m).numpy()
        del x, m
        if (j + 1) % 100 == 0:
            print(f"[bases {time.strftime('%H:%M:%S', time.gmtime())}] clips {j + 1}/{len(clips)} frames {n} ({time.time() - t0:.0f}s)", flush=True)
    mu_c = np.zeros((NCELL, DTOK), np.float32); B_c = np.zeros((NCELL, DTOK, KCELL), np.float32); ev_c = np.zeros((NCELL, KCELL)); tot_c = np.zeros(NCELL)
    for c in range(NCELL):
        mu_c[c], B_c[c], ev_c[c], tot_c[c] = top_dirs(S1[c], S2[c], n, KCELL)
    mu_m, B_m, ev_m, tot_m = top_dirs(M1, M2, n, KMEAN)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, mu_c=mu_c, B_c=B_c, ev_c=ev_c, tot_c=tot_c, mu_m=mu_m, B_m=B_m, ev_m=ev_m, tot_m=np.array(tot_m), clips=np.array(clips), n_frames=np.array(n))
    meta = {"split": a.split, "lo": a.lo, "hi": a.hi, "n_manifest": n_man, "n_clips": len(clips), "n_frames": int(n), "cell_var_share_top16_mean_over_cells": float((ev_c.sum(1) / tot_c).mean()),
            "mean_token_var_share_top128": float(ev_m.sum() / tot_m), "out_sha256": sha_file(a.out), "seconds": round(time.time() - t0, 1)}
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))
    return 0


def cmd_states(a):
    clips, n_man = clip_list(a.cache, a.split, a.lo, a.hi)
    Bz = np.load(a.bases)
    if a.split == "train" and set(clips) & set(Bz["clips"].astype(str).tolist()):
        raise SystemExit("REFUSED: a clip of this read is among the clips the bases were fitted on")
    mu_c = torch.from_numpy(Bz["mu_c"]).double(); B_c = torch.from_numpy(Bz["B_c"]).double(); mu_m = torch.from_numpy(Bz["mu_m"]).double(); B_m = torch.from_numpy(Bz["B_m"]).double()
    G, Mn, Ci, Ts = [], [], [], []
    t0 = time.time()
    for j, c in enumerate(clips):
        x = load_tokens(a.cache, a.split, c)
        g = torch.einsum("tci,cik->tck", x - mu_c[None], B_c).reshape(x.shape[0], NCELL * KCELL)     # cell-major
        m = (x.mean(1) - mu_m[None]) @ B_m
        G.append(g.to(torch.float16).numpy()); Mn.append(m.to(torch.float16).numpy())
        Ci.append(np.full(x.shape[0], j, np.int32)); Ts.append(np.arange(x.shape[0], dtype=np.int16))
        del x, g, m
        if (j + 1) % 200 == 0:
            print(f"[states {time.strftime('%H:%M:%S', time.gmtime())}] clips {j + 1}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
    ci, ts = np.concatenate(Ci), np.concatenate(Ts)
    bsha = sha_file(a.bases)
    for name, mode, Z in (("G", "grid", np.concatenate(G)), ("M", "mean", np.concatenate(Mn))):
        out = f"{a.out_prefix}_{name}.npz"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez(out, Z=Z, clip_idx=ci, t=ts, clips=np.array(clips))
        meta = {"ckpt": None, "ckpt_sha256": None, "step": None, "wm_state": mode, "wm": "untrained principal state (note N9)", "D": int(Z.shape[1]), "n_frames": int(Z.shape[0]), "n_clips": len(clips),
                "split": a.split, "lo": a.lo, "hi": a.hi, "n_manifest": n_man, "bases_sha256": bsha, "finite": bool(np.isfinite(Z.astype(np.float32)).all()), "abs_max": float(np.abs(Z.astype(np.float32)).max()),
                "out_sha256": sha_file(out), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
        with open(out + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        print(json.dumps(meta))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bases"); s = sub.add_parser("states")
    for p in (b, s):
        p.add_argument("--cache", required=True); p.add_argument("--split", required=True, choices=("train", "eval"))
        p.add_argument("--lo", type=int, required=True); p.add_argument("--hi", type=int, required=True)
    b.add_argument("--out", required=True)
    s.add_argument("--bases", required=True); s.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    return cmd_bases(a) if a.cmd == "bases" else cmd_states(a)


if __name__ == "__main__":
    sys.exit(main())
