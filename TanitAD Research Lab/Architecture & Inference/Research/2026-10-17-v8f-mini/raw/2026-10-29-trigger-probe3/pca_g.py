#!/usr/bin/env python3
"""pca_g.py -- the per-token PCA readout of PREREG 2, one population.
  pass A: 16 sampled token positions per window (G8: of 160, G4: of 40; rng 1000 + population index) accumulate exact float64 token second moments per OUTER fold (g_lib.TokenStats) -> 6 PCAs per grid
          (outer fold f = all outer-train windows = total minus block f; fold 5 = all rows); exact covariance eigh, identical in value to g_lib.fit_token_pca on the same tokens;
  pass B: project ALL tokens of ALL windows with all 6 PCAs -> F:/TanitAD-scratch/trigger_probe3_scratch/S8_<pop>.npy [6, 12000, 2560], S4_<pop>.npy [6, 12000, 640] float32 (token-major).
Writes raw/pcaG_<pop>.npz (PCA parameters, spectra, sampled positions) and raw/pcaG_<pop>.json.   CUDA_VISIBLE_DEVICES=-1  python -I pca_g.py --pop pstop|pmove [--mutate-upsample]
--mutate-upsample: the C5(b) content mutation -- the G8 source is replaced by the nearest-neighbour upsample of G4 (same shape); outputs go to mut_* names.
RSS stays below ~1 GB (the box had 2.2 GB available when this was written)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import g_lib as G  # noqa: E402
import numpy as np

CH = 250


def read_g8(mm, a, mutate):
    x = mm.rows(a, a + CH).astype(np.float32).reshape(-1, 160, 1024)
    if mutate:
        x = G.upsample_g4_to_g8(G.g4_from_g8(x))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=G.POPS)
    ap.add_argument("--mutate-upsample", action="store_true")
    a = ap.parse_args()
    pop, mut = a.pop, a.mutate_upsample
    sfx = "mut_" if mut else ""
    d = G.load_pop(pop)
    mm = G.NpyRows(f"{G.SCR}/G8_{pop}.npy")
    N, D = mm.shape
    G.assert_g8_entry(mm.shape)                              # C5(a): the 4x10 file raises here
    assert N == len(d["y"])
    G.require_ram(1500, f"pca_g {pop}")                      # this step streams (RSS < 1 GB); the 3 GB floor applies to loads that hold arrays, see RESULTS "post-freeze choices"
    outer = d["outer"]
    pos8, pos4 = G.sample_positions(N, 1000 + G.POP_INDEX[pop])
    t0 = time.time()
    p8, p4 = G.fit_readout(lambda s: read_g8(mm, s, mut), N, CH, outer, pos8, pos4)
    print(f"pass A + fits {time.time() - t0:.0f}s freeMB {G.free_mb()}", flush=True)
    info = []
    for f in range(6):
        info.append({"fold": f, "windows": int((outer != f).sum()) if f < 5 else int(N), "n_tokens_G8": p8[f]["n"], "n_tokens_G4": p4[f]["n"], "expl_var_ratio_G8": float(p8[f]["ev"].sum() / p8[f]["total_var"]),
                     "expl_var_ratio_G4": float(p4[f]["ev"].sum() / p4[f]["total_var"])})
        print(f"fit fold {f} windows {info[-1]['windows']} n_tok {p8[f]['n']}/{p4[f]['n']} EVR G8 {info[-1]['expl_var_ratio_G8']:.4f} G4 {info[-1]['expl_var_ratio_G4']:.4f}", flush=True)
    S8 = np.zeros((6, N, 160 * G.K_TOK), np.float32)
    S4 = np.zeros((6, N, 40 * G.K_TOK), np.float32)
    G.transform_all(lambda s: read_g8(mm, s, mut), N, CH, p8, p4, S8, S4)
    print(f"pass B {time.time() - t0:.0f}s freeMB {G.free_mb()}", flush=True)
    np.save(f"{G.SCR}/{sfx}S8_{pop}.npy", S8)
    np.save(f"{G.SCR}/{sfx}S4_{pop}.npy", S4)
    fp = {"S8_fold5_sha256_first1000": hashlib.sha256(np.ascontiguousarray(S8[5, :1000]).tobytes()).hexdigest(), "S4_fold5_sha256_first1000": hashlib.sha256(np.ascontiguousarray(S4[5, :1000]).tobytes()).hexdigest()}
    np.savez(f"{G.RAW}/pcaG_{sfx}{pop}.npz", mu8=np.stack([q["mu"] for q in p8]), comp8=np.stack([q["comp"] for q in p8]), ev8=np.stack([q["ev"] for q in p8]), tv8=np.array([q["total_var"] for q in p8]),
             mu4=np.stack([q["mu"] for q in p4]), comp4=np.stack([q["comp"] for q in p4]), ev4=np.stack([q["ev"] for q in p4]), tv4=np.array([q["total_var"] for q in p4]), pos8=pos8, pos4=pos4)
    json.dump({"pop": pop, "mutated": mut, "folds": info, "fingerprints": fp, "seconds": time.time() - t0, "shapes": {"S8": list(S8.shape), "S4": list(S4.shape)}}, open(f"{G.RAW}/pcaG_{sfx}{pop}.json", "w"), indent=1)
    print("done", json.dumps(fp), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
