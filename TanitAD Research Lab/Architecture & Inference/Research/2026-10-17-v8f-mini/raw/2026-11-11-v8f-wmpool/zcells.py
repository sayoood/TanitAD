"""ZCELLS (PREREG_V8F_WMPOOL): per-frame cache of the frozen adapter's 2 x 5 CELLS of v8F-mini (M2b s0 / s1), BEFORE WMAux.proj.

cell[t, R*5 + C] = mean over the 2 x 2 block of tokens (2R+r', 2C+c') of (model.adapter(feats[t].float()) + model.pos)   -- the 4 x 10 grid, token i = r*10 + c.
Mean over the 10 cells == theory/ladder/zcache.py's zmean (equal-sized blocks). Stored float16 [T, 10, 256] (ESTIMATED 2.36 GB per seed).
Built with the SAME code path as zcache.py (clip_inputs, load_model, adapter_sha), one extra pooling after ``adapter + pos``. Inference only.
Modes:
  build    : write <out>/M2b_s{k}/{split}/<clip>.npz (zcell f16 [T,10,D], ok bool [T], t_s f32 [T]); per-clip tmp + rename; resumable.
  manifest : <out>/M2b_s{k}/{split}/CELL_MANIFEST.json (the trainer's --wm-pool att25 reads it: kind, grid, n_cells, d, dtype, adapter_state_sha256, per-file sha256).
  check3   : PREREG_V8F_WMPOOL control 3 on the clips of --clips-file (the form of ZCACHE_CHECK12): mean over the cached cells vs (a) the cached zmean of ``--zmean-dir`` and (b) the recomputed fp32 zmean;
             the fp16 storage error is the measured gap; a wrong-seed adapter must FAIL the identity. Writes --out-json.
Run on box 2 (the raw feature cache lives there). ``ZC_REPO`` / ``ZCACHE_DIR`` as for zcache.py. Pure helpers (``cells_of_tokens``) are importable with torch only.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
import numpy as np
import torch

ZCACHE_DIR = os.environ.get("ZCACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ladder"))
GRID_NAME = "pool4x10->2x5"
N_CELLS, D_CELL = 10, 256


def cells_of_tokens(adapt: torch.Tensor) -> torch.Tensor:
    """adapt [F, 40, d] -> [F, 10, d] (cell R*5 + C = mean of tokens 20R + 10r' + 2C + c'); the same reshape as ``WMAux.cells_of_adapt`` (control 3 compares the two)."""
    F, N, d = adapt.shape
    assert N == 40, N
    return adapt.reshape(F, 2, 2, 5, 2, d).mean(dim=(2, 4)).reshape(F, 10, d)


@torch.no_grad()
def zcell_of(model, feats16: torch.Tensor, ch: int = 16) -> torch.Tensor:
    """feats16 [T, 40, d_feat] (float16) -> [T, 10, d] float32, frames through ``model.adapter`` in chunks, exactly as ``zcache.zmean_of`` and then the 2 x 2 pooling."""
    out = []
    for s in range(0, feats16.shape[0], ch):
        f = feats16[s:s + ch].float()[None]
        adapt = model.adapter(f) + model.pos[None, None]                       # V8FMini._encode
        out.append(cells_of_tokens(adapt[0]))
    return torch.cat(out).float()


def _zc():
    sys.path.insert(0, ZCACHE_DIR)
    import zcache as Z                                                          # noqa: E402
    return Z


def cmd_build(a):
    Z = _zc()
    from tanitad.data import v8f_mini_data as D
    tracks = D.load_native_tracks(a.tracks)
    models = {}
    for kv in a.ckpt:
        k, p = kv.split("=", 1)
        models[int(k)] = Z.load_model(p)[0]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    for split in a.splits:
        clips = Z.list_clips(a.root, split)
        if a.clips_file:
            sel = json.load(open(a.clips_file))
            keep = set(sel[split]) if split in sel else set()
            clips = [c for c in clips if c in keep]
        if a.limit:
            clips = clips[:a.limit]
        for k in models:
            os.makedirs(f"{a.out}/M2b_s{k}/{split}", exist_ok=True)
        t0 = time.time(); n_new = 0
        for i, clip in enumerate(clips):
            todo = [k for k in models if not os.path.exists(f"{a.out}/M2b_s{k}/{split}/{clip}.npz")]
            if not todo:
                continue
            feats, ok, t_s = Z.clip_inputs(a.root, split, clip, tracks)
            for k in todo:
                z = zcell_of(models[k], feats).numpy().astype(np.float16)
                assert np.isfinite(z.astype(np.float32)).all(), (clip, k)
                dst = f"{a.out}/M2b_s{k}/{split}/{clip}.npz"
                tmp = dst + ".tmp"
                with open(tmp, "wb") as fh:
                    np.savez(fh, zcell=z, ok=ok, t_s=t_s)
                os.replace(tmp, dst)
            n_new += 1
            if n_new % 25 == 0 or i == len(clips) - 1:
                print(f"{time.strftime('%H:%M:%S')} {split} {i + 1}/{len(clips)} new={n_new} {(time.time() - t0) / n_new:.2f} s/clip", flush=True)
        print(f"ZZSPLIT-DONE {split} clips={len(clips)} new={n_new} sec={time.time() - t0:.1f}", flush=True)
    print("ZZBUILD-OK", flush=True)


def cmd_manifest(a):
    import torch as T
    Z = _zc()
    for kv in a.ckpt:
        k, p = kv.split("=", 1)
        ck = T.load(p, map_location="cpu", weights_only=False)
        sha, names = Z.adapter_sha(ck)
        for split in ("train", "eval"):
            d = f"{a.out}/M2b_s{k}/{split}"
            fs = sorted(glob.glob(f"{d}/*.npz"))
            if not fs:
                continue
            files, Ts, shapes = {}, [], set()
            for f in fs:
                z = np.load(f)
                assert z["zcell"].dtype == np.float16 and z["zcell"].shape[1:] == (N_CELLS, D_CELL), (f, z["zcell"].dtype, z["zcell"].shape)
                shapes.add(tuple(z["zcell"].shape[1:])); Ts.append(int(z["zcell"].shape[0]))
                files[os.path.basename(f)] = Z.sha_file(f)
            man = {"kind": "zcell", "grid": GRID_NAME, "n_cells": N_CELLS, "d": D_CELL, "dtype": "float16", "cell_index": "R*5 + C, mean of tokens 20R + 10r' + 2C + c' (token i = r*10 + c)",
                   "adapter_state_sha256": sha, "adapter_keys": names, "ckpt_file_sha256": Z.sha_file(p), "code_commit": a.commit, "n_files": len(files), "T_min": min(Ts), "T_max": max(Ts),
                   "T_mean": float(np.mean(Ts)), "source_cache": a.root, "files_sha256": files}
            json.dump(man, open(f"{d}/CELL_MANIFEST.json", "w"), indent=1)
            print("ZZMANIFEST-OK", d, len(files), flush=True)


def cmd_check3(a):
    Z = _zc()
    from tanitad.data import v8f_mini_data as D
    tracks = D.load_native_tracks(a.tracks)
    sel = json.load(open(a.clips_file))
    mods = {int(kv.split("=")[0]): Z.load_model(kv.split("=", 1)[1])[0] for kv in a.ckpt}
    res = {"tol_identity_fp32": 1e-5, "clips": {}}
    worst_id, worst_cache, worst_wrong = 0.0, 0.0, 1e9
    for split, clips in sel.items():
        for clip in clips:
            feats, ok, t_s = Z.clip_inputs(a.root, split, clip, tracks)
            for k, m in mods.items():
                cells = zcell_of(m, feats)                                       # fp32 cells
                zm = Z.zmean_of(m, feats)                                         # fp32 mean (the cache's expression)
                id_err = float((cells.mean(1) - zm).abs().max())                  # (a) mean of cells == token mean, fp32 round-off
                c16 = cells.to(torch.float16).float()
                st_err = float((c16 - cells).abs().max())                         # fp16 storage error of the cells
                mean16 = float((c16.mean(1) - zm).abs().max())                    # (b) mean of the STORED cells vs the token mean
                cached = None
                f = f"{a.zmean_dir}/M2b_s{k}/{split}/{clip}.npz" if a.zmean_dir else ""
                if f and os.path.exists(f):
                    zc = np.load(f)["zmean"]
                    cached = float(np.abs(c16.mean(1).numpy() - zc).max())          # (c) vs today's CACHED zmean on disk
                    worst_cache = max(worst_cache, cached)
                wrong = None
                if len(mods) > 1:
                    zw = Z.zmean_of(mods[1 - k], feats) if (1 - k) in mods else None
                    wrong = None if zw is None else float((cells.mean(1) - zw).abs().max())
                    if wrong is not None:
                        worst_wrong = min(worst_wrong, wrong)
                worst_id = max(worst_id, id_err)
                res["clips"][f"{split}/{clip}/s{k}"] = {"T": int(feats.shape[0]), "cells_mean_vs_zmean_fp32": id_err, "fp16_storage_max_abs": st_err, "stored_cells_mean_vs_zmean": mean16,
                                                       "stored_cells_mean_vs_cached_zmean": cached, "wrongseed_diff": wrong}
    res["max_identity_fp32"] = worst_id
    res["max_stored_vs_cached"] = worst_cache
    res["min_wrongseed_diff"] = worst_wrong if worst_wrong < 1e9 else None
    res["identity_pass"] = worst_id <= res["tol_identity_fp32"]
    res["wrongseed_fails_identity"] = (worst_wrong > res["tol_identity_fp32"]) if worst_wrong < 1e9 else None
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}), flush=True)
    print("ZZCHECK3-OK" if res["identity_pass"] and res["wrongseed_fails_identity"] in (True, None) else "ZZCHECK3-FAIL", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "manifest", "check3"])
    ap.add_argument("--root", default="D:/tanitad/cache_full")
    ap.add_argument("--tracks", default="D:/tanitad/cache_full/native_tracks.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="k=path")
    ap.add_argument("--out", default="D:/tanitad/zcells")
    ap.add_argument("--splits", nargs="+", default=["eval", "train"])
    ap.add_argument("--clips-file")
    ap.add_argument("--zmean-dir", default="D:/tanitad/zcache", help="check3: today's mean cache root (M2b_s{k}/{split}/<clip>.npz)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="ZCELLS_CHECK3.json")
    ap.add_argument("--commit", default="1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c")
    a = ap.parse_args()
    {"build": cmd_build, "manifest": cmd_manifest, "check3": cmd_check3}[a.mode](a)
