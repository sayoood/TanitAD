#!/usr/bin/env python3
"""extract_eval2.py -- eval-141 features for ARMS 2 and 3 (PREREG_FROZEN.md 9): on the 608 eval windows with v0 < 6 (the previous package's eval141_features.npz window set)
  Znn_eval141 / Znl_eval141 : h_O with a TRAIN donor nav (default_rng(1), donor from the canonical TRAIN windows; eval and train clips are disjoint) / TRAIN donor nav + limit
  V4frames_eval141          : the model's own 4 frames t-3..t of the eval ClipStore window (feats[:, -4:]), fp16 [608, 4, 40960]
Controls: the true-nav forward reproduces the banked eval Z; frame t equals the banked V_eval141.   CUDA_VISIBLE_DEVICES=-1  python -I extract_eval2.py"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np

import common2 as C  # noqa: E402
import extract_znn as XZ  # noqa: E402  (this package: swap, draw_donors, nav_hist; it imports the previous package's extract_z unmodified)

torch = XZ.torch


def main():
    t0 = time.time()
    C.require_ram(3000, "extract_eval2")
    M, model, cfg = XZ.EZ.load_model()
    from tanitad.data import v8f_mini_data as D
    A = C.load_windows()
    clips_tr = [str(c) for c in A["clips"]]
    Jt = D.load_join(f"{C.OLDSCR}/join_train_rebuilt.npz")
    jix_t = {c: i for i, c in enumerate(Jt["clip"].tolist())}
    Je = D.load_join(C.EVAL_JOIN)
    jix_e = {c: i for i, c in enumerate(Je["clip"].tolist())}
    ev = np.load(f"{C.OLDRAW}/eval141_features.npz", allow_pickle=False)
    names, t0s = [str(c) for c in ev["clip"]], ev["t0"]
    assert not (set(names) & set(clips_tr)), "eval and train clips overlap"
    n = len(names)
    clip_all = A["clip_i"]
    rng = np.random.default_rng(1)
    don = rng.integers(0, len(clip_all), n)                                        # eval clips are disjoint from train clips, so any TRAIN window is a donor from another episode
    tracks = D.load_native_tracks(C.TRACKS)
    _load = torch.load
    torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})
    store = D.ClipStore(C.EVAL_CACHE, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, Je, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    cap = {}
    model.read_o.register_forward_hook(lambda m, i, o: cap.__setitem__("hO", o.flatten(1).detach().float()))
    Znn, Znl = np.zeros((n, cfg.n_q * cfg.d), np.float32), np.zeros((n, cfg.n_q * cfg.d), np.float32)
    F4 = np.zeros((n, 4, 40 * 1024), np.float16)
    same_nav, same_lim = np.zeros(n, bool), np.zeros(n, bool)
    order = np.argsort(np.array([jix_e[c] for c in names]), kind="stable")
    max_true, max_frame = 0.0, 0.0
    V0 = np.load(f"{C.OLDSCR}/V_eval141.npy", mmap_mode="r")
    for a in range(0, n, 32):
        blk = order[a:a + 32]
        its = [ds.item(jix_e[names[b]], int(t0s[b]), 0, 0) for b in blk]
        batch = D.collate(its)
        feed = {k: batch[k] for k in M.INFERENCE_INPUT_KEYS}
        f1, f2, sn, sl = XZ.swap(feed, Jt, jix_t, clips_tr, A, cfg, don[blk])
        with torch.no_grad():
            model.forward_infer(feed)
            h0 = cap["hO"].clone()
            model.forward_infer(f1)
            h1 = cap["hO"].clone()
            model.forward_infer(f2)
            h2 = cap["hO"].clone()
        max_true = max(max_true, float((h0 - torch.as_tensor(ev["Z"][blk])).abs().max()))
        fr = batch["feats"][:, -4:].reshape(len(blk), 4, -1).numpy().astype(np.float16)
        max_frame = max(max_frame, float(np.abs(fr[:, 3].astype(np.float32) - np.asarray(V0[blk], np.float32)).max()))
        for k_, b in enumerate(blk):
            Znn[b], Znl[b], F4[b] = h1[k_].numpy(), h2[k_].numpy(), fr[k_]
            same_nav[b], same_lim[b] = bool(sn[k_]), bool(sl[k_])
    np.save(f"{C.SCR}/Znn_eval141.npy", Znn)
    np.save(f"{C.SCR}/Znl_eval141.npy", Znl)
    np.save(f"{C.SCR}/V4frames_eval141.npy", F4)
    rep = {"n": n, "seconds": time.time() - t0, "N1_eval_true_forward_vs_banked_Z_max_abs": max_true, "V1_eval_frame_t_vs_banked_V_max_abs": max_frame,
           "share_donor_nav_equals_own": float(same_nav.mean()), "share_donor_limit_equals_own": float(same_lim.mean()), "donor_seed": 1,
           "min_t0": int(t0s.min()), "frames_differ_t_vs_t3_share": float((np.abs(F4[:, 3].astype(np.float32) - F4[:, 0].astype(np.float32)).max(1) > 0).mean())}
    json.dump(rep, open(f"{C.RAW}/control_eval2.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
