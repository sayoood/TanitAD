"""p5_gate: P5's gate (THEORY_CRAWL_AMEND_3.md section 3): reproduce the model's own a0 from the logs on >= 1,000 TRAIN windows, max |a0_reproduced - a0_model| <= 1e-4 m/s^2, else NOT EVALUABLE.

a0_model      = what the training/eval data path hands the model on a window: ``V8FMiniWindows.item(...)["kin3"][1]`` and ``["p10f"][-1][1]`` from ``ClipStore(kin_mode='yaw')`` (the M2b C0 trainer's
                argv: ``--kin-mode yaw --tracks native_tracks.pt``; summary.json levers.kin_mode 'yaw'). The real class, the real cache, the real join; no checkpoint, no GPU.
a0_reproduced = ``L.a0_backward(track speed column, 2t, lag)``, a fresh float64 re-implementation, for each candidate definition of the card's prose ("the 0.2 s backward difference"):
                lag 1 (ONE native frame, 0.1 s: what the source computes), lag 2 (0.2 s: the card's wording), the cache's steer-mode kin3 (0.2 s on the 5 Hz grid), and a forward difference (a mutant).
Also: the logged sampling rate (displacement / speed on the native track). Output: raw/P5_GATE.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

N_GATE, SEED = 1500, 1
TOL = 1e-4


def main():
    t0 = time.time()
    L.assert_card_frozen()
    D = L.setup_tree()
    import torch
    torch.set_num_threads(2)
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    clips, ci, t = tab["clips"], tab["clip_idx"], tab["t"]
    res = {"card_sha256": L.CARD_SHA, "tol": TOL, "n_gate_requested": N_GATE, "gate_seed": SEED}
    # ---- A. whole table (87,942 in-band windows): the model's a0 (yaw_kin_from_track) against the fresh float64 re-implementation
    A = {}
    for nm, col in (("lag1_0.1s_one_native_frame", "a0_indep"), ("lag2_0.2s_card_prose", "a0_lag2"), ("forward_diff_mutant", "a0_fwd")):
        d = np.abs(tab["a0_model"] - tab[col])
        A[nm] = {"n": int(np.isfinite(d).sum()), "max_abs_diff": float(np.nanmax(d)), "median_abs_diff": float(np.nanmedian(d)), "pass": bool(np.nanmax(d) <= TOL)}
    res["A_yaw_kin_from_track_vs_reimplementation_all_inband_windows"] = A
    # ---- B. the real data path on 1,500 seeded windows
    tracks = D.load_native_tracks(L.TRACKS)
    J = D.load_join(L.TRAIN_JOIN)
    _load = torch.load
    torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})      # ClipStore.get reads a 20 MB clip file: mmap it
    store = D.ClipStore(L.TRAIN_CACHE, "train", lru=6, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, J, [], hist=4, hmax=16, max_age_t=2, max_age_s=6, window_start=3)
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(t), size=N_GATE, replace=False))
    sel = sel[np.argsort(ci[sel], kind="stable")]
    a0_batch_kin3, a0_batch_p10f, a0_cache_kin3, a0_ind = [], [], [], {1: [], 2: []}
    for n, i in enumerate(sel):
        c = str(clips[ci[i]])
        it = ds.item(ci_of[c], int(t[i]), 0, 0)
        a0_batch_kin3.append(float(it["kin3"][1]))
        a0_batch_p10f.append(float(it["p10f"][-1][1]))
        o = _load(f"{L.TRAIN_CACHE}/train/clips/{c}.pt", map_location="cpu", mmap=True, weights_only=True)
        a0_cache_kin3.append(float(o["kin3"][int(t[i]), 1]))                                   # the cache's own kin3 (the 'steer' mode: (v[2t] - v[2t-2]) / 0.2)
        del o
        v = tracks[c][:, 3]
        for lag in (1, 2):
            a0_ind[lag].append(float(L.a0_backward(v, np.array([2 * int(t[i])]), lag)[0]))
        if n % 300 == 0:
            print(n, f"{time.time()-t0:.0f}s", flush=True)
    torch.load = _load
    ak, ap, ac = np.array(a0_batch_kin3), np.array(a0_batch_p10f), np.array(a0_cache_kin3)
    a1, a2 = np.array(a0_ind[1]), np.array(a0_ind[2])
    fwd = np.array([float(L.a0_forward(tracks[str(clips[ci[i]])][:, 3], np.array([2 * int(t[i])]))[0]) for i in sel])
    B = {"n_windows": int(len(sel)), "n_clips": int(len(np.unique(ci[sel]))),
         "batch_kin3_vs_batch_p10f_a_feature_max_abs": float(np.abs(ak - ap).max()),
         "model_a0_range": [float(ak.min()), float(ak.max())],
         "candidates": {
             "lag1_0.1s_one_native_frame": {"max_abs_diff": float(np.abs(ak - a1).max()), "median_abs_diff": float(np.median(np.abs(ak - a1))), "pass": bool(np.abs(ak - a1).max() <= TOL)},
             "lag2_0.2s_card_prose": {"max_abs_diff": float(np.abs(ak - a2).max()), "median_abs_diff": float(np.median(np.abs(ak - a2))), "pass": bool(np.abs(ak - a2).max() <= TOL)},
             "cache_kin3_steer_mode_0.2s": {"max_abs_diff": float(np.abs(ak - ac).max()), "median_abs_diff": float(np.median(np.abs(ak - ac))), "pass": bool(np.abs(ak - ac).max() <= TOL)},
             "forward_diff_mutant": {"max_abs_diff": float(np.abs(ak - fwd).max()), "median_abs_diff": float(np.median(np.abs(ak - fwd))), "pass": bool(np.abs(ak - fwd).max() <= TOL)}}}
    res["B_real_data_path_ClipStore_yaw_V8FMiniWindows_item"] = B
    # ---- C. the logged sampling rate: |displacement| / (speed * 0.1 s) on the native track (1.00 = 10 Hz)
    rat = []
    for c in list(tracks)[:600]:
        tr = tracks[c]
        x, y, v = tr[:, 0], tr[:, 1], tr[:, 3]
        d = np.hypot(np.diff(x), np.diff(y)) / 0.1
        vm = 0.5 * (v[1:] + v[:-1])
        m = np.isfinite(d) & np.isfinite(vm) & (vm > 2)
        if m.any():
            rat.append(float(np.median(d[m] / vm[m])))
    res["C_sampling_rate"] = {"native_track_hz": 10.0, "cache_grid_hz": 5.0, "median_over_600_clips_of_median_disp_over_v_dt0.1": float(np.median(rat)), "p5_p95": [float(np.percentile(rat, 5)), float(np.percentile(rat, 95))],
                              "n_clips": len(rat), "note": "1.00 = a 0.1 s frame interval. The cache keeps every second native frame (0.2 s)."}
    gate_ok = bool(B["n_windows"] >= 1000 and B["candidates"]["lag1_0.1s_one_native_frame"]["pass"] and A["lag1_0.1s_one_native_frame"]["pass"])
    res["GATE"] = {"passes_with_the_model_s_own_definition": gate_ok, "model_definition": "lag1_0.1s_one_native_frame",
                   "passes_with_the_card_prose_0.2s_definition": bool(B["candidates"]["lag2_0.2s_card_prose"]["pass"]),
                   "reading": (f"max |Delta| vs the model's a0: one-frame 0.1 s {B['candidates']['lag1_0.1s_one_native_frame']['max_abs_diff']:.3g}, 0.2 s {B['candidates']['lag2_0.2s_card_prose']['max_abs_diff']:.3g}, "
                               f"cache steer-mode kin3 {B['candidates']['cache_kin3_steer_mode_0.2s']['max_abs_diff']:.3g}, forward mutant {B['candidates']['forward_diff_mutant']['max_abs_diff']:.3g} (m/s^2; tolerance {TOL})")}
    res["wall_s"] = time.time() - t0
    json.dump(res, open(L.HERE + "/raw/P5_GATE.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
