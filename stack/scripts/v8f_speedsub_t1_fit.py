#!/usr/bin/env python3
"""v8f_speedsub_t1_fit.py -- the causal ego read-out `mlp_ego_c` of PREREG_V8F_SPEEDSUB_T1 section 2 (fitted once, used as fitted), and its five controls.

THE READ-OUT. The 22-column ego block of SPEEDREAD A1's `mlp_ego` in the same order (p10f 0:8 | nav 8:13 | limit 13:17 | limit_ok 17 | frame_ok 18:22). Only `limit` and `limit_ok` change: for each
TRAIN window (clip, t) they are recomputed with the pinned T1 tool's OWN `causal_bucket` (imported, never re-implemented) over the clip's MEASURED ego speed rows [j - CAUSAL_STEPS, j], j = the window's
NOW row `t` -- the row the tool passes (`build_item`: `causal_bucket(st.d["v"], st.j, CAUSAL_STEPS)`), whatever the tick-frame age `a_t` is. Every other column is read from train_arrays.npz unchanged.
The fit is SPEEDREAD A1's own `standardiser` / `fold_of` / `mlp_fit` with `mlp_ego`'s `mlp_cfg`, folds, holdout fold and seed, all read from fit.json. One fit, no selection, no tuning.
Its TRAIN holdout RMSE is reported beside and decides nothing.

CONTROLS (any failure -> REFUSED, rc 2, nothing is saved as the read-out):
  C-INSTR    the fit functions of the instrument in use equal SPEEDREAD A1's (source text), and the constants (MLP_CFG, LAMBDAS, N_FOLDS, HOLDOUT_FOLD) equal fit.json's
  C-FIT      the SAME code path on the LOGGED columns reproduces the banked `mlp_ego`: holdout predictions within 1e-5 m/s of readout.pt (FINDING_1 measured 9.41e-6)
  C-CAUSAL   on the eval-141 windows at tick 0 (and at tick-frame ages 1, 2 on every 4th window) the bucket this file computes equals, window for window, the `limit` / `limit_ok` that
             `v8f_t1_eval.build_item` builds under --limit causal; the speed rows are read through THIS file's loader, not through the tool's ClipStore
  C-COVER    every TRAIN window has a speed history or is set limit_ok = 0 exactly as the tool does with no finite past; the count is reported
  C-DISJ     eval clips and fit clips are disjoint
  C-PARITY   TRAIN parity 4,512 clips / 306,566 windows / the excluded-clip sha / the window-list sha, as in fit.json and extract.json

USAGE   python v8f_speedsub_t1_fit.py fit --out-dir F:/TanitAD-scratch/speedsub_t1/fit            (defaults are the registered inputs)
CPU only (CUDA_VISIBLE_DEVICES=-1), at most 4 threads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "4"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

SCHEMA = "v8f-speedsub-t1-fit-v1"
READOUT_KEY = "mlp_ego_c"
TOOL_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"       # the pinned T1 tool (PREREG section 3)
INSTR_A1_SHA256 = "fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe"     # SPEEDREAD A1's instrument (the fit.json fit)
BANKED_READOUT_SHA256 = "a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d"
TOL_C_FIT = 1e-5                                                                           # m/s, PREREG section 2 (C-FIT)
COL_LIMIT, COL_LIMIT_OK = slice(13, 17), 17
D_EGO = 22
N_TRAIN_CLIPS, N_TRAIN_WINDOWS = 4512, 306566
FIT_FUNCS = ("standardiser", "fold_of", "mlp_build", "mlp_fit", "mlp_predict", "masked_rmse", "save_readouts", "load_readouts")
#: the container class is compared METHOD BY METHOD: the repo tip's ReadOut = A1's plus exactly these additive methods (PREREG_V8F_SPEEDSUB: the ego-only path)
READOUT_EXTRA_METHODS = {"predict_ego_batch"}
FIT_CONSTS = ("MLP_CFG", "LAMBDAS", "N_FOLDS", "HOLDOUT_FOLD")

TRAIN = "F:/TanitAD-scratch/speedread/train/train_arrays.npz"
TRAIN_EXTRACT = "F:/TanitAD-scratch/speedread/train/extract.json"
FITJSON = "F:/TanitAD-scratch/speedread/fit/fit.json"
READOUT = "F:/TanitAD-scratch/speedread/fit/readout.pt"
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache"
INSTR_A1 = "F:/TanitAD-scratch/speedread/tree_f3971165/stack/scripts/v8f_mini_eval_speedread.py"
ART = "E:/Projects/TanitAD-artifacts"
TOOL_ROOT = f"{ART}/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ the pure part (tested)
def load_v(cache_dir, split, clip):
    """The clip's measured ego speed row array `v` [T] float32, read from the cache file (the field `ClipStore.get` reads as `d["v"]`), by mmap so the scene tensors are never touched."""
    import torch
    o = torch.load(os.path.join(cache_dir, split, "clips", f"{clip}.pt"), map_location="cpu", weights_only=True, mmap=True)
    v = np.array(o["v"].numpy(), copy=True)          # a COPY: a view would keep the whole 20 MB file mapped, and 4,512 live mappings exhaust the address space (access violation, MEASURED)
    del o
    return v


def causal_block(v_of, clips, ts, causal_bucket, steps):
    """-> (limit [N, 4] float32, ok [N] bool): the T1 tool's `causal_bucket(v, j, steps)` at j = ts[i] on clip clips[i]'s measured speed. `v_of(clip)` returns the clip's `v` array.
    j is the window's NOW row, which is what `build_item` passes (`st.j`), independent of the held tick-frame age."""
    n = len(ts)
    lim = np.zeros((n, 4), np.float32)
    ok = np.zeros(n, bool)
    cur, v = None, None
    for i in range(n):
        c = str(clips[i])
        if c != cur:
            cur, v = c, v_of(c)
        lim[i], ok[i] = causal_bucket(v, int(ts[i]), steps)
    return lim, ok


def swap_limit(ego, lim, ok):
    """The causal ego block: `ego` (the logged 22 columns) with ONLY columns 13:17 (limit) and 17 (limit_ok) replaced. Every other column is returned bit-identical."""
    if ego.shape[1] != D_EGO or lim.shape != (len(ego), 4) or ok.shape != (len(ego),):
        raise ValueError(f"ego {ego.shape}, limit {lim.shape}, ok {ok.shape}: not a 22-column block with a 4-wide limit")
    out = np.array(ego, copy=True)
    out[:, COL_LIMIT] = lim
    out[:, COL_LIMIT_OK] = ok.astype(out.dtype)
    return out


def confusion(logged_lim, logged_ok, lim, ok):
    """4 x 4 [logged bucket, causal bucket] counts over windows where both are set, and the counts where either is unset."""
    both = logged_ok & ok
    cm = np.zeros((4, 4), np.int64)
    np.add.at(cm, (logged_lim[both].argmax(1), lim[both].argmax(1)), 1)
    return cm, {"logged_ok_only": int((logged_ok & ~ok).sum()), "causal_ok_only": int((~logged_ok & ok).sum()), "neither": int((~logged_ok & ~ok).sum())}


def fit_ego(sr, ego, Y, M, clips, cfg, holdout_fold):
    """The `mlp_ego` code path of `fit_all`: standardise the 22 ego columns on ALL train rows, fit the MLP on the folds != holdout, early-stop on the holdout fold."""
    E = np.asarray(ego, np.float32)
    em, es = sr.standardiser(E)
    Ez = (E - em) / es
    folds = sr.fold_of(clips)
    tr, ho = np.flatnonzero(folds != holdout_fold), np.flatnonzero(folds == holdout_fold)
    m = sr.mlp_fit(Ez, Y, M, tr, ho, cfg)
    return {"m": m, "em": em, "es": es, "Ez": Ez, "tr": tr, "ho": ho}


def holdout_metrics(sr, fit, Y, M, clips):
    ho = fit["ho"]
    P = sr.mlp_predict(fit["m"], fit["Ez"][ho])
    sd = np.array([np.std(Y[M[:, c], c]) for c in range(Y.shape[1])])
    const = np.array([float(np.mean(Y[M[:, c], c])) for c in range(Y.shape[1])])
    rm = sr.masked_rmse(P, Y[ho], M[ho])
    cr = sr.masked_rmse(np.broadcast_to(const, Y[ho].shape), Y[ho], M[ho])
    return P, {"best_epoch": int(fit["m"]["best_epoch"]), "holdout_rmse_ms": [float(v) for v in rm], "holdout_rmse_sd": [float(v) for v in rm / sd],
               "holdout_skill_vs_trainmean": [float(v) for v in 1.0 - rm ** 2 / cr ** 2], "holdout_rmse_sd_mean": float(np.mean(rm / sd)), "n_holdout_rows": int(len(ho)),
               "n_holdout_clips": int(len(set(map(str, np.asarray(clips)[ho]))))}


def instr_problems(sr, sr_a1, fitrec):
    """C-INSTR: -> list of problems."""
    bad = []
    for nm in FIT_FUNCS:
        a, b = getattr(sr, nm, None), getattr(sr_a1, nm, None)
        if a is None or b is None:
            bad.append(f"{nm}: missing")
        elif inspect.getsource(a) != inspect.getsource(b):
            bad.append(f"{nm}: source differs from SPEEDREAD A1's")
    a_methods = {k for k, v in vars(sr_a1.ReadOut).items() if callable(v)}
    t_methods = {k for k, v in vars(sr.ReadOut).items() if callable(v)}
    for k in sorted(a_methods):
        if k not in t_methods or inspect.getsource(vars(sr.ReadOut)[k]) != inspect.getsource(vars(sr_a1.ReadOut)[k]):
            bad.append(f"ReadOut.{k}: missing or differs from SPEEDREAD A1's")
    if t_methods - a_methods != READOUT_EXTRA_METHODS:
        bad.append(f"ReadOut has methods A1 lacks beyond {sorted(READOUT_EXTRA_METHODS)}: {sorted(t_methods - a_methods - READOUT_EXTRA_METHODS)}")
    for nm in FIT_CONSTS:
        if getattr(sr, nm) != getattr(sr_a1, nm):
            bad.append(f"{nm}: {getattr(sr, nm)!r} != A1 {getattr(sr_a1, nm)!r}")
    if dict(sr.MLP_CFG) != fitrec["mlp_cfg"] or list(sr.LAMBDAS) != fitrec["lambdas"] or sr.N_FOLDS != fitrec["folds"] or sr.HOLDOUT_FOLD != fitrec["holdout_fold"]:
        bad.append("instrument constants != fit.json's mlp_cfg / lambdas / folds / holdout_fold")
    return bad


# ------------------------------------------------------------------------------------------------ controls that need the data
def c_causal(T, D, eval_cache, tracks_path, grid_json, join_npz, v_loader, steps, age_stride=4):
    """C-CAUSAL: this file's bucket == `build_item`'s under --limit causal, window for window, at tick 0 (a_t = 0) and at a_t in (1, 2) on every `age_stride`-th window."""
    tracks = D.load_native_tracks(tracks_path)
    J = D.load_join(join_npz)
    store = D.ClipStore(eval_cache, "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(grid_json, J)
    proxy = T._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=4, hmax=10, max_age_t=2, max_age_s=6, window_start=None)
    clips = [str(J["clip"][ci]) for ci, _, _, _ in wins]
    ts = [t for _, t, _, _ in wins]
    lim, ok = causal_block(lambda c: v_loader(eval_cache, "eval", c), clips, ts, T.causal_bucket, steps)
    bad = {"a_t=0": 0, "a_t=1": 0, "a_t=2": 0}
    n = {"a_t=0": 0, "a_t=1": 0, "a_t=2": 0}
    for i, (ci, t0, _, _) in enumerate(wins):
        clip = str(J["clip"][ci])
        o, nn = int(J["off"][ci]), int(J["n"][ci])
        for a in (0, 1, 2):
            if a and i % age_stride:
                continue
            st = T.WindowState(store.get(clip), tracks[clip], J["nav"][o:o + nn], ci, t0, "deadreckon")
            it = T.build_item(ds, proxy, st, 0, a, a, "logged", "causal")
            key = f"a_t={a}"
            n[key] += 1
            if not (np.array_equal(np.asarray(it["limit"], np.float32), lim[i]) and bool(it["limit_ok"]) == bool(ok[i])):
                bad[key] += 1
    return {"pass": sum(bad.values()) == 0 and n["a_t=0"] == len(wins), "n_windows": len(wins), "n_compared": n, "n_mismatch": bad,
            "n_limit_ok": int(ok.sum()), "bucket_counts": [int(v) for v in lim.sum(0)]}


def main_fit(a):
    t_start = time.time()

    def log(msg):
        print(f"[fit {time.time() - t_start:6.1f}s] {msg}", flush=True)
    import torch
    torch.set_num_threads(4)
    out = {"schema": SCHEMA, "readout_key": READOUT_KEY, "threads": 4, "torch": torch.__version__, "numpy": np.__version__, "controls": {}}
    ctl = out["controls"]
    # ---- pins
    tool_py = f"{a.tool_root}/stack/scripts/v8f_t1_eval.py"
    if sha256_file(tool_py) != TOOL_SHA256:
        raise SystemExit(f"REFUSED: the T1 tool {tool_py} is not the pinned one ({TOOL_SHA256[:12]})")
    sr = load_module(a.instr, "sr_instr")
    out["instrument"] = {"path": a.instr, "sha256": sha256_file(a.instr)}
    fitrec = json.load(open(a.fit_json, encoding="utf-8"))
    ext = json.load(open(a.train_extract, encoding="utf-8"))
    if sha256_file(a.readout) != fitrec["readout_sha256"] or fitrec["readout_sha256"] != BANKED_READOUT_SHA256:
        raise SystemExit("REFUSED: readout.pt is not the banked one named by fit.json")
    if os.path.isfile(a.instr_a1) and sha256_file(a.instr_a1) == INSTR_A1_SHA256:
        ci = instr_problems(sr, load_module(a.instr_a1, "sr_a1"), fitrec)
    else:
        ci = [f"SPEEDREAD A1's instrument {a.instr_a1} is missing or has another sha256"]
    ctl["C-INSTR"] = {"pass": not ci, "problems": ci}
    log("pins + C-INSTR done")
    # ---- arrays
    z = np.load(a.train)
    ego, Y, M, clips, win = z["ego"], z["y"], z["m"], z["clip"], z["win"]
    cpar = {"n_windows": int(len(ego)), "n_clips": int(len(set(map(str, clips)))), "ego_shape": list(ego.shape), "fit_json": [fitrec["n_windows"], fitrec["n_clips"]],
            "extract_parity_ok": bool(ext["parity_ok"]), "window_list_sha256": sha256_text(json.dumps(win.tolist())), "extract_window_list_sha256": ext["window_list_sha256"],
            "excluded_sha256": ext["parity"]["excluded_sha256"], "train_arrays_sha256": sha256_file(a.train)}
    ctl["C-PARITY"] = {"pass": bool(cpar["n_windows"] == N_TRAIN_WINDOWS == fitrec["n_windows"] and cpar["n_clips"] == N_TRAIN_CLIPS == fitrec["n_clips"] and ego.shape[1] == D_EGO
                                   and cpar["extract_parity_ok"] and cpar["window_list_sha256"] == cpar["extract_window_list_sha256"] and fitrec["train_parity_ok"]), **cpar}
    log("C-PARITY done")
    # ---- C-DISJ
    grid = json.load(open(a.eval_grid, encoding="utf-8"))
    ev_clips = set(grid["order"])
    inter = sorted(ev_clips & set(map(str, clips)))
    ctl["C-DISJ"] = {"pass": bool(not inter and len(ev_clips) == 141 and fitrec["eval_train_disjoint"]), "n_eval_clips": len(ev_clips), "n_fit_clips": cpar["n_clips"], "intersection": inter}
    log("C-DISJ done")
    # ---- the tool and the causal limit of every TRAIN window
    T = load_module(tool_py, "v8f_t1_eval_pinned")
    D = T.D
    steps = T.CAUSAL_STEPS
    out["causal_steps"] = steps
    t1 = time.time()
    uniq = sorted(set(map(str, clips)))
    vcache = {}

    def v_of(c):
        if c not in vcache:
            vcache[c] = load_v(a.train_cache, "train", c)
        return vcache[c]
    lim, ok = causal_block(v_of, clips, win[:, 1], T.causal_bucket, steps)
    out["seconds_causal_block"] = round(time.time() - t1, 1)
    log("causal block over the train windows done")
    # second location of the speed source: the native tracks (raw frames, column 3), even rows = the cache grid
    tracks = D.load_native_tracks(a.tracks)
    cmp_ = {"clips_compared": 0, "clips_missing_in_tracks": 0, "len_mismatch": 0, "max_abs_diff_ms": 0.0, "rows_compared": 0}
    for c in uniq:
        if c not in tracks:
            cmp_["clips_missing_in_tracks"] += 1
            continue
        tv, cv = np.asarray(tracks[c], np.float64)[0::2, 3], np.asarray(v_of(c), np.float64)
        if len(tv) != len(cv):
            cmp_["len_mismatch"] += 1
            continue
        m = np.isfinite(tv) & np.isfinite(cv)
        cmp_["clips_compared"] += 1
        cmp_["rows_compared"] += int(m.sum())
        if m.any():
            cmp_["max_abs_diff_ms"] = max(cmp_["max_abs_diff_ms"], float(np.abs(tv[m] - cv[m]).max()))
    out["speed_source"] = {"primary": f"{a.train_cache}/train/clips/<clip>.pt['v'] (the field ClipStore.get reads as d['v'], mmap)", "second_location": f"{a.tracks} tracks[clip][0::2, 3]",
                           "agreement": cmp_}
    logged_lim, logged_ok = ego[:, COL_LIMIT], ego[:, COL_LIMIT_OK] > 0.5
    cm, odd = confusion(logged_lim, logged_ok, lim, ok)
    ego_c = swap_limit(ego, lim, ok)
    chk = {"other_columns_bit_identical": bool(np.array_equal(np.delete(ego_c, list(range(13, 18)), 1), np.delete(ego, list(range(13, 18)), 1))),
           "limit_rowsum_in_0_1": bool(np.isin(ego_c[:, COL_LIMIT].sum(1), [0.0, 1.0]).all()), "limit_ok_iff_onehot": bool(((ego_c[:, COL_LIMIT].sum(1) == 1.0) == (ego_c[:, COL_LIMIT_OK] == 1.0)).all())}
    ctl["C-COVER"] = {"pass": bool(len(ok) == N_TRAIN_WINDOWS and set(uniq) <= set(vcache) and all(chk.values())), "n_windows": int(len(ok)), "n_limit_ok_causal": int(ok.sum()),
                      "n_limit_ok_zero_no_finite_past": int((~ok).sum()), "n_clips_with_v": len(vcache), "n_logged_limit_ok": int(logged_ok.sum()),
                      "confusion_logged_x_causal": cm.tolist(), "unset_counts": odd, "agreement_with_logged": float(np.trace(cm) / max(cm.sum(), 1)), "layout": chk}
    log("C-COVER done")
    # ---- C-CAUSAL
    t1 = time.time()
    ctl["C-CAUSAL"] = c_causal(T, D, a.eval_cache, a.tracks, a.eval_grid, a.eval_join, load_v, steps)
    ctl["C-CAUSAL"]["seconds"] = round(time.time() - t1, 1)
    log("C-CAUSAL done")
    # ---- C-FIT: the same path on the LOGGED columns reproduces the banked mlp_ego
    sd = np.array([np.std(Y[M[:, c], c]) for c in range(Y.shape[1])])
    const = np.array([float(np.mean(Y[M[:, c], c])) for c in range(Y.shape[1])])
    cfg = dict(fitrec["mlp_cfg"])
    t1 = time.time()
    f0 = fit_ego(sr, ego, Y, M, clips, cfg, fitrec["holdout_fold"])
    P0, met0 = holdout_metrics(sr, f0, Y, M, clips)
    banked = sr.load_readouts(a.readout)["mlp_ego"]
    Pb = sr.mlp_predict(banked.mlp, f0["Ez"][f0["ho"]])
    d_pred = float(np.abs(P0 - Pb).max())
    d_xmxs = float(max(np.abs(f0["em"] - banked.xm).max(), np.abs(f0["es"] - banked.xs).max()))
    ctl["C-FIT"] = {"pass": bool(d_pred <= TOL_C_FIT and f0["m"]["best_epoch"] == fitrec["mlp_ego"]["best_epoch"] and d_xmxs == 0.0),
                    "holdout_pred_max_abs_diff_ms": d_pred, "tolerance_ms": TOL_C_FIT, "best_epoch": met0["best_epoch"], "best_epoch_banked": fitrec["mlp_ego"]["best_epoch"],
                    "holdout_rmse_sd_max_abs_diff_vs_fit_json": float(np.abs(np.array(met0["holdout_rmse_sd"]) - np.array(fitrec["mlp_ego"]["holdout_rmse_sd"])).max()),
                    "xm_xs_max_abs_diff_vs_banked": d_xmxs, "seconds": round(time.time() - t1, 1)}
    passed = all(v["pass"] for v in ctl.values())
    out["controls_pass"] = passed
    log("C-FIT done")
    # ---- the fit itself (the same code path on the causal columns)
    t1 = time.time()
    fc = fit_ego(sr, ego_c, Y, M, clips, cfg, fitrec["holdout_fold"])
    Pc, metc = holdout_metrics(sr, fc, Y, M, clips)
    out["fit_seconds"] = round(time.time() - t1, 1)
    out["beside"] = {"note": "TRAIN fold-0 holdout, decides nothing (PREREG section 2). 'logged' = the control refit of mlp_ego (= banked, C-FIT); 'causal' = mlp_ego_c.", "logged": met0, "causal": metc,
                     "train_sd": [float(v) for v in sd], "const": [float(v) for v in const], "fit_json_mlp_ego_holdout_rmse_sd": fitrec["mlp_ego"]["holdout_rmse_sd"]}
    out["inputs"] = {"train_arrays": a.train, "fit_json": a.fit_json, "fit_json_sha256": sha256_file(a.fit_json), "readout_banked": a.readout, "readout_banked_sha256": BANKED_READOUT_SHA256,
                     "tool": tool_py, "tool_sha256": TOOL_SHA256, "tracks": a.tracks, "tracks_sha256": sha256_file(a.tracks), "train_cache": a.train_cache, "eval_cache": a.eval_cache,
                     "eval_grid_sha256": sha256_file(a.eval_grid), "eval_join_sha256": sha256_file(a.eval_join), "mlp_cfg": cfg, "holdout_fold": fitrec["holdout_fold"],
                     "instrument_a1": a.instr_a1}
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez(os.path.join(a.out_dir, "limit_causal.npz"), limit=lim, ok=ok, clip=clips, t=win[:, 1])
    out["limit_causal_npz_sha256"] = sha256_file(os.path.join(a.out_dir, "limit_causal.npz"))
    if not passed:
        out["outcome"] = "REFUSED"
        json.dump(out, open(os.path.join(a.out_dir, "fit_c.REFUSED.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
        print("ZZSPEEDSUB-T1-FIT-STOP REFUSED", {k: v["pass"] for k, v in ctl.items()}, flush=True)
        return 2
    ro = sr.ReadOut("mlp_ego", mlp=fc["m"], ego_only=True, pca_mu=None, pca_V=None, xm=fc["em"], xs=fc["es"])
    path = os.path.join(a.out_dir, "readout_c.pt")
    sr.save_readouts({READOUT_KEY: ro}, path)
    rt = sr.load_readouts(path)[READOUT_KEY]
    out["container"] = {"path": path, "sha256": sha256_file(path), "key": READOUT_KEY, "kind": rt.kind, "ego_only": bool(rt.ego_only), "xm_shape": list(rt.xm.shape),
                        "cols_field": "none: the block stays 22 wide, the same columns as mlp_ego (only the limit / limit_ok values differ)",
                        "roundtrip_pred_max_abs_diff": float(np.abs(sr.mlp_predict(rt.mlp, fc["Ez"][fc["ho"]]) - Pc).max())}
    out["outcome"] = "FITTED"
    out["seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(os.path.join(a.out_dir, "fit_c.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    json.dump({"readout_c_sha256": out["container"]["sha256"], "readout_key": READOUT_KEY, "fit_c_json_sha256": sha256_file(os.path.join(a.out_dir, "fit_c.json"))},
              open(os.path.join(a.out_dir, "READOUT_C_PIN.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZSPEEDSUB-T1-FIT-DONE", out["container"]["sha256"], flush=True)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--out-dir", default="F:/TanitAD-scratch/speedsub_t1/fit")
    p.add_argument("--train", default=TRAIN)
    p.add_argument("--train-extract", default=TRAIN_EXTRACT)
    p.add_argument("--fit-json", default=FITJSON)
    p.add_argument("--readout", default=READOUT)
    p.add_argument("--train-cache", default=TRAIN_CACHE)
    p.add_argument("--tool-root", default=TOOL_ROOT)
    p.add_argument("--instr", default=f"{HERE}/v8f_mini_eval_speedread.py")
    p.add_argument("--instr-a1", default=INSTR_A1)
    p.add_argument("--tracks", default=f"{ART}/sfresh_inputs/inputs/native_tracks.pt")
    p.add_argument("--eval-cache", default=f"{ART}/sfresh_inputs/cache")
    p.add_argument("--eval-grid", default=f"{TOOL_ROOT}/{PK}/eval_grid_windows.json")
    p.add_argument("--eval-join", default=f"{TOOL_ROOT}/{PK}/join_eval141.npz")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return main_fit(a)


if __name__ == "__main__":
    sys.exit(main())
