#!/usr/bin/env python3
"""p1pb_forward.py -- SPEC_P1PB.md section 2: the tick-0 plan of C0 s0 and s1, CPU, zero training, no weight edit, through the RI tool's own functions.
  python -I p1pb_forward.py control   -> P1PB_FW_CONTROL.json : the SAME driver on 240 eval-141 windows must reproduce the T1-kin dumps' tick-0 rows (control FW)
  python -I p1pb_forward.py holdout   -> P1PB_TICK0_s0.npz, P1PB_TICK0_s1.npz, P1PB_F1_CONTROL.json : the 60 RA-1 holdout clips, every window t in [3, n - 30)
CUDA_VISIBLE_DEVICES=-1.  Env NT = torch threads (default 8)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import torch

NT = int(os.environ.get("NT", "8"))
torch.set_num_threads(NT)
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RI_TOOL = f"{ART}/dagland/motion_c/ri/code/v8f_t1_eval_ri.py"
DATA = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CKPT = {0: f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt", 1: f"{ART}/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt"}
DUMPS = {0: "C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz", 1: "C:/Users/Admin/rr_scratch/out/C0C_s1/ticks/ticks_C0C_s1.npz"}
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache"
V9_TRAIN = f"{ART}/v9labels/v9_labels_train.npz"
V9_MD5 = "f63ece410b725febb8a5242cf2b01d3c"
sys.path[:0] = [f"{TREE}/stack", f"{TREE}/stack/scripts", f"{TREE}/taniteval"]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__

spec = importlib.util.spec_from_file_location("ri_tool", RI_TOOL)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
D, M = T.D, T.M
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_models():
    out, cfgs = {}, {}
    for s in (0, 1):
        ck = torch.load(CKPT[s], map_location="cpu", weights_only=False)
        cd = dict(ck["cfg"])
        assert cd.pop("launch_intent_w", 0.0) == 0.0          # a loss-side lever; the inference path is identical (p2b_check.py)
        cfg = M.V8FMiniConfig(**cd)
        assert cfg.kin_mode == "yaw"
        model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        model.load_state_dict(ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out[s], cfgs[s] = model, cfg
        assert ck.get("grid", "pool4x10") == "pool4x10"
    return out, cfgs


def run_windows(models, cfg, store, tracks, J, wins, batch_size=32):
    """-> {seed: dict of arrays} for the tick-0 rows of ``wins`` = [(ci, t0, 0, 0)].  The plan is the tool's ISSUED plan (--smooth sg72 --feasible kbox); plan_raw is the raw network plan."""
    proxy = T._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    dev = torch.device("cpu")
    keys = ("plan", "plan_raw", "plan_s")
    acc = {s: {k: [] for k in keys} for s in (0, 1)}
    common = {k: [] for k in ("v0", "gt", "gt_ok", "ci", "t0", "kin_v0", "nav0", "clip")}
    for b0 in range(0, len(wins), batch_size):
        chunk = wins[b0:b0 + batch_size]
        sts = T.make_states(store, tracks, J, chunk, "deadreckon")
        items = [T.build_item(ds, proxy, st, 0, 0, 0, "logged", "causal", None) for st in sts]
        full = D.collate(items)
        batch = {k: full[k].to(dev) for k in T.MODEL_KEYS}
        v0 = batch["p10f"][:, -1, 0].double().numpy()
        for b, st in enumerate(sts):
            clip = str(J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            g, ok = T.gt_in_frame(store.get(clip)["pose"], pose_now, st.j)
            common["gt"].append(g)
            common["gt_ok"].append(ok)
            common["ci"].append(st.ci)
            common["t0"].append(st.t0)
            common["clip"].append(clip)
            common["kin_v0"].append(float(store.get(clip)["kin3"][st.t0, 0]))
            common["nav0"].append(np.asarray(st.nav0, np.float32))
        common["v0"].append(v0)
        for s in (0, 1):
            mp = T.model_planner(models[s], dev, keep_aux=False, smooth="sg72", feasible="kbox")
            pl = mp(batch)
            acc[s]["plan"].append(np.asarray(pl, np.float64))
            acc[s]["plan_raw"].append(np.asarray(mp.plan_raw, np.float64))
            acc[s]["plan_s"].append(np.asarray(mp.plan_s, np.float64))
        if (b0 // batch_size) % 10 == 0:
            log(f"{b0 + len(chunk)}/{len(wins)}")
    cm = {"v0": np.concatenate(common["v0"]), "gt": np.asarray(common["gt"], np.float64), "gt_ok": np.asarray(common["gt_ok"], bool), "ci": np.asarray(common["ci"], np.int64),
          "t0": np.asarray(common["t0"], np.int64), "kin_v0": np.asarray(common["kin_v0"], np.float64), "nav0": np.asarray(common["nav0"], np.float32), "clip": np.asarray(common["clip"])}
    return {s: {**cm, **{k: np.concatenate(acc[s][k]) for k in keys}} for s in (0, 1)}


def sec(x):
    return np.hypot(x[:, 11, 0] - x[:, 7, 0], x[:, 11, 1] - x[:, 7, 1]) / 0.8


def control():
    """FW: the driver on 240 evenly spaced eval-141 windows reproduces the dump's tick-0 rows."""
    models, cfgs = load_models()
    cfg = cfgs[0]
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(f"{DATA}/join_eval141.npz")
    store = D.ClipStore(f"{ART}/sfresh_inputs/cache", "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins_all = D.eval_grid_windows(f"{DATA}/eval_grid_windows.json", J)
    idx = np.linspace(0, len(wins_all) - 1, 240).round().astype(int)
    wins = [wins_all[i] for i in idx]
    log("control windows", len(wins), "of", len(wins_all))
    R = run_windows(models, cfg, store, tracks, J, wins)
    res = {"n_windows": len(wins), "window_indices": idx.tolist(), "per_seed": {}}
    ok_all = True
    for s in (0, 1):
        Z = np.load(DUMPS[s], allow_pickle=True)
        m0 = np.flatnonzero(Z["tick"] == 0)
        assert len(m0) == len(wins_all)
        rows = m0[idx]
        r = R[s]
        same_win = bool(np.array_equal(Z["ci"][rows], r["ci"]) and np.array_equal(Z["t0"][rows], r["t0"]))
        a_dump = (sec(Z["plan"][rows]) - Z["v0"][rows]) / 2
        a_mine = (sec(r["plan"]) - r["v0"]) / 2
        d = {"same_windows_ci_t0": same_win, "v0_max_abs": float(np.abs(Z["v0"][rows] - r["v0"]).max()), "gt_max_abs": float(np.nanmax(np.abs(Z["gt"][rows] - r["gt"]))),
             "plan_max_abs_m": float(np.abs(Z["plan"][rows] - r["plan"]).max()), "plan_raw_max_abs_m": float(np.abs(Z["plan_raw"][rows] - r["plan_raw"]).max()),
             "plan_s_max_abs_m": float(np.abs(Z["plan_s"][rows] - r["plan_s"]).max()), "a_plan_max_abs": float(np.nanmax(np.abs(a_dump - a_mine))),
             "plan_mean_abs_m": float(np.abs(Z["plan"][rows] - r["plan"]).mean()), "kin_v0_vs_v0_max_abs": float(np.abs(r["kin_v0"] - r["v0"]).max())}
        d["pass"] = bool(same_win and d["v0_max_abs"] <= 1e-6 and d["gt_max_abs"] <= 1e-6 and d["plan_max_abs_m"] <= 5e-3 and d["plan_raw_max_abs_m"] <= 5e-3 and d["a_plan_max_abs"] <= 2e-3)
        ok_all &= d["pass"]
        res["per_seed"][f"s{s}"] = d
        log(f"FW s{s}", json.dumps(d))
    res["pass"] = bool(ok_all)
    res["spec_tolerances"] = {"v0": 1e-6, "gt": 1e-6, "plan_and_plan_raw_m": 5e-3, "a_plan_mps2": 2e-3}
    json.dump(res, open(f"{HERE}/P1PB_FW_CONTROL.json", "w"), indent=1)
    log("FW control pass:", res["pass"])
    return 0 if res["pass"] else 3


def holdout():
    fw = json.load(open(f"{HERE}/P1PB_FW_CONTROL.json"))
    if not fw["pass"]:
        raise SystemExit("REFUSED: control FW failed; the holdout arm is not read")
    ho = json.load(open(f"{HERE}/P1PB_HOLDOUT_60.json"))
    man = {}
    for ln in open(f"{TRAIN_CACHE}/train/manifest.jsonl", encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            man[r["clip"]] = int(r["n_frames"])
    cn = [(c, man[c]) for c in ho]
    models, cfgs = load_models()
    cfg = cfgs[0]
    first = D.effective_window_start(cfg.hist, cfg.window_start)
    assert first == 3 and D.HORIZON == 30
    from tanitad.data.v9_labels import load_v9_release
    rel = load_v9_release(V9_TRAIN, expect_md5=V9_MD5)
    J = D.build_label_join(cn, rel, lead_npz=None, speed=None)
    tracks = D.load_native_tracks(TRACKS)
    store = D.ClipStore(TRAIN_CACHE, "train", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    labelled = np.array([bool(J["has"][int(o):int(o) + int(n)].any()) for o, n in zip(J["off"], J["n"])])
    wins = [(ci, t, 0, 0) for ci, (c, n) in enumerate(cn) for t in range(first, n - D.HORIZON)]
    log("holdout clips", len(cn), "labelled", int(labelled.sum()), "windows", len(wins), "of which labelled-clip windows", int(sum(labelled[w[0]] for w in wins)))
    # F1: the per-clip feature path (the store raises on a track/grid mismatch; the rest is counted here)
    f1 = {"n_clips": len(cn), "clips_without_v9_rows": [c for (c, _), l in zip(cn, labelled) if not l], "n_windows": len(wins), "n_windows_in_labelled_clips": int(sum(labelled[w[0]] for w in wins)),
          "v9_release_md5": rel.md5, "v9_release_md5_expected": V9_MD5, "first_window_t": first, "horizon": D.HORIZON, "per_clip": {}}
    gt_ok_all, n_ok = True, 0
    for ci, (c, n) in enumerate(cn):
        d = store.get(c)                                          # raises if the native track does not match the cache pose grid
        okn = bool(len(d["pose"]) == n and d["feats"].shape[0] == n and d["gt_ok"][first:n - D.HORIZON].all())
        gt_ok_all &= okn
        n_ok += int(okn)
        f1["per_clip"][c] = {"n_frames": n, "labelled": bool(labelled[ci]), "n_windows": n - D.HORIZON - first, "feats_shape_ok": okn}
    f1["all_clips_n_frames_feats_gt_ok"] = bool(gt_ok_all)
    f1["store_track_grid_tie_checked_by_ClipStore_for_all_clips"] = True
    R = run_windows(models, cfg, store, tracks, J, wins)
    for s in (0, 1):
        r = R[s]
        np.savez(f"{HERE}/P1PB_TICK0_s{s}.npz", **r, labelled=labelled[r["ci"]], ckpt=CKPT[s], clip_names=np.asarray(ho))
    r0 = R[0]
    f1["v0_vs_kin3_t0_max_abs"] = float(np.abs(r0["v0"] - r0["kin_v0"]).max())
    f1["gt_ok_all_windows"] = bool(r0["gt_ok"].all())
    f1["finite_plan_s0_s1"] = bool(np.isfinite(R[0]["plan"]).all() and np.isfinite(R[1]["plan"]).all())
    f1["tick0_rows_identical_v0_gt_between_seeds"] = bool(np.array_equal(R[0]["v0"], R[1]["v0"]) and np.array_equal(R[0]["gt"], R[1]["gt"]))
    f1["nav0_all_zero_windows"] = int((np.abs(r0["nav0"]).sum(1) == 0).sum())
    f1["pass"] = bool(f1["all_clips_n_frames_feats_gt_ok"] and f1["v0_vs_kin3_t0_max_abs"] <= 1e-4 and f1["gt_ok_all_windows"] and f1["finite_plan_s0_s1"] and f1["tick0_rows_identical_v0_gt_between_seeds"]
                      and rel.md5 == V9_MD5)
    json.dump(f1, open(f"{HERE}/P1PB_F1_CONTROL.json", "w"), indent=1)
    log("F1", {k: v for k, v in f1.items() if k not in ("per_clip", "clips_without_v9_rows")})
    return 0 if f1["pass"] else 4


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "control":
        sys.exit(control())
    if mode == "holdout":
        sys.exit(holdout())
    raise SystemExit("usage: p1pb_forward.py control | holdout")
