#!/usr/bin/env python3
"""nt_smoke.py -- E1 smoke + controls on ~20 selection-sample tokens from shard 1 (INTERFACE.md section 6).  TANITAD VENV, CPU.

Writes ONLY under --out (default /home/nvidia/r8ns/nt/k1_sel, never the training dir) and SMOKE_REPORT.json.
Controls (each must read its literal, else exit 1):
  E1-K1a  t0 stitch == r8ns navtrain frame bank row (sha16 + array_equal); dataset `frames` == refcv8_bridge.pack_rows(bank, ST)
  E1-K1b  provider poses[0:8] (== pose_hist at t=0) vs ego_history_poses: max |d| == 0.0 (<= 1e-6)
  E1-K1c  mutation wrong_token (neighbour's frame under this token) -> K1a RED
  E1-PNG  every stored PNG decodes bit-exact (the writer's verify)
  E1-INT  analytic: straight constant-velocity track through spline_track -> exact (<= 1e-9 m); circle R=20 m, 8 m/s -> radius err stated
  E1-K2pre future poses at 0.5..4.0 s vs the devkit's human_future_poses: max |dxy| <= 1e-3 m
  E1-LOAD patched loader + the trainer's own dataset class -> one window (t=0 filter), shapes, pose_last == (0,0,0,v0)
  E1-UNPATCHED fb47a58 loader on the same files -> raises (fail loud)
  E1-REG  patched vs fb47a58 loader on real PhysicalAI *.v2ep.pt -> byte-identical
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nt_e1 as E  # noqa: E402

NT = "/home/nvidia/r8ns/nt"
SEL = "/home/nvidia/r8ns/navtrain"
SHARD1 = "/home/nvidia/r8ns/navtrain_dl/navtrain_current_1.tgz"
PAI_CACHE = "/home/nvidia/data/refcv8-b1-416x1024-train-4572"


def load_patched():
    p = os.path.join(HERE, "loader_patch", "v2_dataset_patched.py")
    spec = importlib.util.spec_from_file_location("v2_dataset_patched", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["v2_dataset_patched"] = m
    spec.loader.exec_module(m)
    return m


def pick_tokens(sample: dict, recs: dict, n: int) -> list:
    """The first ``n`` sample tokens in (log, timestamp) order whose 3 t0 jpgs are in the export (all are, KB3)."""
    toks = sorted(sample["tokens"], key=lambda t: (recs[t]["log_name"], recs[t]["timestamps_us"][-1]))
    return toks[:n]


def write_one(tok, rec, frames, pos, fm_by_ft, law_ok, out_dir, shard, mutate_wrong=None):
    """Production write path (nt_e1) for one token; ``fm_by_ft`` = {frame_token: frame meta incl. png + hwc}."""
    tr = E.token_poses(rec, frames, pos)
    fm = fm_by_ft[tok if mutate_wrong is None else mutate_wrong]
    nxt = frames[pos + 1]["token"] if pos + 1 < len(frames) else None
    if law_ok(nxt) and nxt in fm_by_ft:
        law = {"src": "next_token", "token": nxt, "sha16": fm_by_ft[nxt]["sha16"], "png": fm_by_ft[nxt]["png"]}
    else:
        law = {"src": "fallback_t0_not_in_navtrain_current", "token": None, "sha16": None, "png": None}
    meta = E.token_meta(tok, rec, pos, shard, fm, law, tr)
    path = os.path.join(out_dir, E.clip_id_of(tok) + ".v2ep.pt")
    nb = E.save_atomic(E.build_payload(tok, fm["png"], law["png"], tr, meta), path)
    ver = E.verify_file(path, fm["sha16"], law["sha16"], tr)
    return path, nb, ver, tr, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=f"{NT}/k1_sel")
    ap.add_argument("--report", default=f"{NT}/SMOKE_REPORT.json")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t_start = time.time()
    rep = {"tool": "nt_smoke.py", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "threads": a.threads, "controls": {}}
    sample = json.load(open(f"{SEL}/NAVTRAIN_SAMPLE_TOKENS.json"))
    recs = json.load(gzip.open(f"{SEL}/inputs/navtrain_sample_inputs.json.gz", "rt"))["tokens"]
    bank_idx = json.load(open(f"{SEL}/frame_bank/index.json"))
    bank = np.load(f"{SEL}/frame_bank/frames_s00.npy", mmap_mode="r")
    sample_set = set(sample["tokens"])
    toks = pick_tokens(sample, recs, a.n)
    # log positions + LAW frame tokens
    logs, pos_of = {}, {}
    for t in toks:
        ln = recs[t]["log_name"]
        if ln not in logs:
            fr, _ = E.N.load_log_pickle(f"{SEL}/navsim_logs/trainval/{ln}.pkl")
            logs[ln] = fr
        pos_of[t] = next(i for i, f in enumerate(logs[recs[t]["log_name"]]) if f["token"] == t)
    need_ft = set(toks)
    for t in toks:
        fr = logs[recs[t]["log_name"]]
        p = pos_of[t]
        if p + 1 < len(fr) and fr[p + 1]["token"] in sample_set:
            need_ft.add(fr[p + 1]["token"])
    # cams per needed frame token (from the pickle; the export only carries the sample tokens' history paths)
    cams_of = {}
    for t in toks:
        fr = logs[recs[t]["log_name"]]
        for q in (pos_of[t], pos_of[t] + 1):
            if q < len(fr) and fr[q]["token"] in need_ft:
                cams_of[fr[q]["token"]] = fr[q]["cams"]
    need_rel = {str(cams_of[ft][c]["data_path"]).replace("\\", "/"): ft for ft in need_ft for c in ("CAM_L0", "CAM_F0", "CAM_R0")}
    scratch = os.path.join(NT, "scratch_smoke")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    t0 = time.time()
    n_seen = n_got = 0
    for rel, read in E.BNF._tar_readers(SHARD1):
        n_seen += 1
        if rel in need_rel:
            dst = os.path.join(scratch, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(read())
            n_got += 1
    rep["extract"] = {"shard": SHARD1, "cam_jpgs_seen": n_seen, "needed": len(need_rel), "extracted": n_got,
                      "secs": round(time.time() - t0, 1)}
    # stitch every needed frame (timed), encode PNG (timed), level-9 size probe
    rig_cache, fm_by_ft, st_s, enc_s = {}, {}, [], []
    for ft in sorted(need_ft):
        if not all(os.path.exists(os.path.join(scratch, str(cams_of[ft][c]["data_path"]))) for c in ("CAM_L0", "CAM_F0", "CAM_R0")):
            continue
        s0 = time.perf_counter()
        hwc, src, rk, obs, mpx = E.stitch_frame(cams_of[ft], scratch, rig_cache)
        st_s.append(time.perf_counter() - s0)
        s0 = time.perf_counter()
        png = E.encode_png(hwc)
        enc_s.append(time.perf_counter() - s0)
        fm_by_ft[ft] = {"hwc": hwc, "png": png, "sha16": E.sha16(hwc), "rig_key": rk, "observed_frac": round(obs, 6),
                        "mean_px": round(mpx, 2)}
    n_pick = len(toks)
    toks = [t for t in toks if t in fm_by_ft]
    rep["tokens"] = {"picked": n_pick, "with_t0_frame_in_shard1": len(toks), "list": toks}
    import torchvision.io as tvio
    l9 = []
    for ft in sorted(fm_by_ft)[:5]:
        x = torch.from_numpy(fm_by_ft[ft]["hwc"]).permute(2, 0, 1).contiguous()
        s0 = time.perf_counter()
        b9 = tvio.encode_png(x, compression_level=9)
        l9.append({"png6": int(fm_by_ft[ft]["png"].numel()), "png9": int(b9.numel()), "secs9": round(time.perf_counter() - s0, 3)})
    rep["timing"] = {"n_frames_stitched": len(st_s), "stitch_s_per_frame_first": round(st_s[0], 3) if st_s else None,
                     "stitch_s_per_frame_mean_after_first": round(float(np.mean(st_s[1:])), 4) if len(st_s) > 1 else None,
                     "png_encode_s_per_frame_mean": round(float(np.mean(enc_s)), 4),
                     "png_bytes_mean": round(float(np.mean([int(v["png"].numel()) for v in fm_by_ft.values()])), 1),
                     "png_level9_probe": l9}
    # write the files through the production path
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    rows, k1a, k1b, k2 = [], [], [], []
    law_ok = lambda nxt: nxt in sample_set   # noqa: E731  (smoke: LAW exists iff the next frame is a sample token)
    for t in toks:
        rec = recs[t]
        fr = logs[rec["log_name"]]
        path, nb, ver, tr, meta = write_one(t, rec, fr, pos_of[t], fm_by_ft, law_ok, a.out, "navtrain_current_1")
        rows.append({"token": t, "bytes": nb, "verify": ver, "law_src": meta["law_src"], "T_raw": ver["T_raw"]})
        # K1a: the bank row of this token's t0 frame
        bi = bank_idx["tokens"][t]
        brow = np.asarray(bank[bi["rows"][-1]])
        k1a.append({"token": t, "mine": fm_by_ft[t]["sha16"], "bank": E.sha16(brow),
                    "equal": bool(np.array_equal(brow, fm_by_ft[t]["hwc"]))})
        # K1b: provider rows 0..7 = raw 2..9 vs the bridge history
        k1b.append(float(np.abs(tr["poses"][2:10] - tr["hist"]).max()))
        # K2pre: future knots vs devkit human_future_poses (8 x 0.5 s)
        hf = np.asarray(rec["human_future_poses"], np.float64)
        mine = tr["poses"][[RAW for RAW in (E.RAW_NOW + 5 * k for k in range(1, 9))]].astype(np.float64)
        dxy = float(np.abs(mine[:, :2] - hf[:, :2]).max())
        dyaw = float(np.abs(np.arctan2(np.sin(mine[:, 2] - hf[:, 2]), np.cos(mine[:, 2] - hf[:, 2]))).max())
        k2.append({"token": t, "max_dxy_m": dxy, "max_dyaw_rad": dyaw})
    n_law = sum(1 for r in rows if r["law_src"] == "next_token")
    rep["written"] = {"n": len(rows), "n_law_next_token": n_law, "n_fallback": len(rows) - n_law,
                      "bytes_mean": round(float(np.mean([r["bytes"] for r in rows])), 1),
                      "T_raw": sorted({r["T_raw"] for r in rows}), "problems": [r for r in rows if r["verify"]["problems"]]}
    C = rep["controls"]
    C["E1-PNG"] = {"n_files": len(rows), "n_pngs": sum(r["verify"]["pngs"] for r in rows),
                   "pass": all(not r["verify"]["problems"] for r in rows)}
    C["E1-K1a_frame_vs_bank"] = {"n": len(k1a), "n_equal": sum(x["equal"] for x in k1a),
                                 "n_sha_equal": sum(x["mine"] == x["bank"] for x in k1a),
                                 "pass": all(x["equal"] and x["mine"] == x["bank"] for x in k1a), "first": k1a[:3]}
    C["E1-K1b_pose_hist"] = {"n": len(k1b), "max_abs": max(k1b), "pass": max(k1b) <= 1e-6}
    C["E1-K2pre_future_vs_devkit"] = {"n": len(k2), "max_dxy_m": max(x["max_dxy_m"] for x in k2),
                                      "max_dyaw_rad": max(x["max_dyaw_rad"] for x in k2),
                                      "pass": max(x["max_dxy_m"] for x in k2) <= 1e-3}
    # K1c mutation: the neighbour's t0 frame written under this token -> K1a must go RED
    mut_dir = a.out + "_mutation"
    shutil.rmtree(mut_dir, ignore_errors=True)
    os.makedirs(mut_dir)
    t = toks[0]
    rec = recs[t]
    nb_tok = next(x for x in toks[1:] if x in fm_by_ft and fm_by_ft[x]["sha16"] != fm_by_ft[t]["sha16"])
    path, _, _, _, meta = write_one(t, rec, logs[rec["log_name"]], pos_of[t], fm_by_ft, lambda n_: False, mut_dir,
                                    "navtrain_current_1", mutate_wrong=nb_tok)
    dmut = torch.load(path, map_location="cpu", weights_only=False)
    hwc_m = E.decode_png_hwc(dmut["jpeg_buf"][:int(dmut["jpeg_len"][0])])
    brow = np.asarray(bank[bank_idx["tokens"][t]["rows"][-1]])
    C["E1-K1c_mutation_wrong_token"] = {"token": t, "frame_written": nb_tok,
                                        "k1a_equal_under_mutation": bool(np.array_equal(hwc_m, brow)),
                                        "pass": not bool(np.array_equal(hwc_m, brow))}
    shutil.rmtree(mut_dir, ignore_errors=True)
    # E1-INT analytic
    kt = np.arange(-3, 13) * 0.5
    v = 13.7
    straight = np.stack([v * kt, np.zeros_like(kt), np.zeros_like(kt)], -1)
    taus = np.arange(-9, 61) / 10.0
    p, _ = E.spline_track(kt, straight, np.full_like(kt, v), np.zeros_like(kt), taus)
    err_s = float(np.abs(p[:, 0] - v * taus).max() + np.abs(p[:, 1]).max() + np.abs(p[:, 2]).max())
    R, vc = 20.0, 8.0
    om = vc / R
    circ = np.stack([R * np.sin(om * kt), R * (1 - np.cos(om * kt)), om * kt], -1)
    pc, _ = E.spline_track(kt, circ, np.full_like(kt, vc), np.zeros_like(kt), taus)
    rad = np.hypot(pc[:, 0], pc[:, 1] - R)
    C["E1-INT_analytic"] = {"straight_cv_max_abs_err_m": err_s, "straight_pass": err_s <= 1e-9,
                            "circle_R20_v8_max_radius_err_m": float(np.abs(rad - R).max()),
                            "circle_max_pos_err_m": float(np.hypot(pc[:, 0] - R * np.sin(om * taus),
                                                                   pc[:, 1] - R * (1 - np.cos(om * taus))).max()),
                            "pass": err_s <= 1e-9}
    # E1-LOAD (patched loader + the trainer's own dataset class)
    P = load_patched()
    import tanitad.data.v2_dataset as U          # the fb47a58 loader (tree md5 870066f2...)
    eps = P.build_v2_providers(a.out, lru_size=4, rebuild=True, verbose=False)
    load = {"n_providers": len(eps)}
    try:
        sys.path.insert(0, f"{E.C}/tree/stack/scripts")
        import refc_v3_train as RV3
        ds = RV3.V3Dataset(eps, window=8, max_horizon=20, channels=9)
        load["dataset_class"] = "refc_v3_train.V3Dataset"
    except Exception as e:                     # noqa: BLE001
        import refb_train as RBT
        ds = RBT.FailLoudWindowDataset(eps, window=8, max_horizon=20, channels=9)
        load["dataset_class"] = f"refb_train.FailLoudWindowDataset (V3Dataset import failed: {type(e).__name__}: {str(e)[:160]})"
    ds.u8_frames = True
    load["windows_before_filter"] = len(ds.index)
    ds.index = [(e, t) for (e, t) in ds.index if t == 0]          # INTERFACE R2
    load["windows_after_filter"] = len(ds.index)
    it = ds[0]
    e0 = eps[ds.index[0][0]]
    fut_ext = it.get("future_poses_ext")
    load["item"] = {k: [str(v.dtype), list(v.shape)] for k, v in it.items() if torch.is_tensor(v) and v.ndim >= 1}
    # every item: frames == the bridge's R8_LEGAL pack of the bank; LAW row == ST of the next frame; pose_last; validity
    files_sorted = sorted(f for f in os.listdir(a.out) if f.endswith(".v2ep.pt"))
    per = []
    for i, (e_i, t_) in enumerate(ds.index):
        tok_i = files_sorted[e_i][len("navtrain-"):-len(".v2ep.pt")]
        it_i = it if i == 0 else ds[i]
        rec_i = recs[tok_i]
        bank4 = np.asarray(bank[bank_idx["tokens"][tok_i]["rows"]])
        packed = E.B.pack_rows(bank4, E.B.slot_sources8(E.RB7.times_rel_t0(rec_i), "ST"))
        packed = packed if torch.is_tensor(packed) else torch.from_numpy(np.asarray(packed))
        tr_i = E.token_poses(rec_i, logs[rec_i["log_name"]], pos_of[tok_i])
        fr_i = logs[rec_i["log_name"]]
        nxt = fr_i[pos_of[tok_i] + 1]["token"] if pos_of[tok_i] + 1 < len(fr_i) else None
        law_hwc = fm_by_ft[nxt]["hwc"] if (nxt in sample_set and nxt in fm_by_ft) else fm_by_ft[tok_i]["hwc"]
        ff = it_i["future_frames"]
        lawx = torch.from_numpy(law_hwc).permute(2, 0, 1)
        per.append({"token": tok_i, "frames_eq_pack_rows": bool(torch.equal(it_i["frames"], packed)),
                    "law_row_eq_ST_law": bool(all(torch.equal(ff[4][3 * c:3 * c + 3], lawx) for c in range(3))),
                    "law_is_next": nxt in sample_set and nxt in fm_by_ft,
                    "pose_last_ok": bool(torch.equal(it_i["pose_last"][:3], torch.zeros(3)) and
                                         abs(float(it_i["pose_last"][3]) - tr_i["v0"]) < 1e-5),
                    "pose_hist_eq_bridge": bool(np.array_equal(eps[e_i].poses[0:8].numpy(), tr_i["hist"])),
                    "valid_ext": int(it_i["future_valid_ext"].sum()) if "future_valid_ext" in it_i else None,
                    "shapes_ok": list(it_i["frames"].shape) == [8, 9, 416, 1024] and list(ff.shape) == [20, 9, 416, 1024]})
    load["per_item"] = per
    load["frames_equal_bridge_pack_rows"] = all(p["frames_eq_pack_rows"] for p in per)
    load["law_row_is_ST_law_frame"] = all(p["law_row_eq_ST_law"] for p in per)
    load["n_items_with_next_token_law"] = sum(p["law_is_next"] for p in per)
    load["pose_last_is_origin_v0"] = all(p["pose_last_ok"] for p in per)
    load["pose_hist_equal_bridge_all"] = all(p["pose_hist_eq_bridge"] for p in per)
    load["future_valid_ext_true"] = sorted({p["valid_ext"] for p in per})
    load["frames_shape_ok"] = all(p["shapes_ok"] for p in per)
    load["pose_last"] = [float(x) for x in it["pose_last"]]
    C["E1-LOAD"] = {**load, "pass": bool(load["windows_after_filter"] == len(eps) and load["frames_shape_ok"]
                                         and load["frames_equal_bridge_pack_rows"] and load["pose_last_is_origin_v0"]
                                         and load["law_row_is_ST_law_frame"] and load["pose_hist_equal_bridge_all"]
                                         and (fut_ext is None or list(fut_ext.shape) == [60, 4]))}
    # E1-UNPATCHED: the fb47a58 loader on the same dir must fail loud
    try:
        eu = U.build_v2_providers(a.out, lru_size=4, rebuild=True, verbose=False)
        _ = eu[0].frames[0:8]
        C["E1-UNPATCHED"] = {"raised": False, "pass": False}
    except Exception as e:                     # noqa: BLE001
        C["E1-UNPATCHED"] = {"raised": True, "exception": f"{type(e).__name__}: {str(e)[:200]}", "pass": True}
    # E1-REG: patched vs fb47a58 on real PhysicalAI files
    reg_dir = os.path.join(NT, "scratch_reg")
    shutil.rmtree(reg_dir, ignore_errors=True)
    os.makedirs(reg_dir)
    real = sorted(f for f in os.listdir(PAI_CACHE) if f.endswith(".v2ep.pt"))[:3]
    for f in real:
        os.symlink(os.path.realpath(os.path.join(PAI_CACHE, f)), os.path.join(reg_dir, f))
    pe = P.build_v2_providers(reg_dir, lru_size=4, rebuild=True, verbose=False)
    ue = U.build_v2_providers(reg_dir, lru_size=4, rebuild=True, verbose=False)
    regs = []
    for i in range(len(real)):
        T = pe[i].frames.shape[0]
        for (s, e) in ((0, 8), (8, 28), (T - 28, T - 8)):
            regs.append(bool(torch.equal(pe[i].frames[s:e], ue[i].frames[s:e])))
        regs.append(pe[i].frames.shape == ue[i].frames.shape and torch.equal(pe[i].poses, ue[i].poses))
    full_eq = None
    fp = os.path.join(reg_dir, real[0])
    a_ = P.decode_full_episode(fp)
    b_ = U.decode_full_episode(fp)
    full_eq = bool(torch.equal(a_.frames, b_.frames) and torch.equal(a_.poses, b_.poses))
    C["E1-REG_patched_vs_fb47a58_on_PhysicalAI"] = {"files": real, "n_window_checks": len(regs), "n_equal": sum(regs),
                                                     "decode_full_episode_equal": full_eq, "pass": all(regs) and full_eq}
    shutil.rmtree(reg_dir, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    rep["pass_all"] = all(v.get("pass") for v in C.values())
    rep["wall_s"] = round(time.time() - t_start, 1)
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(rep, open(a.report, "w"), indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("pass", "max_abs", "n_equal", "max_dxy_m", "raised",
                                                                   "windows_after_filter", "dataset_class")}
                      for k, v in C.items()}, indent=1, default=str))
    print(f"ZZSMOKE_{'PASS' if rep['pass_all'] else 'FAIL'} wall={rep['wall_s']}s", flush=True)
    return 0 if rep["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
