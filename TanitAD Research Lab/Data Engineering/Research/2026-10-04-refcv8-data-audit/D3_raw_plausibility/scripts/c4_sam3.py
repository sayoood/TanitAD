#!/usr/bin/env python3
"""D3 check 4 -- SAM3 map GT (sam3_gt_v3) plausibility.

usage: c4_sam3.py <gt_dir> <manifest.pt> <out.json> [--max N] [--workers K] [--every E]

Grid (meta_json, schema tanitad.sam3_map_gt/3, 'fine'): 1000 x 600 cells of 0.1 m, rig frame +x fwd / +y LEFT,
x = (i+0.5)*0.1 in [0,100), y = -30 + (j+0.5)*0.1 in [-30,30);  codes 0..7 = {seen-no-class, drivable, lane/road line,
crosswalk, arrow/text, non-drivable edge, hatched, sidewalk/verge}, 255 = not seen.
GT frame axis0 = RAW v2ep index r;  the cache poses row p = r - 2  (n_stack-1 = 2 provider rows are dropped).  The alignment is
VERIFIED here, not assumed (control K1): the speed implied by T_world_rig is cross-correlated with the manifest pose speed.

Literal thresholds / definitions (road geometry, not the builder's constants):
  path test   : the ego's OWN driven rear-axle positions over the NEXT 6 s (60 rows) are, by definition, on the road.  A point is
                scored only if 8 m <= x < 100 m (beyond the bonnet, inside the grid) and |y| < 30 m.
  strict      : code 1                    "drivable"
  roadlike    : codes {1,2,3,4,6}         drivable + painted road furniture
  contradict  : codes {5,7}               non-drivable edge / sidewalk under the ego's own wheels
  unseen      : code 255 ; noclass : code 0
Controls: K1 alignment offset must be +2 rows; K2 mutation -- the SAME path points shifted 6 m sideways (+y) must collapse the
roadlike fraction (the metric can see a misregistration); K3 a clip with every cell == 255 must be reported EMPTY.
Ids are sha12 (the GT files are already named by sha12).
"""
import glob, hashlib, json, math, os, sys, time
import multiprocessing as mp
import numpy as np

ROADLIKE = np.zeros(256, bool); ROADLIKE[[1, 2, 3, 4, 6]] = True
STRICT = np.zeros(256, bool); STRICT[1] = True
CONTRA = np.zeros(256, bool); CONTRA[[5, 7]] = True
H = 60


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def path_cells(Tw, r):
    """next-6s ego positions in the rig frame at row r -> (x, y) arrays"""
    Rr, tr = Tw[r, :3, :3], Tw[r, :3, 3]
    P = Tw[r + 1:r + 1 + H, :3, 3]
    rel = (P - tr) @ Rr          # = R^T (P - t)
    return rel[:, 0], rel[:, 1]


def score_path(codes, x, y, dy=0.0):
    y = y + dy
    ok = (x >= 8.0) & (x < 100.0) & (np.abs(y) < 30.0)
    i = np.floor(x[ok] / 0.1).astype(np.int64); j = np.floor((y[ok] + 30.0) / 0.1).astype(np.int64)
    c = codes[i, j]
    return ok.sum(), STRICT[c].sum(), ROADLIKE[c].sum(), CONTRA[c].sum(), (c == 255).sum(), (c == 0).sum()


def one(args):
    f, vpose, every = args
    sh = os.path.basename(f).split(".")[0]
    try:
        z = np.load(f, allow_pickle=True)
        fine = z["fine_codes"]            # [201,1000,600]
        Tw = z["T_world_rig"].astype(np.float64)
        meta = json.loads(str(z["meta_json"].item()) if z["meta_json"].shape == () else str(z["meta_json"][0]))
    except Exception as e:                # unreadable -> counted, never "absent"
        return {"sha12": sh, "error": repr(e)[:120]}
    T = fine.shape[0]
    out = {"sha12": sh, "split": meta.get("source", {}).get("split"), "T": int(T)}
    # ---- K1 alignment: GT row r <-> cache pose row r-k.  Path-shape error (first point removed) for k = 0..4. ----
    if vpose is not None:
        errs = {}
        for k in range(0, 5):
            n = min(len(vpose), T - k)
            if n < 100:
                continue
            a_ = Tw[k:k + n, :2, 3] - Tw[k, :2, 3]; b_ = vpose[:n, :2] - vpose[0, :2]
            errs[k] = float(np.sqrt(((a_ - b_) ** 2).sum(1)).mean())
        if errs:
            kb = min(errs, key=errs.get)
            out["align_best_k"] = int(kb); out["align_err_k2"] = errs.get(2); out["align_err_min"] = errs[kb]
            # moving clip? (path length > 30 m) -- a parked clip cannot discriminate
            out["align_path_len_m"] = float(np.hypot(*np.diff(vpose[:, :2], axis=0).T).sum())
            out["align_clear_not2"] = bool(kb != 2 and errs.get(2, 9) > 1.5 * errs[kb] and errs.get(2, 9) - errs[kb] > 0.2)
    # ---- image-time vs pose-time jitter: t_img - t_query (cache poses are at the uniform grid t_query) ----
    dlt = (z["t_img_us"].astype(np.float64) - z["t_query_us"].astype(np.float64)) / 1e6
    out["img_minus_query_s"] = {"mean": float(dlt.mean()), "p95": float(np.quantile(dlt, .95)), "max": float(dlt.max()), "min": float(dlt.min())}
    vgt = np.hypot(*(np.diff(Tw[:, :2, 3], axis=0).T)) / 0.1007
    out["mean_speed_gt"] = float(vgt.mean())
    # ---- coverage / class presence per frame ------------------------------------------------------------
    sub = fine[:, ::5, ::5]                       # 0.5 m cells, [201,200,120]
    seen = sub != 255
    out["seen_frac_mean"] = float(seen.mean())
    corridor = sub[:, :120, 52:68]                # x in [0,60), |y| < 4 m  (cols (y+30)/0.5 = 52..68)
    out["corridor_seen_frac_mean"] = float((corridor != 255).mean())
    cnt = np.stack([(sub == c).sum((1, 2)) for c in range(8)], 1)        # [T, 8]
    out["class_cells_total"] = cnt.sum(0).astype(np.int64).tolist()
    out["empty_gt"] = bool((cnt.sum() == 0) or (cnt[:, 1:].sum() == 0))
    out["frames_all_unseen"] = int((seen.sum((1, 2)) == 0).sum())
    out["frames_with_lane_line"] = int((cnt[:, 2] >= 20).sum())            # >= 20 cells of 0.25 m^2 = 5 m^2
    out["frames_with_nondrivable_edge"] = int((cnt[:, 5] >= 20).sum())
    out["frames_with_drivable"] = int((cnt[:, 1] >= 100).sum())
    # ---- ego-path test over windows -----------------------------------------------------------------------
    tot = np.zeros(6, np.int64); mut = np.zeros(6, np.int64)
    rows = list(range(9, min(T - 1 - 1, 179), every))      # GT rows for window NOW = pose rows 7..177
    per_window = []
    for r in rows:
        x, y = path_cells(Tw, r)
        if r + H >= T:                      # truncated future: use what exists
            x, y = x, y
        s = np.array(score_path(fine[r], x, y), np.int64)
        tot += s
        mut += np.array(score_path(fine[r], x, y, dy=6.0), np.int64)
        per_window.append(float(s[2] / s[0]) if s[0] >= 5 else float("nan"))
    out["path"] = {"scored_pts": int(tot[0]), "strict": int(tot[1]), "roadlike": int(tot[2]), "contra": int(tot[3]), "unseen": int(tot[4]), "noclass": int(tot[5]),
                   "mut6m_roadlike": int(mut[2]), "mut6m_scored": int(mut[0]), "n_windows": len(rows),
                   "n_windows_scored": int(np.sum(~np.isnan(per_window))),
                   "windows_roadlike_lt_0.5": int(np.nansum(np.array(per_window) < 0.5)),
                   "windows_roadlike_lt_0.8": int(np.nansum(np.array(per_window) < 0.8))}
    return out


def main():
    gt_dir, man_path, out_path = sys.argv[1:4]
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 10 ** 9
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 10
    import torch
    man = torch.load(man_path, map_location="cpu", weights_only=False)
    vp = {}
    for cid, P in zip(man["clip_id"], man["poses"]):
        vp[sha12(cid)] = P.numpy().astype(np.float64)
    files = sorted(glob.glob(os.path.join(gt_dir, "*.sam3mapgt.npz")))
    t0 = time.time()
    cache_ids = set(vp)
    gt_ids = {os.path.basename(f).split(".")[0] for f in files}
    rng = np.random.default_rng(3)
    sel = [f for f in files if os.path.basename(f).split(".")[0] in cache_ids]
    if len(sel) > mx:
        sel = [sel[i] for i in sorted(rng.choice(len(sel), mx, replace=False))]
    jobs = [(f, vp.get(os.path.basename(f).split(".")[0]), every) for f in sel]
    # K3 control on a synthetic all-255 clip (run through the real function via a temp file)
    import tempfile
    td = tempfile.mkdtemp()
    fake = os.path.join(td, "000000000000.sam3mapgt.npz")
    np.savez(fake, fine_codes=np.full((201, 1000, 600), 255, np.uint8), T_world_rig=np.tile(np.eye(4), (201, 1, 1)) + 0.0,
             meta_json=np.array([json.dumps({"source": {"split": "synthetic"}})]), t_img_us=np.zeros(201), t_query_us=np.zeros(201))
    k3 = one((fake, None, 20))
    assert k3["empty_gt"] is True, k3
    res = []
    with mp.Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            res.append(r)
            if i % 50 == 0:
                print(i, len(jobs), round(time.time() - t0, 1), flush=True)
    err = [r for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    out = {"gt_dir": gt_dir, "n_gt_files": len(files), "n_selected": len(sel), "n_read_ok": len(ok), "n_unreadable": len(err),
           "unreadable_sha12": [r["sha12"] for r in err][:10],
           "cache_clips_without_gt_file": sorted(cache_ids - gt_ids)[:20], "n_cache_clips_without_gt_file": len(cache_ids - gt_ids),
           "gt_files_not_in_cache": len(gt_ids - cache_ids), "K3_synthetic_all_unseen_is_empty": k3["empty_gt"], "every": every}
    if ok:
        pt = np.array([[r["path"]["scored_pts"], r["path"]["strict"], r["path"]["roadlike"], r["path"]["contra"], r["path"]["unseen"], r["path"]["noclass"],
                        r["path"]["mut6m_roadlike"], r["path"]["mut6m_scored"]] for r in ok], np.int64)
        tot = pt.sum(0)
        out["path_pooled"] = {"scored_pts": int(tot[0]), "strict_frac": tot[1] / tot[0], "roadlike_frac": tot[2] / tot[0], "contra_frac": tot[3] / tot[0],
                              "unseen_frac": tot[4] / tot[0], "noclass_frac": tot[5] / tot[0], "K2_mutated_6m_roadlike_frac": tot[6] / tot[7]}
        cf = np.array([r["path"]["roadlike"] / r["path"]["scored_pts"] if r["path"]["scored_pts"] > 50 else np.nan for r in ok])
        sf = np.array([r["path"]["strict"] / r["path"]["scored_pts"] if r["path"]["scored_pts"] > 50 else np.nan for r in ok])
        out["path_per_clip_roadlike"] = {"n_scored": int(np.sum(~np.isnan(cf))), "n_with_too_few_pts": int(np.sum(np.isnan(cf))),
                                         **{f"q{int(q*100):02d}": float(np.nanquantile(cf, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75)},
                                         "n_lt_0.5": int(np.nansum(cf < 0.5)), "n_lt_0.8": int(np.nansum(cf < 0.8)), "n_lt_0.9": int(np.nansum(cf < 0.9)),
                                         "worst10_sha12": [ok[i]["sha12"] for i in np.argsort(np.nan_to_num(cf, nan=9))[:10]],
                                         "worst10_val": [round(float(v), 3) for v in np.sort(np.nan_to_num(cf, nan=9))[:10]]}
        out["path_per_clip_strict"] = {f"q{int(q*100):02d}": float(np.nanquantile(sf, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75)}
        nw_ = np.array([r["path"]["n_windows_scored"] for r in ok]); lt5 = np.array([r["path"]["windows_roadlike_lt_0.5"] for r in ok]); lt8 = np.array([r["path"]["windows_roadlike_lt_0.8"] for r in ok])
        out["windows"] = {"n_scored": int(nw_.sum()), "pct_roadlike_lt_0.5": float(100 * lt5.sum() / nw_.sum()), "pct_roadlike_lt_0.8": float(100 * lt8.sum() / nw_.sum())}
        mv = [r for r in ok if r.get("align_best_k") is not None and r.get("align_path_len_m", 0) > 30]
        ks = [r["align_best_k"] for r in mv]
        out["K1_alignment"] = {"n_moving_testable": len(mv), "hist_best_k": {str(k): ks.count(k) for k in sorted(set(ks))},
                               "n_clear_not_k2": int(sum(r["align_clear_not2"] for r in mv)),
                               "n_not_testable": len(ok) - len(mv)}
        dj = np.array([r["img_minus_query_s"]["mean"] for r in ok]); dm = np.array([r["img_minus_query_s"]["max"] for r in ok]); dmin = np.array([r["img_minus_query_s"]["min"] for r in ok])
        ms_ = np.array([r["mean_speed_gt"] for r in ok])
        out["img_vs_pose_time"] = {"mean_s_over_clips": float(dj.mean()), "max_s_overall": float(dm.max()), "min_s_overall": float(dmin.min()),
                                   "implied_mean_longitudinal_offset_m": float((dj * ms_).mean()), "implied_p95_clip_max_offset_m": float(np.quantile(dm * ms_, .95))}
        out["coverage"] = {"seen_frac_mean_q": {f"q{int(q*100):02d}": float(np.quantile([r["seen_frac_mean"] for r in ok], q)) for q in (0.05, 0.5, 0.95)},
                           "corridor_seen_frac_q": {f"q{int(q*100):02d}": float(np.quantile([r["corridor_seen_frac_mean"] for r in ok], q)) for q in (0.01, 0.05, 0.5, 0.95)},
                           "n_corridor_seen_lt_0.2": int(sum(r["corridor_seen_frac_mean"] < 0.2 for r in ok)),
                           "n_empty_gt": int(sum(r["empty_gt"] for r in ok)), "empty_gt_sha12": [r["sha12"] for r in ok if r["empty_gt"]][:10],
                           "frames_all_unseen_total": int(sum(r["frames_all_unseen"] for r in ok)), "frames_total": int(sum(r["T"] for r in ok))}
        Tt = np.array([r["T"] for r in ok])
        out["thin_classes"] = {
            "pct_frames_with_lane_line": float(100 * sum(r["frames_with_lane_line"] for r in ok) / Tt.sum()),
            "pct_frames_with_nondrivable_edge": float(100 * sum(r["frames_with_nondrivable_edge"] for r in ok) / Tt.sum()),
            "pct_frames_with_drivable": float(100 * sum(r["frames_with_drivable"] for r in ok) / Tt.sum()),
            "n_clips_lane_line_in_lt_10pct_frames": int(sum(r["frames_with_lane_line"] / r["T"] < 0.10 for r in ok)),
            "n_clips_no_lane_line_at_all": int(sum(r["frames_with_lane_line"] == 0 for r in ok)),
            "n_clips_no_edge_at_all": int(sum(r["frames_with_nondrivable_edge"] == 0 for r in ok)),
            "class_cell_share_pct_of_seen": None}
        cc = np.array([r["class_cells_total"] for r in ok], np.int64).sum(0)
        out["thin_classes"]["class_cell_share_pct_of_seen"] = {str(i): round(float(100 * cc[i] / cc.sum()), 3) for i in range(8)}
        out["per_clip"] = [{"sha12": r["sha12"], "road": round(r["path"]["roadlike"] / max(r["path"]["scored_pts"], 1), 3), "pts": r["path"]["scored_pts"],
                            "corr_seen": round(r["corridor_seen_frac_mean"], 3), "lane_frames": r["frames_with_lane_line"], "T": r["T"], "empty": r["empty_gt"]} for r in ok]
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
