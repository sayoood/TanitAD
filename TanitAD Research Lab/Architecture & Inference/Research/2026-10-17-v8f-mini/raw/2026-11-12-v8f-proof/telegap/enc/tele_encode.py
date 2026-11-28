#!/usr/bin/env python3
"""Tele windows T8 and T4 and their tokens for the pilot frames (card ADD122 sections 2 and 5). Pod only (GPU for the encoder).

Per clip: the frames are the frames the pilot (card ADD99) encoded, with the time of the wide video frame that was decoded for each
(wide stage root, enc_common.frame_plan: t_frames[cache_idx]). For each such time the tele frame nearest in time is taken if it is at
most 17000 us away (a frame at exactly 17000 us is matched); a pilot frame with none gets mask False and a zero token row.
Windows are cut from the RAW decoded tele frame by the unchanged routines of the data kit of card ADD99 (window_frame, window_grid,
window_sample, Encoder.window_tokens, imported from enc_common) with the tele camera's own intrinsics (tele stage root):
  T8 = 8 deg, azimuth 0, elevation -2 deg ; T4 = 4 deg, azimuth 0, elevation -1 deg (azimuth left positive, elevation up positive,
  elevation 0 = the tele optical axis, as the data kit's convention).
Output under --out: tok_T/<clip>.npy float16 [n_frames, 2, 65, 1024] (window order T8 then T4; token 0 = class token, 1..64 = 8x8 pooled
patch tokens), match/<clip>.npz (tele_idx int32 with -1 for none, dt_us float64 with nan for none, mask bool, t_wide float64),
tok_T/ledger.jsonl (resume record), tok_T/MANIFEST.json. Frame i of a file is frame i of the pilot's token file.
Nothing is deleted. A file is written whole under a .part name in its own folder and renamed once (the data kit's way).
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("TG_ADD99_KIT", ""), "/workspace/v8proof/cropread/kit/enc", os.path.join(HERE, "..", "..", "cropread", "enc")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from enc_common import (Layout, WIN_PX, decode_selected, frame_plan, read_ids, sha256_file,   # noqa: E402
                        window_frame, window_grid, window_sample, Encoder)

BATCH = 8
TELE_CAM = "camera_front_tele_30fov"
TOL_US = 17000                                   # card section 2: at most 17 ms on the microsecond clock
# (size_deg, azimuth_left_deg, elevation_up_deg), order T8 then T4 (card section 2)
TELE_WINDOWS = ((8.0, 0.0, -2.0), (4.0, 0.0, -1.0))
TELE_SIZES_DEG = (8.0, 4.0)
TOK_DIR = "tok_T"


def log(m):
    print(f"[tele-enc {time.strftime('%H:%M:%S')}] {m}", flush=True)


def tok_path(out, clip):
    return os.path.join(out, TOK_DIR, f"{clip}.npy")


def match_path(out, clip):
    return os.path.join(out, "match", f"{clip}.npz")


def read_ts(path):
    """Timestamps parquet -> float64 array [m] of the first column whose name contains 'time' (microseconds, clip-relative)."""
    import pandas as pd
    ts = pd.read_parquet(path)
    tcol = next(c for c in ts.columns if "time" in str(c).lower())
    return ts[tcol].to_numpy(np.float64)


def pilot_times(plan):
    """Times [us] of the pilot's frames: the wide video frame that was decoded for each cache frame (enc_common.frame_plan
    cache_idx = the builder's frame choice; window_encode.py decodes exactly these video frames)."""
    return np.asarray(plan["t_frames"], np.float64)[np.asarray(plan["cache_idx"], np.int64)]


def match_frames(t_wide, t_tele, tol_us=TOL_US):
    """Nearest tele frame in time for every wide time. -> (tele_idx int64 [n] (-1 where none), dt_us float64 [n] (tele - wide; nan where
    none), mask bool [n]). mask is |dt| <= tol_us (inclusive); a tie between two tele frames takes the earlier one."""
    t_wide = np.asarray(t_wide, np.float64)
    t_tele = np.asarray(t_tele, np.float64)
    n = t_wide.shape[0]
    if t_tele.shape[0] == 0:
        return np.full(n, -1, np.int64), np.full(n, np.nan), np.zeros(n, bool)
    if t_tele.shape[0] > 1 and np.any(np.diff(t_tele) < 0):
        raise ValueError("tele timestamps are not sorted")
    hi = np.searchsorted(t_tele, t_wide).clip(0, t_tele.shape[0] - 1)
    lo = (hi - 1).clip(0, t_tele.shape[0] - 1)
    d_lo = np.abs(t_tele[lo] - t_wide)
    d_hi = np.abs(t_tele[hi] - t_wide)
    idx = np.where(d_lo <= d_hi, lo, hi)
    dt = t_tele[idx] - t_wide
    mask = np.abs(dt) <= float(tol_us)
    return np.where(mask, idx, -1).astype(np.int64), np.where(mask, dt, np.nan), mask


def point_in_other_cam(R_src_to_veh, R_dst_to_veh, az_left_deg, el_up_deg):
    """Direction (azimuth left+, elevation up+) of the centre ray of a window pointed at (az, el) in the SOURCE camera, expressed in the
    DESTINATION camera own (az, el). Both rotations are cam -> vehicle; camera axes +x right, +y down, +z boresight. The same routine
    as in ctl_t2.py (a test pins that they agree)."""
    import math
    az, el = math.radians(az_left_deg), math.radians(el_up_deg)
    d = np.array([-math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az)])
    e = np.asarray(R_dst_to_veh).T @ (np.asarray(R_src_to_veh) @ d)
    return math.degrees(math.atan2(-e[0], e[2])), math.degrees(math.atan2(-e[1], math.hypot(e[0], e[2])))


def rot_cam_to_vehicle(csv_path, clip):
    import pandas as pd
    from tanitad.data.physicalai import FrontWideExtrinsics
    df = pd.read_csv(csv_path, dtype={"clip_id": str})
    r = df[df["clip_id"] == clip].iloc[0]
    return FrontWideExtrinsics(float(r.qx), float(r.qy), float(r.qz), float(r.qw), float(r.x), float(r.y), float(r.z)).rotation_cam_to_vehicle()


def tele_dirs(clip, Lw, Lt):
    """note N1 of card ADD122: the tele windows are centred on the directions (azimuth, elevation) of TELE_WINDOWS given in the WIDE camera
    rectified frame (the frame of the wide windows of card ADD99), carried into the tele camera own frame by the extrinsics of both cameras
    of this clip (centre direction only; a roll between the cameras is not applied). A clip with a missing row raises: no other direction is used."""
    rw = rot_cam_to_vehicle(f"{Lw.root}/calibration/front_wide_extrinsics.csv", clip)
    rt = rot_cam_to_vehicle(f"{Lt.root}/calibration/front_tele_extrinsics.csv", clip)
    return [point_in_other_cam(rw, rt, w[1], w[2]) for w in TELE_WINDOWS]


def windows_from_video(vid, intr, frames, dirs=None):
    """vid [B,3,h,w] (uint8 or float) -> uint8 [B,2,3,256,256]: T8 then T4. Pure CPU. dirs = [(az, el), (az, el)] in the tele camera own
    frame (tele_dirs); None (tests only) takes the numbers of TELE_WINDOWS as they stand."""
    import torch
    h, w = int(vid.shape[-2]), int(vid.shape[-1])
    B = vid.shape[0]
    out = torch.zeros((B, 2, 3, WIN_PX, WIN_PX), dtype=torch.uint8)
    for i in range(2):
        az, el = (TELE_WINDOWS[i][1], TELE_WINDOWS[i][2]) if dirs is None else dirs[i]
        g, m = window_grid(intr, h, w, frames[i], float(az), float(el))
        out[:, i] = window_sample(vid.float(), g.expand(B, -1, -1, -1), m.expand(B, -1, -1))
    return out.numpy()


def make_tele_windows(clip, Lw, Lt, torch_threads=2, wide_tok_dir=None):
    """-> (uint8 [n,2,3,256,256] zeros on unmatched frames, match dict, plan, secs). Pure CPU.
    Lw = layout of the wide stage root (timestamps of the pilot), Lt = layout of the tele stage root."""
    import torch
    from tanitad.data.physicalai import intrinsics_for_clip
    t0 = time.time()
    torch.set_num_threads(torch_threads)
    plan = frame_plan(Lw.ts(clip))
    n = plan["n_cache"]
    if wide_tok_dir:
        wp = os.path.join(wide_tok_dir, f"{clip}.npy")
        if os.path.exists(wp):
            wn = int(np.load(wp, mmap_mode="r").shape[0])
            if wn != n:
                raise ValueError(f"{clip[:8]}: pilot token file has {wn} frames, the plan says {n}")
    t_wide = pilot_times(plan)
    idx, dt, mask = match_frames(t_wide, read_ts(Lt.ts(clip)))
    dirs = tele_dirs(clip, Lw, Lt)
    match = {"tele_idx": idx, "dt_us": dt, "mask": mask, "t_wide": t_wide, "tele_dirs_deg": np.asarray(dirs, np.float64)}
    intr = intrinsics_for_clip(clip, Lt.root)
    assert intr.per_clip, f"{clip[:8]}: corpus-median intrinsics refused (per-clip tele table missing)"
    frames = [window_frame(s) for s in TELE_SIZES_DEG]
    out = np.zeros((n, 2, 3, WIN_PX, WIN_PX), np.uint8)
    by_idx = collections.defaultdict(list)
    for j in np.nonzero(mask)[0]:
        by_idx[int(idx[j])].append(int(j))
    need = sorted(by_idx)
    pend = []

    def flush():
        nonlocal pend
        if not pend:
            return
        js = [j for j, _ in pend]
        vid = torch.stack([f for _, f in pend]).float()
        out[js] = windows_from_video(vid, intr, frames, dirs)
        pend = []

    seen = 0
    if need:
        for vi, fr in decode_selected(Lt.mp4(clip), need):
            for j in by_idx[vi]:
                pend.append((j, fr))
                seen += 1
                if len(pend) >= BATCH:
                    flush()
        flush()
    if seen != int(mask.sum()):
        raise RuntimeError(f"{clip[:8]}: decoded {seen} of {int(mask.sum())} wanted tele frames (video shorter than the timestamp table?)")
    return out, match, plan, time.time() - t0


def encode_tele_tokens(enc, wins, mask):
    """wins uint8 [n,2,3,256,256], mask bool [n] -> float16 [n,2,65,1024]; rows of unmatched frames are zero."""
    import torch
    n = wins.shape[0]
    tok = torch.zeros((n, 2, 65, 1024), dtype=torch.float16)
    idx = np.nonzero(mask)[0]
    if idx.size:
        x = torch.from_numpy(wins[idx].reshape(idx.size * 2, 3, WIN_PX, WIN_PX))
        t = enc.window_tokens(x).reshape(idx.size, 2, 65, -1)
        tok[torch.from_numpy(idx)] = t.to(torch.float16)
    return tok.numpy()


def write_whole(path, writer):
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


def read_ledger(path):
    d = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    d[r["clip"]] = r
                except Exception:                                    # noqa: BLE001
                    pass
    return d


def write_manifest(out):
    led = read_ledger(os.path.join(out, TOK_DIR, "ledger.jsonl"))
    man = {"arm": "T", "layout": "float16 [n_frames, 2, 65, 1024]; window order T8 then T4; token 0 = class token, 1..64 = 8x8 pooled patch "
                                 "tokens row-major; unmatched pilot frames are zero rows (see match/<clip>.npz mask)",
           "windows": [list(w) for w in TELE_WINDOWS], "tol_us": TOL_US, "n_clips": len(led),
           "clips": {c: {"n_frames": r["n_frames"], "n_matched": r["n_matched"], "sha256": r["sha256"], "bytes": r["bytes"]}
                     for c, r in sorted(led.items())}}
    with open(os.path.join(out, TOK_DIR, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def precheck(clips, Lw, Lt):
    """-> (refused [(clip, reason)] for a missing WIDE timestamp file (refuses the whole run), no_tele [(clip, reason)])."""
    refused, no_tele = [], []
    for c in clips:
        if not os.path.exists(Lw.ts(c)):
            refused.append((c, "no wide timestamps file in the wide stage root"))
        elif not (os.path.exists(Lt.mp4(c)) and os.path.exists(Lt.ts(c))):
            no_tele.append((c, "no staged tele clip"))
    return refused, no_tele


def run_clips(clips, Lw, Lt, out, workers=3, torch_threads=2, enc_bs=8, min_free_gb=25.0, enc=None, keep_windows=False,
              limit=0, wide_tok_dir=None):
    """Encode the clips (resumable). -> (n_done, fails, no_tele, rc)."""
    import shutil
    if limit:
        clips = clips[:limit]
    refused, no_tele = precheck(clips, Lw, Lt)
    if refused:
        log(f"REFUSED: {len(refused)} clips have no wide timestamps (first {refused[0][0][:8]})")
        return 0, refused, no_tele, 4
    os.makedirs(os.path.join(out, TOK_DIR), exist_ok=True)
    os.makedirs(os.path.join(out, "match"), exist_ok=True)
    if keep_windows:
        os.makedirs(os.path.join(out, "win"), exist_ok=True)
    skip = {c for c, _ in no_tele}
    led_path = os.path.join(out, TOK_DIR, "ledger.jsonl")
    led = read_ledger(led_path)

    def done(c):
        r = led.get(c)
        p = tok_path(out, c)
        return bool(r) and os.path.exists(p) and os.path.getsize(p) == r["bytes"] and os.path.exists(match_path(out, c))

    todo = [c for c in clips if c not in skip and not done(c)]
    log(f"{len(clips)} clips, {len(clips) - len(skip) - len(todo)} done, {len(skip)} without a tele clip, {len(todo)} to do; workers {workers}")
    if todo and enc is None:
        import torch
        assert torch.cuda.is_available(), "no CUDA: refusing a silent CPU fallback"
        enc = Encoder("cuda", enc_bs)
        log("encoder ready")
    lf = open(led_path, "a", buffering=1, encoding="utf-8")
    fails, n_done, t_start = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        q = collections.deque()
        it = iter(todo)

        def submit_next():
            c = next(it, None)
            if c is not None:
                q.append((c, ex.submit(make_tele_windows, c, Lw, Lt, torch_threads, wide_tok_dir)))
        for _ in range(workers + 1):
            submit_next()
        while q:
            c, fut = q.popleft()
            submit_next()
            try:
                if shutil.disk_usage(out).free / 1e9 < min_free_gb:
                    log("STOP: disk free below the floor")
                    return n_done, fails, no_tele, 7
                wins, match, plan, t_prep = fut.result()
                te = time.time()
                tok = encode_tele_tokens(enc, wins, match["mask"])
                t_enc = time.time() - te
                write_whole(match_path(out, c), lambda fh: np.savez(
                    fh, tele_idx=match["tele_idx"].astype(np.int32), dt_us=match["dt_us"], mask=match["mask"], t_wide=match["t_wide"]))
                p = tok_path(out, c)
                write_whole(p, lambda fh: np.save(fh, tok))
                row = {"clip": c, "n_frames": int(tok.shape[0]), "n_matched": int(match["mask"].sum()), "bytes": os.path.getsize(p),
                       "sha256": sha256_file(p), "t_prep_s": round(t_prep, 2), "t_enc_s": round(t_enc, 2), "t": time.strftime("%F %T")}
                lf.write(json.dumps(row) + "\n")
                if keep_windows:
                    np.save(os.path.join(out, "win", f"{c}.npy"), wins)
                n_done += 1
                if n_done <= 3 or n_done % 25 == 0:
                    log(f"{n_done}/{len(todo)} {c[:8]} n={tok.shape[0]} matched={row['n_matched']} prep={t_prep:.1f}s enc={t_enc:.1f}s "
                        f"wall/clip={(time.time() - t_start) / n_done:.1f}s")
                    write_manifest(out)
            except Exception as e:                                   # noqa: BLE001
                fails.append((c, f"{type(e).__name__}: {str(e)[:200]}"))
                log(f"FAIL {c[:8]} {type(e).__name__}: {str(e)[:200]}")
    lf.close()
    write_manifest(out)
    with open(os.path.join(out, TOK_DIR, "ENCODE_FAILS.json"), "w", encoding="utf-8") as fh:
        json.dump({"fails": fails, "no_tele": no_tele}, fh, indent=1)
    log(f"done: {n_done} encoded, {len(fails)} failed, {len(no_tele)} without a tele clip, wall {time.time() - t_start:.0f}s")
    return n_done, fails, no_tele, (0 if not fails else 5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True)
    ap.add_argument("--wide-root", required=True, help="stage root of the wide clips (timestamps of the pilot): the data kit's stage root")
    ap.add_argument("--tele-root", required=True, help="stage root written by tele_stage.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wide-tok-dir", default="", help="optional: the pilot's tok/D folder; the frame count of each file is cross-checked")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--enc-bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N clips of the list")
    ap.add_argument("--keep-windows", action="store_true", help="smoke only: keep win/<clip>.npy (private camera pixels, pod only)")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    a = ap.parse_args(argv)
    clips = read_ids(a.list)
    n_done, fails, no_tele, rc = run_clips(clips, Layout(a.wide_root), Layout(a.tele_root), a.out, a.workers, a.torch_threads, a.enc_bs,
                                           a.min_free_gb, keep_windows=a.keep_windows, limit=a.limit, wide_tok_dir=a.wide_tok_dir or None)
    print("ZZ TELE ENCODE DONE ZZ" if rc == 0 else f"ZZ TELE ENCODE FAIL ZZ rc={rc} n={len(fails)}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
