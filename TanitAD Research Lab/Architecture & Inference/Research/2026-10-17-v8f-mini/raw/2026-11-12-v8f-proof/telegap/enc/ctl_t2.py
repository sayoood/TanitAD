#!/usr/bin/env python3
"""Control t2 (geometry) of card ADD122, as the card words it. Pod only; reads camera pixels, writes numbers only.

Per evaluation clip (the first --n-clips by ascending sha256 hex of the clip id that have a staged tele clip and at least 2 pilot frames
with a tele match), two pilot frames (the matched frames at the 1/3 and 2/3 position): the tele window T8 (8 deg, azimuth 0, elevation -2,
the card's T8) is built from the tele frame matched to the pilot frame (tele_encode.match_frames), sampled down to 128 px (2x2 mean), and
compared with the central 128 x 128 px of the W16 window (16 deg, 256 px, unchanged enc_common routines) of the wide frame of the same
pilot frame, built at the same direction (azimuth 0, elevation -2: the direction of arm D's W16 in card ADD99 and of T8), by the normalised
correlation (Pearson) of the grey values (0.299 R + 0.587 G + 0.114 B). The same comparison is made with four tele windows turned by 2
degrees (left = azimuth +2, right = azimuth -2, up = elevation +2, down = elevation -2). A frame passes when the correlation at the built
direction is higher than at each of the four. t2 passes when the share of passing frames is at least 0.9.
Note N1 of card ADD122 (lead, before any encode): the tele window of the pass rule is built at the direction the encode uses, the direction
(0, -2) of the WIDE camera's frame carried into the tele camera's own frame through both cameras' extrinsics (tele_encode.tele_dirs); the
W16 window stays at (0, -2) of the wide frame. Also reported, NOT part of the pass rule (information field 'own_frame_variant'): the same
share with both windows at (0, -2) of their own camera.
The wide clips come from the data kit's stage root (--wide-root, layout of stage_clips.py: r0/camera/<clip>.mp4 and ts/<clip>.timestamps
.parquet); --stage-wide runs the UNCHANGED stage_clips.py for chosen clips that are not staged there.
Prints one line:  t2 PASS|FAIL share <x> n <n>.   Exit 0 on PASS, 1 on FAIL.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.environ.get("TG_ADD99_KIT", ""), "/workspace/v8proof/cropread/kit/enc", os.path.join(HERE, "..", "..", "cropread", "enc")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import tele_encode as TE   # noqa: E402
from enc_common import Layout, decode_selected, frame_plan, read_ids, window_frame, window_grid, window_sample   # noqa: E402

THRESHOLD = 0.9
OFFSET_DEG = 2.0
T8_AZ, T8_EL = TELE_DIR = (TE.TELE_WINDOWS[0][1], TE.TELE_WINDOWS[0][2])
# (d_azimuth, d_elevation) of the four turned tele windows: left, right, up, down (left and up positive)
TURNS = ((+OFFSET_DEG, 0.0), (-OFFSET_DEG, 0.0), (0.0, +OFFSET_DEG), (0.0, -OFFSET_DEG))


def log(m):
    print(f"[t2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ pure pieces
def grey(img):
    """float [3,H,W] -> float64 [H,W]."""
    a = np.asarray(img, np.float64)
    return 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]


def ncc(a, b):
    """Pearson correlation of two equally shaped arrays; nan when either has no variance."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def down2(win):
    """[3,256,256] -> float [3,128,128] by 2x2 mean."""
    w = np.asarray(win, np.float64)
    return w.reshape(3, 128, 2, 128, 2).mean(axis=(2, 4))


def central128(win):
    return np.asarray(win, np.float64)[:, 64:192, 64:192]


def frame_passes(built, shifted):
    """The card's per-frame rule: the correlation at the built direction is higher than at each of the four turned windows (nan fails)."""
    return bool(np.isfinite(built) and all(np.isfinite(s) and built > s for s in shifted))


def decide(passes, thr=THRESHOLD):
    """-> (pass bool, share float, n int). No frame = FAIL."""
    n = len(passes)
    share = (sum(1 for p in passes if p) / n) if n else float("nan")
    return bool(n > 0 and share >= thr), share, n


def pick_frames(mask):
    """Two pilot frames among the matched ones: the matched frames at the 1/3 and 2/3 position of the matched list."""
    ok = np.nonzero(np.asarray(mask, bool))[0]
    if ok.size < 2:
        return []
    return sorted({int(ok[int(ok.size / 3.0)]), int(ok[min(ok.size - 1, int(2.0 * ok.size / 3.0))])})


def select_clips(eval_ids, n_clips, has_tele):
    """First n_clips of the evaluation list by ascending sha256 hex of the id that satisfy has_tele(clip)."""
    order = sorted(eval_ids, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    out = [c for c in order if has_tele(c)]
    return out[:n_clips]


# ------------------------------------------------------------------------------------------------ geometry
def cut(vid_u8, intr, frame, az, el):
    """[3,h,w] uint8 -> uint8 [3,256,256] window by the unchanged routines."""
    import torch
    h, w = int(vid_u8.shape[-2]), int(vid_u8.shape[-1])
    g, m = window_grid(intr, h, w, frame, az, el)
    return window_sample(vid_u8.unsqueeze(0).float() if torch.is_tensor(vid_u8) else torch.as_tensor(vid_u8).unsqueeze(0).float(), g, m)[0].numpy()


def frame_corrs(wide_u8, tele_u8, intr_w, intr_t, az=T8_AZ, el=T8_EL, wide_dir=None):
    """-> (corr at the built direction, [corr left, right, up, down]).  wide_dir = (az, el) of the wide window if it differs from the
    tele direction (extrinsics variant), else the same direction."""
    fw, ft = window_frame(16.0), window_frame(8.0)
    wd = wide_dir if wide_dir is not None else (az, el)
    ref = grey(central128(cut(wide_u8, intr_w, fw, wd[0], wd[1])))

    def tele_at(daz, dele):
        return grey(down2(cut(tele_u8, intr_t, ft, az + daz, el + dele)))
    return ncc(ref, tele_at(0.0, 0.0)), [ncc(ref, tele_at(da, de)) for da, de in TURNS]


def point_in_other_cam(R_src_to_veh, R_dst_to_veh, az_left_deg, el_up_deg):
    """Direction (azimuth left+, elevation up+) of the centre ray of a window pointed at (az, el) in the SOURCE camera, expressed in the
    DESTINATION camera's own (az, el). Both rotations are cam -> vehicle (FrontWideExtrinsics.rotation_cam_to_vehicle); camera axes
    +x right, +y down, +z boresight; centre ray (-cos(el) sin(az), -sin(el), cos(el) cos(az))."""
    az, el = math.radians(az_left_deg), math.radians(el_up_deg)
    d = np.array([-math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az)])
    e = np.asarray(R_dst_to_veh).T @ (np.asarray(R_src_to_veh) @ d)
    return math.degrees(math.atan2(-e[0], e[2])), math.degrees(math.atan2(-e[1], math.hypot(e[0], e[2])))


def _rot(csv_path, clip):
    import pandas as pd
    from tanitad.data.physicalai import FrontWideExtrinsics
    df = pd.read_csv(csv_path, dtype={"clip_id": str})
    r = df[df["clip_id"] == clip].iloc[0]
    return FrontWideExtrinsics(float(r.qx), float(r.qy), float(r.qz), float(r.qw), float(r.x), float(r.y), float(r.z)).rotation_cam_to_vehicle()


def _grab(mp4, idxs, threads=1):
    got = {}
    for vi, fr in decode_selected(mp4, sorted(set(idxs)), threads):
        got[int(vi)] = fr
    return got


def t2_clip(clip, Lw, Lt, wide_threads=1):
    """-> list of per-frame dicts for one clip (two frames)."""
    from tanitad.data.physicalai import intrinsics_for_clip
    plan = frame_plan(Lw.ts(clip))
    idx, dt, mask = TE.match_frames(TE.pilot_times(plan), TE.read_ts(Lt.ts(clip)))
    js = pick_frames(mask)
    if not js:
        return []
    intr_w, intr_t = intrinsics_for_clip(clip, Lw.root), intrinsics_for_clip(clip, Lt.root)
    assert intr_w.per_clip and intr_t.per_clip, "per-clip intrinsics missing"
    wid = _grab(Lw.mp4(clip), [int(plan["cache_idx"][j]) for j in js], wide_threads)
    tid = _grab(Lt.mp4(clip), [int(idx[j]) for j in js], wide_threads)
    # note N1 of card ADD122: the pass rule reads the tele window at the direction the encode uses (the direction (0, -2) of the wide frame
    # carried into the tele frame by both extrinsics) against the W16 window at (0, -2); a missing extrinsics row raises and the clip is skipped
    tele_dir = TE.tele_dirs(clip, Lw, Lt)[0]
    rows = []
    for j in js:
        w_img, t_img = wid[int(plan["cache_idx"][j])], tid[int(idx[j])]
        built, sh = frame_corrs(w_img, t_img, intr_w, intr_t, az=tele_dir[0], el=tele_dir[1], wide_dir=(T8_AZ, T8_EL))
        row = {"clip": clip[:8], "j": int(j), "dt_us": float(dt[j]), "tele_dir": [float(tele_dir[0]), float(tele_dir[1])], "corr_built": built, "corr_turned": sh,
               "pass": frame_passes(built, sh)}
        b2, s2 = frame_corrs(w_img, t_img, intr_w, intr_t)                     # information only: both windows at (0, -2) of their own camera
        row.update(own_corr_built=b2, own_pass=frame_passes(b2, s2))
        rows.append(row)
    return rows


def run(eval_ids, Lw, Lt, out_json, n_clips=100, workers=4, stage_wide=None):
    def has_tele(c):
        return os.path.exists(Lt.mp4(c)) and os.path.exists(Lt.ts(c))
    chosen = select_clips(eval_ids, n_clips, has_tele)
    if stage_wide is not None:
        miss = [c for c in chosen if not (os.path.exists(Lw.mp4(c)) and os.path.exists(Lw.ts(c)))]
        if miss:
            stage_wide(miss)
    frames, skipped = [], []

    def one(c):
        try:
            return c, t2_clip(c, Lw, Lt), None
        except Exception as e:                                       # noqa: BLE001
            return c, [], f"{type(e).__name__}: {str(e)[:160]}"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for c, rows, err in ex.map(one, chosen):
            if err or not rows:
                skipped.append({"clip": c[:8], "reason": err or "fewer than 2 matched pilot frames"})
            frames.extend(rows)
    ok, share, n = decide([r["pass"] for r in frames])
    ext = [r["own_pass"] for r in frames if "own_pass" in r]
    rec = {"control": "t2", "rule": "share of frames with corr(T8 down to 128, central 128 of W16) higher at the built direction than at "
                                      "each of the four tele windows turned by 2 deg; pass when share >= 0.9",
           "threshold": THRESHOLD, "pass": ok, "share": share, "n_frames": n, "n_clips_used": len({r["clip"] for r in frames}),
           "n_clips_selected": len(chosen), "skipped": skipped,
           "own_frame_variant": {"info_only": True, "share": (sum(ext) / len(ext)) if ext else None, "n_frames": len(ext)},
           "frames": frames}
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"t2 {'PASS' if ok else 'FAIL'} share {share:.4f} n {n}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--wide-root", required=True)
    ap.add_argument("--tele-root", required=True)
    ap.add_argument("--out", required=True, help="JSON record")
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage-wide", action="store_true", help="run the unchanged stage_clips.py for chosen clips not staged in --wide-root")
    ap.add_argument("--hf-dir", default="/workspace/v8proof/cropread/hfdl")
    a = ap.parse_args(argv)

    def stage_wide(miss):
        import stage_clips
        lst = a.out + ".wide_list.txt"
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(miss) + "\n")
        rc = stage_clips.main(["--list", lst, "--root", a.wide_root, "--hf-dir", a.hf_dir, "--kind", "eval", "--workers", "4"])
        log(f"stage_clips rc {rc} for {len(miss)} clips")
    rec = run(read_ids(a.eval_list), Layout(a.wide_root), Layout(a.tele_root), a.out, a.n_clips, a.workers,
              stage_wide if a.stage_wide else None)
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
