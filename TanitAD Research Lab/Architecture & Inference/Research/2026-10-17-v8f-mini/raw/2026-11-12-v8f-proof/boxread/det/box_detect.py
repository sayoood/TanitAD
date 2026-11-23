#!/usr/bin/env python3
"""BOXREAD-T0 detector pass (card ADD127 sections 2 and 6): one camera per call, one npz per clip. Pod only for real weights (GPU).

For every frame of the token cache (cache index j, the frames the data kit of card ADD99 selects), the frame is decoded from the staged
video and a FIXED window is rendered from the RAW decoded frame by the unchanged window routines of the ADD99 kit (enc_common.window_frame,
window_grid, window_sample), then a torchvision Faster R-CNN (fasterrcnn_resnet50_fpn_v2, COCO weights read from a LOCAL file, never
downloaded) runs on it. Detections with score >= --score-keep and a vehicle label (COCO ids car 3, motorcycle 4, bus 6, truck 8) are kept.
The card's own bar (score >= 0.5) is applied by the reader, which can therefore also see 0.3..0.5.

  wide : the wide camera, window --win-size-deg at --win-px, centre (--az-left-deg, --el-up-deg) in the camera's own rectified frame
         (elevation 0 = the optical axis; the data kit builds no mount pitch). Frame j = video frame plan["cache_idx"][j].
  tele : the tele camera, same window routine with the tele intrinsics. Cache frame j is matched to the tele frame nearest in time to the
         time of the wide video frame decoded for j (tele_encode.pilot_times / match_frames, at most 17000 us); no match = frame_ok False.

Output <out-dir>/<camera>/<clip>.npz (never overwritten) with exactly: n_frames, frame_ok, det_frame, det_box, det_score, det_label, win,
camera, weights_sha256; and <out-dir>/<camera>/BOXDET_RECORD.json (+ BOXDET_RUNS.jsonl, one line per run). Nothing is deleted or moved;
a clip file is written under a .part name in its own folder and renamed once. No frame pixel is written anywhere.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("BOXDET_ENC_KIT", ""), "/workspace/v8proof/cropread/kit/enc",
           os.path.join(HERE, "..", "..", "cropread", "enc")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from enc_common import (Layout, decode_selected, frame_plan, read_ids, sha256_file,   # noqa: E402
                        window_frame, window_grid, window_sample)

VEHICLE_LABELS = (3, 4, 6, 8)             # COCO ids of the torchvision weights: car, motorcycle, bus, truck
CAMERAS = ("wide", "tele")
RENDER_BATCH = 8
MAX_THREADS = 4
NPZ_KEYS = ("n_frames", "frame_ok", "frame_run","det_frame", "det_box", "det_score", "det_label", "win", "camera", "weights_sha256")


def log(m):
    print(f"[boxdet {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ tele helpers (lazy import)
_TELE = None


def _tele():
    global _TELE
    if _TELE is None:
        for p in (os.environ.get("BOXDET_TELE_KIT", ""), "/workspace/v8proof/telegap/kit/enc",
                  os.path.join(HERE, "..", "..", "telegap", "enc")):
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        import tele_encode
        _TELE = tele_encode
    return _TELE


def tele_times(ts_path):
    """Microsecond clock of the tele timestamps table (row index = tele video frame index)."""
    return _tele().read_ts(ts_path)


def pilot_times(plan):
    return _tele().pilot_times(plan)


def match_frames(t_wide, t_tele):
    return _tele().match_frames(t_wide, t_tele)


def get_intr(clip, root):
    """Per-clip camera intrinsics from a stage root (the stack's own reader); a corpus-median fallback is refused."""
    from tanitad.data.physicalai import intrinsics_for_clip
    intr = intrinsics_for_clip(clip, root)
    assert intr.per_clip, f"{clip[:8]}: corpus-median intrinsics refused (per-clip table missing)"
    return intr


# ------------------------------------------------------------------------------------------------ weights and detector
def verify_weights(path, prefix):
    """-> full sha256 of the local weights file. Refuses (ValueError) a missing file or a digest that does not begin with `prefix`."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"weights file missing: {path!r} (this tool never downloads)")
    full = sha256_file(path)
    if not prefix or not full.startswith(prefix.lower()):
        raise ValueError(f"weights sha256 begins {full[:8]}, expected prefix {prefix!r}: refused")
    return full


def build_detector(path, device):
    """fasterrcnn_resnet50_fpn_v2 with NO weight enum (so nothing can be downloaded), state dict from the local file. Eval mode."""
    import torch
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def detect_batch(model, imgs_u8, device, score_keep, labels=VEHICLE_LABELS):
    """imgs_u8 uint8 [B,3,H,W] (RGB, the order window_sample returns) -> list of (box [k,4] f32, score [k] f32, label [k] i16) per image."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).to(device).float() / 255.0     # float 0..1, RGB: what torchvision detectors expect
    with torch.inference_mode():
        outs = model([x[i] for i in range(x.shape[0])])
    res = []
    for o in outs:
        b = o["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        s = o["scores"].detach().float().cpu().numpy().reshape(-1)
        lab = o["labels"].detach().cpu().numpy().reshape(-1).astype(np.int64)
        keep = (s >= float(score_keep)) & np.isin(lab, np.asarray(labels))
        res.append((b[keep].astype(np.float32), s[keep].astype(np.float32), lab[keep].astype(np.int16)))
    return res


# ------------------------------------------------------------------------------------------------ windows (CPU side)
def clip_paths(camera, clip, Lw, Lt):
    """-> (mp4, timestamps file of the video, stage root of the intrinsics)."""
    if camera == "wide":
        return Lw.mp4(clip), Lw.ts(clip), Lw.root
    return Lt.mp4(clip), Lt.ts(clip), Lt.root


def load_has_lead(box_dir, clip, n):
    """has_lead bool [n] of a clip's box file; a missing file or a length different from the cache frame count is refused."""
    path = os.path.join(box_dir, f"{clip}.npz") if box_dir else ""
    if not path or not os.path.exists(path):
        raise ValueError(f"{clip[:8]}: no box file in {box_dir!r}")
    has = np.asarray(np.load(path)["has_lead"]).astype(bool)
    if has.shape != (n,):
        raise ValueError(f"{clip[:8]}: box file has {has.shape[0] if has.ndim else 0} frames, the clip has {n} cache frames")
    return has


def make_windows(camera, clip, Lw, Lt, win, torch_threads=2, box_dir="", only_lead=False):
    """-> (uint8 [n,3,px,px] windows (zeros where not run), frame_run bool [n], info dict (with frame_ok, n_selected)). Pure CPU.
    win = (size_deg, px, az_left_deg, el_up_deg). frame_run = selected, matched (tele) and decoded: where the detector will run."""
    import torch
    t0 = time.time()
    torch.set_num_threads(torch_threads)
    size_deg, px, az, el = win
    plan = frame_plan(Lw.ts(clip))
    n = int(plan["n_cache"])
    cidx = np.asarray(plan["cache_idx"], np.int64)
    assert cidx.shape[0] == n, (cidx.shape, n)
    selected = load_has_lead(box_dir, clip, n) if only_lead else np.ones(n, bool)
    if camera == "wide":
        vidx, wanted = cidx, np.ones(n, bool)
        mp4, root = Lw.mp4(clip), Lw.root
        n_tele_unmatched = 0
    else:
        idx, _dt, mask = match_frames(pilot_times(plan), tele_times(Lt.ts(clip)))
        vidx, wanted = np.asarray(idx, np.int64), np.asarray(mask, bool)
        mp4, root = Lt.mp4(clip), Lt.root
        n_tele_unmatched = int((~wanted).sum())
    base_ok = wanted.copy()                                            # decodable / matched, before the selection
    wanted = wanted & selected
    intr = get_intr(clip, root)
    frame = window_frame(float(size_deg), int(px))
    by_idx = collections.defaultdict(list)
    for j in np.nonzero(wanted)[0]:
        by_idx[int(vidx[j])].append(int(j))
    out = np.zeros((n, 3, int(px), int(px)), np.uint8)
    ok = np.zeros(n, bool)
    grid = None
    pend = []
    err = ""

    def flush():
        nonlocal grid, pend
        if not pend:
            return
        js = [j for j, _ in pend]
        vid = torch.stack([f for _, f in pend]).float()
        if grid is None:
            grid = window_grid(intr, int(vid.shape[-2]), int(vid.shape[-1]), frame, float(az), float(el))
        g, m = grid
        B = len(js)
        out[js] = window_sample(vid, g.expand(B, -1, -1, -1), m.expand(B, -1, -1)).numpy()
        ok[js] = True
        pend = []

    if by_idx:
        try:
            for vi, fr in decode_selected(mp4, sorted(by_idx)):
                for j in by_idx[int(vi)]:
                    pend.append((j, fr))
                    if len(pend) >= RENDER_BATCH:
                        flush()
            flush()
        except Exception as e:                                          # noqa: BLE001  (a corrupt video: rows not reached stay not ok)
            err = f"{type(e).__name__}: {str(e)[:160]}"
            try:
                flush()
            except Exception as e2:                                     # noqa: BLE001
                err += f" | flush {type(e2).__name__}: {str(e2)[:80]}"
    # frame_ok False: could not be decoded, or (tele) no matched frame. A frame the selection left out was not tried: not a failure.
    frame_ok = base_ok & (ok | ~selected)
    info = {"n": n, "n_wanted": int(wanted.sum()), "n_selected": int(selected.sum()), "frame_ok": frame_ok,
            "n_tele_unmatched": n_tele_unmatched, "decode_error": err,
            "prep_s": time.time() - t0}
    return out, ok, info


# ------------------------------------------------------------------------------------------------ one clip
def detect_clip(model, wins, ok, device, batch, score_keep):
    """-> (det dict of arrays, detector seconds, frames run, seconds and frames of the first call (warm-up))."""
    fr, bx, sc, lb = [], [], [], []
    secs, nfr, first_s, first_n = 0.0, 0, 0.0, 0
    js_all = np.nonzero(ok)[0]
    for s in range(0, js_all.size, batch):
        js = js_all[s:s + batch]
        sync(device)
        t = time.time()
        res = detect_batch(model, wins[js], device, score_keep)
        sync(device)
        dt = time.time() - t
        if nfr == 0 and secs == 0.0 and first_n == 0:
            first_s, first_n = dt, int(js.size)
        secs += dt
        nfr += int(js.size)
        for j, (b, sco, lab) in zip(js, res):
            if sco.shape[0]:
                fr.append(np.full(sco.shape[0], int(j), np.int32))
                bx.append(b)
                sc.append(sco)
                lb.append(lab)
    det = {"det_frame": np.concatenate(fr) if fr else np.zeros(0, np.int32),
           "det_box": np.concatenate(bx).astype(np.float32) if bx else np.zeros((0, 4), np.float32),
           "det_score": np.concatenate(sc).astype(np.float32) if sc else np.zeros(0, np.float32),
           "det_label": np.concatenate(lb).astype(np.int16) if lb else np.zeros(0, np.int16)}
    return det, secs, nfr, first_s, first_n


def clip_file(out_dir, camera, clip):
    return os.path.join(out_dir, camera, f"{clip}.npz")


def write_clip(path, n, ok, det, win, camera, wsha, run=None):
    tmp = path + ".part"
    run = ok if run is None else run
    with open(tmp, "wb") as fh:
        np.savez(fh, n_frames=np.int64(n), frame_ok=ok.astype(bool), frame_run=run.astype(bool), det_frame=det["det_frame"].astype(np.int32),
                 det_box=det["det_box"].astype(np.float32), det_score=det["det_score"].astype(np.float32),
                 det_label=det["det_label"].astype(np.int16), win=np.asarray(win, np.float64), camera=np.str_(camera),
                 weights_sha256=np.str_(wsha))
    if os.path.exists(path):                                           # appeared meanwhile: never overwrite, leave the .part
        return False
    os.replace(tmp, path)
    return True


# ------------------------------------------------------------------------------------------------ run
def tool_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def lib_versions():
    out = {}
    for m in ("torch", "torchvision"):
        try:
            out[m] = __import__(m).__version__
        except Exception:                                              # noqa: BLE001
            out[m] = "not importable"
    return out


def select_clips(camera, clips, Lw, Lt, limit):
    """-> (todo-candidates with video [clip], no_video [(clip, reason)]). `limit` counts clips that have a video."""
    have, no_video = [], []
    for c in clips:
        mp4, ts, _root = clip_paths(camera, c, Lw, Lt)
        if not (os.path.exists(mp4) and os.path.exists(ts)):
            no_video.append((c, f"no staged {camera} video or timestamps"))
        elif camera == "tele" and not os.path.exists(Lw.ts(c)):
            no_video.append((c, "no wide timestamps (frame plan) for the tele match"))
        else:
            have.append(c)
    if limit:
        have = have[:limit]
    return have, no_video


def run(a, detector=None):
    """Returns rc: 0 ok, 5 some clip failed, 2 refused (weights, device, arguments)."""
    if a.camera not in CAMERAS:
        log(f"REFUSED: camera {a.camera!r}")
        return 2
    if a.workers > MAX_THREADS or a.workers < 1:
        log(f"REFUSED: --workers {a.workers} (1..{MAX_THREADS})")
        return 2
    if a.camera == "tele" and not a.tele_root:
        log("REFUSED: --tele-root is required for the tele camera")
        return 2
    try:
        wsha = verify_weights(a.weights, a.weights_sha_prefix)
    except ValueError as e:
        log(f"REFUSED: {e}")
        return 2
    t_wall = time.time()
    camdir = os.path.join(a.out_dir, a.camera)
    os.makedirs(camdir, exist_ok=True)
    clips = []
    for lf in a.list:
        for c in read_ids(lf):
            if c not in clips:
                clips.append(c)
    Lw = Layout(a.root)
    Lt = Layout(a.tele_root) if a.tele_root else None
    win = (float(a.win_size_deg), int(a.win_px), float(a.az_left_deg), float(a.el_up_deg))
    have, no_video = select_clips(a.camera, clips, Lw, Lt, a.limit_clips)
    existing = [c for c in have if os.path.exists(clip_file(a.out_dir, a.camera, c))]
    todo = [c for c in have if c not in set(existing)]
    if a.only_lead_frames:
        if not a.box_dir:
            log("REFUSED: --only-lead-frames needs --box-dir")
            return 2
        miss = [c for c in todo if not os.path.exists(os.path.join(a.box_dir, f"{c}.npz"))]
        if miss:
            log(f"REFUSED: {len(miss)} clips have no box file in --box-dir (first {miss[0][:8]})")
            return 2
    log(f"{a.camera}:{len(clips)} clips in the lists, {len(no_video)} without video, {len(existing)} already done, {len(todo)} to do")
    if todo and detector is None:
        if str(a.device).startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                log("REFUSED: no CUDA: refusing a silent CPU fallback")
                return 2
        detector = build_detector(a.weights, a.device)
        log("detector ready")
    tot = {"frames": 0, "frames_not_ok": 0, "detections": 0, "det_s": 0.0, "frames_run": 0, "first_s": 0.0, "first_n": 0,
           "prep_s": 0.0, "clips_done": 0, "decode_errors": 0, "tele_unmatched": 0, "selected": 0}
    fails = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        q = collections.deque()
        it = iter(todo)

        def submit_next():
            c = next(it, None)
            if c is not None:
                q.append((c, ex.submit(make_windows, a.camera, c, Lw, Lt, win, a.torch_threads,
                                                        a.box_dir, bool(a.only_lead_frames))))
        for _ in range(a.workers + 1):
            submit_next()
        while q:
            c, fut = q.popleft()
            submit_next()
            try:
                wins, ok, info = fut.result()
                det, secs, nfr, fs, fn = detect_clip(detector, wins, ok, a.device, a.batch, a.score_keep)
                path = clip_file(a.out_dir, a.camera, c)
                if write_clip(path, info["n"], info["frame_ok"], det, win, a.camera, wsha, ok):
                    tot["clips_done"] += 1
                    tot["frames"] += info["n"]
                    tot["selected"] += info["n_selected"]
                    tot["frames_not_ok"] += int((~info["frame_ok"]).sum())
                    tot["detections"] += int(det["det_frame"].shape[0])
                    tot["det_s"] += secs
                    tot["frames_run"] += nfr
                    tot["prep_s"] += info["prep_s"]
                    tot["tele_unmatched"] += info["n_tele_unmatched"]
                    if tot["first_n"] == 0 and fn:
                        tot["first_s"], tot["first_n"] = fs, fn
                    if info["decode_error"]:
                        tot["decode_errors"] += 1
                        fails.append((c, "decode: " + info["decode_error"]))
                    if tot["clips_done"] <= 3 or tot["clips_done"] % 25 == 0:
                        log(f"{tot['clips_done']}/{len(todo)} {c[:8]} n={info['n']} ok={int(ok.sum())} det={int(det['det_frame'].shape[0])} "
                            f"detector={secs:.1f}s prep={info['prep_s']:.1f}s")
            except Exception as e:                                       # noqa: BLE001
                fails.append((c, f"{type(e).__name__}: {str(e)[:200]}"))
                log(f"FAIL {c[:8]} {type(e).__name__}: {str(e)[:200]}")
    gpu_s = tot["det_s"] if str(a.device).startswith("cuda") else 0.0
    rec = {"arguments": {k: (v if not isinstance(v, list) else list(v)) for k, v in vars(a).items()},
           "camera": a.camera, "win": list(win),
           "clips_in_lists": len(clips), "clips_done": tot["clips_done"], "clips_skipped_existing": len(existing),
           "clips_without_video": len(no_video), "clips_without_video_ids": [c[:8] for c, _ in no_video],
           "frames": tot["frames"], "frames_not_ok": tot["frames_not_ok"], "frames_tele_unmatched": tot["tele_unmatched"],
           "frames_selected": tot["selected"], "frames_run": tot["frames_run"], "box_dir": a.box_dir,
           "only_lead_frames": bool(a.only_lead_frames),
           "frames_detected": tot["frames_run"], "detections_kept": tot["detections"],
           "decode_errors": tot["decode_errors"], "failures": [(c[:8], m) for c, m in fails],
           "detector_seconds": round(tot["det_s"], 3), "gpu_seconds": round(gpu_s, 3),
           "detector_first_call_seconds": round(tot["first_s"], 3), "detector_first_call_frames": tot["first_n"],
           "prep_seconds_sum": round(tot["prep_s"], 3), "wall_seconds": round(time.time() - t_wall, 3),
           "tool_sha256": tool_sha(), "weights_sha256": wsha, "versions": lib_versions(),
           "score_keep": float(a.score_keep), "vehicle_labels": list(VEHICLE_LABELS), "time": time.strftime("%F %T")}
    with open(os.path.join(camdir, "BOXDET_RECORD.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    with open(os.path.join(camdir, "BOXDET_RUNS.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"done: {tot['clips_done']} clips, {tot['frames']} frames, {tot['detections']} detections, {len(fails)} failed, "
        f"{len(no_video)} without video, detector {tot['det_s']:.0f}s")
    print("ZZ BOXDET DONE ZZ" if not fails else f"ZZ BOXDET FAIL ZZ n={len(fails)}", flush=True)
    return 0 if not fails else 5


# ------------------------------------------------------------------------------------------------ cost projection (stop rule, card 6)
def count_lead_frames(box_dir, lists):
    """-> (frames with has_lead summed over the box files of all listed clips, clips without a box file). CPU only."""
    total, missing, seen = 0, 0, set()
    for lf in lists:
        for c in read_ids(lf):
            if c in seen:
                continue
            seen.add(c)
            path = os.path.join(box_dir, f"{c}.npz")
            if not os.path.exists(path):
                missing += 1
                continue
            total += int(np.asarray(np.load(path)["has_lead"]).astype(bool).sum())
    return total, missing


def projected_hours(camdir, n_clips=None, n_frames=None):
    """GPU hours of a full pass, from the last run of `camdir` that ran the detector: seconds per frame (first call left out when more
    frames exist) x frames of the full pass. `n_frames` = the SELECTED frames of the full lists (count_lead_frames); without it, frames
    per clip of the run x n_clips. Returns None when no run of the folder ran the detector."""
    p = os.path.join(camdir, "BOXDET_RUNS.jsonl")
    if not os.path.exists(p):
        return None
    runs = [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]
    runs = [r for r in runs if r.get("frames_detected", 0) > 0 and r.get("clips_done", 0) > 0]
    if not runs:
        return None
    r = runs[-1]
    s, n = float(r["detector_seconds"]), int(r["frames_detected"])
    if n > r.get("detector_first_call_frames", 0):
        s, n = s - float(r["detector_first_call_seconds"]), n - int(r["detector_first_call_frames"])
    spf = s / n
    if n_frames is not None:
        return spf * int(n_frames) / 3600.0
    fpc = r["frames_detected"] / r["clips_done"]
    return spf * fpc * int(n_clips) / 3600.0


def main(argv=None, detector=None):
    """detector: tests only (a stand-in with the torchvision call contract); the weights file is still checked by sha256."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--project" in argv:
        pp = argparse.ArgumentParser()
        pp.add_argument("--project", required=True, help="a <out-dir>/<camera> folder holding BOXDET_RUNS.jsonl")
        pp.add_argument("--project-n-clips", type=int, default=None)
        pp.add_argument("--project-frames", type=int, default=None, help="selected frames of the full lists (--count-lead-frames)")
        pa = pp.parse_args(argv)
        if pa.project_frames is None and pa.project_n_clips is None:
            pp.error("one of --project-frames, --project-n-clips is required")
        h = projected_hours(pa.project, pa.project_n_clips, pa.project_frames)
        print(f"PROJECT_H {h:.4f}" if h is not None else "PROJECT_H nan")
        return 0 if h is not None else 3
    if "--count-lead-frames" in argv:
        cp = argparse.ArgumentParser()
        cp.add_argument("--count-lead-frames", action="store_true")
        cp.add_argument("--box-dir", required=True)
        cp.add_argument("--list", action="append", required=True)
        ca = cp.parse_args(argv)
        tot, miss = count_lead_frames(ca.box_dir, ca.list)
        print(f"LEAD_FRAMES {tot} MISSING_BOX_FILES {miss}")
        return 0 if miss == 0 else 3
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--box-dir", default="", help="folder of <clip>.npz box files with has_lead bool [n_frames] (for --only-lead-frames)")
    ap.add_argument("--only-lead-frames", action="store_true", help="run the detector only on cache frames where has_lead is true")
    ap.add_argument("--camera", required=True, choices=CAMERAS)
    ap.add_argument("--list", action="append", required=True, help="file of clip ids (repeatable)")
    ap.add_argument("--root", "--wide-root", dest="root", required=True,
                    help="stage root of the WIDE clips (stage_clips.py --root): its video for wide, its timestamps (frame plan) for both")
    ap.add_argument("--tele-root", default="", help="stage root written by tele_stage.py (tele only)")
    ap.add_argument("--weights", required=True, help="local .pth of fasterrcnn_resnet50_fpn_v2 COCO weights")
    ap.add_argument("--weights-sha-prefix", required=True, help="the file's sha256 must begin with this (card: dd69338a)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--win-size-deg", type=float, required=True, help="card: wide 32, tele 16")
    ap.add_argument("--win-px", type=int, required=True, help="card: wide 768, tele 1024")
    ap.add_argument("--az-left-deg", type=float, default=0.0)
    ap.add_argument("--el-up-deg", type=float, default=0.0, help="card: wide -6 (6 down), tele 0")
    ap.add_argument("--score-keep", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-clips", type=int, default=0, help="smoke: only the first N clips that have a video")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2, help="clip preparation threads (decode + render), at most 4")
    ap.add_argument("--torch-threads", type=int, default=2)
    return run(ap.parse_args(argv), detector)


if __name__ == "__main__":
    sys.exit(main())
