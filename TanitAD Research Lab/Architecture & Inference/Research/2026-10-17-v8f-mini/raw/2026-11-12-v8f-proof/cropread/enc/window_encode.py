#!/usr/bin/env python3
"""Windows and tokens for a clip list (card sections 3 and 4). Pod only (GPU for the encoder).

Per clip: the frames of the CACHE's own frame times (enc_common.frame_plan = the builder's arithmetic; cache frame j = episode frame 2j),
decoded with the builder's PyAV settings from the staged mp4, four 256x256 windows per frame cut from the RAW decoded frame by the
builder's own projection chain with a centre direction and an angular size (enc_common.window_grid):
  arm D  W32 at (az 0, el -6 deg), W16 at (az 0, el -2 deg), every frame;
  arm B  W32 and W16 both centred on the direction of the centre of the lead's box (box/<clip>.npz from lead_box.py); frames with no lead
         in the label use arm D's windows (their tokens are then arm D's tokens, copied: same pixels, same encoder).
Tokens: the cache builder's encoder path (enc_common.Encoder) per window: token 0 = class token, tokens 1..64 = the 16x16 patch tokens
mean-pooled 2x2 to 8x8 row-major. File tok/<arm>/<clip>.npy float16 [n_frames, 2, 65, 1024], window order W32 then W16; frame i of the
file is frame i of the join. tok/<arm>/ledger.jsonl is the resume record; tok/<arm>/MANIFEST.json has n_frames and sha256 per clip.
No frame image is kept unless --keep-windows (smoke): then win/<clip>.npy uint8 [n,4,3,256,256] (order D32 D16 B32 B16), PRIVATE.
Nothing is deleted. Output files are written whole to a temporary name in the same folder and renamed once (atomic).
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enc_common import (ARM_D_WINDOWS, Layout, SIZES_DEG, WIN_PX, decode_selected, frame_plan, read_ids, sha256_file,   # noqa: E402
                        window_frame, window_grid, window_sample, Encoder)

BATCH = 8


def log(m):
    print(f"[enc {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_box_dirs(box_dir, clip, n):
    """-> (has bool[n], az float[n], el float[n]) of the box centre; all-False when there is no box file."""
    p = os.path.join(box_dir, f"{clip}.npz") if box_dir else None
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"no box file for {clip[:8]} in {box_dir}")
    z = np.load(p)
    has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
    if len(has) != n:
        raise ValueError(f"box file has {len(has)} frames, plan says {n}")
    return has, np.where(has, z["ctr_az"], 0.0), np.where(has, z["ctr_el"], 0.0)


def make_windows(clip, L, box_dir, torch_threads=2):
    """-> (uint8 [n,4,3,256,256] in order D32, D16, B32, B16; has_lead bool[n]; plan; secs). Pure CPU."""
    import torch
    from tanitad.data.physicalai import intrinsics_for_clip
    t0 = time.time()
    torch.set_num_threads(torch_threads)
    plan = frame_plan(L.ts(clip))
    n, cidx = plan["n_cache"], plan["cache_idx"]
    has, baz, bel = load_box_dirs(box_dir, clip, n) if box_dir else (np.zeros(n, bool), np.zeros(n), np.zeros(n))
    intr = intrinsics_for_clip(clip, L.root)
    assert intr.per_clip, f"{clip}: corpus-median intrinsics refused (per-clip table missing)"
    frames = [window_frame(s) for s in SIZES_DEG]
    out = np.zeros((n, 4, 3, WIN_PX, WIN_PX), np.uint8)
    by_idx = collections.defaultdict(list)
    for j, vi in enumerate(cidx):
        by_idx[int(vi)].append(j)
    need = sorted(by_idx)
    gridsD = None
    pend = []                                                     # [(j, frame uint8 [3,H,W])]

    def flush():
        nonlocal gridsD, pend
        if not pend:
            return
        js = [j for j, _ in pend]
        vid = torch.stack([f for _, f in pend]).float()
        h, w = int(vid.shape[-2]), int(vid.shape[-1])
        B = len(js)
        if gridsD is None:
            gridsD = [window_grid(intr, h, w, frames[i], ARM_D_WINDOWS[i][1], ARM_D_WINDOWS[i][2]) for i in range(2)]
        for i in range(2):
            g, m = gridsD[i]
            out[js, i] = window_sample(vid, g.expand(B, -1, -1, -1), m.expand(B, -1, -1)).numpy()
            az = np.where(has[js], baz[js], ARM_D_WINDOWS[i][1])
            el = np.where(has[js], bel[js], ARM_D_WINDOWS[i][2])
            gb, mb = window_grid(intr, h, w, frames[i], az, el)
            out[js, 2 + i] = window_sample(vid, gb, mb).numpy()
        pend = []

    seen = 0
    for vi, fr in decode_selected(L.mp4(clip), need):
        for j in by_idx[vi]:
            pend.append((j, fr))
            seen += 1
            if len(pend) >= BATCH:
                flush()
    flush()
    if seen != n:
        raise RuntimeError(f"{clip}: decoded {seen} of {n} wanted frames (video shorter than the timestamp table?)")
    return out, has, plan, time.time() - t0


def write_npy(path, arr):
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def encode_clip_tokens(enc, wins, has):
    """wins uint8 [n,4,3,256,256] -> (tokD, tokB) float16 [n,2,65,1024] each."""
    import torch
    n = wins.shape[0]
    D = torch.from_numpy(wins[:, 0:2].reshape(n * 2, 3, WIN_PX, WIN_PX))
    tD = enc.window_tokens(D).reshape(n, 2, 65, -1)
    tB = tD.clone()
    idx = np.nonzero(has)[0]
    if idx.size:
        Bw = torch.from_numpy(wins[idx][:, 2:4].reshape(idx.size * 2, 3, WIN_PX, WIN_PX))
        tB[torch.from_numpy(idx)] = enc.window_tokens(Bw).reshape(idx.size, 2, 65, -1)
    return tD.to(torch.float16).numpy(), tB.to(torch.float16).numpy()


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


def write_manifest(out, arm):
    led = read_ledger(os.path.join(out, "tok", arm, "ledger.jsonl"))
    man = {"arm": arm, "layout": "float16 [n_frames, 2, 65, 1024]; window order W32 then W16; token 0 = class token, 1..64 = 8x8 pooled patch tokens row-major",
           "n_clips": len(led), "clips": {c: {"n_frames": r["n_frames"], "sha256": r["sha256"], "bytes": r["bytes"]} for c, r in sorted(led.items())}}
    with open(os.path.join(out, "tok", arm, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True)
    ap.add_argument("--root", required=True, help="stage root (stage_clips.py --root)")
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--enc-bs", type=int, default=8)
    ap.add_argument("--keep-windows", action="store_true", help="smoke only: keep win/<clip>.npy (private camera pixels)")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    a = ap.parse_args(argv)
    import shutil
    import torch
    L = Layout(a.root)
    clips = read_ids(a.list)
    for arm in ("D", "B"):
        os.makedirs(os.path.join(a.out, "tok", arm), exist_ok=True)
    if a.keep_windows:
        os.makedirs(os.path.join(a.out, "win"), exist_ok=True)
    led = {arm: read_ledger(os.path.join(a.out, "tok", arm, "ledger.jsonl")) for arm in ("D", "B")}

    def done(c):
        for arm in ("D", "B"):
            r = led[arm].get(c)
            p = os.path.join(a.out, "tok", arm, f"{c}.npy")
            if not r or not os.path.exists(p) or os.path.getsize(p) != r["bytes"]:
                return False
        return True

    todo = [c for c in clips if not done(c)]
    log(f"{len(clips)} clips, {len(clips) - len(todo)} done, {len(todo)} to do; workers {a.workers}")
    assert torch.cuda.is_available(), "no CUDA: refusing a silent CPU fallback"
    enc = Encoder("cuda", a.enc_bs)
    log("encoder ready")
    lfs = {arm: open(os.path.join(a.out, "tok", arm, "ledger.jsonl"), "a", buffering=1, encoding="utf-8") for arm in ("D", "B")}
    fails = []
    n_done = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        q = collections.deque()
        it = iter(todo)

        def submit_next():
            c = next(it, None)
            if c is not None:
                q.append((c, ex.submit(make_windows, c, L, a.box_dir, a.torch_threads)))
        for _ in range(a.workers + 1):
            submit_next()
        while q:
            c, fut = q.popleft()
            submit_next()
            try:
                if shutil.disk_usage(a.out).free / 1e9 < a.min_free_gb:
                    log("STOP: disk free below the floor")
                    return 7
                wins, has, plan, t_prep = fut.result()
                te = time.time()
                tD, tB = encode_clip_tokens(enc, wins, has)
                t_enc = time.time() - te
                tw = time.time()
                for arm, arr in (("D", tD), ("B", tB)):
                    p = os.path.join(a.out, "tok", arm, f"{c}.npy")
                    write_npy(p, arr)
                    row = {"clip": c, "n_frames": int(arr.shape[0]), "bytes": os.path.getsize(p), "sha256": sha256_file(p),
                           "n_lead_frames": int(has.sum()), "t_prep_s": round(t_prep, 2), "t_enc_s": round(t_enc, 2), "t": time.strftime("%F %T")}
                    lfs[arm].write(json.dumps(row) + "\n")
                if a.keep_windows:
                    np.save(os.path.join(a.out, "win", f"{c}.npy"), wins)
                n_done += 1
                if n_done <= 3 or n_done % 25 == 0:
                    log(f"{n_done}/{len(todo)} {c[:8]} n={wins.shape[0]} lead={int(has.sum())} prep={t_prep:.1f}s enc={t_enc:.1f}s write={time.time() - tw:.1f}s "
                        f"wall/clip={(time.time() - t_start) / n_done:.1f}s")
                    for arm in ("D", "B"):
                        write_manifest(a.out, arm)
            except Exception as e:                                   # noqa: BLE001
                fails.append((c, f"{type(e).__name__}: {str(e)[:200]}"))
                log(f"FAIL {c[:8]} {type(e).__name__}: {str(e)[:200]}")
    for arm in ("D", "B"):
        write_manifest(a.out, arm)
    with open(os.path.join(a.out, "tok", "ENCODE_FAILS.json"), "w", encoding="utf-8") as fh:
        json.dump(fails, fh, indent=1)
    log(f"done: {n_done} encoded, {len(fails)} failed, wall {time.time() - t_start:.0f}s")
    print("ZZ ENCODE DONE ZZ" if not fails else f"ZZ ENCODE FAIL ZZ {len(fails)}", flush=True)
    return 0 if not fails else 5


if __name__ == "__main__":
    sys.exit(main())
