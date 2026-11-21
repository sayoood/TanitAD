#!/usr/bin/env python3
"""Arm S windows and tokens (card ADD100 section 3). Pod only (GPU for the encoder).

Per clip: reads <S_ROOT>/steer/<clip>.npy (float32 [n_frames, 2]; azimuth, elevation in degrees; left and up positive), builds per frame
the two windows W32 and W16, BOTH centred on that frame's steering direction, with the routines of the data kit cropread/enc imported
unchanged (frame_plan, decode_selected, window_frame, window_grid, window_sample, Encoder.window_tokens), and writes
<S_ROOT>/tok/S/<clip>.npy float16 [n_frames, 2, 65, 1024] (window order W32 then W16; token 0 = class token, 1..64 = 8x8 pooled tokens).
<S_ROOT>/tok/S/ledger.jsonl is the resume record, <S_ROOT>/tok/S/MANIFEST.json holds per clip n_frames, bytes, the sha256 of the token file
and the sha256 of the steering file used. Nothing is deleted; an output file is written whole under a .part name and renamed once.
A steering file with a wrong shape or dtype or a non-finite value is refused (all files are checked before the encoder starts).
A token file that exists for a different steering file is refused, never overwritten (use a new S_ROOT).
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
for _p in (os.path.join(HERE, "..", "enc"), "/workspace/v8proof/cropread/kit/enc"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from enc_common import (Layout, SIZES_DEG, WIN_PX, decode_selected, frame_plan, read_ids, sha256_file,   # noqa: E402
                        window_frame, window_grid, window_sample, Encoder)

BATCH = 8
DEFAULT_S_ROOT = "/workspace/v8proof/cropread/s_out"
NO_LEAD_FILL_DEG = (0.0, -4.0)        # direction used by the smoke / control helpers on frames without a lead (any direction is admissible)


def log(m):
    print(f"[steer {time.strftime('%H:%M:%S')}] {m}", flush=True)


def steer_path(s_root, clip):
    return os.path.join(s_root, "steer", f"{clip}.npy")


def tok_path(s_root, clip):
    return os.path.join(s_root, "tok", "S", f"{clip}.npy")


def check_steering(arr, n):
    """Refuse a steering array that is not float32 [n, 2] with every value finite. Returns arr."""
    if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
        raise ValueError(f"steering dtype {getattr(arr, 'dtype', type(arr))}, need float32")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"steering shape {arr.shape}, need [n_frames, 2]")
    if arr.shape[0] != n:
        raise ValueError(f"steering has {arr.shape[0]} frames, the join has {n}")
    if not np.isfinite(arr).all():
        raise ValueError(f"steering has {int((~np.isfinite(arr)).sum())} non-finite values")
    return arr


def read_steer(path, n):
    if not os.path.exists(path):
        raise FileNotFoundError(f"no steering file {path}")
    return check_steering(np.load(path, allow_pickle=False), n)


def box_steering(box_dir, clip):
    """Steering from the box files of card ADD99: the box-centre direction where the frame has a lead, else NO_LEAD_FILL_DEG.
    -> (float32 [n,2], has bool[n]). The box file holds float64; the cast to float32 is the interface's."""
    z = np.load(os.path.join(box_dir, f"{clip}.npz"))
    has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
    az = np.where(has, z["ctr_az"], NO_LEAD_FILL_DEG[0])
    el = np.where(has, z["ctr_el"], NO_LEAD_FILL_DEG[1])
    return np.stack([az, el], 1).astype(np.float32), has


def windows_from_video(vid, intr, frames, az, el):
    """vid [B,3,h,w] (uint8 or float) + steering az, el [B] -> uint8 [B,2,3,256,256]: W32 then W16 both on (az, el). Pure CPU."""
    import torch
    h, w = int(vid.shape[-2]), int(vid.shape[-1])
    B = vid.shape[0]
    out = torch.zeros((B, 2, 3, WIN_PX, WIN_PX), dtype=torch.uint8)
    for i in range(2):
        g, m = window_grid(intr, h, w, frames[i], np.asarray(az, np.float64), np.asarray(el, np.float64))
        out[:, i] = window_sample(vid, g, m)
    return out.numpy()


def make_steer_windows(clip, L, steer, torch_threads=2):
    """-> (uint8 [n,2,3,256,256], plan, secs). Decodes the clip with the builder's settings, cuts the two steered windows per frame."""
    import torch
    from tanitad.data.physicalai import intrinsics_for_clip
    t0 = time.time()
    torch.set_num_threads(torch_threads)
    plan = frame_plan(L.ts(clip))
    n, cidx = plan["n_cache"], plan["cache_idx"]
    check_steering(steer, n)
    intr = intrinsics_for_clip(clip, L.root)
    assert intr.per_clip, f"{clip}: corpus-median intrinsics refused (per-clip table missing)"
    frames = [window_frame(s) for s in SIZES_DEG]
    out = np.zeros((n, 2, 3, WIN_PX, WIN_PX), np.uint8)
    by_idx = collections.defaultdict(list)
    for j, vi in enumerate(cidx):
        by_idx[int(vi)].append(j)
    need = sorted(by_idx)
    pend = []

    def flush():
        nonlocal pend
        if not pend:
            return
        js = [j for j, _ in pend]
        vid = torch.stack([f for _, f in pend]).float()
        out[js] = windows_from_video(vid, intr, frames, steer[js, 0], steer[js, 1])
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
    return out, plan, time.time() - t0


def encode_steer_tokens(enc, wins):
    """wins uint8 [n,2,3,256,256] -> float16 [n,2,65,1024]."""
    import torch
    n = wins.shape[0]
    x = torch.from_numpy(wins.reshape(n * 2, 3, WIN_PX, WIN_PX))
    return enc.window_tokens(x).reshape(n, 2, 65, -1).to(torch.float16).numpy()


def write_npy(path, arr):
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
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


def write_manifest(s_root):
    led = read_ledger(os.path.join(s_root, "tok", "S", "ledger.jsonl"))
    man = {"arm": "S", "layout": "float16 [n_frames, 2, 65, 1024]; window order W32 then W16, both on the frame's steering direction; "
                                 "token 0 = class token, 1..64 = 8x8 pooled patch tokens row-major",
           "n_clips": len(led),
           "clips": {c: {"n_frames": r["n_frames"], "sha256": r["sha256"], "bytes": r["bytes"], "steer_sha256": r["steer_sha256"]}
                     for c, r in sorted(led.items())}}
    with open(os.path.join(s_root, "tok", "S", "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def precheck(clips, L, s_root):
    """Every steering file present, float32 [n,2] with the plan's n, finite. -> list of (clip, reason)."""
    bad = []
    for c in clips:
        try:
            n = frame_plan(L.ts(c))["n_cache"]
            read_steer(steer_path(s_root, c), n)
        except Exception as e:                                        # noqa: BLE001
            bad.append((c, f"{type(e).__name__}: {str(e)[:160]}"))
    return bad


def run_clips(clips, L, s_root, workers=3, torch_threads=2, enc_bs=8, min_free_gb=25.0, enc=None, keep_windows=False):
    """Encode the clips (resumable). -> (n_done, fails list, rc)."""
    import shutil
    os.makedirs(os.path.join(s_root, "tok", "S"), exist_ok=True)
    led_path = os.path.join(s_root, "tok", "S", "ledger.jsonl")
    led = read_ledger(led_path)
    todo, refused = [], []
    for c in clips:
        r = led.get(c)
        p = tok_path(s_root, c)
        sha_steer = sha256_file(steer_path(s_root, c))
        if r and os.path.exists(p) and os.path.getsize(p) == r["bytes"]:
            if r["steer_sha256"] == sha_steer:
                continue
            refused.append((c, "a token file exists for a different steering file; not overwritten"))
            continue
        todo.append(c)
    log(f"{len(clips)} clips, {len(clips) - len(todo) - len(refused)} done, {len(todo)} to do, {len(refused)} refused; workers {workers}")
    if refused:
        return 0, refused, 6
    if enc is None:
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
                n = frame_plan(L.ts(c))["n_cache"]
                q.append((c, ex.submit(make_steer_windows, c, L, read_steer(steer_path(s_root, c), n), torch_threads)))
        for _ in range(workers + 1):
            submit_next()
        while q:
            c, fut = q.popleft()
            submit_next()
            try:
                if shutil.disk_usage(s_root).free / 1e9 < min_free_gb:
                    log("STOP: disk free below the floor")
                    return n_done, fails, 7
                wins, plan, t_prep = fut.result()
                te = time.time()
                tok = encode_steer_tokens(enc, wins)
                t_enc = time.time() - te
                p = tok_path(s_root, c)
                write_npy(p, tok)
                row = {"clip": c, "n_frames": int(tok.shape[0]), "bytes": os.path.getsize(p), "sha256": sha256_file(p),
                       "steer_sha256": sha256_file(steer_path(s_root, c)), "t_prep_s": round(t_prep, 2), "t_enc_s": round(t_enc, 2),
                       "t": time.strftime("%F %T")}
                lf.write(json.dumps(row) + "\n")
                if keep_windows:
                    os.makedirs(os.path.join(s_root, "win"), exist_ok=True)
                    np.save(os.path.join(s_root, "win", f"{c}.npy"), wins)
                n_done += 1
                if n_done <= 3 or n_done % 25 == 0:
                    log(f"{n_done}/{len(todo)} {c[:8]} n={tok.shape[0]} prep={t_prep:.1f}s enc={t_enc:.1f}s wall/clip={(time.time() - t_start) / n_done:.1f}s")
                    write_manifest(s_root)
            except Exception as e:                                   # noqa: BLE001
                fails.append((c, f"{type(e).__name__}: {str(e)[:200]}"))
                log(f"FAIL {c[:8]} {type(e).__name__}: {str(e)[:200]}")
    write_manifest(s_root)
    log(f"done: {n_done} encoded, {len(fails)} failed, wall {time.time() - t_start:.0f}s")
    return n_done, fails, (0 if not fails else 5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True)
    ap.add_argument("--root", required=True, help="stage root (the data kit's layout)")
    ap.add_argument("--s-root", default=os.environ.get("S_ROOT", DEFAULT_S_ROOT))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--enc-bs", type=int, default=8)
    ap.add_argument("--keep-windows", action="store_true", help="smoke only: keep win/<clip>.npy (private camera pixels)")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    a = ap.parse_args(argv)
    L = Layout(a.root)
    clips = read_ids(a.list)
    bad = precheck(clips, L, a.s_root)
    if bad:
        os.makedirs(os.path.join(a.s_root, "tok"), exist_ok=True)
        with open(os.path.join(a.s_root, "tok", "STEER_REFUSED.json"), "w", encoding="utf-8") as fh:
            json.dump(bad, fh, indent=1)
        log(f"REFUSED: {len(bad)} clips have no valid steering file (first: {bad[0][0][:8]} {bad[0][1]})")
        print(f"ZZ STEER ENCODE REFUSED ZZ {len(bad)}", flush=True)
        return 4
    n_done, fails, rc = run_clips(clips, L, a.s_root, a.workers, a.torch_threads, a.enc_bs, a.min_free_gb, keep_windows=a.keep_windows)
    with open(os.path.join(a.s_root, "tok", "ENCODE_FAILS.json"), "w", encoding="utf-8") as fh:
        json.dump(fails, fh, indent=1)
    print("ZZ STEER ENCODE DONE ZZ" if rc == 0 else f"ZZ STEER ENCODE FAIL ZZ rc={rc} n={len(fails)}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
