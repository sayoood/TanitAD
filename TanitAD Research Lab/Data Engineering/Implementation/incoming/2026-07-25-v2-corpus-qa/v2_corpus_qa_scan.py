"""v2 corpus QA scan -- READ-ONLY per-clip integrity + label statistics.

For EVERY ``*.v2ep.pt`` under ``--cache`` this:

  P1 INTEGRITY  torch.load the payload, assert the build_compressed key set,
                decode EVERY JPEG frame (the real corruption test), channel-stack
                them exactly as ``load_compressed`` does, and check every derived
                tensor for shape / dtype / NaN / Inf / length alignment.
  P2 LABELS     recompute the v1 kinematic maneuver histogram
                (``refb_labels.maneuver_labels``, LABEL_HORIZON=20) on exactly the
                poses the trainer sees (``poses[n_stack-1:]``), plus the speed
                regime, per-clip turn/stop/brake presence, net + cumulative
                heading and the v2.1 tight-transient junction probe -- the SAME
                quantities ``score_v2_pool.py`` computed at selection time.

Writes ``<out>.json`` (aggregate + bad-clip list) and ``<out>.csv`` (one row per
clip). NEVER writes inside the cache dir; never mutates a clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchvision.io as tvio

_STACK = os.environ.get("TANITAD_STACK", "/workspace/TanitAD/stack")
# QA_LIB holds the REPO-AUTHORITATIVE refb_labels.py, shipped with this script.
# It goes FIRST on sys.path so a drifted copy inside the pod's stack/scripts can
# never decide a label. (MEASURED 2026-07-25: pod1/pod3 stack/scripts/refb_labels
# .py predates v2.1 and has no route_from_future_v21.)
_QA_LIB = os.environ.get("QA_LIB", "")
if _QA_LIB:
    sys.path.insert(0, _QA_LIB)
sys.path.insert(1 if _QA_LIB else 0, _STACK)
sys.path.append(os.path.join(_STACK, "scripts"))

import refb_labels as rl                                       # noqa: E402
from tanitad.data.comma2k19 import stack_frames                # noqa: E402

assert hasattr(rl, "route_from_future_v21"), (
    f"refb_labels loaded from {rl.__file__} is STALE (no v2.1); set QA_LIB")

REQ_KEYS = ("jpeg_buf", "jpeg_len", "actions", "poses", "n_stack", "image_size",
            "episode_id", "clip_id", "quality")
HZ = 10.0
EXP_SIZE, EXP_NSTACK, EXP_QUALITY = 256, 3, 90


def _f(x):
    return float(x)


def scan_one(path: str, full_stack: bool = True) -> dict:
    r = {"file": os.path.basename(path), "clip_id": "", "ok": False, "err": []}
    try:
        r["bytes"] = os.path.getsize(path)
        r["mtime"] = os.path.getmtime(path)
    except OSError as e:
        r["err"].append(f"STAT:{type(e).__name__}:{e}")
        return r
    if r["bytes"] == 0:
        r["err"].append("ZERO_BYTES")
        return r
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                     # noqa: BLE001
        r["err"].append(f"LOAD:{type(e).__name__}:{str(e)[:180]}")
        return r
    missing = [k for k in REQ_KEYS if k not in d]
    if missing:
        r["err"].append("MISSING_KEYS:" + ",".join(missing))
        return r

    r["clip_id"] = str(d["clip_id"])
    r["episode_id"] = int(d["episode_id"])
    r["quality"] = int(d["quality"])
    r["image_size"] = int(d["image_size"])
    n_stack = int(d["n_stack"])
    r["n_stack"] = n_stack
    if r["image_size"] != EXP_SIZE:
        r["err"].append(f"IMAGE_SIZE:{r['image_size']}")
    if n_stack != EXP_NSTACK:
        r["err"].append(f"N_STACK:{n_stack}")
    if r["quality"] != EXP_QUALITY:
        r["err"].append(f"QUALITY:{r['quality']}")
    if r["clip_id"] and not r["file"].startswith(r["clip_id"]):
        r["err"].append("CLIPID_FILENAME_MISMATCH")

    lens, buf = d["jpeg_len"], d["jpeg_buf"]
    n_raw = int(lens.numel())
    r["n_raw"] = n_raw
    if buf.dtype != torch.uint8:
        r["err"].append(f"BUF_DTYPE:{buf.dtype}")
    if int(lens.sum()) != int(buf.numel()):
        r["err"].append(f"BUF_TRUNCATED:sum={int(lens.sum())}!=buf={int(buf.numel())}")
        return r
    if n_raw == 0:
        r["err"].append("NO_FRAMES")
        return r
    if int(lens.min()) <= 0:
        r["err"].append(f"ZERO_LEN_JPEG:min={int(lens.min())}")

    poses, actions = d["poses"], d["actions"]
    r["poses_shape"] = list(poses.shape)
    r["actions_shape"] = list(actions.shape)
    if poses.ndim != 2 or poses.shape[1] != 4:
        r["err"].append(f"POSES_SHAPE:{list(poses.shape)}")
        return r
    if actions.ndim != 2 or actions.shape[1] != 2:
        r["err"].append(f"ACTIONS_SHAPE:{list(actions.shape)}")
    if poses.shape[0] != n_raw:
        r["err"].append(f"POSE_FRAME_MISALIGN:poses={poses.shape[0]} frames={n_raw}")
    if actions.shape[0] != n_raw:
        r["err"].append(f"ACTION_FRAME_MISALIGN:actions={actions.shape[0]} frames={n_raw}")
    pf, af = poses.float(), actions.float()
    r["n_nan_poses"] = int((~torch.isfinite(pf)).sum())
    r["n_nan_actions"] = int((~torch.isfinite(af)).sum())
    if r["n_nan_poses"]:
        r["err"].append(f"POSES_NONFINITE:{r['n_nan_poses']}")
    if r["n_nan_actions"]:
        r["err"].append(f"ACTIONS_NONFINITE:{r['n_nan_actions']}")

    # ---- P1: decode EVERY JPEG (the real corruption test) -------------------
    offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(lens.to(torch.int64), 0)])
    frames, bad_frames, const_frames = [], 0, 0
    for i in range(n_raw):
        try:
            im = tvio.decode_jpeg(buf[int(offs[i]):int(offs[i + 1])],
                                  mode=tvio.ImageReadMode.RGB)
        except Exception as e:                                 # noqa: BLE001
            bad_frames += 1
            if bad_frames <= 3:
                r["err"].append(f"JPEG_DECODE[{i}]:{type(e).__name__}:{str(e)[:80]}")
            continue
        if tuple(im.shape) != (3, EXP_SIZE, EXP_SIZE) or im.dtype != torch.uint8:
            bad_frames += 1
            if bad_frames <= 3:
                r["err"].append(f"FRAME_SHAPE[{i}]:{tuple(im.shape)}/{im.dtype}")
            continue
        if i % 20 == 0 and float(im.float().std()) < 1.0:      # near-constant image
            const_frames += 1
        frames.append(im)
    r["n_decoded"] = len(frames)
    r["n_bad_frames"] = bad_frames
    r["n_const_frames_sampled"] = const_frames
    if bad_frames:
        r["err"].append(f"BAD_FRAMES:{bad_frames}")
    if const_frames:
        r["err"].append(f"CONSTANT_FRAMES_SAMPLED:{const_frames}")
    if len(frames) != n_raw:
        return r

    # ---- channel-stack exactly as load_compressed does ----------------------
    try:
        vid = torch.stack(frames)
        if full_stack:
            stacked = stack_frames(vid, n_stack)
            r["frames_shape"] = list(stacked.shape)
            if (stacked.ndim != 4 or stacked.shape[1] != 3 * n_stack
                    or stacked.shape[0] != n_raw - (n_stack - 1)
                    or stacked.dtype != torch.uint8):
                r["err"].append(f"STACK_SHAPE:{list(stacked.shape)}/{stacked.dtype}")
            del stacked
        del vid, frames
    except Exception as e:                                     # noqa: BLE001
        r["err"].append(f"STACK:{type(e).__name__}:{str(e)[:120]}")
        return r

    # ---- P2: labels on EXACTLY the trainer-visible poses --------------------
    k = n_stack - 1
    p = pf[k:]                                                 # == load_compressed
    T = int(p.shape[0])
    r["T_out"] = T
    r["hours"] = T / HZ / 3600.0
    if T <= rl.LABEL_HORIZON:
        r["err"].append(f"TOO_SHORT_FOR_LABELS:T={T}")
        r["ok"] = not r["err"]
        return r
    v, yaw = p[:, 3], p[:, 2]
    m1 = rl.maneuver_labels(p, horizon=rl.LABEL_HORIZON)
    m2 = rl.maneuver_labels_v2(p, horizon=rl.LABEL_HORIZON)
    h1 = torch.bincount(m1, minlength=5).tolist()
    h2 = torch.bincount(m2, minlength=5).tolist()
    dyaw = yaw[1:] - yaw[:-1]
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
    junction = 0
    for tt in range(0, T, 5):                                  # v2.1 junction proxy
        if rl.route_from_future_v21(p, tt)["reason"] == "tight_transient":
            junction = 1
            break
    r.update(
        mean_v=_f(v.mean()), max_v=_f(v.max()), min_v=_f(v.min()),
        stop_frac=_f((v < 0.5).float().mean()), dist_m=_f(v.sum()) * 0.1,
        net_head=abs(math.degrees(_f(((yaw[-1] - yaw[0] + math.pi)
                                      % (2 * math.pi)) - math.pi))),
        cum_head=math.degrees(_f(dyaw.abs().sum())), junction=junction,
        stopped=_f((v < 1.0).float().mean()),
        city=_f(((v >= 1) & (v <= 12)).float().mean()),
        hw=_f((v > 12).float().mean()),
        lk=h1[0], tl=h1[1], tr=h1[2], ac=h1[3], bs=h1[4],
        lk2=h2[0], tl2=h2[1], tr2=h2[2], ac2=h2[3], bs2=h2[4],
        nlab=int(sum(h1)),
        has_stop=int(bool((v < 0.5).any())), has_brake=int(h1[4] > 0),
        has_turn=int(h1[1] + h1[2] > 0))
    r["ok"] = not r["err"]
    return r


def _worker(args):
    path, full_stack = args
    try:
        return scan_one(path, full_stack)
    except Exception:                                          # noqa: BLE001
        return {"file": os.path.basename(path), "clip_id": "", "ok": False,
                "err": ["WORKER:" + traceback.format_exc(limit=3)[-300:]]}


ROW_COLS = ["file", "clip_id", "episode_id", "ok", "bytes", "n_raw", "n_decoded",
            "n_bad_frames", "n_const_frames_sampled", "T_out", "nlab", "hours",
            "image_size", "n_stack", "quality", "mean_v", "max_v", "min_v",
            "stop_frac", "dist_m", "net_head", "cum_head", "junction", "stopped",
            "city", "hw", "lk", "tl", "tr", "ac", "bs", "lk2", "tl2", "tr2",
            "ac2", "bs2", "has_stop", "has_brake", "has_turn", "n_nan_poses",
            "n_nan_actions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="output prefix (NOT in --cache)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/K -- scan files with idx%%K==i")
    ap.add_argument("--no-full-stack", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(1)
    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".v2ep.pt"))
    if a.shard:
        i, kk = (int(x) for x in a.shard.split("/"))
        files = [f for j, f in enumerate(files) if j % kk == i]
    if a.limit:
        files = files[:a.limit]
    paths = [(os.path.join(a.cache, f), not a.no_full_stack) for f in files]
    print(f"[qa] {len(paths)} clips in {a.cache}; workers={a.workers}", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(_worker, paths, chunksize=4), 1):
            rows.append(r)
            if n % 250 == 0:
                el = time.time() - t0
                print(f"[qa] {n}/{len(paths)} {el:.0f}s "
                      f"({n/max(el,1e-9):.1f} clip/s) bad={sum(1 for x in rows if not x['ok'])}",
                      flush=True)

    good = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    agg = {
        "cache_dir": a.cache, "host": os.uname().nodename,
        "refb_labels_from": rl.__file__,
        "labeler_thresholds": {
            "LABEL_HORIZON": rl.LABEL_HORIZON, "YAW_TURN_RAD": rl.YAW_TURN_RAD,
            "DV_ACCEL_MS": rl.DV_ACCEL_MS, "DV_BRAKE_MS": rl.DV_BRAKE_MS,
            "STOP_V_MS": rl.STOP_V_MS, "MOVING_V_MS": rl.MOVING_V_MS},
        "scanned": len(rows), "loadable_ok": len(good), "bad": len(bad),
        "scan_seconds": round(time.time() - t0, 1),
        "bad_clips": [{"file": r["file"], "clip_id": r.get("clip_id", ""),
                       "err": r["err"]} for r in bad],
    }
    if good:
        tot = lambda c: sum(int(r[c]) for r in good)           # noqa: E731
        agg.update(
            total_raw_frames=tot("n_raw"),
            total_stacked_frames=tot("T_out"),
            total_label_steps=tot("nlab"),
            hours_stacked=round(tot("T_out") / HZ / 3600.0, 4),
            hours_raw=round(tot("n_raw") / HZ / 3600.0, 4),
            bytes_total=tot("bytes"),
            man_v1_counts={k: tot(k) for k in ("lk", "tl", "tr", "ac", "bs")},
            man_v2_counts={k: tot(k) for k in ("lk2", "tl2", "tr2", "ac2", "bs2")},
            speed_steps={
                "stopped": sum(r["stopped"] * r["T_out"] for r in good),
                "city": sum(r["city"] * r["T_out"] for r in good),
                "hw": sum(r["hw"] * r["T_out"] for r in good)},
            speed_clipmean={k: sum(r[k] for r in good) / len(good)
                            for k in ("stopped", "city", "hw")},
            presence={k: sum(int(r[k]) for r in good) / len(good)
                      for k in ("junction", "has_stop", "has_brake", "has_turn")},
            net_head_gt45=sum(1 for r in good if r["net_head"] > 45) / len(good),
            net_head_gt90=sum(1 for r in good if r["net_head"] > 90) / len(good),
            episode_id_unique=len({r["episode_id"] for r in good}),
            clip_id_unique=len({r["clip_id"] for r in good}),
            T_out_min=min(r["T_out"] for r in good),
            T_out_max=max(r["T_out"] for r in good),
            T_out_mean=tot("T_out") / len(good),
            image_sizes=sorted({r["image_size"] for r in good}),
            n_stacks=sorted({r["n_stack"] for r in good}),
            qualities=sorted({r["quality"] for r in good}),
        )
    with open(a.out + ".json", "w") as f:
        json.dump(agg, f, indent=2)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[qa] DONE ok={len(good)} bad={len(bad)} -> {a.out}.json/.csv", flush=True)
    print(json.dumps({k: v for k, v in agg.items() if k != "bad_clips"}, indent=2)[:2500],
          flush=True)


if __name__ == "__main__":
    main()
