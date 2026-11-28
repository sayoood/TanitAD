"""G-MAP-OVERFIT frame set: select 16 TRAIN frames (4 clips x 4 frames) from the SAM3 10 cm labels ONLY.

Runs ON THOR, read-only (`ssh tanitad-thor-wifi <python> - < this file`), prints one JSON to stdout.
No model, no prediction, no eval clip: the selection sees labels and geometry only, and it is frozen in
`raw/gmo_frameset.json` BEFORE any G-MAP-OVERFIT run exists.

Rule (declared here, applied mechanically):
  * candidates: the K = 300 smallest-sha12 clips of `/home/nvidia/data/refcv6_train_clips.txt` (the TRAIN
    split refcv6 trained on) that have a canonical SAM3 GT file;
  * candidate label frames r = 10, 15, 20, ... <= T - 25 (a full window + 2 s of future exists);
  * counted cells = fine_codes != 255, in the rig x band 0-20 m, inside an APPROXIMATE camera cone
    (|azimuth from the camera| <= 60 deg, horizontal distance from the camera >= 3.0 m; the camera x
    offset from `refcv6_train_extrinsics.json`) -- the harness applies the exact lift mask;
  * per clip, its 4 frames for a target class = greedy best frames for that class, >= 20 frames apart;
  * clips are picked greedily for the rarest classes first: hatched, arrow/text, crosswalk,
    non-drivable edge (one clip each, the one whose 4 frames hold the most cells of that class,
    ties by sha12);
  * requirement: every one of the 8 classes has >= M_MIN cells over the 16 frames (0-20 m band); if a
    class is short, the clip for the most-short class is replaced by the next best (up to 20 swaps).
Clip ids never printed: sha12 only.
"""
import hashlib
import json
import math
import sys
import time

import numpy as np

K = 300
M_MIN = 1000
GT = "/home/nvidia/data/sam3_corpus/semantic_maps/gt"
CLIPS = "/home/nvidia/data/refcv6_train_clips.txt"
EXTR = "/home/nvidia/data/refcv6_train_extrinsics.json"
C8 = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text", "non-drivable edge",
      "hatched", "sidewalk/verge"]
ORDER = [6, 4, 3, 5]                  # hatched, arrow/text, crosswalk, edge


def sha12(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main():
    t0 = time.time()
    clips = [l.strip() for l in open(CLIPS) if l.strip()]
    extr = json.load(open(EXTR))
    cands = sorted(clips, key=sha12)[:K]
    X = (np.arange(600) + 0.5) * 0.1
    Y = -16.0 + (np.arange(320) + 0.5) * 0.1
    per = {}
    n_missing = 0
    for cid in cands:
        s12 = sha12(cid)
        path = f"{GT}/{cid}.sam3mapgt.npz"
        try:
            z = np.load(path)
            fc = z["fine_codes"]
            meta = json.loads(str(z["meta_json"]))
        except FileNotFoundError:
            n_missing += 1
            continue
        if (meta.get("source") or {}).get("clip_sha12") != s12:
            raise SystemExit(f"[gmo] {s12}: GT identity mismatch")
        e = extr.get(cid)
        cx = float(e["x"]) if e else 2.05
        dx = X[:, None] - cx
        az = np.degrees(np.arctan2(Y[None, :], dx))
        cone = (np.abs(az) <= 60.0) & (np.hypot(dx, Y[None, :]) >= 3.0) & (dx > 0)
        T = fc.shape[0]
        fr = {}
        for r in range(10, T - 24, 5):
            c = fc[r]
            row = {}
            for b, (lo, hi) in enumerate(((0, 200), (200, 400), (400, 600))):
                cb = c[lo:hi][cone[lo:hi]]
                cb = cb[cb != 255]
                row[b] = np.bincount(cb, minlength=8)[:8].astype(int).tolist()
            fr[r] = row
        per[s12] = {"T": int(T), "cam_x": cx, "frames": fr}
    def distinct(a, b):
        # v2 (22:45): a STATIONARY ego repeats the same rig-frame label map (v1 picked 3 identical
        # frames of one clip). Two frames are distinct only if their 0-20 m class-count vectors differ
        # by >= 2 % of their seen cells (L1).
        va, vb = np.array(a), np.array(b)
        return np.abs(va - vb).sum() >= 0.02 * max(va.sum(), vb.sum(), 1)

    def best4(s12, cls):
        fr = per[s12]["frames"]
        order = sorted(fr, key=lambda r: (-fr[r][0][cls], r))
        pick = []
        for r in order:
            if all(abs(r - q) >= 20 and distinct(fr[r][0], fr[q][0]) for q in pick):
                pick.append(r)
            if len(pick) == 4:
                break
        return sorted(pick)
    def tot(sel):
        t = np.zeros((3, 8), dtype=np.int64)
        for s12, frs in sel:
            for r in frs:
                for b in range(3):
                    t[b] += np.array(per[s12]["frames"][r][b])
        return t
    ranked = {cls: sorted(per, key=lambda s: (-sum(per[s]["frames"][r][0][cls] for r in best4(s, cls)), s))
              for cls in ORDER}
    ptr = {cls: 0 for cls in ORDER}
    swaps = 0
    while True:
        sel, used = [], set()
        for cls in ORDER:
            k = ptr[cls]
            while ranked[cls][k] in used:
                k += 1
            s = ranked[cls][k]
            used.add(s)
            sel.append((s, best4(s, cls)))
        t = tot(sel)
        short = [c for c in range(8) if t[0, c] < M_MIN]
        if not short or swaps >= 20:
            break
        worst = min(short, key=lambda c: t[0, c])
        # replace the clip picked for the class nearest in rarity order to the short class
        tgt = worst if worst in ORDER else ORDER[-1]
        ptr[tgt] += 1
        swaps += 1
    t = tot(sel)
    out = {"rule": __doc__.strip(), "K": K, "M_MIN": M_MIN, "n_candidates_with_gt": len(per),
           "n_candidates_missing_gt": n_missing, "swaps": swaps,
           "frames": [{"clip_sha12": s, "for_class": C8[cls], "raw_v2ep_frames": frs,
                       "window_t_equals_raw_minus_9": [r - 9 for r in frs],
                       "T": per[s]["T"], "cam_x_m": per[s]["cam_x"],
                       "cells_0_20m_by_frame": {str(r): per[s]["frames"][r][0] for r in frs}}
                      for (s, frs), cls in zip(sel, ORDER)],
           "cells_by_class_0_20m": dict(zip(C8, t[0].tolist())),
           "cells_by_class_20_40m": dict(zip(C8, t[1].tolist())),
           "cells_by_class_40_60m": dict(zip(C8, t[2].tolist())),
           "all_classes_ge_M_MIN_0_20m": bool((t[0] >= M_MIN).all()),
           "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
