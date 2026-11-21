#!/usr/bin/env python3
"""Control s3 (card ADD100 section 5): the arm S route is the bound arm's route.
The 6 evaluation clips first by sha256 hex of the clip id get the box directions of card ADD99 as steering (frames with has_lead; the
direction NO_LEAD_FILL_DEG elsewhere), written as float32 files into a scratch folder, run through the arm S route (steer_encode.run_clips),
and compared with arm B's tokens on the frames with a lead: cosine per frame of the flattened [2,65,1024] tokens; pass when every such frame
is at or above 0.9999. Writes <s-root>/ctl/S3.json with a top-level boolean `pass`. Frames without a lead are reported, never part of the pass
(arm B carries arm D's two different window centres there; one steering direction cannot reproduce both).
Never deletes; the scratch folder <s-root>/s3 stays in place."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_encode as SE   # noqa: E402
from enc_common import Layout, read_ids   # noqa: E402

THRESH = 0.9999


def pick_clips(eval_list, n=6):
    ev = read_ids(eval_list)
    return sorted(ev, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:n]


def cosine_per_frame(a, b):
    """a, b [n,...] -> float64 [n] cosine of the flattened frames (0 when a norm is 0)."""
    x = np.asarray(a, np.float64).reshape(len(a), -1)
    y = np.asarray(b, np.float64).reshape(len(b), -1)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    den = nx * ny
    return np.where(den > 0, (x * y).sum(1) / np.where(den > 0, den, 1.0), 0.0)


def compare_s3(tokS, tokB, has, thresh=THRESH):
    """Per-frame cosine on the lead frames. -> dict (n_lead, min, median, n_below, pass). A frame-count mismatch or no lead frame fails."""
    if tokS.shape != tokB.shape or len(has) != len(tokB):
        return {"n_lead": 0, "min": None, "median": None, "n_below": None, "pass": False,
                "why": f"shape {tokS.shape} vs {tokB.shape}, has {len(has)}"}
    idx = np.nonzero(has)[0]
    if idx.size == 0:
        return {"n_lead": 0, "min": None, "median": None, "n_below": None, "pass": False, "why": "no frame with a lead"}
    c = cosine_per_frame(tokS[idx], tokB[idx])
    return {"n_lead": int(idx.size), "min": float(c.min()), "median": float(np.median(c)), "n_below": int((c < thresh).sum()),
            "pass": bool(c.min() >= thresh), "worst_frames": [int(i) for i in idx[np.argsort(c)[:5]]]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--root", required=True, help="stage root")
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--b-tok-dir", required=True, help="arm B tokens folder (.../tok/B)")
    ap.add_argument("--s-root", required=True, help="output root; scratch in <s-root>/s3, record in <s-root>/ctl/S3.json")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.s_root, "ctl", "S3.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    scratch = os.path.join(a.s_root, "s3")
    clips = pick_clips(a.eval_list, a.n)
    L = Layout(a.root)
    os.makedirs(os.path.join(scratch, "steer"), exist_ok=True)
    has_map, quant = {}, {}
    for c in clips:
        st, has = SE.box_steering(a.box_dir, c)
        np.save(SE.steer_path(scratch, c), st)
        has_map[c] = has
        z = np.load(os.path.join(a.box_dir, f"{c}.npz"))
        d = np.abs(st[has].astype(np.float64) - np.stack([z["ctr_az"], z["ctr_el"]], 1)[has]) if has.any() else np.zeros((0, 2))
        quant[c] = float(d.max()) if d.size else 0.0
    t0 = time.time()
    n_done, fails, rc = SE.run_clips(clips, L, scratch, workers=a.workers, min_free_gb=5.0)
    wall = time.time() - t0
    per, allmin, allmed, ok = {}, [], [], rc == 0 and not fails
    for c in clips:
        r = {}
        try:
            tS = np.load(SE.tok_path(scratch, c))
            pb = os.path.join(a.b_tok_dir, f"{c}.npy")
            tB = np.load(pb)
            r = compare_s3(tS, tB, has_map[c])
            nl = np.nonzero(~has_map[c])[0]
            if nl.size:
                r["no_lead_frames_cos_min_info"] = float(cosine_per_frame(tS[nl], tB[nl]).min())
                r["n_no_lead_frames"] = int(nl.size)
        except Exception as e:                                       # noqa: BLE001
            r = {"pass": False, "why": f"{type(e).__name__}: {str(e)[:160]}"}
        r["max_abs_float32_cast_deg"] = quant[c]
        if r.get("n_lead") == 0 and r.get("why") == "no frame with a lead":
            r["pass"] = True                                         # a clip with no lead frame has nothing to compare; the total is checked below
            r["note"] = "no lead frame on this clip; not counted"
        per[c] = r
        ok = ok and bool(r.get("pass"))
        if r.get("min") is not None:
            allmin.append(r["min"])
            allmed.append(r["median"])
    n_lead_total = int(sum(r.get("n_lead") or 0 for r in per.values()))
    ok = ok and n_lead_total > 0
    res = {"control": "s3", "threshold": THRESH, "clips": clips, "n_clips": len(clips), "encode_rc": rc, "encode_fails": fails,
           "n_lead_frames_total": n_lead_total, "n_clips_with_lead": int(sum(1 for r in per.values() if (r.get("n_lead") or 0) > 0)),
           "cosine_min": min(allmin) if allmin else None, "cosine_median_of_clip_medians": float(np.median(allmed)) if allmed else None,
           "seconds_per_clip": round(wall / max(1, len(clips)), 2), "per_clip": per, "pass": bool(ok)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
