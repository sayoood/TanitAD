"""STEP 3 — ⛔ MAY THE LABELLED CHECK BE DONE ON THE PRE-BRIDGED VIDEOS?

RETRACTION_LOG **C79** says the pipeline's own re-bridge and the pre-bridged
`bridged_w120train_2400/videos/<cid>.mp4` are DIFFERENT ENCODES of the same
content, and that ~7 % of detections sit close enough to the 0.5 threshold to
FLIP between them. That is a real and settled fact — **about detection counts**.

This study does not re-detect. It draws **banked boxes** on a frame and asks a
human *"is there a traffic sign in this box?"*. The question that decides whether
the cheap video is admissible is therefore NOT "do the encodes differ" (they do)
but **"does the difference change what a human sees inside the banked box?"**

⚠️ Two different failure modes, and only one of them would matter here:
  1. **photometric** — compression noise. Visible as a small mean |Δ|.
  2. ⛔ **geometric / temporal** — a different frame count, fps or crop would put
     the banked box on the WRONG MOMENT or the WRONG PLACE. That would silently
     invalidate every adjudication, and it is the one worth being paranoid about.

So this measures BOTH, on the 8 clips for which the exact pipeline bytes already
exist on this box, at the exact frames the records name, and reports the diff
**inside the banked boxes** as well as over the whole frame.

Frames are obtained by calling `ph0_pilot.sample_clip_frames` itself — not by a
reimplementation — so the moment sampled is the pipeline's own by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0-aug120"
PRE = "bridged_w120train_2400/videos/"
LOCAL_BRIDGE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
                r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
                r"\sam3fix_assets\videos")
CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sam3rel\records")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r3_encode_equivalence")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vdir", default=LOCAL_BRIDGE)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args(argv)

    import ph0_pilot
    from huggingface_hub import hf_hub_download
    tok = token()

    have = sorted(f[:-4] for f in os.listdir(a.vdir) if f.endswith(".mp4"))
    print(f"[eq] {len(have)} clips bridged locally by the pipeline path")
    rows = []
    for cid in have:
        rec_p = os.path.join(a.cache, f"{cid}.json")
        if not os.path.exists(rec_p):
            continue
        rec = json.load(open(rec_p, encoding="utf-8"))
        if rec.get("liveness") is None:
            continue                              # stale C77 record
        loc = os.path.join(a.vdir, f"{cid}.mp4")
        pre = hf_hub_download(DS, f"{PRE}{cid}.mp4", repo_type="dataset",
                              token=tok)
        fl, _t, _n = ph0_pilot.sample_clip_frames(loc, t0_s=8.0)
        fp, _t2, _n2 = ph0_pilot.sample_clip_frames(pre, t0_s=8.0)
        row = {"clip_id": cid,
               "n_frames_local": len(fl), "n_frames_pre": len(fp),
               "shape_local": list(fl[0].shape), "shape_pre": list(fp[0].shape),
               "bytes_local": os.path.getsize(loc),
               "bytes_pre": os.path.getsize(pre)}
        if len(fl) != len(fp) or fl[0].shape != fp[0].shape:
            row["VERDICT"] = "GEOMETRY_OR_LENGTH_MISMATCH"
            rows.append(row)
            print(f"  {cid[:8]} ⛔ {row['VERDICT']}")
            continue
        # whole-frame photometric difference, only on the frames actually run
        fis = sorted(int(k) for k in (rec.get("frames") or {}))
        d_all, d_box, boxes_n = [], [], 0
        for fi in fis:
            if fi >= len(fl):
                continue
            A = fl[fi].astype(np.int16)
            B = fp[fi].astype(np.int16)
            d_all.append(float(np.abs(A - B).mean()))
            for d in rec["frames"][str(fi)].get("det", []):
                b = d.get("box_xyxy")
                if not b:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                x0, y0 = max(0, x0), max(0, y0)
                x1 = min(A.shape[1], max(x1, x0 + 1))
                y1 = min(A.shape[0], max(y1, y0 + 1))
                sub = np.abs(A[y0:y1, x0:x1] - B[y0:y1, x0:x1])
                if sub.size:
                    d_box.append(float(sub.mean()))
                    boxes_n += 1
        # ⭐ THE TEMPORAL CHECK. Photometric noise is harmless; a one-frame
        # OFFSET is not, and it would hide inside a small mean. If the encodes
        # were misaligned in time, the SAME-index diff would be no smaller than
        # the NEIGHBOUR-index diff. Measured, not assumed.
        same, neigh = [], []
        for fi in fis:
            if fi + 1 >= len(fl) or fi >= len(fp):
                continue
            A = fl[fi].astype(np.int16)
            same.append(float(np.abs(A - fp[fi].astype(np.int16)).mean()))
            neigh.append(float(np.abs(A - fp[fi + 1].astype(np.int16)).mean()))
        row.update({
            "n_frames_compared": len(d_all),
            "mean_abs_diff_frame_255": round(float(np.mean(d_all)), 3)
            if d_all else None,
            "max_frame_mean_abs_diff_255": round(float(np.max(d_all)), 3)
            if d_all else None,
            "n_boxes": boxes_n,
            "mean_abs_diff_inside_boxes_255": round(float(np.mean(d_box)), 3)
            if d_box else None,
            "max_box_mean_abs_diff_255": round(float(np.max(d_box)), 3)
            if d_box else None,
            "temporal_same_idx_mean": round(float(np.mean(same)), 3)
            if same else None,
            "temporal_next_idx_mean": round(float(np.mean(neigh)), 3)
            if neigh else None})
        row["VERDICT"] = ("ALIGNED" if (row["temporal_next_idx_mean"] or 0) >
                          3.0 * (row["temporal_same_idx_mean"] or 1e9)
                          else "TEMPORAL_ALIGNMENT_NOT_ESTABLISHED")
        rows.append(row)
        print(f"  {cid[:8]} frames {len(fl)} shape {tuple(fl[0].shape)} · "
              f"frame|Δ| {row['mean_abs_diff_frame_255']} · "
              f"box|Δ| {row['mean_abs_diff_inside_boxes_255']} "
              f"(n={boxes_n}) · same {row['temporal_same_idx_mean']} vs "
              f"next {row['temporal_next_idx_mean']} · {row['VERDICT']}",
              flush=True)

    ok = [r for r in rows if r.get("VERDICT") == "ALIGNED"]
    out = {"n_clips": len(rows), "n_aligned": len(ok),
           "pooled_mean_abs_diff_frame_255":
               round(float(np.mean([r["mean_abs_diff_frame_255"]
                                    for r in ok])), 3) if ok else None,
           "pooled_mean_abs_diff_inside_boxes_255":
               round(float(np.mean([r["mean_abs_diff_inside_boxes_255"]
                                    for r in ok
                                    if r["mean_abs_diff_inside_boxes_255"]
                                    is not None])), 3) if ok else None,
           "pooled_same_idx": round(float(np.mean(
               [r["temporal_same_idx_mean"] for r in ok])), 3) if ok else None,
           "pooled_next_idx": round(float(np.mean(
               [r["temporal_next_idx_mean"] for r in ok])), 3) if ok else None,
           "clips": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[eq] aligned {len(ok)}/{len(rows)} · frame|Δ| "
          f"{out['pooled_mean_abs_diff_frame_255']}/255 · box|Δ| "
          f"{out['pooled_mean_abs_diff_inside_boxes_255']}/255 · "
          f"same {out['pooled_same_idx']} vs next {out['pooled_next_idx']}")
    print("EQ_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
