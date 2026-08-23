"""C85, proved on the REAL overlay function rather than argued from the code.

The claim in the retraction is that a v1 record renders NO mask fill because
`draw_camera` does `over[r, a:b]` and numpy clips an out-of-range slice to
nothing. That is a reading of the source. This turns it into a measurement:
pull a real v1 record, render the same detections TWICE through the real
`ph0_rich_overlay.draw_camera` — once with `rle_rows`, once with the key
removed — and count the pixels that differ. If the mask layer draws anything,
that count is large. If C85 holds, it is exactly 0.

⚠️ Deliberately compares against the SAME function with the field removed, not
against a hand-rolled expectation: a difference of 0 then means "the mask layer
contributed nothing", with the boxes, labels and scaling identical on both
sides and therefore cancelled out.

usage:  python c85_overlay_proof.py [--out FILE]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

REPO_ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DS = "Sayood/tanitad-ph0-aug120"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("c85_overlay_proof")
    ap.add_argument("--prefix", default="sam3_backfill/")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "raw", "c85_overlay_proof.json"))
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(REPO_ROOT, "stack", "scripts"))
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    import numpy as np
    import ph0_rich_overlay as ov
    from huggingface_hub import HfApi, hf_hub_download
    with open(os.path.join(REPO_ROOT, "Keys.txt"), encoding="utf-8",
              errors="replace") as fh:
        tok = max(re.findall(r"hf_[A-Za-z0-9]+", fh.read()), key=len)
    api = HfApi(token=tok)
    far = sorted(f.rfilename for f in api.dataset_info(
        DS, files_metadata=True).siblings
        if f.rfilename.startswith(a.prefix) and f.rfilename.endswith(".json")
        and "/_runs/" not in f.rfilename)
    rec = json.load(open(hf_hub_download(DS, far[0], repo_type="dataset",
                                         token=tok), encoding="utf-8"))
    W, H = rec["frame_wh"]
    dets = [d for f in rec["frames"].values() for d in f.get("det", [])
            if d.get("rle_rows")]
    runs = [r for d in dets for r in d["rle_rows"]]
    in_frame = sum(1 for r in runs if 0 <= r[0] < H and r[1] < W)
    frame = np.full((H, W, 3), 100, np.uint8)
    with_rle = np.asarray(ov.draw_camera(frame, dets[:50], []).convert("RGB"))
    without = np.asarray(ov.draw_camera(
        frame, [{k: v for k, v in d.items() if k != "rle_rows"}
                for d in dets[:50]], []).convert("RGB"))
    out = {"class": "MEASURED", "record": far[0], "clip_id": rec["clip_id"],
           "frame_wh": [W, H], "n_det_total": rec.get("n_det_total"),
           "n_dets_with_rle": len(dets), "n_runs": len(runs),
           "distinct_row_indices": sorted({r[0] for r in runs})[:5],
           "max_column_index": max(r[2] for r in runs),
           "runs_that_land_in_frame": in_frame,
           "pixels_the_mask_layer_changed": int(
               (with_rle != without).any(-1).sum())}
    out["C85_CONFIRMED"] = bool(out["pixels_the_mask_layer_changed"] == 0
                                and out["max_column_index"] > W)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
