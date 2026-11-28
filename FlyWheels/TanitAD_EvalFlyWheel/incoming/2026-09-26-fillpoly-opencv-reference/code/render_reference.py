#!/usr/bin/env python3
"""STEP 2 of 2 — draw the banked polygons with REAL cv2.fillPoly 4.5.4, in the throwaway OpenCV env.

Run ONLY with ``D:/venvs/opencv-ref-454/Scripts/python.exe`` (Python 3.9.25, opencv-python-headless
4.5.4.60, numpy 1.26.4, installed offline with --no-index --no-deps; PI approval relayed by the Master
Mind: "you can download opencv on D:").

⛔ It ASSERTS ``cv2.__version__ == "4.5.4"`` BEFORE drawing anything. A wrong-version raster banked as a
reference would pass every later comparison while testing nothing — the "check that shares the defect it
checks for" class. The reference also carries its OWN proof of version (build-info head, wheel sha256s
recomputed from the files on disk, never copied from another receipt).

The call is exactly the one the port claims to reproduce:
    cv2.fillPoly(img, [np.array(poly, np.int32)], 1)      # lineType default LINE_8, shift 0
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np

WHEELS = "D:/venvs/opencv-ref-454_wheels"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    polys_json, out_npz, out_receipt = sys.argv[1:4]
    # ⛔ BEFORE ANY DRAWING
    assert cv2.__version__ == "4.5.4", f"wrong OpenCV: {cv2.__version__} -- refusing to bank a reference"
    build_head = [ln.strip() for ln in cv2.getBuildInformation().splitlines() if ln.strip()][:3]
    assert "OpenCV 4.5.4" in build_head[0], build_head

    doc = json.load(open(polys_json, encoding="utf-8"))
    arrays, groups_meta = {}, {}
    for g, spec in doc["groups"].items():
        h, w = spec["hw"]
        polys = spec["polys"]
        vmax = max(len(p) for p in polys)
        padded = np.zeros((len(polys), vmax, 2), np.int32)
        nverts = np.zeros(len(polys), np.int32)
        rasters = np.zeros((len(polys), h, w), np.uint8)
        ok = np.ones(len(polys), bool)
        raised = []
        for i, p in enumerate(polys):
            nverts[i] = len(p)
            padded[i, :len(p)] = np.asarray(p, np.int32)
            img = np.zeros((h, w), np.uint8)
            try:
                cv2.fillPoly(img, [np.asarray(p, np.int32)], 1)
            except cv2.error as e:                      # record, never guess what cv2 "would" have drawn
                ok[i] = False
                raised.append({"case": i, "error": str(e).splitlines()[-1][:200]})
            rasters[i] = img
        arrays.update({f"{g}_hw": np.array([h, w], np.int32), f"{g}_nverts": nverts,
                       f"{g}_polys": padded, f"{g}_rasters": rasters, f"{g}_ok": ok})
        groups_meta[g] = {"n": len(polys), "hw": [h, w], "cv2_raised": raised,
                          "filled_pixels_total": int(rasters.sum()), "source": spec.get("source")}
    np.savez_compressed(out_npz, **arrays)

    receipt = {
        "_what": "cv2.fillPoly 4.5.4 reference rasters for taniteval.adapters.nuscenes_planning.cv_fill_poly",
        "cv2_version": cv2.__version__,
        "cv2_build_info_head": build_head,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "call": "cv2.fillPoly(img, [np.array(poly, np.int32)], 1)  # lineType LINE_8 (default), shift 0",
        "wheels_sha256_recomputed": {os.path.basename(p): sha256(p) for p in sorted(glob.glob(f"{WHEELS}/*.whl"))},
        "polygons_json_sha256": sha256(polys_json),
        "polygons_generated_with": doc.get("generated_with"),
        "npz_file": os.path.basename(out_npz),
        "npz_sha256": sha256(out_npz),
        "groups": groups_meta,
    }
    with open(out_receipt, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    print(f"cv2 {cv2.__version__} | " + " | ".join(f"{g}: {m['n']} cases, {len(m['cv2_raised'])} raised, "
                                               f"{m['filled_pixels_total']} px" for g, m in groups_meta.items()))
    print(f"npz sha256 {receipt['npz_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
