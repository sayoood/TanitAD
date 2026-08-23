"""Contour fidelity STRATIFIED BY OBJECT SIZE — because the aggregate lies.

The corpus-wide median area error is −1.3 %, which reads as "the contour is
essentially the mask". The p10 is **−23.5 %**, which does not. Both are true and
they are about different objects: a fixed 1.0 px tolerance is a large fraction
of a 30 px² traffic light and a rounding error on a 7 000 px² road curb. A
single number would let a consumer apply the small-object figure's optimism to
the class where it does not hold — and the smallest classes are exactly the ones
the reliability study already flagged as operating below their own audit
resolution (`traffic light` median box **34 px²**).

⇒ Reports error by mask-area bucket, by concept, and separately for the
single-loop and multi-loop cases, since those are two different mechanisms:
RDP loses area (negative), keeping only the largest outer loop gains it
(positive).

Reads the records already in the local HF cache — zero GPU, no re-download.

usage:  python c_contour_fidelity.py [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import statistics
import sys

REPO_ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DS = "Sayood/tanitad-ph0-aug120"
BUCKETS = [(0, 50), (50, 200), (200, 1000), (1000, 5000), (5000, 10 ** 9)]


def q(v, p):
    return round(sorted(v)[int(p * (len(v) - 1))], 4) if v else None


def summarise(errs, pts=None):
    if not errs:
        return None
    out = {"n": len(errs), "p10": q(errs, 0.10), "med": q(errs, 0.50),
           "p90": q(errs, 0.90),
           "mean_abs": round(statistics.mean([abs(e) for e in errs]), 4),
           "frac_within_5pct": round(
               sum(1 for e in errs if abs(e) <= 0.05) / len(errs), 4)}
    if pts:
        out["pts_med"] = q(pts, 0.50)
        out["pts_p90"] = q(pts, 0.90)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("c_contour_fidelity")
    ap.add_argument("--prefix", default="sam3_backfill_v2/")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "raw", "contour_fidelity.json"))
    a = ap.parse_args(argv)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    from huggingface_hub import HfApi, hf_hub_download
    with open(os.path.join(REPO_ROOT, "Keys.txt"), encoding="utf-8",
              errors="replace") as fh:
        tok = max(re.findall(r"hf_[A-Za-z0-9]+", fh.read()), key=len)
    api = HfApi(token=tok)
    far = sorted(f.rfilename for f in api.dataset_info(
        DS, files_metadata=True).siblings
        if f.rfilename.startswith(a.prefix) and f.rfilename.endswith(".json")
        and "/_runs/" not in f.rfilename)

    by_bucket = collections.defaultdict(list)
    by_concept = collections.defaultdict(list)
    pts_bucket = collections.defaultdict(list)
    single, multi, tol_used = [], [], collections.Counter()
    areas = []
    for rf in far:
        rec = json.load(open(hf_hub_download(DS, rf, repo_type="dataset",
                                             token=tok), encoding="utf-8"))
        for f in (rec.get("frames") or {}).values():
            for key in ("det", "scene"):
                for d in f.get(key) or []:
                    ar, ca = d.get("mask_area_px"), d.get("contour_area_px")
                    if not ar or ca is None:
                        continue
                    e = (ca - ar) / ar
                    areas.append(ar)
                    tol_used[str(d.get("contour_tol_px"))] += 1
                    npt = len(d.get("contour_xy") or []) // 2
                    for lo, hi in BUCKETS:
                        if lo <= ar < hi:
                            by_bucket[f"{lo}-{hi}"].append(e)
                            pts_bucket[f"{lo}-{hi}"].append(npt)
                            break
                    by_concept[d.get("concept")].append(e)
                    (multi if (d.get("contour_n_loops") or 1) > 1
                     else single).append(e)

    out = {"class": "MEASURED", "n_contours": len(areas),
           "n_records": len(far),
           "mask_area_px": {"p10": q(areas, .1), "med": q(areas, .5),
                            "p90": q(areas, .9)},
           "tolerance_actually_used": dict(tol_used.most_common()),
           "by_mask_area_px": {k: summarise(v, pts_bucket[k])
                               for k, v in sorted(
                                   by_bucket.items(),
                                   key=lambda kv: int(kv[0].split("-")[0]))},
           "by_concept": {k: summarise(v) for k, v in
                          sorted(by_concept.items(),
                                 key=lambda kv: -len(kv[1]))},
           "single_loop": summarise(single), "multi_loop": summarise(multi)}
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
