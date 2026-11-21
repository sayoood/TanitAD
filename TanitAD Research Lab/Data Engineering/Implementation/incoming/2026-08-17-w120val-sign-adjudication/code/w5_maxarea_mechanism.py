"""STEP 5 — the MECHANISM check the reliability study could not run, because it
only had one corpus.

The study tested *"are `traffic sign` false positives preferentially LARGE?"* on
`aug120` and REFUTED it (FPs median 35.6 px² vs 74.8 px² for true ones), and
concluded that G1's max-area SELECTION could not be the cause. ⚠️ That test
answers the question **on aug120's box-area distribution**. It cannot see a
failure mode that exists only where the distribution has a different TAIL.

Arm C (this package) re-read G1's own 54 tiles and found **six that are
essentially the WHOLE FRAME** — a road, a treeline, a night street. A tight crop
of a detection box cannot be the whole scene unless **the box was the whole
scene**. So the testable mechanism is:

  ⭐ **does `traffic sign` on the w120 VAL leg emit frame-spanning boxes that
     `aug120` does not — and does G1's max-area rule therefore select something
     categorically different there?**

This is pure re-analysis of banked geometry — no GPU, no network, no rendering,
and it is NOT circular: it uses box AREA (a geometric fact) and not score, and
it makes no claim about correctness. Correctness comes from arms A/B/C only.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
LEG_FILES = {"w120val600": os.path.join(SCR, "w120sign", "records",
                                        "w120val600.json"),
             "pilot50": os.path.join(SCR, "w120sign", "records",
                                     "pilot50.json")}
AUG120_CACHE = os.path.join(SCR, "sam3rel", "records")
CONCEPT = "traffic sign"


def quant(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return round(s[lo] + (s[hi] - s[lo]) * (i - lo), 1)


def recs_from_leg(p):
    return {r["clip_id"]: r for r in
            json.load(open(p, encoding="utf-8"))["clips"]}


def recs_from_cache(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        if r.get("liveness") is None:          # the C77 stale filter, as filed
            continue
        out[r["clip_id"]] = r
    return out


def analyse(recs, label):
    frame_px = None
    areas, maxpick, per_clip = [], [], collections.defaultdict(list)
    for cid, r in recs.items():
        fw, fh = (r.get("frame_wh") or [448, 179])
        frame_px = float(fw) * float(fh)
        for f in (r.get("frames") or {}).values():
            for d in f.get("det", []) or []:
                if d.get("concept") != CONCEPT or not d.get("box_xyxy"):
                    continue
                b = d["box_xyxy"]
                ar = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                areas.append(ar)
                per_clip[cid].append(ar)
    for cid, xs in per_clip.items():
        maxpick.append(max(xs))
    fr = lambda a: a / frame_px                                  # noqa: E731
    out = {
        "label": label,
        "n_clips": len(recs),
        "n_clips_with_a_sign": len(per_clip),
        "n_sign_detections": len(areas),
        "frame_px": frame_px,
        "ALL_detections_box_area_px": {
            "p10": quant(areas, .10), "median": quant(areas, .50),
            "p90": quant(areas, .90), "p99": quant(areas, .99),
            "max": round(max(areas), 1) if areas else None},
        "ALL_detections_frac_of_frame": {
            "median": round(fr(quant(areas, .50) or 0), 5),
            "p90": round(fr(quant(areas, .90) or 0), 5),
            "max": round(fr(max(areas)), 5) if areas else None},
        "G1_MAXAREA_PICK_box_area_px": {
            "min": round(min(maxpick), 1) if maxpick else None,
            "p25": quant(maxpick, .25), "median": quant(maxpick, .50),
            "p75": quant(maxpick, .75), "p90": quant(maxpick, .90),
            "max": round(max(maxpick), 1) if maxpick else None},
        "G1_MAXAREA_PICK_frac_of_frame_median": round(
            fr(quant(maxpick, .50) or 0), 5) if maxpick else None,
        # ⭐ the discriminating statistic: how much of the class is a box so big
        # that a TIGHT CROP OF IT IS THE SCENE?
        "frame_spanning": {
            k: {"n_detections": sum(1 for a in areas if fr(a) >= k_),
                "frac_of_class": round(
                    sum(1 for a in areas if fr(a) >= k_) / len(areas), 4)
                if areas else None,
                "n_clips_whose_MAXAREA_PICK_is_this_big":
                    sum(1 for a in maxpick if fr(a) >= k_),
                "frac_of_G1_picks": round(
                    sum(1 for a in maxpick if fr(a) >= k_) / len(maxpick), 4)
                if maxpick else None}
            for k, k_ in (("ge_10pct_of_frame", 0.10),
                          ("ge_25pct_of_frame", 0.25),
                          ("ge_50pct_of_frame", 0.50))}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w5_maxarea_mechanism")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    res = {"question": "Does G1's max-area selection pick a CATEGORICALLY "
                       "different object on the w120 val leg than on aug120?",
           "not_a_correctness_measure": "box AREA only — geometry, not score, "
                                        "not verdicts. Correctness is arms "
                                        "A/B/C and nothing here.",
           "legs": {}}
    for leg, p in LEG_FILES.items():
        res["legs"][leg] = analyse(recs_from_leg(p), leg)
    if os.path.isdir(AUG120_CACHE):
        res["legs"]["aug120"] = analyse(recs_from_cache(AUG120_CACHE),
                                        "aug120 (the reliability study's "
                                        "corpus, recomputed from its own cache)")
    else:
        res["legs"]["aug120"] = {"ERROR": "study record cache absent — "
                                          "aug120 comparison NOT COMPUTED"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)

    print(f"{'leg':<12} {'n_sign':>7} {'medArea':>8} {'p99':>8} {'max':>9} "
          f"{'G1pick_med':>11} {'G1pick_max':>11} {'picks>=25%frame':>16}")
    for leg, d in res["legs"].items():
        if "ERROR" in d:
            print(f"{leg:<12} {d['ERROR']}")
            continue
        print(f"{leg:<12} {d['n_sign_detections']:>7} "
              f"{d['ALL_detections_box_area_px']['median']:>8} "
              f"{d['ALL_detections_box_area_px']['p99']:>8} "
              f"{d['ALL_detections_box_area_px']['max']:>9} "
              f"{d['G1_MAXAREA_PICK_box_area_px']['median']:>11} "
              f"{d['G1_MAXAREA_PICK_box_area_px']['max']:>11} "
              f"{d['frame_spanning']['ge_25pct_of_frame']['n_clips_whose_MAXAREA_PICK_is_this_big']:>7}"
              f"/{d['n_clips_with_a_sign']:<8}")
    print("MECHANISM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
