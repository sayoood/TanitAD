"""STEP 1 — WHICH OF THE PI'S CLASSES ARE ACTUALLY PROMPTABLE? Measure, don't assume.

The PI named: guardrails, road markings, road, ego lane, road curbs. Three
things are unknown before a GPU sees them and all three decide the vocabulary:

 1. **Which SYNONYM grounds.** SAM3's text side is CLIP; `curb`, `kerb` and
    `road curb` are three different embeddings and there is no way to reason
    out which one the model has seen. Ship all of them, count hits.
 2. **Whether `ego lane` grounds at all** — and if it returns something, WHAT.
    The brief's instruction is explicit: prove it with measured output rather
    than assuming, and never silently turn a derived quantity into a prompt.
    A prompt that returns a big plausible mask is the DANGEROUS outcome here,
    not the good one, because nothing downstream could falsify it.
 3. **What each candidate COSTS** — detections per frame, and how big their
    masks are, which is what decides whether the record triples in size.

⛔ ONE ENCODE PER FRAME for the whole candidate list — the same discipline the
production path uses. Probing 16 prompts with 16 encodes would cost 16x and
measure the wrong thing.

Writes /content/out/p1_prompt_probe.json; the driver pulls it into raw/.
"""
import json
import os
import sys
import time

os.chdir("/content")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
from pathlib import Path                                         # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
import ph0_sam3                                                  # noqa: E402

N_CLIPS = int(os.environ.get("P1_CLIPS", "5"))
FRAMES_PER = int(os.environ.get("P1_FRAMES", "2"))
OUT = Path("/content/out")
OUT.mkdir(exist_ok=True)

#: Every candidate, grouped by the PI class it is a candidate FOR. Bare synonyms
#: and modified forms both, because CLIP is sensitive to the modifier and there
#: is no principled way to guess which side wins.
CANDIDATES = {
    "road markings": ["lane marking", "road marking", "lane line",
                      "white lane line", "painted road marking"],
    "road curbs": ["curb", "road curb", "kerb", "sidewalk"],
    "guardrails": ["guardrail", "guard rail", "crash barrier",
                   "metal barrier"],
    "road": ["road", "road surface", "drivable area"],
    # ⚠️ THE ONE THAT SHOULD FAIL. A relation, not an appearance. Recorded so
    # the claim "not promptable" is a measurement rather than an opinion.
    "ego lane": ["ego lane", "my lane", "the lane the car is driving in",
                 "current lane"],
}
ALL = [p for v in CANDIDATES.values() for p in v]
assert len(ALL) == len(set(ALL)), "duplicate candidate prompt"

api = L.hf_api()
gap = L.derive_sam3_gap(api)
clips = gap["absent"][:N_CLIPS]
print(f"[p1] probing {len(ALL)} prompts x {FRAMES_PER} frames x "
      f"{len(clips)} clips = {len(ALL)*FRAMES_PER*len(clips)} scorings")

loc = L.w120_locations(api)
WORK = Path("/content/p1work")
WORK.mkdir(exist_ok=True)
REC_PQ = str(WORK / "records.parquet")
if not Path(REC_PQ).exists():
    import shutil
    shutil.copyfile(L.hf_download(L.DS_ALP, "records.parquet"), REC_PQ)
L.bridge_batch(clips, loc, REC_PQ, WORK)

import ph0_pilot                                                 # noqa: E402
from PIL import Image                                            # noqa: E402
proc, meta = ph0_sam3.build_processor(None)
print(f"[p1] processor up · conf={meta['confidence_threshold']} via "
      f"{meta['confidence_threshold_set_via']} · dtype_fix="
      f"{meta['dtype_fix']['applied']}")
assert meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
L.gpu_mem_report("sam3 load")

stat = {p: {"n_det": 0, "frames_hit": 0, "frames": 0, "scores": [],
            "areas_px": [], "mask_areas_px": []} for p in ALL}
t0 = time.time()
for cid in clips:
    frames = ph0_pilot.sample_clip_frames(
        str(WORK / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]
    h, w = frames[0].shape[:2]
    idx = [len(frames) // 3, 2 * len(frames) // 3][:FRAMES_PER]
    for fi in idx:
        dets = ph0_sam3.detect_many(proc, Image.fromarray(frames[fi]), ALL,
                                    contours=False)
        per = {}
        for d in dets:
            p = d["concept"]
            stat[p]["frames"] = stat[p].get("frames", 0)
            if "error" in d:
                stat[p].setdefault("errors", []).append(d["error"])
                continue
            per[p] = per.get(p, 0) + 1
            stat[p]["scores"].append(round(float(d["score"]), 4))
            b = d.get("box_xyxy")
            if b:
                stat[p]["areas_px"].append(
                    round(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]), 1))
            if d.get("mask_area_px") is not None:
                stat[p]["mask_areas_px"].append(int(d["mask_area_px"]))
        for p in ALL:
            stat[p]["frames"] += 1
            stat[p]["n_det"] += per.get(p, 0)
            stat[p]["frames_hit"] += int(per.get(p, 0) > 0)
    print(f"[p1] {cid[:8]} done ({time.time()-t0:.0f}s)", flush=True)

frame_px = None
for cid in clips[:1]:
    pass


def summ(s):
    sc = sorted(s["scores"])
    ar = sorted(s["mask_areas_px"] or s["areas_px"])
    med = (lambda v: v[len(v) // 2] if v else None)
    return {"n_det": s["n_det"], "frames": s["frames"],
            "frames_hit": s["frames_hit"],
            "hit_rate": round(s["frames_hit"] / s["frames"], 3)
            if s["frames"] else None,
            "det_per_frame": round(s["n_det"] / s["frames"], 2)
            if s["frames"] else None,
            "score_min": sc[0] if sc else None,
            "score_med": med(sc), "score_max": sc[-1] if sc else None,
            "mask_area_med_px": med(sorted(s["mask_areas_px"])),
            "mask_area_max_px": max(s["mask_areas_px"])
            if s["mask_areas_px"] else None,
            "box_area_med_px": med(sorted(s["areas_px"])),
            "n_errors": len(s.get("errors") or [])}


out = {"class": "MEASURED", "n_clips": len(clips), "clips": clips,
       "frames_per_clip": FRAMES_PER, "frame_hw": [h, w],
       "engine": {k: meta[k] for k in ("weights", "confidence_threshold",
                                       "confidence_threshold_set_via")},
       "wall_s": round(time.time() - t0, 1),
       "candidates": CANDIDATES,
       "by_prompt": {p: summ(stat[p]) for p in ALL}}
json.dump(out, open(OUT / "p1_prompt_probe.json", "w"), indent=1)
for grp, ps in CANDIDATES.items():
    print(f"\n== {grp}")
    for p in ps:
        s = out["by_prompt"][p]
        print(f"   {p:32s} det/frame {s['det_per_frame']:>5} · hit "
              f"{s['hit_rate']:>5} · score med {s['score_med']} · "
              f"mask med {s['mask_area_med_px']} px2 / max "
              f"{s['mask_area_max_px']}")
L.gpu_mem_report("p1 end")
print("P1_DONE")
