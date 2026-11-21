"""STEP 2 — SIZE IT BEFORE YOU RUN IT. Three arms, one session, five clips.

⛔ WHY THREE ARMS AND NOT A BEFORE/AFTER. The v2 run changes TWO things at once
— the schema (contours + a scene channel) and the detection floor (0.5 -> 0.25)
— and a single before/after would report their sum as though it were the
schema's cost. That is the `--v2` conflation defect: ten levers on two axes and
a non-attributable result. So:

    A   v1 schema, conf 0.50   <- what the 83 banked records actually are
    B   v1 schema, conf 0.25   <- the floor's cost, alone
    C   v2 schema, conf 0.25   <- what the re-run will bank

C/A is the headline growth the brief asks for; B/A and C/B say which half of it
came from where.

⛔ AND ALL THREE RUN IN ONE SESSION ON THE SAME DECODED BYTES. MEASURED (C79):
the same code on the same clip gives 60 vs 64 detections on two different Colab
VMs, and ~7 % of detections sit close enough to the threshold to flip on
re-encode noise. A growth factor computed across VMs, or against a number
banked by an earlier run, would carry that noise as though it were the change.

⚠️ The threshold is moved by ASSIGNMENT between arms rather than by rebuilding
the processor — a second `build_sam3_image_model` in one kernel adds 3.58 GB of
weights to a 16 GB card and pollutes `max_memory_allocated`, which is a
process-global counter (that is how the first production attempt OOM'd).

Writes /content/out/p2_pilot_size.json.
"""
import json
import os
import statistics
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

N_CLIPS = int(os.environ.get("P2_CLIPS", "5"))
FRAME_STRIDE = 8
OUT = Path("/content/out")
OUT.mkdir(exist_ok=True)
WORK = Path("/content/p2work")
WORK.mkdir(exist_ok=True)

api = L.hf_api()
gap = L.derive_sam3_gap(api)
clips = gap["absent"][:N_CLIPS]
v2_by = L.load_v2_records(api, set(clips))
loc = L.w120_locations(api)
REC_PQ = str(WORK / "records.parquet")
if not Path(REC_PQ).exists():
    import shutil
    shutil.copyfile(L.hf_download(L.DS_ALP, "records.parquet"), REC_PQ)
L.bridge_batch(clips, loc, REC_PQ, WORK)

import ph0_pilot                                                 # noqa: E402
import torch                                                     # noqa: E402
# ⛔ ONE PROCESSOR PER KERNEL. `colab exec` shares globals across invocations,
# so a previous step's processor is still resident and its 3.58 GB of weights
# both count against the 16 GB card and pollute `max_memory_allocated`, which
# is PROCESS-GLOBAL. That is how the first production attempt OOM'd at model
# load with 14.5 GiB already in use, and how a "14.2 GB peak for one set_image"
# got into a report.
for _n in ("proc", "_proc"):
    if _n in globals():
        L.free_leg(globals().pop(_n))
proc, meta = ph0_sam3.build_processor(None)
assert meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
print(f"[p2] conf={meta['confidence_threshold']} via "
      f"{meta['confidence_threshold_set_via']}")
L.gpu_mem_report("sam3 load")

ARMS = [
    ("A_v1_conf050", 0.50, None, False),
    ("B_v1_conf025", 0.25, None, False),
    ("C_v2_conf025", 0.25, ph0_sam3.SCENE_CONCEPTS, True),
]


def sizes(rec):
    """Both encodings, because `bank_json` uses indent=1 today and the choice
    is worth a MEASURED number rather than an assumption."""
    return (len(json.dumps(rec, indent=1).encode()),
            len(json.dumps(rec, separators=(",", ":")).encode()))


def contour_err(rec):
    """|polygon area - mask area| / mask area, per detection.

    ⚠️ Signed on purpose in the raw list: RDP can only LOSE detail, but the
    contour keeps the OUTER loop only, so a detection with a hole comes out
    LARGER than its mask. Averaging the magnitudes would hide which of the two
    is happening, and they mean different things."""
    errs, holes = [], 0
    for f in (rec.get("frames") or {}).values():
        for key in ("det", "scene"):
            for d in f.get(key) or []:
                a, c = d.get("mask_area_px"), d.get("contour_area_px")
                if not a or c is None:
                    continue
                errs.append((c - a) / a)
                holes += int((d.get("contour_n_loops") or 1) > 1)
    return errs, holes


frames_by, res = {}, {}
for cid in clips:
    frames_by[cid] = ph0_pilot.sample_clip_frames(
        str(WORK / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]

for name, conf, scene, contours in ARMS:
    proc.confidence_threshold = conf
    assert proc.confidence_threshold == conf
    torch.cuda.reset_peak_memory_stats()
    arm = {"conf": conf, "scene_concepts": scene, "contours": contours,
           "clips": {}}
    t_arm = time.time()
    for cid in clips:
        t0 = time.time()
        rec = L.sam3_leg(proc, frames_by[cid], v2_by[cid],
                         frame_stride=FRAME_STRIDE, scene_concepts=scene,
                         contours=contours, meta=dict(meta, **{
                             "confidence_threshold": conf}))
        wall = time.time() - t0
        ind, comp = sizes(rec)
        errs, holes = contour_err(rec)
        arm["clips"][cid] = {
            "wall_s": round(wall, 2),
            "n_frames_run": rec["n_frames_run"],
            "n_det_total": rec["n_det_total"],
            "n_scene_det_total": rec.get("n_scene_det_total", 0),
            "n_err_total": rec.get("n_err_total"),
            "bytes_indent1": ind, "bytes_compact": comp,
            "per_concept_hits": rec["per_concept_hits"],
            "per_scene_hits": rec.get("per_scene_hits"),
            "liveness_n_det": (rec.get("liveness") or {}).get("n_det"),
            "n_contours": len(errs),
            "contour_err_med": round(statistics.median(errs), 5) if errs
            else None,
            "contour_err_p90": round(sorted(errs)[int(0.9 * (len(errs) - 1))],
                                     5) if errs else None,
            "contour_err_max_abs": round(max(abs(e) for e in errs), 5)
            if errs else None,
            "n_multiloop": holes,
            "ego_lane_frames_bounded": sum(
                1 for v in ((rec.get("ego_lane") or {}).get("frames")
                            or {}).values() if v.get("lane_idx_est")
                is not None),
        }
        if name == "C_v2_conf025":
            # keep ONE whole record so the schema is auditable off-GPU
            if "sample_record_clip" not in arm:
                arm["sample_record_clip"] = cid
                json.dump(rec, open(OUT / f"p2_sample_{cid[:8]}.json", "w"),
                          indent=1)
        print(f"[{name}] {cid[:8]} {wall:5.1f}s · det {rec['n_det_total']:3d} "
              f"· scene {rec.get('n_scene_det_total', 0):3d} · "
              f"{ind:6d} B (indent1) / {comp:6d} B (compact)", flush=True)
    arm["wall_s_total"] = round(time.time() - t_arm, 1)
    arm["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    cs = arm["clips"]
    arm["mean_wall_s"] = round(sum(c["wall_s"] for c in cs.values())
                               / len(cs), 2)
    arm["mean_bytes_indent1"] = int(sum(c["bytes_indent1"]
                                        for c in cs.values()) / len(cs))
    arm["mean_bytes_compact"] = int(sum(c["bytes_compact"]
                                        for c in cs.values()) / len(cs))
    arm["det_total"] = sum(c["n_det_total"] for c in cs.values())
    arm["scene_total"] = sum(c["n_scene_det_total"] for c in cs.values())
    res[name] = arm
    print(f"[{name}] mean {arm['mean_wall_s']}s/clip · "
          f"{arm['mean_bytes_indent1']} B/clip (indent1) · "
          f"{arm['mean_bytes_compact']} B (compact) · det {arm['det_total']} "
          f"· scene {arm['scene_total']} · peak {arm['peak_gpu_gb']} GB",
          flush=True)

A, B, C = res["A_v1_conf050"], res["B_v1_conf025"], res["C_v2_conf025"]
growth = {
    "wall_B_over_A": round(B["mean_wall_s"] / A["mean_wall_s"], 3),
    "wall_C_over_B": round(C["mean_wall_s"] / B["mean_wall_s"], 3),
    "wall_C_over_A": round(C["mean_wall_s"] / A["mean_wall_s"], 3),
    "bytes_B_over_A": round(B["mean_bytes_indent1"]
                            / A["mean_bytes_indent1"], 3),
    "bytes_C_over_B": round(C["mean_bytes_indent1"]
                            / B["mean_bytes_indent1"], 3),
    "bytes_C_over_A": round(C["mean_bytes_indent1"]
                            / A["mean_bytes_indent1"], 3),
    "bytes_C_over_A_compact": round(C["mean_bytes_compact"]
                                    / A["mean_bytes_indent1"], 3),
    "det_B_over_A": round(B["det_total"] / max(1, A["det_total"]), 3),
    "projected_115_gpu_min_C": round(115 * C["mean_wall_s"] / 60, 1),
}
out = {"class": "MEASURED", "host": "colab T4", "n_clips": len(clips),
       "clips": clips, "frame_stride": FRAME_STRIDE,
       "contour_tol_px": ph0_sam3.CONTOUR_TOL_PX_DEFAULT,
       "contour_max_pts": ph0_sam3.CONTOUR_MAX_PTS_DEFAULT,
       "scene_concepts": ph0_sam3.SCENE_CONCEPTS,
       "arms": res, "growth": growth}
json.dump(out, open(OUT / "p2_pilot_size.json", "w"), indent=1)
print("\n[p2] GROWTH:", json.dumps(growth, indent=1))
print("P2_DONE")
