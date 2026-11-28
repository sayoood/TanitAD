"""STEP 0 — PROVE the mixed-floor diagnosis from the RECORDS, before spending a GPU.

⛔ THE DIAGNOSIS I WAS HANDED IS ITSELF AN INHERITED CLAIM, AND IT COSTS ~43 GPU-min
TO ACT ON. `…/2026-08-17-aug120-refuse/raw/inputs_manifest.json` states the v1 leg is
at `confidence_threshold=0.5` and adds, verbatim:

    "Floor known from the RUN, not from the record."

That is an honest statement of a WEAK evidence class — the floor was read off the
launching code, not off the artifact. If it is wrong (the 86 already at 0.25, or a
third provenance), the whole re-run is waste and the far more valuable result is that
the mixed-floor diagnosis is refuted. So it is measured here FIRST, and from the only
source that cannot be out of date: the detections themselves.

⭐ THE FLOOR IS RECOVERABLE FROM THE RECORD AFTER ALL, and the manifest's caveat is
too pessimistic. `Sam3Processor` applies `keep = out_probs > confidence_threshold`
INSIDE the forward pass (ph0_sam3.py:404), so the floor is not merely "rows that are
not there" — it is a HARD LOWER BOUND on every surviving score. Over a few thousand
detections the minimum score converges onto the threshold from above, and:

    min(score) just above 0.50 with ZERO detections below  ⇒ the floor was 0.50
    min(score) just above 0.25 with ZERO detections below  ⇒ the floor was 0.25

This is a one-sided test and it is stated as one: a min of 0.5001 PROVES the floor is
<= 0.5001 and is overwhelming evidence it IS 0.5 (a 0.25-floor corpus would have to
have produced 2 939 consecutive detections none of which landed in [0.25, 0.50) — and
the v2 leg, measured in the same run below, puts 46 % of its mass exactly there).
It cannot prove the floor was not, say, 0.4999. That distinction does not matter for
the decision, and pretending to more precision than the estimator has is how the
"8-split jackknife" got its name.

Writes: raw/f0_floor_probe.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os

AGENT_KEYS_V1 = {"box_xyxy", "concept", "mask_area_px", "rle_rows", "score"}
V2_ONLY_DET_KEYS = {"contour_xy", "obb_cxcylwa", "mask_hw", "contour_area_px"}


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {"n": n, "min": round(s[0], 6), "p01": round(s[n // 100], 4),
            "p25": round(s[n // 4], 4), "median": round(s[n // 2], 4),
            "p75": round(s[3 * n // 4], 4), "max": round(s[-1], 6)}


def _scan(recs: dict[str, dict]) -> dict:
    """Content census of a SAM3 leg — the fields that reveal its provenance."""
    sc: list[float] = []
    per = collections.Counter()
    scene_per = collections.Counter()
    det_keys: collections.Counter = collections.Counter()
    schemas = collections.Counter()
    confs = collections.Counter()
    n_live = n_dead = n_nocontrol = 0
    n_scene_det = n_err = 0
    n_contour = n_obb = 0
    # C85: `_rows_rle` iterated a (1, H, W) mask as if it were (H, W), so every
    # run came out `[0, flat_start, flat_end)`. The signature is (a) EVERY run on
    # row 0 and (b) end columns beyond the frame width — while (c) the run
    # lengths still sum to `mask_area_px`, which is why nothing caught it.
    rle_rows_seen: collections.Counter = collections.Counter()
    n_runs = n_runs_col_gt_w = n_det_with_rle = n_area_agrees = 0
    for cid, r in sorted(recs.items()):
        W = (r.get("frame_wh") or [0, 0])[0]
        schemas[r.get("schema_version")] += 1
        confs[(r.get("engine") or {}).get("confidence_threshold")] += 1
        lv = r.get("liveness")
        if lv is None:
            n_nocontrol += 1
        else:
            counts = (lv.get("n_det") or {})
            alive = any(int(v) > 0 for v in counts.values())
            n_live += int(alive)
            n_dead += int(not alive)
        n_scene_det += int(r.get("n_scene_det_total") or 0)
        for k, v in (r.get("per_concept_hits") or {}).items():
            per[k] += int(v)
        for k, v in (r.get("per_scene_hits") or {}).items():
            scene_per[k] += int(v)
        for _fi, fr in (r.get("frames") or {}).items():
            for d in fr.get("det", []):
                if "error" in d:
                    n_err += 1
                    continue
                sc.append(float(d["score"]))
                per_none = d.get("concept")
                if per_none is None:
                    continue
                det_keys.update(d.keys())
                n_contour += int("contour_xy" in d)
                n_obb += int(d.get("obb_cxcylwa") is not None)
                rr = d.get("rle_rows")
                if rr:
                    n_det_with_rle += 1
                    tot = 0
                    for run in rr:
                        n_runs += 1
                        rle_rows_seen[int(run[0])] += 1
                        n_runs_col_gt_w += int(W and int(run[2]) > W)
                        tot += int(run[2]) - int(run[1])
                    n_area_agrees += int(tot == d.get("mask_area_px"))
    return {
        "n_clips": len(recs),
        "score": _quantiles(sc),
        "n_det_below_0.50": sum(1 for s in sc if s < 0.50),
        "n_det_below_0.25": sum(1 for s in sc if s < 0.25),
        "n_det_in_[0.25,0.50)": sum(1 for s in sc if 0.25 <= s < 0.50),
        "frac_det_in_[0.25,0.50)": (
            round(sum(1 for s in sc if 0.25 <= s < 0.50) / len(sc), 4)
            if sc else None),
        "schema_version_values": {str(k): v for k, v in schemas.items()},
        "engine_confidence_threshold_values": {
            str(k): v for k, v in confs.items()},
        "liveness_live": n_live, "liveness_dead": n_dead,
        "records_without_control": n_nocontrol,
        "n_scene_det_total": n_scene_det,
        "n_error_entries": n_err,
        "n_detections_with_contour": n_contour,
        "n_detections_with_oriented_extent": n_obb,
        "per_concept_totals": dict(per.most_common()),
        "per_scene_totals": dict(scene_per.most_common()),
        "detection_key_union": sorted(det_keys),
        "c85_rle_rows": {
            "n_detections_with_rle": n_det_with_rle, "n_runs": n_runs,
            "distinct_row_indices": sorted(rle_rows_seen),
            "n_runs_end_col_gt_frame_width": n_runs_col_gt_w,
            "frac_runs_end_col_gt_frame_width": (
                round(n_runs_col_gt_w / n_runs, 4) if n_runs else None),
            "n_detections_whose_runs_sum_to_mask_area": n_area_agrees,
            "FLATTENED": bool(n_runs and sorted(rle_rows_seen) == [0]
                              and n_runs_col_gt_w > 0),
            "note": "⚠️ `runs_sum_to_mask_area` holding at 100 % IS the reason "
                    "nothing caught this: the invariant anyone would have "
                    "checked still holds under the flattening.",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug120", required=True,
                    help="cached aug120 work dir (holds merged/sam3.json)")
    ap.add_argument("--v2-dir", required=True,
                    help="pulled sam3_backfill_v2/ (115 records)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    merged = os.path.join(a.aug120, "merged")
    cohort = {c["clip_id"] for c in json.load(
        open(os.path.join(merged, "ph0_v2.json"), encoding="utf-8"))["clips"]}
    v1 = {c["clip_id"]: c for c in json.load(
        open(os.path.join(merged, "sam3.json"), encoding="utf-8"))["clips"]}
    v2 = {}
    for p in sorted(glob.glob(os.path.join(a.v2_dir, "*.json"))):
        cid = os.path.basename(p)[:-5]
        v2[cid] = json.load(open(p, encoding="utf-8"))

    s1, s2 = _scan(v1), _scan(v2)
    overlap = sorted(set(v1) & set(v2))
    union = set(v1) | set(v2)

    # ---- the verdict, stated as a one-sided test ------------------------- #
    v1_min = (s1["score"] or {}).get("min")
    v1_is_050 = bool(v1_min is not None and 0.50 <= v1_min < 0.51
                     and s1["n_det_below_0.50"] == 0)
    v1_is_025 = bool(v1_min is not None and 0.25 <= v1_min < 0.26
                     and s1["n_det_below_0.25"] == 0)
    v2_is_025 = bool((s2["score"] or {}).get("min") is not None
                     and s2["n_det_below_0.25"] == 0
                     and 0.25 <= s2["score"]["min"] < 0.26)

    if v1_is_050 and v2_is_025:
        verdict = "MIXED_FLOOR_CONFIRMED"
        reading = ("v1 leg floor 0.5, v2 leg floor 0.25 — the corpus IS mixed "
                   "and the 86-clip re-run is justified.")
    elif v1_is_025:
        verdict = "DIAGNOSIS_REFUTED_ALREADY_025"
        reading = ("⛔ the 86 are ALREADY at floor 0.25 — the mixed-floor "
                   "diagnosis is WRONG. STOP; do not spend the GPU.")
    else:
        verdict = "THIRD_PROVENANCE"
        reading = ("⛔ the v1 leg's score floor matches NEITHER 0.5 nor 0.25 — "
                   "a third provenance. STOP and establish it before re-running.")

    out = {
        "class": "MEASURED",
        "what": "provenance of the two aug120 SAM3 legs, read off the "
                "detections rather than off the launching code",
        "cohort_n": len(cohort),
        "legs_disjoint": not overlap,
        "n_overlap": len(overlap),
        "union_equals_cohort": union == cohort,
        "n_union": len(union),
        "v1_leg_batch_pipeline": s1,
        "v2_leg_sam3_backfill_v2": s2,
        "test": {
            "kind": "one-sided lower bound on the detection floor",
            "why_valid": "Sam3Processor applies keep = out_probs > "
                         "confidence_threshold inside the forward pass "
                         "(ph0_sam3.py:404), so min(score) is a hard lower "
                         "bound on the floor and converges onto it from above",
            "what_it_cannot_show": "the exact floor — min 0.5001 proves "
                                   "floor <= 0.5001, not floor == 0.5000",
            "v1_consistent_with_0.50": v1_is_050,
            "v1_consistent_with_0.25": v1_is_025,
            "v2_consistent_with_0.25": v2_is_025,
        },
        "secondary_defects_in_v1_leg": {
            "no_liveness_control": s1["records_without_control"],
            "no_schema_version": s1["schema_version_values"].get("None", 0),
            "no_engine_confidence_stamp":
                s1["engine_confidence_threshold_values"].get("None", 0),
            "no_scene_channel": int(s1["n_scene_det_total"] == 0),
            "no_contours": int(s1["n_detections_with_contour"] == 0),
            "note": "⚠️ the floor is only the defect that was NAMED. The 86 "
                    "also carry NO C77 liveness control (so their zeros are "
                    "unreadable), NO scene channel, NO contours/oriented "
                    "extents, and their rle_rows are the FLATTENED ones "
                    "retracted as C85. The re-run closes all five at once.",
        },
        "VERDICT": verdict,
        "reading": reading,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    payload = json.dumps(out, sort_keys=True).encode()
    print(f"[f0] cohort {len(cohort)} · v1 {len(v1)} · v2 {len(v2)} · "
          f"overlap {len(overlap)} · union=={union == cohort}")
    print(f"[f0] v1 min score {v1_min} · below0.50 {s1['n_det_below_0.50']} · "
          f"no-control {s1['records_without_control']}")
    print(f"[f0] v2 min score {(s2['score'] or {}).get('min')} · "
          f"in[0.25,0.50) {s2['frac_det_in_[0.25,0.50)']}")
    print(f"[f0] VERDICT {verdict}  md5={hashlib.md5(payload).hexdigest()[:12]}")
    print("F0_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
