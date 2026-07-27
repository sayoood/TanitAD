"""THE fully-observed sub-frame, per rig, measured from REAL per-clip intrinsics.

The defect (class C26): today's canonical crop replicate-pads rows that fall
outside the sensor on rig B, and the cylindrical projection converts that
fabrication into a rig-correlated MASK. Neither removes the rig-correlated
signal. The clean fix is a field BOTH rigs fully observe.

For a cylindrical frame the observed mask is a pure function of (per-clip
intrinsics, frame) via the ray map -- exactly the property `rig_mask_census.py`
exploits -- so this needs NO decode and runs over every clip whose calibration
is on the host.

⭐ WHY A 2-D SCAN AND NOT A ROW BAND. A CENTRED sub-rectangle of the built
frame is expressible as a `CanonicalFrame` (same f_ref, same projection) AND is
a pure slice of the pixels already built, because `cylindrical_rays` puts the
boresight at ((W-1)/2, (H-1)/2): rows [ (H-h)/2, (H+h)/2 ) of the parent carry
EXACTLY the v-coordinates of a height-h frame, and likewise for columns. So the
search space is {centred (h, w)} and every point in it is a zero-rebuild slice.
Height alone is not enough if the sensor also fails horizontally, which is why
both axes are scanned and why the instrument can return a FAILING value.

No clip UUID is written to any artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path


def _agg(vals):
    if not vals:
        return None
    return {"n": len(vals), "mean": round(st.mean(vals), 8),
            "median": round(st.median(vals), 8), "min": round(min(vals), 8),
            "max": round(max(vals), 8),
            "stdev": round(st.stdev(vals), 8) if len(vals) > 1 else 0.0}


def _runs(flags) -> list[tuple[int, int]]:
    """Maximal contiguous True runs of a bool list -> [(lo, hi), ...]."""
    out, lo = [], None
    for i, f in enumerate(flags):
        if f and lo is None:
            lo = i
        elif not f and lo is not None:
            out.append((lo, i - 1))
            lo = None
    if lo is not None:
        out.append((lo, len(flags) - 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--projection", default="cylindrical")
    ap.add_argument("--heights", default="256,224,192,176,160,144,128")
    ap.add_argument("--widths", default="640,624,608,592,576")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tanitad.data.calib import CanonicalFrame, cylindrical_grid
    from tanitad.data.physicalai import intrinsics_for_clip

    import pandas as pd
    sel = pd.read_parquet(Path(a.root) / "r0" / "r0_selection.parquet")
    clip_ids = [str(c) for c in sel["clip_id"].tolist()]
    if a.limit:
        clip_ids = clip_ids[:a.limit]

    parent = CanonicalFrame.from_hfov(a.hfov, a.height, a.width, a.projection)
    HS = [int(x) for x in a.heights.split(",")]
    WS = [int(x) for x in a.widths.split(",")]

    out: dict = {
        "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_frame": {**parent.to_dict(), "tag": parent.tag(),
                         "hfov_deg": float(parent.hfov_deg),
                         "vfov_deg": float(parent.vfov_deg)},
        "grid_heights": HS, "grid_widths": WS,
        "clips_in_r0_selection": len(clip_ids),
        "errors": 0, "error_kinds": {},
    }

    cache: dict = {}
    per_clip: list = []
    t0 = time.time()
    for cid in clip_ids:
        try:
            intr = intrinsics_for_clip(cid, a.root)
        except Exception as e:                                  # noqa: BLE001
            out["errors"] += 1
            k = type(e).__name__
            out["error_kinds"][k] = out["error_kinds"].get(k, 0) + 1
            continue
        if not intr.per_clip:
            out["errors"] += 1
            out["error_kinds"]["fallback_not_per_clip"] = \
                out["error_kinds"].get("fallback_not_per_clip", 0) + 1
            continue
        key = (round(float(intr.cx), 6), round(float(intr.cy), 6),
               tuple(round(float(p), 8) for p in intr.poly),
               int(intr.height), int(intr.width))
        if key not in cache:
            _, mask = cylindrical_grid(intr, int(intr.height), int(intr.width),
                                       parent)
            row_ok = [bool(x) for x in mask.all(dim=1).tolist()]
            col_ok = [bool(x) for x in mask.all(dim=0).tolist()]
            rrun, crun = _runs(row_ok), _runs(col_ok)
            # widest symmetric HFOV this sensor fully observes (v == 0 ray)
            half_px = min(float(intr.cx), float(intr.width) - 1 - float(intr.cx))
            th = intr.theta_of_r(half_px)
            # widest DOWNWARD half-field (phi == 0 ray), the rig-B constraint
            th_dn = intr.theta_of_r(float(intr.height) - 1 - float(intr.cy))
            th_up = intr.theta_of_r(float(intr.cy))
            rec = {
                "cx": float(intr.cx), "cy": float(intr.cy),
                "sensor_hw": [int(intr.height), int(intr.width)],
                "poly1": float(intr.poly[1]),
                "masked_frac_parent": float(1.0 - mask.float().mean()),
                "n_rows_fully_observed": sum(row_ok),
                "n_cols_fully_observed": sum(col_ok),
                "row_runs": rrun[:6], "col_runs": crun[:6],
                "max_hfov_deg_observed": math.degrees(2.0 * th),
                "down_halffield_deg_observed": math.degrees(th_dn),
                "up_halffield_deg_observed": math.degrees(th_up),
                "centre_row_masked_px": int((~mask[a.height // 2 - 1]).sum()),
                "bottom_row_masked_px": int((~mask[a.height - 1]).sum()),
                "top_row_masked_px": int((~mask[0]).sum()),
            }
            grid = {}
            for h in HS:
                r0 = (a.height - h) // 2
                for w in WS:
                    c0 = (a.width - w) // 2
                    sub = mask[r0:r0 + h, c0:c0 + w]
                    grid[f"{h}x{w}"] = float(1.0 - sub.float().mean())
            rec["centred_masked"] = grid
            cache[key] = rec
        per_clip.append(dict(cache[key]))
    out["seconds"] = round(time.time() - t0, 1)
    out["distinct_sensor_geometries"] = len(cache)
    out["n_clips_measured"] = len(per_clip)

    cys = sorted(r["cy"] for r in per_clip)
    gaps = [(cys[i + 1] - cys[i], i) for i in range(len(cys) - 1)]
    gmax, gi = max(gaps) if gaps else (0.0, 0)
    boundary = (cys[gi] + cys[gi + 1]) / 2.0 if gaps else (cys[0] if cys else 0.0)
    out["rig_boundary"] = {"cy_min": cys[0] if cys else None,
                           "cy_max": cys[-1] if cys else None,
                           "largest_gap": round(gmax, 3),
                           "boundary_cy": round(boundary, 3),
                           "bimodal": bool(gmax > 50.0)}
    for r in per_clip:
        r["rig"] = "A" if r["cy"] < boundary else "B"

    by = {}
    for rg in ("A", "B"):
        rows = [r for r in per_clip if r["rig"] == rg]
        if not rows:
            by[rg] = {"n": 0}
            continue
        by[rg] = {
            "n": len(rows),
            "share": round(len(rows) / max(len(per_clip), 1), 6),
            "cy": _agg([r["cy"] for r in rows]),
            "cx": _agg([r["cx"] for r in rows]),
            "poly1": _agg([r["poly1"] for r in rows]),
            "masked_frac_parent": _agg([r["masked_frac_parent"] for r in rows]),
            "max_hfov_deg_observed": _agg([r["max_hfov_deg_observed"]
                                           for r in rows]),
            "down_halffield_deg_observed": _agg(
                [r["down_halffield_deg_observed"] for r in rows]),
            "up_halffield_deg_observed": _agg(
                [r["up_halffield_deg_observed"] for r in rows]),
            "n_clips_with_horizontal_deficit": sum(
                1 for r in rows if r["centre_row_masked_px"] > 0),
            "centre_row_masked_px": _agg([float(r["centre_row_masked_px"])
                                          for r in rows]),
            # WORST case over the rig — this is what "the rig fully observes"
            "centred_masked_max": {k: max(r["centred_masked"][k] for r in rows)
                                   for k in rows[0]["centred_masked"]},
            "centred_masked_mean": {
                k: sum(r["centred_masked"][k] for r in rows) / len(rows)
                for k in rows[0]["centred_masked"]},
        }
    if per_clip:
        keys = list(per_clip[0]["centred_masked"])
        by["pooled"] = {
            "n": len(per_clip),
            "centred_masked_max": {k: max(r["centred_masked"][k]
                                          for r in per_clip) for k in keys},
            "max_hfov_deg_observed": _agg([r["max_hfov_deg_observed"]
                                           for r in per_clip]),
            "down_halffield_deg_observed": _agg(
                [r["down_halffield_deg_observed"] for r in per_clip]),
        }
    out["by_rig"] = by

    # ---- THE ANSWER: every centred (h, w) with mask EXACTLY 0 on both rigs --
    zero = [k for k, v in by["pooled"]["centred_masked_max"].items() if v == 0.0]
    def _area(k):
        h, w = k.split("x")
        return int(h) * int(w)
    zero.sort(key=_area, reverse=True)
    out["answer"] = {
        "zero_mask_centred_frames": zero,
        "largest_zero_mask_frame": zero[0] if zero else None,
        "zero_mask_exists": bool(zero),
        "tiling_ok_zero_mask": [k for k in zero
                                if int(k.split("x")[0]) % 64 == 0
                                and int(k.split("x")[1]) % 64 == 0],
    }
    out["distinct_probes"] = list(cache.values())
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("RIG_BAND_SCAN_DONE " + json.dumps({
        "n": out["n_clips_measured"], "n_A": by["A"].get("n"),
        "n_B": by["B"].get("n"), "distinct": out["distinct_sensor_geometries"],
        "answer": out["answer"], "seconds": out["seconds"],
        "errors": out["errors"], "error_kinds": out["error_kinds"]}), flush=True)


if __name__ == "__main__":
    main()
