"""Rig-B masked-periphery fraction — a CENSUS, not an n=6 sample.

The ~9 % figure in FLEET_REFILL.md 3.4 is n=6 with ONE rig-A clip. That is
DIRECTIONAL, not a rate. This measures it two ways:

  A. GEOMETRY CENSUS (every clip on this host). ``observed_frac`` is a property
     of the per-clip intrinsics + the requested frame, NOT of the pixels: it
     comes from the ray map, so a ``torch.zeros`` probe gives the EXACT value
     with no decode. n = every clip with per-clip intrinsics.
  B. REAL-DECODE SUBSAMPLE. Decodes real mp4s and counts genuinely-zero output
     pixels, which is the honest upper bound (it also catches real black
     pixels). Validates that A predicts B.

Rig is assigned from the per-clip principal point cy (rig A ~543, rig B ~755),
the split the D-016 R1 two-rig fix uses. The boundary is derived from the
largest gap in the observed cy distribution and its bimodality is CHECKED, not
assumed.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path


def _agg(rows, key):
    v = [r[key] for r in rows if key in r]
    if not v:
        return None
    return {"n": len(v), "mean": round(st.mean(v), 6),
            "median": round(st.median(v), 6), "min": round(min(v), 6),
            "max": round(max(v), 6),
            "stdev": round(st.stdev(v), 6) if len(v) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--projection-mode", default="cylindrical")
    ap.add_argument("--decode-n", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tanitad.data.calib import (CanonicalFrame, cylindrical_rectify,
                                    ftheta_crop_resize)
    from tanitad.data.physicalai import (_decode_mp4, discover_r0_clips,
                                         intrinsics_for_clip)

    proj = "cylindrical" if a.projection_mode == "cylindrical" else "pinhole"
    frame = CanonicalFrame.from_hfov(a.hfov, a.height, a.width, proj)
    clips = discover_r0_clips(a.root)
    out: dict = {
        "request": {"hfov_deg": a.hfov, "height": a.height, "width": a.width,
                    "projection_mode": a.projection_mode},
        "frame": {**frame.to_dict(), "tag": frame.tag(),
                  "hfov_deg": float(frame.hfov_deg)},
        "clips_available": len(clips),
        "geometry_census": [], "real_decode": [],
    }

    # ---- A. geometry census over EVERY clip on this host -------------------
    probe_cache: dict = {}
    t0 = time.time()
    for i, c in enumerate(clips):
        cid = c["clip_id"]
        try:
            intr = intrinsics_for_clip(cid, a.root)
        except Exception as e:
            out["geometry_census"].append(
                {"i": i, "error": type(e).__name__ + ": " + str(e)})
            continue
        cy = float(getattr(intr, "cy", float("nan")))
        cx = float(getattr(intr, "cx", float("nan")))
        # one probe per DISTINCT sensor geometry — the ray map is identical
        k = (round(cx, 3), round(cy, 3), int(intr.height), int(intr.width),
             bool(intr.per_clip))
        if k not in probe_cache:
            probe = torch.zeros(1, 3, intr.height, intr.width, dtype=torch.uint8)
            if proj == "cylindrical":
                cylindrical_rectify(probe, intr, frame,
                                    require_per_clip=intr.per_clip)
                obs = float(cylindrical_rectify.last_observed_frac)
                feff = float(cylindrical_rectify.last_f_eff)
            else:
                ftheta_crop_resize(probe, intr, frame=frame)
                obs = 1.0
                feff = float(ftheta_crop_resize.last_f_eff)
            probe_cache[k] = (obs, feff)
        obs, feff = probe_cache[k]
        out["geometry_census"].append(
            {"i": i, "cx": cx, "cy": cy, "per_clip": bool(intr.per_clip),
             "sensor_hw": [int(intr.height), int(intr.width)],
             "observed_frac": obs, "masked_frac": 1.0 - obs, "f_eff": feff})
    out["geometry_census_seconds"] = round(time.time() - t0, 1)
    out["distinct_sensor_geometries"] = len(probe_cache)

    # ---- rig assignment, with a bimodality CHECK ---------------------------
    cys = sorted(r["cy"] for r in out["geometry_census"]
                 if "cy" in r and not math.isnan(r["cy"]))
    if cys:
        gaps = [(cys[i + 1] - cys[i], i) for i in range(len(cys) - 1)]
        gmax, gi = max(gaps) if gaps else (0.0, 0)
        boundary = (cys[gi] + cys[gi + 1]) / 2.0 if gaps else cys[0]
        out["rig_boundary"] = {"cy_min": cys[0], "cy_max": cys[-1],
                               "largest_gap": round(gmax, 2),
                               "boundary_cy": round(boundary, 2),
                               "bimodal": bool(gmax > 50.0)}
        for r in out["geometry_census"]:
            if "cy" in r and not math.isnan(r["cy"]):
                r["rig"] = "A" if r["cy"] < boundary else "B"

    ok = [r for r in out["geometry_census"] if "rig" in r]
    out["by_rig"] = {
        "A": {"n": sum(1 for r in ok if r["rig"] == "A"),
              "masked_frac": _agg([r for r in ok if r["rig"] == "A"],
                                  "masked_frac"),
              "cy": _agg([r for r in ok if r["rig"] == "A"], "cy")},
        "B": {"n": sum(1 for r in ok if r["rig"] == "B"),
              "masked_frac": _agg([r for r in ok if r["rig"] == "B"],
                                  "masked_frac"),
              "cy": _agg([r for r in ok if r["rig"] == "B"], "cy")},
        "pooled": {"n": len(ok), "masked_frac": _agg(ok, "masked_frac")},
    }

    # ---- B. real-decode subsample, balanced across rigs --------------------
    byrig = {"A": [r["i"] for r in ok if r["rig"] == "A"],
             "B": [r["i"] for r in ok if r["rig"] == "B"]}
    pick: list[int] = []
    for rg in ("A", "B"):
        lst = byrig[rg]
        want = min(a.decode_n // 2, len(lst))
        if want:
            pick += [lst[round(j * (len(lst) - 1) / max(want - 1, 1))]
                     for j in range(want)]
    pick = sorted(set(pick))
    rig_of = {r["i"]: r["rig"] for r in ok}
    for i in pick:
        c = clips[i]
        rec = {"i": i, "rig": rig_of[i]}
        try:
            t1 = time.time()
            vid = _decode_mp4(Path(c["mp4"]), size=a.height, frame=frame,
                              projection_mode=a.projection_mode)
            rec["decode_s"] = round(time.time() - t1, 3)
            rec["frames"] = int(vid.shape[0])
            rec["out_shape"] = list(vid.shape[1:])
            allzero = (vid == 0).all(dim=1)          # [n,H,W], all 3 channels 0
            rec["zero_pixel_frac"] = round(float(allzero.float().mean()), 6)
            rec["s_per_frame"] = round(rec["decode_s"] / max(rec["frames"], 1), 5)
        except Exception as e:
            rec["error"] = type(e).__name__ + ": " + str(e)
        out["real_decode"].append(rec)
        print("[decode] " + json.dumps(rec), flush=True)

    dec = [r for r in out["real_decode"] if "zero_pixel_frac" in r]
    out["real_decode_by_rig"] = {
        rg: _agg([r for r in dec if r["rig"] == rg], "zero_pixel_frac")
        for rg in ("A", "B")}
    out["decode_s_per_frame"] = _agg(dec, "s_per_frame")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("RIG_MASK_CENSUS_DONE " + json.dumps({
        "clips": out["clips_available"],
        "n_A": out["by_rig"]["A"]["n"], "n_B": out["by_rig"]["B"]["n"],
        "masked_A": (out["by_rig"]["A"]["masked_frac"] or {}).get("mean"),
        "masked_B": (out["by_rig"]["B"]["masked_frac"] or {}).get("mean"),
        "real_decode_by_rig": out["real_decode_by_rig"],
        "decode_s_per_frame": out["decode_s_per_frame"]}), flush=True)


if __name__ == "__main__":
    main()
