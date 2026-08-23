"""Wide-FOV REAL-FRAME validation — the gap the preflight leaves open.

`wfov_preflight.py` (and `build_pai_cache.py`'s own pre-decode assert) validate
the geometry against a `torch.zeros` probe. That proves the ray maths and the
delivered field, but it decodes NOTHING — so it cannot catch a failure in the
actual mp4 -> resample path, and it measures `observed_frac` for exactly ONE
clip.

This decodes a handful of REAL clips at the requested geometry and reports:
  * that the decode path works end-to-end at a non-canonical frame,
  * the per-clip distribution of `observed_frac` (the masked periphery),
  * the per-clip delivered `f_eff` / HFOV (per-clip intrinsics vary, and this
    corpus has TWO camera rigs, so a single clip is not representative),
  * decode wall-clock per clip, for a grounded build-time estimate.

⛔ Writes NOTHING into `_epcache` and mints no corpus key. It is a read-only
probe, deliberately, because on a host that lacks the full raw corpus a cache
build would be an episode RE-SELECTION and must be refused.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--projection-mode", default="cylindrical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.physicalai import (_decode_mp4, discover_r0_clips,
                                         intrinsics_for_clip)

    proj = "cylindrical" if a.projection_mode == "cylindrical" else "pinhole"
    frame = CanonicalFrame.from_hfov(a.hfov, a.height, a.width, proj)
    clips = discover_r0_clips(a.root)
    res: dict = {
        "request": {"height": a.height, "width": a.width, "hfov_deg": a.hfov,
                    "projection_mode": a.projection_mode,
                    "clips_probed": min(a.clips, len(clips)),
                    "clips_available": len(clips)},
        "frame": {"height": frame.height, "width": frame.width,
                  "f_ref": float(frame.f_ref), "projection": frame.projection,
                  "hfov_deg": float(frame.hfov_deg)},
        "per_clip": [],
    }

    # spread the sample across the clip list so both camera rigs are represented
    n = min(a.clips, len(clips))
    idx = [round(i * (len(clips) - 1) / max(n - 1, 1)) for i in range(n)]
    for i in idx:
        c = clips[i]
        cid = c["clip_id"]
        rec: dict = {"clip_index": i}
        try:
            intr = intrinsics_for_clip(cid, a.root)
            rec["sensor_hw"] = [int(intr.height), int(intr.width)]
            rec["cy"] = float(getattr(intr, "cy", float("nan")))
            rec["per_clip"] = bool(intr.per_clip)
            t0 = time.time()
            vid = _decode_mp4(Path(c["mp4"]), size=a.height, frame=frame,
                              projection_mode=a.projection_mode)
            dt = time.time() - t0
            rec["decode_s"] = round(dt, 3)
            rec["frames"] = int(vid.shape[0])
            rec["out_shape"] = list(vid.shape[1:])
            rec["s_per_frame"] = round(dt / max(int(vid.shape[0]), 1), 5)
            if a.projection_mode == "cylindrical":
                f_eff = float(cylindrical_rectify.last_f_eff)
                obs = float(cylindrical_rectify.last_observed_frac)
                rec["f_eff"] = f_eff
                rec["observed_frac"] = obs
                rec["hfov_deg"] = math.degrees(2 * (frame.width / 2) / f_eff)
            # honest black for unobserved pixels -> measure how much is black
            v0 = vid[0].float()
            rec["zero_pixel_frac"] = round(
                float((v0.sum(0) == 0).float().mean()), 5)
            rec["mean_luma"] = round(float(v0.mean()), 3)
            rec["ok"] = True
        except Exception as e:                                # noqa: BLE001
            rec["ok"] = False
            rec["error"] = repr(e)[:400]
        # NOTE: clip_id deliberately NOT recorded — PhysicalAI-AV is gated and
        # clip identifiers must not leave the pod.
        res["per_clip"].append(rec)

    ok = [r for r in res["per_clip"] if r.get("ok")]
    if ok:
        def agg(k):
            v = [r[k] for r in ok if k in r]
            if not v:
                return None
            return {"min": round(min(v), 5), "max": round(max(v), 5),
                    "mean": round(st.mean(v), 5),
                    "stdev": round(st.stdev(v), 5) if len(v) > 1 else 0.0}
        res["summary"] = {
            "n_ok": len(ok), "n_fail": len(res["per_clip"]) - len(ok),
            "observed_frac": agg("observed_frac"),
            "f_eff": agg("f_eff"),
            "zero_pixel_frac": agg("zero_pixel_frac"),
            "s_per_frame": agg("s_per_frame"),
            "decode_s": agg("decode_s"),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("summary", {}), indent=1), flush=True)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
