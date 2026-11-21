"""Wide-FOV cache PREFLIGHT — the cheap gate that runs before a multi-hour decode.

This mirrors the geometry self-check inside `stack/scripts/build_pai_cache.py`
(the branch that runs when the frame is non-canonical) WITHOUT building anything,
plus it measures the parity precondition that decides whether a rebuild on this
host is admissible at all.

It answers three questions and writes them as raw JSON:

  1. Is the requested field actually DELIVERED at this frame size, or does the
     resampler clamp at the sensor edge and silently ZOOM instead of widening?
     (MEASURED precedent: a 100 deg SQUARE 256 frame delivers only 67.1 deg.)
  2. What build params / geometry key fragment would the rebuild mint?
  3. Does this host hold the raw sources for the FULL canonical corpus? The
     parity sibling is minted only if uid digest + count + skip indices match
     `physicalai-train-e438721ae894` exactly, so a host with a partial clip set
     CANNOT produce a parity-preserving cache -- and building anyway would be an
     episode RE-SELECTION, which the program's parity rule requires be refused.

Deliberately read-only: it decodes nothing and writes only its own JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--projection-mode", default="cylindrical")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch  # noqa: F401
    from tanitad.config import base250cam_config
    from tanitad.geometry import apply_frame, geometry_report
    from tanitad.data.calib import (CanonicalFrame, cylindrical_rectify,
                                    ftheta_crop_resize)
    from tanitad.data.physicalai import (discover_r0_clips, geometry_build_params,
                                         intrinsics_for_clip)

    res: dict = {"request": {"height": a.height, "width": a.width,
                             "hfov_deg": a.hfov,
                             "projection_mode": a.projection_mode,
                             "patch": a.patch, "root": a.root}}

    cfg = base250cam_config()
    clips = discover_r0_clips(a.root)
    res["clips_discovered"] = len(clips)

    # ---- geometry delivery -------------------------------------------------
    proj = "cylindrical" if a.projection_mode == "cylindrical" else "pinhole"
    frame = CanonicalFrame.from_hfov(a.hfov, a.height, a.width, proj)
    res["frame"] = {k: getattr(frame, k) for k in
                    ("height", "width", "hfov_deg", "f_ref", "projection")
                    if hasattr(frame, k)}
    res["frame"]["is_canonical"] = bool(frame.is_canonical)
    res["tokens_at_patch"] = (a.height // a.patch) * (a.width // a.patch)
    try:
        apply_frame(cfg, frame)
        res["geometry_report"] = geometry_report(cfg)
    except Exception as e:                                    # noqa: BLE001
        res["geometry_report_error"] = repr(e)[:400]

    if clips:
        cid = clips[0]["clip_id"]
        intr = intrinsics_for_clip(cid, a.root)
        probe = torch.zeros(1, 3, intr.height, intr.width, dtype=torch.uint8)
        if a.projection_mode == "cylindrical":
            cylindrical_rectify(probe, intr, frame,
                                require_per_clip=intr.per_clip)
            got_f = cylindrical_rectify.last_f_eff
            obs = cylindrical_rectify.last_observed_frac
        else:
            ftheta_crop_resize(probe, intr, frame=frame)
            got_f = ftheta_crop_resize.last_f_eff
            obs = 1.0
        got_hfov = math.degrees(2 * (
            math.atan((frame.width / 2) / got_f)
            if frame.projection == "pinhole" else (frame.width / 2) / got_f))
        res["delivered"] = {
            "sensor_hw": [int(intr.height), int(intr.width)],
            "per_clip_intrinsics": bool(intr.per_clip),
            "f_eff": float(got_f),
            "hfov_deg": float(got_hfov),
            "observed_frac": float(obs),
            "shortfall_deg": float(frame.hfov_deg - got_hfov),
            # this is the assertion build_pai_cache.py makes before decoding
            "PASSES_BUILD_ASSERT": bool(abs(got_hfov - frame.hfov_deg) < 0.5),
        }
        res["build_params"] = {
            "size": cfg.encoder.image_size, "n_stack": 3, "hz": 10,
            "calib": "ftheta_v2",
            **geometry_build_params(frame, a.projection_mode),
        }

    # ---- parity precondition ----------------------------------------------
    cache = Path(a.root) / "_epcache"
    canon = cache / "physicalai-train-e438721ae894"
    done = canon / "DONE"
    par: dict = {"canonical_dir": str(canon), "exists": canon.is_dir()}
    if done.is_file():
        try:
            par["DONE"] = json.loads(done.read_text())
        except Exception:                                    # noqa: BLE001
            par["DONE_raw"] = done.read_text()[:200]
    if canon.is_dir():
        par["ep_files"] = len(list(canon.glob("ep_*.pt")))
    need = par.get("DONE", {}).get("episodes")
    skipped = par.get("DONE", {}).get("skipped")
    par["clips_present_on_host"] = len(clips)
    if need is not None:
        par["clips_required_min"] = need + (skipped or 0)
        par["parity_rebuild_possible"] = len(clips) >= (need + (skipped or 0))
        par["coverage_frac"] = round(len(clips) / (need + (skipped or 0)), 4)
    res["parity"] = par

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
