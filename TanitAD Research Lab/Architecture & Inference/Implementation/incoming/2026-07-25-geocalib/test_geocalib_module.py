"""Pod test for geocalib_intrinsics.py — proves the importable interface + the
drop-in geometry path run on REAL video (MEASURED). Also surfaces the crop-size /
retained-field difference vs the pilot's fixed-100deg HFOV (the geometry the swap
actually changes)."""
import glob, json, math
import numpy as np
from PIL import Image

import geocalib_intrinsics as gi
from tanitad.data.calib import focal_crop_resize, focal_crop_size, nominal_focal_from_hfov

FR = "gc_frames"


def load_pngs(d):
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(glob.glob(f"{FR}/{d}/*.png"))]


class StubAnon:
    """identity anonymizer for the geometry test (real pipeline uses yt_pilot_common.Anonymizer)."""
    stats = {"faces": 0, "plates": 0, "bodies": 0, "frames": 0}
    def __call__(self, rgb):
        self.stats["frames"] += 1
        return rgb


def crop_report(est, W, H, hfov_fixed=100.0):
    f_gc = est.focal_px(W, H)
    c_gc = focal_crop_size(f_gc, H, W, 256)
    f_fx = nominal_focal_from_hfov(W, hfov_fixed)
    c_fx = focal_crop_size(f_fx, H, W, 256)
    return {"focal_geocalib_px": round(f_gc, 1), "crop_side_geocalib": c_gc,
            "retained_width_frac_geocalib": round(c_gc / W, 3),
            "focal_fixed100_px": round(f_fx, 1), "crop_side_fixed100": c_fx,
            "retained_width_frac_fixed100": round(c_fx / W, 3),
            "crop_ratio_fixed_over_gc": round(c_fx / c_gc, 3)}


def main():
    est = gi.GeoCalibEstimator()   # distorted, cuda

    print("=== estimate_from_frames: comma_native (12) — GT vfov 51.28 deg ===")
    e_comma = est.estimate_from_frames(load_pngs("comma_native"))
    print(json.dumps({k: v for k, v in e_comma.as_dict().items() if k != "per_frame_vfov_deg"}, indent=2))
    print("  crop vs fixed-100:", json.dumps(crop_report(e_comma, e_comma.est_width, e_comma.est_height)))

    print("=== estimate_from_frames: physicalai_native (8) — 120deg fisheye ===")
    e_pai = est.estimate_from_frames(load_pngs("physicalai_native"))
    print(json.dumps({k: v for k, v in e_pai.as_dict().items() if k != "per_frame_vfov_deg"}, indent=2))

    print("=== estimate_from_video: real comma video.hevc (16 frames) ===")
    e_vid = est.estimate_from_video("testvid_comma.hevc", n_frames=16)
    print(json.dumps({k: v for k, v in e_vid.as_dict().items() if k != "per_frame_vfov_deg"}, indent=2))
    print("  crop vs fixed-100:", json.dumps(crop_report(e_vid, e_vid.est_width, e_vid.est_height)))

    print("=== estimate_from_video: real physicalai mp4 (16 frames) ===")
    e_pvid = est.estimate_from_video("testvid_pai.mp4", n_frames=16)
    print(json.dumps({k: v for k, v in e_pvid.as_dict().items() if k != "per_frame_vfov_deg"}, indent=2))

    print("=== decode_canonical_geocalib drop-in on real comma video (8 frames) ===")
    frames_u8, meta = gi.decode_canonical_geocalib(
        "testvid_comma.hevc", StubAnon(), estimator=est, max_frames=8)
    print("  out shape:", tuple(frames_u8.shape), "dtype", frames_u8.dtype)
    print("  focal_crop_resize.last_f_eff:", round(float(focal_crop_resize.last_f_eff), 2),
          "(canonical target 266)")
    print("  meta.geometry:", meta["geometry"], "| fallback:", meta["geocalib_fallback_used"],
          "| hfov_used:", round(meta["hfov_used_deg"], 2))
    assert tuple(frames_u8.shape)[1:] == (3, 256, 256), "canonical shape wrong"
    assert abs(float(focal_crop_resize.last_f_eff) - 266.0) / 266.0 < 0.02, "f_eff not canonical"
    print("\nALL MODULE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
