"""P4 — MEASURE GeoCalib on the pilot's REAL YouTube clips vs the fixed-100deg HFOV.

Re-downloads a bounded sample of the pilot's CC videos (from the staged
pointers.jsonl), runs GeoCalib per video, and quantifies how far the true FoV is
from the assumed 100 deg and how much the fixed assumption mis-crops (the geometry
error the swap removes). PRIVACY: no imagery persists — each mp4 is deleted right
after estimation; only intrinsics NUMBERS are written.

Output: youtube_geocalib_measurement.json
"""
from __future__ import annotations
import json, math, os, sys, time, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geocalib_intrinsics as gi
from tanitad.data.calib import focal_crop_size, nominal_focal_from_hfov

CC = "Creative Commons Attribution license (reuse allowed)"
TMP = HERE / "_ytmp"
TMP.mkdir(exist_ok=True)
MAX_VIDEOS = int(os.environ.get("MAX_VIDEOS", "12"))
ASSUMED_HFOV = 100.0


def download(url):
    from yt_dlp import YoutubeDL
    out = str(TMP / "%(id)s.%(ext)s")
    opts = {"format": "best[height<=480][ext=mp4]/best[height<=480]/worst[ext=mp4]/worst",
            "outtmpl": out, "quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 30, "retries": 2}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fp = ydl.prepare_filename(info)
        if not os.path.exists(fp):
            cands = glob.glob(str(TMP / (info.get("id", "*") + ".*")))
            fp = cands[0] if cands else None
        return fp, info.get("license"), info.get("id")


def crop_geometry(est, W, H):
    f_gc = est.focal_px(W, H)
    c_gc = focal_crop_size(f_gc, H, W, 256)
    f_fx = nominal_focal_from_hfov(W, ASSUMED_HFOV)
    c_fx = focal_crop_size(f_fx, H, W, 256)
    return {"focal_geocalib_px": round(f_gc, 1), "crop_side_geocalib": c_gc,
            "retained_w_frac_geocalib": round(c_gc / W, 3),
            "focal_fixed100_px": round(f_fx, 1), "crop_side_fixed100": c_fx,
            "retained_w_frac_fixed100": round(c_fx / W, 3),
            # >1 => fixed-100 crops TIGHTER than GeoCalib (inflates apparent motion)
            "linear_zoom_fixed_over_gc": round(c_gc / c_fx, 3)}


def main():
    recs = [json.loads(l) for l in open(HERE / "pointers.jsonl", encoding="utf-8") if l.strip()]
    vids = []
    seen = set()
    for r in recs:
        vid = r.get("video_id")
        if vid and vid not in seen and r.get("is_cc"):
            seen.add(vid); vids.append((vid, r.get("url")))
    print(f"unique CC videos in pointers: {len(vids)}; sampling up to {MAX_VIDEOS}")

    est = gi.GeoCalibEstimator()   # distorted, cuda
    results = []
    tried = 0
    for vid, url in vids:
        if len(results) >= MAX_VIDEOS:
            break
        tried += 1
        try:
            fp, lic, gotid = download(url)
        except Exception as e:
            print(f"  [dl-fail] {vid}: {str(e)[:80]}"); continue
        if not fp or not os.path.exists(fp):
            print(f"  [dl-none] {vid}"); continue
        if lic != CC:
            print(f"  [not-cc] {vid}: {lic}"); os.remove(fp); continue
        try:
            e = est.estimate_from_video(fp, n_frames=16)
            geo = crop_geometry(e, e.est_width, e.est_height)
            rec = {"video_id": vid, "url": url, "license_ok": True,
                   "est_res": [e.est_width, e.est_height],
                   "vfov_deg": round(e.vfov_deg, 2), "hfov_deg": round(e.hfov_deg, 2),
                   "vfov_mad_deg": round(e.vfov_mad_deg, 2) if e.vfov_mad_deg == e.vfov_mad_deg else None,
                   "confidence": e.confidence, "fallback_used": e.fallback_used,
                   "n_used": e.n_frames_used, **geo}
            results.append(rec)
            print(f"  [ok] {vid}: hfov {rec['hfov_deg']:.1f} (assumed {ASSUMED_HFOV}) "
                  f"conf {e.confidence} zoom_fixed/gc {geo['linear_zoom_fixed_over_gc']}")
        except Exception as e:
            print(f"  [est-fail] {vid}: {str(e)[:80]}")
        finally:
            try:
                os.remove(fp)     # PRIVACY: no imagery persists
            except OSError:
                pass

    # aggregate
    ok = [r for r in results if not r["fallback_used"]]
    def stats(key, rows):
        xs = [r[key] for r in rows]
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
        return {"n": n, "mean": round(mean, 2), "median": round(med, 2),
                "min": round(xs[0], 2), "max": round(xs[-1], 2)}
    summary = {
        "assumed_hfov_deg": ASSUMED_HFOV, "n_videos_tried": tried,
        "n_measured": len(results), "n_geocalib_confident": len(ok),
        "hfov_deg": stats("hfov_deg", results),
        "hfov_deg_confident_only": stats("hfov_deg", ok),
        "linear_zoom_fixed_over_gc": stats("linear_zoom_fixed_over_gc", results),
        "confidence_counts": {c: sum(1 for r in results if r["confidence"] == c)
                              for c in ("high", "medium", "low")},
        "n_fixed100_within_10deg": sum(1 for r in results if abs(r["hfov_deg"] - ASSUMED_HFOV) <= 10),
    }
    out = {"summary": summary, "per_video": results}
    (HERE / "youtube_geocalib_measurement.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("WROTE youtube_geocalib_measurement.json")
    # cleanup tmp
    for f in glob.glob(str(TMP / "*")):
        try: os.remove(f)
        except OSError: pass


if __name__ == "__main__":
    main()
