"""P2 — GeoCalib intrinsics-recovery validation on KNOWN-intrinsics data.

Runs GeoCalib (pinhole + distorted weights) on the prepared known-GT frames and
measures estimated focal / vFoV vs ground truth:
  comma_native     : rectilinear pinhole, GT focal=910 px  (PRIMARY absolute test)
  comma_480p       : same, ~480p (YouTube-pilot res)       (resolution robustness)
  comma_focalsweep : exact known focal 910*s, s in [1,2]    (CONTROLLED relative test)
  physicalai_native: 120deg f-theta fisheye                 (distortion robustness)

Evidence class: MEASURED (this script + gc_frames/manifest.json GT).
Output: geocalib_validation_results.json
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

import numpy as np
import torch
from geocalib import GeoCalib

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "gc_frames"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def summarize(errs):
    errs = [e for e in errs if e is not None and not math.isnan(e)]
    if not errs:
        return {}
    a = np.array(errs, float)
    return {"n": len(a), "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "mean_abs": round(float(np.abs(a).mean()), 3)}


def run_model(weights, manifest):
    model = GeoCalib(weights=weights).to(DEV)
    per_image = []
    for m in manifest:
        img = model.load_image(str(FRAMES / m["file"])).to(DEV)
        with torch.no_grad():
            res = model.calibrate(img)
        cam = res["camera"]
        f_est = float(cam.f.detach().cpu().numpy().ravel()[0])
        vfov_est = math.degrees(float(cam.vfov))
        hfov_est = math.degrees(float(cam.hfov))
        rec = {"file": m["file"], "set": m["set"], "src": m.get("src"),
               "f_est_px": round(f_est, 2), "vfov_est_deg": round(vfov_est, 3),
               "hfov_est_deg": round(hfov_est, 3)}
        if "gt_focal_px" in m:                        # pinhole GT sets
            gt_f = m["gt_focal_px"]; gt_v = m["gt_vfov_deg"]
            rec.update({
                "gt_focal_px": gt_f, "gt_vfov_deg": gt_v,
                "f_err_pct": round(100 * (f_est - gt_f) / gt_f, 3),
                "vfov_err_deg": round(vfov_est - gt_v, 3),
                "f_ratio": round(f_est / gt_f, 4)})
        else:                                         # fisheye robustness set
            rec.update({"paraxial_focal_px": m.get("paraxial_focal_px"),
                        "f_over_paraxial": round(f_est / m["paraxial_focal_px"], 4)})
        per_image.append(rec)
    return per_image


def analyse(per_image):
    out = {}
    for s in ("comma_native", "comma_480p", "comma_focalsweep", "physicalai_native"):
        rows = [r for r in per_image if r["set"] == s]
        if not rows:
            continue
        d = {"per_image": rows}
        if s != "physicalai_native":
            d["f_err_pct"] = summarize([r["f_err_pct"] for r in rows])
            d["vfov_err_deg"] = summarize([r["vfov_err_deg"] for r in rows])
            d["f_ratio"] = summarize([r["f_ratio"] for r in rows])
            if s == "comma_focalsweep":
                gt = [r["gt_focal_px"] for r in rows]
                es = [r["f_est_px"] for r in rows]
                d["focal_sweep_pearson_r"] = round(pearson(gt, es), 4)
                d["focal_sweep_points"] = [
                    {"gt_focal": r["gt_focal_px"], "est_focal": r["f_est_px"],
                     "ratio": r["f_ratio"]} for r in rows]
        else:
            d["f_est_px"] = summarize([r["f_est_px"] for r in rows])
            d["vfov_est_deg"] = summarize([r["vfov_est_deg"] for r in rows])
            d["f_over_paraxial"] = summarize([r["f_over_paraxial"] for r in rows])
        out[s] = d
    return out


def main():
    manifest = json.load(open(FRAMES / "manifest.json"))
    results = {"device": DEV, "n_images": len(manifest), "models": {}}
    for weights in ("pinhole", "distorted"):
        print(f"=== running GeoCalib weights={weights} on {len(manifest)} images ===")
        per_image = run_model(weights, manifest)
        results["models"][weights] = analyse(per_image)
        # quick console read
        for s, d in results["models"][weights].items():
            if "f_err_pct" in d:
                print(f"  [{weights}] {s}: f_err% mean {d['f_err_pct'].get('mean')} "
                      f"(|.| {d['f_err_pct'].get('mean_abs')}), vfov_err_deg mean "
                      f"{d['vfov_err_deg'].get('mean')}"
                      + (f", sweep_r {d.get('focal_sweep_pearson_r')}" if s == 'comma_focalsweep' else ""))
            else:
                print(f"  [{weights}] {s}: f_est mean {d['f_est_px'].get('mean')} px, "
                      f"vfov_est mean {d['vfov_est_deg'].get('mean')} deg, "
                      f"f/paraxial {d['f_over_paraxial'].get('mean')}")
    (HERE / "geocalib_validation_results.json").write_text(json.dumps(results, indent=2))
    print("WROTE geocalib_validation_results.json")


if __name__ == "__main__":
    main()
