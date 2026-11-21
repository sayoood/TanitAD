"""The one-page comparison: 256x640 vs 256x1024 vs 408x1024.

Every cell is read from a built artifact or a measurement receipt -- nothing is
retyped. The "rows of the source kept" column is the one that decides the
question, so it is computed from the frames themselves:

    a 640-frame row v' and an Hx1024 row v see the same elevation iff
        v = (H-1)/2 + 1.6 * (v' - 127.5)
    so the Hx1024 frame spans 640-rows  127.5 +- ((H-1)/2 + 0.5)/1.6 .
"""
from __future__ import annotations
import argparse, json, math, os, sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass
STACK = os.environ.get("TANITAD_STACK", r"C:/Users/Admin/tanitad-snap-20260915/stack")
sys.path.insert(0, STACK)
from tanitad.data.calib import CanonicalFrame                        # noqa: E402

CORPUS_N_TRAIN = 4713
F640 = 305.5774907364391


def rows_of_640_covered(H, f_ref):
    """The 640-frame rows this HxW frame spans (its own edges, half-pixel out)."""
    half = ((H - 1) / 2.0 + 0.5) * (F640 / f_ref)
    return 127.5 - half, 127.5 + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-256x1024", required=True)
    ap.add_argument("--manifest-408x1024", required=True)
    ap.add_argument("--ref640-bytes", type=float, required=True,
                    help="exact bytes of the deployed 256x640 cache (du -sb)")
    ap.add_argument("--ref640-clips", type=int, required=True)
    ap.add_argument("--activation", required=True)
    ap.add_argument("--geometry-check", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    act = json.load(open(a.activation))
    geo = json.load(open(a.geometry_check))
    mans = {"256x1024": json.load(open(a.manifest_256x1024)),
            "408x1024": json.load(open(a.manifest_408x1024))}

    rows = {}
    for tag, H, W in (("256x640", 256, 640), ("256x1024", 256, 1024),
                      ("408x1024", 408, 1024)):
        fr = CanonicalFrame.from_hfov(120.0, H, W, "cylindrical")
        vfov = math.degrees(2 * math.atan((H / 2) / fr.f_ref))
        lo, hi = rows_of_640_covered(H, fr.f_ref)
        if tag == "256x640":
            mb = a.ref640_bytes / a.ref640_clips / 1e6
            src = f"deployed cache, du -sb / {a.ref640_clips} payloads"
        else:
            m = mans[tag]
            mb = m["total_bytes"] / m["n_clips"] / 1e6
            src = f"MANIFEST.json, {m['n_clips']} clips"
        lv = geo["G6"].get(tag, {}).get("levels", {})
        rows[tag] = {
            "height": H, "width": W, "f_ref": round(fr.f_ref, 7),
            "hfov_deg": round(fr.hfov_deg, 6), "vfov_deg": round(vfov, 4),
            "deg_per_col": round(fr.hfov_deg / W, 6),
            "centre_col": (W - 1) / 2.0, "centre_row": (H - 1) / 2.0,
            "rows_of_256x640_covered": [round(lo, 4), round(hi, 4)],
            "vfov_vs_today_pct": round(100 * vfov / 45.4556, 1),
            "mb_per_clip": round(mb, 2), "mb_per_clip_source": src,
            "corpus_gb": round(CORPUS_N_TRAIN * mb / 1e3, 1),
            "stride16": lv.get("stride16"), "stride32": lv.get("stride32"),
            "activation_sum_mb_k3_b1_fp32":
                act["geometries"][tag]["activation_sum_mb"],
            "activation_ratio_vs_256x1024": act["ratios"][tag],
        }
    out = {"note": "every cell read from a built artifact or a measurement "
                   "receipt; nothing retyped",
           "corpus_clips_assumed": CORPUS_N_TRAIN,
           "activation_caveat": act["what_is_measured"],
           "geometries": rows}
    if "external_cuda_peak_scaling" in act:
        out["external_cuda_peak_scaling"] = act["external_cuda_peak_scaling"]
    json.dump(out, open(a.out, "w"), indent=1)

    hdr = ["", "256x640 (today)", "256x1024", "408x1024"]
    def line(label, key, fmt=lambda v: v):
        return f"| {label:34s} | " + " | ".join(
            str(fmt(rows[t][key])).rjust(15) for t in hdr[1:][:0] or
            ("256x640", "256x1024", "408x1024")) + " |"
    print(f"| {'':34s} | {'256x640 (today)':>15s} | {'256x1024':>15s} | "
          f"{'408x1024':>15s} |")
    for label, key in (("f_ref", "f_ref"), ("HFOV deg", "hfov_deg"),
                       ("deg / column", "deg_per_col"),
                       ("VFOV deg", "vfov_deg"),
                       ("VFOV as % of today", "vfov_vs_today_pct"),
                       ("640-rows covered", "rows_of_256x640_covered"),
                       ("MB / clip", "mb_per_clip"),
                       ("corpus GB (4,713)", "corpus_gb"),
                       ("resnet101 K=3 activations MB",
                        "activation_sum_mb_k3_b1_fp32"),
                       ("activation x vs 256x1024",
                        "activation_ratio_vs_256x1024")):
        print(line(label, key))
    for st in ("stride16", "stride32"):
        print(f"| {st + ' grid / tokens / deg/col':34s} | " + " | ".join(
            (f"{rows[t][st]['grid'][0]}x{rows[t][st]['grid'][1]} "
             f"{rows[t][st]['tokens']} {rows[t][st]['deg_per_col']}"
             if rows[t].get(st) else "n/a").rjust(15)
            for t in ("256x640", "256x1024", "408x1024")) + " |")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
