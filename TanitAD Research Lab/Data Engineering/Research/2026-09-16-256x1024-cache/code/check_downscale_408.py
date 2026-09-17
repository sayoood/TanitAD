"""G1/G2/G3 for the 408x1024 cache, plus C1 -- the half-pixel control.

Reuses the comparison primitives from ``check_downscale_identity`` verbatim
(``compare`` / ``stats`` / ``xcorr_peak`` / ``frames_of`` / ``pick_frames``) so
the two geometries are judged by the SAME instrument. Only the row
correspondence differs, and it is derived, not fitted (PREREG ADDENDUM A1):

    256x1024 -> resize to (160,640), compare against frame640[48:208]  (integer)
    408x1024 -> resize to (255,640), compare against frame640 rows r+0.5
                i.e. the half-pixel-interpolated band
                0.5*(frame640[0:255] + frame640[1:256])

C1 prices that half-pixel on the ALREADY-PASSED 256 case, and the 408 tolerance
is the base tolerance widened by exactly that price -- fixed before the 408
numbers were seen.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                         # noqa: E402
import torch.nn.functional as F                                      # noqa: E402
from check_downscale_identity import (TOL, compare, frames_of,       # noqa: E402
                                      load_payload, pick_frames)


def down_to(x_u8, hw):
    return F.interpolate(x_u8.float().unsqueeze(0), size=hw, mode="bilinear",
                         align_corners=False, antialias=True
                         ).squeeze(0).round().clamp(0, 255).to(torch.uint8)


def halfpix(ref_u8, lo, hi):
    """Rows lo+0.5 .. hi-0.5 of the reference, bilinear (one half-pixel shift)."""
    a = ref_u8[:, lo:hi, :].float()
    b = ref_u8[:, lo + 1:hi + 1, :].float()
    return (0.5 * (a + b)).round().clamp(0, 255).to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide408", required=True)
    ap.add_argument("--wide256", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ids = [l.strip() for l in open(a.ids) if l.strip()][:a.clips]
    R, FAIL = {"tolerances_base": TOL["G2"]}, []

    # ---- C1: price the half-pixel on the already-passed 256x1024 case --------
    print("C1 half-pixel control, measured on the PASSED 256x1024 comparison",
          flush=True)
    d0, d1 = [], []
    for cid in ids:
        pa = os.path.join(a.wide256, f"{cid}.v2ep.pt")
        pb = os.path.join(a.ref, f"{cid}.v2ep.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        A, B = load_payload(pa), load_payload(pb)
        n = min(int(A["jpeg_len"].shape[0]), int(B["jpeg_len"].shape[0]))
        idx = pick_frames(n, a.frames)
        fa, fb = frames_of(A, idx), frames_of(B, idx)
        for i in range(len(idx)):
            dn = down_to(fa[i], (160, 640))
            d0.append(compare(dn, fb[i][:, 48:208, :], "aligned", TOL["G2"]))
            d1.append(compare(dn, halfpix(fb[i], 48, 208), "halfpix", TOL["G2"]))
    pen = {"median": round(max(x["median"] for x in d1)
                           - max(y["median"] for y in d0), 4),
           "mae": round(max(x["mae"] for x in d1) - max(y["mae"] for y in d0), 4),
           "psnr_db": round(min(y["psnr_db"] for y in d0)
                            - min(x["psnr_db"] for x in d1), 4)}
    pen = {k: max(v, 0.0) for k, v in pen.items()}
    R["C1_half_pixel_penalty"] = {
        "n_frames": len(d0),
        "aligned_worst": {"median": max(x["median"] for x in d0),
                          "mae": round(max(x["mae"] for x in d0), 4),
                          "psnr_db": round(min(x["psnr_db"] for x in d0), 3)},
        "halfpix_worst": {"median": max(x["median"] for x in d1),
                          "mae": round(max(x["mae"] for x in d1), 4),
                          "psnr_db": round(min(x["psnr_db"] for x in d1), 3)},
        "penalty": pen}
    print(f"  aligned worst : median {R['C1_half_pixel_penalty']['aligned_worst']['median']} "
          f"mae {R['C1_half_pixel_penalty']['aligned_worst']['mae']} "
          f"psnr {R['C1_half_pixel_penalty']['aligned_worst']['psnr_db']} dB", flush=True)
    print(f"  half-pixel    : median {R['C1_half_pixel_penalty']['halfpix_worst']['median']} "
          f"mae {R['C1_half_pixel_penalty']['halfpix_worst']['mae']} "
          f"psnr {R['C1_half_pixel_penalty']['halfpix_worst']['psnr_db']} dB", flush=True)
    print(f"  PENALTY       : {pen}", flush=True)

    tol408 = {"median": TOL["G2"]["median"] + pen["median"],
              "mae": TOL["G2"]["mae"] + pen["mae"],
              "psnr": TOL["G2"]["psnr"] - pen["psnr_db"]}
    R["tolerances_408"] = tol408
    print(f"\nG1/G2 at 408x1024 (tolerance = base widened by C1): {tol408}",
          flush=True)

    rows, mut = [], {}
    for cid in ids:
        s12 = hashlib.sha256(cid.encode()).hexdigest()[:12]
        pa = os.path.join(a.wide408, f"{cid}.v2ep.pt")
        pb = os.path.join(a.ref, f"{cid}.v2ep.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  [SKIP] {s12}", flush=True); continue
        A, B = load_payload(pa), load_payload(pb)
        assert (int(A["image_w"]), int(A["image_h"])) == (1024, 408), \
            (A["image_w"], A["image_h"])
        assert abs(A["frame"]["f_ref"] - 488.9239851782515) < 1e-6
        n = min(int(A["jpeg_len"].shape[0]), int(B["jpeg_len"].shape[0]))
        idx = pick_frames(n, a.frames)
        fa, fb = frames_of(A, idx), frames_of(B, idx)
        per = []
        for i, fi in enumerate(idx):
            dn = down_to(fa[i], (255, 640))
            v = compare(dn, halfpix(fb[i], 0, 255), f"{s12}@{fi}", tol408)
            v["G2_photometric_pass"] = (v["median"] <= tol408["median"]
                                        and v["mae"] <= tol408["mae"]
                                        and v["psnr_db"] >= tol408["psnr"])
            per.append(v)
        ok = all(p["G1_geometry_pass"] and p["G2_photometric_pass"] for p in per)
        rows.append({"clip_sha12": s12, "frames": idx, "pass": ok,
                     "per_frame": per})
        if not ok:
            FAIL.append(s12)
        print(f"  [{'PASS' if ok else 'FAIL'}] {s12}: worst median "
              f"{max(p['median'] for p in per):.3f} mae "
              f"{max(p['mae'] for p in per):.3f} psnr "
              f"{min(p['psnr_db'] for p in per):.2f} dB peaks="
              f"{[p['xcorr_peak'] for p in per]}", flush=True)

        if cid == ids[0]:
            print("\nG3 mutations re-run at 408 (each MUST fail)", flush=True)
            dn = down_to(fa[0], (255, 640))
            m2 = compare(dn, halfpix(fb[0], 0, 255)[:, 0:255, :].roll(40, dims=1),
                         "M2_wrong_row_crop_408", tol408)
            m3 = compare(torch.flip(dn, dims=[2]), halfpix(fb[0], 0, 255),
                         "M3_mirrored_408", tol408)
            for k, v in (("M2_wrong_row_crop_408", m2), ("M3_mirrored_408", m3)):
                v["G2_photometric_pass"] = (v["median"] <= tol408["median"]
                                            and v["mae"] <= tol408["mae"]
                                            and v["psnr_db"] >= tol408["psnr"])
                broke = not (v["G1_geometry_pass"] and v["G2_photometric_pass"])
                v["MUTATION_DETECTED"] = broke
                mut[k] = v
                print(f"  [{'DETECTED' if broke else 'MISSED(BUG)'}] {k}: "
                      f"peak={v['xcorr_peak']} median={v['median']} "
                      f"mae={v['mae']} psnr={v['psnr_db']} dB", flush=True)
                if not broke:
                    FAIL.append(k)
    R["G1_G2_408"] = {"clips": rows, "failures": FAIL}
    R["G3_mutations_408"] = mut
    R["failures"] = FAIL
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ','.join(map(str, FAIL))}"
          f" -> {a.out}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
