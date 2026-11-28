"""Census the fully-black BOTTOM ROWS of every payload in a v2ep cache.

WHY (coordinator, 2026-09-23): on the gated eval-139 cache at 416x1024 a
reviewer MEASURED 74/139 clips carrying 27-37 fully-black bottom rows (mean
31.1 = 7.5 % of height), 65 carrying ZERO and nothing in between, and showed the
strip is a USABLE SHORTCUT -- painting a 31-row strip on a random half of the
clean clips makes that arbitrary label 0.899 +- 0.046 balanced-accuracy
decodable from an 8x16 thumbnail against three controls at chance.
``calib.py:1031-1036`` names the shape as retraction class C26: *"a
rig-correlated BLACK region is still a rig-correlated signal, and this model
eats shortcuts."*

⛔ THIS FILE DOES NOT CHANGE THE GEOMETRY. 416x1024 is the PI's instruction and
the SPEC's ruling; the mitigation is a loader/model-side mask owned by another
stream. The job here is to make the property VISIBLE AND MEASURED IN THE
ARTIFACT, so a consumer asks the cache "does this episode carry the strip?"
instead of re-deriving it -- and so the TRAIN split's number is our own, rather
than the eval-139 split's number quoted for a corpus it does not describe.

⭐ THE CORRELATE IS THE POINT, NOT THE COUNT. A uniform strip is a nuisance; a
strip that tracks the rig is a rig label painted into the pixels. This reports
the 2x2 contingency against the per-clip principal point (rig A ``cy`` ~543 /
rig B ``cy`` ~755, the two-rig split already established for this corpus) and
against clip length, and prints NOT ESTABLISHED when a correlate cannot be
resolved rather than guessing one.

Sampled, not exhaustive: ``--frames`` evenly spaced frames per clip (default 5).
The strip is a geometric property of the rig's vertical field, so it is
constant within a clip; the spread ACROSS sampled frames is reported so that
assumption is checked rather than assumed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def _load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False,
                          mmap=True)                 # only the pages we touch
    except Exception:                                            # noqa: BLE001
        return torch.load(path, map_location="cpu", weights_only=False)


def census_one(args):
    """(clip_id, path, n_frames_to_sample) -> per-clip strip record."""
    cid, path, nf = args
    import torch, torchvision.io as tvio
    torch.set_num_threads(1)
    try:
        d = _load(path)
        lens = d["jpeg_len"]
        n = int(lens.shape[0])
        if n <= 0:
            return cid, None, "jpeg_len empty"
        off = torch.cumsum(lens, 0) - lens
        idx = sorted({int(round(i * (n - 1) / max(nf - 1, 1)))
                      for i in range(nf)})
        buf = d["jpeg_buf"]
        rows, zfrac = [], []
        for i in idx:
            o, l = int(off[i]), int(lens[i])
            img = tvio.decode_png(buf[o:o + l], mode=tvio.ImageReadMode.RGB)
            H, W = int(img.shape[1]), int(img.shape[2])
            # a row is BLACK only if every pixel of every channel is exactly 0
            rowmax = img.amax(dim=(0, 2))            # [H]
            black = (rowmax == 0)
            k = 0
            for r in range(H - 1, -1, -1):           # from the BOTTOM upward
                if bool(black[r]):
                    k += 1
                else:
                    break
            rows.append(k)
            zfrac.append(float((img.amax(dim=0) == 0).float().mean()))
        return cid, {"black_bottom_rows": int(max(rows)),
                     "black_bottom_rows_min": int(min(rows)),
                     "black_bottom_rows_median": int(statistics.median(rows)),
                     "constant_across_frames": bool(min(rows) == max(rows)),
                     "zero_pixel_frac": round(max(zfrac), 6),
                     "n_frames": n, "height": H, "width": W,
                     "n_sampled": len(idx)}, None
    except Exception as e:                                       # noqa: BLE001
        return cid, None, f"{type(e).__name__}: {str(e)[:160]}"


def rig_table(root, ids):
    """clip_id -> principal point cy, via the deployed resolution order."""
    sys.path.insert(0, os.environ.get("TANITAD_STACK",
                                      "/home/nvidia/TanitAD/stack"))
    out = {}
    try:
        from tanitad.data import physicalai as PA
    except Exception as e:                                       # noqa: BLE001
        print(f"[census] intrinsics UNAVAILABLE: {type(e).__name__}: {e}")
        return out
    for c in ids:
        try:
            i = PA.intrinsics_for_clip(c, root)
            if i is not None:
                out[c] = float(i.cy)
        except Exception:                                        # noqa: BLE001
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--root", default="/home/nvidia/data/_b1stage416")
    p.add_argument("--out", default="")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cy-split", type=float, default=650.0,
                   help="rig A (cy~543) below, rig B (cy~755) above")
    a = p.parse_args()
    out_p = a.out or os.path.join(a.cache, "_black_rows_census.json")

    ids = sorted(f[: -len(".v2ep.pt")] for f in os.listdir(a.cache)
                 if f.endswith(".v2ep.pt"))
    print(f"[census] {len(ids)} payloads in {a.cache}", flush=True)
    recs, errs, t0 = {}, {}, time.time()
    jobs = [(c, os.path.join(a.cache, f"{c}.v2ep.pt"), a.frames) for c in ids]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(census_one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            cid, r, err = fu.result()
            if err:
                errs[sha12(cid)] = err
            else:
                recs[cid] = r
            if k % 250 == 0 or k == len(jobs):
                print(f"[census] {k}/{len(jobs)} ok={len(recs)} "
                      f"err={len(errs)} {time.time()-t0:.0f}s", flush=True)

    vals = [r["black_bottom_rows"] for r in recs.values()]
    carriers = [c for c, r in recs.items() if r["black_bottom_rows"] > 0]
    clean = [c for c, r in recs.items() if r["black_bottom_rows"] == 0]
    hist = {}
    for v in vals:
        hist[str(v)] = hist.get(str(v), 0) + 1
    nzv = [v for v in vals if v > 0]

    # ---- the correlate ------------------------------------------------- #
    cy = rig_table(a.root, list(recs))
    corr = {"status": "NOT ESTABLISHED",
            "note": "per-clip intrinsics could not be resolved"}
    if cy:
        tab = {"A_strip": 0, "A_clean": 0, "B_strip": 0, "B_clean": 0}
        nocy = 0
        for c, r in recs.items():
            if c not in cy:
                nocy += 1
                continue
            rig = "B" if cy[c] >= a.cy_split else "A"
            tab[f"{rig}_{'strip' if r['black_bottom_rows'] > 0 else 'clean'}"] += 1
        nA, nB = tab["A_strip"] + tab["A_clean"], tab["B_strip"] + tab["B_clean"]
        corr = {"status": "MEASURED", "cy_split": a.cy_split,
                "contingency_2x2": tab, "n_without_cy": nocy,
                "rigA_n": nA, "rigB_n": nB,
                "rigA_strip_frac": round(tab["A_strip"] / nA, 6) if nA else None,
                "rigB_strip_frac": round(tab["B_strip"] / nB, 6) if nB else None}
        # ⭐ THE CONTINUOUS RELATION IS THE MECHANISM TEST, not the 2x2. If the
        # strip is the frame running off the bottom of THIS rig's valid field,
        # the row COUNT must move with cy among the carriers. A 2x2 alone is
        # equally consistent with "rig B clips were shot somewhere darker".
        cys = [cy[c] for c in recs if c in cy]
        corr["cy_min"] = round(min(cys), 2) if cys else None
        corr["cy_max"] = round(max(cys), 2) if cys else None
        car = [(cy[c], recs[c]["black_bottom_rows"])
               for c in carriers if c in cy]
        if len(car) >= 3:
            xs = [x for x, _ in car]
            ys = [float(y) for _, y in car]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            sx = sum((x - mx) ** 2 for x in xs) ** 0.5
            sy = sum((y - my) ** 2 for y in ys) ** 0.5
            r = (sum((x - mx) * (y - my) for x, y in car) / (sx * sy)
                 if sx > 0 and sy > 0 else None)
            corr["carriers_cy_vs_rows_pearson_r"] = round(r, 4) if r is not None else None
            corr["carriers_n"] = len(car)
            corr["carriers_cy_mean"] = round(mx, 2)
            corr["carriers_rows_mean"] = round(my, 3)
            corr["carriers_cy_sd_zero"] = sx == 0.0
        corr["clean_cy_mean"] = (round(statistics.mean(
            [cy[c] for c in clean if c in cy]), 2)
            if any(c in cy for c in clean) else None)
        # clip length: is the strip confounded with duration?
        ln_s = [recs[c]["n_frames"] for c in carriers]
        ln_c = [recs[c]["n_frames"] for c in clean]
        corr["n_frames_mean_strip"] = round(statistics.mean(ln_s), 2) if ln_s else None
        corr["n_frames_mean_clean"] = round(statistics.mean(ln_c), 2) if ln_c else None

    rep = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache": os.path.basename(a.cache.rstrip("/")),
        "definition": "count of consecutive rows from the BOTTOM of the frame "
                      "whose every pixel in every channel is exactly 0; max "
                      "over evenly spaced sampled frames per clip",
        "frames_sampled_per_clip": a.frames,
        "n_clips": len(recs), "n_errors": len(errs),
        "n_carrying_strip": len(carriers), "n_clean": len(clean),
        "frac_carrying_strip": round(len(carriers) / max(len(recs), 1), 6),
        "strip_rows_min": min(nzv) if nzv else None,
        "strip_rows_max": max(nzv) if nzv else None,
        "strip_rows_mean": round(statistics.mean(nzv), 3) if nzv else None,
        "strip_rows_median": statistics.median(nzv) if nzv else None,
        "strip_frac_of_height_mean": (round(statistics.mean(nzv) / 416, 5)
                                      if nzv else None),
        "n_not_constant_across_frames": sum(
            1 for r in recs.values() if not r["constant_across_frames"]),
        "histogram_rows_to_count": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "bimodal_check": {
            "n_zero": len(clean), "n_in_1_to_10": sum(1 for v in vals if 1 <= v <= 10),
            "n_gt_10": sum(1 for v in vals if v > 10),
            "gap_is_empty_1_to_10": sum(1 for v in vals if 1 <= v <= 10) == 0},
        "rig_correlation": corr,
        "errors_sha12": errs,
        "per_clip": {sha12(c): r for c, r in sorted(recs.items())},
    }
    json.dump(rep, open(out_p, "w"), indent=1)
    brief = {k: v for k, v in rep.items()
             if k not in ("per_clip", "errors_sha12")}
    print(json.dumps(brief, indent=1), flush=True)
    print(f"ZZCENSUS-{len(recs)}-{len(carriers)}-{len(errs)}ZZ -> {out_p}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
