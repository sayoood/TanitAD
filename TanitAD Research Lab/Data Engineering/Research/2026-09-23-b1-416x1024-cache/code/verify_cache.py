"""INDEPENDENT content verification of the 416x1024 v2ep cache.

⛔ This is deliberately NOT a re-run of the builder's own checks. The builder
verifies each payload against ``pod_pull_b1_epcache.EXPECT``, which it also
SETS -- re-running that derivation and finding agreement measures determinism,
not correctness. Every expectation here is written as a LITERAL, typed from the
geometry definition rather than read from the code under test:

    image_h 416 · image_w 1024 · f_ref 488.92398517830253 · HFOV 120.0000
    codec png · n_stack 3 · projection cylindrical

⭐ The analytic cross-check is the one that cannot agree by construction:
``f_ref`` must equal ``512 / radians(60)`` to the last bit, and on a CYLINDRICAL
projection HFOV is ``degrees(2*(W/2)/f_ref)``, exactly 120 -- NOT the pinhole
``2*atan((W/2)/f_ref)``, which reads 92.6414 here and would look plausible.

⛔ And a decode that raises into a pre-allocated buffer leaves a full-size file
of ZEROS that still loads: every sampled payload must decode to a frame with a
NON-ZERO max, and its mean is printed so "it passed" is a number, not a word.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

H, W = 416, 1024
F_REF = 488.92398517830253
HFOV = 120.0
CODEC, N_STACK, PROJ = "png", 3, "cylindrical"


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def check(args):
    cid, path = args
    import torch, torchvision.io as tvio
    torch.set_num_threads(1)
    bad = []
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                       # noqa: BLE001
        return cid, ["load: %s: %s" % (type(e).__name__, str(e)[:120])], None
    try:
        n = int(d["jpeg_len"].shape[0])
        if n <= 0:
            bad.append("jpeg_len empty")
        for k, want in (("image_h", H), ("image_w", W), ("codec", CODEC),
                        ("n_stack", N_STACK), ("projection_mode", PROJ)):
            if d.get(k) != want:
                bad.append(f"{k}={d.get(k)!r} != {want!r}")
        if int(d["poses"].shape[0]) != n:
            bad.append(f"poses {int(d['poses'].shape[0])} != {n}")
        if int(d["actions"].shape[0]) != n:
            bad.append(f"actions {int(d['actions'].shape[0])} != {n}")
        if int(d["jpeg_buf"].numel()) != int(d["jpeg_len"].sum()):
            bad.append("jpeg_buf != sum(jpeg_len)")
        fr = d["frame"]
        if abs(float(fr["f_ref"]) - F_REF) > 0:
            bad.append(f"f_ref {fr['f_ref']!r} != {F_REF!r}")
        if int(fr["height"]) != H or int(fr["width"]) != W:
            bad.append(f"frame {fr['height']}x{fr['width']}")
        if fr["projection"] != PROJ:
            bad.append(f"projection {fr['projection']!r}")
        got = math.degrees(2.0 * (float(fr["width"]) / 2.0) / float(fr["f_ref"]))
        if abs(got - HFOV) > 1e-4:
            bad.append(f"HFOV {got:.6f} != 120")
        # the all-zero-bank trap, on a RANDOM frame (not always frame 0)
        import torch as T
        off = T.cumsum(d["jpeg_len"], 0) - d["jpeg_len"]
        i = random.randrange(n) if n > 1 else 0
        o, l = int(off[i]), int(d["jpeg_len"][i])
        img = tvio.decode_png(d["jpeg_buf"][o:o + l],
                              mode=tvio.ImageReadMode.RGB)
        if tuple(img.shape) != (3, H, W):
            bad.append(f"decoded {tuple(img.shape)}")
        mx, mean = int(img.max()), float(img.float().mean())
        if mx == 0:
            bad.append("decoded frame ALL ZERO")
        return cid, bad, {"n_frames": n, "frame_idx": i, "max": mx,
                          "mean": round(mean, 3), "hfov": round(got, 9),
                          "bytes": os.path.getsize(path)}
    except Exception as e:                                       # noqa: BLE001
        return cid, bad + ["%s: %s" % (type(e).__name__, str(e)[:120])], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--sample", type=int, default=0, help="0 = ALL payloads")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="")
    a = p.parse_args()
    ids = sorted(f[: -len(".v2ep.pt")] for f in os.listdir(a.cache)
                 if f.endswith(".v2ep.pt"))
    if a.sample and a.sample < len(ids):
        random.seed(20260923)
        ids = sorted(random.sample(ids, a.sample))
    print(f"[verify] {len(ids)} payloads, literals "
          f"{H}x{W} f_ref={F_REF!r} HFOV={HFOV}", flush=True)
    print(f"[verify] analytic f_ref 512/radians(60) = "
          f"{512.0/math.radians(60.0)!r} -> agrees: "
          f"{512.0/math.radians(60.0) == F_REF}", flush=True)
    ok, fails, stats, t0 = 0, {}, [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(check, (c, os.path.join(a.cache, f"{c}.v2ep.pt")))
                for c in ids]
        for k, fu in enumerate(as_completed(futs), 1):
            cid, bad, st = fu.result()
            if bad:
                fails[sha12(cid)] = bad
            else:
                ok += 1
                stats.append(st)
            if k % 500 == 0 or k == len(ids):
                print(f"[verify] {k}/{len(ids)} ok={ok} fail={len(fails)} "
                      f"{time.time()-t0:.0f}s", flush=True)
    by = [s["bytes"] for s in stats]
    mn = [s["mean"] for s in stats]
    rep = {"verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cache": os.path.basename(a.cache.rstrip("/")),
           "expect_literals": {"image_h": H, "image_w": W, "f_ref": F_REF,
                               "hfov_deg": HFOV, "codec": CODEC,
                               "n_stack": N_STACK, "projection_mode": PROJ},
           "analytic_f_ref": 512.0 / math.radians(60.0),
           "analytic_agrees_bitwise": 512.0 / math.radians(60.0) == F_REF,
           "hfov_pinhole_WRONG_formula": round(
               math.degrees(2 * math.atan((W / 2) / F_REF)), 4),
           "n_checked": len(ids), "n_ok": ok, "n_failed": len(fails),
           "bytes_total": sum(by), "bytes_mean": round(statistics.mean(by), 1)
           if by else None,
           "MB_per_ep_decimal": round(statistics.mean(by) / 1e6, 3) if by else None,
           "MiB_per_ep": round(statistics.mean(by) / 1048576, 3) if by else None,
           "decoded_mean_min": min(mn) if mn else None,
           "decoded_mean_max": max(mn) if mn else None,
           "decoded_mean_mean": round(statistics.mean(mn), 3) if mn else None,
           "n_frames_min": min(s["n_frames"] for s in stats) if stats else None,
           "n_frames_max": max(s["n_frames"] for s in stats) if stats else None,
           "failures_sha12": fails}
    out = a.out or os.path.join(a.cache, "_verify_report.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "failures_sha12"},
                     indent=1), flush=True)
    print(f"ZZVERIFY-{ok}-{len(fails)}ZZ -> {out}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
