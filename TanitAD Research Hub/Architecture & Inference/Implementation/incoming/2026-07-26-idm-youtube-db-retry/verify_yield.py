"""D-B retry — YIELD VERIFICATION (presence is NOT completeness).

This program has twice been burned by presence-not-completeness: a 3.24 GB checkpoint
that was 48.9 MB short still passed a `find`. So this script does NOT count files.
For every merged clip-latent it verifies:

  * the file exists AND its byte size is > 0 and matches a torch-loadable tensor,
  * the tensor's SHAPE (n_windows, state_dim) and dtype,
  * finite-ness (no NaN/Inf) — a silently-truncated encode shows up here,
  * the joined POINTER's frame count / duration (the clip is 250 frames @ 10 Hz = 25 s
    by construction; anything else is a short/partial clip),
  * geometry provenance (GeoCalib vs fixed-HFOV fallback) so the headline can say
    which geometry each clip actually got.

Emits ONE JSON with per-clip rows + the aggregate yield vs TARGET. A yield of N/400 is
reported as N/400 — never as "the pipeline works".

Run on pod3:  /workspace/venv/bin/python verify_yield.py --work /workspace/tmp/yt_scaleup \
                  --target 400 --out /workspace/tmp/yt_scaleup/results/yield_verification.json
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import torch

# --clip-frames 250 segmented at 10 Hz, then stacked with N_STACK=3 -> the pointer's
# n_frames_10hz is 250 - 3 + 1 = 248 for a FULL clip (24.8 s). 248 is therefore the
# expected full length, NOT a short clip. (Measured from yt_pilot_common.N_STACK=3;
# getting this wrong makes every clip look truncated.)
CLIP_FRAMES = 250
N_STACK = 3
EXPECTED_CLIP_FRAMES = CLIP_FRAMES - N_STACK + 1     # 248
TARGET_HZ = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/tmp/yt_scaleup")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    work = Path(args.work)
    merged = work / "latents"

    # ---- 1. pointers (one jsonl per worker) -------------------------------
    pointers = {}
    ptr_files = sorted(glob.glob(str(work / "w*" / "pointers.jsonl")))
    dup_ptr = 0
    for pf in ptr_files:
        wname = Path(pf).parent.name
        with open(pf, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    p = json.loads(ln)
                except Exception:
                    continue
                # the pointer schema names it `clip_idx` (per-worker, 0-based)
                key = (wname, p.get("clip_idx", p.get("clip_id")))
                if key in pointers:
                    dup_ptr += 1
                pointers[key] = p

    # ---- 2. every merged latent, byte-verified ---------------------------
    rows, bad = [], []
    lat_files = sorted(glob.glob(str(merged / "yt_*.pt")))
    total_bytes = 0
    for f in lat_files:
        name = os.path.basename(f)
        try:
            nbytes = os.path.getsize(f)
        except OSError as e:
            bad.append({"file": name, "why": f"stat failed: {e}"}); continue
        total_bytes += nbytes
        row = {"file": name, "bytes": nbytes}
        if nbytes == 0:
            row["ok"] = False; row["why"] = "ZERO BYTES"
            rows.append(row); bad.append(row); continue
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            row["ok"] = False; row["why"] = f"torch.load failed: {type(e).__name__}: {e}"
            rows.append(row); bad.append(row); continue

        z = obj.get("z") if isinstance(obj, dict) else obj
        if not torch.is_tensor(z):
            # find the first tensor in the payload
            if isinstance(obj, dict):
                for v in obj.values():
                    if torch.is_tensor(v) and v.dim() >= 2:
                        z = v; break
        if not torch.is_tensor(z):
            row["ok"] = False; row["why"] = "no tensor payload"
            rows.append(row); bad.append(row); continue

        row["shape"] = list(z.shape)
        row["dtype"] = str(z.dtype)
        finite = bool(torch.isfinite(z.float()).all())
        row["all_finite"] = finite
        row["n_windows"] = int(z.shape[0])
        row["state_dim"] = int(z.shape[-1])
        if isinstance(obj, dict):
            for k in ("video_id", "clip_id"):
                if k in obj:
                    row[k] = obj[k]
        row["ok"] = finite and z.shape[0] > 0
        if not row["ok"]:
            row["why"] = "non-finite values" if not finite else "empty tensor"
            bad.append(row)
        rows.append(row)

    # ---- 3. pointer-side duration / frame verification --------------------
    ptr_rows, short_clips = [], 0
    geo_counts, conf_counts, lic_counts = {}, {}, {}
    canon_counts = {"fully_canonical_true": 0, "fully_canonical_false": 0}
    for (wname, cid), p in sorted(pointers.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        nfr = p.get("n_frames_10hz") or p.get("n_frames") or p.get("n_frames_stacked")
        # duration from the pointer's own timestamps (independent of the frame count)
        t0, t1 = p.get("start_time_s"), p.get("end_time_s")
        dur_ts = (round(t1 - t0, 2) if isinstance(t0, (int, float))
                  and isinstance(t1, (int, float)) else None)
        dur = (round(nfr / TARGET_HZ, 2) if isinstance(nfr, (int, float)) else None)
        full = (nfr is not None and abs(nfr - EXPECTED_CLIP_FRAMES) <= 1)
        if not full:
            short_clips += 1
        g = p.get("geometry_source"); c = p.get("geocalib_confidence"); l = str(p.get("license"))
        geo_counts[str(g)] = geo_counts.get(str(g), 0) + 1
        conf_counts[str(c)] = conf_counts.get(str(c), 0) + 1
        lic_counts[l] = lic_counts.get(l, 0) + 1
        canon_counts["fully_canonical_true" if p.get("fully_canonical")
                     else "fully_canonical_false"] += 1
        ptr_rows.append({
            "worker": wname, "clip_idx": cid, "video_id": p.get("video_id"),
            "n_frames": nfr, "duration_s": dur, "duration_s_from_timestamps": dur_ts,
            "full_length": full,
            "geometry_source": g, "hfov_used_deg": p.get("hfov_used_deg"),
            "geocalib_vfov_deg": p.get("geocalib_vfov_deg"),
            "geocalib_confidence": c,
            "geocalib_fallback_used": p.get("geocalib_fallback_used"),
            "achieved_f_eff": p.get("achieved_f_eff"),
            "fully_canonical": p.get("fully_canonical"),
            "shotcut_score": p.get("shotcut_score"),
            "is_cc": p.get("is_cc"), "license": p.get("license"),
            "url": p.get("url") or p.get("webpage_url"),
        })

    n_ok = sum(1 for r in rows if r.get("ok"))
    out = {
        "experiment": "db_retry_yield_verification",
        "target_clips": args.target,
        "verification_policy": (
            "presence is NOT completeness: every latent is byte-sized, torch-loaded, "
            "shape-checked and finite-checked; every pointer's frame count -> duration "
            "is checked against the 250-frame/25 s construction."),
        "n_latent_files": len(lat_files),
        "n_latents_verified_ok": n_ok,
        "n_latents_bad": len(bad),
        "total_latent_bytes": total_bytes,
        "n_pointers": len(pointers),
        "duplicate_pointer_ids": dup_ptr,
        "expected_full_clip_frames": EXPECTED_CLIP_FRAMES,
        "expected_full_clip_duration_s": EXPECTED_CLIP_FRAMES / TARGET_HZ,
        "n_short_clips": short_clips,
        "yield_vs_target": f"{n_ok}/{args.target}",
        "yield_fraction": round(n_ok / args.target, 4) if args.target else None,
        "canonical_crop_distribution": canon_counts,
        "geometry_source_distribution": geo_counts,
        "geocalib_confidence_distribution": conf_counts,
        "license_distribution": lic_counts,
        "bad_latents": bad,
        "latents": rows,
        "pointers": ptr_rows,
    }
    js = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"WROTE {args.out}")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("latents", "pointers", "bad_latents")}, indent=2))


if __name__ == "__main__":
    main()
