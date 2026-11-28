"""Correct a shipped v2 cache's `_geometry.json` — LEGIBLY, never silently.

THE DEFECT. `v2_compressed.py build` probed ONE clip before the first decode and
wrote its `observed_frac` into `_geometry.json`. Both v5 caches therefore declare
`"observed_frac": 1.0`, which is true of that clip and FALSE OF THE CORPUS: the
PhysicalAI front-wide camera comes in two rigs, and at 256x640/120 deg the
rectifier masks ~8.9 % of every rig-B frame. The probed clip was rig A.

⚠️ WHY IT MATTERS MORE THAN IT LOOKS. Nothing hashes pixels, so `_geometry.json`
IS what a consumer checks — and `register_v2_sibling.py` copies the whole dict
into `parity_manifest.json`, i.e. into a COMMITTED parity record.

WHAT THIS DOES. Recomputes the fraction over the WHOLE cache, PER RIG, from each
clip's own f-theta intrinsics via `calib.observed_report` (a ray-map property —
no decode, exact), and rewrites the sidecar with:

  * the original file preserved at `_geometry.json.orig-<date>`;
  * a `corrections` list carrying OLD value, NEW value, both bases, and why;
  * `observed_frac` replaced by the population mean, its old value kept beside
    it as `observed_frac_superseded`;
  * a `rig_observability` block (per rig: n, mean/min/max) — because a single
    corpus mean would hide the very asymmetry that made 1.0 wrong;
  * `subframe_observability`: the same numbers for each candidate rig-clean
    sub-frame, so a consumer can see what the loader slice buys.

⛔ It changes NO pixel and NO payload. `--dry-run` writes nothing.

Usage:
  PYTHONPATH=<stack> python3 restamp_geometry_manifest.py \
      --cache <dir> [--cache <dir>] --root /workspace/data/physicalai_phase0 \
      --out <json> [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

#: rig assignment boundary in native principal-point rows, MEASURED over the
#: 3,000-clip canonical selection: rig A cy 533.95-554.20, rig B cy 747.55-764.52,
#: largest gap 193.35 px, boundary 650.872. No clip is ambiguous.
RIG_CY_BOUNDARY = 650.872


def _rig(cy: float) -> str:
    return "A" if float(cy) < RIG_CY_BOUNDARY else "B"


def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(sum(vals) / len(vals), 10),
            "min": round(min(vals), 10), "max": round(max(vals), 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--root", required=True, help="PhysicalAI root (calibration/)")
    ap.add_argument("--stack", default=os.environ.get(
        "TANITAD_STACK", "/workspace/TanitAD/stack"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--limit", type=int, default=0, help="0 = every clip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, os.path.join(a.stack, "scripts"))
    from tanitad.data.calib import (CanonicalFrame,
                                    PHYSICALAI_RIG_CLEAN_128x576,
                                    PHYSICALAI_RIG_CLEAN_176x624,
                                    observed_report)
    from tanitad.data.physicalai import intrinsics_for_clip

    report: dict = {"date": a.date, "root": a.root, "dry_run": a.dry_run,
                    "caches": {}}

    for cache in a.cache:
        gpath = os.path.join(cache, "_geometry.json")
        geo = json.load(open(gpath, encoding="utf-8"))
        frame = CanonicalFrame.from_dict(geo["frame"])
        subs = {f"{f.height}x{f.width}": f
                for f in (PHYSICALAI_RIG_CLEAN_176x624,
                          PHYSICALAI_RIG_CLEAN_128x576)
                if f.height <= frame.height and f.width <= frame.width}

        clips = sorted(os.path.basename(p).split(".v2ep")[0]
                       for p in glob.glob(os.path.join(cache, "*.v2ep.pt")))
        if a.limit:
            clips = clips[:a.limit]

        # one ray-map per DISTINCT sensor geometry (the census found 121 over
        # 3,000 clips), so the scan is cheap and exact rather than sampled.
        cache_by_geom: dict[tuple, dict] = {}
        per_rig: dict[str, list[float]] = {"A": [], "B": []}
        per_rig_sub: dict[str, dict[str, list[float]]] = {
            k: {"A": [], "B": []} for k in subs}
        pop: list[float] = []
        n_fallback = 0
        t0 = time.time()
        for j, cid in enumerate(clips):
            intr = intrinsics_for_clip(cid, a.root)
            if not getattr(intr, "per_clip", False):
                n_fallback += 1
                continue
            key = (round(float(intr.cx), 6), round(float(intr.cy), 6),
                   tuple(round(float(c), 6) for c in intr.poly))
            hit = cache_by_geom.get(key)
            if hit is None:
                hit = {"rig": _rig(intr.cy),
                       "obs": float(observed_report(intr, frame)["observed_frac"])}
                for nm, f in subs.items():
                    hit[nm] = float(observed_report(intr, f)["observed_frac"])
                cache_by_geom[key] = hit
            rig = hit["rig"]
            per_rig[rig].append(hit["obs"])
            pop.append(hit["obs"])
            for nm in subs:
                per_rig_sub[nm][rig].append(hit[nm])
            if (j + 1) % 400 == 0:
                print(f"  [{os.path.basename(cache)}] {j + 1}/{len(clips)} "
                      f"({time.time() - t0:.0f}s, {len(cache_by_geom)} distinct "
                      f"geometries)", flush=True)

        old = geo.get("geometry_check", {}).get("observed_frac")
        new = round(sum(pop) / len(pop), 10) if pop else None
        rig_obs = {r: _agg(v) for r, v in per_rig.items()}
        sub_obs = {nm: {r: _agg(v) for r, v in d.items()}
                   for nm, d in per_rig_sub.items()}

        rec = {"cache": cache, "n_clips_scanned": len(pop),
               "n_clips_in_cache": len(clips),
               "n_no_per_clip_intrinsics": n_fallback,
               "distinct_sensor_geometries": len(cache_by_geom),
               "frame": geo["frame"], "old_observed_frac": old,
               "new_observed_frac": new, "rig_observability": rig_obs,
               "subframe_observability": sub_obs,
               "scan_s": round(time.time() - t0, 1)}
        report["caches"][cache] = rec
        print(json.dumps({k: v for k, v in rec.items()
                          if k != "subframe_observability"}, indent=1),
              flush=True)

        if a.dry_run or not pop:
            continue

        backup = f"{gpath}.orig-{a.date}"
        if not os.path.exists(backup):
            shutil.copy2(gpath, backup)

        gc = dict(geo.get("geometry_check") or {})
        gc["observed_frac_superseded"] = old
        gc["observed_frac"] = new
        gc["observed_frac_basis"] = (
            f"POPULATION: every clip in this cache ({len(pop)}), each clip's own "
            f"f-theta intrinsics, calib.observed_report ray map, no decode")
        gc["observed_frac_note"] = (
            f"CORRECTED {a.date}. The superseded value came from a ONE-CLIP "
            f"pre-decode probe and is false of this corpus. See corrections[].")
        geo["geometry_check"] = gc
        geo["rig_observability"] = {
            **rig_obs,
            "rig_assignment": f"native principal-point row cy, boundary "
                              f"{RIG_CY_BOUNDARY} (MEASURED: largest gap 193.35 px "
                              f"over 3,000 clips; no clip ambiguous)",
            "fully_observed_by_all": all(v.get("min", 0.0) >= 1.0
                                         for v in rig_obs.values() if v["n"]),
        }
        geo["subframe_observability"] = {
            **sub_obs,
            "_": ("what a CENTRED sub-frame of this cache would deliver. It is a "
                  "pure pixel slice — `v2_dataset.build_v2_providers(frame=…)` / "
                  "`v2_compressed.load_compressed(path, frame=…)`. Nothing needs "
                  "rebuilding to reach these numbers."),
        }
        geo.setdefault("corrections", []).append({
            "date": a.date,
            "field": "geometry_check.observed_frac",
            "old": old, "new": new,
            "old_basis": ("ONE clip. `v2_compressed._assert_geometry_deliverable` "
                          "probed clips[0] before the first decode; that clip was "
                          "rig A, which this frame fully observes."),
            "new_basis": rec["n_clips_scanned"],
            "new_basis_note": gc["observed_frac_basis"],
            "why": ("the front-wide camera is TWO RIGS whose principal points "
                    "differ ~211 px vertically; a 120 deg x 45.456 deg request "
                    "over-runs rig B's sensor, so the rectifier masks ~8.9 % of "
                    "every rig-B frame. A single-clip probe cannot see that, and "
                    "nothing hashes pixels — this declaration is what a consumer "
                    "checks, and `register_v2_sibling.py` copies it verbatim into "
                    "parity_manifest.json."),
            "changes_no_pixels": True,
            "backup_of_original": os.path.basename(backup),
            "instrument": "restamp_geometry_manifest.py (rig-fix-wiring stream)",
            "raw": os.path.basename(a.out),
        })
        tmp = gpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(geo, fh, indent=1)
        os.replace(tmp, gpath)
        rec["written"] = gpath
        rec["backup"] = backup
        print(f"  -> restamped {gpath} (original kept at {backup})", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
