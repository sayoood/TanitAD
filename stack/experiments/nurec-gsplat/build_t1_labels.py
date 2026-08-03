#!/usr/bin/env python3
"""Build map-derived STRATEGIC option-set labels for the T1 branch scenes, in bulk.

⚠️ **THE DOWNLOAD COST THE BRIEF WARNED ABOUT IS AVOIDABLE, AND THIS IS THE PROOF.**
Each NuRec scene is a ~1.8-2.0 GB ``.usdz``.  But a USDZ is a zip whose members are all
**STORED, never deflated** (39/39 verified on ``00040136``), so an HTTP ``Range`` request
pulls one member out of the archive without moving the rest.  The label for one scene
needs ``map.xodr`` + ``pose_record.json`` + two clipgt parquets — MEASURED at roughly
**1-4 MB per scene**, i.e. ~0.15 % of the archive.  Scoring 141 scenes therefore costs
about the same traffic as downloading **one third of one scene**.

The camera video (``camera_front_wide_120fov.mp4``, ~40 MB) is a SEPARATE file in the
scene directory — it is *not* inside the usdz — so the observation side is cheap too.
The only genuinely expensive member is ``volume.nurec`` (~600 MB, the gaussians), and a
route-head evaluation on the LOGGED track does not need it.

Emits, per scene, exactly what :mod:`taniteval.strategic_optionset` consumes:
``results/strategic_gt/strategic_gt_<scene>.json``.

Usage::

    python3 build_t1_labels.py --tier T1 --limit 0 --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nurec_hf_survey import open_scene_members  # noqa: E402

#: everything the label + the later camera join need.  ``parsed_config.yaml`` carries the
#: f-theta intrinsics and ``rig_trajectories.json`` the per-frame rig poses/timestamps, so
#: they are pulled now rather than in a second pass over the same archives.
MEMBERS = ["map.xodr", "pose_record.json", "data_info.json", "parsed_config.yaml",
           "rig_trajectories.json",
           "clipgt/egomotion_estimate.parquet", "clipgt/intersection_area.parquet",
           "clipgt/lane.parquet"]


def read_tier(tsv: Path, tier="T1"):
    """``[(turn_deg, scene_id, category)]`` for one tier of the survey shortlist."""
    rows = []
    for line in tsv.read_text().splitlines():
        if line.startswith("#") or line.startswith("tier\t"):
            continue
        f = line.split("\t")
        if len(f) < 4 or f[0] != tier:
            continue
        rows.append((float(f[1]), f[2], f[3]))
    return rows


def pull(sid: str, root: Path, block=1 << 16):
    d = root / sid
    need = [m for m in MEMBERS if not (d / m).exists()]
    if not need:
        return {"scene_id": sid, "cached": True, "fetched_bytes": 0}
    blobs, meta = open_scene_members(sid, MEMBERS, block=block)
    (d / "clipgt").mkdir(parents=True, exist_ok=True)
    for k, v in blobs.items():
        (d / k).write_bytes(v)
    return {"scene_id": sid, "cached": False, "fetched_bytes": meta["fetched_bytes"],
            "usdz_bytes": meta["usdz_bytes"], "members": sorted(blobs)}


def label(sid: str, root: Path, out: Path, horizon_m=60.0):
    from strategic_gt import strategic_gt
    d = root / sid
    rep = strategic_gt(d / "clipgt", d / "map.xodr", d / "pose_record.json",
                       horizon_m=horizon_m)
    p = out / f"strategic_gt_{sid}.json"
    p.write_text(json.dumps(rep, indent=1, default=str))
    return {
        "scene_id": sid,
        "ADMISSIBLE": rep["ADMISSIBLE"],
        "selfconsistency_worst_deg": rep["SELFCONSISTENCY_CONTROL"][
            "worst_abs_err_deg_untruncated"],
        "alignment_rms_m": rep["alignment_rms_m"],
        "n_poses": rep["n_poses"],
        "n_decision_events": rep["n_decision_events"],
        "n_SCOREABLE_events": rep["n_SCOREABLE_events"],
        "branching_max": rep["branching_factor"]["max"],
        "n_admissible_poses": rep["n_admissible_poses"],
        "gt_class_share": rep["route_gt_class_share_over_events"],
        "path": str(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str(HERE / "results" / "junction_turn_scenes.tsv"))
    ap.add_argument("--tier", default="T1")
    ap.add_argument("--members-root", default=str(HERE / "scene_members"))
    ap.add_argument("--out", default=str(HERE / "results" / "strategic_gt_t1"))
    ap.add_argument("--index", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--horizon-m", type=float, default=60.0)
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()

    root = Path(a.members_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_tier(Path(a.tsv), a.tier)
    rows = rows[a.offset:]
    if a.limit:
        rows = rows[:a.limit]
    ids = [r[1] for r in rows]
    print(f"{len(ids)} {a.tier} scenes", flush=True)

    t0 = time.time()
    pulls, errs = [], []

    def work(sid):
        try:
            return pull(sid, root)
        except Exception as e:                                    # noqa: BLE001
            return {"scene_id": sid, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, ids), 1):
            pulls.append(r)
            if i % 20 == 0:
                mb = sum(p.get("fetched_bytes", 0) for p in pulls) / 1e6
                print(f"  pull {i}/{len(ids)}  {time.time()-t0:.0f}s  {mb:.1f} MB",
                      flush=True)
    mb = sum(p.get("fetched_bytes", 0) for p in pulls) / 1e6
    n_err = sum(1 for p in pulls if "error" in p)
    print(f"PULL done: {len(pulls)} scenes, {n_err} errors, {mb:.1f} MB, "
          f"{time.time()-t0:.0f}s", flush=True)
    if a.pull_only:
        return

    # labels are CPU-bound and use numpy internally -> serial, but fast enough
    t1 = time.time()
    labs = []
    for i, sid in enumerate(ids, 1):
        if any(p["scene_id"] == sid and "error" in p for p in pulls):
            errs.append({"scene_id": sid, "stage": "pull"})
            continue
        try:
            labs.append(label(sid, root, out, a.horizon_m))
        except Exception as e:                                    # noqa: BLE001
            errs.append({"scene_id": sid, "stage": "label",
                         "error": f"{type(e).__name__}: {e}",
                         "tb": traceback.format_exc()[-600:]})
        if i % 10 == 0:
            print(f"  label {i}/{len(ids)}  {time.time()-t1:.0f}s", flush=True)

    adm = [x for x in labs if x["ADMISSIBLE"]]
    with_branch = [x for x in adm if x["n_SCOREABLE_events"] > 0]
    idx = {
        "tool": "build_t1_labels.py",
        "evidence_class": "MEASURED",
        "tier": a.tier,
        "horizon_m": a.horizon_m,
        "n_scenes_attempted": len(ids),
        "n_pull_or_label_errors": len(errs),
        "n_labelled": len(labs),
        "n_admissible_selfconsistency": len(adm),
        "n_refused_selfconsistency": len(labs) - len(adm),
        "n_scenes_with_a_SCOREABLE_branch": len(with_branch),
        "n_SCOREABLE_events_total": sum(x["n_SCOREABLE_events"] for x in adm),
        "fetched_MB": round(mb, 1),
        "usdz_MB_avoided": round(
            sum(p.get("usdz_bytes", 0) for p in pulls) / 1e6, 1),
        "seconds": round(time.time() - t0, 1),
        "scenes": labs,
        "errors": errs,
    }
    p = Path(a.index or (out.parent / f"strategic_gt_{a.tier}_index.json"))
    p.write_text(json.dumps(idx, indent=1, default=str))
    print(json.dumps({k: v for k, v in idx.items() if k not in ("scenes", "errors")},
                     indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
