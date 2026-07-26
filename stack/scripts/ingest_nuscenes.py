"""nuScenes: metadata-only value analysis, then (optionally) lake ingest.

THE CHEAP DISCRIMINATING STEP FIRST. ``--analyze`` runs on the ~0.4 GB metadata
archive alone and answers the value question — roundabout / turn / traffic-light
scene counts, the cross-camera off-front-only statistic, and the CAM_FRONT
geometry verdict — for near-zero cost and with **no image byte on disk**. Only if
that justifies it should anyone pull the ~60 GB keyframe blobs and run
``--ingest``.

    python scripts/ingest_nuscenes.py --analyze --root <nuscenes_root> \
        --out analysis.json
    python scripts/ingest_nuscenes.py --ingest  --root <nuscenes_root> \
        --lake  <lake_root>

🔴 LICENSE. nuScenes is CC BY-NC-SA 4.0 -> `nc-research` + `share_alike`. Records
land in the segregated copyleft shard and can NEVER enter TanitDataSet-C. The
export guard refuses them on three independent grounds; do not weaken it.

🔴 ACQUISITION. The corpus cannot be fetched by an agent. A HUMAN must register at
nuscenes.org and accept the Terms of Use. This script never downloads anything —
it reads a root that a human has already populated, and fails with the
acquisition steps if that root is empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parents[1]
if str(STACK) not in sys.path:
    sys.path.insert(0, str(STACK))


def analyze(root: str, version: str, out: str | None, max_scenes: int | None,
            categories: list[str] | None) -> dict:
    """Metadata-only. Touches no image bytes."""
    from tanitad.data import nuscenes as ns
    from tanitad.data.calib import (NUSCENES_CAM_FRONT_INTR_NOMINAL,
                                    pinhole_geometry_report)

    idx = ns.NuScenesIndex(ns.load_tables(root, version))
    scen = ns.corpus_scenario_report(idx)
    vis = ns.corpus_camera_visibility_report(idx, max_scenes=max_scenes,
                                             categories=categories)

    # geometry: report on a REAL per-sample intrinsic, not the nominal constant
    geo_real = None
    for sc in idx.scenes():
        samples = idx.samples_of_scene(sc["token"])
        if not samples:
            continue
        sd = idx.sample_data(samples[0]["token"], ns.EGO_CAMERA)
        if sd is None:
            continue
        intr = idx.camera_intrinsics_of(sd)
        geo_real = {"source": "per-sample calibrated_sensor (MEASURED)",
                    "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                    "width": intr.width, "height": intr.height,
                    **pinhole_geometry_report(intr)}
        break

    rep = {
        "corpus": "nuscenes", "version": version, "root": str(root),
        "license": {"name": "CC-BY-NC-SA-4.0", "class": "nc-research",
                    "share_alike": True, "commercial_ok": False,
                    "tier": "nc",
                    "shard_prefix": "shards/nc-research/sharealike/nuscenes/"},
        "scenario": {k: v for k, v in scen.items() if k != "scenes"},
        "camera_visibility": vis,
        "geometry_cam_front_measured": geo_real,
        "geometry_cam_front_nominal": pinhole_geometry_report(
            NUSCENES_CAM_FRONT_INTR_NOMINAL),
        "n_scene_rows": len(scen.get("scenes", [])),
    }
    if out:
        Path(out).write_text(json.dumps({**rep, "scenes": scen["scenes"]},
                                        indent=2), encoding="utf-8")
        print(f"[nuscenes] analysis -> {out}")
    print(json.dumps(rep, indent=2))
    return rep


def ingest(root: str, lake: str, version: str, size: int, val_frac: float,
           max_units: int | None) -> dict:
    """Full ingest. REQUIRES the keyframe blobs, not just the metadata."""
    from tanitad.lake.ingest import NuScenesIngestor, ingest_source
    ing = NuScenesIngestor(size=size, version=version, val_frac=val_frac)
    summary = ingest_source(ing, root, lake, max_units=max_units)
    assert summary["license_class"] == "nc-research", summary
    assert summary["share_alike"] is True, summary
    for shard in summary["shards"]:
        assert shard.startswith("shards/nc-research/sharealike/nuscenes/"), shard
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"},
                     indent=2))
    print(f"[nuscenes] {len(summary['skipped'])} unit(s) skipped")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="nuScenes root (the dir containing v1.0-trainval/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--analyze", action="store_true",
                    help="metadata-only value analysis (no images needed)")
    ap.add_argument("--ingest", action="store_true",
                    help="build lake records (REQUIRES the ~60 GB keyframes)")
    ap.add_argument("--lake", help="lake root (with --ingest)")
    ap.add_argument("--out", help="write the analysis JSON here")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="cap scenes in the visibility aggregate (it is O(N))")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="category prefixes for the visibility aggregate, "
                         "e.g. vehicle human")
    a = ap.parse_args()

    from tanitad.data.nuscenes import NuScenesTermsError
    try:
        if a.analyze or not a.ingest:
            analyze(a.root, a.version, a.out, a.max_scenes, a.categories)
        if a.ingest:
            if not a.lake:
                ap.error("--ingest requires --lake")
            ingest(a.root, a.lake, a.version, a.size, a.val_frac, a.max_units)
    except NuScenesTermsError as e:
        print(f"\n[nuscenes] NOT ACQUIRED\n{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
