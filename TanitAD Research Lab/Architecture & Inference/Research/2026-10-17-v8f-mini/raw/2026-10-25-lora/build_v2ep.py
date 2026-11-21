"""Rebuild ``<clip>.v2ep.pt`` (256x640 cylindrical, PNG) from the LOCAL mp4s, into a scratch directory only.

The route is ``pod_build_b1_epcache.build_one`` (read, not edited): ``v2_compressed.build_compressed(clip, out, size=256, n_stack=3, frame=CanonicalFrame.from_hfov(120, 256,
640, 'cylindrical'), projection_mode='cylindrical', codec='png')``. The corpus layout the builder expects is recreated under ``--root`` with COPIES of the inputs
(``r0/camera_front_wide/<clip>.mp4`` + ``.timestamps.parquet``, ``calibration/physicalai_front_wide_intrinsics.csv``, a STORED ``egomotion_all.zip`` of
``<clip>.egomotion.parquet`` made from ``labels/egomotion_alpamayo/<clip>.parquet``). Nothing is read from or written to any other place.

Usage: python build_v2ep.py --out <dir> --clips <id> [<id> ...] [--root <scratch root>] [--frame 256|416]
``--frame 416`` builds the 416x1024 cylinder and resamples it to 256x640 with the cache builder's own ``resize416_to_256`` (the route the dev-box test cache took).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

DATA = Path("C:/Users/Admin/tanitad-data/physicalai")
REPO = Path(__file__).resolve().parents[6]


def stage_root(root: Path, clips) -> Path:
    cam = root / "r0" / "camera_front_wide"
    cam.mkdir(parents=True, exist_ok=True)
    (root / "calibration").mkdir(exist_ok=True)
    (root / "labels" / "egomotion").mkdir(parents=True, exist_ok=True)
    csv = root / "calibration" / "physicalai_front_wide_intrinsics.csv"
    if not csv.exists():
        shutil.copy2(DATA / "calibration" / "physicalai_front_wide_intrinsics.csv", csv)
    z = root / "labels" / "egomotion" / "egomotion_all.zip"
    have = set()
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            have = set(zf.namelist())
    with zipfile.ZipFile(z, "a", compression=zipfile.ZIP_STORED) as zf:
        for c in clips:
            for ext in ("mp4", "timestamps.parquet"):
                dst = cam / f"{c}.{ext}"
                src = DATA / "camera" / "camera_front_wide_120fov" / f"{c}.{ext}"
                if not src.is_file():
                    raise FileNotFoundError(f"{src} (looked in {src.parent}; the clip is not among the local mp4s)")
                if not dst.exists():
                    shutil.copy2(src, dst)
            name = f"{c}.egomotion.parquet"
            if name not in have:
                src = DATA / "labels" / "egomotion_alpamayo" / f"{c}.parquet"
                if not src.is_file():
                    raise FileNotFoundError(f"{src} (the clip has no egomotion parquet locally)")
                zf.write(src, name)
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--frame", choices=["256", "416"], default="256")
    ap.add_argument("--stack", default=str(REPO / "stack"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = stage_root(Path(a.root) if a.root else out.parent / "build_root", a.clips)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, a.stack)
    sys.path.insert(0, os.path.join(a.stack, "scripts"))
    import torch
    import v2_compressed as V
    from tanitad.data.calib import CanonicalFrame
    H, W = (256, 640) if a.frame == "256" else (416, 1024)
    frame = CanonicalFrame.from_hfov(120.0, H, W, "cylindrical")
    for c in a.clips:
        t0 = time.time()
        clip = {"clip_id": c, "mp4": str(root / "r0" / "camera_front_wide" / f"{c}.mp4"),
                "timestamps": str(root / "r0" / "camera_front_wide" / f"{c}.timestamps.parquet"),
                "ego_zip": str(root / "labels" / "egomotion" / "egomotion_all.zip")}
        dst = out / f"{c}.v2ep.pt"
        tmp = out / f"{c}.v2ep.pt.build"
        V.build_compressed(clip, str(tmp), size=256, n_stack=3, frame=frame, projection_mode="cylindrical", codec="png")
        if a.frame == "416":
            import importlib.util
            pep_p = REPO / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py"
            spec = importlib.util.spec_from_file_location("pull_encode_pooled", str(pep_p))
            pep = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pep)
            o = torch.load(tmp, map_location="cpu", weights_only=False)
            torch.save(pep.resize416_to_256(o), dst)
            tmp.unlink()
        else:
            os.replace(tmp, dst)
        print(f"built {c} ({a.frame}) {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
