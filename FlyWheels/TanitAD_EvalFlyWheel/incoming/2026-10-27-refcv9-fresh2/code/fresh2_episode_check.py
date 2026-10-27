"""fresh2_episode_check.py -- CONTENT assertions on every usable FRESH2 episode file (L1C FEASIBILITY s3's checks, re-run here): codec png, 416x1024,
cylindrical geometry manifest, poses / actions finite, first / middle / last RAW frame decoded and NOT flat (pixel std >= 5).  A file of zeros must not pass.
Writes result/EPISODE_CHECKS.json (counts + min/median std; sha12 of any failure).  CPU only.  No clip id is printed.  NEW file (FRESH2 agent, 2026-10-27).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.io as tvio  # noqa: E402
from tanitad.data.v2_dataset import _jpeg_offsets  # noqa: E402

EPS = Path("F:/l1c/kit_fresh2/data/refcv6-b1-416x1024-eval139")
IDS = Path("F:/l1c/fresh2/fresh2_ids.json")
PKG = Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(2)
    usable = [r["clip_id"] for r in json.load(open(IDS, encoding="utf-8"))]
    geo = json.load(open(EPS / "_geometry.json", encoding="utf-8"))
    stds, bad = [], []
    for c in usable:
        s12 = hashlib.sha256(c.encode()).hexdigest()[:12]
        d = torch.load(EPS / f"{c}.v2ep.pt", map_location="cpu", weights_only=False, mmap=True)
        why = []
        if str(d.get("codec")) != "png":
            why.append("codec")
        if (int(d.get("image_h", -1)), int(d.get("image_w", -1))) != (416, 1024):
            why.append("geometry")
        if not (torch.isfinite(d["poses"]).all() and torch.isfinite(d["actions"]).all()):
            why.append("nonfinite poses/actions")
        offs = _jpeg_offsets(d["jpeg_len"])
        n = len(d["jpeg_len"])
        ss = []
        for i in (0, n // 2, n - 1):
            f = tvio.decode_png(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])], mode=tvio.ImageReadMode.RGB)
            if tuple(f.shape) != (3, 416, 1024):
                why.append("frame shape")
            ss.append(float(f.float().std()))
        if min(ss) < 5.0:
            why.append("flat frame")
        stds.append(min(ss))
        if why:
            bad.append({"sha12": s12, "why": why})
    out = {"n_checked": len(usable), "n_bad": len(bad), "bad": bad, "frame_std_min_over_clips": float(np.min(stds)), "frame_std_median": float(np.median(stds)),
           "geometry_manifest": {k: geo.get(k) for k in ("frame_tag", "projection_mode", "codec")}, "ALL_PASS": len(bad) == 0 and geo.get("projection_mode") == "cylindrical"}
    (PKG / "result").mkdir(exist_ok=True)
    (PKG / "result" / "EPISODE_CHECKS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"ZZEPCHECK{'PASS' if out['ALL_PASS'] else 'FAIL'}ZZ n {len(usable)} bad {len(bad)} std min {out['frame_std_min_over_clips']:.1f} med {out['frame_std_median']:.1f}", flush=True)


if __name__ == "__main__":
    main()
