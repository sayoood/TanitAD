"""Prepare KNOWN-intrinsics validation frames for GeoCalib (P2).

Builds four sets, each with per-image ground-truth intrinsics in manifest.json:
  comma_native/     : comma2k19 road cam, native res, GT rectilinear focal=910 px
  comma_480p/       : same frames downscaled ~480p (matches YouTube pilot res),
                      GT focal scales with height, GT vfov unchanged
  comma_focalsweep/ : ONE clean comma frame, centre-cropped by factor s and
                      resized back to native -> EXACT known focal = 910*s
  physicalai_native/: PhysicalAI front-wide 120deg f-theta fisheye, native res,
                      records per-clip fw_poly / paraxial focal (fisheye, NOT a
                      clean pinhole GT -> robustness datapoint)

Output dir: scratchpad/gc_frames/  (PNGs + manifest.json). Then scp to eval pod.
Content-independent geometry; frames are only a concrete substrate.
"""
from __future__ import annotations
import glob, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.data.calib import COMMA2K19_FOCAL_PX, focal_crop_resize  # noqa

OUT = Path(r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/gc_frames")
COMMA = Path(r"C:/Users/Admin/tanitad-data/comma2k19/extracted")
PAI = Path(r"C:/Users/Admin/tanitad-data/physicalai")


def vfov_deg(f, h):
    return math.degrees(2 * math.atan((h / 2.0) / f))


def hfov_deg(f, w):
    return math.degrees(2 * math.atan((w / 2.0) / f))


def save_png(arr_hw3_u8, path):
    from PIL import Image
    Image.fromarray(arr_hw3_u8).save(path)


def decode_one(video_path, skip=120):
    """Decode a single mid-clip RGB frame -> uint8 [H,W,3]."""
    import av
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]; st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i >= skip:
                return fr.to_ndarray(format="rgb24")
    return None  # too short


def main():
    manifest = []
    for sub in ("comma_native", "comma_480p", "comma_focalsweep", "physicalai_native"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # ---- comma2k19 native + 480p ------------------------------------------- #
    segs = sorted(glob.glob(str(COMMA / "**" / "video.hevc"), recursive=True))
    print(f"comma segments found: {len(segs)}")
    picks = segs[:: max(1, len(segs) // 14)][:12]  # ~12 spread across the set
    clean_frame = None
    for k, seg in enumerate(picks):
        try:
            rgb = decode_one(seg)
        except Exception as e:
            print("comma decode fail", seg, e); continue
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        name = f"comma_{k:02d}.png"
        save_png(rgb, OUT / "comma_native" / name)
        manifest.append({
            "set": "comma_native", "file": f"comma_native/{name}",
            "src": os.path.relpath(seg, COMMA).replace("\\", "/"),
            "width": W, "height": H, "model": "pinhole",
            "gt_focal_px": COMMA2K19_FOCAL_PX,
            "gt_vfov_deg": round(vfov_deg(COMMA2K19_FOCAL_PX, H), 3),
            "gt_hfov_deg": round(hfov_deg(COMMA2K19_FOCAL_PX, W), 3)})
        # 480p downscale (short side ~480), preserve intrinsics scaling
        s = 480.0 / H
        Wd, Hd = int(round(W * s)), int(round(H * s))
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
        td = F.interpolate(t, size=(Hd, Wd), mode="bilinear", align_corners=False)
        rgbd = td[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
        f_d = COMMA2K19_FOCAL_PX * s
        save_png(rgbd, OUT / "comma_480p" / name)
        manifest.append({
            "set": "comma_480p", "file": f"comma_480p/{name}",
            "src": os.path.relpath(seg, COMMA).replace("\\", "/"),
            "width": Wd, "height": Hd, "model": "pinhole",
            "gt_focal_px": round(f_d, 3),
            "gt_vfov_deg": round(vfov_deg(f_d, Hd), 3),
            "gt_hfov_deg": round(hfov_deg(f_d, Wd), 3)})
        if clean_frame is None:
            clean_frame = (rgb, W, H, seg)

    # ---- controlled focal sweep (exact relative GT) ------------------------ #
    if clean_frame is not None:
        rgb, W, H, seg = clean_frame
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
        for s in (1.0, 1.25, 1.5, 2.0):
            # centre-crop by 1/s then resize back to (H,W): focal -> 910*s
            cw, ch = int(round(W / s)), int(round(H / s))
            top, left = (H - ch) // 2, (W - cw) // 2
            crop = t[..., top:top + ch, left:left + cw]
            up = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
            rgbs = up[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
            f_s = COMMA2K19_FOCAL_PX * s
            name = f"sweep_x{str(s).replace('.', 'p')}.png"
            save_png(rgbs, OUT / "comma_focalsweep" / name)
            manifest.append({
                "set": "comma_focalsweep", "file": f"comma_focalsweep/{name}",
                "src": os.path.relpath(seg, COMMA).replace("\\", "/"),
                "width": W, "height": H, "model": "pinhole",
                "sweep_factor": s, "gt_focal_px": round(f_s, 3),
                "gt_vfov_deg": round(vfov_deg(f_s, H), 3),
                "gt_hfov_deg": round(hfov_deg(f_s, W), 3)})

    # ---- PhysicalAI front-wide (f-theta fisheye) --------------------------- #
    try:
        from tanitad.data.physicalai import intrinsics_for_clip
        mp4s = sorted(glob.glob(str(PAI / "r0" / "camera_front_wide" / "*.mp4")))
        print(f"physicalai r0 front-wide mp4s: {len(mp4s)}")
        pick = mp4s[:: max(1, len(mp4s) // 10)][:8]
        for k, mp4 in enumerate(pick):
            clip_id = Path(mp4).name.split(".camera_front_wide")[0]
            try:
                rgb = decode_one(mp4, skip=60)
                intr = intrinsics_for_clip(clip_id, str(PAI))
            except Exception as e:
                print("pai fail", clip_id, e); continue
            if rgb is None or intr is None:
                print("pai skip (no frame/intr)", clip_id); continue
            H, W = rgb.shape[:2]
            name = f"pai_{k:02d}.png"
            save_png(rgb, OUT / "physicalai_native" / name)
            manifest.append({
                "set": "physicalai_native", "file": f"physicalai_native/{name}",
                "src": clip_id, "width": W, "height": H, "model": "ftheta_fisheye",
                "paraxial_focal_px": round(intr.paraxial_focal, 3),
                "cx": round(intr.cx, 2), "cy": round(intr.cy, 2),
                "intr_width": intr.width, "intr_height": intr.height,
                "fw_poly": [round(p, 5) for p in intr.poly],
                "note": "120deg f-theta fisheye; paraxial focal is CENTRE focal, "
                        "NOT a pinhole GT (GeoCalib pinhole fit will differ)"})
    except Exception as e:
        print("physicalai block failed:", e)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # summary
    from collections import Counter
    c = Counter(m["set"] for m in manifest)
    print("WROTE", dict(c), "-> total", len(manifest), "images at", OUT)


if __name__ == "__main__":
    main()
