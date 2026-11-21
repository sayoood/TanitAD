"""Annotate the upstream visualizer's panels for PhysicalAI frames and assemble videos.

The panels themselves are rendered by Qwen-Drive's OWN `visualize.render_frame`, unmodified.
This script only ADDS a banner that says what each panel is -- because two of the upstream
titles would otherwise mislead on PhysicalAI data:
  * "occupancy ground truth" is a LiDAR PSEUDO-occupancy here (ground drawn as driveable),
  * "map ground truth" is EMPTY: PhysicalAI-AV ships no map.
Clip ids are gated-confidential: only the sha12 digest is written into any artifact.
"""
import json, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FFMPEG = r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"


def font(sz, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate(png: Path, meta: dict, out: Path, width=1600, extra=""):
    im = Image.open(png).convert("RGB")
    # ⛔ the upstream header prints the frame token, whose prefix is 8 chars of a gated-confidential
    # clip id -- paint it out and write the sha12-based token instead
    red = ImageDraw.Draw(im)
    red.rectangle([0, 0, 420, 26], fill=(255, 255, 255))
    red.text((12, 3), meta["clip_sha12"] + "_" + meta["token"].split("_", 1)[1] + "   nuplan-rig v2", fill=(47, 52, 55), font=font(17))
    s = width / im.width
    im = im.resize((width, int(round(im.height * s / 2)) * 2), Image.LANCZOS)
    banner_h = 118
    canvas = Image.new("RGB", (width, im.height + banner_h), (246, 247, 249))
    canvas.paste(im, (0, banner_h))
    d = ImageDraw.Draw(canvas)
    v = meta["views"]
    t_s = meta.get("t_s")
    d.text((14, 8), f"Qwen-Drive-1.0-4B perception on PhysicalAI-AV  |  clip {meta['clip_sha12']}"
                    + (f"  t = {t_s:5.1f} s" if t_s is not None else "") + f"  |  {extra}",
           fill=(25, 30, 35), font=font(21, True))
    d.text((14, 38), "Rendered by Qwen-Drive's own run_perception.py + visualize_perception.py.  "
                     "Input: virtual nuPlan 8-camera rig (v2), exact reprojection of each PhysicalAI camera.",
           fill=(55, 60, 66), font=font(16))
    d.text((14, 60), "BEV + occupancy 'ground truth' = OUR data: green boxes = obstacle.offline labels; occupancy GT = LiDAR "
                     "pseudo-occupancy (ground drawn as driveable, NOT semantic GT).  Map GT: none (PhysicalAI ships no map).",
           fill=(150, 40, 30), font=font(16))
    d.text((14, 82), f"View yaws: R1/L1 {v['CAM_R1']['yaw_deg']:+.0f}/{v['CAM_L1']['yaw_deg']:+.0f} deg (nuPlan -111/+111), "
                     f"R2/L2 {v['CAM_R2']['yaw_deg']:+.0f}/{v['CAM_L2']['yaw_deg']:+.0f} (nuPlan -141/+141).  "
                     f"B0 = rear tele, {100 * v['CAM_B0']['observed_frac']:.0f} % observed (black = no camera sees it).  "
                     f"GT boxes <= 200 m: {meta['n_gt_boxes']}",
           fill=(55, 60, 66), font=font(16))
    canvas.save(out)


def main():
    vis_dir, frames_dir, out_dir, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(q for q in vis_dir.glob("*.png") if (frames_dir / q.stem / "meta.json").exists())
    t0 = None
    seq = []
    for i, p in enumerate(pngs):
        meta = json.loads((frames_dir / p.stem / "meta.json").read_text(encoding="utf-8"))
        if t0 is None:
            t0 = meta["t_ref_us"]
        meta["t_s"] = (meta["t_ref_us"] - t0) / 1e6 if label.startswith("seq") else None
        o = out_dir / f"f{i:04d}.png"
        annotate(p, meta, o, extra=label)
        seq.append(o)
    print("annotated", len(seq))
    if label.startswith("seq") and seq:
        mp4 = out_dir.parent / f"{label}.mp4"
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", "5", "-i", str(out_dir / "f%04d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("ffmpeg rc", r.returncode, r.stderr[-400:], mp4, mp4.stat().st_size if mp4.exists() else 0)


if __name__ == "__main__":
    main()
