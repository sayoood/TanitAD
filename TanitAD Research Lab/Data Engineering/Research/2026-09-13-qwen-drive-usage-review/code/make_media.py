"""Package media for the review: every image that carries a frame token is redacted to sha12."""
import hashlib, json, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SP = Path(__file__).resolve().parent.parent
PKG = Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-09-13-qwen-drive-usage-review/media")
PKG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SP / "v2"))
from annotate_and_video import annotate  # noqa: E402

FRAMES = Path(r"C:/Users/Admin/qwenvis/v2/frames_legacy")


def font(sz):
    for n in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def redact_header(im: Image.Image, token_redacted: str, label: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 420, 26], fill=(255, 255, 255))
    d.text((12, 3), f"{token_redacted}   {label}", fill=(47, 52, 55), font=font(17))
    return im


def meta_of(tok):
    return json.loads((FRAMES / tok / "meta.json").read_text(encoding="utf-8"))


# 1) the demo control: A0 vs A2 (Qwen demo token, not a PhysicalAI clip)
shutil.copy2(SP / "ctrl" / "pull" / "A0_vs_A2.png", PKG / "demo_A0_as_shipped_vs_A2_focal_changed.png")

# 2) v1 vs v2 at the same instant, both decoded by the upstream visualizer, headers redacted
tok = "4fbd97b6a4b7_f0"; m = meta_of(tok); red = m["clip_sha12"] + "_f0"
a = redact_header(Image.open(SP / "v2" / "cmp" / "v1_4fbd97b6a4b7_f0.png").convert("RGB"), red, "v1 packing")
b = redact_header(Image.open(SP / "v2" / "pull_legacy" / "4fbd97b6a4b7_f0.png").convert("RGB"), red, "v2 packing")
h = min(a.height, b.height)
im = Image.new("RGB", (a.width + b.width + 40, h + 100), "white")
im.paste(a.crop((0, 0, a.width, h)), (0, 100)); im.paste(b.crop((0, 0, b.width, h)), (a.width + 40, 100))
d = ImageDraw.Draw(im)
d.text((20, 18), "v1 packing (2026-09-11, REJECTED): 6 views at 110/68/29 deg, off-centre principal point", fill=(190, 30, 30), font=font(40))
d.text((20, 60), "same instant, same weights, decoded by Qwen-Drive's own visualizer", fill=(90, 90, 90), font=font(30))
d.text((a.width + 60, 18), "v2 packing: virtual nuPlan 8-camera rig (f = 1545 px @1920, 63.7 deg)", fill=(20, 110, 40), font=font(40))
d.text((a.width + 60, 60), "green = obstacle.offline GT; occupancy GT = LiDAR pseudo-occupancy; no map GT exists", fill=(90, 90, 90), font=font(30))
im.thumbnail((2600, 2600)); im.save(PKG / f"v1_vs_v2_{red}.png")

# 3) geometry check sheet
shutil.copy2(SP / "v2" / "gtproj_4fbd97b6a4b7_f0.png", PKG / f"gt_projection_check_{red}.png")

# 4) annotated legacy panels for the near-empty control and the night/snow scene
for tok in ("6924358fafe0_f0", "2bb37d62b419_f1"):
    m = meta_of(tok)
    annotate(SP / "v2" / "pull_legacy" / f"{tok}.png", m, PKG / f"v2_{m['clip_sha12']}_{tok.split('_')[1]}.png", extra="legacy instant")

# 5) sequence stills already annotated + redacted by annotate_and_video.py
for src, dst in ((SP / "v2" / "ann_seq_4fbd97b6a4b7" / "f0060.png", "v2_seq_4fbd97b6a4b7_t12.0s.png"),):
    shutil.copy2(src, PKG / dst)
print("media:", sorted(p.name for p in PKG.iterdir()))
