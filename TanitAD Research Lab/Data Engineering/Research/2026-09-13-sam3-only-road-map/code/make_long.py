"""Join rendered SAM3-map clip videos into one long sequence: a 2 s title card, then the clip's frames, per clip."""
import json, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SM = Path("/home/nvidia/sam3map")


def font(sz, bold=False):
    for n in (("DejaVuSans-Bold.ttf",) if bold else ("DejaVuSans.ttf",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main(c8s, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    mined = json.load(open(SM / "data" / "mining_counts.json"))
    k = 0
    for q, c8 in enumerate(c8s):
        frames = sorted((SM / f"render_{c8}_v3").glob("f*.png"))
        if not frames:
            print("skip (no frames)", c8); continue
        meta = json.loads(next((SM / "front" / f"seq_{c8}").glob("*/meta.json")).read_text())
        cnt = next((v["counts"] for cid, v in mined.items() if cid.startswith(c8) and "counts" in v), {})
        card = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(card)
        d.text((120, 380), f"example {q + 1} of {len(c8s)}", fill=(170, 178, 174), font=font(34))
        d.text((120, 440), f"clip {meta['clip_sha12']}", fill=(240, 242, 241), font=font(64, True))
        d.text((120, 540), "mined for road paint: SAM3 on 3 frames of the front camera found "
               f"{cnt.get('crosswalk', 0) + cnt.get('zebra crossing', 0)} crosswalk, {cnt.get('arrow painted on road', 0)} arrow, "
               f"{cnt.get('hatched road marking', 0)} hatched-prompt instances", fill=(200, 205, 202), font=font(28))
        d.text((120, 590), "front camera only · ground from the ego path · SAM3-only map v3", fill=(150, 158, 154), font=font(26))
        for _ in range(10):
            card.save(out / f"f{k:05d}.png"); k += 1
        for f in frames:
            os.symlink(f, out / f"f{k:05d}.png"); k += 1
    print("long frames", k)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
