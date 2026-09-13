"""Convert the package media PNGs to JPEG (q 90) to keep the repo light; rewrite RESULT references; verify."""
from pathlib import Path
from PIL import Image

PK = Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-09-13-qwen-drive-usage-review")
R = PK / "RESULT.md"
s = R.read_text(encoding="utf-8")
before = after = 0
for png in sorted((PK / "media").glob("*.png")):
    jpg = png.with_suffix(".jpg")
    Image.open(png).convert("RGB").save(jpg, quality=90, optimize=True)
    a, b = png.stat().st_size, jpg.stat().st_size
    assert b > 50_000, f"suspiciously small {jpg.name}"
    before += a; after += b
    s = s.replace(png.name, jpg.name)
    png.unlink()
R.write_text(s, encoding="utf-8", newline="\n")
left = [n for n in ("v1_vs_v2_4fbd97b6a4b7_f0.png", "demo_A0_as_shipped_vs_A2_focal_changed.png") if n in s]
print(f"PNG {before:,} B -> JPG {after:,} B; png refs left in RESULT: {left}; media now: {sorted(p.name for p in (PK / 'media').iterdir())}")
