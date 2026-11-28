# -*- coding: utf-8 -*-
"""Measure the strategic-panel nav caption's wrapped line count with the EXACT
glyphs and the EXACT font the renderer will use. The ASCII-substituted probe
read 6 lines where the renderer read 7 -- the em dash and U+2212 MINUS are the
difference, so only the real characters are admissible here."""
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = sys.argv[1] if len(sys.argv) > 1 else \
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f = ImageFont.truetype(FONT, 13)
im = Image.new("RGB", (10, 10))
d = ImageDraw.Draw(im)


def wrap(text, max_w=330):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=f) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


ARCH = ("GIVEN INPUT, not a prediction — the clip's v7.2 "
        "nav_command (ego-future); an ORACLE absent at deployment. "
        "Both heads are WIRED to it (E13), but the route head is "
        "nav-INSENSITIVE BY CONSTRUCTION: it reads pooled VISION, "
        "not this token.")

CANDS = {
    "CURRENT arch+unmeasured": ARCH
    + " (true−shuffled NOT measured for this step.)",
    "arch only": ARCH,
    "A no-egofuture +unmeas": ("GIVEN INPUT, not a prediction — the clip's "
                               "v7.2 nav_command; an ORACLE absent at "
                               "deployment. Both heads are WIRED to it (E13), "
                               "but the route head is nav-INSENSITIVE BY "
                               "CONSTRUCTION: it reads pooled VISION, not this "
                               "token. (true−shuffled NOT measured for "
                               "this step.)"),
    "B ascii-minus +unmeas": ARCH + " (true-shuffled NOT measured here.)",
    "C tight +unmeas": ("GIVEN INPUT, not a prediction — the clip's v7.2 "
                        "nav_command; an ORACLE absent at deployment. Both "
                        "heads are WIRED to it (E13), but the route head is "
                        "nav-INSENSITIVE BY CONSTRUCTION: it reads pooled "
                        "VISION, not this token. (true-shuffled not measured "
                        "for this step.)"),
}

print("font:", FONT)
for k, v in CANDS.items():
    ls = wrap(v)
    print("%-26s lines=%d   last=%r" % (k, len(ls), ls[-1]))

# CONTROL: a string that MUST wrap to more than 6 lines, so a "6" above is a
# real measurement and not a probe that silently returns a constant.
ctrl = wrap(" ".join(["controlword"] * 60))
print("CONTROL (must be > 6): lines=%d" % len(ctrl))
