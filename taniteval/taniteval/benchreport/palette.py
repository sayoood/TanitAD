"""Colour tokens for the W5 reports — the `dataviz` skill's VALIDATED reference instance, no invented hex.

Validator output (both modes) is banked in the W5 package: ``raw/palette_validation.txt``
(``validate_palette.js``: slots 1-3 ``--pairs all`` PASS light + dark, light aqua carries a contrast WARN
⇒ every chart ships direct labels AND a table view; win/loss poles PASS both modes).

Roles, not hues: charts reference CSS custom properties (``var(--s1)``), so light / dark are two
SELECTED token sets (dark steps from the same ramps), never an automatic flip.
"""
from __future__ import annotations

# chart chrome & ink (palette.md "Chart chrome & ink")
LIGHT = {
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink-2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)", "good-text": "#006300",
    # categorical slots 1-3 (validated --pairs all): primary arm / plan, CV, STOP
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
    # diverging poles for paired win / loss (validated adjacent), neutral tie
    "win": "#2a78d6", "loss": "#e34948", "tie": "#898781",
    "deemph": "#898781", "ref": "#52514e",
}
DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink-2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)", "good-text": "#0ca30c",
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
    "win": "#3987e5", "loss": "#e66767", "tie": "#898781",
    "deemph": "#898781", "ref": "#c3c2b7",
}
# status (fixed, never themed; always icon + label)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "✓", "warning": "!", "serious": "▲", "critical": "⛔"}

# sequential blue ramp (palette.md "Sequential hue"); light: near-zero recedes to the light surface,
# dark: the anchor flips so near-zero recedes to the dark surface
SEQ_BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
            350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
            600: "#184f95", 650: "#104281", 700: "#0d366b"}
HEAT_STEPS_LIGHT = (100, 200, 300, 400, 500, 600, 700)
HEAT_STEPS_DARK = (700, 600, 500, 400, 300, 200, 100)
N_HEAT = len(HEAT_STEPS_LIGHT)

# the gallery is a raster on its own light surface (not themed): light-mode slots
GALLERY = {"plan": LIGHT["s1"], "cv": LIGHT["s2"], "stop": LIGHT["s3"], "human": LIGHT["ink"],
           "ink": LIGHT["ink"], "ink2": LIGHT["ink-2"], "surface": "#ffffff"}


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def best_ink(fill: str) -> str:
    """Ink for text set INSIDE a coloured fill: whichever of ink / white clears more contrast."""
    return "#0b0b0b" if contrast(fill, "#0b0b0b") >= contrast(fill, "#ffffff") else "#ffffff"


def heat_bin(v: float | None) -> int | None:
    if v is None:
        return None
    x = min(max(float(v), 0.0), 1.0)
    return min(int(x * N_HEAT), N_HEAT - 1)


def css_tokens() -> str:
    """The token block: light default, dark under the OS preference (unless forced light), and both
    forced by ``data-theme`` (the viewer's toggle wins both ways)."""
    def block(t: dict, heat_steps) -> str:
        parts = [f"--{k}:{v};" for k, v in t.items()]
        for k in ("win", "tie", "loss", "s1"):         # text set INSIDE these fills
            parts.append(f"--{k}-ink:{best_ink(t[k])};")
        for i, st in enumerate(heat_steps):
            fill = SEQ_BLUE[st]
            parts.append(f"--q{i}:{fill};--q{i}-ink:{best_ink(fill)};")
        for k, v in STATUS.items():
            parts.append(f"--st-{k}:{v};")
        return "".join(parts)
    light, dark = block(LIGHT, HEAT_STEPS_LIGHT), block(DARK, HEAT_STEPS_DARK)
    return (f":root{{color-scheme:light;{light}}}"
            f"@media (prefers-color-scheme: dark){{:root:not([data-theme=\"light\"]){{color-scheme:dark;{dark}}}}}"
            f":root[data-theme=\"dark\"]{{color-scheme:dark;{dark}}}"
            f":root[data-theme=\"light\"]{{color-scheme:light;{light}}}")
