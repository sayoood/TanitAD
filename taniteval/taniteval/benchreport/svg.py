"""A tiny SVG builder for the W5 charts (stdlib only).

Marks reference CSS custom properties through classes (``m-s1``, ``m-win`` …) so one SVG string renders
correctly inside the themed HTML page (light / dark / print) AND as a standalone ``fig/*.svg`` (which
embeds the same token block). Every mark can carry ``data-*`` attributes; numbers carry ``data-k`` /
``data-v`` / ``data-fmt`` for the parse-back verifier, and ``data-tip`` for the hover / focus tooltip.
"""
from __future__ import annotations

import html
import math

CHART_CSS = (
    ".viz{font:12px/1.35 system-ui,-apple-system,'Segoe UI',sans-serif;fill:var(--ink)}"
    # a surface-coloured halo keeps labels legible where a reference line or gridline crosses them
    ".viz text{paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}"
    ".viz text.in-win,.viz text.in-tie,.viz text.in-loss{stroke:none}"
    ".viz .t-ink{fill:var(--ink)}.viz .t-2{fill:var(--ink-2)}.viz .t-mut{fill:var(--muted)}"
    ".viz .t-num{font-variant-numeric:tabular-nums}.viz .t-b{font-weight:600}.viz .t-sm{font-size:11px}"
    ".viz .grid{stroke:var(--grid);stroke-width:1}.viz .axis{stroke:var(--axis);stroke-width:1}"
    ".viz .ref{stroke:var(--ref);stroke-width:1.5}"
    ".viz .m-s1{fill:var(--s1)}.viz .m-s2{fill:var(--s2)}.viz .m-s3{fill:var(--s3)}"
    ".viz .m-deemph{fill:var(--deemph)}.viz .m-win{fill:var(--win)}.viz .m-loss{fill:var(--loss)}"
    ".viz .m-tie{fill:var(--tie)}.viz .m-pub{fill:none;stroke:var(--ink-2);stroke-width:1.5}"
    ".viz .dot{stroke:var(--surface);stroke-width:2}"
    ".viz .l-s1{stroke:var(--s1)}.viz .l-s2{stroke:var(--s2)}.viz .l-s3{stroke:var(--s3)}"
    ".viz .in-win{fill:var(--win-ink)}.viz .in-tie{fill:var(--tie-ink)}.viz .in-loss{fill:var(--loss-ink)}"
    ".viz [data-tip]{cursor:default}.viz [data-tip]:hover,.viz [data-tip]:focus{opacity:.82;outline:none}"
    ".viz .hit{fill:transparent}"
)


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def attrs(d: dict | None) -> str:
    if not d:
        return ""
    return "".join(f' {k}="{esc(v)}"' for k, v in d.items() if v is not None)


def fnum(x: float) -> str:
    """Compact coordinate formatting (geometry only — never a reported number)."""
    return f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


class Svg:
    def __init__(self, width: float, height: float, title: str, desc: str = "", fig_id: str = ""):
        self.w, self.h, self.title, self.desc, self.fig_id = width, height, title, desc, fig_id
        self.parts: list[str] = []

    # --- primitives ------------------------------------------------------------------------
    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, cls: str, rx: float = 0.0, data: dict | None = None, tip: str | None = None,
             focusable: bool = False) -> None:
        w = max(float(w), 0.0)
        extra = dict(data or {})
        if tip:
            extra["data-tip"] = tip
        if focusable:
            extra["tabindex"] = "0"
        r = f' rx="{fnum(rx)}"' if rx else ""
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<rect class="{cls}" x="{fnum(x)}" y="{fnum(y)}" width="{fnum(w)}" height="{fnum(h)}"'
                 f'{r}{attrs(extra)}>{t}</rect>')

    def bar_h(self, x0, y, length, thickness, cls: str, data=None, tip=None) -> None:
        """Horizontal bar: square at the baseline (x0), 4px rounded data-end (dataviz mark spec)."""
        length = float(length)
        if length <= 0.5:
            self.rect(x0, y, max(length, 0.0), thickness, cls, data=data, tip=tip, focusable=bool(tip))
            return
        r = min(4.0, thickness / 2, length / 2)
        x1 = x0 + length
        d = (f"M{fnum(x0)},{fnum(y)} H{fnum(x1 - r)} Q{fnum(x1)},{fnum(y)} {fnum(x1)},{fnum(y + r)} "
             f"V{fnum(y + thickness - r)} Q{fnum(x1)},{fnum(y + thickness)} {fnum(x1 - r)},{fnum(y + thickness)} "
             f"H{fnum(x0)} Z")
        extra = dict(data or {})
        if tip:
            extra["data-tip"] = tip
            extra["tabindex"] = "0"
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<path class="{cls}" d="{d}"{attrs(extra)}>{t}</path>')

    def line(self, x1, y1, x2, y2, cls: str, data=None) -> None:
        self.add(f'<line class="{cls}" x1="{fnum(x1)}" y1="{fnum(y1)}" x2="{fnum(x2)}" y2="{fnum(y2)}"'
                 f'{attrs(data)}/>')

    def circle(self, cx, cy, r, cls: str, data=None, tip: str | None = None) -> None:
        extra = dict(data or {})
        if tip:
            extra["data-tip"] = tip
            extra["tabindex"] = "0"
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<circle class="{cls}" cx="{fnum(cx)}" cy="{fnum(cy)}" r="{fnum(r)}"{attrs(extra)}>{t}</circle>')

    def text(self, x, y, s: str, cls: str = "t-ink", anchor: str = "start", data: dict | None = None,
             baseline: str = "middle") -> None:
        self.add(f'<text class="{cls}" x="{fnum(x)}" y="{fnum(y)}" text-anchor="{anchor}" '
                 f'dominant-baseline="{baseline}"{attrs(data)}>{esc(s)}</text>')

    def open_g(self, data: dict | None = None, cls: str = "") -> None:
        c = f' class="{cls}"' if cls else ""
        self.add(f"<g{c}{attrs(data)}>")

    def close_g(self) -> None:
        self.add("</g>")

    # --- output ----------------------------------------------------------------------------
    def to_string(self, standalone_css: str | None = None) -> str:
        style = f"<style>{standalone_css}</style>" if standalone_css else ""
        fid = f' id="{esc(self.fig_id)}"' if self.fig_id else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" class="viz"{fid} viewBox="0 0 {fnum(self.w)} '
                f'{fnum(self.h)}" width="{fnum(self.w)}" height="{fnum(self.h)}" role="img" '
                f'aria-labelledby="{esc(self.fig_id)}-t"><title id="{esc(self.fig_id)}-t">{esc(self.title)}'
                f"</title>{('<desc>' + esc(self.desc) + '</desc>') if self.desc else ''}{style}"
                + "".join(self.parts) + "</svg>")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering [lo, hi] (axis geometry only)."""
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step          # the axis always ENDS on a labelled tick
    ticks, v = [], start
    while v <= stop + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def tick_label(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s
