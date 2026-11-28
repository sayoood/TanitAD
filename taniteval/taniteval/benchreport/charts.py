"""Figure builders for the W5 run report and the W4 leaderboard components.

Form choices (the `dataviz` skill's heuristic): magnitude per arm -> horizontal bars, EMPHASIS (the
primary arm in the accent hue, the rest de-emphasised); floors -> labelled reference lines; paired
win / tie / loss -> diverging stacked bars (blue win, neutral tie, red loss); per-log / per-speed-band ->
dot plots capped at THREE series (the validated all-pairs cap: primary arm, CV, STOP); grids of numbers
-> heat tables (one sequential hue). Every chart has a table twin; every printed number carries
``data-k`` (its summary.json pointer), ``data-v`` (full precision) and ``data-fmt`` for the verifier.
"""
from __future__ import annotations

import math

from .svg import Svg, esc, nice_ticks, tick_label

# ---- renderer-side number formats (the verifier re-implements these independently) --------------
FMT = {
    "f4": lambda v: f"{v:.4f}", "f3": lambda v: f"{v:.3f}", "f2": lambda v: f"{v:.2f}",
    "f1": lambda v: f"{v:.1f}", "int": lambda v: f"{int(round(v))}",
    "pct1": lambda v: f"{100.0 * v:.1f} %", "sgn4": lambda v: f"{v:+.4f}", "sgn3": lambda v: f"{v:+.3f}",
}
CHAR_W = 6.4          # px per character at 12px system-ui (layout estimate only)


def fmt(v: float, f: str) -> str:
    return FMT[f](float(v))


def numattrs(k: str | None, v: float, f: str | None) -> dict:
    d = {"data-v": repr(float(v))}
    if k:
        d["data-k"] = k
    if f:
        d["data-fmt"] = f
    return d


def num_span(k: str | None, v: float, f: str, cls: str = "num") -> str:
    a = numattrs(k, v, f)
    at = "".join(f' {kk}="{esc(vv)}"' for kk, vv in a.items())
    return f'<span class="{cls}"{at}>{esc(fmt(v, f))}</span>'


def _x_scale(d0: float, d1: float, x0: float, w: float):
    span = (d1 - d0) or 1.0
    return lambda v: x0 + (float(v) - d0) / span * w


def _tiers(items: list, xs: list, widths: list, gap: float = 8.0) -> list:
    """Assign label tiers so no two labels on one tier overlap."""
    order = sorted(range(len(items)), key=lambda i: xs[i])
    ends: list[float] = []
    tier = [0] * len(items)
    for i in order:
        left = xs[i] - widths[i] / 2
        for t, e in enumerate(ends):
            if left > e + gap:
                tier[i] = t
                ends[t] = xs[i] + widths[i] / 2
                break
        else:
            tier[i] = len(ends)
            ends.append(xs[i] + widths[i] / 2)
    return tier


# =================================================================================== horizontal bars
def hbar_chart(fig_id: str, title: str, rows: list, *, domain: tuple, unit: str, refs: tuple = (),
               desc: str = "", label_w: int = 250, plot_w: int = 440, value_w: int = 190,
               row_h: int = 26, bar_t: int = 14) -> str:
    """rows: {arm|pub, label, value|None, k, fmt, cls, tip, tag, refusal_k, refusal}
    refs: {floor, label, value, k, fmt} drawn as labelled vertical reference lines."""
    d0, d1 = domain
    ticks = nice_ticks(d0, d1)
    d0, d1 = min(d0, ticks[0]), max(d1, ticks[-1])
    X = _x_scale(d0, d1, label_w, plot_w)
    ref_txt = [f"{r['label']}  {fmt(r['value'], r['fmt'])}" for r in refs]
    ref_x = [X(r["value"]) for r in refs]
    # the label pair straddles its line (label left, value right): width = 2 x the longer half
    ref_w = [2 * max(len(r["label"]), len(fmt(r["value"], r["fmt"]))) * CHAR_W + 8 for r in refs]
    ref_tier = _tiers(list(refs), ref_x, ref_w) if refs else []
    n_t = (max(ref_tier) + 1) if ref_tier else 0
    top = 10 + 16 * n_t + (6 if n_t else 0)
    H = top + row_h * len(rows) + 34
    W = label_w + plot_w + value_w
    s = Svg(W, H, title, desc, fig_id)
    y_plot0, y_plot1 = top, top + row_h * len(rows)
    for t in ticks:                                   # recessive hairline grid + axis labels
        x = X(t)
        s.line(x, y_plot0, x, y_plot1, "grid")
        s.text(x, y_plot1 + 14, tick_label(t), "t-mut t-num t-sm", "middle")
    s.text(label_w + plot_w / 2, y_plot1 + 29, unit, "t-mut t-sm", "middle")
    x_base = X(max(d0, 0.0) if d0 <= 0 <= d1 else d0)
    s.line(x_base, y_plot0, x_base, y_plot1, "axis")
    for i, r in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2
        grp = {"data-arm": r.get("arm"), "data-pub": r.get("pub")}
        s.open_g(grp)
        role = "arm-label" if r.get("arm") else "pub-label"
        lab = r["label"]
        max_ch = int((label_w - 14) / CHAR_W)
        if not r.get("arm") and len(lab) > max_ch:          # published names can be long; arm labels never trimmed
            s.text(label_w - 10, cy, lab[: max_ch - 1] + "…", "t-ink", "end",
                   data={"data-role": role, "data-tip": lab})
        else:
            s.text(label_w - 10, cy, lab, "t-ink", "end", data={"data-role": role})
        if r.get("value") is None:
            msg = r.get("refusal") or "UNAVAILABLE"
            short = msg if len(msg) <= 58 else msg[:55] + "…"
            s.text(label_w + 6, cy, short, "t-mut t-sm", "start",
                   data={"data-refusal-k": r.get("refusal_k"), "data-tip": msg})
        else:
            v = float(r["value"])
            x1 = X(v)
            s.bar_h(min(x_base, x1), cy - bar_t / 2, abs(x1 - x_base), bar_t, r.get("cls", "m-s1"),
                    data=numattrs(r.get("k"), v, None), tip=r.get("tip"))
            s.text(max(x1, x_base) + 6, cy, fmt(v, r["fmt"]), "t-ink t-num", "start",
                   data=numattrs(r.get("k"), v, r["fmt"]))
            if r.get("tag"):
                s.text(max(x1, x_base) + 10 + len(fmt(v, r["fmt"])) * CHAR_W, cy, r["tag"], "t-mut t-sm")
        s.close_g()
    for r, x, t, txt in zip(refs, ref_x, ref_tier, ref_txt):
        s.open_g({"data-arm": r.get("floor"), "data-ref": "1"})
        s.line(x, top - 4 - 16 * (n_t - 1 - t) + 6, x, y_plot1, "ref")
        ty = 10 + 16 * t
        s.text(x - 3, ty, r["label"], "t-2 t-sm t-b", "end", data={"data-role": "arm-short"})
        s.text(x + 3, ty, fmt(r["value"], r["fmt"]), "t-2 t-num t-sm", "start",
               data=numattrs(r.get("k"), r["value"], r["fmt"]))
        s.close_g()
    return s.to_string()


# =================================================================================== win / tie / loss
def wtl_chart(fig_id: str, title: str, rows: list, *, floor_label: str, desc: str = "",
              label_w: int = 250, plot_w: int = 420, value_w: int = 170, row_h: int = 26,
              bar_t: int = 16) -> str:
    """rows: {arm, label, wins, ties, losses, n, k (pointer to the paired block)} or {arm, label, refusal}."""
    H = 30 + row_h * len(rows) + 8
    W = label_w + plot_w + value_w
    s = Svg(W, H, title, desc, fig_id)
    # legend (always present: 3 classes)
    lx = label_w
    for cls, txt in (("m-win", f"win (arm > {floor_label})"), ("m-tie", "tie"),
                     ("m-loss", f"loss (arm < {floor_label})")):
        s.rect(lx, 6, 10, 10, cls, rx=2)
        s.text(lx + 14, 11, txt, "t-2 t-sm")
        lx += 20 + len(txt) * CHAR_W
    for i, r in enumerate(rows):
        y = 28 + i * row_h
        cy = y + row_h / 2
        s.open_g({"data-arm": r["arm"]})
        s.text(label_w - 10, cy, r["label"], "t-ink", "end", data={"data-role": "arm-label"})
        if r.get("refusal"):
            s.text(label_w + 6, cy, r["refusal"][:70], "t-mut t-sm", "start",
                   data={"data-refusal-k": r.get("refusal_k"), "data-tip": r["refusal"]})
            s.close_g()
            continue
        n = r["n"] or 1
        x = float(label_w)
        for key, cls, ink in (("wins", "m-win", "in-win"), ("ties", "m-tie", "in-tie"),
                              ("losses", "m-loss", "in-loss")):
            c = r[key]
            w = plot_w * c / n
            if c > 0:
                gap = 2.0 if w > 3 else 0.0            # 2px surface gap between touching segments
                s.rect(x, cy - bar_t / 2, max(w - gap, 0.5), bar_t, cls,
                       data=numattrs(f"{r['k']}/{key}", c, None),
                       tip=f"{r['label']}: {key} {c} of {r['n']} vs {floor_label}", focusable=True)
                txt = str(c)
                if w - gap >= len(txt) * CHAR_W + 8:
                    s.text(x + (w - gap) / 2, cy, txt, f"{ink} t-num t-sm", "middle",
                           data=numattrs(f"{r['k']}/{key}", c, "int"))
            x += w
        vx = label_w + plot_w + 10
        for j, key in enumerate(("wins", "ties", "losses")):
            s.text(vx, cy, str(r[key]), "t-ink t-num t-sm", "start",
                   data=numattrs(f"{r['k']}/{key}", r[key], "int"))
            vx += len(str(r[key])) * CHAR_W + 4
            if j < 2:
                s.text(vx, cy, "/", "t-mut t-sm")
                vx += 10
        # the n key differs by layout (`n_common` pooled, `n` per stage) — the caller names it, and when
        # it names none the count is printed WITHOUT a verifier key rather than with a guessed one
        s.text(vx + 6, cy, f"n {r['n']}", "t-mut t-num t-sm", "start",
               data=numattrs(r.get("n_k"), r["n"], None) if r.get("n_k") else None)
        s.close_g()
    return s.to_string()


# =================================================================================== dot plot
def dot_chart(fig_id: str, title: str, cats: list, series: list, *, domain: tuple, unit: str,
              desc: str = "", label_w: int = 300, plot_w: int = 470, right_w: int = 60, row_h: int = 28) -> str:
    """cats: {id, label, n}; series (<= 3): {arm, label, cls, points: {cat_id: (value, k)}}."""
    if len(series) > 3:
        raise ValueError("dot_chart: more than 3 series breaks the validated all-pairs palette cap")
    d0, d1 = domain
    ticks = nice_ticks(d0, d1)
    d0, d1 = min(d0, ticks[0]), max(d1, ticks[-1])
    X = _x_scale(d0, d1, label_w, plot_w)
    top = 30
    H = top + row_h * len(cats) + 34
    W = label_w + plot_w + right_w
    s = Svg(W, H, title, desc, fig_id)
    lx = label_w
    for sr in series:                                   # legend: always present for >= 2 series
        s.open_g({"data-arm": sr["arm"], "data-legend": "1"})
        s.circle(lx + 5, 12, 5, sr["cls"] + " dot")
        s.text(lx + 14, 12, sr["label"], "t-2 t-sm", data={"data-role": "arm-label"})
        s.close_g()
        lx += 26 + len(sr["label"]) * CHAR_W
    y1 = top + row_h * len(cats)
    for t in ticks:
        x = X(t)
        s.line(x, top, x, y1, "grid")
        s.text(x, y1 + 14, tick_label(t), "t-mut t-num t-sm", "middle")
    s.text(label_w + plot_w / 2, y1 + 29, unit, "t-mut t-sm", "middle")
    offs = {1: (0.0,), 2: (-4.0, 4.0), 3: (-6.0, 0.0, 6.0)}[max(len(series), 1)]
    for i, c in enumerate(cats):
        cy = top + i * row_h + row_h / 2
        s.line(label_w, cy, label_w + plot_w, cy, "grid")
        s.text(label_w - 10, cy, c["label"], "t-ink t-sm", "end")
        s.text(label_w + plot_w + 8, cy, f"n {c['n']}", "t-mut t-num t-sm", "start")
        for sr, dy in zip(series, offs):
            p = sr["points"].get(c["id"])
            if p is None or p[0] is None:
                continue
            v, k = p
            s.open_g({"data-arm": sr["arm"]})
            s.circle(X(v), cy + dy, 5, sr["cls"] + " dot", data=numattrs(k, v, None),
                     tip=f"{sr['label']} · {c['label']}: {fmt(v, 'f4')} (n {c['n']})")
            s.close_g()
    return s.to_string()


# =================================================================================== HTML tables
def heat_table(table_id: str, caption: str, cols: list, rows: list, *, f: str = "f3",
               first_col: str = "arm") -> str:
    """cols: [(key, header, tooltip)]; rows: {arm, label, cells: {key: (value, k) | ('REFUSED', text, k)}}."""
    from .palette import heat_bin
    th = "".join(f'<th scope="col" title="{esc(tip)}">{esc(h)}</th>' for _k, h, tip in cols)
    out = [f'<table class="heat" id="{esc(table_id)}"><caption>{esc(caption)}</caption>'
           f'<thead><tr><th scope="col">{esc(first_col)}</th>{th}</tr></thead><tbody>']
    for r in rows:
        tds = []
        for key, _h, _t in cols:
            c = r["cells"].get(key)
            if c is None:
                tds.append('<td class="na">—</td>')
            elif c[0] == "REFUSED":
                tds.append(f'<td class="refused" data-refusal-k="{esc(c[2] or "")}" title="{esc(c[1])}">'
                           f'{esc(c[1][:18])}</td>')
            else:
                v, k = c
                if v is None:
                    tds.append('<td class="na">—</td>')
                    continue
                q = heat_bin(v)
                a = numattrs(k, v, f)
                at = "".join(f' {kk}="{esc(vv)}"' for kk, vv in a.items())
                tds.append(f'<td class="h q{q}"{at}>{esc(fmt(v, f))}</td>')
        out.append(f'<tr data-arm="{esc(r["arm"])}"><th scope="row" data-role="arm-label">{esc(r["label"])}'
                   f'</th>{"".join(tds)}</tr>')
    out.append("</tbody></table>")
    return "".join(out)


def plain_table(table_id: str, caption: str, header: list, rows: list) -> str:
    """rows: {arm|None, label, cells: [html strings]} — cells are pre-rendered (num_span or text)."""
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in header)
    out = [f'<table class="plain" id="{esc(table_id)}"><caption>{esc(caption)}</caption>'
           f"<thead><tr>{th}</tr></thead><tbody>"]
    for r in rows:
        arm = f' data-arm="{esc(r["arm"])}"' if r.get("arm") else ""
        role = ' data-role="arm-label"' if r.get("arm") else ""
        out.append(f'<tr{arm}><th scope="row"{role}>{esc(r["label"])}</th>'
                   + "".join(f"<td>{c}</td>" for c in r["cells"]) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def nice_domain(values: list, *, lo: float | None = None, hi: float | None = None, pad: float = 0.04) -> tuple:
    vs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    a = min(vs) if vs else 0.0
    b = max(vs) if vs else 1.0
    a = lo if lo is not None else min(a, 0.0)
    b = hi if hi is not None else b + (b - a) * pad
    return (a, b if b > a else a + 1.0)
