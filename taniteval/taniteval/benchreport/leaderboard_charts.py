"""Chart components for W4's leaderboard — ``taniteval.leaderboard.build`` calls
:func:`render_leaderboard_charts` with its canonical model and embeds the returned HTML fragment.

Two components, both built from the W5 chart primitives (validated palette, table twin, tooltips):

* **rank bars per protocol** — one axis per protocol tag, NEVER two (``navsim.cross_protocol``): our rows
  highlighted, published rows hollow, our STOP / CV floors as labelled reference lines, and every row
  the protocol has but the chart cannot draw (a HYBRID number, a missing value) listed with its reason;
* **params vs score scatter** — only for rows that carry a parameter count; rows without one are named
  under the chart instead of being silently dropped (and a count that is an ENCODER size is labelled as
  such, never merged with a total).

The fragment carries its own token ``<style>`` so it renders in W4's page (light and dark) unchanged.
"""
from __future__ import annotations

import math
import re

from .charts import CHAR_W, fmt, hbar_chart, nice_domain, numattrs, plain_table
from .page import PAGE_CSS
from .palette import N_HEAT, css_tokens
from .svg import CHART_CSS, Svg, esc, nice_ticks, tick_label

_PARAM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([MB])\b", re.I)


def parse_params(row: dict) -> tuple:
    """-> (params, kind, text) or (None, None, why). Explicit numbers win; an ``encoder`` string is
    parsed only as an ENCODER count and labelled as such."""
    for key, kind in (("params", "total"), ("params_total", "total"), ("params_m", "total_m"),
                      ("params_encoder", "encoder")):
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            n = float(v) * (1e6 if kind == "total_m" else 1.0)
            return n, ("total" if kind != "encoder" else "encoder"), f"{n / 1e6:.0f}M"
    enc = row.get("encoder")
    if isinstance(enc, str):
        m = _PARAM_RE.match(enc)
        if m:
            n = float(m.group(1)) * (1e6 if m.group(2).upper() == "M" else 1e9)
            return n, "encoder", enc.strip()
    return None, None, "no parameter count in the source row"


def _our_rows(model: dict, protocol: str) -> list:
    return [r for r in (model.get("navsim") or []) if r.get("protocol") == protocol]


def _pub_rows(model: dict, protocol: str) -> list:
    pub = model.get("published") or {}
    return [r for r in (pub.get("results") or []) if r.get("protocol") == protocol
            and isinstance(r.get("value"), (int, float))]


def rank_bars(model: dict, protocol: str, fig_id: str = "") -> tuple:
    """-> (svg, table twin html, notes html) for ONE protocol."""
    pub = model.get("published") or {}
    meta = (pub.get("protocols") or {}).get(protocol, {})
    dec = int(meta.get("decimals", 2))
    f = {1: "f1", 2: "f2", 3: "f3", 4: "f4"}.get(dec, "f2")
    ours, pubs = _our_rows(model, protocol), _pub_rows(model, protocol)
    rows, refs, vals, refused, twin = [], [], [], [], []
    for r in sorted(pubs, key=lambda x: -float(x["value"])):
        v = float(r["value"])
        rows.append({"pub": r["id"], "label": r.get("system", r["id"]), "value": v, "fmt": f, "cls": "m-pub",
                     "tag": r.get("role", ""), "k": None,
                     "tip": f"{r.get('system')} — {r.get('source', {}).get('table', '')} "
                            f"({r.get('evidence_class', pub.get('evidence_class_default', ''))})"})
        vals.append(v)
        twin.append({"arm": None, "label": f"{r.get('system')} [{r['id']}]",
                     "cells": [f'<span class="num" data-src="published:{esc(r["id"])}">{esc(fmt(v, f))}</span>',
                               esc(r.get("modality", "")), esc(r.get("evidence_class", pub.get("evidence_class_default", "")))]})
    for r in sorted(ours, key=lambda x: -(x.get("official_x100") or -math.inf)):
        v = r.get("official_x100")
        label = f"TanitAD · {r.get('label', r.get('key'))}"
        if v is None or r.get("hybrid"):
            refused.append((label, "HYBRID — not this arm's own number" if r.get("hybrid") else
                            "no value for this protocol's official statistic"))
            continue
        v = float(v)
        vals.append(v)
        if r.get("kind") == "floor":
            refs.append({"floor": r.get("key"), "label": r.get("key"), "value": v, "fmt": f, "k": None})
        else:
            rows.append({"pub": f"ours:{r.get('key')}", "label": label, "value": v, "fmt": f, "cls": "m-s1",
                         "k": None, "tag": "ours", "tip": f"{label}: {fmt(v, f)} · {r.get('source', '')}"})
        twin.append({"arm": None, "label": label,
                     "cells": [f'<span class="num" data-src="run:{esc(str(r.get("source", "")))}#{esc(str(r.get("key")))}">'
                               f'{esc(fmt(v, f))}</span>', esc(r.get("declared", "")), esc(str(r.get("tier_loop", "")))]})
    if not rows and not refs:
        return "", "", (f'<p class="sub">No drawable row for <code>{esc(protocol)}</code>.</p>')
    unit = f"{meta.get('metric', 'score')} {meta.get('unit', '')} — {protocol}"
    svg = hbar_chart(fig_id or f"lb-{protocol}", meta.get("title", protocol), rows,
                     domain=nice_domain(vals, lo=0.0), unit=unit, refs=tuple(refs),
                     desc=meta.get("definition", ""), label_w=290, plot_w=430, value_w=150)
    notes = ""
    if refused:
        notes = '<p class="sub">not drawn: ' + " · ".join(f"<b>{esc(a)}</b> — {esc(b)}" for a, b in refused) + "</p>"
    twin_html = plain_table(f"lbtab-{protocol}", f"{protocol} — every row on this axis",
                            ["entry", meta.get("metric", "score"), "modality / declared", "evidence / tier"], twin)
    return svg, twin_html, notes


def params_scatter(model: dict, protocol: str, fig_id: str = "") -> tuple:
    """-> (svg, notes html). x = parameters (log10), y = the protocol's score. Ours highlighted."""
    pub = model.get("published") or {}
    meta = (pub.get("protocols") or {}).get(protocol, {})
    dec = int(meta.get("decimals", 2))
    f = {1: "f1", 2: "f2", 3: "f3", 4: "f4"}.get(dec, "f2")
    pts, missing, kinds = [], [], set()
    for r in _pub_rows(model, protocol):
        n, kind, txt = parse_params(r)
        name = r.get("system", r["id"])
        if n:
            pts.append((name, n, float(r["value"]), "m-pub", kind, txt))
            kinds.add(kind)
        else:
            missing.append((name, txt))
    for r in _our_rows(model, protocol):
        v = r.get("official_x100")
        if v is None or r.get("hybrid"):
            continue
        n, kind, txt = parse_params(r)
        label = f"TanitAD · {r.get('label', r.get('key'))}"
        if n:
            pts.append((label, n, float(v), "m-s1", kind, txt))
            kinds.add(kind)
        else:
            missing.append((label, txt))
    if len(pts) < 2:
        names = ", ".join(f"<b>{esc(a)}</b>" for a, _b in missing[:6])
        more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        return "", (f'<p class="sub">params-vs-score scatter not drawn for <code>{esc(protocol)}</code>: '
                    f'{len(pts)} of {len(pts) + len(missing)} rows carry a parameter count '
                    f'(missing: {names}{more}).</p>')
    W, H = 880, 420
    left, right, top, bottom = 64, 24, 28, 52
    xs = [math.log10(p[1]) for p in pts]
    ys = [p[2] for p in pts]
    x0, x1 = min(xs) - 0.15, max(xs) + 0.15
    yt = nice_ticks(min(min(ys), 0.0), max(ys))
    y0, y1 = yt[0], yt[-1]
    X = lambda v: left + (v - x0) / ((x1 - x0) or 1) * (W - left - right)          # noqa: E731
    Y = lambda v: H - bottom - (v - y0) / ((y1 - y0) or 1) * (H - top - bottom)     # noqa: E731
    s = Svg(W, H, f"parameters vs {meta.get('metric', 'score')} — {protocol}", meta.get("definition", ""),
            fig_id or f"lb-scatter-{protocol}")
    for t in yt:
        s.line(left, Y(t), W - right, Y(t), "grid")
        s.text(left - 8, Y(t), tick_label(t), "t-mut t-num t-sm", "end")
    for dcd in range(int(math.floor(x0)), int(math.ceil(x1)) + 1):
        if x0 <= dcd <= x1:
            s.line(X(dcd), top, X(dcd), H - bottom, "grid")
            s.text(X(dcd), H - bottom + 14, f"{10 ** dcd / 1e6:g}M" if dcd >= 6 else f"{10 ** dcd:g}",
                   "t-mut t-num t-sm", "middle")
    s.text((left + W - right) / 2, H - bottom + 32, "parameters (log scale) — "
           + ("encoder counts as published" if kinds == {"encoder"} else "/".join(sorted(kinds))),
           "t-mut t-sm", "middle")
    s.text(left - 8, top - 14, f"{meta.get('metric', 'score')} {meta.get('unit', '')}", "t-mut t-sm", "start")
    for label, n, v, cls, kind, txt in sorted(pts, key=lambda p: p[1]):
        s.open_g({"data-pub": label})
        s.circle(X(math.log10(n)), Y(v), 6, cls + (" dot" if cls == "m-s1" else ""),
                 data=numattrs(None, v, None), tip=f"{label}: {fmt(v, f)} at {txt} ({kind})")
        s.text(X(math.log10(n)) + 10, Y(v), f"{label} {fmt(v, f)}", "t-2 t-sm", "start",
               data={"data-role": "pub-label"})
        s.close_g()
    note = ""
    if missing:
        names = ", ".join(f"<b>{esc(a)}</b>" for a, _b in missing[:6])
        more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        note = (f'<p class="sub">{len(missing)} row(s) carry no parameter count and are not plotted: {names}{more}. '
                "W4 integration ask: numeric <code>params_total</code> / <code>params_encoder</code> with sources.</p>")
    return s.to_string(), note


def render_leaderboard_charts(model: dict) -> str:
    """The entry point W4 calls. Never raises: an unexpected model shape returns a visible note."""
    try:
        protocols = sorted({r.get("protocol") for r in (model.get("navsim") or []) if r.get("protocol")}
                           | {r.get("protocol") for r in ((model.get("published") or {}).get("results") or [])
                              if r.get("protocol")})
        blocks = []
        for p in protocols:
            svg, twin, notes = rank_bars(model, p)
            if not svg:
                continue
            sc, sc_note = params_scatter(model, p)
            blocks.append(f'<section class="card"><h3>{esc(p)}</h3><figure>{svg}</figure>{notes}'
                          f'<details class="twin"><summary>table view</summary>{twin}</details>'
                          + (f'<figure>{sc}</figure>' if sc else "") + sc_note + "</section>")
        if not blocks:
            return '<p>W5 charts: no protocol in the leaderboard model carries a drawable row.</p>'
        qcss = "".join(f".q{i}{{background:var(--q{i});color:var(--q{i}-ink)}}" for i in range(N_HEAT))
        css = css_tokens() + PAGE_CSS.replace("QCSS", qcss).replace("PRINTLIGHT", "") + CHART_CSS
        return (f'<style>{css}</style><div class="w5-charts"><h2>Charts (W5 components)</h2>'
                f'<p class="lede">One axis per protocol tag — never two protocols on one axis. Our rows are '
                f'solid, published rows hollow, our STOP / CV floors are the reference lines.</p>'
                + "".join(blocks) + "</div>")
    except Exception as e:                                                  # noqa: BLE001
        return (f'<p>W5 charts unavailable: {esc(type(e).__name__)}: {esc(str(e)[:300])} — the leaderboard '
                f'tables below are the complete record.</p>')


_ = CHAR_W
