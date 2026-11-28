"""Assemble one run's report: ``<run_dir>/report/index.html`` + ``report/fig/*.svg`` (+ gallery PNGs).

Every number printed comes out of ``summary.json`` (or, for published rows, out of the published-results
file) through :func:`_n` / the chart builders, which stamp it with its JSON pointer; ``verify.py`` then
re-reads the written HTML and checks every one. Refusals are printed with their reason and n.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path

from . import REPORT_VERSION
from .charts import (dot_chart, fmt, hbar_chart, heat_table, nice_domain, num_span, plain_table, wtl_chart)
from .contract import Run
from .page import page, standalone_svg
from .metrics import SPEED_BANDS, metric_set_for
from .svg import esc

SELF_FLOORS_ORDER = ("STOP", "CV", "ECHO")

# ⛔ BINDING (W2, 2026-09-19/20 — `estimator-and-gates`): the NavSim driving command is a ROUTE-LEVEL
# ORACLE, and an arm fed it may never be presented as evidence of route-following. The caveat is
# rendered BESIDE the headline of every such arm, never in a footnote. A run may carry its own text in
# `arms.<arm>.caveats[]` / `summary.caveats[]`; this one fires from the DECLARED INPUTS, so it also
# covers runs written before the finding landed.
ORACLE_COMMAND_CAVEAT = {
    "id": "navsim.driving_command_is_a_route_oracle",
    "level": "serious",
    "short": "command = route-level ORACLE",
    "text": ("The NavSim `driving_command` is a ROUTE-LEVEL ORACLE, not a driver input: it is a function "
             "of the ego pose now + the nuPlan route + the map (W2 reproduced it 1,902/1,902 with "
             "OpenScene's own function), and that route is the EXPERT'S DRIVEN PATH at roadblock "
             "granularity (79.8 % of forward blocks driven vs a 34.6 % null control). On stage 2 the "
             "command and the route are COPIED from the expert's own frame (5,462/5,462 navhard). ⇒ an "
             "arm that declares the command is NOT evidence of route-following, and its nav-compliance "
             "readout is not a route claim."),
    "source": "W2 (EvalFlyWheel, estimator-and-gates), 2026-09-19/20",
}


def _reads_command(run: Run, arm: str) -> bool:
    return any("driving_command" in str(d) for d in (run.arm(arm).get("declared_inputs") or []))


def _oracle_arms(run: Run) -> list:
    return [a for a in run.arm_names if _reads_command(run, a)]


def _oracle_tag(run: Run, arm: str) -> str:
    if not _reads_command(run, arm):
        return ""
    return (f'<span class="pill oracle" data-caveat="{esc(ORACLE_COMMAND_CAVEAT["id"])}" '
            f'title="{esc(ORACLE_COMMAND_CAVEAT["text"])}">⚠ {esc(ORACLE_COMMAND_CAVEAT["short"])}</span>')


def interval_html(run: Run, arm: str) -> str:
    """An interval is ALWAYS printed: the value with its estimator, or UNAVAILABLE with its reason."""
    iv = run.arm(arm).get("interval")
    if not isinstance(iv, dict):
        return _refusal("no interval block in summary.json for this arm", P("arms", arm, "interval"))
    if iv.get("status") == "OK" and _is_num(iv.get("lo")) and _is_num(iv.get("hi")):
        return (f'[{_n(run, iv["lo"], "f4", "arms", arm, "interval", "lo")}, '
                f'{_n(run, iv["hi"], "f4", "arms", arm, "interval", "hi")}] '
                f'<span class="mut">{esc(iv.get("estimator", "?"))} · clusters {esc(iv.get("cluster_unit", "?"))} '
                f'· n {esc(iv.get("n_clusters", "?"))}</span>')
    return _refusal(iv.get("reason", "no reason recorded"), P("arms", arm, "interval"), iv.get("n"))


# ------------------------------------------------------------------------------------ helpers
def P(*parts) -> str:
    """JSON-pointer-style key (``/`` separated, ``~``/``/`` escaped as in RFC 6901, no leading slash)."""
    return "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _n(run: Run, v, f: str, *path) -> str:
    """A number from summary.json at ``path`` rendered with its verifier stamp."""
    return num_span(P(*path), v, f)


def _na(run: Run, arm: str, v, f: str, *path) -> str:
    """A number in a MULTI-ARM row: wrapped in its own arm scope so the verifier can check ownership."""
    return f'<span data-arm="{esc(arm)}">{num_span(P(*path), v, f)}</span>'


def _arm_label(run: Run, arm: str) -> str:
    """The ONE place an arm's display label is resolved (the label-swap mutation test patches this)."""
    return run.label(arm)


def _pill(status: str) -> str:
    cls = {"OK": "ok", "PARTIAL": "partial"}.get(status, "unavailable")
    ic = {"OK": "✓", "PARTIAL": "◐"}.get(status, "–")
    return f'<span class="pill {cls}">{ic} {esc(status)}</span>'


def _short(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _refusal(reason: str, k: str | None, n=None, cls: str = "mut") -> str:
    nn = f" (n {n})" if n is not None else ""
    return (f'<span class="{cls}" data-refusal-k="{esc(k or "")}" title="{esc(reason)}">'
            f"UNAVAILABLE{nn} — {esc(_short(reason))}</span>")


def _grouped_refusals(run: Run, items: list, prefix: str = "") -> str:
    """items: [(arm, reason, n, pointer)] -> one line per DISTINCT reason, one chip per arm (each chip keeps
    its own refusal pointer, so the verifier checks every arm individually)."""
    groups: dict = {}
    for arm, why, n, k in items:
        groups.setdefault((why, n), []).append((arm, k))
    out = []
    for (why, n), arms in groups.items():
        chips = ", ".join(f'<span class="chip" data-arm="{esc(a)}" data-refusal-k="{esc(k)}">'
                          f'<b data-role="arm-label">{esc(_arm_label(run, a))}</b></span>' for a, k in arms)
        nn = f" (n {n})" if n is not None else ""
        out.append(f'<p class="sub">{prefix}{chips} — <span class="mut">UNAVAILABLE{nn}: </span>'
                   f'<span class="ink2" title="{esc(why)}">{esc(_short(why, 420))}</span></p>')
    return "".join(out)


def _floors_present(run: Run) -> list:
    fl = [f for f in SELF_FLOORS_ORDER if f in run.floors] + [f for f in run.floors if f not in SELF_FLOORS_ORDER]
    return [f for f in fl if f in run.arm_names]


def _non_floor_arms(run: Run) -> list:
    return [a for a in run.arm_names if a not in run.floors]


def _published_rows(published: dict | None, protocol: str) -> tuple[list, dict]:
    """The published rows of THIS protocol only (a second protocol never reaches an axis)."""
    if not published:
        return [], {}
    meta = (published.get("protocols") or {}).get(protocol, {})
    rows = [r for r in published.get("results", []) if r.get("protocol") == protocol and _is_num(r.get("value"))]
    return rows, meta


# ------------------------------------------------------------------------------------ sections
def stamp_html(run: Run) -> str:
    st = run.stamps
    loop = st.get("loop") or {}
    loop_s = " · ".join(f"{k.replace('_', ' ')} {v.split(' (')[0]}" for k, v in loop.items()) or "MISSING"
    dk = run.devkit
    sha = (dk.get("sha") or "UNAVAILABLE")[:10]
    npatch = len(dk.get("patches") or [])
    fix = ' · <b style="color:var(--st-serious)">FIXTURE (not a bench run)</b>' if run.is_fixture else ""
    return (f"<span>protocol <b>{esc(run.protocol)}</b></span>"
            f"<span>devkit <b>{esc(dk.get('repo', '?'))}@{esc(sha)}</b>{f' +{npatch} patches' if npatch else ''}</span>"
            f"<span>tier <b>{esc(st.get('tier', 'MISSING'))}</b></span>"
            f"<span>loop <b>{esc(loop_s)}</b>{' · closed loop NO' if st.get('closed_loop') is False else ''}</span>"
            f"<span>evidence <b>{esc(run.evidence_class)}</b></span>"
            f"<span>run <b>{esc(run.run_id)}</b></span>{fix}")


def banners_html(run: Run, extra: list) -> str:
    out = []
    order = {"critical": 0, "serious": 1, "warning": 2, "good": 3}
    items = [(i.level, i.what) for i in run.issues] + list(extra)
    for lvl, what in sorted(items, key=lambda t: order.get(t[0], 9)):
        ic = {"critical": "⛔", "serious": "▲", "warning": "!", "good": "✓"}.get(lvl, "•")
        word = {"critical": "BLOCKING", "serious": "SERIOUS", "warning": "NOTE", "good": "OK"}.get(lvl, lvl)
        out.append(f'<div class="banner {lvl}"><span class="ic">{ic}</span><b>{word}</b> — {esc(what)}</div>')
    return "".join(out)


def overview_html(run: Run) -> str:
    pa = run.primary_arm
    a = run.arm(pa)
    bench = run.bench
    ck = bench.get("ckpt") or {}
    split = bench.get("split") or {}
    tiles = []
    s2 = g(a, "statistics", "S2_EPDMS_u")
    if s2 and _is_num(s2.get("value")):
        deltas = []
        for f in ("STOP", "CV"):
            d = g(a, "paired", f, "statistic_deltas", "S2_EPDMS_u")
            if _is_num(d):
                deltas.append(f"Δ vs {f} {_n(run, d, 'sgn4', 'arms', pa, 'paired', f, 'statistic_deltas', 'S2_EPDMS_u')}")
        tiles.append(f'<div class="kpi" data-arm="{esc(pa)}"><div class="l">S2-EPDMS-u · '
                     f'<span data-role="arm-label">{esc(_arm_label(run, pa))}</span></div>'
                     f'<div class="v">{_n(run, s2["value"], "f4", "arms", pa, "statistics", "S2_EPDMS_u", "value")}</div>'
                     f'<div class="d">{" · ".join(deltas)}</div></div>')
    h = a.get("headline") or {}
    if _is_num(h.get("value")):
        tiles.append(f'<div class="kpi" data-arm="{esc(pa)}"><div class="l">{esc(run.summary["headline_metric"]["name"])} '
                     f'(official) · <span data-role="arm-label">{esc(_arm_label(run, pa))}</span></div>'
                     f'<div class="v">{_n(run, h["value"], "f4", "arms", pa, "headline", "value")}</div>'
                     f'<div class="d">n {esc(h.get("n"))}</div></div>')
    else:
        tiles.append(f'<div class="kpi" data-arm="{esc(pa)}"><div class="l">{esc(run.summary.get("headline_metric", {}).get("name", "headline"))} '
                     f'(official) · <span data-role="arm-label">{esc(_arm_label(run, pa))}</span></div>'
                     f'<div class="v mut" style="font-size:20px">UNAVAILABLE</div>'
                     f'<div class="d">{_refusal(h.get("reason", "no headline"), P("arms", pa, "headline"), h.get("n"))}</div></div>')
    for f in ("STOP", "CV"):
        if f not in run.arm_names or f == pa:
            continue
        # the MOST SPECIFIC scope the run carries: per stage where the stages differ, else pooled
        stages = run.paired_stages(pa, f)
        for stage in (stages or [None]):
            c = run.paired_counts(pa, f, stage)
            if not c:
                continue
            n_txt = (num_span(c["n"][1], c["n"][0], "int") if c["n"][0] is not None else "?")
            tiles.append(f'<div class="kpi" data-arm="{esc(pa)}"><div class="l">paired W / T / L vs {esc(f)} · '
                         f'<span data-role="arm-label">{esc(_arm_label(run, pa))}</span></div>'
                         f'<div class="v">{num_span(c["wins"][1], c["wins"][0], "int")} / '
                         f'{num_span(c["ties"][1], c["ties"][0], "int")} / '
                         f'{num_span(c["losses"][1], c["losses"][0], "int")}</div>'
                         f'<div class="d">{esc(c["scope"])} · n {n_txt}</div></div>')
    if _reads_command(run, pa):
        tiles.append(f'<div class="kpi caveat" data-arm="{esc(pa)}"><div class="l">inputs · '
                     f'<span data-role="arm-label">{esc(_arm_label(run, pa))}</span></div>'
                     f'<div class="v" style="font-size:15px;line-height:1.35">⚠ {esc(ORACLE_COMMAND_CAVEAT["short"])}</div>'
                     f'<div class="d">{esc(_short(ORACLE_COMMAND_CAVEAT["text"], 210))}</div></div>')
    facts = [("benchmark / split", f"{run.benchmark} / {split.get('name', run.summary.get('split'))}"),
             ("scenes · logs", f"{split.get('n_scenes', '?')} · {split.get('n_logs', '?')}"),
             ("checkpoint", f"{ck.get('registry_key') or '—'}"),
             ("device", f"{g(bench, 'device', 'used', default='?')}"),
             ("interval", g(run.summary, "estimator", "interval", "status", default="?")),
             ("claim-bearing", str(run.summary.get("claim_bearing")))]
    facts_html = "".join(f"<span><b>{esc(k)}</b> {esc(v)}</span>" for k, v in facts)
    return (f'<h1>{esc(run.benchmark)} · {esc(run.summary.get("split", ""))} — {esc(_arm_label(run, pa))}</h1>'
            f'<p class="sub legend">{facts_html}</p><div class="kpis">{"".join(tiles)}</div>')


def headline_html(run: Run, published: dict | None, figs: dict) -> str:
    hm = run.summary.get("headline_metric") or {}
    pub_rows, pmeta = _published_rows(published, run.protocol)
    unit_x100 = pmeta.get("unit") == "x100"
    out = [f'<h2 id="headline">1 · Headline vs the mandatory floors</h2>'
           f'<p class="lede">Each axis carries ONE statistic of ONE protocol. Floors (STOP, CV{", ECHO" if "ECHO" in run.floors else ""}) '
           f'are the labelled reference lines; our arms are bars (the primary arm in the accent colour); published rows are '
           f'hollow bars and appear ONLY on the axis of their own protocol tag.</p>']
    # ---- (a) the protocol's official statistic, with the published rows of the same protocol
    rows, refused = [], []
    vals = []
    for arm in _non_floor_arms(run):
        a = run.arm(arm)
        h = a.get("headline") or {}
        if _is_num(h.get("value")):
            v, k = (h["x100"], P("arms", arm, "headline", "x100")) if unit_x100 and _is_num(h.get("x100")) else \
                   (h["value"], P("arms", arm, "headline", "value"))
            rows.append({"arm": arm, "label": _arm_label(run, arm), "value": v, "k": k, "fmt": "f4",
                         "cls": "m-s1" if arm == run.primary_arm else "m-deemph",
                         "tag": "⚠ oracle cmd" if _reads_command(run, arm) else None,
                         "tip": (f"{_arm_label(run, arm)}: {fmt(v, 'f4')} (n {h.get('n')})"
                                 + (f"\n⚠ {ORACLE_COMMAND_CAVEAT['text']}" if _reads_command(run, arm) else ""))})
            vals.append(v)
        else:                                   # never drawn on the axis; listed under the chart instead
            refused.append((arm, h.get("reason", "no headline value"), h.get("n"), P("arms", arm, "headline")))
    for r in pub_rows:
        v = float(r["value"])
        rows.append({"pub": r["id"], "label": r.get("system", r["id"]), "value": v, "k": None, "fmt": "f4",
                     "cls": "m-pub", "tag": f"published · {r.get('evidence_class', published.get('evidence_class_default', '?'))}",
                     "tip": f"{r.get('system')}: {r['value']} — {g(r, 'source', 'table', default='')}"})
        vals.append(v)
    refs = []
    for f in _floors_present(run):
        h = run.arm(f).get("headline") or {}
        if _is_num(h.get("value")):
            v, k = (h["x100"], P("arms", f, "headline", "x100")) if unit_x100 and _is_num(h.get("x100")) else \
                   (h["value"], P("arms", f, "headline", "value"))
            refs.append({"floor": f, "label": _arm_label(run, f).split(" · ")[0], "value": v, "k": k, "fmt": "f4"})
            vals.append(v)
    # the unit comes from the protocol's own metadata; NEVER call a metric a "fraction" on the guess
    # (MEASURED 2026-09-20: the internal_t1 run's ADE — metres, lower-is-better — read "ADE (fraction)")
    name = hm.get("name", "headline")
    if unit_x100:
        unit_txt = f"{name} ×100"
    elif pmeta.get("unit"):
        unit_txt = f"{name} {pmeta['unit']}"
    else:
        unit_txt = f"{name} (column `{hm.get('column', 'value')}`)"
    if not run.higher_is_better:
        unit_txt += " · LOWER is better"
    unit = f"{unit_txt} — {run.protocol}"
    svg = hbar_chart("fig-headline-official", f"{hm.get('name')} (official) — {run.protocol}", rows,
                     domain=nice_domain(vals, lo=0.0), unit=unit, refs=tuple(refs),
                     desc=hm.get("statistic", ""))
    figs["headline_official"] = svg
    floor_ref_refusals = [(f, g(run.arm(f), "headline", "reason", default="no headline value"),
                           g(run.arm(f), "headline", "n"), P("arms", f, "headline"))
                          for f in _floors_present(run) if not _is_num(g(run.arm(f), "headline", "value"))]
    off_axis = (_grouped_refusals(run, refused, "Not on this axis: ") if refused else "") + \
        (_grouped_refusals(run, floor_ref_refusals, "Floor without a reference line: ") if floor_ref_refusals else "")
    note = ""
    pub_note = (f"published rows: {len(pub_rows)} for <code>{esc(run.protocol)}</code> from "
                f"<code>{esc((published or {}).get('_path', 'no published file'))}</code>"
                + (" (other protocols in that file are never drawn here)" if published else ""))
    # the table twin carries BOTH scales: the protocol's published unit (x100) and the fraction the
    # registry and our own artifacts speak in — so neither has to be converted by the reader
    def _both(arm: str) -> list:
        h = run.arm(arm).get("headline") or {}
        cells = [num_span(P("arms", arm, "headline", "x100"), h["x100"], "f4")] if unit_x100 and _is_num(h.get("x100")) else []
        cells.append(num_span(P("arms", arm, "headline", "value"), h["value"], "f4"))
        cells.append(interval_html(run, arm))           # never an empty cell (W2)
        cells.append(_oracle_tag(run, arm) or "—")
        return cells

    twin_rows = []
    for r in rows:
        if r.get("arm"):
            twin_rows.append({"arm": r["arm"], "label": r["label"],
                              "cells": _both(r["arm"]) + [esc(run.arm(r["arm"]).get("kind"))]})
    for arm, why, n, k in refused:
        twin_rows.append({"arm": arm, "label": _arm_label(run, arm),
                          "cells": [_refusal(why, k, n)] + ([""] if unit_x100 else [])
                                   + [interval_html(run, arm), _oracle_tag(run, arm) or "—",
                                      esc(run.arm(arm).get("kind"))]})
    for rf in refs:
        twin_rows.append({"arm": rf["floor"], "label": _arm_label(run, rf["floor"]),
                          "cells": _both(rf["floor"]) + ["floor"]})
    for r in pub_rows:
        twin_rows.append({"arm": None, "label": f"{r.get('system')} [published · {r['id']}]",
                          "cells": [f'<span class="num" data-pub="{esc(r["id"])}" data-pub-v="{esc(repr(float(r["value"])))}">'
                                    f'{esc(fmt(r["value"], "f4"))}</span>'] + ([""] if unit_x100 else [])
                                   + ["—", esc(r.get("ego_status", "")),
                                      esc(r.get("evidence_class", published.get("evidence_class_default")))]})
    oracle = _oracle_arms(run)
    caveat_html = ""
    if oracle:
        chips = ", ".join(f'<span class="chip" data-arm="{esc(a)}"><b data-role="arm-label">'
                          f'{esc(_arm_label(run, a))}</b></span>' for a in oracle)
        caveat_html = (f'<div class="banner serious" data-caveat="{esc(ORACLE_COMMAND_CAVEAT["id"])}">'
                       f'<span class="ic">▲</span><b>{esc(ORACLE_COMMAND_CAVEAT["short"].upper())}</b> — '
                       f'{chips} declare the NavSim command. {esc(ORACLE_COMMAND_CAVEAT["text"])} '
                       f'<span class="mut">({esc(ORACLE_COMMAND_CAVEAT["source"])})</span></div>')
    # one line per DISTINCT interval verdict (a run where no arm has one repeats a single reason)
    unav = [(a, (run.arm(a).get("interval") or {}).get("reason", "no interval block in summary.json for this arm"),
             (run.arm(a).get("interval") or {}).get("n"), P("arms", a, "interval"))
            for a in run.arm_names
            if not ((run.arm(a).get("interval") or {}).get("status") == "OK")]
    have = [a for a in run.arm_names if (run.arm(a).get("interval") or {}).get("status") == "OK"]
    iv_lines = "".join(f'<li><span data-arm="{esc(a)}"><b data-role="arm-label">{esc(_arm_label(run, a))}</b></span>: '
                       f'{interval_html(run, a)}</li>' for a in have)
    interval_block = (f'<p class="sub">Interval per arm (never an empty cell — the settled estimator is the '
                      f'LOG-CLUSTER bootstrap, clusters = <code>log_name</code>, B = 2000, n ≥ 8 clusters):</p>'
                      + (f'<ul class="sub cols">{iv_lines}</ul>' if iv_lines else "")
                      + (_grouped_refusals(run, unav, "No interval: ") if unav else ""))
    out.append(f'<div class="card"><h3>{esc(hm.get("name", "headline"))} — the protocol\'s official statistic</h3>'
               f'<figure>{svg}<figcaption>{esc(hm.get("statistic", ""))}. {pub_note}. {note}</figcaption></figure>'
               f'{caveat_html}{off_axis}{interval_block}'
               f'<details class="twin"><summary>table view</summary>'
               f'{plain_table("tab-headline-official", unit, ["arm"] + (["×100", "fraction"] if unit_x100 else ["value"]) + ["interval", "inputs", "kind"], twin_rows)}</details></div>')
    # ---- (b) S2-EPDMS-u (every arm has it on a stage-2-only run)
    if any(_is_num(g(run.arm(a), "statistics", "S2_EPDMS_u", "value")) for a in run.arm_names):
        sdef = g(run.summary, "statistics", "S2_EPDMS_u", "definition", default="")
        rows2, vals2 = [], []
        for arm in _non_floor_arms(run):
            s = g(run.arm(arm), "statistics", "S2_EPDMS_u") or {}
            if _is_num(s.get("value")):
                rows2.append({"arm": arm, "label": _arm_label(run, arm), "value": s["value"],
                              "k": P("arms", arm, "statistics", "S2_EPDMS_u", "value"), "fmt": "f4",
                              "cls": "m-s1" if arm == run.primary_arm else "m-deemph",
                              "tag": "⚠ oracle cmd" if _reads_command(run, arm) else None,
                              "tip": (f"{_arm_label(run, arm)}: {fmt(s['value'], 'f4')} ({s.get('n_scenes')} scenes, "
                                      f"{s.get('n_groups')} groups)"
                                      + (f"\n⚠ {ORACLE_COMMAND_CAVEAT['text']}" if _reads_command(run, arm) else ""))})
                vals2.append(s["value"])
            else:
                has_blk = isinstance(g(run.arm(arm), "statistics", "S2_EPDMS_u"), dict)
                why = s.get("reason") or (g(run.arm(arm), "headline", "reason") if not has_blk else None) \
                    or "S2-EPDMS-u absent for this arm"
                rows2.append({"arm": arm, "label": _arm_label(run, arm), "value": None, "refusal": why,
                              "refusal_k": (P("arms", arm, "statistics", "S2_EPDMS_u") if has_blk
                                            else P("arms", arm, "headline"))})
        refs2 = []
        for f in _floors_present(run):
            s = g(run.arm(f), "statistics", "S2_EPDMS_u") or {}
            if _is_num(s.get("value")):
                refs2.append({"floor": f, "label": _arm_label(run, f).split(" · ")[0], "value": s["value"],
                              "k": P("arms", f, "statistics", "S2_EPDMS_u", "value"), "fmt": "f4"})
                vals2.append(s["value"])
        svg2 = hbar_chart("fig-headline-s2u", "S2-EPDMS-u (stage-2 statistic, not a two-stage EPDMS)", rows2,
                          domain=nice_domain(vals2, lo=0.0), unit="S2-EPDMS-u (fraction) — stage-2 tokens only",
                          refs=tuple(refs2), desc=sdef)
        figs["headline_s2u"] = svg2
        twin2 = [{"arm": r["arm"], "label": r["label"],
                  "cells": [num_span(r["k"], r["value"], "f4") if r.get("value") is not None else _refusal(r["refusal"], r["refusal_k"])]}
                 for r in rows2] + [{"arm": rf["floor"], "label": _arm_label(run, rf["floor"]), "cells": [num_span(rf["k"], rf["value"], "f4")]}
                                    for rf in refs2]
        out.append(f'<div class="card"><h3>S2-EPDMS-u — the statistic every arm has on this run</h3>'
                   f'<figure>{svg2}<figcaption>{esc(sdef)} No published result exists for this statistic, so none is drawn.'
                   f'</figcaption></figure><details class="twin"><summary>table view</summary>'
                   f'{plain_table("tab-headline-s2u", "S2-EPDMS-u", ["arm", "S2-EPDMS-u"], twin2)}</details></div>')
    return "".join(out)


def _wtl_rows(run: Run, floor: str, stage):
    """Chart rows for one scope; an arm without counts at that scope becomes a named refusal."""
    rows = []
    for arm in run.arm_names:
        if arm == floor:
            continue
        p = run.paired(arm, floor)
        c = run.paired_counts(arm, floor, stage)
        if c:
            rows.append({"arm": arm, "label": _arm_label(run, arm), "wins": c["wins"][0], "ties": c["ties"][0],
                         "losses": c["losses"][0], "n": c["n"][0] or (c["wins"][0] + c["ties"][0] + c["losses"][0]),
                         "k": c["wins"][1].rsplit("/", 1)[0], "n_k": c["n"][1], "scope": c["scope"]})
        elif p.get("status") == "OK":
            rows.append({"arm": arm, "label": _arm_label(run, arm),
                         "refusal": f"no win/tie/loss counts at this scope in summary.json",
                         "refusal_k": P("arms", arm, "paired", floor)})
        else:
            rows.append({"arm": arm, "label": _arm_label(run, arm),
                         "refusal": f"UNAVAILABLE: {p.get('reason') or p.get('status') or 'not paired'}",
                         "refusal_k": P("arms", arm, "paired", floor)})
    return rows


def paired_html(run: Run, figs: dict) -> str:
    direction = ("" if run.higher_is_better else
                 " ⚠ On this protocol LOWER is better, so a NEGATIVE Δ is an improvement and a 'win' is the "
                 "arm scoring lower than the floor. ")
    out = ['<h2 id="paired">2 · Paired per-scene win / tie / loss vs each floor</h2>'
           '<p class="lede">Per scene: arm − floor on the identical tokens. A win on more scenes can still lose on the '
           'mean when its losses are multiplicative zeros — read the count beside the Δ, never alone. ' + direction +
           '⛔ On a two-stage protocol the stages have DIFFERENT token sets, so a per-stage count and a pooled count '
           'are different quantities: every chart below states the SCOPE it was counted over.</p>']
    for f in _floors_present(run):
        # which scopes does this run actually carry? per stage first (that is where the meaning is)
        stages = sorted({s for arm in run.arm_names for s in run.paired_stages(arm, f)},
                        key=lambda s: {"stage_two": 0, "stage_one": 1}.get(s, 2))
        scopes = [(s, f"stage {s.split('_')[-1]}") for s in stages]
        if any(run.paired_counts(arm, f, None) for arm in run.arm_names if arm != f):
            scopes.append((None, "pooled"))
        cards = []
        for stage, tag in scopes:
            rows = _wtl_rows(run, f, stage)
            if not any("wins" in r for r in rows):
                continue
            scope_txt = next((r["scope"] for r in rows if r.get("scope")), "scope not declared")
            fid = f"fig-wtl-{f}-{tag.replace(' ', '_')}"
            svg = wtl_chart(fid, f"Paired W/T/L vs {f} — {tag}", rows, floor_label=f)
            figs[f"wtl_{f}_{tag.replace(' ', '_')}"] = svg
            n_hint = next((r.get("n") for r in rows if r.get("n")), "?")
            cards.append(f'<figure>{svg}<figcaption><b>Scope: {esc(scope_txt)}</b> · n {esc(n_hint)} tokens. '
                         f'Tie = |Δ| ≤ the run\'s tie band.</figcaption></figure>')
        if not cards:
            # no countable scope for this floor — the REFUSALS still have to be printed (the verifier
            # caught exactly this on W1's FAILED run: a floor silently vanished from the page)
            unav = [(arm, (run.paired(arm, f).get("reason") or str(run.paired(arm, f).get("status") or "not paired")),
                     run.paired(arm, f).get("n"), P("arms", arm, "paired", f))
                    for arm in run.arm_names if arm != f]
            out.append(f'<div class="card" data-floor="{esc(f)}"><h3>vs {esc(_arm_label(run, f))}</h3>'
                       f'<p class="sub">No win/tie/loss counts in this run at any scope.</p>'
                       + _grouped_refusals(run, unav, "Paired: ") + "</div>")
            continue
        # ---- table twin: the deltas, per stage where the run holds them
        hdr = ["arm", "Δ official", "Δ S2-EPDMS-u"]
        for s in stages:
            hdr += [f"Δ mean ({s.split('_')[-1]})", f"W/T/L ({s.split('_')[-1]})"]
        if (None, "pooled") in scopes:
            hdr += ["Δ mean (pooled)", "W/T/L (pooled)"]
        twin = []
        for arm in run.arm_names:
            if arm == f:
                continue
            p = run.paired(arm, f)
            if p.get("status") != "OK":
                twin.append({"arm": arm, "label": _arm_label(run, arm),
                             "cells": [_refusal(p.get("reason") or str(p.get("status")), P("arms", arm, "paired", f),
                                                p.get("n"))] + [""] * (len(hdr) - 2)})
                continue
            hd = p.get("headline_delta")
            cells = [_n(run, hd, "sgn4", "arms", arm, "paired", f, "headline_delta") if _is_num(hd) else
                     f'<span class="mut" title="{esc(p.get("headline_delta_reason", ""))}">n/a</span>']
            sd, sk = run.paired_stat_delta(arm, f)
            cells.append(num_span(sk, sd, "sgn4") if sd is not None else "—")
            for stage, _tag in scopes:
                md, mk = run.paired_mean_delta(arm, f, stage)
                cells.append(num_span(mk, md, "sgn4") if md is not None else "—")
                c = run.paired_counts(arm, f, stage)
                cells.append(" / ".join(num_span(c[x][1], c[x][0], "int") for x in ("wins", "ties", "losses"))
                             if c else "—")
            twin.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
        out.append(f'<div class="card" data-floor="{esc(f)}"><h3>vs {esc(_arm_label(run, f))}</h3>'
                   + "".join(cards)
                   + f'<details class="twin"><summary>table view (Δ per scope)</summary>'
                   + plain_table(f"tab-wtl-{f}", f"arm − {f}: every Δ and count the run holds, per scope", hdr, twin)
                   + "</details></div>")
    return "".join(out)


def submetrics_html(run: Run) -> str:
    mset = metric_set_for(run.protocol)
    if mset is None:
        return ('<h2 id="submetrics">3 · Sub-metrics per stage</h2><p class="lede">No NavSim metric set for protocol '
                f'<code>{esc(run.protocol)}</code> — this section applies to EPDMS / PDMS runs only.</p>')
    out = ['<h2 id="submetrics">3 · Sub-metrics per stage, and the multiplicative zeros</h2>'
           f'<p class="lede">{esc(mset.name)}: multipliers {", ".join(mset.multipliers)} (a zero anywhere zeroes the scene) × '
           f'weighted {", ".join(f"{k}·{w}" for k, w in mset.weighted.items())} / {mset.denominator:g}. Uniform scene means '
           f'of the devkit per-token rows; a stage an arm did not answer itself is refused with its reason.</p>']
    stages = run.stages()
    # one polarity per table: every column here is "higher is better" (the failure rates get their own table)
    cols = [("scene_mean", "score", "uniform scene mean of `score`")] + \
           [(k, k, mset.long_names.get(k, k)) for k in mset.submetrics]
    for st in stages:
        rows, refused = [], []
        for arm in run.arm_names:
            b = g(run.arm(arm), "per_stage", st)
            score, score_k = run.stage_score(arm, st)
            subs = {k: run.stage_submetric(arm, st, k) for k in mset.submetrics}
            if score is not None or any(v is not None for v, _k in subs.values()):
                cells = {"scene_mean": (score, score_k)}
                cells.update(subs)
                rows.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
            else:
                if isinstance(b, dict):
                    why, n, kk = b.get("reason"), b.get("n"), P("arms", arm, "per_stage", st)
                elif isinstance(g(run.arm(arm), "per_stage"), dict):
                    ps = g(run.arm(arm), "per_stage")
                    why, n, kk = ps.get("reason"), ps.get("n"), P("arms", arm, "per_stage")
                else:
                    why, n, kk = None, None, P("arms", arm)
                refused.append((arm, why or "no per-stage block for this arm", n, kk))
        n_hint = next((run.stage_n(a, st) for a in run.arm_names if _is_num(run.stage_n(a, st))), None)
        tab = heat_table(f"tab-sub-{st}", f"{st.replace('_', ' ')} · n {n_hint} scenes · uniform means "
                         f"(darker = higher = better)", cols, rows)
        ref = _grouped_refusals(run, refused, "Refused on this stage: ") if refused else ""
        out.append(f'<div class="card"><h3>{esc(st.replace("_", " "))}</h3>{tab}{ref}</div>')
    # multiplicative-zero rates (stage two, and stage one where the arm answered it)
    zcols = [(f"z_{k}", f"{k}=0", f"fraction of scenes with {k} exactly 0") for k in mset.multipliers] + \
            [(f"p_{k}", f"{k}=½", f"fraction with {k} in (0, 1) — the partial-credit band") for k in ("NC", "DDC")
             if k in mset.multipliers] + [("zero_score_rate", "score=0", "any multiplier zero")]
    for st in stages:
        rows = []
        for arm in run.arm_names:
            cells = {f"z_{k}": run.stage_rate(arm, st, "zero_rates", k) for k in mset.multipliers}
            for k in ("NC", "DDC"):
                if k in mset.multipliers:
                    cells[f"p_{k}"] = run.stage_rate(arm, st, "partial_rates", k)
            zs = g(run.arm(arm), "per_stage", st, "zero_score_rate")
            cells["zero_score_rate"] = ((zs, P("arms", arm, "per_stage", st, "zero_score_rate"))
                                        if _is_num(zs) else (None, None))
            if not any(v is not None for v, _k in cells.values()):
                continue
            rows.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
        if rows:
            out.append(f'<div class="card"><h3>multiplicative-zero rates · {esc(st.replace("_", " "))}</h3>'
                       + heat_table(f"tab-zero-{st}", "fraction of scenes (darker = more)", zcols, rows, f="pct1") + "</div>")
    # the devkit's own summary rows (kernel-weighted stage 2) — admissible or HYBRID
    trows = []
    for arm in run.arm_names:
        orow = g(run.arm(arm), "statistics", "official_rows")
        if not isinstance(orow, dict):
            continue
        adm = g(run.arm(arm), "statistics", "EPDMS_two_stage_combined", "admissible")
        cells = []
        if adm:
            for key in ("stage_one", "stage_two", "combined"):
                v = g(orow, key, "score")
                cells.append(_n(run, v, "f4", "arms", arm, "statistics", "official_rows", key, "score") if _is_num(v) else "—")
            cells.append("admissible")
        else:
            # ⛔ a HYBRID row is never printed as a number: its stage 1 is another agent's and its stage-2
            # kernel weights come from that agent's endpoints (E2 RESULT §2)
            why = g(run.arm(arm), "statistics", "EPDMS_two_stage_combined", "reason", default="HYBRID")
            cells = ['<span class="mut">CV stand-in</span>', '<span class="mut">HYBRID</span>', '<span class="mut">HYBRID</span>',
                     f'<span class="mut" data-refusal-k="{esc(P("arms", arm, "headline"))}" title="{esc(why)}">'
                     f'not this arm\'s — not reported</span>']
        trows.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
    if trows:
        out.append('<div class="card"><h3>the devkit\'s own summary rows</h3>'
                   + plain_table("tab-official-rows", "extended_pdm_score_{stage_one, stage_two, combined} — stage 2 is "
                                 "kernel-weighted by the stage-1 endpoints, so an arm with a stand-in stage 1 is HYBRID",
                                 ["arm", "stage one", "stage two", "combined", "status"], trows) + "</div>")
    return "".join(out)


def _floor_series(run: Run) -> list:
    """<= 3 series for the dot plots: the primary arm, CV, STOP (the validated all-pairs cap)."""
    ser = [(run.primary_arm, "m-s1")]
    for f, cls in (("CV", "m-s2"), ("STOP", "m-s3")):
        if f in run.arm_names and f != run.primary_arm:
            ser.append((f, cls))
    return ser


def per_log_html(run: Run, figs: dict) -> str:
    out = ['<h2 id="per-log">4 · Per log</h2>']
    series = _floor_series(run)
    logs = run.logs()
    if not logs:
        return out[0] + '<p class="lede mut">No per-log block in this run.</p>'
    pa = run.primary_arm
    cats = [{"id": lg, "label": lg, "n": run.per_log_n(pa, lg) or "?"} for lg in logs]
    ser = []
    vals = []
    for arm, cls in series:
        pts = {}
        for lg in logs:
            v, k = run.per_log(arm, lg)
            if v is not None:
                pts[lg] = (v, k)
                vals.append(v)
        ser.append({"arm": arm, "label": _arm_label(run, arm), "cls": cls, "points": pts})
    svg = dot_chart("fig-per-log", "Per-log scene mean: primary arm vs CV vs STOP", cats, ser,
                    domain=nice_domain(vals, lo=0.0, hi=1.0), unit="uniform scene mean of `score` (stage 2)")
    figs["per_log"] = svg
    cols = [(lg, lg.split("_")[0][5:] + "·" + lg.split("_")[-2], lg) for lg in logs]
    rows = []
    for arm in run.arm_names:
        cells = {lg: run.per_log(arm, lg) for lg in logs}
        if any(v is not None for v, _k in cells.values()):
            rows.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
    drows = []
    for f in ("CV", "STOP"):
        bl = g(run.arm(pa), "paired", f, "by_log") or {}
        if not bl or f == pa:
            continue
        cells = []
        for lg in logs:
            b = bl.get(lg) or {}
            if _is_num(b.get("scene_mean_delta")):
                k = ("arms", pa, "paired", f, "by_log", lg)
                cells.append(f'{_na(run, pa, b["scene_mean_delta"], "sgn3", *k, "scene_mean_delta")}<br><span class="mut">'
                             f'{_na(run, pa, b["wins"], "int", *k, "wins")}/{_na(run, pa, b["ties"], "int", *k, "ties")}/'
                             f'{_na(run, pa, b["losses"], "int", *k, "losses")}</span>')
            else:
                cells.append("—")
        drows.append({"arm": None, "label": f"{_arm_label(run, pa)} − {f}", "cells": cells})
    out.append(f'<div class="card"><figure>{svg}<figcaption>Dots: {", ".join(esc(_arm_label(run, a)) for a, _ in series)}. '
               f'n = the primary arm\'s scenes in that log. No interval: this split has fewer log groups than the RG-14 floor.'
               f'</figcaption></figure><details class="twin"><summary>table view (all arms × logs)</summary>'
               + heat_table("tab-per-log", "uniform scene mean per log", cols, rows)
               + (plain_table("tab-per-log-delta", "paired Δ scene mean per log (W/T/L below)", ["pair"] + [c[1] for c in cols], drows) if drows else "")
               + "</details></div>")
    return "".join(out)


def per_speed_html(run: Run, figs: dict) -> str:
    out = ['<h2 id="per-speed">5 · Per t0-speed band</h2>']
    pa = run.primary_arm
    bands = g(run.arm(pa), "per_speed_band", "bands") or {}
    if not bands:
        return out[0] + '<p class="lede mut">No per-speed-band block in this run.</p>'
    order = [b[0] for b in SPEED_BANDS if b[0] in bands] + [b for b in bands if b not in {x[0] for x in SPEED_BANDS}]
    cats = [{"id": b, "label": bands[b].get("label", b), "n": bands[b].get("n")} for b in order]
    ser, vals = [], []
    for arm, cls in _floor_series(run):
        pts = {}
        for b in order:
            v = g(run.arm(arm), "per_speed_band", "bands", b, "value")
            if _is_num(v):
                pts[b] = (v, P("arms", arm, "per_speed_band", "bands", b, "value"))
                vals.append(v)
        ser.append({"arm": arm, "label": _arm_label(run, arm), "cls": cls, "points": pts})
    svg = dot_chart("fig-per-speed", "Per t0-speed band: primary arm vs CV vs STOP", cats, ser,
                    domain=nice_domain(vals, lo=0.0, hi=1.0), unit="uniform scene mean of `score` (stage 2)")
    figs["per_speed"] = svg
    rows = []
    for b in order:
        bb = bands[b]
        k0 = ("arms", pa, "per_speed_band", "bands", b)
        cells = [_na(run, pa, bb["n"], "int", *k0, "n")]
        for arm, _c in _floor_series(run):
            v = g(run.arm(arm), "per_speed_band", "bands", b, "value")
            cells.append(_na(run, arm, v, "f3", "arms", arm, "per_speed_band", "bands", b, "value") if _is_num(v) else "—")
        for f in ("CV", "STOP"):
            p = g(run.arm(pa), "paired", f, "by_speed_band", b) or {}
            if _is_num(p.get("scene_mean_delta")):
                k = ("arms", pa, "paired", f, "by_speed_band", b)
                cells.append(f'{_na(run, pa, p["scene_mean_delta"], "sgn3", *k, "scene_mean_delta")} '
                             f'<span class="mut">({_na(run, pa, p["wins"], "int", *k, "wins")}/{_na(run, pa, p["ties"], "int", *k, "ties")}/'
                             f'{_na(run, pa, p["losses"], "int", *k, "losses")})</span>')
            else:
                cells.append("—")
        rows.append({"arm": None, "label": bb.get("label", b), "cells": cells})
    hdr = ["band", "n"] + [_arm_label(run, a) for a, _ in _floor_series(run)] + \
          [f"Δ vs CV (W/T/L)", f"Δ vs STOP (W/T/L)"]
    src = g(run.arm(pa), "per_speed_band", "v0_source", default="")
    out.append(f'<div class="card"><figure>{svg}<figcaption>Bands of |v0| at t0 ({esc(src)}). '
               f'Δ are the primary arm <span data-arm="{esc(pa)}"><span data-role="arm-label">{esc(_arm_label(run, pa))}'
               f'</span></span> minus the floor, per scene.'
               f'</figcaption></figure><details class="twin" open><summary>table view</summary>'
               + plain_table("tab-per-speed", "per t0-speed band", hdr, rows) + "</details></div>")
    return "".join(out)


FAMILY_METRICS = {
    "longitudinal": [("speed_mae_mps", "speed MAE", "m/s", "f3"), ("along_mae_m", "along-track MAE", "m", "f3"),
                     ("accel_mae_mps2", "accel MAE", "m/s²", "f3"),
                     ("target_speed_acc/within_1.0_mps", "speed within 1 m/s", "frac", "f3"),
                     ("ego_progress/progress_ratio_mean", "progress ratio", "×", "f3")],
    "lateral": [("cross_mae_m", "cross-track MAE", "m", "f3"), ("heading_mae_deg", "heading MAE", "°", "f2"),
                ("curvature_mae_1pm", "curvature MAE", "1/m", "f4"), ("yaw_rate_mae_degps", "yaw-rate MAE", "°/s", "f2")],
    "tactical": [("lateral_decision/accuracy", "lateral decision acc", "frac", "f3"),
                 ("lateral_decision/kappa", "lateral κ", "", "f3"),
                 ("longitudinal_decision/accuracy", "longitudinal decision acc", "frac", "f3"),
                 ("longitudinal_decision/kappa", "longitudinal κ", "", "f3"),
                 ("goal_setting/goal_point_error_m", "goal-point error", "m", "f2"),
                 ("goal_setting/goal_bearing_mae_deg", "goal bearing MAE", "°", "f2")],
    "strategic": [("nav_compliance/readouts/plan/conditionings/nav_true/compliance_with_TRUE_command",
                   "nav compliance (true cmd)", "frac", "f3"),
                  ("nav_compliance/readouts/plan/paired_true_minus_zero", "Δ vs nav withheld", "", "sgn3")],
}


# ⛔ BINDING (W2, 2026-09-20): `cross_mae_m` is a LATERAL OFFSET AT MATCHED TIME INDEX (the ego-frame y
# column differenced, `four_families.py:187`/`:776`) — NOT a distance to the path. Two arms whose plans
# keep y ≈ 0 score identically however differently they drive (MEASURED on W1's warmup run: CV and STOP
# both 1.0658 m). The qualifier is rendered BESIDE the lateral headline, with the along-track context
# number that says whether the offset is informative at all, and the projection-based alternative named.
CROSS_QUALIFIER = ("`cross_mae_m` is a lateral offset at matched time index, informative only while the "
                   "along-track error is small; read it with the LONGITUDINAL family, and use "
                   "`headline.pathgeom_crosstrack_m` when a distance-to-path is meant.")
CROSS_ALTERNATIVE = "headline.pathgeom_crosstrack_m (lateral.py::frenet_dense)"


def _lateral_qualifier(run: Run) -> str:
    """The qualifier + per-arm along-track context + the projection-based alternative + any TIE."""
    qual = next((v for a in run.arm_names
                 for v, _k in [run.family_field(a, "lateral", "_cross_is")] if isinstance(v, str)), CROSS_QUALIFIER)
    alt = next((v for a in run.arm_names
                for v, _k in [run.family_field(a, "lateral", "_projection_based_alternative")]
                if isinstance(v, str)), CROSS_ALTERNATIVE)
    rows, by_value = [], {}
    for arm in run.arm_names:
        cv, ck, _s = run.family_metric(arm, "lateral", "cross_mae_m")
        if not _is_num(cv):
            continue
        by_value.setdefault(round(float(cv), 6), []).append(arm)
        ctx, ctx_k = run.family_field(arm, "lateral", "_along_mae_m_for_context")
        src = "the run's own lateral block"
        if not _is_num(ctx):                      # fall back to the LONGITUDINAL family, and SAY so
            ctx, ctx_k, _sc = run.family_metric(arm, "longitudinal", "along_mae_m")
            src = "LONGITUDINAL family (along_mae_m)"
        alt_v, alt_k, _sc2 = run.family_metric(arm, "lateral", "pathgeom_crosstrack_m")
        if not _is_num(alt_v):
            alt_v, alt_k = run.family_field(arm, "lateral", "pathgeom_crosstrack_m")
        rows.append({"arm": arm, "label": _arm_label(run, arm), "cells": [
            num_span(ck, cv, "f4"),
            (num_span(ctx_k, ctx, "f4") + f'<span class="mut"> · {esc(src)}</span>') if _is_num(ctx)
            else '<span class="mut">— no along-track context in this run</span>',
            num_span(alt_k, alt_v, "f4") if _is_num(alt_v) else
            f'<span class="mut">not in this run — {esc(alt)}</span>']})
    if not rows:
        return ""
    ties = [arms for arms in by_value.values() if len(arms) > 1]
    tie_html = ""
    for arms in ties:
        chips = ", ".join(f'<span class="chip" data-arm="{esc(a)}"><b data-role="arm-label">'
                          f'{esc(_arm_label(run, a))}</b></span>' for a in arms)
        v = next(val for val, ar in by_value.items() if ar is arms)
        tie_html += (f'<p class="sub">⚠ {chips} read the SAME cross-track ({fmt(v, "f4")} m) — the offset '
                     f'cannot separate them; a moving arm and a standing one tie here by construction.</p>')
    return (f'<div class="banner serious" data-caveat="lateral.cross_mae_is_an_offset">'
            f'<span class="ic">▲</span><b>CROSS-TRACK IS AN OFFSET, NOT A DISTANCE TO THE PATH</b> — {esc(qual)} '
            f'<span class="mut">(W2, EvalFlyWheel, 2026-09-20)</span></div>{tie_html}'
            + plain_table("tab-fam-lateral-qual", "cross-track read WITH its along-track context, and the "
                          "projection-based alternative to use when a distance-to-path is meant",
                          ["arm", "cross_mae_m (m)", "along-track context (m)", alt], rows))


def families_html(run: Run) -> str:
    out = ['<h2 id="families">6 · The four metric families</h2>'
           '<p class="lede">OUR instruments on trajectory geometry — not NavSim\'s score. Binding (PI 2026-08-02): all four, '
           'per family, never pooled; a family that cannot be computed is shown with its reason and n, never dropped.</p>']
    for fam, metrics in FAMILY_METRICS.items():
        hdr = ["arm", "status", "n"] + [f"{lab}{f' ({u})' if u else ''}" for _k, lab, u, _f in metrics] + ["reason / scope"]
        rows = []
        for arm in run.arm_names:
            fb = g(run.arm(arm), "families", fam) or {"status": "UNAVAILABLE", "reason": "family block absent", "n": 0}
            status = fb.get("status", "UNAVAILABLE")
            cells = [_pill(status), _n(run, fb["n"], "int", "arms", arm, "families", fam, "n") if _is_num(fb.get("n")) else "—"]
            scopes = fb.get("scopes") or {}
            for key, _lab, _u, f in metrics:
                val, kpath, sc = run.family_metric(arm, fam, key)
                if isinstance(val, dict):          # a REFUSAL leaf (W2's stationary-plan shape) — print it
                    why = val.get("reason", "no reason recorded")
                    extra = " · ".join(f"{k} {val[k]}" for k in ("n_steps_total", "min_ds_m") if k in val)
                    n_txt = f" (n {val['n']})" if _is_num(val.get("n")) else ""
                    tip = why + (f" [{extra}]" if extra else "")
                    cells.append(f'<span class="mut" data-refusal-k="{esc(kpath)}" title="{esc(tip)}">'
                                 f'UNAVAILABLE{esc(n_txt)}</span>')
                elif val is None:
                    cells.append('<span class="mut">—</span>')
                else:
                    suffix = {"stage_one": "S1", "stage_two": "S2"}.get(sc, sc or "")
                    cells.append(num_span(kpath, val, f)
                                 + (f'<span class="mut"> {esc(suffix)}</span>' if suffix else ""))
            reason = fb.get("reason", "")
            scopes_txt = ", ".join(f"{s}: {g(b, 'status', default='values') if isinstance(b, dict) else '?'}"
                                   for s, b in scopes.items())
            cells.append((f'<span data-refusal-k="{esc(P("arms", arm, "families", fam))}" title="{esc(reason)}">'
                          f'{esc(_short(reason, 96))}</span>' if reason else "")
                         + (f'<br><span class="mut">scopes — {esc(scopes_txt)}</span>' if scopes_txt else ""))
            rows.append({"arm": arm, "label": _arm_label(run, arm), "cells": cells})
        tab = plain_table(f"tab-fam-{fam}", f"{fam.upper()} (stage suffix = the scope the value came from)", hdr, rows)
        extra = ""
        if fam == "lateral":
            extra = _lateral_qualifier(run)
        if fam == "strategic" and _oracle_arms(run):
            chips = ", ".join(f'<span class="chip" data-arm="{esc(a)}"><b data-role="arm-label">'
                              f'{esc(_arm_label(run, a))}</b></span>' for a in _oracle_arms(run))
            extra = (f'<div class="banner serious" data-caveat="{esc(ORACLE_COMMAND_CAVEAT["id"])}">'
                     f'<span class="ic">▲</span><b>NOT A ROUTE CLAIM</b> — {chips} are fed the NavSim command, '
                     f'which is a route-level ORACLE: {esc(ORACLE_COMMAND_CAVEAT["text"])} A nav-compliance '
                     f'readout of such an arm measures whether the plan follows the ORACLE it was given, not '
                     f'whether the model can find the route. <span class="mut">'
                     f'({esc(ORACLE_COMMAND_CAVEAT["source"])})</span></div>')
        out.append(f'<div class="card fam"><h3>{esc(fam.upper())}</h3>{tab}{extra}</div>')
    return "".join(out).replace('<td><span data-refusal-k', '<td class="reason"><span data-refusal-k')


def provenance_html(run: Run, verify_note: str, gallery_note: str) -> str:
    b = run.bench
    ck = b.get("ckpt") or {}
    dk = run.devkit
    patches = "".join(f"<li>{esc(p.get('name'))} <span class='mut'>({esc(p.get('kind'))})</span></li>" for p in dk.get("patches") or [])
    arms = []
    for arm in run.arm_names:
        a = run.arm(arm)
        files = " · ".join(f'<a href="../{esc(v)}">{esc(k)}</a>' for k, v in (a.get("files") or {}).items()
                           if isinstance(v, str))
        arms.append({"arm": arm, "label": _arm_label(run, arm),
                     "cells": [esc(a.get("kind")), esc(a.get("role", "")), esc(a.get("status")),
                               esc(", ".join(a.get("declared_inputs") or []) or "none"), files]})
    ctrl = run.summary.get("controls") or {}
    ctrl_rows = "".join(f"<tr><th scope='row'>{esc(k)}</th><td class='wrap'><code>{esc(_short(json.dumps(v, ensure_ascii=False), 400))}</code></td></tr>"
                        for k, v in ctrl.items())
    sch = "".join(f"<li>{esc(k)}.json: <b>{esc(v.get('status'))}</b>{(' — ' + esc('; '.join(v.get('errors')[:3]))) if v.get('errors') else ''}</li>"
                  for k, v in run.schema_status.items())
    prov = run.summary.get("provenance") or {}
    return (f'<h2 id="provenance">8 · Provenance and integrity</h2><div class="card">'
            f'<table class="plain"><tbody>'
            f"<tr><th scope='row'>run</th><td class='wrap'>{esc(run.run_id)} · utc {esc(b.get('utc', '?'))} · status "
            f"{esc(b.get('status', '?'))}{(' — ' + esc(b.get('status_reason'))) if b.get('status_reason') else ''}</td></tr>"
            f"<tr><th scope='row'>git HEAD</th><td class='wrap'><code>{esc(b.get('git_head', '?'))}</code> {esc(g(b, 'git', 'note', default='') or g(b, 'git', 'reason', default=''))}</td></tr>"
            f"<tr><th scope='row'>checkpoint</th><td class='wrap'>{esc(ck.get('registry_key') or '—')} · <code>{esc(ck.get('path') or '—')}</code> · "
            f"sha256 {esc(ck.get('sha256') or ck.get('sha256_status') or '—')}{(' · md5 ' + esc(ck.get('md5'))) if ck.get('md5') else ''}</td></tr>"
            f"<tr><th scope='row'>devkit</th><td class='wrap'>{esc(dk.get('repo'))}@<code>{esc(dk.get('sha'))}</code><ul class='sub'>{patches}</ul></td></tr>"
            f"<tr><th scope='row'>schema (W1 v1)</th><td class='wrap'><ul class='sub'>{sch}</ul></td></tr>"
            f"<tr><th scope='row'>source</th><td class='wrap'>{esc(prov.get('adapter', 'taniteval.bench'))} · "
            f"{esc(prov.get('source_dir', ''))} · leaderboard-eligible <b>{esc(prov.get('leaderboard_eligible', True))}</b></td></tr>"
            f"<tr><th scope='row'>number check</th><td class='wrap'>{esc(verify_note)}</td></tr>"
            f"<tr><th scope='row'>gallery</th><td class='wrap'>{esc(gallery_note)}</td></tr>"
            f"</tbody></table></div>"
            f'<div class="card"><h3>arms</h3>'
            + plain_table("tab-arms", "declared inputs = what the arm may read at inference", ["arm", "kind", "role", "status",
                                                                                          "declared inputs", "files"], arms)
            + f'</div><div class="card"><h3>controls carried by the run</h3><table class="plain"><tbody>{ctrl_rows}</tbody></table></div>')


# ------------------------------------------------------------------------------------ entry
def render_report(run_dir, *, published_path=None, gallery: dict | None = None, out_dir=None,
                  verify_note: str = "(pending)", extra_banners: list | None = None) -> dict:
    """Render the report; returns {index, figs, n_numbers(estimated), ...}. Does NOT verify (see verify.py)."""
    from .contract import load_run
    run = load_run(run_dir)
    out = Path(out_dir) if out_dir else run.root / "report"
    (out / "fig").mkdir(parents=True, exist_ok=True)
    published = None
    if published_path and Path(published_path).is_file():
        published = json.loads(Path(published_path).read_text(encoding="utf-8"))
        published["_path"] = str(published_path)
    extra = list(extra_banners or [])
    if run.is_fixture:
        extra.append(("warning", "FIXTURE: this run directory is a W5 adapter conversion of banked outputs "
                                 f"({g(run.summary, 'provenance', 'source_dir', default='?')}); leaderboard-eligible = false."))
    iv = g(run.summary, "estimator", "interval") or {}
    if iv.get("status") == "UNAVAILABLE":
        extra.append(("warning", f"No interval on this run: {iv.get('reason')} (n {iv.get('n')}). Point estimates and paired "
                                 "per-scene counts only."))
    if run.summary.get("claim_bearing") is False:
        extra.append(("serious", "NOT CLAIM-BEARING: this protocol's numbers are diagnostic only (never a capability claim)."))
    figs: dict = {}
    gal_html, gal_note = "", "not requested"
    if gallery is not None:
        from .gallery import gallery_html
        gal_html, gal_note = gallery_html(run, gallery)
    body = (banners_html(run, extra) + overview_html(run) + headline_html(run, published, figs) + paired_html(run, figs)
            + submetrics_html(run) + per_log_html(run, figs) + per_speed_html(run, figs) + families_html(run)
            + (gal_html or '<h2 id="gallery">7 · Failure gallery</h2><p class="lede mut">Not rendered for this report '
                            f'({esc(gal_note)}).</p>')
            + provenance_html(run, verify_note, gal_note)
            + f'<div class="foot">TanitEval bench report {esc(REPORT_VERSION)} · rendered '
              f'{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ} from <code>summary.json</code> '
              f'sha256 {hashlib.sha256((run.root / "summary.json").read_bytes()).hexdigest()[:16]} · every number above is '
              f'stamped with its summary.json pointer and re-checked by parsing this file (taniteval.benchreport.verify).</div>')
    meta = {"tanitad-run-id": run.run_id, "tanitad-protocol": run.protocol,
            "tanitad-summary-sha256": hashlib.sha256((run.root / "summary.json").read_bytes()).hexdigest(),
            "tanitad-published": (published or {}).get("_path", ""), "generator": f"taniteval.benchreport {REPORT_VERSION}"}
    title = f"TanitEval · {run.protocol} · {run.run_id}"
    html_text = page(title, stamp_html(run), body, meta)
    (out / "index.html").write_text(html_text, encoding="utf-8")
    for name, svg in figs.items():
        (out / "fig" / f"{name}.svg").write_text(standalone_svg(svg), encoding="utf-8")
    return {"index": str(out / "index.html"), "figs": sorted(figs), "run": run, "published": published,
            "gallery_note": gal_note}
