"""Assemble the data model, render it, and splice it between the BENCH markers.

The splice is the only write to ``LEADERBOARD.md``: bytes before ``<!-- BENCH:BEGIN -->`` and after the
line holding ``<!-- BENCH:END -->`` are copied verbatim. If the markers are absent, the section is
inserted (followed by one blank line) immediately before the first line starting with
``insert_before_line_prefix``. The host file's newline convention (CRLF on this checkout) is detected
and preserved, so ``delete the section → build`` returns the file byte-identical.
"""
from __future__ import annotations

from pathlib import Path

from . import render, sources
from .sources import SourceError


def load_model(root: Path, cfg: dict) -> dict:
    """Every value the page will print, read from disk now (nothing is cached between builds)."""
    pub = sources.load_published(root, cfg)
    reg = sources.load_registry(root, cfg)
    internal = sources.load_internal(root, cfg, reg)
    w1, _ = sources.load_bench_runs(root, cfg)
    # only a REPORTED value supersedes a legacy row (a failed W1 run must not delete a measured one)
    reported = {(r["protocol"], r["key"]): r["official_x100"] for r in w1 if r["official_x100"] is not None}
    legacy, notes = sources.load_legacy(root, cfg, set(reported), reported)
    audit, audit_path = sources.load_audit(root, cfg)
    return {"published": pub, "internal": internal, "navsim": w1 + legacy, "notes": notes,
            "audit": audit, "audit_path": audit_path}


def render_section(root: Path | None = None, cfg: dict | None = None) -> str:
    root = Path(root) if root else sources.REPO
    cfg = cfg or sources.load_config(root)
    return render.render_md(load_model(root, cfg), cfg)


def _newline_of(text: str) -> str:
    return "\r\n" if text.count("\r\n") and text.count("\r\n") == text.count("\n") else "\n"


def splice(host: str, section_lf: str, cfg: dict) -> str:
    """Return ``host`` with the BENCH section replaced (or inserted). ``section_lf`` uses '\\n'."""
    nl = _newline_of(host)
    sec = section_lf.replace("\n", nl)
    b, e = cfg["markers"]["begin"], cfg["markers"]["end"]
    nb, ne = host.count(b), host.count(e)
    if nb != ne or nb > 1:
        raise SourceError(f"LEADERBOARD.md has {nb} BEGIN and {ne} END markers — refusing to guess")
    if nb == 1:
        i = host.index(b)
        # the section starts at the beginning of the BEGIN marker's line
        i = host.rfind(nl, 0, i) + len(nl) if host.rfind(nl, 0, i) >= 0 else 0
        j = host.index(e, i)
        j_end = host.find(nl, j)
        j_end = len(host) if j_end < 0 else j_end + len(nl)
        return host[:i] + sec + host[j_end:]
    prefix = cfg["insert_before_line_prefix"]
    pos = 0
    for line in host.split(nl):
        if line.startswith(prefix):
            return host[:pos] + sec + nl + host[pos:]
        pos += len(line) + len(nl)
    raise SourceError(f"no BENCH markers and no line starting with {prefix!r} to insert before")


def _w5_charts(model: dict) -> str:
    """Leaderboard figures from W5's components (``taniteval.benchreport.charts``), imported, never
    edited. If W5 later publishes its own ``render_leaderboard_charts(model)``, that wins."""
    import importlib
    for modname in ("taniteval.benchreport.leaderboard_charts", "taniteval.benchreport"):
        try:
            fn = getattr(importlib.import_module(modname), "render_leaderboard_charts", None)
        except Exception:
            continue
        if callable(fn):
            return fn(model)                     # W5's own components, given W4's canonical model
    return render.w5_charts(model)               # fallback: W5 chart primitives, or tables only


def build(root: Path | None = None, check: bool = False, write_html: bool = True,
          md_path: Path | None = None, html_path: Path | None = None, cfg: dict | None = None) -> dict:
    """Regenerate the BENCH section. ``md_path`` / ``html_path`` / ``cfg`` override the targets and the
    configuration (tests run the acceptance round-trip on a temporary copy of the page)."""
    root = Path(root) if root else sources.REPO
    cfg = cfg or sources.load_config(root)
    before = sources.bench_tree_state(root, cfg)
    model = load_model(root, cfg)
    section = render.render_md(model, cfg)
    # ⛔ A PAGE MAY NOT BE ASSEMBLED FROM A MOVING TREE. MEASURED 2026-09-20: W1's warmup run dirs went
    # 4 → 2 (one cited run deleted, a new one added) between two builds minutes apart, so the acceptance
    # round-trip read DIFFERS for a generator that was working perfectly. Fail LOUD and leave the page
    # alone; a silently re-rendered page would cite a run that no longer exists.
    after = sources.bench_tree_state(root, cfg)
    if after != before:
        gone, new_ = sorted(set(before) - set(after)), sorted(set(after) - set(before))
        raise SourceError(
            "the W1 results tree CHANGED WHILE THIS PAGE WAS BEING BUILT — refusing to write. "
            f"removed: {gone or '—'} · added: {new_ or '—'} · "
            f"changed: {sorted(k for k in set(before) & set(after) if before[k] != after[k]) or '—'}. "
            "Re-run when the producing stream is idle.")
    md_path = Path(md_path) if md_path else root / cfg["target_md"]
    host = md_path.read_bytes().decode("utf-8")
    new = splice(host, section, cfg)
    html_text = render.render_html(section, _w5_charts(model)).replace("\n", _newline_of(host))
    html_path = Path(html_path) if html_path else root / cfg["target_html"]
    changed_md = new != host
    changed_html = (not html_path.exists()) or html_path.read_bytes().decode("utf-8") != html_text
    if check:
        return {"md_up_to_date": not changed_md, "html_up_to_date": not changed_html, "digest": render.canonical_digest(model)}
    if changed_md:
        md_path.write_bytes(new.encode("utf-8"))
    if write_html and changed_html:
        html_path.write_bytes(html_text.encode("utf-8"))
    return {"md": str(md_path), "md_changed": changed_md, "html": str(html_path), "html_changed": changed_html,
            "digest": render.canonical_digest(model), "section_bytes": len(section.encode("utf-8"))}
