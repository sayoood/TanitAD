#!/usr/bin/env python3
"""LIBRARY INGEST — bank a paper locally, with metadata, and index it.

⛔ WHY THIS EXISTS (PI directive, 2026-08-18). This programme runs literature
research constantly and has **seven** `KNOWLEDGE_BASE.md` files full of
`[PUBLISHED]` findings — every one of them citing a URL and **keeping nothing**.
Three failure modes follow, and we have hit all three:

* **Link rot / paywall drift.** A citation that cannot be re-opened cannot be
  re-checked, and an unre-checkable claim silently becomes folklore.
* **Secondary-source drift.** The 2026-08-18 frozen-encoder review had to mark
  five numeric claims `PUBLISHED-SECONDARY` — read from aggregator summaries,
  **inadmissible for the registry** — precisely because the primaries were not
  to hand. Banking the PDF is what converts SECONDARY into PRIMARY.
* **Re-downloading the same paper per stream.** Two streams citing one paper
  kept two copies, or none.

⭐ THE RULE THIS TOOL MAKES EXECUTABLE (CLAUDE.md "Research banking"): *a
research deliverable that cites a paper it did not bank is incomplete.* A rule
in prose decays — C126 measured exactly that, where a duplication documented in
a report was re-counted by every later census because **no prose correction can
reach a glob**. So the rule ships as a tool plus a test, not as a paragraph.

Usage
-----
    python tools/kb_add.py 2411.04983 --note "DINO-WM: frozen DINOv2 + CEM/MPC"
    python tools/kb_add.py https://arxiv.org/abs/2601.03460 --tag frozen-encoder
    python tools/kb_add.py --local Ressources/paper.pdf --key unece-r157 \
        --title "UNECE R157" --note "regulatory ceiling on ALKS"
    python tools/kb_add.py --reindex      # rebuild LIBRARY.md from library.json
    python tools/kb_add.py --verify       # sha256 every entry against disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve_lib(root: Path) -> Path:
    """Locate the Library across the 2026-08-22 "Research Hub" -> "Research Lab"
    rename. Resolving by CONTENT (an existing library.json) rather than by the
    current name keeps a stale checkout and a renamed one banking into the SAME
    library; picking a name blindly forks it into a fresh empty dir and the split
    is invisible until a --verify months later."""
    candidates = [root / "TanitAD Research Lab" / "Library",
                  root / "TanitAD Research Hub" / "Library"]
    for cand in candidates:
        if (cand / "library.json").exists():
            return cand
    for cand in candidates:
        if cand.parent.is_dir():
            return cand
    return candidates[0]


LIB = _resolve_lib(ROOT)
PAPERS = LIB / "papers"
INDEX_JSON = LIB / "library.json"
INDEX_MD = LIB / "LIBRARY.md"

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s[:n].rstrip("-")


def _load() -> dict:
    if INDEX_JSON.exists():
        return json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    return {"schema": 1, "entries": {}}


def _save(db: dict) -> None:
    LIB.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(json.dumps(db, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")


class _IndexLock:
    """Serialise the index read-modify-write across concurrent bankers.

    MEASURED 2026-08-23: two research streams banking at the same time each read
    `library.json`, added their own entry, and wrote the whole file back — so the
    later write erased the earlier stream's entries. **7 PDFs landed on disk with
    no index row.** The failure is invisible to `--verify`, which iterates the
    entries: an entry that was never written cannot be found missing. A lock plus
    the orphan scan below closes both halves.

    `O_CREAT|O_EXCL` is used rather than `fcntl`/`msvcrt` because the library sits
    on a Google Drive mount where byte-range locking is unreliable, and because
    the same code must run on Windows and Linux.
    """

    def __init__(self, path: Path, timeout: float = 30.0):
        self.path, self.timeout, self.fd = Path(str(path) + ".lock"), timeout, None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                # A lock with no live holder is debris, not contention — the same
                # shape as a stale .git/index.lock. Break it once it is clearly old.
                try:
                    if time.time() - self.path.stat().st_mtime > self.timeout:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(
                        f"could not acquire {self.path} within {self.timeout}s. "
                        f"If no other kb_add is running, delete it — it is debris.")
                time.sleep(0.15)
            except OSError:
                time.sleep(0.15)
                if time.time() > deadline:
                    raise

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


def reindex_orphans(tags: list[str], cited_by: list[str]) -> int:
    """Recover banked PDFs the concurrent-write race left out of the index.

    The bytes are already on disk and are the primary source; only the row was
    lost. Metadata is re-fetched for an arXiv key (a small XML call, no PDF
    re-download); anything else is recorded from its filename with the title
    marked UNVERIFIED rather than invented.
    """
    found = orphans()
    if not found:
        print("no orphans")
        return 0
    for p in found:
        key = p.name.split("_", 1)[0]
        entry = {"key": key, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                 "bytes": p.stat().st_size, "sha256": _sha256(p),
                 "banked": date.today().isoformat(),
                 "tags": sorted(set(tags + ["recovered-orphan"])),
                 "cited_by": sorted(set(cited_by)),
                 "note": ("index row lost to the pre-2026-08-23 concurrent-write "
                          "race; bytes verified by sha256, row rebuilt")}
        if ARXIV_ID.fullmatch(key):
            try:
                _ssl()
                meta = arxiv_meta(key)
                entry.update({"kind": "arxiv", "title": meta.get("title", ""),
                              "authors": meta.get("authors", []),
                              "published": meta.get("published", ""),
                              "abstract": meta.get("abstract", ""),
                              "url": f"https://arxiv.org/abs/{key}"})
            except Exception as e:                       # noqa: BLE001
                entry.update({"kind": "arxiv", "url": f"https://arxiv.org/abs/{key}",
                              "title": p.stem.split("_", 1)[-1].replace("-", " "),
                              "authors": [], "published": "", "abstract": "",
                              "meta_error": f"UNVERIFIED title: {type(e).__name__}"})
        else:
            entry.update({"kind": "local", "url": "", "authors": [],
                          "published": "", "abstract": "",
                          "title": p.stem.split("_", 1)[-1].replace("-", " ")})
        _commit_entry(key, entry)
        print(f"recovered {key}: {entry['bytes']//1024} KB — {entry['title'][:60]}")
    return len(found)


def _commit_entry(key: str, entry: dict) -> None:
    """Insert one entry under the lock, RE-READING the index first.

    ⛔ The race was not the write, it was the STALE SNAPSHOT: `_load()` runs, then
    a download takes up to 180 s, then the whole file is written back from the old
    snapshot — erasing every entry a concurrent stream added in between. Locking
    the write alone would not fix it; the index must be re-read inside the lock so
    this becomes a merge rather than a clobber.
    """
    with _IndexLock(INDEX_JSON):
        db = _load()
        db.setdefault("entries", {})[key] = entry
        _save(db)


def _merge_entry(key: str, **updates) -> None:
    """Update an existing entry's tags/citations/note under the same discipline."""
    with _IndexLock(INDEX_JSON):
        db = _load()
        e = db.setdefault("entries", {}).get(key)
        if e is None:
            return
        for field in ("tags", "cited_by"):
            if field in updates:
                e[field] = sorted(set(e.get(field, [])) | set(updates[field]))
        note = updates.get("note")
        if note and note not in e.get("note", ""):
            e["note"] = (e.get("note", "") + " | " + note).strip(" |")
        _save(db)


def orphans() -> list[Path]:
    """Banked PDFs with no row in the index — the race's fingerprint.

    ⛔ `verify()` iterates ENTRIES, so it is structurally unable to see a file the
    index never recorded. Absence has to be probed from the OTHER side."""
    if not PAPERS.is_dir():
        return []
    indexed = {(ROOT / e["path"]).resolve()
               for e in _load().get("entries", {}).values() if e.get("path")}
    return sorted(p for p in PAPERS.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf"
                  and p.resolve() not in indexed)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ssl():
    """The dev box sits behind a TLS proxy: certifi fails, truststore works.
    Bare urllib/curl otherwise reports a transport error that reads exactly like
    an outage or a paywall — a false negative we have paid for before."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:                                        # noqa: BLE001
        pass


def arxiv_meta(aid: str) -> dict:
    """Title / authors / abstract from the arXiv API. Best-effort: a metadata
    failure must never block the BYTES from being banked, because the bytes are
    the thing that stops being available."""
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            xml = r.read().decode("utf-8", "replace")
        # ⛔ SCOPE TO <entry>, NEVER THE WHOLE FEED. The Atom response opens with
        # a FEED-level <title> that echoes the query ("arXiv Query:
        # search_query=&id_list=..."), so a naive first-match `re.search` over
        # the document banks that string as the paper's title — MEASURED on the
        # first migration run, which filed all 11 papers under the query echo
        # and reported success. The tool "worked"; the content was garbage.
        em = re.search(r"<entry>(.*?)</entry>", xml, re.S)
        ent = em.group(1) if em else ""
        def grab(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", ent, re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        authors = re.findall(r"<name>(.*?)</name>", ent)
        return {"title": grab("title"), "abstract": grab("summary")[:900],
                "authors": authors[:12], "published": grab("published")[:10]}
    except Exception as e:                                   # noqa: BLE001
        return {"title": "", "abstract": "", "authors": [],
                "published": "", "meta_error": f"{type(e).__name__}: {e}"}


def add_arxiv(ident: str, *, tags: list[str], note: str,
              cited_by: list[str]) -> str:
    m = ARXIV_ID.search(ident)
    if not m:
        raise SystemExit(f"no arXiv id found in {ident!r}")
    aid = m.group(1)
    db = _load()
    if aid in db["entries"] and Path(db["entries"][aid]["path"]).name:
        p = ROOT / db["entries"][aid]["path"]
        if p.exists():
            _merge_entry(aid, tags=tags, cited_by=cited_by, note=note)
            print(f"already banked: {aid} — tags/citations updated")
            return aid

    _ssl()
    meta = arxiv_meta(aid)
    PAPERS.mkdir(parents=True, exist_ok=True)
    name = f"{aid}_{_slug(meta.get('title') or 'untitled')}.pdf"
    dest = PAPERS / name
    url = f"https://arxiv.org/pdf/{aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "TanitAD-KB/1"})
    with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    size = dest.stat().st_size
    if size < 20_000:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"refusing {aid}: got {size} B — not a PDF")

    _commit_entry(aid, {
        "key": aid, "kind": "arxiv",
        "title": meta.get("title", ""), "authors": meta.get("authors", []),
        "published": meta.get("published", ""),
        "abstract": meta.get("abstract", ""),
        "url": f"https://arxiv.org/abs/{aid}",
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "bytes": size, "sha256": _sha256(dest),
        "banked": date.today().isoformat(),
        "tags": sorted(set(tags)), "note": note,
        "cited_by": sorted(set(cited_by)),
    })
    print(f"banked {aid}: {size//1024} KB — {meta.get('title','')[:70]}")
    return aid


def add_local(path: Path, *, key: str, title: str, tags: list[str],
              note: str, cited_by: list[str], url: str = "") -> str:
    src = (ROOT / path) if not path.is_absolute() else path
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    PAPERS.mkdir(parents=True, exist_ok=True)
    dest = PAPERS / f"{key}_{_slug(title or src.stem)}{src.suffix}"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    _commit_entry(key, {
        "key": key, "kind": "local", "title": title or src.stem,
        "authors": [], "published": "", "abstract": "", "url": url,
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "bytes": dest.stat().st_size, "sha256": _sha256(dest),
        "banked": date.today().isoformat(),
        "tags": sorted(set(tags)), "note": note,
        "cited_by": sorted(set(cited_by)),
    })
    print(f"banked {key}: {dest.stat().st_size//1024} KB — {title or src.stem}")
    return key


def reindex() -> None:
    db = _load()
    es = sorted(db["entries"].values(), key=lambda e: e.get("banked", ""),
                reverse=True)
    tags: dict[str, list] = {}
    for e in es:
        for t in e.get("tags") or ["untagged"]:
            tags.setdefault(t, []).append(e)
    tot = sum(e["bytes"] for e in es)
    out = [
        "# LIBRARY — primary sources, banked locally",
        "",
        "> ⛔ **The rule** (CLAUDE.md, *Research banking*): **a research",
        "> deliverable that cites a paper it did not bank is incomplete.** Add",
        "> with `python tools/kb_add.py <arxiv-id-or-url> --tag <topic>",
        "> --cited-by <path-of-your-report>`; never hand-edit this file — it is",
        "> generated from `library.json` by `--reindex`.",
        "",
        "> A `[PUBLISHED]` entry in any `KNOWLEDGE_BASE.md` should cite the",
        "> library key, not only a URL. A URL is a claim about the internet; a",
        "> banked sha256 is a claim about a file we hold.",
        "",
        f"**{len(es)} entries · {tot/1e6:.1f} MB · index generated by "
        f"`kb_add.py --reindex`**",
        "",
        "## By topic",
        "",
    ]
    for t in sorted(tags):
        out.append(f"### `{t}` ({len(tags[t])})")
        out.append("")
        for e in tags[t]:
            title = e["title"] or e["key"]
            out.append(f"- **[{e['key']}]({e['path'].replace(' ', '%20')})** — "
                       f"{title}"
                       + (f"  \n  *{e['note']}*" if e.get("note") else ""))
        out.append("")
    out += ["## Full index (newest first)", "",
            "| key | title | banked | size | sha256 | cited by |",
            "|---|---|---|---:|---|---|"]
    for e in es:
        cb = ", ".join(Path(c).name for c in e.get("cited_by", [])) or "—"
        out.append(f"| [{e['key']}]({e['path'].replace(' ', '%20')}) | "
                   f"{(e['title'] or '')[:70]} | {e.get('banked','')} | "
                   f"{e['bytes']//1024} KB | `{e['sha256'][:12]}` | {cb} |")
    out.append("")
    INDEX_MD.write_text("\n".join(out), encoding="utf-8")
    print(f"reindexed: {len(es)} entries, {tot/1e6:.1f} MB -> {INDEX_MD}")


def verify() -> int:
    """⛔ Verify by CONTENT, never by presence — the programme's standing rule.
    A file that exists but whose bytes changed is worse than a missing one."""
    db = _load()
    bad = 0
    for k, e in sorted(db["entries"].items()):
        p = ROOT / e["path"]
        if not p.exists():
            print(f"MISSING  {k}: {e['path']}")
            bad += 1
            continue
        got = _sha256(p)
        if got != e["sha256"]:
            print(f"MISMATCH {k}: recorded {e['sha256'][:12]} got {got[:12]}")
            bad += 1

    # ⛔ Probe the absence from the OTHER side. Iterating entries can never reveal
    # an entry that was never written — which is exactly what the concurrent-write
    # race produced (7 banked PDFs, no index rows, --verify reporting 0 problems).
    orph = orphans()
    for p in orph:
        print(f"ORPHAN   {p.name}: on disk, absent from the index "
              f"(re-bank it, or --reindex-orphans)")
    bad += len(orph)

    print(f"verified {len(db['entries'])} entries, {len(orph)} orphan(s), "
          f"{bad} problem(s)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ident", nargs="?", help="arXiv id or URL")
    ap.add_argument("--local", type=Path, help="bank a local file instead")
    ap.add_argument("--key", default="", help="key for --local")
    ap.add_argument("--title", default="", help="title for --local")
    ap.add_argument("--url", default="", help="source URL for --local")
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--cited-by", action="append", default=[],
                    help="repo path of the report that cites it")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reindex-orphans", action="store_true",
                    help="rebuild index rows for banked PDFs the concurrent-write "
                         "race left unindexed. Pass --cited-by to attribute them.")
    a = ap.parse_args(argv)

    if a.reindex_orphans:
        n = reindex_orphans(a.tag or [], a.cited_by or [])
        if n:
            reindex()
        return 0
    if a.verify:
        return verify()
    if a.reindex and not (a.ident or a.local):
        reindex()
        return 0
    if a.local:
        if not a.key:
            raise SystemExit("--local needs --key")
        add_local(a.local, key=a.key, title=a.title, tags=a.tag, note=a.note,
                  cited_by=a.cited_by, url=a.url)
    elif a.ident:
        add_arxiv(a.ident, tags=a.tag, note=a.note, cited_by=a.cited_by)
    else:
        ap.print_help()
        return 2
    reindex()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
