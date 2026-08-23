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

⛔ TWO DEFECTS MEASURED 2026-08-23 (Research Lab run 001, "Process hazards
measured today" §3 and §4) — both are fixed here, and both are pinned by
`tools/tests/test_library.py`:

1. **A mid-download `ConnectionResetError` (WinError 10054, arXiv) left a
   correctly-named but TRUNCATED PDF** — exactly 5,242,880 B, no `%%EOF`
   trailer — **that was never recorded in `library.json`**, and a `| tail` pipe
   masked the non-zero exit. *A file that exists with the wrong bytes is worse
   than a missing one, because it defeats every presence check downstream.*
   ⇒ The bytes now land on a **temp path in the same directory**, are verified
   (`Content-Length` when the header is present · `%PDF` header · `%%EOF`
   trailer), and only then `os.replace` into the destination. On ANY failure the
   temp is deleted and the tool exits non-zero with the reason. **The
   destination is never opened for writing.**
2. **`library.json` was read-modify-written with no lock**, so two agents
   running `kb_add.py` at once clobbered each other — measured: a concurrent add
   of `1707.06347` landed between a path-repair's write and its verify and
   overwrote it. ⇒ Every load→modify→write now runs inside an **exclusive lock
   file** (`library.json.lock`, `O_EXCL`, backoff to ~60 s, stale locks broken
   by age), the index is **re-read inside the lock** — a lock that guards only
   the write still loses the edit — and the JSON is written to a temp file and
   `os.replace`d, so a reader never sees a torn index.

⚠️ The Library lives on the G: Drive mount, which serves recently-written files
as `Errno 22` / "Invalid request code" for ~a minute while they sync (same run,
hazard §1). Reads retry with backoff, and every index write is verified by
read-back — an exit code is not evidence that anything was banked.

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
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "TanitAD Research Lab" / "Library"
PAPERS = LIB / "papers"
INDEX_JSON = LIB / "library.json"
INDEX_MD = LIB / "LIBRARY.md"

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

# --- hardening constants (each earned by a measured failure, see the docstring)
LOCK_TIMEOUT_S = 60.0      # how long to wait for a concurrent kb_add to finish
LOCK_STALE_S = 120.0       # older than this with no progress => debris, break it
PDF_MIN_BYTES = 20_000     # below this it is an error page, not a paper
PDF_TAIL_SCAN = 4096       # how far back to look for the %%EOF trailer
TMP_PREFIX = ".kb_add-"    # reserved prefix: never mistakable for a banked paper
_PDF_TRAILING = b" \t\r\n\x0b\x0c\x00"      # "trailer, allowing trailing space"


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s[:n].rstrip("-")


# ------------------------------------------------------------ resilient I/O --
def _read_text_resilient(p: Path, *, attempts: int = 8, what: str = "") -> str:
    """⚠️ The G: Drive mount serves a just-written file as `Errno 22` /
    "Invalid request code" for ~a minute while it syncs (MEASURED 2026-08-23:
    every `kb_add` rewrite of `library.json` poisoned the NEXT read, and `git`
    itself flapped to "not a git repository" twice). One failed read is not
    evidence the file is bad — retry with backoff, then fail loudly."""
    delay, last = 0.25, None
    for i in range(attempts):
        try:
            return p.read_text(encoding="utf-8")
        except OSError as e:                                 # noqa: PERF203
            last = e
            if i == attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    raise SystemExit(
        f"kb_add: cannot read {what or p} after {attempts} attempts: "
        f"{type(last).__name__}: {last}")


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the SAME directory, then `os.replace`.

    `os.replace` is atomic, so a concurrent reader sees either the old file or
    the new one — never a half-written index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f"{TMP_PREFIX}{path.name}.", suffix=".tmp")
    try:
        # ⚠️ newline=None (the default) on purpose: `Path.write_text` — what this
        # replaced — translates to the platform ending, and both index files are
        # CRLF on disk today. Forcing "\n" here would rewrite `library.json` and
        # `LIBRARY.md` end to end on the next add, for no content change.
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            with contextlib.suppress(OSError):     # Drive-FS may refuse fsync
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _discard(tmp: Path) -> None:
    """Remove a temp file, loudly if it will not go — a leftover with a paper's
    name is the defect we are fixing, so never let one linger silently."""
    try:
        tmp.unlink(missing_ok=True)
    except OSError as e:                                     # noqa: BLE001
        print(f"kb_add: WARNING could not remove temp {tmp}: {e}",
              file=sys.stderr)


# ------------------------------------------------------------- the index lock --
def _lock_path() -> Path:
    """Derived from INDEX_JSON so the lock always follows the index it guards
    (a test that retargets the index must not race the real Library)."""
    return INDEX_JSON.with_name(INDEX_JSON.name + ".lock")


def _lock_age(lock: Path) -> float | None:
    try:
        return max(0.0, time.time() - lock.stat().st_mtime)
    except OSError:            # mount flap: unknown age, so never call it stale
        return None


def _lock_holder(lock: Path) -> str:
    try:
        return lock.read_text(encoding="utf-8", errors="replace").strip() or "?"
    except OSError:
        return "?"


@contextlib.contextmanager
def _index_lock(timeout: float | None = None, stale_s: float | None = None):
    """Exclusive lock around load->modify->write of `library.json`.

    ⛔ MEASURED 2026-08-23: with no lock, a concurrent add of `1707.06347`
    landed between a path-repair's write and its verify and **overwrote it**.
    `O_EXCL` is the atomic test-and-set that works across processes AND across
    the two machines that share this Drive.

    A lock file left by a crashed run is DEBRIS, not contention — the same shape
    as the stale `.git/index.lock` rule in CLAUDE.md — so one older than
    `stale_s` is broken rather than waited on."""
    timeout = LOCK_TIMEOUT_S if timeout is None else timeout
    stale_s = LOCK_STALE_S if stale_s is None else stale_s
    lock = _lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = (f"pid={os.getpid()} host={platform.node()} "
               f"t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n").encode("utf-8")
    deadline = time.monotonic() + timeout
    delay, fd, last_err = 0.05, None, None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            last_err = None                       # plain contention, not a fault
        except OSError as e:
            # ⚠️ A Drive-FS flap (Errno 22) is indistinguishable from contention
            # at this call, so keep retrying to the deadline rather than raising
            # a false hard failure — but carry the reason into the timeout
            # message, or a broken path reads as "another kb_add is running".
            last_err = e
        age = _lock_age(lock)
        if age is not None and age > stale_s:
            print(f"kb_add: breaking a stale lock ({age:.0f}s old > "
                  f"{stale_s:.0f}s), holder was: {_lock_holder(lock)}",
                  file=sys.stderr)
            with contextlib.suppress(OSError):
                lock.unlink()
            delay = 0.05
        if time.monotonic() >= deadline:
            why = (f" (last error: {type(last_err).__name__}: {last_err})"
                   if last_err is not None else "")
            raise SystemExit(
                f"kb_add: could not take {lock} within {timeout:.0f}s — another "
                f"kb_add is running (holder: {_lock_holder(lock)}){why}. If that "
                f"process is gone the file is debris: delete it and retry.")
        time.sleep(delay)
        delay = min(delay * 1.6, 2.0)
    try:
        with contextlib.suppress(OSError):
            os.write(fd, payload)
        os.close(fd)
        fd = None
        yield lock
    finally:
        if fd is not None:                                   # pragma: no cover
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            lock.unlink()


def _load() -> dict:
    if INDEX_JSON.exists():
        return json.loads(_read_text_resilient(INDEX_JSON, what="library.json"))
    return {"schema": 1, "entries": {}}


def _save(db: dict) -> None:
    """Atomic write + read-back. ⚠️ On this mount an exit code is not evidence
    that anything landed (MEASURED: `kb_add` exited 0, printed nothing and banked
    nothing while the mount was mid-sync), so the write is confirmed by reading
    it back and re-parsing it."""
    LIB.mkdir(parents=True, exist_ok=True)
    want = len(db.get("entries", {}))
    _write_atomic(INDEX_JSON, json.dumps(db, indent=1, ensure_ascii=False) + "\n")
    try:
        back = json.loads(_read_text_resilient(
            INDEX_JSON, what="library.json (write read-back)"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"kb_add: library.json did not read back as JSON "
                         f"after the write: {e}") from None
    got = len(back.get("entries", {}))
    if got != want:
        raise SystemExit(f"kb_add: library.json read back with {got} entries, "
                         f"expected {want} — the write did not land")


def _update(mutate):
    """load -> mutate -> save, all inside the lock.

    ⛔ The load is INSIDE the lock on purpose. A lock that guards only the write
    still loses the edit: that is exactly how the `1707.06347` clobber happened,
    with the loser holding an index it had read before the winner's write."""
    with _index_lock():
        db = _load()
        out = mutate(db)
        _save(db)
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------- download verification --
def pdf_defect(path: Path, *, expected_bytes: int | None = None,
               min_bytes: int = PDF_MIN_BYTES) -> str | None:
    """Return a human-readable defect, or None if the bytes are a whole PDF.

    ⛔ Verify by CONTENT, never by presence. The 2026-08-23 truncation was a
    correctly-named 5,242,880 B file with no `%%EOF` — every presence check
    downstream would have called it banked.

    MEASURED 2026-08-23: all 67 papers already in the Library pass both the
    `%PDF` header check and the strict trailer check (nothing but whitespace or
    NUL after the last `%%EOF` in the final 4 KiB), so this validator rejects
    nothing we hold."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return f"cannot stat the download: {type(e).__name__}: {e}"
    if expected_bytes is not None and size != expected_bytes:
        return (f"TRUNCATED: {size:,} B on disk but Content-Length said "
                f"{expected_bytes:,} B (short by {expected_bytes - size:,} B)")
    if size < min_bytes:
        return (f"only {size:,} B — too small to be a paper "
                f"(floor {min_bytes:,} B); this is usually an error page")
    try:
        with path.open("rb") as f:
            head = f.read(5)
            f.seek(max(0, size - PDF_TAIL_SCAN))
            tail = f.read()
    except OSError as e:
        return f"cannot read the download: {type(e).__name__}: {e}"
    if not head.startswith(b"%PDF"):
        return f"not a PDF: starts with {head!r}, not b'%PDF'"
    i = tail.rfind(b"%%EOF")
    if i == -1:
        return (f"TRUNCATED: no %%EOF trailer in the last "
                f"{min(size, PDF_TAIL_SCAN):,} B of {size:,} B")
    rest = tail[i + 5:].strip(_PDF_TRAILING)
    if rest:
        return f"{len(rest):,} B of junk after the %%EOF trailer: {rest[:48]!r}"
    return None


def download_verified(url: str, dest: Path, *, timeout: int = 180,
                      min_bytes: int = PDF_MIN_BYTES) -> int:
    """Fetch `url` to `dest` atomically, or leave nothing behind and exit non-zero.

    ⛔ NEVER open `dest` for writing. The bytes go to a temp file with the
    reserved `TMP_PREFIX` in the SAME directory (so `os.replace` stays atomic),
    are verified, and only then replace the destination. A reset mid-stream —
    MEASURED: `ConnectionResetError` WinError 10054 from arXiv — therefore
    cannot leave a correctly-named truncated PDF where a presence check will
    find it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=TMP_PREFIX,
                                    suffix=".part")
    tmp = Path(tmp_name)
    expected: int | None = None
    try:
        with os.fdopen(fd, "wb") as f:
            req = urllib.request.Request(
                url, headers={"User-Agent": "TanitAD-KB/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = None
                with contextlib.suppress(Exception):         # noqa: BLE001
                    raw = r.headers.get("Content-Length")
                if raw is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        expected = int(str(raw).strip())
                shutil.copyfileobj(r, f)
                f.flush()
                with contextlib.suppress(OSError):
                    os.fsync(f.fileno())
    except BaseException as e:                               # noqa: BLE001
        _discard(tmp)
        raise SystemExit(
            f"kb_add: download failed for {url}: {type(e).__name__}: {e} "
            f"— nothing written, temp removed") from None

    defect = pdf_defect(tmp, expected_bytes=expected, min_bytes=min_bytes)
    if defect:
        _discard(tmp)
        raise SystemExit(f"kb_add: refusing {dest.name}: {defect} "
                         f"— nothing written, temp removed")
    size = tmp.stat().st_size
    try:
        os.replace(tmp, dest)
    except OSError as e:
        _discard(tmp)
        raise SystemExit(f"kb_add: could not put the download in place at "
                         f"{dest}: {e} — nothing written, temp removed") from None
    return size


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

    # Lock-free fast path: only DECIDING whether we need the network. The real
    # read that the write depends on happens inside the lock, below.
    e0 = _load()["entries"].get(aid)
    if e0 and Path(e0.get("path", "")).name and (ROOT / e0["path"]).exists():
        def _merge(db: dict) -> bool:
            e = db["entries"].get(aid)
            if e is None:            # a concurrent run removed it — go download
                return False
            e["tags"] = sorted(set(e.get("tags", []) + tags))
            e["cited_by"] = sorted(set(e.get("cited_by", []) + cited_by))
            if note and note not in e.get("note", ""):
                e["note"] = (e.get("note", "") + " | " + note).strip(" |")
            return True
        if _update(_merge):
            print(f"already banked: {aid} — tags/citations updated")
            return aid

    _ssl()
    meta = arxiv_meta(aid)
    PAPERS.mkdir(parents=True, exist_ok=True)
    name = f"{aid}_{_slug(meta.get('title') or 'untitled')}.pdf"
    dest = PAPERS / name
    size = download_verified(f"https://arxiv.org/pdf/{aid}", dest)

    entry = {
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
    }

    def _insert(db: dict) -> int:
        db["entries"][aid] = entry
        return len(db["entries"])
    n = _update(_insert)
    print(f"banked {aid}: {size//1024} KB — {meta.get('title','')[:70]} "
          f"[index: {n} entries]")
    return aid


def add_local(path: Path, *, key: str, title: str, tags: list[str],
              note: str, cited_by: list[str], url: str = "") -> str:
    src = (ROOT / path) if not path.is_absolute() else path
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    PAPERS.mkdir(parents=True, exist_ok=True)
    dest = PAPERS / f"{key}_{_slug(title or src.stem)}{src.suffix}"
    if src.resolve() != dest.resolve():
        # Same temp-then-replace discipline as the download: a copy that dies
        # half-way must not leave a correctly-named partial file behind.
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=TMP_PREFIX,
                                        suffix=src.suffix)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copyfile(src, tmp)
        except BaseException as e:                           # noqa: BLE001
            _discard(tmp)
            raise SystemExit(f"kb_add: could not copy {src}: "
                             f"{type(e).__name__}: {e} — nothing written") from None
        if src.suffix.lower() == ".pdf":
            defect = pdf_defect(tmp)
            if defect:
                _discard(tmp)
                raise SystemExit(f"kb_add: refusing {dest.name}: {defect} "
                                 f"— nothing written, temp removed")
        try:
            os.replace(tmp, dest)
        except OSError as e:
            _discard(tmp)
            raise SystemExit(f"kb_add: could not put {dest} in place: {e}") from None
        shutil.copystat(src, dest, follow_symlinks=True)

    entry = {
        "key": key, "kind": "local", "title": title or src.stem,
        "authors": [], "published": "", "abstract": "", "url": url,
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "bytes": dest.stat().st_size, "sha256": _sha256(dest),
        "banked": date.today().isoformat(),
        "tags": sorted(set(tags)), "note": note,
        "cited_by": sorted(set(cited_by)),
    }

    def _insert(db: dict) -> None:
        db["entries"][key] = entry
    _update(_insert)
    print(f"banked {key}: {dest.stat().st_size//1024} KB — {title or src.stem}")
    return key


def reindex() -> None:
    """Regenerate LIBRARY.md from library.json.

    Under the lock, and written atomically: two agents finishing an add at the
    same moment must not interleave a stale read with a fresh write and leave
    LIBRARY.md describing an index that no longer exists."""
    with _index_lock():
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
        _write_atomic(INDEX_MD, "\n".join(out))
    print(f"reindexed: {len(es)} entries, {tot/1e6:.1f} MB -> {INDEX_MD}")


def verify() -> int:
    """⛔ Verify by CONTENT, never by presence — the programme's standing rule.
    A file that exists but whose bytes changed is worse than a missing one.

    ORPHANS COUNT AS PROBLEMS. The 2026-08-23 truncation was a file on disk with
    a paper's name and NO index entry: a presence check ("is 2411.04983 banked?
    the file is right there") returns the wrong answer for it. Anything in
    `papers/` that could be mistaken for a banked paper must be in the index.
    Files carrying the reserved `TMP_PREFIX` cannot be mistaken for one, so they
    are reported as removable debris and not counted."""
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
    indexed = {Path(e["path"]).name for e in db["entries"].values()}
    if PAPERS.is_dir():
        for p in sorted(PAPERS.iterdir()):
            if not p.is_file():
                continue
            if p.name.startswith(TMP_PREFIX):
                print(f"DEBRIS   {p.name}: an interrupted kb_add — safe to delete")
                continue
            if p.name not in indexed:
                print(f"ORPHAN   {p.name}: {p.stat().st_size:,} B on disk but "
                      f"NOT in library.json — presence checks will lie about it")
                bad += 1
    print(f"verified {len(db['entries'])} entries, {bad} problem(s)")
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
    a = ap.parse_args(argv)

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
