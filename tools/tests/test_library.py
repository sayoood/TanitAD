"""The research-banking rule, pinned so it cannot decay into prose.

C126 measured the decay mechanism directly: a correction that lived only in a
report was re-counted by every later census, because **no prose correction can
reach a glob**. The banking rule (CLAUDE.md, *Research banking*) is therefore
enforced here — the index must be generated, the hashes must match the bytes,
and the arXiv metadata parser must not re-acquire the bug that filed eleven
papers under the query echo.

The second half of this file pins the two defects MEASURED on 2026-08-23
(Research Lab run 001, "Process hazards measured today" §3 and §4): a truncated
download that survived on disk without an index entry, and an unlocked
read-modify-write of `library.json` that let two agents clobber each other.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import kb_add as kb  # noqa: E402

LIB = ROOT / "TanitAD Research Lab" / "Library"


def _db() -> dict:
    assert kb.INDEX_JSON.exists(), "library.json is missing — the Library is the evidence layer"
    return json.loads(kb.INDEX_JSON.read_text(encoding="utf-8"))


# ------------------------------------------------------------- the library --
def test_the_library_exists_and_is_not_empty():
    assert LIB.is_dir() and kb.PAPERS.is_dir()
    assert len(_db()["entries"]) >= 14


def test_every_entry_has_the_fields_a_citation_needs():
    for k, e in _db()["entries"].items():
        for f in ("key", "title", "path", "bytes", "sha256", "banked", "tags"):
            assert e.get(f) not in (None, ""), f"{k} missing {f}"
        assert len(e["sha256"]) == 64, k
        assert e["bytes"] > 20_000, f"{k}: {e['bytes']} B is not a paper"


def test_every_banked_file_is_present_and_hashes_to_its_record():
    """⛔ Content, never presence. `--verify` is the same check the CLI runs."""
    assert kb.verify() == 0


def test_the_generated_index_is_in_sync_with_the_json():
    md = kb.INDEX_MD.read_text(encoding="utf-8")
    for k, e in _db()["entries"].items():
        assert k in md, f"{k} banked but absent from LIBRARY.md — run --reindex"
        assert e["sha256"][:12] in md, f"{k}: LIBRARY.md carries a stale hash"


def test_the_index_declares_it_is_generated_so_nobody_hand_edits_it():
    md = kb.INDEX_MD.read_text(encoding="utf-8")
    assert "reindex" in md and "never hand-edit" in md.lower()


# ------------------------------------------- the bug that shipped once, once --
def test_arxiv_metadata_is_scoped_to_the_entry_not_the_feed():
    """MEASURED 2026-08-18: the first migration filed all 11 papers under
    'arXiv Query: search_query=...' — the FEED-level <title> — and reported
    success. The parser must read inside <entry>."""
    src = (ROOT / "tools" / "kb_add.py").read_text(encoding="utf-8")
    assert "<entry>(.*?)</entry>" in src, "arxiv_meta must scope to <entry>"


def test_no_banked_title_is_the_arxiv_query_echo():
    """The regression guard on the DATA, not just the code."""
    for k, e in _db()["entries"].items():
        assert "arXiv Query" not in e["title"], f"{k} carries the feed echo"
        assert "search_query" not in e["title"], k


def test_arxiv_metadata_parses_a_realistic_feed_offline():
    """No network: the exact shape that broke it, as a fixture."""
    feed = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <title>arXiv Query: search_query=&amp;id_list=1234.5678&amp;start=0</title>
      <entry><id>http://arxiv.org/abs/1234.5678v1</id>
        <published>2024-01-02T00:00:00Z</published>
        <title>The Real Paper Title</title>
        <summary>An abstract.</summary>
        <author><name>A. Person</name></author>
      </entry></feed>"""
    import re
    em = re.search(r"<entry>(.*?)</entry>", feed, re.S)
    ent = em.group(1)
    title = re.sub(r"\s+", " ",
                   re.search(r"<title>(.*?)</title>", ent, re.S).group(1)).strip()
    assert title == "The Real Paper Title"


# --------------------------------------------------------------- the rule ---
@pytest.mark.parametrize("doc,token", [
    ("CLAUDE.md", "RESEARCH BANKING"),
    ("CLAUDE.md", "kb_add.py"),
    ("Project Steering/AGENT_OPERATING_STANDARD.md", "Research banking"),
    ("Project Steering/AGENT_OPERATING_STANDARD.md", "kb_add.py"),
])
def test_the_rule_is_stated_where_an_agent_will_read_it(doc, token):
    assert token in (ROOT / doc).read_text(encoding="utf-8"), \
        f"{doc} must carry {token!r} — an unstated rule is not a rule"


def test_the_rule_names_the_secondary_source_consequence():
    """The rule must say WHY, or it reads as bureaucracy and gets skipped."""
    t = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "PUBLISHED-SECONDARY" in t
    assert "inadmissible" in t.lower()


def test_the_findings_layer_still_exists_and_is_separate():
    """The Library is the EVIDENCE layer; the seven KNOWLEDGE_BASE.md files are
    the FINDINGS layer. Collapsing them would lose the curation."""
    kbs = list((ROOT / "TanitAD Research Lab").glob("*/Research/KNOWLEDGE_BASE.md"))
    assert len(kbs) >= 6, f"expected the per-area knowledge bases, found {len(kbs)}"


# ============================================================================
# HARDENING — the two defects MEASURED 2026-08-23 (Research Lab run 001).
#
#   §3  a mid-download ConnectionResetError (WinError 10054) left a correctly
#       NAMED but truncated PDF (exactly 5,242,880 B, no %%EOF) that was never
#       recorded in library.json, and a `| tail` pipe masked the non-zero exit.
#   §4  library.json had no lock, so a concurrent add of 1707.06347 clobbered a
#       path-repair edit that had already been written.
#
# These tests never touch the real Library: `sandbox` retargets the module's
# path constants at a tmp_path, and the lock follows the index by construction.
# ============================================================================
PDF_HEAD = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
PDF_TAIL = b"\ntrailer\n<< /Size 1 >>\nstartxref\n9\n%%EOF\n"


def _pdf(n_bytes: int = 40_000, *, trailer: bool = True) -> bytes:
    """A byte string that passes (or, with trailer=False, fails) the validator
    for the same reason a real PDF does: header, bulk, and an %%EOF terminator."""
    tail = PDF_TAIL if trailer else b"\n% cut here, no terminator\n"
    pad = max(0, n_bytes - len(PDF_HEAD) - len(tail))
    unit = b"%" + b"p" * 78 + b"\n"                      # exactly 80 bytes
    return PDF_HEAD + unit * (pad // 80) + b"q" * (pad % 80) + tail


class _FakeHTTP:
    """A urlopen() stand-in. `raise_after` reproduces the measured failure: the
    connection dies mid-body, after some bytes have already been written."""

    def __init__(self, payload: bytes, *, content_length: int | None = None,
                 raise_after: int | None = None):
        self._buf = payload
        self._sent = 0
        self._raise_after = raise_after
        self.headers = ({} if content_length is None
                        else {"Content-Length": str(content_length)})

    def read(self, n: int = -1) -> bytes:
        if not self._buf:
            return b""
        chunk = self._buf if n in (-1, None) else self._buf[:n]
        if (self._raise_after is not None
                and self._sent + len(chunk) > self._raise_after):
            keep = max(0, self._raise_after - self._sent)
            if keep:
                self._sent += keep
                self._buf = self._buf[keep:]
                return chunk[:keep]
            raise ConnectionResetError(
                10054, "An existing connection was forcibly closed by the "
                       "remote host")
        self._buf = b"" if n in (-1, None) else self._buf[n:]
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Retarget kb_add at a throwaway Library. ⛔ Never let these tests write to
    the real one — they are about clobbering, and that is what they would do."""
    lib = tmp_path / "TanitAD Research Lab" / "Library"
    papers = lib / "papers"
    papers.mkdir(parents=True)
    monkeypatch.setattr(kb, "ROOT", tmp_path)
    monkeypatch.setattr(kb, "LIB", lib)
    monkeypatch.setattr(kb, "PAPERS", papers)
    monkeypatch.setattr(kb, "INDEX_JSON", lib / "library.json")
    monkeypatch.setattr(kb, "INDEX_MD", lib / "LIBRARY.md")
    monkeypatch.setattr(kb, "arxiv_meta", lambda aid: {
        "title": "A Fake Paper", "abstract": "", "authors": ["A. Person"],
        "published": "2026-01-01"})
    return lib


def _serve(monkeypatch, response_factory):
    monkeypatch.setattr(kb.urllib.request, "urlopen",
                        lambda *a, **k: response_factory())


# -------------------------------------------------- §3 truncated downloads ---
def test_a_reset_mid_download_leaves_nothing_behind_and_exits_non_zero(
        sandbox, monkeypatch):
    """THE MEASURED DEFECT. WinError 10054 part-way through the body must not
    leave a correctly-named PDF — not the paper, and not a temp file either."""
    body = _pdf(60_000)
    _serve(monkeypatch, lambda: _FakeHTTP(body, content_length=len(body),
                                          raise_after=12_000))
    with pytest.raises(SystemExit) as ei:
        kb.add_arxiv("1234.5678", tags=[], note="", cited_by=[])

    assert ei.value.code not in (0, None), "a failed bank must exit non-zero"
    assert list(kb.PAPERS.iterdir()) == [], \
        f"left behind: {[p.name for p in kb.PAPERS.iterdir()]}"
    assert not kb.INDEX_JSON.exists(), "nothing should have been indexed"


def test_a_short_body_against_content_length_is_refused(sandbox, monkeypatch):
    """The clean-EOF case: the socket closes tidily but the body is short. The
    Content-Length check is the one that catches this."""
    full = _pdf(80_000)
    _serve(monkeypatch, lambda: _FakeHTTP(full[:30_000],
                                          content_length=len(full)))
    with pytest.raises(SystemExit) as ei:
        kb.add_arxiv("1234.5678", tags=[], note="", cited_by=[])
    assert "TRUNCATED" in str(ei.value.code)
    assert list(kb.PAPERS.iterdir()) == []


def test_a_body_without_the_eof_trailer_is_refused_when_no_length_header(
        sandbox, monkeypatch):
    """No Content-Length (chunked transfer) — then the %%EOF trailer is the ONLY
    thing standing between us and a banked half-paper."""
    _serve(monkeypatch, lambda: _FakeHTTP(_pdf(60_000, trailer=False)))
    with pytest.raises(SystemExit) as ei:
        kb.add_arxiv("1234.5678", tags=[], note="", cited_by=[])
    assert "%%EOF" in str(ei.value.code)
    assert list(kb.PAPERS.iterdir()) == []


def test_the_exact_5mib_signature_from_run_001_is_refused(sandbox, monkeypatch):
    """The incident's fingerprint, byte for byte: 5,242,880 B, no trailer, no
    length header. It was accepted once; it must never be accepted again."""
    body = _pdf(5_242_880, trailer=False)
    assert len(body) == 5_242_880
    _serve(monkeypatch, lambda: _FakeHTTP(body))
    with pytest.raises(SystemExit):
        kb.add_arxiv("1234.5678", tags=[], note="", cited_by=[])
    assert list(kb.PAPERS.iterdir()) == []


def test_an_error_page_instead_of_a_pdf_is_refused(sandbox, monkeypatch):
    _serve(monkeypatch, lambda: _FakeHTTP(b"<html>403 Forbidden</html>"))
    with pytest.raises(SystemExit) as ei:
        kb.add_arxiv("1234.5678", tags=[], note="", cited_by=[])
    assert "too small" in str(ei.value.code) or "not a PDF" in str(ei.value.code)
    assert list(kb.PAPERS.iterdir()) == []


def test_a_good_download_lands_with_its_trailer_and_is_indexed(
        sandbox, monkeypatch):
    """The positive control — without it, "nothing was written" proves nothing."""
    body = _pdf(60_000)
    _serve(monkeypatch, lambda: _FakeHTTP(body, content_length=len(body)))
    kb.add_arxiv("1234.5678", tags=["t"], note="n", cited_by=["report.md"])

    files = [p for p in kb.PAPERS.iterdir()]
    assert len(files) == 1, [p.name for p in files]
    got = files[0]
    assert got.name.startswith("1234.5678_"), got.name
    assert not got.name.startswith(kb.TMP_PREFIX), "a temp name was published"
    raw = got.read_bytes()
    assert raw[:4] == b"%PDF" and raw.rstrip().endswith(b"%%EOF")
    assert raw == body, "the banked bytes are not the bytes we served"

    e = json.loads(kb.INDEX_JSON.read_text(encoding="utf-8"))["entries"]["1234.5678"]
    assert e["bytes"] == len(body) == got.stat().st_size
    assert e["sha256"] == kb._sha256(got)
    assert Path(e["path"]).name == got.name
    assert kb.pdf_defect(got) is None


def test_a_local_pdf_is_banked_through_the_same_temp_then_replace_path(
        sandbox, tmp_path):
    """`--local` copies rather than downloads, but a copy that dies half-way
    leaves the same correctly-named partial file, so it gets the same
    discipline: temp, validate, replace."""
    src = tmp_path / "hand-filed.pdf"
    src.write_bytes(_pdf(40_000))
    kb.add_local(src, key="unece-r157", title="UNECE R157", tags=["reg"],
                 note="", cited_by=[])

    files = list(kb.PAPERS.iterdir())
    assert [p.name for p in files] == ["unece-r157_UNECE-R157.pdf"]
    assert kb.pdf_defect(files[0]) is None
    e = json.loads(kb.INDEX_JSON.read_text(encoding="utf-8"))["entries"]["unece-r157"]
    assert e["sha256"] == kb._sha256(files[0]) and e["bytes"] == 40_000


def test_a_truncated_local_pdf_is_refused_and_leaves_nothing(sandbox, tmp_path):
    src = tmp_path / "half.pdf"
    src.write_bytes(_pdf(40_000, trailer=False))
    with pytest.raises(SystemExit) as ei:
        kb.add_local(src, key="half", title="Half A Paper", tags=[], note="",
                     cited_by=[])
    assert "%%EOF" in str(ei.value.code)
    assert list(kb.PAPERS.iterdir()) == []
    assert src.exists(), "the SOURCE must never be touched"


def test_a_local_non_pdf_is_not_put_through_the_pdf_validator(sandbox, tmp_path):
    """The Library also holds regulatory text and notes; only PDFs get the
    PDF check, or banking a .md would fail for having no %%EOF."""
    src = tmp_path / "notes.md"
    src.write_text("# a note that is not a PDF\n", encoding="utf-8")
    kb.add_local(src, key="note-1", title="A Note", tags=[], note="",
                 cited_by=[])
    assert (kb.PAPERS / "note-1_A-Note.md").is_file()


def test_the_destination_is_never_opened_for_writing(sandbox):
    """A source guard, because the defect returns the moment someone
    "simplifies" the temp file away. Same shape as the <entry> guard above."""
    src = (ROOT / "tools" / "kb_add.py").read_text(encoding="utf-8")
    assert "os.replace" in src, "the download must be published atomically"
    assert 'dest.open("wb")' not in src, "never stream into the destination"
    assert "TMP_PREFIX" in src


def test_a_truncated_download_exits_non_zero_at_the_process_level(tmp_path):
    """⛔ The `| tail` pipe in run 001 hid the failure, so the exit CODE — not an
    in-process exception — is what has to be right. This runs the real CLI."""
    lib = tmp_path / "TanitAD Research Lab" / "Library"
    (lib / "papers").mkdir(parents=True)
    driver = tmp_path / "drive_kb_add.py"
    driver.write_text(
        "import sys, urllib.request\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, __TOOLS__)\n"
        "import kb_add as kb\n"
        "root = Path(__ROOT__)\n"
        "kb.ROOT = root\n"
        "kb.LIB = root / 'TanitAD Research Lab' / 'Library'\n"
        "kb.PAPERS = kb.LIB / 'papers'\n"
        "kb.INDEX_JSON = kb.LIB / 'library.json'\n"
        "kb.INDEX_MD = kb.LIB / 'LIBRARY.md'\n"
        "kb.arxiv_meta = lambda aid: {'title': 'Fake', 'abstract': '',\n"
        "                             'authors': [], 'published': '2026-01-01'}\n"
        "PAYLOAD = b'%PDF-1.4' + b'x' * 60000        # no %%EOF trailer\n"
        "class R:\n"
        "    headers = {}\n"
        "    def __init__(self): self.b = PAYLOAD\n"
        "    def read(self, n=-1):\n"
        "        c = self.b if n in (-1, None) else self.b[:n]\n"
        "        self.b = b'' if n in (-1, None) else self.b[n:]\n"
        "        return c\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "urllib.request.urlopen = lambda *a, **k: R()\n"
        "sys.exit(kb.main(['1234.5678']))\n"
        .replace("__TOOLS__", repr(str(ROOT / "tools")))
        .replace("__ROOT__", repr(str(tmp_path))),
        encoding="utf-8")

    p = subprocess.run([sys.executable, str(driver)], capture_output=True,
                       text=True, encoding="utf-8", timeout=120,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert p.returncode != 0, f"exit 0 on a truncated download\n{p.stdout}{p.stderr}"
    assert "refusing" in (p.stdout + p.stderr)
    assert list((lib / "papers").iterdir()) == []
    assert not (lib / "library.json").exists()


# ------------------------------------------------------------ §4 the lock ---
def test_the_lock_sits_next_to_the_index_it_guards(sandbox):
    assert kb._lock_path() == sandbox / "library.json.lock"


def test_two_interleaved_load_modify_write_cycles_both_survive(sandbox):
    """THE MEASURED DEFECT. Two agents, each doing load->modify->write with the
    window held open. Unlocked (or locked only around the WRITE) the second
    load precedes the first write and one entry disappears — that is exactly how
    the concurrent add of 1707.06347 erased the path repair."""
    kb._save({"schema": 1, "entries": {}})
    ready = threading.Barrier(2, timeout=10)
    errors: list[BaseException] = []

    def worker(key: str):
        def mutate(db):
            db["entries"][key] = {"key": key, "bytes": 1}
            time.sleep(0.30)          # hold the read-modify-write window open
        try:
            ready.wait()
            kb._update(mutate)
        except BaseException as e:                           # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in ("AAA", "BBB")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
        assert not t.is_alive(), "a writer never finished — deadlocked lock?"
    assert not errors, errors

    entries = json.loads(kb.INDEX_JSON.read_text(encoding="utf-8"))["entries"]
    assert set(entries) == {"AAA", "BBB"}, \
        f"a concurrent writer was clobbered: {sorted(entries)}"
    assert not kb._lock_path().exists(), "the lock outlived its writers"


def test_the_lock_actually_excludes_and_is_released_afterwards(sandbox):
    with kb._index_lock(timeout=5.0):
        assert kb._lock_path().exists()
        t0 = time.monotonic()
        with pytest.raises(SystemExit) as ei:
            with kb._index_lock(timeout=0.5, stale_s=3600.0):
                pass                                         # pragma: no cover
        waited = time.monotonic() - t0
        assert waited >= 0.4, f"gave up after {waited:.2f}s — it did not wait"
        assert "could not take" in str(ei.value.code)
    assert not kb._lock_path().exists(), "the lock must be released on exit"


def test_a_lock_left_by_a_dead_run_is_broken_not_waited_on(sandbox):
    """Debris, not contention — the same rule as the stale .git/index.lock."""
    lock = kb._lock_path()
    lock.write_text("pid=999999 host=dead-pod t=1970-01-01T00:00:00\n",
                    encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    t0 = time.monotonic()
    with kb._index_lock(timeout=5.0, stale_s=60.0):
        pass
    assert time.monotonic() - t0 < 5.0, "it waited out a stale lock"
    assert not lock.exists()


def test_a_failed_write_never_leaves_a_half_written_index(sandbox):
    """`os.replace` means a reader sees the old index or the new one — never a
    torn one — and a crash mid-write leaves no temp behind."""
    kb._save({"schema": 1, "entries": {"AAA": {"key": "AAA"}}})
    before = kb.INDEX_JSON.read_text(encoding="utf-8")

    class Boom(RuntimeError):
        pass

    def explode(db):
        db["entries"]["BBB"] = {"key": "BBB"}
        raise Boom("write died half-way")

    with pytest.raises(Boom):
        kb._update(explode)

    assert kb.INDEX_JSON.read_text(encoding="utf-8") == before
    assert [p.name for p in sandbox.iterdir() if p.name.startswith(kb.TMP_PREFIX)] == []
    assert not kb._lock_path().exists(), "the lock must be released on error"


def test_an_index_write_that_does_not_land_is_caught_not_called_success(
        sandbox, monkeypatch):
    """⛔ An exit code is not evidence on this mount — run 001 §2 measured
    `kb_add` exiting 0, printing nothing and banking nothing while the Drive was
    mid-sync. _save must read back what it wrote and refuse to call a vanished
    write a success."""
    kb._save({"schema": 1, "entries": {"AAA": {"key": "AAA", "bytes": 1}}})
    back = json.loads(kb.INDEX_JSON.read_text(encoding="utf-8"))
    assert list(back["entries"]) == ["AAA"]

    monkeypatch.setattr(kb, "_write_atomic", lambda p, t: None)   # mount eats it
    with pytest.raises(SystemExit) as ei:
        kb._save({"schema": 1, "entries": {"AAA": {"key": "AAA"},
                                           "BBB": {"key": "BBB"}}})
    assert "did not land" in str(ei.value.code)


def test_a_read_that_flaps_with_errno_22_is_retried_not_believed(
        sandbox, monkeypatch):
    """The G: mount serves a just-written file as Errno 22 for ~a minute. One
    failed read is not evidence the index is gone."""
    kb._save({"schema": 1, "entries": {"AAA": {"key": "AAA"}}})
    real = Path.read_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        if self == kb.INDEX_JSON and calls["n"] < 2:
            calls["n"] += 1
            raise OSError(22, "Invalid request code")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert list(kb._load()["entries"]) == ["AAA"]
    assert calls["n"] == 2, "the retry never happened"


# ------------------------------------------------- the orphan the fix leaves --
def test_verify_reports_a_file_that_is_on_disk_but_not_in_the_index(sandbox):
    """The incident's leftover was a PDF with a paper's name and NO entry — a
    presence check calls that "banked". --verify must not."""
    kb._save({"schema": 1, "entries": {}})
    (kb.PAPERS / "1234.5678_Ghost.pdf").write_bytes(_pdf(30_000))
    assert kb.verify() == 1
    (kb.PAPERS / "1234.5678_Ghost.pdf").unlink()
    (kb.PAPERS / f"{kb.TMP_PREFIX}abc.part").write_bytes(b"x")
    assert kb.verify() == 0, "reserved-prefix debris cannot be mistaken for a paper"
