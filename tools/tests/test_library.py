"""The research-banking rule, pinned so it cannot decay into prose.

C126 measured the decay mechanism directly: a correction that lived only in a
report was re-counted by every later census, because **no prose correction can
reach a glob**. The banking rule (CLAUDE.md, *Research banking*) is therefore
enforced here — the index must be generated, the hashes must match the bytes,
and the arXiv metadata parser must not re-acquire the bug that filed eleven
papers under the query echo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import kb_add as kb  # noqa: E402

# post-migration (2026-08-27): the library lives under "TanitAD Research
# Lab"; derive from the resolver instead of hardcoding either era's name.
LIB = kb.PAPERS.parent


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


def _research_dir() -> Path:
    """The research tree, under either side of the 2026-08-22 rename."""
    for name in ("TanitAD Research Lab", "TanitAD Research Hub"):
        if (ROOT / name).is_dir():
            return ROOT / name
    raise AssertionError("no research tree found under either name")


def test_the_findings_layer_still_exists_and_is_separate():
    """The Library is the EVIDENCE layer; the seven KNOWLEDGE_BASE.md files are
    the FINDINGS layer. Collapsing them would lose the curation."""
    kbs = list(_research_dir().glob("*/Research/KNOWLEDGE_BASE.md"))
    assert len(kbs) >= 6, f"expected the per-area knowledge bases, found {len(kbs)}"


# ------------------------------------------- the rename must not fork the lib ---
def test_lib_resolves_to_the_dir_that_actually_holds_the_library(tmp_path):
    """Renaming the research tree must NOT fork the library into a fresh empty
    dir. Resolution is by CONTENT (library.json), so a stale checkout and a
    renamed one bank into the same place.

    The defect this pins: kb_add hardcoded "TanitAD Research Hub" while the tree
    had been renamed to "TanitAD Research Lab" — banking would have silently
    created a second, empty library and every later --verify would have read the
    wrong one.
    """
    for holder, other in (("TanitAD Research Lab", "TanitAD Research Hub"),
                          ("TanitAD Research Hub", "TanitAD Research Lab")):
        root = tmp_path / holder.replace(" ", "_")
        (root / holder / "Library").mkdir(parents=True)
        (root / holder / "Library" / "library.json").write_text("{}", encoding="utf-8")
        (root / other).mkdir(parents=True)          # the decoy: exists, but empty
        assert kb._resolve_lib(root) == root / holder / "Library", \
            f"resolved away from the dir holding library.json ({holder})"


def test_lib_resolution_prefers_the_new_name_when_neither_holds_a_library(tmp_path):
    """Greenfield: both names may exist with no library yet — prefer the CURRENT
    name so a fresh bank lands in the renamed tree, not the deprecated one."""
    (tmp_path / "TanitAD Research Lab").mkdir(parents=True)
    (tmp_path / "TanitAD Research Hub").mkdir(parents=True)
    assert kb._resolve_lib(tmp_path) == tmp_path / "TanitAD Research Lab" / "Library"


def test_the_live_library_is_the_one_with_the_papers():
    """Guard against the fork existing RIGHT NOW in this checkout."""
    assert kb.INDEX_JSON.exists(), f"kb_add resolved LIB to {kb.LIB} — no library.json there"
    assert kb.LIB.parent.name in ("TanitAD Research Lab", "TanitAD Research Hub")


# ------------------------------------- the concurrent-write race (2026-08-23) ---
# MEASURED: two streams banking at once each read library.json, downloaded for up
# to 180 s, then wrote the WHOLE file back from their stale snapshot. 7 PDFs
# landed on disk with no index row, and --verify reported "0 problems" because it
# iterates entries — an entry never written cannot be found missing.

def test_commit_entry_does_not_clobber_a_concurrent_write(tmp_path, monkeypatch):
    """The regression arm for the race: an entry added AFTER our snapshot was taken
    must survive our commit. Before the fix this test loses `other`."""
    lib = tmp_path / "Library"
    (lib / "papers").mkdir(parents=True)
    idx = lib / "library.json"
    idx.write_text('{"schema": 1, "entries": {}}', encoding="utf-8")
    monkeypatch.setattr(kb, "LIB", lib)
    monkeypatch.setattr(kb, "INDEX_JSON", idx)
    monkeypatch.setattr(kb, "PAPERS", lib / "papers")

    kb._load()                                   # our (now stale) snapshot
    idx.write_text(json.dumps({"schema": 1, "entries": {"other": {"key": "other"}}}),
                   encoding="utf-8")             # a sibling stream commits
    kb._commit_entry("mine", {"key": "mine"})    # we commit afterwards

    entries = json.loads(idx.read_text(encoding="utf-8"))["entries"]
    assert "mine" in entries
    assert "other" in entries, "a concurrent entry was ERASED - the race is back"


def test_orphans_sees_a_pdf_the_index_never_recorded(tmp_path, monkeypatch):
    """--verify iterates entries, so absence must be probed from the DISK side."""
    lib = tmp_path / "Library"
    papers = lib / "papers"
    papers.mkdir(parents=True)
    (lib / "library.json").write_text('{"schema": 1, "entries": {}}', encoding="utf-8")
    (papers / "9999.00001_ghost.pdf").write_bytes(b"%PDF-1.4 ghost")
    monkeypatch.setattr(kb, "LIB", lib)
    monkeypatch.setattr(kb, "INDEX_JSON", lib / "library.json")
    monkeypatch.setattr(kb, "PAPERS", papers)

    found = [p.name for p in kb.orphans()]
    assert "9999.00001_ghost.pdf" in found, "the orphan scan is blind - the race stays invisible"


def test_orphans_is_quiet_when_every_pdf_is_indexed(tmp_path, monkeypatch):
    """The regression arm for the arm above: it must not cry orphan on a clean lib."""
    lib = tmp_path / "Library"
    papers = lib / "papers"
    papers.mkdir(parents=True)
    pdf = papers / "1234.56789_ok.pdf"
    pdf.write_bytes(b"%PDF-1.4 ok")
    monkeypatch.setattr(kb, "LIB", lib)
    monkeypatch.setattr(kb, "INDEX_JSON", lib / "library.json")
    monkeypatch.setattr(kb, "PAPERS", papers)
    monkeypatch.setattr(kb, "ROOT", tmp_path)
    (lib / "library.json").write_text(json.dumps(
        {"schema": 1, "entries": {"1234.56789": {
            "key": "1234.56789",
            "path": str(pdf.relative_to(tmp_path)).replace("\\", "/")}}}),
        encoding="utf-8")
    assert kb.orphans() == []


def test_index_lock_is_released_and_reentrant(tmp_path):
    """A lock left behind would block every later bank - the stale .git/index.lock
    shape. It must clear on exit."""
    target = tmp_path / "library.json"
    with kb._IndexLock(target):
        assert (tmp_path / "library.json.lock").exists()
    assert not (tmp_path / "library.json.lock").exists()
    with kb._IndexLock(target):
        pass


def test_the_live_library_has_no_orphans():
    """Guard against the race's damage existing RIGHT NOW in this checkout."""
    assert kb.orphans() == [],         f"unindexed banked PDFs present: {[p.name for p in kb.orphans()]}"
