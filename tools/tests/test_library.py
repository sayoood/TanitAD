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

LIB = ROOT / "TanitAD Research Hub" / "Library"


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
    kbs = list((ROOT / "TanitAD Research Hub").glob("*/Research/KNOWLEDGE_BASE.md"))
    assert len(kbs) >= 6, f"expected the per-area knowledge bases, found {len(kbs)}"
