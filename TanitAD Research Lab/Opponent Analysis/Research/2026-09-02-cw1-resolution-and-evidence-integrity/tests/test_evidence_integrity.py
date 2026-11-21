"""Pin E-OPP-CW1-1's integrity finding AND guard the repaired primary.

⭐ The second test is the one that earns its place: it re-reads the banked
DriveFuture PDF by CONTENT. `kb_add.py --verify` cannot catch a truncation (it
re-hashes the hash it recorded at bank time), so this is the check that would
notice if the file regressed.

    pytest -q test_evidence_integrity.py
"""
import io
import json
import os

import pytest

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw", "library_integrity.json")
REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
CW1_PDF = os.path.join(
    REPO, "TanitAD Research Lab", "Library", "papers",
    "2605.09701_DriveFuture-Future-Aware-Latent-World-Models-for-Autonomous.pdf")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(RAW):
        pytest.skip(f"artifact not present: {RAW}")
    return json.load(io.open(RAW, encoding="utf-8"))


def test_the_pre_repair_survey_found_exactly_one_corrupt_primary(r):
    """The measured finding, kept as measured — this artifact is PRE-repair."""
    assert r["tally"].get("OK") == 309
    assert r["tally"].get("TRUNCATED_NO_EOF") == 1
    assert list(r["unreadable"]) == ["2605.09701"]


def test_the_survey_is_not_quotable_as_library_rot(r):
    """⚠️ 309/310 is a 0.3 % rate. Pin it so the finding cannot drift upward."""
    total = r["n_entries_checked"]
    assert total == 310
    assert len(r["unreadable"]) / total < 0.01


@pytest.mark.skipif(not os.path.exists(CW1_PDF), reason="Library not mounted")
def test_the_repaired_primary_is_a_WHOLE_document():
    """⛔ CONTENT, not presence — the check `--verify` structurally cannot make."""
    size = os.path.getsize(CW1_PDF)
    assert size > 9_000_000, f"{size} B — the truncated copy was 1,373,148 B"
    with open(CW1_PDF, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, size - 4096))
        tail = fh.read(4096)
    assert head.startswith(b"%PDF")
    assert b"%%EOF" in tail, "no EOF marker — the file is truncated again"


@pytest.mark.skipif(not os.path.exists(CW1_PDF), reason="Library not mounted")
def test_the_reconciling_sentence_CW1_needed_is_actually_in_the_file():
    """The resolution must be re-derivable from the artifact, not from RESULT.md."""
    pypdf = pytest.importorskip("pypdf")
    r = pypdf.PdfReader(CW1_PDF)
    assert len(r.pages) >= 20
    text = "\n".join((p.extract_text() or "") for p in r.pages)
    flat = " ".join(text.split())
    assert "without GTRS-Dense scorer" in flat, "the CW-1 resolution is missing"
    assert "55.5" in flat and "34.6" in flat and "30.9" in flat
