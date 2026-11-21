"""Tests for tools/media_verify.py and Media/MEDIA_MANIFEST.json.

The bug class these guard against is not "the code throws". It is
**a verifier that reports success while verifying nothing** — the same family
as the library tool that filed 11 papers under the query string and reported
success, and as the memmap that produced 2.76 GB of zeros and exited 0.

So the weight is on:

* a **DELIBERATE REGRESSION ARM** — corrupt a fixture's bytes and assert the
  verifier says ``MISMATCH``. A checker that cannot fail on known-bad input is
  not evidence of anything, and this is the only test here that proves the
  hash comparison is actually load-bearing.
* the **ORPHAN** probe, which is the half of the check that iterating manifest
  rows structurally cannot perform.
* the **MISSING vs UNREADABLE** boundary. A Drive-dehydrated file read as
  "gone" is a manufactured loss report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import media_verify as mv  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST = REPO_ROOT / "Media" / "MEDIA_MANIFEST.json"


# ---------------------------------------------------------------------------
# Fixture: a tiny self-consistent Media/ tree
# ---------------------------------------------------------------------------
def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return mv.sha256_file(path)[0]


@pytest.fixture()
def media_tree(tmp_path):
    """Build ``<tmp>/Media/`` with two campaigns and a matching manifest."""
    media = tmp_path / "Media"
    a = _write(media / "2026-01-01-alpha" / "one.mp4", b"alpha-one-payload")
    b = _write(media / "2026-01-02-beta" / "two.mp4", b"beta-two-payload-longer")
    manifest = {
        "schema_version": 1,
        "media_root": "Media",
        "totals": {"unique_assets": 2},
        "campaigns": {"2026-01-01-alpha": {"description": "a", "n_assets": 1},
                      "2026-01-02-beta": {"description": "b", "n_assets": 1}},
        "assets": [
            {"sha256": a, "bytes": len(b"alpha-one-payload"),
             "campaign": "2026-01-01-alpha", "filename": "one.mp4",
             "media_path": "Media/2026-01-01-alpha/one.mp4", "shows": "x",
             "aliases": [], "tracked_in_git": False, "sources": []},
            {"sha256": b, "bytes": len(b"beta-two-payload-longer"),
             "campaign": "2026-01-02-beta", "filename": "two.mp4",
             "media_path": "Media/2026-01-02-beta/two.mp4", "shows": "y",
             "aliases": [], "tracked_in_git": True, "sources": []},
        ],
    }
    path = media / "MEDIA_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return media, manifest


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_clean_tree_verifies_ok(media_tree):
    media, manifest = media_tree
    rows = mv.verify_rows(manifest, media)
    assert [r["status"] for r in rows] == [mv.OK, mv.OK]
    counts = mv.summarise(rows, mv.find_orphans(manifest, media))
    assert counts[mv.OK] == 2
    assert not mv.is_failure(counts)


# ---------------------------------------------------------------------------
# ⭐ DELIBERATE REGRESSION ARM
# ---------------------------------------------------------------------------
def test_corrupted_bytes_are_reported_as_mismatch(media_tree):
    """Same length, different bytes: only a content check can see this."""
    media, manifest = media_tree
    target = media / "2026-01-01-alpha" / "one.mp4"
    original = target.read_bytes()
    target.write_bytes(b"XXXXX-one-payload")          # identical length
    assert len(target.read_bytes()) == len(original)   # size check would pass

    rows = mv.verify_rows(manifest, media)
    bad = [r for r in rows if r["media_path"].endswith("one.mp4")]
    assert bad[0]["status"] == mv.MISMATCH, "verifier failed to detect corrupted content"
    assert "sha256" in bad[0]["reason"]
    counts = mv.summarise(rows, mv.find_orphans(manifest, media))
    assert counts[mv.MISMATCH] == 1
    assert mv.is_failure(counts), "a MISMATCH must gate non-zero"


def test_corrupted_manifest_hash_is_reported_as_mismatch(media_tree):
    """The regression arm from the other side: rot the RECORD, not the file."""
    media, manifest = media_tree
    manifest["assets"][1]["sha256"] = "0" * 64
    rows = mv.verify_rows(manifest, media)
    bad = [r for r in rows if r["media_path"].endswith("two.mp4")]
    assert bad[0]["status"] == mv.MISMATCH


def test_quick_mode_cannot_see_same_size_corruption(media_tree):
    """Documents the limit of --quick so nobody quotes it as a content check."""
    media, manifest = media_tree
    (media / "2026-01-01-alpha" / "one.mp4").write_bytes(b"XXXXX-one-payload")
    rows = mv.verify_rows(manifest, media, quick=True)
    assert all(r["status"] == mv.OK for r in rows)
    assert all(r.get("checked") == "size-only" for r in rows)


# ---------------------------------------------------------------------------
# ORPHAN: the probe from the disk side
# ---------------------------------------------------------------------------
def test_orphan_detected_for_unindexed_file(media_tree):
    media, manifest = media_tree
    (media / "2026-01-02-beta" / "stowaway.mp4").write_bytes(b"nobody indexed me")

    rows = mv.verify_rows(manifest, media)
    assert all(r["status"] == mv.OK for r in rows), (
        "iterating manifest rows must NOT see the orphan - that is the point")

    orphans = mv.find_orphans(manifest, media)
    assert orphans == ["Media/2026-01-02-beta/stowaway.mp4"]
    counts = mv.summarise(rows, orphans)
    assert counts[mv.ORPHAN] == 1
    assert mv.is_failure(counts), "an ORPHAN must gate non-zero"


def test_non_video_files_are_not_orphans(media_tree):
    media, manifest = media_tree
    (media / "README.md").write_text("notes", encoding="utf-8")
    (media / "2026-01-01-alpha" / "notes.txt").write_text("x", encoding="utf-8")
    assert mv.find_orphans(manifest, media) == []


# ---------------------------------------------------------------------------
# MISSING vs UNREADABLE — absence of a file vs absence of information
# ---------------------------------------------------------------------------
def test_deleted_file_is_missing(media_tree):
    media, manifest = media_tree
    (media / "2026-01-01-alpha" / "one.mp4").unlink()
    rows = mv.verify_rows(manifest, media)
    bad = [r for r in rows if r["media_path"].endswith("one.mp4")]
    assert bad[0]["status"] == mv.MISSING
    assert mv.is_failure(mv.summarise(rows, []))


def test_unreadable_file_is_not_reported_as_missing(media_tree, monkeypatch):
    """A Drive-mount outage must never be read as data loss."""
    media, manifest = media_tree

    def boom(path, retries=4, sleep=None):
        return None, "OSError:errno=22"

    monkeypatch.setattr(mv, "sha256_file", boom)
    rows = mv.verify_rows(manifest, media)
    assert all(r["status"] == mv.UNREADABLE for r in rows)
    counts = mv.summarise(rows, [])
    assert counts[mv.MISSING] == 0
    assert counts[mv.UNREADABLE] == 2
    assert not mv.is_failure(counts), "UNREADABLE alone must NOT gate - it is not a conclusion"


def test_sha256_file_retries_before_giving_up(tmp_path):
    calls = []
    missing = tmp_path / "nope.mp4"
    digest, reason = mv.sha256_file(missing, retries=3, sleep=calls.append)
    assert digest is None
    assert "errno" in reason
    assert len(calls) == 2, "should back off between attempts, not on the last one"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mutate, needle", [
    (lambda m: m["assets"][0].update(sha256=""), "malformed sha256"),
    (lambda m: m["assets"][0].update(sha256="z" * 64), "not hex"),
    (lambda m: m["assets"][0].update(campaign="  "), "empty campaign"),
    (lambda m: m["assets"][0].pop("media_path"), "missing required key"),
    (lambda m: m["assets"][0].update(bytes=-1), "malformed byte count"),
    (lambda m: m.pop("assets"), "missing required key"),
])
def test_invalid_manifests_are_rejected(media_tree, mutate, needle):
    _media, manifest = media_tree
    mutate(manifest)
    with pytest.raises(ValueError) as exc:
        mv.validate_manifest(manifest)
    assert needle in str(exc.value)


def test_duplicate_media_path_is_rejected(media_tree):
    _media, manifest = media_tree
    manifest["assets"][1]["media_path"] = manifest["assets"][0]["media_path"]
    with pytest.raises(ValueError) as exc:
        mv.validate_manifest(manifest)
    assert "duplicate media_path" in str(exc.value)


# ---------------------------------------------------------------------------
# The real manifest — schema only; hashing 597 MB does not belong in a unit test
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="Media/MEDIA_MANIFEST.json not present")
def test_real_manifest_is_schema_valid():
    manifest = mv.load_manifest(REAL_MANIFEST)
    assert manifest["assets"], "the real manifest must not be empty"
    for row in manifest["assets"]:
        assert len(row["sha256"]) == 64
        assert row["campaign"].strip()
        assert row["bytes"] > 0
        assert row["media_path"].startswith("Media/")
        assert row["sources"], "every asset must record where it came from"


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="Media/MEDIA_MANIFEST.json not present")
def test_real_manifest_totals_match_rows():
    manifest = mv.load_manifest(REAL_MANIFEST)
    assets = manifest["assets"]
    totals = manifest["totals"]
    assert totals["unique_assets"] == len(assets)
    assert totals["bytes"] == sum(a["bytes"] for a in assets)
    assert totals["campaigns"] == len(set(a["campaign"] for a in assets))
    assert totals["tracked_in_git"] == sum(1 for a in assets if a["tracked_in_git"])
    assert len(set(a["sha256"] for a in assets)) == len(assets), "assets must be deduped by sha256"


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="Media/MEDIA_MANIFEST.json not present")
def test_real_manifest_campaign_index_matches_rows():
    manifest = mv.load_manifest(REAL_MANIFEST)
    from collections import Counter
    counted = Counter(a["campaign"] for a in manifest["assets"])
    for slug, meta in manifest["campaigns"].items():
        assert meta["n_assets"] == counted[slug], "campaign index disagrees with rows: %s" % slug
    assert set(manifest["campaigns"]) == set(counted)
