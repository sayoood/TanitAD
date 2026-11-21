"""tools/verify_s3_etag.py — the check the nuScenes fetch actually needed.

⭐ Non-circular by construction: the positive and negative controls use the LIVE S3 ETag of a real
object as a LITERAL (S3 computed it, not this code), and the single-part branch uses the textbook md5
constants of b"" and b"abc". A verifier whose expected values come from the verifier measures its own
determinism, not its correctness (`CLAUDE.md`).
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import verify_s3_etag as V  # noqa: E402

REAL = "D:/Archive/devbox-C/nuscenes/archives/v1.0-trainval_meta.tgz"
REAL_SIZE = 461_678_030
#: the live object's ETag, from a HEAD on 2026-09-26 — produced by S3, not by us
LIVE_ETAG = '"fd4ea76d8701fb567a67a65025d0b022-56"'
#: what the bucket's published md5.checksum claims for the same archive — WRONG for the served object
STALE_CHECKSUM_MD5 = "3eee698806fcf52330faa2e682b9f3a1"

needs_real = pytest.mark.skipif(not (os.path.isfile(REAL) and os.path.getsize(REAL) == REAL_SIZE),
                                reason="nuScenes metadata archive not on this box")


class _FlipOneByte:
    """Streams a real file but XORs ONE byte at ``pos`` — a mutation without copying 0.46 GB."""

    def __init__(self, path: str, pos: int):
        self._f, self._pos, self._off = open(path, "rb"), pos, 0

    def read(self, n: int = -1) -> bytes:
        b = self._f.read(n)
        if b and self._off <= self._pos < self._off + len(b):
            m = bytearray(b)
            m[self._pos - self._off] ^= 0x01
            b = bytes(m)
        self._off += len(b)
        return b

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()


@needs_real
def test_POSITIVE_the_real_archive_matches_the_live_multipart_etag():
    verdict, detail = V.verify(lambda: open(REAL, "rb"), REAL_SIZE, LIVE_ETAG)
    assert verdict == "MATCH", detail
    assert "8 MiB" in detail


@needs_real
def test_NEGATIVE_one_flipped_byte_is_a_mismatch():
    """⛔ The control that proves the check can fail at all: one bit, mid-file, must be caught."""
    verdict, detail = V.verify(lambda: _FlipOneByte(REAL, REAL_SIZE // 2), REAL_SIZE, LIVE_ETAG)
    assert verdict == "MISMATCH", detail


@needs_real
def test_the_bucket_md5_checksum_value_does_NOT_verify_the_served_object():
    """Pins the finding: the publisher's checksum file does not describe the object it sits beside."""
    verdict, _ = V.verify(lambda: open(REAL, "rb"), REAL_SIZE, STALE_CHECKSUM_MD5)
    assert verdict == "MISMATCH"


def test_INCONCLUSIVE_when_no_part_size_fits_is_never_a_match_or_mismatch():
    """A failure to RECONSTRUCT the ETag is not evidence about the bytes — say so, don't guess."""
    verdict, detail = V.verify(lambda: io.BytesIO(b"x" * 100), 100, '"0123456789abcdef0123456789abcdef-56"')
    assert verdict == "INCONCLUSIVE", detail


@pytest.mark.parametrize("data,md5", [(b"", "d41d8cd98f00b204e9800998ecf8427e"),
                                      (b"abc", "900150983cd24fb0d6963f7d28e17f72")])
def test_single_part_etag_is_the_plain_md5(data, md5):
    """Textbook md5 constants, written as literals: the single-part branch."""
    assert V.verify(lambda: io.BytesIO(data), len(data), f'"{md5}"')[0] == "MATCH"
    assert V.verify(lambda: io.BytesIO(data + b"!"), len(data) + 1, f'"{md5}"')[0] == "MISMATCH"


def test_cli_exit_codes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert V.main([str(p), "900150983cd24fb0d6963f7d28e17f72"]) == 0
    assert V.main([str(p), "00000000000000000000000000000000"]) == 1
    assert V.main([str(tmp_path / "absent.bin"), "x"]) == 3
    assert V.main([]) == 2
