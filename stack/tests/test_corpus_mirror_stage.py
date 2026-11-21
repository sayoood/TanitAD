"""`corpus_mirror_stage`: a local mirror is staged ONLY if every file matches HF's hashes.

The load-bearing case is `test_one_byte_tamper_at_the_same_size_refuses_and_stages_nothing`:
a size check passes it and only a content hash catches it, so it goes RED if the sha256
comparison is ever removed from `verify_sources` (verified by mutation, 2026-09-19).
Expectations are literals, never expressions over the module under test.
No network: the expected hashes are supplied directly.
"""
import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import corpus_mirror_stage as S  # noqa: E402

CLIPS = ["00000000-aaaa-4bbb-8ccc-000000000001",
         "00000000-aaaa-4bbb-8ccc-000000000002",
         "00000000-aaaa-4bbb-8ccc-000000000003"]
#: three payloads of DIFFERENT sizes, so a byte-count check has something to see
PAYLOAD = {CLIPS[0]: b"\x00\x01mp4-one" * 100,
           CLIPS[1]: b"\x02\x03mp4-two" * 250,
           CLIPS[2]: b"\x04\x05mp4-three" * 40}


def _mirror(tmp_path):
    m = tmp_path / "mirror"
    m.mkdir()
    for c, data in PAYLOAD.items():
        (m / ("%s.mp4" % c)).write_bytes(data)
    return m


def _expected():
    """What HF would say, computed from the payload by hashlib -- not by the module."""
    return {c: (hashlib.sha256(d).hexdigest(), len(d)) for c, d in PAYLOAD.items()}


def _staged(cam):
    return sorted(p.name for p in Path(cam).glob("*.mp4")) if Path(cam).exists() else []


def test_matching_mirror_stages_every_file_and_the_copies_match(tmp_path):
    m, cam = _mirror(tmp_path), tmp_path / "stage" / "r0" / "camera_front_wide"
    rows = S.stage(CLIPS, m, cam, _expected(), mode="copy")
    assert len(rows) == 3                                               # literal
    assert [r["staged_by"] for r in rows] == ["copy", "copy", "copy"]   # literal
    for c, data in PAYLOAD.items():
        assert (cam / ("%s.mp4" % c)).read_bytes() == data
    assert not list(cam.glob("*.part"))                                 # no debris


def test_one_byte_tamper_at_the_same_size_refuses_and_stages_nothing(tmp_path):
    m, cam = _mirror(tmp_path), tmp_path / "stage" / "r0" / "camera_front_wide"
    victim = m / ("%s.mp4" % CLIPS[1])
    b = bytearray(victim.read_bytes())
    b[len(b) // 2] ^= 0xFF                      # one byte flipped, SIZE UNCHANGED
    victim.write_bytes(bytes(b))
    assert victim.stat().st_size == 2250                                # literal: same size
    with pytest.raises(S.MirrorMismatch, match="NOTHING staged"):
        S.stage(CLIPS, m, cam, _expected(), mode="copy")
    assert _staged(cam) == []                                           # literal: none


def test_truncated_file_refuses(tmp_path):
    m, cam = _mirror(tmp_path), tmp_path / "stage"
    p = m / ("%s.mp4" % CLIPS[0])
    p.write_bytes(p.read_bytes()[:-7])
    with pytest.raises(S.MirrorMismatch, match="1 mirror file"):
        S.stage(CLIPS, m, cam, _expected())
    assert _staged(cam) == []


def test_missing_file_refuses(tmp_path):
    m, cam = _mirror(tmp_path), tmp_path / "stage"
    (m / ("%s.mp4" % CLIPS[2])).unlink()
    with pytest.raises(S.MirrorMismatch, match="missing from the mirror"):
        S.stage(CLIPS, m, cam, _expected())
    assert _staged(cam) == []


def test_hf_disagreeing_with_itself_refuses():
    table = {c: {"sha256": s, "bytes": n} for c, (s, n) in _expected().items()}
    lfs = dict(_expected())
    lfs[CLIPS[0]] = ("f" * 64, lfs[CLIPS[0]][1])            # the Hub says something else
    with pytest.raises(S.MirrorMismatch, match="table and LFS disagree"):
        S.reconcile_expected(table, lfs, CLIPS)
    # control: when both sources agree, all three reconcile
    assert len(S.reconcile_expected(table, dict(_expected()), CLIPS)) == 3  # literal


def test_a_clip_absent_from_hf_refuses():
    table = {c: {"sha256": s, "bytes": n} for c, (s, n) in _expected().items()}
    with pytest.raises(S.MirrorMismatch, match="absent from"):
        S.reconcile_expected(table, dict(_expected()), CLIPS + ["not-a-clip"])


def test_hardlink_mode_links_rather_than_copies(tmp_path):
    m, cam = _mirror(tmp_path), tmp_path / "stage"
    try:
        rows = S.stage(CLIPS, m, cam, _expected(), mode="hardlink")
    except OSError:
        pytest.skip("this filesystem cannot hardlink (exFAT) -- stage() refused, as intended")
    assert [r["staged_by"] for r in rows] == ["hardlink"] * 3            # literal
    assert os.path.samefile(m / ("%s.mp4" % CLIPS[0]), cam / ("%s.mp4" % CLIPS[0]))


def test_one_byte_tamper_refuses_in_hardlink_mode_too(tmp_path):
    """In hardlink mode there is NO copy to re-hash, so the SOURCE check is the only guard.
    MEASURED by mutation 2026-09-19: with the source sha256 comparison removed, copy mode
    was still refused by the copy re-hash (after staging a file -- so not atomic), while
    this mode would stage the tampered file silently. This test is what catches that."""
    m, cam = _mirror(tmp_path), tmp_path / "stage"
    victim = m / ("%s.mp4" % CLIPS[2])
    b = bytearray(victim.read_bytes())
    b[0] ^= 0xFF
    victim.write_bytes(bytes(b))
    try:
        os.link(m / ("%s.mp4" % CLIPS[0]), tmp_path / "probe_link")
    except OSError:
        pytest.skip("this filesystem cannot hardlink (exFAT)")
    with pytest.raises(S.MirrorMismatch, match="NOTHING staged"):
        S.stage(CLIPS, m, cam, _expected(), mode="hardlink")
    assert _staged(cam) == []                                           # literal: none


def test_verify_file_refuses_a_wrong_hash(tmp_path):
    p = tmp_path / "timestamps.tar"
    p.write_bytes(b"tar-bytes" * 10)
    good = hashlib.sha256(b"tar-bytes" * 10).hexdigest()
    S.verify_file(p, good, 90, "timestamps.tar")                       # control: passes
    with pytest.raises(S.MirrorMismatch, match="HF says"):
        S.verify_file(p, "0" * 64, 90, "timestamps.tar")
