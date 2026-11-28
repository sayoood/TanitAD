"""Stage the corpus's source mp4s from a LOCAL MIRROR instead of HF -- verified by content.

E17 / C3 (2026-09-19): the 4,713 train clips' source mp4s are ALREADY on the dev box
(`C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov`), byte-identical
to the private HF corpus -- measured, 4,719 / 4,719 sha256 matches
(`…/Data Engineering/Research/2026-09-19-c3-source-transfer-price/`). Pulling them again
from HF costs ~3.4 h for nothing. This module is the local half of
`fetch_corpus_clips.py --local-mirror`.

⛔ THE CONTENT CHECK IS NOT RELAXED BECAUSE THE SOURCE IS LOCAL. A local file is exactly
as able to be stale, truncated or edited as a download is corrupt, and a mirror nobody
re-hashed is an assumption. So:

* the expected hashes come from HF, from TWO independently produced sources that must
  AGREE: the corpus's own `camera/camera_sha256.json` (written by the uploader) and the
  Hub's LFS metadata (computed by the Hub on upload) -- `reconcile_expected`;
* EVERY requested source file is hashed BEFORE anything is staged, and one mismatch
  stages NOTHING -- `stage` (a half-staged root is worse than none: it looks usable);
* every COPY is hashed again after it is written, so a bad copy is caught too;
* a size-only check is not a content check: the test suite tampers one byte at the
  SAME size, which a size check passes and only the hash catches.

Pure functions, no network: the caller supplies the expected hashes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


class MirrorMismatch(RuntimeError):
    """The local mirror disagrees with HF, or HF disagrees with itself. Never a warning."""


def sha256f(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()[:12]


def reconcile_expected(table: dict, lfs: dict, clips) -> dict:
    """{clip: (sha256, bytes)} where the sha table and the LFS metadata AGREE.

    `table`: {clip: {"sha256": str, "bytes": int|str}} -- `camera/camera_sha256.json`.
    `lfs`:   {clip: (sha256, bytes)} -- the Hub's per-file LFS sha256 and size.
    Refuses (MirrorMismatch) if a requested clip is absent from either source, or the two
    sources disagree on either field -- HF disagreeing with itself is not something to
    resolve by picking one.
    """
    out, problems = {}, []
    for c in clips:
        if c not in table or c not in lfs:
            problems.append("%s absent from %s" % (sha12(c), "table" if c not in table else "LFS"))
            continue
        t_sha, t_bytes = table[c]["sha256"], int(table[c]["bytes"])
        l_sha, l_bytes = lfs[c][0], int(lfs[c][1])
        if t_sha != l_sha or t_bytes != l_bytes:
            problems.append("%s: table and LFS disagree" % sha12(c))
            continue
        out[c] = (t_sha, t_bytes)
    if problems:
        raise MirrorMismatch("HARD FAIL: %d clip(s) without an agreed HF hash: %s"
                             % (len(problems), problems[:5]))
    return out


def verify_sources(clips, mirror, expected: dict) -> list:
    """Every problem with the mirror copies of `clips`, as sha12-keyed strings. Hashes all."""
    mirror, problems = Path(mirror), []
    for c in clips:
        src = mirror / ("%s.mp4" % c)
        if not src.is_file():
            problems.append("%s: missing from the mirror" % sha12(c))
            continue
        want_sha, want_bytes = expected[c]
        nb = src.stat().st_size
        if nb != want_bytes:
            problems.append("%s: %d bytes, HF says %d" % (sha12(c), nb, want_bytes))
            continue
        got = sha256f(src)
        if got != want_sha:
            problems.append("%s: sha256 %s, HF says %s" % (sha12(c), got[:16], want_sha[:16]))
    return problems


def stage(clips, mirror, cam, expected: dict, mode: str = "copy") -> list:
    """Verify ALL sources, then place each as `<cam>/<clip>.mp4`. One mismatch stages nothing.

    mode "copy": copy, then hash the COPY. mode "hardlink": os.link (same NTFS volume only;
    exFAT refuses, and this raises rather than silently copying -- the caller chose).
    A destination that already exists with the right bytes and hash is left in place.
    Returns receipt rows (sha12, bytes, sha256, staged_by).
    """
    if mode not in ("copy", "hardlink"):
        raise ValueError("mode must be 'copy' or 'hardlink'")
    problems = verify_sources(clips, mirror, expected)
    if problems:
        raise MirrorMismatch("HARD FAIL: %d mirror file(s) disagree with HF -- NOTHING staged: %s"
                             % (len(problems), problems[:5]))
    mirror, cam = Path(mirror), Path(cam)
    cam.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in clips:
        src, dst = mirror / ("%s.mp4" % c), cam / ("%s.mp4" % c)
        want_sha, want_bytes = expected[c]
        if dst.exists() and dst.stat().st_size == want_bytes and sha256f(dst) == want_sha:
            rows.append({"sha12": sha12(c), "bytes": want_bytes, "sha256": want_sha,
                         "staged_by": "already-present"})
            continue
        if dst.exists():
            dst.unlink()
        if mode == "hardlink":
            os.link(src, dst)                     # raises on exFAT / cross-volume: intended
        else:
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            got = sha256f(tmp)
            if got != want_sha or tmp.stat().st_size != want_bytes:
                tmp.unlink()
                raise MirrorMismatch("HARD FAIL: the COPY of %s does not match HF (%s)"
                                     % (sha12(c), got[:16]))
            os.replace(tmp, dst)
        rows.append({"sha12": sha12(c), "bytes": want_bytes, "sha256": want_sha,
                     "staged_by": mode})
    return rows


def verify_file(path, want_sha: str, want_bytes: int, name: str) -> None:
    """A single artifact (e.g. `timestamps.tar`) against its HF LFS hash. Raises on mismatch."""
    p = Path(path)
    if not p.is_file():
        raise MirrorMismatch("HARD FAIL: %s missing at the given local path" % name)
    nb = p.stat().st_size
    got = sha256f(p)
    if nb != int(want_bytes) or got != want_sha:
        raise MirrorMismatch("HARD FAIL: local %s is %d bytes / %s, HF says %s / %s"
                             % (name, nb, got[:16], want_bytes, want_sha[:16]))
