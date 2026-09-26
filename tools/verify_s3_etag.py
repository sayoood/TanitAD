#!/usr/bin/env python3
"""Verify a downloaded file against the S3 object's own ETag — including MULTIPART ETags.

⛔ WHY THIS EXISTS (MEASURED 2026-09-26, nuScenes `v1.0-trainval_meta.tgz`). The bucket's published
`md5.checksum` lists `3eee6988…` for that archive; the file we received hashes to `537d3954…` at an
EXACTLY matching byte count. It was not corrupt: `gzip -t` passed, the tar index listed cleanly, and the
S3 multipart ETag recomputed from our bytes (8 MiB parts) was `fd4ea76d8701fb567a67a65025d0b022-56` —
identical to the live object's. The checksum FILE was wrong. A verifier trusting it would have flagged a
perfect 0.46 GB download as corrupt — and on the 45 GB tier, trained everyone to switch verification off.

⭐ And the fetch script it replaces fetched `md5.checksum` and **never compared against it** — the same
"built, reachable, never used" class as the nuScenes category audit.

HOW AN S3 ETAG IS FORMED
* single-part upload: ``ETag = md5(bytes)`` — no suffix;
* multipart upload with n parts: ``ETag = md5(md5(part_1) ‖ … ‖ md5(part_n)) + "-" + str(n)``.
The part size is not in the ETag, so we try the sizes uploaders actually use and keep only those whose
part COUNT matches the suffix. If none does we cannot reconstruct it, and we say INCONCLUSIVE — never
MATCH, and never MISMATCH, because a failure to reconstruct is not evidence about the bytes.

Exit codes: 0 MATCH · 1 MISMATCH · 3 INCONCLUSIVE · 2 usage.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

#: part sizes in MiB, most common first (8 MiB is the AWS CLI / boto3 default)
CANDIDATE_PART_MIB = (8, 16, 5, 10, 15, 32, 50, 64, 100, 128, 256, 512)
CHUNK = 1 << 20


def _parse(etag: str) -> tuple[str, int | None]:
    e = etag.strip().strip('"').strip()
    if "-" in e:
        h, n = e.rsplit("-", 1)
        return h.lower(), int(n)
    return e.lower(), None


def _read_parts(fp, part_size: int):
    """Yield successive parts of exactly ``part_size`` bytes (the last may be shorter)."""
    while True:
        buf = bytearray()
        while len(buf) < part_size:
            chunk = fp.read(min(CHUNK, part_size - len(buf)))
            if not chunk:
                break
            buf += chunk
        if not buf:
            return
        yield bytes(buf)
        if len(buf) < part_size:
            return


def verify(opener, size: int, etag: str) -> tuple[str, str]:
    """``opener()`` returns a fresh binary stream over the file (called once per candidate).

    Returns ``(verdict, detail)`` with verdict in MATCH / MISMATCH / INCONCLUSIVE.
    """
    want, n = _parse(etag)
    if n is None:                                          # single-part: the ETag IS the md5
        h = hashlib.md5()
        with opener() as fp:
            for chunk in iter(lambda: fp.read(CHUNK), b""):
                h.update(chunk)
        got = h.hexdigest()
        return ("MATCH", f"single-part md5 {got}") if got == want else \
               ("MISMATCH", f"single-part md5 {got} != ETag {want}")
    fits = [m for m in CANDIDATE_PART_MIB if math.ceil(size / (m << 20)) == n]
    if not fits:
        return "INCONCLUSIVE", (f"no candidate part size gives {n} parts for {size} B "
                                f"(tried {list(CANDIDATE_PART_MIB)} MiB) — cannot reconstruct the ETag")
    tried = []
    for m in fits:
        digests = b""
        with opener() as fp:
            for part in _read_parts(fp, m << 20):
                digests += hashlib.md5(part).digest()
        got = f"{hashlib.md5(digests).hexdigest()}-{n}"
        if got == f"{want}-{n}":
            return "MATCH", f"multipart ETag {got} ({m} MiB parts)"
        tried.append(f"{m} MiB -> {got}")
    return "MISMATCH", f"no fitting part size reproduces {want}-{n}: " + "; ".join(tried)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 2:
        print("usage: verify_s3_etag.py <file> <etag>", file=sys.stderr)
        return 2
    path, etag = a
    if not os.path.isfile(path):
        print(f"INCONCLUSIVE: cannot read {path}")
        return 3
    verdict, detail = verify(lambda: open(path, "rb"), os.path.getsize(path), etag)
    print(f"{verdict}: {detail}")
    return {"MATCH": 0, "MISMATCH": 1}.get(verdict, 3)


if __name__ == "__main__":
    sys.exit(main())
