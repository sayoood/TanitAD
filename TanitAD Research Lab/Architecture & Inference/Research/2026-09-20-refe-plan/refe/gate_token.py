#!/usr/bin/env python3
"""The REFe launch-gate TOKEN: mint, verify, and the primitives they stand on.  Route and rationale: eval/REFE_LAUNCH_GATE.md.

SAME SCHEMA AND KEY AS THE PROGRAMME'S GENERIC GATE (`tanitad.launch_gate/1`, banked at products/P4-training-pipelines/
2026-09-27-v7f-refav1-state/v7f_ckpt/code/fix/stack/scripts/launch_gate.py): `canonical_json`, `argv_sha256`, `sign`, `signature_ok`,
`fingerprint`, `fp_key` are byte-identical re-implementations, and test_launch_gate.py loads the banked module (when present) and
asserts equal outputs, so a token minted here verifies there once that gate knows the refe profiles.  Never hand-written: the
HMAC refuses an edited token.

WHY A SEPARATE MODULE (not a copy of the 7,806-line generic gate): the pod ships only `code/ refe/ splits/` (no git, no `stack/`),
and REFe is not a `tanitad.train` trainer, so everything the generic gate wraps around the token (trainer loader, declared_vs_built,
smoke runners) does not apply.  What REFe needs from it is exactly this: a token bound to (profile, commit, tree, argv, data).

THE COMMIT WITHOUT GIT ON THE POD: `ship-manifest` (dev box) writes SHIP_MANIFEST.json = {commit, files: {path: sha256}} after proving
every shipped file equals its blob at that commit; the pod's `mint` proves the pod's tree equals the manifest.  So the commit in the
token is a measured fact about the bytes, not a label.
"""
from __future__ import annotations

import calendar
import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

GATE_SCHEMA = "tanitad.launch_gate/1"
KEY_ENV = "TANITAD_GATE_KEY_FILE"
FP_THREADS_ENV = "TANITAD_GATE_FP_THREADS"          # worker threads of the metadata walk / the concurrent fingerprints; 1 = fully serial (the pre-2026-10-15 shape)
FP_THREADS_DEFAULT = 16
FP_STAT_CHUNK = 256                                  # files per stat task of a directory listing
SHIP_MANIFEST = "SHIP_MANIFEST.json"
TREE_ROOTS = ("refe", "code", "splits")
_TREE_SUFFIX = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".patch")
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})
CONTENT_MANIFEST = "content_manifest.json"          # HNSCREEN-CACHE-CONTENT-FP: the build-time sha256-per-file manifest of a data DIRECTORY whose CONTENT the gate must bind (the HN screen's --ctx-cache)
CONTENT_MANIFEST_TOOL = "content_manifest/1"
HASH_THREADS_ENV = "TANITAD_GATE_HASH_THREADS"      # worker threads that sha256 the files of a content fingerprint; 1 = serial
HASH_THREADS_DEFAULT = 4
CONTENT_MAX_PROBLEMS = 20                            # problems listed in a content fingerprint (the count is exact; the list is the first ones)


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    """sha256 of the file's bytes.  `readinto` a reusable buffer: no 4 MiB allocation per read (the digest is the same bytes' digest, whatever the read size)."""
    h = hashlib.sha256()
    buf = bytearray(chunk)
    mv = memoryview(buf)
    with open(p, "rb", buffering=0) as fh:
        while True:
            n = fh.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Key-sorted, compact, UTF-8: the ONE byte form every hash and HMAC here is taken over (== the generic gate's)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def argv_sha256(argv: list) -> str:
    """sha256 of the argv as an ORDERED JSON array (order is meaning; never sorted) (== the generic gate's)."""
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise GateError("argv must be a list of strings")
    return sha256_bytes(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def argv_from_string(s: str) -> list:
    """The shell word-splitting of an unquoted `$ARGS` (default IFS) -- what `train.py $ARGS` passes.  No path in ARGS holds a blank
    or a glob character; test_launch_gate pins that on the chain's real ARGS."""
    return s.split()


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ key and signature
def default_key_file() -> Path:
    env = os.environ.get(KEY_ENV)
    return Path(env) if env else Path.home() / ".config" / "tanitad" / "launch_gate.key"


def load_key(path=None, *, create: bool = True, forbid_under: Iterable = ()) -> bytes:
    """32 random bytes, hex on disk, created on first use with mode 0600.  Refused inside a code tree: a key that can be shipped is
    not a key.  On the pod the key is created by the first gate run on THAT pod; a token minted elsewhere does not verify there."""
    p = Path(path) if path else default_key_file()
    rp = os.path.normcase(str(p.resolve()))
    for t in forbid_under:
        rt = os.path.normcase(str(Path(t).resolve()))
        if rp == rt or rp.startswith(rt + os.sep):
            raise GateError(f"refusing a key file inside the code tree {t}: {p}")
    if not p.exists():
        if not create:
            raise GateError(f"no gate key at {p} (run the gate on this host first)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(secrets.token_hex(32) + "\n")
    txt = p.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", txt):
        raise GateError(f"gate key {p} is not 64 hex chars")
    return bytes.fromhex(txt)


def key_id(key: bytes) -> str:
    return sha256_bytes(b"tanitad-launch-gate-key-id:" + key)[:16]


def sign(token_wo_sig: dict, key: bytes) -> str:
    return hmac.new(key, canonical_json(token_wo_sig), hashlib.sha256).hexdigest()


def signature_ok(token: dict, key: bytes) -> bool:
    body = {k: v for k, v in token.items() if k != "hmac_sha256"}
    got = str(token.get("hmac_sha256", ""))
    return len(got) == 64 and hmac.compare_digest(sign(body, key), got)


# ------------------------------------------------------------------------------------------ fingerprints
def fp_threads() -> int:
    """Worker threads for the directory walk and for the concurrent fingerprints of `data_manifest` (env TANITAD_GATE_FP_THREADS; 1 = serial)."""
    try:
        n = int(os.environ.get(FP_THREADS_ENV, ""))
    except ValueError:
        n = 0
    return n if n >= 1 else FP_THREADS_DEFAULT


def _stat_lines(dirpath: str, rel_prefix: str, names: list) -> tuple:
    """([`relpath\\tsize\\n`...], total bytes) of `names` in `dirpath`.  os.stat FOLLOWS symlinks, like Path.stat() did; it raises the same errors."""
    lines, total = [], 0
    for fn in names:
        size = os.stat(os.path.join(dirpath, fn)).st_size
        lines.append(f"{rel_prefix}{fn}\t{size}\n")
        total += size
    return lines, total


def _dir_listing(p: Path, pool) -> tuple:
    """(lines, n_files, total_bytes) of the LISTING of directory `p`: exactly the walk the gate has always taken (os.walk unchanged: symlinked dirs not followed, unreadable
    dirs skipped, `_DIR_SKIP_NAMES` pruned), one `relpath\\tsize` line per file.  What changed is only HOW the per-file sizes are read: string ops instead of a Path object per file,
    and, when `pool` is given, the stat calls of a directory's files run on worker threads (os.stat releases the GIL; on a network filesystem the per-file latency, not the CPU,
    is the cost) while os.walk goes on listing the next directory.  The digest taken over the sorted lines is bit-identical to the serial one (test_launch_gate.py: fingerprint_identity_static)."""
    root = os.fspath(p)
    posix_fix = os.sep == "\\"
    lines, n, total, futs = [], 0, 0, []
    try:
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _DIR_SKIP_NAMES)
            names = [fn for fn in filenames if fn not in _DIR_SKIP_NAMES]
            if not names:
                continue
            sub = dirpath[len(root):]
            if sub.startswith(os.sep):
                sub = sub[len(os.sep):]
            if posix_fix:
                sub = sub.replace("\\", "/")
            rel_prefix = sub + "/" if sub else ""
            n += len(names)
            if pool is None:
                ls, t = _stat_lines(dirpath, rel_prefix, names)
                lines += ls
                total += t
            else:
                for i in range(0, len(names), FP_STAT_CHUNK):
                    futs.append(pool.submit(_stat_lines, dirpath, rel_prefix, names[i:i + FP_STAT_CHUNK]))
        for f in futs:
            ls, t = f.result()
            lines += ls
            total += t
    finally:
        for f in futs:
            f.cancel()
    return lines, n, total


def fingerprint(path, *, _pool=None) -> dict:
    """A file: its sha256.  A directory: its LISTING (relpath + size of every file; contents NOT hashed -- a 449 GB bank is not re-read
    per launch; stated in `method`).  Missing: missing.  (== the generic gate's)  `_pool`: a ThreadPoolExecutor for the listing's stat calls (default: a private one of
    `fp_threads()` workers; `TANITAD_GATE_FP_THREADS=1` = serial).  The value returned is the same for every thread count."""
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": sha256_file(p), "method": "sha256-v1"}
    if p.is_dir():
        own = None
        if _pool is None and fp_threads() > 1:
            own = _pool = concurrent.futures.ThreadPoolExecutor(max_workers=fp_threads(), thread_name_prefix="fp-stat")
        try:
            lines, n, total = _dir_listing(p, _pool)
        finally:
            if own is not None:
                own.shutdown(wait=True)
        lines.sort()
        return {"kind": "dir", "n_files": n, "bytes": total, "listing_sha256": sha256_bytes("".join(lines).encode("utf-8")),
                "method": "dir-listing-v1 (relpath + size of every file; contents NOT hashed; _v2manifest.pt excluded -- a cache "
                          "rewrites it on first use)"}
    return {"kind": "missing"}


def fp_key(fp: dict) -> str:
    if fp.get("kind") == "file":
        return f"file:{fp.get('sha256')}"
    if fp.get("kind") == "dir":
        return f"dir:{fp.get('n_files')}:{fp.get('bytes')}:{fp.get('listing_sha256')}"
    if fp.get("kind") == "dir-content":                                     # content_fingerprint(): a valid one is `dirc:`; a directory with ANY problem is `dir-content-INVALID:` and can equal no registered key
        if fp.get("n_problems") or not fp.get("content_sha256"):
            return f"dir-content-INVALID:{fp.get('n_problems')}:{sha256_bytes(chr(10).join(str(x) for x in (fp.get('problems') or [])).encode('utf-8'))[:16]}"
        return f"dirc:{fp.get('n_files')}:{fp.get('bytes')}:{fp.get('content_sha256')}"
    if fp.get("kind") == "dir-anchor":                                      # anchor_fingerprint(): a valid one carries the ANCHOR's content key (= the registered `dirc:` pin); any problem is `dir-anchor-INVALID:` and equals no registered key
        if fp.get("n_problems") or not fp.get("content_sha256"):
            return f"dir-anchor-INVALID:{fp.get('n_problems')}:{sha256_bytes(chr(10).join(str(x) for x in (fp.get('problems') or [])).encode('utf-8'))[:16]}"
        return f"dirc:{fp.get('n_files')}:{fp.get('bytes')}:{fp.get('content_sha256')}"
    return f"{fp.get('kind')}:{fp.get('sha256', fp.get('why', ''))}"


# ------------------------------------------------------------------------------------------ content fingerprints (HNSCREEN-CACHE-CONTENT-FP)
# `fingerprint(dir)` above is a LISTING (relpath + size): a same-size byte flip INSIDE a file passes it (MEASURED 2026-10-19, raw/2026-10-19-hnscreen-fp/out_probe_dir_fp_limit.log).  A directory whose CONTENT
# the launch must bind (the P6-HN screen's --ctx-cache) is fingerprinted HERE instead.  The BUILD seals it ONCE (`seal_content_manifest`: sha256 of every file's bytes -> CONTENT_MANIFEST, sorted relpaths); every
# gate check re-hashes EVERY file against that manifest (`content_fingerprint`) -- there is no way to bind bytes without reading them -- and the fingerprint is the sha256 of the manifest's own canonical bytes,
# recomputed from the LIVE files (so a hand-edited manifest cannot vouch for itself).  Fail closed: a missing / unreadable / non-canonical manifest, a missing, extra, renamed, resized or changed file all
# make the fingerprint `dir-content-INVALID:...`, which equals no registered key.  Nothing above changes: `fingerprint` / `fp_key` of every other kind are byte-identical, and a profile opts in PER FLAG
# (`data_manifest(content_flags=...)`, `verify_token(content_flags=...)`).
def hash_threads() -> int:
    """Worker threads that sha256 the files of a content fingerprint (env TANITAD_GATE_HASH_THREADS; 1 = serial).  hashlib drops the GIL on large buffers; the value is the same for every thread count."""
    try:
        n = int(os.environ.get(HASH_THREADS_ENV, ""))
    except ValueError:
        n = 0
    return n if n >= 1 else HASH_THREADS_DEFAULT


def _sha256_and_size(path, chunk: int = 1 << 22) -> tuple:
    """(bytes actually read, sha256 hex) of one file: the size recorded is the size HASHED, so a manifest is self-consistent even if a file grows mid-read."""
    h = hashlib.sha256()
    buf = bytearray(chunk)
    mv = memoryview(buf)
    n = 0
    with open(path, "rb", buffering=0) as fh:
        while True:
            k = fh.readinto(mv)
            if not k:
                break
            h.update(mv[:k])
            n += k
    return n, h.hexdigest()


def _content_listing(root: Path) -> tuple:
    """({relpath: size} of EVERY file under `root` except the manifest itself, [problems]).  Unlike the listing fingerprint nothing is pruned (an extra file is a finding), and a directory that cannot be
    listed is a PROBLEM, never a silent skip (os.walk's default)."""
    rootp = os.fspath(root)
    sizes: dict = {}
    problems: list = []

    def onerr(e):
        problems.append(f"cannot list {getattr(e, 'filename', '?')}: {type(e).__name__}: {e}")
    for dirpath, dirnames, filenames in os.walk(rootp, onerror=onerr):
        dirnames[:] = sorted(dirnames)
        problems += [f"a symlinked directory is not followed, so its files cannot be bound: {os.path.relpath(os.path.join(dirpath, d), rootp).replace(os.sep, '/')}"
                     for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, rootp).replace(os.sep, "/")
            if rel == CONTENT_MANIFEST:
                continue
            try:
                sizes[rel] = os.stat(full).st_size
            except OSError as e:
                problems.append(f"cannot stat {rel}: {type(e).__name__}: {e}")
    return sizes, problems


def manifest_bytes(files: dict) -> bytes:
    """The canonical bytes of a content manifest over `files` = {relpath: (size, sha256)}: key-sorted JSON, indent 1, LF, UTF-8.  The sha256 of THESE bytes is the directory's content fingerprint."""
    body = {"tool": CONTENT_MANIFEST_TOOL, "algorithm": "sha256", "n_files": len(files), "bytes": sum(v[0] for v in files.values()),
            "files": {rel: {"bytes": int(v[0]), "sha256": v[1]} for rel, v in sorted(files.items())}}
    return (json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _parse_manifest(raw: bytes) -> tuple:
    """({relpath: (size, sha256)}, [problems]) of a stored manifest; any malformation is a problem and nothing is trusted."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, [f"{CONTENT_MANIFEST} is unreadable: {type(e).__name__}: {str(e)[:100]}"]
    if not isinstance(doc, dict) or doc.get("tool") != CONTENT_MANIFEST_TOOL or doc.get("algorithm") != "sha256" or not isinstance(doc.get("files"), dict) or not doc["files"]:
        return None, [f"{CONTENT_MANIFEST} is not a non-empty {CONTENT_MANIFEST_TOOL} (sha256) manifest"]
    out, bad = {}, []
    for rel, ent in doc["files"].items():
        parts = str(rel).split("/")
        if not rel or rel.startswith("/") or "\\" in rel or "" in parts or ".." in parts or not isinstance(ent, dict) or type(ent.get("bytes")) is not int or ent["bytes"] < 0 \
                or not re.fullmatch(r"[0-9a-f]{64}", str(ent.get("sha256", ""))):
            bad.append(str(rel)[:80])
        else:
            out[rel] = (ent["bytes"], ent["sha256"])
    if bad:
        return None, [f"{CONTENT_MANIFEST} has malformed entries: {bad[:5]}"]
    return out, []


def _hash_files(root: Path, rels: list, pool) -> tuple:
    """({rel: (size, sha256)}, [problems]) of `rels` under `root`; the files are hashed on `pool`'s workers when given."""
    live, problems = {}, []
    futs = [(rel, (pool.submit(_sha256_and_size, root / rel) if pool is not None else None)) for rel in rels]
    try:
        for rel, f in futs:
            try:
                live[rel] = f.result() if f is not None else _sha256_and_size(root / rel)
            except OSError as e:
                problems.append(f"cannot read {rel}: {type(e).__name__}: {e}")
    finally:
        for _, f in futs:
            if f is not None:
                f.cancel()
    return live, problems


def content_check(root, pool=None) -> dict:
    """Check directory `root` against its CONTENT_MANIFEST.  {"problems": [...], "live": {rel: (size, sha256)} | None, "manifest_file_sha256": str | None}.  Structural problems (a missing / malformed / non-canonical
    manifest, a missing / extra / renamed / resized file) end the check BEFORE any byte is hashed (fail fast: a 70 GB cache is not re-read to learn a file is missing); otherwise every file is re-hashed and a
    byte change at the same size is a problem."""
    root = Path(root)
    mp = root / CONTENT_MANIFEST
    if not mp.is_file():
        return {"problems": [f"no {CONTENT_MANIFEST} in {root}: the directory was not sealed at build (ctx_cache.finalize writes it; a cache built without it: `python refe/ctx_cache.py seal --dir <dir>`)"],
                "live": None, "manifest_file_sha256": None}
    try:
        raw = mp.read_bytes()
    except OSError as e:
        return {"problems": [f"{CONTENT_MANIFEST} cannot be read: {type(e).__name__}: {e}"], "live": None, "manifest_file_sha256": None}
    stored, problems = _parse_manifest(raw)
    if stored is not None and raw != manifest_bytes(stored):
        problems.append(f"{CONTENT_MANIFEST} is not in the canonical form it is written in (edited by hand, or written by another tool)")
    if stored is None:
        return {"problems": problems, "live": None, "manifest_file_sha256": sha256_bytes(raw)}
    sizes, lprobs = _content_listing(root)
    problems += lprobs
    problems += [f"file in the manifest is MISSING: {r}" for r in sorted(set(stored) - set(sizes))]
    problems += [f"file NOT in the manifest (extra, or a renamed file): {r}" for r in sorted(set(sizes) - set(stored))]
    problems += [f"SIZE changed: {r} (manifest {stored[r][0]} bytes, now {sizes[r]})" for r in sorted(set(stored) & set(sizes)) if stored[r][0] != sizes[r]]
    if not sizes:
        problems.append("the directory holds no file besides its manifest")
    if problems:
        return {"problems": problems, "live": None, "manifest_file_sha256": sha256_bytes(raw)}
    live, hprobs = _hash_files(root, sorted(sizes), pool)
    problems += hprobs
    problems += [f"BYTES CHANGED at the same size ({live[r][0]}): {r} (sha256 manifest {stored[r][1][:12]}, now {live[r][1][:12]})" for r in sorted(live)
                 if live[r][1] != stored[r][1] or live[r][0] != stored[r][0]]
    return {"problems": problems, "live": live if not problems else None, "manifest_file_sha256": sha256_bytes(raw)}


def content_fingerprint(path, *, _pool=None) -> dict:
    """The CONTENT fingerprint of `path`.  A file: `fingerprint` (its sha256).  Missing: missing.  A directory: `kind: dir-content` -- valid only if it carries a CONTENT_MANIFEST that every file's bytes match
    exactly (no extra, missing, renamed, resized or changed file); then `content_sha256` is the sha256 of the manifest's canonical bytes recomputed from the LIVE files, and `fp_key` is `dirc:<n>:<bytes>:<sha>`.
    Any problem -> `n_problems > 0` and the key is `dir-content-INVALID:...`.  Every call re-reads every byte (no cache): that is what makes it a content check."""
    p = Path(path)
    if p.is_file():
        return fingerprint(p)
    if not p.is_dir():
        return {"kind": "missing"}
    own = None
    if _pool is None and hash_threads() > 1:
        own = _pool = concurrent.futures.ThreadPoolExecutor(max_workers=hash_threads(), thread_name_prefix="fp-hash")
    try:
        r = content_check(p, _pool)
    finally:
        if own is not None:
            own.shutdown(wait=True)
    fp = {"kind": "dir-content", "n_problems": len(r["problems"]), "problems": r["problems"][:CONTENT_MAX_PROBLEMS], "manifest_file_sha256": r["manifest_file_sha256"],
          "method": "dir-content-v1 (sha256 of the BYTES of every file, all re-read at every check, against the build-time content_manifest.json; the key is the sha256 of the manifest's canonical bytes "
                    "recomputed from the live files; a missing manifest or any missing / extra / renamed / resized / changed file makes the key INVALID)"}
    if r["live"] is not None:
        fp.update(n_files=len(r["live"]), bytes=sum(v[0] for v in r["live"].values()), content_sha256=sha256_bytes(manifest_bytes(r["live"])))
    return fp


def seal_content_manifest(root, *, pool=None) -> dict:
    """THE BUILD SIDE, run once when a directory is finished: sha256 every file under `root` and write CONTENT_MANIFEST atomically (LF, canonical bytes).  Refuses an empty or unreadable directory.  Re-sealing
    rewrites it (the old manifest is never one of the files hashed).  -> {"path", "sha256" (of the manifest file == the directory's content fingerprint), "n_files", "bytes"}."""
    root = Path(root)
    if not root.is_dir():
        raise GateError(f"cannot seal {root}: not a directory")
    tmp = root / (CONTENT_MANIFEST + ".tmp")
    if tmp.exists():
        tmp.unlink()                                          # a crashed earlier seal: never one of the files hashed
    sizes, problems = _content_listing(root)
    if problems or not sizes:
        raise GateError(f"cannot seal {root}: " + ("; ".join(problems[:3]) if problems else "no file under it"))
    own = None
    if pool is None and hash_threads() > 1:
        own = pool = concurrent.futures.ThreadPoolExecutor(max_workers=hash_threads(), thread_name_prefix="fp-hash")
    try:
        live, problems = _hash_files(root, sorted(sizes), pool)
    finally:
        if own is not None:
            own.shutdown(wait=True)
    if problems:
        raise GateError(f"cannot seal {root}: " + "; ".join(problems[:3]))
    raw = manifest_bytes(live)
    with open(tmp, "wb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, root / CONTENT_MANIFEST)
    return {"path": str(root / CONTENT_MANIFEST), "sha256": sha256_bytes(raw), "n_files": len(live), "bytes": sum(v[0] for v in live.values())}


# ------------------------------------------------------------------------------------------ the CTX ANCHOR (HNSCREEN-FP-COST, MM ruling (b'), 2026-10-19)
# A 3-seed screen re-read the ~70 GiB cache SIX times per seed (`launch_gate.py run` = run_checks 2 + finalize 2, then `verify` at the mint and again at the pass start) = 20 full reads (CALLMAP.md).  The content check
# is KEPT IN FULL where it is the evidence (the seal, the fill, the START of the chain, the END of the chain); only its REPETITION in between is bounded.  An anchor is the chain's start-of-chain record: ONE full
# content fingerprint (compared with the registered pin) + the (relpath, size, mtime_ns) of EVERY file, HMAC-signed with the gate key.  An INTERMEDIATE gate call in `anchor-bound` mode compares the live stat
# listing with it and reads no byte: any missing, extra, resized or re-timestamped file is a REFUSAL.  The residual -- a same-size edit that also restores the mtime_ns -- is DETECTED at the END (a second full
# fingerprint vs the anchor + a final stat compare): a mismatch writes a VOID record, and the reader (`ctx_chain_refusals`) refuses any chain without an END_VERIFY_OK record bound to the same anchor.
# Nothing here changes `content` mode (the default everywhere): `data_manifest(ctx_anchor=None)` is the call it always was.
CTX_MODE_CONTENT, CTX_MODE_ANCHOR = "content", "anchor-bound"
CTX_MODES = (CTX_MODE_CONTENT, CTX_MODE_ANCHOR)
ANCHOR_SCHEMA, ANCHOR_KIND = "tanitad.hn_ctx_anchor/1", "hn-ctx-anchor"
END_SCHEMA, END_KIND = "tanitad.hn_ctx_end_verify/1", "hn-ctx-end-verify"
END_OK, END_VOID = "END_VERIFY_OK", "VOID"
ANCHOR_REL, END_REL, CHAIN_REL = "ctx_chain/anchor.json", "ctx_chain/end_verify.json", "ctx_chain/chain.json"      # under the chain root: where the chain writes them and the reader reads them


def norm_dir(p) -> str:
    """The identity of a directory for the anchor's 'same cache directory' test: absolute, symlinks resolved, case- and separator-normalised."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(p))))


def _stat_listing(root) -> tuple:
    """({relpath: [size, mtime_ns]} of EVERY file under `root`, the content manifest INCLUDED and nothing pruned, [problems]).  Reads no byte.  A directory that cannot be listed, a file that cannot be stat'ed and
    a symlinked directory (not followed, so its files would be bound by nothing) are PROBLEMS, never a silent skip."""
    rootp = os.fspath(root)
    files: dict = {}
    problems: list = []

    def onerr(e):
        problems.append(f"cannot list {getattr(e, 'filename', '?')}: {type(e).__name__}: {e}")
    for dirpath, dirnames, filenames in os.walk(rootp, onerror=onerr):
        dirnames[:] = sorted(dirnames)
        problems += [f"a symlinked directory is not followed, so its files cannot be bound: {os.path.relpath(os.path.join(dirpath, d), rootp).replace(os.sep, '/')}"
                     for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, rootp).replace(os.sep, "/")
            try:
                st = os.stat(full)
                files[rel] = [int(st.st_size), int(st.st_mtime_ns)]
            except OSError as e:
                problems.append(f"cannot stat {rel}: {type(e).__name__}: {e}")
    return files, problems


def mint_anchor(cache_dir, *, chain_id: str, key: bytes, pin, out_path) -> dict:
    """THE START OF THE CHAIN: ONE full content fingerprint of `cache_dir` (every file re-hashed against its seal), compared with the REGISTERED pin, bracketed by two stat listings that must be equal (a file edited
    while it is being anchored is not anchored).  Writes the HMAC-signed anchor to `out_path` and returns it.  Raises GateError on ANY problem: nothing is written then."""
    if not re.fullmatch(r"[0-9a-f]{16,64}", str(chain_id)):
        raise GateError("the anchor needs a chain id (16-64 hex)")
    if not (isinstance(pin, str) and pin.startswith("dirc:")):
        raise GateError(f"no registered content pin to anchor against ({str(pin)[:24]!r} is not a dirc: key): fill fingerprints_expected with hn_fingerprints.py first")
    if not Path(cache_dir).is_dir():
        raise GateError(f"cannot anchor {cache_dir}: not a directory")
    before, p1 = _stat_listing(cache_dir)
    fp = content_fingerprint(cache_dir)                                                      # the ONE full read of the anchor
    after, p2 = _stat_listing(cache_dir)
    if p1 or p2:
        raise GateError("cannot anchor: " + "; ".join((p1 + p2)[:3]))
    if fp.get("n_problems"):
        raise GateError(f"cannot anchor: the cache FAILS its content check ({fp['n_problems']} problem(s)): " + "; ".join(str(x) for x in (fp.get("problems") or [])[:3]))
    if before != after:
        raise GateError("cannot anchor: the cache changed while it was being anchored (its stat listing moved across the content read)")
    k = fp_key(fp)
    if k != pin:
        raise GateError(f"cannot anchor: the cache's content fingerprint {k[:40]} is not the registered pin {pin[:40]}")
    doc = {"schema": ANCHOR_SCHEMA, "kind": ANCHOR_KIND, "chain_id": chain_id, "cache_dir": norm_dir(cache_dir), "minted_ns": time.time_ns(), "minted_utc": utc_now(), "key_id": key_id(key),
           "content": {"key": k, "n_files": fp["n_files"], "bytes": fp["bytes"], "content_sha256": fp["content_sha256"], "manifest_file_sha256": fp.get("manifest_file_sha256")},
           "pin": pin, "stat": {"n_files": len(before), "files": before}}
    doc["hmac_sha256"] = sign(doc, key)
    write_json(out_path, doc)
    return doc


def load_anchor(path, key: bytes) -> tuple:
    """(doc | None, reasons, sha256 of the file's bytes).  Refuses an unreadable, wrong-kind, wrong-schema, unsigned / edited or malformed anchor."""
    try:
        raw = Path(path).read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        return None, [f"the anchor {path} is unreadable: {type(e).__name__}: {str(e)[:120]}"], None
    sha = sha256_bytes(raw)
    if not isinstance(doc, dict) or doc.get("kind") != ANCHOR_KIND or doc.get("schema") != ANCHOR_SCHEMA:
        return None, [f"{path} is not a {ANCHOR_KIND} ({ANCHOR_SCHEMA})"], sha
    if not signature_ok(doc, key):
        return None, [f"the anchor's HMAC does not verify -- edited, or signed with another key (anchor key_id {str(doc.get('key_id'))[:20]}, this host's {key_id(key)})"], sha
    st, ct = doc.get("stat"), doc.get("content")
    if not (isinstance(st, dict) and isinstance(st.get("files"), dict) and st["files"] and isinstance(ct, dict) and ct.get("content_sha256") and type(doc.get("minted_ns")) is int
            and all(isinstance(v, list) and len(v) == 2 and all(type(x) is int for x in v) for v in st["files"].values())):
        return None, ["the anchor is malformed (no stat listing / content block / minted_ns)"], sha
    return doc, [], sha


def anchor_binding_refusals(doc: dict, *, cache_dir, chain_id, chain_start_ns) -> list:
    """Why this anchor may not serve THIS call: another cache directory, another chain, or minted BEFORE the chain started."""
    out = []
    if norm_dir(cache_dir) != doc.get("cache_dir"):
        out.append(f"the anchor was minted from another cache directory ({str(doc.get('cache_dir'))[-60:]!r}), this call names {norm_dir(cache_dir)[-60:]!r}")
    if chain_id != doc.get("chain_id"):
        out.append(f"the anchor belongs to another chain (anchor {str(doc.get('chain_id'))[:12]}, this call {str(chain_id)[:12]})")
    if not isinstance(chain_start_ns, int) or doc["minted_ns"] < chain_start_ns:
        out.append(f"the anchor is OLDER than the chain (minted_ns {doc['minted_ns']} < chain start {chain_start_ns}): it cannot vouch for a chain that began after it")
    return out


def anchor_listing_problems(want: dict, got: dict) -> list:
    """Every difference between the anchor's (size, mtime_ns) listing `want` and the live one `got`: a missing file, an extra file, a changed size, a changed mtime_ns."""
    out = [f"file in the anchor is MISSING: {r}" for r in sorted(set(want) - set(got))]
    out += [f"file NOT in the anchor (extra, or a renamed file): {r}" for r in sorted(set(got) - set(want))]
    out += [f"SIZE changed since the anchor: {r} (anchor {want[r][0]} bytes, now {got[r][0]})" for r in sorted(set(want) & set(got)) if want[r][0] != got[r][0]]
    out += [f"MTIME changed since the anchor: {r} (anchor {want[r][1]} ns, now {got[r][1]} ns): the bytes are not re-read at an intermediate call, so a re-timestamped file is refused as a possible edit"
            for r in sorted(set(want) & set(got)) if want[r][1] != got[r][1]]
    return out


def anchor_fingerprint(path, anchor: dict) -> dict:
    """The ANCHOR-BOUND fingerprint of `path` (`anchor` = {"doc": <load_anchor doc>, "sha256": <anchor file sha256>}): the live (relpath, size, mtime_ns) listing is compared with the anchor's and NO BYTE IS READ.
    Valid iff there is no difference at all; then its `fp_key` is the anchor's content key (`dirc:...`), i.e. the registered one.  Any problem -> `n_problems > 0` and the key is `dir-anchor-INVALID:...`."""
    p = Path(path)
    if not p.is_dir():
        return {"kind": "missing"}
    doc = anchor["doc"]
    got, problems = _stat_listing(p)
    problems += anchor_listing_problems(doc["stat"]["files"], got)
    fp = {"kind": "dir-anchor", "mode": CTX_MODE_ANCHOR, "n_problems": len(problems), "problems": problems[:CONTENT_MAX_PROBLEMS], "anchor_sha256": anchor["sha256"], "chain_id": doc["chain_id"],
          "method": "dir-anchor-v1 (relpath + size + mtime_ns of EVERY file compared with the chain's start anchor; NO byte is read; the anchor is one full content fingerprint compared with the registered pin; "
                    "a same-size, same-mtime_ns edit is detected only by the chain's END verify)"}
    if not problems:
        c = doc["content"]
        fp.update(n_files=c["n_files"], bytes=c["bytes"], content_sha256=c["content_sha256"])
    return fp


def end_verify(cache_dir, anchor: dict, key: bytes, out_path, *, seeds=()) -> dict:
    """THE END OF THE CHAIN: a second FULL content fingerprint of `cache_dir`, compared with the START anchor, bracketed by stat listings that must still equal the anchor's.  Writes the signed record -- status
    `END_VERIFY_OK` only if the content key AND every (size, mtime_ns) equal the anchor's, otherwise `VOID` -- and returns it.  A VOID chain's outputs are never read (`ctx_chain_refusals`)."""
    doc = anchor["doc"]
    before, p1 = _stat_listing(cache_dir)
    fp = content_fingerprint(cache_dir)                                                      # the ONE full read of the end
    after, p2 = _stat_listing(cache_dir)
    problems = list(p1 + p2)
    if fp.get("kind") != "dir-content":
        problems.append(f"the cache is {fp.get('kind')} at the end of the chain")
    if fp.get("n_problems"):
        problems.append(f"the cache FAILS its content check at the end of the chain ({fp['n_problems']} problem(s)): " + "; ".join(str(x) for x in (fp.get("problems") or [])[:3]))
    k_end = fp_key(fp)
    content_equal = k_end == doc["content"]["key"]
    if not content_equal:
        problems.append(f"CONTENT differs from the start anchor (start {doc['content']['key'][:40]}, end {k_end[:40]})")
    stat_problems = anchor_listing_problems(doc["stat"]["files"], before) + anchor_listing_problems(doc["stat"]["files"], after)
    stat_equal = not stat_problems and not p1 and not p2
    problems += stat_problems[:5]
    rec = {"schema": END_SCHEMA, "kind": END_KIND, "status": END_OK if not problems else END_VOID, "chain_id": doc["chain_id"], "anchor_sha256": anchor["sha256"], "cache_dir": norm_dir(cache_dir),
           "ended_ns": time.time_ns(), "ended_utc": utc_now(), "seeds": sorted(int(s) for s in seeds), "content_key_start": doc["content"]["key"], "content_key_end": k_end,
           "content_equal": content_equal, "stat_equal": stat_equal, "n_problems": len(problems), "problems": problems[:CONTENT_MAX_PROBLEMS], "key_id": key_id(key)}
    rec["hmac_sha256"] = sign(rec, key)
    write_json(out_path, rec)
    return rec


def ctx_chain_refusals(chain_root, seeds, key: bytes) -> list:
    """THE READER'S GATE.  [] iff every pass in `seeds` carries a signed PASS token whose `--ctx-cache` is bound (a) by CONTENT (`dir-content`: every call re-read the cache), or (b) by an ANCHOR (`dir-anchor`) and
    then the chain's END record is signed, `END_VERIFY_OK`, bound to THE SAME anchor (sha256 of the anchor file), covers every seed asked for, and both its content and its stat comparisons were equal.  Fail closed:
    a missing token, a missing / deleted / VOID / foreign / edited END record, two different anchors, or a listing-only token is a refusal.  Layout under `chain_root`: passes/s<seed>/lg/PASS_*.json,
    ctx_chain/anchor.json, ctx_chain/end_verify.json."""
    import glob
    root = Path(chain_root)
    seeds = sorted({int(s) for s in seeds})
    if not seeds:
        return ["no seed to read: there is no pass token to check the cache binding of"]
    out, anchors = [], {}
    for s in seeds:
        toks = sorted(glob.glob(str(root / "passes" / f"s{s}" / "lg" / "PASS_*.json")))
        if not toks:
            out.append(f"seed {s}: no PASS token under {root / 'passes' / f's{s}' / 'lg'}: the cache binding of this pass cannot be established")
            continue
        try:
            tok = read_json(toks[-1])
        except (OSError, ValueError) as e:
            out.append(f"seed {s}: the token is unreadable ({type(e).__name__})")
            continue
        if not signature_ok(tok, key) or tok.get("verdict") != "PASS" or tok.get("kind") != "launch-gate-token":
            out.append(f"seed {s}: the token is not a signed PASS launch-gate token")
            continue
        d = (((tok.get("binding") or {}).get("data")) or {}).get("--ctx-cache")
        kind = d.get("kind") if isinstance(d, dict) else None
        if kind == "dir-content" and not d.get("n_problems") and d.get("content_sha256"):
            continue
        if kind == "dir-anchor" and d.get("mode") == CTX_MODE_ANCHOR and not d.get("n_problems") and d.get("content_sha256") and re.fullmatch(r"[0-9a-f]{64}", str(d.get("anchor_sha256"))):
            anchors[s] = d["anchor_sha256"]
            continue
        out.append(f"seed {s}: the token binds --ctx-cache as {kind!r}, which is neither a valid content fingerprint (dir-content) nor an anchor-bound one (dir-anchor)")
    if out or not anchors:
        return out
    shas = sorted(set(anchors.values()))
    if len(shas) != 1:
        return [f"the pass tokens are bound to {len(shas)} different anchors ({[x[:12] for x in shas]}): one chain has one anchor"]
    adoc, areasons, asha = load_anchor(root / ANCHOR_REL, key)
    if areasons:
        return areasons
    if asha != shas[0]:
        return [f"the anchor on disk ({asha[:12]}) is not the anchor the tokens are bound to ({shas[0][:12]})"]
    try:
        rec = read_json(root / END_REL)
    except (OSError, ValueError):
        return [f"no END_VERIFY_OK record ({END_REL}): the chain's end verify did not run, or its record was deleted -- the screen's outputs are not read"]
    if not (isinstance(rec, dict) and rec.get("kind") == END_KIND and rec.get("schema") == END_SCHEMA) or not signature_ok(rec, key):
        return [f"{END_REL} is not a signed {END_KIND} record (edited, foreign or malformed)"]
    if rec.get("status") != END_OK:
        return [f"the chain is {rec.get('status')!r}, not {END_OK}: the cache changed under the chain ({'; '.join(str(x) for x in (rec.get('problems') or [])[:2])[:300]})"]
    out = []
    if rec.get("anchor_sha256") != asha or rec.get("chain_id") != adoc["chain_id"]:
        out.append(f"the {END_OK} record is bound to another anchor/chain (record anchor {str(rec.get('anchor_sha256'))[:12]}, the tokens' {asha[:12]})")
    if rec.get("content_equal") is not True or rec.get("stat_equal") is not True or rec.get("content_key_end") != adoc["content"]["key"]:
        out.append(f"the {END_OK} record does not carry equal content AND stat comparisons")
    if not set(seeds) <= {int(x) for x in (rec.get("seeds") or [])}:
        out.append(f"the {END_OK} record covers seeds {rec.get('seeds')}, the read asks for {seeds}")
    if not (type(rec.get("ended_ns")) is int and rec["ended_ns"] >= adoc["minted_ns"]):
        out.append(f"the {END_OK} record is not later than its anchor")
    return out


def flag_pairs(argv: list) -> list:
    """[(flag, [values...])] -- a token starting with `--` opens a flag, the tokens after it are its values."""
    out, cur = [], None
    for t in argv:
        if t.startswith("--"):
            cur = (t, [])
            out.append(cur)
        elif cur is not None:
            cur[1].append(t)
    return out


def flag_values(argv: list, flag: str):
    for f, v in flag_pairs(argv):
        if f == flag:
            return v
    return None


def _timed_fingerprint(path, pool=None, content=False, anchor=None) -> tuple:
    t0 = time.perf_counter()
    if content and anchor is not None:                                                  # ANCHOR-BOUND (HNSCREEN-FP-COST): stat compare against the chain's start anchor, no byte read
        fp = anchor_fingerprint(path, anchor)
    else:
        fp = content_fingerprint(path) if content else fingerprint(path, _pool=pool)       # content: a directory's BYTES are hashed (on its own hash pool); `pool` is the listing's stat pool
    return fp, time.perf_counter() - t0


def data_manifest(argv: list, data_flags: Iterable[str], gate_inputs: dict | None = None, path_map=None, timings: dict | None = None, content_flags: Iterable[str] = (), ctx_anchor: dict | None = None) -> dict:
    """Every input the launch READS, fingerprinted on THIS host: the value of each data flag in the argv, plus the gate's own inputs
    (`gate:*`).  A data flag that is passed and missing on the host is a finding (`data_problems`).  `timings`, when given, is filled {input: seconds}: evidence of where the
    time went, never part of the manifest.  `content_flags`: the flags whose DIRECTORY value is fingerprinted by CONTENT (`content_fingerprint`: every byte re-hashed against the build-time manifest)
    instead of by listing; default (): every flag is fingerprinted exactly as it always was.  `ctx_anchor` (HNSCREEN-FP-COST; default None = every content flag is re-read in full, today's behaviour):
    {"doc", "sha256"} of a chain's start anchor; a content flag is then fingerprinted ANCHOR-BOUND (`anchor_fingerprint`: a stat compare, no byte read)."""
    want = set(data_flags)
    cflags = set(content_flags)
    jobs = []                                                         # (key, extra fields, path to fingerprint, by content?), in the order the manifest has always been built
    for flag, vals in flag_pairs(argv):
        if flag in want and vals:
            local = vals[0]
            for a, b in (path_map or []):
                if local.startswith(a):
                    local = b + local[len(a):]
            jobs.append((flag, {"path": vals[0], **({"local": local} if local != vals[0] else {})}, local, flag in cflags))
    for key, path in sorted((gate_inputs or {}).items()):
        jobs.append((key, {"path": str(path)}, path, False))
    # The inputs are independent: the 1.29 GB checkpoint's sha256 and the 413,184-file image listing do not wait for each other.  Every job still runs (a flag named twice is
    # still fingerprinted twice, the later one wins, a raise still raises); one pool of `fp_threads()` workers serves all the listings.
    nthr = fp_threads()
    man: dict = {}
    if nthr <= 1 or not jobs:
        for key, extra, path, byc in jobs:
            fp, dt = _timed_fingerprint(path, None, byc, ctx_anchor)
            man[key] = {**extra, **fp}
            if timings is not None:
                timings[key] = round(dt, 2)
        return man
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthr, thread_name_prefix="fp-stat") as stat_pool, \
            concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs), 8), thread_name_prefix="fp-job") as job_pool:
        futs = [job_pool.submit(_timed_fingerprint, path, stat_pool, byc, ctx_anchor) for _, _, path, byc in jobs]
        try:
            for (key, extra, _, _), f in zip(jobs, futs):
                fp, dt = f.result()
                man[key] = {**extra, **fp}
                if timings is not None:
                    timings[key] = round(dt, 2)
        finally:
            for f in futs:
                f.cancel()
    return man


def data_problems(data: dict) -> list:
    return [f"data input {k} ({v.get('path', '')}) is {v.get('kind')} on this host" for k, v in sorted(data.items())
            if v.get("kind") in ("missing", "unresolved")] + \
           [f"data input {k} ({v.get('path', '')}) FAILS its content check ({v.get('n_problems')} problem(s)): {'; '.join(str(x) for x in (v.get('problems') or [])[:3])}" for k, v in sorted(data.items())
            if v.get("kind") == "dir-content" and v.get("n_problems")] + \
           [f"data input {k} ({v.get('path', '')}) FAILS its anchor check ({v.get('n_problems')} problem(s)): {'; '.join(str(x) for x in (v.get('problems') or [])[:3])}" for k, v in sorted(data.items())
            if v.get("kind") == "dir-anchor" and v.get("n_problems")]


# ------------------------------------------------------------------------------------------ the tree
def tree_manifest(pkg) -> dict:
    """{relpath: sha256} of the shipped tree: refe/ code/ splits/ under `pkg`, source-like files only, never __pycache__."""
    pkg = Path(pkg)
    out: dict = {}
    for root in TREE_ROOTS:
        base = pkg / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(_TREE_SUFFIX):
                    f = Path(dirpath) / fn
                    out[f.relative_to(pkg).as_posix()] = sha256_file(f)
    return out


def tree_digest(man: dict) -> str:
    return sha256_bytes(canonical_json(man))


def tree_diff(want: dict, got: dict, limit: int = 12) -> dict:
    return {"missing": sorted(set(want) - set(got))[:limit], "extra": sorted(set(got) - set(want))[:limit],
            "changed": sorted(k for k in set(want) & set(got) if want[k] != got[k])[:limit]}


def ship_manifest(pkg, *, commit: str | None = None, git_dir: str | None = None, repo_prefix: str = "") -> dict:
    """DEV BOX: write the manifest the pod's `mint` will be held to.  With `commit`, every shipped file must equal its blob at that
    commit (`git show <commit>:<repo_prefix><path>`): a file that is not landed, or edited since, REFUSES -- so the commit in the token
    names bytes that exist in the repository."""
    man = tree_manifest(pkg)
    if commit:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise GateError("commit must be a full 40-hex sha")
        bad = []
        for rel, sha in sorted(man.items()):
            cmd = ["git"] + (["--git-dir", git_dir] if git_dir else []) + ["show", f"{commit}:{repo_prefix}{rel}"]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                bad.append(f"{rel}: not in {commit[:12]}")
            elif sha256_bytes(r.stdout) != sha:
                bad.append(f"{rel}: differs from its blob at {commit[:12]}")
        if bad:
            raise GateError(f"{len(bad)} shipped file(s) are not the committed bytes: {bad[:6]}")
    return {"schema": "refe.ship_manifest/1", "commit": commit, "created_utc": utc_now(), "files": man,
            "tree_sha256": tree_digest(man)}


# ------------------------------------------------------------------------------------------ mint and verify
def binding_of(profile: str, commit: str | None, tree_sha: str, argv: list) -> dict:
    return {"profile": profile, "commit": commit, "tree_sha256": tree_sha, "argv_sha256": argv_sha256(argv)}


def mint(*, profile: str, commit: str | None, tree_man: dict, argv: list, trainer: str, data: dict, checks: dict, reasons: list,
         key: bytes, out_dir, gate_inputs: dict | None = None, gate_self_sha: str = "", verdict_override: str | None = None):
    """Assemble, sign and write the token.  PASS only when every entry of `checks` is PASS and there is no reason; the file is
    `<verdict>_<commit12 or tree12>.json`, exactly ONE current token per binding (a stale PASS is DEMOTED to superseded/)."""
    tsha = tree_digest(tree_man)
    binding = {**binding_of(profile, commit, tsha, argv), "argv": list(argv), "trainer": trainer, "gate_inputs": dict(gate_inputs or {}),
               "data": data, "data_sha256": sha256_bytes(canonical_json(data))}
    reasons = list(reasons)
    bad = [c for c, v in checks.items() if v.get("status") != "PASS"]
    if bad:
        reasons.append(f"checks not PASS: {bad}")
    verdict = verdict_override or ("PASS" if not reasons else "FAIL")
    token = {"schema": GATE_SCHEMA, "kind": "launch-gate-token", "verdict": verdict, "binding": binding, "checks": checks,
             "reasons": reasons, "gate_sha256": gate_self_sha, "created_utc": utc_now(), "key_id": key_id(key)}
    token["hmac_sha256"] = sign(token, key)
    out = Path(out_dir)
    tag = (commit or "nocommit")[:12] if commit else tsha[:12]
    stamp = utc_now().replace(":", "")
    for other in ("PASS", "FAIL", "INCOMPLETE"):
        op = out / f"{other}_{tag}.json"
        if op.exists():
            (out / "superseded").mkdir(parents=True, exist_ok=True)
            os.replace(op, out / "superseded" / f"{other}_{tag}.{stamp}.json")
    path = write_json(out / f"{verdict}_{tag}.json", token)
    return verdict, path, token


def verify_token(token_path, argv: list, pkg, *, key: bytes, profile: str | None = None, data_flags: Iterable[str] = (),
                 commit: str | None = None, path_map=None, content_flags: Iterable[str] = (), ctx_anchor: dict | None = None):
    """THE SUPERVISOR'S SIDE.  Every field the token binds is RE-MEASURED here; any difference refuses.  Returns (ok, reasons, info)."""
    reasons: list = []
    info: dict = {"token": str(token_path)}
    try:
        tok = read_json(token_path)
    except Exception as e:                                                  # noqa: BLE001 -- a refusal, never a pass
        return False, [f"token unreadable: {type(e).__name__}: {e}"], info
    if tok.get("kind") != "launch-gate-token" or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not signature_ok(tok, key):
        reasons.append(f"HMAC does not verify -- edited, or signed with another key (token key_id {tok.get('key_id')}, this host's "
                       f"{key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "tree_sha256", "argv_sha256")}
    if profile and b.get("profile") != profile:
        reasons.append(f"token profile {b.get('profile')!r} != launch profile {profile!r}")
    if commit and b.get("commit") != commit:
        reasons.append(f"token commit {str(b.get('commit'))[:12]} != launch commit {commit[:12]}")
    a_sha = argv_sha256(argv)
    info["argv_sha256"] = a_sha
    if b.get("argv_sha256") != a_sha or b.get("argv") != argv:
        changed = [f"{x} -> {y}" for x, y in zip(b.get("argv") or [], argv) if x != y][:5]
        reasons.append(f"argv differs from the gated argv (sha {a_sha[:12]} vs {str(b.get('argv_sha256'))[:12]}; "
                       f"{changed or 'length differs'})")
    try:
        t_tree = time.perf_counter()
        tsha = tree_digest(tree_manifest(pkg))
        info["tree_sha256"] = tsha
        info["timing_s"] = {"tree_manifest": round(time.perf_counter() - t_tree, 2)}          # where the time went (evidence only: no verdict reads it, the HMAC does not bind it)
        if tsha != b.get("tree_sha256"):
            reasons.append(f"code tree differs from the gated tree ({tsha[:16]} vs {str(b.get('tree_sha256'))[:16]})")
    except Exception as e:                                                  # noqa: BLE001
        reasons.append(f"cannot measure the code tree: {e}")
    flags = list(data_flags) or sorted({k for k in (b.get("data") or {}) if k.startswith("--")})
    t_data, dtim = time.perf_counter(), {}
    data = data_manifest(argv, flags, b.get("gate_inputs") or {}, path_map, timings=dtim, content_flags=content_flags, ctx_anchor=ctx_anchor)
    if ctx_anchor is not None:                                                  # the call RECORDS which mode it used (content-mode records are exactly what they always were)
        info["ctx_cache"] = {"mode": CTX_MODE_ANCHOR, "anchor_sha256": ctx_anchor["sha256"], "chain_id": ctx_anchor["doc"]["chain_id"]}
    info.setdefault("timing_s", {})["data_manifest"] = round(time.perf_counter() - t_data, 2)
    info["timing_s"]["data_inputs"] = dict(sorted(dtim.items(), key=lambda kv: -kv[1])[:6])
    want = b.get("data") or {}
    d_bad = [k for k in sorted(set(want) | set(data)) if k not in want or k not in data or fp_key(want[k]) != fp_key(data[k])]
    info["data_differs"] = d_bad
    if d_bad:
        reasons.append(f"data differs from the gated data on {d_bad[:8]}")
        reasons += [p_ for p_ in data_problems(data) if "content check" in p_ or "anchor check" in p_]       # WHICH file, when the input that differs is a content- or anchor-fingerprinted directory
    return not reasons, reasons, info


# ------------------------------------------------------------------------------------------ G-SUITE pre-run evidence
# G-SUITE is 22 CPU-only minutes (MEASURED 2026-10-15, P2-s1's launch: GPU free 15:33Z -> TRAIN_START 16:13Z, G-SUITE alone 22 min) that depend only on the shipped tree and the environment, so they
# can run BEFORE the GPU frees.  `launch_gate.py suite-prerun` runs exactly `check_suite(profile)` and writes the result as SIGNED evidence; `launch_gate.py run --suite-evidence <file>` ACCEPTS it in
# place of the live run only if EVERY binding below re-measures equal NOW (`verify_suite_evidence`).  Anything else -> the suite RUNS LIVE, exactly as without the flag, and the token records why.
# A bad evidence file never makes the gate weaker; it only costs time.  The evidence is NOT a token: its schema and kind (both inside the signed body) differ, verify_token refuses it and
# verify_suite_evidence refuses a token.  Wiring: raw/2026-10-16-gate-suite-prerun/WIRING.md; the regression arms and the source-mutation matrix: refe/test_gate_suite_prerun.py.
SUITE_EVIDENCE_SCHEMA = "tanitad.launch_gate.suite_prerun/1"
SUITE_EVIDENCE_KIND = "gate-suite-prerun-evidence"
SUITE_EVIDENCE_MAX_AGE_S = 12 * 3600                 # accepted at <= 12 h, refused beyond
SUITE_EVIDENCE_FUTURE_S = 120                        # an evidence dated more than this far in the FUTURE is refused (clock skew or forgery)
# Environment variables the shipped code READS and that can change a test's outcome (MEASURED 2026-10-16: test_s1c.py failed on the pod on a path AND needed AUX_NAVTEST).  A matching name is bound by a
# short hash of its VALUE (never the value).  Secrets are never read; thread-count and key-file variables are performance / location only and are not bound.
SUITE_ENV_PREFIXES = ("REFE_", "AUX_", "NUPLAN_", "P2REP_", "P6CPU_", "S1C_", "DZ_", "TANITAD_")
SUITE_ENV_NAMES = ("PYTHONPATH", "PYTHONHASHSEED", "PKG")
SUITE_ENV_IGNORE = frozenset({"TANITAD_GATE_ARM", "TANITAD_GATE_KEY_FILE", "TANITAD_GATE_FP_THREADS", "REFE_GATE_NESTED"})
_SECRET_MARKS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def parse_utc(s) -> float | None:
    """Seconds since the epoch of `YYYY-MM-DDTHH:MM:SSZ` (what `utc_now` writes), or None when it is anything else."""
    try:
        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, TypeError, OverflowError):
        return None


def suite_entries_canonical(suite) -> list:
    """The suite entries as the gate runs them: (script, token, args) -- and `raw` for a raw-argv entry -- in order.  Order is meaning."""
    return [{"script": e.get("script"), "token": e.get("token"), "args": [str(x) for x in e.get("args", ())],
             "raw": [str(x) for x in e["raw"]] if "raw" in e else None} for e in suite]


def suite_entries_sha256(suite) -> str:
    return sha256_bytes(canonical_json(suite_entries_canonical(suite)))


def gate_code_shas() -> dict:
    """sha256 of the gate's own code.  `gate_sha256` is launch_gate.py -- the SAME notion `mint` binds (`gate_self_sha`); gate_checks.py (it holds `check_suite`) and this file are bound too."""
    here = Path(__file__).resolve().parent
    return {"gate_sha256": sha256_file(here / "launch_gate.py"), "gate_checks_sha256": sha256_file(here / "gate_checks.py"), "gate_token_sha256": sha256_file(here / "gate_token.py")}


def suite_env(suite_dir) -> dict:
    """What the suite runs IN: the interpreter, the installed distributions (name==version; a `uv pip install` during the wait would change them), the directory the scripts run from (MEASURED 2026-10-16:
    a repo-relative path that indexed fine at a deep dev-box path killed test_s1c on the pod's shallow one), and the value hashes of the environment variables the code reads (SUITE_ENV_*)."""
    import importlib.metadata as md

    def _dist(d):
        try:
            return f"{str(d.metadata['Name']).lower().replace('_', '-')}=={d.version}" if d.metadata["Name"] else None
        except Exception:                                                   # noqa: BLE001 -- a broken dist-info is skipped identically by the pre-run and by the check
            return None
    dists = sorted({x for x in map(_dist, md.distributions()) if x})
    names = sorted(n for n in os.environ if n not in SUITE_ENV_IGNORE and (n in SUITE_ENV_NAMES or n.startswith(SUITE_ENV_PREFIXES)) and not any(m in n.upper() for m in _SECRET_MARKS))
    return {"python": sys.version, "executable": os.path.normcase(os.path.abspath(sys.executable)), "dists_sha256": sha256_bytes(canonical_json(dists)), "n_dists": len(dists),
            "suite_dir": os.path.normcase(os.path.abspath(str(suite_dir))), "vars": {n: sha256_bytes(os.environ[n].encode("utf-8", "replace"))[:16] for n in names}}


def suite_evidence_binding(*, profile_name: str, commit: str | None, pkg, suite, suite_dir) -> tuple:
    """(binding, env): every fact an evidence is held to, MEASURED NOW.  `suite-prerun` takes it before and after the suite; `verify_suite_evidence` takes it when the evidence is offered."""
    env = suite_env(suite_dir)
    return {"profile": profile_name, "commit": commit, "tree_sha256": tree_digest(tree_manifest(pkg)), "suite_sha256": suite_entries_sha256(suite), **gate_code_shas(),
            "env_sha256": sha256_bytes(canonical_json(env))}, env


def make_suite_evidence(*, binding: dict, env: dict, suite_result: dict, key: bytes, arm: str | None = None, extra_reasons: Iterable[str] = ()) -> dict:
    """The SIGNED evidence of one `check_suite` run (`suite_result` is its evidence dict, whole: per-script rc / token / skips / seconds).  `binding` / `env` are what was measured BEFORE the suite ran.
    The verdict is PASS only if the suite passed, no regression arm was active and nothing in `extra_reasons` (e.g. the tree changed while the suite ran)."""
    reasons = list(extra_reasons) + ([f"regression arm {arm} active"] if arm else [])
    verdict = "PASS" if (suite_result.get("status") == "PASS" and not reasons) else "FAIL"
    ev = {"schema": SUITE_EVIDENCE_SCHEMA, "kind": SUITE_EVIDENCE_KIND, "verdict": verdict, "reasons": reasons, "binding": binding, "env": env, "arm": arm or None, "suite": suite_result,
          "created_utc": utc_now(), "key_id": key_id(key)}
    ev["hmac_sha256"] = sign(ev, key)
    return ev


def suite_result_refusals(res) -> list:
    """Why a recorded `check_suite` result is not a clean pass, RE-JUDGED from its raw fields (its own `status` is a claim, the per-script rows are the evidence): every script exited 0, printed its
    token, and not one printed a skip."""
    if not isinstance(res, dict) or not isinstance(res.get("details"), dict) or not isinstance(res["details"].get("scripts"), list):
        return ["the recorded suite result has no per-script rows"]
    out = []
    if res.get("check") != "G-SUITE":
        out.append(f"the recorded result is for check {str(res.get('check'))[:20]!r}, not G-SUITE")
    if res.get("status") != "PASS" or res.get("reasons"):
        out.append(f"the recorded suite status is {str(res.get('status'))[:20]!r} with reasons {[str(x)[:80] for x in list(res.get('reasons') or [])[:2]]}")
    for r in res["details"]["scripts"]:
        r = r if isinstance(r, dict) else {}
        if r.get("rc") != 0 or r.get("token_seen") is not True:
            out.append(f"{str(r.get('script'))[-40:]}: rc {r.get('rc')}, token_seen {r.get('token_seen')}")
        if r.get("skips"):
            out.append(f"{str(r.get('script'))[-40:]}: skip(s) {str(r['skips'][0])[:80]!r} -- a skip is not a pass")
    return out


def verify_suite_evidence(path, *, profile_name: str, commit: str | None, pkg, suite, suite_dir, key: bytes, arm: str | None = None, now: float | None = None,
                          max_age_s: float = SUITE_EVIDENCE_MAX_AGE_S) -> tuple:
    """THE GATE'S SIDE.  Returns (ok, reasons, info); `info["evidence"]` is the PARSED evidence (the bytes that were verified -- the file is read once).  Every field the evidence binds is RE-MEASURED
    here, now, on this host, and ALL differences are collected (not the first).  ok only if: the signature verifies; profile, commit, tree digest (recomputed), suite entries, gate code and
    environment all equal; the age is within `max_age_s`; no regression arm is involved; and the recorded result is PASS with every script rc 0, its token seen and 0 skips.  Any False ->
    the caller RUNS THE SUITE LIVE."""
    reasons: list = []
    info: dict = {"path": str(path)}
    try:
        raw = Path(path).read_bytes()
        info["evidence_sha256"] = sha256_bytes(raw)
        ev = json.loads(raw.decode("utf-8"))
    except Exception as e:                                                  # noqa: BLE001 -- a refusal, never a pass
        return False, [f"suite evidence unreadable: {type(e).__name__}: {str(e)[:120]}"], info
    if not isinstance(ev, dict):
        return False, ["suite evidence is not a JSON object"], info
    info["evidence"] = ev
    if ev.get("kind") != SUITE_EVIDENCE_KIND or ev.get("schema") != SUITE_EVIDENCE_SCHEMA:
        reasons.append(f"not a gate-suite pre-run evidence (kind {str(ev.get('kind'))[:40]!r}, schema {str(ev.get('schema'))[:40]!r})")
    if not signature_ok(ev, key):
        reasons.append(f"HMAC does not verify -- edited, or signed with another key (evidence key_id {str(ev.get('key_id'))[:20]}, this host's {key_id(key)})")
    b = ev.get("binding") if isinstance(ev.get("binding"), dict) else {}
    want, env = suite_evidence_binding(profile_name=profile_name, commit=commit, pkg=pkg, suite=suite, suite_dir=suite_dir)
    info["binding_now"] = want
    if b.get("profile") != want["profile"]:
        reasons.append(f"evidence profile {str(b.get('profile'))[:40]!r} != launch profile {want['profile']!r}")
    if b.get("commit") != want["commit"]:
        reasons.append(f"evidence commit {str(b.get('commit'))[:12]} != launch commit {str(want['commit'])[:12]}")
    if b.get("tree_sha256") != want["tree_sha256"]:
        reasons.append(f"code tree differs from the pre-run's ({want['tree_sha256'][:16]} now vs {str(b.get('tree_sha256'))[:16]})")
    if b.get("suite_sha256") != want["suite_sha256"]:
        reasons.append(f"suite entries differ from the pre-run's ({want['suite_sha256'][:16]} now vs {str(b.get('suite_sha256'))[:16]})")
    for k in ("gate_sha256", "gate_checks_sha256", "gate_token_sha256"):
        if b.get(k) != want[k]:
            reasons.append(f"gate code differs from the pre-run's: {k} {want[k][:16]} now vs {str(b.get(k))[:16]}")
    if b.get("env_sha256") != want["env_sha256"]:
        reasons.append(f"environment differs from the pre-run's (python / installed distributions / suite dir / env vars): now {env['n_dists']} dists, suite dir {env['suite_dir']}")
    created = parse_utc(ev.get("created_utc"))
    t_now = time.time() if now is None else float(now)
    info["age_s"] = None if created is None else round(t_now - created, 1)
    if created is None:
        reasons.append(f"created_utc {str(ev.get('created_utc'))[:30]!r} is not a UTC timestamp")
    elif t_now - created > max_age_s:
        reasons.append(f"evidence is {(t_now - created) / 3600:.2f} h old (limit {max_age_s / 3600:g} h)")
    elif created - t_now > SUITE_EVIDENCE_FUTURE_S:
        reasons.append(f"evidence is dated {created - t_now:.0f} s in the future")
    if arm or ev.get("arm"):
        reasons.append(f"a regression arm is involved (now {arm!r}, evidence {str(ev.get('arm'))[:30]!r}): an arm run is never short-cut")
    if ev.get("verdict") != "PASS":
        reasons.append(f"evidence verdict is {str(ev.get('verdict'))[:20]!r}, not PASS")
    reasons += suite_result_refusals(ev.get("suite"))
    rows = ((ev.get("suite") or {}).get("details") or {}).get("scripts") if isinstance(ev.get("suite"), dict) else None
    if isinstance(rows, list):
        suite = list(suite)
        if len(rows) != len(suite):
            reasons.append(f"the recorded suite ran {len(rows)} script(s), the profile lists {len(suite)}")
        else:
            for r, e in zip(rows, suite):
                if not isinstance(r, dict) or r.get("script") != e.get("script", "raw") or r.get("token") != e.get("token"):
                    reasons.append(f"a recorded script row ({str((r or {}).get('script') if isinstance(r, dict) else r)[-40:]}) is not the profile's entry {str(e.get('script', 'raw'))[-40:]}")
    return not reasons, reasons, info
