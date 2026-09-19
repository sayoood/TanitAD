"""Stop the identifier count GROWING. Grandfather what is already banked.

**PI ruling 2026-09-17**, asked and answered: the sha12 rule is about **hygiene and
reproducibility**, not confidentiality. ⇒ redacting 53 files of banked evidence
would buy **appearance only** — git history keeps every one of them regardless —
so the right move is a guard that stops the count growing.

MEASURED 2026-09-17 across ``Project Steering/**/*.md`` and
``TanitAD Research Lab/**/*.md``: **64 full clip UUIDs and 530 eight-char prefixes
across 53 files**. A guard that simply refused all of them would be **red from its
first run and switched off within a day**, which is worse than no guard.

## What is checked, and why this shape

⭐ **A bare UUID in a repo artifact is an UNSTABLE HANDLE.** ``sha12`` is
``sha256(clip_id)[:12]`` — a *hash*, stable, and the identity every GT file already
stores. That is the whole reproducibility argument, and it applies to any UUID in
prose, not only to clip ids. So the always-runnable half needs **no corpus list**
and cannot rot when one moves.

⛔ **THE BASELINE STORES COUNTS, NEVER IDENTIFIERS.** An allowlist of the leaked ids
would itself be a list of clip ids — the exact thing this guard exists to stop
growing. Per file it records *how many*, and nothing else.

The optional second half (``--clips``) counts 8-char prefixes against a known clip
list. It is optional precisely because it needs external data; the UUID half is the
one that always runs.

⭐ **Its floor is recorded against ONE named list** (2026-09-19: the 4,719-clip v7
corpus, sha256 ``a48251e89c7a8603…``, 530 prefixes). The baseline keeps the list's size
and digest, never the list, and a run with any other list is REFUSED
(``ClipListMismatch``): the same tree reads 530 against the corpus and 870 against the
306,152-clip index, so a cross-list count is meaningless. Rebuild the list in any order
from the local mirror's mp4 names (``tanitad-data/physicalai/camera/
camera_front_wide_120fov/<clip id>.mp4``) or the corpus's ``camera/camera_sha256.json``
keys; both reproduce the digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

__all__ = ["UUID_RE", "scan_text", "scan_tree", "compare_to_baseline",
           "LeakGrew", "DEFAULT_GLOBS", "ClipListMismatch", "list_identity"]

#: A canonical UUID. ⛔ Word-anchored so a longer hex run is not a match.
UUID_RE = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])")

#: An 8-hex token that is not part of a longer hex run.
_P8 = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")

DEFAULT_GLOBS = ("Project Steering/**/*.md", "TanitAD Research Lab/**/*.md")


class LeakGrew(RuntimeError):
    """A file gained identifiers, or a new file arrived carrying them."""


class ClipListMismatch(RuntimeError):
    """The prefix half was run against a different clip list than its floor was.

    ⛔ Added 2026-09-19 with the first recorded prefix floor. A prefix count is only
    meaningful against the list it was counted with: the same tree reads 530 prefixes
    against the 4,719-clip v7 corpus and 870 against the 306,152-clip index. Comparing
    across lists manufactures growth (or hides it), so the comparison is REFUSED.
    """


def list_identity(ids) -> dict:
    """``{"n", "sha256_sorted"}`` — the list's identity WITHOUT the list.

    ``sha256("\\n".join(sorted(set(ids))))`` — the programme's existing corpus-id
    convention (the v7 corpus's `corpus_id_sha256_16` is its first 16 hex). A digest
    names the list and verifies it; it cannot be turned back into a single clip id.
    """
    s = sorted(set(ids))
    return {"n": len(s), "sha256_sorted": hashlib.sha256("\n".join(s).encode("utf-8")).hexdigest()}


def _read_any(p: Path):
    """utf-8, then cp1252, then utf-8 with replacement — or ``None``.

    ⚠️ This box writes some banked markdown in **cp1252** (a real file in this
    repo carries byte ``0x97``, an em-dash, and is not valid utf-8). Refusing it
    outright would make the guard permanently red on a file nobody is going to
    re-encode, and a permanently red guard gets switched off.

    ⭐ Replacement is SAFE FOR THIS SCAN specifically: a UUID is ASCII hex, and a
    byte that failed to decode becomes U+FFFD, which cannot form one. So a
    replaced character can hide nothing the patterns look for. ⛔ That argument
    does NOT generalise to other scanners — a UTF-16 file would decode to
    interleaved nulls and under-count, which is why an undecodable file is still
    reported rather than silently accepted.
    """
    for enc in ("utf-8", "cp1252"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def scan_text(text: str, clip_prefixes=None) -> dict:
    """``{"uuids": n, "prefixes": n}`` for one document."""
    n_uuid = len(UUID_RE.findall(text))
    n_pre = 0
    if clip_prefixes:
        n_pre = sum(1 for m in _P8.finditer(text) if m.group(0) in clip_prefixes)
    return {"uuids": n_uuid, "prefixes": n_pre}


def scan_tree(root, globs=DEFAULT_GLOBS, clip_prefixes=None) -> dict:
    """``{relative path: counts}`` for every file that carries any identifier.

    ⛔ A file that cannot be READ is reported under ``"_unreadable"`` rather than
    counted as clean: a count of 0 from a file that could not be opened is
    indistinguishable from a genuine absence, and this repo has been bitten by
    exactly that on a flaky mount.
    """
    root = Path(root)
    out, unreadable = {}, []
    for g in globs:
        for p in sorted(root.glob(g)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel in out:
                continue
            t = _read_any(p)
            if t is None:
                unreadable.append(rel)
                continue
            c = scan_text(t, clip_prefixes)
            if c["uuids"] or c["prefixes"]:
                out[rel] = c
    if unreadable:
        out["_unreadable"] = sorted(unreadable)
    return out


def compare_to_baseline(current: dict, baseline: dict, prefixes_counted: bool = True) -> dict:
    """Refuse GROWTH, allow what is grandfathered.

    ⛔ Three ways to fail, and the third is the one a naive check misses:

    1. a file **not in the baseline** carries identifiers — a new leak;
    2. a file's count **exceeds** its baseline — an existing file gained some;
    3. any file was **unreadable** — the scan cannot claim a clean result over
       documents it never opened.

    ⭐ A count that FALLS is fine and is reported, not refused: somebody cleaning
    a file up must never be blocked by the guard that protects the cleanup.
    """
    cur = {k: v for k, v in current.items() if k != "_unreadable"}
    # ⛔ Named distinctly from `scan_tree`'s local. Both functions had a guard
    # reading `if unreadable:`, so the mutation audit removed the WRONG one and
    # reported this branch ESCAPED — the same duplicate-anchor trap that hit
    # `if wrong:` an hour earlier. An audit that mutates the wrong line proves
    # nothing, so every guard gets a unique anchor.
    could_not_read = current.get("_unreadable") or []
    if could_not_read:
        raise LeakGrew(
            f"{len(could_not_read)} file(s) could not be read, so this scan "
            f"cannot report a clean result: {could_not_read[:5]}. A zero from an "
            f"unopened file is indistinguishable from a genuine absence.")
    new_files, grew, shrank = [], [], []
    for rel, c in sorted(cur.items()):
        b = baseline.get(rel)
        if b is None:
            new_files.append(f"{rel} (+{c['uuids']} uuid, +{c['prefixes']} prefix)")
            continue
        # ⚠️ Without a clip list the prefix half was never COUNTED, so every current
        # prefix count is 0 — comparing it would report a recorded floor as a "cleanup".
        for kind in (("uuids", "prefixes") if prefixes_counted else ("uuids",)):
            if c[kind] > b.get(kind, 0):
                grew.append(f"{rel}:{kind} {b.get(kind, 0)} -> {c[kind]}")
            elif c[kind] < b.get(kind, 0):
                shrank.append(f"{rel}:{kind} {b.get(kind, 0)} -> {c[kind]}")
    if new_files or grew:
        raise LeakGrew(
            "identifiers were ADDED to the record. ⛔ Use sha12 "
            "(`sha256(clip_id)[:12]`) — a UUID in prose is an unstable handle and "
            "the reason this guard exists.\n"
            + ("  new files: " + "; ".join(new_files) + "\n" if new_files else "")
            + ("  grew: " + "; ".join(grew) if grew else ""))
    return {"files_with_ids": len(cur),
            "total_uuids": sum(c["uuids"] for c in cur.values()),
            "total_prefixes": sum(c["prefixes"] for c in cur.values()),
            "shrank": shrank, "verdict": "NO-GROWTH"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--clips", default=None,
                    help="optional file of clip ids, one per line, enabling the "
                         "8-char prefix half")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record today's counts as the grandfathered floor")
    ap.add_argument("--clips-name", default="",
                    help="a human label recorded beside the list's digest when writing")
    a = ap.parse_args(argv)
    prefixes, ident = None, None
    if a.clips:
        # utf-8-sig: a list written by PowerShell carries a BOM, which would otherwise
        # glue itself to the first id and turn the right list into a "different" one.
        ids = [l.strip() for l in Path(a.clips).read_text(encoding="utf-8-sig").split()
               if l.strip()]
        prefixes = {i[:8] for i in ids}
        ident = list_identity(ids)
    cur = scan_tree(a.root, clip_prefixes=prefixes)
    if a.write_baseline:
        # ⛔ A floor recorded over files that were never read under-counts, and the
        # first real edit to one of them would then read as a leak.
        if cur.get("_unreadable"):
            raise LeakGrew(f"refusing to write a baseline: {len(cur['_unreadable'])} "
                           f"file(s) could not be read: {cur['_unreadable'][:5]}")
        out = dict(cur)
        if ident:
            # the list is NAMED by its digest, never stored — see `list_identity`
            out["_clips_list"] = dict(ident, name=a.clips_name)
        Path(a.baseline).write_text(
            json.dumps(out, indent=1, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        print(f"baseline written: {len(cur)} entries"
              + (f", prefix floor against a {ident['n']}-id list" if ident else ""))
        return 0
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    if ident:
        rec = base.get("_clips_list")
        if rec is None:
            raise ClipListMismatch(
                "this baseline records NO prefix floor, so the prefix half has nothing "
                "to compare against. Record one with --write-baseline --clips <list>.")
        if (rec.get("sha256_sorted"), rec.get("n")) != (ident["sha256_sorted"], ident["n"]):
            raise ClipListMismatch(
                f"the prefix floor was recorded against the list {rec.get('name') or '?'} "
                f"(n={rec.get('n')}, sha256 {str(rec.get('sha256_sorted'))[:16]}…); this run's "
                f"list is n={ident['n']}, sha256 {ident['sha256_sorted'][:16]}…. Prefix counts "
                f"are only comparable against the SAME list.")
    rep = compare_to_baseline(cur, {k: v for k, v in base.items() if not k.startswith("_")},
                              prefixes_counted=bool(ident))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
