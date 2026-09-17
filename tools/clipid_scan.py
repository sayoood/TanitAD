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
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

__all__ = ["UUID_RE", "scan_text", "scan_tree", "compare_to_baseline",
           "LeakGrew", "DEFAULT_GLOBS"]

#: A canonical UUID. ⛔ Word-anchored so a longer hex run is not a match.
UUID_RE = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-fA-F])")

#: An 8-hex token that is not part of a longer hex run.
_P8 = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")

DEFAULT_GLOBS = ("Project Steering/**/*.md", "TanitAD Research Lab/**/*.md")


class LeakGrew(RuntimeError):
    """A file gained identifiers, or a new file arrived carrying them."""


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


def compare_to_baseline(current: dict, baseline: dict) -> dict:
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
        for kind in ("uuids", "prefixes"):
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
    a = ap.parse_args(argv)
    prefixes = None
    if a.clips:
        prefixes = {l.strip()[:8] for l in Path(a.clips).read_text(
            encoding="utf-8").split() if l.strip()}
    cur = scan_tree(a.root, clip_prefixes=prefixes)
    if a.write_baseline:
        Path(a.baseline).write_text(
            json.dumps(cur, indent=1, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        print(f"baseline written: {len(cur)} entries")
        return 0
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    rep = compare_to_baseline(cur, base)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
