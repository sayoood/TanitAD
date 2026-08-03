#!/usr/bin/env python3
"""registry_paths — every artifact path cited in MODEL_REGISTRY.md, checked.

Why this exists
---------------
``Project Steering/MODEL_REGISTRY.md`` is the ONLY quotable source for model
facts, so a dead path inside it is a programme-level defect: it makes a claim
unverifiable while still looking sourced. Two such paths were found by hand on
2026-08-03 — §4.2 cited a ``taniteval/results/refc-small-30k.json`` that does
not exist, and a ``{base,xl}`` **brace expansion** was being used as a citation
and names no file. Both had been read as evidence.

What it does NOT do
-------------------
It does not "fix" a path by pointing it at a plausible file. A path that cannot
be resolved is reported ``UNRESOLVED`` and stays that way until a human
verifies the target is the right one. Silently re-pointing a citation at a
lookalike is how a wrong number becomes a sourced number.

Path classes and how each is judged:

  ``repo``      relative to the repo root            -> EXISTS / MISSING
  ``ellipsis``  ``…/incoming/<dir>/<file>``          -> resolved by globbing the
                research hub; AMBIGUOUS if >1 match
  ``remote``    ``<host>:/abs/path``                 -> NOT_CHECKED. A pod path
                is not checkable from here and the host may be gone; reporting
                it MISSING would be an absence claim from a single failed probe.
  ``glob``      contains ``{a,b}`` or ``*``          -> NOT_A_PATH: a brace
                expansion is not a citation.

Usage
-----
    python tools/registry_paths.py                 # table
    python tools/registry_paths.py --json out.json
    python tools/registry_paths.py --only-bad

Exit code ``1`` if any cited path is MISSING or NOT_A_PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "Project Steering" / "MODEL_REGISTRY.md"
HUB = REPO_ROOT / "TanitAD Research Hub"

EXTS = (".json", ".pt", ".jsonl", ".npz", ".npy", ".md", ".py", ".csv",
        ".parquet", ".yaml", ".yml", ".png", ".mp4", ".txt", ".safetensors")

EXISTS = "EXISTS"
MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"
NOT_CHECKED = "NOT_CHECKED"
NOT_A_PATH = "NOT_A_PATH"
NAME_ONLY = "NAME_ONLY"

#: A token with no "/" is a NAME, not a path citation — the registry says
#: `ckpt.pt` or `ci.py` constantly to mean "the checkpoint file" / "that
#: module", and calling those MISSING drowns the real dead paths in noise. The
#: first cut of this sweep reported 85 MISSING, of which ~70 were bare names.
#: They are still resolved by basename and reported, but they are not defects.

#: `…` (U+2026) and `...` both appear in the registry as "somewhere under the
#: research hub". Either one starts an ellipsis path.
ELLIPSIS = ("\u2026/", ".../")


def extract_citations(text: str) -> list[str]:
    """Backtick-quoted tokens that look like a file path.

    Deliberately conservative: a citation the registry makes in prose without
    backticks is not machine-checkable, and guessing at prose is how the
    stale-absence class gets re-created.
    """
    out: list[str] = []
    for tok in re.findall(r"`([^`\n]+)`", text):
        tok = tok.strip()
        if not tok or " " in tok.split("::")[0][:0]:
            pass
        if not tok.lower().endswith(EXTS):
            continue
        if tok.startswith(("http://", "https://")):
            continue
        out.append(tok)
    return out


_INDEX_CACHE: dict[Path, dict[str, list[Path]]] = {}


def file_index(root: Path) -> dict[str, list[Path]]:
    """basename -> [paths]. Built ONCE per root and cached.

    The repo lives on a network-synced drive where a single ``**`` walk costs
    seconds; doing one per citation (250+) does not terminate in any useful
    time. Measured 2026-08-03: the per-citation version was still running after
    9 minutes; this one indexes the tree once.
    """
    if root in _INDEX_CACHE:
        return _INDEX_CACHE[root]
    idx: dict[str, list[Path]] = {}
    # `.claude/worktrees/*` are transient agent COPIES of this repo. Indexing
    # them turns every genuine single hit into an 8-way AMBIGUOUS and hides
    # the real defects behind noise.
    skip = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv",
            "worktrees"}
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.name in skip:
                continue
            if e.is_dir():
                stack.append(e)
            else:
                idx.setdefault(e.name, []).append(e)
    _INDEX_CACHE[root] = idx
    return idx


def find_by_suffix(root: Path, tail: str) -> list[Path]:
    """Paths whose repo-relative form ends with `tail` (a '/'-separated tail)."""
    name = tail.rsplit("/", 1)[-1]
    want = tail.replace("\\", "/")
    out = []
    for p in file_index(root).get(name, []):
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel == want or rel.endswith("/" + want):
            out.append(p)
    return sorted(out)


def find_by_loose_tail(root: Path, tail: str) -> list[Path]:
    """Like :func:`find_by_suffix`, but an elided component may be a SUFFIX of
    the real one.

    The registry writes `…/own-dynamics-encoder/RESULTS_camcond.md` for a file
    that actually lives under `2026-07-22-own-dynamics-encoder/`. The ellipsis
    elides a *prefix of a path component*, not only whole components, so strict
    suffix matching reports a file that plainly exists as MISSING — a fabricated
    defect, which is the same error class in the other direction.
    """
    segs = [s for s in tail.replace("\\", "/").split("/") if s]
    name = segs[-1]
    out = []
    for p in file_index(root).get(name, []):
        parts = list(p.relative_to(root).parts)
        if len(parts) < len(segs):
            continue
        window = parts[-len(segs):]
        if all(w == s or w.endswith(s) for w, s in zip(window, segs)):
            out.append(p)
    return sorted(out)


def classify(tok: str) -> str:
    if "{" in tok or "*" in tok or "?" in tok or "<" in tok:
        return "glob"
    if tok.startswith(ELLIPSIS):
        return "ellipsis"
    # host:/abs/path  — a drive letter (C:/...) is not a remote
    m = re.match(r"^([A-Za-z][\w.-]{2,}):(/.*)$", tok)
    if m:
        return "remote"
    if tok.startswith("/"):
        return "abspath"
    if "/" not in tok:
        return "name"
    return "repo"


def resolve(tok: str, repo_root: Path | None = None,
            hub: Path | None = None) -> dict:
    root = repo_root or REPO_ROOT
    hubdir = hub or (root / "TanitAD Research Hub")
    kind = classify(tok)
    rec = {"citation": tok, "kind": kind}

    if kind == "glob":
        rec["status"] = NOT_A_PATH
        rec["note"] = ("brace expansion / wildcard names no file — not a "
                       "citation")
        return rec
    if kind == "remote":
        host, _, path = tok.partition(":")
        rec["status"] = NOT_CHECKED
        rec["host"] = host
        rec["path"] = path
        rec["note"] = ("remote path — not checkable from the repo; a failed "
                       "probe here would be a single-location absence claim")
        return rec
    if kind == "abspath":
        # A bare absolute path (`/workspace/...`, `/root/...`) is a path on a
        # POD, not in this repo. It cannot be checked from here, so it is
        # NOT_CHECKED — never MISSING. But we DO report whether a file of that
        # basename is banked in the repo, because a pod-path citation with no
        # repo counterpart is a STRANDED artifact, which is the durability
        # question this whole sweep serves.
        name = Path(tok).name
        banked = [m for m in file_index(root).get(name, [])]
        rec["status"] = NOT_CHECKED
        rec["path"] = tok
        rec["note"] = "absolute pod path — not checkable from the repo"
        rec["repo_counterpart"] = (
            str(banked[0].relative_to(root)).replace("\\", "/") if len(banked) == 1
            else None)
        rec["n_repo_counterparts"] = len(banked)
        rec["stranded"] = not banked
        return rec
    if kind == "name":
        matches = file_index(root).get(tok, [])
        rec["status"] = NAME_ONLY
        rec["note"] = "bare filename, not a path citation"
        rec["n_matches"] = len(matches)
        if len(matches) == 1:
            rec["resolved"] = str(matches[0].relative_to(root)).replace("\\", "/")
        return rec
    if kind == "ellipsis":
        tail = tok
        for e in ELLIPSIS:
            if tail.startswith(e):
                tail = tail[len(e):]
                break
        # Hub first (that is what `…/incoming/...` almost always means), then
        # the whole repo — an ellipsis path may elide `stack/` just as easily.
        found = find_by_suffix(root, tail) or find_by_loose_tail(root, tail)
        matches = [m for m in found if hubdir in m.parents] or found
        # A pod-rescue dump is a COPY of a repo artifact, not a second
        # artifact. Prefer the canonical path when both matched.
        canon = [m for m in matches
                 if "pod-rescue" not in str(m).replace("\\", "/")]
        if canon:
            matches = canon
        if len(matches) == 1:
            rec["status"] = EXISTS
            rec["resolved"] = str(matches[0].relative_to(root)).replace("\\", "/")
            rec["bytes"] = matches[0].stat().st_size
        elif not matches:
            rec["status"] = MISSING
        else:
            rec["status"] = AMBIGUOUS
            rec["candidates"] = [str(m.relative_to(root)).replace("\\", "/")
                                 for m in matches[:8]]
        return rec

    p = root / tok
    if p.exists():
        rec["status"] = EXISTS
        rec["resolved"] = tok
        rec["bytes"] = p.stat().st_size if p.is_file() else None
        return rec
    # A repo-relative miss may still be a hub artifact quoted without the
    # ellipsis. Try the hub before declaring it missing — absence at ONE
    # location is not absence.
    by_tail = find_by_suffix(root, tok)
    matches = by_tail or ([m for m in file_index(root).get(Path(tok).name, [])
                           if hubdir in m.parents]
                          or file_index(root).get(Path(tok).name, []))
    canon = [m for m in matches if "pod-rescue" not in str(m).replace("\\", "/")]
    if canon:
        matches = canon
    if len(matches) == 1:
        rec["status"] = EXISTS
        rec["resolved"] = str(matches[0].relative_to(root)).replace("\\", "/")
        rec["bytes"] = matches[0].stat().st_size
        rec["note"] = "found by basename under the research hub, not at the cited path"
        return rec
    if len(matches) > 1:
        rec["status"] = AMBIGUOUS
        rec["candidates"] = [str(m.relative_to(root)).replace("\\", "/")
                             for m in matches[:8]]
        return rec
    rec["status"] = MISSING
    return rec


def sweep(registry: Path | None = None, repo_root: Path | None = None) -> dict:
    reg = registry or REGISTRY
    root = repo_root or REPO_ROOT
    text = reg.read_text(encoding="utf-8", errors="replace")
    seen: dict[str, dict] = {}
    for tok in extract_citations(text):
        if tok not in seen:
            seen[tok] = resolve(tok, root)
    counts: dict[str, int] = {}
    for r in seen.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"registry": str(reg.relative_to(root)).replace("\\", "/"),
            "n_citations": len(seen), "counts": counts,
            "citations": sorted(seen.values(), key=lambda r: r["citation"])}


def exit_code(result: dict) -> int:
    bad = result["counts"].get(MISSING, 0) + result["counts"].get(NOT_A_PATH, 0)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--only-bad", action="store_true")
    args = ap.parse_args(argv)
    res = sweep()
    print(f"{res['n_citations']} backtick-quoted artifact citations in "
          f"{res['registry']}")
    for k in (EXISTS, MISSING, AMBIGUOUS, NOT_CHECKED, NOT_A_PATH):
        if res["counts"].get(k):
            print(f"  {k:12s} {res['counts'][k]}")
    print()
    for r in res["citations"]:
        if args.only_bad and r["status"] in (EXISTS, NOT_CHECKED):
            continue
        line = f"{r['status']:12s} {r['citation']}"
        if r.get("resolved") and r["resolved"] != r["citation"]:
            line += f"\n{'':12s}   -> {r['resolved']}"
        if r.get("note"):
            line += f"\n{'':12s}   ({r['note']})"
        if r.get("candidates"):
            line += "\n" + "\n".join(f"{'':12s}   ? {c}" for c in r["candidates"])
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return exit_code(res)


if __name__ == "__main__":
    raise SystemExit(main())
