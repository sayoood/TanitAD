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

The sweep that followed found **17** more brace/glob citations and 4 malformed
ones. That is the real lesson: **the defects accumulated invisibly**, in the one
document every reader is required to quote from. A one-shot audit does not fix
that — a standing check does. Hence the allowlist ratchet below.

What it does NOT do
-------------------
It does not "fix" a path by pointing it at a plausible file. A path that cannot
be resolved is reported and stays that way until a human verifies the target is
the right one. Silently re-pointing a citation at a lookalike is how a wrong
number becomes a sourced number.

Path classes and how each is judged:

  ``repo``      relative to the repo root            -> EXISTS / MISSING
  ``ellipsis``  ``…/incoming/<dir>/<file>``          -> resolved by globbing the
                research hub; AMBIGUOUS if >1 match
  ``remote``    ``<host>:/abs/path``                 -> NOT_CHECKED. A pod path
                is not checkable from here and the host may be gone; reporting
                it MISSING would be an absence claim from a single failed probe.
  ``glob``      contains ``{a,b}`` or ``*``          -> NOT_A_PATH: a brace
                expansion is not a citation.

The allowlist — why a defect list needs a ratchet
-------------------------------------------------
A handful of ``NOT_A_PATH`` tokens are **not** defects: the registry legitimately
writes ``ep_*.pt`` to state a *naming contract*, quotes ``git status`` output
verbatim, and quotes a brace expansion *as the worked example of the rule that
bans it*. Those live in ``tools/registry_paths_allow.json`` with a reason each.

Two properties stop that allowlist from becoming the rug the defects sweep under:

1. **Every entry declares how many times it occurs** in the registry. Re-introduce
   the token at a new site and the count no longer matches -> ``ALLOW_COUNT_MISMATCH``
   -> exit 1. An entry whose token has disappeared is ``ALLOW_STALE`` -> exit 1.
2. **``unresolved`` entries are ratcheted.** A citation that is genuinely dead and
   genuinely unfixable (the artifact is on a terminated host and in no backup) is
   recorded as ``unresolved`` — visible forever, never silently closed — and the
   file declares ``max_unresolved``. Adding one without raising that number fails
   the check, and raising it is a one-line reviewable diff.

Usage
-----
    python tools/registry_paths.py                 # table
    python tools/registry_paths.py --json out.json
    python tools/registry_paths.py --only-bad
    python tools/registry_paths.py --strict        # AMBIGUOUS + UNRESOLVED also fail

Exit codes
----------
  ``0``  clean
  ``1``  a hard defect: MISSING, NOT_A_PATH, a stale/miscounted allowlist entry,
         or more ``unresolved`` entries than the declared ratchet allows
  ``2``  no hard defect, but known-UNRESOLVED citations remain (CI may warn on
         this and fail on 1); ``--strict`` turns it into 1
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
ALLOWLIST = Path(__file__).resolve().parent / "registry_paths_allow.json"

EXTS = (".json", ".pt", ".jsonl", ".npz", ".npy", ".md", ".py", ".csv",
        ".parquet", ".yaml", ".yml", ".png", ".mp4", ".txt", ".safetensors")

EXISTS = "EXISTS"
MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"
NOT_CHECKED = "NOT_CHECKED"
NOT_A_PATH = "NOT_A_PATH"
NAME_ONLY = "NAME_ONLY"
#: allowlisted, ``status: pattern`` — a naming convention / command template /
#: verbatim quotation of tool output. Never was a citation; not a defect.
PATTERN = "PATTERN"
#: allowlisted, ``status: unresolved`` — a REAL defect that cannot be repaired
#: because the artifact is unreachable. Kept visible and counted, never closed.
UNRESOLVED = "UNRESOLVED"

#: allowlist bookkeeping failures — both are hard defects
ALLOW_STALE = "ALLOW_STALE"
ALLOW_COUNT_MISMATCH = "ALLOW_COUNT_MISMATCH"

_ALLOW_STATUS = {"pattern": PATTERN, "unresolved": UNRESOLVED}

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
        if not tok:
            continue
        if not tok.lower().endswith(EXTS):
            continue
        if tok.startswith(("http://", "https://")):
            continue
        out.append(tok)
    return out


def count_occurrences(text: str, tok: str) -> int:
    """How many times ``tok`` appears in the registry as a backticked token.

    Exact, not fuzzy: the allowlist ratchet is only meaningful if re-introducing
    the same token at a NEW site changes this number.
    """
    return text.count("`" + tok + "`")


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


# --------------------------------------------------------------------------
# allowlist
# --------------------------------------------------------------------------

def load_allow(path: Path | None = None) -> dict:
    """Read the allowlist. A MISSING allowlist is not an error — it means
    'nothing is excused', which is the strictest state, not a broken one."""
    p = path or ALLOWLIST
    if not p.exists():
        return {"entries": [], "max_unresolved": 0}
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("entries", [])
    data.setdefault("max_unresolved", 0)
    for e in data["entries"]:
        st = e.get("status")
        if st not in _ALLOW_STATUS:
            raise ValueError(
                f"allowlist entry {e.get('citation')!r} has status {st!r}; "
                f"must be one of {sorted(_ALLOW_STATUS)}")
        if not e.get("reason"):
            raise ValueError(
                f"allowlist entry {e.get('citation')!r} has no reason. "
                "An excuse with no reason teaches nobody anything.")
        if "occurrences" not in e:
            raise ValueError(
                f"allowlist entry {e.get('citation')!r} declares no "
                "'occurrences'; without it the token can be re-introduced at a "
                "new site and the check will not notice.")
    return data


def apply_allow(citations: dict[str, dict], text: str, allow: dict) -> list[dict]:
    """Re-label allowlisted citations and return the allowlist's own problems."""
    issues: list[dict] = []
    for e in allow["entries"]:
        tok = e["citation"]
        seen = count_occurrences(text, tok)
        if seen == 0:
            issues.append({"status": ALLOW_STALE, "citation": tok,
                           "note": ("allowlisted but no longer cited in the "
                                    "registry — delete the entry")})
            continue
        if seen != e["occurrences"]:
            issues.append({"status": ALLOW_COUNT_MISMATCH, "citation": tok,
                           "expected": e["occurrences"], "found": seen,
                           "note": ("this token is excused at a declared number "
                                    "of sites; the count changed")})
        rec = citations.get(tok)
        if rec is not None:
            rec["status"] = _ALLOW_STATUS[e["status"]]
            rec["allow_reason"] = e["reason"]
            rec["allow_added"] = e.get("added")
            rec["occurrences"] = seen
    return issues


EMPTY_ALLOW = {"entries": [], "max_unresolved": 0}


def sweep(registry: Path | None = None, repo_root: Path | None = None,
          allow: dict | None = None, allow_path: Path | None = None) -> dict:
    """Sweep a registry.

    The allowlist is resolved in this order, and the last rule is the one that
    matters: an explicit ``allow``/``allow_path`` wins; otherwise the REAL
    allowlist applies only to the REAL registry. A synthetic registry (every
    test fixture) gets an EMPTY allowlist — excusing a token in a fixture it
    does not contain would report the whole allowlist as stale, and a check that
    cries wolf on its own fixtures gets switched off.
    """
    reg = registry or REGISTRY
    root = repo_root or REPO_ROOT
    text = reg.read_text(encoding="utf-8", errors="replace")
    if allow is None:
        if allow_path is not None:
            allow = load_allow(allow_path)
        elif reg.resolve() == REGISTRY.resolve():
            allow = load_allow()
        else:
            allow = dict(EMPTY_ALLOW)
    seen: dict[str, dict] = {}
    for tok in extract_citations(text):
        if tok not in seen:
            seen[tok] = resolve(tok, root)
    allow_issues = apply_allow(seen, text, allow)
    counts: dict[str, int] = {}
    for r in seen.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for i in allow_issues:
        counts[i["status"]] = counts.get(i["status"], 0) + 1
    n_unresolved = counts.get(UNRESOLVED, 0)
    ratchet = allow.get("max_unresolved", 0)
    return {"registry": str(reg.relative_to(root)).replace("\\", "/"),
            "n_citations": len(seen), "counts": counts,
            "allow_issues": allow_issues,
            "n_unresolved": n_unresolved,
            "max_unresolved": ratchet,
            "ratchet_exceeded": n_unresolved > ratchet,
            "ratchet_loose": n_unresolved < ratchet,
            "citations": sorted(seen.values(), key=lambda r: r["citation"])}


def exit_code(result: dict, strict: bool = False) -> int:
    """1 = hard defect. 2 = only known-UNRESOLVED remain. 0 = clean."""
    hard = (result["counts"].get(MISSING, 0)
            + result["counts"].get(NOT_A_PATH, 0)
            + result["counts"].get(ALLOW_STALE, 0)
            + result["counts"].get(ALLOW_COUNT_MISMATCH, 0))
    if result.get("ratchet_exceeded"):
        hard += 1
    if strict:
        hard += result["counts"].get(AMBIGUOUS, 0)
        hard += result["counts"].get(UNRESOLVED, 0)
    if hard:
        return 1
    return 2 if result["counts"].get(UNRESOLVED, 0) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--only-bad", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="AMBIGUOUS and known-UNRESOLVED citations also fail")
    args = ap.parse_args(argv)
    res = sweep()
    print(f"{res['n_citations']} backtick-quoted artifact citations in "
          f"{res['registry']}")
    for k in (EXISTS, MISSING, AMBIGUOUS, NOT_CHECKED, NOT_A_PATH, PATTERN,
              UNRESOLVED, NAME_ONLY, ALLOW_STALE, ALLOW_COUNT_MISMATCH):
        if res["counts"].get(k):
            print(f"  {k:20s} {res['counts'][k]}")
    print(f"  {'unresolved ratchet':20s} {res['n_unresolved']}/"
          f"{res['max_unresolved']}"
          + ("  ⛔ EXCEEDED" if res["ratchet_exceeded"] else
             ("  (ratchet can be tightened)" if res["ratchet_loose"] else "")))
    print()
    for i in res["allow_issues"]:
        print(f"{i['status']:20s} {i['citation']}\n{'':20s}   ({i['note']})")
    for r in res["citations"]:
        if args.only_bad and r["status"] in (EXISTS, NOT_CHECKED, NAME_ONLY,
                                             PATTERN):
            continue
        line = f"{r['status']:20s} {r['citation']}"
        if r.get("resolved") and r["resolved"] != r["citation"]:
            line += f"\n{'':20s}   -> {r['resolved']}"
        if r.get("allow_reason"):
            line += f"\n{'':20s}   ALLOWED: {r['allow_reason']}"
        elif r.get("note"):
            line += f"\n{'':20s}   ({r['note']})"
        if r.get("candidates"):
            line += "\n" + "\n".join(f"{'':20s}   ? {c}" for c in r["candidates"])
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return exit_code(res, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
