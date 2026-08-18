#!/usr/bin/env python3
"""Report code that exists on a BOX but not in git — "a box is not storage".

The 2026-07-20 audit found the program's dominant failure mode was good work
stranded outside the repo: REF-B v2's architecture and the ENTIRE TanitEval
harness each lived on a single pod disk, with no copy in git and no backup.
Losing either pod would have destroyed a top-3 arm and the evidence base for
every headline number we have published. That is the question this answers, and
**no other instrument in the programme answers it.**

⛔ WHAT THIS DOES **NOT** ANSWER — read this before using it as a sync check
==========================================================================
Its direction is **box → repo**. *"Is this file on the box also in git?"*

**A repo file MISSING from the box, or STALE on the box, is invisible here by
construction** — and that is the failure class that has actually cost us:

* C99: a 2.6×-stale ``refc_dump_latents.py`` on Thor, three green md5s.
* C102/C105: ``train_v6_staged.py`` itself 234,845 B against the repo's 252,691 B,
  still exporting the symbol its caller imports, missing all six ``probe_applies``
  references — **the S-T gate would have run with no applicability filtering.**

⇒ ⭐ **For "is the box running the repo?", use** ``stack/scripts/launch_closure_audit.py``,
which computes the launch import closure, classifies ``SAME`` / ``CRLF_ONLY`` /
``DRIFT`` / ``MISSING_REMOTE``, and verifies with a **real import**. The two tools
are converses and both are needed. Do not substitute one for the other.

⚠️ REPAIRED 2026-08-18 — four measured defects, each of which produced a clean,
plausible, wrong answer (the ``df``-on-a-pod family)
====================================================================
1. ⛔ **The index walked ``.claude/worktrees/``** — 14 stale full copies of the
   repo. MEASURED: **8,079 indexed files where the repo holds 2,132** (3.8×), and
   ambiguous basenames went from **5.6 % to 89.4 % of names**. Every judgement
   was being made against a pool of stale duplicates. Now excluded.
2. ⛔ **It matched by BASENAME anywhere in the repo**, so it compared unrelated
   same-named files. Even with the worktrees gone, ``__init__.py`` has dozens of
   copies. ⇒ Matching is now by **longest path suffix**; a basename-only hit is a
   separate, weaker verdict (:data:`NAME_ONLY`) and is **never** called drift.
3. ⛔ **No CRLF normalisation.** The repo tree is CRLF, every box is LF.
   MEASURED on the trees a box actually holds: **28.1 % of ``stack/`` and 19.3 %
   of ``taniteval/`` source files contain CRLF** — so roughly a quarter of every
   file it inspected would be reported as drift that does not exist. Stated on
   the other denominator, the one that matters when a box is well-synced: **of
   the rows it would print as drift, ~94 % were artifacts** (C105, 47 of 50).
   ⇒ Both digests are indexed and a line-ending difference is
   :data:`CRLF_ONLY`, its own category, never :data:`DRIFTED`.
4. ⛔ **``DEFAULT_PODS`` were four dead machines** (``tanitad-pod``/``-pod2``/
   ``-pod3``/``-eval``; the RunPod fleet was released 2026-08-15). It would have
   printed four unreachable hosts and a reassuring ``TOTAL: 0``. ⇒ Repointed at
   the live fleet, and **an unreachable host is now a loud non-zero exit**, not a
   quiet zero — *absence of a finding is not a finding of absence.*

⚠️ **A fifth, from ``PRODUCED_GOAL_PATH.md`` and now fixed:** ``-maxdepth 3`` put
``/root/TanitAD/stack/tanitad/**`` (depth 5) below the horizon, so the two
``stack/`` trees that were 52 % and worse wrong were never scanned at all.

Usage:
    python3 stack/scripts/pod_git_drift.py                 # the live fleet
    python3 stack/scripts/pod_git_drift.py --hosts tanitad-thor-wifi
    python3 stack/scripts/pod_git_drift.py --json report.json --show-drifted

Read-only: it runs only ``find``/``sha256sum`` over ``ssh -n`` and never writes
to a box. (``-n`` is mandatory: a nested ssh inside a pipe eats the rest of the
caller's stdin and the tail silently never runs.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

#: ⚠️ The RunPod fleet (``tanitad-pod``, ``-pod2``, ``-pod3``, ``-eval``) was
#: released 2026-08-15. The fleet is the Jetson Thor plus the dev box. A default
#: naming dead machines does not fail — it reports nothing and exits 0.
DEFAULT_HOSTS = ["tanitad-thor-wifi"]

#: Where humans and agents actually stash things. ``/home/nvidia`` is Thor's.
SEARCH_ROOTS = ["/home/nvidia", "/root", "/workspace"]

#: ⚠️ Was 3. At depth 3, ``/root/TanitAD/stack/tanitad/**`` (depth 5) and
#: ``/root/v4eval/stack/scripts/*`` (depth 4) are below the horizon — the two
#: trees that were later found 52 % and worse wrong were never scanned.
SEARCH_MAXDEPTH = 6

SUFFIXES = (".py", ".sh")

#: Noise that is never a deliverable.
#: ⛔ ``.claude/worktrees/`` is the important one: 14 stale full repo copies,
#: which inflated the index 3.8× and made basename matching meaningless.
EXCLUDE_PARTS = (
    "__pycache__", "site-packages", "dist-packages", "/.git/", "/node_modules/",
    "/.cache/", "/venv/", "/.venv/", "/miniconda", "/anaconda",
    ".claude/worktrees/", "/_pod_backup/",
)

#: The same exclusions, as an ``egrep`` alternation for the remote ``find``.
#: ⚠️ ``/.cargo/``, ``/.rustup/`` etc. are package-manager caches — third-party
#: source that is genuinely "on the box and not in git" and is genuinely not our
#: problem. MEASURED 2026-08-18: 5 ``glam``/``numpy`` build scripts were being
#: reported as stranded deliverables.
REMOTE_EXCLUDE_RE = (r"__pycache__|site-packages|dist-packages|/\.git/|"
                     r"/\.cache/|/venv/|/\.venv/|/miniconda|/anaconda|"
                     r"/node_modules/|/\.cargo/|/\.rustup/|/\.npm/|/\.conda/|"
                     r"/\.local/lib/")

# ---------------------------------------------------------------- verdicts
HOST_ONLY = "HOST_ONLY"    # no file of this name in the repo at all — rescue it
POD_ONLY = HOST_ONLY       # legacy alias; the fleet is no longer pods
DRIFTED = "DRIFTED"        # a repo file at the SAME PATH differs in content
NAME_ONLY = "NAME_ONLY"    # only a same-BASENAME file exists — weak, not drift
CRLF_ONLY = "CRLF_ONLY"    # differs only by line endings — NOT drift
IN_GIT = "IN_GIT"          # byte-identical to a repo copy

VERDICTS = (IN_GIT, CRLF_ONLY, DRIFTED, NAME_ONLY, HOST_ONLY)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_lf(data: bytes) -> str:
    """Digest after ``\\r\\n`` -> ``\\n`` **only**.

    No trailing-whitespace strip, no final-newline fixup, no encoding round
    trip. A normaliser that rewrites more than line endings would hide real
    drift, which is the opposite failure and a worse one.
    """
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _path_suffixes(posix_path: str) -> list[str]:
    """``a/b/c.py`` -> ``['a/b/c.py', 'b/c.py', 'c.py']``, longest first."""
    parts = [p for p in posix_path.split("/") if p not in ("", ".", "..")]
    return ["/".join(parts[i:]) for i in range(len(parts))]


def repo_index(repo_root: str | Path) -> dict:
    """Index the repo's source files by raw hash, LF hash, path and basename.

    Pure function over the filesystem so the comparison logic stays testable.

    Returns ``{"by_hash", "by_lf", "by_name", "by_path"}`` where ``by_path`` maps
    a repo-relative path to its two digests — that is what makes a **path**
    comparison possible instead of a basename lottery.
    """
    root = Path(repo_root)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_lf: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, set[str]] = defaultdict(set)
    by_path: dict[str, dict[str, str]] = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        posix = path.as_posix()
        if any(part in posix for part in EXCLUDE_PARTS):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        raw, lf = sha256_bytes(data), sha256_lf(data)
        rel = path.relative_to(root).as_posix()
        by_hash[raw].append(rel)
        by_lf[lf].append(rel)
        by_name[path.name].add(raw)
        by_path[rel] = {"raw": raw, "lf": lf}

    return {"by_hash": dict(by_hash), "by_lf": dict(by_lf),
            "by_name": {k: v for k, v in by_name.items()},
            "by_path": by_path}


def _match_by_path(host_path: str, by_path: dict) -> str | None:
    """The repo file this box path most specifically corresponds to.

    ``/home/nvidia/TanitAD/stack/scripts/v6_chain.py`` -> ``stack/scripts/
    v6_chain.py``. **Longest suffix wins**, so a two-segment coincidence can
    never outrank a full-path agreement. ⚠️ Returns ``None`` rather than a
    best-effort guess — C105 cost three rounds to a locator that returned
    something plausible instead of failing.
    """
    for suffix in _path_suffixes(host_path):
        if suffix in by_path:
            return suffix
    return None


def classify(host_files: list[tuple[str, str]], index: dict) -> list[dict]:
    """Classify ``(sha256, host_path)`` pairs against a repo index.

    Deliberately free of ssh/IO so it can be unit-tested directly.

    Order is load-bearing: **content agreement outranks path agreement** (a
    rescued file legitimately lands at a new repo path), and **line endings are
    resolved before drift is ever considered**, because on this repo a naive
    comparison calls ~a quarter of all files drifted and, on a well-synced box,
    ~94 % of the rows it prints as drift are that artifact.
    """
    by_hash = index["by_hash"]
    by_lf = index.get("by_lf", {})
    by_name = index["by_name"]
    by_path = index.get("by_path", {})
    out = []
    for digest, host_path in host_files:
        name = host_path.rsplit("/", 1)[-1]
        peer = _match_by_path(host_path, by_path)
        if digest in by_hash:
            verdict, note = IN_GIT, by_hash[digest][0]
        elif digest in by_lf:
            # The box is LF, the repo tree is CRLF. Not drift, and saying so is
            # this repair's single largest effect on the report.
            verdict, note = CRLF_ONLY, (
                f"line endings only vs {by_lf[digest][0]} — NOT drift")
        elif peer is not None:
            verdict, note = DRIFTED, (
                f"{peer} exists in the repo with different content")
        elif name in by_name:
            # ⚠️ Same basename, different tree. This is NOT evidence the box is
            # running something stale — it is usually two unrelated files.
            verdict, note = NAME_ONLY, (
                f"only a same-named file elsewhere in the repo ({name}); "
                "weak evidence, not drift")
        else:
            verdict, note = HOST_ONLY, "no file of this name anywhere in the repo"
        out.append({"path": host_path, "sha256": digest,
                    "verdict": verdict, "note": note})
    return out


def _ssh_bin() -> str:
    win = Path("C:/Windows/System32/OpenSSH/ssh.exe")
    return str(win) if win.is_file() else (shutil.which("ssh") or "ssh")


def scan_host(host: str, roots: list[str] | None = None,
              maxdepth: int = SEARCH_MAXDEPTH,
              timeout: int = 240) -> tuple[list[tuple[str, str]], str | None]:
    """ssh into ``host`` and hash candidate source files. Read-only.

    Returns ``(files, error)``. ⚠️ **The error is returned, not swallowed.** The
    old version printed to stderr and returned ``[]``, which reads downstream as
    *"this box is clean"* — four dead hosts would have produced a confident
    ``TOTAL POD-ONLY FILES: 0``.
    """
    roots = roots or SEARCH_ROOTS
    names = " -o ".join(f"-name '*{s}'" for s in SUFFIXES)
    # -size -2M keeps us to source, not data blobs.
    remote = (
        f"find {' '.join(roots)} -maxdepth {maxdepth} -type f \\( {names} \\) "
        f"-size -2M 2>/dev/null "
        f"| grep -vE '{REMOTE_EXCLUDE_RE}' "
        f"| head -4000 | xargs -r sha256sum 2>/dev/null"
    )
    try:
        res = subprocess.run(
            [_ssh_bin(), "-n", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
             host, remote],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [], f"unreachable ({type(exc).__name__})"
    if res.returncode != 0 and not res.stdout.strip():
        return [], f"ssh failed rc={res.returncode}: {res.stderr.strip()[:160]}"

    files = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            files.append((parts[0], parts[1]))
    if not files:
        return [], "no candidate source files found (check --search-roots)"
    return files, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Find code stranded on a box. For the CONVERSE check "
                    "(is the box running the repo?) use launch_closure_audit.py.")
    ap.add_argument("--hosts", "--pods", nargs="*", dest="hosts",
                    default=DEFAULT_HOSTS,
                    help="ssh aliases of the live fleet")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--search-roots", nargs="*", default=SEARCH_ROOTS)
    ap.add_argument("--maxdepth", type=int, default=SEARCH_MAXDEPTH)
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--show-drifted", action="store_true",
                    help="also list DRIFTED / NAME_ONLY files (default: counts)")
    args = ap.parse_args(argv)

    if not args.hosts:
        # ⛔ `--hosts` with no values scanned nothing and printed
        # "TOTAL HOST-ONLY FILES: 0", exit 0 — a reassuring answer to a question
        # never asked. That is the same shape as the four dead default pods this
        # repair exists to remove, so it is a hard error, not a warning.
        print("[drift] no hosts given — nothing was scanned. This is NOT a "
              "clean result.", file=sys.stderr)
        return 2

    print("=" * 74)
    print("box -> repo ONLY. A repo file MISSING or STALE on the box is invisible")
    print("here by construction — that is launch_closure_audit.py's question.")
    print("=" * 74)
    print(f"[drift] hosts={args.hosts}  roots={args.search_roots}  "
          f"maxdepth={args.maxdepth}")
    print(f"[drift] indexing repo at {args.repo} ...")
    index = repo_index(args.repo)
    print(f"[drift] {len(index['by_path'])} source files · "
          f"{len(index['by_hash'])} distinct blobs · "
          f"{len(index['by_name'])} distinct filenames")

    report: dict = {"hosts": {}, "unreachable": []}
    host_only_total = 0
    for host in args.hosts:
        print(f"\n=== {host} ===")
        found, err = scan_host(host, args.search_roots, args.maxdepth)
        if err:
            # ⛔ Loud. A host we could not read is NOT a clean host.
            print(f"  !! UNREADABLE: {err}", file=sys.stderr)
            report["unreachable"].append({"host": host, "reason": err})
            continue
        found = classify(found, index)
        report["hosts"][host] = found
        counts = defaultdict(int)
        for f in found:
            counts[f["verdict"]] += 1
        print(f"  {len(found)} files · " +
              " · ".join(f"{counts[v]} {v.lower()}" for v in VERDICTS))
        # ⚠️ What the UNREPAIRED tool would have printed. Kept in the output
        # because a repair whose effect is invisible gets undone by the next
        # person who thinks the old numbers looked more thorough.
        old_drift = counts[CRLF_ONLY] + counts[DRIFTED] + counts[NAME_ONLY]
        if old_drift:
            false_n = counts[CRLF_ONLY] + counts[NAME_ONLY]
            print(f"  (pre-repair this printed {old_drift} DRIFTED, of which "
                  f"{false_n} = {100 * false_n / old_drift:.1f}% were artifacts: "
                  f"{counts[CRLF_ONLY]} line-ending, {counts[NAME_ONLY]} "
                  f"basename collisions)")

        # per-directory shape FIRST — 260 files in one vendored tree is one
        # fact, not 260, and printing it as 260 buries the 47 that matter.
        for verdict in (HOST_ONLY, DRIFTED):
            group: dict[str, int] = defaultdict(int)
            for f in found:
                if f["verdict"] == verdict:
                    group[f["path"].rsplit("/", 1)[0]] += 1
            if group:
                print(f"  {verdict} by directory:")
                for d, n in sorted(group.items(), key=lambda kv: -kv[1])[:15]:
                    print(f"    {n:>4}  {d}")

        for f in found:
            if f["verdict"] == HOST_ONLY:
                host_only_total += 1
                print(f"    HOST_ONLY  {f['path']}")
            elif f["verdict"] in (DRIFTED, NAME_ONLY) and args.show_drifted:
                print(f"    {f['verdict']:<10} {f['path']}  ({f['note']})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[drift] wrote {args.json}")

    print(f"\n[drift] TOTAL HOST-ONLY FILES: {host_only_total}")
    if report["unreachable"]:
        print(f"[drift] {len(report['unreachable'])} host(s) UNREADABLE — this "
              f"run proves nothing about them: "
              f"{[u['host'] for u in report['unreachable']]}")
    if host_only_total:
        print("[drift] These exist in exactly one place. Rescue them into git.")
    return 1 if (host_only_total or report["unreachable"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
