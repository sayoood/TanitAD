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

⛔ SIXTH — C110: **THE COUNT IT PRINTED WAS A CLAIM ABOUT ITS OWN FILTER**
==========================================================================
``SUFFIXES`` was ``(".py", ".sh")``, so every stranded result JSON, run log,
``.md``, ``.yaml`` and ``.bak`` was invisible **by construction**. It reported
**45** stranded files on Thor; a content-hash sweep of the same directories found
**102**. ⇒ **It missed 46 of 102 — 45 %** — and the published "45" was therefore
a statement about the filter, not about Thor.

⇒ **RULE: before quoting a count, read the instrument's inclusion rule.** A
census is a claim about the FILTER until proven otherwise. This tool now
**prints its inclusion rule with every run** and stamps it into ``--json``, so
the next reader cannot mistake a filtered view for a census.

⚠️ **AND THE OBVIOUS OVER-CORRECTION IS ALSO A FAILURE.** Widening the suffix set
pulls in caches, venvs, generated dumps and third-party trees, and the ~94 %
false-positive rate scales with it. *A tool that prints 361 rows of which 293 are
noise is not more useful than one that printed 45.* The four things that keep the
widening honest:

1. Line endings are normalised **before** any comparison (defect 3 above).
2. Files are split into :data:`SOURCE_SUFFIXES` and :data:`ARTIFACT_SUFFIXES`
   and **counted separately** — a run log is not noise (C110 kept all 17 as raw
   measurement transcripts) but it needs a different judgement from a ``.py``.
3. A hard :data:`MAX_BYTES` cap on **both** sides, so the two never disagree
   merely because one side skipped a large file.
4. ⛔ The remote list cap is **reported**, not silently applied. A truncated
   scan that prints a total is the C110 failure recreated one layer down.

⛔ SEVENTH — C110: ``NAME_ONLY`` WAS HOW THE MOST CONSEQUENTIAL FIND ESCAPED
============================================================================
The banked ``…/2026-08-02-thor-deployment-profile/thor_profile.py`` **cannot have
produced its own co-banked ``thor_profile.json``** — the JSON carries
``"frame": "176x624 hfov 117.0"`` and the banked script never assigns
``out['frame']``; Thor's copy does. **A different model was profiled than the
banked script describes**, and *a banked script that cannot produce its banked
result is worse than a missing one: the pair looks like provenance.*

⛔ **This tool saw that file and downgraded it to** ``NAME_ONLY`` — *"weak
evidence, not drift"* — **which is exactly how it escaped.** ⇒ A same-basename
hit is now split: an **unambiguous, program-authored** basename becomes
:data:`NAME_DRIFT` (a real finding, needing an owner's adjudication), and only a
genuinely ambiguous name (``__init__.py``, ``utils.py``, dozens of copies) stays
the weak :data:`NAME_ONLY`.

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
import re
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

#: Things a human or an agent WROTE. Losing one destroys work.
SOURCE_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cfg",
                   ".ini", ".patch", ".diff", ".sql", ".bash")

#: Things a program EMITTED. ⚠️ NOT noise — C110 kept all 17 stranded run logs
#: because they are raw measurement transcripts, and the stranded result JSONs
#: are the evidence base for published numbers. But they need a different
#: judgement (is it regenerable? is it superseded?), so they are counted
#: separately rather than swelling one undifferentiated total.
ARTIFACT_SUFFIXES = (".json", ".jsonl", ".log", ".txt", ".csv", ".tsv",
                     ".bak", ".out", ".err")

#: ⛔ WAS ``(".py", ".sh")`` — C110: that filter missed **46 of 102** stranded
#: files on Thor (45 %), and the "45" it printed was a claim about itself.
SUFFIXES = SOURCE_SUFFIXES + ARTIFACT_SUFFIXES

#: Applied to BOTH sides. ⚠️ If only the remote were capped, a large repo file
#: would go unindexed while its box copy was scanned, and the tool would invent
#: a HOST_ONLY finding out of its own asymmetry.
MAX_BYTES = 2 * 1024 * 1024

#: Basenames too common for a same-name hit to mean anything, even when the
#: index happens to hold exactly one. Uniqueness does most of the work; this is
#: the short backstop.
UBIQUITOUS_NAMES = frozenset({
    "__init__.py", "__main__.py", "conftest.py", "setup.py", "utils.py",
    "main.py", "run.sh", "README.md", "requirements.txt", "config.json",
    "metrics.json", "summary.json", "notes.md", "TODO.md",
})

#: A same-basename hit is a real finding only when the repo side is OURS.
#: ``thor_profile.py`` lives under the research hub; a stray ``glam`` build
#: script does not.
AUTHORED_ROOTS = ("stack/", "taniteval/", "alpasim/", "TanitAD Research Hub/",
                  "Project Steering/", "scripts/", "tools/")

#: ⚠️ Paths that must NEVER be pulled without reading them first. C111: the
#: 117-file Thor rescue banked a run log whose line 11 held a **live Hugging
#: Face token**, and GitHub's push protection — not ours — caught it. The
#: invariant we hold is about ``Keys.txt``; it says nothing about the EXHAUST,
#: and a widened suffix set sweeps in exactly that exhaust.
#: ⚠️ This tool prints paths and digests only, never contents.
SENSITIVE_HINTS = ("keys.txt", "credential", "secret", "token", ".env",
                   ".pem", ".netrc", "id_rsa", "id_ed25519", ".htpasswd")

#: Noise that is never a deliverable.
#: ⛔ ``.claude/worktrees/`` is the important one: 14 stale full repo copies,
#: which inflated the index 3.8× and made basename matching meaningless.
#: ⚠️ The cache entries matter far more now that ``.json``/``.txt`` are in
#: scope — they are where an over-correction would drown the signal.
EXCLUDE_PARTS = (
    "__pycache__", "site-packages", "dist-packages", "/.git/", "/node_modules/",
    "/.cache/", "/venv/", "/.venv/", "/miniconda", "/anaconda",
    ".claude/worktrees/", "/_pod_backup/", "/.pytest_cache/", "/.mypy_cache/",
    "/.ruff_cache/", "/.ipynb_checkpoints/", ".egg-info", "/.tox/",
    # ⚠️ Thor is a WORKSTATION with a desktop, and a widened suffix set sweeps
    # in its application data. MEASURED 2026-08-18: 9 Thunderbird profile files
    # were reported as stranded deliverables. User app data is not program work.
    "/snap/", "/.thunderbird/", "/.mozilla/", "/.config/", "/.local/share/",
    "/.jupyter/", "/.vscode", "/.gnupg/", "/.dbus/",
)

#: The same exclusions, as an ``egrep`` alternation for the remote ``find``.
#: ⚠️ ``/.cargo/``, ``/.rustup/`` etc. are package-manager caches — third-party
#: source that is genuinely "on the box and not in git" and is genuinely not our
#: problem. MEASURED 2026-08-18: 5 ``glam``/``numpy`` build scripts were being
#: reported as stranded deliverables.
REMOTE_EXCLUDE_RE = (r"__pycache__|site-packages|dist-packages|/\.git/|"
                     r"/\.cache/|/venv/|/\.venv/|/miniconda|/anaconda|"
                     r"/node_modules/|/\.cargo/|/\.rustup/|/\.npm/|/\.conda/|"
                     r"/\.local/lib/|/\.pytest_cache/|/\.mypy_cache/|"
                     r"/\.ipynb_checkpoints/|\.egg-info|/\.tox/|/snap/|"
                     r"/\.thunderbird/|/\.mozilla/|/\.config/|/\.local/share/|"
                     r"/\.jupyter/|/\.vscode|/\.gnupg/|/\.dbus/")

#: ⚠️ Was ``head -4000``, applied SILENTLY. With the widened suffix set a real
#: fleet can exceed it, and a truncated scan that prints a confident total is
#: C110 recreated one layer down. The remote now reports its own pre-cap count.
REMOTE_LIST_CAP = 20000

# ---------------------------------------------------------------- verdicts
HOST_ONLY = "HOST_ONLY"    # no file of this name in the repo at all — rescue it
POD_ONLY = HOST_ONLY       # legacy alias; the fleet is no longer pods
DRIFTED = "DRIFTED"        # a repo file at the SAME PATH differs in content
#: ⛔ C110: an UNAMBIGUOUS, program-authored basename whose content differs.
#: ``thor_profile.py`` was called NAME_ONLY and escaped on that word "weak";
#: it turned out to prove a DIFFERENT MODEL had been profiled than the banked
#: script describes. Which side is canonical is the owner's call — but it is a
#: finding, not a shrug.
NAME_DRIFT = "NAME_DRIFT"
NAME_ONLY = "NAME_ONLY"    # an AMBIGUOUS same-basename hit — genuinely weak
CRLF_ONLY = "CRLF_ONLY"    # differs only by line endings — NOT drift
IN_GIT = "IN_GIT"          # byte-identical to a repo copy

VERDICTS = (IN_GIT, CRLF_ONLY, DRIFTED, NAME_DRIFT, NAME_ONLY, HOST_ONLY)

#: Verdicts a human must act on. NAME_ONLY and CRLF_ONLY are deliberately out.
ACTIONABLE = (HOST_ONLY, DRIFTED, NAME_DRIFT)

SOURCE, ARTIFACT = "source", "artifact"


def kind_of(path: str) -> str:
    """``source`` (someone wrote it) vs ``artifact`` (a program emitted it)."""
    lower = path.lower()
    return ARTIFACT if lower.endswith(ARTIFACT_SUFFIXES) else SOURCE


def is_sensitive(path: str) -> bool:
    """⚠️ C111 — never pull one of these without reading it first."""
    lower = path.lower()
    return any(hint in lower for hint in SENSITIVE_HINTS)


def inclusion_rule(roots=None, maxdepth: int = None) -> dict:
    """The filter, as data — so it can be printed AND stamped into the JSON.

    ⛔ C110's whole lesson: a count is a claim about the filter until the filter
    is stated next to it.
    """
    return {
        "source_suffixes": list(SOURCE_SUFFIXES),
        "artifact_suffixes": list(ARTIFACT_SUFFIXES),
        "max_bytes": MAX_BYTES,
        "search_roots": list(roots if roots is not None else SEARCH_ROOTS),
        "maxdepth": SEARCH_MAXDEPTH if maxdepth is None else maxdepth,
        "excluded": list(EXCLUDE_PARTS),
        "remote_list_cap": REMOTE_LIST_CAP,
        "_warning": (
            "THIS IS A FILTERED VIEW, NOT A CENSUS. A file outside this rule is "
            "invisible here BY CONSTRUCTION. C110: the previous rule was "
            "(.py, .sh) and missed 46 of 102 stranded files (45 %)."),
    }


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
    name_paths: dict[str, set[str]] = defaultdict(set)
    by_path: dict[str, dict[str, str]] = {}
    skipped_large = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        posix = path.as_posix()
        if any(part in posix for part in EXCLUDE_PARTS):
            continue
        try:
            # ⚠️ Same cap the remote applies. An asymmetric cap would invent
            # HOST_ONLY findings out of the tool's own inconsistency.
            if path.stat().st_size >= MAX_BYTES:
                skipped_large += 1
                continue
            data = path.read_bytes()
        except OSError:
            continue
        raw, lf = sha256_bytes(data), sha256_lf(data)
        rel = path.relative_to(root).as_posix()
        by_hash[raw].append(rel)
        by_lf[lf].append(rel)
        by_name[path.name].add(raw)
        name_paths[path.name].add(rel)
        by_path[rel] = {"raw": raw, "lf": lf}

    return {"by_hash": dict(by_hash), "by_lf": dict(by_lf),
            "by_name": {k: v for k, v in by_name.items()},
            "name_paths": {k: v for k, v in name_paths.items()},
            "by_path": by_path, "skipped_large": skipped_large}


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
    name_paths = index.get("name_paths", {})
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
            twin = _authored_twin(name, name_paths, kind_of(host_path))
            if twin is not None:
                # ⛔ C110. One repo path, ours, different content. This is how
                # thor_profile.py escaped as "weak evidence".
                verdict, note = NAME_DRIFT, (
                    f"{twin} is the repo's ONLY {name} and its content "
                    f"DIFFERS. Paths do not align, so which side is canonical "
                    f"needs that package's owner — but this is a finding, not "
                    f"weak evidence (C110: the banked thor_profile.py could "
                    f"not have produced its own banked thor_profile.json).")
            else:
                # ⚠️ Genuinely ambiguous: several repo copies, or a name so
                # common the match carries no information.
                copies = len(name_paths.get(name, ())) or "several"
                verdict, note = NAME_ONLY, (
                    f"{copies} same-named file(s) elsewhere in the repo "
                    f"({name}); ambiguous — weak evidence, not drift")
        else:
            verdict, note = HOST_ONLY, "no file of this name anywhere in the repo"
        row = {"path": host_path, "sha256": digest, "verdict": verdict,
               "kind": kind_of(host_path), "note": note}
        if is_sensitive(host_path):
            # ⚠️ C111: a rescued run log carried a live HF token to GitHub.
            row["sensitive"] = True
            row["note"] += (" ⚠️ SENSITIVE-LOOKING PATH — read it before "
                            "pulling; C111 banked a live token from a run log.")
        out.append(row)
    return out


def _authored_twin(name: str, name_paths: dict,
                   kind: str = SOURCE) -> str | None:
    """The repo path a same-basename hit unambiguously refers to, or ``None``.

    Four conditions, all required — the point is to promote ``thor_profile.py``
    without promoting ``__init__.py``, and without promoting 304 result JSONs:

    1. The box file is :data:`SOURCE`, not an emitted :data:`ARTIFACT`.
    2. **Exactly one** repo file carries the basename (no lottery).
    3. It lives under a root **we author** (not a vendored build script).
    4. The basename is not on the short :data:`UBIQUITOUS_NAMES` backstop.

    ⛔ CONDITION 1 IS THE ANTI-OVER-CORRECTION CLAUSE, and it is MEASURED, not
    assumed. Without it the widened filter promoted **478** basename hits on
    Thor, of which **304 (63.6 %) were artifacts** — result JSONs and logs whose
    names merely coincide with a repo file. *A tool that prints 478 rows of
    which 304 are noise is not more useful than one that printed 45.* C110's
    actual case was a ``.py``: an artifact's basename colliding says nothing
    about provenance, while a SOURCE file's does.
    """
    if kind != SOURCE or name in UBIQUITOUS_NAMES:
        return None
    paths = name_paths.get(name)
    if not paths or len(paths) != 1:
        return None
    only = next(iter(paths))
    return only if only.startswith(AUTHORED_ROOTS) else None


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
    # ⚠️ ``-size -{N}c`` is EXACT bytes. ``-size -2M`` rounds up to whole MiB
    # and would actually cap at 1 MiB, disagreeing with repo_index's MAX_BYTES
    # — an asymmetry that manufactures HOST_ONLY rows.
    # ⚠️ The count is computed POD-SIDE and emitted as an opaque ``ZZ<n>ZZ``
    # marker. CLAUDE.md: a filter that contains the token it searches for
    # matches its own echoed command. ``ZZ%sZZ`` in the command text cannot
    # match ``ZZ\d+ZZ``, so the emitted token is disjoint from the searched one.
    remote = (
        f"L=$(find {' '.join(roots)} -maxdepth {maxdepth} -type f "
        f"\\( {names} \\) -size -{MAX_BYTES}c 2>/dev/null "
        f"| grep -vE '{REMOTE_EXCLUDE_RE}'); "
        f"printf 'ZZ%sZZ\\n' \"$(printf '%s\\n' \"$L\" | grep -c .)\"; "
        # ⛔ MEASURED 2026-08-18: plain `xargs` splits on WHITESPACE, so every
        # path containing a space was torn into fragments that hashed to
        # nothing — 415 of 3,397 files on Thor (12.2 %) vanished silently. The
        # old tool had this too and could not report it; the pre-cap count
        # above is what made it visible. NUL-delimit instead.
        f"printf '%s\\n' \"$L\" | head -{REMOTE_LIST_CAP} "
        f"| tr '\\n' '\\0' | xargs -0 -r sha256sum 2>/dev/null"
    )
    try:
        res = subprocess.run(
            [_ssh_bin(), "-n", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
             host, remote],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [], f"unreachable ({type(exc).__name__})", 0
    if res.returncode != 0 and not res.stdout.strip():
        return [], f"ssh failed rc={res.returncode}: {res.stderr.strip()[:160]}", 0

    total = 0
    files = []
    for line in res.stdout.splitlines():
        marker = re.search(r"ZZ(\d+)ZZ", line)
        if marker:
            total = int(marker.group(1))
            continue
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            files.append((parts[0], parts[1]))
    if not files:
        return [], "no candidate files found (check --search-roots)", total
    return files, None, total


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

    rule = inclusion_rule(args.search_roots, args.maxdepth)
    print("=" * 74)
    print("box -> repo ONLY. A repo file MISSING or STALE on the box is invisible")
    print("here by construction — that is launch_closure_audit.py's question.")
    print("=" * 74)
    # ⛔ C110. The inclusion rule prints BEFORE any count, every run. The old
    # tool printed "45" with no statement of what it could see, and that number
    # was published as a fact about Thor when it was a fact about this filter.
    print("INCLUSION RULE — THIS IS A FILTERED VIEW, NOT A CENSUS:")
    print(f"  source suffixes   {' '.join(SOURCE_SUFFIXES)}")
    print(f"  artifact suffixes {' '.join(ARTIFACT_SUFFIXES)}")
    print(f"  size cap          < {MAX_BYTES} bytes (both sides)")
    print(f"  roots             {args.search_roots}  maxdepth={args.maxdepth}")
    print(f"  excluded          {', '.join(EXCLUDE_PARTS)}")
    print("  ⛔ Anything outside this rule is invisible. C110: the previous "
          "rule was")
    print("     (.py, .sh) and missed 46 of 102 stranded files — 45 %.")
    print("=" * 74)
    print(f"[drift] hosts={args.hosts}")
    print(f"[drift] indexing repo at {args.repo} ...")
    index = repo_index(args.repo)
    print(f"[drift] {len(index['by_path'])} indexed files · "
          f"{len(index['by_hash'])} distinct blobs · "
          f"{len(index['by_name'])} distinct filenames · "
          f"{index.get('skipped_large', 0)} skipped as >= {MAX_BYTES} B")

    report: dict = {"inclusion_rule": rule, "hosts": {}, "unreachable": [],
                    "_evidence_class": "MEASURED (ours)"}
    host_only_total = 0
    actionable_total = 0
    for host in args.hosts:
        print(f"\n=== {host} ===")
        found, err, total_found = scan_host(host, args.search_roots,
                                            args.maxdepth)
        if err:
            # ⛔ Loud. A host we could not read is NOT a clean host.
            print(f"  !! UNREADABLE: {err}", file=sys.stderr)
            report["unreachable"].append({"host": host, "reason": err})
            continue
        if total_found > len(found):
            # ⛔ An incomplete scan that prints a confident total is C110 again,
            # one layer down. Never silent — and the two causes are different
            # bugs with different fixes, so they are never merged into one
            # message. MEASURED 2026-08-18: the second one was real (415 of
            # 3,397 files lost to whitespace-splitting in `xargs`).
            why = ("the list cap" if total_found > REMOTE_LIST_CAP
                   else "hashing (unreadable, vanished, or an odd filename)")
            print(f"  !! INCOMPLETE: the box matched {total_found} files but "
                  f"only {len(found)} were hashed — lost to {why}. "
                  f"THIS RUN IS NOT A CENSUS.", file=sys.stderr)
            report["unreachable"].append(
                {"host": host,
                 "reason": f"incomplete: {len(found)} of {total_found} hashed "
                           f"(lost to {why})"})
        found = classify(found, index)
        report["hosts"][host] = found
        counts = defaultdict(int)
        kinds: dict[tuple, int] = defaultdict(int)
        for f in found:
            counts[f["verdict"]] += 1
            kinds[(f["verdict"], f["kind"])] += 1
        print(f"  {len(found)} files · " +
              " · ".join(f"{counts[v]} {v.lower()}" for v in VERDICTS))
        # ⚠️ ARTIFACTS AS THEIR OWN CATEGORY. A run log is not noise — C110 kept
        # all 17 as raw measurement transcripts — but it needs a different
        # judgement from a .py, and merging them into one total is how a
        # widened filter turns into 293 rows of undifferentiated noise.
        for verdict in ACTIONABLE:
            n_src = kinds[(verdict, SOURCE)]
            n_art = kinds[(verdict, ARTIFACT)]
            if n_src or n_art:
                print(f"    {verdict:<11} {n_src:>4} source · {n_art:>4} "
                      f"artifact (regenerable? superseded? judge by content)")
        sensitive = [f for f in found if f.get("sensitive")]
        if sensitive:
            # ⚠️ C111: a live HF token reached a commit inside a rescued log.
            print(f"  ⚠️ {len(sensitive)} SENSITIVE-LOOKING path(s) — READ "
                  f"BEFORE PULLING, do not bank blind:")
            for f in sensitive[:10]:
                print(f"      {f['path']}")

        # ⚠️ What the UNREPAIRED tool would have printed. Kept in the output
        # because a repair whose effect is invisible gets undone by the next
        # person who thinks the old numbers looked more thorough.
        old_drift = (counts[CRLF_ONLY] + counts[DRIFTED] + counts[NAME_ONLY]
                     + counts[NAME_DRIFT])
        if old_drift:
            false_n = counts[CRLF_ONLY] + counts[NAME_ONLY]
            print(f"  (pre-repair this printed {old_drift} DRIFTED, of which "
                  f"{false_n} = {100 * false_n / old_drift:.1f}% were artifacts: "
                  f"{counts[CRLF_ONLY]} line-ending, {counts[NAME_ONLY]} "
                  f"ambiguous basename collisions)")

        # per-directory shape FIRST — 260 files in one vendored tree is one
        # fact, not 260, and printing it as 260 buries the 47 that matter.
        for verdict in ACTIONABLE:
            group: dict[str, int] = defaultdict(int)
            for f in found:
                if f["verdict"] == verdict:
                    group[f["path"].rsplit("/", 1)[0]] += 1
            if group:
                ranked = sorted(group.items(), key=lambda kv: -kv[1])
                total_v = sum(group.values())
                top5 = sum(n for _, n in ranked[:5])
                # ⚠️ THE ANTI-NOISE AFFORDANCE. 629 rows reads as a crisis;
                # "629, of which 74 % sit in 5 directories" reads as three
                # vendored trees and a results dump — which is what it is.
                # A count without its concentration invites the reader to treat
                # a widened filter as a regression.
                print(f"  {verdict} by directory ({total_v} total, "
                      f"{100 * top5 / total_v:.0f}% in the top 5 — a vendored "
                      f"or generated tree is ONE fact, not {top5}):")
                for d, n in ranked[:15]:
                    print(f"    {n:>4}  {d}")

        for f in found:
            if f["verdict"] in ACTIONABLE:
                actionable_total += 1
            if f["verdict"] == HOST_ONLY:
                host_only_total += 1
                print(f"    HOST_ONLY  [{f['kind']:<8}] {f['path']}")
            elif f["verdict"] == NAME_DRIFT:
                # ⛔ Never hidden behind --show-drifted. This is the class that
                # escaped as "weak evidence" and turned out to be the most
                # consequential find of the C110 sweep.
                print(f"    NAME_DRIFT [{f['kind']:<8}] {f['path']}\n"
                      f"               {f['note']}")
            elif f["verdict"] in (DRIFTED, NAME_ONLY) and args.show_drifted:
                print(f"    {f['verdict']:<10} {f['path']}  ({f['note']})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              default=str), encoding="utf-8")
        print(f"\n[drift] wrote {args.json}")

    print(f"\n[drift] TOTAL HOST-ONLY FILES (within the inclusion rule "
          f"printed above — NOT a census): {host_only_total}")
    print(f"[drift] TOTAL ACTIONABLE ({', '.join(ACTIONABLE)}): "
          f"{actionable_total}")
    if report["unreachable"]:
        print(f"[drift] {len(report['unreachable'])} host(s) UNREADABLE — this "
              f"run proves nothing about them: "
              f"{[u['host'] for u in report['unreachable']]}")
    if host_only_total:
        print("[drift] These exist in exactly one place. Rescue them into git.")
    return 1 if (actionable_total or report["unreachable"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
