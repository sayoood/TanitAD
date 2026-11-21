"""Report findings registered since LAB_BACKLOG.md was last written.

⛔ WHY THIS EXISTS. `LAB_BACKLOG.md`'s own update contract says any session that
produces a finding proposes a row IN THE SAME TURN. On 2026-08-31 that contract was
measured against reality and had failed silently for two days: the backlog had not been
touched since 2026-08-29 while MM-E10..MM-E19 and five gate problems were registered.
Its own text scored `hold-action` 0 hits, `action-deaf` 0, `L3` 0, `teacher-forced` 0,
`target construction` 0, `O11` 0 -- i.e. the Research Lab's seed list contained NONE of
the problems then gating v7, and the daily agent would have researched a stale list.

⭐ THE LESSON THE FIX ENCODES. The rule already existed and *I* broke it. A rule that
depends on the author remembering is not a mechanism; the same day produced the general
form -- "generate + assert, never type" -- and this is that applied to the backlog. It
makes the gap VISIBLE and non-zero-exit, so it fires instead of being remembered.

⚠️ GIT IS THE SOURCE OF TRUTH, NOT PROSE. Parsing the register for "new rows" would need
a heuristic for what counts as a finding, and heuristics rot. Commit history over the two
files that carry findings is exact, cheap, and cannot drift from what actually happened.

⚠️ OUTPUT IS DELIBERATELY ASCII-ONLY. The dev box is cp1252 and a script that dies on its
own house glyphs is a guard that fails closed for the wrong reason (MEASURED 2026-08-21).
Glyphs live in comments, never in anything printed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: Files whose commits ARE findings. Keep this list short and explicit -- a wildcard
#: would sweep in formatting commits and train readers to ignore the output.
FINDING_FILES = (
    "Project Steering/GOALS_AND_CLAIMS.md",
    "Project Steering/V7_LAUNCH_GATE.md",
)
BACKLOG = "TanitAD Research Lab/LAB_BACKLOG.md"


def _git(repo: Path, *args: str) -> str:
    """Run git with explicit dirs.

    ⚠️ The Drive mount makes a plain `cd && git` report "not a repository"
    intermittently; --git-dir/--work-tree is the form that survives it.
    """
    out = subprocess.run(
        ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"[drift] git failed: {' '.join(args)}\n{out.stderr.strip()}")
    return out.stdout


def last_touch(repo: Path, path: str) -> str | None:
    """The commit hash that last modified `path`, or None if it has none."""
    out = _git(repo, "log", "-1", "--format=%H", "--", path).strip()
    return out or None


def findings_since(repo: Path, since: str | None) -> list[tuple[str, str, str]]:
    """(hash, date, subject) for finding-commits after `since`, newest first."""
    rng = [f"{since}..HEAD"] if since else []
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for f in FINDING_FILES:
        out = _git(repo, "log", *rng, "--format=%H\x1f%ad\x1f%s", "--date=short",
                   "--", f)
        for line in out.splitlines():
            if not line.strip():
                continue
            h, d, s = line.split("\x1f", 2)
            if h not in seen:
                seen.add(h)
                rows.append((h, d, s))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--max", type=int, default=25, help="rows to print")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    b = last_touch(repo, BACKLOG)
    if b is None:
        print(f"[drift] {BACKLOG} has no commit history -- cannot compute drift.")
        return 2
    b_date = _git(repo, "log", "-1", "--format=%ad", "--date=short", b).strip()
    drift = findings_since(repo, b)

    print(f"[drift] backlog last written : {b[:9]}  {b_date}")
    print(f"[drift] finding-commits since: {len(drift)}")
    if not drift:
        print("[drift] OK - no findings registered since the backlog was last written.")
        return 0

    print()
    print("UNPROPOSED FINDINGS - each of these should have proposed a backlog row")
    print("in the same turn, per LAB_BACKLOG's update contract:")
    for h, d, s in drift[:a.max]:
        print(f"  {h[:9]}  {d}  {s[:96]}")
    if len(drift) > a.max:
        print(f"  ... and {len(drift) - a.max} more")
    print()
    print("=> Propose the missing rows under '## PROPOSED (unranked)', then re-run.")
    print("   The finder proposes and never self-ranks; the Master Mind ranks.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
