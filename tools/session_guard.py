"""session_guard — the D-026 stranded-work guardrail every agent runs at session end.

The 2026-07-17 fleet review found ~15k lines stranded across 8 unmerged agent
branches, uncommitted hub deliverables that never reached a commit, and INTAKE
packages sitting for days with an empty orchestrator verdict. That debt class is
structural, so this makes it a mechanical, self-explaining gate. It reports three
things and turns the blocking one into a non-zero exit code:

  (a) UNMERGED agent branches vs the integration tip — every ``agent/*`` branch
      with commits the tip does not contain. WARN + the exact merge command so the
      orchestrator (or the agent) can land it. Not blocking by default (the current
      session branch is legitimately ahead until its own end-of-run commit).

  (b) UNCOMMITTED deliverable files in the hub areas — any modified/untracked file
      under the research hub / steering docs / decision log. This is the "results
      left only in the working tree" failure. BLOCKING: a clean session-end tree is
      the whole point of the guard.

  (c) STALE INTAKE verdicts — ``Implementation/incoming/<date>-<slug>/INTAKE.md``
      files whose ``ORCHESTRATOR VERDICT`` is still the unfilled template, older than
      ``--max-intake-age-days`` (default 3), by the folder's date prefix. ESCALATE
      list (WARN) so the orchestrator triages them.

Exit codes: 0 = all clear (warnings may still print); 1 = a BLOCKING condition
(uncommitted hub deliverables, or any warning when ``--strict``); 3 = the tool
could not run git at all. ``--json`` emits the machine-readable report instead.

Stdlib-only, ASCII-clean stdout (the Windows cp1252 console lesson from ci_gate),
OS-agnostic: ``python tools/session_guard.py`` on any box, ``session_guard.ps1``
wraps it on the Windows dev machine.

Usage::

    python tools/session_guard.py                 # gate the current worktree
    python tools/session_guard.py --strict         # branches + stale INTAKEs also block
    python tools/session_guard.py --base origin/main  # tip = a different ref
    python tools/session_guard.py --json           # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

# Working-tree paths that hold agent deliverables. An uncommitted change under any
# of these at session end is the stranded-results failure the guard blocks on.
HUB_PREFIXES = (
    "TanitAD Research Hub/",
    "Project Steering/",
    "DECISIONS.md",
    "PROJECT_STATE.md",
)

# The unfilled verdict placeholder from INTAKE_TEMPLATE.md. A verdict still equal to
# (or containing) this — or empty — has not been triaged.
VERDICT_PLACEHOLDER = "integrate / integrate-with-changes / defer / reject"

# Non-committal tokens some authors drop into the verdict line before triage; these
# are "no decision yet" just as much as the empty placeholder.
VERDICT_UNFILLED_TOKENS = {"", "-", "--", "_pending_", "pending", "tbd", "todo", "none", "n/a"}


# --------------------------------------------------------------------------- git


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stdout with trailing whitespace
    removed. Leading whitespace is preserved on purpose: ``git status --porcelain``
    encodes the state in the first two columns, so its first line begins with a
    space for unstaged edits (` M path`) — a full ``.strip()`` would shift that
    line's fixed-offset path parse. Raise on non-zero exit."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} -> {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout.rstrip()


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel"))


def resolve_base(repo: Path, requested: str) -> str:
    """Resolve the integration tip. Try the requested ref, then a sensible chain."""
    candidates = [requested, "origin/main", "main", "HEAD"]
    for ref in candidates:
        if not ref:
            continue
        try:
            git(repo, "rev-parse", "--verify", "--quiet", ref)
            return ref
        except GitError:
            continue
    return "HEAD"


# ------------------------------------------------------------------------- checks


@dataclass
class UnmergedBranch:
    name: str
    ahead: int          # commits on the branch not in the tip
    is_current: bool
    merge_cmd: str


@dataclass
class StaleIntake:
    path: str
    slug: str
    age_days: int
    verdict: str        # the raw (unfilled) verdict text found


@dataclass
class Report:
    base: str
    base_sha: str
    current_branch: str
    unmerged: list[UnmergedBranch] = field(default_factory=list)
    uncommitted_hub: list[str] = field(default_factory=list)
    stale_intakes: list[StaleIntake] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def check_unmerged_branches(repo: Path, base: str) -> list[UnmergedBranch]:
    """Every ``agent/*`` branch with commits the tip (``base``) does not contain."""
    try:
        current = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except GitError:
        current = ""
    names = [
        ln.strip().lstrip("*+ ").strip()
        for ln in git(repo, "branch", "--list", "agent/*").splitlines()
        if ln.strip()
    ]
    out: list[UnmergedBranch] = []
    for name in names:
        # commits on the branch not reachable from the tip
        ahead = int(git(repo, "rev-list", "--count", f"{base}..{name}"))
        if ahead == 0:
            continue
        out.append(
            UnmergedBranch(
                name=name,
                ahead=ahead,
                is_current=(name == current),
                merge_cmd=f"git merge --no-ff {name}",
            )
        )
    out.sort(key=lambda b: (-b.ahead, b.name))
    return out


def check_uncommitted_hub(repo: Path, prefixes: tuple[str, ...] = HUB_PREFIXES) -> list[str]:
    """Modified/untracked working-tree paths under the hub areas (porcelain v1)."""
    status = git(repo, "status", "--porcelain")
    hits: list[str] = []
    for ln in status.splitlines():
        if not ln:
            continue
        path = ln[3:].strip()
        # rename lines look like "old -> new": keep the destination
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if any(path.startswith(p) for p in prefixes):
            hits.append(path)
    return sorted(set(hits))


def _slug_date(slug: str) -> date | None:
    """Parse the leading YYYY-MM-DD from an incoming-package folder name."""
    parts = slug.split("-")
    if len(parts) < 3:
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _verdict_unfilled(text: str) -> tuple[bool, str]:
    """Return (is_unfilled, raw_verdict). Reads the '- **Verdict:**' line under the
    ORCHESTRATOR VERDICT heading; unfilled if empty, still the '/'-menu, or the
    placeholder string."""
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("## orchestrator verdict"):
            in_section = True
            continue
        if in_section and line.startswith("- **Verdict:**"):
            value = line.split("**Verdict:**", 1)[1].strip()
            norm = value.strip("*_` ").lower()
            unfilled = (
                "/" in value
                or VERDICT_PLACEHOLDER in value
                or norm in VERDICT_UNFILLED_TOKENS
            )
            return unfilled, value
    # No verdict line at all -> treat as unfilled.
    return True, ""


def check_stale_intakes(repo: Path, now: date, max_age_days: int) -> list[StaleIntake]:
    hub = repo / "TanitAD Research Hub"
    out: list[StaleIntake] = []
    if not hub.is_dir():
        return out
    for intake in hub.glob("*/Implementation/incoming/*/INTAKE.md"):
        slug = intake.parent.name
        try:
            text = intake.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        unfilled, verdict = _verdict_unfilled(text)
        if not unfilled:
            continue
        d = _slug_date(slug)
        age = (now - d).days if d else max_age_days + 1  # undateable -> assume stale
        if age > max_age_days:
            rel = intake.relative_to(repo).as_posix()
            out.append(StaleIntake(path=rel, slug=slug, age_days=age, verdict=verdict))
    out.sort(key=lambda s: (-s.age_days, s.path))
    return out


def build_report(repo: Path, base: str, now: date, max_intake_age_days: int) -> Report:
    resolved = resolve_base(repo, base)
    try:
        current = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except GitError:
        current = ""
    return Report(
        base=resolved,
        base_sha=git(repo, "rev-parse", "--short", resolved),
        current_branch=current,
        unmerged=check_unmerged_branches(repo, resolved),
        uncommitted_hub=check_uncommitted_hub(repo),
        stale_intakes=check_stale_intakes(repo, now, max_intake_age_days),
    )


# ------------------------------------------------------------------------- render


def render(report: Report, strict: bool) -> tuple[str, int]:
    """Return (human text, exit code). Exit 1 if a blocking condition trips."""
    lines: list[str] = []
    blocking = False

    lines.append(f"session_guard: tip = {report.base} ({report.base_sha}); "
                 f"branch = {report.current_branch or '(detached)'}")

    # (b) uncommitted hub deliverables -- BLOCKING
    if report.uncommitted_hub:
        blocking = True
        lines.append("")
        lines.append(f"[BLOCK] {len(report.uncommitted_hub)} uncommitted hub "
                     f"deliverable file(s) -- commit or discard before session end:")
        for p in report.uncommitted_hub:
            lines.append(f"    {p}")
    else:
        lines.append("[ok]    hub working tree clean (no stranded deliverables)")

    # (a) unmerged agent branches -- WARN (block only in --strict)
    strays = [b for b in report.unmerged if not b.is_current]
    if strays:
        if strict:
            blocking = True
        tag = "[BLOCK]" if strict else "[WARN] "
        lines.append("")
        lines.append(f"{tag} {len(strays)} unmerged agent branch(es) vs tip "
                     f"(stranded work -- open the merge):")
        for b in strays:
            lines.append(f"    {b.name}  (+{b.ahead})   {b.merge_cmd}")
    else:
        lines.append("[ok]    no stranded agent branches vs tip")
    cur = [b for b in report.unmerged if b.is_current]
    if cur:
        lines.append(f"[info]  current branch {cur[0].name} is +{cur[0].ahead} vs tip "
                     f"(expected -- merge at session end)")

    # (c) stale INTAKE verdicts -- WARN (block only in --strict)
    if report.stale_intakes:
        if strict:
            blocking = True
        tag = "[BLOCK]" if strict else "[WARN] "
        lines.append("")
        lines.append(f"{tag} {len(report.stale_intakes)} INTAKE package(s) with an "
                     f"unfilled verdict older than the age budget -- escalate:")
        for s in report.stale_intakes:
            lines.append(f"    {s.path}  ({s.age_days}d old)")
    else:
        lines.append("[ok]    no stale INTAKE verdicts")

    lines.append("")
    lines.append("RESULT: " + ("BLOCK (session-end gate failed)" if blocking
                               else "PASS (clear to end session)"))
    return "\n".join(lines), (1 if blocking else 0)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D-026 stranded-work session-end guard.")
    parser.add_argument("--repo", default=".", help="repo/worktree root (default: cwd)")
    parser.add_argument("--base", default="HEAD",
                        help="integration tip ref (default: HEAD; falls back "
                             "origin/main -> main -> HEAD if absent)")
    parser.add_argument("--max-intake-age-days", type=int, default=3,
                        help="stale-verdict age budget in days (default: 3)")
    parser.add_argument("--now", default="",
                        help="reference date YYYY-MM-DD for age math (default: today)")
    parser.add_argument("--strict", action="store_true",
                        help="also block on unmerged branches and stale INTAKEs")
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable report instead of text")
    args = parser.parse_args(argv)

    try:
        repo = repo_root(Path(args.repo).resolve())
    except (GitError, FileNotFoundError) as exc:
        print(f"session_guard: not a git repo / git unavailable: {exc}", file=sys.stderr)
        return 3

    now = date.today()
    if args.now:
        now = datetime.strptime(args.now, "%Y-%m-%d").date()

    try:
        report = build_report(repo, args.base, now, args.max_intake_age_days)
    except GitError as exc:
        print(f"session_guard: git failed: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(report.to_json())
        # JSON mode still returns the gate exit code for scripting.
        _, code = render(report, args.strict)
        return code

    text, code = render(report, args.strict)
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
