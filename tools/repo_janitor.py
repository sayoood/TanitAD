"""repo_janitor — make search in this repo trustworthy again.

The failure this exists for is not untidiness, it is a **false claim that reached
the PI**. On 2026-07-24 two modules were reported STRANDED on worktrees and
absent from main. Both were in main. Root cause (RETRACTION_LOG 07-24, class C8):
``Glob`` sorts by mtime and truncates at 100, so with 51 registered worktrees and
94 ``Implementation/incoming/`` bundles, the freshly-touched worktree copies
filled the result window and the older main-tree originals fell off the end. The
sprawl IS the measurement error.

Three reports, one command:

  (a) **WORKTREES.** ``git worktree prune`` clears records whose directory is
      gone, then every remaining worktree is scored for *unique commits* against
      the integration tip. A worktree whose branch is 0 commits ahead and whose
      tree is clean holds nothing that is not already in the tip -- a
      safe-to-delete candidate. **It is REPORTED, not deleted.** Deletion needs
      an explicit ``--delete``, and even then only candidates qualify: this
      program has lost work to single-disk artifacts too often to let a janitor
      guess (AGENT_OPERATING_STANDARD, "finish before you start").

  (b) **THE INCOMING LEDGER.** Every ``**/Implementation/incoming/<date>-<slug>/``
      bundle, dated, aged, sized, with its INTAKE verdict state. ~94 of them are
      live right now; without a ledger the old ones are indistinguishable from
      the new ones in any directory listing.

  (c) **FUTURE-DATED ITEMS.** Files and bundle folders dated AHEAD of the system
      clock. This repo genuinely produces them -- the narrative clock in the hub
      notes runs ahead of wall-clock -- and a future mtime poisons every
      mtime-sorted tool, which is precisely how (a) went wrong.

Exit codes: 0 = report produced (findings are informational) · 1 = a hard
condition under ``--fail-on`` · 2 = usage · 3 = git unavailable.

Usage::

    python tools/repo_janitor.py                        # report everything
    python tools/repo_janitor.py --ledger-out LEDGER.md # write the dated ledger
    python tools/repo_janitor.py --worktrees-only
    python tools/repo_janitor.py --delete               # remove ONLY safe candidates
    python tools/repo_janitor.py --json janitor.json

Stdlib-only, ASCII-clean stdout, OS-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

INCOMING_GLOB = "*/Implementation/incoming/*"
HUB_DIR = "TanitAD Research Hub"
VERDICT_PLACEHOLDER = "integrate / integrate-with-changes / defer / reject"
VERDICT_UNFILLED = {"", "-", "--", "_pending_", "pending", "tbd", "todo", "none", "n/a"}


def ascii_safe(s: str) -> str:
    """Windows cp1252 console guard -- see tools/README.md. Bundle slugs and
    commit subjects in this repo carry em dashes and emoji."""
    return s.encode("ascii", "replace").decode("ascii")


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, errors="replace", encoding="utf-8")
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} -> {proc.returncode}: "
                       f"{proc.stderr.strip()}")
    return proc.stdout.rstrip("\n")


# ------------------------------------------------------------------------ worktrees


@dataclass
class Worktree:
    path: str
    branch: str
    head: str
    ahead: int = -1          # commits not in the tip; -1 = could not compute
    dirty: int = -1          # changed paths; -1 = not checked
    exists: bool = True
    candidate: bool = False
    reason: str = ""


def parse_worktrees(repo: Path) -> list[Worktree]:
    out: list[Worktree] = []
    cur: dict[str, str] = {}
    for line in git(repo, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line.strip():
            if cur.get("worktree"):
                out.append(Worktree(
                    path=cur["worktree"],
                    branch=cur.get("branch", "").replace("refs/heads/", ""),
                    head=cur.get("HEAD", "")[:8],
                    exists=Path(cur["worktree"]).is_dir()))
            cur = {}
            continue
        key, _, val = line.partition(" ")
        cur[key] = val
    return out


def score_worktrees(repo: Path, wts: list[Worktree], tip: str,
                    check_status: bool) -> list[Worktree]:
    """Ahead-count + dirtiness for every worktree except the one we run in.

    ``--fast`` skips the status probe: 51 ``git status`` calls across a Google
    Drive mount is the slow part, and the ahead-count alone already answers
    "does this hold anything unique"."""
    main_path = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    for w in wts:
        same = Path(w.path).resolve() == main_path
        if same:
            w.reason = "the worktree this command is running in"
            continue
        if not w.exists:
            w.reason = "directory is GONE (prune clears the record)"
            continue
        ref = w.branch or w.head
        if ref:
            try:
                w.ahead = int(git(repo, "rev-list", "--count", f"{tip}..{ref}"))
            except (GitError, ValueError):
                w.ahead = -1
        if check_status:
            try:
                st = git(Path(w.path), "status", "--porcelain",
                         "--untracked-files=normal", check=False)
                w.dirty = len([ln for ln in st.splitlines() if ln.strip()])
            except (GitError, OSError):
                w.dirty = -1
        if w.ahead == 0 and w.dirty == 0:
            w.candidate = True
            w.reason = f"0 unique commits vs {tip}, clean tree"
        elif w.ahead == 0 and not check_status:
            w.reason = (f"0 unique commits vs {tip}; tree not checked "
                        f"(--fast) -- re-run without --fast before deleting")
        elif w.ahead > 0:
            w.reason = f"HOLDS {w.ahead} commit(s) not in {tip} -- do NOT delete"
        elif w.dirty and w.dirty > 0:
            w.reason = f"{w.dirty} uncommitted path(s) in the tree -- do NOT delete"
        else:
            w.reason = "could not score (missing branch/ref)"
    return wts


# ------------------------------------------------------------------ incoming ledger


@dataclass
class Bundle:
    path: str
    area: str
    slug: str
    dated: str          # YYYY-MM-DD or ""
    age_days: int | None
    n_files: int
    bytes: int
    intake: str         # "missing" | "unfilled" | verdict text
    future: bool = False


def _slug_date(slug: str) -> date | None:
    parts = slug.split("-")
    if len(parts) < 3:
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _verdict(text: str) -> str:
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("## orchestrator verdict"):
            in_section = True
            continue
        if in_section and line.startswith("- **Verdict:**"):
            value = line.split("**Verdict:**", 1)[1].strip()
            norm = value.strip("*_` ").lower()
            if "/" in value or VERDICT_PLACEHOLDER in value or norm in VERDICT_UNFILLED:
                return "unfilled"
            return value[:60]
    return "unfilled"


def scan_incoming(repo: Path, today: date) -> list[Bundle]:
    hub = repo / HUB_DIR
    out: list[Bundle] = []
    if not hub.is_dir():
        return out
    for d in sorted(hub.glob(INCOMING_GLOB)):
        if not d.is_dir():
            continue
        slug = d.name
        n_files = 0
        total = 0
        for root, _dirs, files in os.walk(d):
            for f in files:
                n_files += 1
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
        intake_p = d / "INTAKE.md"
        if intake_p.is_file():
            try:
                intake = _verdict(intake_p.read_text(encoding="utf-8",
                                                     errors="replace"))
            except OSError:
                intake = "unfilled"
        else:
            intake = "missing"
        dt = _slug_date(slug)
        age = (today - dt).days if dt else None
        out.append(Bundle(
            path=d.relative_to(repo).as_posix(),
            area=d.relative_to(hub).parts[0],
            slug=slug,
            dated=dt.isoformat() if dt else "",
            age_days=age,
            n_files=n_files,
            bytes=total,
            intake=intake,
            future=bool(age is not None and age < 0)))
    out.sort(key=lambda b: (-(b.age_days if b.age_days is not None else 9999),
                            b.path))
    return out


# ------------------------------------------------------------------- future-dated


@dataclass
class FutureItem:
    path: str
    kind: str        # "mtime" | "slug"
    stamp: str
    ahead_s: int


def scan_future(repo: Path, now: datetime, skew_s: int,
                limit: int) -> list[FutureItem]:
    """Anything stamped after the system clock.

    The hub's narrative clock legitimately runs ahead of wall-clock here, so
    these are reported as an ARTIFACT to be aware of, not as corruption -- but
    they must be visible, because a future mtime sorts to the top of every
    mtime-ordered tool and that is the mechanism behind the 07-24 stranding
    claim. Only git-tracked files are stat'd: walking the whole Drive tree costs
    minutes and adds nothing (an untracked temp file poisons nobody's Glob of
    the source tree)."""
    out: list[FutureItem] = []
    cutoff = now.timestamp() + skew_s
    try:
        tracked = git(repo, "ls-files", "-z").split("\0")
    except GitError:
        tracked = []
    for rel in tracked:
        if not rel:
            continue
        try:
            st = (repo / rel).stat()
        except OSError:
            continue
        if st.st_mtime > cutoff:
            out.append(FutureItem(
                path=rel, kind="mtime",
                stamp=datetime.fromtimestamp(st.st_mtime,
                                             timezone.utc).isoformat(timespec="seconds"),
                ahead_s=int(st.st_mtime - now.timestamp())))
        if len(out) >= limit:
            break
    out.sort(key=lambda f: -f.ahead_s)
    return out


# ---------------------------------------------------------------------- the report


@dataclass
class JanitorReport:
    repo: str
    tip: str
    generated: str
    pruned: str = ""
    worktrees: list[Worktree] = field(default_factory=list)
    bundles: list[Bundle] = field(default_factory=list)
    future: list[FutureItem] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"repo": self.repo, "tip": self.tip, "generated": self.generated,
                "pruned": self.pruned,
                "worktrees": [asdict(w) for w in self.worktrees],
                "bundles": [asdict(b) for b in self.bundles],
                "future": [asdict(f) for f in self.future],
                "deleted": self.deleted}


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def render(rep: JanitorReport, stale_days: int) -> str:
    lines = [f"repo_janitor: {rep.repo}",
             f"  tip = {rep.tip} | generated {rep.generated}"]
    if rep.pruned:
        lines.append(f"  git worktree prune: {rep.pruned}")

    cands = [w for w in rep.worktrees if w.candidate]
    holders = [w for w in rep.worktrees if w.ahead > 0]
    lines.append("")
    lines.append(f"[worktrees] {len(rep.worktrees)} registered | "
                 f"{len(cands)} SAFE-TO-DELETE candidate(s) | "
                 f"{len(holders)} hold unique commits")
    for w in cands:
        lines.append(f"    CANDIDATE  {w.path}  [{w.branch or w.head}]  {w.reason}")
    for w in sorted(holders, key=lambda x: -x.ahead):
        lines.append(f"    KEEP  +{w.ahead:<4} {w.path}  [{w.branch or w.head}]")
    others = [w for w in rep.worktrees if not w.candidate and w.ahead <= 0]
    for w in others[:8]:
        lines.append(f"    ----  {w.path}  [{w.branch or w.head}]  {w.reason}")
    if len(others) > 8:
        lines.append(f"    ---- ... {len(others) - 8} more unscored/neutral "
                     f"worktree(s); --json for the full list")
    if rep.deleted:
        lines.append(f"    DELETED {len(rep.deleted)}: " + ", ".join(rep.deleted))
    elif cands:
        lines.append("    (reported only -- pass --delete to remove the "
                     "CANDIDATE rows; nothing else is ever touched)")

    stale = [b for b in rep.bundles
             if b.age_days is not None and b.age_days > stale_days]
    untriaged = [b for b in rep.bundles if b.intake in ("missing", "unfilled")]
    lines.append("")
    lines.append(f"[incoming]  {len(rep.bundles)} bundle(s) | "
                 f"{len(stale)} older than {stale_days}d | "
                 f"{len(untriaged)} with no filled INTAKE verdict | "
                 f"{_mb(sum(b.bytes for b in rep.bundles))} total")
    for b in rep.bundles[:12]:
        age = f"{b.age_days}d" if b.age_days is not None else "undated"
        lines.append(f"    {age:>7}  {b.intake:<10} {b.n_files:>4}f "
                     f"{_mb(b.bytes):>9}  {b.path}")
    if len(rep.bundles) > 12:
        lines.append(f"    ... {len(rep.bundles) - 12} more "
                     f"(use --ledger-out for the full dated ledger)")

    fut_bundles = [b for b in rep.bundles if b.future]
    lines.append("")
    lines.append(f"[future]    {len(rep.future)} tracked file(s) with a future "
                 f"mtime | {len(fut_bundles)} bundle(s) with a future date slug")
    for f in rep.future[:10]:
        lines.append(f"    +{f.ahead_s / 3600:6.1f} h  {f.stamp}  {f.path}")
    for b in fut_bundles[:10]:
        lines.append(f"    slug {b.dated} ({-(b.age_days or 0)}d ahead)  {b.path}")
    if rep.future or fut_bundles:
        lines.append("    NOTE: this repo's narrative clock legitimately runs "
                     "ahead of wall-clock. These are not corruption -- but they "
                     "sort FIRST in every mtime-ordered tool (the 07-24 "
                     "'stranded modules' false claim), so verify presence with "
                     "`git ls-files`, never with a truncated Glob.")
    return ascii_safe("\n".join(lines))


def render_ledger(rep: JanitorReport, stale_days: int) -> str:
    rows = ["# Incoming-bundle ledger",
            "",
            f"- **Generated:** {rep.generated}",
            f"- **Repo:** `{rep.repo}`  |  **tip:** `{rep.tip}`",
            f"- **Bundles:** {len(rep.bundles)}  |  "
            f"**total size:** {_mb(sum(b.bytes for b in rep.bundles))}",
            f"- **Stale threshold:** {stale_days} days",
            "",
            "Produced by `tools/repo_janitor.py --ledger-out`. A bundle sitting "
            "in `incoming/` is work that has not been integrated; age is the "
            "signal, and an unfilled INTAKE verdict older than a few days is an "
            "escalation, not a backlog item.",
            "",
            "| age | date | area | bundle | files | size | INTAKE verdict |",
            "|---:|---|---|---|---:|---:|---|"]
    for b in rep.bundles:
        age = f"{b.age_days}d" if b.age_days is not None else "?"
        if b.future:
            age = f"{-b.age_days}d AHEAD"
        rows.append(f"| {age} | {b.dated or '-'} | {b.area} | `{b.slug}` | "
                    f"{b.n_files} | {_mb(b.bytes)} | {b.intake} |")
    return ascii_safe("\n".join(rows) + "\n")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Worktree/incoming sprawl janitor -- reports by default, "
                    "deletes only with --delete.")
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument("--tip", default="HEAD",
                    help="integration tip a worktree is scored against "
                         "(default HEAD; origin/main is intentionally diverged "
                         "in this repo)")
    ap.add_argument("--no-prune", action="store_true",
                    help="skip `git worktree prune` (it only drops records whose "
                         "directory is already gone)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the per-worktree `git status` probe (faster on a "
                         "Drive mount; --delete then refuses to act)")
    ap.add_argument("--delete", action="store_true",
                    help="remove worktrees that are 0 commits ahead AND clean. "
                         "Never uses --force; anything holding work is skipped.")
    ap.add_argument("--stale-days", type=int, default=14,
                    help="incoming bundles older than this are called out "
                         "(default 14)")
    ap.add_argument("--skew-s", type=int, default=300,
                    help="clock skew tolerated before an mtime counts as "
                         "future-dated (default 300 s)")
    ap.add_argument("--future-limit", type=int, default=500,
                    help="stop after this many future-dated files (default 500)")
    ap.add_argument("--worktrees-only", action="store_true")
    ap.add_argument("--incoming-only", action="store_true")
    ap.add_argument("--ledger-out", default=None,
                    help="write the dated incoming ledger to this markdown file")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--fail-on", default="none",
                    choices=("none", "future", "stale", "any"),
                    help="exit 1 when the named condition is present "
                         "(default none: this is a reporter, not a gate)")
    args = ap.parse_args(argv)

    try:
        repo = Path(git(Path(args.repo).resolve(), "rev-parse", "--show-toplevel"))
    except (GitError, FileNotFoundError, OSError) as exc:
        print(f"repo_janitor: not a git repo / git unavailable: {exc}",
              file=sys.stderr)
        return 3

    now = datetime.now()
    rep = JanitorReport(repo=str(repo), tip=args.tip,
                        generated=now.isoformat(timespec="seconds"))

    do_wt = not args.incoming_only
    do_in = not args.worktrees_only

    if do_wt:
        if not args.no_prune:
            before = len(parse_worktrees(repo))
            git(repo, "worktree", "prune", check=False)
            after_list = parse_worktrees(repo)
            rep.pruned = (f"{before - len(after_list)} record(s) removed "
                          f"({before} -> {len(after_list)})")
        rep.worktrees = score_worktrees(repo, parse_worktrees(repo), args.tip,
                                        check_status=not args.fast)
        if args.delete:
            if args.fast:
                print("repo_janitor: --delete refuses to run with --fast "
                      "(a clean-tree check is mandatory before removing "
                      "anything)", file=sys.stderr)
                return 2
            for w in rep.worktrees:
                if not w.candidate:
                    continue
                out = git(repo, "worktree", "remove", w.path, check=False)
                if Path(w.path).is_dir():
                    print(f"repo_janitor: could not remove {w.path}: "
                          f"{ascii_safe(out)}", file=sys.stderr)
                else:
                    rep.deleted.append(w.path)

    if do_in:
        rep.bundles = scan_incoming(repo, now.date())
    rep.future = scan_future(repo, now, args.skew_s, args.future_limit)

    print(render(rep, args.stale_days))

    if args.ledger_out:
        out = Path(args.ledger_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_ledger(rep, args.stale_days), encoding="utf-8")
        print(f"\nrepo_janitor: ledger written to {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=2),
                                   encoding="utf-8")

    stale = [b for b in rep.bundles
             if b.age_days is not None and b.age_days > args.stale_days]
    fut = rep.future or [b for b in rep.bundles if b.future]
    trip = ((args.fail_on == "future" and fut)
            or (args.fail_on == "stale" and stale)
            or (args.fail_on == "any" and (fut or stale)))
    return 1 if trip else 0


if __name__ == "__main__":
    raise SystemExit(main())
