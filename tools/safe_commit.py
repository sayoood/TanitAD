"""safe_commit — the ONLY sanctioned way to make a commit in this repo.

It mechanises `CLAUDE.md` section "Git hygiene", which exists because the same two
failures keep costing us work:

  (1) **The whole-index sweep.** ``git commit`` commits the ENTIRE INDEX, not the
      files you just ``git add``ed. With six agents staging concurrently -- the
      normal state here -- a "quick commit of my thing" silently swallows a
      sibling's half-finished code under the wrong message. Measured twice in one
      session (``60265d3`` swallowed the eval tooling; ``3d41bd0`` swallowed
      REF-C v1.2's in-progress rescorer).

  (2) **The pathspec segfault.** The obvious fix -- ``git commit -- <pathspec>``,
      git's *partial-commit* path -- **CRASHES INTERMITTENTLY on this repo**
      (measured 2026-07-25: exit 139 under MSYS git and 0xC0000005 under native
      Windows git, so it is not the shell; fsmonitor was already false). Three
      root-cause theories were asserted and falsified in one session
      (RETRACTION_LOG 07-25, class C8). **This tool therefore NEVER emits a
      pathspec.** It uses the non-partial ``git commit -F <msgfile>`` path, which
      has never crashed, and makes the whole-index risk safe by *declaration*
      instead: you say what you staged, and the tool aborts if the index holds
      anything else.

  (3) **The phantom lock.** Every crash leaves a stale ``.git/index.lock``, so the
      next attempt dies with *"Another git process seems to be running"* -- which
      reads like contention but is debris. The tool confirms no git process is
      alive, removes the lock, and retries.

  (4) **Secrets.** ``Keys.txt`` is git-ignored and must never be committed. Any
      staged path that git *itself* considers ignored, any credential-shaped
      filename, and any staged *content* matching a provider token pattern is a
      hard refusal. Findings are printed REDACTED -- the tool never echoes a
      token, not even in a diagnostic.

Exit codes: 0 = committed (or --dry-run OK) · 1 = REFUSED by a guard
2 = usage error · 3 = git unavailable / not a repo · 4 = commit failed after
retries.

Usage::

    # the normal call: declare what you staged, pass a message
    python tools/safe_commit.py -p tools/ -p "TanitAD Research Hub/Data Engineering" \\
        -m "tools: wave-1 ops tooling"

    # a long message lives in a file (never shell-quote a multi-line message)
    python tools/safe_commit.py --paths-from staged.txt -F .git/COMMIT_MSG

    # see exactly what would happen, touch nothing
    python tools/safe_commit.py -p tools/ -m "..." --dry-run

    # the index legitimately holds a sibling's work and you have READ it:
    # this is CLAUDE.md step 1 -- the tool prints the full index, names the
    # foreign entries in the commit message, and proceeds.
    python tools/safe_commit.py --accept-index -m "..."

    python tools/safe_commit.py --print-index      # just show the index, exit 0

Stdlib-only, ASCII-clean stdout, OS-agnostic (same file runs on the pods).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

# --------------------------------------------------------------------------- secrets

# Tier A -- provider token SHAPES. A hit here is an unconditional refusal: no
# flag overrides it. The HF pattern demands >=30 trailing chars because a real
# token is `hf_` + 34 alnum, while the loose `hf_[A-Za-z0-9]+` from the brief
# also matches ordinary identifiers this repo genuinely commits
# (`hf_export.py`, `hf_repo_state_2026-07-25.json`, `hf_relay`). Tier C below
# keeps the loose pattern as an advisory so nothing is silently dropped.
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("aws-akid", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Tier C -- the loose pattern from the brief. Advisory only (it fires on this
# repo's own legitimate `hf_*` filenames), but a count is always printed so an
# operator can look.
LOOSE_HF = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")

# Tier B -- credential-shaped PATHS. Refusable, overridable only with the
# explicit --allow-secret-path flag (there is a legitimate case: committing a
# `*.pem` fixture for a test).
SECRET_PATH_GLOBS = (
    "Keys.txt", "keys.txt", "KEYS.txt",
    "*.pem", "*.key", "*.pfx", "*.p12",
    ".env", ".env.*", "*.env",
    "*credential*", "*secret*", "*_token.txt", "*token.json",
    "id_rsa", "id_ed25519", "*.ppk",
    ".netrc", "_netrc",
)

# Exit codes that mean "the git process died", not "git reported an error".
# 139   = 128+SIGSEGV under MSYS/POSIX shells
# 3221225477 = 0xC0000005 STATUS_ACCESS_VIOLATION under native Windows git
CRASH_CODES = {139, -11, 3221225477, -1073741819}


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(repo),
                          capture_output=True, text=True, errors="replace")
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} -> {proc.returncode}: "
                       f"{proc.stderr.strip()}")
    return proc


def git_out(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.rstrip("\n")


def repo_root(start: Path) -> Path:
    return Path(git_out(start, "rev-parse", "--show-toplevel"))


# ------------------------------------------------------------------- the index lock


def git_processes_alive() -> tuple[bool, str]:
    """Best-effort 'is any git process running right now'.

    Deliberately matched on the IMAGE NAME, never on a command line: CLAUDE.md's
    trap list records that ``pgrep -f <name>`` self-matches the caller's own ssh
    command and silently kills the session. Returns (alive, how_we_know); an
    undeterminable answer is reported as alive=False with how="unknown" so the
    caller can fall back on the lock's age."""
    try:
        if os.name == "nt":
            proc = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq git.exe", "/NH"],
                capture_output=True, text=True, errors="replace", timeout=20)
            if proc.returncode != 0:
                return False, "unknown"
            return ("git.exe" in proc.stdout), "tasklist"
        proc = subprocess.run(["ps", "-e", "-o", "comm="],
                              capture_output=True, text=True,
                              errors="replace", timeout=20)
        if proc.returncode != 0:
            return False, "unknown"
        names = {ln.strip().rsplit("/", 1)[-1] for ln in proc.stdout.splitlines()}
        return ("git" in names), "ps"
    except (OSError, subprocess.SubprocessError):
        return False, "unknown"


def clear_stale_lock(repo: Path, max_age_s: float, force: bool,
                     log=print) -> bool:
    """Remove ``.git/index.lock`` when it is provably debris. Returns True if a
    lock was removed.

    A crashed ``git commit`` always leaves this file, and the NEXT attempt then
    fails with *"Another git process seems to be running"* -- an error that
    describes contention while the actual cause is a corpse. Guard rails: never
    remove it while a git process is alive (that WOULD be contention), and, when
    liveness cannot be determined, only remove a lock older than ``max_age_s``."""
    gitdir = Path(git_out(repo, "rev-parse", "--git-dir"))
    if not gitdir.is_absolute():
        gitdir = repo / gitdir
    lock = gitdir / "index.lock"
    if not lock.exists():
        return False
    alive, how = git_processes_alive()
    age = max(0.0, time.time() - lock.stat().st_mtime)
    if alive and not force:
        log(f"[safe_commit] index.lock present and a git process IS alive "
            f"(via {how}) -- NOT removing; this looks like real contention")
        return False
    if how == "unknown" and age < max_age_s and not force:
        log(f"[safe_commit] index.lock present, {age:.1f}s old, liveness "
            f"undeterminable -- waiting (needs >{max_age_s:.0f}s or --force-lock)")
        return False
    try:
        lock.unlink()
    except OSError as exc:
        log(f"[safe_commit] could not remove {lock}: {exc}")
        return False
    log(f"[safe_commit] removed stale {lock} ({age:.1f}s old, git alive={alive} "
        f"via {how}) -- the index itself survives this intact")
    return True


# ------------------------------------------------------------------------- the index


def staged_paths(repo: Path) -> list[str]:
    """The index, as repo-relative POSIX paths.

    ``-z`` because git otherwise C-quotes any path with a space -- and the hub
    tree is full of them (``TanitAD Research Hub/...``)."""
    out = git(repo, "diff", "--cached", "--name-only", "-z").stdout
    return [p for p in out.split("\0") if p]


def _glob_match(path: str, pattern: str) -> bool:
    """Segment-wise glob: ``*`` stops at ``/``, ``**`` spans directories.

    Plain ``fnmatch`` lets ``*`` cross a separator, so ``tools/*.py`` would also
    cover ``tools/sub/deep.py``. For a DECLARATION guard that is the wrong
    direction of error -- an over-broad declaration silently re-admits exactly
    the sibling's-work-swept-in failure this tool exists to stop."""
    pp, pt = path.split("/"), pattern.split("/")

    def rec(i: int, j: int) -> bool:
        while j < len(pt):
            if pt[j] == "**":
                if j + 1 == len(pt):
                    return True
                return any(rec(k, j + 1) for k in range(i, len(pp) + 1))
            if i >= len(pp) or not fnmatch(pp[i], pt[j]):
                return False
            i += 1
            j += 1
        return i == len(pp)

    return rec(0, 0)


def matches_declared(path: str, declared: list[str]) -> bool:
    """A declared entry covers ``path`` if it equals it, is a directory prefix of
    it, or is a segment-wise glob it satisfies."""
    for d in declared:
        d = d.replace("\\", "/").strip()
        if not d:
            continue
        if path == d:
            return True
        if d in (".", "./") or path.startswith(d.rstrip("/") + "/"):
            return True
        if _glob_match(path, d):
            return True
    return False


@dataclass
class SecretFinding:
    kind: str          # "ignored-path" | "secret-path" | "token" | "loose"
    path: str
    detail: str        # ALWAYS redacted


def _redact(match: str) -> str:
    """Never echo a credential. Keep only the scheme prefix and the length."""
    head = match[:3] if len(match) > 3 else "?"
    return f"{head}***<{len(match)} chars, redacted>"


def scan_secrets(repo: Path, paths: list[str]) -> tuple[list[SecretFinding],
                                                        list[SecretFinding]]:
    """Return (blocking, advisory) findings for the staged set.

    Three independent probes, because one is not absence (CLAUDE.md operating
    standard 2): git's OWN ignore verdict on each staged path (the only way
    ``Keys.txt`` can reach the index is ``git add -f``, and that leaves this
    fingerprint), the filename shape, and the staged CONTENT."""
    blocking: list[SecretFinding] = []
    advisory: list[SecretFinding] = []
    if not paths:
        return blocking, advisory

    # (1) git's own ignore verdict -- the tool that OWNS the fact, per CLAUDE.md
    #     operating standard 2. `--no-index` makes check-ignore answer for paths
    #     that are ALREADY staged, which is exactly the Keys.txt case; without it
    #     git reports "not ignored" for anything in the index and the probe is
    #     vacuous. Paths go in over stdin, NUL-separated, so a path containing a
    #     space (the whole hub tree) survives.
    proc = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                          cwd=str(repo), input="\0".join(paths) + "\0",
                          capture_output=True, text=True, errors="replace")
    for p in (x for x in proc.stdout.split("\0") if x):
        blocking.append(SecretFinding(
            "ignored-path", p,
            "staged despite being git-IGNORED (only 'git add -f' does this)"))

    # (2) credential-shaped filenames.
    for p in paths:
        base = p.rsplit("/", 1)[-1]
        if any(fnmatch(base, g) for g in SECRET_PATH_GLOBS):
            blocking.append(SecretFinding(
                "secret-path", p, f"filename matches a credential glob ({base})"))

    # (3) staged CONTENT. Read the cached diff once; only added lines matter.
    diff = git(repo, "diff", "--cached", "--unified=0", "--no-color",
               check=False).stdout
    cur = "<unknown>"
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for name, pat in TOKEN_PATTERNS:
            m = pat.search(line)
            if m:
                blocking.append(SecretFinding(
                    "token", cur, f"{name} token shape: {_redact(m.group(0))}"))
        if not any(pat.search(line) for _, pat in TOKEN_PATTERNS):
            m = LOOSE_HF.search(line)
            if m:
                advisory.append(SecretFinding(
                    "loose", cur,
                    f"loose hf_* identifier {_redact(m.group(0))} "
                    f"(usually a legitimate filename/symbol in this repo)"))
    # de-duplicate, keep order
    def _dedup(rows: list[SecretFinding]) -> list[SecretFinding]:
        seen, out = set(), []
        for r in rows:
            k = (r.kind, r.path, r.detail)
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out
    return _dedup(blocking), _dedup(advisory)


# ------------------------------------------------------------------------- the commit


@dataclass
class Plan:
    repo: Path
    index: list[str] = field(default_factory=list)
    declared: list[str] = field(default_factory=list)
    foreign: list[str] = field(default_factory=list)
    blocking: list[SecretFinding] = field(default_factory=list)
    advisory: list[SecretFinding] = field(default_factory=list)
    branch: str = ""
    head: str = ""
    refusals: list[str] = field(default_factory=list)


def build_plan(repo: Path, declared: list[str], accept_index: bool,
               allow_main: bool, allow_secret_path: bool,
               allow_empty: bool) -> Plan:
    plan = Plan(repo=repo, declared=declared)
    plan.index = staged_paths(repo)
    plan.branch = git_out(repo, "rev-parse", "--abbrev-ref", "HEAD")
    plan.head = git_out(repo, "rev-parse", "HEAD")
    plan.blocking, plan.advisory = scan_secrets(repo, plan.index)

    if not plan.index and not allow_empty:
        plan.refusals.append("the index is EMPTY -- nothing staged to commit "
                             "(git add your deliverables first, or --allow-empty)")

    if plan.branch == "main" and not allow_main:
        plan.refusals.append("refusing to commit on 'main' -- CLAUDE.md invariant "
                             "'agents never commit to main' (--allow-main to override)")

    if declared:
        plan.foreign = [p for p in plan.index if not matches_declared(p, declared)]
        if plan.foreign and not accept_index:
            plan.refusals.append(
                f"{len(plan.foreign)} staged path(s) are NOT covered by your "
                f"--path declarations. `git commit` commits the WHOLE INDEX, so "
                f"these WOULD ride along under your message. Read the list, then "
                f"either widen --path or pass --accept-index (which names them in "
                f"the commit message).")
    elif not accept_index:
        plan.refusals.append(
            "no --path declared. Because this tool never uses the crash-prone "
            "`git commit -- <pathspec>` form, the commit takes the WHOLE INDEX; "
            "declare what you staged with -p, or pass --accept-index after "
            "reading the index listing above.")
    else:
        plan.foreign = list(plan.index)

    hard = [f for f in plan.blocking
            if f.kind != "secret-path" or not allow_secret_path]
    for f in hard:
        plan.refusals.append(f"SECRET GUARD [{f.kind}] {f.path}: {f.detail}")
    return plan


def render_plan(plan: Plan) -> str:
    lines = [f"[safe_commit] repo   = {plan.repo}",
             f"[safe_commit] branch = {plan.branch}  HEAD = {plan.head[:8]}",
             f"[safe_commit] index  = {len(plan.index)} path(s)"]
    for p in plan.index:
        mark = "  <-- NOT DECLARED" if p in plan.foreign and plan.declared else ""
        lines.append(f"    {p}{mark}")
    if plan.advisory:
        lines.append(f"[safe_commit] {len(plan.advisory)} advisory secret "
                     f"finding(s) (not blocking):")
        for f in plan.advisory:
            lines.append(f"    ? {f.path}: {f.detail}")
    return "\n".join(lines)


def foreign_trailer(foreign: list[str]) -> str:
    """CLAUDE.md: when a sibling agent's deliverables are in the index, SAY SO in
    the commit message rather than splitting them out (splitting them out is the
    pathspec path, which segfaults)."""
    head = ("\n\nAlso-staged-by-a-sibling-agent (swept in by the whole-index "
            "commit path, declared not hidden):\n")
    return head + "\n".join(f"  - {p}" for p in sorted(foreign))


def do_commit(repo: Path, msg_file: Path, retries: int, lock_max_age_s: float,
              force_lock: bool, log=print) -> tuple[int, str]:
    """Run the ONE sanctioned commit form and survive the intermittent crash.

    NOTE the shape of this command: ``git commit -F <file>`` and NOTHING else.
    No pathspec (the partial-commit path segfaults ~50 % of the time on this
    repo, independent of file count and pathspec shape -- RETRACTION_LOG 07-25
    C8), and no ``--amend`` (it re-opens the whole index and defeats every check
    above). Returns (exit_code, new_head)."""
    before = git_out(repo, "rev-parse", "HEAD")
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        clear_stale_lock(repo, lock_max_age_s, force_lock, log=log)
        proc = subprocess.run(["git", "commit", "-F", str(msg_file)],
                              cwd=str(repo), capture_output=True, text=True,
                              errors="replace")
        after = git_out(repo, "rev-parse", "HEAD")
        if after != before:
            # A crash can still have written the commit. HEAD movement is the
            # only trustworthy success signal -- the exit code is not.
            if proc.returncode != 0:
                log(f"[safe_commit] attempt {attempt}: git exited "
                    f"{proc.returncode} but HEAD MOVED {before[:8]}->{after[:8]} "
                    f"-- the commit landed; not retrying")
            return 0, after
        last = (proc.stdout + proc.stderr).strip()
        if proc.returncode in CRASH_CODES:
            log(f"[safe_commit] attempt {attempt}/{retries}: git CRASHED "
                f"(exit {proc.returncode}) -- the known intermittent "
                f"partial-commit-adjacent crash. Clearing debris and retrying.")
            continue
        if "index.lock" in last or "Another git process" in last:
            log(f"[safe_commit] attempt {attempt}/{retries}: phantom lock error "
                f"-- clearing and retrying")
            continue
        log(f"[safe_commit] attempt {attempt}/{retries}: git exited "
            f"{proc.returncode}\n{last}")
        return proc.returncode, before
    return 4, before


def write_message(text: str) -> Path:
    """Messages always go through a FILE. A long or multi-line message passed as
    ``-m`` on a Windows shell is a quoting minefield, and CLAUDE.md already
    mandates ``-F <msgfile>``."""
    fd, name = tempfile.mkstemp(prefix="safe_commit_msg_", suffix=".txt")
    os.close(fd)
    p = Path(name)
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="The sanctioned TanitAD commit path (CLAUDE.md 'Git hygiene').")
    ap.add_argument("--repo", default=".", help="repo/worktree root (default: cwd)")
    ap.add_argument("-p", "--path", action="append", default=[], dest="paths",
                    metavar="PATH",
                    help="a path you deliberately staged (repeatable). Accepts a "
                         "file, a directory prefix, or an fnmatch glob.")
    ap.add_argument("--paths-from", default=None, metavar="FILE",
                    help="read declared paths from FILE, one per line")
    ap.add_argument("-m", "--message", default=None,
                    help="commit message text (written to a temp file; this tool "
                         "never passes -m to git)")
    ap.add_argument("-F", "--message-file", default=None,
                    help="path to an existing commit-message file")
    ap.add_argument("--accept-index", action="store_true",
                    help="acknowledge the WHOLE index after reading the listing "
                         "(CLAUDE.md step 1). Foreign entries are named in the "
                         "commit message, not hidden.")
    ap.add_argument("--allow-main", action="store_true",
                    help="permit a commit on main (agents must not)")
    ap.add_argument("--allow-secret-path", action="store_true",
                    help="permit a credential-SHAPED filename (e.g. a test "
                         "fixture .pem). Never overrides a token-content hit.")
    ap.add_argument("--allow-empty", action="store_true",
                    help="permit an empty index (git --allow-empty is NOT passed)")
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts against the intermittent git crash (default 3)")
    ap.add_argument("--lock-max-age-s", type=float, default=30.0,
                    help="when git liveness is undeterminable, only clear an "
                         "index.lock older than this (default 30 s)")
    ap.add_argument("--force-lock", action="store_true",
                    help="clear index.lock regardless of age/liveness")
    ap.add_argument("--print-index", action="store_true",
                    help="print the index and exit 0 (the check CLAUDE.md "
                         "requires you to run FIRST)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every guard, print the exact command, commit nothing")
    args = ap.parse_args(argv)

    try:
        repo = repo_root(Path(args.repo).resolve())
    except (GitError, FileNotFoundError, OSError) as exc:
        print(f"[safe_commit] not a git repo / git unavailable: {exc}",
              file=sys.stderr)
        return 3

    declared = list(args.paths)
    if args.paths_from:
        try:
            declared += [ln.strip() for ln in
                         Path(args.paths_from).read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.lstrip().startswith("#")]
        except OSError as exc:
            print(f"[safe_commit] --paths-from unreadable: {exc}", file=sys.stderr)
            return 2

    if args.print_index:
        plan = build_plan(repo, declared, True, True, True, True)
        print(render_plan(plan))
        return 0

    if not args.message and not args.message_file:
        print("[safe_commit] need -m/--message or -F/--message-file", file=sys.stderr)
        return 2
    if args.message and args.message_file:
        print("[safe_commit] pass exactly one of -m / -F", file=sys.stderr)
        return 2

    plan = build_plan(repo, declared, args.accept_index, args.allow_main,
                      args.allow_secret_path, args.allow_empty)
    print(render_plan(plan))

    if plan.refusals:
        print(f"[safe_commit] REFUSED ({len(plan.refusals)}):")
        for r in plan.refusals:
            print(f"    - {r}")
        if plan.foreign and plan.declared:
            print("[safe_commit] undeclared staged paths, in full:")
            for p in plan.foreign:
                print(f"    {p}")
            print("[safe_commit] their diffstat:")
            stat = git(repo, "diff", "--cached", "--stat", "--", *plan.foreign,
                       check=False).stdout.rstrip()
            for ln in stat.splitlines():
                print(f"    {ln}")
        return 1

    # --- message -------------------------------------------------------------
    tmp: Path | None = None
    if args.message_file:
        msg_path = Path(args.message_file).resolve()
        if not msg_path.is_file():
            print(f"[safe_commit] message file not found: {msg_path}",
                  file=sys.stderr)
            return 2
        if args.accept_index and plan.foreign and declared:
            text = msg_path.read_text(encoding="utf-8") + foreign_trailer(plan.foreign)
            tmp = msg_path = write_message(text)
    else:
        text = args.message
        if args.accept_index and plan.foreign and declared:
            text += foreign_trailer(plan.foreign)
        tmp = msg_path = write_message(text)

    cmd = f"git commit -F {msg_path}"
    print(f"[safe_commit] command: {cmd}")
    print("[safe_commit] (no pathspec, no --amend -- both are the crash-prone "
          "partial-commit path)")

    if args.dry_run:
        print("[safe_commit] DRY RUN -- nothing committed")
        if tmp:
            print(f"[safe_commit] message file kept for inspection: {tmp}")
        return 0

    try:
        code, head = do_commit(repo, msg_path, args.retries,
                               args.lock_max_age_s, args.force_lock)
    finally:
        if tmp and not args.dry_run:
            try:
                tmp.unlink()
            except OSError:
                pass
    if code == 0:
        print(f"[safe_commit] COMMITTED {head[:8]} on {plan.branch}")
        return 0
    print(f"[safe_commit] commit FAILED after {args.retries} attempt(s)")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
