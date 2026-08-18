#!/usr/bin/env python3
"""Commit ONLY the paths you name, without disturbing other agents' staged work.

⛔ THE PROBLEM THIS SOLVES, AND WHY "BE MORE CAREFUL" CANNOT (2026-08-18).

Two rules this programme follows, both correctly, COLLIDE:

  * `AGENT_OPERATING_STANDARD.md` tells agents to **stage as they go** and bank
    incrementally, so nothing is stranded in one context.
  * `CLAUDE.md` tells committers `git commit` takes the **ENTIRE INDEX**.

Obey both and an agent's incremental `git add` is *guaranteed* to be swept into
whatever commit lands next. It is not carelessness — it is the documented
procedure working as specified. MEASURED: one agent's 44-file deliverable landed
across THREE commits, 41 of them under subjects about unrelated work. The
whole-index sweep has now fired at least five times.

⛔ AND THE OBVIOUS FIX IS UNAVAILABLE: `git commit -- <pathspec>` (the partial-
commit path) **SEGFAULTS on this repo** — exit 139 under MSYS git and 0xC0000005
under native Windows git, so it is not a shell issue. Three separate root-cause
theories for that crash were each falsified, so `CLAUDE.md` deliberately states
no mechanism. It is simply not usable.

⇒ THIS SCRIPT TAKES A THIRD ROUTE: a **temporary index**. `GIT_INDEX_FILE` points
git at a scratch index seeded from HEAD; only the named paths are added to it;
then `write-tree` / `commit-tree` / `update-ref` create the commit through
PLUMBING, which never enters the porcelain partial-commit code that crashes.

The real `.git/index` is never opened for writing, so **every other agent's
staged work stays staged, exactly as they left it.**

⚠️ WHAT THIS DOES NOT DO. It does not rewrite history, and it is not a substitute
for reading `git diff --cached --name-only` before committing. When foreign work
genuinely belongs in your commit, name it in the message — a recorded sweep is
recoverable, a silent one is not.

Usage:
    python stack/scripts/scoped_commit.py -F msg.txt -- <path> [<path> ...]
    python stack/scripts/scoped_commit.py --dry-run -F msg.txt -- <path> ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TMP_INDEX = ".git/scoped-commit-index"


def _git(*args, env=None, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       env=env, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.stderr.write(f"git {' '.join(args)} -> {r.returncode}\n{r.stderr}\n")
        raise SystemExit(r.returncode)
    return r.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-F", "--file", required=True,
                    help="path to the commit message file")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the tree and report what WOULD be committed, "
                         "without moving HEAD")
    ap.add_argument("paths", nargs="+", help="repo-relative paths to commit")
    a = ap.parse_args(argv)

    msg = Path(a.file).read_text(encoding="utf-8")
    if not msg.strip():
        sys.stderr.write("refusing an empty commit message\n")
        return 2

    # ⚠️ Record the shared index BEFORE and AFTER, and assert it did not move.
    # The whole point is non-interference, so it is checked rather than assumed.
    before = _git("diff", "--cached", "--name-only")

    tmp = Path(TMP_INDEX)
    if tmp.exists():
        tmp.unlink()
    env = dict(os.environ, GIT_INDEX_FILE=TMP_INDEX)
    try:
        _git("read-tree", "HEAD", env=env)
        _git("add", "--", *a.paths, env=env)
        tree = _git("write-tree", env=env)

        head = _git("rev-parse", "HEAD")
        changed = _git("diff", "--name-only", head, tree)
        changed_list = [c for c in changed.splitlines() if c.strip()]

        if not changed_list:
            sys.stderr.write(
                "nothing to commit: the named paths match HEAD already\n")
            return 1

        print(f"tree {tree}")
        print(f"paths this commit will change ({len(changed_list)}):")
        for c in changed_list:
            print(f"  {c}")

        # ⛔ THE ASSERTION THAT MAKES THIS WORTH USING: nothing outside the
        # named paths may appear. Without it this is just a slower `git commit`.
        named = set()
        for p in a.paths:
            pp = Path(p)
            if pp.is_dir():
                named.update(
                    str(q).replace("\\", "/") for q in pp.rglob("*") if q.is_file())
            else:
                named.add(str(pp).replace("\\", "/"))
        stray = [c for c in changed_list if c not in named]
        if stray:
            sys.stderr.write(
                "REFUSING: the tree changes paths that were not named — "
                f"{stray}\n")
            return 3

        if a.dry_run:
            print("dry-run: HEAD not moved")
            return 0

        commit = _git("commit-tree", tree, "-p", head, "-F", a.file)
        _git("update-ref", "HEAD", commit)
        print(f"committed {commit[:9]} on {_git('rev-parse', '--abbrev-ref', 'HEAD')}")
    finally:
        if tmp.exists():
            tmp.unlink()

    after = _git("diff", "--cached", "--name-only")
    # Paths we just committed legitimately drop out of the shared index
    # (index blob now equals HEAD), so compare only the ones we did NOT name.
    b = {x for x in before.splitlines() if x.strip()}
    aft = {x for x in after.splitlines() if x.strip()}
    lost = (b - aft) - set(changed_list)
    if lost:
        sys.stderr.write(
            f"⚠️ shared index lost entries it should have kept: {sorted(lost)}\n")
        return 4
    print(f"shared index intact: {len(b)} staged before, {len(aft)} after "
          f"({len(changed_list)} committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
