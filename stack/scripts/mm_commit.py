"""Reusable scoped committer for the Master Mind, hardened for THIS repo tonight.

Why not stack/scripts/scoped_commit.py: it seeds a scratch index from HEAD, then takes
~10-13 min of read-tree over 8,952 files on the G: mount, during which HEAD moves under
it (five agents are committing). Its stray-path guard then correctly REFUSES, because the
tree it built would revert a sibling's file. Correct, but livelocked.

This does the same thing with two changes:
  * a PRIVATE index path, so it never contends on .git/scoped-commit-index.lock;
  * it RE-SEEDS and retries when HEAD moves, instead of refusing, and passes the old
    value to update-ref so the final write is a compare-and-swap.

Usage: python mm_commit.py <msgfile> <path> [<path> ...]
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
GIT = ["git", f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

import shutil


def _rm(d):
    shutil.rmtree(d, ignore_errors=True)


msgfile, paths = sys.argv[1], sys.argv[2:]
if not paths:
    raise SystemExit("usage: mm_commit.py <msgfile> <path> [<path> ...]")


def git(*a, env=None, check=True):
    for i in range(8):
        r = subprocess.run([*GIT, *a], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode == 0:
            return r.stdout.strip()
        # A git child that died mid-command (the 0xC0000006 page-out below, or a
        # mount drop) leaves OUR scratch index's `.lock` behind; the child has
        # exited, so it is debris, and every retry would then fail with
        # "Unable to create '<scratch>.lock': File exists". MEASURED 2026-09-05:
        # one page-out on read-tree poisoned all 7 retries. Only THIS run's own
        # scratch lock is touched -- never another agent's.
        _idx = (env or {}).get("GIT_INDEX_FILE")
        if _idx and os.path.exists(_idx + ".lock"):
            try:
                os.unlink(_idx + ".lock")
            except OSError:
                pass
        # 0xC0000006 STATUS_IN_PAGE_ERROR (and its signed form): the memory-mapped
        # read of a git object could not be paged in because the G: mount dropped
        # mid-read. It arrives as an EXIT CODE WITH EMPTY STDERR, so string matching
        # cannot see it. MEASURED 2026-09-04 on read-tree; CLAUDE.md records the same
        # code from a bulk `git add`. Always transient - retry.
        if r.returncode in (3221225478, -1073741818, 3221225477, -1073741819):
            print(f"    transient {a[0]}: exit 0x{r.returncode & 0xFFFFFFFF:08X} "
                  f"(mount paged out) (retry {i})", flush=True)
            time.sleep(15)
            continue
        # "Could not read <sha>" on this mount is a TRANSIENT READ FAILURE, not
        # corruption: MEASURED 2026-09-04 -- read-tree died on a tree object that
        # `git cat-file -t` read successfully seconds later, with HEAD's whole tree
        # and every load-bearing path intact. On the G: mount an empty or failed git
        # result is indistinguishable from absence, so it must be retried, never
        # believed.
        if any(s in r.stderr for s in ("Invalid request code", "Errno 22", "busy",
                                       "Invalid argument", "unable to", "File exists",
                                       "index.lock", "Could not read", "could not read",
                                       "unable to read", "short read",
                                       # the G: mount drops out entirely for seconds at a
                                       # time; git then reports the repo itself as absent.
                                       # MEASURED 2026-09-04. Absence on this mount is
                                       # indistinguishable from a failed query -- retry.
                                       "not a git repository", "Permission denied",
                                       "cannot open", "No such file or directory",
                                       # the mount can drop a .git metadata file
                                       # mid-command; git then blames the file
                                       # rather than the mount. MEASURED 2026-09-04
                                       # on `add`: "cannot use .git/info/exclude as
                                       # an exclude file". Transient.
                                       "as an exclude file", "cannot use",
                                       "Invalid request code")):
            print(f"    transient {a[0]}: {r.stderr.strip()[:60]} (retry {i})", flush=True)
            time.sleep(10)
            continue
        if not check:
            return None
        raise SystemExit(f"git {a[0]} failed [{r.returncode}]: {r.stderr.strip()[:200]}")
    raise SystemExit(f"git {a[0]} kept failing")


for attempt in range(8):
    # MEASURED by the eval stream 2026-09-03: a scratch index ON THE G: MOUNT makes
    # `read-tree HEAD` write a 1.5 MB index across the mount -- 13 MINUTES under ~97
    # concurrent git processes, long enough for another agent to land a commit inside
    # the window, after which the stray-path guard refuses forever. On LOCAL DISK the
    # same read-tree is 3m39s and the race closes. This is the root cause of tonight's
    # commit thrash, including one commit that silently reverted a sibling's five files.
    # ⚠️ A SIBLING AGENT DELETED `mm-index-*` FILES FROM %TEMP% WHILE CLEARING LOCKS
    # and said so; `read-tree` then failed 8/8 with "Unable to create <scratch>".
    # A flat, guessable name in a SHARED temp dir is not a private path. Use a
    # per-run DIRECTORY with a random component so no sweep can match a pattern,
    # and so a stale file can never block creation.
    scratch_dir = Path(tempfile.mkdtemp(prefix="mmidx-"))
    scratch = scratch_dir / f"index-{attempt}"
    env = dict(os.environ, GIT_INDEX_FILE=str(scratch))
    head = git("rev-parse", "HEAD")
    print(f"[attempt {attempt}] HEAD = {head[:9]}", flush=True)
    git("read-tree", head, env=env)
    for p in paths:
        git("add", "--", p, env=env)
    delta = git("diff", "--cached", "--name-only", head, env=env).splitlines()
    extra = [d for d in delta if d not in paths]
    if extra:
        raise SystemExit(f"REFUSING - would touch unnamed paths: {extra}")
    if not delta:
        print("  nothing to commit (already identical to HEAD)")
        _rm(scratch_dir)
        raise SystemExit(0)
    print("  will change:", delta, flush=True)
    tree = git("write-tree", env=env)
    bad = [p for p in delta
           if git("rev-parse", f"{tree}:{p}") != git("hash-object", p)]
    if bad:
        raise SystemExit(f"tree/worktree mismatch on {bad}")

    now = git("rev-parse", "HEAD")
    if now != head:
        print(f"  HEAD moved {head[:9]} -> {now[:9]} during staging; re-seeding",
              flush=True)
        _rm(scratch_dir)
        continue
    c = git("commit-tree", tree, "-p", head, "-F", msgfile)
    ok = git("update-ref", "HEAD", c, head, check=False)   # compare-and-swap
    _rm(scratch_dir)
    if ok is None:
        print("  update-ref lost the race; retrying", flush=True)
        continue
    print("committed", c[:9], flush=True)
    print(git("log", "--oneline", "-1"), flush=True)
    break
else:
    raise SystemExit("could not land the commit in 4 attempts - HEAD too hot")
