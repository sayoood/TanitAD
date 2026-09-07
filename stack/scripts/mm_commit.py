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
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The repo this commits into. Overridable ONLY so the tool can be exercised
# against a throwaway repo by its own self-test -- a committer whose failure
# modes cannot be reproduced is a committer whose fixes cannot be proven.
# Unset (the normal case) it is exactly the constant it has always been.
REPO = Path(os.environ.get("MM_COMMIT_REPO") or r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
# The git invocation. `MM_COMMIT_GIT` exists ONLY so the self-test can put a
# shim at the real subprocess boundary and reproduce the historical failure
# (add exiting 0xC0000006) instead of mocking it. MEASURED: a PATH shim is
# NOT reachable here -- Windows CreateProcess appends only ".exe", so a
# git.bat on PATH is silently ignored and the real git answers. Unset (the
# normal case) this is exactly ["git", ...] as before.
# Retry sleeps are REAL by default. The self-test scales them down because the
# duration is not the behaviour under test -- "the tool dies on a page-faulting
# add" is -- and a 183 s test is a test that gets skipped. Unset = production.
_SLEEP = float(os.environ.get("MM_COMMIT_SLEEP_SCALE") or "1")

GIT = shlex.split(os.environ.get("MM_COMMIT_GIT") or "git") + [f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

import shutil


def _rm(d):
    shutil.rmtree(d, ignore_errors=True)


msgfile, paths = sys.argv[1], sys.argv[2:]
if not paths:
    raise SystemExit("usage: mm_commit.py <msgfile> <path> [<path> ...]")


def git(*a, env=None, check=True, soft=False, tries=8):
    for i in range(tries):
        # cwd=REPO, because `git hash-object -- <relative path>` resolves the
        # path against the CURRENT WORKING DIRECTORY, not against --work-tree.
        # The tool has always been invoked from the repo root, so this never
        # showed -- but nothing enforced it, and the cacheinfo fallback below
        # made it load-bearing. Caught by test_mm_commit_page_fault_fallback.
        r = subprocess.run([*GIT, *a], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env,
                           cwd=str(REPO))
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
            time.sleep(15 * _SLEEP)
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
            time.sleep(10 * _SLEEP)
            continue
        if not check:
            return None
        raise SystemExit(f"git {a[0]} failed [{r.returncode}]: {r.stderr.strip()[:200]}")
    if soft:
        return None
    raise SystemExit(f"git {a[0]} kept failing")


def _stage(p, env):
    """Put ONE path into the scratch index.

    `git add` reads the WORKING FILE through a memory map. MEASURED 2026-09-07:
    that read died with 0xC0000006 (STATUS_IN_PAGE_ERROR) on all 8 retries, for
    minutes, on 13 paths -- while `git hash-object` on THE SAME PATHS in the same
    seconds succeeded every time. So the mount was not down; one code path was.
    The tool used to die there, which blocks committing entirely.

    Fallback: hash the blob into the object store, then write the index entry
    from the OBJECT via `update-index --cacheinfo`. --cacheinfo is also strictly
    safer than `add` for a scoped commit -- it writes exactly the entry named and
    cannot pick up a sibling file or a directory's contents.
    """
    # ⛔ A SHORT ladder, not the usual 8x15s. The long ladder is for operations
    #    with NO alternative; `add` now has one. MEASURED: waiting the full
    #    120 s per path before falling back costs 26 MINUTES on a 13-path
    #    commit, all of it sleeping, for a condition whose remedy is known.
    #    Caught by test_mm_commit_page_fault_fallback, which timed out
    #    against a tool that was technically working.
    if not _stage.add_is_dead:
        if git("add", "--", p, env=env, soft=True, tries=2) is not None:
            return "add"
        # Once `add` has page-faulted, it is the MOUNT that is in that state,
        # not this one path. Re-probing it per path costs ~15 s each -- over 3
        # minutes on a 13-path commit -- to re-learn the same fact. Latch it.
        _stage.add_is_dead = True
        print("    `git add` is page-faulting; using cacheinfo for the rest",
              flush=True)
    return _stage_via_cacheinfo(p, env)


def _stage_via_cacheinfo(p, env):
    sha = git("hash-object", "-w", "--", p)
    if len(sha) != 40:
        raise SystemExit(f"REFUSING: hash-object gave {len(sha)} chars for {p}")
    mode = "100755" if os.access(REPO / p, os.X_OK) else "100644"
    git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{p}", env=env)
    print(f"    staged via cacheinfo (add was page-faulting): {p}", flush=True)
    return "cacheinfo"


_stage.add_is_dead = False

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
        _stage(p, env)
    delta = git("diff", "--cached", "--name-only", head, env=env).splitlines()
    extra = [d for d in delta if d not in paths]
    if extra:
        raise SystemExit(f"REFUSING - would touch unnamed paths: {extra}")
    if not delta:
        # An EMPTY delta is NOT evidence that there is nothing to commit.
        # CLAUDE.md: `git diff --name-only` on this mount returned empty for a
        # commit that really changed 146 files, and three probes agreed on that
        # wrong answer because they share one failure mode. This branch used to
        # print "nothing to commit" and exit 0 -- a FALSE SUCCESS that tells an
        # agent its work landed when HEAD never moved.
        # Settle it by POSITIVE per-path assertion instead, and require both
        # operands to be 40 chars: two empty strings compare EQUAL.
        really = []
        for p in paths:
            inhead = git("rev-parse", f"{head}:{p}", check=False) or ""
            mine = git("hash-object", "--", p) or ""
            if len(inhead) != 40 or len(mine) != 40:
                really.append(p)          # INCONCLUSIVE -> treat as work, retry
            elif inhead != mine:
                really.append(p)
        if really:
            print(f"  empty delta but {len(really)} path(s) differ from HEAD "
                  f"or read INCONCLUSIVE -- the mount lied; retrying", flush=True)
            _rm(scratch_dir)
            time.sleep(10 * _SLEEP)
            continue
        print("  nothing to commit (verified per path against HEAD)")
        _rm(scratch_dir)
        raise SystemExit(0)
    print("  will change:", delta, flush=True)
    tree = git("write-tree", env=env)
    # ASSERT THE SHAPE BEFORE COMPARING. Both operands come through the SAME
    # channel, so in a mount outage both return "" and `!=` is False -- the
    # check then PASSES on a commit it never verified. MEASURED 2026-09-04:
    # exactly this reported a commit as landed while HEAD held 887e9aab and the
    # worktree held 507ba1ad. Report INCONCLUSIVE, never MATCH.
    bad, inconclusive = [], []
    for p in delta:
        a = git("rev-parse", f"{tree}:{p}", check=False) or ""
        b = git("hash-object", "--", p) or ""
        if len(a) != 40 or len(b) != 40:
            inconclusive.append(p)
        elif a != b:
            bad.append(p)
    if inconclusive:
        raise SystemExit(f"INCONCLUSIVE - could not read blobs for {inconclusive}; "
                         "re-run rather than trusting this")
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
