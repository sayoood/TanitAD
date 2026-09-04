#!/usr/bin/env python3
"""mktree_commit.py -- commit named paths WITHOUT ever seeding an index.

    python stack/scripts/mktree_commit.py <msgfile> <path> [<path> ...]

WHY IT EXISTS. `stack/scripts/scoped_commit.py` and `mm_commit.py` both start by
seeding a scratch index from HEAD (`git read-tree HEAD`). On this repo that is
minutes of work over ~9,000 files on the G: mount, and it fails for two
independent reasons:

  * the mount flaps on a ~2-5 minute period -- MEASURED 2026-09-04: git read
    HEAD at 10:36 and reported "not a git repository" at 10:41, while PowerShell
    listed `.git/objects` fine and `.git/HEAD` itself returned "Incorrect
    function". A read-tree cannot reliably survive a window shorter than itself;
  * a path in HEAD that Windows cannot represent makes `read-tree` fail
    OUTRIGHT and forever -- MEASURED the same day:
    `error: invalid path '.claude?/hooks/no-idle-guard.sh'`.

This never builds an index. It rewrites only the directories on the paths' own
spines:

    blob   = git hash-object -w <file>          (one call per file)
    tree_D = git mktree  <-- git ls-tree HEAD:D with that entry replaced
                                                (one call per directory level)
    commit = git commit-tree <root> -p HEAD -F msg
    ref    = git update-ref HEAD <new> <old>    (COMPARE-AND-SWAP)

Every call reads ONE directory listing, so the window that must stay open is
seconds. It commits ONLY the named paths and never opens the shared index, so
the "commit swept a sibling's staged work" failure is structurally impossible.

=============================================================================
THE FAILURE THIS FILE'S GUARDS EXIST TO PREVENT -- IT ALREADY HAPPENED ONCE
=============================================================================
MEASURED 2026-09-04: an earlier version of this script passed `git mktree` its
records through `subprocess.run(..., input=<str>, text=True)`. On Windows a
TEXT-mode pipe translates every LF written into CRLF. `mktree`'s record format
is `<mode> <type> <hash><TAB><name><LF>`, so the stray CR landed INSIDE THE
NAME. The commit's root tree held `.claude<CR>`, `CLAUDE.md<CR>`,
`taniteval<CR>` ... for EVERY entry: a commit whose every path is wrong. Nothing
was lost -- the subtree hashes were byte-identical to the parent's -- but from
that moment `read-tree` failed for EVERY agent on the repo, so NOBODY could
commit. Recovery was a compare-and-swap `update-ref` back to the parent.

Two lessons are now enforced in code:

  A. ALL git I/O here is BINARY. stdin is encoded to bytes in `git()` and stdout
     is decoded there, so no newline translation can happen in between.
  B. The tree is checked BEFORE the ref moves. The old version verified only
     AFTER `update-ref`, which is how a corrupt tree reached a shared branch at
     all. `assert_names_preserved` compares every rebuilt tree's entry NAMES and
     child hashes against the originals and REFUSES on any difference other than
     the paths the caller named.

FOUR GUARDS IN TOTAL:
  1. binary I/O (above);
  2. pre-commit name/hash preservation check on every rewritten tree;
  3. HEAD re-read immediately before `commit-tree`, and the final write is a
     COMPARE-AND-SWAP so a lost race FAILS instead of clobbering -- a plain
     `update-ref HEAD <new>` is how a sibling's commit disappears, and that has
     happened three times in this repo;
  4. POSITIVE post-commit verification of every named path, every sibling blob
     in each rewritten directory, and every ROOT entry. On this mount an empty
     or failing git result is indistinguishable from absence, so it is retried,
     never read as an answer.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("MKTREE_REPO",
                           r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"))
GIT = ["git", f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

#: substrings that mean "the mount blinked", not "the answer is no".
TRANSIENT = ("not a git repository", "Invalid request code", "Invalid argument",
             "Errno 22", "Permission denied", "cannot open", "unable to",
             "Unable to", "could not read", "Could not read", "short read",
             "No such file or directory", "index.lock", "busy",
             "cannot use", "as an exclude file")


def git(*args: str, stdin: str | None = None, check: bool = True,
        tries: int = 10) -> str:
    """One git command, with BINARY I/O and retries across mount flaps."""
    payload = stdin.encode("utf-8") if stdin is not None else None
    for i in range(tries):
        r = subprocess.run([*GIT, *args], input=payload, capture_output=True,
                           cwd=str(REPO))
        out = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
        if r.returncode == 0:
            return out.strip()
        # 0xC0000006 STATUS_IN_PAGE_ERROR and its signed form arrive as an exit
        # code with EMPTY stderr, so string matching cannot see them.
        transient = (r.returncode in (3221225478, -1073741818,
                                      3221225477, -1073741819)
                     or any(s in err for s in TRANSIENT))
        if transient and i < tries - 1:
            print(f"    transient {args[0]}: "
                  f"{(err.strip() or f'exit {r.returncode}')[:70]} (retry {i})",
                  flush=True)
            time.sleep(8)
            continue
        if not check:
            return ""
        raise SystemExit(f"git {args[0]} failed [{r.returncode}]: {err[:300]}")
    raise SystemExit(f"git {args[0]} kept failing (mount never stayed up)")


def read_tree_entries(spec: str, missing_ok: bool = False) -> dict[str, str]:
    """`git ls-tree <spec>` -> {name: '<mode> <type> <hash>'} for ONE level.

    ``missing_ok`` is for a directory that does not exist in HEAD yet — a NEW
    directory is an empty listing, not an error. ⛔ It is NOT a general "ignore
    failures" switch: only git's own "Not a valid object name" / "not a tree"
    is treated as empty, and every transient mount error still retries inside
    ``git()``. Confusing the two would turn a mount blink into a silently
    EMPTIED directory — i.e. a commit that deletes a subtree.
    """
    try:
        listing = git("ls-tree", spec)
    except SystemExit as ex:
        msg = str(ex)
        if missing_ok and ("Not a valid object name" in msg
                           or "not a tree object" in msg):
            return {}
        raise
    entries: dict[str, str] = {}
    for line in listing.splitlines():
        if not line.strip():
            continue
        meta, name = line.split("\t", 1)
        entries[name] = meta
    return entries


def assert_names_preserved(label: str, old: dict[str, str],
                           new_tree_hash: str, intended: set[str]) -> None:
    """REFUSE unless the rebuilt tree differs from the original ONLY in the
    entries the caller intended to change.

    This is the guard that was missing when a CRLF-mangled `mktree` produced a
    root tree in which every NAME had gained a trailing CR. The names are the
    thing to check: the hashes can legitimately change, the names cannot.
    """
    new = read_tree_entries(new_tree_hash)
    lost = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old) - intended)
    if lost or added:
        raise SystemExit(
            f"⛔ REFUSING to commit: rebuilding {label!r} changed its ENTRY "
            f"NAMES. lost={lost[:8]} unexpected_new={added[:8]}. This is the "
            f"CRLF-into-mktree corruption class — a tree whose every path is "
            f"wrong. Nothing has been committed.")
    moved = sorted(n for n in old
                   if n not in intended and old[n] != new.get(n))
    # a child directory we rewrote legitimately changes hash; those are passed
    # in `intended` by the caller.
    if moved:
        raise SystemExit(
            f"⛔ REFUSING to commit: {label!r} entries changed that were not "
            f"named: {moved[:8]}. Nothing has been committed.")


def build_tree(head: str, updates: dict[str, str]) -> str:
    """Rewrite only the directories on the updated paths' spines; return the new
    ROOT tree. ``updates`` maps repo-relative POSIX paths to blob hashes."""
    by_dir: dict[str, dict[str, str]] = {}
    dirs: set[str] = set()
    for p in updates:
        d = str(Path(p).parent.as_posix())
        d = "" if d == "." else d
        by_dir.setdefault(d, {})[Path(p).name] = updates[p]
        while d:
            dirs.add(d)
            d = str(Path(d).parent.as_posix())
            d = "" if d == "." else d
    new_tree: dict[str, str] = {}
    for d in sorted(dirs, key=lambda x: -x.count("/")) + [""]:
        spec = f"{head}:{d}" if d else f"{head}^{{tree}}"
        # a directory that does not exist in HEAD yet is an empty listing
        entries = read_tree_entries(spec, missing_ok=bool(d))
        old = dict(entries)
        intended: set[str] = set()
        for name, blob in by_dir.get(d, {}).items():
            mode = entries.get(name, "100644 blob x").split()[0]
            entries[name] = f"{mode} blob {blob}"
            intended.add(name)
        for child, h in new_tree.items():
            parent = str(Path(child).parent.as_posix())
            parent = "" if parent == "." else parent
            if parent == d:
                entries[Path(child).name] = f"040000 tree {h}"
                intended.add(Path(child).name)
        body = "".join(f"{meta}\t{name}\n"
                       for name, meta in sorted(entries.items()))
        new_tree[d] = git("mktree", stdin=body)
        assert_names_preserved(d or "<root>", old, new_tree[d], intended)
    return new_tree[""]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit("usage: mktree_commit.py <msgfile> <path> [<path> ...]")
    msgfile, paths = argv[1], [p.replace("\\", "/") for p in argv[2:]]
    for attempt in range(6):
        head = git("rev-parse", "HEAD")
        print(f"[attempt {attempt}] HEAD = {head[:9]}", flush=True)
        blobs = {p: git("hash-object", "-w", "--", p) for p in paths}
        # non-clobber snapshot: siblings in each rewritten directory, and every
        # ROOT entry (the root is the tree whose entries are whole subtrees).
        sibs: dict[str, str] = {}
        for p in paths:
            d = str(Path(p).parent.as_posix())
            for name, meta in read_tree_entries(f"{head}:{d}",
                                                missing_ok=True).items():
                rel = f"{d}/{name}"
                if rel not in paths and meta.split()[1] == "blob":
                    sibs[rel] = meta.split()[2]
        roots = {n: m.split()[2]
                 for n, m in read_tree_entries(f"{head}^{{tree}}").items()}
        touched = {p.split("/")[0] for p in paths}

        root = build_tree(head, blobs)
        if root == git("rev-parse", f"{head}^{{tree}}"):
            print("  nothing to commit (tree identical to HEAD)")
            return 0
        now = git("rev-parse", "HEAD")
        if now != head:
            print(f"  HEAD moved {head[:9]} -> {now[:9]} while building; restart")
            continue
        new = git("commit-tree", root, "-p", head, "-F", msgfile)
        r = subprocess.run([*GIT, "update-ref", "HEAD", new, head],
                           capture_output=True, cwd=str(REPO))
        if r.returncode != 0:
            print("  CAS lost ("
                  + r.stderr.decode('utf-8', 'replace').strip()[:120]
                  + "); restart")
            time.sleep(5)
            continue
        bad = [p for p in paths
               if git("rev-parse", f"HEAD:{p}") != git("hash-object", "--", p)]
        moved = [s for s, h in sibs.items() if git("rev-parse", f"HEAD:{s}") != h]
        now_roots = {n: m.split()[2]
                     for n, m in read_tree_entries("HEAD^{tree}").items()}
        lost = [r_ for r_, h in roots.items()
                if r_ not in touched and now_roots.get(r_) != h]
        lost += [f"MISSING:{r_}" for r_ in roots if r_ not in now_roots]
        print(f"committed {new[:9]}  paths_verified={len(paths) - len(bad)}/"
              f"{len(paths)}  siblings_unchanged={len(sibs) - len(moved)}/{len(sibs)}"
              f"  root_entries_unchanged={len(roots) - len(lost)}/{len(roots)}")
        if lost:
            print(f"  ⛔ ROOT TREE ENTRIES CHANGED OR LOST: {lost}")
            return 6
        if bad:
            print(f"  ⛔ NOT in HEAD after commit: {bad}")
            return 4
        if moved:
            print(f"  ⛔ SIBLING BLOBS MOVED (would have clobbered): {moved}")
            return 5
        return 0
    raise SystemExit("HEAD kept moving / CAS kept losing after 6 attempts")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
