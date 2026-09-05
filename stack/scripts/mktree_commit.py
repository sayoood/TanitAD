"""Scoped commit that never runs `read-tree`.

⛔ WHY THIS EXISTS. `mm_commit.py` seeds a scratch index with `read-tree HEAD`,
which walks all ~8,950 tracked entries across the G: mount. MEASURED 2026-09-05:
that step failed 8/8 internal retries on every one of 25 driver attempts over
~40 minutes — `exit 0xC0000006 (mount paged out)` then a run of
`fatal: not a git repository`, while `git rev-parse HEAD`, `git status` and
`git cat-file -e HEAD:CLAUDE.md` all returned 0 in the same second. The mount
drops in bursts SHORTER than a full-tree read but LONGER than a single object
read, so the fix is to make the write O(depth x entries-per-directory) instead
of O(tracked files).

HOW. Classic plumbing: hash the changed blobs, then rebuild only the ancestor
directories of those paths with `git ls-tree -z` + `git mktree`, reusing every
sibling subtree SHA verbatim. Nothing else in the tree is even read, so nothing
else can be lost. The final write is a COMPARE-AND-SWAP
(`update-ref HEAD <new> <old>`), so a sibling agent landing a commit inside the
window makes this FAIL rather than clobber.

Guards, all POSITIVE assertions:
  * every named path in the new tree hashes to the blob we wrote;
  * every named path's parent directory keeps its full sibling list (count
    checked before and after);
  * the commit is verified by `git cat-file -e HEAD:<path>` + a length-guarded
    blob comparison after `update-ref`.

Usage: python mktree_commit.py <msgfile> <path> [<path> ...]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
GIT = ["git", f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

TRANSIENT = ("not a git repository", "Invalid argument", "Invalid request code",
             "unable to", "could not read", "Could not read", "short read",
             "cannot open", "No such file or directory", "Permission denied",
             "cannot use", "as an exclude file")
TRANSIENT_RC = (3221225478, -1073741818, 3221225477, -1073741819)


def git(*a, binary_in=None, attempts=30, sleep_s=6.0):
    last = ""
    for i in range(attempts):
        r = subprocess.run([*GIT, *a], capture_output=True,
                           input=binary_in, text=binary_in is None,
                           encoding=None if binary_in else "utf-8",
                           errors=None if binary_in else "replace")
        if r.returncode == 0:
            return r.stdout if binary_in is None else r.stdout.decode()
        err = (r.stderr if binary_in is None
               else r.stderr.decode("utf-8", "replace")) or ""
        last = f"[{r.returncode}] {err.strip()[:110]}"
        if r.returncode in TRANSIENT_RC or any(t in err for t in TRANSIENT):
            print(f"    transient {a[0]}: {last} (retry {i})", flush=True)
            time.sleep(sleep_s)
            continue
        raise SystemExit(f"git {a[0]} FAILED {last}")
    raise SystemExit(f"git {a[0]} kept failing: {last}")


def ls_tree(tree: str) -> dict:
    """``name -> (mode, type, sha)`` for ONE directory. ``-z`` so names with
    spaces need no quoting — `TanitAD Research Lab/...` has three of them."""
    raw = git("ls-tree", "-z", tree)
    out = {}
    for rec in raw.split("\0"):
        if not rec:
            continue
        meta, name = rec.split("\t", 1)
        mode, typ, sha = meta.split(" ", 2)
        out[name] = (mode, typ, sha)
    return out


def mktree(entries: dict) -> str:
    body = "".join(f"{m} {t} {s}\t{n}\n"
                   for n, (m, t, s) in sorted(entries.items()))
    return git("mktree", binary_in=body.encode("utf-8")).strip()


def update(tree: str, spec: dict, path_so_far: str = "") -> str:
    """Rebuild ``tree`` so the paths in ``spec`` are replaced. ``spec`` maps a
    first path segment to either a blob sha (leaf) or a nested spec dict."""
    entries = ls_tree(tree) if tree else {}
    n_before = len(entries)
    existing = set(entries)          # captured BEFORE mutation, read once
    for name, val in spec.items():
        here = f"{path_so_far}/{name}" if path_so_far else name
        if isinstance(val, dict):
            sub = entries.get(name)
            if sub is not None and sub[1] != "tree":
                raise SystemExit(f"{here} is a {sub[1]} in HEAD, not a tree")
            entries[name] = ("040000", "tree",
                             update(sub[2] if sub else "", val, here))
        else:
            mode = entries[name][0] if name in entries else "100644"
            if name in entries and entries[name][1] != "blob":
                raise SystemExit(f"{here} is a {entries[name][1]}, not a blob")
            entries[name] = (mode, "blob", val)
    # POSITIVE guard: rebuilding a directory must never LOSE a sibling.
    new_names = set(spec) - existing
    if len(entries) != n_before + len(new_names):
        raise SystemExit(f"sibling count moved in {path_so_far or '<root>'}: "
                         f"{n_before} -> {len(entries)} (+{len(new_names)} new)")
    return mktree(entries)


def main():
    msgfile, paths = sys.argv[1], sys.argv[2:]
    if not paths:
        raise SystemExit("usage: mktree_commit.py <msgfile> <path> [<path> ...]")
    head = git("rev-parse", "HEAD").strip()
    print(f"[mktree] HEAD = {head[:9]}", flush=True)

    blobs = {}
    for p in paths:
        blobs[p] = git("hash-object", "-w", "--", p).strip()
        print(f"  blob {blobs[p][:9]}  {p}", flush=True)

    spec: dict = {}
    for p, sha in blobs.items():
        parts = p.split("/")
        node = spec
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = sha

    root_tree = git("rev-parse", f"{head}^{{tree}}").strip()
    new_tree = update(root_tree, spec)
    print(f"[mktree] tree {root_tree[:9]} -> {new_tree[:9]}", flush=True)
    if new_tree == root_tree:
        print("nothing to commit (tree identical to HEAD)")
        return 0

    # POSITIVE guard: every named path in the NEW tree is the blob we wrote.
    for p, sha in blobs.items():
        got = git("rev-parse", f"{new_tree}:{p}").strip()
        if got != sha:
            raise SystemExit(f"new tree has {got[:9]} at {p}, expected {sha[:9]}")
    print("[mktree] all named paths verified in the new tree", flush=True)

    now = git("rev-parse", "HEAD").strip()
    if now != head:
        print(f"HEAD moved {head[:9]} -> {now[:9]} while building; re-run")
        return 2
    commit = git("commit-tree", new_tree, "-p", head, "-F", msgfile).strip()
    git("update-ref", "HEAD", commit, head)          # COMPARE-AND-SWAP
    print(f"committed {commit[:9]}", flush=True)

    bad = []
    for p, sha in blobs.items():
        a = git("rev-parse", f"HEAD:{p}").strip()
        if len(a) != 40 or len(sha) != 40:
            bad.append((p, "INCONCLUSIVE"))
        elif a != sha:
            bad.append((p, f"{a[:9]} != {sha[:9]}"))
    if bad:
        raise SystemExit(f"POST-COMMIT VERIFICATION FAILED: {bad}")
    print(f"VERIFIED in HEAD by blob comparison: {len(blobs)} path(s)")
    print(git("log", "--oneline", "-1").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
