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

import os
import subprocess
import sys
import time
from pathlib import Path

#: ⛔ CORRECTED 2026-09-18. This was a HARDCODED `G:\...\Projects\TanitAD` literal,
#: wrong twice over: the live project moved to `D:/Projects/TanitAD` (so the tool pointed
#: at a path whose .git does not hydrate, and EVERY call burned the full 30x6s retry
#: ladder before failing — that is the 12 minutes its test file took), and
#: `test_mktree_commit.py` sets `MKTREE_REPO` to a throwaway repo, which a literal cannot
#: honour — so all six of its tests ran against the dead mount instead of their fixture.
#: Derived-from-`__file__` is the only spelling that is correct in the repo, in a
#: worktree, and under the test.
REPO = Path(os.environ.get("MKTREE_REPO") or Path(__file__).resolve().parents[2])
GIT = ["git", f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

TRANSIENT = ("not a git repository", "Invalid argument", "Invalid request code",
             "unable to", "could not read", "Could not read", "short read",
             "cannot open", "No such file or directory", "Permission denied",
             "cannot use", "as an exclude file")
TRANSIENT_RC = (3221225478, -1073741818, 3221225477, -1073741819)


def git(*a, stdin=None, binary_in=None, attempts=30, sleep_s=6.0):
    """Run one git command, retrying only the TRANSIENT mount failures.

    ``stdin`` is the caller-facing name (str or bytes, as ``mktree`` wants) and is
    the spelling `test_mktree_commit.py` uses; ``binary_in`` is the older internal
    name, kept so existing call sites do not move. Both at once is a caller bug.
    """
    if stdin is not None:
        if binary_in is not None:
            raise SystemExit("git(): pass stdin= or binary_in=, not both")
        binary_in = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
    last = ""
    for i in range(attempts):
        r = subprocess.run([*GIT, *a], capture_output=True, cwd=str(REPO),
                           input=binary_in, text=binary_in is None,
                           encoding=None if binary_in else "utf-8",
                           errors=None if binary_in else "replace")
        if r.returncode == 0:
            out = r.stdout if binary_in is None else r.stdout.decode()
            # ⚠️ Strip the TRAILING NEWLINE only. Every caller here wants an OID, and
            # an unstripped one silently becomes part of the NEXT revision expression:
            # f"{head}^{{tree}}" built from a newline-terminated sha asks git for a ref
            # with a line break inside it and fails as `ambiguous argument`, which reads
            # like a bad ref rather than a formatting bug (MEASURED 2026-09-18: five of
            # this file's six tests failed that way). rstrip of CR/LF and NOT strip(),
            # so `ls-tree -z` output, whose records end in NUL, is untouched.
            return out.rstrip("\r\n")
        err = (r.stderr if binary_in is None
               else r.stderr.decode("utf-8", "replace")) or ""
        last = f"[{r.returncode}] {err.strip()[:110]}"
        if r.returncode in TRANSIENT_RC or any(t in err for t in TRANSIENT):
            print(f"    transient {a[0]}: {last} (retry {i})", flush=True)
            time.sleep(sleep_s)
            continue
        raise SystemExit(f"git {a[0]} FAILED {last}")
    raise SystemExit(f"git {a[0]} kept failing: {last}")


class _Remove:
    """Sentinel: this path is DELETED in the new tree."""
    def __repr__(self):
        return "<REMOVE>"


REMOVE = _Remove()


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


def read_tree_entries(tree: str, missing_ok: bool = False) -> dict:
    """``name -> "<mode> <type> <sha>"`` for ONE directory, plus an EXPLICIT choice
    about a tree that is not there.

    ⛔ THE missing_ok DISTINCTION IS A SAFETY PROPERTY, not ergonomics. "this tree is
    absent" and "this tree is empty" must never collapse into one answer: confusing
    them turns a mount blink into a commit that silently empties a subtree. With
    ``missing_ok=False`` (the default) an absent tree RAISES; only an explicit
    ``missing_ok=True`` may read ``{}``.

    ⚠️ THE VALUE IS A STRING, NOT :func:`ls_tree`'s TUPLE — and I got this wrong once.
    Restored 2026-09-18 alongside :func:`build_tree`; both were this module's API on
    2026-09-04 (`e685d90`) and `aad4088` renamed them to :func:`ls_tree` / :func:`update`
    the next day, orphaning six tests. My first restoration described itself as "a thin
    layer over the current internals, behaviour unchanged" and delegated straight to
    :func:`ls_tree`. That was WRONG: the caller rebuilds an `mktree` body with
    ``f"{meta}	{name}"``, so ``meta`` must already be the ``"100644 blob <sha>"`` line
    git emits — a tuple renders as its repr and produces a corrupt tree. The two
    functions genuinely differ in their return type, and saying otherwise was a claim
    about the code I had not checked against its consumer.
    """
    if missing_ok:
        # ⛔ Probed WITHOUT `git()` on purpose: `git()` raises SystemExit on a
        # non-zero return, which is the behaviour we are deliberately opting out
        # of here — and catching that exception instead would also swallow a
        # genuine failure and report it as "absent".
        probe = subprocess.run([*GIT, "cat-file", "-e", tree],
                               capture_output=True, cwd=str(REPO))
        if probe.returncode != 0:
            return {}
    return {name: f"{mode} {typ} {sha}"
            for name, (mode, typ, sha) in ls_tree(tree).items()}


def assert_names_preserved(label: str, old: dict, new_tree: str,
                           changed: set) -> None:
    """REFUSE a rebuilt tree that lost, gained or MANGLED a name.

    ⛔ WHAT THIS IS FOR, and it is not hypothetical: on 2026-09-04 a rebuilt tree
    came back with a CARRIAGE RETURN appended to every entry name. Git accepted it —
    `mktree` does not validate names — so `a.txt
` and `a.txt` are two different
    files, the commit "succeeded", and every original path was gone while the tree
    looked the right size. A count check passes that; only comparing the NAME SETS
    catches it.

    ``changed`` names may differ in CONTENT; no name may appear or disappear. The
    check is on names only, because content is verified separately by blob hash.
    """
    got = set(read_tree_entries(new_tree))
    want = set(old)
    lost, gained = sorted(want - got), sorted(got - want)
    if lost or gained:
        raise SystemExit(
            f"REFUSING to commit: rebuilding {label} changed its NAME SET — "
            f"lost {lost or '[]'}, gained {gained or '[]'}. "
            "A name that vanishes or grows a stray byte (CR, 2026-09-04) is a "
            "corrupt tree that git will happily accept.")
    unexpected = sorted(n for n in want
                        if n not in changed and old[n] != read_tree_entries(new_tree)[n])
    if unexpected:
        raise SystemExit(
            f"REFUSING to commit: rebuilding {label} changed entries nobody named: "
            f"{unexpected}")


def build_tree(commit_or_tree: str, flat: dict) -> str:
    """New tree from ``commit_or_tree`` with the FLAT ``path -> blob`` map applied.

    The caller-facing shape: full slash-separated paths, a commit accepted as well
    as a tree. :func:`update` is the internal form and takes a TREE plus a NESTED
    spec; this splits the paths and resolves the commit, reusing ``main``'s own
    conventions so there is one spelling of each.
    ``REMOVE`` is passed through as a value, so a deletion spec works here too.
    """
    root = git("rev-parse", f"{commit_or_tree}^{{tree}}").strip()
    spec: dict = {}
    for p, sha in flat.items():
        parts = p.split("/")
        node = spec
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = sha
    return update(root, spec)


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
    removed = set()
    for name, val in spec.items():
        here = f"{path_so_far}/{name}" if path_so_far else name
        if val is REMOVE:
            #: ⛔ A REMOVE of a path that is not there is a REFUSAL, not a
            #: no-op: silently accepting it is how a typo'd rename ships as a
            #: successful commit that removed nothing.
            if name not in entries:
                raise SystemExit(f"cannot remove {here}: not in HEAD")
            del entries[name]
            removed.add(name)
            continue
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
    new_names = set(spec) - existing - removed
    if len(entries) != n_before + len(new_names) - len(removed):
        raise SystemExit(f"sibling count moved in {path_so_far or '<root>'}: "
                         f"{n_before} -> {len(entries)} (+{len(new_names)} new, "
                         f"-{len(removed)} removed)")
    return mktree(entries)


def main():
    msgfile, rest = sys.argv[1], sys.argv[2:]
    if "--rm" in rest:
        i = rest.index("--rm")
        paths, removals = rest[:i], rest[i + 1:]
    else:
        paths, removals = rest, []
    if not paths and not removals:
        raise SystemExit("usage: mktree_commit.py <msgfile> [<path> ...] "
                         "[--rm <path-to-delete> ...]")
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
    for p in removals:
        parts = p.split("/")
        node = spec
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = REMOVE
        print(f"  REMOVE  {p}", flush=True)

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
    #: ⛔ POSITIVE guard for removals too: `rev-parse <tree>:<path>` must
    #: FAIL. Trusting that the delete happened is the same error as trusting a
    #: `git add` exit code.
    for p in removals:
        chk = subprocess.run([*GIT, "rev-parse", f"{new_tree}:{p}"],
                             capture_output=True, text=True, encoding="utf-8")
        if chk.returncode == 0:
            raise SystemExit(f"removal did not take: {p} still resolves in the new tree")
    print(f"[mktree] all named paths verified in the new tree "
          f"({len(blobs)} added/changed, {len(removals)} removed)", flush=True)

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
    for p in removals:
        chk = subprocess.run([*GIT, "rev-parse", f"HEAD:{p}"],
                             capture_output=True, text=True, encoding="utf-8")
        if chk.returncode == 0:
            raise SystemExit(f"POST-COMMIT: {p} still resolves in HEAD")
    print(f"VERIFIED in HEAD by blob comparison: {len(blobs)} path(s); "
          f"{len(removals)} removal(s) verified absent")
    print(git("log", "--oneline", "-1").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
