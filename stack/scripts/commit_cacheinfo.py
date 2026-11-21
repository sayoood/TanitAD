"""Commit named paths WITHOUT `git add`.

`git add` reads the WORKING FILE, and on this mount that read is dying with
0xC0000006 (STATUS_IN_PAGE_ERROR) while `git hash-object` on the same paths
succeeds.  The blobs are already in the object store (the paths are staged in
the shared index), so the index entry can be written from the OBJECT alone via
`update-index --cacheinfo` -- no working-tree read at all.

Verification is POSITIVE per path, never an empty diff: CLAUDE.md records that
`git diff --name-only` returns EMPTY on this mount for a commit that really
changed 146 files, and mm_commit.py:121 turns exactly that into
"nothing to commit" + exit 0.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
GIT = ["git", f"--git-dir={REPO / '.git'}", f"--work-tree={REPO}"]

msgfile = sys.argv[1]
paths = [a for a in sys.argv[2:] if not a.startswith("--")]
if not paths:
    raise SystemExit("usage: commit_cacheinfo.py <msgfile> <path> [<path> ...]")


def git(*a, env=None, check=True):
    r = subprocess.run([*GIT, *a], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode and check:
        raise SystemExit(f"git {a[0]} failed [{r.returncode}]: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


# ---- 1. the blob + mode for each path, FROM THE SHARED INDEX (no file read) ----
entries = {}
for p in paths:
    out = git("ls-files", "--stage", "--", p)
    if not out:
        raise SystemExit(f"REFUSING: {p} is not in the shared index")
    mode, sha, _stage = out.split()[0], out.split()[1], out.split()[2]
    if len(sha) != 40:
        raise SystemExit(f"REFUSING: {p} sha is {len(sha)} chars, not 40")
    # prove the object really exists before we build a tree that points at it
    git("cat-file", "-e", sha)
    # ⛔ THE PRECONDITION, ENFORCED RATHER THAN DOCUMENTED. A staged blob can be
    #    STALE -- CLAUDE.md: `--cached` answers "is this path in the index?",
    #    which is YES for a tracked file whose index blob is the PRE-EDIT
    #    version. Committing that would silently ship old content. So compare
    #    the index blob to the worktree; and when the worktree cannot be READ
    #    (the very outage that sends you to this tool) say INCONCLUSIVE and
    #    require the operator to say so out loud. ⛔ Never treat an unreadable
    #    file as agreement -- two failures compare EQUAL.
    wt = git("hash-object", "--", p, check=False) or ""
    if len(wt) != 40:
        if "--assume-staged-is-current" not in sys.argv:
            raise SystemExit(
                f"INCONCLUSIVE: cannot read {p} to confirm the staged blob is "
                f"current. Re-run when the mount settles, or pass "
                f"--assume-staged-is-current if you have verified staging by "
                f"another route.")
        print(f"  ⚠️  {p}: worktree unreadable; trusting the staged blob "
              f"ON THE OPERATOR'S ASSERTION", flush=True)
    elif wt != sha:
        raise SystemExit(
            f"REFUSING: {p} is staged as {sha[:9]} but the worktree holds "
            f"{wt[:9]} -- the index is STALE. `git add` it first.")
    entries[p] = (mode, sha)
print(f"resolved {len(entries)} blobs from the shared index", flush=True)

scratch = Path(tempfile.mkdtemp(prefix="mmci-"))       # LOCAL DISK, private
env = dict(os.environ, GIT_INDEX_FILE=str(scratch / "index"))

head = git("rev-parse", "HEAD")
print(f"HEAD before = {head[:9]}", flush=True)
git("read-tree", head, env=env)
for p, (mode, sha) in entries.items():
    git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{p}", env=env)
tree = git("write-tree", env=env)
print(f"tree = {tree[:9]}", flush=True)

# ---- 2. POSITIVE per-path assertions against CURRENT HEAD ----
now = git("rev-parse", "HEAD")
if now != head:
    raise SystemExit(f"REFUSING: HEAD moved {head[:9]} -> {now[:9]} during the build; re-run")

named = set(entries)
changed = []
for p, (_mode, sha) in entries.items():
    inmine = git("rev-parse", f"{tree}:{p}", check=False)
    if inmine != sha:
        raise SystemExit(f"REFUSING: {p} is {inmine[:9]} in my tree, expected {sha[:9]}")
    inhead = git("rev-parse", f"{head}:{p}", check=False)
    if len(inhead) == 40 and inhead == sha:
        print(f"  unchanged vs HEAD: {p}")
    else:
        changed.append(p)
if not changed:
    raise SystemExit("REFUSING: every named path already equals HEAD -- nothing to do")
print(f"  will change {len(changed)} of {len(named)} named paths", flush=True)

# Every UNNAMED path must equal HEAD's. Two arguments, in order of strength:
#
#  (1) BY CONSTRUCTION: the tree is `read-tree HEAD` plus exactly len(named)
#      `update-index --cacheinfo` writes. Unlike `git add`, --cacheinfo cannot
#      pick up a sibling file or a directory -- it writes the one entry named.
#      This is the guarantee; the listing below only corroborates it.
#  (2) A SMALL LISTING, because the -r form is not trustworthy here: CLAUDE.md
#      records `git ls-tree -r` truncating SILENTLY and CONSISTENTLY on this
#      mount (7,535 five times, while cat-file proved six "missing" files
#      present). A size floor cannot catch that -- both the true and truncated
#      counts are large -- and a refusal built on it livelocks, which is
#      exactly scoped_commit.py's failure. So compare the TOP LEVEL only: ~20
#      entries, small enough to trust, and any change under a directory
#      bubbles up into its subtree hash.
roots = {p.split("/", 1)[0] for p in named}
mine = {l.split("	", 1)[1]: l.split()[2]
        for l in git("ls-tree", tree, env=env).splitlines() if "	" in l}
theirs = {l.split("	", 1)[1]: l.split()[2]
          for l in git("ls-tree", head).splitlines() if "	" in l}
if not mine or not theirs:
    raise SystemExit("REFUSING: top-level ls-tree came back empty -- INCONCLUSIVE, re-run")
if set(mine) != set(theirs):
    raise SystemExit(f"REFUSING: top-level entries differ: "
                     f"{set(mine) ^ set(theirs)}")
stray = [k for k, v in mine.items() if theirs[k] != v and k not in roots]
if stray:
    raise SystemExit(f"REFUSING: would touch unnamed top-level trees: {stray}")
moved = [k for k in roots if mine[k] != theirs[k]]
print(f"  tree guard OK: {len(mine)} top-level entries, "
      f"changed exactly {moved}, 0 stray", flush=True)

# ---- 3. commit-tree + compare-and-swap ----
msg = Path(msgfile).read_text(encoding="utf-8")
new = git("commit-tree", tree, "-p", head, "-m", msg)
git("update-ref", "HEAD", new, head)          # CAS: fails if HEAD moved
print(f"COMMITTED {new[:9]}", flush=True)
