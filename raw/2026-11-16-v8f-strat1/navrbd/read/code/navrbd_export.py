"""navrbd_export.py -- exports the minimal NAVRBD read tree of one commit and verifies it by git blob hash.
Only the directories the evaluators need: stack/scripts, stack/tanitad, taniteval/taniteval + taniteval/tools (t1_eval.py is loaded by the T1 analysis step; the T1 / T0 wrappers, the model, the data loaders, the pinned evaluators, ci.py).
`git -c core.autocrlf=false archive <commit> <paths>` from the mirror repo, extracted into DEST; then every file listed by `git ls-tree -r <commit> <paths>` must exist in DEST with the SAME git blob sha1
(sha1 of "blob <n>\\0" + bytes: a CRLF conversion changes it), and every repo path must be <= 200 characters.  Exit 1 with the problems printed otherwise.
Usage: python navrbd_export.py COMMIT DEST [--repo C:/Users/Admin/tanitad-push]"""
import hashlib
import os
import subprocess
import sys

PATHS = ("stack/scripts", "stack/tanitad", "taniteval/taniteval", "taniteval/tools")
MAX_REPO_PATH = 200


def blob_sha1(p):
    b = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def parse_ls_tree(text):
    """`git ls-tree -r` lines '<mode> blob <sha>\\t<path>' -> {path: sha}.  Pure."""
    out = {}
    for ln in text.splitlines():
        if "\t" not in ln:
            continue
        meta, path = ln.split("\t", 1)
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            out[path] = parts[2]
    return out


def problems(want, dest):
    """-> [problem]: a listed file missing, a blob sha1 that differs, a repo path over the limit.  Pure given files."""
    bad = []
    for path, sha in want.items():
        if len(path) > MAX_REPO_PATH:
            bad.append(f"repo path {len(path)} > {MAX_REPO_PATH}: {path[:60]}...")
        q = os.path.join(dest, path)
        if not os.path.isfile(q):
            bad.append(f"missing {path}")
        elif blob_sha1(q) != sha:
            bad.append(f"blob differs {path}")
    if not want:
        bad.append("the commit lists no files under the exported paths")
    return bad


def export(commit, dest, repo="C:/Users/Admin/tanitad-push"):
    git = ["git", "-C", repo, "-c", "core.autocrlf=false"]
    ls = subprocess.run(git + ["ls-tree", "-r", commit, "--"] + list(PATHS), capture_output=True, encoding="utf-8", check=True).stdout
    want = parse_ls_tree(ls)
    os.makedirs(dest, exist_ok=True)
    tarp = dest.rstrip("/\\") + ".tar.part"
    with open(tarp, "wb") as f:
        subprocess.run(git + ["archive", "--format=tar", commit, "--"] + list(PATHS), stdout=f, check=True)
    subprocess.run(["tar", "xf", tarp, "-C", dest], check=True)
    os.remove(tarp)
    return problems(want, dest), len(want)


if __name__ == "__main__":
    a = sys.argv[1:]
    repo = a[a.index("--repo") + 1] if "--repo" in a else "C:/Users/Admin/tanitad-push"
    bad, n = export(a[0], a[1], repo)
    for b in bad:
        print("PROBLEM", b)
    print("exported", n, "files;", "OK" if not bad else f"{len(bad)} problems")
    sys.exit(1 if bad else 0)
