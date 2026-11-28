"""mkpatch_n1dh.py -- n1dm.patch against 1f217bf0 from the build tree. The base side is read from the mirror's blobs (git show 1f217bf0:<path>, raw bytes), never from a working copy.
Writes: pd/a, pd/b (the changed files on each side), n1dm.patch (LF, a/ b/ prefixes), TREE_FILES_SHA256.txt (per changed file, sha256 of the new content, '-' for none), changed_files.txt,
and verifies that `git apply` of the patch onto the base blobs reproduces the tree files byte for byte (apply result == tree)."""
import hashlib
import os
import shutil
import subprocess
import sys

BASE = "1f217bf0"
MIRROR = "C:/Users/Admin/tanitad-push"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TREE = HERE + "/tree"
CHANGED = [l.strip() for l in open(HERE + "/changed_files.txt", encoding="utf-8").read().split("\n") if l.strip()]


def base_blob(rel):
    r = subprocess.run(["git", "-C", MIRROR, "show", f"{BASE}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


shutil.rmtree(HERE + "/pd", ignore_errors=True)
rows = []
for rel in CHANGED:
    new = open(f"{TREE}/{rel}", "rb").read()
    assert b"\r\n" not in new, ("CRLF in", rel)
    old = base_blob(rel)
    if old is not None:
        assert b"\r\n" not in old, ("CRLF in base", rel)
        os.makedirs(os.path.dirname(f"{HERE}/pd/a/{rel}"), exist_ok=True)
        open(f"{HERE}/pd/a/{rel}", "wb").write(old)
    os.makedirs(os.path.dirname(f"{HERE}/pd/b/{rel}"), exist_ok=True)
    open(f"{HERE}/pd/b/{rel}", "wb").write(new)
    rows.append((rel, hashlib.sha256(new).hexdigest(), "modified" if old is not None else "new"))
r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "diff", "--no-index", "--no-color", "--src-prefix=a/", "--dst-prefix=b/", "pd/a", "pd/b"], cwd=HERE, capture_output=True)
out = r.stdout.decode("utf-8")
out = out.replace("a/pd/a/", "a/").replace("b/pd/b/", "b/").replace("a/pd/b/", "b/")
open(HERE + "/n1dm.patch", "wb").write(out.encode("utf-8"))
open(HERE + "/TREE_FILES_SHA256.txt", "w", encoding="utf-8").write("".join(f"{h}  {rel}  ({k})\n" for rel, h, k in rows))
# verify: base blobs + the patch == the tree files
chk = HERE + "/pd/chk"
for rel in CHANGED:
    old = base_blob(rel)
    if old is not None:
        os.makedirs(os.path.dirname(f"{chk}/{rel}"), exist_ok=True)
        open(f"{chk}/{rel}", "wb").write(old)
os.makedirs(chk, exist_ok=True)
subprocess.run(["git", "init", "-q", chk], check=True)
ap = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", HERE + "/n1dm.patch"], cwd=chk, capture_output=True)
assert ap.returncode == 0, ap.stderr.decode()
bad = [rel for rel in CHANGED if open(f"{chk}/{rel}", "rb").read() != open(f"{TREE}/{rel}", "rb").read()]
print("files", len(CHANGED), "patch bytes", len(out), "apply==tree", not bad, bad)
shutil.rmtree(chk + "/.git", ignore_errors=True)
sys.exit(0 if not bad else 1)
