"""md5 manifest of a tree, byte-exact, portable between Windows and Linux.

Emits {relpath: md5} for every regular file under --root, skipping __pycache__,
*.pyc and .git. Used to prove a synced tree on a pod is byte-identical to the
repo working tree -- the check that would have caught the eval pod being 62%
stale with corridor.py entirely missing.
"""
import argparse
import hashlib
import json
import os
import sys

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            yield rel, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    files = {}
    total = 0
    for rel, full in walk(a.root):
        try:
            with open(full, "rb") as fh:
                h = hashlib.md5()
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            files[rel] = h.hexdigest()
            total += os.path.getsize(full)
        except OSError as ex:
            files[rel] = f"ERROR:{type(ex).__name__}"
    out = {"label": a.label, "root": os.path.abspath(a.root),
           "n_files": len(files), "total_bytes": total,
           "python": sys.version.split()[0], "files": files}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[md5] {a.label or a.root}: {len(files)} files, {total/1e6:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
