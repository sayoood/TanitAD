"""Enumerate every published citation of the 84x curvature figure.

⛔ grep/rg under-report on the G: mount while exiting 0, so this reads each file
with retries and prints a SAME-BREATH CONTROL that must read non-zero.
"""
import io
import json
import os
import subprocess
import time

REPO = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
NEEDLES = ["2.30973", "0.02737", "84x", "84\u00d7", "2.32114", "2.90288", "0.70839",
           "1.58320", "curvature MAE", "never steers"]
CONTROL = "the"          # must appear in every prose file; a 0 here means a flap


def read_retry(path, tries=10):
    for _ in range(tries):
        try:
            with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            time.sleep(3)
    return None


SKIP = {".git", "__pycache__", "node_modules", "Media", ".venv"}


def tracked_files():
    """os.walk, because `.git` is unreadable while the mount re-hydrates."""
    got = []
    for root, dirs, fs in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in fs:
            if f.endswith((".md", ".txt", ".json")):
                got.append(os.path.relpath(os.path.join(root, f), REPO).replace("\\", "/"))
    return got


files = [f for f in tracked_files() if not f.startswith("Media/")]
print("tracked prose/json files scanned: %d" % len(files))

hits, unreadable, control_zero = {}, [], []
for rel in files:
    txt = read_retry(os.path.join(REPO, rel))
    if txt is None:
        unreadable.append(rel)
        continue
    if CONTROL not in txt and rel.endswith(".md"):
        control_zero.append(rel)
    for nd in NEEDLES:
        if nd in txt:
            lines = [i + 1 for i, ln in enumerate(txt.splitlines()) if nd in ln]
            hits.setdefault(nd, []).append({"file": rel, "lines": lines})

print("\n=== UNREADABLE (mount) : %d ===" % len(unreadable))
for u in unreadable[:20]:
    print("   ", u)
print("=== CONTROL 'the' missing from (should be ~0 md files): %d ===" % len(control_zero))

for nd in NEEDLES:
    rows = hits.get(nd, [])
    print("\n### %-16s -> %d file(s)" % (repr(nd), len(rows)))
    for r in rows:
        print("    %-100s %s" % (r["file"], r["lines"]))

json.dump({"hits": hits, "unreadable": unreadable, "n_scanned": len(files)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "enumerate_out.json"), "w"), indent=1)
