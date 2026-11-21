#!/usr/bin/env python3
"""Decode thor_head_probe.out -> thor_head.json  {path: {head_blob, worktree_blob}}."""
import base64
import gzip
import hashlib
import json
import re
import sys

raw = open("thor_head_probe.out", encoding="utf-8", errors="replace").read()


def marker(name):
    m = re.search(r"WW" + name + r"(.*?)WW", raw)
    if not m:
        raise SystemExit("marker %s absent -- truncated pull" % name)
    return m.group(1)


head = marker("HEAD")
autocrlf = marker("AUTOCRLF")
n_tree, n_exist, n_join = int(marker("TREE")), int(marker("EXIST")), int(marker("JOIN"))
n_lines, b64len, b64md5 = int(marker("LINES")), int(marker("B64LEN")), marker("B64MD5")

body, grab = [], False
for line in raw.splitlines():
    s = line.strip()
    if s == "WWPAYLOADSTARTWW":
        grab = True
        continue
    if s == "WWPAYLOADENDWW":
        grab = False
        continue
    if grab:
        body.append(s)
b64 = "".join(body)
assert len(b64) == b64len, "TRUNCATED payload %d != %d" % (len(b64), b64len)
assert hashlib.md5(b64.encode()).hexdigest() == b64md5, "CORRUPT payload"
txt = gzip.decompress(base64.b64decode(b64)).decode("utf-8", "replace")
lines = txt.split("\n")
if lines and lines[-1] == "":
    lines.pop()
assert len(lines) == n_lines, "row count %d != %d" % (len(lines), n_lines)
i = lines.index("===SPLIT===")
tree_rows, wt_rows = lines[:i], lines[i + 1:]
assert len(tree_rows) == n_tree, "tree %d != %d" % (len(tree_rows), n_tree)
assert len(wt_rows) == n_join, "join %d != %d" % (len(wt_rows), n_join)

tree = {}
for ln in tree_rows:
    sha, path = ln.split("\t", 1)
    tree[path] = sha.strip()
wt = {}
for ln in wt_rows:
    sha, path = ln.split("\t", 1)
    wt[path] = sha.strip()

same = sum(1 for p in tree if tree[p] == wt.get(p))
diff = sorted(p for p in tree if p in wt and tree[p] != wt[p])
out = {"thor_head": head, "core_autocrlf": autocrlf, "n_tracked": len(tree),
       "n_worktree_present": len(wt), "n_worktree_equals_thor_head": same,
       "n_worktree_differs_from_thor_head": len(diff),
       "differs_from_thor_head": diff,
       "tree": tree, "worktree": wt}
json.dump(out, open("thor_head.json", "w", encoding="utf-8"), indent=1)
print("thor HEAD      :", head)
print("core.autocrlf  :", autocrlf)
print("tracked files  :", len(tree))
print("worktree found :", len(wt))
print("== Thor HEAD   :", same)
print("!= Thor HEAD   :", len(diff))
for p in diff[:80]:
    print("   ", p)
if len(diff) > 80:
    print("    ... and %d more" % (len(diff) - 80))
