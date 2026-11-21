"""Everything in D: (index, worktree, untracked) that is NOT on the tip, by blob id.
The tip tree is read from the push mirror by PLUMBING ONLY (its index is never used)."""
import collections
import json
import subprocess

D = "D:/Projects/TanitAD"
M = "C:/Users/Admin/tanitad-push"
TIP = open("C:/Users/Admin/AppData/Local/Temp/claude/tip.txt").read().strip()


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True).stdout.decode("utf-8", "replace")


tip = {}
for line in g(M, "ls-tree", "-r", "-z", TIP).split("\0"):
    if "\t" in line:
        meta, path = line.split("\t", 1)
        tip[path] = meta.split()[2]
# positive control against silent truncation: named load-bearing paths must be listed
for must in ("CLAUDE.md", "Project Steering/GOALS_AND_CLAIMS.md",
             "TanitAD Research Lab/Library/library.json", "tools/kb_add.py"):
    assert must in tip, "tip listing is missing %s -- truncated listing" % must

idx = {}
for line in g(D, "ls-files", "--stage", "-z").split("\0"):
    if "\t" in line:
        meta, path = line.split("\t", 1)
        idx[path] = meta.split()[1]
untracked = [p for p in g(D, "ls-files", "--others", "--exclude-standard", "-z").split("\0") if p]
unstaged = [p for p in g(D, "diff", "--name-only", "-z").split("\0") if p]

rows = []
for p, b in idx.items():
    if tip.get(p) != b:
        rows.append({"path": p, "where": "index", "on_tip": p in tip})
for p in untracked:
    rows.append({"path": p, "where": "untracked", "on_tip": p in tip})
for p in unstaged:
    rows.append({"path": p, "where": "worktree-modified", "on_tip": p in tip})
deleted_in_index = [p for p in tip if p not in idx]

print("tip %s: %d paths | D: index %d | differing index entries %d | untracked %d | unstaged %d"
      % (TIP[:7], len(tip), len(idx), sum(r["where"] == "index" for r in rows),
         len(untracked), len(unstaged)))
print("tip paths ABSENT from D:'s index (the index predates them): %d" % len(deleted_in_index))
grp = collections.Counter()
for r in rows:
    parts = r["path"].split("/")
    key = "/".join(parts[:3]) if parts[0] == "TanitAD Research Lab" else "/".join(parts[:2])
    grp[(r["where"], "new" if not r["on_tip"] else "differs", key)] += 1
for (w, s, k), n in sorted(grp.items(), key=lambda x: (x[0][0], x[0][2])):
    print("  %-17s %-7s %4d  %s" % (w, s, n, k))
json.dump({"tip": TIP, "rows": rows, "tip_paths_absent_from_index": deleted_in_index},
          open("C:/Users/Admin/AppData/Local/Temp/claude/landscape.json", "w"), indent=1)
