"""Build code/fix (FULL modified files, LF like the repo blobs) and code/diffs_vs_c36b6ddd
(unified diffs, LF-normalised, so they apply to the repo's LF blobs), plus a hunk index."""
import difflib
import json
import os
import sys

TIP = os.environ.get("V7F_TIP_SNAPSHOT", "C:/Users/Admin/tipsnap/c36b6ddd/")
MINE = os.environ.get("V7F_WORK_COPY", "C:/Users/Admin/v7f_r1r4/")
DELIV = sys.argv[1]
FILES = ["stack/tanitad/models/v6.py", "stack/scripts/train_v6_staged.py",
         "stack/tests/test_v6_stage_init_introduction.py",
         "stack/tanitad/models/plan_speed_cap.py",
         "stack/tanitad/train/declared_vs_built_v6.py", "stack/tests/test_v7f_r1r4.py"]


def lf(p):
    if not os.path.exists(p):
        return None
    return open(p, "rb").read().decode("utf-8").replace("\r\n", "\n")


index = []
for rel in FILES:
    old, new = lf(TIP + rel), lf(MINE + rel)
    fix = os.path.join(DELIV, "code", "fix", rel)
    os.makedirs(os.path.dirname(fix), exist_ok=True)
    open(fix, "w", encoding="utf-8", newline="\n").write(new)
    a = [] if old is None else old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    d = list(difflib.unified_diff(a, b, fromfile=("/dev/null" if old is None else f"a/{rel}"),
                                  tofile=f"b/{rel}", n=3))
    dp = os.path.join(DELIV, "code", "diffs_vs_c36b6ddd", rel.replace("/", "_") + ".diff")
    os.makedirs(os.path.dirname(dp), exist_ok=True)
    head = (f"# {rel} -- vs origin tip c36b6ddd (LF blobs) -- R1/R4 2026-09-27; "
            f"apply from the repo root with: git apply <this file>\n")
    open(dp, "w", encoding="utf-8", newline="\n").write(head + "".join(d))
    hunks = [h.strip() for h in d if h.startswith("@@")]
    added = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    index.append({"file": rel, "status": "NEW" if old is None else "MODIFIED",
                  "n_hunks": len(hunks), "lines_added": added, "lines_removed": removed,
                  "hunks": hunks})
json.dump(index, open(os.path.join(DELIV, "raw", "hunk_index.json"), "w"), indent=1)
for r in index:
    print(r["status"], r["file"], "hunks", r["n_hunks"], "+", r["lines_added"], "-", r["lines_removed"])
    for h in r["hunks"]:
        print("    ", h)
