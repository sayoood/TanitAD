#!/usr/bin/env python3
"""W3's end-of-turn blob check — over MY paths AND the files I depend on but do not own.

    python code/w3_tree_state.py            # prints a summary, writes raw/w3_tree_state.json

⭐ WHY THE DEPENDENCIES ARE IN HERE (orchestrator rule, 2026-09-20): *an ownership boundary is not
a lock.* A MISMATCH on a file I did not touch is evidence about the TREE, not about my edit — two
siblings edited W1-owned files after its code freeze today and only an end-of-turn check saw it.
So this reports, per path, the three blobs (HEAD / index / worktree) and CLASSIFIES rather than
assuming:

    clean            index == worktree == HEAD
    mine_staged      index == worktree != HEAD      (a staged edit — mine if the path is mine)
    worktree_ahead   worktree != index              (an edit not yet staged — REPORT, never assume)
    stale_index      worktree == HEAD != index      (the revert-waiting-to-happen shape)
    INCONCLUSIVE     any blob that is not 40 hex    (a failed probe is NOT agreement)

⛔ Every comparison asserts BOTH sides are 40 characters first: two empty strings compare equal,
and that is how a mount outage reads as a match.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PKG, *[".."] * 4))
PKG_REL = os.path.relpath(PKG, REPO).replace(os.sep, "/")

#: files W3 CALLS or IMPORTS from other streams — not mine to edit, mine to notice
DEPENDENCIES = [
    "taniteval/taniteval/bench/contract.py",
    "taniteval/taniteval/bench/cli.py",
    "taniteval/taniteval/bench/gpu_gap.py",
    "taniteval/taniteval/bench/schema_check.py",
    "taniteval/taniteval/bench/schema/bench_run.schema.json",
    "taniteval/taniteval/bench/schema/summary.schema.json",
    "taniteval/taniteval/bench/navsim/summarize.py",
    "taniteval/taniteval/bench/navsim/profiles.py",
    "taniteval/adapters/navsim.py",
    "taniteval/adapters/navsim_ci.py",
    "tools/criteria_check.py",
    "products/P7-TanitEval/CRITERIA_REGISTRY.json",
    ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/"
     "code/navsim_win.py"),
    ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/"
     "code/tanitad_navsim_bridge.py"),
    ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/"
     "code/build_frames.py"),
    ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/"
     "raw/cluster_maps/navtest.json"),
]
MINE_EXTRA = ["taniteval/taniteval/bench/plugins/navsim_v1.py"]


def git(*args) -> tuple:
    r = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", REPO, *args],
                       capture_output=True, text=True)
    return r.stdout.strip(), r.returncode, r.stderr.strip()


def out(*args) -> str:
    return git(*args)[0]


def head_blob(path: str) -> tuple:
    """(blob, absent_from_head). ⛔ `rev-parse HEAD:<path>` exits 128 BOTH for a path HEAD does
    not contain (a FACT — the file was added today and is not committed yet) and for a broken
    probe. `cat-file -e` separates them: only a clean 'does not exist' is an absence."""
    o, rc, err = git("rev-parse", f"HEAD:{path}")
    if len(o) == 40:
        return o, False
    _, rc2, err2 = git("cat-file", "-e", f"HEAD:{path}")
    absent = rc2 != 0 and ("does not exist" in (err + err2).lower()
                           or "path" in (err + err2).lower() and "exist" in (err + err2).lower())
    return "", absent


def blobs(path: str) -> dict:
    head, head_absent = head_blob(path)
    line = out("ls-files", "--stage", "--", path)
    idx = line.split()[1] if line else ""
    work = out("hash-object", path) if os.path.exists(os.path.join(REPO, path)) else ""
    have_iw = len(idx) == 40 and len(work) == 40
    if head_absent and have_iw:
        state = "new_staged" if idx == work else "new_worktree_ahead"
    elif head_absent and len(work) == 40 and not idx:
        state = "untracked"
    elif not (have_iw and len(head) == 40):
        state = "INCONCLUSIVE"
    elif idx == work == head:
        state = "clean"
    elif idx == work != head:
        state = "staged"
    elif work == head != idx:
        state = "stale_index"
    elif work != idx:
        state = "worktree_ahead"
    else:
        state = "other"
    return {"path": path, "head": head or None, "head_absent_from_HEAD": head_absent,
            "index": idx or None, "worktree": work or None, "state": state}


def main() -> int:
    mine = [l.split()[3] for l in out("ls-files", "--stage", "--", PKG_REL).splitlines() if l]
    rows = {"mine": [blobs(p) for p in mine + MINE_EXTRA],
            "dependencies": [blobs(p) for p in DEPENDENCIES]}
    summary = {}
    for group, rs in rows.items():
        c = {}
        for r in rs:
            c[r["state"]] = c.get(r["state"], 0) + 1
        summary[group] = c
    # a dependency that is merely NOT COMMITTED YET (new_staged) is not "moved": the whole
    # EvalFlyWheel build landed today. What matters is an edit in flight on top of it.
    moved = [r for r in rows["dependencies"]
             if r["state"] not in ("clean", "new_staged")]
    rec = {"package": PKG_REL, "utc": out("log", "-1", "--format=%cI") or None,
           "head": out("rev-parse", "HEAD"), "summary": summary,
           "dependencies_in_flight": moved, "rows": rows,
           "_reading": ("`worktree_ahead`/`stale_index` on a DEPENDENCY is a sibling's live edit — "
                        "evidence about the TREE, not about W3 (an ownership boundary is not a "
                        "lock). `new_staged` only means the file was added today and is not "
                        "committed yet. `INCONCLUSIVE` is never agreement.")}
    with open(os.path.join(PKG, "raw", "w3_tree_state.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({"package": PKG_REL, "head": rec["head"][:12], "summary": summary,
                      "dependencies_in_flight": [(r["path"], r["state"]) for r in moved]}, indent=1))
    bad = [r for r in rows["mine"] if r["state"] == "INCONCLUSIVE"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
