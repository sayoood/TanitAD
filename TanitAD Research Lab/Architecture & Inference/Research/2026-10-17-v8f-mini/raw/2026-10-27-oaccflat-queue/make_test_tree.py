#!/usr/bin/env python3
"""Build a test tree on a SHORT path (C:/Users/Admin/oq*, never the long scratchpad path: MAX_PATH made deep raw/ files fail spuriously) for the 15 queue-reading repo test files: junctions to the Master Mind's patched export
(C:/Users/Admin/ofx, never edited: stack/*, taniteval, Project Steering, raw/*), REAL copies of stack/tests (+ the overlay test files) and of the v8f-mini code/ dir (+ the given queue).

    python make_test_tree.py TREE_DIR QUEUE_FILE [TESTS_OVERLAY_DIR]
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("C:/Users/Admin/ofx")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def junction(link: Path, target: Path):
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def main():
    t, queue = Path(sys.argv[1]), Path(sys.argv[2])
    overlay = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if t.exists():
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(t)], check=True)
    (t / "stack").mkdir(parents=True)
    for c in (SRC / "stack").iterdir():
        if c.name == "tests":
            continue
        (junction(t / "stack" / c.name, c) if c.is_dir() else shutil.copy2(c, t / "stack" / c.name))
    shutil.copytree(SRC / "stack" / "tests", t / "stack" / "tests")
    if overlay:
        for f in overlay.glob("test_v8f_*.py"):
            shutil.copy2(f, t / "stack" / "tests" / f.name)
    junction(t / "taniteval", SRC / "taniteval")
    junction(t / "Project Steering", SRC / "Project Steering")
    raw = t / PK / "raw"
    raw.mkdir(parents=True)
    for c in (SRC / PK / "raw").iterdir():
        (junction(raw / c.name, c) if c.is_dir() else shutil.copy2(c, raw / c.name))
    shutil.copytree(SRC / PK / "code", t / PK / "code")
    shutil.copy2(queue, t / PK / "code" / "v8f_mini_box2_queue.cmd")
    print("tree", t, "queue", queue.name, "overlay", sorted(p.name for p in overlay.glob("test_v8f_*.py")) if overlay else None)


if __name__ == "__main__":
    main()
