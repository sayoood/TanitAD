"""Build the gate's evidence trees on the dev box (never a worktree: `git archive` of the tip).

    python mk_trees.py --tip <sha> --gate-files <dir with stack/...> [--fixes <pkg code/fix dir>]
                       --dest <tree dir>

* `git archive <tip> stack taniteval tools` with core.autocrlf=false (the repo's bytes);
* then, in order, the fixes agent's `code/fix/` overlay (when given) and the gate's own files;
* every overlaid file is verified by sha256 after the copy, and the tree's `stack/`+`taniteval/`
  digest is printed with the gate's own definition.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

GIT_DIR = "C:/Users/Admin/tanitad-push/.git"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def overlay(src_root: Path, dest: Path) -> list[str]:
    done = []
    for f in sorted(src_root.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src_root)
        if rel.parts[0] not in ("stack", "taniteval", "tools"):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
        if sha(out) != sha(f):
            raise SystemExit(f"copy mismatch {rel}")
        done.append(rel.as_posix())
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tip", required=True)
    ap.add_argument("--gate-files", required=True)
    ap.add_argument("--fixes", default=None)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    if dest.exists():
        raise SystemExit(f"{dest} exists -- build into a fresh directory")
    env = dict(os.environ, GIT_DIR=GIT_DIR)
    raw = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", a.tip, "stack",
                          "taniteval", "tools"], env=env, capture_output=True, check=True).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(dest)
    rep = {"tip": a.tip, "fixes": [], "gate": []}
    if a.fixes:
        rep["fixes"] = overlay(Path(a.fixes), dest)
    rep["gate"] = overlay(Path(a.gate_files), dest)
    sys.path.insert(0, str(Path(a.gate_files) / "stack" / "scripts"))
    import launch_gate as LG
    man = LG.tree_manifest(dest)
    print({"dest": str(dest), "n_files": len(man), "tree_sha256": LG.tree_digest(man),
           "fixes_overlaid": len(rep["fixes"]), "gate_overlaid": rep["gate"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
