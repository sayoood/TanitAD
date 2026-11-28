#!/usr/bin/env python3
"""Stage the package for landing: copies into out/<repo path> and writes out/paths.json ([repo path, absolute source, "raw"] per file) and out/RESULT.md is written by hand.  Re-run after any new file.  Never commits or pushes.
    python make_stage.py"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-p2-confirm"
OUT = os.path.join(PKG, "out")
TOP = ("PREREG_SEARCH_P2_CONFIRM_DRAFT.md", "HANDOFF.md", "PREREG_SEARCH_P2_CONFIRM.md", "PREREG_SEARCH_P2_CONFIRM.sha256", "SEALED_AT.txt")
DIRS = {"code": (".py",), "raw": (".json", ".md"), "src": (".md",), "logs": (".log", ".json")}
SKIP_DIRS = {"__pycache__", "scratch", "out"}


def main():
    shutil.rmtree(os.path.join(OUT, "TanitAD Research Lab"), ignore_errors=True)
    paths, manifest = [], []
    def add(rel, src):
        dst = os.path.join(OUT, REPO, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        paths.append([f"{REPO}/{rel}", src.replace("\\", "/"), "raw"])
        if rel.startswith("code/") and rel.endswith(".py"):
            h = hashlib.sha256(open(src, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
            manifest.append(h + "  " + rel)
    for f in TOP:
        if os.path.isfile(os.path.join(PKG, f)):
            add(f, os.path.join(PKG, f))
    for d, exts in DIRS.items():
        for root, dirs, files in os.walk(os.path.join(PKG, d)):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for fn in sorted(files):
                if fn.endswith(exts):
                    add(os.path.relpath(os.path.join(root, fn), PKG).replace("\\", "/"), os.path.join(root, fn))
    mp = os.path.join(PKG, "code", "MANIFEST.sha256")
    open(mp, "w", encoding="utf-8", newline="\n").write("\n".join(sorted(manifest)) + "\n")
    add("code/MANIFEST.sha256", mp)
    json.dump(paths, open(os.path.join(OUT, "paths.json"), "w"), indent=1)
    print("ZZSTAGE_OK", len(paths), "files ->", os.path.join(OUT, REPO))


if __name__ == "__main__":
    main()
