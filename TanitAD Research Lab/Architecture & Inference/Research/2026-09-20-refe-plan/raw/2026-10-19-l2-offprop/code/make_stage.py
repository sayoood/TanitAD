#!/usr/bin/env python3
"""Stage the L2-OFFPROP build for landing: copies every deliverable into C:/Users/Admin/par18/l2/out/repo/<repo path> and writes out/paths.json = [[repo path, abs source, "raw"], ...].
Nothing is committed or pushed.  Large arrays (feats.npz and the shards) stay on D:; only their manifest is staged."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil

W = "C:/Users/Admin/par18/l2"
OUT = f"{W}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-l2-offprop"
ART = "D:/Projects/TanitAD-artifacts/l2_offprop"

CODE = ["l2lib.py", "outcome.py", "synthetic.py", "gates.py", "run_l2.py", "export_feats.py", "merge_feats.py", "design_sweep.py", "cost_synthetic.py", "test_l2.py", "test_outcome.py", "make_stage.py", "render_docs.py",
        "finalize_recipe.py", "seed_spread.py", "planted_margin.py"]
DOCS = {"DESIGN_CHOICES.md": f"{W}/DESIGN_CHOICES.md", "RECIPE_SELECTION.md": f"{W}/RECIPE_SELECTION.md", "READY.md": f"{W}/READY.md", "RESULT_GATES.md": f"{W}/RESULT_GATES.md", "DRAFT_PREREG_L2.md": f"{W}/DRAFT_PREREG_L2.md",
        "MM_RULING_L2_BUILD.md": f"{W}/MM_RULING_L2_BUILD.md"}
RAW = {**{f"raw/gates/{os.path.basename(p)}": p for p in glob.glob(f"{OUT}/gates/*.json")},
       **{f"raw/superseded/{os.path.basename(p)}": p for p in glob.glob(f"{OUT}/superseded/*")},
       "raw/feats_manifest.json": f"{ART}/feats_manifest.json", "raw/sweep_semi.json": f"{OUT}/sweep_semi.json", "raw/INSTRUMENT_DEFECT.json": f"{OUT}/INSTRUMENT_DEFECT.json",
       "raw/seed_spread.json": f"{OUT}/seed_spread.json", "raw/recipe_chosen.json": f"{OUT}/recipe_chosen.json", "raw/planted_margin.json": f"{OUT}/planted_margin.json",
       "raw/cost.json": f"{OUT}/cost.json", "raw/cost_semi.json": f"{OUT}/cost_semi.json", "raw/pytest_test_l2.log": f"{W}/logs/pytest_test_l2.log",
       "raw/test_outcome_mutations.log": f"{W}/logs/test_outcome.log", "raw/export_w1.log": f"{W}/logs/export_w1.log", "raw/export_w2.log": f"{W}/logs/export_w2.log"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    stage = f"{OUT}/repo/{REPO}"
    if os.path.exists(f"{OUT}/repo"):
        shutil.rmtree(f"{OUT}/repo")
    items = []
    man = []
    for fn in CODE:
        items.append((f"code/{fn}", f"{W}/code/{fn}"))
    for rel, src in DOCS.items():
        items.append((rel, src))
    for rel, src in RAW.items():
        items.append((rel, src))
    paths, missing = [], []
    for rel, src in items:
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = f"{stage}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if src.endswith((".py", ".md", ".json", ".log", ".txt")):        # staged as LF text (a Windows text-mode write had left a CRLF/LF mix); the gate stamps hash LF-normalised bytes
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read().replace(b"\r\n", b"\n"))
        else:
            shutil.copyfile(src, dst)
        if rel.startswith("code/"):
            man.append(f"{sha(dst)}  {rel}")
    open(f"{stage}/code/MANIFEST.sha256", "w", encoding="utf-8").write("\n".join(man) + "\n")
    for p in sorted(glob.glob(f"{stage}/**/*", recursive=True)):
        if os.path.isfile(p):
            rel = os.path.relpath(p, stage).replace("\\", "/")
            paths.append([f"{REPO}/{rel}", p.replace("\\", "/"), "raw"])
    json.dump(paths, open(f"{OUT}/paths.json", "w", encoding="utf-8"), indent=1)
    print("staged", len(paths), "files; missing:", missing)


if __name__ == "__main__":
    main()
