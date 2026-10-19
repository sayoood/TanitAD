#!/usr/bin/env python3
"""Stage the L2 Step B build for landing: copies every deliverable into C:/Users/Admin/par18/l2stepb/out/repo/<repo path> as LF text and writes out/paths.json = [[repo path, abs source, "raw"], ...].
Nothing is committed or pushed; the repo working tree is not touched."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil

W = "C:/Users/Admin/par18/l2stepb"
OUT = f"{W}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-l2-stepB"
CODE = ["l2_scorer.py", "stepB.py", "g_reach_stepb.py", "label_stepB.py", "analyse_stepB.py", "cost_stepB.py", "make_test_head.py", "fake_scorer.py", "test_stepB.py", "stepB_chain.py", "test_stepB_chain.py", "make_stage.py"]
DOCS = {"PREREG_L2_STEPB.md": f"{W}/PREREG_L2_STEPB.md", "PREREG_L2_STEPB.sha256": f"{W}/PREREG_L2_STEPB.sha256", "SEALED_AT.txt": f"{W}/SEALED_AT.txt", "MM_RULING_STEPB.md": f"{W}/MM_RULING_STEPB.md",
        "PREREG_L2_STEPB_DRAFT.md": f"{W}/PREREG_L2_STEPB_DRAFT.md", "READY.md": f"{W}/READY.md", "HANDOFF.md": f"{W}/HANDOFF.md", "seal_stepB.py": f"{W}/seal_stepB.py"}
RAW = {"raw/stepB_config.json": f"{W}/raw/stepB_config.json", "raw/stepB_config_O4.json": f"{W}/raw/stepB_config_O4.json", "raw/cost.json": f"{OUT}/cost.json", "raw/smoke_label_timing.json": f"{OUT}/smoke_label_timing.json",
       "raw/testheads/test_head_s0.npz": f"{W}/raw/testheads/test_head_s0.npz", "raw/pytest_test_stepB.log": f"{W}/logs/pytest_test_stepB.log", "raw/pytest_test_stepB_sealed.log": f"{W}/logs/pytest_test_stepB_sealed.log", "raw/gate_real_testhead.log": f"{W}/logs/gate_real_testhead.log",
       "raw/gate_real_testhead_rev1.log": f"{W}/logs/gate_real_testhead_rev1.log",
       **{f"raw/gate_testhead/{os.path.basename(p)}": p for p in glob.glob(f"{OUT}/gate_testhead/G-REACH-STEPB_*.json")},
       **{f"raw/gate_testhead_rev1/{os.path.basename(p)}": p for p in glob.glob(f"{W}/raw/gate_testhead_rev1/G-REACH-STEPB_*.json")}}


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
    run_items = {}
    if os.path.exists(f"{W}/run/RESULT_STEPB.json"):                       # after ZZSTEPB_DONE: the run's small artifacts (stamps, merged search files, labels, the result, the logs); nothing large lives in run/
        for p in glob.glob(f"{W}/run/**/*", recursive=True):
            rel = os.path.relpath(p, f"{W}/run").replace(os.sep, "/")
            if os.path.isfile(p) and os.path.getsize(p) < 20_000_000 and not rel.startswith("gate_s") or (os.path.isfile(p) and rel.startswith("gate_s") and os.path.basename(p).startswith("G-REACH-STEPB_PASS")):
                run_items[f"raw/run/{rel}"] = p
    items = [(f"code/{fn}", f"{W}/code/{fn}") for fn in CODE] + list(DOCS.items()) + list(RAW.items()) + list(run_items.items())
    man, missing = [], []
    for rel, src in items:
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = f"{stage}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if src.endswith((".py", ".md", ".json", ".log", ".txt", ".sha256")):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read().replace(b"\r\n", b"\n"))
        else:
            shutil.copyfile(src, dst)
        if rel.startswith("code/"):
            man.append(f"{sha(dst)}  {rel}")
    open(f"{stage}/code/MANIFEST.sha256", "w", encoding="utf-8", newline="\n").write("\n".join(man) + "\n")
    paths = []
    for p in sorted(glob.glob(f"{stage}/**/*", recursive=True)):
        if os.path.isfile(p):
            rel = os.path.relpath(p, stage).replace("\\", "/")
            paths.append([f"{REPO}/{rel}", p.replace("\\", "/"), "raw"])
    json.dump(paths, open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("staged", len(paths), "files; missing:", missing)


if __name__ == "__main__":
    main()
