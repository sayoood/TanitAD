#!/usr/bin/env python3
"""Stage the L2-FRESH-STEPA package for landing: copies every deliverable (LF text) into C:/Users/Admin/par18/l2fresh/out/repo/<repo path> and writes out/paths.json = [[repo path, abs source, "raw"], ...]
plus code/MANIFEST.sha256 (LF sha256 of every staged code file).  Nothing is committed or pushed.  Large arrays stay on D: (listed in READY.md / HANDOFF.md)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil

W = "C:/Users/Admin/par18/l2fresh"
OUT = f"{W}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-l2-fresh-stepA"
ART = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA"

CODE = ["fit_final_heads.py", "fresh_carve.py", "probe_fresh_universe.py", "probe_logless251.py", "score_fresh.py", "analyse_fresh.py", "logoof_companion.py", "seal_prereg.py", "prove_guard.py", "readtwo_prep.py",
        "readtwo_geometry.py", "bridge_g2b.py", "bridge_g2b.cmd", "ram_gate.py", "nsplit.py", "fresh_chain.cmd", "test_fresh.py", "test_readtwo.py", "make_stage.py",
        "design_calc_ball.py", "scope_unseen_stratum.py", "seal_amendment2.py", "amendment2_check.py", "test_amendment2.py"]
DOCS = {n: f"{W}/{n}" for n in ("PREREG_L2_FRESH.md", "PREREG_L2_FRESH.sha256", "SEALED_AT.txt", "PREREG_L2_FRESH.src.md", "PREREG_L2_FRESH_DRAFT.md", "PREREG_L2_FRESH_D_DRAFT.md", "MM_RULING_L2_FRESH.md", "MM_RULING_L2_FRESH_D.md", "PREREG_L2_FRESH_AMENDMENT2.md", "PREREG_L2_FRESH_AMENDMENT2.sha256", "PREREG_L2_FRESH_AMENDMENT2.src.md", "SEALED_AT_AMENDMENT2.txt", "READY.md", "HANDOFF.md")}
RAW = {**{f"raw/{os.path.basename(p)}": p for p in sorted(glob.glob(f"{W}/raw/*.json"))},
       "raw/heads/final_heads.json": f"{ART}/heads/final_heads.json", "raw/heads/final_head_s0.npz": f"{ART}/heads/final_head_s0.npz", "raw/heads/final_head_s1.npz": f"{ART}/heads/final_head_s1.npz",
       **{f"raw/logs/{os.path.basename(p)}": p for p in sorted(glob.glob(f"{W}/logs/*.log"))}, "raw/pytest_test_fresh.log": f"{W}/logs/pytest_test_fresh.out", "raw/pytest_test_readtwo.log": f"{W}/logs/pytest_test_readtwo.out", "raw/pytest_test_amendment2.log": f"{W}/logs/pytest_test_amendment2.out",
       "raw/readtwo_fetch_list.txt": f"{W}/raw/readtwo_fetch_list.txt", "raw/logoof/logoof_heads.npz": f"{W}/raw/logoof/logoof_heads.npz", "raw/logoof/logoof_heads.json": f"{W}/raw/logoof/logoof_heads.json"}


def sha(p, lf=False):
    b = open(p, "rb").read()
    return hashlib.sha256(b.replace(b"\r\n", b"\n") if lf else b).hexdigest()


def main():
    stage = f"{OUT}/repo/{REPO}"
    if os.path.exists(f"{OUT}/repo"):
        shutil.rmtree(f"{OUT}/repo")
    items = [(f"code/{fn}", f"{W}/code/{fn}") for fn in CODE] + list(DOCS.items()) + list(RAW.items())
    paths, missing, man = [], [], []
    for rel, src in items:
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = f"{stage}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if src.endswith((".py", ".md", ".json", ".log", ".txt", ".out")):        # .cmd stays CRLF: a batch file with LF endings can mis-parse labels / goto
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read().replace(b"\r\n", b"\n"))
        else:
            shutil.copyfile(src, dst)
        if rel.startswith("code/"):
            man.append(f"{sha(dst)}  {rel}")
        paths.append([f"{REPO}/{rel}", dst, "raw"])
    mp = f"{stage}/code/MANIFEST.sha256"
    open(mp, "w", newline="\n").write("\n".join(man) + "\n")
    paths.append([f"{REPO}/code/MANIFEST.sha256", mp, "raw"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(paths, open(f"{OUT}/paths.json", "w"), indent=1)
    print("staged", len(paths), "files; MISSING:", missing, "; MANIFEST sha256", hashlib.sha256(open(mp, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
