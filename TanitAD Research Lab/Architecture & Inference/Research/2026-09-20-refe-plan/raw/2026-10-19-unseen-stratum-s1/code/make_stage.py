#!/usr/bin/env python3
"""Stage the UNSEEN-STRATUM S1 package for landing: copies every deliverable (LF text) into C:/Users/Admin/par18/unseen_s1/out/repo/<repo path> and writes
out/paths.json = [[repo path, staged abs path, "raw"], ...] (the same shape as l2fresh/out/paths.json) + code/MANIFEST.sha256.  Nothing is committed or pushed."""
import glob
import hashlib
import json
import os
import shutil

W = "C:/Users/Admin/par18/unseen_s1"
OUT = f"{W}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-unseen-stratum-s1"
CODE = ["unseen_token_rule.py", "test_unseen_token_rule.py", "identity_check.py", "bias_measure.py", "sampler_check.py", "apply_U_and_fetch.py", "make_stage.py"]
EXPLORE = ["_explore7.py", "_explore8.py", "_explore9.py", "_explore12.py"]      # the phase-hypothesis search (kept: it is the evidence for the stated C1 limit)
TEXT_EXT = (".py", ".json", ".md", ".txt", ".log")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    stage = f"{OUT}/repo/{REPO}"
    if os.path.exists(f"{OUT}/repo"):
        shutil.rmtree(f"{OUT}/repo")
    items = [(f"code/{f}", f"{W}/code/{f}") for f in CODE] + [(f"code/explore/{f.lstrip('_')}", f"{W}/code/{f}") for f in EXPLORE]
    items += [("RESULT.md", f"{W}/RESULT.md")]
    skip = {"_phase_features.json"}
    items += [(f"raw/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{W}/raw/*")) if os.path.basename(p) not in skip and os.path.isfile(p)]
    items += [(f"raw/logs/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{W}/logs/*")) if os.path.isfile(p)]
    paths, man, missing = [], [], []
    for rel, src in items:
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = f"{stage}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        b = open(src, "rb").read()
        if rel.endswith(TEXT_EXT):
            b = b.replace(b"\r\n", b"\n")
        open(dst, "wb").write(b)
        paths.append([f"{REPO}/{rel}", dst, "raw"])
        if rel.startswith("code/") and rel.endswith(".py"):
            man.append(f"{sha(b)}  {rel}")
    mp = f"{stage}/code/MANIFEST.sha256"
    open(mp, "w", newline="\n").write("\n".join(man) + "\n")
    paths.append([f"{REPO}/code/MANIFEST.sha256", mp, "raw"])
    json.dump(paths, open(f"{OUT}/paths.json", "w"), indent=0)
    tot = sum(os.path.getsize(p[1]) for p in paths)
    print("staged", len(paths), "files", round(tot / 1e6, 1), "MB; missing", missing)


if __name__ == "__main__":
    main()
