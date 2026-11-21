#!/usr/bin/env python3
"""Stage the UNSEEN-STRATUM S1b package for landing: LF copies into C:/Users/Admin/par18/unseen_s1b/out/repo/<repo path> + out/paths.json ([[repo path, staged abs path, "raw"], ...]).
Nothing is committed or pushed.  The scorer files (refe/navsim_pdm_targets.py ...) are NOT copied: they are the tip's own, byte-identical (tree/refe/ checked by sha256)."""
import glob
import hashlib
import json
import os
import shutil

W = "C:/Users/Admin/par18/unseen_s1b"
OUT = f"{W}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-unseen-stratum-s1b"
CODE = ["s1b_lib.py", "s1b_rules.py", "s1b_worker.py", "s1b_chain.py", "analyze_identity.py", "make_filtered_fetch_list.py", "test_s1b_rules.py", "make_stage.py", "headline.py"]
EXPLORE = ["t_timing.py", "t_prof.py", "t_rss.py"]
TEXT_EXT = (".py", ".json", ".md", ".txt", ".log", ".jsonl", ".done", ".logs")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    stage = f"{OUT}/repo/{REPO}"
    if os.path.exists(f"{OUT}/repo"):
        shutil.rmtree(f"{OUT}/repo")
    items = [(f"code/{f}", f"{W}/code/{f}") for f in CODE] + [(f"code/explore/{f}", f"{W}/code/{f}") for f in EXPLORE]
    items += [(f, f"{W}/{f}") for f in ("RESULT.md", "HANDOFF.md", "state.json") if os.path.exists(f"{W}/{f}")]
    items += [(os.path.relpath(p, W).replace("\\", "/"), p) for p in sorted(glob.glob(f"{W}/raw/**", recursive=True)) if os.path.isfile(p)]
    items += [(f"logs/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{W}/logs/*")) if os.path.isfile(p)]
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
