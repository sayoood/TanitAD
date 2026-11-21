#!/usr/bin/env python3
"""Assemble RESULT.md from the hand-written parts + the generated raw/tables.md, and write inputs.json and paths.json.
    python assemble_result.py <parts dir>      (run from C:/Users/Admin/hncensus)
"""
import hashlib
import json
import os
import sys

SP = sys.argv[1].rstrip("/") + "/"
ROOT = "C:/Users/Admin/hncensus"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-hn-census"
os.chdir(ROOT)


def rd(n):
    return open(SP + n, encoding="utf-8").read()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


tb = open("raw/tables.md", encoding="utf-8").read()
i_k2, i_t8, i_t10 = tb.index("### K2"), tb.index("### T8"), tb.index("### T10")
A, K, Bt, C = tb[:i_k2], tb[i_k2:i_t8], tb[i_t8:i_t10], tb[i_t10:]
out = rd("head.md") + A + rd("s3.md") + K + rd("s5.md") + Bt + rd("s6.md") + C + rd("s7.md")
open("RESULT.md", "w", encoding="utf-8", newline="\n").write(out)
print(len(out), "chars in RESULT.md")

meta = json.load(open("raw/hn_joined_meta.json"))
inputs = {"prereg_sha256": open("PREREG_HN_CENSUS.md.sha256").read().split()[0],
          "derived_cache": {"raw/hn_joined.npz": sha("raw/hn_joined.npz"), "raw/human_future.npz": sha("raw/human_future.npz")},
          "banked_inputs_sha256": meta["inputs_sha256"], "queue_files": meta["queue_files"],
          "paths": {"labels": "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r*_hw*.jsonl",
                    "sets": "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl",
                    "teacher_queue": "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/queue/props_r*",
                    "human_dbs": "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/**/<log>.db (13 of 24 logs)",
                    "sibling_scorers": "D:/Projects/TanitAD-artifacts/s1_read_cache/arms/N-s{0,1,2}/dump/scores_update400_B.npz",
                    "navtest_table": "NOT OPENED"},
          "mirror_tip_commit_read": "5448a05", "interpreter": "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe (numpy 1.26.4, python 3.11.13)"}
json.dump(inputs, open("inputs.json", "w"), indent=1)

res = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in sorted(fn):
        if f == "paths.json" or f.startswith("_") or f == "hn_joined.npz" or f.endswith(".pyc") or f == "HANDOFF.md":
            continue
        ab = os.path.join(dp, f).replace(chr(92), "/")
        rel = os.path.relpath(ab, ROOT).replace(chr(92), "/")
        res.append([f"{REPO}/{rel}", ab, "raw"])
json.dump(sorted(res), open(os.path.join(ROOT, "paths.json"), "w", encoding="utf-8"), indent=1)
print(len(res), "files in paths.json")
