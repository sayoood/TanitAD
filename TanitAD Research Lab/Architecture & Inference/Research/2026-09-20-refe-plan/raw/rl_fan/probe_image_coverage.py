#!/usr/bin/env python3
"""RL-S1 priority 2 probe: do the 51,000 partition scenes have all four camera frames on the pod?  (read-only; no pod is touched)

    python raw/rl_fan/probe_image_coverage.py --bank <train_grow/targets_rank0.jsonl> --train-log <refe-final-2026-10-01/train.log> \\
        --token-table raw/rl_pool/navtrain_token_table.tsv.gz --scenes <rl_train_scenes.tsv> <rl_val_scenes.tsv> [--out image_coverage.json]

The question has TWO answers, kept apart (CLAUDE.md "absence at one location is not absence"):
  1. THE ORIGINAL A40 POD (the tree the main run trained from).  MEASURED two ways that do not share an assumption:
       a. the trainer's own load-time line, in the banked train.log, for every grow epoch: "images: N/N tuples resolve ALL cameras under
          /workspace/data/navtrain_pixels (413,152 loose frames indexed ...)" -- the bank's rows all resolved, none dropped;
       b. closed-form arithmetic: the pod indexed 413,152 frames = 103,288 navtrain tokens x 4 cameras exactly; the rank-0 bank names 103,037 x 4 = 412,148
          distinct frames; the difference, 1,004 = 251 x 4, is exactly the 251 yaml tokens that have no bank row (and so no log).  Every partition scene
          is a bank row, hence one of the resolved ones.
  2. THE POD THE RL RUN WILL USE (the PI's P2 pod).  NOT MEASURABLE from here (no ssh by instruction).  It builds navtrain_pixels from the same 32 public
     OpenScene navtrain_current_N.tgz shards (code/pod_dataprep.sh STAGES=pix, 32/32 required), so coverage is INHERITED, not measured.  The measurement is
     `rl_fan_dump.py --plan-only` on that pod: exit 3 and unavailable_scenes_*.tsv on any gap, in seconds.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--token-table", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"tool": "probe_image_coverage/1"}
    refs, toks, ncam = set(), set(), collections.Counter()
    for ln in open(a.bank, encoding="utf-8"):
        r = json.loads(ln)
        toks.add(r["token"])
        im = r["image"] if isinstance(r["image"], list) else [r["image"]]
        ncam[len(im)] += 1
        refs.update(x.split("/")[-1] for x in im)
    res["bank_rank0"] = {"rows": sum(ncam.values()), "distinct_tokens": len(toks), "cameras_per_row": dict(ncam), "distinct_frame_basenames": len(refs)}
    with gzip.open(a.token_table, "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        it, il = head.index("token"), head.index("log")
        pool = [l.rstrip("\n").split("\t") for l in f]
    res["token_table"] = {"scenes": len(pool), "with_log": sum(1 for p in pool if p[il]), "without_log": sum(1 for p in pool if not p[il])}
    lines = []
    for ln in open(a.train_log, encoding="utf-8"):
        m = re.match(r"\s*images: ([\d,]+)/([\d,]+) tuples resolve ALL cameras under (\S+)\s+\(([\d,]+) loose frames indexed", ln)
        if m:
            lines.append({"resolved": int(m.group(1).replace(",", "")), "of": int(m.group(2).replace(",", "")), "root": m.group(3), "indexed_frames": int(m.group(4).replace(",", ""))})
    res["trainer_load_lines"] = {"n_lines": len(lines), "all_resolved": all(l["resolved"] == l["of"] for l in lines), "last": lines[-1] if lines else None,
                                 "indexed_frames_values": sorted({l["indexed_frames"] for l in lines})}
    idx = lines[-1]["indexed_frames"] if lines else None
    res["arithmetic"] = {"tokens_x4": 4 * len(pool), "pod_indexed_frames": idx, "equal": idx == 4 * len(pool),
                         "bank_distinct_frames": len(refs), "frames_of_tokens_without_a_bank_row": 4 * (len(pool) - len(toks)),
                         "bank_plus_unbanked_equals_indexed": (len(refs) + 4 * (len(pool) - len(toks)) == idx) if idx else None}
    sc = {}
    for p in a.scenes:
        rows = [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8")][1:]
        for t, lg in rows:
            sc[t] = lg
    bank_logs = {}
    for ln in open(a.bank, encoding="utf-8"):
        r = json.loads(ln)
        bank_logs[r["token"]] = r["log_name"]
    res["partition_scenes"] = {"n": len(sc), "in_bank": sum(1 for t in sc if t in bank_logs), "log_agrees_with_bank": sum(1 for t, lg in sc.items() if bank_logs.get(t) == lg),
                               "frames_named": 4 * len(sc)}
    ok = (res["trainer_load_lines"]["all_resolved"] and res["arithmetic"]["equal"] and res["arithmetic"]["bank_plus_unbanked_equals_indexed"]
          and res["partition_scenes"]["in_bank"] == len(sc) == res["partition_scenes"]["log_agrees_with_bank"])
    res["original_a40_pod_all_partition_frames_present"] = bool(ok)
    res["p2_pod"] = "UNVERIFIED (inherited: same 32 shards by code/pod_dataprep.sh); measure with rl_fan_dump.py --plan-only"
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZCOVERAGE_" + ("COMPLETE" if ok else "GAP"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
