#!/usr/bin/env python3
"""SEARCH-PESSIMISTIC feasibility memo: the session-level exposure probe (CPU, read-only, no GPU, no network).

Inputs (both exported from the mirror at 0738fa3 with `git show`):
  navtrain_token_table.tsv.gz  <- PK/raw/rl_pool/navtrain_token_table.tsv.gz
  navtest_plan.json            <- PK/raw/2026-10-19-search-p2-confirm/c2/result/navtest_plan.json
A drive SESSION is the log name minus its two trailing segment indices (`2021.06.14.18.42.45_veh-12_01345_01523` -> `2021.06.14.18.42.45_veh-12`).
Writes session_probe.json beside itself.

    python session_probe.py <token_table.tsv.gz> <navtest_plan.json>
"""
import collections
import csv
import gzip
import json
import os
import sys


def sess(log):
    return log.rsplit('_', 2)[0]


def main(tt, nt):
    rows = list(csv.DictReader(gzip.open(tt, 'rt'), delimiter='\t'))
    out = {"table_rows": len(rows)}
    mp_tok, mp_sets, mp_seg = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    ho_tok, ho_seg = collections.Counter(), collections.defaultdict(set)
    for r in rows:
        if int(r['served_train_sets']) > 0:
            mp_tok[r['log']] += 1
            mp_sets[r['log']] += int(r['served_train_sets'])
            mp_seg[sess(r['log'])].add(r['log'])
        if int(r['heldout_log']) == 1:
            ho_tok[r['log']] += 1
            ho_seg[sess(r['log'])].add(r['log'])
    out["M_PDM"] = {"tokens": sum(mp_tok.values()), "segments": len(mp_tok), "sets": sum(mp_sets.values()), "sessions": len(mp_seg),
                    "multi_segment_sessions": sum(1 for v in mp_seg.values() if len(v) > 1)}
    shared = [s for s in ho_seg if s in mp_seg]
    mp_tok_by_sess = collections.Counter()
    for lg, n in mp_tok.items():
        mp_tok_by_sess[sess(lg)] += n
    out["heldout"] = {"tokens": sum(ho_tok.values()), "segments": len(ho_tok), "sessions": len(ho_seg),
                      "sessions_shared_with_M_PDM": len(shared), "M_PDM_tokens_in_shared_sessions": sum(mp_tok_by_sess[s] for s in shared)}
    # L2's segment folds (rank mod 5 over the 24 sorted segment names) vs session folds (rank mod 5 over the sorted sessions)
    segs = sorted(ho_tok)
    segfold = {lg: i % 5 for i, lg in enumerate(segs)}
    out["segment_folds_tokens"] = [sum(ho_tok[lg] for lg in segs if segfold[lg] == f) for f in range(5)]
    out["sessions_split_across_segment_folds"] = sum(1 for s, v in ho_seg.items() if len({segfold[lg] for lg in v}) > 1)
    ss = sorted(ho_seg)
    out["session_folds_tokens_rank_mod5"] = [sum(ho_tok[lg] for i, s in enumerate(ss) if i % 5 == f for lg in ho_seg[s]) for f in range(5)]
    p = json.load(open(nt))
    nlogs = set(p['token_log'].values())
    nsess = {sess(lg) for lg in nlogs}
    allsess = {sess(r['log']) for r in rows}
    out["navtest"] = {"tokens": len(p['token_log']), "segments": len(nlogs), "sessions": len(nsess), "sessions_shared_with_navtrain": len(nsess & allsess)}
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_probe.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
