"""PRIVGEOM-CLEAN coverage (LABEL-FREE): which of the 117 f10 logs resolve to a nuPlan DB, and how many sets / tokens the 76 not seen by PRIVGEOM-SAFE hold.

READS NO LABEL.  Store keys only (`fastneg_lib.read_members` refuses any member whose name contains 'label'); the census meta is read through its NON-label members only
(`log_idx`, `fold`, `code`, `key_*`, `log_names`), exactly as `pg_select.py` does.  The 41 logs PRIVGEOM-SAFE read are taken from ITS subset.json and are EXCLUDED from the read.

    python pgc_coverage.py --safe-subset <privgeom/run/subset.json> --out <run/coverage.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_extract as X                                                                   # noqa: E402

META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
META_NONLABEL = ("log_idx", "fold", "code", "key_log", "key_token", "key_step", "key_rank", "key_ckpt_step", "log_rank", "log_names")
DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
SCHEMA = "privgeom-clean-coverage-v1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe-subset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    safe = json.load(open(a.safe_subset, encoding="utf-8"))
    seen = list(safe["logs_in_order"])
    elig = X.eligible_logs()                                                             # {log: db path}: the landed `index_dbs` AND an f10 store exists
    all_logs = sorted(os.path.basename(p)[:-4] for p in __import__("glob").glob(f"{X.SETS_DIR}/*.npz"))
    unresolved = [n for n in all_logs if n not in elig]
    unseen = [n for n in all_logs if n not in set(seen)]
    unseen_ok = [n for n in unseen if n in elig]
    z = np.load(META, allow_pickle=False)
    M = {k: z[k] for k in META_NONLABEL}                                                  # never indexes 'labels6'
    where = {(str(l), str(t), int(s), int(r), int(c)): i for i, (l, t, s, r, c) in enumerate(zip(M["key_log"], M["key_token"], M["key_step"], M["key_rank"], M["key_ckpt_step"]))}
    per_log, census_idx = {}, []
    for n in unseen_ok:
        k = X.store_keys(n)
        toks = k["token"].astype("U")
        idx = [where[(n, str(toks[j]), int(k["step"][j]), int(k["rank"][j]), int(k["ckpt_step"][j]))] for j in range(len(toks))]
        census_idx += idx
        fold = sorted(set(int(M["fold"][i]) for i in idx))
        per_log[n] = {"n_sets": int(len(toks)), "n_tokens": int(len(set(toks.tolist()))), "fold": fold, "db": os.path.basename(elig[n]), "db_bytes": int(os.path.getsize(elig[n]))}
    census_idx = np.array(sorted(census_idx), np.int64)
    in_sub = np.zeros(len(M["fold"]), bool)
    in_sub[census_idx] = True
    code = M["code"][in_sub]
    folds = [int(len({int(x) for x in M["log_idx"][in_sub][M["fold"][in_sub] == f]})) for f in range(5)]
    sets_by_fold = [int((M["fold"][in_sub] == f).sum()) for f in range(5)]
    overlap = sorted(set(seen) & set(unseen_ok))
    doc = {"schema": SCHEMA, "n_f10_logs": len(all_logs), "n_f10_sets": int(len(M["fold"])), "n_resolved_to_db": len(elig), "n_unresolved": len(unresolved), "unresolved": unresolved,
           "n_seen_by_safe": len(seen), "seen_subset_sha256": hashlib.sha256(open(a.safe_subset, "rb").read()).hexdigest(),
           "seen_all_resolve": bool(all(n in elig for n in seen)),
           "n_unseen_logs": len(unseen), "n_unseen_resolved": len(unseen_ok), "overlap_with_seen": overlap,
           "n_sets": int(len(census_idx)), "n_tokens": int(sum(v["n_tokens"] for v in per_log.values())),
           "logs_per_fold": folds, "sets_per_fold": sets_by_fold, "proposals_per_stratum_S_Q1_Q2_Q3_Q4": [int((code == c).sum()) for c in range(5)],
           "db_bytes_total": int(sum(v["db_bytes"] for v in per_log.values())), "logs_in_order": unseen_ok, "per_log": per_log, "census_idx": census_idx.tolist(),
           "sets_per_log_min_med_max": [int(min(v["n_sets"] for v in per_log.values())), float(np.median([v["n_sets"] for v in per_log.values()])), int(max(v["n_sets"] for v in per_log.values()))]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("per_log", "census_idx", "logs_in_order")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
