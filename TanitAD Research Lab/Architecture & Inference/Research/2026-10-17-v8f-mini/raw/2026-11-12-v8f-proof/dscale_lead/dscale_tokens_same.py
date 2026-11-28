#!/usr/bin/env python
"""dscale_tokens_same.py -- control d2e and report d3t of card PREREG_V8F_PROOF_ADD45 (note N2). Standard library only, CPU, seconds.

d2e (gates): the evaluation cache on this pod holds the study's evaluation set: the clip ids of its manifest, in manifest order, equal those of the study's evaluation
manifest (file STUDY_EVAL_MD5.tsv of the kit: clip id, content md5, one line per clip in the study's manifest order). The reader pairs the extracted states with the
evaluation pack of card ADD43 by position, so another order would misalign every window. ``--plant-swap`` swaps two ids before the check: it must then fail.

d3t (reported, gates nothing): how many clips carry the same content md5 as in the study's caches, for the evaluation clips and for the corpus clips of the union
(file STUDY_TRAIN_MD5.tsv). Equal md5 = the probe on this pod reads the study's tokens for those clips; different md5 is stated in the row as a limit of the link to
row 97 (the named difference, union minus corpus, is taken inside this pod's cache either way).

Exit codes: 0 d2e passes, 2 d2e fails, 1 an input is missing.
"""
import argparse
import hashlib
import json
import os
import sys


def find_manifest(cache, split):
    for p in (os.path.join(cache, split, "manifest.jsonl"), os.path.join(cache, "manifest.jsonl")):
        if os.path.exists(p):
            return p
    raise SystemExit("MISSING: no manifest.jsonl under %s (looked in %s/ and the root)" % (cache, split))


def read_manifest(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            rows.append((str(d["clip"]), str(d.get("content_md5", ""))))
    return rows


def read_tsv(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n").rstrip("\r")
        if ln and not ln.startswith("#"):
            a = ln.split("\t")
            rows.append((a[0], a[1] if len(a) > 1 else ""))
    return rows


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--union-cache", required=True)
    ap.add_argument("--bset-list", required=True)
    ap.add_argument("--study-eval-tsv", required=True)
    ap.add_argument("--study-train-tsv", required=True)
    ap.add_argument("--expect-eval", type=int, default=141)
    ap.add_argument("--plant-swap", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for p in (a.bset_list, a.study_eval_tsv, a.study_train_tsv):
        if not os.path.exists(p):
            print("MISSING:", p)
            return 1
    ev = read_manifest(find_manifest(a.eval_cache, "eval"))
    st = read_tsv(a.study_eval_tsv)
    ids = [c for c, _ in ev]
    if a.plant_swap and len(ids) >= 2:
        ids[0], ids[1] = ids[1], ids[0]
    want = [c for c, _ in st]
    same_order = ids == want
    md5_here, md5_study = dict(ev), dict(st)
    ev_same = sum(1 for c in want if md5_here.get(c) and md5_here.get(c) == md5_study.get(c))
    d2e = {"pass": bool(same_order and len(ids) == a.expect_eval and len(want) == a.expect_eval), "n_here": len(ids), "n_study": len(want), "same_ids_same_order": same_order,
           "same_id_set": sorted(ids) == sorted(want), "sha256_ids_here": sha_ids(ids), "sha256_ids_study": sha_ids(want), "planted_swap": bool(a.plant_swap)}
    un = dict(read_manifest(find_manifest(a.union_cache, "train")))
    tr = dict(read_tsv(a.study_train_tsv))
    corpus = [ln.strip() for ln in open(a.bset_list, encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]
    in_study = [c for c in corpus if c in tr]
    in_union = [c for c in in_study if c in un]
    have_md5 = [c for c in in_union if un[c] and tr[c]]
    d3t = {"gates": False, "eval_clips_same_content_md5": ev_same, "eval_clips": len(want), "corpus_ids": len(corpus), "corpus_ids_in_the_study_manifest": len(in_study),
           "corpus_ids_in_the_union_manifest": len(in_union), "corpus_clips_with_md5_on_both_sides": len(have_md5),
           "corpus_clips_same_content_md5": sum(1 for c in have_md5 if un[c] == tr[c])}
    res = {"tool": "dscale_tokens_same.py", "pass": d2e["pass"], "d2e_eval_set_is_the_study_set": d2e, "d3t_tokens_same": d3t}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("evalset: pass %s -> %s" % (res["pass"], a.out))
    print("   d2e", json.dumps(d2e)[:400])
    print("   d3t", json.dumps(d3t)[:400])
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
