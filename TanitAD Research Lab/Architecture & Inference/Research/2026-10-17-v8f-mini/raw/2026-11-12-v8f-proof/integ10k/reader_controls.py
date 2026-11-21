#!/usr/bin/env python
"""reader_controls.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE): the pod-side reader controls k1 (reader quality on the 534 reference windows), k6 (no leak) and k7 (causal reader), per reader arm.

  python reader_controls.py run --reader CKPT --arm LR16_s0 --eval-cache DIR_WITH_eval --join join_eval141.npz --ref LS_REF40_WINDOWS.json --train-manifest CACHE_TR/train/manifest.jsonl --out OUT.json

Reading of k1 (card; note of the lead on the keys): a window of the reference file is scored at CACHE step g0 with the five-frame history g0-4 .. g0 (the key t is not used).
  gap error = mean |min(g_read, 40) - gap_t| over the 534 windows; presence AUROC over all labelled evaluation rows with a complete history (positive: label lead present with a gap of at most 40 m).
  label check: the join label gap at (clip eid, cache step g0) equals gap_t within 1e-3; the share that matches is stated.
Tier T0 (a read on logged tokens); never driving performance.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tok_reader as TR  # noqa: E402

GAP_CLIP_M = 40.0
MAE_BAR_M = 5.80
AUROC_BAR = 0.85
LABEL_TOL = 1e-3


def auroc(score, pos):
    """rank-based AUROC (ties averaged); NaN when a class is empty."""
    s = np.asarray(score, np.float64)
    y = np.asarray(pos, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gap_error(g_read, gap_t, clip_m=GAP_CLIP_M):
    return float(np.mean(np.abs(np.minimum(np.asarray(g_read, np.float64), clip_m) - np.asarray(gap_t, np.float64))))


def leak_count(reader_clip_ids, eval_clip_ids):
    """k6: number of evaluation clips among the reader's training or held-out clips (exactly 0 is the bar)."""
    return len(set(map(str, reader_clip_ids)) & set(map(str, eval_clip_ids)))


def causal_unchanged(reader, tokens, tick, rng_seed=0, read=TR.read_clip):
    """k7: the reading at `tick` is the same when every frame after the tick is replaced by noise."""
    p0, g0, ok0 = read(reader, tokens)
    t2 = np.array(tokens, np.float32, copy=True)
    rng = np.random.default_rng(rng_seed)
    t2[tick + 1:] = rng.normal(0.0, 3.0, size=t2[tick + 1:].shape).astype(np.float32)
    p1, g1, ok1 = read(reader, t2)
    return bool(ok0[tick] and p0[tick] == p1[tick] and g0[tick] == g1[tick])


def cmd_run(a):
    import torch
    reader = TR.TokReader.from_ckpt(a.reader)
    ref = json.load(open(a.ref, encoding="utf-8"))
    J = np.load(a.join, allow_pickle=True)
    jclip = [str(c) for c in J["clip"]]
    row = {c: i for i, c in enumerate(jclip)}
    eman = [json.loads(l)["clip"] for l in open(os.path.join(a.eval_cache, "eval", "manifest.jsonl"), encoding="utf-8") if l.strip()]
    tman = [json.loads(l)["clip"] for l in open(a.train_manifest, encoding="utf-8") if l.strip()]
    cache = {}

    def clip_rows(c):
        if c not in cache:
            d = torch.load(os.path.join(a.eval_cache, "eval", "clips", c + ".pt"), map_location="cpu")
            x = d["pool4x10"].float().numpy()
            cache[c] = (x,) + TR.read_clip(reader, x)
        return cache[c]
    eids = [str(e) for e in ref["eid"]]
    g0 = np.asarray(ref["g0"], np.int64)
    gap_t = np.asarray(ref["gap_t"], np.float64)
    gr = np.empty(len(eids))
    pr = np.empty(len(eids))
    okr = np.zeros(len(eids), bool)
    for i, (e, s) in enumerate(zip(eids, g0)):
        x, p, g, ok = clip_rows(e)
        pr[i], gr[i], okr[i] = p[s], g[s], ok[s]
    ci = np.array([row[e] for e in eids], np.int64)
    lp, lg = TR.label_rows(J, ci, g0)
    match = lp & (np.abs(lg - gap_t) <= LABEL_TOL)
    # AUROC over every labelled evaluation row with a complete history
    sc, ps = [], []
    for c in eman:
        x, p, g, ok = clip_rows(c)
        t = np.arange(len(p))
        lpc, lgc = TR.label_rows(J, np.full(len(p), row[c], np.int64), t)
        sel = ok
        sc.append(p[sel])
        ps.append((lpc & (lgc <= GAP_CLIP_M))[sel])
        if len(cache) > 20:
            for k in list(cache)[:10]:
                cache.pop(k)
    au = auroc(np.concatenate(sc), np.concatenate(ps))
    mae = gap_error(gr[okr], gap_t[okr])
    # k7 on the first 40 reference windows
    k7 = []
    for e, s in list(zip(eids, g0))[:40]:
        d = torch.load(os.path.join(a.eval_cache, "eval", "clips", e + ".pt"), map_location="cpu")
        k7.append(causal_unchanged(reader, d["pool4x10"].float().numpy(), int(s)))
    leak = leak_count(tman, eman)
    out = {"arm": a.arm, "reader": a.reader, "reader_sha256": TR.sha256_file(a.reader), "card": "PREREG_V8F_PROOF_ADD46", "tier": "T0",
           "k1": {"n_windows": int(len(eids)), "n_scored": int(okr.sum()), "gap_mae_m_clipped_40": mae, "auroc": au, "label_match_share": float(match.mean()), "label_match_n": int(match.sum()),
                  "auroc_rows": int(sum(len(x) for x in sc)), "auroc_positives": int(sum(int(x.sum()) for x in ps)), "mae_bar_m": MAE_BAR_M, "auroc_bar": AUROC_BAR},
           "k6": {"n_eval_clips": len(eman), "n_reader_clips": len(tman), "leak_count": leak, "n_heldout": int(sum(TR.is_heldout(c) for c in tman)),
                  "planted_shared_id_detected": bool(leak_count(list(tman) + [eman[0]], eman) >= 1)},
           "k7": {"n_checked": len(k7), "all_unchanged": bool(all(k7))}}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({"arm": a.arm, "mae": mae, "auroc": au, "match": out["k1"]["label_match_share"], "leak": leak, "k7": out["k7"]["all_unchanged"]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for k in ("reader", "arm", "eval-cache", "join", "ref", "train-manifest", "out"):
        r.add_argument("--" + k, required=True)
    return cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
