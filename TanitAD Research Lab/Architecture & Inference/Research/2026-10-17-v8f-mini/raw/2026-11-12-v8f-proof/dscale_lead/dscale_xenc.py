#!/usr/bin/env python
"""dscale_xenc.py -- report d3x of card PREREG_V8F_PROOF_ADD45 (note N3): how close are two encodes of the same evaluation clips? Reported, gates nothing. CPU, torch + numpy.

The caches of two pods hold the same clips under the same pipeline tag with different bytes (control d3t reads 0 clips with equal content md5). This tool measures
the size of that difference on a fixed sample: the first --clips evaluation clips of STUDY_EVAL_MD5.tsv, token frames 0, 10, ..., 90, cache key pool4x10.

    sample   writes the sample of THIS cache (float16 holds every fp8 value exactly):   <out>.npz  (ids, frames, tok [clips, frames, 40, 1024], content_md5)
    compare  reads another pod's sample and this cache, writes a json:
             cos_token_*           cosine between the two encodes of one token, over all tokens of the sample (mean, 1st percentile, minimum)
             rel_abs_diff          mean absolute difference over mean absolute value
             pearson_r, share_equal, max_abs_diff
             cos_token_2s_apart_*  the same cosine between two token frames of ONE encode 10 token frames (2.0 s) apart, same clip, same token position: the scale to
                                   read the first number against (how much the tokens move in 2.0 s of driving)

Exit codes: 0 written, 1 an input is missing or a clip of the sample is not in this cache.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

FRAMES = list(range(0, 100, 10))
KEY = "pool4x10"


def find_manifest(cache, split="eval"):
    for p in (os.path.join(cache, split, "manifest.jsonl"), os.path.join(cache, "manifest.jsonl")):
        if os.path.exists(p):
            return p
    raise SystemExit("MISSING: no manifest.jsonl under %s" % cache)


def manifest_rows(cache):
    p = find_manifest(cache)
    rows = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            rows[str(d["clip"])] = d
    return os.path.dirname(os.path.abspath(p)), rows


def study_ids(tsv, n):
    ids = []
    for ln in open(tsv, encoding="utf-8"):
        ln = ln.rstrip("\n").rstrip("\r")
        if ln and not ln.startswith("#"):
            ids.append(ln.split("\t")[0])
    return ids[:n]


def load_tokens(root, row, frames):
    import torch
    o = torch.load(os.path.join(root, row.get("file") or ("clips/%s.pt" % row["clip"])), map_location="cpu", weights_only=True)
    t = o[KEY].to(torch.float32)
    if t.ndim != 3 or t.shape[1] != 40 or t.shape[2] != 1024 or t.shape[0] <= max(frames):
        raise SystemExit("REFUSED: %s of %s has shape %s" % (KEY, row["clip"], tuple(t.shape)))
    return t[frames].numpy()


def read_sample(cache, ids, frames):
    root, rows = manifest_rows(cache)
    miss = [c for c in ids if c not in rows]
    if miss:
        raise SystemExit("MISSING: %d clips of the sample are not in this cache, first %s" % (len(miss), miss[0]))
    tok = np.stack([load_tokens(root, rows[c], frames) for c in ids]).astype(np.float32)
    return tok, [str(rows[c].get("content_md5", "")) for c in ids]


def cosines(a, b):
    a = a.reshape(-1, a.shape[-1]).astype(np.float64)
    b = b.reshape(-1, b.shape[-1]).astype(np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.where(den > 0, den, 1.0)


def stats(a, b):
    c = cosines(a, b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"n_tokens": int(c.size), "cos_token_mean": float(c.mean()), "cos_token_p01": float(np.percentile(c, 1)), "cos_token_min": float(c.min()),
            "rel_abs_diff": float(d.mean() / max(float(np.abs(a).mean()), 1e-12)), "pearson_r": float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1]),
            "share_equal": float((d == 0).mean()), "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d))}


def apart(tok):
    c = cosines(tok[:, :-1], tok[:, 1:])
    return {"cos_token_2s_apart_mean": float(c.mean()), "cos_token_2s_apart_p01": float(np.percentile(c, 1)), "n_token_pairs": int(c.size)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    c = sub.add_parser("compare")
    for p in (s, c):
        p.add_argument("--eval-cache", required=True)
        p.add_argument("--out", required=True)
    s.add_argument("--study-eval-tsv", required=True)
    s.add_argument("--clips", type=int, default=20)
    c.add_argument("--sample", required=True, help="the npz another pod wrote with the sample subcommand")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        ids = study_ids(a.study_eval_tsv, a.clips)
        tok, md5 = read_sample(a.eval_cache, ids, FRAMES)
        t16 = tok.astype(np.float16)
        if not np.array_equal(t16.astype(np.float32), tok):
            raise SystemExit("REFUSED: float16 does not hold the tokens exactly")
        np.savez_compressed(a.out, ids=np.array(ids), frames=np.array(FRAMES, dtype=np.int64), tok=t16, content_md5=np.array(md5))
        print("xenc sample: %d clips x %d frames, key %s, mean abs %.4f -> %s" % (len(ids), len(FRAMES), KEY, float(np.abs(tok).mean()), a.out))
        return 0
    if not os.path.exists(a.sample):
        print("MISSING:", a.sample)
        return 1
    z = np.load(a.sample)
    ids, frames, other = [str(x) for x in z["ids"]], [int(x) for x in z["frames"]], z["tok"].astype(np.float32)
    here, md5 = read_sample(a.eval_cache, ids, frames)
    res = {"tool": "dscale_xenc.py", "gates": False, "key": KEY, "clips": len(ids), "frames": frames, "sample_sha256": hashlib.sha256(open(a.sample, "rb").read()).hexdigest(),
           "clips_same_content_md5": int(sum(1 for x, y in zip(md5, [str(v) for v in z["content_md5"]]) if x and x == y)),
           "between_the_two_encodes": stats(here, other), "inside_this_encode": apart(here), "inside_the_other_encode": apart(other),
           "self_check_this_encode_against_itself": stats(here, here)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    b = res["between_the_two_encodes"]
    print("xenc compare: cosine per token mean %.6f p01 %.6f min %.6f; rel abs diff %.5f; share equal %.4f; 2.0 s apart inside this encode: cosine mean %.4f -> %s" % (
        b["cos_token_mean"], b["cos_token_p01"], b["cos_token_min"], b["rel_abs_diff"], b["share_equal"], res["inside_this_encode"]["cos_token_2s_apart_mean"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
