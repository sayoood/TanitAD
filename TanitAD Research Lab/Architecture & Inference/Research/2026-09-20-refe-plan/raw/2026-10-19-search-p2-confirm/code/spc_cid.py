#!/usr/bin/env python3
"""C-ID of SEARCH-P2-CONFIRM: the arm re-run on 16 carve-B tokens must reproduce Step B's banked P2HEAD outputs BIT FOR BIT, through the same path (a fresh GPU bundle for those 16 tokens, then the CPU search).

Three comparisons, all by token (the re-run bundle has its own token order and indices), all bitwise on the raw bytes (a NaN equals itself; a -0.0 differs from 0.0, which is the point):
  search  every array the search banks (exec, scorer_in, theta, moved, hist_*, pred_logit_out, pred_logit_pick, arm_argmax, pick_exec, pick)
  bundle  traj_raw, logits, pick, sub, pdms, valid  (the banked-fan side of the re-built bundle vs carve B's bundle)
  ctx     the visual-context rows (uint16 bf16 bits) of the re-built bundle vs carve B's ctx.npy
Non-vacuity (a bit-for-bit identity on tokens the search never moved is an identity of the pick with itself): the sample must contain at least `min_moved` tokens the BANKED search moved in >= 1 seed AND at least one it never moved.
    python spc_cid.py check --banked-search A.npz --rerun-search B.npz --banked-bundle a/bundle.npz --rerun-bundle b/bundle.npz --banked-ctx a/ctx.npy --rerun-ctx b/ctx.npy --out cid.json
Exit 0 PASS, 3 FAIL (the chain stops).  Reads no B-REST token and no label of one.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEARCH_KEYS = ("exec", "scorer_in", "theta", "moved", "hist_best_agg", "hist_mu", "pred_logit_out", "pred_logit_pick", "arm_argmax", "pick_exec", "pick")
BUNDLE_KEYS = ("traj_raw", "logits", "pick", "sub", "pdms", "valid")


def _row_map(ref_tokens, new_tokens):
    ix = {str(t): i for i, t in enumerate(ref_tokens)}
    miss = [str(t) for t in new_tokens if str(t) not in ix]
    if miss:
        raise ValueError(f"{len(miss)} re-run tokens are not in the banked file: {miss[:3]}")
    return np.array([ix[str(t)] for t in new_tokens], np.int64)


def _same_bytes(a, b) -> tuple:
    """-> (identical, n_rows_differing, max_abs_diff_or_None)"""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, int(a.shape[0]) if a.ndim else 1, None
    n = a.shape[0] if a.ndim else 1
    ra, rb = a.reshape(n, -1), b.reshape(n, -1)
    bad = [i for i in range(n) if ra[i].tobytes() != rb[i].tobytes()]
    mx = None
    if bad and np.issubdtype(a.dtype, np.number):
        try:
            mx = float(np.nanmax(np.abs(ra[bad].astype(np.float64) - rb[bad].astype(np.float64))))
        except Exception:
            mx = None
    return not bad, len(bad), mx


def compare_npz(banked: dict, rerun: dict, keys, ref_tokens, new_tokens, what: str) -> dict:
    rows = _row_map(ref_tokens, new_tokens)
    out = {}
    for k in keys:
        if k not in banked or k not in rerun:
            out[k] = {"identical": False, "why": f"key {k!r} missing in {'banked' if k not in banked else 're-run'} {what}"}
            continue
        ok, nbad, mx = _same_bytes(np.asarray(banked[k])[rows], np.asarray(rerun[k]))
        out[k] = {"identical": bool(ok), "rows_differing": nbad, "max_abs_diff": mx}
    return out


def compare_ctx(banked_ctx, rerun_ctx, ref_tokens, new_tokens) -> dict:
    rows = _row_map(ref_tokens, new_tokens)
    bad = [int(j) for j, r in enumerate(rows) if not np.array_equal(np.asarray(banked_ctx[r]), np.asarray(rerun_ctx[j]))]
    return {"identical": not bad, "rows_differing": len(bad), "n_rows": int(len(rows)), "shape_banked_row": list(np.asarray(banked_ctx[int(rows[0])]).shape) if len(rows) else None}


def check(banked_search: dict, rerun_search: dict, banked_bundle: dict, rerun_bundle: dict, banked_ctx, rerun_ctx, min_moved: int = 1) -> dict:
    stoks = [str(t) for t in rerun_search["token"]]
    btoks = [str(t) for t in rerun_bundle["token"]]
    res = {"n_tokens": len(stoks), "failures": []}
    if len(stoks) == 0:
        res["failures"].append("empty re-run")
    if sorted(stoks) != sorted(btoks):
        res["failures"].append("the re-run search and the re-run bundle are not on the same tokens")
    moved_any = np.asarray(banked_search["moved"]).any(1)
    try:
        ix = _row_map(banked_search["token"], stoks) if stoks else np.zeros(0, np.int64)
    except ValueError as e:                                  # a re-run token the banked search never saw: refused, not a crash
        res["failures"].append(str(e))
        res["pass"] = False
        return res
    n_moved, n_unmoved = int(moved_any[ix].sum()), int((~moved_any[ix]).sum())
    res["non_vacuity"] = {"tokens_banked_search_moved": n_moved, "tokens_never_moved": n_unmoved, "min_moved": min_moved}
    if n_moved < min_moved or n_unmoved < 1:
        res["failures"].append(f"vacuous sample: {n_moved} moved / {n_unmoved} never-moved tokens (need >= {min_moved} and >= 1)")
    try:
        s = compare_npz(banked_search, rerun_search, SEARCH_KEYS, banked_search["token"], stoks, "search")
        b = compare_npz(banked_bundle, rerun_bundle, BUNDLE_KEYS, banked_bundle["token"], btoks, "bundle")
        c = compare_ctx(banked_ctx, rerun_ctx, banked_bundle["token"], btoks)
    except ValueError as e:
        res["failures"].append(str(e))
        res["pass"] = False
        return res
    res["search"], res["bundle"], res["ctx"] = s, b, c
    for name, d in (("search", s), ("bundle", b)):
        for k, v in d.items():
            if not v["identical"]:
                res["failures"].append(f"{name}.{k} differs ({v.get('rows_differing', '?')} rows; {v.get('why', '')})")
    if not c["identical"]:
        res["failures"].append(f"ctx differs on {c['rows_differing']} of {c['n_rows']} rows")
    res["pass"] = not res["failures"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    for k in ("--banked-search", "--rerun-search", "--banked-bundle", "--rerun-bundle", "--banked-ctx", "--rerun-ctx", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--min-moved", type=int, default=1)
    a = ap.parse_args(argv)
    ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=False).items()}   # noqa: E731
    res = check(ld(a.banked_search), ld(a.rerun_search), ld(a.banked_bundle), ld(a.rerun_bundle), np.load(a.banked_ctx, mmap_mode="r"), np.load(a.rerun_ctx, mmap_mode="r"), a.min_moved)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZCID_PASS" if res["pass"] else "ZZCID_FAIL " + "; ".join(res["failures"])[:400], flush=True)
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
