#!/usr/bin/env python
"""dscale_clockcheck.py -- control d2c of card PREREG_V8F_PROOF_ADD45 (note N2): are the rows of the union lead file on the clock of the cached tokens?

CPU only (run with CUDA_VISIBLE_DEVICES=-1). The lead file stores, per raw frame k of a clip, the ego speed ``v`` beside the lead flag and the gap. The token cache stores,
per token frame j of the same clip (raw frame 2 j), the ego speed ``v`` of that frame. If the lead file is on the clock of the tokens, the two speeds agree at k = 2 j and
disagree more at every shifted pairing k = 2 (j + s), s != 0. The check reads, per group (corpus clips, new clips), the mean absolute speed difference at the shifts
-5, -2, -1, 0, 1, 2, 5 token frames over ``--n-clips`` clips drawn at random, and passes when in BOTH groups

  * the smallest mean absolute difference sits at shift 0, and
  * the value at shift 0 is at most ``--ratio`` (0.5) times the smaller of the values at shifts -1 and +1.

``--plant-shift P`` moves the lead rows by P token frames before the check (k -> k + 2 P): the check must then fail (exit code 2). Exit codes: 0 pass, 2 fail, 1 refused.
It names nothing of the card: it is a control of the inputs.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

SHIFTS = (-5, -2, -1, 0, 1, 2, 5)
MIN_CLIPS = 100


def sha12(clip_id):
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def find_manifest(cache):
    for p in (os.path.join(cache, "train", "manifest.jsonl"), os.path.join(cache, "manifest.jsonl")):
        if os.path.exists(p):
            return p
    raise SystemExit("MISSING: no manifest.jsonl under %s (looked in train/ and the root)" % cache)


def read_manifest(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            out.setdefault(d["clip"], d)
    return out


def read_ids(path):
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]


class LeadSpeed:
    """The lead npz: per clip (by sha12) the raw frame index k and the ego speed v."""

    def __init__(self, path, column):
        z = np.load(path, allow_pickle=True)
        for k in ("sha12", "k", column):
            if k not in z.files:
                raise SystemExit("MISSING: lead file %s has no key %s (has %s)" % (path, k, z.files))
        sh = np.asarray(z["sha12"]).astype(str)
        order = np.argsort(sh, kind="stable")
        self.sh = sh[order]
        self.k = np.asarray(z["k"], np.int64)[order]
        self.v = np.asarray(z[column], np.float64)[order]
        uniq, start = np.unique(self.sh, return_index=True)
        stop = np.append(start[1:], len(self.sh))
        self.where = {str(u): (int(a), int(b)) for u, a, b in zip(uniq, start, stop)}

    def clip(self, cid):
        r = self.where.get(sha12(cid))
        if r is None:
            return None
        return self.k[r[0]:r[1]], self.v[r[0]:r[1]]


def cache_speed(path, key):
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        d = torch.load(path, map_location="cpu", weights_only=False)
    if key not in d:
        raise SystemExit("MISSING: cache clip %s has no key %s (has %s)" % (path, key, sorted(d.keys())))
    return d[key].to(dtype=torch.float64).numpy()


def group_read(ids, man, root, lead, key, plant):
    """Sums of |v_lead(k) - v_cache(k/2 + s)| per shift over the clips ``ids``. Returns (per-shift sum, per-shift count, clips read, clips without rows)."""
    tot = {s: 0.0 for s in SHIFTS}
    cnt = {s: 0 for s in SHIFTS}
    read = no_rows = 0
    for c in ids:
        r = lead.clip(c)
        if r is None:
            no_rows += 1
            continue
        k, v = r
        k = k + 2 * plant
        even = (k % 2 == 0) & np.isfinite(v)
        j0, vl = (k[even] // 2), v[even]
        vc = cache_speed(os.path.join(root, man[c]["file"]), key)
        n = len(vc)
        for s in SHIFTS:
            j = j0 + s
            ok = (j >= 0) & (j < n)
            if ok.any():
                d = np.abs(vl[ok] - vc[j[ok]])
                d = d[np.isfinite(d)]
                tot[s] += float(d.sum())
                cnt[s] += int(len(d))
        read += 1
    return tot, cnt, read, no_rows


def judge(tot, cnt, ratio):
    mad = {s: (tot[s] / cnt[s] if cnt[s] else float("nan")) for s in SHIFTS}
    vals = [mad[s] for s in SHIFTS]
    if not all(np.isfinite(vals)):
        return {"pass": False, "reason": "a shift has no rows", "mean_abs_diff_mps": {str(s): mad[s] for s in SHIFTS}, "rows": {str(s): cnt[s] for s in SHIFTS}}
    best = SHIFTS[int(np.argmin(vals))]
    near = min(mad[-1], mad[1])
    ok = bool(best == 0 and mad[0] <= ratio * near)
    return {"pass": ok, "argmin_shift": int(best), "ratio_shift0_to_nearest": (mad[0] / near if near > 0 else float("inf")), "ratio_bar": ratio,
            "mean_abs_diff_mps": {str(s): round(mad[s], 6) for s in SHIFTS}, "rows": {str(s): cnt[s] for s in SHIFTS}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-cache", required=True)
    ap.add_argument("--bset-list", required=True, help="the corpus clip ids inside the union")
    ap.add_argument("--lead-union", required=True)
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--lead-speed-column", default="v")
    ap.add_argument("--cache-speed-key", default="v")
    ap.add_argument("--plant-shift", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.n_clips < MIN_CLIPS:
        raise SystemExit("REFUSED: --n-clips %d is under the minimum %d" % (a.n_clips, MIN_CLIPS))
    if a.ratio != 0.5 and not os.environ.get("DSCALE_CLOCK_TEST"):
        raise SystemExit("REFUSED: the bar of the check is fixed at 0.5 (note N2)")
    t0 = time.time()
    mpath = find_manifest(a.union_cache)
    root = os.path.dirname(mpath)
    man = read_manifest(mpath)
    corpus = set(read_ids(a.bset_list))
    lead = LeadSpeed(a.lead_union, a.lead_speed_column)
    rng = np.random.default_rng(a.seed)
    res = {"tool": "dscale_clockcheck.py", "sha256_tool": sha256_file(os.path.abspath(__file__)), "manifest": mpath, "lead_union": a.lead_union,
           "sha256_lead_union": sha256_file(a.lead_union), "n_clips": a.n_clips, "seed": a.seed, "shifts": list(SHIFTS), "plant_shift": a.plant_shift, "groups": {}}
    ok_all = True
    for name, pool in (("corpus", [c for c in man if c in corpus]), ("new", [c for c in man if c not in corpus])):
        pool = [c for c in pool if sha12(c) in lead.where]
        if len(pool) < a.n_clips:
            res["groups"][name] = {"pass": False, "reason": "only %d clips with rows in the lead file, %d asked" % (len(pool), a.n_clips)}
            ok_all = False
            continue
        pick = [pool[i] for i in rng.choice(len(pool), size=a.n_clips, replace=False)]
        tot, cnt, read, no_rows = group_read(pick, man, root, lead, a.cache_speed_key, a.plant_shift)
        g = judge(tot, cnt, a.ratio)
        g.update({"clips_read": read, "clips_without_rows": no_rows, "pool": len(pool)})
        res["groups"][name] = g
        ok_all = ok_all and g["pass"]
    res["pass"] = bool(ok_all)
    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("clockcheck: pass %s (plant %d) -> %s" % (res["pass"], a.plant_shift, a.out))
    for name, g in res["groups"].items():
        print("  ", name, json.dumps({k: g[k] for k in g if k != "rows"})[:400])
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
