"""SPEC §3.4: fit the positional-prior control (`taniteval.map_hires_metrics.PositionalPrior`) on the
300-clip TRAIN fit set (`prior_select.py`), from the `/3` GT pulled read-only from Thor.

    python prior_fit.py --fit-set prior_fit_set.json --gt-dir gt_train300 --md5-thor thor_md5_fit.txt \
        --md5-local local_md5_fit.txt --out prior_train300.npz

Every frame of every fit clip enters the counts (the prior is 'each cell's train-set majority class').
The md5 of every fit file must match Thor's, and the fit set must not intersect the eval clips (the
prior itself refuses to predict a clip it was fitted on). The extent is refcv7's (100 m x +-30 m).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-set", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--md5-thor", required=True)
    ap.add_argument("--md5-local", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    fs = json.load(open(a.fit_set, encoding="utf-8"))

    def md5map(p):
        d = {}
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                m, f = ln.split(None, 1)
                d[os.path.basename(f.lstrip("*"))] = m
        return d
    th, lo = md5map(a.md5_thor), md5map(a.md5_local)
    names = [s + ".sam3mapgt.npz" for s in fs["sha12"]]
    bad = [n for n in names if not (len(th.get(n, "")) == 32 and th.get(n) == lo.get(n))]
    if bad:
        raise SystemExit(f"[prior] {len(bad)} fit files fail the Thor/local md5 check -- refusing")
    ext = EXTENT_REFCV7                   # 100 m x +-30 m, 1000 x 600 (SPEC_REFCV7 A7)
    pr = MH.PositionalPrior(extent=ext)
    t0 = time.time()
    for i, (s, n) in enumerate(zip(fs["sha12"], names)):
        with np.load(os.path.join(a.gt_dir, n), allow_pickle=False) as z:
            fc = z["fine_codes"]
        pr.add(s, fc)
        if i % 50 == 0:
            print(f"[prior] {i + 1}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
    pr.save(a.out)
    fp = pr.fingerprint()
    rec = {"tool": "prior_fit.py", "fit_set": a.fit_set, "fit_set_sha256": fs["sha12_list_sha256"],
           "n_files_md5_verified": len(names), "extent_m": [100.0, 30.0], **fp,
           "out": a.out, "out_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
           "wall_s": round(time.time() - t0, 1)}
    json.dump(rec, open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
