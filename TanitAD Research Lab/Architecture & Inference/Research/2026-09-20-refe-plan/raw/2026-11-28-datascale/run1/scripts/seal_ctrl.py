"""DATA-SCALE seal controls (PREREG_DATASCALE.md 2.2 / 6), MM 2026-11-28. Read-only on F:/safehead_cache; writes only --out.

  CTRL-INDEX  index_185 holds EXACTLY the parent's 2,272 tokens (same token, log, split, sha256) plus x150 rows; no log or token twice;
              every row's file re-hashes to its sha256 (build_index_185.py already checks; re-checked here on the parent rows).
  CTRL-LOGIT  C1's score_head on each x150 npz's cached `s` reproduces its cached `logits`. GATE <= 1e-4 (the card's CTRL-FEAT bar; the
              same alignment as navtest AMENDMENT NT-1, fixed here BEFORE any x150 logit is compared). The old 1e-5 bar of sh_ctrl.py is
              recorded as descriptive (`old_bar_1e-5_pass`), never as the gate.
usage: python seal_ctrl.py --index185 <run>/index_185.csv --parent F:/safehead_cache/index.csv --ckpt F:/safehead_cache/ckpt/C1/model_final.pt --out <json>
"""
import argparse
import csv
import hashlib
import json
import sys

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--index185", required=True)
ap.add_argument("--parent", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

new = list(csv.DictReader(open(a.index185, encoding="utf-8")))
par = list(csv.DictReader(open(a.parent, encoding="utf-8")))
res = {"n_rows": len(new), "n_parent": len(par)}
key = lambda r: (r["token"], r["log"], r["split"], r["sha256"])
pk = {key(r) for r in par}
nk = {key(r) for r in new}
toks = [r["token"] for r in new]
log_src = {}
for r in new:
    log_src.setdefault(r["log"], set()).add(r["file"].replace("\\", "/").rsplit("/out/", 1)[0])
res["CTRL-INDEX"] = {
    "parent_rows_all_present_bit_identical": pk <= nk,
    "parent_rows_missing": len(pk - nk),
    "duplicate_tokens": len(toks) - len(set(toks)),
    "logs_in_two_roots": sorted(l for l, s in log_src.items() if len(s) > 1),
    "n_logs": len(log_src),
    "n_heldout_logs": len({r["log"] for r in new if r["split"] == "heldout"}),
    "n_new_rows": len(nk - pk),
}
ci = res["CTRL-INDEX"]
ci["pass"] = ci["parent_rows_all_present_bit_identical"] and ci["duplicate_tokens"] == 0 and not ci["logs_in_two_roots"] and ci["n_new_rows"] == len(new) - len(par)

import torch  # noqa: E402  (after the cheap checks)

torch.set_num_threads(2)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)["model"]
W = sd["score_head.weight"].float()
B = sd["score_head.bias"].float()
worst, n, files = 0.0, 0, 0
x150_files = sorted({r["file"] for r in new if "/x150/out/" in r["file"].replace("\\", "/")})
for p in x150_files:
    d = np.load(p)
    s = torch.from_numpy(np.asarray(d["s"])).float()
    with torch.no_grad():
        y = (s @ W.T + B).numpy()
    worst = max(worst, float(np.abs(y - np.asarray(d["logits"])).max()))
    n += len(d["token"])
    files += 1
res["CTRL-LOGIT"] = {"pass": worst <= 1e-4, "bar": "<= 1e-4 (card CTRL-FEAT; as navtest AMENDMENT NT-1)", "max_abs_diff": worst,
                     "old_bar_1e-5_pass": worst <= 1e-5, "n_tokens": n, "n_files": files}
res["pass"] = ci["pass"] and res["CTRL-LOGIT"]["pass"]
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print("ZZSEALCTRL", "PASS" if res["pass"] else "FAIL", json.dumps({k: v for k, v in res.items() if k != "pass"})[:700])
sys.exit(0 if res["pass"] else 1)
