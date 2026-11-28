#!/usr/bin/env python3
"""NAVTEST controls (each can FAIL). No PDMS, no aggregate.
  nt_ctrl.py smoke [--n 2]   controls on the first n logs' files found in navtest/  -> navtest/ctrl_smoke.json
  nt_ctrl.py full            controls on all 136 logs, requires 12,146 tokens, writes navtest/index.csv (token,log,split,file,sha256) -> navtest/ctrl_full.json
CTRL-PICK-FULL: my pick == anchor table pick (bar: 0 mismatches, 0 missing). CTRL-LOGIT: score_head(s) from the checkpoint's head == cached logits (<= 1e-4, AMENDMENT NT-1; was 1e-5).
CTRL-SELECT: navsim_v1 aggregate argmax of cached logits == pick (the contract's validate()). CTRL-SHAPE: shapes, finite, pick in 0..63, log/split scalars."""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "F:/safehead_cache/code")
import sh_lib as L  # noqa: E402

ND = L.wp("navtest")
PL = ND + "/plan"


def agg32(Lg):
    Lg = np.asarray(Lg, np.float32)
    p = 1.0 / (1.0 + np.exp(-Lg))
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5.0 * p[..., 3] + 2 * p[..., 4]) / (7 + 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    job = json.load(open(PL + "/job_navtest.json"))
    anchor = json.load(open(PL + "/anchor_pick.json"))
    logs = [e["log"] for e in job["logs"]]
    want = logs[:a.n] if a.mode == "smoke" else logs
    import torch
    sd = torch.load(L.C1_CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
    Wt, Bt = sd["score_head.weight"].float(), sd["score_head.bias"].float()
    rows, ntok, mism, missing, wl, bad_sel, shape_bad, nexp = [], 0, [], [], 0.0, 0, [], 0
    for lg in want:
        p = f"{ND}/{lg}.npz"
        exp_toks = [e for e in job["logs"] if e["log"] == lg][0]["tokens"]
        nexp += len(exp_toks)
        if not os.path.exists(p):
            missing += exp_toks
            shape_bad.append(f"{lg}: file missing")
            continue
        d = np.load(p, allow_pickle=False)
        tk = [str(t) for t in d["token"]]
        n = len(tk)
        ok = (d["s"].shape == (n, 64, 256) and d["logits"].shape == (n, 64, 6) and d["props"].ndim == 4 and d["props"].shape[:2] == (n, 64)
              and all(np.isfinite(d[k]).all() for k in ("s", "logits", "props")) and d["pick"].min() >= 0 and d["pick"].max() < 64
              and str(d["log"]) == lg and str(d["split"]) == "navtest" and d["s"].dtype == np.float32 and d["logits"].dtype == np.float32)
        if not ok:
            shape_bad.append(lg)
        have = set(tk)
        missing += [t for t in exp_toks if t not in have]
        for i, t in enumerate(tk):
            if int(d["pick"][i]) != anchor[t]:
                mism.append(t)
        with torch.no_grad():
            y = (torch.from_numpy(d["s"]).float() @ Wt.T + Bt).numpy()
        wl = max(wl, float(np.abs(y - d["logits"]).max()))
        bad_sel += int((agg32(d["logits"]).argmax(-1) != d["pick"]).sum())
        ntok += n
        sha = L.sha256_file(p)
        rows += [(t, lg, "navtest", os.path.basename(p), sha) for t in tk]
    res = {}
    res["CTRL-PICK-FULL"] = {"pass": not mism and not missing and ntok > 0, "n_tokens_checked": ntok, "n_expected": nexp, "mismatches": len(mism), "missing": len(missing),
                             "first_mismatch": mism[:5], "bar": "0 mismatches vs anchor full_C1_s{0,1}of2.jsonl pick, 0 missing"}
    res["CTRL-LOGIT"] = {"pass": wl <= 1e-4 and ntok > 0, "max_abs_diff": wl, "bar": "<= 1e-4 (s float32; AMENDMENT NT-1 2026-11-27: = PREREG_SAFEHEAD CTRL-FEAT, the binding bar; the old 1e-5 sat inside float32 accumulation noise, 8 ulp at |logit| 19.8)", "old_bar_1e-5_pass": wl <= 1e-5}
    res["CTRL-SELECT"] = {"pass": bad_sel == 0 and ntok > 0, "n_pick_ne_argmax_agg32": bad_sel}
    res["CTRL-SHAPE"] = {"pass": not shape_bad, "problems": shape_bad[:10]}
    res["COVERAGE"] = {"pass": a.mode == "smoke" or ntok == 12146, "mode": a.mode, "logs": len(want), "tokens": ntok}
    res["overall_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    out = ND + ("/ctrl_smoke.json" if a.mode == "smoke" else "/ctrl_full.json")
    json.dump(res, open(out, "w"), indent=1)
    if a.mode == "full" and res["overall_pass"]:
        with open(ND + "/index.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["token", "log", "split", "file", "sha256"])
            w.writerows(rows)
    print("ZZNTCTRL " + json.dumps(res)[:1500], flush=True)
    sys.exit(0 if res["overall_pass"] else 1)


main()
