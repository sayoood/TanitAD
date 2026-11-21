#!/usr/bin/env python3
"""K6 (PREREG_NT_DECIDE sec. 6): the estimator reads its known values.

The REGISTERED pairing (``navsim_bars8.pair_record`` -> ``navsim_ci.paired_log_cluster_bootstrap``, single-stage, B 2000,
seed 0, cluster ``log_name``, sensitivity ``nuplan_drive``), on a real per-token score table:

* SELF-PAIR: arm A against ITSELF on the identical tokens -> delta EXACTLY 0.0 and the interval EXACTLY [0, 0];
* TOKEN-SHUFFLED PAIRING (the join defect, deliberately reintroduced): A against A with its per-token scores permuted
  across tokens -> the delta stays 0 (same multiset) but the interval WIDENS (lo < 0 < hi). An estimator that read
  [0, 0] here would be blind to a mis-joined pair.

Input: any score CSV of the registered sample (default: the banked A0 = theta_DEC R8_LEGAL seed-0 read,
``wiseft/scores/r8sel_wmod2_DEC_R8_LEGAL/*.csv``) + the sample file's ``token_log`` map. 0 GPU; seconds of CPU.
For A1 the same call runs on A1's own snapshot CSVs (and on the navtest CSV after the PI's go)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEF_BARS = "C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/pkg/code/navsim_bars8.py"
DEF_CI = str(HERE.parent / "tree_fb47a58/taniteval/adapters/navsim_ci.py")
DEF_CSV = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/scores/r8sel_wmod2_DEC_R8_LEGAL/r8sel_wmod2_DEC_R8_LEGAL.csv"
DEF_SAMPLE = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run/sample/NAVTRAIN_SAMPLE_TOKENS.json"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bars", default=DEF_BARS)
    ap.add_argument("--ci", default=DEF_CI)
    ap.add_argument("--csv", default=DEF_CSV)
    ap.add_argument("--sample", default=DEF_SAMPLE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["NAVSIM_CI_PATH"] = a.ci
    spec = importlib.util.spec_from_file_location("navsim_bars8_k6", a.bars)
    B = importlib.util.module_from_spec(spec)
    sys.modules["navsim_bars8_k6"] = B
    spec.loader.exec_module(B)
    df = B.read_scores_csv(a.csv)
    smp = json.load(open(a.sample, encoding="utf-8"))
    tl = smp["token_log"]
    toks = sorted(t for t in df.index if t in tl)
    clusters = [str(tl[t]) for t in toks]
    cfg = dict(B.DEFAULT_CFG)
    self_pair = B.pair_record(df, df, toks, clusters, cfg)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(toks))
    shuf = df.loc[toks].copy()
    shuf.loc[:, :] = shuf.to_numpy()[perm]
    shuf_pair = B.pair_record(df, shuf, toks, clusters, cfg)
    sp, sh = self_pair["interval"], shuf_pair["interval"]
    ok_self = (sp.get("status") == "OK" and self_pair["delta_x100"] == 0.0 and sp["lo"] == 0.0 and sp["hi"] == 0.0)
    ok_shuf = (sh.get("status") == "OK" and sh["lo"] < 0.0 < sh["hi"])
    rec = {"control": "K6-estimator", "PASS": bool(ok_self and ok_shuf),
           "self_pair": self_pair, "token_shuffled_pair": shuf_pair,
           "n_tokens": len(toks), "n_logs": len(set(clusters)),
           "inputs": {"csv": [a.csv, _sha(a.csv)], "sample": [a.sample, _sha(a.sample)],
                      "bars": [a.bars, _sha(a.bars)], "ci": [a.ci, _sha(a.ci)]},
           "estimator": {"cfg": {k: cfg[k] for k in ("n_boot", "seed")}, "cluster": "log_name",
                         "sensitivity": "nuplan_drive", "aggregation": "AGG_SINGLE_STAGE"},
           "tool": {"nt_k6.py": hashlib.md5(Path(__file__).read_bytes()).hexdigest()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"PASS": rec["PASS"], "n": len(toks), "logs": rec["n_logs"],
                      "self": {k: sp.get(k) for k in ("delta", "lo", "hi", "n_clusters")},
                      "shuffled": {k: sh.get(k) for k in ("delta", "lo", "hi")}}))
    return 0 if rec["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
