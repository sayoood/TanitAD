"""Assemble gates.json for `sh5_train_eval.py analyze` from the EVIDENCE files (a gate is never hand-written; it is read from the artifact that proves it - the launch-gate lesson).

    python sh5_gates.py --cid-dir R_cid --glive R/g_live_A1_s0.json R/g_live_A1_s1.json --ctx-manifest C/ctx_manifest.json --tests tests_summary.json --out gates.json

  c_id       the E=0 pre-flight (`fit --stage A1 --epochs 0`, both seeds) wrote OOF logits that are EXACTLY 0.0 on every set (the zero-initialised branch emits nothing), and the 2-parameter stacker returns
             (1, 0) on them, so L+S == L bit for bit (re-checked here from the arrays, not from a flag)
  g_live     every G-LIVE json handed in says ok (finite terms, every trainable group has gradient, the frozen-ctx_ln zero control reads exactly 0.0)
  c_fid      the ctx build's fidelity gate passed (>= 99 % of sets reproduce the census logits to <= 1e-3)
  red_table  the decision table passes AND every RED mutation test ran green (tests_summary.json from run_tests.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402


def check_cid(cid_dir: str, lab_path: str = None, inputs_path: str = None) -> dict:
    out = {"ok": False, "seeds": {}}
    for s in L.SEEDS:
        p = f"{cid_dir}/OOF_S_s{s}.npz"
        if not os.path.exists(p):
            out["seeds"][f"s{s}"] = {"present": False}
            return out
        meta = json.load(open(p.replace(".npz", ".json"), encoding="utf-8"))
        z = np.load(p)["z"]
        exact_zero = bool(np.isfinite(z).all() and np.count_nonzero(z) == 0)
        zl = np.random.default_rng(s).normal(size=z.shape).astype(np.float32)           # any L: the identity must hold for an arbitrary P2 logit
        y = (np.random.default_rng(s + 7).random(z.shape) < 0.5).astype(np.float32)
        fold = np.arange(len(z)) % L.FOLDS
        comb, par = L.oof_stack(zl, z, y, fold)
        out["seeds"][f"s{s}"] = {"present": True, "epochs": meta.get("epochs"), "exact_zero": exact_zero, "stacker_identity": bool(np.array_equal(comb, zl)), "n_sets": int(len(z))}
    out["ok"] = all(v.get("present") and v["epochs"] == 0 and v["exact_zero"] and v["stacker_identity"] for v in out["seeds"].values()) and len(out["seeds"]) == len(L.SEEDS)
    return out


def build(cid_dir: str, glive_files: list, ctx_manifest: str, tests_summary: str) -> dict:
    cid = check_cid(cid_dir)
    gl = [json.load(open(f, encoding="utf-8")) for f in glive_files]
    man = json.load(open(ctx_manifest, encoding="utf-8"))
    ts = json.load(open(tests_summary, encoding="utf-8"))
    return {"c_id": bool(cid["ok"]), "g_live": bool(gl) and all(g.get("ok") for g in gl), "c_fid": bool(man.get("c_fid", {}).get("pass")),
            "red_table": bool(ts.get("all_passed")) and L.branch_table_failures() == [], "evidence": {"c_id": cid, "g_live_files": glive_files, "c_fid": man.get("c_fid"), "tests": {k: v for k, v in ts.items() if k != "log"}}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid-dir", required=True)
    ap.add_argument("--glive", nargs="+", required=True)
    ap.add_argument("--ctx-manifest", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g = build(a.cid_dir, a.glive, a.ctx_manifest, a.tests)
    json.dump(g, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: g[k] for k in ("c_id", "g_live", "c_fid", "red_table")}))
    return 0 if all(g[k] for k in ("c_id", "g_live", "c_fid", "red_table")) else 1


if __name__ == "__main__":
    sys.exit(main())
