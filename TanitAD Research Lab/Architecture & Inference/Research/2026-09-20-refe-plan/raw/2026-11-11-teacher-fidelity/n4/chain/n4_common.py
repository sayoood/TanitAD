"""N4-DAC shared helpers (navtrain held-out ONLY; never navtest). Loads the banked W-TRANSP-format score rows of the 469-token re-roll set
(arms R0, RJ3, RJ5, RJ5H5, RG8, R5HZ; RJ8 is bit-identical to RJ5 and excluded) and exposes the paired log-cluster bootstrap
(= tfn2_analyze.boot: B 10,000, seed 20260927, 95 % percentile, mean of token deltas resampled by log cluster)."""
import glob
import json
import os

import numpy as np

ROWS = os.environ.get("N4_ROWS", "F:/tanitad_scratch/n4dac/rows")
B, SEED = 10000, 20260927
RUNS = ("R0", "RJ3", "RJ5", "RJ5H5", "RG8", "R5HZ")
FORMS = ("R", "C", "T")
SUB = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")


def load_scores(rows=ROWS, runs=RUNS):
    out = {}
    for run in runs:
        d = {}
        for fp in sorted(glob.glob(f"{rows}/scores/{run}_s*.jsonl")):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                d[r["token"]] = r
        out[run] = d
    return out


def load_rolls(rows=ROWS, runs=RUNS):
    out = {}
    for run in runs:
        d = {}
        for fp in sorted(glob.glob(f"{rows}/rolls/{run}_s*.jsonl")):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "traj20" in r:
                    d[r["token"]] = r
        out[run] = d
    return out


def boot(vals, logs, paired=None, b=B, seed=SEED):
    """log-cluster bootstrap of the token mean of vals (or vals - paired) -> (point, lo, hi); identical to tfn2_analyze.boot."""
    v = np.asarray(vals, float) - (0.0 if paired is None else np.asarray(paired, float))
    ul = sorted(set(logs))
    idx = {l: np.array([i for i, x in enumerate(logs) if x == l]) for l in ul}
    s = np.array([v[idx[l]].sum() for l in ul])
    n = np.array([len(idx[l]) for l in ul])
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(ul), (b, len(ul)))
    m = s[d].sum(1) / n[d].sum(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
