#!/usr/bin/env python3
"""E3 A4 — the two-stage aggregation, cross-checked against the DEVKIT'S OWN CODE.

`taniteval/adapters/navsim_ci.py` re-derives NavSim's two-stage EPDMS aggregation
per mapping key. A re-derivation checked only against itself measures determinism,
not correctness ("a check that shares the defect it checks is green forever"). So
this script EXECUTES the devkit's own `calculate_weighted_average_score` and
`calculate_individual_mapping_scores` — their source text taken verbatim from the
pinned blob `navsim/planning/script/run_pdm_score.py@0a380a9` — on synthetic
per-token frames, and banks the devkit's outputs as LITERALS that
`taniteval/tests/test_navsim_ci.py` compares against.

Four fixtures, each chosen for the property it can falsify:
  F1  one key, hand-computed: the devkit must return 0.44 exactly (a literal
      derived on paper, not from either implementation).
  F2  40 keys / 10 logs, random, WITH the NaN cases the devkit handles implicitly
      (a NaN stage-2 score, a missing prev row, an all-zero-weight group, a
      duplicated synthetic token): per-column combined / stage-1 / stage-2 means.
  F3  every subset of F2 obtained by DROPPING whole logs: the devkit's aggregate on
      the subset must equal nanmean(our contributions restricted to the subset).
  F4  a BOOTSTRAP REPLICATE built physically — logs drawn with replacement, a
      repeated log's tokens RENAMED so the devkit sees two copies — the devkit's
      aggregate on that dataset must equal our resampled statistic. This is the
      property the log-cluster bootstrap relies on, checked on the devkit itself.

Run (TanitAD venv; the NavSim venv also works):
    python code/devkit_aggregation_crosscheck.py --out raw/devkit_aggregation_reference.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))
from adapters import navsim_ci as NC  # noqa: E402

DEVKIT = Path(r"C:/Users/Admin/navsim/devkit")
DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
RPS = "navsim/planning/script/run_pdm_score.py"
COLS = ["score", "ego_progress"]


def devkit_functions():
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(DEVKIT), "show",
                        f"{DEVKIT_SHA}:{RPS}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("cannot read the pinned run_pdm_score.py blob")
    src = r.stdout.decode("utf-8")
    mod = types.ModuleType("navsim_rps_pinned")
    sys.modules[mod.__name__] = mod
    import typing
    mod.__dict__.update({"np": np, "pd": pd, "Dict": typing.Dict, "List": typing.List,
                         "Tuple": typing.Tuple})
    prov = {}
    for name in ("calculate_weighted_average_score", "calculate_individual_mapping_scores"):
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        seg = ast.get_source_segment(src, node)
        exec(compile(seg, f"{RPS}@{DEVKIT_SHA[:7]}::{name}", "exec"), mod.__dict__)
        prov[name] = f"{RPS}@{DEVKIT_SHA[:7]} L{node.lineno}-{node.end_lineno}"
    prov["blob_sha256"] = hashlib.sha256(r.stdout).hexdigest()
    return mod, prov


def frame(rows: dict) -> pd.DataFrame:
    return pd.DataFrame([{"token": t, "weight": r["weight"], **{c: r[c] for c in COLS}}
                         for t, r in rows.items()])


def run_devkit(mod, rows, mapping):
    """The devkit call exactly as run_pdm_score.main makes it (L403-405)."""
    all_mappings = {(o, p): [tuple(x) for x in pairs] for o, p, pairs in mapping}
    g, s1, s2 = mod.calculate_individual_mapping_scores(frame(rows)[COLS + ["token", "weight"]],
                                                        all_mappings)
    return ({c: float(g[c]) for c in COLS}, {c: float(s1[c]) for c in COLS},
            {c: float(s2[c]) for c in COLS})


def ours(rows, mapping):
    return {c: NC.official_aggregate(NC.two_stage_key_contributions(rows, mapping, column=c))
            for c in COLS}


def _eq(a, b, tol=1e-12):
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------- #
def f1():
    """Hand-computed on paper (SPEC §4 worked example):
    s1(orig)=0.8, now-group scores [1.0, 0.5] weights [0.75, 0.25] -> 0.875;
    s1(prev)=0.6, prev-group scores [0.4, 0.2] weights [0.5, 0.5]  -> 0.3;
    key = (0.8*0.875 + 0.6*0.3)/2 = (0.7 + 0.18)/2 = 0.44."""
    rows = {"o": {"score": 0.8, "ego_progress": 0.8, "weight": 1.0},
            "p": {"score": 0.6, "ego_progress": 0.6, "weight": 1.0},
            "n1": {"score": 1.0, "ego_progress": 1.0, "weight": 0.75},
            "n2": {"score": 0.5, "ego_progress": 0.5, "weight": 0.25},
            "q1": {"score": 0.4, "ego_progress": 0.4, "weight": 0.5},
            "q2": {"score": 0.2, "ego_progress": 0.2, "weight": 0.5}}
    mapping = [("o", "p", [("n1", "q1"), ("n2", "q2")])]
    return rows, mapping, 0.44


def f2(seed=20260919, n_logs=10, keys_per_log=4):
    rng = np.random.default_rng(seed)
    rows, mapping, key_log = {}, [], {}
    for lg in range(n_logs):
        for k in range(keys_per_log):
            o, p = f"L{lg}K{k}o", f"L{lg}K{k}p"
            for t in (o, p):
                rows[t] = {"score": float(rng.uniform(0.2, 1.0)),
                           "ego_progress": float(rng.uniform(0.2, 1.0)), "weight": 1.0}
            n = int(rng.integers(3, 13))
            pairs = []
            wn = rng.gamma(1.0, size=n)
            wn = wn / wn.sum()
            wp = rng.gamma(1.0, size=n)
            wp = wp / wp.sum()
            for j in range(n):
                a, b = f"L{lg}K{k}s{j}a", f"L{lg}K{k}s{j}b"
                rows[a] = {"score": float(rng.uniform(0.0, 1.0)),
                           "ego_progress": float(rng.uniform(0.0, 1.0)), "weight": float(wn[j])}
                rows[b] = {"score": float(rng.uniform(0.0, 1.0)),
                           "ego_progress": float(rng.uniform(0.0, 1.0)), "weight": float(wp[j])}
                pairs.append((a, b))
            mapping.append((o, p, pairs))
            key_log[o] = f"log{lg}"
    # --- the NaN cases the devkit handles implicitly --------------------------- #
    rows["L1K0s0a"]["score"] = float("nan")          # a failed stage-2 scene
    del rows["L2K1p"]                                 # a missing prev row
    for (a, b) in mapping[3 * keys_per_log + 2][2]:   # an all-zero-weight 'now' group
        rows[a]["weight"] = 0.0
    m = mapping[5 * keys_per_log + 1]                 # a duplicated synthetic token
    m[2].append(m[2][0])
    return rows, mapping, key_log


def rename_copy(rows, mapping, key_log, logs_drawn):
    """A physical bootstrap replicate: each drawn log's keys and tokens copied under
    a draw-specific suffix, so a log drawn twice appears twice to the devkit."""
    new_rows, new_map, new_log = {}, [], {}
    for d, lg in enumerate(logs_drawn):
        suf = f"#{d}"
        for o, p, pairs in mapping:
            if key_log[o] != lg:
                continue
            toks = [o, p] + [t for pr in pairs for t in pr]
            for t in toks:
                if t in rows:
                    new_rows[t + suf] = dict(rows[t])
            new_map.append((o + suf, p + suf, [(a + suf, b + suf) for a, b in pairs]))
            new_log[o + suf] = lg
    return new_rows, new_map, new_log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mod, prov = devkit_functions()
    rep = {"_what": __doc__.split("\n")[0], "devkit_provenance": prov,
           "pandas": pd.__version__, "numpy": np.__version__,
           "python": sys.version.split()[0], "columns": COLS}

    # F1
    rows, mapping, lit = f1()
    g, s1, s2 = run_devkit(mod, rows, mapping)
    o = ours(rows, mapping)
    rep["F1"] = {"literal_on_paper": lit, "devkit_combined": g, "ours": o,
                 "devkit_equals_literal": abs(g["score"] - lit) < 1e-12,
                 "ours_equals_devkit": all(_eq(g[c], o[c]) for c in COLS)}

    # F2
    rows, mapping, key_log = f2()
    g, s1, s2 = run_devkit(mod, rows, mapping)
    o = ours(rows, mapping)
    contrib = {c: NC.two_stage_key_contributions(rows, mapping, column=c).tolist() for c in COLS}
    rep["F2"] = {"n_keys": len(mapping), "n_logs": len(set(key_log.values())),
                 "n_rows": len(rows),
                 "n_nan_contributions": {c: int(np.isnan(contrib[c]).sum()) for c in COLS},
                 "devkit_combined": g, "devkit_stage1": s1, "devkit_stage2": s2, "ours": o,
                 "ours_equals_devkit": all(_eq(g[c], o[c]) for c in COLS),
                 "inputs": {"rows": rows, "mapping": mapping, "key_log": key_log},
                 "ours_contributions": contrib}

    # F3 — dropping whole logs
    logs = sorted(set(key_log.values()))
    f3 = []
    rng = np.random.default_rng(7)
    for trial in range(12):
        keep = sorted(rng.choice(logs, size=int(rng.integers(2, len(logs))), replace=False).tolist())
        sub_map = [m for m in mapping if key_log[m[0]] in keep]
        gd, _, _ = run_devkit(mod, rows, sub_map)
        mask = np.array([key_log[m[0]] in keep for m in mapping])
        mine = {c: NC.official_aggregate(np.asarray(contrib[c])[mask]) for c in COLS}
        f3.append({"logs_kept": keep, "devkit": gd, "ours": mine,
                   "equal": all(_eq(gd[c], mine[c]) for c in COLS)})
    rep["F3"] = {"n_trials": len(f3), "n_equal": sum(t["equal"] for t in f3), "trials": f3}

    # F4 — physical bootstrap replicates (logs drawn WITH replacement)
    f4 = []
    for trial in range(12):
        drawn = rng.choice(logs, size=len(logs), replace=True).tolist()
        r2, m2, _ = rename_copy(rows, mapping, key_log, drawn)
        gd, _, _ = run_devkit(mod, r2, m2)
        # our resampled statistic: nanmean over the drawn logs' contributions, with multiplicity
        sel = np.concatenate([np.flatnonzero(np.array([key_log[m[0]] == lg for m in mapping]))
                              for lg in drawn])
        mine = {c: NC.official_aggregate(np.asarray(contrib[c])[sel]) for c in COLS}
        f4.append({"logs_drawn": drawn, "n_distinct": len(set(drawn)), "devkit": gd,
                   "ours": mine, "equal": all(_eq(gd[c], mine[c]) for c in COLS)})
    rep["F4"] = {"n_trials": len(f4), "n_equal": sum(t["equal"] for t in f4), "trials": f4}

    ok = (rep["F1"]["devkit_equals_literal"] and rep["F1"]["ours_equals_devkit"]
          and rep["F2"]["ours_equals_devkit"] and rep["F3"]["n_equal"] == rep["F3"]["n_trials"]
          and rep["F4"]["n_equal"] == rep["F4"]["n_trials"])
    rep["ALL_AGREE"] = bool(ok)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps({"F1": {k: rep["F1"][k] for k in ("devkit_combined", "ours",
                                                       "devkit_equals_literal",
                                                       "ours_equals_devkit")},
                      "F2": {k: rep["F2"][k] for k in ("devkit_combined", "ours",
                                                       "n_nan_contributions",
                                                       "ours_equals_devkit")},
                      "F3": f"{rep['F3']['n_equal']}/{rep['F3']['n_trials']}",
                      "F4": f"{rep['F4']['n_equal']}/{rep['F4']['n_trials']}",
                      "ALL_AGREE": rep["ALL_AGREE"], "pandas": pd.__version__},
                     indent=1, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
