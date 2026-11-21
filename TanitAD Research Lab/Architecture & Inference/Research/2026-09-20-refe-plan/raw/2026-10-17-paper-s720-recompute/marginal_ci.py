"""Marginal log-cluster bootstrap intervals for the REFe ladder rows (read-only; writes marginal_ci.json).

Estimator: identical to zero_tail.Boot (refe-plan/raw/2026-10-17-zero-tail-mechanism/code/zero_tail.py):
ratio-of-sums over 136 navtest logs, B = 10,000 resamples, seed 20260927, percentile 2.5 / 97.5.
Controls (must read the banked values or the script's numbers are not quotable):
  human  94.55 [93.80, 95.24]   (RESULT_P4_TEACHER_NAVTEST.md C2)
  teacher 89.79 [88.55, 90.97]  (same file, all 12,146 tokens)
  u400 mean 88.2469, P2-s0 88.8184, P2-s1 89.1493 (RESULT_P2REP.md)
Inputs: harness CSVs under D:/Projects/TanitAD/data/refe_navtest/score/<run>/<run>.csv and
D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json (token -> log).
"""
import json, sys
import numpy as np

SEED, B = 20260927, 10000
S = "D:/Projects/TanitAD/data/refe_navtest/score"
TOK = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
RUNS = {
    "model_final": "refe_navtest_final",            # unfixed; may be absent
    "model_final_routefix_A9": "refe_navtest_final_routefix",
    "sft4_B_u400": "refe_stage2_sft4_B_u400_navtest_full",
    "p2_s0": "refe_stage2_p2_full",
    "teacher_driverl": "refe_p4_teacher_navtest",
    "human": "refe_p4_human_navtest",
}


def load(run, tokens):
    import csv, os
    p = f"{S}/{run}/{run}.csv"
    if not os.path.exists(p):
        return None
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["token"]] = float(r["score"]) if r["score"] not in ("", "nan") else float("nan")
    return np.array([d[t] for t in tokens], float)


def main():
    tk = json.load(open(TOK, encoding="utf-8"))
    tokens = sorted(tk["tokens"])
    logs = np.array([tk["token_log"][t] for t in tokens])
    names, idx = np.unique(logs, return_inverse=True)
    n_logs = len(names)
    assert len(tokens) == 12146 and n_logs == 136, (len(tokens), n_logs)
    draws = np.random.default_rng(SEED).integers(0, n_logs, size=(B, n_logs))
    ones = np.ones(len(tokens))
    s_d = np.bincount(idx, weights=ones, minlength=n_logs)
    out = {"estimator": "log-cluster bootstrap, ratio of sums, 136 logs, B=10000, seed 20260927 (zero_tail.Boot)",
           "answers": "another draw of LOGS only (one training seed per arm)", "tier": "T1 NAVSIM v1.1 PDMS x100",
           "rows": {}}
    for name, run in RUNS.items():
        x = load(run, tokens)
        if x is None:
            out["rows"][name] = {"run": run, "status": "CSV ABSENT on this box"}
            continue
        ok = ~np.isnan(x)
        s_n = np.bincount(idx, weights=np.where(ok, x, 0.0), minlength=n_logs)
        s_dd = np.bincount(idx, weights=ok.astype(float), minlength=n_logs)
        point = s_n.sum() / s_dd.sum() * 100
        v = s_n[draws].sum(axis=1) / s_dd[draws].sum(axis=1) * 100
        out["rows"][name] = {"run": run, "n": int(ok.sum()), "point": round(float(point), 4),
                             "lo": round(float(np.percentile(v, 2.5)), 4), "hi": round(float(np.percentile(v, 97.5)), 4),
                             "simple_mean": round(float(x[ok].mean() * 100), 4)}
    json.dump(out, open("C:/Users/Admin/paper_pkg/marginal_ci.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
