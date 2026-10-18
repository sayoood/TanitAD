"""Marginal log-cluster bootstrap interval for P2-s1 -- the one REFe arm on the navtest board that had NO level interval.

WHY THIS FILE EXISTS (2026-10-18, leaderboard currency for PDMS_v1_navtest)
    `MODEL_REGISTRY.md` section 14.2 (P2-s1 row of the marginal-PDMS table) records P2-s1's interval as "NOT COMPUTED":
    `raw/2026-10-17-paper-s720-recompute/marginal_ci.json` has rows for model_final, the route-fixed deployed system,
    SFT-4 B u400, P2-s0, the teacher and the human -- and none for P2-s1. The leaderboard prints a level interval beside
    every PDMS, so this script computes the missing one with the SAME estimator and writes a sibling artifact.

ESTIMATOR (copied from `marginal_ci.py`, which copies `zero_tail.Boot`; nothing is re-designed here)
    ratio-of-sums over the 136 navtest logs, B = 10,000 resamples, seed 20260927, percentile 2.5 / 97.5, tokens sorted,
    logs = np.unique(token_log[sorted tokens]), draws = default_rng(SEED).integers(0, n_logs, (B, n_logs)).
    It answers "would another draw of LOGS say this?" and nothing else -- ONE training seed, deterministic inference.

CONTROLS (the script refuses to write an interval whose controls do not read)
    C1  the token file's md5 equals the pin recorded in P2's r2_result.json (3a8b3653...), 12,146 tokens, 136 logs.
    C2  each input CSV's md5, after CRLF normalisation, equals the md5 pinned in its own MD5SUMS.txt
        (the git blob of the P2-s0 CSV is LF; the pin is the CRLF form the harness wrote).
    C3  P2-s0 is recomputed through the SAME code path and must equal the banked `marginal_ci.json` row `p2_s0`
        (point 88.8184, lo 87.8611, hi 89.7701) to all four printed decimals. A mismatch means this script is not the
        banked estimator, and the P2-s1 interval is then NOT written.
    C4  the P2-s1 point must equal the harness summary in `p2s1_full/result.json` (89.1493) at 4 dp.

Usage:  python p2s1_marginal_ci.py --repo <repo root> --tokens <navtest_full_tokens.json> --out <marginal_ci_p2s1.json>
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEED, B = 20260927, 10000
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TOKENS_MD5_PIN = "3a8b3653fd70891e50a230a37eb76ae8"
ESTIMATOR = "log-cluster bootstrap, ratio of sums, 136 logs, B=10000, seed 20260927 (zero_tail.Boot)"


def md5_crlf(path: Path) -> str:
    b = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.md5(b).hexdigest()


def read_scores(path: Path, tokens: list) -> np.ndarray:
    d = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            s = r["score"]
            d[r["token"]] = float("nan") if s in ("", "nan") else float(s)
    return np.array([d[t] for t in tokens], float)


def interval(x: np.ndarray, idx: np.ndarray, n_logs: int, draws: np.ndarray) -> dict:
    ok = ~np.isnan(x)
    s_n = np.bincount(idx, weights=np.where(ok, x, 0.0), minlength=n_logs)
    s_d = np.bincount(idx, weights=ok.astype(float), minlength=n_logs)
    point = s_n.sum() / s_d.sum() * 100
    v = s_n[draws].sum(axis=1) / s_d[draws].sum(axis=1) * 100
    return {"n": int(ok.sum()), "point": round(float(point), 4),
            "lo": round(float(np.percentile(v, 2.5)), 4), "hi": round(float(np.percentile(v, 97.5)), 4),
            "simple_mean": round(float(x[ok].mean() * 100), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, pk = Path(a.repo), Path(a.repo) / PK
    tok_path = Path(a.tokens)
    tok_md5 = hashlib.md5(tok_path.read_bytes()).hexdigest()
    assert tok_md5 == TOKENS_MD5_PIN, f"C1 FAILED: token file md5 {tok_md5} != pin {TOKENS_MD5_PIN}"
    tk = json.loads(tok_path.read_text(encoding="utf-8"))
    tokens = sorted(tk["tokens"])
    logs = np.array([tk["token_log"][t] for t in tokens])
    names, idx = np.unique(logs, return_inverse=True)
    n_logs = len(names)
    assert len(tokens) == 12146 and n_logs == 136, (len(tokens), n_logs)
    draws = np.random.default_rng(SEED).integers(0, n_logs, size=(B, n_logs))

    arms = {"p2_s0": pk / "raw/p2_readout/p2_full/arm", "p2_s1": pk / "raw/p2_readout/p2s1_full/arm"}
    csvname = {"p2_s0": "refe_stage2_p2_full.csv", "p2_s1": "refe_stage2_p2s1_full.csv"}
    inputs, rows = {}, {}
    for k, d in arms.items():
        p = d / csvname[k]
        pins = {ln.split()[-1].lstrip("*"): ln.split()[0] for ln in (d / "MD5SUMS.txt").read_text().splitlines() if ln.strip()}
        got = md5_crlf(p)
        assert got == pins[csvname[k]], f"C2 FAILED for {k}: CRLF-normalised md5 {got} != pinned {pins[csvname[k]]}"
        inputs[k] = {"csv": f"{PK}/raw/p2_readout/{'p2_full' if k == 'p2_s0' else 'p2s1_full'}/arm/{csvname[k]}",
                     "md5_pinned_in_MD5SUMS": pins[csvname[k]], "md5_crlf_normalised_equals_pin": True}
        rows[k] = interval(read_scores(p, tokens), idx, n_logs, draws)

    banked = json.loads((pk / "raw/2026-10-17-paper-s720-recompute/marginal_ci.json").read_text(encoding="utf-8"))
    b0 = banked["rows"]["p2_s0"]
    for f in ("n", "point", "lo", "hi", "simple_mean"):
        assert rows["p2_s0"][f] == b0[f], f"C3 FAILED: recomputed p2_s0 {f} {rows['p2_s0'][f]} != banked {b0[f]}"
    assert banked["estimator"] == ESTIMATOR, "C3 FAILED: the banked estimator string moved"
    res = json.loads((pk / "raw/2026-10-05-sft-stage2/p2s1_full/result.json").read_text(encoding="utf-8"))
    harness = res["harness_status"]["summary_x100_4dp"]["PDMS"]
    assert round(rows["p2_s1"]["point"], 4) == round(harness, 4), f"C4 FAILED: {rows['p2_s1']['point']} != harness {harness}"

    out = {"estimator": ESTIMATOR, "answers": "another draw of LOGS only (one training seed per arm)",
           "tier": "T1 NAVSIM v1.1 PDMS x100",
           "why": "marginal_ci.json (raw/2026-10-17-paper-s720-recompute) has no P2-s1 row; this file adds it with the identical estimator",
           "controls": {"C1_tokens_md5": tok_md5, "C1_pin": TOKENS_MD5_PIN, "C1_n_tokens": len(tokens), "C1_n_logs": n_logs,
                        "C2_inputs": inputs,
                        "C3_p2_s0_recomputed_equals_banked": {"recomputed": rows["p2_s0"], "banked": b0, "equal": True},
                        "C4_p2_s1_point_equals_harness_summary": {"point": rows["p2_s1"]["point"], "harness_x100_4dp": harness}},
           "rows": {"p2_s1": {"run": "refe_stage2_p2s1_full", **rows["p2_s1"]}}}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["rows"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
