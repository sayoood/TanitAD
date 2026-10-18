#!/usr/bin/env python3
"""The anatomy of P2's zero tail on the full navtest (T1, NAVSIM v1.1). CPU only, banked data only, NO model.

Pre-registration: PREREG.md in this directory (written before any count was computed; Amendment 0 corrects two provenance facts).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe zero_tail.py [--out RESULT.json]

WHY A PROXY.  P2 has NO all-64 per-proposal dump (its seam keeps ONE pose per token). So SEL / PROP is MEASURED on two arms that do
have an all-64 NAVSIM table (u400: C3's table; the pre-SFT final model: lane_census) and INFERRED for P2 through a pre-registered
bridge. Every number in RESULT.json says which of the two it is.

Definitions (PREREG section 3): zero token = score == 0.0; SEL = some of the 64 proposals has PDMS > 0; PROP = all 64 == 0.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parents[1]                      # .../2026-09-20-refe-plan
RAW = PK / "raw"

B_BOOT = 10000
SEED = 20260927
M = 64
V0_BINS = [(0.0, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 14.0), (14.0, 99.0)]
CMD_NAMES = {0: "left", 1: "straight", 2: "right", 3: "unknown"}
CAUSES = ("DAC_only", "NC_only", "both", "neither")
PROP_KINDS = ("all_DAC0", "all_NC0", "all_both", "mixed", "other")

DEFAULTS = {
    "p2_csv": "D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv",
    "p2_csv_repo": RAW / "p2_readout/p2_full/arm/refe_stage2_p2_full.csv",
    "u400_csv_repo": RAW / "p2_r2_instrument/baseline/u400_navtest_full.csv",
    "tokens": RAW / "p2_r2_instrument/baseline/navtest_full_tokens.json",
    "p2_seam": RAW / "p2_readout/p2_full/arm/stage2_p2_full.npz",
    "c3_table": RAW / "2026-10-11-c3-selection-headroom/table.npz",
    "c3_tokens": RAW / "2026-10-11-c3-selection-headroom/tokens.json",
    "c3_props": RAW / "2026-10-11-c3-selection-headroom/proposals.npz",
    "census": RAW / "2026-10-04-lane-discipline/lane_census.jsonl.xz",
    "inputs_export": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
}
EXPECT = {"p2_mean_x100": 88.82, "p2_mean_tol": 0.01, "p2_zero": 509, "u400_mean_x100": 88.25, "u400_zero": 529,
          "final_zero_reviewed": 983, "n_tokens": 12146, "n_logs": 136,
          "p2_csv_sha256": "61073a61e44a3dcf41fad6dc8c5b261ce97eb8f8d8b49c1adc077dd7c04b61e6"}


# ================================================================== pure functions (unit-tested)
def sel_any(pdms: np.ndarray) -> np.ndarray:
    """SEL: at least one of the 64 proposals scores PDMS > 0.  pdms: [n, 64]."""
    return (pdms > 0).any(axis=1)


def cause_labels(nc: np.ndarray, dac: np.ndarray) -> np.ndarray:
    """Which gate zeroes a token at its pick. 'DAC_only' = DAC 0 and NC non-zero; 'NC_only' = NC 0 and DAC non-zero."""
    return np.where((nc == 0) & (dac == 0), "both",
                    np.where(dac == 0, "DAC_only", np.where(nc == 0, "NC_only", "neither")))


def fan_stats(pdms, nc, dac, ttc, sel_fn=sel_any) -> dict:
    """Everything the anatomy needs from an all-64 table. All arrays [n, 64]."""
    sel = sel_fn(pdms)
    ok_ttc = (ttc == 1) & (pdms > 0)
    all_dac0 = (dac == 0).all(axis=1)
    all_nc0 = (nc == 0).all(axis=1)
    other = ((nc * dac > 0) & (pdms == 0)).any(axis=1)
    kind = np.where(other, "other", np.where(all_dac0 & all_nc0, "all_both", np.where(all_dac0, "all_DAC0",
                    np.where(all_nc0, "all_NC0", "mixed"))))
    return {"sel": sel,
            "best": pdms.max(axis=1),
            "n_nonzero": (pdms > 0).sum(axis=1),
            "safe": ((nc == 1) & (dac == 1)).any(axis=1),
            "sel_ttc": ok_ttc.any(axis=1),
            "best_ttcok": np.where(ok_ttc, pdms, -1.0).max(axis=1),
            "prop_kind": kind}


def verdict(n_sel: int, n_zero: int, gain_pts: float) -> str:
    """The committed reading (PREREG section 6), integer-exact on the share. SELECTION is evaluated first."""
    if n_zero <= 0:
        return "NO_ZERO_TOKENS"
    if 2 * n_sel >= n_zero and gain_pts >= 2.0:
        return "SELECTION"
    if 2 * (n_zero - n_sel) >= n_zero:
        return "PROPOSAL"
    return "MIXED"


def k50(counts: np.ndarray) -> int:
    """Smallest number of logs whose zero counts sum to >= ceil(total / 2)."""
    total = int(counts.sum())
    if total == 0:
        return 0
    cum = np.cumsum(np.sort(counts)[::-1])
    return int(np.searchsorted(cum, (total + 1) // 2, side="left") + 1)


class Boot:
    """Log-cluster bootstrap of ratio estimators sum(num) / sum(den); every call reuses the SAME log draws (paired by construction)."""

    def __init__(self, log_idx: np.ndarray, n_logs: int, B: int = B_BOOT, seed: int = SEED):
        self.log_idx, self.n_logs, self.B = log_idx, n_logs, B
        self.draws = np.random.default_rng(seed).integers(0, n_logs, size=(B, n_logs))

    def ratio(self, num, den, scale: float = 1.0) -> dict:
        num = np.asarray(num, float)
        den = np.asarray(den, float)
        s_n = np.bincount(self.log_idx, weights=num, minlength=self.n_logs)
        s_d = np.bincount(self.log_idx, weights=den, minlength=self.n_logs)
        d_all = s_d.sum()
        point = float(s_n.sum() / d_all * scale) if d_all > 0 else float("nan")
        bd = s_d[self.draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(bd > 0, s_n[self.draws].sum(axis=1) / bd * scale, np.nan)
        ok = ~np.isnan(v)
        if not ok.any():
            return {"point": point, "lo": float("nan"), "hi": float("nan"), "B": self.B, "n_nan": int((~ok).sum())}
        return {"point": point, "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)),
                "B": self.B, "n_nan": int((~ok).sum())}


def concentration(zero: np.ndarray, log_idx: np.ndarray, n_logs: int, log_names: np.ndarray) -> dict:
    cnt = np.bincount(log_idx[zero], minlength=n_logs)
    ntok = np.bincount(log_idx, minlength=n_logs)
    order = np.argsort(-cnt, kind="stable")          # logs are indexed in sorted-name order, so ties break by name
    top = [{"log": str(log_names[j]), "zero": int(cnt[j]), "tokens": int(ntok[j]),
            "zero_rate": float(cnt[j] / ntok[j])} for j in order[:10]]
    return {"n_zero": int(cnt.sum()), "n_logs_with_zero": int((cnt > 0).sum()), "k50": k50(cnt),
            "top10": top, "top10_share_of_zeros": float(cnt[order[:10]].sum() / max(cnt.sum(), 1)), "counts": cnt}


def permutation_null(n_tokens: int, n_zero: int, log_idx: np.ndarray, n_logs: int, B: int = B_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    ks, tops = np.empty(B, int), np.empty(B, float)
    for b in range(B):
        idx = rng.choice(n_tokens, size=n_zero, replace=False)
        c = np.bincount(log_idx[idx], minlength=n_logs)
        ks[b] = k50(c)
        tops[b] = np.sort(c)[::-1][:10].sum() / n_zero
    return {"B": B, "k50_median": float(np.median(ks)), "k50_lo": float(np.percentile(ks, 2.5)), "k50_hi": float(np.percentile(ks, 97.5)),
            "top10_share_median": float(np.median(tops)), "top10_share_lo": float(np.percentile(tops, 2.5)),
            "top10_share_hi": float(np.percentile(tops, 97.5)), "ks": ks}


def embed(n_total: int, idx: np.ndarray, vals: np.ndarray, fill):
    out = np.full((n_total,) + vals.shape[1:], fill, dtype=vals.dtype)
    out[idx] = vals
    return out


def cleanliness(zero, sel, n_nonzero) -> dict:
    """POST-HOC descriptive: how many of the 64 proposals are non-zero on the SEL zero tokens. A SEL token with ONE non-zero proposal in 64
    is SEL by the definition and almost unselectable in practice, so the oracle gain is a ceiling, not a target."""
    k = n_nonzero[zero & sel]
    if len(k) == 0:
        return {"n": 0}
    return {"n": int(len(k)), "exactly_1_of_64": int((k == 1).sum()), "2_to_4_of_64": int(((k >= 2) & (k <= 4)).sum()),
            "5_to_31_of_64": int(((k >= 5) & (k <= 31)).sum()), "32_or_more_of_64": int((k >= 32).sum()),
            "median_nonzero_of_64": float(np.median(k)), "mean_nonzero_share": float(k.mean() / M)}


def q1_block(zero, sel, best, boot, n_total) -> dict:
    """Counts exact; share and gain carry the log-cluster bootstrap."""
    z = zero.astype(float)
    n_zero = int(zero.sum())
    n_sel = int((zero & sel).sum())
    gain = boot.ratio(np.where(zero & sel, best, 0.0), np.ones(n_total), scale=100.0)
    return {"n_zero": n_zero, "n_SEL": n_sel, "n_PROP": n_zero - n_sel,
            "SEL_share": boot.ratio((zero & sel).astype(float), z),
            "PROP_share": boot.ratio((zero & ~sel).astype(float), z),
            "oracle_on_SEL_gain_pdms_pts": gain,
            "verdict": verdict(n_sel, n_zero, gain["point"])}


def cause_block(zero, cause, sel, boot) -> dict:
    z = zero.astype(float)
    out = {"all": {c: {"n": int((zero & (cause == c)).sum()), "share": boot.ratio((zero & (cause == c)).astype(float), z)} for c in CAUSES}}
    for name, m in (("SEL", sel), ("PROP", ~sel)):
        zz = zero & m
        out[name] = {c: {"n": int((zz & (cause == c)).sum())} for c in CAUSES}
        out[name]["n_total"] = int(zz.sum())
    return out


def ttc_block(ttc_tok, sel_ttc, best_ttcok, pick_pdms, boot, n_total) -> dict:
    """TTC tokens: score > 0 and TTC == 0 at the pick. headroom = max(0, best TTC-ok proposal - pick) summed over SEL_TTC tokens / N."""
    t = ttc_tok.astype(float)
    n_t = int(ttc_tok.sum())
    n_s = int((ttc_tok & sel_ttc).sum())
    head = np.where(ttc_tok & sel_ttc, np.maximum(0.0, best_ttcok - pick_pdms), 0.0)
    return {"n_TTC_tokens": n_t, "n_SEL_TTC": n_s, "n_PROP_TTC": n_t - n_s,
            "SEL_TTC_share": boot.ratio((ttc_tok & sel_ttc).astype(float), t),
            "headroom_pdms_pts": boot.ratio(head, np.ones(n_total), scale=100.0)}


# ================================================================== loaders
def sha256_of(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_scores(path, tokens) -> dict:
    rows, avg = {}, None
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["token"] == "average":
                avg = float(r["score"])
                continue
            rows[r["token"]] = r
    if set(rows) != set(tokens):
        raise AssertionError(f"token set differs: csv {len(rows)} vs canonical {len(tokens)}")

    def col(k):
        return np.array([float(rows[t][k]) for t in tokens])

    return {"nc": col("no_at_fault_collisions"), "dac": col("drivable_area_compliance"), "ep": col("ego_progress"),
            "ttc": col("time_to_collision_within_bound"), "c": col("comfort"), "ddc": col("driving_direction_compliance"),
            "score": col("score"), "average": avg, "valid": np.array([rows[t]["valid"] == "True" for t in tokens])}


def load_census(path, tokens) -> dict:
    pos = {t: i for i, t in enumerate(tokens)}
    n = len(tokens)
    A = {k: np.full((n, M), np.nan) for k in ("pdms", "nc", "dac", "ttc", "c")}
    pick = np.full(n, -1, int)
    csv_pdms = np.full(n, np.nan)
    pick_ok = np.zeros(n, bool)
    seen = 0
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = pos.get(r["token"])
            if i is None:
                continue
            seen += 1
            for k in A:
                A[k][i] = np.asarray(r[k][1:M + 1], float)
            pick[i], csv_pdms[i], pick_ok[i] = r["pick"], r["csv_pdms"], bool(r["pick_ok"])
    A.update({"pick": pick, "csv_pdms": csv_pdms, "pick_ok": pick_ok, "seen": seen})
    return A


# ================================================================== the run
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=str(v))
    ap.add_argument("--out", default=str(HERE / "RESULT.json"))
    a = ap.parse_args()
    P = {k: getattr(a, k) for k in DEFAULTS}
    p2_csv = P["p2_csv"] if os.path.exists(P["p2_csv"]) else str(P["p2_csv_repo"])
    gates, R = {}, {"meta": {}}

    # ---- canonical token order + logs
    tk = json.load(open(P["tokens"], encoding="utf-8"))
    tokens = sorted(tk["tokens"])
    N = len(tokens)
    pos = {t: i for i, t in enumerate(tokens)}
    logs = np.array([tk["token_log"][t] for t in tokens])
    log_names, log_idx = np.unique(logs, return_inverse=True)
    n_logs = len(log_names)
    gates["G2_tokens_logs"] = {"n_tokens": N, "n_logs": n_logs, "pass": (N == EXPECT["n_tokens"] and n_logs == EXPECT["n_logs"])}
    boot = Boot(log_idx, n_logs)

    # ---- P2 and u400 per-token scores
    p2 = load_scores(p2_csv, tokens)
    u4 = load_scores(P["u400_csv_repo"], tokens)
    p2_mean, u4_mean = float(p2["score"].mean() * 100), float(u4["score"].mean() * 100)
    gates["G1_p2_mean"] = {"recomputed_x100": p2_mean, "expected": EXPECT["p2_mean_x100"], "csv_average_row_x100": p2["average"] * 100,
                           "pass": bool(abs(p2_mean - EXPECT["p2_mean_x100"]) <= EXPECT["p2_mean_tol"] and abs(p2_mean - p2["average"] * 100) < 1e-6)}
    gates["G2_u400_mean"] = {"recomputed_x100": u4_mean, "expected": EXPECT["u400_mean_x100"], "pass": bool(abs(u4_mean - EXPECT["u400_mean_x100"]) <= 0.01)}
    p2z, u4z = p2["score"] == 0.0, u4["score"] == 0.0
    gates["G3_zero_counts"] = {"p2": int(p2z.sum()), "p2_expected": EXPECT["p2_zero"], "u400": int(u4z.sum()), "u400_expected": EXPECT["u400_zero"],
                               "pass": bool(p2z.sum() == EXPECT["p2_zero"] and u4z.sum() == EXPECT["u400_zero"])}
    R["meta"]["p2_csv"] = {"path": p2_csv, "sha256": sha256_of(p2_csv), "expected_sha256_original": EXPECT["p2_csv_sha256"],
                           "is_the_banked_original": sha256_of(p2_csv) == EXPECT["p2_csv_sha256"]}

    # ---- u400 all-64 table (C3)
    T = np.load(P["c3_table"], allow_pickle=True)
    c3 = json.load(open(P["c3_tokens"], encoding="utf-8"))
    t_tok = [str(x) for x in T["token"]]
    t_idx = np.array([pos[t] for t in t_tok])
    sub = T["sub"].astype(float)                       # [m, 64, 6]: nc dac ep ttc c ddc
    names = [str(x) for x in T["sub_names"]]
    assert names[0].startswith("no_at_fault") and names[1].startswith("drivable") and names[3].startswith("time_to_collision")
    u_pdms = T["pdms"].astype(float)
    pick_t = T["pick"].astype(int)
    at_pick = u_pdms[np.arange(len(t_idx)), pick_t]
    d_g4 = float(np.max(np.abs(at_pick - u4["score"][t_idx])))
    gates["G4_u400_table_at_pick"] = {"max_abs_diff": d_g4, "n": int(len(t_idx)), "all_valid": bool(T["valid"].all()), "pass": bool(d_g4 <= 1e-9 and T["valid"].all())}
    fs_u = fan_stats(u_pdms, sub[..., 0], sub[..., 1], sub[..., 3])
    stratum = np.array([c3["stratum"][t] for t in t_tok])
    in_u = np.zeros(N, bool)
    in_u[t_idx] = True
    U = {k: embed(N, t_idx, v, {"sel": False, "best": 0.0, "safe": False, "sel_ttc": False, "best_ttcok": -1.0, "prop_kind": "", "n_nonzero": 0}[k]) for k, v in fs_u.items()}
    U["sel"] = U["sel"].astype(bool)
    U["prop_kind"] = np.array([x if x else "" for x in U["prop_kind"]])

    # ---- final-model census (all 12,146)
    C = load_census(P["census"], tokens)
    csv_diff = float(np.nanmax(np.abs(C["csv_pdms"] - C["pdms"][np.arange(N), C["pick"]])))
    pk_p = C["pdms"][np.arange(N), C["pick"]]
    status_ok = bool(((C["csv_pdms"] == 0) == (pk_p == 0)).all())
    n_hidden = int(((C["nc"] * C["dac"] > 0) & (C["pdms"] == 0)).sum())    # a tiny positive PDMS rounded to 0 would sit here
    gates["G5_census"] = {"records": int(C["seen"]), "max_abs_pick_vs_csv_pdms": csv_diff, "tolerance": 5.0e-5 + 1e-9,
                          "zero_status_agrees": status_ok, "proposals_ncdac_pos_but_pdms0": n_hidden, "pick_ok_all": bool(C["pick_ok"].all()),
                          "pass": bool(C["seen"] == N and csv_diff <= 5.0e-5 + 1e-9 and status_ok and n_hidden == 0)}
    fs_c = fan_stats(C["pdms"], C["nc"], C["dac"], C["ttc"])
    fz = C["csv_pdms"] == 0.0
    nc_f = C["nc"][np.arange(N), C["pick"]]
    dac_f = C["dac"][np.arange(N), C["pick"]]
    ttc_f = C["ttc"][np.arange(N), C["pick"]]
    f_score = C["csv_pdms"]
    R["meta"]["final_census"] = {"zero_picks": int(fz.sum()), "zero_picks_reviewed_in_REVIEW_7": EXPECT["final_zero_reviewed"],
                                 "mean_x100": float(f_score.mean() * 100), "note": "pre-route-fix final model; informational check, not a hard gate"}
    R["gates"] = gates
    if not all(g["pass"] for g in gates.values()):
        R["STOP"] = "a gate failed; no result is written"
        json.dump(R, open(a.out, "w"), indent=1, default=str)
        print(json.dumps(gates, indent=1, default=str))
        sys.exit(2)

    ones = np.ones(N)
    # ================= Q2 at the pick (all arms) and the DDC test
    p2_cause, u4_cause = cause_labels(p2["nc"], p2["dac"]), cause_labels(u4["nc"], u4["dac"])
    f_cause = cause_labels(nc_f, dac_f)
    R["ddc_gate_test"] = {"p2_rows_ddc0_and_score_gt0": int(((p2["ddc"] == 0) & (p2["score"] > 0)).sum()),
                          "p2_rows_ddc0": int((p2["ddc"] == 0).sum()),
                          "reading": "> 0 means DDC does not gate the score (NAVSIM v1.1 driving_direction_weight = 0.0)"}

    # ================= Q1a  u400 on its own zero tokens (MEASURED)
    assert (in_u & u4z).sum() == u4z.sum(), "u400 table must contain every u400 zero token"
    R["Q1a_u400_own"] = {"evidence": "MEASURED on u400 (table.npz, all 529 zero tokens are in the table)",
                         "SEL_def": q1_block(u4z, U["sel"], U["best"], boot, N),
                         "SAFE_sensitivity": {"n_SAFE": int((u4z & U["safe"]).sum()), "n_zero": int(u4z.sum())},
                         "Q2_cause": cause_block(u4z, u4_cause, U["sel"], boot),
                         "fan_cleanliness_POSTHOC": cleanliness(u4z, U["sel"], U["n_nonzero"]),
                         "PROP_failure_kinds": {k: int((u4z & ~U["sel"] & (U["prop_kind"] == k)).sum()) for k in PROP_KINDS}}

    # ================= Q1b  final model on its own zero tokens (MEASURED)
    R["Q1b_final_own"] = {"evidence": "MEASURED on the pre-SFT final model (lane_census, all 12,146 tokens)",
                          "SEL_def": q1_block(fz, fs_c["sel"], fs_c["best"], boot, N),
                          "SAFE_sensitivity": {"n_SAFE": int((fz & fs_c["safe"]).sum()), "n_zero": int(fz.sum())},
                          "Q2_cause": cause_block(fz, f_cause, fs_c["sel"], boot),
                          "fan_cleanliness_POSTHOC": cleanliness(fz, fs_c["sel"], fs_c["n_nonzero"]),
                          "PROP_failure_kinds": {k: int((fz & ~fs_c["sel"] & (fs_c["prop_kind"] == k)).sum()) for k in PROP_KINDS}}

    # ================= Q1c  P2 through the bridge (INFERRED)
    tier = np.where(in_u, 1, 2)                       # the census covers every token (G5)
    sel_v1 = np.where(in_u, U["sel"], fs_c["sel"])
    best_v1 = np.where(in_u, U["best"], fs_c["best"])
    sel_v2, best_v2 = fs_c["sel"], fs_c["best"]
    sel_lo, best_lo = np.where(in_u, U["sel"], False), np.where(in_u, U["best"], 0.0)
    sel_hi, best_hi = np.where(in_u, U["sel"], True), np.where(in_u, U["best"], fs_c["best"])
    var = {"V1_hierarchy": (sel_v1, best_v1), "V2_final_census_only": (sel_v2, best_v2),
           "V3_lo_u400_uncovered_as_PROP": (sel_lo, best_lo), "V3_hi_u400_uncovered_as_SEL": (sel_hi, best_hi)}
    n_p2z = int(p2z.sum())
    tier1 = int((p2z & in_u).sum())
    strength = "strong" if tier1 / n_p2z >= 0.75 else ("partial" if tier1 / n_p2z >= 0.5 else "weak")
    both_cov = in_u                                    # census covers all, so the overlap is the u400 table
    t1b_all = float((U["sel"][both_cov] == fs_c["sel"][both_cov]).mean())
    zz = both_cov & u4z
    t1b_zero = float((U["sel"][zz] == fs_c["sel"][zz]).mean())
    cont = {f"u400_{'SEL' if a1 else 'PROP'}__final_{'SEL' if b1 else 'PROP'}": int((both_cov & (U["sel"] == a1) & (fs_c["sel"] == b1)).sum())
            for a1 in (True, False) for b1 in (True, False)}
    cont_z = {f"u400_{'SEL' if a1 else 'PROP'}__final_{'SEL' if b1 else 'PROP'}": int((zz & (U["sel"] == a1) & (fs_c["sel"] == b1)).sum())
              for a1 in (True, False) for b1 in (True, False)}
    # T2: P2's pick against u400's 64 proposals on the 1,058 C3 tokens
    S = np.load(P["p2_seam"], allow_pickle=True)
    s_pos = {str(t): i for i, t in enumerate(S["token"])}
    Pr = np.load(P["c3_props"], allow_pickle=True)
    assert [str(x) for x in Pr["token"]] == t_tok
    p2_pose = np.stack([S["poses"][s_pos[t]] for t in t_tok]).astype(float)
    dmin = np.abs(Pr["proposals"].astype(float) - p2_pose[:, None]).max(axis=(2, 3)).min(axis=1)
    in_zero_stratum = stratum == "zero"
    R["bridge"] = {
        "evidence": "INFERRED: P2 has no all-64 table; classification comes from other models' fans",
        "tier1_coverage": {"P2_zero_in_u400_table": tier1, "P2_zero": n_p2z, "share": tier1 / n_p2z, "strength": strength},
        "persistence": {"P2z_and_u400z": int((p2z & u4z).sum()), "P2z_only": int((p2z & ~u4z).sum()), "u400z_only": int((u4z & ~p2z).sum()),
                        "P2z_and_finalz": int((p2z & fz).sum()), "share_of_P2z_also_u400z": float((p2z & u4z).sum() / n_p2z),
                        "share_of_P2z_also_finalz": float((p2z & fz).sum() / n_p2z)},
        "T1b_label_agreement_u400_table_vs_final_census": {"tokens_both": int(both_cov.sum()), "agreement_all": t1b_all,
                                                           "agreement_on_u400_zero_tokens": t1b_zero, "contingency_all": cont, "contingency_u400_zero": cont_z},
        "T2_P2_pick_vs_u400_fan_on_C3_tokens": {"n": int(len(dmin)), "share_within_1e-3_m": float((dmin <= 1e-3).mean()),
                                                "share_within_0.5_m": float((dmin <= 0.5).mean()), "median_min_dist_m": float(np.median(dmin)),
                                                "on_u400_zero_tokens": {"n": int(in_zero_stratum.sum()), "share_within_1e-3_m": float((dmin[in_zero_stratum] <= 1e-3).mean())}},
    }
    safe_v1 = np.where(in_u, U["safe"], fs_c["safe"])
    p2v, rob = {}, {}
    for name, (s, b) in var.items():
        blk = {"SEL_def": q1_block(p2z, s, b, boot, N),
               "Q2_cause": cause_block(p2z, p2_cause, s, boot),
               "gain_by_cause_pdms_pts": {c: boot.ratio(np.where(p2z & s & (p2_cause == c), b, 0.0), ones, scale=100.0) for c in CAUSES[:3]}}
        if name == "V1_hierarchy":
            blk["SAFE_sensitivity_n"] = int((p2z & safe_v1).sum())
            blk["fan_cleanliness_POSTHOC"] = cleanliness(p2z, s, np.where(in_u, U["n_nonzero"], fs_c["n_nonzero"]))
        if name == "V2_final_census_only":
            blk["SAFE_sensitivity_n"] = int((p2z & fs_c["safe"]).sum())
        pk = np.where(in_u, U["prop_kind"], fs_c["prop_kind"])
        blk["PROP_failure_kinds"] = {k: int((p2z & ~s & (pk == k)).sum()) for k in PROP_KINDS} if name in ("V1_hierarchy", "V2_final_census_only") else None
        if name == "V1_hierarchy":
            blk["by_tier"] = {f"tier{t}": {"n": int((p2z & (tier == t)).sum()), "n_SEL": int((p2z & (tier == t) & s).sum())} for t in (1, 2)}
        p2v[name] = blk
        rob[name] = blk["SEL_def"]["verdict"]
    R["Q1c_P2_bridged"] = {"evidence": "INFERRED from proxy fans (never MEASURED on P2)", "variants": p2v, "verdicts": rob,
                           "robust": len(set(rob.values())) == 1,
                           "reading_if_bridge_weak": "UNDETERMINED from banked data" if strength == "weak" else None}
    # POST-HOC descriptive detail (not pre-registered; it explains the bridge, it decides nothing)
    pers = p2z & u4z
    p2_only = p2z & ~u4z
    R["bridge"]["persistence_detail_POSTHOC"] = {
        "persistent_zeros": int(pers.sum()),
        "persistent_SEL_under_u400_fan": int((pers & U["sel"]).sum()),
        "persistent_PROP_under_u400_fan": int((pers & ~U["sel"]).sum()),
        "P2_only_zeros": int(p2_only.sum()),
        "P2_only_zeros_in_u400_table": int((p2_only & in_u).sum()),
        "P2_only_zeros_where_u400_PICK_scored_gt0": int((p2_only & (u4["score"] > 0)).sum()),
        "reading": "on a P2-only zero token u400's own pick scored > 0, so u400's fan contains a non-zero plan BY CONSTRUCTION (MEASURED on u400); "
                   "that it transfers to P2's fan is the bridge assumption (T2 reports how close the fans are)",
        "u400_only_zeros_rescued_by_P2": int((u4z & ~p2z).sum()),
    }

    # ================= Q2 at P2's pick (MEASURED, pick level)
    R["Q2_P2_pick_level"] = {"evidence": "MEASURED on P2 (per-token csv)",
                             "cause": {c: {"n": int((p2z & (p2_cause == c)).sum()), "share": boot.ratio((p2z & (p2_cause == c)).astype(float), p2z.astype(float))} for c in CAUSES},
                             "NC_values_among_zero": {str(v): int((p2z & (p2["nc"] == v)).sum()) for v in (0.0, 0.5, 1.0)},
                             "u400_cause": {c: int((u4z & (u4_cause == c)).sum()) for c in CAUSES},
                             "final_cause": {c: int((fz & (f_cause == c)).sum()) for c in CAUSES}}

    # ================= Q3 concentration (+ tags)
    conc_p2 = concentration(p2z, log_idx, n_logs, log_names)
    conc_u4 = concentration(u4z, log_idx, n_logs, log_names)
    null = permutation_null(N, n_p2z, log_idx, n_logs)
    top_p2 = {x["log"] for x in conc_p2["top10"]}
    top_u4 = {x["log"] for x in conc_u4["top10"]}
    obs_cnt = conc_p2["counts"]
    R["Q3_concentration"] = {
        "evidence": "MEASURED (counts exact); null = permutation of the zero labels, B=10000, seed 20260927",
        "P2": {k: v for k, v in conc_p2.items() if k != "counts"},
        "u400": {k: v for k, v in conc_u4.items() if k != "counts"},
        "null_k50": {k: v for k, v in null.items() if k != "ks"},
        "P2_k50_percentile_in_null": float((null["ks"] <= conc_p2["k50"]).mean()),
        "top10_overlap_P2_u400": len(top_p2 & top_u4),
        "logs_total": n_logs,
        "logs_with_zero_P2": conc_p2["n_logs_with_zero"],
    }
    tags = {"available": False}
    if os.path.exists(P["inputs_export"]):
        exp = json.load(gzip.open(P["inputs_export"], "rt", encoding="utf-8"))["tokens"]
        miss = [t for t in tokens if t not in exp]
        if not miss:
            v0 = np.array([float(np.hypot(*exp[t]["ego_statuses"][-1]["ego_velocity"])) for t in tokens])
            cmd = np.array([int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in tokens])
            tags = {"available": True, "source": P["inputs_export"], "sha256": sha256_of(P["inputs_export"]),
                    "evidence": "MEASURED; tags come from the navtest input export joined by token, NOT from the P2 dump",
                    "by_command": {CMD_NAMES[c]: {"tokens": int((cmd == c).sum()), "P2_zero": int((p2z & (cmd == c)).sum()),
                                                  "P2_zero_rate": boot.ratio((p2z & (cmd == c)).astype(float), (cmd == c).astype(float)),
                                                  "u400_zero": int((u4z & (cmd == c)).sum())} for c in CMD_NAMES if (cmd == c).any()},
                    "by_v0_mps": {f"[{lo},{hi})": {"tokens": int(((v0 >= lo) & (v0 < hi)).sum()), "P2_zero": int((p2z & (v0 >= lo) & (v0 < hi)).sum()),
                                                   "P2_zero_rate": boot.ratio((p2z & (v0 >= lo) & (v0 < hi)).astype(float), ((v0 >= lo) & (v0 < hi)).astype(float)),
                                                   "u400_zero": int((u4z & (v0 >= lo) & (v0 < hi)).sum())} for lo, hi in V0_BINS}}
        else:
            tags = {"available": False, "reason": f"{len(miss)} tokens missing from the export"}
    R["Q3_tags"] = tags

    # ================= Q4 TTC
    p2_ttc_tok = (p2["score"] > 0) & (p2["ttc"] == 0)
    u4_ttc_tok = (u4["score"] > 0) & (u4["ttc"] == 0)
    f_ttc_tok = (f_score > 0) & (ttc_f == 0)
    sel_ttc_v1 = np.where(in_u, U["sel_ttc"], fs_c["sel_ttc"])
    best_ttc_v1 = np.where(in_u, U["best_ttcok"], fs_c["best_ttcok"])
    q4 = {"evidence": "see per-block labels",
          "P2_pick_level": {"evidence": "MEASURED on P2", "n_TTC_tokens": int(p2_ttc_tok.sum()),
                            "also_zeroed_and_TTC0": int((p2z & (p2["ttc"] == 0)).sum()),
                            "TTC_token_rate": boot.ratio(p2_ttc_tok.astype(float), ones)},
          "P2_bridged_V1": {"evidence": "INFERRED", **ttc_block(p2_ttc_tok, sel_ttc_v1, best_ttc_v1, p2["score"], boot, N),
                            "by_tier": {f"tier{t}": {"n": int((p2_ttc_tok & (tier == t)).sum()), "n_SEL_TTC": int((p2_ttc_tok & (tier == t) & sel_ttc_v1).sum())} for t in (1, 2)}},
          "P2_bridged_V2_final_census": {"evidence": "INFERRED", **ttc_block(p2_ttc_tok, fs_c["sel_ttc"], fs_c["best_ttcok"], p2["score"], boot, N)},
          "final_own": {"evidence": "MEASURED on the final model (full census)", **ttc_block(f_ttc_tok, fs_c["sel_ttc"], fs_c["best_ttcok"], f_score, boot, N),
                        "also_zeroed_and_TTC0": int((fz & (ttc_f == 0)).sum())}}
    # u400: TTC tokens are all in the non-zeroed stratum; the C3 random stratum is a uniform 529 of 11,617
    rs = np.zeros(N, bool)
    rs[t_idx[stratum == "random"]] = True
    n_zero_u = int(u4z.sum())
    n_nonzero_u = N - n_zero_u
    samp_tt = rs & u4_ttc_tok
    sel_s = float((samp_tt & U["sel_ttc"]).sum())
    head_s = np.where(samp_tt & U["sel_ttc"], np.maximum(0.0, U["best_ttcok"] - u4["score"]), 0.0)
    q4["u400_own"] = {"evidence": "ESTIMATED: the SEL_TTC share comes from the C3 random stratum (a uniform 529 of 11,617 non-zeroed tokens); the TTC token COUNT is exact from the full csv",
                      "n_TTC_tokens_exact": int(u4_ttc_tok.sum()), "n_TTC_tokens_in_sample": int(samp_tt.sum()),
                      "n_SEL_TTC_in_sample": int(sel_s),
                      "SEL_TTC_share": boot.ratio((samp_tt & U["sel_ttc"]).astype(float), samp_tt.astype(float)),
                      "headroom_pdms_pts": boot.ratio(head_s, rs.astype(float), scale=100.0 * n_nonzero_u / N),
                      "also_zeroed_and_TTC0_exact": int((u4z & (u4["ttc"] == 0)).sum())}
    R["Q4_TTC"] = q4

    # ================= context: how big the zero tail is against the gap (DESCRIPTIVE arithmetic on the P2 csv)
    nz = ~p2z
    mean_nz = float(p2["score"][nz].mean() * 100)
    R["context"] = {"evidence": "DESCRIPTIVE arithmetic on the P2 csv; 94.55 is PUBLISHED (DriveZero Table A13, per the brief)",
                    "P2_pdms_x100": p2_mean, "paper_bar": 94.55, "gap_to_paper": 94.55 - p2_mean,
                    "zero_rate": float(p2z.mean()), "mean_pdms_of_nonzero_tokens_x100": mean_nz,
                    "zero_tail_cost_pts": float(p2z.sum() / N * mean_nz),
                    "cost_def": "zero tokens at the mean of the non-zero tokens instead of 0: n_zero / N * mean_nonzero",
                    "u400_zero_tail_cost_pts": float(u4z.sum() / N * float(u4["score"][~u4z].mean() * 100)),
                    "proposal_side_ceiling_pts_ESTIMATED": float(p2v["V1_hierarchy"]["SEL_def"]["n_PROP"] / N * mean_nz),
                    "proposal_side_ceiling_def": "V1 n_PROP / N * mean non-zero PDMS: what fixing every PROP token to a typical score would add; an ESTIMATE, not a measurement",
                    "selection_side_ceiling_pts_INFERRED": p2v["V1_hierarchy"]["SEL_def"]["oracle_on_SEL_gain_pdms_pts"]["point"]}

    # ================= the committed reading, assembled
    r_u = R["Q1a_u400_own"]["SEL_def"]
    r_f = R["Q1b_final_own"]["SEL_def"]
    R["reading"] = {
        "u400_MEASURED": {"verdict": r_u["verdict"], "SEL_share": r_u["SEL_share"]["point"], "gain": r_u["oracle_on_SEL_gain_pdms_pts"]["point"]},
        "final_MEASURED": {"verdict": r_f["verdict"], "SEL_share": r_f["SEL_share"]["point"], "gain": r_f["oracle_on_SEL_gain_pdms_pts"]["point"]},
        "P2_INFERRED": {"V1_verdict": rob["V1_hierarchy"], "variants": rob, "robust_across_variants": R["Q1c_P2_bridged"]["robust"],
                        "bridge_strength": strength,
                        "V1_SEL_share": p2v["V1_hierarchy"]["SEL_def"]["SEL_share"], "V1_gain_pts": p2v["V1_hierarchy"]["SEL_def"]["oracle_on_SEL_gain_pdms_pts"],
                        "T1b_agreement_ok_ge_80pct": bool(t1b_all >= 0.8 and t1b_zero >= 0.8)},
    }
    R["meta"].update({"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1), "python": platform.python_version(),
                      "numpy": np.__version__, "bootstrap": {"B": B_BOOT, "seed": SEED, "n_logs": n_logs, "estimand": "another draw of LOGS only"},
                      "inputs": {k: {"path": str(v), "sha256": sha256_of(v)} for k, v in P.items() if k not in ("inputs_export", "p2_csv", "p2_csv_repo") and os.path.exists(str(v))},
                      "tier": "T1"})
    json.dump(R, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s)")
    print(json.dumps({"gates": gates, "reading": R["reading"]}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
