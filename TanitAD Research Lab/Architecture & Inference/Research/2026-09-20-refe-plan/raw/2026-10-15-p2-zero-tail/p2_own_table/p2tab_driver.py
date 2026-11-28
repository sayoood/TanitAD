#!/usr/bin/env python3
"""P2 OWN 64-PROPOSAL TABLE -- the driver.  Turns `D-REFE-ZEROTAIL-1`'s P2 row from INFERRED into MEASURED.

    --tag smoke8   CPU, no GPU lock: dump 8 of the 509 P2 zero tokens, gate the pick against P2's banked seam, score the 8 pick columns through
                   the W3 harness and require table-at-pick == the banked P2 per-token row.  The SAME-PICK PROOF on a smoke.  -> _smoke8/SMOKE8_PROOF.json
    --tag main     dump (GPU, behind the shared dev-box lock) -> fidelity gate (STOPS before any CPU scoring if P2's pick is not reproduced)
                   -> 64 harness columns (4 workers) -> table (G2) -> SAME-PICK PROOF (strict) -> P2_MEASURED.json -> D: backup
                   Markers in this directory: P2TABLE_DONE or P2TABLE_FAILED (with the traceback).

The measurement pipeline is `c3_run_p2.py` (= the C3 `c3_run.py` re-pointed at P2; see c3_run_vs_p2.diff).  The SEL / PROP classifier, the cause split and
the log-cluster bootstrap are IMPORTED from the landed `../zero_tail.py` (ZT.fan_stats, ZT.q1_block, ZT.cause_block, ZT.Boot ...), not re-implemented.

Evidence class of every number written: MEASURED on P2 (own 64-proposal table of its 509 zero tokens), tier T1 (NAVSIM v1.1 pseudo-simulation of an
open-loop plan).  Estimator: log-cluster bootstrap over the 136 navtest logs, B 10,000, seed 20260927 -- it answers "another draw of LOGS" only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZT_DIR = os.path.dirname(HERE)                       # .../raw/2026-10-15-p2-zero-tail  (zero_tail.py, RESULT.json)
sys.path.insert(0, HERE)
sys.path.insert(0, ZT_DIR)
import c3_run_p2 as C   # noqa: E402
import zero_tail as ZT  # noqa: E402

POSE_TOL_M = C.FIDELITY_TOL_M                        # 1e-3 m, the pre-registered C3 fidelity bar
PDMS_TOL = 5.0e-5 + 1e-9                             # the 4-decimal rounding quantum (the same tolerance PREREG Amendment 1 uses)
BACKUP = "D:/Projects/TanitAD-artifacts/p2_zero_tail_table"
SMOKE_TOKENS = os.path.join(HERE, "tokens_p2zero_smoke8.json")
SMOKE_POSE_TOL_M = 5e-3                              # CPU-vs-GPU float noise bar for the smoke only (measured 1.2e-4 .. 1.4e-3 on the 8)
SMOKE_MARGIN = 100.0                                 # the nearest other proposal must be >= 100x farther from the banked pick than the dump's pick is


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def run_c3(argv):
    sys.argv = ["c3_run_p2.py"] + [str(x) for x in argv]
    return C.main()


# ================================================================================ the same-pick proof
def banked():
    """P2's banked seam (pick poses) and per-token csv rows (valid, score, six sub-scores)."""
    B = np.load(C.BANKED_SEAM)
    bidx = {str(t): i for i, t in enumerate(B["token"])}
    return B, bidx, C.read_csv(C.BANKED_CSV)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def pick_from_logits(logits):
    """planner.REFePlanner.aggregate, rule navsim_v1: sigmoid, NC x DAC x (5 EP + 5 TTC + 2 C) / 12, argmax.  INFORMATIONAL cross-check only."""
    p = sig(logits)
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
    return agg.argmax(-1)


def same_pick_proof(wd, out_json):
    """Re-derive P2's own pick from the table and require it to EQUAL P2's banked pick.  Three independent equalities per token:
       (1) POSE   the dump's picked proposal == the banked P2 seam row (max |d| <= 1e-3 m; n bit-identical reported)
       (2) SCORE  the table's PDMS at the pick == the banked P2 csv score (<= 5e-5 = the 4-decimal rounding quantum; n exact reported), AND every token
                  reads EXACTLY 0.0 there (these are P2's zero tokens)
       (3) SUB    the six sub-scores at the pick == the banked csv's six sub-scores (<= 5e-5)
       plus (informational) argmax of the navsim_v1 aggregate of the dumped logits == the planner's pick."""
    D = np.load(os.path.join(wd, "proposals.npz"))
    T = np.load(os.path.join(wd, "table.npz"), allow_pickle=False)
    tok = [str(t) for t in T["token"]]
    assert tok == [str(t) for t in D["token"]], "table token order differs from the dump"
    N = len(tok)
    P, pick = D["proposals"], D["pick"].astype(int)
    B, bidx, lr = banked()
    ban = B["poses"][[bidx[t] for t in tok]].astype(np.float64)
    sel = P[np.arange(N), pick].astype(np.float64)
    dpose = np.abs(sel - ban).reshape(N, -1).max(1)
    pd = T["pdms"][np.arange(N), pick]
    csv_score = np.array([lr[t][1] for t in tok])
    dsc = np.abs(pd - csv_score)
    sub_at = T["sub"][np.arange(N), pick]                                      # [N, 6]
    sub_csv = np.array([lr[t][2] for t in tok])
    dsub = np.abs(sub_at - sub_csv).max(1)
    pl = pick_from_logits(D["logits"])
    want = set(json.load(open(os.path.join(HERE, "tokens_p2zero.json"), encoding="utf-8"))["tokens"]) if wd == HERE else set(tok)
    worst = np.argsort(-np.maximum(dpose, dsc))[:5]
    R = {"evidence": "MEASURED on P2 (dump + harness through the C3 pipeline); tier T1",
         "n_tokens": N, "token_set_equals_expected": bool(set(tok) == want), "table_all_valid": bool(T["valid"].all()),
         "pose": {"max_abs_m": float(dpose.max()), "n_bit_identical": int((dpose == 0).sum()), "n_gt_1e-4": int((dpose > 1e-4).sum()),
                  "n_gt_tol": int((dpose > POSE_TOL_M).sum()), "tol_m": POSE_TOL_M},
         "pdms_at_pick_vs_banked_csv": {"max_abs": float(dsc.max()), "n_exact": int((dsc == 0).sum()), "n_gt_tol": int((dsc > PDMS_TOL).sum()),
                                        "tol": PDMS_TOL, "n_table_pdms_at_pick_exactly_0": int((pd == 0.0).sum()),
                                        "n_banked_csv_exactly_0": int((csv_score == 0.0).sum())},
         "subscores_at_pick_vs_banked_csv": {"max_abs": float(dsub.max()), "n_exact": int((dsub == 0).sum()), "n_gt_tol": int((dsub > PDMS_TOL).sum())},
         "pick_from_logits_informational": {"n_equal_to_planner_pick": int((pl == pick).sum()), "n": N,
                                            "what": "argmax of sigmoid(logits) navsim_v1 aggregate (float64, numpy) vs the planner's torch float32 pick; not gating"},
         "distinct_pick_indices": int(len(set(pick.tolist()))),
         "worst_tokens": [(tok[i], float(dpose[i]), float(dsc[i])) for i in worst]}
    R["PASS"] = bool(R["token_set_equals_expected"] and R["table_all_valid"] and dpose.max() <= POSE_TOL_M and dsc.max() <= PDMS_TOL
                     and (pd == 0.0).all() and (csv_score == 0.0).all() and dsub.max() <= PDMS_TOL)
    R["reading"] = ("P2's own pick is REPRODUCED: the table belongs to P2" if R["PASS"] else
                    "NOT the same pick -- the checkpoint, rule or seam is wrong; no SEL/PROP number from this table is admissible")
    jdump(R, out_json)
    return R


def smoke_proof(wd):
    """CPU smoke: the pick column of each of the 8 tokens, scored through the harness, against the banked P2 row.  CPU vs the banked GPU pick can differ
    by up to ~4e-4 m (C3 plumbing measured 4.1e-4), so the pose bar is the pre-registered 1e-3 m and the score bar is the rounding quantum."""
    ctx = C.Ctx(argparse.Namespace(tag="smoke8", tokens=SMOKE_TOKENS, m=C.M, only=None, device="cpu", workers=4))
    D = np.load(ctx.dump)
    tok = [str(t) for t in D["token"]]
    pick = D["pick"].astype(int)
    P = D["proposals"]
    B, bidx, lr = banked()
    ban = B["poses"][[bidx[t] for t in tok]].astype(np.float64)
    dall = np.abs(P.astype(np.float64) - ban[:, None]).reshape(len(tok), P.shape[1], -1).max(2)        # [n, 64]: every proposal vs the banked pick
    dpose = dall[np.arange(len(tok)), pick]
    d2 = np.sort(np.where(np.arange(P.shape[1])[None] == pick[:, None], np.inf, dall), axis=1)[:, 0]    # nearest OTHER proposal
    same_prop = (dall.argmin(1) == pick) & (dpose <= SMOKE_POSE_TOL_M) & (d2 >= SMOKE_MARGIN * np.maximum(dpose, 1e-9))
    rows, per = {}, []
    for i, t in enumerate(tok):
        k = int(pick[i])
        if k not in rows:
            rows[k] = C.read_csv(ctx.csv_k(k))
        v, s, c = rows[k][t]
        bv, bs, bc = lr[t]
        per.append({"token": t, "pick": k, "pose_max_abs_m": float(dpose[i]), "nearest_other_proposal_m": float(d2[i]), "same_proposal": bool(same_prop[i]),
                    "table_pdms_at_pick": s, "banked_csv_score": bs, "abs_diff": abs(s - bs),
                    "subscore_max_abs_diff": float(np.abs(np.array(c) - np.array(bc)).max()), "valid": v})
    ok = bool(same_prop.all() and all(p["valid"] for p in per) and all(p["table_pdms_at_pick"] == 0.0 and p["banked_csv_score"] == 0.0 for p in per))
    R = {"evidence": "MEASURED on P2, CPU dump (not the GPU that produced the banked seam); tier T1", "n_tokens": len(tok), "tokens_asked": len(ctx.toks),
         "criteria": {"same_proposal": f"argmin over the 64 proposals of max|pose - banked P2 pick| == the dump's pick, d_pick <= {SMOKE_POSE_TOL_M} m, and the nearest OTHER proposal >= {SMOKE_MARGIN}x farther",
                      "score": "table PDMS at the pick == 0.0 and the banked P2 csv score == 0.0 (all 8 are P2 zero tokens), all valid; sub-score differences REPORTED, not gating",
                      "why_not_the_C3_1e-3_m_pose_bar": "the first CPU attempt read max |d| 1.36e-3 m (1 of 8 tokens above 1e-3 m; attempt1_gate_fidelity_cpu.json). The 1e-3 m bar was registered for GPU-vs-GPU "
                                                        "(C3 u400: 0.0 on 1,058). This CPU-vs-GPU smoke cannot be bit-exact, so it asks the question the bar stands for -- is it the SAME PROPOSAL -- "
                                                        "and the strict 1e-3 m + score-equality proof is applied to all 509 on the GPU in the main run. Criterion pinned after seeing the 8 pose diffs (post-hoc, flagged)."},
         "pose_max_abs_m": float(dpose.max()), "pose_n_bit_identical": int((dpose == 0).sum()), "n_same_proposal": int(same_prop.sum()),
         "score_n_exact": int(sum(1 for p in per if p["abs_diff"] == 0)), "score_max_abs_diff": max(p["abs_diff"] for p in per),
         "subscore_max_abs_diff": max(p["subscore_max_abs_diff"] for p in per), "distinct_pick_columns": sorted(rows),
         "per_token": per, "PASS": ok,
         "reading": ("P2's own pick is REPRODUCED on the smoke" if ok else "NOT the same pick on the smoke -- STOP: checkpoint, rule or seam is wrong")}
    jdump(R, os.path.join(wd, "SMOKE8_PROOF.json"))
    return R


# ================================================================================ P2_MEASURED
def bridge_v1(tokens, pos, N, P):
    """Rebuild the landed V1 bridge labels per token (u400 table if the token is in it, else the pre-SFT census) with ZT's own functions,
    to compare token by token with the new MEASURED labels.  Checked against RESULT.json's V1 counts before it is trusted."""
    Tu = np.load(P["c3_table"], allow_pickle=True)
    t_idx = np.array([pos[str(t)] for t in Tu["token"]])
    sub = Tu["sub"].astype(float)
    fs_u = ZT.fan_stats(Tu["pdms"].astype(float), sub[..., 0], sub[..., 1], sub[..., 3])
    in_u = np.zeros(N, bool)
    in_u[t_idx] = True
    Cn = ZT.load_census(P["census"], tokens)
    fs_c = ZT.fan_stats(Cn["pdms"], Cn["nc"], Cn["dac"], Cn["ttc"])
    sel = np.where(in_u, ZT.embed(N, t_idx, fs_u["sel"], False), fs_c["sel"])
    best = np.where(in_u, ZT.embed(N, t_idx, fs_u["best"], 0.0), fs_c["best"])
    return sel, best, np.where(in_u, 1, 2)


def measured(wd, out_json, out_csv, proof):
    P = {k: str(v) for k, v in ZT.DEFAULTS.items()}
    tk = json.load(open(P["tokens"], encoding="utf-8"))
    tokens = sorted(tk["tokens"])
    N = len(tokens)
    pos = {t: i for i, t in enumerate(tokens)}
    logs = np.array([tk["token_log"][t] for t in tokens])
    log_names, log_idx = np.unique(logs, return_inverse=True)
    n_logs = len(log_names)
    assert N == ZT.EXPECT["n_tokens"] and n_logs == ZT.EXPECT["n_logs"], (N, n_logs)
    boot = ZT.Boot(log_idx, n_logs)                                        # B 10,000, seed 20260927, 136 logs: the landed analysis's own draws
    p2_csv = P["p2_csv"] if os.path.exists(P["p2_csv"]) else P["p2_csv_repo"]
    assert sha256_file(p2_csv) == ZT.EXPECT["p2_csv_sha256"], "the P2 csv is not the banked original"
    p2 = ZT.load_scores(p2_csv, tokens)
    p2z = p2["score"] == 0.0
    assert int(p2z.sum()) == ZT.EXPECT["p2_zero"], int(p2z.sum())
    T = np.load(os.path.join(wd, "table.npz"), allow_pickle=False)
    tok_t = [str(x) for x in T["token"]]
    t_idx = np.array([pos[t] for t in tok_t])
    assert set(t_idx.tolist()) == set(np.where(p2z)[0].tolist()), "the table's tokens are not exactly P2's zero tokens"
    sub = T["sub"].astype(float)
    names = [str(x) for x in T["sub_names"]]
    assert names[0].startswith("no_at_fault") and names[1].startswith("drivable") and names[3].startswith("time_to_collision")
    pdms = T["pdms"].astype(float)
    assert np.isfinite(pdms).all() and bool(T["valid"].all())
    fs = ZT.fan_stats(pdms, sub[..., 0], sub[..., 1], sub[..., 3])
    fill = {"sel": False, "best": 0.0, "safe": False, "n_nonzero": 0, "prop_kind": ""}
    S = {k: ZT.embed(N, t_idx, fs[k], fill[k]) for k in fill}
    S["sel"] = S["sel"].astype(bool)
    ones = np.ones(N)

    q1 = ZT.q1_block(p2z, S["sel"], S["best"], boot, N)
    cause = ZT.cause_labels(p2["nc"], p2["dac"])
    cb = ZT.cause_block(p2z, cause, S["sel"], boot)
    gain_by_cause = {c: boot.ratio(np.where(p2z & S["sel"] & (cause == c), S["best"], 0.0), ones, scale=100.0) for c in ZT.CAUSES[:3]}
    prop_kinds = {k: int((p2z & ~S["sel"] & (S["prop_kind"] == k)).sum()) for k in ZT.PROP_KINDS}
    safe_n = int((p2z & S["safe"]).sum())
    mean_nz = float(p2["score"][~p2z].mean() * 100)

    # ---- against the landed INFERRED V1 row (RESULT.json) and the proxy labels, token by token
    old = json.load(open(os.path.join(ZT_DIR, "RESULT.json"), encoding="utf-8"))
    v1 = old["Q1c_P2_bridged"]["variants"]["V1_hierarchy"]
    sel_v1, best_v1, tier = bridge_v1(tokens, pos, N, P)
    recon_ok = bool(int((p2z & sel_v1).sum()) == v1["SEL_def"]["n_SEL"])
    cmp_ = {"inferred_V1_row_in_RESULT_json": {"n_SEL": v1["SEL_def"]["n_SEL"], "n_PROP": v1["SEL_def"]["n_PROP"],
                                              "SEL_share": v1["SEL_def"]["SEL_share"], "oracle_on_SEL_gain_pdms_pts": v1["SEL_def"]["oracle_on_SEL_gain_pdms_pts"],
                                              "Q2_cause": v1["Q2_cause"], "evidence": "INFERRED (proxy fans)"},
            "bridge_reconstruction_reproduces_RESULT_json_V1_n_SEL": recon_ok}
    if recon_ok:
        mz = p2z
        cmp_["token_by_token_inferred_V1_vs_measured"] = {
            "contingency_n": {f"inferred_{'SEL' if a else 'PROP'}__measured_{'SEL' if b else 'PROP'}": int((mz & (sel_v1 == a) & (S['sel'] == b)).sum())
                              for a in (True, False) for b in (True, False)},
            "agreement": float((sel_v1[mz] == S["sel"][mz]).mean()),
            "by_tier": {f"tier{t}": {"n": int((mz & (tier == t)).sum()), "agreement": float((sel_v1[mz & (tier == t)] == S["sel"][mz & (tier == t)]).mean())
                                     if (mz & (tier == t)).any() else None} for t in (1, 2)},
            "both_SEL_tokens": {"n": int((mz & sel_v1 & S["sel"]).sum()),
                                "median_abs_diff_best_pdms": float(np.median(np.abs(best_v1[mz & sel_v1 & S["sel"]] - S["best"][mz & sel_v1 & S["sel"]])))
                                if (mz & sel_v1 & S["sel"]).any() else None,
                                "sum_best_inferred_over_N_x100": float(best_v1[mz & sel_v1 & S["sel"]].sum() / N * 100),
                                "sum_best_measured_over_N_x100": float(S["best"][mz & sel_v1 & S["sel"]].sum() / N * 100)}}
    ci = lambda d: [round(d["lo"], 3), round(d["hi"], 3)]  # noqa: E731
    reading = {
        "verdict_PREREG_section_6_rule": q1["verdict"],
        "P2_MEASURED": {"n_zero": q1["n_zero"], "n_SEL": q1["n_SEL"], "n_PROP": q1["n_PROP"], "SEL_share": q1["SEL_share"]["point"], "SEL_share_ci95": ci(q1["SEL_share"]),
                        "oracle_on_SEL_ceiling_pdms_pts": q1["oracle_on_SEL_gain_pdms_pts"]["point"], "ceiling_ci95": ci(q1["oracle_on_SEL_gain_pdms_pts"])},
        "P2_INFERRED_V1_for_comparison": {"n_SEL": v1["SEL_def"]["n_SEL"], "n_PROP": v1["SEL_def"]["n_PROP"], "SEL_share": v1["SEL_def"]["SEL_share"]["point"],
                                          "ceiling": v1["SEL_def"]["oracle_on_SEL_gain_pdms_pts"]["point"]},
        "ceiling_is_a_CEILING_not_an_expected_gain": True}
    R = {"meta": {"title": "P2's OWN 64-proposal table on its 509 zero-PDMS navtest tokens", "when": now(), "tier": "T1 (NAVSIM v1.1 PDMS, pseudo-simulation of an open-loop plan)",
                  "evidence": "MEASURED on P2 (own table); the INFERRED row it replaces is in comparison_to_inferred",
                  "estimator": {"name": "log-cluster bootstrap (zero_tail.Boot)", "B": ZT.B_BOOT, "seed": ZT.SEED, "n_logs": n_logs,
                                "answers": "another draw of LOGS only; NOT another training run (H-ESTIM-SEED-1: P2 is one seed) and NOT another inference run (P2's inference is deterministic)"},
                  "ckpt": C.CKPT, "ckpt_sha256": C.CKPT_SHA256, "rule": C.RULE, "banked_p2_csv_sha256": ZT.EXPECT["p2_csv_sha256"],
                  "table_npz_sha256": sha256_file(os.path.join(wd, "table.npz")), "tokens_json_sha256": sha256_file(os.path.join(HERE, "tokens_p2zero.json")),
                  "same_pick_proof": {"PASS": proof["PASS"], "pose_max_abs_m": proof["pose"]["max_abs_m"], "pose_n_bit_identical": proof["pose"]["n_bit_identical"],
                                      "pdms_n_exact": proof["pdms_at_pick_vs_banked_csv"]["n_exact"], "n": proof["n_tokens"]},
                  "code": {"classifier": "zero_tail.py (landed ddc9ce9): fan_stats, q1_block, cause_block, cause_labels, Boot, verdict, cleanliness",
                           "pipeline": "c3_run_p2.py = c3_run.py + c3_run_vs_p2.diff"}},
         "Q1_SEL_PROP": {"evidence": "MEASURED on P2", **q1},
         "Q2_cause_split": {"evidence": "MEASURED on P2 (cause at the pick from the banked csv; SEL/PROP from P2's own table)", **cb,
                            "gain_by_cause_pdms_pts_SEL": gain_by_cause},
         "SAFE_sensitivity": {"n_SAFE_NC1_DAC1_some_proposal": safe_n, "n_zero": int(p2z.sum()), "what": "C3's 'rescuable' definition, beside SEL, never in place of it"},
         "fan_cleanliness_POSTHOC": ZT.cleanliness(p2z, S["sel"], S["n_nonzero"]),
         "PROP_failure_kinds": prop_kinds,
         "context": {"mean_pdms_of_nonzero_tokens_x100": mean_nz,
                     "proposal_side_ceiling_pts_ESTIMATED": float(q1["n_PROP"] / N * mean_nz),
                     "proposal_side_ceiling_def": "n_PROP / N * mean non-zero PDMS: an ESTIMATE, not a measurement"},
         "comparison_to_inferred": cmp_, "reading": reading}
    jdump(R, out_json)
    # per-token csv (so the labels can be joined without loading the npz)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "log", "pick", "p2_score", "cause_at_pick", "SEL", "n_nonzero_of_64", "best_pdms_of_64", "safe_NC1_DAC1_exists", "prop_kind",
                    "inferred_V1_SEL", "inferred_V1_tier"])
        pk = T["pick"].astype(int)
        for j, t in enumerate(tok_t):
            i = t_idx[j]
            w.writerow([t, tk["token_log"][t], int(pk[j]), p2["score"][i], cause[i], bool(S["sel"][i]), int(S["n_nonzero"][i]), round(float(S["best"][i]), 6),
                        bool(S["safe"][i]), S["prop_kind"][i], bool(sel_v1[i]), int(tier[i])])
    return R


# ================================================================================ markers + backup
def backup(wd, files):
    os.makedirs(BACKUP, exist_ok=True)
    man = {}
    for src in files:
        if os.path.exists(src):
            dst = os.path.join(BACKUP, os.path.basename(src))
            shutil.copy2(src, dst)
            man[os.path.basename(src)] = {"sha256": sha256_file(dst), "bytes": os.path.getsize(dst)}
    jdump({"written": now(), "what": "P2 own 64-proposal table (the 509 P2 zero tokens): durable second copy; the package dir is the primary", "files": man},
          os.path.join(BACKUP, "MANIFEST.sha256.json"))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=("main", "smoke8"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from-stage", default="all", choices=("all", "proof"), help="proof = skip the C3 pipeline (table.npz exists); rerun the proof + analysis only")
    a = ap.parse_args()
    smoke = a.tag == "smoke8"
    wd = os.path.join(HERE, "_smoke8") if smoke else HERE
    os.makedirs(wd, exist_ok=True)
    done, failed = [os.path.join(wd, n) for n in (("SMOKE8_DONE", "SMOKE8_FAILED") if smoke else ("P2TABLE_DONE", "P2TABLE_FAILED"))]
    for p in (done, failed):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{now()}] p2tab_driver start tag={a.tag} pid={os.getpid()}", flush=True)

    def fail(msg):
        with open(failed, "w", encoding="utf-8") as f:
            f.write(f"{now()} {msg}\n--- STATUS.json ---\n")
            sp = os.path.join(wd, "STATUS.json")
            f.write(open(sp, encoding="utf-8").read() if os.path.exists(sp) else "(none)")
        print(f"[{now()}] FAILED: {msg}", flush=True)
        return 1

    try:
        if smoke:
            for st, extra in (("dump", ["--device", "cpu"]), ("gate", [])):
                rc = run_c3([st, "--tag", "smoke8", "--tokens", SMOKE_TOKENS] + extra)
                if st == "gate" and rc == 2:                       # CPU smoke: the GPU-registered 1e-3 m bar is informational here (see smoke_proof criteria)
                    sp = os.path.join(wd, "STATUS.json")
                    stt = json.load(open(sp, encoding="utf-8"))
                    stt["result"] = "CPU SMOKE: the C3 1e-3 m GPU pose bar is informational on CPU; the criterion is smoke_proof's same-proposal test (SMOKE8_PROOF.json)"
                    jdump(stt, sp)
                    rc = 0
                if rc != 0:
                    return fail(f"smoke8 stage {st} rc={rc} (gate_fidelity.json in _smoke8)")
            picks = sorted(set(int(k) for k in np.load(os.path.join(wd, "proposals.npz"))["pick"]))
            rc = run_c3(["score", "--tag", "smoke8", "--tokens", SMOKE_TOKENS, "--workers", a.workers, "--only"] + picks)
            if rc != 0:
                return fail(f"smoke8 score rc={rc}")
            R = smoke_proof(wd)
            print(json.dumps({k: v for k, v in R.items() if k != "per_token"}, indent=1), flush=True)
            if not R["PASS"]:
                return fail("SMOKE8 SAME-PICK PROOF FAILED: " + R["reading"])
            open(done, "w").write(f"{now()} SMOKE8 PASS\n")
            return 0
        if a.from_stage == "all":
            rc = run_c3(["all", "--tag", "main", "--workers", a.workers])
            if rc != 0:
                return fail(f"c3_run_p2 all rc={rc} (1=dump/preflight, 2=FIDELITY gate = P2's pick not reproduced, 3=score, 4=table G2/G3)")
        proof = same_pick_proof(wd, os.path.join(wd, "same_pick_proof.json"))
        print(json.dumps({k: proof[k] for k in ("PASS", "pose", "pdms_at_pick_vs_banked_csv", "subscores_at_pick_vs_banked_csv")}, indent=1), flush=True)
        if not proof["PASS"]:
            return fail("SAME-PICK PROOF FAILED: " + proof["reading"])
        R = measured(wd, os.path.join(wd, "P2_MEASURED.json"), os.path.join(wd, "p2_zero_tokens_sel_prop.csv"), proof)
        print(json.dumps(R["reading"], indent=1), flush=True)
        man = backup(wd, [os.path.join(wd, n) for n in ("table.npz", "proposals.npz", "same_pick_proof.json", "P2_MEASURED.json", "p2_zero_tokens_sel_prop.csv",
                                                       "gate_fidelity.json", "gate_table.json", "STATUS.json", "tokens_p2zero.json")])
        with open(done, "w", encoding="utf-8") as f:
            json.dump({"when": now(), "headline": R["reading"], "backup_dir": BACKUP, "backup_files": sorted(man)}, f, indent=1)
        print(f"[{now()}] P2TABLE_DONE", flush=True)
        return 0
    except BaseException:  # noqa: BLE001
        return fail("EXCEPTION\n" + traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())
