#!/usr/bin/env python3
"""READ 2 interval geometry, ESTIMATED from Step A's OWN per-token terms (no D label or score exists; none is read).  CPU only.

Question: with 99 tokens over 84 log clusters (universe D) instead of 400 tokens over 24 (Step A / B), how wide is the A1 interval the `interval_aware` rule will see, and so what A1 point does O1 need?
Method (a variance model, stated so it can be checked; every number is ESTIMATED, not MEASURED on D):
  1. per-token (num_i, den_i) of A1 on Step A's 400 tokens for an arm (P2; the banked L2 out-of-fold read, seeds 0 and 1);  r = sum num / sum den;  e_i = num_i - r * den_i.
  2. token-level (iid) half-width of the 95 % interval at m tokens:  HW_tok(m) = 1.96 * sqrt(mean(e^2) / m) / mean(den).
  3. design effect of the 24-log clustering at Step A:  DEFF_A = (HW_logcluster / HW_tok(400))^2, with HW_logcluster = half the width of Step A's own interval (paired log-cluster bootstrap, 10,000, seed 20260927);
     intra-cluster correlation  rho = (DEFF_A - 1) / (mbar_A - 1),  mbar_A = 400 / 24 tokens per log.
  4. D: m = 99 tokens in 84 logs, mbar_D = 99 / 84;  DEFF_D = 1 + (mbar_D - 1) * rho;  HW_D = HW_tok(99) * sqrt(DEFF_D)  (a log with 2 tokens is one cluster).
  ASSUMPTION (the weak point): D's per-token A1 terms (mean den, mean e^2) are those of Step A's tokens.  D is a different set of navtrain scenes; if its per-token variance differs, HW_D scales with it.
The O1 requirement under `interval_aware` is a LOWER interval end > 0.80, i.e. an A1 POINT >= 0.80 + HW_D.
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

sys.path.insert(0, "C:/Users/Admin/par18/l2/code")
import l2lib  # noqa: E402
import stepA  # noqa: E402

FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
ARMS = "D:/Projects/TanitAD-artifacts/l2_offprop/run/arms"
M_D, LOGS_D = 99, 84


def geometry(num_tok, den_tok, hw_logcluster, n_tokens_A, n_logs_A, m_D=M_D, logs_D=LOGS_D) -> dict:
    r = float(num_tok.sum() / den_tok.sum())
    e = num_tok - r * den_tok
    hw_tok = lambda m: 1.96 * np.sqrt(np.mean(e ** 2) / m) / np.mean(den_tok)
    deff_A = float((hw_logcluster / hw_tok(n_tokens_A)) ** 2)
    mbar_A, mbar_D = n_tokens_A / n_logs_A, m_D / logs_D
    rho = (deff_A - 1.0) / (mbar_A - 1.0)
    deff_D = 1.0 + (mbar_D - 1.0) * rho
    hw_D = float(hw_tok(m_D) * np.sqrt(max(deff_D, 1.0)))
    return {"A1_point_StepA": r, "HW_logcluster_StepA_measured": float(hw_logcluster), "HW_tok_iid_StepA_400": float(hw_tok(n_tokens_A)), "DEFF_A": deff_A, "rho_hat": float(rho), "DEFF_D": float(deff_D),
            "HW_tok_iid_D_99": float(hw_tok(m_D)), "HW_D_estimated": hw_D, "O1_needs_A1_point_at_least": 0.80 + hw_D}


def main():
    D = l2lib.load_real(FEATS)
    out = {"what": "READ 2 (universe D: 99 tokens / 84 logs) A1 interval half-width, ESTIMATED from Step A's per-token terms (variance model in the module docstring); NOT a D measurement", "evidence": "ESTIMATED",
           "n_tokens_StepA": int(D.N), "n_logs_StepA": int(D.n_logs), "m_D": M_D, "logs_D": LOGS_D, "arms": {}}
    arms = {"P2": (l2lib.ep_apply(D.tok_in, D.W6, D.b6), l2lib.ep_apply(D.tok_p, D.W6, D.b6))}
    for s in (0, 1):
        z = np.load(f"{ARMS}/H_s{s}.npz", allow_pickle=False)
        arms[f"L2_OOF_s{s}"] = (z["ep_in"], z["ep_p"])
    for name, (ei, ep) in arms.items():
        P = l2lib._arm_pairs(D, ei, ep)
        num, den = stepA.admission_terms(P)["A1"]
        nt, dt = np.bincount(D.pair_tok, weights=num, minlength=D.N), np.bincount(D.pair_tok, weights=den, minlength=D.N)
        sums = stepA.per_log_sums({"A1": (num, den)}, D.pair_log, D.n_logs)["A1"]
        ci = stepA.ratio_ci(sums[0], sums[1], D.boot_idx)
        g = geometry(nt, dt, (ci["hi"] - ci["lo"]) / 2.0, D.N, D.n_logs)
        g["StepA_interval"] = [ci["lo"], ci["hi"]]
        g["mean_den_per_token"] = float(dt.mean())
        g["expected_A1_pairs_in_D"] = float(dt.mean() * M_D)
        out["arms"][name] = g
    json.dump(out, open("C:/Users/Admin/par18/l2fresh/raw/readtwo_geometry.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
