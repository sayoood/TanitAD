"""Paired log-cluster intervals for the D-REFE-GAP-1 decomposition (read-only; writes gap_decomp_ci.json).

Decomposition formula copied from refe-plan/raw/2026-10-11-ep-gap-decomposition/ep_lowtail.py (exact, sequential,
gate first): human - u400 = gate + EP + TTC + C with
  gate = mean((g_h - g_u) * w_h), EP = mean(g_u * 5 (ep_h - ep_u) / 12), TTC = mean(g_u * 5 (ttc_h - ttc_u) / 12),
  C = mean(g_u * 2 (c_h - c_u) / 12), g = NC * DAC, w = (5 EP + 5 TTC + 2 C) / 12.
Control: the printed line of ep_lowtail.out must be reproduced:
  "human - u400 = 6.30 PDMS = gate(NC*DAC zeros) 4.15 + EP 0.86 + TTC 1.31 + C -0.01".
Estimator: zero_tail.Boot (ratio of sums over 136 logs, B = 10,000, seed 20260927), the SAME log draws for every term.
"""
import csv, json
import numpy as np

SEED, B = 20260927, 10000
S = "D:/Projects/TanitAD/data/refe_navtest/score"
TOK = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
U, H = "refe_stage2_sft4_B_u400_navtest_full", "refe_p4_human_navtest"
COLS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "score")


def load(run, tokens):
    d = {}
    with open(f"{S}/{run}/{run}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["token"]] = [float(r[c]) for c in COLS]
    return np.array([d[t] for t in tokens], float)


def main():
    tk = json.load(open(TOK, encoding="utf-8"))
    tokens = sorted(tk["tokens"])
    logs = np.array([tk["token_log"][t] for t in tokens])
    names, idx = np.unique(logs, return_inverse=True)
    n_logs = len(names)
    assert len(tokens) == 12146 and n_logs == 136
    draws = np.random.default_rng(SEED).integers(0, n_logs, size=(B, n_logs))
    u, h = load(U, tokens), load(H, tokens)
    gu, gh = u[:, 0] * u[:, 1], h[:, 0] * h[:, 1]
    wu = (5 * u[:, 2] + 5 * u[:, 3] + 2 * u[:, 4]) / 12
    wh = (5 * h[:, 2] + 5 * h[:, 3] + 2 * h[:, 4]) / 12
    terms = {
        "total_human_minus_u400": gh * wh - gu * wu,
        "gate_zeroed_tokens": (gh - gu) * wh,
        "EP": gu * 5 * (h[:, 2] - u[:, 2]) / 12,
        "TTC": gu * 5 * (h[:, 3] - u[:, 3]) / 12,
        "C": gu * 2 * (h[:, 4] - u[:, 4]) / 12,
    }
    ones = np.ones(len(tokens))
    s_d = np.bincount(idx, weights=ones, minlength=n_logs)
    bd = s_d[draws].sum(axis=1)
    out = {"estimator": "paired log-cluster bootstrap, ratio of sums, 136 logs, B=10000, seed 20260927, same draws for all terms",
           "answers": "another draw of LOGS only (u400 = one training seed)", "tier": "T1 NAVSIM v1.1 PDMS x100",
           "control_printed_by_ep_lowtail_out": "6.30 = 4.15 + EP 0.86 + TTC 1.31 + C -0.01", "terms": {}}
    for k, x in terms.items():
        s_n = np.bincount(idx, weights=x, minlength=n_logs)
        point = s_n.sum() / s_d.sum() * 100
        v = s_n[draws].sum(axis=1) / bd * 100
        out["terms"][k] = {"point": round(float(point), 4), "lo": round(float(np.percentile(v, 2.5)), 4),
                           "hi": round(float(np.percentile(v, 97.5)), 4)}
    t = out["terms"]
    out["sum_of_terms"] = round(sum(t[k]["point"] for k in ("gate_zeroed_tokens", "EP", "TTC", "C")), 4)
    out["zero_picks_u400"] = int(np.sum(gu == 0))
    out["zero_picks_human"] = int(np.sum(gh == 0))
    out["gate_share_of_gap_pct"] = round(t["gate_zeroed_tokens"]["point"] / t["total_human_minus_u400"]["point"] * 100, 2)
    json.dump(out, open("C:/Users/Admin/paper_pkg/gap_decomp_ci.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
