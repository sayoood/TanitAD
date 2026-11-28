#!/usr/bin/env python3
"""C3 (eval/P6_GO_SIGNAL.md section 3) -- STEP 1: write tokens.json BEFORE any u400 proposal is run or scored.

Population (as pre-registered):
  * ZERO stratum  : every token whose u400 NAVSIM v1.1 navtest row has NC x DAC == 0 (the banked 529).
  * RANDOM stratum: an equal number (529) of the OTHER tokens, drawn uniformly without replacement with
                    np.random.default_rng(20261011) from the NON-zeroed tokens sorted ascending by token string
                    (the sort is the only operationalisation: the prereg fixes the seed, not the base ordering).
Output: tokens.json in W3 format {'rule','tokens','token_log'} (so refe_navtest_seam.py and
score_navtest_refe.py both consume it unchanged) + 'stratum' {token: 'zero'|'random'} + provenance.
Nothing here reads a model output other than the already-banked u400 CSV.
"""
import csv, gzip, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = ("D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/"
       "refe_stage2_sft4_B_u400_navtest_full.csv")
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEED = 20261011
N_ZERO_EXPECTED = 529


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (float(r["no_at_fault_collisions"]), float(r["drivable_area_compliance"]), float(r["score"]))
    n = len(rows)
    zero = sorted(t for t, (nc, dac, s) in rows.items() if nc * dac == 0.0)
    score0 = sorted(t for t, (nc, dac, s) in rows.items() if s == 0.0)
    nz = sorted(t for t in rows if t not in set(zero))
    assert n == 12146, n
    assert len(zero) == N_ZERO_EXPECTED, f"NC x DAC == 0 count {len(zero)} != 529"
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(nz), size=len(zero), replace=False))
    rand = [nz[i] for i in idx]
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(zero + rand)
    assert len(set(toks)) == 2 * N_ZERO_EXPECTED
    token_log = {t: exp[t]["log_name"] for t in toks}
    out = {
        "rule": "C3 selection headroom (eval/P6_GO_SIGNAL.md section 3): 529 u400 NC x DAC = 0 tokens + 529 uniform non-zeroed tokens",
        "tokens": toks,
        "token_log": token_log,
        "stratum": {t: ("zero" if t in set(zero) else "random") for t in toks},
        "provenance": {
            "csv": CSV, "csv_sha256": sha(CSV), "n_csv_tokens": n,
            "n_zero_NCxDAC": len(zero), "n_score_exactly_zero": len(score0),
            "zero_set_equals_score_zero_set": zero == score0,
            "n_nonzero": len(nz), "seed": SEED,
            "draw": "np.random.default_rng(20261011).choice(len(nz_sorted), 529, replace=False); nz_sorted = non-zeroed tokens ascending",
            "n_logs_total": len(set(token_log.values())),
            "n_logs_zero": len({token_log[t] for t in zero}),
            "n_logs_random": len({token_log[t] for t in rand}),
        },
    }
    p = os.path.join(HERE, "tokens.json")
    assert not os.path.exists(p), "tokens.json already exists -- it is written ONCE, before any u400 run"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=0)
    print(json.dumps(out["provenance"], indent=1))
    print("tokens.json sha256", sha(p))


if __name__ == "__main__":
    main()
