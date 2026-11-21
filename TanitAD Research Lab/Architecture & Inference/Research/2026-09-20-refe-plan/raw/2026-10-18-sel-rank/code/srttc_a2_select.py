"""SR-TTC Amendment 2 (DRAFT): deterministic log-subsample of the FIT set, from METADATA ONLY.

Reads ONE file: rl_pool/navtrain_token_table.tsv.gz (sha256 c7456c88..., the token table the
p2-heldout-table disjointness proof cross-checks), and from it ONLY five columns:
token, log, in_navtest, heldout_log, served_train_sets. No feature, label, score, PDM value,
reward or scorer output is read. served_keys.tsv.gz is NOT read.

Rule (fixed before any SR-TTC fit datum is read):
  fit logs   = logs with >= 1 token whose served_train_sets > 0   (expected 1,167 / 99,495 / 174,912)
  key(log)   = sha256( SALT + "\n" + log ).hexdigest()            SALT = "SR-TTC/A2/v1"
  order      = ascending key
  S_f        = first ceil(f * N) logs of the order                 (nested: S_10 < S_25 < S_50 < S_100)
  fold(log)  = rank_in_order mod 5                                 (stable under scale-up)
  LC prefix  = within each CV training part, the first ceil(p * n) logs in the same order, p in {.25,.5,1}

Usage: python -I srttc_a2_select.py <navtrain_token_table.tsv.gz> <out_dir>
"""
import gzip, hashlib, json, math, sys
from collections import defaultdict

SALT = "SR-TTC/A2/v1"
TABLE_SHA256 = "c7456c882b3d297c5ba202947d82b850e640a5b08648a3f4dea3c7c3dd176cd1"
EXPECT = {"n_logs": 1167, "n_tokens": 99495, "n_sets": 174912}  # disjointness_proof.json
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
K_FOLDS, MIN_LOGS_PER_FOLD = 5, 20
LC_PREFIXES = [0.25, 0.50, 1.00]
COLS = ["token", "log", "in_navtest", "heldout_log", "served_train_sets"]

# Rates (s per forward). Provenance in the amendment, section A2.4.
RATE_CENTRAL = 1.44   # prereg sec 4 / Amendment 1: 907 s / 629 navtest tokens (MEASURED there)
# 11 held-out dump units, (tokens, seconds), from raw/2026-10-18-p2-heldout-table/state/unit_*.json "n"
# and logs/dump_*.log "seconds" (MEASURED, dev box, contended GPU):
HO_UNITS = [(48, 70.5), (14, 334.4), (9, 35.3), (36, 52.8), (17, 188.3), (5, 10.3),
            (87, 128.0), (35, 52.1), (23, 48.6), (9, 15.3), (50, 78.3)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ols(xy):
    n = len(xy); mx = sum(x for x, _ in xy) / n; my = sum(y for _, y in xy) / n
    b = sum((x - mx) * (y - my) for x, y in xy) / sum((x - mx) ** 2 for x, _ in xy)
    return my - b * mx, b


def main(table, out):
    got = sha256_file(table)
    assert got == TABLE_SHA256, f"token table sha256 {got} != {TABLE_SHA256}"
    tok_log, log_tok, log_sets = {}, defaultdict(int), defaultdict(int)
    bad = {"navtest": 0, "heldout": 0}
    with gzip.open(table, "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        idx = [hdr.index(c) for c in COLS]
        for line in f:
            r = line.rstrip("\n").split("\t")
            tok, log, nav, ho, s = (r[i] for i in idx)
            s = int(s)
            if s <= 0:
                continue
            if nav not in ("0", "False", "false", ""):
                bad["navtest"] += 1
            if ho not in ("0", "False", "false", ""):
                bad["heldout"] += 1
            tok_log[tok] = log; log_tok[log] += 1; log_sets[log] += s
    meas = {"n_logs": len(log_tok), "n_tokens": len(tok_log), "n_sets": sum(log_sets.values())}
    assert meas == EXPECT, f"fit-set metadata does not reproduce Amendment 1: {meas} vs {EXPECT}"
    assert bad == {"navtest": 0, "heldout": 0}, f"fit tokens flagged navtest/heldout: {bad}"

    keyed = sorted((hashlib.sha256((SALT + "\n" + lg).encode("utf-8")).hexdigest(), lg) for lg in log_tok)
    N = len(keyed)
    with open(f"{out}/ranked_fit_logs.tsv", "w", encoding="utf-8", newline="\n") as w:
        w.write("rank\tlog\tkey_sha256\tfold\tfirst_fraction\tn_fit_tokens\tn_fit_sets\n")
        for r, (k, lg) in enumerate(keyed):
            ff = next(fr for fr in FRACTIONS if r < math.ceil(fr * N))
            w.write(f"{r}\t{lg}\t{k}\t{r % K_FOLDS}\t{ff:.2f}\t{log_tok[lg]}\t{log_sets[lg]}\n")

    a_all, b_all = ols(HO_UNITS)
    clean = [u for u in HO_UNITS if u[1] / u[0] < 2.5]
    a_cl, b_cl = ols(clean)
    rate_contended = sum(s for _, s in HO_UNITS) / sum(n for n, _ in HO_UNITS)
    rates = {"central_s_per_token": RATE_CENTRAL,
             "heldout_marginal_s_per_token_clean_units": round(b_cl, 3),
             "heldout_clean_units_n": len(clean),
             "heldout_ols_all_units": [round(a_all, 1), round(b_all, 3)],
             "contended_s_per_token_aggregate_11_units": round(rate_contended, 3)}

    res = {"what": "SR-TTC Amendment 2 DRAFT: log-subsample selection from metadata only",
           "salt": SALT, "token_table_sha256": got, "columns_read": COLS,
           "fit_set_reproduced": meas, "rates": rates, "fractions": {}}
    for fr in FRACTIONS:
        n = math.ceil(fr * N)
        sel = [lg for _, lg in keyed[:n]]
        folds = []
        for k in range(K_FOLDS):
            fl = [lg for r, lg in enumerate(sel) if r % K_FOLDS == k]
            folds.append({"logs": len(fl), "tokens": sum(log_tok[x] for x in fl), "sets": sum(log_sets[x] for x in fl)})
        tr_logs_min = min(n - f_["logs"] for f_ in folds)
        tokens = sum(log_tok[x] for x in sel); sets = sum(log_sets[x] for x in sel)
        res["fractions"][f"{int(fr*100)}%"] = {
            "n_logs": n, "n_tokens": tokens, "n_sets": sets,
            "token_share": round(tokens / EXPECT["n_tokens"], 4), "set_share": round(sets / EXPECT["n_sets"], 4),
            "selected_logs_sha256": hashlib.sha256("\n".join(sorted(sel)).encode("utf-8")).hexdigest(),
            "folds": folds,
            "fold_min_logs": min(f_["logs"] for f_ in folds), "fold_max_logs": max(f_["logs"] for f_ in folds),
            "fold_sets_min_max": [min(f_["sets"] for f_ in folds), max(f_["sets"] for f_ in folds)],
            "fold_min_logs_ge_k": min(f_["logs"] for f_ in folds) >= MIN_LOGS_PER_FOLD,
            "lc_smallest_prefix_logs": math.ceil(LC_PREFIXES[0] * tr_logs_min),
            "S0_gpu_h": {
                "central_per_token": round(tokens * RATE_CENTRAL / 3600, 2),
                "contended_per_token": round(tokens * rate_contended / 3600, 2),
                "upper_per_set": round(sets * RATE_CENTRAL / 3600, 2)}}
    with open(f"{out}/selection.json", "w", encoding="utf-8", newline="\n") as w:
        json.dump(res, w, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
