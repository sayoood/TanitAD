# diag_tail2.py : runs ON THE POD, CPU only, no model. Second step of the diagnosis of the stop TRAIN-RC of the reader training of card PREREG_V8F_PROOF_ADD46.
# diag_tail.py counted an upper bound of the gap-labelled frames per training step from the labels alone. This step computes the EXACT count (the mask of
# lead_sup_terms: frame_ok over the five frames behind an end frame, and a finite gap label) for the candidate steps, by building the items as the trainer does:
#   (1) every short batch (fewer rows than the batch size) with at most SMALL rows, (2) the batch with the smallest upper bound of every flagged log interval,
#   (3) the batches of the first N_LOGGED log records (exact count against the logged n_gap: the check that the exact count is the trainer's count).
# Usage: python diag_tail2.py <build dir> <reader out dir> <own out dir> <lead tree>
import hashlib
import json
import os
import sys

import numpy as np

B, LR, OUT, TREE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SMALL, N_LOGGED = 8, 8
sys.path.insert(0, B)
import tok_probe_lr as P  # noqa: E402

meta = json.load(open(os.path.join(LR, "train_meta.json"), encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
argv = list(meta["trainer_argv"])
argv[argv.index("--out") + 1] = OUT
T = P.use_tree(TREE)
D, TR = T["D"], T["TR"]
a = TR.build_parser().parse_args(argv)
a.steps = int(meta["total"])
cfg = TR.make_config(a)
ds, wins, S, total, info = P.build_training_data(a, cfg)
assert total == meta["total"] and info["n_windows"] == meta["n_windows"], (total, info["n_windows"])
chunk, batch = int(meta["sampler_chunk"]), int(a.batch)
sampler = D.ChunkedSampler(wins, batch, seed=a.seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
LS, off, hist, hm = ds.lead_sup_labels, ds.J["off"], int(cfg.lead_hist), int(cfg.hmax)
gap_fin = np.isfinite(np.asarray(LS["gap"]))
LOG_EVERY = int(P.LOG_EVERY)


def n_gap_ub(b):
    n = 0
    for ci, t, _, _ in b:
        idx = np.arange(t - hm + 1, t + 1)
        ok = idx >= 0
        g = gap_fin[int(off[ci]) + np.maximum(idx, 0)]
        for j in range(hist - 1, hm):
            if ok[j - hist + 1:j + 1].all() and g[j]:
                n += 1
    return n


def n_gap_exact(b):
    n = 0
    for w in b:
        it = ds.item(*w)
        ok = np.asarray(it["frame_ok"], bool)
        g = np.isfinite(np.asarray(it["lead_gapm"], np.float64))
        for j in range(hist - 1, len(ok)):
            if ok[j - hist + 1:j + 1].all() and g[j]:
                n += 1
    return n


def interval(s):
    return 1 if s == 1 else (total if s > (total // LOG_EVERY) * LOG_EVERY else ((s + LOG_EVERY - 1) // LOG_EVERY) * LOG_EVERY)


batches, full_hash = [], []
ep = 0
while len(batches) < total:
    for b in sampler.epoch(ep):
        if len(batches) >= total:
            break
        if len(full_hash) < P.R1_BATCHES:
            full_hash.append(P.batch_hash(b))
        batches.append(b)
    ep += 1
seq = hashlib.sha256("".join(full_hash).encode()).hexdigest()
r1 = json.load(open(os.path.join(LR, "r1_batches.json"), encoding="utf-8"))
rows = np.array([len(b) for b in batches])
ub = np.array([n_gap_ub(b) for b in batches])
arms = sorted(d for d in os.listdir(LR) if os.path.isfile(os.path.join(LR, d, "train_log.jsonl")))
logs = {arm: [json.loads(ln) for ln in open(os.path.join(LR, arm, "train_log.jsonl"), encoding="utf-8") if ln.strip()][1:] for arm in arms}
recs = logs[arms[0]]
flagged = sorted(int(r["step"]) for r in recs if r["n_gap_min_interval"] == 0)
same_flags = all(sorted(int(r["step"]) for r in logs[x] if r["n_gap_min_interval"] == 0) == flagged for x in arms)
cand = set(int(i) for i in np.flatnonzero((rows < batch) & (rows <= SMALL)))
by_iv = {}
for i in range(total):
    by_iv.setdefault(interval(i + 1), []).append(i)
for s in flagged:
    ii = by_iv[s]
    cand.add(int(ii[int(np.argmin(ub[ii]))]))
logged_steps = [int(r["step"]) for r in recs[:N_LOGGED]]
exact = {i: n_gap_exact(batches[i]) for i in sorted(cand | set(s - 1 for s in logged_steps))}
zero = sorted(i for i in cand if exact[i] == 0)
iv_zero = sorted(set(interval(i + 1) for i in zero))
res = {"what": "exact count of gap-labelled frames (mask of lead_sup_terms, items built as the trainer builds them) on candidate steps of the reader training",
       "same_order_first_200_batches": seq == r1.get("sequence_sha256"), "total_steps": int(total), "batch": batch, "sampler_chunk": chunk,
       "n_short_batches": int((rows < batch).sum()), "n_candidates": len(cand), "small_rows_bound": SMALL,
       "logged_check": [{"step": s, "logged_n_gap": float(r["n_gap"]), "exact": int(exact[s - 1]), "ub": int(ub[s - 1])} for s, r in zip(logged_steps, recs[:N_LOGGED])],
       "logged_check_all_equal": all(int(r["n_gap"]) == exact[s - 1] for s, r in zip(logged_steps, recs[:N_LOGGED])),
       "flagged_log_intervals": flagged, "flags_equal_in_all_arms": bool(same_flags), "n_flagged": len(flagged),
       "exact_zero_steps": [i + 1 for i in zero], "exact_zero_rows": [int(rows[i]) for i in zero], "exact_zero_ub": [int(ub[i]) for i in zero],
       "n_exact_zero_batches_among_candidates": len(zero), "exact_zero_all_short": bool(all(rows[i] < batch for i in zero)),
       "exact_zero_max_rows": int(max(rows[i] for i in zero)) if zero else None,
       "intervals_with_exact_zero": iv_zero, "flagged_intervals_all_explained": bool(set(flagged) <= set(iv_zero)), "intervals_equal": iv_zero == flagged,
       "full_batches_min_ub": int(ub[rows == batch].min()),
       "share_of_steps_exact_zero_among_candidates": len(zero) / total, "share_of_window_rows_in_exact_zero_batches": float(sum(int(rows[i]) for i in zero) / rows.sum()),
       "small_short_batches": [{"step": i + 1, "rows": int(rows[i]), "exact": int(exact[i]), "ub": int(ub[i])} for i in sorted(cand) if rows[i] <= SMALL and rows[i] < batch]}
json.dump(res, open(os.path.join(OUT, "DIAG_TAIL2.json"), "w", encoding="utf-8", newline="\n"), indent=1)
keep = {k: res[k] for k in res if k not in ("small_short_batches", "flagged_log_intervals", "intervals_with_exact_zero")}
print(json.dumps(keep)[:2200])
print("ZZ DIAG-TAIL2 DONE")
