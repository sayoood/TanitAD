# diag_tail3.py : runs ON THE POD, CPU only, no model. Third step of the diagnosis of the stop TRAIN-RC of the reader training of card PREREG_V8F_PROOF_ADD46.
# The EXACT count of gap-labelled frames (the mask of lead_sup_terms, items built as the trainer builds them) for EVERY training step of
#   (1) every log interval whose logged interval minimum is 0 (the flagged intervals), and
#   (2) the first N_CTRL log intervals whose logged interval minimum is above 0 (known-value check: the exact interval minimum equals the logged one).
# Outside the flagged intervals the train log itself says that every step has a count above 0, so (1) gives the exact number of steps with no gap-labelled frame.
# Usage: python diag_tail3.py <build dir> <reader out dir> <own out dir> <lead tree>
import hashlib
import json
import os
import sys

import numpy as np

B, LR, OUT, TREE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N_CTRL = 5
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
hist = int(cfg.lead_hist)
LOG_EVERY = int(P.LOG_EVERY)


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
arms = sorted(d for d in os.listdir(LR) if os.path.isfile(os.path.join(LR, d, "train_log.jsonl")))
logs = {arm: [json.loads(ln) for ln in open(os.path.join(LR, arm, "train_log.jsonl"), encoding="utf-8") if ln.strip()][1:] for arm in arms}
recs = logs[arms[0]]
logged_min = {int(r["step"]): float(r["n_gap_min_interval"]) for r in recs}
flagged = sorted(s for s, v in logged_min.items() if v == 0)
same_flags = all(sorted(int(r["step"]) for r in logs[x] if r["n_gap_min_interval"] == 0) == flagged for x in arms)
ctrl = [s for s in sorted(logged_min) if logged_min[s] > 0 and s > 1][:N_CTRL]
by_iv = {}
for i in range(total):
    by_iv.setdefault(interval(i + 1), []).append(i)
exact = {}
for s in flagged + ctrl:
    for i in by_iv[s]:
        exact[i] = n_gap_exact(batches[i])
zero = sorted(i for s in flagged for i in by_iv[s] if exact[i] == 0)
full_in_flagged = [exact[i] for s in flagged for i in by_iv[s] if rows[i] == batch]
res = {"what": "exact count of gap-labelled frames for every training step of the flagged log intervals and of five control intervals (reader training of card ADD46)",
       "same_order_first_200_batches": seq == r1.get("sequence_sha256"), "total_steps": int(total), "batch": batch, "sampler_chunk": chunk,
       "n_log_records": len(recs), "flagged_log_intervals": flagged, "n_flagged": len(flagged), "flags_equal_in_all_arms": bool(same_flags),
       "n_steps_counted_exactly": len(exact), "n_steps_in_flagged_intervals": int(sum(len(by_iv[s]) for s in flagged)),
       "zero_steps": [i + 1 for i in zero], "zero_rows": [int(rows[i]) for i in zero], "n_zero_steps": len(zero),
       "zero_all_short": bool(all(rows[i] < batch for i in zero)), "zero_max_rows": int(max(rows[i] for i in zero)) if zero else None,
       "every_flagged_interval_has_a_zero_step": bool(all(any(exact[i] == 0 for i in by_iv[s]) for s in flagged)),
       "full_batches_in_flagged_intervals": len(full_in_flagged), "full_batches_in_flagged_intervals_min_exact": int(min(full_in_flagged)) if full_in_flagged else None,
       "share_of_steps_zero": len(zero) / total, "share_of_window_rows_in_zero_steps": float(sum(int(rows[i]) for i in zero) / rows.sum()),
       "control_intervals": [{"step": s, "logged_interval_min": logged_min[s], "exact_interval_min": int(min(exact[i] for i in by_iv[s]))} for s in ctrl],
       "control_intervals_all_equal": bool(all(int(logged_min[s]) == min(exact[i] for i in by_iv[s]) for s in ctrl))}
json.dump(res, open(os.path.join(OUT, "DIAG_TAIL3.json"), "w", encoding="utf-8", newline="\n"), indent=1)
keep = {k: res[k] for k in res if k not in ("flagged_log_intervals",)}
print(json.dumps(keep)[:2200])
print("ZZ DIAG-TAIL3 DONE")
