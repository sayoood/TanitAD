# diag_tail.py : runs ON THE POD, CPU only, no model, no token read. Diagnosis of the stop TRAIN-RC of the reader training of card PREREG_V8F_PROOF_ADD46.
# It replays the batch order of the reader training (the trainer's own data builder and sampler, as tok_probe_lr.py train does) and counts, for every
# training step, the rows of the batch and the gap-labelled frames the lead term would count, from the label arrays only. The count ignores one condition
# of frame_ok (non-finite p10f of a frame), so it is an upper bound; it is compared with the logged n_gap of every log record.
# Usage: python diag_tail.py <build dir with tok_probe_lr.py> <reader out dir with train_meta.json and the arm folders> <own out dir> <lead tree>
import json
import os
import sys

import numpy as np

B, LR, OUT, TREE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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
assert batch == meta["batch"]
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


rows, ub, full_hash = [], [], []
ep = 0
while len(rows) < total:
    for b in sampler.epoch(ep):
        if len(rows) >= total:
            break
        if len(full_hash) < P.R1_BATCHES:
            full_hash.append(P.batch_hash(b))
        rows.append(len(b))
        ub.append(n_gap_ub(b))
    ep += 1
rows, ub = np.array(rows), np.array(ub)
import hashlib  # noqa: E402

seq = hashlib.sha256("".join(full_hash).encode()).hexdigest()
r1 = json.load(open(os.path.join(LR, "r1_batches.json"), encoding="utf-8"))
res = {"what": "replay of the batch order of the reader training (labels only, CPU); n_gap_ub ignores the p10f part of frame_ok",
       "total_steps": int(total), "batch": batch, "sampler_chunk": chunk, "epochs_touched": ep, "n_windows": int(info["n_windows"]),
       "replay_sequence_sha256": seq, "trained_sequence_sha256": r1.get("sequence_sha256"), "same_order_first_200_batches": seq == r1.get("sequence_sha256")}
short = rows < batch
zero = ub == 0
res["n_short_batches"] = int(short.sum())
res["short_rows_min_median_max"] = [int(rows[short].min()), float(np.median(rows[short])), int(rows[short].max())] if short.any() else None
res["n_zero_gap_batches"] = int(zero.sum())
res["zero_gap_batches_all_short"] = bool((short[zero]).all()) if zero.any() else None
res["zero_gap_batch_rows"] = sorted(int(x) for x in rows[zero])
res["zero_gap_steps"] = [int(i) + 1 for i in np.flatnonzero(zero)]
res["full_batches_min_n_gap_ub"] = int(ub[~short].min())
res["full_batches_n"] = int((~short).sum())
res["share_of_steps_zero_gap"] = float(zero.mean())
res["share_of_window_rows_in_zero_gap_batches"] = float(rows[zero].sum() / rows.sum())
# intervals of the train log: a record at step 1, at every multiple of LOG_EVERY and at the last step; the interval minimum is reset after every record
iv_zero = set()
for s in res["zero_gap_steps"]:
    iv_zero.add(1 if s == 1 else (total if (s > (total // LOG_EVERY) * LOG_EVERY) else ((s + LOG_EVERY - 1) // LOG_EVERY) * LOG_EVERY))
per_arm = {}
for arm in sorted(d for d in os.listdir(LR) if os.path.isfile(os.path.join(LR, d, "train_log.jsonl"))):
    recs = [json.loads(ln) for ln in open(os.path.join(LR, arm, "train_log.jsonl"), encoding="utf-8") if ln.strip()][1:]
    flagged = sorted(int(r["step"]) for r in recs if r["n_gap_min_interval"] == 0)
    inst_equal = sum(1 for r in recs if int(r["n_gap"]) == int(ub[int(r["step"]) - 1]))
    per_arm[arm] = {"n_records": len(recs), "flagged_records": len(flagged), "flagged_equal_replay_intervals": flagged == sorted(iv_zero),
                    "min_logged_instantaneous_n_gap": float(min(r["n_gap"] for r in recs)), "records_where_logged_n_gap_equals_replay_ub": inst_equal,
                    "records_where_logged_n_gap_above_replay_ub": sum(1 for r in recs if int(r["n_gap"]) > int(ub[int(r["step"]) - 1]))}
res["replay_intervals_with_zero"] = sorted(iv_zero)
res["per_arm"] = per_arm
json.dump(res, open(os.path.join(OUT, "DIAG_TAIL.json"), "w", encoding="utf-8", newline="\n"), indent=1)
np.savez_compressed(os.path.join(OUT, "diag_tail_steps.npz"), rows=rows.astype(np.int16), n_gap_ub=ub.astype(np.int32))
keep = {k: res[k] for k in res if k not in ("zero_gap_steps", "replay_intervals_with_zero")}
print(json.dumps(keep)[:1800])
print("ZZ DIAG-TAIL DONE")
