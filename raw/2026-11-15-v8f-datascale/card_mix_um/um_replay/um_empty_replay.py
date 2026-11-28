"""MM 2026-11-18: does UM_s0's K-TACSRC miss (tac_src_empty_batches 10 > 5) come from the SAMPLER alone?
Replays the trainer's own data build and ChunkedSampler (tree_um = commit 648d4ca3, UM_s0's argv, seed, chunk) for exactly `total` steps,
indices only (no features, no model), and counts batches with NO listed (corpus) clip -- the batches whose T block is exactly 0 at lambda 0.
If the count equals the run's 10, the miss is a deterministic property of the sampler's partial pool-end batches, not a reach defect.
Also reports how many of them are pool-end partial batches and their sizes. Pod CPU only. usage: python um_empty_replay.py <argv json> <out json>"""
import json
import math
import os
import sys

TREE = "/root/dscale10k/tree_um"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE):
    sys.path.insert(0, p)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import v8f_mini_train as T  # noqa: E402
import tanitad.data.v8f_mini_data as D  # noqa: E402

assert os.path.realpath(T.__file__).startswith(TREE + "/"), T.__file__
argv_path, out_path = sys.argv[1], sys.argv[2]
toks = json.load(open(argv_path, encoding="utf-8"))
toks = toks["argv"] if isinstance(toks, dict) else toks
a = T.build_parser().parse_args([str(t) for t in toks])
cfg = T.make_config(a)
store, J, wins, clips, info = T.build_data(a, cfg)
tsrc = T.make_tac_src(a, J, clips, wins, info)
full = np.asarray(tsrc.full_ci, dtype=bool)
chunk = max(4, min(48, info["lru"] - 2))
sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
S = math.ceil(info["n_windows_all_clips"] / a.batch)
total = a.steps or S * a.steps_mult
step, ep, empty, sizes, empty_sizes, n_partial, rows_l, rows_o = 0, 0, 0, {}, [], 0, 0, 0
while step < total:
    for idx in sampler.epoch(ep):
        n = len(idx)
        nl = int(sum(bool(full[int(w[0])]) for w in idx))
        rows_l += nl
        rows_o += n - nl
        if n < a.batch:
            n_partial += 1
        if nl == 0:
            empty += 1
            empty_sizes.append(n)
        step += 1
        if step >= total:
            break
    else:
        ep += 1
        continue
    break
rec = {"tree": TREE, "argv": argv_path, "seed": a.seed, "batch": a.batch, "chunk_clips": chunk, "total_steps": total, "epochs_entered": ep + 1,
       "empty_batches_replayed": empty, "empty_batch_sizes": empty_sizes, "partial_batches": n_partial,
       "rows_listed_replayed": rows_l, "rows_other_replayed": rows_o, "n_listed_clips": int(full.sum()), "n_windows": len(wins)}
json.dump(rec, open(out_path, "w", encoding="utf-8"), indent=1)
print("ZZ UM REPLAY DONE ZZ", json.dumps({k: rec[k] for k in ("empty_batches_replayed", "empty_batch_sizes", "partial_batches", "rows_listed_replayed", "rows_other_replayed", "total_steps", "epochs_entered")}))
