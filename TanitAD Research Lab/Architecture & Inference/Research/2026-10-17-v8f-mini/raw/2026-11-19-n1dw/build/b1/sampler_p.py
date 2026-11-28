"""K-SHARE / Z-MASK helper (PREREG_V8F_N1DW section 6): a CPU replay of the arm's OWN ChunkedSampler, no model, no GPU, no image cache.

It rebuilds exactly what ``v8f_mini_train.py`` ``build_data`` + ``main`` feed the sampler -- the trainer's own parser and ``make_config`` on the arm's argv, ``D.select_train_clips`` after ``--exclude``, ``D.train_windows``, the
``lru`` formula and the ``ChunkedSampler(..., chunk=max(4, min(48, lru - 2)), seed, max_age_t, max_age_s)`` call -- and replays ``sampler.epoch(0)`` (one epoch; the 3,000-step runs take about the first half of it). For every sampled
NOW tick (clip, t) it reads c0 of the route-text table's ``train_nav_rb`` row ``off[clip] + t`` (the row O's nav token reads at NOW: ``rb_mask`` in v8f_mini_train.py reads the same row through the batch).

Prints (and writes as JSON): the RB share p of the sampled NOW ticks over the whole epoch and over the first ``--steps`` batches (the run's own), the per-batch RB-count distribution, the number of leftover (short) batches and
how many hold no RB sample, and K-SHARE's w p / (w p + 1 - p) for w = 2 and 4. GT-valid is every window here (``train_windows`` lists only 6 s-complete windows).

usage: python sampler_p.py --argv-json <N1D_s0.argv.json> --table <navtext_nav_rbd.npz> [--steps 3000] [--out sampler_p.json]
The clip order is the table's ``train_clip`` (row-aligned to the join, which is the cache manifest's order: the trainer asserts it with ``_assert_table_aligned``); ``--join-train`` may name the join npz to use ITS clip / n instead."""
import argparse
import collections
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.environ.get("N1DW_TREE", ""), ):
    if p:
        sys.path.insert(0, os.path.join(p, "stack"))
        sys.path.insert(0, os.path.join(p, "stack", "scripts"))
import v8f_mini_train as TR  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402


def replay(argv, table_path, steps=None, join_path=None):
    a = TR.build_parser().parse_args(argv)
    cfg = TR.make_config(a)
    z = np.load(table_path)
    if join_path:
        J = D.load_join(join_path)
        clip_all, n_of = [str(c) for c in J["clip"].tolist()], np.asarray(J["n"])
        assert [str(c) for c in z["train_clip"].tolist()] == clip_all, "table and join clip order differ"
        off = np.asarray(J["off"])
    else:
        clip_all, n_of, off = [str(c) for c in z["train_clip"].tolist()], np.asarray(z["train_n"]), np.asarray(z["train_off"])
    rb = np.asarray(z["train_nav_rb"], np.float32)
    assert int(off[-1] + n_of[-1]) == rb.shape[0]
    # build_data's selection
    clips_all = list(clip_all)
    if a.exclude:
        xlo, xhi = (int(x) for x in a.exclude.split(":"))
        ex = set(sorted(clips_all, key=D.sha12)[xlo:xhi])
        clips_all = [c for c in clips_all if c not in ex]
    clips = D.select_train_clips(clips_all, a.data_frac)
    if a.max_clips:
        clips = clips[:a.max_clips]
    ci_of = {c: i for i, c in enumerate(clip_all)}
    Jm = {"clip": np.asarray(clip_all), "n": n_of}
    wins = D.train_windows(None, Jm, clips, hist=cfg.hist, start=cfg.window_start)
    lru = a.lru or max(8, int(1.5e9 // (int(n_of.max()) * cfg.n_tok * 1024 * 2)))
    if a.encoder == "lora":
        lru = a.lru or 50
    chunk = max(4, min(48, lru - 2))
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    steps = a.steps if steps is None else steps
    all_cnt, run_cnt, sizes = [], [], []
    for bi, batch in enumerate(sampler.epoch(0)):
        c = [int(rb[int(off[ci]) + t, 0] > 0.5) for ci, t, _, _ in batch]
        all_cnt.append(sum(c))
        sizes.append(len(c))
    sizes = np.asarray(sizes)
    all_cnt = np.asarray(all_cnt)

    def stats(cnt, sz):
        rbs, rows = int(cnt.sum()), int(sz.sum())
        p = rbs / rows
        dist = collections.Counter(int(x) for x in cnt)
        short = sz < a.batch
        return {"batches": int(len(cnt)), "rows": rows, "rb_rows": rbs, "p": p, "batches_with_no_rb": int((cnt == 0).sum()), "share_batches_with_no_rb": float((cnt == 0).mean()),
                "rb_count_per_batch": {str(k): dist[k] for k in sorted(dist)}, "mean_rb_per_batch": float(cnt.mean()), "min_rb_per_batch": int(cnt.min()), "max_rb_per_batch": int(cnt.max()),
                "leftover_batches": int(short.sum()), "leftover_sizes": sorted(int(x) for x in set(sz[short].tolist())), "leftover_batches_with_no_rb": int(((cnt == 0) & short).sum()),
                "k_share_w2": (2 * p) / (2 * p + 1 - p), "k_share_w4": (4 * p) / (4 * p + 1 - p)}

    out = {"argv_sha256": hashlib.sha256(json.dumps(list(argv)).encode("utf-8")).hexdigest(), "table": os.path.abspath(table_path), "table_sha256": hashlib.sha256(open(table_path, "rb").read()).hexdigest(),
           "n_train_clips_all": len(clip_all), "n_clips_trained": len(clips), "n_windows": len(wins), "batch": a.batch, "seed": a.seed, "hist": cfg.hist, "window_start_effective": cfg.window_start_effective,
           "lru": int(lru), "chunk": chunk, "steps": steps, "epoch_batches": int(len(all_cnt)), "run_covers_epoch_share": min(1.0, steps / len(all_cnt)),
           "p_table_all_rows": float(rb[:, 0].mean()), "one_epoch": stats(all_cnt, sizes), "first_steps_batches": stats(all_cnt[:steps], sizes[:steps]),
           "note": "ChunkedSampler.epoch(0) replay; the run's batches are the first `steps` of epoch 0 (it never reaches epoch 1 when steps < epoch_batches); DAgger / ego stages do not change the NOW tick or nav_rb"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--argv-json", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--join-train", default="")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    tok = json.load(open(a.argv_json, encoding="utf-8"))
    steps = a.steps
    if steps is None:
        steps = int(tok[tok.index("--steps") + 1]) if "--steps" in tok else None
    r = replay(tok, a.table, steps, a.join_train or None)
    s = json.dumps(r, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    o, f = r["one_epoch"], r["first_steps_batches"]
    print(f"windows {r['n_windows']} clips {r['n_clips_trained']} batches/epoch {r['epoch_batches']} (run covers {r['run_covers_epoch_share']:.3f} of epoch 0)")
    print(f"table p (all rows) {r['p_table_all_rows']:.6f}; epoch p {o['p']:.6f}; first-{r['steps']} p {f['p']:.6f}; batches with no RB {f['share_batches_with_no_rb']:.4f}; leftover batches {o['leftover_batches']} (sizes {o['leftover_sizes']}, no RB in {o['leftover_batches_with_no_rb']})")
    print(f"K-SHARE (first {r['steps']} batches): w=2 {f['k_share_w2']:.4f}  w=4 {f['k_share_w4']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
