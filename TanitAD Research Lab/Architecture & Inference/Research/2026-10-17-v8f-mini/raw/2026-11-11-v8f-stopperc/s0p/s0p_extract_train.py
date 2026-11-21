"""s0p_extract_train.py -- S0-P step 2: TRAIN-side extraction (card section 8 S0-P; AMENDMENT_1_S0.md).  Built for box 2's CPU; CPU only, no model pass, no training.

What it does (the card's rule, fixed here):
  1. Pool = STOPBAL's prevalence rule (`stopbal/prevalence_stopbal.py`, imported by logic not by file): the join's clips minus `--exclude 540:600` (sorted by sha12), `select_train_clips(.., 1.0)`,
     `train_windows(hist 4)` (t in [3, T-31)); goal-observed = has & goal_w bit 6; positive = observed & goal_y bit 6; negative = observed & not positive.
     Asserted against the banked counts 286,035 / 13,116 / 272,919 (`--expect-pool`, default those; `none` skips -- tests and smokes only).
  2. Sample = `s0p_lib.draw_sample` (default_rng(20261111), positives first, sorted by (clip id, window start), 4,000 without replacement; then 12,000 negatives the same way).
  3. For each sampled window: E exactly as the trainer builds the item (`V8FMiniWindows.item` through a ClipStore with the trainer's own kin_mode yaw + native tracks): p10f last row, nav last row;
     limit / limit_ok = `v8f_t1_eval.causal_bucket(v, t, 20)` (AMENDMENT 1: the causal bucket on both sides; the logged sidecar is never read).  The NOW frame's `pool4x10` / `pool8x20` / `mean` rows are read
     from the clip file (memory-mapped, only the sampled rows), stored as the raw fp8 bytes (mean fp16).
  4. Controls (printed and written into meta.json): pool counts vs the banked ones; second route for E (ClipStore.get -> p10f[t], J["nav"][row], causal_bucket) max |diff| per block; NOW pool4x10 row vs the
     item's feats is not available here (the grid is never loaded) -> the stored-bytes route is the only one; train / eval clip ids disjoint (D.assert_disjoint, 0 shared); the fraction of windows whose causal bucket is not ok.

Output: a DIRECTORY (`--out-dir`) of .npy files (E, v40, v160, mean, y, clip, ws, eid), `meta.json`, `progress.json` and `MANIFEST.sha256`.  The same field names and layouts as the eval npz (`s0p_lib.FIELDS`).
RESUMABLE: rows are written to open_memmap files and `progress.json` records the number of finished clips; a power cut loses at most 100 clips.

Box 2 (do NOT launch from here; the Master Mind does), from `D:\\tanitad\\repo`:
  D:\\tanitad\\venv\\Scripts\\python.exe <repo>\\...\\s0p_extract_train.py --tree D:\\tanitad\\repo --cache-dir D:\\tanitad\\cache_full --split train ^
     --join D:\\tanitad\\runs\\v8f_mini_full\\join_train.npz --tracks D:\\tanitad\\cache_full\\native_tracks.pt --out-dir <OUT>\\s0p_train --eval-cache-dir D:\\tanitad\\cache_full
(`--dry-run` stops after the pool + sample and writes only sample.json.)
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import s0p_lib as L  # noqa: E402

BANKED_POOL = (286035, 13116, 272919)          # prevalence_stopbal.json goal_stop n_observed / n_pos / n_neg
CKPT_EVERY = 100


def parse_exclude(s):
    if not s or s.lower() == "none":
        return None
    a, b = s.split(":")
    return int(a), int(b)


def parse_shard(s):
    if not s:
        return None
    i, k = (int(x) for x in s.split("/"))
    assert k >= 1 and 0 <= i < k, f"--shard {s}: need 0 <= i < K"
    return i, k


def shard_bounds(n, i, k):
    """contiguous slice i of k of n sorted windows; the k slices tile [0, n) exactly."""
    return (i * n) // k, ((i + 1) * n) // k


def build_pool(J, D, exclude):
    """-> clips (list, train clips in select order), wins (list of (ci, t, 0, 0)), clip_of_win (str array), ws (int array), pos, neg (bool arrays), info dict."""
    clips_all = [str(c) for c in J["clip"].tolist()]
    ex = set()
    if exclude:
        ex = set(sorted(clips_all, key=D.sha12)[exclude[0]:exclude[1]])
    clips = [c for c in clips_all if c not in ex]
    sel = D.select_train_clips(clips, 1.0)
    wins = D.train_windows(None, J, sel, hist=4, start=None)
    ci = np.array([w[0] for w in wins], np.int64)
    ws = np.array([w[1] for w in wins], np.int64)
    rows = np.array([int(J["off"][c]) + int(t) for c, t in zip(ci, ws)], np.int64)
    has = J["has"][rows].astype(bool)
    gy, gw = J["goal_y"][rows].astype(np.int64), J["goal_w"][rows].astype(np.int64)
    obs = has & (((gw >> L.GOAL_STOP_BIT) & 1) == 1)
    pos = obs & (((gy >> L.GOAL_STOP_BIT) & 1) == 1)
    neg = obs & ~pos
    clip_of = np.asarray([str(J["clip"][c]) for c in ci])
    info = {"n_clips_all": len(clips_all), "n_excluded": len(ex), "n_clips_train": len(sel), "n_windows": len(wins), "n_observed": int(obs.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    return sel, wins, clip_of, ws, ci, pos, neg, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree"), help="repo root of the code of record (box 2: D:\\tanitad\\repo)")
    ap.add_argument("--cache-dir", required=True, help="the dir holding <split>/clips/*.pt (box 2: D:\\tanitad\\cache_full)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--join", required=True, help="the join npz the trainer names with --join-train")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude", default="540:600")
    ap.add_argument("--n-pos", type=int, default=L.N_POS)
    ap.add_argument("--n-neg", type=int, default=L.N_NEG)
    ap.add_argument("--seed", type=int, default=L.SEED)
    ap.add_argument("--expect-pool", default=",".join(map(str, BANKED_POOL)), help="n_observed,n_pos,n_neg, or 'none' (tests / smokes only)")
    ap.add_argument("--eval-cache-dir", default=None, help="a dir holding eval/clips/*.pt: train/eval clip ids must be disjoint (D.assert_disjoint). Skipped with --split eval.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/K: after the deterministic sample is drawn, process only contiguous slice i of K of the sorted sample (window i*N//K .. (i+1)*N//K) into <out-dir>/shard_<i>/. Concatenated in order, the K shards equal an unsharded run byte for byte.")
    a = ap.parse_args(argv)
    shard = parse_shard(a.shard)

    tree = L.setup_tree(a.tree)
    from tanitad.data import v8f_mini_data as D
    import torch
    E = L.load_eval_module(tree)
    J = D.load_join(a.join)
    clips, wins, clip_of, ws_all, ci_all, pos, neg, info = build_pool(J, D, parse_exclude(a.exclude))
    print("pool", json.dumps(info), flush=True)
    ctl = {"pool": info, "amendment": "AMENDMENT_1_S0.md: limit = causal bucket both sides", "causal_steps": E.CAUSAL_STEPS, "seed": a.seed}
    if a.expect_pool.lower() != "none":
        exp = tuple(int(x) for x in a.expect_pool.split(","))
        got = (info["n_observed"], info["n_pos"], info["n_neg"])
        assert got == exp, f"pool counts {got} != banked {exp} (prevalence_stopbal.json): the pool rule drifted"
        ctl["pool_equals_banked_prevalence"] = True
    if a.eval_cache_dir and a.split != "eval":
        ev = [c for c, _ in D.read_manifest(a.eval_cache_dir, "eval")]
        ctl["disjoint"] = D.assert_disjoint(clips, ev)
        assert ctl["disjoint"]["overlap"] == 0
    ip, ineg = L.draw_sample(clip_of, ws_all, pos, neg, seed=a.seed, n_pos=a.n_pos, n_neg=a.n_neg)
    sel = np.sort(np.concatenate([ip, ineg]))                       # pool order = clip by clip: contiguous reads
    y = pos[sel].astype(np.uint8)
    assert int(y.sum()) == len(ip) and len(sel) == len(ip) + len(ineg)
    ctl["n_sample"], ctl["n_pos_sample"], ctl["n_neg_sample"] = int(len(sel)), int(len(ip)), int(len(ineg))
    sample_sha = hashlib.sha256((json.dumps([[str(clip_of[i]), int(ws_all[i])] for i in sel]) + str(y.tolist())).encode()).hexdigest()
    ctl["sample_sha256"] = sample_sha
    prog_key = sample_sha
    if shard:
        lo_s, hi_s = shard_bounds(len(sel), *shard)
        ctl["shard"] = {"i": shard[0], "k": shard[1], "lo": lo_s, "hi": hi_s, "n_total": int(len(sel))}
        prog_key = f"{sample_sha}#shard{shard[0]}/{shard[1]}"
        if a.dry_run:                                               # the full sample, once, at the root: pool + sample + counts only
            os.makedirs(a.out_dir, exist_ok=True)
            with open(os.path.join(a.out_dir, "sample.json"), "w", encoding="utf-8") as f:
                json.dump({"clip": [str(clip_of[i]) for i in sel], "ws": [int(ws_all[i]) for i in sel], "y": y.tolist(), "sha256": sample_sha}, f)
            ctl["shard"]["n_shard_pos"] = int(y[lo_s:hi_s].sum())
        sel, y = sel[lo_s:hi_s], y[lo_s:hi_s]
        a.out_dir = os.path.join(a.out_dir, f"shard_{shard[0]}")
        ctl["n_shard"], ctl["n_pos_shard"] = int(len(sel)), int(y.sum())
        assert len(sel) > 0, "empty shard"
    os.makedirs(a.out_dir, exist_ok=True)
    if not shard or not a.dry_run:                                  # a shard's own sample.json = its slice; sample_sha256 is the FULL sample's
        with open(os.path.join(a.out_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump({"clip": [str(clip_of[i]) for i in sel], "ws": [int(ws_all[i]) for i in sel], "y": y.tolist(), "sha256": sample_sha}, f)
    if a.dry_run:
        print(json.dumps(ctl, indent=1))
        return ctl

    tracks = D.load_native_tracks(a.tracks)
    proxy = L.MmapProxyLoad(grid="pool4x10")
    store = D.ClipStore(a.cache_dir, a.split, lru=2, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, J, [], hist=4, hmax=16, max_age_t=2, max_age_s=6, window_start=None)
    cdir = os.path.join(a.cache_dir, a.split, "clips")
    N = len(sel)
    # open the arrays (resumable)
    prog_path = os.path.join(a.out_dir, "progress.json")
    prog = json.load(open(prog_path, encoding="utf-8")) if os.path.exists(prog_path) else {"n_clips_done": 0, "sample_sha256": prog_key, "c2": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}, "n_limit_not_ok": 0}
    assert prog["sample_sha256"] == prog_key, "progress.json belongs to a different sample"
    mode = "r+" if prog["n_clips_done"] > 0 and os.path.exists(os.path.join(a.out_dir, "E.npy")) else "w+"
    if mode == "w+":
        prog = {"n_clips_done": 0, "sample_sha256": prog_key, "c2": {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}, "n_limit_not_ok": 0}
    mm = {}
    for k, shape, dt in (("E", (N, L.E_DIM), np.float32), ("v40", (N, 40, 1024), np.uint8), ("v160", (N, 160, 1024), np.uint8), ("mean", (N, 1024), np.float16)):
        mm[k] = np.lib.format.open_memmap(os.path.join(a.out_dir, k + ".npy"), mode=mode, dtype=dt, shape=shape)
    # per-clip groups in the sample (sel is in pool order = clip order)
    sel_clips = clip_of[sel]
    bounds = [0] + [i for i in range(1, N) if sel_clips[i] != sel_clips[i - 1]] + [N]
    groups = [(int(bounds[g]), int(bounds[g + 1])) for g in range(len(bounds) - 1)]
    t0 = time.time()
    first = True
    for gi in range(prog["n_clips_done"], len(groups)):
        lo, hi = groups[gi]
        clip = str(sel_clips[lo])
        with proxy:
            d = store.get(clip)                                     # the trainer's per-clip arrays (grid replaced by zeros: never read)
            real = proxy.last[str(store.dir / f"{clip}.pt")]
            if first:
                for k in ("pool4x10", "pool8x20", "mean"):
                    if k not in real:
                        raise SystemExit(f"NO-GRID: {k} absent from {clip}.pt: the train cache lacks it (card section 4 S0-PRE: PI question, no re-encode approved)")
                first = False
            for i in range(lo, hi):
                ci, t = int(ci_all[sel[i]]), int(ws_all[sel[i]])
                ds.store = _One(d)
                it = ds.item(ci, t, 0, 0)
                lim, lok = E.causal_bucket(d["v"], t, E.CAUSAL_STEPS)
                eb = L.e_from_item_arrays(it["p10f"][-1:], it["nav"][-1:], lim[None], np.array([lok]))
                mm["E"][i] = eb[0]
                mm["v40"][i] = real["pool4x10"][t].view(torch.uint8).numpy()
                mm["v160"][i] = real["pool8x20"][t].view(torch.uint8).numpy()
                mm["mean"][i] = real["mean"][t].numpy().astype(np.float16)
                # second route (C2)
                row = int(J["off"][ci]) + t
                p10 = np.nan_to_num(d["p10f"][t], nan=0.0).astype(np.float32)
                c2 = prog["c2"]
                c2["p10f"] = max(c2["p10f"], float(np.abs(eb[0, :8] - p10).max()))
                c2["nav"] = max(c2["nav"], float(np.abs(eb[0, 8:13] - np.asarray(J["nav"][row], np.float32)).max()))
                c2["limit"] = max(c2["limit"], float(np.abs(eb[0, 13:17] - lim).max()))
                c2["limit_ok"] = max(c2["limit_ok"], float(abs(eb[0, 17] - float(lok))))
                prog["n_limit_not_ok"] += int(not lok)
            del real
        proxy.last.clear()
        prog["n_clips_done"] = gi + 1
        if (gi + 1) % CKPT_EVERY == 0 or gi + 1 == len(groups):
            for m in mm.values():
                m.flush()
            json.dump(prog, open(prog_path, "w", encoding="utf-8"))
            print(f"  [{gi + 1}/{len(groups)} clips] {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out_dir, "y.npy"), y)
    np.save(os.path.join(a.out_dir, "clip.npy"), np.asarray(clip_of[sel]).astype("U40"))
    np.save(os.path.join(a.out_dir, "ws.npy"), ws_all[sel].astype(np.int64))
    np.save(os.path.join(a.out_dir, "eid.npy"), np.asarray([D.stable_episode_id(str(c)) for c in clip_of[sel]], np.int64))
    ctl["c2_second_route_max_abs"] = prog["c2"]
    ctl["causal_limit_ok_false_fraction"] = prog["n_limit_not_ok"] / N
    ctl["n_clips_in_sample"] = len(groups)
    ctl["E_layout"] = "p10f_last(8) nav_last(5) limit_causal(4) limit_ok_causal(1)"
    ctl["elapsed_s"] = round(time.time() - t0, 1)
    ctl["split"] = a.split
    with open(os.path.join(a.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(ctl, f, indent=1, sort_keys=True)
    for m in mm.values():
        m.flush()
    del mm
    names = ["E.npy", "v40.npy", "v160.npy", "mean.npy", "y.npy", "clip.npy", "ws.npy", "eid.npy", "meta.json", "sample.json"]
    with open(os.path.join(a.out_dir, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{L.sha256_file(os.path.join(a.out_dir, n))}  {n}\n")
    print(json.dumps(ctl, indent=1), flush=True)
    return ctl


class _One:
    """ClipStore stand-in serving one clip's dict to V8FMiniWindows.item."""

    def __init__(self, d):
        self.d = d

    def get(self, _clip):
        return self.d


if __name__ == "__main__":
    main()
