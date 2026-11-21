"""PREREG_V8F_DSCALE, "Proofs, before any run": the REACHABILITY of `--data-frac 0.5`, MEASURED by running the trainer's OWN `build_data` on the REAL train clip list. CPU only, no GPU, no box 2.

  python measure_dscale.py [--manifest <cache_full/train/manifest.jsonl>] [--v2-manifest <train4572_v2manifest.pt>] [--out measure_dscale.json]

What it does (nothing here restates the trainer: the numbers come out of `v8f_mini_train.build_data`, `D.select_train_clips` and `D.train_windows` themselves):
  1. takes the argv the QUEUE prints for B (`train M2b 0`) and for Q (`traindscale 0`) -- the actual caller -- and changes only the PATH flags to stubs;
  2. writes a stub cache (a `train/manifest.jsonl` of the real clip ids and their frame counts, a stub join and a stub tracks file), so `build_data` runs its real code path;
  3. calls `TR.build_data(a, cfg)` for B (data_frac 1.0) and Q (data_frac 0.5) and reads back n_clips, n_clips_all, n_clips_expected, train_sha256, excluded_sha256, the window counts,
     then sizes the step budget the way `main` does (S = ceil(n_windows_all_clips / batch), total = S x steps-mult);
  4. checks every number against B's banked record (`raw/full/runs/M2b_s0/summary.json`) -- the whole pipeline is only trusted if it REPRODUCES B -- and the nesting: Q's clips are the
     first round(0.5 x N_B) of B's list in sha12 order.

THE CLIP LIST. The trainer reads `<cache>/train/manifest.jsonl` (box 2: D:\\tanitad\\cache_full). This machine does not hold that file outside `box2_ship` (off limits to this build), so the default
source is `train4572_v2manifest.pt` (`C:/Users/Admin/tanitad-data/add203/in/`), whose `clip_id` list is the same 4,572 clips. Its IDENTITY is MEASURED, not assumed: after the trainer's
`--exclude 540:600` the sha256 of the sorted 4,512-clip list and of the 60 excluded ids reproduce B's recorded `train_sha256` and `excluded_sha256` EXACTLY (asserted below).
Its FRAME COUNTS are DERIVED (the v2 file has raw frame counts `T_out`, the cache manifest has `n_frames`): n_frames = ceil(T_out / 2) + 1, a relation checked against B's record -- the
resulting n_windows_all_clips must equal B's recorded 306,566 and the budget its 19,164 steps, exactly (asserted). With `--manifest <the cache manifest>` the frame counts are exact and nothing is derived.
UNVERIFIED stays attached to the per-clip frame counts (not to the clip list) until the run is made with the cache manifest; the chain's gate re-checks every window count on box 2's own manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[2]
REPO = HERE.parents[6]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
QUEUE = PKG / "code" / "v8f_mini_box2_queue.cmd"
BANKED_B = PKG / "raw" / "full" / "runs" / "M2b_s0" / "summary.json"
DEFAULT_V2 = "C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt"
PATH_FLAGS = ("--cache-dir", "--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--out")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_list(ids):
    return hashlib.sha256(chr(10).join(ids).encode("utf-8")).hexdigest()


def queue_argv(stage_args):
    """The argv the queue PRINTS for a stage (V8F_DRYRUN=1), python and script dropped. Windows only: the queue is a batch file."""
    env = dict(os.environ, V8F_DRYRUN="1")
    r = subprocess.run(f'cmd /c ""{QUEUE}" {stage_args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    lines = [x for x in r.stdout.splitlines() if ".py" in x]
    if r.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"the queue printed no single command for {stage_args!r} (rc {r.returncode}): {r.stdout[-300:]}")
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', lines[0].strip())][2:]


def with_stub_paths(argv, cache, join, tracks, out):
    argv = list(argv)
    for flag, val in (("--cache-dir", cache), ("--join-train", join), ("--tracks", tracks), ("--out", out)):
        argv[argv.index(flag) + 1] = str(val)
    return argv


def load_clips(a):
    """-> (ids in file order, n_frames list, source dict)."""
    if a.manifest:
        ents = {}
        for ln in open(a.manifest, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                ents[d["clip"]] = int(d["n_frames"])
        return list(ents), list(ents.values()), {"kind": "cache manifest.jsonl (n_frames EXACT)", "path": str(a.manifest), "sha256": sha256_file(a.manifest), "n_frames": "exact"}
    import torch
    o = torch.load(a.v2_manifest, map_location="cpu", weights_only=True)
    ids, t = list(o["clip_id"]), [int(x) for x in o["T_out"]]
    nf = [(x + 1) // 2 + 1 for x in t]
    return ids, nf, {"kind": "train4572_v2manifest.pt (clip_id exact, n_frames DERIVED = ceil(T_out / 2) + 1)", "path": str(a.v2_manifest), "sha256": sha256_file(a.v2_manifest),
                     "n_frames": "derived; validated by reproducing B's n_windows_all_clips exactly"}


def build(TR, D, argv, ids, nf, tmp):
    """The trainer's own build_data on a stub cache holding the real clip list; returns its info, the step budget and the window list."""
    cache = Path(tmp) / "cache"
    (cache / "train").mkdir(parents=True, exist_ok=True)
    with open(cache / "train" / "manifest.jsonl", "w", encoding="utf-8") as f:
        for c, n in zip(ids, nf):
            f.write(json.dumps({"clip": c, "n_frames": int(n)}) + "\n")
    n = np.asarray(nf, np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    join = Path(tmp) / "join_stub.npz"
    np.savez(join, clip=np.array(ids), off=off, n=n, n_clips_missing_labels=np.int64(0))
    import torch
    tracks = Path(tmp) / "tracks_stub.pt"
    torch.save({"tracks": {ids[0]: torch.zeros(4, 4)}}, tracks)
    a = TR.build_parser().parse_args(with_stub_paths(argv, cache, join, tracks, Path(tmp) / "out"))
    cfg = TR.make_config(a)
    store, J, wins, clips, info = TR.build_data(a, cfg)
    S = math.ceil(info["n_windows_all_clips"] / a.batch)
    total = a.steps or S * a.steps_mult
    sha12 = D.sha12
    return {"argv_flags": {k: getattr(a, k) for k in ("arm", "seed", "data_frac", "steps_mult", "batch", "exclude", "hist", "window_start", "grid")},
            "n_clips": info["n_clips"], "n_clips_all": info["n_clips_all"], "n_clips_expected": info["n_clips_expected"],
            "n_windows_trained": info["n_windows"], "n_windows_all_clips": info["n_windows_all_clips"], "S": S, "steps_total": total,
            "train_sha256": info["train_sha256"], "excluded_sha256": info["excluded_sha256"], "n_excluded": info["n_excluded"], "excluded_in_train": info["excluded_in_train"],
            "window_t_min": info["window_t_min"], "first_sha12": sha12(sorted(clips, key=sha12)[0]), "last_sha12": sha12(sorted(clips, key=sha12)[-1]),
            "passes_per_window": total * a.batch / max(len(wins), 1)}, clips, wins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=None, help="the cache's train/manifest.jsonl (exact n_frames); default: --v2-manifest")
    ap.add_argument("--v2-manifest", default=DEFAULT_V2)
    ap.add_argument("--out", default=str(HERE.with_suffix(".json")))
    a = ap.parse_args(argv)
    import v8f_mini_train as TR
    from tanitad.data import v8f_mini_data as D
    b_rec = json.loads(BANKED_B.read_text(encoding="utf-8"))
    ids, nf, src = load_clips(a)
    res = {"schema": "v8f-dscale-measure-v1", "source": src, "n_manifest_clips": len(ids), "banked_b": {k: b_rec[k] for k in ("n_clips", "n_windows", "S", "steps", "train_sha256", "excluded_sha256", "data_frac")},
           "queue_sha256": sha256_file(QUEUE), "checks": {}}
    with tempfile.TemporaryDirectory(prefix="dscale_measure_") as tmp:
        argv_b = queue_argv("train M2b 0")
        argv_q = queue_argv("traindscale 0")
        res["queue_argv"] = {"B (train M2b 0)": argv_b, "Q (traindscale 0)": argv_q}
        res["checks"]["queue_argv_q_is_b_plus_data_frac_0.5_only"] = bool(_flagdict(argv_q, {"--data-frac", "--out"}) == _flagdict(argv_b, {"--out"}) and _flagdict(argv_q, set())["--data-frac"] == "0.5"
                                                                      and "--data-frac" not in _flagdict(argv_b, set()))
        os.makedirs(tmp + "/b")
        os.makedirs(tmp + "/q")
        B_, clips_b, wins_b = build(TR, D, argv_b, ids, nf, tmp + "/b")
        Q_, clips_q, wins_q = build(TR, D, argv_q, ids, nf, tmp + "/q")
    res["B"], res["Q"] = B_, Q_
    n_b, half = B_["n_clips"], int(round(0.5 * B_["n_clips"]))
    order_b = sorted(clips_b, key=D.sha12)
    first_half = order_b[:half]
    ck = res["checks"]
    ck["B_reproduces_banked_record"] = {k: (B_[k2] == b_rec[k], B_[k2], b_rec[k]) for k, k2 in (("n_clips", "n_clips"), ("n_windows", "n_windows_trained"), ("S", "S"), ("steps", "steps_total"),
                                                                                                  ("train_sha256", "train_sha256"), ("excluded_sha256", "excluded_sha256"))}
    ck["B_n_windows_all_clips_equals_banked"] = bool(B_["n_windows_all_clips"] == b_rec["n_windows"])
    ck["Q_n_clips_is_round_half_of_N_B"] = bool(Q_["n_clips"] == half == Q_["n_clips_expected"] and Q_["n_clips_all"] == n_b == 4512)
    ck["Q_clips_are_the_first_half_of_B_in_sha12_order"] = bool(sorted(clips_q, key=D.sha12) == first_half and Q_["train_sha256"] == sha_list(first_half))
    ck["Q_nested_in_B"] = bool(set(clips_q) <= set(clips_b))
    ck["Q_steps_are_B"] = bool(Q_["S"] == B_["S"] == b_rec["S"] and Q_["steps_total"] == B_["steps_total"] == b_rec["steps"] == 19164)
    ck["Q_window_budget_is_sized_from_all_clips"] = bool(Q_["n_windows_all_clips"] == B_["n_windows_all_clips"] and Q_["n_windows_trained"] < B_["n_windows_trained"])
    ck["Q_excluded_set_is_B"] = bool(Q_["excluded_sha256"] == B_["excluded_sha256"] == b_rec["excluded_sha256"] and Q_["excluded_in_train"] == 0)
    ck["Q_sees_each_window_about_twice_as_often"] = {"B": B_["passes_per_window"], "Q": Q_["passes_per_window"], "ratio": Q_["passes_per_window"] / B_["passes_per_window"]}
    ck["PASS"] = bool(all(v[0] for v in ck["B_reproduces_banked_record"].values()) and all(v is True for k, v in ck.items() if isinstance(v, bool)))
    res["half_facts"] = {"N_B": n_b, "round_half": half, "first_sha12": D.sha12(first_half[0]), "last_sha12": D.sha12(first_half[-1]), "first_clip": first_half[0], "last_clip": first_half[-1],
                         "half_train_sha256": sha_list(first_half), "n_windows_Q": Q_["n_windows_trained"], "n_windows_B": B_["n_windows_trained"],
                         "q_over_b_windows": Q_["n_windows_trained"] / B_["n_windows_trained"]}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"source: {src['kind']} (sha256 {src['sha256'][:16]})")
    print(f"B (data_frac 1.0): clips {B_['n_clips']} of {B_['n_clips_all']}, windows {B_['n_windows_trained']}, S {B_['S']}, steps {B_['steps_total']}, train_sha256 {B_['train_sha256'][:16]}, excluded_sha256 {B_['excluded_sha256'][:16]}")
    print(f"Q (data_frac 0.5): clips {Q_['n_clips']} of {Q_['n_clips_all']} (expected {Q_['n_clips_expected']}), windows {Q_['n_windows_trained']} (all clips {Q_['n_windows_all_clips']}), S {Q_['S']}, "
          f"steps {Q_['steps_total']}, first/last sha12 {res['half_facts']['first_sha12']} / {res['half_facts']['last_sha12']}, train_sha256 {Q_['train_sha256'][:16]}")
    print(f"passes per window: B {B_['passes_per_window']:.3f}, Q {Q_['passes_per_window']:.3f} (x{ck['Q_sees_each_window_about_twice_as_often']['ratio']:.3f})")
    for k, v in ck.items():
        print(f"  {k}: {v}")
    print("MEASURE", "PASS" if ck["PASS"] else "FAIL")
    return 0 if ck["PASS"] else 1


def _flagdict(argv, drop):
    out, i = {}, 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[t] = argv[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return {k: v for k, v in out.items() if k not in drop}


if __name__ == "__main__":
    sys.exit(main())
