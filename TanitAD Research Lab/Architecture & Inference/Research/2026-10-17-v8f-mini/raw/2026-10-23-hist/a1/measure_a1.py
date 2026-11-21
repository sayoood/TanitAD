"""PREREG_V8F_HIST Amendment 1, MEASURED (CPU only): what `--hist 10 --window-start 3` does to the window list, the step budget and the held ages, with the trainer's OWN functions.

  python measure_a1.py [--manifest <box 2 train manifest.jsonl>] [--out measure_a1.json]     (PYTHONPATH must reach <repo>/stack)

  W  the window list, S and the step budget on box 2's real train manifest (4,572 clips, minus `--exclude 540:600` in sha12 order = 4,512): hist 4 (B), hist 10 alone, hist 10 + window-start 3.
     `train_windows` is the trainer's function (v8f_mini_data.py); S = ceil(n / batch), total = S x steps_mult is v8f_mini_train.py's formula (proved against the trainer's selfcheck in
     stack/tests/test_v8f_window_start.py::test_the_budget_formula_is_the_trainers_own_on_the_synthetic_world).
  A  the held ages: on a synthetic world, `V8FMiniWindows.item` at hist 4 vs hist 10 + window-start 3 vs hist 10 alone, every t in 0..40 and every one of the sampler's 15 draws:
     every key of every item equal (hist 10 + window-start 3), and the cells where hist 10 ALONE differs.
UNVERIFIED is said, per item, where an input is not on this disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[7]
for p in (REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402

DEFAULT_MANIFEST = "C:/Users/Admin/box2_ship/logs/v8f_grid/manifest_train.jsonl"
DRAWS = [(a, a + k) for a in range(3) for k in range(5)]
B_BANKED = {"n_windows_all_clips": 306566, "S": 4791, "steps": 19164, "source": "raw/full/runs/M2b_s0/summary.json (n_windows, steps)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def windows_block(manifest):
    mp = Path(manifest)
    if not mp.exists():
        return f"UNVERIFIED: {manifest} is not on this disk"
    ents = {}
    with open(mp, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                ents[d["clip"]] = int(d["n_frames"])
    excluded = set(sorted(ents, key=D.sha12)[540:600])
    clips = [c for c in ents if c not in excluded]
    J = {"clip": np.array(list(ents)), "n": np.array([ents[c] for c in ents], np.int64)}
    budget = lambda n: {"n_windows": n, "S": math.ceil(n / 64), "steps": 4 * math.ceil(n / 64)}          # --batch 64 --steps-mult 4
    w4 = D.train_windows(None, J, clips, hist=4)
    w10 = D.train_windows(None, J, clips, hist=10)
    w10s = D.train_windows(None, J, clips, hist=10, start=3)
    return {"manifest": {"path": str(mp), "sha256": sha256(mp), "n_clips_total": len(ents), "n_clips_after_exclude": len(clips)},
            "B_hist4": budget(len(w4)), "H_hist10_alone": budget(len(w10)), "H_hist10_window_start3": budget(len(w10s)),
            "banked_b": B_BANKED, "hist4_equals_banked": budget(len(w4)) == {k: B_BANKED[k] if k != "n_windows" else B_BANKED["n_windows_all_clips"] for k in ("n_windows", "S", "steps")},
            "window_start3_list_equals_hist4_list": bool(w10s == w4), "hist10_alone_is_hist4_minus_t3_to_t8": bool(w10 == [w for w in w4 if w[1] >= 9]),
            "windows_lost_by_hist10_alone": len(w4) - len(w10), "steps_lost_by_hist10_alone": budget(len(w4))["steps"] - budget(len(w10))["steps"]}


def ages_block():
    root = Path(tempfile.mkdtemp(prefix="v8fhist_a1_"))             # left behind on purpose (nothing is deleted); a few KB of synthetic clips
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(root / "native_tracks.pt"))
    ds4 = D.V8FMiniWindows(store, J, [], hist=4, hmax=16)
    ds10s = D.V8FMiniWindows(store, J, [], hist=10, hmax=16, window_start=3)
    ds10 = D.V8FMiniWindows(store, J, [], hist=10, hmax=16)

    def same(x, y):
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, np.ndarray):
            return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
        return bool(x == y)
    n = bad_s = bad_alone = 0
    alone_t = set()
    for ci in range(len(w["names"]["train"])):
        for t in range(0, 41):
            for a_t, a_s in DRAWS:
                i4 = ds4.item(ci, t, a_t, a_s)
                n += 1
                bad_s += not same(ds10s.item(ci, t, a_t, a_s), i4)
                i10 = ds10.item(ci, t, a_t, a_s)
                if not same(i10, i4):
                    bad_alone += 1
                    alone_t.add(t)
    return {"world": str(root), "n_cells": n, "hist10_window_start3_cells_with_any_key_different_from_hist4": int(bad_s),
            "hist10_alone_cells_different_from_hist4": int(bad_alone), "hist10_alone_differing_t": sorted(alone_t)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=str(HERE.with_suffix(".json")))
    a = ap.parse_args(argv)
    res = {"schema": "v8f-hist-a1-measure-v1", "W": windows_block(a.manifest), "A": ages_block()}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    W, A = res["W"], res["A"]
    if isinstance(W, dict):
        print("W B (hist 4):", W["B_hist4"], "| equals banked:", W["hist4_equals_banked"])
        print("W H (hist 10 + window-start 3):", W["H_hist10_window_start3"], "| list == B's:", W["window_start3_list_equals_hist4_list"])
        print("W H (hist 10 alone):", W["H_hist10_alone"], "| windows lost", W["windows_lost_by_hist10_alone"], "steps lost", W["steps_lost_by_hist10_alone"])
    else:
        print("W", W)
    print("A", {k: v for k, v in A.items() if k != "world"})
    ok = isinstance(W, dict) and W["hist4_equals_banked"] and W["window_start3_list_equals_hist4_list"] and A["hist10_window_start3_cells_with_any_key_different_from_hist4"] == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
