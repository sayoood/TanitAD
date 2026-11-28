"""PREREG_V8F_HIST Amendment 2 input, MEASURED (CPU only): how many of the 2,399 REAL eval windows would have had different held ages for an H checkpoint (hist 10 + window-start 3)
if the evaluator had NOT been given `window_start` (clamp origin hist - 1 = 9 instead of 3).

  python measure_a2fix.py [--grid <eval_grid_windows.json>] [--out measure_a2fix.json]     (PYTHONPATH must reach <repo>/stack)

The ages are `v8f_mini_eval.py cmd_run`'s own `--ages cycle` rule (the DEFAULT): window i of the grid (grid["order"] x grid["windows"][clip], in that order, which is what
`eval_grid_windows` returns) is held at a_t = i % 3, a_s = a_t + (i // 3) % 5 (`v8f_mini_eval.py:291-295`). The effective ages are what the REAL `V8FMiniWindows.item` returns after its
clamp (`room = max(t - window_start, 0)`, a_t <= min(room, 2), a_s <= min(room, 6), a_s >= a_t), on a synthetic clip long enough for every t (the clamp reads only t and the origin),
for the dataset the FIXED evaluator builds (hist 10, window_start 3: what H trained with) and the dataset the UNFIXED one built (hist 10, window_start absent = 9).
A pure-arithmetic re-derivation of the clamp is checked equal on every window, so the count does not rest on one implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[7]
for p in (REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402

DEFAULT_GRID = HERE.parents[1].parent / "eval_grid_windows.json"          # <PK>/raw/eval_grid_windows.json


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cycle_ages(i):
    """`v8f_mini_eval.py:291-295` (--ages cycle), restated."""
    a_t = i % 3
    return a_t, a_t + (i // 3) % 5


def clamp(t, a_t, a_s, ws, max_age_t=2, max_age_s=6):
    """`V8FMiniWindows.item`'s clamp, re-derived (the cross-check)."""
    room = max(t - ws, 0)
    a_t = int(min(a_t, room, max_age_t))
    return a_t, int(max(a_t, min(a_s, room, max_age_s)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    ap.add_argument("--out", default=str(HERE.with_suffix(".json")))
    a = ap.parse_args(argv)
    g = json.load(open(a.grid, encoding="utf-8"))
    order = list(g["order"])
    wins = [(ci, int(t)) for ci, c in enumerate(order) for t in g["windows"][c]]            # eval_grid_windows' order (ci = position in `order` here; only t matters for the clamp)
    assert len(wins) == g["n_windows"] == 2399, (len(wins), g["n_windows"])
    tmax = max(t for _, t in wins)

    root = Path(tempfile.mkdtemp(prefix="v8fhist_a2fix_"))                                 # left behind on purpose (nothing is deleted); a few KB of synthetic clips
    D.make_synthetic_world(root, n_train=1, n_eval=1, T=tmax + 40, seed=0)
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=2, kin_mode="yaw", tracks=D.load_native_tracks(root / "native_tracks.pt"))
    fixed = D.V8FMiniWindows(store, J, [], hist=10, hmax=16, window_start=3)                # the evaluator AFTER the fix, for an H checkpoint (cfg.window_start == 3)
    unfixed = D.V8FMiniWindows(store, J, [], hist=10, hmax=16)                              # the evaluator BEFORE the fix (no window_start passed)
    assert (fixed.window_start, unfixed.window_start) == (3, 9)

    diff, by_t, n_item_diff = [], Counter(), 0
    n_rederive_bad = 0
    for i, (_, t) in enumerate(wins):
        a_t, a_s = cycle_ages(i)
        f, u = fixed.item(0, t, a_t, a_s), unfixed.item(0, t, a_t, a_s)
        ef, eu = (f["a_t"], f["a_s"]), (u["a_t"], u["a_s"])
        n_rederive_bad += (ef != clamp(t, a_t, a_s, 3)) or (eu != clamp(t, a_t, a_s, 9))
        if ef != eu:
            diff.append(i)
            by_t[t] += 1
            n_item_diff += not (np.array_equal(f["pose_s"], u["pose_s"]) and np.array_equal(f["pose_t"], u["pose_t"]))
    n_t = Counter(t for _, t in wins)
    n_t_lt15 = sum(v for k, v in n_t.items() if k < 15)
    res = {"schema": "v8f-hist-a2fix-measure-v1", "grid": {"path": str(a.grid), "sha256": sha256(a.grid), "n_windows": len(wins), "n_episodes": len(order),
                                                         "t_min": min(t for _, t in wins), "t_max": tmax, "t_values_below_15": sorted(k for k in n_t if k < 15)},
           "rule": "v8f_mini_eval.py:291-295 --ages cycle (default): a_t = i % 3, a_s = a_t + (i // 3) % 5, i = position in eval_grid_windows' list",
           "origins": {"fixed_H": 3, "unfixed_H": 9},
           "n_windows_with_different_effective_held_ages": len(diff), "fraction": len(diff) / len(wins),
           "by_t": {str(k): {"n_windows_at_t": n_t[k], "n_different": by_t[k]} for k in sorted(n_t) if k < 15},
           "n_windows_at_t_below_15": n_t_lt15,
           "n_of_those_whose_tick_poses_also_differ_on_the_synthetic_clip": n_item_diff,
           "arithmetic_rederivation_disagreements": int(n_rederive_bad),
           "t1_note": "v8f_t1_eval.py rolls each window forward and draws ages_for(step) = (step % 3, step % 6) at j = t0 + step; the same clamp applies at every step, so the count above is "
                      "the --ages cycle read only (the T1 read was affected at every early window and step; not counted here)."}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("grid:", res["grid"]["n_windows"], "windows,", res["grid"]["n_episodes"], "episodes, t =", res["grid"]["t_min"], "..", res["grid"]["t_max"])
    print("windows at t < 15:", n_t_lt15, "| with DIFFERENT effective held ages without the fix:", len(diff), f"({100 * len(diff) / len(wins):.2f} %)")
    print("by t:", res["by_t"])
    print("arithmetic re-derivation disagrees on", n_rederive_bad, "windows")
    return 0 if n_rederive_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
