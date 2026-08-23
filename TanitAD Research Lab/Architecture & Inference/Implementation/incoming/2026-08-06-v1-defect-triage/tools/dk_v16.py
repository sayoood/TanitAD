"""Distance-keeping for v1.6 (flagship-v16-unicycle) — closes the family half that
`v16_full_eval.json` reports UNAVAILABLE.

Join: the 40-episode lead block (`build_lead_block.py`, stride-8 grid, 880 rows) attaches
to the STRIDE-8 SUBSET of the banked stride-1 eval dump (`/workspace/v16_eval/dump/ep*.npz`).
`lead_source.window_last_indices(T)` reproduces the grid; each npz's `ws` holds every
stride-1 window origin, so the stride-8 origins must be an exact subset — asserted, not
assumed. NO re-inference: the paths scored here are byte-identical to the ones already
scored for every other family.

Arms: a = v1arch step-readout, b = v1.6 unicycle readout, g = GT (reference — GT vs the
same lead tracks bounds what "correct" distance-keeping looks like on this corpus).

Estimator: paired episode-cluster bootstrap via `lead_metrics.paired_distance_keeping`.
"""
import glob
import json

import numpy as np

import sys
sys.path.insert(0, "/workspace/TanitAD/taniteval")
from taniteval import lead_source as ls
from taniteval.lead_metrics import (distance_keeping, distance_keeping_by_speed,
                                    paired_distance_keeping)

import torch
blk = torch.load("/workspace/v16_eval/lead_block_40.npz", map_location="cpu", weights_only=False)
leads, lead_lens = blk["leads"], blk["lead_lens"]
speeds, state, eid = blk["speeds"], blk["state"], list(blk["eid"])
n_blk = leads.shape[0]
print(f"[block] {n_blk} rows, states "
      f"{ {s: int((np.asarray(state) == s).sum()) for s in set(state)} }", flush=True)

dumps = sorted(glob.glob("/workspace/v16_eval/dump/ep*.npz"))
assert len(dumps) == 40, len(dumps)

paths = {"v1arch": [], "v16": [], "gt": []}
row = 0
for fi, f in enumerate(dumps):
    d = np.load(f)
    ws = d["ws"]
    # T is not stored; invert the stride-1 grid instead: ws = w0..w0+n-1 with
    # w0 = WINDOW-1, n = T-WINDOW-K_MAX  =>  T = len(ws) + WINDOW + K_MAX.
    assert ws[0] == ls.WINDOW - 1 and np.all(np.diff(ws) == 1), f"ep{fi}: ws not stride-1"
    T = len(ws) + ls.WINDOW + ls.K_MAX
    grid = ls.window_last_indices(T)
    pos = np.searchsorted(ws, grid)
    assert np.array_equal(ws[pos], grid), f"ep{fi}: stride-8 grid not inside ws"
    # this episode owns the next len(grid) block rows — verify via eid runs
    assert all(e == eid[row] for e in eid[row:row + len(grid)]), f"ep{fi}: eid run mismatch"
    for k, key in (("a", "v1arch"), ("b", "v16"), ("g", "gt")):
        paths[key].append(d[k][pos, :, :2].astype(np.float64))
    row += len(grid)
assert row == n_blk, (row, n_blk)
print(f"[join] {row} rows matched across 40 episodes", flush=True)

eid_arr = eid
out = {"_grid": "stride-8 subset of the stride-1 v16 eval dump; "
                "lead block lead_block_40.npz (build_lead_block.py, 40 eps)",
       "_join_rows": int(row)}
dks = {}
for key in ("v1arch", "v16", "gt"):
    P = np.concatenate(paths[key])
    dk = distance_keeping(P, leads, lead_lens, speeds, dt=0.1)
    dks[key] = dk
    out[key] = {k: (v if isinstance(v, str) else float(v) if isinstance(v, (int, float, np.floating, np.integer)) else None)
                for k, v in dk.items() if not isinstance(v, np.ndarray)}
    out[key]["by_speed"] = distance_keeping_by_speed(dk, speeds, eid_arr, states=state)
    print(f"[{key}] n={dk['n']}  headway_min={dk.get('mean_headway_min_m')}  "
          f"time_gap={dk.get('mean_time_gap_min_s')}  ttc={dk.get('mean_min_ttc_s')}",
          flush=True)

out["paired_v16_minus_v1arch"] = paired_distance_keeping(
    dks["v1arch"], dks["v16"], eid_arr, names=("v1arch", "v16"))
out["paired_v16_minus_gt"] = paired_distance_keeping(
    dks["gt"], dks["v16"], eid_arr, names=("gt", "v16"))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(_clean(out), open("/workspace/v16_eval/v16_distance_keeping.json", "w"), indent=1)
print("DK_DONE", flush=True)
