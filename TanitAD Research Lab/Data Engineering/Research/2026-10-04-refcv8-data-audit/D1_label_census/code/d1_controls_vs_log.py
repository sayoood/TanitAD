"""Known-value controls against the TRAINING RUN'S OWN LOG (independent of the census code):

  K1  the 128 windows the in-run eval scored (`torch.randperm(N, seed 12345)[:8*16]`, the trainer's own rule) -> the
      census table must reproduce the logged `eval_tacv6_n_supervised_goal_cells` (33.875/batch = 271) and
      `eval_tacv6_n_supervised_lat` (3.625/batch = 29) EXACTLY, and `eval_agent_n_*` / `eval_box3d_n_*` counts.
  K2  the TRAIN log (1,008 logged steps, batch 16): mean tactical rows / window, agent-labelled fraction, map-labelled
      fraction against the census fractions; and cells per tactical row against BOTH label-module states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pc(x):
    return np.array([bin(int(v)).count("1") for v in x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    cfg = json.load(open(a.config, encoding="utf-8"))
    rows = [json.loads(l) for l in open(a.metrics, encoding="utf-8") if l.strip()]
    rec_ev = json.load(open(d / "../thor_eval139/record_eval139.json", encoding="utf-8"))
    toks = rec_ev["vocab"]["goal_tokens"]
    tm = sum(1 << toks.index(t) for t in cfg["tac_goal_stats"]["trainable"])
    out = {}
    # ---------------- K1: the in-run eval windows ---------------------------------------------------
    import torch
    ze = np.load(d / "../thor_eval139/windows_eval139.npz")
    N = len(ze["t"])
    g = torch.Generator().manual_seed(12345)
    idx = np.array(torch.randperm(N, generator=g)[:8 * 16].tolist())
    ev = [r for r in rows if "eval_tacv6_n_supervised_goal_cells" in r]
    last = ev[-1]
    mine = {
        "eval_tacv6_n_supervised_goal_cells": float(pc(ze["goal_w_bits"][idx] & tm).sum() / 8),
        "eval_tacv6_n_supervised_lat": float((ze["lat_v7"][idx] != -100).sum() / 8),
        "eval_tacv6_n_supervised_lon": float((ze["lon_v7"][idx] != -100).sum() / 8),
        "eval_agent_n_windows_total_128": int(len(idx)),
        "agent_labelled_in_128": int(ze["agent_labelled"][idx].sum()),
        "box3d_any_in_128": int(ze["box3d_labelled"][idx].sum()),
        "map_label_in_128": int(ze["map_label"][idx].sum())}
    logged = {k: last.get(k) for k in ("eval_tacv6_n_supervised_goal_cells", "eval_tacv6_n_supervised_lat",
                                       "eval_tacv6_n_supervised_lon", "eval_agent_n_windows", "eval_agent_n_labelled",
                                       "eval_box3d_n_windows", "eval_box3d_n_labelled", "eval_map_hires_n_windows",
                                       "eval_map_hires_n_labelled", "eval_agent_n_raw")}
    distinct = sorted({(r["eval_tacv6_n_supervised_goal_cells"], r["eval_tacv6_n_supervised_lat"]) for r in ev})
    out["K1_inrun_eval_windows"] = {
        "n_evals_logged": len(ev), "distinct_logged_(cells,lat)_over_all_evals": distinct,
        "census_per_batch": mine, "logged_last_eval_step": last.get("step"), "logged": logged,
        "agent_labelled_logged_x8": None if logged["eval_agent_n_labelled"] is None else float(logged["eval_agent_n_labelled"]) * 8,
        "match_tactical": bool(abs(mine["eval_tacv6_n_supervised_goal_cells"] - logged["eval_tacv6_n_supervised_goal_cells"]) < 1e-9
                               and abs(mine["eval_tacv6_n_supervised_lat"] - logged["eval_tacv6_n_supervised_lat"]) < 1e-9)}
    # ---------------- K2: the train log vs the train census -----------------------------------------
    zt = np.load(d / "../thor_train/windows_train.npz")
    tr = [r for r in rows if "tacv6_n_supervised_lat" in r]
    cells = np.array([r["tacv6_n_supervised_goal_cells"] for r in tr], np.float64)
    lat = np.array([r["tacv6_n_supervised_lat"] for r in tr], np.float64)
    ag = np.array([r["agent_n_labelled"] for r in tr], np.float64)
    mp = np.array([r["map_hires_n_labelled"] for r in tr], np.float64)
    bs = np.array([r.get("agent_n_windows", 16) for r in tr], np.float64)
    rng = np.random.default_rng(0)
    def boot(f):
        v = []
        for _ in range(2000):
            k = rng.integers(0, len(tr), len(tr))
            v.append(f(k))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    m = zt["in_band"]
    A = float(pc(zt["goal_w_bits_censusstate"][m] & tm).mean())
    B = float(pc(zt["goal_w_bits"][m] & tm).mean())
    out["K2_train_log"] = {
        "n_logged_steps": len(tr), "batch": 16,
        "logged_tactical_rows_per_window": float(lat.sum() / bs.sum()),
        "logged_tactical_rows_ci95_over_steps": boot(lambda k: lat[k].sum() / bs[k].sum()),
        "census_in_band_fraction": float(m.mean()),
        "logged_agent_labelled_fraction": float(ag.sum() / bs.sum()),
        "census_agent_labelled_fraction": float(zt["agent_labelled"].mean()),
        "logged_map_labelled_fraction": float(mp.sum() / bs.sum()),
        "census_map_labelled_fraction": float(zt["map_label"].mean()),
        "logged_trainable_cells_per_tactical_row": float(cells.sum() / lat.sum()),
        "logged_cells_per_row_ci95_over_steps": boot(lambda k: cells[k].sum() / lat[k].sum()),
        "census_cells_per_tactical_row_STATE_A_train_blob_census": A,
        "census_cells_per_tactical_row_STATE_B_eval_blob_in_workers": B,
        "closer_state": "B" if abs(B - cells.sum() / lat.sum()) < abs(A - cells.sum() / lat.sum()) else "A",
        "trainable_mask_tokens": cfg["tac_goal_stats"]["trainable"]}
    # the per-token view: LANE_CHANGE_L supervised cells per tactical row under each state
    i = toks.index("LANE_CHANGE_L")
    out["K2_train_log"]["LANE_CHANGE_L_scored_fraction_of_in_band_windows"] = {
        "state_A": float(((zt["goal_w_bits_censusstate"][m] >> i) & 1).mean()),
        "state_B": float(((zt["goal_w_bits"][m] >> i) & 1).mean()),
        "positive": float(((zt["goal_y_bits"][m] >> i) & 1).mean()),
        "positive_windows": int(((zt["goal_y_bits"][m] >> i) & 1).sum()),
        "positive_clips": int(np.unique(zt["clip_ix"][m & (((zt["goal_y_bits"] >> i) & 1) > 0)]).size)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
