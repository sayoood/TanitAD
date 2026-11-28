"""R1 identity report (CPU, Thor): I-0 (capture FULL forward vs the route package's banked eval_s0.npz on the GRID),
I-X / I-S summaries (from the capture's own per-window rows), window-set agreement with A6's dense list, cache size.
Writes <cache>/eval/identity_report.json. Prints sha12 only."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

TOL = 1e-5            # DESIGN.md sec. 5, fixed before the GPU run
IS_FLIP_MAX = 0.01    # I-S: differing E9 picks <= 1 % of windows
IS_FAN_P99 = 0.05     # I-S: p99 fan |diff| <= 0.05 m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--bank", default="/home/nvidia/refcv7_post/route/out/eval_s0.npz")
    ap.add_argument("--a6-list", default="/home/nvidia/refcv7_post/route_a6/out/a6_window_list.json")
    a = ap.parse_args()
    import torch
    d = Path(a.cache) / "eval"
    sel = json.loads((d / "selection.json").read_text())
    shards = sorted(glob.glob(str(d / "shard_*.pt"))) + sorted(glob.glob(str(d / "gridonly_*.pt")))
    keys = ("full_fan", "full_s_core", "full_s_e9", "full_sel", "full_selb", "full_p_lat", "full_p_lon", "b_r1_i")
    acc = {k: [] for k in keys}
    sha = []
    for s in shards:
        r = torch.load(s, map_location="cpu", weights_only=False)
        for k in keys:
            acc[k].append(r[k])
        sha += r["win_sha12"]
    cat = {k: torch.cat(v).numpy() for k, v in acc.items()}
    t_of_i = {w["i"]: w["t"] for w in sel["windows"]}
    wt = np.array([t_of_i[int(i)] for i in cat["b_r1_i"]])
    pos = {(s_, int(t_)): j for j, (s_, t_) in enumerate(zip(sha, wt))}
    out = {"n_cached_windows": len(sha), "n_selected": sel["n_windows"], "n_shards": len(shards)}
    # ---- I-0 ---------------------------------------------------------------------------------- #
    z = np.load(a.bank, allow_pickle=True)
    bs, bt = z["win_sha12"].astype(str), z["win_t"].astype(int)
    rows = [(j, pos.get((s_, int(t_)))) for j, (s_, t_) in enumerate(zip(bs, bt))]
    miss = sum(1 for _j, p in rows if p is None)
    jj = np.array([j for j, p in rows if p is not None])
    pp = np.array([p for _j, p in rows if p is not None])
    i0 = {"n_bank": int(len(bs)), "n_joined": int(len(jj)), "n_missing": int(miss)}
    for bk, ck in (("fan", "full_fan"), ("s_core", "full_s_core"), ("s_e9", "full_s_e9"), ("p_lat", "full_p_lat"),
                   ("p_lon", "full_p_lon")):
        i0[f"maxabs_{bk}"] = float(np.abs(z[bk][jj].astype(np.float64) - cat[ck][pp].astype(np.float64)).max())
    i0["n_sel_differ"] = int((z["sel_idx"][jj] != cat["full_sel"][pp]).sum())
    i0["n_selbase_differ"] = int((z["sel_idx_base"][jj] != cat["full_selb"][pp]).sum())
    i0["PASS"] = bool(miss == 0 and i0["n_sel_differ"] == 0 and i0["n_selbase_differ"] == 0
                      and all(i0[f"maxabs_{k}"] <= TOL for k in ("fan", "s_core", "s_e9", "p_lat", "p_lon")))
    out["I0_capture_vs_bank_eval_s0"] = i0
    # ---- I-X / I-S ------------------------------------------------------------------------------ #
    rec = json.loads((d / "record.json").read_text())
    idn = rec["identity"]
    n = idn["n_windows"]
    ix = {k: v for k, v in idn.items() if k.startswith("x_")}
    is_ = {k: v for k, v in idn.items() if k.startswith("s_")}
    ix["PASS"] = bool(idn["x_n_sel_differ"] == 0 and idn["x_n_selbase_differ"] == 0 and all(
        idn[f"x_maxabs_{k}"] <= TOL for k in ("fan", "traj", "s_core", "s_e9", "p_lat", "p_lon", "p_goal", "g_tac")))
    is_["flip_rate"] = idn["s_n_sel_differ"] / max(n, 1)
    is_["PASS"] = bool(is_["flip_rate"] <= IS_FLIP_MAX and idn["s_p99abs_fan"] <= IS_FAN_P99)
    out["n_identity_windows"] = n
    out["IX_exact_tensors"] = ix
    out["IS_stored_tensors"] = is_
    out["e9_top2_margin_p01_p05_p50"] = idn.get("e9_top2_margin_p01_p05_p50")
    # ---- window set vs A6 ------------------------------------------------------------------------ #
    if os.path.exists(a.a6_list):
        a6 = json.loads(Path(a.a6_list).read_text())
        A = {(w[0], int(w[1])) for w in a6["windows"]}
        mine = {(w["sha12"], w["t"]) for w in sel["windows"] if ("turn" in w["flags"] or "sample" in w["flags"])}
        mt = {(w["sha12"], w["t"]) for w in sel["windows"] if "turn" in w["flags"]}
        at = {(w[0], int(w[1])) for w in a6["windows"] if w[2] == "turn"}
        out["vs_A6_window_list"] = {"n_A6": len(A), "n_R1_scored": len(mine), "n_common": len(A & mine),
                                    "turn_sets_equal": mt == at, "n_turn_R1": len(mt), "n_turn_A6": len(at)}
    sz = sum(os.path.getsize(p) for p in glob.glob(str(Path(a.cache) / "*" / "*_*.pt")))
    out["cache_bytes_all_shards"] = int(sz)
    (d / "identity_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
