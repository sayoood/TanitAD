#!/usr/bin/env python3
"""ri_receipts.py -- collect the scene_reindex receipts of the four REINDEXED production rollouts into raw/RI_RECEIPTS.json and gate them.
Usage: python -I code/ri_receipts.py --out-dir F:/TanitAD-scratch/motion_c/ri/out --raw E:/.../motion_c/ri/raw
Per tag (RIC0_s0, RIC0_s1, RIC0C_s0, RIC0C_s1): reads out/<tag>/t1v8f_<tag>.json (header under 't1v8f') and done_<tag>.txt, and checks (a failure is a FAIL, exit 1, every line printed):
  header.scene == reindexed;  n_windows == 2399;  scene_reindex.past_rows_bit_identical true (rows compared > 0, changed 0);  future_frame_rows == 0 (model and floor);  rows_reindexed > 0 (REACH);
  rows by class sum to rows;  the composite arms carry feasible.mode == kbox and the others carry no feasible block;  the ticks file exists, scene == reindexed, and carries the head columns (--dump-heads)."""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

TAGS = ("RIC0_s0", "RIC0_s1", "RIC0C_s0", "RIC0C_s1")
N_WINDOWS = 2399


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()
    res, bad = {}, []
    for tag in TAGS:
        p = f"{a.out_dir}/{tag}/t1v8f_{tag}.json"
        d = f"{a.out_dir}/done_{tag}.txt"
        if not (os.path.isfile(p) and os.path.isfile(d)):
            bad.append(f"{tag}: result JSON or done marker absent ({p}, {d})")
            continue
        h = json.load(open(p, encoding="utf-8"))["t1v8f"]
        done = json.load(open(d, encoding="utf-8"))
        sr = h.get("scene_reindex", {})
        composite = tag.startswith("RIC0C")
        tf = h.get("ticks_file", "")
        z = np.load(tf, allow_pickle=True) if os.path.isfile(tf) else None
        chk = {
            "scene_is_reindexed": h.get("scene") == "reindexed",
            "n_windows_2399": h.get("n_windows") == N_WINDOWS,
            "past_rows_bit_identical": bool(sr.get("past_rows_bit_identical")) and sr.get("past_rows_checked", 0) > 0 and sr.get("past_rows_changed", -1) == 0,
            "no_future_frame_model_and_floor": sr.get("future_frame_rows", -1) == 0 and sr.get("floor", {}).get("future_frame_rows", -1) == 0,
            "rows_reindexed_gt_0": sr.get("rows_reindexed", 0) > 0,
            "rows_by_class_sum": sr.get("rows_logged", 0) + sr.get("rows_reindexed", 0) + sr.get("rows_on_time", 0) + sr.get("rows_clamped", 0) == sr.get("rows", -1),
            "feasible_as_declared": (h.get("feasible", {}).get("mode") == "kbox") if composite else ("feasible" not in h),
            "ticks_scene_reindexed_and_heads": z is not None and str(z["scene"]) == "reindexed" and "s_ckpt" in z.files and "t_lat_logits" in z.files,
        }
        for k, v in chk.items():
            if not v:
                bad.append(f"{tag}: check {k} FAILED")
        res[tag] = {"checks": chk, "tag": h.get("tag"), "ckpt": h.get("ckpt"), "n_windows": h.get("n_windows"), "feasible": h.get("feasible", {}).get("mode"),
                    "ticks_rows": int(len(z["tick"])) if z is not None else None, "scene_reindex": sr, "done": done}
        s = sr
        lag = s.get("lag_cache_steps_all_rows_after_t0", {})
        lr = s.get("lag_cache_steps_reindexed_rows", {})
        print(f"RIRECEIPT {tag}: items {s.get('items')} rows {s.get('rows')} logged {s.get('rows_logged')} reindexed {s.get('rows_reindexed')} on_time {s.get('rows_on_time')} clamped {s.get('rows_clamped')} "
              f"raised {s.get('rows_monotone_raised')} future {s.get('future_frame_rows')} | lag all-after-t0 mean {lag.get('mean')} p50 {lag.get('p50')} p90 {lag.get('p90')} max {lag.get('max')} | "
              f"lag reindexed n {lr.get('n')} mean {lr.get('mean')} max {lr.get('max')} | past bit-identical {s.get('past_rows_bit_identical')} ({s.get('past_rows_checked')} rows) | wall {done.get('wall_s')} s")
    os.makedirs(a.raw, exist_ok=True)
    json.dump({"tags": res, "failures": bad, "PASS": not bad}, open(f"{a.raw}/RI_RECEIPTS.json", "w", encoding="utf-8"), indent=1)
    for b in bad:
        print("RIRECEIPT FAIL", b)
    print("RIRECEIPT", "PASS" if not bad else f"FAIL ({len(bad)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
