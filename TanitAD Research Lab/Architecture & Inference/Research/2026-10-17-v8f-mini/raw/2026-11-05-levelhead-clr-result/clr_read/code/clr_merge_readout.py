"""Merge CL_READOUT_s0.json and CL_READOUT_s1.json into raw/CL_READOUT.json (per seed, with the committed labels) and print the table RESULT.md quotes."""
import json
import os
import sys

RAW = "E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/readout"


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def cell(d, n=3):
    return f"{f(d['point'], n)} [{f(d['ci95'][0], n)}, {f(d['ci95'][1], n)}]"


def main():
    per = {}
    for s in (0, 1):
        per[f"s{s}"] = json.load(open(f"{RAW}/CLR_READOUT_s{s}.json"))
    out = {"prereg": per["s0"]["prereg"], "per_seed": per,
           "labels_per_seed": {k: v["labels"] for k, v in per.items()},
           "control_failures": {k: v["control_failures"] for k, v in per.items()}}
    json.dump(out, open(f"{RAW}/CLR_READOUT.json", "w", newline="\n"), indent=1)
    rows = []
    for k, v in per.items():
        t, sn, m = v["tracking"], v["sensitivity"], v["m1_windows"]
        rows.append((k, v["n_windows"], cell(t["r2_all"]), cell(t["band_mae"]), t["n_band"],
                     (cell(m["crawl_share_gt0"]["mae"]) + f" (n={m['crawl_share_gt0']['n']})") if m.get("status") == "computed" else m.get("status"),
                     cell(sn["plus3"]["mean_signed_change_ms"], 4), cell(sn["minus3"]["mean_signed_change_ms"], 4), cell(sn["plus3"]["mean_abs_change_ms"], 4),
                     cell(sn["e_level_over_ego_norm"]["mean_of_per_window_ratio"], 4), t["label"], sn["label"]))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    for k, v in per.items():
        print(k, "controls", v["control_failures"], "tick0 dev", v["banked_tick0_check"].get("max_abs_dev_m"), "reach", v["sensitivity"]["controls"]["reach_control_big_e_level_output_perturbation"]["mean_abs_plan_speed_change_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
