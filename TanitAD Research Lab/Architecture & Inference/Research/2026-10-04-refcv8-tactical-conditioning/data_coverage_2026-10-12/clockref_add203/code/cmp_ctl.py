import json, sys
a = json.load(open("raw/q4c_grid_vs_egolog_ALLTRAIN.json"))["splits"]["train"]["per_clip"]
a = {r["clip"]: r for r in a}
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "raw/q4c_CTL.json"))
print("controls:", d["controls"])
new = {r["clip"]: r for r in d["splits"]["train"]["per_clip"]}
sp = d["splits"]["train"]
print("requested", sp["n_clips_requested"], "fit", sp["n_clips_fit"], "no_egolog", sp["n_clips_no_egolog"])
both = [c for c in new if c in a]
only_new = [c for c in new if c not in a]
print("fit in both:", len(both), "| fit only in NEW:", len(only_new), "| fit in ALLTRAIN but not requested here: n/a")
keys = ["grid_start_on_label_timeline_s", "dt_s", "fit_resid_max_ms", "fit_resid_rms_ms", "K3_speed_maxabs_ms", "n_moving_rows", "anchor_row_true_time_s", "span_log_s", "T_out", "proj_dist_max_m"]
for k in keys:
    mx = max(abs(new[c][k] - a[c][k]) for c in both)
    print(f"  max|d {k}| = {mx:.3e}")
