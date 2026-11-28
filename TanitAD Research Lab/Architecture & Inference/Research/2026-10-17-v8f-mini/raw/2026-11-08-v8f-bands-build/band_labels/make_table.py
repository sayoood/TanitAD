"""make_table.py <compare_json> -- markdown per-field change table (v9 -> v10)."""
import json, sys
r = json.load(open(sys.argv[1]))
f = r["fields"]
pc = lambda v: "-" if v is None else f"{100 * v:.1f} %"
print(f"rows {r['n_rows']:,}; v9 full-band rows (h_obs >= 8 s) {r['n_rows_v9_fullband']:,}; rows with any constraint cell changed {pc(r['any_constraint_cell_changed'])}\n")
print("| field | defined v9 | defined v10 | changed, all rows | changed among v9-defined | ... full-band rows | defined->IGNORE | IGNORE->defined |")
print("|---|---|---|---|---|---|---|---|")
for k in ("lat_cls_a", "lat_cls_b", "lon_cls"):
    v = f[k]
    print(f"| {k} | {pc(v['v9_defined'])} | {pc(v['v10_defined'])} | {pc(v['changed_all'])} | {pc(v['changed_among_v9_defined'])} | {pc(v['changed_among_v9_defined_fullband'])} | {v['defined_to_ignore']:,} | {v['ignore_to_defined']:,} |")
for k in ("lat_allowed_a", "lat_allowed_b", "lon_allowed", "goal_y(22-bit mask)", "goal_w(22-bit mask)", "h_obs_s"):
    print(f"| {k} | | | {pc(f[k]['changed_all'])} | | {pc(f[k]['changed_fullband'])} | | |")
print()
print("| constraint cell | defined v9 | defined v10 | changed among v9-defined | defined->undefined | undefined->defined |")
print("|---|---|---|---|---|---|")
for k, v in f.items():
    if "v9_defined" in v and k not in ("lat_cls_a", "lat_cls_b", "lon_cls"):
        print(f"| {k} | {pc(v['v9_defined'])} | {pc(v['v10_defined'])} | {pc(v['changed_among_v9_defined'])} | {v['defined_to_ignore']:,} | {v['ignore_to_defined']:,} |")
print()
print("| goal token | weighted v9 | weighted v10 | w changed | positive v9 | positive v10 |")
print("|---|---|---|---|---|---|")
for k, v in r["goal_bits"].items():
    print(f"| {k} | {pc(v['w_v9'])} | {pc(v['w_v10'])} | {pc(v['w_changed'])} | {pc(v['pos_v9'])} | {pc(v['pos_v10'])} |")
