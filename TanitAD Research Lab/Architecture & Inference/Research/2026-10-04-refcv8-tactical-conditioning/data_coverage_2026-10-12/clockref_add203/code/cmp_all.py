"""Full-scale replication control: q4c main() on all 4,572 clips (Thor) vs the stored ALLTRAIN (dev box, 4,357 fitted old clips) and vs the ADD203 file."""
import json
A = {r['clip']: r for r in json.load(open('raw/q4c_grid_vs_egolog_ALLTRAIN.json'))['splits']['train']['per_clip']}
N = json.load(open('raw/q4c_ALL4572_rawmain.json'))
sp = N['splits']['train']; print({k: v for k, v in sp.items() if k.startswith('n_')}, 'ego_dir_n_files', N['ego_dir_n_files'])
P = {r['clip']: r for r in sp['per_clip']}
D = {r['clip']: r for r in json.load(open('raw/q4c_ADD203_rawmain.json'))['splits']['train']['per_clip']}
inboth = [c for c in A if c in P]
print('ALLTRAIN fitted clips', len(A), '| also fitted in ALL4572', len(inboth), '| fitted in ALLTRAIN only', len([c for c in A if c not in P]), '| fitted in ALL4572 only (= the ADD fits)', len([c for c in P if c not in A]))
keys = ["grid_start_on_label_timeline_s", "dt_s", "fit_resid_max_ms", "fit_resid_rms_ms", "K3_speed_maxabs_ms", "n_moving_rows", "anchor_row_true_time_s", "span_log_s", "T_out", "proj_dist_max_m"]
for k in keys: print(f'  old fitted clips, Thor-ALL4572 vs stored ALLTRAIN: max|d {k}| = {max(abs(P[c][k]-A[c][k]) for c in inboth):.3e}')
print('ADD fits: ALL4572 set == ADD203 file set:', set(c for c in P if c not in A) == set(D))
for k in keys: print(f'  ALL4572 vs ADD203 file max|d {k}| = {max(abs(P[c][k]-D[c][k]) for c in D):.3e}')
