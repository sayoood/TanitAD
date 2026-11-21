# p4 exploratory blend reads (2026-09-27, AFTER Amendment A1 was staged at 01:32:05; EXPLORATORY)
Synthetic dumps built from the ORIGINAL dev-box dumps (`C:/Users/Admin/refav1_margin/p4out/dump_<arm>/`):
every `ep*.npz` copied with ONLY its `cl` array replaced (positions [n, 10, 2], float64 blend then cast back):
- `b50_<arm>`: cl := 0.5*cl + 0.5*ha0_ext   (arms: loncomb3, lonshift, lonshift_s1)
- `w0_loncomb3`: cl := ha0_ext               (known-value control: must read exactly 0.0000 vs ha0_ext -- it does)
- `damp50`: cl := 0.5*ha0 + 0.5*ha0_ext      (the planner-free blend -- the matched floor, Amendment A2)
Floors untouched (the paired tool's bit-identical floor check passed). Scored with
`taniteval/tools/refav1_paired_delta.py` (tip b3f7ea6f), paired episode-cluster bootstrap, n_boot 2000, 8 clusters.
The synthetic dumps themselves are not banked (regenerable in seconds from the banked originals + this recipe).
