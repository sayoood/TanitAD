# Full-grid damped-floor read (2026-09-27, EXPLORATORY, zero GPU)
`shipped_full_dump/` = Thor `/home/nvidia/refav1_evalrun/full_dump` (the 2026-09-04 shipped-cost run of
`refav1-b1-v72-ep3-speed` @ 21,109 over the 141-episode v7.2 EVAL grid, stride 40 -> 282 windows), copied read-only by
scp on 2026-09-27; md5 of ep000/ep070/ep140 matched Thor. Until today this dump existed only on Thor. The episode and decision `.npz` files carry integer
indices only (binary-safe grep: 0 clip ids); the dump's `manifest.json` carried the 141 clip ids (282 occurrences) and
is banked with every one rewritten as sha12. CORRECTION: the first version of this note said "no clip ids (checked)"
before that check had been run; the post-write package scan found the 282 and they were rewritten before anything
was committed. It predates the `ha0_ext` floor.
Synthetic dumps (not banked, regenerable in seconds): `dampha` = cl := 0.5*ha0 + 0.5*ha; `w0ha` = cl := ha (known-value
control: reads exactly 0.0000 vs ha). Scored with `taniteval/tools/refav1_paired_delta.py` (tip b3f7ea6f), paired
episode-cluster bootstrap over 141 clusters, n_boot 2000 -> `pd_full_dampha.{md,json,log}`.
