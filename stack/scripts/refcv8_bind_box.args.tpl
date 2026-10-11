# G-BOX-OVERFIT binding harness argv for closure_run.py --binding (one token per line; `#` lines are comments).
# INHERITED from the refcv7 binding run: /home/nvidia/refcv7_bind/run_box (closure argv MEASURED in
# raw/thor_bind/gbo_closure.REDACTED.json, elapsed 5,065.7 s, MAIN arm only per SPEC_REFCV7 A19).
# ⚠ The data roots of the closure are the dirs the harness reads (frames, GT, prereg); the MM / WP-B VERIFY them against the
#   refcv7 bind command line on Thor before the first refcv8 bind run (the command line itself is not in the repo).
stack/scripts/g_box_overfit.py
--launch-argv
{ARGV}
--audit-dir
{OUT}/audit
--out
{OUT}/gbo_binding.json
--device
cuda
--binding
--commit
{COMMIT}
--candidate
BINDING: launch commit {COMMIT} (refcv8-r101-s0; A19 MAIN-only)
--arms
main
DATA-ROOT /home/nvidia/data/refcv6-b1-416x1024-train
DATA-ROOT /home/nvidia/data/sam3_gt_v3
DATA-ROOT {INPUTS}
