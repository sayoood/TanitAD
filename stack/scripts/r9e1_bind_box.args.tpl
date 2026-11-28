# refcv9-e1rig COPY of refcv8_bind_box.args.tpl (ONLY the --candidate text differs: it named refcv8-r101-s0).
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
BINDING (refcv9-e1rig, rig argv {ARGV_SHA}): base commit {COMMIT} + the rig-aware g_box_overfit.py overlay (hold_r9_ovf); A19 MAIN-only
--arms
main
# VERIFIED 2026-10-14 (MM, launch commit b30f924) against the MEASURED refcv7 bind command lines on Thor
# (/home/nvidia/refcv7_bind/run_gbo_binding_main.sh:48, run_gmo_binding_main.sh:86): those declared
# `--data-root /home/nvidia/data` + the run's own dir. The narrower roots this template carried left files the
# harness READS outside every root (closure_run records only files opened UNDER a root; it does not refuse),
# and the box template's {INPUTS} root does not exist on Thor. Roots below = refcv7's measured set. refcv8's third root (the v9 label release)
# is DROPPED: the rig argv reads no --r8-v9-labels.
DATA-ROOT /home/nvidia/data
DATA-ROOT {OUT}/audit
