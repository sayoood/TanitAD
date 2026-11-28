# G-MAP-OVERFIT binding harness argv for closure_run.py --binding (one token per line; `#` lines are comments).
# INHERITED from the refcv7 binding run: /home/nvidia/refcv7_bind/run_map (closure argv MEASURED in
# raw/thor_bind/gmo_closure.REDACTED.json, elapsed 2,047.3 s, MAIN arm only per SPEC_REFCV7 A19).
# ⚠ Before the refcv8 bind run, the MM / WP-B VERIFY that the near-lift / refine / decision-rule / class-weights values
#   below equal the refcv8 LAUNCH argv's map settings, and that {INPUTS} holds the A19 MAIN spec + the frameset.
# {CODE} {COMMIT} {ARGV} {ARGV_SHA} {OUT} {INPUTS} are filled by `refcv8_prelaunch.sh bind`.
stack/scripts/map_hires_overfit.py
--spec
{INPUTS}/gmo_spec_A19_MAP_MAIN.json
--frameset
{INPUTS}/gmo_frameset.json
--class-weights
/home/nvidia/data/refcv7/map_hires_class_weights_train_100x30.json
--decision-rule
prior_corrected
--near-lift-m
20
--near-refine-blocks
1
--v2-cache
/home/nvidia/data/refcv6-b1-416x1024-train
--gt-root
/home/nvidia/data/sam3_gt_v3
--extrinsics
/home/nvidia/data/refcv6_train_eval139_extrinsics.json
--cam-ts-dir
/home/nvidia/data/_b1stage416/r0/camera_front_wide
--launch-commit
{COMMIT}
--launch-argv-sha256
{ARGV_SHA}
--arms
healthy
--out
{OUT}
--device
cuda
# data roots whose OPENED files the closure records (every file read under them is hashed into the closure)
# VERIFIED 2026-10-14 (MM, launch commit b30f924) against the MEASURED refcv7 bind command lines on Thor
# (/home/nvidia/refcv7_bind/run_gbo_binding_main.sh:48, run_gmo_binding_main.sh:86): those declared
# `--data-root /home/nvidia/data` + the run's own dir. The narrower roots this template carried left files the
# harness READS outside every root (closure_run records only files opened UNDER a root; it does not refuse),
# and the box template's {INPUTS} root does not exist on Thor. Roots below = refcv7's measured set.
DATA-ROOT /home/nvidia/data
DATA-ROOT {INPUTS}
