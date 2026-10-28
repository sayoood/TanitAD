#!/bin/bash
# the FINAL green-on-build and red-on-tip runs, then the existing v8F suites ONE FILE AT A TIME on the build (each through run_k.sh, which waits for a pytest slot). Single instance. Writes logs/suites.done.
S=F:/TanitAD-scratch/launch_intent
mkdir $S/tmp/lock_suites 2>/dev/null || { echo "pipe_suites_final already running or done (remove $S/tmp/lock_suites to rerun)"; exit 0; }
R=E:/Projects/TanitAD-artifacts/dagland/launch_intent/tools/run_k.sh
bash $R $S/tree final_green ""
bash $R $S/pristine final_red_on_tip ""
for f in test_v8f_ego_dagger_compose test_v8f_ego_perturb test_v8f_dagger_train test_v8f_dagger test_v8f_init_ckpt_o_interface test_launch_gate_v8f test_v8f_orc_chain test_v8f_mini test_v8f_pgt test_v8f_flat test_v8f_oacc; do
  bash $R $S/tree suite_$f "" $f
done
echo done > $S/logs/suites.done
