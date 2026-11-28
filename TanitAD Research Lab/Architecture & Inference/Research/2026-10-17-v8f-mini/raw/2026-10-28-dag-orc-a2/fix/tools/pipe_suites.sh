#!/bin/bash
# DETACHED pipeline A: (1) the RED-on-old proof (new tests on the UNMODIFIED e7ea20d1 code), (2) the seven existing suites + the changed file on the fixed tree, ONE pytest at a time.
export PATH="/usr/bin:/bin:$PATH"
T=F:/TanitAD-scratch/compose_fix/tmp
L=F:/TanitAD-scratch/compose_fix/logs
mkdir -p $L
rm -f $L/suites.done
# the FINAL test file onto the pristine e7ea20d1 code
cp F:/TanitAD-scratch/compose_fix/tree/stack/tests/test_v8f_ego_dagger_compose.py F:/TanitAD-scratch/compose_fix/oldnew/stack/tests/test_v8f_ego_dagger_compose.py
NEWSEL="equal_seeds_every_sample or selection_uniforms_are_not_one_stream or seed_of_stream_zero or box2_shape or run_record_names_the_perturbation_stream or not_on_the_declared_stream or not_resumed_on_another_generator"
bash $T/run_k.sh F:/TanitAD-scratch/compose_fix/oldnew red_on_old "$NEWSEL"
bash $T/run_k.sh F:/TanitAD-scratch/compose_fix/tree green_new "$NEWSEL"
for f in test_v8f_ego_dagger_compose test_v8f_ego_perturb test_v8f_dagger_train test_v8f_dagger test_v8f_init_ckpt_o_interface test_launch_gate_v8f test_v8f_orc_chain; do
  bash $T/run_k.sh F:/TanitAD-scratch/compose_fix/tree suite_$f "" $f
done
echo done > $L/suites.done
