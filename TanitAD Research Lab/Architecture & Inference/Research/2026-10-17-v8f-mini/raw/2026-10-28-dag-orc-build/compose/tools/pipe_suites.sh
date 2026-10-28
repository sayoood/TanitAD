#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
L=F:/TanitAD-scratch/compose_tmp/logs
rm -f $L/suites_progress.txt $L/suites.done
bash F:/TanitAD-scratch/compose_tmp/run_suites.sh test_v8f_ego_dagger_compose test_v8f_ego_perturb test_v8f_dagger_train test_v8f_dagger test_v8f_init_ckpt_o_interface test_launch_gate_v8f test_v8f_orc_chain
echo done > $L/suites.done
