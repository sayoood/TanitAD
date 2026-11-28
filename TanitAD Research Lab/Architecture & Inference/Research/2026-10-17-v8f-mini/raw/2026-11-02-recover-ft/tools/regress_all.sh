#!/bin/sh
# six detached pytest groups on tree_patched, then the verdict (re-runs failures on the clean tip tree). Writes logs/patched_*.log and E:/.../recover_ft/logs/REGRESSION_VERDICT.txt
cd F:/TanitAD-scratch/recover_ft || exit 2
T=F:/TanitAD-scratch/recover_ft/tree_patched
rm -f logs/patched_*.log
sh regress2.sh $T patched_A1a stack/tests/test_v8f_mini.py stack/tests/test_v8f_ego_perturb.py stack/tests/test_v8f_ego_dagger_compose.py &
sh regress2.sh $T patched_A1b stack/tests/test_v8f_launch_intent.py stack/tests/test_v8f_pgt.py stack/tests/test_v8f_init_ckpt_o_interface.py stack/tests/test_v8f_tac.py stack/tests/test_v8f_orc_chain.py stack/tests/test_v8f_cz.py &
sh regress2.sh $T patched_G1 stack/tests/test_v8f_dagger_train.py &
sh regress2.sh $T patched_G2 stack/tests/test_v8f_oacc.py stack/tests/test_v8f_window_start.py &
sh regress2.sh $T patched_G3 stack/tests/test_v8f_hist.py stack/tests/test_v8f_grid.py &
sh regress2.sh $T patched_R stack/tests/test_v8f_recover.py stack/tests/test_v8f_recover_wiring.py &
wait
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I regress_verdict.py > logs/verdict_runner.log 2>&1
