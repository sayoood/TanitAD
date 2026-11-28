#!/usr/bin/env bash
# the A5 verification run: every suite the amendment names, output beside this script.   cd <refe-plan> && PYTHONIOENCODING=utf-8 bash raw/2026-10-14-s1-g12-bf16/run_all_tests.sh
PY="${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
R=raw/2026-10-14-s1-g12-bf16
run() { name="$1"; shift; "$PY" "$@" > "$R/$name.out" 2>&1; echo "EXIT $?" >> "$R/$name.out"; }
run test_aux_g12_bf16 refe/test_aux_g12_bf16.py
run test_rl_s1b refe/test_rl_s1b.py
run test_aux_drivable_head refe/test_aux_drivable_head.py
run test_rl_epo refe/test_rl_epo.py
run test_s1aux_chain refe/test_s1aux_chain.py
run test_launch_gate_static refe/test_launch_gate.py --static
run test_aux_s1_wiring refe/test_aux_s1_wiring.py
echo ALLDONE > "$R/ALLDONE"
