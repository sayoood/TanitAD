#!/usr/bin/env bash
# MM independent re-run of the A3.3 suites on a COPY (F:/tanitad_scratch/mm_a33chk/a33w), own TMP; serial, CPU
C=/f/tanitad_scratch/mm_a33chk; H=$C/a33w; L=$C/logs; mkdir -p $L
export TMP=F:/tanitad_scratch/mm_a33chk/tmp TEMP=F:/tanitad_scratch/mm_a33chk/tmp TMPDIR=F:/tanitad_scratch/mm_a33chk/tmp CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export P5C_TMP=F:/tanitad_scratch/mm_a33chk/gt
export P5C_TABLE_MOD=$H/refe/traj_label_table_gatesuite.py P5C_A31_TABLE_MOD=$H/refe/traj_label_table.py P5C_A3_TABLE_MOD=$H/refe/traj_label_table.py
export P5C_A1_RUNNER=$H/a1_legacy/p5c_pod_label.sh
export P5C_A31_RJ5PATH=$H/a3/p5c_rj5_path.py P5C_A31_GATES=$H/a1/p5c_table_gates.py P5C_A3_GATES_MOD=$H/a1/p5c_table_gates.py
cd $H/a1; python test_p5c_table_gates.py > $L/gates_legacy.log 2>&1; echo "gates_legacy rc=$? $(tail -1 $L/gates_legacy.log | cut -c1-160)"
python -m pytest -q test_p5c_a33_gates.py -p no:cacheprovider > $L/a33_gates.log 2>&1; echo "a33_gates rc=$? $(tail -1 $L/a33_gates.log)"
python -m pytest -q test_p5c_pod_label_a31.py -p no:cacheprovider > $L/runner.log 2>&1; echo "runner rc=$? $(tail -1 $L/runner.log)"
cd $H/a3; python -m pytest -q test_p5c_rj5_path.py test_p5c_a33_rj5.py -p no:cacheprovider > $L/rj5.log 2>&1; echo "rj5 rc=$? $(tail -1 $L/rj5.log)"
echo ZZA33CHK_END
