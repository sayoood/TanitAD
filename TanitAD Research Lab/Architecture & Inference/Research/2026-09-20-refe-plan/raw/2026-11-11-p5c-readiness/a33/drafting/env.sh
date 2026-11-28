# source this: test env for the A3.3 suites
H=/e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a33/a33w
export TMP=F:/tanitad_scratch/p5c_a33_tmp/t1 TEMP=F:/tanitad_scratch/p5c_a33_tmp/t1 TMPDIR=F:/tanitad_scratch/p5c_a33_tmp/t1 CUDA_VISIBLE_DEVICES=-1
export P5C_TMP=F:/tanitad_scratch/p5c_a33_tmp/gt
export P5C_TABLE_MOD=$H/refe/traj_label_table_gatesuite.py P5C_A31_TABLE_MOD=$H/refe/traj_label_table.py P5C_A3_TABLE_MOD=$H/refe/traj_label_table.py
export P5C_A1_RUNNER=$H/a1_legacy/p5c_pod_label.sh
export P5C_A31_RJ5PATH=$H/a3/p5c_rj5_path.py P5C_A31_GATES=$H/a1/p5c_table_gates.py P5C_A3_GATES_MOD=$H/a1/p5c_table_gates.py
