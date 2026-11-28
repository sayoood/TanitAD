#!/usr/bin/env bash
# the three suites of the A3.3 files, on the dev box (CUDA hidden; TMP on F:)
cd "$(dirname "$0")"; . ./env.sh
cd a1; python test_p5c_table_gates.py > ../proof/logs/TEST_RUN_a32_gates_legacy_suite.log 2>&1; echo "gates_legacy rc=$?" ; tail -1 ../proof/logs/TEST_RUN_a32_gates_legacy_suite.log | cut -c1-200
python -m pytest -q test_p5c_a33_gates.py -p no:cacheprovider > ../proof/logs/TEST_RUN_a33_gates.log 2>&1; echo "a33_gates rc=$?"; tail -2 ../proof/logs/TEST_RUN_a33_gates.log
python -m pytest -q test_p5c_pod_label_a31.py -p no:cacheprovider > ../proof/logs/TEST_RUN_runner_a31_a33.log 2>&1; echo "runner rc=$?"; tail -2 ../proof/logs/TEST_RUN_runner_a31_a33.log
cd ../a3; python -m pytest -q test_p5c_rj5_path.py test_p5c_a33_rj5.py -p no:cacheprovider > ../proof/logs/TEST_RUN_rj5_a3_a33.log 2>&1; echo "rj5 rc=$?"; tail -2 ../proof/logs/TEST_RUN_rj5_a3_a33.log
