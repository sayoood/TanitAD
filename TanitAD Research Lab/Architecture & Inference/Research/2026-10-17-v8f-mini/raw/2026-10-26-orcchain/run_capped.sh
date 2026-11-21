#!/usr/bin/env bash
# run_capped.sh <label> <cmd...>: waits until pytest_runs.py prints 0 or 1, then runs the given run script with the label (the repo's pytest-cap rule)
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
while true; do n=$($PY C:/Users/Admin/orcread/pytest_runs.py 2>/dev/null | tr -d '\r\n '); if [ "$n" = "0" ] || [ "$n" = "1" ]; then break; fi; sleep 4; done
"$@"
