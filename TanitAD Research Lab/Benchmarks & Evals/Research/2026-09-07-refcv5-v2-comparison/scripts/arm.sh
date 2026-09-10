#!/bin/sh
# Run taniteval/tools/refcv3_arm.py OFF the G: Drive on the dev-box RTX 4060.
# ⛔ G: cannot RUN the stack (Errno 22 mid-import) -- everything below is local.
R=/c/Users/Admin/refcv5cmp
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\Users\Admin\refcv5cmp\repo\stack;C:\Users\Admin\refcv5cmp\repo\taniteval;C:\Users\Admin\refcv5cmp\repo"
cd $R/repo
exec /c/Users/Admin/venvs/tanitad/Scripts/python.exe taniteval/tools/refcv3_arm.py "$@"
