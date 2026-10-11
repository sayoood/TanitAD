#!/bin/bash
# waits for post_chain.sh, then runs the extra arms (cached_r4..r6) and re-aggregates
cd "$(dirname "$0")"
until grep -q ZZMLAB_POST_DONE chain.log 2>/dev/null; do sleep 20; done
PYTHONIOENCODING=utf-8 C:/Users/Admin/navsim-crun/venv/Scripts/python.exe mlab_driver.py > chain2.log 2>&1
PYTHONIOENCODING=utf-8 C:/Users/Admin/navsim-crun/venv/Scripts/python.exe mlab_summarize.py > summarize2.log 2>&1
echo ZZMLAB_POST2_DONE $(date +%FT%T) >> chain2.log
