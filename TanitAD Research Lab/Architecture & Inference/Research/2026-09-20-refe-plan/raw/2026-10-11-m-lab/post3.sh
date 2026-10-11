#!/bin/bash
# waits for post2.sh, then renders the DRAFT result (the finishing agent owns the final eval/RESULT_M_LAB.md)
cd "$(dirname "$0")"
until grep -q ZZMLAB_POST2_DONE chain2.log 2>/dev/null; do sleep 20; done
PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe mlab_render.py --out "$(pwd -W)/RESULT_M_LAB.draft.md" > render.log 2>&1
echo ZZMLAB_POST3_DONE $(date +%FT%T) >> chain2.log
