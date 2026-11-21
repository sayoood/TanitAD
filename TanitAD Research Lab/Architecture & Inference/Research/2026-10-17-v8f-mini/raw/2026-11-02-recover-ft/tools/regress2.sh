#!/bin/sh
# usage: regress2.sh <tree> <tag> <test files...> -- run the named files on one tree; logs/<tag>.log (CPU, tip interpreter)
T="$1"; TAG="$2"; shift; shift
export CUDA_VISIBLE_DEVICES=-1
cd "$T" || exit 2
mkdir -p "F:/TanitAD-scratch/recover_ft/bt_$TAG"
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q -rfE --durations=5 --basetemp="F:/TanitAD-scratch/recover_ft/bt_$TAG" -p no:cacheprovider "$@" > "F:/TanitAD-scratch/recover_ft/logs/$TAG.log" 2>&1
echo "exit $?" >> "F:/TanitAD-scratch/recover_ft/logs/$TAG.log"
