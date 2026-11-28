#!/bin/sh
# usage: pt.sh <tree> <pytest args...>   (CPU only, tip interpreter)
T="$1"; shift
export CUDA_VISIBLE_DEVICES=-1
cd "$T" || exit 2
exec C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q --basetemp=F:/TanitAD-scratch/recover_ft/bt -p no:cacheprovider "$@"
