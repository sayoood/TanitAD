#!/bin/bash
# usage: envrun.sh <tree> <tmpname> <cmd...>   (CPU only; python by path)
TREE="$1"; TMPN="$2"; shift 2
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1
export PYTHONPATH="$(cygpath -w "$TREE/stack")"
T="F:/TanitAD-scratch/n1dc/build/n1dh_b1/tmp/$TMPN"; mkdir -p "$T"
export TMP="$(cygpath -w "$T")" TEMP="$(cygpath -w "$T")"
cd "$TREE/stack"
"C:/Users/Admin/venvs/tanitad/Scripts/python.exe" "$@"
