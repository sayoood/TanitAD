#!/bin/bash
# MM 2026-11-15: read ONE banked ACTSENS arm with the patched driver (torch threads 1 for windows and plans, as phase 1; C7 PASS on s0 and s1).
# usage: bash read_arm.sh <tag> <chassis>. Arm bank = C:/Users/Admin/t2v8f_pulls/actsens_<tag> (the chain's verified bank; BANK.done required).
# Marker: out/<tag>/ARM_DONE_<tag>.txt or ARM_FAILED_<tag>.txt; exit line runs/read_<tag>.exit.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
TAG=${1:?tag}; CH=${2:?chassis}
EX=/f/TanitAD-scratch/actsens/read/runs/read_$TAG.exit
[ -s "C:/Users/Admin/t2v8f_pulls/actsens_$TAG/BANK.done" ] || { echo "read_arm $TAG REFUSED: no BANK.done $(date -Iseconds)" >> $EX; exit 3; }
export HWM_WORKERS=2
bash /f/TanitAD-scratch/actsens/read/read_actsens.sh $TAG $CH
echo "read_arm $TAG exit $? $(date -Iseconds)" >> $EX
