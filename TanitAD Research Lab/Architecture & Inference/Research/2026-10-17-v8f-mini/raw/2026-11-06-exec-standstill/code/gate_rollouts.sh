#!/usr/bin/env bash
# gate_rollouts.sh: the four exec-tool rollouts of the gate subset (C0_s0 checkpoint, 123 windows). G_R0_orig (the unpatched LF6 tool) is run separately.
S=C:/Users/Admin/exec_standstill_scratch
T=$S/tree/stack/scripts/v8f_t1_eval_exec.py
R=E:/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/code/run_tool.sh
CK=E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt
G=$S/subset/grid_subset.json
O=$S/gate
X() { local tag=$1; shift; bash $R $T $tag $CK $O $G --no-analyze --dump-exec-ch $O/$tag/exec_ch_$tag.npz "$@"; echo "$(date +%T) $tag rc=$?" >> $O/gate_rollouts.log; }
X G_R1_default
X G_R2_eps1e-3 --exec-eps 1e-3
X G_R3_hold01 --exec-eps 0.1
X G_R4_over2 --exec-eps 2.0
echo "$(date +%T) ALL DONE" >> $O/gate_rollouts.log
