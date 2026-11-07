#!/usr/bin/env bash
# MM 2026-11-06: re-run of the SECONDARY pswpedge M1 arm read -- mapthr_read.sh's attempt (20:30:27Z) hit ALREADYRUNNING (the psw re-run held
# read_v3/arm_M1/instance.lock). Same env as mapthr_arm_psw.sh, LF kit.
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; K=$R/kit_mapcal_2408e921; LV=$RD/LIFT_VALID_M1.npz; AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $F/MAPTHR_STATUS.txt; }
( cd "$RD" && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_M1/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60     R9E1_MAP_THR_ARM="$F/thr_M1_pswpedge.json" R9E1_MAP_THR_REF="$F/thr_C0_pswpedge.json" R9E1_MAP_THR_FLOOR_DIR="$F/floor_thr_pswpedge"     bash "$AR" M1 > "$F/arm_M1_thr_pswpedge.rerun.out" 2>&1 < /dev/null )
ev "ARM pswpedge-rerun M1: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_M1_thr_pswpedge.rerun.out | tail -1)"
