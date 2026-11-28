#!/bin/bash
# run_4d_read.sh -- AMENDMENT 4d PHASE 2: one read of the H2cn stage at fit seed S on the registered battery tree: forward -> TAC rows -> verdict.  CPU only.  No git, nothing on G:.
#   bash run_4d_read.sh 1     # H2CN_FS1_DEPLOY8  (stage ship_r8_sub_fs/H2CN_FS1_STAGE_FROZEN.pt, registration REGISTERED_H2CN_FS1_STAGE.sha256)  -> out_4d_fs1/
#   bash run_4d_read.sh 2     # H2CN_FS2_DEPLOY8                                                                                                  -> out_4d_fs2/
#   bash run_4d_read.sh 0     # the REGISTERED seed-0 stage (read-only ship_r8_sub/H2CN_STAGE_FROZEN.pt) re-read from THIS copy of the code     -> out_4d_fs0/  (reproduction)
# Every step is judged on its ARTIFACT (the json exists and carries the field), never on an exit code read through a pipe.  Markers ZZRUN<..>ZZ in logs/run_4d_read_fs<S>.log.
# NEVER writes in ship_r8_sub/ (REG is read only).
set -u
S=${1:?usage: run_4d_read.sh 0|1|2}
W=C:/Users/Admin/refcv8_pf/ship_r8_sub_fs
REG=C:/Users/Admin/refcv8_pf/ship_r8_sub
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TREE=step48852ema_pf
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "F:/refcv9_4d_scratch/tmp" "$W/logs"
export TMP=F:/refcv9_4d_scratch/tmp TEMP=F:/refcv9_4d_scratch/tmp TMPDIR=F:/refcv9_4d_scratch/tmp
OUTR=$W/out_4d_fs$S
LOG=$W/logs/run_4d_read_fs$S.log
mk() { echo "ZZRUN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
case "$S" in
  0) FWD_ARGS="--mode h2cn --fit-seed 0 --stage-file $REG/H2CN_STAGE_FROZEN.pt --freeze-info $REG/H2CN_FREEZE_INFO.json --registered-stage $REG/REGISTERED_H2CN_STAGE.sha256 --out-root $OUTR"
     VER_ARGS="--fit-seed 0 --stage-file $REG/H2CN_STAGE_FROZEN.pt --registered-stage $REG/REGISTERED_H2CN_STAGE.sha256"; LABEL=H2CN_DEPLOY8 ;;
  1|2) FWD_ARGS="--mode h2cn --fit-seed $S --out-root $OUTR"; VER_ARGS="--fit-seed $S --label H2CN_FS${S}_DEPLOY8"; LABEL=H2CN_FS${S}_DEPLOY8 ;;
  *) echo "unknown fit seed $S"; exit 2 ;;
esac
TAG=h2cn
FWD=$OUTR/forward/FORWARD_H2CN_$TREE.json
mk START "fit seed $S -> $OUTR"
t0=$(date +%s)
rm -f "$FWD"
"$PY" -X utf8 "$W/ship_r8_sub_forward.py" $FWD_ARGS --tree $TREE > "$W/logs/fwd_4d_fs$S.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd_4d_fs$S.log; a WITHHELD_*.json names the control)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in H2CN_DEPLOY8 H2CN_NEAR RETIME_DEPLOY8; do
  t1=$(date +%s)
  AD=$OUTR/arms/$TAG/$A
  rm -f "$AD/TAC_$TREE.json" "$AD/TAC_RECORD_$TREE.json"
  "$PY" -X utf8 "$W/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_4d_fs${S}_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_4d_fs${S}_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
"$PY" -X utf8 "$W/ship_r8_sub_verdict.py" --arm H2CN_DEPLOY8 --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" --out-dir "$OUTR/ship_r8" $VER_ARGS > "$W/logs/verdict_4d_fs$S.log" 2>&1
has "$OUTR/ship_r8/SHIP_R8_$LABEL.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_$LABEL.json"; exit 4; }
mk END "$LABEL overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$OUTR/ship_r8/SHIP_R8_$LABEL.json") wall $(( $(date +%s) - t0 )) s"
