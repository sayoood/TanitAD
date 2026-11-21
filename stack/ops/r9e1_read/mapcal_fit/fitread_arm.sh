#!/usr/bin/env bash
# fitread_arm.sh <ARM> <KIT dir> <REF> [<REF> ...] -- MM 2026-11-06 (Thor). A map arm's pswpedgeF read (PREREG_M1_PEDGEFIT protocol + A1; for
# M1N20: PREREG_M1_NEAR20.md -- REFs "C0 M1": C0 DECISIVE, M1 attribution), once the chain's own fixed-128 reader has dumped the arm:
#   1 WAIT   read_v2/arm_<ARM>/READ_DONE (the chain reader's DEV dump exists) -> read_v3/arm_<ARM>/dump_<ARM> symlink + its STATUS_SNAPSHOT
#   2 DUMP   the arm's FIT dump (1,112 TRAIN-DIAG windows) under the Thor GPU lock with an idle-GPU check (mapcal_q.sh's command, <KIT>'s dumper)
#   3 FIT    thr_<ARM>_psw.json (IoU) and thr_<ARM>_pswpedgeF.json (psw IoU + pedge f_tol); the A1 check: both files' psw levels EQUAL
#   4 READ   per REF: r9e1_arm_read.sh <ARM> <REF> on read_v3 with R9E1_MAP_THR_ARM/_REF = thr_*_pswpedgeF, floor floor_thr_pswpedgeF, lift-valid
# Ends ZZFITREAD-DONEZZ or ZZFITREAD-ABORT-<why>ZZ.  Log: mapcal_fit/FITREAD_<ARM>.txt.  CPU except step 2.
export PATH=/usr/bin:/bin:$PATH; set -u
ARM="$1"; K="$2"; shift 2; REFS="$*"
R=/home/nvidia/r8ns/r9e1; RV=$R/read_v2; RD=$R/read_v3; F=$R/mapcal_fit; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock; LV=$RD/LIFT_VALID_M1.npz
AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh; FT=$K/stack/scripts/r9_map_fit_thresholds.py; FD=$F/floor_thr_pswpedgeF
ST=$F/FITREAD_$ARM.txt
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZFITREAD-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES=$F/fit_remap.json "$@"; }
PP="$K/stack:$K/stack/scripts:$K/taniteval"; PPC="$PP:$K/stack/ops/r9e1_read/shim_scipy"
[ -n "$REFS" ] || die ARGS "no reference given"
for f in "$AR" "$FT" "$LV" "$FD/FLOOR.json" "$F/fit_clips.json" "$F/fit_remap.json"; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ "$(grep -c $'\r' "$AR")" = 0 ] && bash -n "$AR" || die CRLF-KIT "$AR (export kits with git -c core.autocrlf=false archive)"
grep -q f_tol "$FT" || die OLD-KIT "$FT has no f_tol"
for REF in $REFS; do [ -s "$F/thr_${REF}_pswpedgeF.json" ] || die NO-REF-THR "$F/thr_${REF}_pswpedgeF.json"; done
ev START "arm $ARM refs [$REFS] kit $K"
# ---- 1 WAIT for the chain reader's dump ----
t0=$(date +%s)
until [ -e "$RV/arm_$ARM/READ_DONE" ]; do
  [ -e "$RV/arm_$ARM/READ_ABORTED" ] && die READ-ABORTED "$(head -1 $RV/arm_$ARM/READ_ABORTED | cut -c1-160)"
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && die WAIT "no read_v2/arm_$ARM/READ_DONE in 48 h"; sleep 300; done
mkdir -p "$RD/arm_$ARM"; [ -e "$RD/arm_$ARM/dump_$ARM" ] || ln -s "$(readlink -f $RV/arm_$ARM/dump_$ARM)" "$RD/arm_$ARM/dump_$ARM"
awk -v A="$ARM" '{L[NR]=$0} / START commit /{n++; s[n]=NR; i=index($0,"arms ["); if(i){ r=substr($0,i+6); j=index(r,"]"); r=substr(r,1,j-1); k=split(r,xs," ");
  for(q=1;q<=k;q++) if(xs[q]==A) last=n } } END{ if(!last) exit 1; e=(last<n)? s[last+1]-1 : NR; for(i2=1;i2<=e;i2++) print L[i2] }' \
  "$R/gate3/CHAIN_STATUS.txt" > "$RD/arm_$ARM/STATUS_SNAPSHOT.txt" || die NO-SEGMENT "no chain segment lists $ARM"
ev DEV-DUMP-READY "$RD/arm_$ARM/dump_$ARM -> $(readlink -f $RD/arm_$ARM/dump_$ARM)"
# ---- 2 FIT dump under the lock ----
RUN=$R/runs/r9e1-$ARM; O=$F/dump_$ARM
[ -s $RUN/model_ema.pt ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN"
if [ ! -s $O/dump.json ]; then
  exec 9>$LOCK; ev LOCK-WAIT "$LOCK"
  flock -w 86400 9 || die LOCK-TIMEOUT "no lock in 24 h"
  napp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$napp" = "0" ] || { flock -u 9; die GPU-BUSY "$napp compute app(s) under the lock"; }
  [ -e $O ] && mv $O $O.partial_$(date +%s)
  ev DUMP-START "$ARM"
  ( cd $K/stack && penv "$PP" $PY -X utf8 scripts/r9_e1_dump.py --ckpt $RUN/model_ema.pt --config $RUN/config.json --windows $F/fit_clips.json \
      --exclude-split $SPLIT --fit-k-per-ep 8 --require-ema --out $O > $F/dump_$ARM.log 2>&1 < /dev/null ); rc=$?
  flock -u 9; exec 9>&-; ev LOCK-RELEASED "rc $rc"
  [ -s $O/dump.json ] || die DUMP "rc $rc (read $F/dump_$ARM.log)"
  ev DUMP-DONE "$ARM rasters $(ls $O/rast | grep -c 'npz$')"
fi
# ---- 3 FIT ----
fitone() { local out="$1"; shift
  [ -s "$out" ] && return 0
  ( cd $K/stack && penv "$PPC" $PY -X utf8 "$FT" fit --rast-dir $O/rast --dev-hold $SPLIT --arm $ARM "$@" --out "$out" > "$out.log" 2>&1 < /dev/null ) || die FIT "(read $out.log)"; }
fitone $F/thr_${ARM}_psw.json --planes psw
fitone $F/thr_${ARM}_pswpedgeF.json --planes psw,pedge --plane-criterion pedge=f_tol
CHK=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8"))["planes"]; b=json.load(open(sys.argv[2],encoding="utf-8"))["planes"]
print(("EQUAL" if a["psw"]==b["psw"] else "DIFFER"), "psw", a["psw"], "pedge", b.get("pedge"))' $F/thr_${ARM}_psw.json $F/thr_${ARM}_pswpedgeF.json)
[ "${CHK%% *}" = EQUAL ] || die A1-PSW "$CHK"
ev FIT-DONE "$ARM ($CHK)"
# ---- 4 READS ----
for REF in $REFS; do
  ( cd "$RD" && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_$ARM/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60 \
      R9E1_MAP_THR_ARM="$F/thr_${ARM}_pswpedgeF.json" R9E1_MAP_THR_REF="$F/thr_${REF}_pswpedgeF.json" R9E1_MAP_THR_FLOOR_DIR="$FD" \
      bash "$AR" "$ARM" "$REF" > "$F/arm_${ARM}_vs_${REF}_pswpedgeF.out" 2>&1 < /dev/null )
  ev ARM "pswpedgeF $ARM vs $REF: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_${ARM}_vs_${REF}_pswpedgeF.out | tail -1) $(grep -ohE 'thr_[0-9a-f]{8}_[0-9a-f]{8}' $F/arm_${ARM}_vs_${REF}_pswpedgeF.out | tail -1) $(grep -ohE -- '-> (KEEP|DROP)' $F/arm_${ARM}_vs_${REF}_pswpedgeF.out | tail -1)"
done
ev DONE "arm $ARM read vs [$REFS]"
