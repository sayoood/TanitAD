#!/usr/bin/env bash
# V-R8-noLW post-processing, end to end, so the PREREG_VR8_NOLW verdict exists the moment the arm does.
# Waits (10-min polls, <= 30 h) for Thor's noLW chain to log ZZNOLW-ALLDONE and the run's PULL_READY, then:
#   1 pull model_final.pt (+md5, config.json, metrics.jsonl) and the Thor batch-8 base dumps; md5-verify the model
#   2 dev-box eval: ladder_eval_diag.py --save-cands, base row, seeds 0,1, batch 4 (the recipe the V-R8 / V0 / V-R8d
#     comparison dumps were made with -- never compare against Thor batch-8 dumps for P1/P2)
#   3 oracle_variants.py -> oraclebase / oracleall / randombase
#   4 nolw_readout.py (pre-registered P1 / P2, written 2026-10-12 before the data) -> readout.json
# Exits 0 with ZZNOLWPOST-DONE-<verdict>; non-zero with ZZNOLWPOST-FAIL-<step> (never reads $? through a pipe).
set -u
NW=/home/nvidia/refcv8_ladder/noLW; RUN=$NW/arms/V-R8-noLW/run; EV=$NW/eval/V-R8-noLW
L=E:/lgdiag; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; T=tanitad-thor-wifi
fail() { echo "ZZNOLWPOST-FAIL-$1 $2"; exit 3; }
i=0
while :; do
  q=$(timeout 90 ssh -n -o ConnectTimeout=20 $T "a=\$(grep -c ZZNOLW-ALLDONE $NW/chain.log 2>/dev/null); x=\$(grep -c 'ZZNOLW-NOT-STARTED\|ZZSTOPZZ\|ZZDIE' $NW/chain.log 2>/dev/null); p=\$(ls $RUN/PULL_READY 2>/dev/null | wc -l); echo \"PP\${a:-0}-\${x:-0}-\${p}PP\"" 2>/dev/null)
  tok=$(printf '%s' "$q" | grep -o 'PP[0-9]*-[0-9]*-[0-9]*PP' | tail -1)
  echo "[$(date -u +%H:%MZ)] ${tok:-ssh-fail}"
  case "$tok" in PP[1-9]*-0-1PP) break ;; PP*-[1-9]*-*PP) fail wait "noLW chain logged a stop/not-started/die marker" ;; esac
  i=$((i + 1)); [ $i -ge 180 ] && fail wait "timeout after 30 h"
  sleep 600
done
mkdir -p $L/V-R8-noLW $L/thor/V-R8-noLW
for f in config.json metrics.jsonl model_final.pt model_final.pt.md5; do
  timeout 1800 scp -q $T:$RUN/$f $L/V-R8-noLW/$f || fail pull "$f"
done
for f in base_s0.npz base_s0.json base_s1.npz base_s1.json; do
  timeout 600 scp -q $T:$EV/$f $L/thor/V-R8-noLW/$f || fail pull "thor dump $f"
done
want=$(cut -c1-32 $L/V-R8-noLW/model_final.pt.md5); got=$(md5sum $L/V-R8-noLW/model_final.pt | cut -c1-32)
[ ${#want} -eq 32 ] && [ "$want" = "$got" ] || fail md5 "want=$want got=$got"
(cd $L/thor/V-R8-noLW && md5sum base_s*.npz base_s*.json > thor.md5)
echo "ZZNOLWPOST-PULLED md5 $got"
cd $L/code || fail cd code
MSYS_NO_PATHCONV=1 env -u GIT_DIR REFCV6_REPO=E:/lgt1df3 REFCV6_KIT=D:/refcv6_eval_kit \
  REFCV6_REMAP_OVERRIDES=E:/lgdiag/remap_overrides_devbox.json PYTHONPATH="E:/lgt1df3/stack;E:/lgt1df3/taniteval" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  $PY ladder_eval_diag.py --run $L/V-R8-noLW --ckpt $L/V-R8-noLW/model_final.pt --expect-step 3000 --rows base \
  --seeds 0,1 --batch 4 --save-cands --out-dir $L/eval_cands/V-R8-noLW --workers 0 > $L/eval_cands_V-R8-noLW.log 2>&1
for s in 0 1; do [ -f $L/eval_cands/V-R8-noLW/base_s$s.json ] || fail eval "no base_s$s.json (see eval_cands_V-R8-noLW.log)"; done
echo "ZZNOLWPOST-EVALED"
PYTHONPATH="E:/lgt1df3/stack;E:/lgt1df3/taniteval" $PY oracle_variants.py $L/eval_cands/V-R8-noLW > $L/oracle_V-R8-noLW.log 2>&1
[ -f $L/eval_cands/V-R8-noLW/oraclebase_s1.json ] || fail oracle "no oraclebase_s1.json"
mkdir -p $L/read/nolw
PYTHONPATH="E:/lgt1df3/stack;E:/lgt1df3/taniteval" $PY nolw_readout.py $L/eval_cands/V-R8-noLW $L/eval_cands/V-R8 \
  $L/thor/V-R8 $L/thor/V-R8r --out $L/read/nolw/readout.json > $L/read/nolw/readout.txt 2>&1
[ -s $L/read/nolw/readout.json ] || fail readout "no readout.json (see read/nolw/readout.txt)"
v=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" $L/read/nolw/readout.json)
echo "ZZNOLWPOST-DONE-$v"
