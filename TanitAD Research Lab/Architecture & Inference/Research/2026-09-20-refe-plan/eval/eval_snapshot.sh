#!/bin/bash
# Banked copy of the dev-box snapshot-eval runner (the working copy lives in the session scratchpad as eval_snap.sh).
# One checkpoint through the whole dev-box eval: md5-verified fetch -> learning-curve point (PDMS, --prev chain)
# -> SPEC E-6 (every proposal scored, selection readout) -> RESULT_E6 doc + raw copies -> SPEC Amendment 4.
# usage: eval_snap.sh NNN|final WORKERS      prints ZZSNAP_DONE <N> or ZZSNAP_FAIL <N> <stage>
#   NNN   an epoch snapshot snap_epochNNN.pt (written at the boundary INTO epoch NNN)
#   final model_final.pt, the model after the last step -- fetched only once summary.json says done, and checked
#         against the sha256 the trainer wrote into summary.json as well as against the pod's md5
N="$1"; W="${2:-2}"
PY="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
E="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
D="D:/Projects/TanitAD/data/refe_runs_eval"
P="D:/Projects/TanitAD/data/refe_navtest/proptable"
R="/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3"
T="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json"
K="-i $HOME/.ssh/tanitad_pod"
fail() { echo "ZZSNAP_FAIL $N $1"; exit 1; }
if [ "$N" = final ]; then F="model_final.pt"; NAME="sub200_final"; TAG="final"
elif [[ "$N" =~ ^[0-9]{3}$ ]]; then F="snap_epoch$N.pt"; NAME="sub200_ep$N"; TAG="ep$N"
else echo "usage: eval_snap.sh NNN|final [WORKERS]"; exit 2; fi
echo "start $(date -u +%FT%TZ) $F workers $W"
if [ "$N" = final ]; then
  scp -q $K -P <port> "root@<redacted-ip>:$R/summary.json" "$D/final_summary.json" || fail summary_fetch
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True else 1)" \
    "$D/final_summary.json" || fail "summary_not_done"
fi
if [ ! -f "$D/$F" ]; then
  scp -q $K -P <port> "root@<redacted-ip>:$R/$F" "$D/$F.part" || fail scp
  mv "$D/$F.part" "$D/$F"
fi
l=$(md5sum "$D/$F" | cut -c1-32)
r=$(ssh -n $K -p <port> -o ConnectTimeout=20 root@<redacted-ip> "md5sum $R/$F" | cut -c1-32)
if [ ${#l} -ne 32 ] || [ "$l" != "$r" ]; then rm -f "$D/$F"; fail "md5 l=$l r=$r"; fi
if [ "$N" = final ]; then
  s=$(sha256sum "$D/$F" | cut -c1-64)
  x=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model_final_sha256'])" "$D/final_summary.json")
  if [ ${#s} -ne 64 ] || [ "$s" != "$x" ]; then fail "sha256 local=$s summary=$x"; fi
  echo "sha256 matches summary.json: ${s:0:16}"
fi
echo "fetched $F md5 $l $(date -u +%FT%TZ)"
args=()
for f in "$D"/snap_epoch[0-9][0-9][0-9].pt; do
  [ -f "$f" ] || continue
  n=${f##*snap_epoch}; n=${n%.pt}
  if [ "$N" = final ] || [ "$((10#$n))" -le "$((10#$N))" ]; then args+=(--ckpt "ep$n=$f"); fi
done
[ "$N" = final ] && args+=(--ckpt "final=$D/$F")
cd "$E" || fail cd
"$PY" learning_curve.py --tokens "$T" --prefix sub200 "${args[@]}" || fail curve
echo "curve done $(date -u +%FT%TZ)"
# W3's harness aborts a run when the dev box's free RAM stays under 3 GB (W3_RAM_GUARD_ABORT, rc=3; other sessions'
# jobs share the box). proposal_table skips runs that already PASSED, so a retry redoes only the aborted ones.
ok=0
for w in "$W" 1 1; do
  if "$PY" proposal_table.py --ckpt "$D/$F" --name "$NAME" --tokens "$T" --reuse-dump --workers "$w"; then ok=1; break; fi
  echo "proptable pass failed; retrying the failed runs with 1 worker in 60 s"; sleep 60
done
[ $ok = 1 ] || fail proptable
"$PY" selection_readout.py --name "$NAME" --tokens "$T" > "$P/$NAME/readout.log" 2>&1 \
  || { cat "$P/$NAME/readout.log"; fail readout; }
tail -3 "$P/$NAME/readout.log"
"$PY" write_result_e6.py --name "$NAME" || fail result_doc
mkdir -p "$E/raw/e6_$NAME"
cp "$P/$NAME/readout.json" "$P/$NAME/gates.json" "$P/$NAME/readout.log" "$E/raw/e6_$NAME/" || fail raw_copy
# SPEC_NAVTEST AMENDMENT 4 from the first on-policy snapshot on (the trainer's own bank lines decide eligibility;
# exit 2 = NOT_ELIGIBLE is an answer, not a failure)
if [ "$N" = final ] || [ "$((10#$N))" -ge 13 ]; then
  scp -q $K -P <port> "root@<redacted-ip>:$R/metrics.jsonl" "$P/$NAME/metrics_at_eval.jsonl" || fail metrics_fetch
  "$PY" amendment4_readout.py --name "$NAME" --metrics "$P/$NAME/metrics_at_eval.jsonl" > "$P/$NAME/a4.log" 2>&1
  rc=$?
  cat "$P/$NAME/a4.log"
  { [ $rc = 0 ] || [ $rc = 2 ]; } || fail amendment4
  if [ -f "$P/$NAME/amendment4.json" ]; then cp "$P/$NAME/amendment4.json" "$E/raw/e6_$NAME/"; fi
fi
echo "ZZSNAP_DONE $N $TAG $(date -u +%FT%TZ)"
