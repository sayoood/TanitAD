#!/usr/bin/env bash
# e2abl_roll_v2.sh <kit dir> <run_battery_r9_e2.py blob> <refcv3_arm.py blob> ["abl:tag ..."] -- v2 (21:30Z): the knockout list is arg 4; MM 4a3d05, 2026-11-12: ZERO-TRAINING diagnosis of the E2 NO-GO's LATERAL
# regression (GOALS H-R9-E2-DT).  Rolls K2 (runs/r9e2-K2M1N20) through the SAME E2 per-frame battery with ONE planner-wire knockout (R9E2_ABLATE), DEV, inference
# seeds 0,1, under the Thor GPU lock: map-zero (W-MAP-2) FIRST, then box-zero (W-BOX).  Tree = a COPY of the frozen E2 read tree with exactly two files replaced
# by the landed blobs (run_battery_r9_e2.py, refcv3_arm.py); env = the E2 read's env file; the battery invocation = r9e2_roll.sh roll_arm, step for step.
# After each arm the child's census (roll_s{0,1}.json r9e2_stage.ablation.applied) must name the knockout, or the arm ABORTS (a knockout that did not reach
# the model would read exactly like a K2 baseline).  Writes runs/E2_ABL_K2M1N20/{E2_K2abl_mapzero_pf,E2_K2abl_boxzero_pf}/ + STATUS.txt.
# Ends ZZE2ABL-DONEZZ or ZZE2ABL-ABORT-<why>ZZ (also in STATUS.txt).
export PATH=/usr/bin:/bin:$PATH; set -u
KIT=${1:?kit dir}; BATT=${2:?run_battery_r9_e2.py blob}; ARMB=${3:?refcv3_arm.py blob}
PAIRS=${4:-r9_map_chan_zero:E2_K2abl_mapzero_pf r9_box_tok_zero:E2_K2abl_boxzero_pf}
R=/home/nvidia/r8ns/r9e2; SRC=$R/read_tree_cf653ca4bb47_lfix_e77c39d4_g55f14ef2; TREE=$R/read_tree_cf653ca4bb47_abl_${BATT:0:8}
ENVF=$R/inputs/r9e2_read_env.thor.sh_760d28d7; OUT=$R/runs/E2_ABL_K2M1N20; ST=$OUT/STATUS.txt; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir -p $OUT/logs
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZE2ABL-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
[ "$(b $KIT/run_battery_r9_e2.py)" = "$BATT" ] || die KIT "kit run_battery_r9_e2.py is not blob $BATT"
[ "$(b $KIT/refcv3_arm.py)" = "$ARMB" ] || die KIT "kit refcv3_arm.py is not blob $ARMB"
[ -d $SRC/stack ] || die NO-SRC "$SRC"
if [ ! -d $TREE ]; then
  [ -e $TREE.tmp ] && die TREE-TMP "$TREE.tmp exists (a crashed copy): inspect it"
  cp -a $SRC $TREE.tmp && cp $KIT/run_battery_r9_e2.py $TREE.tmp/stack/ops/r9e2_read/run_battery_r9_e2.py && cp $KIT/refcv3_arm.py $TREE.tmp/taniteval/tools/refcv3_arm.py \
    && mv $TREE.tmp $TREE || die TREE "cannot build $TREE"
  printf 'copy of %s with stack/ops/r9e2_read/run_battery_r9_e2.py (%s) and taniteval/tools/refcv3_arm.py (%s) from the landed tip; built %s by e2abl_roll.sh\n' \
    "$SRC" "$BATT" "$ARMB" "$(date -u +%FT%TZ)" > $TREE.provenance
fi
[ "$(b $TREE/stack/ops/r9e2_read/run_battery_r9_e2.py)" = "$BATT" ] && [ "$(b $TREE/taniteval/tools/refcv3_arm.py)" = "$ARMB" ] || die TREE-BLOB "$TREE does not carry the kit blobs"
ev START "tree $TREE out $OUT knockouts [$PAIRS] seeds 0,1"
exec 9>$LOCK; ev LOCK-WAIT "$LOCK"
flock -w 43200 9 || die LOCK-TIMEOUT "no lock in 12 h"
NA=1; for _i in $(seq 1 20); do NA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'); [ "$NA" = 0 ] && break; sleep 15; done
[ "$NA" = 0 ] || { flock -u 9; die GPU-BUSY "$NA compute app(s) under the lock after 5 min"; }
ev LOCK-TAKEN "GPU empty"
for pair in $PAIRS; do
  ABL=${pair%%:*}; TAG=${pair##*:}
  if [ -s $OUT/$TAG/dump_s0/manifest.json ] && [ -s $OUT/$TAG/dump_s1/manifest.json ] && [ -s $OUT/$TAG/ABL_CENSUS_OK ]; then ev SKIP "$TAG complete"; continue; fi
  ev ROLL-START "$ABL tag $TAG"
  ( export R9E2_TREE=$TREE; source $ENVF K2M1N20 > $OUT/logs/env_$TAG.log 2>&1 || exit 7
    export R8_LAUNCH_TREE=$TREE R9E2_OUT=$OUT R9E2_ABLATE=$ABL
    PP="$TREE/stack:$TREE/stack/scripts:$TREE/taniteval"
    cd "${R9E2_BASE_CWD:-.}" || exit 8
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
        -u REFCV6_KIT -u REFCV6_REMAP_OVERRIDES -u REFCV6_REPO -u R8_NO_POSE_SYNC \
        PYTHONPATH="$PP" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
        HF_TOKEN_PATH=/nonexistent/hf_token $PY -X utf8 $TREE/stack/ops/r9e2_read/run_battery_r9_e2.py --ckpt "$R9E2_K2_RUN/model_ema.pt" \
        --config "$R9E2_K2_RUN/config.json" --no-g0 --no-baselines --tag $TAG --infer-seeds 0,1 --lock-held --out-root $OUT
  ) > $OUT/logs/roll_$TAG.out 2> $OUT/logs/roll_$TAG.err < /dev/null; rc=$?
  for s in 0 1; do [ -s $OUT/$TAG/dump_s$s/manifest.json ] && [ -s $OUT/$TAG/roll_s$s.json ] || { flock -u 9; die ROLL "$ABL rc $rc: seed $s dump/roll missing (read $OUT/logs/roll_$TAG.err)"; }; done
  C=$(python3 -c 'import json,sys
out=[]
for s in ("0","1"):
    st=json.load(open(sys.argv[1]+"/roll_s"+s+".json",encoding="utf-8")).get("r9e2_stage") or {}
    ab=(st.get("ablation") or {}); ap=ab.get("applied") or {}
    if ab.get("requested")!=sys.argv[2] or ap.get("name")!=sys.argv[2]: print("BAD seed",s,json.dumps(ab)[:300]); sys.exit(1)
    out.append("s%s:%s%s"%(s,ap["name"]," INERT" if (ap.get("record") or {}).get("INERT") else ""))
print(" ".join(out))' $OUT/$TAG $ABL) || { flock -u 9; die CENSUS "$ABL: $C"; }
  echo "$C" > $OUT/$TAG/ABL_CENSUS_OK
  ev ROLL-DONE "$ABL rc $rc census $C"
done
flock -u 9; exec 9>&-; ev LOCK-RELEASED "$LOCK"
ev DONE "all knockout arms rolled [$PAIRS]: $OUT"
