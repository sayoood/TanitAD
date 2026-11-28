#!/bin/bash
# k2dc02nhtail_repro_roll.sh [DRY] -- PREREG_K2DC02S1 sec. 2 item 6 (the REPRODUCTION roll), zero training, ~0.65 GPU-h. Derived from k2dc02tail_cmp_reroll.sh, which re-made BOTH comparators;
# THIS kit re-makes NOTHING but one roll: K0t at inference seed 0 ONLY, in the K2DC02nh tail environment (same read tree, roll blob 38161299, env blob d4bdcf07, OMP_NUM_THREADS=6 as stage B):
#   $W/repro/E2_K0_pf   = K0's TAIL weights (runs/r9e2-K0/model_ema_tail5k6k.pt), dump_s0 only.
# The devbox check (k2dc02nhtail_repro_check.py) then compares: pinned K2DC02t - re-rolled K0t  vs  pinned K2DC02t - pinned K0t, LAT_cross_mae_m, within +-1 mm, else REPRO-VOID.
# The pinned comparator rolls under runs/E2TAIL_K2DC02/ are NEVER written, moved or re-rolled; this script writes only under $W (runs/E2TAIL_K2DC02nh).
# The roll is EXACTLY r9e2_roll.sh's roll_arm command (run_battery_r9_e2.py --no-g0 --no-baselines --lock-held, tag E2_K0_pf, cwd = R9E2_BASE_CWD) with --infer-seeds 0 instead of 0,1, under ONE GPU-lock hold
# with the GPU-empty check (the shared lock /home/nvidia/refcv7_post/thor_gpu.lock). Refuses if the output exists (a read is never overwritten).
# DRY (K2S_DRY=1 or first arg DRY): every guard runs; the lock is NOT taken, nothing is written (status lines go to stdout only), the roll command is printed. Ends REPRO-DRY-OK or the first ABORT.
# Ends ZZK2DC02NHTAILREPRO-DONEZZ / -ABORT-<why>ZZ.
export PATH=/usr/bin:/bin:$PATH; set -u
DRY=0; if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then DRY=1; fi
TC=${K2S_TRAIN_COMMIT:-}
R=/home/nvidia/r8ns/r9e2; W=$R/runs/E2TAIL_K2DC02nh; TREE=$R/read_tree_${TC}_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_d4bdcf07
ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb; ENV_BLOB=d4bdcf07cf337758ee0770f72091a23e6bf4eb34
CK=model_ema_tail5k6k.pt; ST=$W/REPRO_STATUS.txt; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; PY=/home/nvidia/venvs/tanitad-train/bin/python
ev() { [ $DRY = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZK2DC02NHTAILREPRO-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
# ---- guard 1 (the FIRST): the trained arm exists; this reproduction is only meaningful beside its tail roll ----
[ -s $R/runs/r9e2-K2DC02nh/model_ema.pt ] && [ -s $R/runs/r9e2-K2DC02nh/config.json ] || die NO-RUN "$R/runs/r9e2-K2DC02nh lacks model_ema.pt / config.json: K2DC02nh has not trained"
[ ${#TC} = 12 ] || die NO-TRAIN-COMMIT "K2S_TRAIN_COMMIT (the 12-char commit K2DC02nh trained from) is required, got '$TC'"
[ -d $TREE/stack ] || die NO-TREE "$TREE: build the EMA read tree first (e2read_launch_k2dc02nh.sh)"
[ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die ROLL-BLOB "tree roll blob != $ROLL_BLOB (the reproduction must run in the K2DC02nh tail's environment)"
[ "$(b $ENVF)" = $ENV_BLOB ] || die ENV-BLOB "$ENVF blob != $ENV_BLOB"
# shellcheck disable=SC1090
export R9E2_TREE=$TREE R8_LAUNCH_TREE=$TREE   # export BEFORE sourcing; the env derives PF_NF_HOLD / R9E2_REMAP from R9E2_TREE
source $ENVF K2DC02nh || die ENV "cannot source $ENVF"
T=$TREE; export R9E2_TREE=$TREE; export R8_LAUNCH_TREE=$TREE
PYPATH="$T/stack:$T/stack/scripts:$T/taniteval"; OPS=$T/stack/ops/r9e2_read
renv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO -u REFCV6_KIT -u REFCV6_REMAP_OVERRIDES -u REFCV6_REPO -u R8_NO_POSE_SYNC \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token "$@"; }
RUN=$R/runs/r9e2-K0
[ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN/$CK"
[ ! -e $W/repro ] || die EXISTS "$W/repro exists (a read is never overwritten)"
FREE=$(df -BM --output=avail /home/nvidia | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -gt $((15360 + 300)) ] || die DISK "free ${FREE} MB"
OUTROOT=$W/repro
CMD=( "$PY" -X utf8 "$OPS/run_battery_r9_e2.py" --ckpt "$RUN/$CK" --config "$RUN/config.json" --no-g0 --no-baselines --tag E2_K0_pf --infer-seeds 0 --lock-held --out-root "$OUTROOT" )
if [ $DRY = 1 ]; then
  echo "[dry] would: take $LOCKF, require 0 GPU compute apps, then ( cd ${R9E2_BASE_CWD:-.} && OMP_NUM_THREADS=6 ${CMD[*]} )  # K0t, inference seed 0 only"
  ev DRY-OK "all guards pass; the roll was not started"; exit 0
fi
mkdir -p $W
ev START "reproduction roll K0t seed 0 only -> $OUTROOT/E2_K0_pf (tree $TREE, env blob d4bdcf07, roll blob 38161299)"
exec 9>$LOCKF; ev LOCK-WAIT "$LOCKF"; flock -w 43200 9 || die LOCK-TIMEOUT "no lock in 12 h"
NA=1; for _i in $(seq 1 20); do NA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'); [ "$NA" = 0 ] && break; sleep 15; done
[ "$NA" = 0 ] || die GPU-NOT-EMPTY "$NA compute app(s) after 5 min; never share the GPU"
mkdir -p $OUTROOT/logs || die MKDIR "$OUTROOT"
ev ROLL-START "K0 tag E2_K0_pf ckpt $RUN/$CK"
( cd "${R9E2_BASE_CWD:-.}" && renv "$PYPATH" "${CMD[@]}" > "$OUTROOT/logs/roll_K0.out" 2> "$OUTROOT/logs/roll_K0.err" ); rc=$?
[ -s "$OUTROOT/E2_K0_pf/dump_s0/manifest.json" ] && [ -s "$OUTROOT/E2_K0_pf/windows_s0.json" ] || die ROLL-FAILED "K0 rc $rc seed 0 (read $OUTROOT/logs/roll_K0.err)"
[ -s "$OUTROOT/E2_K0_pf/r9e2_dev_filter.json" ] || die ROLL-FAILED "K0 rc $rc: no r9e2_dev_filter.json"
ev ROLL-DONE "K0 rc $rc $OUTROOT/E2_K0_pf"
flock -u 9; exec 9>&-
ev DONE "reproduction roll done; pull $OUTROOT/E2_K0_pf, then k2dc02nhtail_repro_check.py"
