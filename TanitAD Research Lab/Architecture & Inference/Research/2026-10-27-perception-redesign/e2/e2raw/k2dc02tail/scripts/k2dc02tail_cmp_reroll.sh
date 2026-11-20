#!/bin/bash
# k2dc02tail_cmp_reroll.sh -- OPTIONAL, NOT LAUNCHED BY THE KIT (an MM decision, ~2.6 GPU-h, zero training). Re-makes the two COMPARATOR planner rolls that E2-TAIL deleted from Thor
# after its pull and never banked (MEASURED 2026-11-20: no E2_K0_pf / E2_K2_pf of the E2-TAIL read exists on Thor or under F:/refcv9_e3/e2tail_*):
#   $W/cmp/E2_K0_pf          = K0's TAIL weights  (runs/r9e2-K0/model_ema_tail5k6k.pt)   -> the K0 slot of stage B's families / distkeep, and the dose tool's --k0-pf
#   $W/cmp_k2dc/E2_K2_pf     = K2DC's TAIL weights (runs/r9e2-K2DC/model_ema_tail5k6k.pt)  -> the dose tool's --k2-pf (the K2DCt comparator of PREREG_K2DC02TAIL sec. 3)
# Each is EXACTLY r9e2_roll.sh's roll_arm command (run_battery_r9_e2.py --no-g0 --no-baselines --infer-seeds 0,1 --lock-held, tag E2_K0_pf / E2_K2_pf, cwd = R9E2_BASE_CWD), under ONE GPU-lock
# hold with the GPU-empty check, same read tree / env as stage B (read_tree_e454e96c6a84_k2dc, env d4bdcf07). INTEGRITY CHECK (run by the dev-box wrapper after the pull): families on these two
# rerolled dumps must reproduce E2-TAIL's banked e2tail/stageB/read/families_s{0,1}.json (K2DCt - K0t); a mismatch voids the re-roll as a comparator.
# Ends ZZK2DC02TAILCMP-DONEZZ / -ABORT-<why>ZZ. Refuses if an output exists (a read is never overwritten).
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e2; W=$R/runs/E2TAIL_K2DC02; TREE=$R/read_tree_e454e96c6a84_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_d4bdcf07
CK=model_ema_tail5k6k.pt; ST=$W/CMP_STATUS.txt; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; PY=/home/nvidia/venvs/tanitad-train/bin/python
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZK2DC02TAILCMP-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
mkdir -p $W
# shellcheck disable=SC1090
export R9E2_TREE=$TREE R8_LAUNCH_TREE=$TREE   # MM 11-20 fix: export BEFORE sourcing; the env derives PF_NF_HOLD / R9E2_REMAP from R9E2_TREE (else the deleted cf653ca4bb47_lfix default)
source $ENVF K2DC02 || die ENV "cannot source $ENVF"
T=$TREE; export R9E2_TREE=$TREE; export R8_LAUNCH_TREE=$TREE
PYPATH="$T/stack:$T/stack/scripts:$T/taniteval"; OPS=$T/stack/ops/r9e2_read
renv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO -u REFCV6_KIT -u REFCV6_REMAP_OVERRIDES -u REFCV6_REPO -u R8_NO_POSE_SYNC \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token "$@"; }
for RUN in $R/runs/r9e2-K0 $R/runs/r9e2-K2DC; do [ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN/$CK"; done
[ ! -e $W/cmp ] && [ ! -e $W/cmp_k2dc ] || die EXISTS "$W/cmp or $W/cmp_k2dc exists"
FREE=$(df -BM --output=avail /home/nvidia | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -gt $((15360 + 300)) ] || die DISK "free ${FREE} MB"
ev START "comparator re-roll K0t -> $W/cmp/E2_K0_pf and K2DCt -> $W/cmp_k2dc/E2_K2_pf (tree $TREE)"
exec 9>$LOCKF; ev LOCK-WAIT "$LOCKF"; flock -w 43200 9 || die LOCK-TIMEOUT "no lock in 12 h"
NA=1; for _i in $(seq 1 20); do NA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'); [ "$NA" = 0 ] && break; sleep 15; done
[ "$NA" = 0 ] || die GPU-NOT-EMPTY "$NA compute app(s) after 5 min; never share the GPU"
roll() {   # $1 label, $2 run dir, $3 tag, $4 out root
  mkdir -p $4/logs || die MKDIR "$4"
  ev ROLL-START "$1 tag $3 ckpt $2/$CK"
  ( cd "${R9E2_BASE_CWD:-.}" && renv "$PYPATH" "$PY" -X utf8 "$OPS/run_battery_r9_e2.py" --ckpt "$2/$CK" --config "$2/config.json" --no-g0 --no-baselines \
      --tag "$3" --infer-seeds 0,1 --lock-held --out-root "$4" > "$4/logs/roll_$1.out" 2> "$4/logs/roll_$1.err" ); rc=$?
  for s in 0 1; do [ -s "$4/$3/dump_s$s/manifest.json" ] && [ -s "$4/$3/windows_s$s.json" ] || die ROLL-FAILED "$1 rc $rc seed $s (read $4/logs/roll_$1.err)"; done
  [ -s "$4/$3/r9e2_dev_filter.json" ] || die ROLL-FAILED "$1 rc $rc: no r9e2_dev_filter.json"
  ev ROLL-DONE "$1 rc $rc $4/$3"
}
roll K0 $R/runs/r9e2-K0 E2_K0_pf $W/cmp
roll K2DC $R/runs/r9e2-K2DC E2_K2_pf $W/cmp_k2dc
flock -u 9; exec 9>&-
ev DONE "both comparator rolls done; pull $W/cmp/E2_K0_pf and $W/cmp_k2dc/E2_K2_pf, then the integrity check"
