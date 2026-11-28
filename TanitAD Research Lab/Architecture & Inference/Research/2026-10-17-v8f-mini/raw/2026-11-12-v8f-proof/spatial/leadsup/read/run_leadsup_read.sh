#!/usr/bin/env bash
# run_leadsup_read.sh [IN_DIR] [OUT_DIR]  -- the lead's ONE command for the lead read (section 3a of card PREREG_V8F_PROOF_ADD24), dev box, CPU only.
# It does not contact the pod. It checks the pulled files against their records and runs `leadsup_read.py run`. Exit 0 = every control passed, 2 = a control failed
# (no outcome is named then), 3 = a file is missing or does not match its record.
#
# ===================================================================================================================================================
# STEP A (pod side, per arm A in LS_s0 LS_s1 LSH_s0 LSH_s1 LSD_s0 LSD_s1; the lead runs these by the usual pod route, after the arm is DONE in its chain log)
# ===================================================================================================================================================
#   Queue Q = leadsup_a | leadsup_b | leadsup_c (the queue the arm was trained in), tree = tree_leadsup, PB = /workspace/v8proof/probe,
#   RR = /workspace/v8proof/runs/$Q, checkpoint = $RR/$A/ckpt_last.pt.
#
#   A1. Put pod_lead_apply.py on the pod once (the pod has no git credentials; use the usual file push), at $PB/pod_lead_apply.py.
#       Verify on both sides:  sha256sum pod_lead_apply.py   (this box: run `sha256sum F:/TanitAD-scratch/v8proof/spatial/leadsup/read/pod_lead_apply.py`).
#
#   A2. State extraction, exactly as step 2 of F:/TanitAD-scratch/v8proof/spatial/s1/gw_read_arm_q.sh with queue leadsup_a|b|c and tree tree_leadsup
#       (idempotent: the .npz.json record is written last):
#         cd $PB && mkdir -p states && for sp in 'train 600 1800' 'eval 0 141'; do set -- $sp; [ -f states/${A}_$1.npz.json ] || \
#           nice -n 19 ionice -c3 /workspace/venv/bin/python extract_states.py --tree /workspace/v8proof/tree_leadsup --ckpt $RR/$A/ckpt_last.pt \
#           --cache /workspace/v8proof/data/cache_pod --split $1 --lo $2 --hi $3 --out states/${A}_$1.npz > states/${A}_$1.log 2>&1; done
#       (or run the same script with GW_QUEUE=leadsup_a GW_TREE=tree_leadsup bash gw_read_arm_q.sh <ARM>: that also starts the dev-box s1 reads, which this
#        card does not need; the extraction block alone is enough).
#
#   A3. Apply the trained head to the two state files (CPU, no GPU; nice):
#         cd $PB/states && for sp in eval train; do nice -n 19 /workspace/venv/bin/python $PB/pod_lead_apply.py --tree /workspace/v8proof/tree_leadsup \
#           --ckpt $RR/$A/ckpt_last.pt --states ${A}_$sp.npz --out ${A}_${sp}_lead.npz; done
#       It refuses a checkpoint without a head and a state file extracted from another checkpoint. It writes, per state file, ${A}_${sp}_lead.npz,
#       ${A}_${sp}_lead.npz.head.npz (the six head tensors) and ${A}_${sp}_lead.npz.json (sha256 of checkpoint, states, outputs, head tensors; counts).
#
# ===================================================================================================================================================
# STEP B (pull to IN_DIR, default F:/TanitAD-scratch/v8proof/spatial/leadsup/read/in; about 175 MB per arm; F: has about 7 GB free, C: is too small)
# ===================================================================================================================================================
#   Per arm, binary-clean (pcat.sh as in gw_read_arm_q.sh: bash /c/Users/Admin/v8pod/pcat.sh <pod path> <local path>), 10 files:
#       $PB/states/${A}_eval.npz           ${A}_eval.npz.json           ${A}_train.npz            ${A}_train.npz.json
#       $PB/states/${A}_eval_lead.npz      ${A}_eval_lead.npz.json      ${A}_eval_lead.npz.head.npz
#       $PB/states/${A}_train_lead.npz     ${A}_train_lead.npz.json     ${A}_train_lead.npz.head.npz
#   This script then checks, per file: sha256 of the file = the record (out_sha256 for the four state files and the two lead outputs; head_sha256 for the head files),
#   the state file the head was applied to = the state file pulled here (states_sha256 of the lead record), and one checkpoint sha256 across the four records of the arm.
#   Retry a pull whose sha256 does not match (the gotty pty can corrupt a plain pull: frame the lines as in the pod notes).
#
# ===================================================================================================================================================
# STEP C (this script)
# ===================================================================================================================================================
#   bash run_leadsup_read.sh [IN_DIR] [OUT_DIR]      OUT_DIR default F:/TanitAD-scratch/v8proof/spatial/leadsup/read/out (RESULT_LEADSUP_T0.json, VERDICT_LEADSUP_T0.md)
#   Env (rarely needed): ARMS (comma list, default the six), B (default 2000), STATES_DIR + ALIAS (plumbing: ALIAS=PLUMB_s0=G_s0), EXTRA (e.g. --no-train --plumbing).
#   Needs the packs eval_pack.npz and train_pack.npz next to this script (built once by `python leadsup_read.py pack`; this script builds them if missing).
#   Memory: one Python job under 3 GB; the train probe states are 155 MB each as float16. About 1.5 minutes per arm at B 2000 (measured on the plumbing run, one arm).
#
# Controls k5 (train log), k6 (checkpoint identity with the T1-kin dump) and k7 (q3_read) belong to the lead and are not run here: the checkpoint sha256 of every arm
# is printed in RESULT_LEADSUP_T0.json (arms.<ARM>.ckpt_sha256) for k6.
set -u
HERE=/f/TanitAD-scratch/v8proof/spatial/leadsup/read
IN=${1:-$HERE/in}
OUT=${2:-$HERE/out}
ARMS=${ARMS:-LS_s0,LS_s1,LSH_s0,LSH_s1,LSD_s0,LSD_s1}
B=${B:-2000}
STATES_DIR=${STATES_DIR:-$IN}
ALIAS=${ALIAS:-}
EXTRA=${EXTRA:-}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
sha() { sha256sum "$1" | cut -c1-64; }
jget() { grep -o "\"$2\": \"[0-9a-f]*\"" "$1" | head -n 1 | cut -d'"' -f4; }
fail=0
chk() {  # file record-field record-file
  if [ ! -f "$1" ]; then echo "MISSING $1"; fail=1; return; fi
  if [ ! -f "$3" ]; then echo "MISSING $3"; fail=1; return; fi
  want=$(jget "$3" "$2"); got=$(sha "$1")
  if [ -z "$want" ] || [ "$want" != "$got" ]; then echo "SHA MISMATCH $1 (record $2 ${want:-none}, file $got)"; fail=1; fi
}
alias_of() { a=$1; for kv in ${ALIAS//,/ }; do [ "${kv%%=*}" = "$a" ] && { echo "${kv#*=}"; return; }; done; echo "$a"; }
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_leadsup_read: in=$IN out=$OUT arms=$ARMS B=$B"
for A in ${ARMS//,/ }; do
  S=$(alias_of "$A")
  for sp in eval train; do
    case " $EXTRA " in *" --no-train "*) [ "$sp" = train ] && continue;; esac
    chk "$STATES_DIR/${S}_$sp.npz" out_sha256 "$STATES_DIR/${S}_$sp.npz.json"
    chk "$IN/${A}_${sp}_lead.npz" out_sha256 "$IN/${A}_${sp}_lead.npz.json"
    chk "$IN/${A}_${sp}_lead.npz.head.npz" head_sha256 "$IN/${A}_${sp}_lead.npz.json"
    if [ -f "$IN/${A}_${sp}_lead.npz.json" ] && [ -f "$STATES_DIR/${S}_$sp.npz" ]; then
      [ "$(jget "$IN/${A}_${sp}_lead.npz.json" states_sha256)" = "$(sha "$STATES_DIR/${S}_$sp.npz")" ] || { echo "HEAD WAS APPLIED TO ANOTHER STATE FILE than $STATES_DIR/${S}_$sp.npz"; fail=1; }
    fi
  done
done
if [ $fail != 0 ]; then echo "RUN-LEADSUP-FAIL: input files missing or not matching their records"; exit 3; fi
[ -f "$HERE/eval_pack.npz" ] && [ -f "$HERE/train_pack.npz" ] || "$PY" "$HERE/leadsup_read.py" pack --out-dir "$HERE" || { echo "RUN-LEADSUP-FAIL: pack"; exit 3; }
mkdir -p "$OUT"
ALIAS_ARG=""; [ -n "$ALIAS" ] && ALIAS_ARG="--alias $ALIAS"
"$PY" "$HERE/leadsup_read.py" run --arms "$ARMS" --in-dir "$IN" --out-dir "$OUT" --states-dir "$STATES_DIR" --pack-dir "$HERE" --B "$B" $ALIAS_ARG $EXTRA 2>&1 | grep -v -e RuntimeWarning -e 'jump = '
rc=${PIPESTATUS[0]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] leadsup_read rc $rc; results: $OUT/RESULT_LEADSUP_T0.json $OUT/VERDICT_LEADSUP_T0.md"
exit $rc
