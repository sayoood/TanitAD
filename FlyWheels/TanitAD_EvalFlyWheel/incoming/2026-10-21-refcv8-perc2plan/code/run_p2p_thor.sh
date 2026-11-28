#!/bin/bash
# run_p2p_thor.sh <ARM> <SEED> -- ONE PERC2PLAN-R8 os-only roll on Thor (PREREG_PERC2PLAN.md sha256 d1a2eb53...).
# Run ONLY under the Thor GPU lock (chain_p2p_thor.sh does: flock -w 7200 /home/nvidia/refcv7_post/thor_gpu.lock ...).
# Opaque markers in out/<ARM>_s<SEED>/p2p.log: ZZRSTART / ZZRBUSY / ZZRSKIP / ZZREND rc=<n>.
set -u
ARM=${1:?arm}; SEED=${2:?seed}
P=/home/nvidia/r8ns/perc2plan; R=$P/ref; O=$P/out/${ARM}_s$SEED; mkdir -p $O
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION P2P_DEBUG_NOHOOK
export R8_LAUNCH_TREE=$R/tree R8_EVAL_KIT=/home/nvidia PF_R8_CODE=$R/code/batt/code PF_NF_HOLD=$R/code/navfix
export R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/tree/stack:$R/tree/taniteval
LOG=$O/p2p.log
if [ -s $O/meta_s$SEED.json ] && $PY -I -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m['C4']['all_pass'] and not m['child_rc'] else 1)" $O/meta_s$SEED.json; then
  echo "ZZRSKIP $(date -Is) already complete" >> $LOG; exit 0; fi
echo "ZZRSTART $(date -Is) pid $$ arm $ARM seed $SEED" >> $LOG
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
pys=$(ps -u nvidia -o pid=,args= | awk '$2 ~ /python/' | grep -vE 'r8ns/nt/|code/nt_[a-z0-9_]+[.]py' | wc -l)  # NT-DECIDE CPU pipeline excluded (MM 2026-10-21)
echo "precheck compute_apps=$apps other_python=$pys" >> $LOG
if [ "$apps" != "0" ] || [ "$pys" != "0" ]; then echo "ZZRBUSY" >> $LOG; exit 3; fi
BANK=""
case $ARM in
  BASE) BANK="--donor-bank-out $P/out/BASE_s$SEED/donor_bank_s$SEED.pt";;
  *SHUF) [ -s $P/out/BASE_s$SEED/donor_bank_s$SEED.pt ] || { echo "ZZRNOBANK" >> $LOG; exit 5; }
         BANK="--donor-bank-in $P/out/BASE_s$SEED/donor_bank_s$SEED.pt";;
esac
rm -rf $O/dump
cd $O
$PY -X utf8 $P/code/perc2plan_roll_r8_pf.py --arm $ARM --pf-code $R/code/pf/code --remap-json $P/code/refcv8_eval_remap_thor_p2p.json \
  --ckpt /home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt --config /home/nvidia/r8ns/ckpt/config.json --seed $SEED \
  --dump-dir $O/dump --windows-json $R/inputs/windows_s1.json --out-json $O/roll_s$SEED.json \
  --meta-json $O/meta_s$SEED.json --device cuda --os-only $BANK >> $LOG 2>&1 200>&-
echo "ZZREND rc=$? $(date -Is)" >> $LOG
