#!/bin/bash
# DATASCALE-10K streaming driver launcher (pod). Idempotent/resumable: re-run the same command after any kill.
#   full : nohup setsid bash /root/datascale/ds_run10k.sh < /dev/null > /root/datascale/run10k.stdout 2>&1 & disown
#   smoke: bash /root/datascale/ds_run10k.sh --smoke
# Final line of /root/datascale/run10k.log (smoke: smoke10k.log): 'ZZ DATASCALE-10K DONE ZZ ...' or 'ZZ DATASCALE-10K FAIL ZZ <why>'.
# No address, port or pod id in this file.
B=/root/datascale
. /root/ops/lorapod2_env.sh
env_lora
# the orchestrator itself needs the hub ONLINE (stage + upload); the build/encode/join children get the offline vars re-added by ds_stream.py
unset HF_HUB_OFFLINE HF_HUB_CACHE TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_XET=1
cd $B || exit 1
LOGF=$B/run10k.log
ARGS=""
if [ "$1" = "--smoke" ]; then LOGF=$B/smoke10k.log; ARGS="--smoke --log $LOGF"; fi
if [ -z "$ARGS" ]; then ARGS="--log $LOGF"; fi
/root/venv/bin/python $B/ds_stream.py $ARGS
rc=$?
if ! tail -3 $LOGF | grep -q "ZZ DATASCALE-10K "; then
  echo "[ds10k $(date '+%F %T')] ZZ DATASCALE-10K FAIL ZZ driver exited rc=$rc without a marker" >> $LOGF
fi
exit $rc
