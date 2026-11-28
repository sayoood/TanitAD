#!/bin/bash
# DATASCALE pilot driver (pod). Idempotent: every phase resumes from its ledger. Launch detached:
#   nohup setsid bash /root/datascale/ds_run.sh < /dev/null > /root/datascale/pilot.stdout 2>&1 &
# Final line of /root/datascale/pilot.log is 'ZZ DATASCALE-PILOT DONE ZZ' or 'ZZ DATASCALE-PILOT FAIL ZZ <why>'.
# No address, port or pod id in this file.
B=/root/datascale
LOG=$B/pilot.log
. /root/ops/lorapod2_env.sh
env_lora
PY=/root/venv/bin/python
cd $B || exit 1
say() { echo "[pilot $(date '+%F %T')] $*" >> $LOG; }
fail() { say "ZZ DATASCALE-PILOT FAIL ZZ $*"; exit 1; }
phase() { # name cmd...
  local n=$1; shift
  local t0=$(date +%s.%N)
  say "PHASE $n start"
  "$@" >> $B/phase_$n.out 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  echo "$n $t0 $t1" >> $B/phases.tsv
  say "PHASE $n rc=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.0f", b-a}')s"
  return $rc
}
say "start pid=$$ scope_sha=$(sha256sum $B/SCOPE_DATASCALE.md | cut -c1-16)"
[ -s $B/selection/pilot_clips.txt ] || fail "no pilot list"
# --- stage controls + pilot (range reads)
phase stage_pilot env -u HF_HUB_OFFLINE -u HF_HUB_CACHE -u TRANSFORMERS_OFFLINE $PY ds_stage.py --lists selection/control_clips.txt selection/pilot_clips.txt --threads 8 || fail "stage rc"
grep -q "ZZ DS-STAGE DONE ZZ" phase_stage_pilot.out || say "WARN stage partial (clips with FAIL stay in ledger_stage.jsonl)"
# --- v2ep builds (controls first: the gate)
: > /dev/null
for s in ctl pilot; do
  if [ $s = ctl ]; then L=selection/control_clips.txt; else L=$B/selection/pilot_ok.txt; fi
  if [ $s = pilot ]; then
    $PY - <<'EOF'
import json
ok=set()
for ln in open("/root/datascale/ledger_stage.jsonl",encoding="utf-8"):
    r=json.loads(ln)
    if r.get("status")=="OK": ok.add(r["clip"])
p=[c.strip() for c in open("/root/datascale/selection/pilot_clips.txt",encoding="utf-8") if c.strip()]
open("/root/datascale/selection/pilot_ok.txt","w",encoding="utf-8").write("\n".join(c for c in p if c in ok)+"\n")
EOF
  fi
  phase build_$s $PY ds_build.py --clips $L --v2ep-root $B/v2ep_$s --ledger $B/ledger_$s.jsonl --log $B/build_$s.log --workers 13 || fail "build $s rc"
  if [ $s = ctl ]; then
    mkdir -p $B/ctl_ref/train
    for c in $(cat selection/control_clips.txt); do cp -n /root/data/cache_pod/train/clips/$c.pt $B/ctl_ref/train/$c.pt; done
    phase encode_ctl $PY ds_encode.py --set ctl || fail "encode ctl rc"
  fi
done
phase encode_pilot $PY ds_encode.py --set pilot || fail "encode pilot rc"
# --- obstacle join (best effort) and label probe (best effort)
mkdir -p $B/stage/obsroot/labels/egomotion
cp -n $B/stage/labels/egomotion/egomotion_all.zip $B/stage/obsroot/labels/egomotion/egomotion.chunk_0000.zip
NV=$(ls $B/v2ep_pilot/train/*.v2ep.pt | wc -l)
phase join_pilot $PY /root/tad/stack/scripts/build_obstacle_join.py --corpus $B/v2ep_pilot/train --episodes $NV --out $B/join_pilot.jsonl \
   --hf-cache $B/hfc --labels-root $B/stage/obsroot --no-download || say "WARN obstacle join rc!=0 (see phase_join_pilot.out)"
phase labelprobe $PY $B/ds_labelprobe.py || say "WARN label probe rc!=0"
$PY $B/ds_report.py > $B/phase_report.out 2>&1 || say "WARN report rc!=0"
NE=$(wc -l < $B/cache_pilot/train/manifest.jsonl)
if grep -q "ZZ CACHEPOD DONE ZZ" $B/phase_encode_pilot.out && [ "$NE" -ge 1 ]; then
  say "ZZ DATASCALE-PILOT DONE ZZ encoded=$NE report=$B/PILOT_REPORT.json"
else
  say "ZZ DATASCALE-PILOT FAIL ZZ encode marker missing or no cache rows (encoded=$NE)"
fi
