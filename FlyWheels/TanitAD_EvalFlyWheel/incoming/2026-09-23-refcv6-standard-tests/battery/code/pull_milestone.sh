#!/bin/sh
# READ-ONLY pull of a refcv6 milestone checkpoint from Thor (brief: scp + ssh -n ls/md5sum only).
#   sh pull_milestone.sh 5000            -> D:/refcv6_eval_kit/ckpt/ckpt_5000.pt (md5-verified)
# Refuses unless (a) the file exists, (b) its size is stable over two reads 20 s apart, and
# (c) metrics.jsonl already carries the eval row for that step (the trainer writes the milestone
# BEFORE the eval, so an eval row at the step means the save has finished).
# Emits ZZPULLOK<step>ZZ / ZZPULLWAIT<step>ZZ / ZZPULLFAIL<step>ZZ -- never the words it searches.
STEP="$1"
H=tanitad-thor-wifi
RUN=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
DST=/d/refcv6_eval_kit/ckpt
RAW=/c/Users/Admin/ev6_battery/raw/thor_reads
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=20"
mkdir -p "$RAW"
s1=$(timeout 60 $SSH $H "stat -c %s $RUN/ckpt_${STEP}.pt" 2>/dev/null)
[ -n "$s1" ] || { echo "ZZPULLWAIT${STEP}ZZ no-file"; exit 2; }
sleep 20
s2=$(timeout 60 $SSH $H "stat -c %s $RUN/ckpt_${STEP}.pt" 2>/dev/null)
[ "$s1" = "$s2" ] || { echo "ZZPULLWAIT${STEP}ZZ size-moving $s1 $s2"; exit 2; }
TS=$(date +%Y%m%dT%H%M)
timeout 300 scp -q -o BatchMode=yes $H:$RUN/metrics.jsonl "$RAW/metrics_${TS}.jsonl" || { echo "ZZPULLFAIL${STEP}ZZ metrics-scp"; exit 3; }
PYW=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
have=$($PYW -c "
import json,sys
rows=[json.loads(l) for l in open(r'C:/Users/Admin/ev6_battery/raw/thor_reads/metrics_${TS}.jsonl',encoding='utf-8') if l.strip()]
print(sum(1 for r in rows if r.get('step')==${STEP} and 'eval_loss' in r))")
[ "$have" = "1" ] || { echo "ZZPULLWAIT${STEP}ZZ no-eval-row-yet ($have)"; exit 2; }
rm=$(timeout 300 $SSH $H "md5sum $RUN/ckpt_${STEP}.pt" 2>/dev/null | cut -d' ' -f1)
[ ${#rm} -eq 32 ] || { echo "ZZPULLFAIL${STEP}ZZ remote-md5-unreadable"; exit 3; }
timeout 1800 scp -q -o BatchMode=yes $H:$RUN/ckpt_${STEP}.pt "$DST/ckpt_${STEP}.pt.part" || { echo "ZZPULLFAIL${STEP}ZZ scp"; exit 3; }
lm=$(md5sum "$DST/ckpt_${STEP}.pt.part" | cut -d' ' -f1)
if [ "$lm" = "$rm" ] && [ ${#lm} -eq 32 ]; then
  mv "$DST/ckpt_${STEP}.pt.part" "$DST/ckpt_${STEP}.pt"
  echo "$lm  ckpt_${STEP}.pt" >> "$DST/MD5SUMS"
  echo "ZZPULLOK${STEP}ZZ md5 $lm size $s2 metrics metrics_${TS}.jsonl"
else
  echo "ZZPULLFAIL${STEP}ZZ md5-mismatch local=$lm remote=$rm"; exit 3
fi
