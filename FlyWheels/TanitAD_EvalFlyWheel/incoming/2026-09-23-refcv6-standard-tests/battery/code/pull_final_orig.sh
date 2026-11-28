#!/bin/sh
# READ-ONLY pull of the FINAL refcv6 checkpoint (Thor run dir ckpt.pt), only after summary.json says
# done. md5 on Thor BEFORE and AFTER the copy plus on the dev box: all three must be equal, and the
# checkpoint's own `step` field must equal summary.json's step. Opaque markers (ZZ...ZZ).
H=tanitad-thor-wifi
RUN=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
DST=/d/refcv6_eval_kit/ckpt
RAW=/c/Users/Admin/ev6_battery/raw/thor_reads
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=20"
PYW=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$RAW"
i=0
while [ $i -lt 300 ]; do
  d=$(timeout 60 $SSH $H "cat $RUN/summary.json" 2>/dev/null)
  case "$d" in *'"done": true'*) break;; esac
  echo "ZZFINALWAITZZ $(date +%FT%T)"
  sleep 600; i=$((i+1))
done
case "$d" in *'"done": true'*) ;; *) echo "ZZFINALFAILZZ no summary.json"; exit 3;; esac
echo "$d" > "$RAW/summary_final.json"
m1=$(timeout 600 $SSH $H "md5sum $RUN/ckpt.pt" 2>/dev/null | cut -d' ' -f1)
[ ${#m1} -eq 32 ] || { echo "ZZFINALFAILZZ md5-before unreadable"; exit 3; }
timeout 3600 scp -q -o BatchMode=yes $H:$RUN/ckpt.pt "$DST/ckpt_final.pt.part" || { echo "ZZFINALFAILZZ scp"; exit 3; }
m2=$(timeout 600 $SSH $H "md5sum $RUN/ckpt.pt" 2>/dev/null | cut -d' ' -f1)
m3=$(md5sum "$DST/ckpt_final.pt.part" | cut -d' ' -f1)
if [ ${#m2} -ne 32 ] || [ ${#m3} -ne 32 ] || [ "$m1" != "$m2" ] || [ "$m1" != "$m3" ]; then
  echo "ZZFINALFAILZZ md5 before=$m1 after=$m2 local=$m3"; exit 3
fi
st=$($PYW -c "import torch; print(int(torch.load(r'D:/refcv6_eval_kit/ckpt/ckpt_final.pt.part', map_location='cpu', weights_only=False)['step']))")
want=$($PYW -c "import json; print(int(json.load(open(r'C:/Users/Admin/ev6_battery/raw/thor_reads/summary_final.json'))['step']))")
[ "$st" = "$want" ] || { echo "ZZFINALFAILZZ ckpt step $st != summary step $want"; exit 3; }
mv "$DST/ckpt_final.pt.part" "$DST/ckpt_final.pt"
echo "$m3  ckpt_final.pt  (step $st; md5 on Thor before=after=$m1)" >> "$DST/MD5SUMS"
TS=$(date +%Y%m%dT%H%M)
timeout 300 scp -q -o BatchMode=yes $H:$RUN/metrics.jsonl "$RAW/metrics_${TS}.jsonl" || { echo "ZZFINALFAILZZ metrics-scp"; exit 3; }
echo "ZZFINALOKZZ md5 $m3 step $st metrics metrics_${TS}.jsonl"
