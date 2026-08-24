#!/usr/bin/env bash
# Wait for the first POST-BREAK checkpoint, then stop.
#
# ⛔ `--save-every 2500` OVERWRITES a single rolling `ckpt.pt`; there are NO
# numbered snapshots. My first watcher globbed for `ckpt_step*.pt` and would have
# waited forever while looking perfectly healthy. Watch the MTIME and the STEP
# together — the filename carries no step information at all, so "which step is
# this checkpoint?" can only be answered by pairing the two.
#
# The break was at step 5400 (o11_loss 1.3956 -> 0.4024, pick 0.25 -> 1.00,
# sep_rel 0.01 -> 12-17). The step-5000 save on disk is PRE-break and useless for
# scoring the arm; the step-7500 save will be the first post-break state.
set -u
BASE=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'stat -c %Y /home/nvidia/v7tiny/o11p30k/ckpt.pt' 2>/dev/null | tr -d '\r')
echo "$(date +%H:%M) baseline ckpt mtime=$BASE  (the PRE-break step-5000 save)"
while true; do
  R=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
    'echo "ZZM-$(stat -c %Y /home/nvidia/v7tiny/o11p30k/ckpt.pt)ZZ";
     echo "ZZS-$(tail -1 /home/nvidia/v7tiny/o11p30k/train_log.jsonl | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get(\"step\",0))")ZZ";
     echo "ZZP-$(ps -eo args | grep -c "[t]rain_v6_staged")ZZ"' 2>/dev/null)
  M=$(echo "$R" | grep -oE "ZZM-[0-9]+ZZ" | grep -oE "[0-9]+")
  S=$(echo "$R" | grep -oE "ZZS-[0-9]+ZZ" | grep -oE "[0-9]+")
  P=$(echo "$R" | grep -oE "ZZP-[0-9]+ZZ" | grep -oE "[0-9]+")
  echo "$(date +%H:%M) step=${S:-?} mtime=${M:-?} alive=${P:-?}"
  if [ -n "$M" ] && [ -n "$BASE" ] && [ "$M" -gt "$BASE" ] 2>/dev/null; then
    echo "ZZCKPT-REFRESHED-at-step-${S}ZZ"
    break
  fi
  # ⚠️ cover the failure path too: a filter that only matches the happy outcome
  # stays silent through a crash, and silence looks identical to "still running".
  if [ "${P:-1}" = "0" ]; then
    echo "ZZTRAINER-GONE-at-step-${S}ZZ"
    break
  fi
  sleep 240
done
