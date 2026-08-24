#!/usr/bin/env bash
# O11 close-out: PULL and STOP first, SCORE later.
#
# ⭐ WHY THE STOP DOES NOT WAIT FOR THE SCORING. The DEGENERATE verdict is already
# sustained across steps 5400-7200 (o11_loss 1.3956 -> 0.0005, pick_acc 0.25 ->
# 1.000, sep_rel 0.01 -> 12-17, o5 +18.7 % against the adjacent window). The
# pre-registration's DEGENERATE branch is decided by THAT, not by actchan —
# actchan is a bonus finding about whether the PRIMARY METRIC CAN BE GAMED, and it
# can be run from the checkpoint at any time. Letting the arm run to 30,000 would
# spend ~6 more hours of the ONLY GPU we have re-confirming a settled verdict.
# ⇒ pull (md5-verified) -> write the done-marker -> stop the trainer -> score when
# the dev box is free.
#
# ⛔ THE DONE-MARKER IS WRITTEN BEFORE THE KILL, NOT AFTER. A supervised run whose
# marker is missing gets RESURRECTED the moment whatever blocked its relaunch is
# fixed — that cost this programme 2 days and a corrupted run directory. There is
# no supervisor here, but the marker is also the record of WHY the run stopped,
# and a run that stops without one is indistinguishable from a crash.
set -u
SPD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
cd "$SPD"

while ! grep -qE "ZZCKPT-REFRESHED|ZZTRAINER-GONE" o11_ckpt_watch.log 2>/dev/null; do sleep 60; done
echo "ZZWATCH-FIRED $(grep -oE 'ZZ(CKPT-REFRESHED|TRAINER-GONE)-at-step-[0-9]+ZZ' o11_ckpt_watch.log | tail -1) ZZ"

DEST="$SPD/v7tiny_o11p30k"; mkdir -p "$DEST"
STEP=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'tail -1 /home/nvidia/v7tiny/o11p30k/train_log.jsonl | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get(\"step\",0))"' 2>/dev/null | tr -d '\r')
RMD5=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'md5sum /home/nvidia/v7tiny/o11p30k/ckpt.pt | cut -d" " -f1' 2>/dev/null | tr -d '\r')
scp -q tanitad-thor-wifi:/home/nvidia/v7tiny/o11p30k/ckpt.pt "$DEST/ckpt.pt"
scp -q tanitad-thor-wifi:/home/nvidia/v7tiny/o11p30k/config.json "$DEST/config.json"
scp -q tanitad-thor-wifi:/home/nvidia/v7tiny/o11p30k/train_log.jsonl "$DEST/train_log.jsonl"
LMD5=$(/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile \
  -Command "(Get-FileHash -Algorithm MD5 '$DEST/ckpt.pt').Hash.ToLower()" 2>/dev/null | tr -d '\r')
if [ "$RMD5" != "$LMD5" ]; then echo "ZZPULL-MD5-MISMATCH r=$RMD5 l=$LMD5 ZZ"; exit 1; fi
echo "{\"provenance\":\"o11p30k rolling ckpt.pt, log step $STEP at pull time\",\"md5\":\"$LMD5\"}" > "$DEST/summary.json"
echo "ZZPULL-OK step $STEP md5 $LMD5 ZZ"

# done-marker on Thor, then stop by EXPLICIT PID (never pkill -f, which
# self-matches the ssh command and kills the session instead of the trainer)
ssh -n -o ConnectTimeout=25 tanitad-thor-wifi "cat > /home/nvidia/v7tiny/o11p30k/summary.json <<'EOF'
{\"done\": true, \"run\": \"o11p30k\", \"stopped_at_step\": $STEP, \"planned_steps\": 30000,
 \"stopped_by\": \"PREREG DEGENERATE branch, 2026-08-24\",
 \"reason\": \"MEASURED: o11_loss 1.3956 -> 0.00046 against the ln4 floor with pick_acc 0.25 -> 1.000, but sep_rel 0.01 -> 12-17 and o5 +18.7 percent against the adjacent window. This is zhat = f(z) + lambda*a, the trivial minimiser named in the term's own docstring. E-DEC-39 shows why it was the only solution available: the action is not recoverably present in the latent transition.\",
 \"_evidence_class\": \"MEASURED (ours; Thor)\"}
EOF
echo wrote-marker" 2>&1 | tail -1

PID=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'ps -eo pid,args | grep "[t]rain_v6_staged" | awk "{print \$1}" | head -1' 2>/dev/null | tr -d '\r')
if [ -n "$PID" ]; then
  ssh -n -o ConnectTimeout=25 tanitad-thor-wifi "kill $PID" 2>/dev/null
  sleep 20
  ALIVE=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi 'ps -eo args | grep -c "[t]rain_v6_staged"' 2>/dev/null | tr -d '\r')
  echo "ZZSTOPPED pid $PID, remaining $ALIVE ZZ"
else
  echo "ZZNO-PID trainer already gone ZZ"
fi
echo "ZZTHOR-FREE ZZ"

# scoring waits for the dev GPU; the arm is already stopped so nothing is burning
while /c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'idm_oracle|rangeprobe|physics|deltaz' }" 2>/dev/null | grep -q python; do sleep 60; done
export PYTHONPATH="C:/Users/Admin/tanitad-mirror/stack" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
SPD_ARMS="o11p30k,rdw8p30k" SPD_OUT="$SPD/actchan_o11.json" \
  /c/Users/Admin/venvs/tanitad/Scripts/python.exe actchan.py > actchan_o11.log 2>&1
echo "ZZACTCHAN-$?ZZ"
SPD_ARMS="o11p30k" SPD_OUT="$SPD/physics_o11.json" \
  /c/Users/Admin/venvs/tanitad/Scripts/python.exe physics.py > physics_o11.log 2>&1
echo "ZZPHYSICS-$?ZZ"
echo "ZZCLOSEOUT-DONEZZ"
