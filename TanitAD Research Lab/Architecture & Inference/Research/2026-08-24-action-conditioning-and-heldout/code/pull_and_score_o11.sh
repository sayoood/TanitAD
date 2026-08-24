#!/usr/bin/env bash
# Pull the first POST-BREAK o11p30k checkpoint and run the PRE-REGISTERED PRIMARY
# READ on it, then stop the run.
#
# ⭐ WHY SCORE A CHECKPOINT WE ALREADY BELIEVE IS DEGENERATE. The prereg commits
# to `actchan` — d_out(shuffle_all)/d_out(latent +10 % control) — NOT to the
# training log. A degenerate arm (zhat = f(z) + lambda*a) should score VERY HIGH
# there, because replacing the actions genuinely moves its prediction a great
# deal. **If it does, that is a finding about THE METRIC, not only about the arm:
# the pre-registered primary read can be maxed out by a model that predicts
# WORSE.** That is worth knowing before the metric is used to accept a future arm.
#
# ⛔ `--save-every 2500` OVERWRITES a single rolling ckpt.pt; the filename carries
# no step, so the step is recorded here from the log at pull time and written
# beside the file. A checkpoint whose step is unknown is not quotable.
set -u
SPD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
cd "$SPD"
DEST="$SPD/v7tiny_o11p30k"
mkdir -p "$DEST"

STEP=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'tail -1 /home/nvidia/v7tiny/o11p30k/train_log.jsonl | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get(\"step\",0))"' 2>/dev/null | tr -d '\r')
RMD5=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
  'md5sum /home/nvidia/v7tiny/o11p30k/ckpt.pt | cut -d" " -f1' 2>/dev/null | tr -d '\r')
echo "pulling ckpt.pt (log step $STEP, remote md5 $RMD5)"
scp -q tanitad-thor-wifi:/home/nvidia/v7tiny/o11p30k/ckpt.pt "$DEST/ckpt.pt"
scp -q tanitad-thor-wifi:/home/nvidia/v7tiny/o11p30k/config.json "$DEST/config.json"
LMD5=$(/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile \
  -Command "(Get-FileHash -Algorithm MD5 '$DEST/ckpt.pt').Hash.ToLower()" 2>/dev/null | tr -d '\r')
if [ "$RMD5" != "$LMD5" ]; then
  echo "ZZPULL-MD5-MISMATCH remote=$RMD5 local=$LMD5 ZZ"; exit 1
fi
echo "{\"provenance\":\"o11p30k rolling ckpt.pt pulled at log step $STEP\",\"md5\":\"$LMD5\"}" \
  > "$DEST/summary.json"
echo "ZZPULL-OK-step-$STEP-md5-$LMD5 ZZ"

export PYTHONPATH="C:/Users/Admin/tanitad-mirror/stack" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
# score the O11 arm against the SAME three baselines actchan already measured, so
# the comparison is on identical windows, clips and seed.
SPD_ARMS="o11p30k,rdw8p30k" /c/Users/Admin/venvs/tanitad/Scripts/python.exe actchan.py > actchan_o11.log 2>&1
echo "ZZACTCHAN-$?ZZ"
SPD_ARMS="o11p30k" /c/Users/Admin/venvs/tanitad/Scripts/python.exe physics.py > physics_o11.log 2>&1
echo "ZZPHYSICS-$?ZZ"
