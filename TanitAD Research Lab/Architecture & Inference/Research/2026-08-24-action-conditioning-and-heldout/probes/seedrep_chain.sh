#!/usr/bin/env bash
# TRACK A'S REPLICATION TEST — does the distilled-init lever survive a seed change?
#
# ⭐ WHY IT DECIDES SOMETHING. Collapse and representation are the campaign's only
# POSITIVE results, and the lever behind both is `--init-from <distilled ckpt>`.
# Every number supporting it is SINGLE-SEED. The claim is a drift fraction of
# 0.175-0.365 for distilled arms against 0.614-0.642 for scratch — non-overlapping
# ranges, which is why it survived tonight's null recalibration when the t-based
# claims did not. If seed 1 lands outside the distilled band, the one solved axis
# is not solved and v7-full must not launch.
#
# ⛔ THE WAIT IS CONTENT-BASED, NOT PROCESS-BASED. CLAUDE.md records that a poll
# whose filter contains the pattern it searches for matches its own echoed command
# — measured three times, and I hit the PowerShell version of it today (a chain
# waited forever because Get-CimInstance enumerated the shell running the query).
# Waiting on `summary.json` sidesteps the whole family: it is a FACT about the run,
# not a string about the poller.
set -u
SPD="$(cd "$(dirname "$0")" && pwd)"
cd "$SPD" || exit 1
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
S=/home/nvidia/v7tiny/postrain30k_seed1

while true; do
  ok=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi \
        "test -f $S/summary.json && echo YES || echo NO" 2>/dev/null | tr -d '\r')
  [ "$ok" = "YES" ] && break
  sleep 120
done
echo "ZZSEED1-DONE ZZ"

mkdir -p "$SPD/v7tiny_postrain30k_seed1"
ssh -n -o ConnectTimeout=30 tanitad-thor-wifi "cat $S/ckpt.pt" \
  > "$SPD/v7tiny_postrain30k_seed1/ckpt.pt" 2>/dev/null
ssh -n -o ConnectTimeout=30 tanitad-thor-wifi "cat $S/config.json" \
  > "$SPD/v7tiny_postrain30k_seed1/config.json" 2>/dev/null
want=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi "md5sum $S/ckpt.pt | cut -c1-32" 2>/dev/null | tr -d '\r')
got=$(md5sum "$SPD/v7tiny_postrain30k_seed1/ckpt.pt" | cut -c1-32)
echo "ZZMD5 want=$want got=$got ZZ"
[ "$want" = "$got" ] || { echo "ZZREFUSE md5 mismatch ZZ"; exit 1; }

# ⭐ the drift column of latentmotion.py IS the drift fraction, and it is the same
# instrument that measured rdw8p30k at +0.674 (a scratch arm, in the scratch band)
# — so the comparison is on one instrument, not across two.
SPD_ARMS="postrain30k,postrain30k_seed1" SPD_OUT="$SPD/seedrep.json" \
  PYTHONIOENCODING=utf-8 "$PY" latentmotion.py > "$SPD/seedrep.log" 2>&1
echo "ZZSEEDREP rc=$? ZZ"
