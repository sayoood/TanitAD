#!/bin/bash
# nt_download.sh -- H-NT-DECIDE-1 E1 data: stream the 31 remaining OpenScene navtrain_current shards (2..32) to Thor.
# PI-authorised 2026-10-21 ~20:40 Berlin (registered f2f604f). Public HF OpenDriveLab/OpenScene @ the SAME pinned revision the
# S1 download used (a76f840b...). Each shard is sha256-verified against its HF LFS oid (raw_hf_tree_navsim.json) before it is
# offered to the E1 consumer (nt_e1_run.py), which deletes it after its tokens are written AND verified.
#
# STREAMING RULE (brief): at most 2 shards in flight = (downloading) + (downloaded, not yet consumed).  A shard file exists in
# $DL as <name>.part (downloading) -> <name> + <name>.ok (verified, offered).  The consumer removes <name> and <name>.ok.
# DISK FLOOR: a download starts only if free bytes - its size >= FLOOR_GB (default 35).  Resumable: rerun picks up .part files
# (curl -C -) and skips shards already consumed (listed in $DL/CONSUMED.txt by the consumer) or already verified.
# Markers (opaque, disjoint from any searched word): ZZDLSTART / ZZDLFILE <st> ... / ZZDLEND.
set -u
NT=/home/nvidia/r8ns/nt; DL=$NT/dl; LOG=$NT/logs/dl.log; PROG=$NT/logs/DL_PROGRESS.json
REV=a76f840b65e972bc45e56c2adced897498e9a026
B="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/$REV"
FLOOR_GB=${FLOOR_GB:-35}
MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-2}
mkdir -p "$DL" "$NT/logs"; touch "$DL/CONSUMED.txt"
echo "ZZDLSTART $(date -Is) pid $$ floor_gb=$FLOOR_GB max_in_flight=$MAX_IN_FLIGHT" >> "$LOG"
in_flight() { ls "$DL"/navtrain_current_*.tgz "$DL"/navtrain_current_*.tgz.part 2>/dev/null | wc -l; }
free_b() { df -B1 --output=avail /home/nvidia | tail -1 | tr -d ' '; }
n_ok=0; n_bad=0; tot_b=0; tot_s=0
while read -r P SZ SHA; do
  N=$(basename "$P"); F="$DL/$N"
  if grep -qx "$N" "$DL/CONSUMED.txt"; then continue; fi
  if [ -f "$F.ok" ]; then continue; fi
  # wait for a slot (the .part of THIS shard, if resuming, already counts as one)
  while :; do
    k=$(in_flight); [ -f "$F.part" ] && k=$((k-1))
    fb=$(free_b); need=$(( SZ + FLOOR_GB*1000000000 ))
    [ -f "$F.part" ] && need=$(( need - $(stat -c %s "$F.part") ))
    if [ "$k" -lt "$MAX_IN_FLIGHT" ] && [ "$fb" -ge "$need" ]; then break; fi
    sleep 30
  done
  for attempt in 1 2 3; do
    t0=$(date +%s)
    nice -n 19 ionice -c3 curl -L --fail --silent --show-error -C - --retry 8 --retry-delay 10 --speed-limit 100000 --speed-time 120 \
      -o "$F.part" "$B/$P" >> "$LOG" 2>&1
    rc=$?; t1=$(date +%s); got=$(stat -c %s "$F.part" 2>/dev/null || echo 0)
    if [ "$got" = "$SZ" ]; then
      h=$(nice -n 19 ionice -c3 sha256sum "$F.part" | cut -d' ' -f1)
    else h=size_mismatch; fi
    secs=$((t1-t0)); [ $secs -lt 1 ] && secs=1
    if [ "$h" = "$SHA" ]; then
      mv "$F.part" "$F"; echo "{\"name\":\"$N\",\"bytes\":$SZ,\"sha256\":\"$SHA\",\"verified\":\"$(date -Is)\"}" > "$F.ok"
      n_ok=$((n_ok+1)); tot_b=$((tot_b+SZ)); tot_s=$((tot_s+secs))
      echo "ZZDLFILE OK $N attempt=$attempt rc=$rc bytes=$got/$SZ secs=$secs MBps=$(echo "scale=2; $got/$secs/1000000" | bc) sha=${h:0:12} free=$(free_b) at $(date -Is)" >> "$LOG"
      break
    else
      echo "ZZDLFILE BAD $N attempt=$attempt rc=$rc bytes=$got/$SZ secs=$secs sha=${h:0:12} at $(date -Is)" >> "$LOG"
      if [ "$got" = "$SZ" ] || [ "$rc" != 0 -a "$got" -gt "$SZ" ]; then rm -f "$F.part"; fi
      n_bad=$((n_bad+1)); sleep 20
    fi
  done
  printf '{"updated":"%s","n_ok_this_run":%d,"n_bad_attempts":%d,"bytes_ok_this_run":%d,"secs_ok_this_run":%d,"mean_MBps":%s}\n' \
    "$(date -Is)" $n_ok $n_bad $tot_b $tot_s "$( [ $tot_s -gt 0 ] && echo "scale=2; $tot_b/$tot_s/1000000" | bc || echo null)" > "$PROG"
done < "$NT/code/shards_2_32.txt"
echo "ZZDLEND $(date -Is) ok=$n_ok bad_attempts=$n_bad bytes=$tot_b secs=$tot_s" >> "$LOG"
