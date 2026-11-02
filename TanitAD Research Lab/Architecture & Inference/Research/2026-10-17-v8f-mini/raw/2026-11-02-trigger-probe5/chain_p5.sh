#!/bin/bash
# chain_p5.sh -- dev-box script chain (no model turns): wait for the staged shards -> REQUIRE the align gate -> ship to box 2 (size + sha256 verified) -> launch the encode via launch.ps1
# -> poll by script -> pull the features -> sha256 verify against the box-2 json -> write raw/CHAIN_P5.done (or .failed).  Detached start: powershell -ExecutionPolicy Bypass -File launch_dev.ps1 (see HANDOFF.md).
set -u
P=/e/Projects/TanitAD-artifacts/dagland/trigger_probe5
SCR=/f/TanitAD-scratch/trigger_probe5_scratch
LOG=$P/raw/logs/chain_p5.log
BUSY=/c/Users/Admin/box2_ship/BOX2_BUSY.txt
B2="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" >> "$LOG"; }
fail() { say "FAILED: $*"; echo "$*" > "$P/raw/CHAIN_P5.failed"; rm -f "$BUSY"; exit 1; }
rm -f "$P/raw/CHAIN_P5.failed" "$P/raw/CHAIN_P5.done"

say "chain start; waiting for SHARDS-DONE"
for i in $(seq 1 600); do
  grep -q SHARDS-DONE "$P/raw/logs/stage_shards.log" && [ "$(ls $SCR/shards/*.done 2>/dev/null | wc -l)" -ge 23 ] && break
  sleep 60
done
[ "$(ls $SCR/shards/*.done 2>/dev/null | wc -l)" -ge 23 ] || fail "staging did not finish (23 shards expected)"
[ -f "$P/raw/ALIGN_GATE_PASS" ] || fail "REFUSED: the known-answer alignment gate has not passed (raw/ALIGN_GATE_PASS missing)"

echo "probe5 trigger crops encode, started $(date +%Y-%m-%dT%H:%M:%S)" > "$BUSY"
say "shipping shards"
cd "$SCR/shards" || fail "no shards dir"
for f in shard_*.zip shard_*.idx.json; do
  for t in 1 2 3 4 5; do
    $SCP "$f" "tanitad-box2:D:/tanitad/jobs/probe5/shards/" && break
    sleep 20
  done
done
$SCP "$P/hash_shards.ps1" tanitad-box2:D:/tanitad/jobs/probe5/ || fail "could not ship hash_shards.ps1"
for round in 1 2 3; do
  $B2 "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\jobs\\probe5\\hash_shards.ps1" 2>/dev/null | grep -E "^[0-9a-f]{64}  shard_" | sort -k2 > "$SCR/remote_sha.txt"
  ( sha256sum shard_*.zip | sed 's/ \*/  /' | sort -k2 ) > "$SCR/local_sha.txt"
  if diff -q "$SCR/local_sha.txt" "$SCR/remote_sha.txt" > /dev/null; then say "ship verified: $(wc -l < $SCR/local_sha.txt) shards, sha256 equal both ends"; break; fi
  say "ship mismatch round $round; re-sending differing files"
  for f in $(diff <(cat "$SCR/local_sha.txt") <(cat "$SCR/remote_sha.txt") | grep '^<' | awk '{print $3}'); do $SCP "$f" "tanitad-box2:D:/tanitad/jobs/probe5/shards/"; done
  [ "$round" = 3 ] && fail "shard ship sha256 mismatch after 3 rounds"
done

launch() {
  n=$($B2 "tasklist /FI \"IMAGENAME eq python.exe\" | find /c \"python\"" 2>/dev/null | tail -1 | tr -d '\r ')
  for w in $(seq 1 30); do [ "$n" = "0" ] && break; say "box 2 python busy ($n); waiting"; sleep 60; n=$($B2 "tasklist /FI \"IMAGENAME eq python.exe\" | find /c \"python\"" 2>/dev/null | tail -1 | tr -d '\r '); done
  [ "$n" = "0" ] || fail "box 2 never became idle"
  say "launch: $($B2 'powershell -NoProfile -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\probe5\p5_crops.cmd -Log D:\tanitad\logs\p5_crops.log' 2>/dev/null | tail -1)"
}
launch
relaunch=0
for i in $(seq 1 360); do
  sleep 60
  $SCP tanitad-box2:D:/tanitad/logs/p5_crops.log "$P/raw/logs/box2_crops.log" 2>/dev/null
  grep -q "ZZJOB-END" "$P/raw/logs/box2_crops.log" 2>/dev/null && break
  if [ $((i % 10)) = 0 ]; then
    n=$($B2 "tasklist /FI \"IMAGENAME eq python.exe\" | find /c \"python\"" 2>/dev/null | tail -1 | tr -d '\r ')
    if [ "$n" = "0" ] && [ "$relaunch" -lt 3 ]; then relaunch=$((relaunch + 1)); say "job dead without ZZJOB-END (reboot/crash?); relaunch $relaunch (resumes per shard)"; launch; fi
  fi
done
grep -q "ZZJOB-END" "$P/raw/logs/box2_crops.log" || fail "encode never finished (6 h)"
say "box 2 job ended: $(grep ZZJOB-END $P/raw/logs/box2_crops.log | tail -1)"

for t in 1 2 3; do $SCP -r "tanitad-box2:D:/tanitad/jobs/probe5/out" "$SCR/" && break; sleep 30; done
cd "$SCR/out" || fail "no pulled out dir"
bad=0; n=0
for j in feat_shard_*.json; do
  want=$(C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json;print(json.load(open('$j'))['sha256'])")
  got=$(sha256sum "${j%.json}.npy" | cut -d' ' -f1)
  n=$((n + 1)); [ "$want" = "$got" ] || { bad=$((bad + 1)); say "SHA MISMATCH $j"; }
done
[ "$n" -ge 23 ] && [ "$bad" = 0 ] || fail "pulled $n feature shards, $bad sha256 mismatches"
sha256sum feat_shard_*.npy > "$P/raw/FEAT_SHA256.txt"
say "pulled and verified $n feature shards"
rm -f "$BUSY"
echo "ok $n shards" > "$P/raw/CHAIN_P5.done"
