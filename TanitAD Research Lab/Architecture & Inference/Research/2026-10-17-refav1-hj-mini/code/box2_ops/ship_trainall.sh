#!/usr/bin/env bash
# PREREG_RA1_FULL: ship the full-corpus build INPUTS to box 2 (disk only -- no job runs there while this runs).
#   control clips (2) + clips_rest (3,972): mp4 + timestamps.parquet;  egomotion_all.zip;  intrinsics csv;  lists + HF sha ledger.
# Resumable (ledger of shipped files). Each scp batch is retried; at the end every file's SIZE is compared to the dev box's,
# and the zip/csv/ledger by sha256. mp4 content is sha-checked again by ra1_full_box2.py before each chunk builds.
S=/c/Users/Admin/tanitad-data/refav1-train4572-256_stage
B=/c/Users/Admin/box2_ship; L=$B/shipped_trainall.txt; touch $L
R=D:/tanitad/data/trainall_stage
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
q() { grep -v -i -E "post-quantum|store now|upgraded"; }
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/tanitad/data/trainall_stage/r0/camera_front_wide, D:/tanitad/data/trainall_stage/labels/egomotion, D:/tanitad/data/trainall_stage/calibration | Out-Null; 'dirs ok'\"" 2>&1 | q
put() {  # put <remote dir> <files...>  -- 4 tries
  local d=$1; shift
  for t in 1 2 3 4; do $SCP "$@" "tanitad-box2:$d/" 2>/dev/null && return 0; sleep 15; done; return 1
}
t0=$(date +%s)
for f in calibration/physicalai_front_wide_intrinsics.csv labels/egomotion/egomotion_all.zip; do
  grep -qxF "$f" $L && continue
  put "$R/$(dirname $f)" "$S/$f" || { echo "ZZSHIP-FAIL $f"; exit 1; }; echo "$f" >> $L
  echo "$(date +%H:%M:%S) shipped $f ($(( $(date +%s) - t0 ))s)"
done
for f in clips_rest.txt clips_all.txt ctl_expected.json; do put "$R" "$S/$f" || { echo "ZZSHIP-FAIL $f"; exit 1; }; done
put "$R" "$S/_hf/camera/camera_sha256.json" || { echo "ZZSHIP-FAIL ledger"; exit 1; }
mapfile -t CL < <(python -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))))" "$(cygpath -m $S/ctl_expected.json)" | tr -d '\r'; tr -d '\r' < $S/clips_rest.txt)
echo "clips to ship: ${#CL[@]}"
batch=(); n=0
flush() {
  [ ${#batch[@]} -eq 0 ] && return 0
  put "$R/r0/camera_front_wide" "${batch[@]}" || { echo "ZZSHIP-FAIL batch at $n"; exit 1; }
  for f in "${batch[@]}"; do basename "$f" >> $L; done; batch=()
}
for c in "${CL[@]}"; do
  [ -z "$c" ] && continue
  for e in mp4 timestamps.parquet; do
    grep -qxF "$c.$e" $L && continue
    batch+=("$S/r0/camera_front_wide/$c.$e")
  done
  n=$((n + 1))
  if [ ${#batch[@]} -ge 60 ]; then flush; el=$(( $(date +%s) - t0 )); echo "$(date +%H:%M:%S) $n/${#CL[@]} clips (${el}s)"; fi
done
flush
echo "$(date +%H:%M:%S) all shipped, checking sizes"
# the clip list is read INSIDE python: 3,974 ids as argv overflow the command line, and an empty local list makes every
# comparison below read 0 mismatches (MEASURED 2026-10-18: "Argument list too long", then "0 mismatches" against nothing)
python - "$(cygpath -m $S)" > $B/trainall_local_sizes.txt <<'EOF'
import json, os, sys
s = sys.argv[1]
cl = list(json.load(open(f"{s}/ctl_expected.json"))) + [l.strip() for l in open(f"{s}/clips_rest.txt") if l.strip()]
for c in cl:
    for e in ("mp4", "timestamps.parquet"):
        print(f"{c}.{e} {os.path.getsize(f'{s}/r0/camera_front_wide/{c}.{e}')}")
EOF
$SSH "powershell -NoProfile -Command \"Get-ChildItem D:/tanitad/data/trainall_stage/r0/camera_front_wide | ForEach-Object { \$_.Name + ' ' + \$_.Length }\"" 2>/dev/null | tr -d '\r' | grep -E ' [0-9]+$' | sort > $B/trainall_remote_sizes.txt
tr -d '\r' < $B/trainall_local_sizes.txt | sort > $B/trainall_local_sizes.sorted     # Windows python prints CRLF
[ "$(wc -l < $B/trainall_local_sizes.sorted)" = $((2 * ${#CL[@]})) ] || { echo "ZZSHIP-FAIL the local size list is incomplete"; exit 1; }
nd=$(comm -23 $B/trainall_local_sizes.sorted $B/trainall_remote_sizes.txt | wc -l)
echo "size mismatches or missing: $nd (local $(wc -l < $B/trainall_local_sizes.sorted), remote $(wc -l < $B/trainall_remote_sizes.txt))"
[ "$nd" = 0 ] || { comm -23 $B/trainall_local_sizes.sorted $B/trainall_remote_sizes.txt | head -5; echo "ZZSHIP-FAIL sizes"; exit 1; }
for f in labels/egomotion/egomotion_all.zip calibration/physicalai_front_wide_intrinsics.csv; do
  l=$(sha256sum "$S/$f" | cut -c1-64)
  r=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 D:/tanitad/data/trainall_stage/$f).Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
  [ "$l" = "$r" ] && echo "sha ok $f" || { echo "ZZSHIP-FAIL sha $f local=$l remote=$r"; exit 1; }
done
echo "ZZSHIP-DONE $(date +%H:%M:%S) in $(( $(date +%s) - t0 ))s"
