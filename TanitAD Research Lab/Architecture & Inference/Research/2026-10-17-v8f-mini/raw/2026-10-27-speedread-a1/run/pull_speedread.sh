#!/usr/bin/env bash
# PREREG_V8F_SPEEDREAD s2: pull box 2's TRAIN clip features (D:\tanitad\cache_full\train\clips) to F:, md5-verified against the train
# manifest, ONLY while box 2 is idle between chains (waits for DAG-OACC r2's chain end and for no BOX2_BUSY.txt; holds BOX2_BUSY.txt itself
# while it reads). All 4,572 manifest clips are pulled (the 4,512 the extract uses + the 60 its parity rule drops, 1.25 GB) so that the
# manifest-driven rule sees every file it lists. Disk reads only on box 2; nothing is written there.
# Markers in $W/pull.out: ZZSPEEDREAD-PULL-DONE <n verified> | ZZSPEEDREAD-PULL-STOP <reason>. On DONE it runs run_speedread.sh in the foreground.
set -u
B=/c/Users/Admin/box2_ship; L=$B/logs
HERE=/e/Projects/TanitAD-artifacts/speedread_run
W=/f/TanitAD-scratch/speedread; DST=$W/cache/train/clips
MAN=$L/v8f_lora_ar2/cache/train/manifest.jsonl
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=$W/pull.out
mkdir -p $DST
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $OUT; }
stop() { log "ZZSPEEDREAD-PULL-STOP $*"; exit 3; }
log "pull start; waiting for DAG-OACC r2's chain end and no busy marker"
for i in $(seq 1 720); do
  if grep -aq '^ZZV8FMINI-DONE\|^ZZV8FMINI-STOP\|^ZZDAGOACC-CO3-STOP\|^ZZDISK-FLOOR-STOP\|^ZZDAGOACC-CONTROL-FAIL' $L/chain_v8f_dagoacc_r2.out 2>/dev/null \
     && [ ! -f $B/BOX2_BUSY.txt ]; then break; fi
  sleep 60
done
grep -aq '^ZZV8FMINI-DONE\|^ZZV8FMINI-STOP\|^ZZDAGOACC-\|^ZZDISK-FLOOR-STOP' $L/chain_v8f_dagoacc_r2.out || stop "DAG-OACC r2 never ended (12 h)"
[ -f $B/BOX2_BUSY.txt ] && stop "busy marker still present: $(cat $B/BOX2_BUSY.txt)"
echo "speedread train-cache pull (session bfe78918, PREREG_V8F_SPEEDREAD s2) since $(date '+%m-%d %H:%M'): disk reads only. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
trap 'rm -f $B/BOX2_BUSY.txt' EXIT
cp -n $MAN $W/cache/train/manifest.jsonl
cmp -s $MAN $W/cache/train/manifest.jsonl || stop "the local manifest copy differs from $MAN"
free=$(df -m /f | tail -1 | awk '{print $4}'); [ "$free" -gt 110000 ] || stop "F: has $free MB free (< 110 GB)"
for pass in 1 2 3 4 5 6; do
  "$PY" -I - "$(cygpath -m $MAN)" "$(cygpath -m $DST)" "$(cygpath -m $W)" <<'EOF' || stop "verify step failed (pass $pass)"
import hashlib, json, os, sys
man, dst, w = sys.argv[1:4]
rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
assert len(rows) == 4572, len(rows)
vf = os.path.join(w, "verified.txt")
ok = set(open(vf).read().split()) if os.path.exists(vf) else set()
need, newok = [], []
for r in rows:
    c = r["clip"]; p = os.path.join(dst, c + ".pt")
    if c in ok and os.path.getsize(p) == r["bytes"]:
        continue
    if os.path.exists(p) and os.path.getsize(p) == r["bytes"]:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() == r["md5"]:
            newok.append(c); continue
    need.append(c)
with open(vf, "a") as f:
    for c in newok:
        f.write(c + "\n")
with open(os.path.join(w, "need.txt"), "w") as f:
    for c in need:
        f.write("-get /D:/tanitad/cache_full/train/clips/%s.pt %s/%s.pt\n" % (c, dst, c))
print("verified %d, need %d" % (len(ok) + len(newok), len(need)))
EOF
  n=$(wc -l < $W/need.txt)
  log "pass $pass: $n clips to fetch"
  [ "$n" -eq 0 ] && break
  timeout 21600 sftp -q -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -b $W/need.txt tanitad-box2 >> $W/sftp_pass$pass.log 2>&1
  log "pass $pass sftp rc=$?"
done
[ "$(wc -l < $W/need.txt)" -eq 0 ] || stop "$(wc -l < $W/need.txt) clips still unverified after 6 passes: no partial fit"
nv=$(sort -u $W/verified.txt | wc -l); [ "$nv" -eq 4572 ] || stop "verified list has $nv entries, not 4572"
rm -f $B/BOX2_BUSY.txt; trap - EXIT
log "ZZSPEEDREAD-PULL-DONE $nv clips md5-verified against the manifest; box 2 released"
log "run_speedread.sh starting in the foreground (log $W/run.out)"
bash $HERE/run_speedread.sh >> $W/run.out 2>&1
log "run_speedread.sh rc=$?"
