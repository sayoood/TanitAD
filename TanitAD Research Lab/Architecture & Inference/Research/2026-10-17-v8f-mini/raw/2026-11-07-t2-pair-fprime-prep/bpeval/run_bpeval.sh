#!/usr/bin/env bash
# run_bpeval.sh -- F' for PREREG_V8F_LORA Amendment 4 item 3 on box 2. NOT RUN by its author: the Master Mind launches it once box 2 is free.
# Launch (detached; ~25-40 min, ESTIMATED from banked eval times of 2m10s-2m24s per eval + ship/hash/pull):
#   setsid nohup bash run_bpeval.sh < /dev/null > bpeval.out 2>&1 &        (or a Bash run_in_background job)   progress: grep ZZ bpeval.out
# Needs on the dev box: E:/Projects/TanitAD-artifacts/lorapod_eval/bprime/M2b_s{0,1}/{ckpt_last.pt,argv.json,summary.json},
#   lorapod_eval/eval/ (pod eval cache: clips/, manifest.jsonl) + lorapod_eval/cache_pod_eval.sha256, optional lorapod_eval/native_tracks.pt (the pod's copy).
# Order: local checks -> tip -> claim BOX2_BUSY.txt atomically (refuse if present) -> box-2 preconditions (idle, C: free >= 8 GiB, banked inputs on D: read-only)
#   -> ship to box 2 C: only (D: is reserved for T2 stages and is only READ) -> verify every shipped file by sha256 -> detached jobs: cache hash, then the 3 evals
#   -> pull to lorapod_eval/out -> compute_fprime.py -> move the busy marker aside. Every abort after the claim moves the marker aside; nothing on box 2 is deleted.
set -u
export PATH="/usr/bin:/bin:$PATH"
DRY=${BPEVAL_DRY:-0}                       # 1 = no ssh at all: a fake box-2 root, stub evals (copies of the banked files); never touches the real busy marker
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=${BPEVAL_ART:-E:/Projects/TanitAD-artifacts}
LE=$ART/lorapod_eval
CK=${BPEVAL_CKPT_ROOT:-$LE/bprime}
CACHE=${BPEVAL_CACHE:-$LE/eval}
SHAF=${BPEVAL_CACHE_SHA:-$LE/cache_pod_eval.sha256}
PODTRK=${BPEVAL_PODTRK:-$LE/native_tracks.pt}
OUT=${BPEVAL_OUT:-$LE/out}
WORK=${BPEVAL_WORK:-$LE/work}
BUSY=${BPEVAL_BUSY:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}
ASIDE=$(dirname "$BUSY")/busy_aside
FAKE=${BPEVAL_FAKE:-}
MIRROR=C:/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RR=C:/lorapod_eval                          # box 2 root, C: only
FLOOR_GIB=8                                 # ship ~3.2 GiB (2.8 pod cache + 2 ckpt + code) + ~4.8 GiB headroom (outputs, pagefile, temp)
BANKED_CKPT_SHA=601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8   # raw/full/MANIFEST.md: M2b_s0 ckpt_last.pt (D:/tanitad/runs/v8f_mini_full/M2b_s0)
NOISE='post-quantum|store now|upgraded|pq.html'
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
mkdir -p "$WORK" "$OUT" 2>/dev/null
say() { echo "[$(date +%H:%M:%S)] $*"; }
claimed=0
die() { say "ZZ BPEVAL $1 ZZ ${2:-}"; [ $claimed = 1 ] && aside "stopped_$1"; exit "${3:-4}"; }
aside() { mkdir -p "$ASIDE"; mv "$BUSY" "$ASIDE/BOX2_BUSY.bpeval.$1.$(date +%Y%m%d_%H%M%S).txt"; claimed=0; }

# ---- remote primitives (real = ssh/sftp; DRY = a plain directory tree standing in for box 2). Remote paths are Windows-style with a drive letter.
fk() { echo "$FAKE/$(echo "$1" | sed 's#^/##; s#:##')"; }
rsh()   { if [ $DRY = 1 ]; then return 0; fi; $SSH "powershell -NoProfile -Command \"$1\"" 2>&1 | grep -vE "$NOISE" | tr -d '\r'; }
rfree() { if [ $DRY = 1 ]; then echo "${BPEVAL_FAKE_FREE:-100}"; else rsh "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr ',' '.' | grep -E '^[0-9.]+$' | head -1; fi; }
rexists() { if [ $DRY = 1 ]; then [ -e "$(fk "$1")" ] && echo True || echo False; else rsh "Test-Path -LiteralPath '$1'" | head -1; fi; }
rmk()   { if [ $DRY = 1 ]; then mkdir -p "$(fk "$1")"; else rsh "New-Item -ItemType Directory -Force -Path '$1' | Out-Null"; fi; }
rput()  { if [ $DRY = 1 ]; then mkdir -p "$(dirname "$(fk "$2")")"; cp -r "$1" "$(fk "$2")"; else printf 'put -r "%s" "/%s"\n' "$1" "$2" | sftp -b - tanitad-box2 2>&1 | grep -vE "$NOISE" | tail -2; fi; }
rputdir() { if [ $DRY = 1 ]; then cp -r "$1" "$(fk "$2")"/; else printf 'put -r "%s" "/%s"
' "$1" "$2" | sftp -b - tanitad-box2 2>&1 | grep -vE "$NOISE" | tail -2; fi; }
rget()  { if [ $DRY = 1 ]; then mkdir -p "$2"; cp -r "$(fk "$1")" "$2"/; else printf 'get -r "/%s" "%s"\n' "$1" "$2" | sftp -b - tanitad-box2 2>&1 | grep -vE "$NOISE" | tail -2; fi; }
rhash() { if [ $DRY = 1 ]; then sha256sum "$(fk "$1")" | cut -c1-64; else $SSH "certutil -hashfile $(echo "$1" | sed 's#/#\\#g') SHA256" 2>&1 | grep -vE "$NOISE" | tr -d '\r ' | sed -n 2p | tr 'A-F' 'a-f'; fi; }
rboot() { if [ $DRY = 1 ]; then echo "2026-10-17 18:43:26"; else rsh "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')" | grep -E '^[0-9]{4}-' | head -1; fi; }
rlaunch() { # rlaunch <job.cmd on box 2> <log on box 2>
  if [ $DRY = 1 ]; then local jb; jb=$(fk "$1"); dry_job "$jb" "$(fk "$2")"; return 0; fi
  for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job $(echo "$1" | sed 's#/#\\#g') -Log $(echo "$2" | sed 's#/#\\#g')" 2>&1 | grep -vE "$NOISE" | tr -d '\r') && { say "launched: $o"; return 0; }; sleep 20; done; return 1; }
rwait() {   # rwait <log> <polls of 30 s> : ZZJOB-END rc, a reboot (crash) or unreachable stop it. Script polling, not model turns.
  local miss=0 lg; lg=$(echo "$1" | sed 's#/#\\#g')
  for i in $(seq 1 "$2"); do
    [ $DRY = 1 ] || sleep 30
    if [ $DRY = 1 ]; then b=$BOOT0; else b=$(rboot); fi
    [ -z "$b" ] && { miss=$((miss+1)); [ $miss -ge 24 ] && return 3; continue; }; miss=0
    [ "$b" != "$BOOT0" ] && return 2
    if [ $DRY = 1 ]; then t=$(tr '\r' '\n' < "$(fk "$1")" 2>/dev/null); else t=$($SSH "type $lg" 2>/dev/null | tr '\r' '\n'); fi
    echo "$t" | grep -aq ZZJOB-END && { WAITRC=$(echo "$t" | grep -a ZZJOB-END | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p'); return 0; }
  done; return 4; }
dry_job() { # the stand-in for a box-2 job: the cache hash job and the evals (stub: copies of the banked files, so compute_fprime's whole path runs)
  local jb=$1 lg=$2 o
  case "$jb" in
    *bp_hash*) (cd "$FAKE/C/lorapod_eval/cache_pod/eval" && find . -type f | sed 's#^\./##' | sort | while read -r f; do echo "$f $(sha256sum "$f" | cut -c1-64)"; done) > "$FAKE/C/lorapod_eval/logs/cache_hash.txt";;
    *bp_job*) o="$FAKE/C/lorapod_eval/out"; mkdir -p "$o"; for p in "M2b_s0-idcheck:0" "M2bp_s0:0" "M2bp_s1:1"; do t=${p%%:*}; s=${p##*:}; cp "$HERE/banked/eval_M2b_s$s.json" "$o/eval_$t.json"; cp "$HERE/banked/eval_M2b_s$s.npz" "$o/eval_$t.npz"; echo "stub eval $t" > "$o/eval_$t.log"; done;;
  esac; mkdir -p "$(dirname "$lg")"; echo "===== ZZJOB-END dry rc=0" >> "$lg"; }

# ================================================================ 0. local inputs
say "BPEVAL start (DRY=$DRY)"
for s in 0 1; do for f in ckpt_last.pt argv.json summary.json; do [ -s "$CK/M2b_s$s/$f" ] || die LOCAL-INPUT-MISSING "$CK/M2b_s$s/$f" 2; done
  grep -q '"M2b"' "$CK/M2b_s$s/argv.json" && ! grep -q -- '--encoder' "$CK/M2b_s$s/argv.json" || die ARGV-NOT-PLAIN-M2B "$CK/M2b_s$s/argv.json" 2; done
[ "$(basename "$CACHE")" = eval ] && [ -d "$CACHE/clips" ] && [ -s "$CACHE/manifest.jsonl" ] && [ -s "$SHAF" ] || die POD-CACHE-MISSING "$CACHE (clips/, manifest.jsonl) or $SHAF" 2
NCLIP=$(ls "$CACHE/clips" | wc -l); NMAN=$(wc -l < "$CACHE/manifest.jsonl" | tr -d ' ')
[ "$NCLIP" = 141 ] && [ "$NMAN" = 141 ] || die POD-CACHE-COUNT "clips=$NCLIP manifest rows=$NMAN (want 141)" 2
[ -s "$HERE/banked/SHA256.txt" ] || die BANKED-REF-MISSING "" 2
# local sha of the pod cache against the pod's own list (tolerant to the list's layout: 'hash  [*]path', matched by the clip file name)
$PY -I - "$SHAF" "$CACHE" "$WORK/cache_local.sha" <<'EOF' || die POD-CACHE-SHA-MISMATCH "the local copy differs from cache_pod_eval.sha256" 2
import hashlib, os, re, sys
lst, root, outp = sys.argv[1:4]
want = {}
for l in open(lst, encoding="utf-8", errors="replace"):
    m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", l)
    if m: want[m.group(2).replace("\\", "/").split("eval/")[-1].lstrip("./")] = m.group(1).lower()
have = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(dp, f); h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
        have[os.path.relpath(p, root).replace("\\", "/")] = h.hexdigest()
bad = [k for k, v in want.items() if have.get(k) != v]
print("sha list entries", len(want), "local files", len(have), "bad/missing", len(bad), bad[:3])
if not want or bad: sys.exit(1)
open(outp, "w").write("".join("%s %s\n" % (k, have[k]) for k in sorted(have)))
EOF
# ================================================================ 1. the tip
if [ $DRY = 1 ]; then T=${BPEVAL_TIP:-$(git -C $MIRROR ls-remote origin $BR | cut -f1)}; else git -C $MIRROR fetch -q origin $BR; T=${BPEVAL_TIP:-$(git -C $MIRROR ls-remote origin $BR | cut -f1)}; fi
git -C $MIRROR cat-file -e "$T^{commit}" 2>/dev/null || die TIP-NOT-IN-MIRROR "$T" 2
say "tip $T"
# ================================================================ 2. claim the box atomically
mkdir -p "$(dirname "$BUSY")"
( set -C; printf 'owner=BPEVAL (F-prime, LoRA card Amendment 4)\nstart=%s\neta=0.5-0.7 h\n' "$(date '+%F %T')" > "$BUSY" ) 2>/dev/null || { say "ZZ BPEVAL-BUSY: $BUSY exists, STOP (read it, do not delete it) ZZ"; cat "$BUSY"; exit 3; }
claimed=1; say "box claimed"
# ================================================================ 3. box-2 preconditions
BOOT0=$(rboot); [ -n "$BOOT0" ] || die BOX2-UNREACHABLE
if [ $DRY != 1 ]; then npy=$(rsh "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count" | head -1); [ "$npy" = 0 ] || die BOX2-NOT-IDLE "python.exe processes on box 2: $npy"; fi
FR=$(rfree); say "free C: ${FR:-?} GiB (floor $FLOOR_GIB)"
awk -v d="${FR:-0}" -v n="$FLOOR_GIB" 'BEGIN{exit !(d>=n)}' || die DISK-LOW "C: free=$FR GiB < $FLOOR_GIB GiB"
[ "$(rexists "$RR")" = True ] && die REMOTE-ROOT-EXISTS "$RR exists (a leftover run? read it, move it aside, relaunch)"
if [ $DRY != 1 ]; then
  for p in D:/tanitad/cache_full/eval/manifest.jsonl D:/tanitad/cache_full/native_tracks.pt D:/tanitad/data/v9/v9_labels_eval139.npz D:/tanitad/data/v9/lead_eval139.npz \
           D:/tanitad/data/v9/refcv6_speed_max_v8_eval.jsonl D:/tanitad/venv/Scripts/python.exe D:/tanitad/runs/v8f_mini_full/M2b_s0/ckpt_last.pt; do
    [ "$(rexists "$p")" = True ] || die BOX2-INPUT-MISSING "$p"; done
  g=$(rhash D:/tanitad/runs/v8f_mini_full/M2b_s0/ckpt_last.pt); [ "$g" = "$BANKED_CKPT_SHA" ] || die BANKED-CKPT-SHA "box 2 M2b_s0 ckpt sha=$g want=$BANKED_CKPT_SHA"
fi
# ================================================================ 4. build + ship (C: only)
rm -f "$WORK/bp_code.tar"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o "$WORK/bp_code.tar" "$T" -- stack/tanitad stack/scripts/v8f_mini_eval.py taniteval/taniteval \
  "$PK/raw/eval_grid_windows.json" "$PK/raw/eval141_kin.npz" "$PK/raw/join_eval141.npz" || die ARCHIVE
for f in env_bp bp_job bp_evals bp_hash; do sed 's/$/\r/' "$HERE/box2/$f.cmd" > "$WORK/$f.cmd"; done      # box 2 .cmd files get CRLF here, the repo copies stay LF
for d in "" /code /ckpt/M2bp_s0 /ckpt/M2bp_s1 /out /logs /tmp /jobs; do rmk "$RR$d"; done
declare -A SHIP=( ["$WORK/bp_code.tar"]="$RR/bp_code.tar" ["$WORK/env_bp.cmd"]="$RR/env_bp.cmd" ["$WORK/bp_job.cmd"]="$RR/jobs/bp_job.cmd" ["$WORK/bp_evals.cmd"]="$RR/bp_evals.cmd" ["$WORK/bp_hash.cmd"]="$RR/jobs/bp_hash.cmd"
  ["$CK/M2b_s0/ckpt_last.pt"]="$RR/ckpt/M2bp_s0/ckpt_last.pt" ["$CK/M2b_s1/ckpt_last.pt"]="$RR/ckpt/M2bp_s1/ckpt_last.pt" )
[ -s "$PODTRK" ] && SHIP["$PODTRK"]="$RR/cache_pod/native_tracks.pt"
rmk "$RR/cache_pod"
for src in "${!SHIP[@]}"; do rput "$src" "${SHIP[$src]}"; done
say "shipped files; shipping the pod eval cache (2.8 GB)"
rputdir "$CACHE" "$RR/cache_pod"          # the pod's eval split directory goes in under its own name: cache_pod/eval/{clips,manifest.jsonl}
for src in "${!SHIP[@]}"; do w=$(sha256sum "$src" | cut -c1-64); g=$(rhash "${SHIP[$src]}"); [ "$w" = "$g" ] || die SHIP-SHA-MISMATCH "${SHIP[$src]} want=$w got=$g"; done
say "small files + checkpoints sha256 verified"
if [ $DRY = 1 ]; then mkdir -p "$(fk $RR/code)"; (cd "$(fk $RR/code)" && tar --force-local -xf "$WORK/bp_code.tar"); else $SSH "tar -xf C:\lorapod_eval\bp_code.tar -C C:\lorapod_eval\code" 2>&1 | grep -vE "$NOISE"; fi
for f in stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/ci.py; do
  w=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64); g=$(rhash "$RR/code/$f"); [ "$w" = "$g" ] || die CODE-SHA-MISMATCH "$f want=$w got=$g"; done
say "code tree = tip $T (4 key files sha-checked)"
# the pod cache, every file: hashed ON box 2 by a detached job, compared with the local manifest
rlaunch "$RR/jobs/bp_hash.cmd" "$RR/logs/bp_hash.log" || die LAUNCH-HASH
rwait "$RR/logs/bp_hash.log" 60; wrc=$?; [ $wrc = 4 ] && { say "ZZ BPEVAL HASH-WAIT-TIMEOUT: the box stays CLAIMED, a job may still run ZZ"; exit 9; }; [ $wrc = 0 ] || die HASH-WAIT "rc=$wrc (2 = box rebooted, 3 = unreachable)"
if [ $DRY = 1 ]; then cp "$(fk $RR/logs/cache_hash.txt)" "$WORK/cache_remote.sha"; else printf 'get "/C:/lorapod_eval/logs/cache_hash.txt" "%s"\n' "$WORK/cache_remote.sha" | sftp -b - tanitad-box2 2>&1 | grep -vE "$NOISE" | tail -1; fi
cmp -s <(sort "$WORK/cache_local.sha" | tr -d '\r') <(sort "$WORK/cache_remote.sha" | tr -d '\r') || die CACHE-SHA-MISMATCH "box-2 copy of the pod cache differs from the local copy (diff $WORK/cache_local.sha $WORK/cache_remote.sha)"
say "pod eval cache: $(wc -l < "$WORK/cache_remote.sha" | tr -d ' ') files sha256-equal on both sides"
# tracks: the pod's copy if supplied, else box 2's (the identity eval and the B' evals then read the SAME tracks file)
if [ -s "$PODTRK" ]; then
  [ $DRY = 1 ] || { g=$(rhash D:/tanitad/cache_full/native_tracks.pt); w=$(sha256sum "$PODTRK" | cut -c1-64)
    [ "$g" = "$w" ] && say "TRACKS-SAME pod copy == box-2 cache_full copy" || die TRACKS-DIFFER "pod native_tracks $w vs box-2 $g: the identity and B' evals would differ in an input" ; }
else say "TRACKS-POD-NOT-SUPPLIED: box 2's native_tracks.pt is read for all three evals (the pod's copy is UNVERIFIED equal)"; fi
# ================================================================ 5. the three evals
rlaunch "$RR/jobs/bp_job.cmd" "$RR/logs/bp_job.log" || die LAUNCH-EVALS
say "evals launched (identity, B' s0, B' s1)"
rwait "$RR/logs/bp_job.log" 120; wrc=$?; [ $wrc = 4 ] && { say "ZZ BPEVAL EVAL-WAIT-TIMEOUT: the box stays CLAIMED, the job may still run ZZ"; exit 9; }; [ $wrc = 0 ] || die EVAL-WAIT "rc=$wrc (2 = box rebooted, 3 = unreachable)" 4
[ "${WAITRC:-x}" = 0 ] || { rget "$RR/out" "$WORK/failed_run_$(date +%H%M%S)"; die EVAL-RC "job rc=${WAITRC:-?}; logs pulled to $WORK/failed_run_*" 5; }
# ================================================================ 6. pull, verify, compute, release
ls "$OUT"/eval_* >/dev/null 2>&1 && die OUT-NOT-EMPTY "$OUT already holds eval_* (move it aside)" 3
PULL="$WORK/pull_$(date +%H%M%S)"; rget "$RR/out" "$PULL"; [ -d "$PULL/out" ] || die PULL-FAILED "$PULL/out" 6; mv "$PULL"/out/* "$OUT"/
for t in M2b_s0-idcheck M2bp_s0 M2bp_s1; do for e in json npz log; do [ -s "$OUT/eval_$t.$e" ] || die PULL-INCOMPLETE "$OUT/eval_$t.$e" 6; done
  for e in json npz; do w=$(sha256sum "$OUT/eval_$t.$e" | cut -c1-64); g=$(rhash "$RR/out/eval_$t.$e"); [ "$w" = "$g" ] || die PULL-SHA-MISMATCH "$t.$e" 6; done; done
say "pulled and sha256-verified: 3 evals (json, npz, log)"
$PY -I "$HERE/compute_fprime.py" --out-dir "$OUT" > "$OUT/FPRIME.txt" 2>&1; frc=$?
cat "$OUT/FPRIME.txt"
aside "pulled"
say "ZZ BPEVAL DONE ZZ compute_fprime rc=$frc (0 = F' written to $OUT/FPRIME.json; 3 = the identity eval missed or an input is wrong: F' not computed)"
exit $frc
