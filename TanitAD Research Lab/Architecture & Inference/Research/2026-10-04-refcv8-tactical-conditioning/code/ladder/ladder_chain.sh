#!/usr/bin/env bash
# SPEC_WPB_LADDER (registered e1b6aff7...) + A1 (cbd64fa2...) -- the Thor chain. ONE GPU job at a time under the
# programme lock (`flock $LOCK`, the same lock every stream uses; no second lock, no queue jumping: flock is not FIFO
# and a waiter here simply waits). Detached; leaves no python alive; every chunk <= 40 min.
#
#   S0   30 steps each of V0 / V-R8 / V-MAP4 at --log-every 1 (nothing kept) -> s/step -> N (sec. 4.1), and the I-0
#        step-1 identity + the V0/V-R8 fed-speed CRC rows (MM assertion 2)
#   ARMS in ORDER (ladder_arms.ORDER); after V-R8: k from its first 20 % of gs rows (sec. 4.2) -> the V-TACk* argv
#        (every earlier argv is re-derived and must hash IDENTICALLY)
#   each arm: resumable chunks, killed only RIGHT AFTER a ckpt write once the next save would pass the 40-min cap
#   EVAL per arm (sampler seeds 0,1; rows per SPEC sec. 2/3), then the checkpoint is stripped to a model-only
#        `model_final.pt` (+md5) for the dev-box puller, and ckpt.pt / ckpt_5000.pt are DELETED (Thor disk)
#   I0 + SCORE at the end (CPU).
# Disk: no job starts below FREE_MIN_GB free (R1 stops below 20 GB). Markers: ZZ<stage>ZZ in $W/chain.log.
# Usage (Thor): nohup bash ladder_chain.sh > $W/chain.out 2>&1 < /dev/null &
set -u
W=${W:-/home/nvidia/refcv8_ladder/W}
TREE=${TREE:-/home/nvidia/refcv8_ladder/tree}
CODE=${CODE:-/home/nvidia/refcv8_ladder/code}
PY=${PY:-/home/nvidia/venvs/tanitad-train/bin/python}
LOCK=${LOCK:-/home/nvidia/refcv7_post/thor_gpu.lock}
CAP_S=${CAP_S:-2400}
FREE_MIN_GB=${FREE_MIN_GB:-22}
DRY=${DRY:-0}
V9TRAIN=${V9TRAIN:-/home/nvidia/refcv8_v9labels/release/v9_labels_train.npz}
V9TRAIN_MD5=${V9TRAIN_MD5:-f63ece410b725febb8a5242cf2b01d3c}   # refcv8_train.V9_RELEASE_MD5["train"]
PSEP=${PSEP:-:}               # ";" only for a dev-box DRY run (Windows python)
export PYTHONPATH="$TREE/stack$PSEP$TREE/taniteval" REFCV6_REPO="$TREE" REFCV6_KIT=/home/nvidia OMP_NUM_THREADS=6
mkdir -p "$W"
cd "$W" || exit 3            # never run from a directory that could shadow the `taniteval` namespace package
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$W/chain.log"; }
die() { log "ZZSTOPZZ $*"; exit 3; }

disk_wait() {
  local f
  while :; do
    f=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    [ -n "$f" ] && [ "$f" -ge "$FREE_MIN_GB" ] && return 0
    log "disk ${f:-?} GB < $FREE_MIN_GB GB -- waiting (R1 stops below 20)"
    sleep 300
  done
}

# ---------------------------------------------------------------------------------------------------- preflight
preflight() {
  [ -d "$TREE/stack" ] || die "no launch tree at $TREE"
  grep -q "def speed_only_fwd" "$TREE/stack/tanitad/train/refcv8_train.py" || die "tree lacks L3 (speed_only_fwd)"
  grep -q "r8-critic-drivable" "$TREE/stack/scripts/refc_v3_train.py" || die "tree lacks L2 (drivable critic)"
  local got
  got=$("$PY" -c 'import sys, tanitad, taniteval.ci; from pathlib import Path as P
t, f = P(sys.argv[1]).resolve(), P(tanitad.__file__).resolve()
print(f.as_posix()); sys.exit(0 if t in f.parents else 7)' "$TREE" 2>&1) || die "import probe (tanitad not from $TREE): $got"
  for f in ladder_arms.py ladder_eval.py ladder_i0.py ladder_score.py; do
    [ -f "$CODE/$f" ] || die "missing $CODE/$f"
  done
  log "preflight OK tree=$TREE tanitad=$got"
}

argv_of() {   # arm -> NUL-separated argv on stdout
  "$PY" -c 'import json,sys; sys.stdout.write("\0".join(json.load(open(sys.argv[1]))["argv"]))' "$W/argv/$1.json"
}

build_argv() {   # n [k]
  local k=${2:-}
  "$PY" "$CODE/ladder_arms.py" --tree "$TREE" --out "$W" --n "$1" --root "$W/arms" ${k:+--k "$k"} \
      >> "$W/chain.log" 2>&1 || die "ladder_arms AUDIT FAIL (n=$1 k=$k)"
}

# run one trainer chunk under the lock; kill RIGHT AFTER a checkpoint once the next save would pass the cap
train_chunk() {   # out_dir log final_step argv...
  local out=$1 lg=$2 fin=$3; shift 3
  disk_wait
  if [ "$DRY" = 1 ]; then log "DRY train $out"; mkdir -p "$out"; echo '{"done": true}' > "$out/summary.json"; return 0; fi
  flock "$LOCK" timeout "$CAP_S" "$PY" "$TREE/stack/scripts/refc_v3_train.py" "$@" >> "$lg" 2>&1 200>&- &
  local fl=$! t0 now py="" last_e=-1 prev_e=0 n0 n1 e iv
  t0=$(date +%s)
  n0=$(grep -c "ckpt step" "$lg" 2>/dev/null); n0=${n0:-0}
  while kill -0 "$fl" 2>/dev/null; do
    sleep 10
    [ -z "$py" ] && py=$(pgrep -P "$fl" -x timeout 2>/dev/null | head -1)
    n1=$(grep -c "ckpt step" "$lg" 2>/dev/null); n1=${n1:-0}
    if [ "$n1" -gt "$n0" ]; then
      now=$(date +%s); e=$((now - t0)); iv=$((e - prev_e)); prev_e=$e; n0=$n1
      local cs; cs=$(grep -o "ckpt step [0-9]*" "$lg" | tail -1 | tr -dc 0-9)
      if [ $((e + iv + iv / 10)) -gt $((CAP_S - 60)) ] && [ "$cs" != "$fin" ] && [ ! -f "$out/summary.json" ]; then
        local tp; tp=$(pgrep -P "$py" 2>/dev/null | head -1)
        log "chunk end right after ckpt step $cs (elapsed ${e}s, interval ${iv}s): SIGTERM python=$tp"
        [ -n "$tp" ] && kill -TERM "$tp"
      fi
    fi
  done
  wait "$fl"
  return $?
}

train_arm() {   # arm
  local arm=$1 out="$W/arms/$1/run" lg="$W/arms/$1/train.log" i=0 argv=()
  mkdir -p "$out"
  [ -f "$out/summary.json" ] && { log "$arm already done"; return 0; }
  mapfile -d '' argv < <(argv_of "$arm")
  [ "${#argv[@]}" -gt 20 ] || die "$arm argv empty"
  while [ ! -f "$out/summary.json" ]; do
    i=$((i + 1)); [ "$i" -gt 15 ] && die "$arm: 15 chunks without summary.json"
    log "ZZTRAIN-$arm-chunk$i-ZZ"
    train_chunk "$out" "$lg" "$N" "${argv[@]}"
    local rc=$?
    if [ ! -f "$out/summary.json" ] && [ ! -f "$out/ckpt.pt" ] && [ "$rc" != 0 ]; then
      die "$arm chunk $i rc=$rc with no checkpoint (see $lg)"
    fi
  done
  grep -q '"done": true' "$out/summary.json" || die "$arm summary.json carries no done marker"
  log "ZZDONE-$arm-ZZ"
}

rows_for() {   # arm -> comma rows (SPEC sec. 2/3: eval-only rows on every refcv8 arm; VMAX rows on V-R8 for L4)
  case "$1" in
    V0|V0r) echo "base";;
    V-R8) echo "base,legal,rc_off,rc_shuf,vmax_off,vmax_shuf";;
    *) echo "base,legal,rc_off,rc_shuf";;
  esac
}

eval_arm() {   # arm n
  local arm=$1 n=$2 out="$W/arms/$1/run" ed="$W/eval/$1" r
  mkdir -p "$ed"
  for r in $(rows_for "$arm" | tr , ' '); do
    [ -f "$ed/${r}_s1.json" ] && [ -f "$ed/${r}_s0.json" ] && continue
    disk_wait
    log "ZZEVAL-$arm-$r-ZZ"
    if [ "$DRY" = 1 ]; then log "DRY eval $arm $r"; continue; fi
    flock "$LOCK" timeout "$CAP_S" "$PY" "$CODE/ladder_eval.py" --run "$out" --expect-step "$n" --rows "$r" \
        --seeds 0,1 --out-dir "$ed" --device cuda >> "$W/eval/$arm.log" 2>&1 200>&-
    [ -f "$ed/${r}_s0.json" ] && [ -f "$ed/${r}_s1.json" ] || die "eval $arm/$r wrote no artifact (see $W/eval/$arm.log)"
  done
  # strip -> model-only for the dev-box puller; the optimiser-carrying ckpt.pt and the milestone are DELETED (disk)
  if [ "$DRY" != 1 ] && [ -f "$out/ckpt.pt" ]; then
    "$PY" - "$out" <<'EOF' || die "strip failed for $arm"
import hashlib, sys, torch
from pathlib import Path
o = Path(sys.argv[1])
ck = torch.load(o / "ckpt.pt", map_location="cpu", weights_only=False)
torch.save({"model": ck["model"], "step": ck["step"]}, o / "model_final.pt")
h = hashlib.md5((o / "model_final.pt").read_bytes()).hexdigest()
(o / "model_final.pt.md5").write_text(h + "\n")
print("stripped", ck["step"], h)
EOF
    rm -f "$out/ckpt.pt" "$out/ckpt_5000.pt"
    touch "$out/PULL_READY"
  fi
  log "ZZEVALDONE-$arm-ZZ"
}

# ---------------------------------------------------------------------------------------------------- S0
s0() {
  build_argv 2000
  local arm argv=()
  for arm in V0 V-R8 V-MAP4; do
    local out="$W/s0/$arm/run"
    [ -f "$out/summary.json" ] && continue
    mkdir -p "$out"
    mapfile -d '' argv < <(argv_of "$arm")
    argv=("${argv[@]}")
    # S0 tokens (stated): 30 steps, every step logged, no eval, nothing kept; the out dir is the S0 one
    local a2=() i=0
    while [ $i -lt ${#argv[@]} ]; do
      case "${argv[$i]}" in
        --steps|--log-every|--eval-every|--save-every|--out) i=$((i + 2));;
        *) a2+=("${argv[$i]}"); i=$((i + 1));;
      esac
    done
    a2+=(--steps 30 --log-every 1 --eval-every 0 --save-every 100000 --out "$out")
    log "ZZS0-$arm-ZZ"
    train_chunk "$out" "$W/s0/$arm/train.log" 30 "${a2[@]}"
    rm -f "$out/ckpt.pt"
    [ -f "$out/summary.json" ] || die "S0 $arm did not finish (see $W/s0/$arm/train.log)"
  done
  [ "$DRY" = 1 ] && { echo 3000 > "$W/N.txt"; return 0; }
  "$PY" - "$W" <<'EOF' || die "N rule failed"
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
rows = [json.loads(l) for l in (w / "s0/V-R8/run/metrics.jsonl").read_text().splitlines() if l.startswith("{")]
e = {int(r["step"]): float(r["elapsed_s"]) for r in rows if "elapsed_s" in r}
s = (e[30] - e[5]) / 25.0
n = int((3 * 3600) // s) // 250 * 250
n = min(n, 6000)
rec = {"s_step_VR8": s, "rule": "largest multiple of 250 with N*s <= 3 h, cap 6000, floor 2000", "N": n}
if n < 2000:
    rec["STOP"] = "N < 2000: the rig is too slow -- the rig change goes to the MM (SPEC sec. 4.1)"
(w / "S0.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
sys.exit(0 if n >= 2000 else 5)
EOF
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["N"])' "$W/S0.json" > "$W/N.txt"
}

# ---------------------------------------------------------------------------------------------------- main
preflight
log "ZZSTARTZZ W=$W"
[ -f "$W/N.txt" ] || s0
N=$(cat "$W/N.txt"); [ -n "$N" ] || die "no N"
log "N=$N"
build_argv "$N"
cp "$W/argv/AUDIT.json" "$W/argv/AUDIT_n.json"
argv_sha() { "$PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]+"/"+a+".json"))["argv_sha256"][:16] for a in ("V0","V-R8")))' "$W/argv"; }
sha_before=$(argv_sha)
for arm in V0 V-R8; do train_arm "$arm"; eval_arm "$arm" "$N"; done
if [ ! -f "$W/SIZE_K.json" ]; then
  [ "$DRY" = 1 ] && echo '{"k": 51.0}' > "$W/SIZE_K.json" || "$PY" "$CODE/ladder_score.py" size-k --w "$W" >> "$W/chain.log" 2>&1
fi
K=$("$PY" -c 'import json,sys; k=json.load(open(sys.argv[1]))["k"]; print("" if k is None else ("%g" % k))' "$W/SIZE_K.json")
[ -n "$K" ] || log "k: NO ROOT above 1 -- the L2 arms are not built (named in the RESULT)"
build_argv "$N" "$K"
sha_after=$(argv_sha)
[ "$sha_before" = "$sha_after" ] || die "re-deriving the argv with k changed V0/V-R8 ($sha_before vs $sha_after)"
for arm in V-R8d V0r V-R8r V-TACk V-TACk-roll V-TACkr V-MAP4 V-MAP4-roll V-MAP4r V-VSHUF V-R8-E8 V-R8-E8-roll \
           V-R8-DRV V-R8-DRV-roll; do
  [ -f "$W/argv/$arm.json" ] || { log "skip $arm (no argv)"; continue; }
  train_arm "$arm"; eval_arm "$arm" "$N"
done
"$PY" "$CODE/ladder_i0.py" --w "$W" --pytest-rc "$W/I0b_pytest_rc.txt" >> "$W/chain.log" 2>&1
"$PY" "$CODE/ladder_score.py" score --w "$W" --v9-train "$V9TRAIN" --v9-train-md5 "$V9TRAIN_MD5" >> "$W/chain.log" 2>&1
[ -f "$W/LADDER_SCORE.json" ] || die "the scorer wrote no LADDER_SCORE.json"
log "ZZALLDONEZZ"
