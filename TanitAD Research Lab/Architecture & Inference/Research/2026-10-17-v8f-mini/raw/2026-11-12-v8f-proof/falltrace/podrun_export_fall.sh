#!/usr/bin/env bash
# podrun_export_fall.sh -- pod side, CPU only: runs export_fall.py on the banked T1-kin rollouts (RR, RLc and the arms in ARMS, both model seeds). ARMS is a comma list, default VD5c,F2,F3,F4,E5c; G2,G3,G4,GEc,E3c are accepted.
# Root: /workspace/v8proof/falltrace (a new root; nothing is written elsewhere). Makes $R/pull (links to the run folders), $R/out (refused if it exists), the logs and the markers START_FALLTRACE.txt, DONE_FALLTRACE.txt, STOP_FALLTRACE.txt.
# DRY=1 only checks that every input folder and file exists and prints what is missing; it makes nothing. It deletes, moves and stops nothing.
# Last line: ZZ PODRUN-FALLTRACE DONE rc <n>   or   ZZ PODRUN-FALLTRACE STOP <why>   or (DRY) ZZ PODRUN-FALLTRACE DRY missing <n>.
set -u
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; CODE=$P/code
R=$V/falltrace; K=$R/kit; OUT=$R/out; LINKS=$R/pull
PY=/workspace/venv/bin/python
ARMS=${ARMS:-VD5c,F2,F3,F4,E5c}
JOIN=$P/data/join_eval141.npz
BANKED=$V/cropscale/read/CROPSCALE10K_RESULT.json
B=RQ50KF5P30Hb20
run_of() {
  case "$1" in
    RR) echo "$B";; RLc) echo RLcKF5P30Hb20;; VD5c) echo "${B}vVD5c";; F2|F3|F4) echo "${B}f$1";; E5c) echo "${B}eE5c";;
    G2|G3|G4) echo "${B}g$1";; GEc) echo "${B}gGEc";; E3c) echo "${B}gE3c";; *) echo "";;
  esac
}
dir_of() { for d in "$W/t1/$1" "$RD/pull/$1"; do if [ -d "$d" ]; then echo "$d"; return 0; fi; done; return 1; }
stop() { echo "ZZ PODRUN-FALLTRACE STOP $*"; [ "${DRY:-0}" = 1 ] || { [ -d "$R" ] && echo "$*" > "$R/STOP_FALLTRACE.txt"; }; exit 3; }
FORMS="RR RLc $(echo "$ARMS" | tr ',' ' ')"
for f in $FORMS; do [ -n "$(run_of "$f")" ] || stop "UNKNOWN-ARM $f"; done
missing=0
miss() { echo "MISSING $1"; missing=$((missing + 1)); }
chk() { [ -e "$1" ] || miss "$1"; }
chk "$RD/read_longroll10k.py"; chk "$W/longroll/grid_p30.json"; chk "$JOIN"; chk "$CODE"; chk "$K/export_fall.py"
for f in $FORMS; do
  n=$(run_of "$f")
  for s in 0 1; do
    k=I10${n}_s$s
    d=$(dir_of "$k") || { miss "run folder $k (looked in $W/t1 and $RD/pull)"; continue; }
    chk "$d/dump_${k}_m10_causal_heads"
    if [ "$f" != RR ] && [ "$f" != RLc ]; then chk "$d/ticks/rule_log_$k.npz"; chk "$d/ticks/rule_log_mid_$k.npz"; fi
  done
done
if [ "${DRY:-0}" = 1 ]; then
  echo "DRY: no folder is made, nothing is run; arms $ARMS; output folder $OUT (refused if it exists)"
  [ -f "$BANKED" ] && echo "banked result file present" || echo "banked result file absent (only the contact values of the brief are checked)"
  echo "ZZ PODRUN-FALLTRACE DRY missing $missing"
  exit 0
fi
[ -e "$OUT" ] && stop "ALREADY-EXPORTED $OUT exists"
[ "$missing" = 0 ] || stop "MISSING-INPUT $missing (run with DRY=1 to list)"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "container free ${free_gb} GB"; [ "$free_gb" -ge 8 ] || stop "MEMORY free ${free_gb} GB"; fi
mkdir -p "$LINKS" || stop "CANNOT-MAKE $LINKS"
for f in $FORMS; do
  n=$(run_of "$f")
  for s in 0 1; do k=I10${n}_s$s; [ -e "$LINKS/$k" ] || ln -s "$(dir_of "$k")" "$LINKS/$k"; done
done
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ) arms $ARMS" > "$R/START_FALLTRACE.txt"
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$JOIN HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
( cd "$RD" && "$PY" "$K/export_fall.py" --selftest --code "$CODE" > "$R/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$R/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
BARG=""
[ -f "$BANKED" ] && BARG="--banked $BANKED"
( cd "$RD" && "$PY" "$K/export_fall.py" --pull "$LINKS" --grid-dir "$W/longroll" --join "$JOIN" --code "$CODE" --out "$OUT" --arms "$ARMS" $BARG > "$R/export_fall.log" 2>&1 ); rc=$?
grep -E '^FALLTRACE|^ZZ FALLTRACE|Traceback|Error|STOP|MISSING' "$R/export_fall.log" | cut -c1-300 | head -n 60
echo "ZZ PODRUN-FALLTRACE DONE rc $rc"
if [ "$rc" = 0 ]; then echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$R/DONE_FALLTRACE.txt"; else echo "rc $rc" > "$R/STOP_FALLTRACE.txt"; fi
exit "$rc"
