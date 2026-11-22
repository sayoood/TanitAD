#!/usr/bin/env bash
# pod_lookgap_fit_queue.sh -- FIT queue of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0), stage C. Pod, detached, resumable. Deletes and moves nothing, kills nothing.
# What it does (the pilot's fit tool and dump tool of card ADD99 are used unchanged, by path, from the pilot's kit folder):
#   gate    : $ROOT/DONE_ENC.txt exists, $ROOT/STOP_ENC.txt does not, $ROOT/ENC_M1.json has "pass_m1": true   -> else exit 4 (REFUSED)
#   run root: $RUN, made once, nothing overwritten: lists copied from the pilot's run (sha256 compared), tok/B = symbolic link to $ROOT/tok/P (arm P is handed to the
#             tool under the arm name B; no token file is copied), present_tpl/ links (seed 2 has no present dump of its own: it reads the one of seed 0, as card ADD104 did),
#             RUNROOT.json written once.
#   fits    : fit_s0, fit_s1, fit_s2, fit_perm0   (head H5; at most 3 at a time; slot rule before each start)
#   dumps   : dump_s0, dump_s1, dump_s2, dump_perm0   (one at a time, after all four fits)
#   xdumps  : xdump_ident (pilot head of seed 0 on the pilot's tokens of arm B, compared with the pilot's own dump; a failing identity is logged and the queue goes on),
#             xdump_s0, xdump_s1 (pilot heads of seeds 0 and 1 on the tokens of arm P)
# Stop rule: before each job start, the sum of the .SECS files of the finished jobs must be at most LG_STOP_S (default 5400), else STOP_FIT.txt. A failed job -> STOP_FIT.txt.
# Markers in $ROOT: START_FIT.txt, DONE_FIT.txt, STOP_FIT.txt. Log: $ROOT/fit_queue.log. Times: $ROOT/FIT_TIMES.txt. Jobs: $ROOT/queue_fit/jobs, logs: $ROOT/queue_fit/logs.
# Environment: DRY=1 prints every command and creates nothing at all; RESUME=1 continues after STOP_FIT.txt; LG_STOP_S=5400; LG_STAGGER_S=60 (pause after a fit start so its GPU process shows).
# Start (lead): nohup setsid bash /workspace/v8proof/lookgap/kit_fit/pod_lookgap_fit_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/workspace/v8proof/lookgap
RUN=$ROOT/run
R1=/workspace/v8proof/cropread/run1
K=/workspace/v8proof/cropread/kit/fit
KF=$ROOT/kit_fit
PY=/workspace/venv/bin/python
JOIN=/workspace/v8proof/data/inputs/join_train.npz
PRES=/workspace/v8proof/closeread/dump/both
Q=$ROOT/queue_fit
JD=$Q/jobs
LD=$Q/logs
LOG=$ROOT/fit_queue.log
TIMES=$ROOT/FIT_TIMES.txt
DRY=${DRY:-0}
RESUME=${RESUME:-0}
STOP_S=${LG_STOP_S:-5400}
STAGGER_S=${LG_STAGGER_S:-60}
MAXJ=3
MINGB=28
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fitcmd()  { echo "$PY $K/crop_fit.py --arm B --head H5 --seed $1${2:+ $2} --root $RUN --join-train $JOIN --device cuda"; }
dumpcmd() { echo "$PY $K/crop_dump.py --arm B --head H5 --seed $1${2:+ $2} --root $RUN --present-dump $RUN/present_tpl --device cuda"; }
xcmd()    { # fit dir, tok dir, present file, seed index, out, then optional extra words
  local fd=$1 td=$2 pr=$3 si=$4 out=$5; shift 5
  echo "$PY $KF/cross_dump.py --kit-fit $K --fit-dir $fd --tok-dir $td --present $pr --seed-index $si --out $out${*:+ $*}"
}
XIDENT_EXTRA="--compare $R1/dump/B_H5/reader_s0_score.npz --ident-out $ROOT/xdump/XDUMP_IDENT.json"
cmd_ident() { xcmd "$R1/fits/B_H5_s0" "$R1/tok/B" "$PRES/reader_s0_score.npz" 0 "$ROOT/xdump/B_on_B_ident/reader_s0_score.npz" $XIDENT_EXTRA; }
cmd_xs0()   { xcmd "$R1/fits/B_H5_s0" "$ROOT/tok/P" "$PRES/reader_s0_score.npz" 0 "$ROOT/xdump/B_on_P/reader_s0_score.npz"; }
cmd_xs1()   { xcmd "$R1/fits/B_H5_s1" "$ROOT/tok/P" "$PRES/reader_s1_score.npz" 1 "$ROOT/xdump/B_on_P/reader_s1_score.npz"; }

# ------------------------------------------------------------------------------------------------ DRY: print, make nothing (no folder, no lock, no marker)
if [ "$DRY" = 1 ]; then
  echo "DRY=1: nothing is created; commands are printed"
  echo "CMD: gate: REFUSED (exit 4) unless $ROOT/DONE_ENC.txt exists, $ROOT/STOP_ENC.txt does not, and $ROOT/ENC_M1.json has pass_m1 true"
  echo "CMD: ulimit -n 16384"
  echo "CMD: run root $RUN: cp (no overwrite) $R1/lists/train_final.txt and $R1/lists/eval141.txt to $RUN/lists/, sha256 compared"
  echo "CMD: ln -s $ROOT/tok/P $RUN/tok/B"
  echo "CMD: ln -s $PRES/reader_s0_score.npz $RUN/present_tpl/reader_s0_score.npz"
  echo "CMD: ln -s $PRES/reader_s1_score.npz $RUN/present_tpl/reader_s1_score.npz"
  echo "CMD: ln -s $PRES/reader_s0_score.npz $RUN/present_tpl/reader_s2_score.npz"
  echo "CMD: write $RUN/RUNROOT.json once"
  echo "CMD: slot rule before each job: fewer than $MAXJ GPU compute processes (nvidia-smi --query-compute-apps) and more than $MINGB GB free (cgroup memory.max minus anon)"
  echo "CMD: stop rule before each job: sum of the .SECS of the finished jobs at most $STOP_S s, else STOP_FIT.txt"
  echo "JOB fit_s0:";    echo "  CMD: $(fitcmd 0)"
  echo "JOB fit_s1:";    echo "  CMD: $(fitcmd 1)"
  echo "JOB fit_s2:";    echo "  CMD: $(fitcmd 2)"
  echo "JOB fit_perm0:"; echo "  CMD: $(fitcmd 0 --permute-labels)"
  echo "JOB dump_s0 (after the four fits):";    echo "  CMD: $(dumpcmd 0)"
  echo "JOB dump_s1:";    echo "  CMD: $(dumpcmd 1)"
  echo "JOB dump_s2:";    echo "  CMD: $(dumpcmd 2)"
  echo "JOB dump_perm0:"; echo "  CMD: $(dumpcmd 0 --permute-labels)"
  echo "JOB xdump_ident (a failing identity is logged, the queue goes on):"; echo "  CMD: $(cmd_ident)"
  echo "JOB xdump_s0:"; echo "  CMD: $(cmd_xs0)"
  echo "JOB xdump_s1:"; echo "  CMD: $(cmd_xs1)"
  echo "DRY run complete: at most $MAXJ fit jobs at a time, dumps and cross dumps one at a time"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ gate, lock, markers
[ -d "$ROOT" ] || { echo "no $ROOT"; exit 2; }
[ -f "$ROOT/DONE_ENC.txt" ] || { echo "REFUSED: $ROOT/DONE_ENC.txt is missing (the encode is not finished)"; exit 4; }
[ ! -f "$ROOT/STOP_ENC.txt" ] || { echo "REFUSED: $ROOT/STOP_ENC.txt is present"; exit 4; }
[ -f "$ROOT/ENC_M1.json" ] || { echo "REFUSED: $ROOT/ENC_M1.json is missing"; exit 4; }
gate=$("$PY" - "$ROOT/ENC_M1.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as ex:
    print("UNREADABLE %s" % type(ex).__name__); sys.exit(1)
if isinstance(d, dict) and d.get("pass_m1") is True:
    print("GATE-OK pass_m1 true"); sys.exit(0)
print("pass_m1 is not true"); sys.exit(1)
PYEOF
) || { echo "REFUSED: ENC_M1.json: $gate"; exit 4; }
echo "$gate"
[ ! -f "$ROOT/DONE_FIT.txt" ] || { echo "DONE_FIT.txt is present: the fit queue is finished"; exit 3; }
if [ -f "$ROOT/STOP_FIT.txt" ] && [ "$RESUME" != 1 ]; then echo "STOP_FIT.txt is present: RESUME=1 to continue (lead decision)"; exit 3; fi
exec 9>"$ROOT/fit_queue.lock"
flock -n 9 || { echo "queue already running: second start refused"; exit 1; }
mkdir -p "$JD" "$LD" || { echo "cannot make $Q"; exit 2; }
say()   { echo "[queue $(ts)] $*" >> "$LOG"; }
stamp() { echo "$(ts) $*" >> "$TIMES"; }
STOP_BASE=0; [ -f "$ROOT/STOP_FIT.txt" ] && STOP_BASE=$(wc -l < "$ROOT/STOP_FIT.txt")
stopq() { echo "$(ts) $*" >> "$ROOT/STOP_FIT.txt"; say "STOP $*"; stamp "STOP $*"; echo "STOP $*"; exit 3; }
blocked() { [ -f "$ROOT/STOP_FIT.txt" ] && [ "$(wc -l < "$ROOT/STOP_FIT.txt")" -gt "$STOP_BASE" ]; }
echo "$(ts) pid $$ resume $RESUME stop_s $STOP_S" >> "$ROOT/START_FIT.txt"
stamp "START_FIT"
say "start: RESUME=$RESUME LG_STOP_S=$STOP_S"

# ------------------------------------------------------------------------------------------------ inputs
for f in "$K/crop_fit.py" "$K/crop_dump.py" "$KF/cross_dump.py" "$JOIN" "$PRES/reader_s0_score.npz" "$PRES/reader_s1_score.npz" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" \
         "$R1/fits/B_H5_s0/head.pt" "$R1/fits/B_H5_s1/head.pt" "$R1/dump/B_H5/reader_s0_score.npz"; do
  [ -f "$f" ] || stopq "MISSING-INPUT $f"
done
[ -d "$ROOT/tok/P" ] || stopq "MISSING-INPUT $ROOT/tok/P"
[ -d "$R1/tok/B" ] || stopq "MISSING-INPUT $R1/tok/B"
ulimit -n 16384 || stopq "ULIMIT-FAILED"

# ------------------------------------------------------------------------------------------------ run root (once; nothing overwritten)
mkdir -p "$RUN/lists" "$RUN/tok" "$RUN/fits" "$RUN/dump" "$RUN/present_tpl" "$ROOT/xdump" || stopq "CANNOT-MAKE-RUN-ROOT"
sha() { sha256sum "$1" | cut -d' ' -f1; }
for l in train_final.txt eval141.txt; do
  [ -e "$RUN/lists/$l" ] || cp "$R1/lists/$l" "$RUN/lists/$l" || stopq "COPY-FAILED $l"
  [ "$(sha "$RUN/lists/$l")" = "$(sha "$R1/lists/$l")" ] || stopq "LIST-DIFFERS-FROM-PILOT $l"
done
mklink() { # TARGET LINK: makes the link, or accepts the one already there when it points at TARGET
  if [ -L "$2" ] || [ -e "$2" ]; then [ "$(readlink "$2")" = "$1" ] || stopq "LINK-IS-NOT-AS-EXPECTED $2"
  else ln -s "$1" "$2" 2>/dev/null || [ "$(readlink "$2" 2>/dev/null)" = "$1" ] || stopq "LINK-FAILED $2"; fi
}
mklink "$ROOT/tok/P" "$RUN/tok/B"
mklink "$PRES/reader_s0_score.npz" "$RUN/present_tpl/reader_s0_score.npz"
mklink "$PRES/reader_s1_score.npz" "$RUN/present_tpl/reader_s1_score.npz"
mklink "$PRES/reader_s0_score.npz" "$RUN/present_tpl/reader_s2_score.npz"
if [ ! -e "$RUN/RUNROOT.json" ]; then
  "$PY" - "$RUN" "$R1" "$K" "$ROOT" <<'PYEOF' || stopq "RUNROOT-JSON-FAILED"
import hashlib, json, os, sys, time
run, r1, k, root = sys.argv[1:5]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
rec = {"tok_b_realpath": os.path.realpath(os.path.join(run, "tok", "B")),
       "train_list_sha256": sha(os.path.join(run, "lists", "train_final.txt")), "eval_list_sha256": sha(os.path.join(run, "lists", "eval141.txt")),
       "pilot_train_list_sha256": sha(os.path.join(r1, "lists", "train_final.txt")), "pilot_eval_list_sha256": sha(os.path.join(r1, "lists", "eval141.txt")),
       "crop_fit_sha256": sha(os.path.join(k, "crop_fit.py")), "crop_dump_sha256": sha(os.path.join(k, "crop_dump.py")),
       "present_links": {"s%d" % s: os.path.realpath(os.path.join(run, "present_tpl", "reader_s%d_score.npz" % s)) for s in (0, 1, 2)},
       "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
with open(os.path.join(run, "RUNROOT.json"), "x", encoding="utf-8", newline="\n") as fh:
    json.dump(rec, fh, indent=1)
PYEOF
fi
say "run root ready: $RUN"

# ------------------------------------------------------------------------------------------------ slot, stop rule, job runner
mem_free_gb() { local mx an; mx=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); an=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
  if [[ "$mx" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then echo $(( (mx - an) / 1073741824 )); else echo 999; fi; }
wait_slot() {
  local waited=0 n fr
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    fr=$(mem_free_gb)
    if [ "${n:-9}" -lt "$MAXJ" ] && [ "${fr:-0}" -gt "$MINGB" ]; then [ "$waited" -gt 0 ] && say "slot free after ${waited} s (gpu procs $n, memory free ${fr} GB)"; return 0; fi
    [ "$waited" -ge 21600 ] && stopq "no slot after ${waited} s (gpu procs $n, memory free ${fr} GB)"
    [ "$waited" = 0 ] && say "waiting for a slot (gpu procs $n, memory free ${fr} GB)"
    sleep 60; waited=$((waited + 60))
  done
}
secs_sum() { cat "$JD"/*.SECS 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
run_job() { # name, command words (runs in a subshell)
  local j=$1; shift
  local t0 rc; t0=$(date +%s)
  echo "$(ts) START $j" > "$JD/$j.START"
  "$@" > "$LD/$j.log" 2>&1; rc=$?
  echo $(( $(date +%s) - t0 )) > "$JD/$j.SECS"
  if [ "$rc" = 0 ]; then echo "$(ts) DONE $j" > "$JD/$j.DONE"; say "DONE $j"
  else echo "$(ts) rc $rc" > "$JD/$j.STOP"; echo "$(ts) job $j failed rc $rc" >> "$ROOT/STOP_FIT.txt"; say "FAILED $j rc $rc"; fi
  stamp "$j rc $rc"
  return $rc
}
# start_job NAME DONEFILE HALFFILE cmd...: DONEFILE exists -> job counted done (no run). HALFFILE ("-" for none) exists without DONEFILE -> stop. Returns 10 when skipped.
start_job() {
  local j=$1 donef=$2 halff=$3; shift 3
  [ -f "$JD/$j.DONE" ] && return 10
  if [ -f "$donef" ]; then echo "$(ts) DONE $j (output exists)" > "$JD/$j.DONE"; [ -f "$JD/$j.SECS" ] || echo 0 > "$JD/$j.SECS"; return 10; fi
  if [ "$halff" != "-" ] && [ -f "$halff" ]; then echo "$(ts) half dump" > "$JD/$j.STOP"; stopq "half dump: $halff exists without $donef"; fi
  [ -f "$JD/$j.STOP" ] && echo "$(ts) retry $j (an old .STOP stays on disk)" >> "$ROOT/START_FIT.txt"
  local s; s=$(secs_sum)
  [ "$s" -le "$STOP_S" ] || stopq "finished jobs used $s s, above LG_STOP_S $STOP_S"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && return 11
  wait_slot
  ( run_job "$j" "$@" ) &
  return 0
}

# ------------------------------------------------------------------------------------------------ fits (at most 3 at a time)
say "fits: fit_s0 fit_s1 fit_s2 fit_perm0"
start_job fit_s0    "$RUN/fits/B_H5_s0/FIT_RECORD.json"      - $(fitcmd 0);                    [ $? = 0 ] && sleep "$STAGGER_S"
start_job fit_s1    "$RUN/fits/B_H5_s1/FIT_RECORD.json"      - $(fitcmd 1);                    [ $? = 0 ] && sleep "$STAGGER_S"
start_job fit_s2    "$RUN/fits/B_H5_s2/FIT_RECORD.json"      - $(fitcmd 2);                    [ $? = 0 ] && sleep "$STAGGER_S"
start_job fit_perm0 "$RUN/fits/B_H5_s0_perm/FIT_RECORD.json" - $(fitcmd 0 --permute-labels);   [ $? = 0 ] && sleep "$STAGGER_S"
wait
blocked && { echo "STOPPED: see $ROOT/STOP_FIT.txt"; exit 3; }
for j in fit_s0 fit_s1 fit_s2 fit_perm0; do [ -f "$JD/$j.DONE" ] || stopq "fit $j is not done"; done

# ------------------------------------------------------------------------------------------------ dumps and cross dumps, one at a time
seq_job() { # NAME DONEFILE HALFFILE cmd...
  start_job "$@"; local rc=$?
  wait
  blocked && { echo "STOPPED: see $ROOT/STOP_FIT.txt"; exit 3; }
  [ "$rc" = 11 ] && { echo "STOPPED: see $ROOT/STOP_FIT.txt"; exit 3; }
  return 0
}
D=$RUN/dump
seq_job dump_s0    "$D/B_H5/reader_s0_meta.json"      "$D/B_H5/reader_s0_score.npz"      $(dumpcmd 0)
seq_job dump_s1    "$D/B_H5/reader_s1_meta.json"      "$D/B_H5/reader_s1_score.npz"      $(dumpcmd 1)
seq_job dump_s2    "$D/B_H5/reader_s2_meta.json"      "$D/B_H5/reader_s2_score.npz"      $(dumpcmd 2)
seq_job dump_perm0 "$D/B_H5_perm/reader_s0_meta.json" "$D/B_H5_perm/reader_s0_score.npz" $(dumpcmd 0 --permute-labels)
X=$ROOT/xdump
seq_job xdump_ident "$X/B_on_B_ident/reader_s0_meta.json" "$X/B_on_B_ident/reader_s0_score.npz" $(cmd_ident)
if [ -f "$X/XDUMP_IDENT.json" ]; then say "identity record: $(tr -d '\n' < "$X/XDUMP_IDENT.json" | tr -s ' ' | cut -c1-300)"; else say "identity record is missing"; fi
seq_job xdump_s0 "$X/B_on_P/reader_s0_meta.json" "$X/B_on_P/reader_s0_score.npz" $(cmd_xs0)
seq_job xdump_s1 "$X/B_on_P/reader_s1_meta.json" "$X/B_on_P/reader_s1_score.npz" $(cmd_xs1)

ok=1
for j in fit_s0 fit_s1 fit_s2 fit_perm0 dump_s0 dump_s1 dump_s2 dump_perm0 xdump_ident xdump_s0 xdump_s1; do [ -f "$JD/$j.DONE" ] || ok=0; done
if [ "$ok" = 1 ]; then
  echo "ZZ LOOKGAP FIT DONE ZZ $(ts) seconds_sum=$(secs_sum) root=$ROOT" > "$ROOT/DONE_FIT.txt"; stamp "DONE_FIT"; say "DONE"; echo "ZZ LOOKGAP-FIT DONE"
else
  echo "ZZ LOOKGAP-FIT INCOMPLETE"; exit 3
fi
exit 0
