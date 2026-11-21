#!/usr/bin/env bash
# podread_price.sh : pod-side read of card PREREG_V8F_PROOF_ADD47 (PRICE10K). Same readers, same pairs, same rules as read_price10k.sh; only the machine differs
# (the dev box had under 1 GB of commit memory free, held by jobs of other sessions; logged in TIMES_NOTE.md before this ran).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 cell for cell, same reader command); only if e1 passes, the read of card ADD47.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-PRICE DONE or ZZ PODREAD-PRICE STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-PRICE STOP $*"; exit 3; }
# memory guard: 20 GB of the container limit must be free
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1" "$OUT/read_price"
# the pull layout of the dev box, as links (read_price10k.py and control m5 read it)
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_price" ] || ln -s "$W/small_price" "$PL/small_price"
[ -f "$PL/small_price/PRICE_MAPS_CHECK.json" ] || stop "NO-MAPS-CHECK-FILE"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1/hw" > "$OUT/e1/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1/q3" > "$OUT/e1/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1/hw.log" "$OUT/e1/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1" --out "$OUT/e1/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD47 (pairs of read_price10k.sh)
PAIRS="I10RQ50:I10R0,I10RQ50:I10RLb,I10RQ50:I10RR,I10RQ35:I10R0,I10RQ35:I10RLb,I10RRaw:I10R0,I10RRaw:I10RLb,I10RR:I10R0,I10RR:I10RLb"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_price/hw" > "$OUT/read_price/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_price/q3" > "$OUT/read_price/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_price10k.py" --read-dir "$OUT/read_price" --pull "$PL" --reused "$P/root/integ10k/PRICE10K_REUSED.json" --out "$OUT/read_price/READ_PRICE10K.json" > "$OUT/read_price/read_price10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 64 "$OUT/read_price/read_price10k.log" | cut -c1-430
echo "ZZ PODREAD-PRICE DONE $(ts)"
