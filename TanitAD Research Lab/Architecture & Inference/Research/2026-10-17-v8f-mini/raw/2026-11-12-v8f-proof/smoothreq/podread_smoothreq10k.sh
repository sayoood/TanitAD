#!/usr/bin/env bash
# podread_smoothreq10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD85 (SMOOTHREQ10K): tables with a smooth or white planted error, a smooth spread alone, the matched surrogate and the head without its mean bias
# fed to the rule (form RRe, --closing-table), at age. New runs: I10RQ50e{STKM,TKD,Z00,S90,S50,S7B,W7B}KF5P30Hb20. Banked: the 18 runs of card ADD84, RR, RRc, RLc, R0K, the runs of card ADD79, R0Hb20.
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.
# Usage: bash podread_smoothreq10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --tabclose-json FILE --tab-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16
#          [--read-dir DIR] [--py PYTHON]
#   --w: queue folder; it must hold SMOOTHREQ10K_DONE.txt (or both SMOOTHREQ10K_DONE_s0.txt and _s1.txt) and no SMOOTHREQ10K_STOP*.txt.  --tab-dir: folder of the tables and SMOOTH_TABLES_RECORD.json (closeread/est_smooth).
#   --tabclose-json: the result file of the read of card ADD84 (required).  --estclose-json: READ_ESTCLOSE10K.json of card ADD79 (required, controls_all_pass true).
# Output folder OUT: READ_SMOOTHREQ10K.json (+ .txt), read_smoothreq10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-SMOOTHREQ10K DONE or ZZ PODREAD-SMOOTHREQ10K STOP <why>.
set -u
PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; ESTJ=""; TBJ=""; TABDIR=""; W=""; TREE=""; OUT=""; WSHA=""; RD="$PWD"; PY=/workspace/venv/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=$2; shift 2;;
    --code) CODE=$2; shift 2;;
    --grid-dir) GRIDDIR=$2; shift 2;;
    --fresh-grid) FG=$2; shift 2;;
    --midlook-json) MIDJ=$2; shift 2;;
    --estclose-json) ESTJ=$2; shift 2;;
    --tab-dir) TABDIR=$2; shift 2;;
    --tabclose-json) TBJ=$2; shift 2;;
    --w) W=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --wrapper-sha) WSHA=$2; shift 2;;
    --read-dir) RD=$2; shift 2;;
    --py) PY=$2; shift 2;;
    *) echo "ZZ PODREAD-SMOOTHREQ10K STOP unknown-argument $1"; exit 3;;
  esac
done
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-SMOOTHREQ10K STOP $*"; exit 3; }
for v in PULL CODE GRIDDIR FG MIDJ ESTJ TBJ TABDIR W TREE OUT WSHA; do eval "val=\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev
[ -n "${HEADWAY_JOIN:-}" ] || export HEADWAY_JOIN=$CODE/../data/join_eval141.npz
[ -n "${HEADWAY_BANK:-}" ] || export HEADWAY_BANK=$CODE/../data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
{ [ -f "$W/SMOOTHREQ10K_DONE.txt" ] || { [ -f "$W/SMOOTHREQ10K_DONE_s0.txt" ] && [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]; }; } || stop "QUEUE-NOT-DONE"
for sf in "$W"/SMOOTHREQ10K_STOP*.txt; do [ -e "$sf" ] && stop "QUEUE-STOPPED"; done
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
[ -e "$OUT/READ_SMOOTHREQ10K.json" ] && stop "ALREADY-READ $OUT/READ_SMOOTHREQ10K.json exists"
case "$WSHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) stop "WRAPPER-SHA must be 16 lowercase hex digits";; esac
MODS="read_smoothreq10k.py read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
for f in "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$TABDIR/SMOOTH_TABLES_RECORD.json" "$TBJ"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
grep -q '"controls_all_pass": true' "$MIDJ" || stop "ADD76-READ-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$ESTJ" || stop "ADD79-READ-NOT-ADMITTED"
grep -q 'PREREG_V8F_PROOF_ADD79' "$ESTJ" || stop "ESTCLOSE-JSON-IS-NOT-THE-READ-OF-ADD79"
grep -q 'PREREG_V8F_PROOF_ADD84' "$TBJ" || stop "ADD84-JSON-IS-NOT-THE-READ-OF-ADD84"
ARMS_N="STKM TKD Z00 S90 S50 S7B W7B"
for x in $ARMS_N; do for s in 0 1; do [ -f "$TABDIR/tab_${x}_s$s.npz" ] || stop "MISSING-INPUT $TABDIR/tab_${x}_s$s.npz"; done; done
NEWRUNS=""; for x in $ARMS_N; do NEWRUNS="$NEWRUNS RQ50e${x}KF5P30Hb20"; done
BANK84=""; for x in TK STK L100 W00 W50 W70 W90 S70; do BANK84="$BANK84 RQ50e${x}KF5P30Hb20"; done
for r in R0Hb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 $BANK84 $NEWRUNS; do
  for s in 0 1; do [ -d "$PULL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
mkdir -p "$OUT"
( cd "$RD" && sha256sum $MODS "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$TBJ" "$TABDIR"/SMOOTH_TABLES_RECORD.json "$TABDIR"/tab_*.npz ) > "$OUT/READER_HASHES.txt"
cut -c1-16,65- "$OUT/READER_HASHES.txt"
( cd "$RD" && "$PY" read_smoothreq10k.py --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" read_smoothreq10k.py --pull "$PULL" --code "$CODE" --grid-dir "$GRIDDIR" --fresh-grid "$FG" --midlook-json "$MIDJ" --estclose-json "$ESTJ" --tabclose-json "$TBJ" --tab-dir "$TABDIR" --w "$W" --tree "$TREE" \
    --wrapper-sha "$WSHA" --out "$OUT/READ_SMOOTHREQ10K.json" > "$OUT/read_smoothreq10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-SMOOTHREQ|a control did not pass| FAIL |Traceback|Error' "$OUT/read_smoothreq10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-SMOOTHREQ10K DONE $(ts)"
