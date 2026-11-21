#!/usr/bin/env bash
# podread_tabclose10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD84 (TABCLOSE10K): the token-head table and the planted-quality tables fed to the rule (form RRe, --closing-table), at age and fresh.
# New runs by part (TB_PART / --part): A = I10RQ50e{TK,STK}KF5P30Hb20 and I10RQ50eTKKF5Hb20; B = I10RQ50e{L100,W70,W50,W90,W00,S70}KF5P30Hb20; ALL = both. Banked: the runs of the estimates (card ADD79), RR, RRc, RLc, R0K at both settings, R0Hb20.
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.
# Usage: bash podread_tabclose10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --tab-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16
#          [--part A|B|ALL] [--read-dir DIR] [--py PYTHON]
#   --read-dir: folder that holds read_tabclose10k.py, read_estclose10k_n2.py and the engine modules; default $PWD.  --w: queue folder; it must hold TABCLOSE10K_DONE[_part].txt and not TABCLOSE10K_STOP.txt.
#   --tab-dir: folder of the tables and TAB_TABLES_RECORD.json (closeread/est_tab).  --estclose-json: READ_ESTCLOSE10K.json of card ADD79 (required, controls_all_pass true).
# Output folder OUT: READ_TABCLOSE10K.json (+ .txt), read_tabclose10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-TABCLOSE10K DONE or ZZ PODREAD-TABCLOSE10K STOP <why>.
set -u
PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; ESTJ=""; TABDIR=""; W=""; TREE=""; OUT=""; WSHA=""; RD="$PWD"; PY=/workspace/venv/bin/python; PART=ALL
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=$2; shift 2;;
    --code) CODE=$2; shift 2;;
    --grid-dir) GRIDDIR=$2; shift 2;;
    --fresh-grid) FG=$2; shift 2;;
    --midlook-json) MIDJ=$2; shift 2;;
    --estclose-json) ESTJ=$2; shift 2;;
    --tab-dir) TABDIR=$2; shift 2;;
    --part) PART=$2; shift 2;;
    --w) W=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --wrapper-sha) WSHA=$2; shift 2;;
    --read-dir) RD=$2; shift 2;;
    --py) PY=$2; shift 2;;
    *) echo "ZZ PODREAD-TABCLOSE10K STOP unknown-argument $1"; exit 3;;
  esac
done
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-TABCLOSE10K STOP $*"; exit 3; }
for v in PULL CODE GRIDDIR FG MIDJ ESTJ TABDIR W TREE OUT WSHA; do eval "val=\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done
case "$PART" in A|B|ALL) ;; *) stop "PART must be A, B or ALL";; esac
PS=""; [ "$PART" = ALL ] || PS="_$PART"
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev
[ -n "${HEADWAY_JOIN:-}" ] || export HEADWAY_JOIN=$CODE/../data/join_eval141.npz
[ -n "${HEADWAY_BANK:-}" ] || export HEADWAY_BANK=$CODE/../data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/TABCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
[ -e "$OUT/READ_TABCLOSE10K.json" ] && stop "ALREADY-READ $OUT/READ_TABCLOSE10K.json exists"
case "$WSHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) stop "WRAPPER-SHA must be 16 lowercase hex digits";; esac
MODS="read_tabclose10k.py read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
for f in "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$TABDIR/TAB_TABLES_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
grep -q '"controls_all_pass": true' "$MIDJ" || stop "ADD76-READ-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$ESTJ" || stop "ADD79-READ-NOT-ADMITTED"
grep -q 'PREREG_V8F_PROOF_ADD79' "$ESTJ" || stop "ESTCLOSE-JSON-IS-NOT-THE-READ-OF-ADD79"
ARMS_N=""; [ "$PART" != B ] && ARMS_N="TK STK"; [ "$PART" != A ] && ARMS_N="$ARMS_N L100 W70 W50 W90 W00 S70"
for x in $ARMS_N; do for s in 0 1; do [ -f "$TABDIR/tab_${x}_s$s.npz" ] || stop "MISSING-INPUT $TABDIR/tab_${x}_s$s.npz"; done; done
NEWRUNS=""; [ "$PART" != B ] && NEWRUNS="RQ50eTKKF5P30Hb20 RQ50eSTKKF5P30Hb20 RQ50eTKKF5Hb20"
[ "$PART" != A ] && NEWRUNS="$NEWRUNS RQ50eL100KF5P30Hb20 RQ50eW70KF5P30Hb20 RQ50eW50KF5P30Hb20 RQ50eW90KF5P30Hb20 RQ50eW00KF5P30Hb20 RQ50eS70KF5P30Hb20"
for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50cKF5Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20 $NEWRUNS; do
  for s in 0 1; do [ -d "$PULL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
mkdir -p "$OUT"
( cd "$RD" && sha256sum $MODS "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$TABDIR"/TAB_TABLES_RECORD.json "$TABDIR"/tab_*.npz ) > "$OUT/READER_HASHES.txt"
cut -c1-16,65- "$OUT/READER_HASHES.txt"
( cd "$RD" && "$PY" read_tabclose10k.py --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" read_tabclose10k.py --pull "$PULL" --code "$CODE" --grid-dir "$GRIDDIR" --fresh-grid "$FG" --midlook-json "$MIDJ" --estclose-json "$ESTJ" --tab-dir "$TABDIR" --part "$PART" --w "$W" --tree "$TREE" \
    --wrapper-sha "$WSHA" --out "$OUT/READ_TABCLOSE10K.json" > "$OUT/read_tabclose10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-TABCLOSE|a control did not pass| FAIL |Traceback|Error' "$OUT/read_tabclose10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-TABCLOSE10K DONE $(ts)"
