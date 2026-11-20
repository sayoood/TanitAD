#!/usr/bin/env bash
# podread_shufclose10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K): the lead rule fed with the estimate of ANOTHER clip (surrogate tables, form RRe with --closing-table), at age and fresh.
# New runs I10RQ50eSL25KF5P30Hb20, I10RQ50eSN10KF5P30Hb20, I10RQ50eSL25KF5Hb20, I10RQ50eSN10KF5Hb20; banked: the runs of the estimates themselves (card ADD79: I10RQ50eN10KF5P30Hb20, I10RQ50eL25KF5P30Hb20,
# I10RQ50eN10KF5Hb20, I10RQ50eL25KF5Hb20) and, row 137 and card ADD76, RR, RRc, RRh, RLc, no rule R0K at both settings, spread R0Hb20.
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.
# Usage: bash podread_shufclose10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --shuf-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16
#          [--read-dir DIR] [--py PYTHON]
#   --read-dir: folder that holds read_shufclose10k.py, read_estclose10k_n2.py and the engine modules (read_handsrc_decomp.py, read_closrc10k.py, read_agedose_decomp.py, read_agerule_decomp.py, read_longroll10k*.py,
#               strat_roll_read.py); default $PWD
#   --w: queue folder; it must hold SHUFCLOSE10K_DONE.txt and not SHUFCLOSE10K_STOP.txt.  --pull: folder of the run folders I10<run>_s<k> (a link folder as in the reads of cards ADD79 and ADD80 is fine).
#   --estclose-json: READ_ESTCLOSE10K.json of card ADD79 (required, controls_all_pass true).  --shuf-dir: folder of the surrogate tables and SURROGATE_RECORD.json (closeread/est_shuf).
# Output folder OUT: READ_SHUFCLOSE10K.json, read_shufclose10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-SHUFCLOSE10K DONE or ZZ PODREAD-SHUFCLOSE10K STOP <why>.
set -u
PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; ESTJ=""; SHUFDIR=""; W=""; TREE=""; OUT=""; WSHA=""; RD="$PWD"; PY=/workspace/venv/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=$2; shift 2;;
    --code) CODE=$2; shift 2;;
    --grid-dir) GRIDDIR=$2; shift 2;;
    --fresh-grid) FG=$2; shift 2;;
    --midlook-json) MIDJ=$2; shift 2;;
    --estclose-json) ESTJ=$2; shift 2;;
    --shuf-dir) SHUFDIR=$2; shift 2;;
    --w) W=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --wrapper-sha) WSHA=$2; shift 2;;
    --read-dir) RD=$2; shift 2;;
    --py) PY=$2; shift 2;;
    *) echo "ZZ PODREAD-SHUFCLOSE10K STOP unknown-argument $1"; exit 3;;
  esac
done
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-SHUFCLOSE10K STOP $*"; exit 3; }
for v in PULL CODE GRIDDIR FG MIDJ ESTJ SHUFDIR W TREE OUT WSHA; do eval "val=\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev
[ -n "${HEADWAY_JOIN:-}" ] || export HEADWAY_JOIN=$CODE/../data/join_eval141.npz
[ -n "${HEADWAY_BANK:-}" ] || export HEADWAY_BANK=$CODE/../data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
[ -f "$W/SHUFCLOSE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/SHUFCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
[ -e "$OUT/READ_SHUFCLOSE10K.json" ] && stop "ALREADY-READ $OUT/READ_SHUFCLOSE10K.json exists"
case "$WSHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) stop "WRAPPER-SHA must be 16 lowercase hex digits";; esac
MODS="read_shufclose10k.py read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
for f in "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$SHUFDIR/SURROGATE_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
grep -q '"controls_all_pass": true' "$MIDJ" || stop "ADD76-READ-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$ESTJ" || stop "ADD79-READ-NOT-ADMITTED"
grep -q 'PREREG_V8F_PROOF_ADD79' "$ESTJ" || stop "ESTCLOSE-JSON-IS-NOT-THE-READ-OF-ADD79"
grep -q '^ "pass": true' "$SHUFDIR/SURROGATE_RECORD.json" || stop "SURROGATE-RECORD-NOT-PASS"
for e in N10 L25; do for s in 0 1; do [ -f "$SHUFDIR/surr_${e}_s$s.npz" ] || stop "MISSING-INPUT $SHUFDIR/surr_${e}_s$s.npz"; done; done
for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50hKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eSN10KF5P30Hb20 RQ50eSL25KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50hKF5Hb20 RQ50cKF5Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20 RQ50eSN10KF5Hb20 RQ50eSL25KF5Hb20; do
  for s in 0 1; do [ -d "$PULL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
mkdir -p "$OUT"
( cd "$RD" && sha256sum $MODS "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$ESTJ" "$SHUFDIR"/SURROGATE_RECORD.json "$SHUFDIR"/surr_*.npz ) > "$OUT/READER_HASHES.txt"
cut -c1-16,65- "$OUT/READER_HASHES.txt"
( cd "$RD" && "$PY" read_shufclose10k.py --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" read_shufclose10k.py --pull "$PULL" --code "$CODE" --grid-dir "$GRIDDIR" --fresh-grid "$FG" --midlook-json "$MIDJ" --estclose-json "$ESTJ" --shuf-dir "$SHUFDIR" --w "$W" --tree "$TREE" \
    --wrapper-sha "$WSHA" --out "$OUT/READ_SHUFCLOSE10K.json" > "$OUT/read_shufclose10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-SHUFCLOSE|a control did not pass| FAIL |Traceback|Error' "$OUT/read_shufclose10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-SHUFCLOSE10K DONE $(ts)"
