#!/usr/bin/env bash
# podread_constclose10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD80 (CONSTCLOSE10K): the reader-fed lead rule with a constant closing input (form RRe, 1.0 / 2.0 / 4.0 m/s), at age and fresh.
# New runs I10RQ50e<C10|C20|C40>KF5P30Hb20 and I10RQ50e<C10|C20|C40>KF5Hb20; banked (row 137 and card ADD76): RR, RRc, RRh, RLc, no rule R0K at both settings, spread R0Hb20. With --estclose-json (the result of
# the read of card ADD79) that says controls_all_pass true, also the runs RQ50eN10KF5P30Hb20, RQ50eL25KF5P30Hb20, RQ50eN10KF5Hb20, RQ50eL25KF5Hb20 are read and the pairs with the estimators are written.
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.
# Usage: bash podread_constclose10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --const-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16 [--estclose-json FILE] [--read-dir DIR] [--py PYTHON]
#   --read-dir: folder that holds read_constclose10k_n1.py, read_estclose10k.py, closest.py and the engine modules (read_handsrc_decomp.py, read_closrc10k.py, read_agedose_decomp.py, read_agerule_decomp.py,
#               read_longroll10k*.py, strat_roll_read.py); default $PWD
#   --w: queue folder; it must hold CONSTCLOSE10K_DONE.txt and not CONSTCLOSE10K_STOP.txt.  --pull: folder of the run folders I10<run>_s<k>.  --const-dir: the folder of the three constant files of the queue.
#   --estclose-json: optional; absent or not admitted, the pairs with the estimators and their verdicts are not written (the reason is printed; this is not a failed control).
# Output folder OUT: READ_CONSTCLOSE10K.json, read_constclose10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-CONSTCLOSE10K DONE or ZZ PODREAD-CONSTCLOSE10K STOP <why>.
set -u
PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; CONSTDIR=""; W=""; TREE=""; OUT=""; WSHA=""; ESTJ=""; RD="$PWD"; PY=/workspace/venv/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=$2; shift 2;;
    --code) CODE=$2; shift 2;;
    --grid-dir) GRIDDIR=$2; shift 2;;
    --fresh-grid) FG=$2; shift 2;;
    --midlook-json) MIDJ=$2; shift 2;;
    --const-dir) CONSTDIR=$2; shift 2;;
    --estclose-json) ESTJ=$2; shift 2;;
    --w) W=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --wrapper-sha) WSHA=$2; shift 2;;
    --read-dir) RD=$2; shift 2;;
    --py) PY=$2; shift 2;;
    *) echo "ZZ PODREAD-CONSTCLOSE10K STOP unknown-argument $1"; exit 3;;
  esac
done
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CONSTCLOSE10K STOP $*"; exit 3; }
for v in PULL CODE GRIDDIR FG MIDJ CONSTDIR W TREE OUT WSHA; do eval "val=\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev
[ -n "${HEADWAY_JOIN:-}" ] || export HEADWAY_JOIN=$CODE/../data/join_eval141.npz
[ -n "${HEADWAY_BANK:-}" ] || export HEADWAY_BANK=$CODE/../data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export CC_ESTREAD_DIR=$RD CLOSEST_DIR=$RD
[ -f "$W/CONSTCLOSE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/CONSTCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
[ -e "$OUT/READ_CONSTCLOSE10K.json" ] && stop "ALREADY-READ $OUT/READ_CONSTCLOSE10K.json exists"
case "$WSHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) stop "WRAPPER-SHA must be 16 lowercase hex digits";; esac
MODS="read_constclose10k_n1.py read_estclose10k.py closest.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
for f in "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
grep -q '"controls_all_pass": true' "$MIDJ" || stop "ADD76-READ-NOT-ADMITTED"
for c in C10 C20 C40; do [ -f "$CONSTDIR/const_${c}_H1.npz" ] || stop "MISSING-INPUT $CONSTDIR/const_${c}_H1.npz"; done
EARGS=""; ERUNS=""
if [ -n "$ESTJ" ]; then
  [ -f "$ESTJ" ] || stop "MISSING-INPUT $ESTJ"
  EARGS="--estclose-json $ESTJ"
  if grep -q '"controls_all_pass": true' "$ESTJ"; then ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20"; echo "[$(ts)] the read of card ADD79 is admitted: the pairs with N10 and L25 are written"
  else echo "[$(ts)] the read of card ADD79 has not all its controls passed: the pairs with N10 and L25 are not written (not a failed control)"; fi
else echo "[$(ts)] no --estclose-json: the pairs with N10 and L25 are not written (not a failed control)"; fi
for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50hKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eC10KF5P30Hb20 RQ50eC20KF5P30Hb20 RQ50eC40KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50hKF5Hb20 RQ50cKF5Hb20 RQ50eC10KF5Hb20 RQ50eC20KF5Hb20 RQ50eC40KF5Hb20 $ERUNS; do
  for s in 0 1; do [ -d "$PULL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
mkdir -p "$OUT"
( cd "$RD" && sha256sum $MODS "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$CONSTDIR"/const_C*_H1.npz $ESTJ ) > "$OUT/READER_HASHES.txt"
cut -c1-16,65- "$OUT/READER_HASHES.txt"
( cd "$RD" && "$PY" read_constclose10k_n1.py --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$OUT/selftest_pod.log" | cut -c1-160
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" read_constclose10k_n1.py --pull "$PULL" --code "$CODE" --grid-dir "$GRIDDIR" --fresh-grid "$FG" --midlook-json "$MIDJ" --const-dir "$CONSTDIR" --w "$W" --tree "$TREE" --wrapper-sha "$WSHA" $EARGS --out "$OUT/READ_CONSTCLOSE10K.json" > "$OUT/read_constclose10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-CONSTCLOSE|a control did not pass| FAIL |Traceback|Error|^NOTE' "$OUT/read_constclose10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-CONSTCLOSE10K DONE $(ts)"
