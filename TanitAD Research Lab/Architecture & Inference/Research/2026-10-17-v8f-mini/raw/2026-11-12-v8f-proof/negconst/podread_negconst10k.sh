#!/usr/bin/env bash
# podread_negconst10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD82 (NEGCONST10K): the reader-fed lead rule with a constant closing input of the cautious sign (form RRe, -1.0 / -2.0 / -4.0 m/s), at age and fresh.
# New runs I10RQ50e<M10|M20|M40>KF5P30Hb20 and I10RQ50e<M10|M20|M40>KF5Hb20; banked (row 137 and card ADD76): RR, RRc, RRh, RLc, no rule R0K at both settings, spread R0Hb20. REQUIRED beside them, with their result files:
# the runs RQ50eN10 and RQ50eL25 of card ADD79 (--estclose-json, controls_all_pass true) and the runs RQ50eC10, RQ50eC20, RQ50eC40 of card ADD80 (--constclose-json, controls_all_pass true), both settings.
# A missing or not admitted result file stops this script (it is a failed control k3 of the reader, so no name could be written anyway).
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing. Derived from the read script of card ADD80 by make_podread_negconst.py.
# Usage: bash podread_negconst10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --const-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16 --estclose-json FILE --constclose-json FILE [--read-dir DIR] [--py PYTHON]
#   --read-dir: folder that holds read_negconst10k.py, read_estclose10k.py, closest.py and the engine modules (read_handsrc_decomp.py, read_closrc10k.py, read_agedose_decomp.py, read_agerule_decomp.py,
#               read_longroll10k*.py, strat_roll_read.py); default $PWD
#   --w: queue folder; it must hold NEGCONST10K_DONE.txt and not NEGCONST10K_STOP.txt.  --pull: folder of the run folders I10<run>_s<k>.  --const-dir: the folder of the three constant files of the queue (est_negconst).
# Output folder OUT: READ_NEGCONST10K.json, read_negconst10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-NEGCONST10K DONE or ZZ PODREAD-NEGCONST10K STOP <why>.
set -u
PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; CONSTDIR=""; W=""; TREE=""; OUT=""; WSHA=""; ESTJ=""; CONSTJ=""; RD="$PWD"; PY=/workspace/venv/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=$2; shift 2;;
    --code) CODE=$2; shift 2;;
    --grid-dir) GRIDDIR=$2; shift 2;;
    --fresh-grid) FG=$2; shift 2;;
    --midlook-json) MIDJ=$2; shift 2;;
    --const-dir) CONSTDIR=$2; shift 2;;
    --estclose-json) ESTJ=$2; shift 2;;
    --constclose-json) CONSTJ=$2; shift 2;;
    --w) W=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --wrapper-sha) WSHA=$2; shift 2;;
    --read-dir) RD=$2; shift 2;;
    --py) PY=$2; shift 2;;
    *) echo "ZZ PODREAD-NEGCONST10K STOP unknown-argument $1"; exit 3;;
  esac
done
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-NEGCONST10K STOP $*"; exit 3; }
for v in PULL CODE GRIDDIR FG MIDJ CONSTDIR W TREE OUT WSHA ESTJ CONSTJ; do eval "val=\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev
[ -n "${HEADWAY_JOIN:-}" ] || export HEADWAY_JOIN=$CODE/../data/join_eval141.npz
[ -n "${HEADWAY_BANK:-}" ] || export HEADWAY_BANK=$CODE/../data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export CC_ESTREAD_DIR=$RD CLOSEST_DIR=$RD
[ -f "$W/NEGCONST10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/NEGCONST10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
[ -e "$OUT/READ_NEGCONST10K.json" ] && stop "ALREADY-READ $OUT/READ_NEGCONST10K.json exists"
case "$WSHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) stop "WRAPPER-SHA must be 16 lowercase hex digits";; esac
MODS="read_negconst10k.py read_estclose10k.py closest.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
for f in "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
grep -q '"controls_all_pass": true' "$MIDJ" || stop "ADD76-READ-NOT-ADMITTED"
for c in M10 M20 M40; do [ -f "$CONSTDIR/const_${c}_H1.npz" ] || stop "MISSING-INPUT $CONSTDIR/const_${c}_H1.npz"; done
[ -f "$ESTJ" ] || stop "MISSING-INPUT $ESTJ"
[ -f "$CONSTJ" ] || stop "MISSING-INPUT $CONSTJ"
grep -q '"controls_all_pass": true' "$ESTJ" || stop "ADD79-READ-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$CONSTJ" || stop "ADD80-READ-NOT-ADMITTED"
grep -q 'PREREG_V8F_PROOF_ADD79' "$ESTJ" || stop "ADD79-JSON-IS-NOT-THE-RESULT-OF-ADD79"
grep -q 'PREREG_V8F_PROOF_ADD80' "$CONSTJ" || stop "ADD80-JSON-IS-NOT-THE-RESULT-OF-ADD80"
EARGS="--estclose-json $ESTJ --constclose-json $CONSTJ"
ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20"
OLDRUNS="RQ50eC10KF5P30Hb20 RQ50eC20KF5P30Hb20 RQ50eC40KF5P30Hb20 RQ50eC10KF5Hb20 RQ50eC20KF5Hb20 RQ50eC40KF5Hb20"
echo "[$(ts)] the reads of cards ADD79 and ADD80 are admitted: the pairs with N10 and L25 and the runs of +1, +2, +4 m/s are read"
for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50hKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eM10KF5P30Hb20 RQ50eM20KF5P30Hb20 RQ50eM40KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50hKF5Hb20 RQ50cKF5Hb20 RQ50eM10KF5Hb20 RQ50eM20KF5Hb20 RQ50eM40KF5Hb20 $ERUNS $OLDRUNS; do
  for s in 0 1; do [ -d "$PULL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
mkdir -p "$OUT"
( cd "$RD" && sha256sum $MODS "$GRIDDIR/grid_p30.json" "$FG" "$MIDJ" "$CONSTDIR"/const_M*_H1.npz $ESTJ $CONSTJ ) > "$OUT/READER_HASHES.txt"
cut -c1-16,65- "$OUT/READER_HASHES.txt"
( cd "$RD" && "$PY" read_negconst10k.py --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$OUT/selftest_pod.log" | cut -c1-160
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" read_negconst10k.py --pull "$PULL" --code "$CODE" --grid-dir "$GRIDDIR" --fresh-grid "$FG" --midlook-json "$MIDJ" --const-dir "$CONSTDIR" --w "$W" --tree "$TREE" --wrapper-sha "$WSHA" $EARGS --out "$OUT/READ_NEGCONST10K.json" > "$OUT/read_negconst10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-NEGCONST|a control did not pass| FAIL |Traceback|Error|^NOTE' "$OUT/read_negconst10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-NEGCONST10K DONE $(ts)"
