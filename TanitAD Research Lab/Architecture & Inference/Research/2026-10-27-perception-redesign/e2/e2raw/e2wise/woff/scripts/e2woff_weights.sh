#!/bin/bash
# e2woff_weights.sh -- MM 4a3d05, 2026-11-18 (Thor; CPU). model_wireoff.pt for K2DC and K0 (PREREG_E2WISE_2026-11-17_ADDENDUM4.md, 81d409d9):
# K2DC's tail with exactly 8 wire tensors at 0 (6 box_gates + map_chan_proj.proj.weight/.bias); K0 has none -> a byte copy of its tail.
# CONTROL (from the tool's re-load check + here): K2DC n_zeroed 8, K0 n_zeroed 0 and md5_out == md5 of its tail; every other tensor bit-equal.
# Idempotent: an existing output is re-verified, not re-made. Ends ZZWOFFW-DONEZZ or ZZWOFFW-FAIL-<why>ZZ.
set -u
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python; TOOL=$R/inputs/wire_off.py
cd $R || { echo "ZZWOFFW-FAIL-CDZZ"; exit 2; }
for A in K2DC K0; do
  case $A in K2DC) N=8;; K0) N=0;; esac
  T=runs/r9e2-$A/model_ema_tail5k6k.pt; O=runs/r9e2-$A/model_wireoff.pt
  [ -s $T ] || { echo "ZZWOFFW-FAIL-NOTAILZZ $T"; exit 3; }
  if [ -e $O ]; then echo "  $A: $O exists (made by an earlier run of this script): re-verifying, not re-making"
  else $PY -I $TOOL --run $T --out $O --expect $N > logs/e2woff_weights_$A.log 2>&1 || { echo "ZZWOFFW-FAIL-TOOL-$A""ZZ"; tail -3 logs/e2woff_weights_$A.log; exit 5; }; fi
  $PY -I -c '
import hashlib, json, sys
d = json.load(open(sys.argv[1] + ".provenance.json"))
m = hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest()
t = hashlib.md5(open(sys.argv[2], "rb").read()).hexdigest()
n = int(sys.argv[3])
ok = d["n_zeroed"] == n and d["all_others_bit_equal"] is True and d["md5_out"] == m and d["md5_run"] == t and ((m == t) == (n == 0))
print("CONTROL-" + ("OK" if ok else "BAD"), "zeroed", d["n_zeroed"], "untouched", d["n_untouched_bit_equal"], "md5", m, "tail", t)
sys.exit(0 if ok else 1)' $O $T $N || { echo "ZZWOFFW-FAIL-CONTROL-$A""ZZ"; exit 6; }
done
echo "ZZWOFFW-DONEZZ $(date -u +%H:%M:%SZ)"
