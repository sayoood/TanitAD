#!/bin/bash
# mm_hb_overlap.sh -- the N1DW D1-HO overlap read, T1-kin/old vs T1-kin/hb0.2 (card PREREG_N1DW_HB_OVERLAP.md; PI ruling D16-D, 2026-11-20).
# Starts only after the ADD45 GPU chain ended, then holds the pod GPU queue lock for the whole CPU read, so no training runs on the pod meanwhile.
# Log /root/strat1/n1dw/hb/mm_hb_overlap.log ; last line "ZZ HBOVERLAP DONE ZZ <status>" or "ZZ HBOVERLAP FAIL ZZ <why>".
HB=/root/strat1/n1dw/hb
LOG=$HB/mm_hb_overlap.log
PY=/root/venv/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
fail() { echo "ZZ HBOVERLAP FAIL ZZ $1" >> $LOG; exit 1; }
say "start; waiting for the ADD45 GPU chain to end (cap 30 h)"
t=""
for i in $(seq 1 1800); do
  t=$(tail -n1 /root/dscale_lead/mm_gpu_chain.log 2>/dev/null)
  case "$t" in "ZZ ADD45GPU "*) break;; esac
  sleep 60
done
case "$t" in "ZZ ADD45GPU "*) say "ADD45 chain ended: $t";; *) fail "the ADD45 chain did not end within 30 h";; esac
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9
say "GPU queue lock held: no training on the pod during this read"
cd $HB || fail "no $HB"
( cd $HB/ckpts && sha256sum -c SHA256_PODS.txt ) >> $LOG 2>&1 || fail "N1DW checkpoint copies differ from the pod sha256 record"
say "K-HB0: N1R_s0 through the wrapper with the old executor"
bash $HB/hb_dump.sh n1r N1R_s0 follow none || fail "K-HB0 dump failed (see $HB/out/hbnone/n1r/t1/log_N1R_s0.txt)"
$PY $HB/khb0_compare.py /root/strat1/navrbd_step0/out/t1/dump_NAVRBD-HO-N1R_s0_m10_causal_heads $HB/out/hbnone/n1r/t1/dump_NAVRBD-HO-N1R_s0_m10_causal_heads $HB/K-HB0.json >> $LOG 2>&1 \
  || fail "K-HB0: the wrapper with --kappa-max none does not reproduce the banked N1R_s0 dump bit for bit; no hb read"
say "K-HB0 PASS; 14 dumps at kappa 0.2, three at a time"
cat > $HB/dumplist.txt <<'EOL'
n1r N1R_s0 follow 0.2
n1r N1R_s1 follow 0.2
n1d N1D_s0 follow 0.2
n1d N1D_s1 follow 0.2
n1d N1D_s0 zero 0.2
n1d N1D_s1 zero 0.2
n1dw N1DW2_s0 follow 0.2
n1dw N1DW2_s1 follow 0.2
n1dw N1DW4_s0 follow 0.2
n1dw N1DW4_s1 follow 0.2
n1dw N1DW2_s0 zero 0.2
n1dw N1DW2_s1 zero 0.2
n1dw N1DW4_s0 zero 0.2
n1dw N1DW4_s1 zero 0.2
EOL
xargs -P 3 -L 1 bash $HB/hb_dump.sh < $HB/dumplist.txt >> $LOG 2>&1
nf=$(ls $HB/out/hb0.2/*/t1*/FAIL_* 2>/dev/null | wc -l)
nd=$(ls $HB/out/hb0.2/*/t1*/DONE_* 2>/dev/null | wc -l)
say "dumps: $nd DONE, $nf FAIL"
[ "$nd" = 14 ] && [ "$nf" = 0 ] || fail "dumps: $nd of 14 DONE, $nf FAIL"
say "read"
$PY -u $HB/n1dw_ho_read_hb.py > $HB/read_hb.log 2>&1 || fail "read failed (see $HB/read_hb.log)"
st=$($PY -c "import json; print(json.load(open('$HB/read/RESULT_N1DW_D1HO_HB.json'))['status_hb'])")
echo "ZZ HBOVERLAP DONE ZZ $st" >> $LOG
