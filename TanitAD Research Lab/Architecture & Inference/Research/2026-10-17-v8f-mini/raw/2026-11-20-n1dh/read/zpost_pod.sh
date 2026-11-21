#!/bin/bash
# MM 2026-11-21: Z-POST of the N1DH card on the pod CPU, in the gap after the RHp3 chain and before the GPU goes to Ruled, so it never runs on a training pod.
# Waits (cap 90 min) until the RHp3 chain's last line is ZZ ROLLHIST-P03 DONE/FAIL and no rh_p03_chain.sh runs; then takes /root/gpu_queue.lock (nothing trains
# while it runs), runs n1dh_zpost.py --profile ho (1 process, 2 threads, nice 10) with K-OL-REAL on both seeds, deletes its own work dir, releases the lock.
# Log /root/strat1/n1dh/zpost/zpost.log; last line ZZ ZPOST DONE ZZ or ZZ ZPOST FAIL ZZ <why>.
N=/root/strat1/n1dh; Z=$N/zpost; PY=/root/venv/bin/python
mkdir -p $Z
exec >> $Z/zpost.log 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
stop() { echo "ZZ ZPOST FAIL ZZ $1"; exit 1; }
say "start; waiting for the RHp3 chain to end"
for i in $(seq 1 180); do
  l=$(tail -n1 /root/rollhist/rh_p03_chain.log 2>/dev/null)
  case "$l" in "ZZ ROLLHIST-P03 "*) break;; esac
  sleep 30
done
case "$(tail -n1 /root/rollhist/rh_p03_chain.log)" in "ZZ ROLLHIST-P03 "*) say "RHp3 chain ended: $(tail -n1 /root/rollhist/rh_p03_chain.log)";; *) stop "RHp3 chain did not end within 90 min";; esac
grep -q 'rh_p03_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null && stop "an rh_p03_chain.sh process still exists"
exec 9>/root/gpu_queue.lock
flock -w 300 9 || stop "GPU queue lock busy for 5 min"
say "GPU queue lock held (no training during Z-POST)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
D=$N/out/hb0.2/t1
cd $N/read_tools || stop "no read_tools"
nice -n 10 $PY -I -u n1dh_zpost.py --tree $N/tree --arm N1DH_s0 $N/runs/N1DH_s0/train/ckpt_last.pt --arm N1DH_s1 $N/runs/N1DH_s1/train/ckpt_last.pt \
  --profile ho --out $Z/RESULT_N1DH_ZPOST.json --work $Z/work --hb-dir $N/hb \
  --real-dump N1DH_s0 $D/dump_N1DH-HO-N1DH_s0_m10_causal_heads --real-dump N1DH_s1 $D/dump_N1DH-HO-N1DH_s1_m10_causal_heads > $Z/zpost_run.log 2>&1
rc=$?
tail -n 25 $Z/zpost_run.log
[ $rc -eq 0 ] || stop "n1dh_zpost.py rc $rc"
[ -s $Z/RESULT_N1DH_ZPOST.json ] || stop "no RESULT_N1DH_ZPOST.json"
sz=$(du -sk $Z/work 2>/dev/null | cut -f1)
rm -rf /root/strat1/n1dh/zpost/work && echo "$(date -u +%FT%TZ) MM zpost_pod.sh deleted own work dir /root/strat1/n1dh/zpost/work (${sz} KB, one-tick dumps, re-creatable)" >> $N/deletions.log
echo "ZZ ZPOST DONE ZZ"
