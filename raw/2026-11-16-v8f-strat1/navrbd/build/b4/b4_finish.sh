#!/bin/bash
# b4_finish.sh: waits (sleep 30, at most 3 h) for the NAVRBD G-SUITE sentinel, then runs the CPU binding preflight and copies the suite result next to it. Last line of /root/strat1/navrbd/b4/finish.log: ZZ B4 FINISH DONE ZZ or ZZ B4 FINISH FAIL ZZ <why>.
L=/root/strat1/logs/navrbd_gsuite.log; O=/root/strat1/navrbd/b4/finish.log
for i in $(seq 1 360); do tail -n1 $L 2>/dev/null | grep -q "ZZ NAVRBD GSUITE" && break; sleep 30; done
tail -n1 $L | grep -q "ZZ NAVRBD GSUITE DONE ZZ" || { echo "ZZ B4 FINISH FAIL ZZ suite: $(tail -n1 $L)" > $O; exit 1; }
cp /root/strat1/gsuite_result_navrbd.json /root/strat1/navrbd/b4/ ; cp /root/strat1/logs/navrbd_gsuite.log /root/strat1/navrbd/b4/gsuite_log.txt
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/root/venv/bin/python /root/strat1/navrbd/binding_preflight.py > /root/strat1/navrbd/b4/binding_preflight.out 2>&1 && echo "ZZ B4 FINISH DONE ZZ" > $O || echo "ZZ B4 FINISH FAIL ZZ preflight (see binding_preflight.out)" > $O
