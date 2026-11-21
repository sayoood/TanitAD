#!/bin/bash
cd /root/strat1/n1dw/read/code
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
rm -f /root/strat1/n1dw/read/HO_READ.done /root/strat1/n1dw/read/HO_READ.fail
nice -n 10 /usr/local/bin/python -u n1dw_ho_read.py > /root/strat1/n1dw/read/ho_read.log 2>&1
rc=$?
echo "rc $rc" > /root/strat1/n1dw/read/HO_READ.done
