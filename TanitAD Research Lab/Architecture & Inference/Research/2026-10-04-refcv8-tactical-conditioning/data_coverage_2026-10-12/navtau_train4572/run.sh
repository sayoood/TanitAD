#!/bin/bash
D=/home/nvidia/navtau_4572; T=/home/nvidia/refcv6_run/fa43d9c5ce; PY=/home/nvidia/venvs/tanitad-train/bin/python
L=/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz
cd $D || exit 2
md5sum derive_navc_tau.py $L /home/nvidia/data/refcv6-b1-416x1024-train/_v2manifest.pt /home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt > inputs_md5.txt
free -g | awk '/Mem:/{print "avail_gb", $7}' > mem_before.txt
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
nice -n 19 $PY derive_navc_tau.py $T /home/nvidia/data/refcv6-b1-416x1024-train/_v2manifest.pt $L $D/ctl_old4369.json > ctl.log 2>&1
echo "ctl_rc=$?" > rc.txt
nice -n 19 $PY derive_navc_tau.py $T /home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt $L $D/nav_compliance_tau_train4572.json > new.log 2>&1
echo "new_rc=$?" >> rc.txt
echo DONE >> rc.txt
