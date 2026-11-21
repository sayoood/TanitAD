#!/bin/bash
echo "ZZ-PSALL"; ps -eo pid,etime,pcpu,rss,comm,args --sort=-pcpu | head -15 | cut -c1-200
echo "ZZ-R8"; ls -la /home/nvidia/data/refcv8 | head -30
cd /home/nvidia/tanit-astra-research/data 2>/dev/null && { echo "ZZ-TAR"; du -sh * 2>/dev/null; }
echo "ZZ-PILOT"; ls -la navsim_pilot_20261003 navsim_mini_front_20261003 2>&1 | head -30
echo "ZZ-PILOTTAR"; timeout 60 tar -tzf navsim_pilot_20261003/navsim_metadata.tar.gz 2>/dev/null | head -12; echo "n_members=$(timeout 120 tar -tzf navsim_pilot_20261003/navsim_metadata.tar.gz 2>/dev/null | wc -l)"
echo "ZZ-REPLAY"; ls -la /home/nvidia/tanit-astra-research/runtime/thor_cpu_replay_v1/ 2>&1 | head; ls /home/nvidia/tanit-astra-research/runtime/thor_cpu_replay_v1/nuplan 2>&1 | head
echo "ZZ-NSVENV"; ls ~/venvs/navsim-cpu/lib/python*/site-packages 2>/dev/null | grep -i -E "navsim|nuplan" | head; ls ~/venvs/navsim-cpu-python 2>&1 | head -5
echo "ZZ-END"
