#!/bin/bash
cd /home/nvidia/sam3map
for C8 in 73495082f98b 4fbd97b6a4b7; do
  PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_extract_v3.py $C8 all > run_v3_$C8.log 2>&1
  echo "extract_exit=$?" >> run_v3_$C8.log
done
echo ZZV3CHAIN-DONEZZ >> v3_extract_chain.log
