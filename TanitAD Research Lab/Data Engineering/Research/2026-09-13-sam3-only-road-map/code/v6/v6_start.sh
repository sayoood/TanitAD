#!/bin/bash
# Thor: re-smoke frame 41 with the evidence-bitfield extractor, compare its classes with the first smoke, then run the chain.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
cd $SM
[ -d $SM/4fbd97b6a4b7_v6raw ] && mv $SM/4fbd97b6a4b7_v6raw $SM/smoke_v6a_4fbd97b6a4b7_raw
env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_extract_v6.py 4fbd97b6a4b7 41 > $SM/smoke_v6b_4fbd97b6a4b7.log 2>&1
$PY - > $SM/smoke_v6b_compare.log 2>&1 <<'EOF'
import numpy as np
a = np.load("/home/nvidia/sam3map/smoke_v6a_4fbd97b6a4b7_raw/041.npz", allow_pickle=True)
b = np.load("/home/nvidia/sam3map/4fbd97b6a4b7_v6raw/041.npz", allow_pickle=True)
cams = sorted(k[4:] for k in a.files if k.startswith("cls_"))
diff = {c: int((a[f"cls_{c}"] != b[f"cls_{c}"]).sum()) for c in cams}
pts = {k: (len(a[f"pts_{k}"]), len(b[f"pts_{k}"])) for k in range(1, 8)}
ev = {c: {bit: round(100 * float((b[f"evid_{c}"] & bit > 0).mean()), 2) for bit in (1, 2, 4, 8, 16, 32, 64, 128)} for c in cams}
print("class pixels differing per camera (of 518400):", diff)
print("points per class (first smoke, this smoke):", pts)
print("evidence bit coverage %:", ev)
print("ZZCMP-%dZZ" % sum(diff.values()))
EOF
nohup bash $SM/v6_chain.sh 4fbd97b6a4b7 73495082f98b > $SM/v6_chain.out 2>&1 < /dev/null &
echo "ZZV6START-$!ZZ" >> $SM/smoke_v6b_compare.log
