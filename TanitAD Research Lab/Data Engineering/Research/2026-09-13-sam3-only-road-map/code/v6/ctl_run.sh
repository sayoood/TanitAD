#!/bin/bash
# Thor: the two pre-launch controls for the v6 chain + a 2-frame all-camera extraction smoke on the day clip.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
cd $SM/eval && OMP_NUM_THREADS=2 $PY score_ctl.py > $SM/ctl_score.log 2>&1
cd $SM/eval && env SAM3MAP_ROOT=$SM/front_native OMP_NUM_THREADS=2 $PY render_ctl.py 1f1f05ca011d $SM/1f1f05ca011d_v5 $SM/ctl_render_1f1f05ca011d_v5 > $SM/ctl_render.log 2>&1
(cd $SM/render_1f1f05ca011d_v5 && md5sum f*.png) > $SM/ctl_render_old.md5
(cd $SM/ctl_render_1f1f05ca011d_v5 && md5sum f*.png) > $SM/ctl_render_new.md5
NSAME=$(comm -12 <(sort $SM/ctl_render_old.md5) <(sort $SM/ctl_render_new.md5) | wc -l)
NOLD=$(wc -l < $SM/ctl_render_old.md5)
echo "ZZRENDERCTL-$NSAME-of-${NOLD}ZZ" >> $SM/ctl_render.log
cd $SM && env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_extract_v6.py 4fbd97b6a4b7 40,41 > $SM/smoke_v6_4fbd97b6a4b7.log 2>&1
echo "ZZCTLRUN-ENDZZ" >> $SM/smoke_v6_4fbd97b6a4b7.log
