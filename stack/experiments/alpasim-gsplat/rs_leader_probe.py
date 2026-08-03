#!/usr/bin/env python3
"""Where do the extra mp4 frames SIT?  Full-resolution leader/trailer test, GPU-free."""
import sys, numpy as np, cv2
sys.path.insert(0, "/home/nvidia/nurec-gsplat")
from nurec_loader import RigTrajectories
ROOT="/home/nvidia/nurec_scenes/sample_set/26.04_release"; CAM="camera_front_wide_120fov"
for scene in sys.argv[1:]:
    rig=RigTrajectories(f"{ROOT}/{scene}/rig_trajectories.json"); nf=rig.n_frames(CAM)
    P=np.array([rig.T_rig_world(CAM,f,1)[:3,3] for f in range(nf)])
    step=np.linalg.norm(np.diff(P,axis=0),axis=1)          # metres per 33.333 ms
    spd=step/0.0333330
    cap=cv2.VideoCapture(f"{ROOT}/{scene}/{CAM}.mp4")
    n_meta=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    head=[]; 
    for i in range(16):
        ok,img=cap.read()
        if not ok: break
        head.append(img.astype(np.int16))
    # full-res consecutive diffs at the head
    hd=[float(np.abs(head[i]-head[i-1]).mean()) for i in range(1,len(head))]
    # tail
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,n_meta-16)); tail=[]
    while True:
        ok,img=cap.read()
        if not ok: break
        tail.append(img.astype(np.int16))
    td=[float(np.abs(tail[i]-tail[i-1]).mean()) for i in range(1,len(tail))]
    cap.release()
    delta=n_meta-nf
    print("="*72)
    print(f"SCENE {scene[:8]}  mp4={n_meta} rig={nf} delta={delta}")
    print(f"  rig ego speed m/s  f0..f9 : {np.round(spd[:9],2)}")
    print(f"  rig ego speed m/s  last 9 : {np.round(spd[-9:],2)}")
    print(f"  FULL-RES |dI| head (mp4 idx 1..{len(hd)}): {np.round(hd,4)}")
    print(f"  FULL-RES |dI| tail (last {len(td)} steps): {np.round(td,4)}")
    thr=0.5
    lead=next((i for i,v in enumerate(hd,start=1) if v>thr), None)
    print(f"  first mp4 index whose |dI| vs predecessor > {thr}: {lead}  "
          f"=> static head block = indices 0..{(lead-1) if lead else '?'} "
          f"({lead} frames) ; pad = {(lead-1) if lead else '?'}")
    print(f"  PREDICTION if pad==delta: pad should equal {delta}")
