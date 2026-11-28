import sys, gzip, json
OV='C:/Users/Admin/par18/p6adh/work/overlay'  # the mirror-tip overlay (raw/make_overlay.sh)
sys.path[:0]=[OV+'/stack', OV+'/taniteval']
import numpy as np, torch
from taniteval import four_families as ff
from tanitad.refs import refc_tactical as RT
exp=json.load(gzip.open('E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz','rt',encoding='utf-8'))['tokens']
S='D:/Projects/TanitAD/data/refe_navtest/seams/'
for name,f in (('p2','stage2_p2_full.npz'),('u400','stage2_sft4_B_u400_navtest_full.npz')):
    z=np.load(S+f, allow_pickle=False)
    tok=[str(t) for t in z['token']]
    pr=torch.as_tensor(z['poses'][...,:2]).float()
    gt=torch.as_tensor(np.asarray([exp[t]['human_future_poses'] for t in tok],dtype=np.float64)[...,:2]).float()
    out={}
    for nm,w in (('pred',pr),('gt',gt)):
        dyaw,dv,v0,v1,prov=ff.maneuver_kinematics(w,0.5)
        lat,lon=RT.factor_from_kinematics(dyaw,dv,v0,v1)
        out[nm]=lat.numpy(); print(name,nm,prov['n_heading_held_from_last_moving_step'],prov['n_never_moved'])
    cm=np.zeros((3,3),int)
    for g,p in zip(out['gt'],out['pred']): cm[g,p]+=1
    print(name, cm.tolist())
