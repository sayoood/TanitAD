import sys, glob, json, numpy as np
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
ROOT=sys.argv[1]; seed=sys.argv[2]
files=sorted(glob.glob(f"{ROOT}/dump_s{seed}/decisions/ep*.npz"))
D={}
for i,f in enumerate(files):
    z=np.load(f,allow_pickle=True)
    for k in z.files:
        D.setdefault(k,[]).append(z[k])
    D.setdefault("eid",[]).append(np.full(len(z['ws']),i))
Dc={k:np.concatenate(v) for k,v in D.items() if k not in ("sel_bank_nav_true","sel_bank_nav_shuffled","sel_bank_nav_flipped","sel_bank_nav_zero") and v[0].ndim>=1 and v[0].shape[0]==len(D['ws'][0]) or k=="eid"}
# above filter may drop; rebuild cleanly
keep=["ws","lat_pred_nav_true","lon_pred_nav_true","lat_pred_nav_zero","lon_pred_nav_zero","lat_pred_nav_shuffled","lon_pred_nav_shuffled","lat_label","lon_label","route_label","nav_cmd","nav_cmd_clip","plan_full_nav_true","plan_full_nav_zero","ego_t0","gt_future_ext","gt_future_valid_ext","eid","route_pred_nav_true","r8_nav_known"]
X={k:np.concatenate(D[k]) for k in keep if k in D}
print("windows",len(X['ws']),"episodes",len(files))
np.savez_compressed(S+f"X_s{seed}_{ROOT.split('/')[-1]}.npz",**X)
