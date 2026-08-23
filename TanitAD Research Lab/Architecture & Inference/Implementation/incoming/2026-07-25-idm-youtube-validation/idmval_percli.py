"""Per-clip from-video IDM metrics over all 40 0c5f7dac3b11 val eps (stride 1)."""
import sys, json, torch, numpy as np
sys.path.insert(0,"/root/v4eval/stack"); sys.path.insert(0,"/root/v4eval/stack/scripts")
import idm_head as ih, run_idm_proof as R
dev="cuda"; VAL="/root/valdata/physicalai-val-0c5f7dac3b11"
hd=torch.load("/root/idmval/idm_head_v1.pt",map_location="cpu",weights_only=False)
head=ih.IDMHead(**hd["config"]["head_kwargs"]).to(dev); head.load_state_dict(hd["state_dict"]); head.eval()
enc,ro,_=R.load_encoder("/root/models/flagship-30k/ckpt.pt",dev)
rows=[]
for i in range(40):
    d=torch.load(f"{VAL}/ep_{i:05d}.pt",weights_only=False)
    z=R.encode_frames(enc,ro,d["frames_u8"],dev,32).float()
    Z,S,T=ih.build_windows(z,d["poses"].float(),d["actions"].float(),k=4,stride=1)
    m=ih.evaluate(head,Z,S,T,device=dev)
    v=d["poses"][:,3].numpy()
    rows.append(dict(idx=i,epid=int(d.get("episode_id",-1)),n=m["n"],
                     vmean=float(v.mean()),vmin=float(v.min()),vmax=float(v.max()),
                     speed_r2=m["r2"]["speed"],yaw_r2=m["r2"]["yaw_rate"],
                     speed_mae=m["mae"]["speed"],ade_2s=m["ade_2s"]))
json.dump(rows,open("/root/idmval/results/percli.json","w"),indent=1)
print("%3s %6s %4s %5s %5s %6s %6s %6s %6s"%("idx","epid","n","vmn","vmx","spR2","yawR2","spMAE","ade2s"))
for r in sorted(rows,key=lambda x:x["ade_2s"]):
    print("%3d %6d %4d %5.1f %5.1f %6.2f %6.2f %6.2f %6.2f"%(r["idx"],r["epid"],r["n"],r["vmin"],r["vmax"],r["speed_r2"],r["yaw_r2"],r["speed_mae"],r["ade_2s"]))
