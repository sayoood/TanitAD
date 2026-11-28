import sys, glob, torch
sys.path.insert(0,"/root/v4eval/stack"); sys.path.insert(0,"/root/v4eval/stack/scripts")
import idm_head as ih, run_idm_proof as R
dev="cuda"
hd=torch.load("/root/idmval/idm_head_v1.pt",map_location="cpu",weights_only=False)
head=ih.IDMHead(**hd["config"]["head_kwargs"]).to(dev); head.load_state_dict(hd["state_dict"]); head.eval()
files=sorted(glob.glob("/root/idmval/va40/*.pt"))
Z,Sc,Tj=R.windows_from_latents(files,k=4,stride=2)
m=ih.evaluate(head,Z,Sc,Tj,device=dev)
print("MATCHED-SUBSTRATE (pod3 va_* latents, f1b378f295ae, stride2)")
print("n",m["n"],"n_clips",len(files))
print("speed_r2 %.4f  yaw_r2 %.4f  steer_r2 %.4f"%(m["r2"]["speed"],m["r2"]["yaw_rate"],m["r2"]["steer"]))
print("speed_mae %.4f  yaw_mae %.4f"%(m["mae"]["speed"],m["mae"]["yaw_rate"]))
print("ade_2s %.4f  de@h %s"%(m["ade_2s"],[round(x,3) for x in m["de_per_horizon"]]))
print("CARD  speed_r2 0.8853  yaw_r2 0.8075  ade_2s 2.7032  speed_mae 2.0726  n 3517")
