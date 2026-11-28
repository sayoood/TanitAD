"""The ONLY resident configurations this 8.19 GiB card can hold, measured properly.
Writes each row as it completes (unbuffered) -- the previous probe hid everything behind a pipe."""
import sys, time
import torch
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFeConfig, REFe, param_report, wta_loss
out=open("cost2.txt","w",encoding="utf-8")
def P(s):
    print(s); out.write(s+"\n"); out.flush(); sys.stdout.flush()
free0,total=torch.cuda.mem_get_info()
P(f"device {torch.cuda.get_device_name(0)}  total {total/1e9:.2f} GB  free at start {free0/1e9:.2f} GB")
P(f"{'cams':>4s} {'batch':>5s} {'imgs':>4s} {'peak GB':>8s} {'resident?':>10s} {'s (best of 3)':>14s} {'s/image':>8s}")
res={}
for nc,b in [(1,1),(1,2)]:
    c=REFeConfig.for_backbone("vitl16"); c.n_cameras=nc
    m=REFe(c).cuda()
    img=torch.randn(b,nc,3,c.img_h,c.img_w,device="cuda"); ego=torch.randn(b,c.ego_dim,device="cuda")
    goal=torch.randn(b,2*c.n_goal_points,device="cuda"); tgt=torch.randn(b,c.horizon_steps,3,device="cuda")
    ts=[]
    for i in range(4):
        torch.cuda.synchronize()
        if i==1: torch.cuda.reset_peak_memory_stats()
        t0=time.time()
        tr,sc=m(img,ego,goal); l,_=wta_loss(tr,tgt); l.backward()
        torch.cuda.synchronize(); ts.append(time.time()-t0); m.zero_grad(set_to_none=True)
    pk=torch.cuda.max_memory_allocated()/1e9
    res[(nc,b)]=(pk,min(ts[1:]))
    P(f"{nc:4d} {b:5d} {nc*b:4d} {pk:8.2f} {('YES' if pk<free0/1e9 else 'NO -- SPILLS'):>10s} "
      f"{min(ts[1:]):14.3f} {min(ts[1:])/(nc*b):8.3f}")
    del m,img,ego,goal,tgt,tr,sc,l; torch.cuda.empty_cache()
a=res[(1,1)]; bb=res[(1,2)]
marg=(bb[0]-a[0]); P("")
P(f"marginal memory per IMAGE (1 cam): {marg:.2f} GB   fixed (weights+optimiser+grads): {a[0]-marg:.2f} GB")
P(f"per-image time at batch 2: {bb[1]/2:.3f} s   (batch 1: {a[1]:.3f} s)")
P(f"=> a RESIDENT 4-image step would cost about {4*marg + (a[0]-marg):.2f} GB and "
  f"{4*bb[1]/2:.2f} s on this card's arithmetic -- but {4*marg+(a[0]-marg):.2f} GB > {free0/1e9:.2f} GB free,")
P(f"   so NO 4-image configuration is resident here and the 4-camera seconds cannot be measured on this box.")
P("COST2_DONE")
