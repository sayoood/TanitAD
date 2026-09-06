import torch
from tanitad.refs import refc
hz=(5,10,15,20,25,30,40,60); n=32
torch.manual_seed(0)
cfg = refc.DecoderConfig(d=32,n_heads=4,layers=2,ff_mult=2,sampler="ddim")
dec = refc.AnchoredDiffusionDecoder(feat_dim=16,n_steps=len(hz),d_meas=8,d_ctx=4,
    tac_latent_dim=4,anchors=torch.randn(n,len(hz),2),cfg=cfg,hierarchy=False,
    graft_maneuver=False,graft_target_latent=False,grounded_selector=False,
    horizons=hz,v0_conditioned=True,control_units="alat").eval()
torch.manual_seed(1)
dec.anchor_controls.copy_(torch.stack([torch.linspace(-4,2,n),torch.linspace(-3,3,n)],-1))
torch.nn.init.normal_(dec.control_head.weight,std=0.01)   # an OPEN denoiser
fm,m,v = torch.randn(4,16,3,5),torch.randn(4,8),torch.tensor([5.,12.,20.,30.])
outs={}
for k in (1,2,3):
    with torch.no_grad():
        torch.manual_seed(7)             # SAME eps -> only the ladder differs
        outs[k]=dec(fm,m,steps=k,v_ms=v)
print("=== does adding a DDIM rung change the OUTPUT? (refcv3's loop did not) ===")
for a,b in ((1,2),(2,3)):
    du=float((outs[a]["u0_hat"]-outs[b]["u0_hat"]).abs().mean())
    dx=float((outs[a]["anchor_traj"]-outs[b]["anchor_traj"]).abs().mean())
    ch=int((outs[a]["u0_hat"]!=outs[b]["u0_hat"]).any(-1).any(-1).sum())
    print(f"  steps {a} vs {b}: ladder {outs[a]['sel_tele']['sampler_ladder']} vs "
          f"{outs[b]['sel_tele']['sampler_ladder']}")
    print(f"     mean|d u0_hat| {du:.5f}   mean|d fan| {dx:.5f} m   rows changed {ch}/{4*n}")
