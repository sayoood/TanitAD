import torch
from tanitad.refs import refc
from tanitad.refs import refc_sampler as rs

hz=(5,10,15,20,25,30,40,60); n=32
torch.manual_seed(0)
cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="ddim")
dec = refc.AnchoredDiffusionDecoder(feat_dim=16, n_steps=len(hz), d_meas=8, d_ctx=4,
    tac_latent_dim=4, anchors=torch.randn(n,len(hz),2), cfg=cfg, hierarchy=False,
    graft_maneuver=False, graft_target_latent=False, grounded_selector=False,
    horizons=hz, v0_conditioned=True, control_units="alat").eval()
torch.manual_seed(1)
dec.anchor_controls.copy_(torch.stack([torch.linspace(-4,2,n), torch.linspace(-3,3,n)],-1))

fm,m,v = torch.randn(4,16,3,5), torch.randn(4,8), torch.tensor([5.,12.,20.,30.])
norm = torch.tensor(cfg.control_norm)
u_anchor = dec.anchor_control_seq(4, torch.float32)

print("=== A. ZERO-INIT head: u0_hat must be EXACTLY the noised anchor ===")
with torch.no_grad():
    torch.manual_seed(7); a = dec(fm,m,steps=2,v_ms=v)
d = ((a["u0_hat"]-u_anchor)/norm).abs()
s8=float(dec.sched.sqrt_one_minus_abar(8)); a8=float(dec.sched.sqrt_abar(8))
print(f"  |u0_hat - anchor|/norm : mean {float(d.mean()):.5f} max {float(d.max()):.5f}")
print(f"  schedule says sigma(8)={s8:.5f}  sqrt_abar(8)={a8:.5f}  -> deviation IS the anchored Gaussian")

print()
print("=== B. OPEN the denoiser: does the LOOP move the state? (refcv3: 201/201 UNCHANGED) ===")
torch.nn.init.normal_(dec.control_head.weight, std=0.01)
sd = dec.sched
x0n = u_anchor/norm
torch.manual_seed(7); eps = torch.randn_like(x0n)
xt = sd.add_noise(x0n, eps, torch.tensor(8))
kv = dec.feat_proj(fm.flatten(2).transpose(1,2)); cond = dec.cond_proj(m)
ladder = sd.infer_timesteps(8,2); print(f"  ladder {ladder}")
with torch.no_grad():
    for i,t in enumerate(ladder):
        tp = ladder[i+1] if i+1<len(ladder) else 0
        xp = dec._state_to_path(xt*norm, v, False)
        c,du = dec._decode_ctrl(kv,cond,xp,torch.full((4,),float(t)),None,None)
        x0h = xt+du
        nxt = sd.step(x0h, xt, torch.tensor(t), torch.tensor(tp))
        print(f"  step t={t:2d}->{tp:2d}: |du| mean {float(du.abs().mean()):.5f} | "
              f"state moved {float((nxt-xt).abs().mean()):.5f} | "
              f"rows changed {int((nxt!=xt).any(-1).any(-1).sum())}/4")
        xt = nxt

print()
print("=== C. ranking on the SAMPLER's own surface (sel.refined) ===")
from tanitad.refs.refc import SelectionConfig
print("  SelectionConfig().refined default =", SelectionConfig().refined)
