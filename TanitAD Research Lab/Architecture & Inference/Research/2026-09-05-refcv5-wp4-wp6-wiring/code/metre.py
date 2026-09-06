import torch
from tanitad.refs import refc
from tanitad.refs import refc_sampler as rs

hz=(5,10,15,20,25,30,35,40); n=32
def mk(space):
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32,n_heads=4,layers=2,ff_mult=2,
                             sampler="ddim", sampler_space=space)
    d = refc.AnchoredDiffusionDecoder(feat_dim=16,n_steps=len(hz),d_meas=8,d_ctx=4,
        tac_latent_dim=4,anchors=torch.randn(n,len(hz),2),cfg=cfg,hierarchy=False,
        graft_maneuver=False,graft_target_latent=False,grounded_selector=False,
        horizons=hz,v0_conditioned=True,control_units="alat").eval()
    torch.manual_seed(1)
    d.anchor_controls.copy_(torch.stack(
        [torch.linspace(-1.0,1.0,n), torch.linspace(-1.0,1.0,n)],-1))
    return d

def implied_alat(p):
    """|d2y/dt2| between consecutive slots, dt = 0.5 s."""
    y=p[...,1]; d1=(y[...,1:]-y[...,:-1])/0.5
    return ((d1[...,1:]-d1[...,:-1])/0.5).abs()

fm,m,v = torch.randn(1,16,3,5), torch.randn(1,8), torch.tensor([15.0])
print("=== THE DELIBERATE REGRESSION: can the metre arm FAIL the flyability gate? ===")
print("   (zero-init control_head, so this is the SAMPLER's noise alone)")
for space in ("control","metre"):
    d = mk(space)
    with torch.no_grad():
        torch.manual_seed(7); o = d(fm,m,steps=2,v_ms=v)
    a = implied_alat(o["anchor_traj"])
    print(f"  {space:8s}: implied |a_lat| mean {float(a.mean()):7.4f}  "
          f"max {float(a.max()):8.4f} m/s^2  ({float(a.max())/9.81:6.2f} g)  "
          f"| frac over mu=0.7 ({0.7*9.81:.2f}): {float((a>0.7*9.81).float().mean()):.4f}")
    print(f"            tele: {o['sel_tele']['sampler_space']}")
print()
print("  DD's published per-waypoint sigma is 0.90 m / 0.73 m; over a 0.5 s slot a")
print("  0.73 m lateral excursion implies ~5.8 m/s^2 (0.6 g). The metre arm must")
print("  land in that neighbourhood, and the control arm must not.")
