import torch
from tanitad.refs import refc
from tanitad.refs import refc_sampler as rs

def mk(units, n=117, horizons=(5,10,15,20,25,30,40,60), seed=0):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="ddim")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(horizons), d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(n, len(horizons), 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False, grounded_selector=False,
        horizons=horizons, v0_conditioned=True, control_units=units)
    torch.manual_seed(seed+1)
    dec.anchor_controls.copy_(torch.stack(
        [torch.linspace(-4.0, 2.0, n), torch.linspace(-3.0, 3.0, n)], dim=-1))
    return dec, horizons

print("=== THE LOAD-BEARING IDENTITY: roll_controls(constant) == roll_bank ===")
for units in ("alat", "kappa"):
    dec, hz = mk(units)
    v0 = torch.tensor([0.0, 3.0, 12.0, 25.0, 36.0])   # incl. standstill + 36 m/s
    bank = dec.roll_bank(v0, None, 5, torch.float32)
    u = dec.anchor_control_seq(5, torch.float32)
    rolled = rs.roll_controls(u, v0, hz, control_units=units, tick=dec.anchor_dt,
                              alat_v_floor=dec.anchor_alat_v_floor,
                              kappa_cap=dec.anchor_kappa_cap)
    eq = torch.equal(rolled, bank)
    md = float((rolled - bank).abs().max())
    ulp = int((rolled.view(torch.int32) != bank.view(torch.int32)).sum())
    print(f"  units={units:5s} shape={tuple(bank.shape)} n_elem={bank.numel()} "
          f"BIT_IDENTICAL={eq} max|diff|={md:.3e} differing_bit_patterns={ulp}")
    assert eq

print()
print("=== the sampler MOVES its own output (refcv3 read 201/201 UNCHANGED) ===")
dec, hz = mk("alat", n=32)
dec.eval()
torch.manual_seed(3)
with torch.no_grad():
    fm, m, v = torch.randn(4,16,3,5), torch.randn(4,8), torch.tensor([5.,12.,20.,30.])
    a = dec(fm, m, steps=2, v_ms=v)
    # now open the denoiser by hand (as training would) and re-run
    torch.nn.init.normal_(dec.control_head.weight, std=0.02)
    torch.nn.init.normal_(dec.control_head.bias, std=0.02)
    torch.manual_seed(3)
    b = dec(fm, m, steps=2, v_ms=v)
rk_a = a["sel_score"].argsort(dim=1); rk_b = b["sel_score"].argsort(dim=1)
moved = int((rk_a != rk_b).any(dim=1).sum())
print(f"  windows whose RANKING moved when the denoiser opened: {moved}/4")
print(f"  max |fan delta| (m): {float((a['anchor_traj']-b['anchor_traj']).abs().max()):.4f}")
print(f"  u0_hat present: {'u0_hat' in a}  shape {tuple(a['u0_hat'].shape)}")
print(f"  sel_tele: {{k: a['sel_tele'][k] for k in ('sampler','sampler_space','sampler_ladder')}}")
print("  ->", {k: a['sel_tele'][k] for k in ('sampler','sampler_space','sampler_ladder')})

print()
print("=== diffusers cross-check available? ===")
try:
    import diffusers
    print("  diffusers", diffusers.__version__)
    print("  assert_matches_diffusers ->", rs.assert_matches_diffusers(rs.DDIMSchedule()))
except Exception as e:
    print("  NOT INSTALLED:", type(e).__name__, e)
