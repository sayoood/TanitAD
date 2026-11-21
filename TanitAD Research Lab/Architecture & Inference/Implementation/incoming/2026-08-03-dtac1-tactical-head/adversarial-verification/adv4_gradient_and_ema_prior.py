import json, math, torch
from tanitad.refs import refc, refc_tactical as tac
out={}
sm = refc.refc_smoke_config(); sm.factored_maneuver=True; sm.tactical_speed_input=True
def fresh():
    torch.manual_seed(5); return refc.RefCModel(sm)
fr=torch.randn(2,sm.window,1,64,64); v0=torch.rand(2)*10; nc=torch.tensor([0,1])

# G1: gradient through TRAJ only  (what the traj L1 loss gives)
g=fresh(); o=g(fr,v0=v0,nav_cmd=nc); o['traj'].sum().backward()
out['via_traj'] = {k:(None if v is None else round(float(v.abs().max()),8)) for k,v in
   [('lon_to_anchor',g.decoder.lon_to_anchor.weight.grad),
    ('lat_to_anchor',g.decoder.lat_to_anchor.weight.grad),
    ('lon_head',g.lon_head.weight.grad),('lat_head',g.lat_head.weight.grad),
    ('tactical_trunk',g.tactical_trunk[0].weight.grad)]}

# G2: gradient through ANCHOR_LOGITS (what the anchor CE gives)
g=fresh(); o=g(fr,v0=v0,nav_cmd=nc); o['anchor_logits'].sum().backward()
out['via_anchor_logits'] = {k:(None if v is None else round(float(v.abs().max()),8)) for k,v in
   [('lon_to_anchor',g.decoder.lon_to_anchor.weight.grad),
    ('lat_to_anchor',g.decoder.lat_to_anchor.weight.grad),
    ('lon_head',g.lon_head.weight.grad),('lat_head',g.lat_head.weight.grad),
    ('tactical_trunk',g.tactical_trunk[0].weight.grad)]}

# G3: gradient through the LAT/LON CE (the aux losses)
g=fresh(); o=g(fr,v0=v0,nav_cmd=nc)
(torch.nn.functional.cross_entropy(o['lat_logits'],torch.tensor([0,1]))
 +torch.nn.functional.cross_entropy(o['lon_logits'],torch.tensor([0,2]))).backward()
out['via_lat_lon_CE'] = {k:(None if v is None else round(float(v.abs().max()),8)) for k,v in
   [('lon_to_anchor',g.decoder.lon_to_anchor.weight.grad),
    ('lat_to_anchor',g.decoder.lat_to_anchor.weight.grad),
    ('lon_head',g.lon_head.weight.grad),('lat_head',g.lat_head.weight.grad)]}

# G4: does anything at all in the FULL model make traj depend on conf?
g=fresh(); o=g(fr,v0=v0,nav_cmd=nc)
out['traj_requires_grad']=bool(o['traj'].requires_grad)
out['anchor_logits_requires_grad']=bool(o['anchor_logits'].requires_grad)
out['sel_idx_dtype']=str(o['sel_idx'].dtype)

# G5: BASELINE — did the OLD maneuver_to_anchor behave the same via traj?
sm0 = refc.refc_smoke_config()
torch.manual_seed(5); g0=refc.RefCModel(sm0); o0=g0(fr,v0=v0,nav_cmd=nc)
o0['traj'].sum().backward()
out['BASELINE_via_traj_maneuver_to_anchor']= (None if g0.decoder.maneuver_to_anchor.weight.grad is None
    else round(float(g0.decoder.maneuver_to_anchor.weight.grad.abs().max()),8))
out['BASELINE_via_traj_maneuver_head']= (None if g0.maneuver_head[0].weight.grad is None
    else round(float(g0.maneuver_head[0].weight.grad.abs().max()),8))

# G6: EMA prior updater — does it actually move, stay normalised, and is it
#     an identity for logit_adjust when never called?
g=fresh()
before=g.lon_log_prior.clone()
idx=torch.tensor([0]*10+[1]*80+[2]*10)
g.update_tactical_prior(torch.zeros(100,dtype=torch.long), idx, momentum=0.0)
out['prior_after_mom0']=[round(float(x),5) for x in g.lon_log_prior.exp()]
out['prior_sums_to_1']=round(float(g.lon_log_prior.exp().sum()),6)
out['prior_moved']=round(float((g.lon_log_prior-before).abs().max()),5)
g2=fresh()
for _ in range(200): g2.update_tactical_prior(torch.zeros(100,dtype=torch.long), idx)
out['prior_after_200_ema_default_momentum']=[round(float(x),5) for x in g2.lon_log_prior.exp()]

# G7: man_prior_tau in the CONFIG -> does refc_factored_config's tau=1.0 change
#     maneuver_decision vs tau=0?
smA=refc.refc_smoke_config(); smA.factored_maneuver=True; smA.man_prior_tau=0.0
smB=refc.refc_smoke_config(); smB.factored_maneuver=True; smB.man_prior_tau=1.0
torch.manual_seed(9); mA=refc.RefCModel(smA).eval()
torch.manual_seed(9); mB=refc.RefCModel(smB).eval()
# give both a NON-uniform prior so tau can bite
pri=torch.tensor([0.1122,0.7104,0.1774]).log()
mA.lon_log_prior.copy_(pri); mB.lon_log_prior.copy_(pri)
f=torch.randn(64,sm.window,1,64,64)
with torch.no_grad():
    oa=mA(f,v0=torch.rand(64)*10,nav_cmd=torch.zeros(64,dtype=torch.long))
    ob=mB(f,v0=torch.rand(64)*10,nav_cmd=torch.zeros(64,dtype=torch.long))
out['tau0_lon_decision_hist']=[int((oa['lon_decision']==i).sum()) for i in range(3)]
out['tau1_lon_decision_hist']=[int((ob['lon_decision']==i).sum()) for i in range(3)]
out['tau_changes_decision']= bool((oa['lon_decision']!=ob['lon_decision']).any())
# and does tau leak into the SELECTION path (it must not: graft uses lat/lon_prior)
out['tau_changes_traj']= round(float((oa['traj']-ob['traj']).abs().max()),8)
out['tau_changes_anchor_logits']= round(float((oa['anchor_logits']-ob['anchor_logits']).abs().max()),8)
print(json.dumps(out,indent=1,default=str))
