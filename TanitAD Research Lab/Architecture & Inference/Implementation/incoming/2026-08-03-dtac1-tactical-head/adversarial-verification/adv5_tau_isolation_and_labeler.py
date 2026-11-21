"""Clean re-test of the tau isolation claim + the labeler-guard + loss scale."""
import json, math, torch, torch.nn.functional as F
from tanitad.refs import refc, refc_tactical as tac
out={}
sm=lambda tau:(lambda c:(setattr(c,'factored_maneuver',True),
                         setattr(c,'man_prior_tau',tau), c)[-1])(refc.refc_smoke_config())
A,B = sm(0.0), sm(1.0)
torch.manual_seed(9); mA=refc.RefCModel(A).eval()
torch.manual_seed(9); mB=refc.RefCModel(B).eval()
# identical weights?
out['weights_identical']= all(torch.equal(a,b) for a,b in
    zip(mA.state_dict().values(), mB.state_dict().values()))
pri=torch.tensor([0.1122,0.7104,0.1774]).log()
with torch.no_grad():
    mA.lon_log_prior.copy_(pri); mB.lon_log_prior.copy_(pri)
# ONE fixed input for BOTH (the previous test drew v0 twice -> confounded)
torch.manual_seed(11)
f=torch.randn(96,A.window,1,64,64); v=torch.rand(96)*10; nc=torch.zeros(96,dtype=torch.long)
with torch.no_grad():
    oa=mA(f,v0=v,nav_cmd=nc); ob=mB(f,v0=v,nav_cmd=nc)
out['tau_delta_traj']=float((oa['traj']-ob['traj']).abs().max())
out['tau_delta_anchor_logits']=float((oa['anchor_logits']-ob['anchor_logits']).abs().max())
out['tau_delta_maneuver_logits']=float((oa['maneuver_logits']-ob['maneuver_logits']).abs().max())
out['tau_delta_lon_logits']=float((oa['lon_logits']-ob['lon_logits']).abs().max())
out['tau0_lon_decision_hist']=[int((oa['lon_decision']==i).sum()) for i in range(3)]
out['tau1_lon_decision_hist']=[int((ob['lon_decision']==i).sum()) for i in range(3)]
out['tau_changes_lon_decision_n']=int((oa['lon_decision']!=ob['lon_decision']).sum())

# does tau bite when the raw posterior is NOT degenerate? force a spread
torch.manual_seed(13); L=torch.randn(2000,3)*1.2
d0=tac.logit_adjust(L,pri,0.0).argmax(-1); d1=tac.logit_adjust(L,pri,1.0).argmax(-1)
out['synthetic_tau0_hist']=[int((d0==i).sum()) for i in range(3)]
out['synthetic_tau1_hist']=[int((d1==i).sum()) for i in range(3)]

# ---- LOSS-SCALE check: is "aux pressure held at 0.10" true in GRADIENT terms?
n=4096
lat_t=torch.randint(0,3,(n,)); lon_t=torch.randint(0,3,(n,))
man5_t=tac.collapse(lat_t,lon_t)
z5=torch.zeros(n,5); z3a=torch.zeros(n,3); z3b=torch.zeros(n,3)
ce5=float(F.cross_entropy(z5,man5_t))
ce3=0.5*float(F.cross_entropy(z3a,lat_t))+0.5*float(F.cross_entropy(z3b,lon_t))
out['CE_at_uniform_5way']=round(ce5,4); out['CE_at_uniform_factored_halfsum']=round(ce3,4)
out['effective_aux_loss_ratio_factored_over_5way']=round(ce3/ce5,4)
out['log5']=round(math.log(5),4); out['log3']=round(math.log(3),4)

# ---- the labeler guard: does window_factored_labels collapse to v1 or v2? ---
import importlib.util,sys
sys.path.insert(0,'G:\\Meine Ablage\\SayBouBase\\raw\\Projects\\TanitAD\\stack\\scripts')
import refb_labels
torch.manual_seed(17)
P=torch.randn(3000,4); P[:,3]=P[:,3].abs()*12
FUT=torch.randn(3000,20,4); FUT[:,:,3]=FUT[:,:,3].abs()*12
H=20
lat_g,lon_g = tac.window_factored_labels(P,FUT,horizon=H)
coll=tac.collapse(lat_g,lon_g)
v1=refb_labels.window_maneuver_labels(P,FUT,horizon=H)
v2=refb_labels.window_maneuver_labels_v2(P,FUT,horizon=H)
out['factored_collapse_eq_v1']=float((coll==v1).float().mean())
out['factored_collapse_eq_v2']=float((coll==v2).float().mean())
out['v1_eq_v2']=float((v1==v2).float().mean())
print(json.dumps(out,indent=1,default=str))
