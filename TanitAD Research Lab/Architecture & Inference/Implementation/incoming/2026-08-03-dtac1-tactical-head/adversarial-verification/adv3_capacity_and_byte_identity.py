"""ADVERSARIAL verification of the D-TAC1 CODE claims, run against the STAGED files."""
import json, math, copy, hashlib
import torch
from tanitad.refs import refc, refc_tactical as tac
out={}

# ---- C1. CAPACITY: +897 params, claimed by refc_factored_config docstring ----
base = refc.RefCModel(refc.refc_config())
fact = refc.RefCModel(refc.refc_factored_config())
nb = sum(p.numel() for p in base.parameters())
nf = sum(p.numel() for p in fact.parameters())
out['params_base']=nb; out['params_factored']=nf; out['delta']=nf-nb
out['delta_pct']=round(100*(nf-nb)/nb,6)
out['claim_base_104191577']= (nb==104191577)
out['claim_fact_104192474']= (nf==104192474)
out['claim_delta_897']= (nf-nb==897)
pb=refc.param_breakdown(base); pf=refc.param_breakdown(fact)
out['breakdown_base_aux']=pb['aux']; out['breakdown_fact_aux']=pf['aux']
out['breakdown_total_matches_sum_params']= (pb['total']==nb, pf['total']==nf)

# ---- C1b. do BUFFERS count? params vs state_dict ----------------------------
out['n_buffers_base']=sum(b.numel() for b in base.buffers())
out['n_buffers_fact']=sum(b.numel() for b in fact.buffers())

# ---- C2. FLAG OFF => byte-identical state_dict AND outputs -------------------
cfg_off = refc.refc_config()
cfg_off2 = refc.refc_config(); cfg_off2.factored_maneuver=False
torch.manual_seed(0); m1 = refc.RefCModel(cfg_off)
torch.manual_seed(0); m2 = refc.RefCModel(cfg_off2)
k1=sorted(m1.state_dict().keys()); k2=sorted(m2.state_dict().keys())
out['flagoff_same_keys']= (k1==k2)
out['flagoff_key_count']=len(k1)
out['flagoff_all_tensors_equal']= all(torch.equal(m1.state_dict()[k], m2.state_dict()[k]) for k in k1)
# and vs the FACTORED key set
kf=sorted(fact.state_dict().keys())
out['factored_extra_keys']=[k for k in kf if k not in k1]
out['factored_removed_keys']=[k for k in k1 if k not in kf]

# forward equality (deterministic eval)
sm = refc.refc_smoke_config()
torch.manual_seed(1); a = refc.RefCModel(sm).eval()
torch.manual_seed(1); b = refc.RefCModel(sm).eval()
W = sm.window; C,H = 1, 64
fr = torch.randn(3, W, C, H, H); v0=torch.rand(3)*10; nc=torch.tensor([0,1,2])
with torch.no_grad():
    o1=a(fr,v0=v0,nav_cmd=nc); o2=b(fr,v0=v0,nav_cmd=nc)
out['flagoff_forward_traj_maxdiff']=float((o1['traj']-o2['traj']).abs().max())
out['flagoff_forward_man_maxdiff']=float((o1['maneuver_logits']-o2['maneuver_logits']).abs().max())

# ---- C3. derive_man5_logprobs is a NORMALISED 5-way logprob + exact roundtrip
torch.manual_seed(3)
ll=torch.randn(500,3)*3; lo=torch.randn(500,3)*3
m5 = tac.derive_man5_logprobs(ll,lo)
out['derive_sums_to_1_maxerr']=float((m5.exp().sum(-1)-1).abs().max())
il,io = tac.invert_man5(m5)
out['invert_lat_maxerr']=float((il.exp()-torch.softmax(ll,-1)).abs().max())
out['invert_lon_maxerr']=float((io.exp()-torch.softmax(lo,-1)).abs().max())
m5b = tac.derive_man5_logprobs(il,io)
out['roundtrip_m5_maxerr']=float((m5b-m5).abs().max())
# does derive match the ALGEBRA the docstring states?
pl=torch.softmax(ll,-1); pn=torch.softmax(lo,-1); p5=m5.exp()
ref=torch.stack([pl[:,0]*pn[:,1], pl[:,1], pl[:,2], pl[:,0]*pn[:,2], pl[:,0]*pn[:,0]],1)
out['derive_matches_stated_algebra_maxerr']=float((p5-ref).abs().max())

# ---- C4. logit_adjust with UNIFORM prior is the identity at any tau ----------
u=torch.full((3,), -math.log(3))
out['logit_adjust_uniform_is_identity']= all(
    float((tac.logit_adjust(lo,u,t).argmax(-1)!=lo.argmax(-1)).sum())==0
    for t in [0.0,0.5,1.0,2.0])

# ---- C5. lon_to_anchor ZERO-INIT and it RECEIVES GRADIENT -------------------
f2 = refc.RefCModel(refc.refc_factored_config())
out['lon_to_anchor_zero_at_init']=float(f2.decoder.lon_to_anchor.weight.abs().max())
out['lat_to_anchor_nonzero_at_init']=float(f2.decoder.lat_to_anchor.weight.abs().max())
out['maneuver_to_anchor_is_None_when_factored']= (f2.decoder.maneuver_to_anchor is None)
smf = refc.refc_smoke_config(); smf.factored_maneuver=True; smf.tactical_speed_input=True
g = refc.RefCModel(smf)
o = g(torch.randn(2,smf.window,1,64,64), v0=torch.rand(2)*10, nav_cmd=torch.tensor([0,1]))
o['traj'].sum().backward()
gl = g.decoder.lon_to_anchor.weight.grad
out['lon_to_anchor_grad_is_nonzero']= bool(gl is not None and float(gl.abs().max())>0)
out['lon_to_anchor_grad_absmax']=float(gl.abs().max()) if gl is not None else None
out['lat_head_grad_nonzero']=bool(g.lat_head.weight.grad is not None and float(g.lat_head.weight.grad.abs().max())>0)
out['lon_head_grad_nonzero']=bool(g.lon_head.weight.grad is not None and float(g.lon_head.weight.grad.abs().max())>0)

# ---- C6. does the SPEED input actually change the tactical logits? -----------
g2 = refc.RefCModel(smf).eval()
fr2=torch.randn(4,smf.window,1,64,64)
with torch.no_grad():
    oa=g2(fr2, v0=torch.zeros(4), nav_cmd=torch.zeros(4,dtype=torch.long))
    ob=g2(fr2, v0=torch.full((4,),30.0), nav_cmd=torch.zeros(4,dtype=torch.long))
out['speed_input_moves_lon_logits']=float((oa['lon_logits']-ob['lon_logits']).abs().max())
smn = refc.refc_smoke_config(); smn.factored_maneuver=True; smn.tactical_speed_input=False
g3 = refc.RefCModel(smn).eval()
with torch.no_grad():
    oc=g3(fr2, v0=torch.zeros(4), nav_cmd=torch.zeros(4,dtype=torch.long))
    od=g3(fr2, v0=torch.full((4,),30.0), nav_cmd=torch.zeros(4,dtype=torch.long))
out['NEGCTRL_no_speed_input_lon_logit_delta']=float((oc['lon_logits']-od['lon_logits']).abs().max())

# ---- C7. maneuver_decision RE-APPLIES the collapse: can it emit a lon class
#          on a window where lat says turn?  (the defect, still present?)
lat_d=torch.tensor([1,2,0,0]); lon_d=torch.tensor([0,2,0,2])
coll=tac.collapse(lat_d,lon_d)
out['collapse_turn_plus_brake_gives']=int(coll[0]); out['collapse_turn_plus_accel_gives']=int(coll[1])
out['maneuver_decision_still_destroys_lon_on_turns']= (int(coll[0])==refc_TL if False else int(coll[0])==1 and int(coll[1])==2)

# ---- C8. trainer weights: 0.05+0.05 == MANEUVER_WEIGHT 0.10 -----------------
import importlib.util, sys
spec=importlib.util.spec_from_file_location("refc_train","/tmp/advstack/scripts/refc_train.py")
mt=importlib.util.module_from_spec(spec); sys.modules['refc_train']=mt
try:
    spec.loader.exec_module(mt)
    out['MANEUVER_WEIGHT']=getattr(mt,'MANEUVER_WEIGHT',None)
    for nm in ['LAT_WEIGHT','LON_WEIGHT','MAN_LAT_WEIGHT','MAN_LON_WEIGHT']:
        if hasattr(mt,nm): out[nm]=getattr(mt,nm)
except Exception as e:
    out['trainer_import_error']=repr(e)[:300]
print(json.dumps(out,indent=1,default=str))
