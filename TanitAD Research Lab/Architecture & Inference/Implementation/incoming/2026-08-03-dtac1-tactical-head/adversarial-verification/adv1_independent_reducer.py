# ADVERSARIAL independent re-reduction of the D-TAC1 substrate.
# Deliberately does NOT import refc_tactical -- the algebra is re-implemented
# from the collapse map stated in the report, so a bug in the stream's module
# cannot propagate into this check.
import json, collections
import torch

D = torch.load('/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt',
               map_location='cpu', weights_only=False)
log5, lat, lon, man5, man_banked = D['log5'], D['lat'], D['lon'], D['man5'], D['man_banked']
eid = D['eid']; v0 = D['v0']; pooled = D['pooled']
n = log5.shape[0]
out = {}

# ---- 0. integrity -------------------------------------------------------
out['n'] = n
out['n_unique_eid'] = len(set(int(e) for e in eid))
cnt = collections.Counter(int(e) for e in eid)
out['eid_min_windows'] = min(cnt.values()); out['eid_max_windows'] = max(cnt.values())
out['eid_decoded_sample'] = [int(e).to_bytes(4,'little').decode('latin1') for e in list(dict.fromkeys(int(x) for x in eid))[:5]]
out['log5_finite'] = bool(torch.isfinite(log5).all())
out['man5_eq_man_banked'] = float((man5 == man_banked).float().mean())
out['n_man5_ne_banked'] = int((man5 != man_banked).sum())

# ---- 1. 5-way argmax confusion, my own tally ----------------------------
LK,TL,TR,AC,BS = 0,1,2,3,4
pred5 = log5.argmax(1)
C = torch.zeros(5,5, dtype=torch.long)
for t,p in zip(man5.tolist(), pred5.tolist()): C[t,p]+=1
out['conf5'] = C.tolist()
out['conf5_rowsum'] = C.sum(1).tolist(); out['conf5_colsum'] = C.sum(0).tolist()
out['acc5'] = round(float(C.diag().sum())/n, 4)
rec5 = [ (float(C[i,i])/float(C[i].sum()) if C[i].sum()>0 else float('nan')) for i in range(5)]
out['recall5'] = [round(r,4) for r in rec5]
out['macro_recall5'] = round(sum(rec5)/5, 4)
# where does the longitudinal mass land?
lon_true = C[AC].sum()+C[BS].sum()
out['lon_true_n'] = int(lon_true)
out['lon_true_to_lane_keep'] = int(C[AC,LK]+C[BS,LK])
out['lon_true_to_a_LATERAL_class'] = int(C[AC,TL]+C[AC,TR]+C[BS,TL]+C[BS,TR])
out['lon_true_to_lane_keep_frac'] = round(float(C[AC,LK]+C[BS,LK])/float(lon_true),4)

# ---- 2. exact inversion of the 5-way softmax into lat x lon -------------
lp = torch.log_softmax(log5.double(), dim=-1)
p = lp.exp()
p_lat = torch.stack([p[:,LK]+p[:,AC]+p[:,BS], p[:,TL], p[:,TR]], dim=1)   # keep,left,right
den = p_lat[:,0].clamp_min(1e-30)
p_lon = torch.stack([p[:,BS]/den, p[:,LK]/den, p[:,AC]/den], dim=1)       # brake,steady,accel
out['inv_plat_sums_to_1_maxerr'] = float((p_lat.sum(1)-1).abs().max())
out['inv_plon_sums_to_1_maxerr'] = float((p_lon.sum(1)-1).abs().max())
# round trip: reconstruct P5 from the factorisation
rec = torch.stack([p_lat[:,0]*p_lon[:,1], p_lat[:,1], p_lat[:,2],
                   p_lat[:,0]*p_lon[:,2], p_lat[:,0]*p_lon[:,0]], dim=1)
out['roundtrip_maxabs_err'] = float((rec-p).abs().max())

# ---- 3. lateral readout (marginalised), my tally ------------------------
pred_lat = p_lat.argmax(1)
CL = torch.zeros(3,3,dtype=torch.long)
for t,q in zip(lat.tolist(), pred_lat.tolist()): CL[t,q]+=1
out['conf_lat'] = CL.tolist()
out['acc_lat'] = round(float(CL.diag().sum())/n,4)
rl = [float(CL[i,i])/float(CL[i].sum()) for i in range(3)]
out['recall_lat'] = [round(x,4) for x in rl]; out['macro_recall_lat'] = round(sum(rl)/3,4)
out['lat_n_pred'] = CL.sum(0).tolist(); out['lat_n_true'] = CL.sum(1).tolist()

# ---- 4. factored-raw longitudinal decode (tau=0) ------------------------
def lon_conf(pred):
    M = torch.zeros(3,3,dtype=torch.long)
    for t,q in zip(lon.tolist(), pred.tolist()): M[t,q]+=1
    return M
M0 = lon_conf(p_lon.argmax(1))
out['conf_lon_tau0'] = M0.tolist()
r0=[float(M0[i,i])/float(M0[i].sum()) for i in range(3)]
out['recall_lon_tau0']=[round(x,4) for x in r0]; out['macro_recall_lon_tau0']=round(sum(r0)/3,4)
out['acc_lon_tau0']=round(float(M0.diag().sum())/n,4)
out['lon_n_true']=M0.sum(1).tolist(); out['lon_n_pred_tau0']=M0.sum(0).tolist()

# ---- 5. tau frontier, prior = EMPIRICAL VAL PRIOR of lon ----------------
prior = torch.tensor([float((lon==i).sum())/n for i in range(3)], dtype=torch.double)
out['lon_prior_used'] = [round(float(x),4) for x in prior]
frontier=[]
for tau in [0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0]:
    adj = p_lon.clamp_min(1e-30).log() - tau*prior.log()[None,:]
    M = lon_conf(adj.argmax(1))
    r=[float(M[i,i])/float(M[i].sum()) for i in range(3)]
    frontier.append({'tau':tau,'acc':round(float(M.diag().sum())/n,4),
                     'macroR':round(sum(r)/3,4),
                     'rec':{'brake':round(r[0],4),'steady':round(r[1],4),'accel':round(r[2],4)},
                     'npred':M.sum(0).tolist()})
out['tau_frontier_mine'] = frontier

# ---- 6. AUC (rank-based, own implementation) ----------------------------
def auc(score, pos):
    s=score.double(); y=pos.double()
    order=torch.argsort(s); r=torch.empty_like(s); r[order]=torch.arange(len(s),dtype=torch.double)+1
    # average ranks for ties
    su,inv=torch.unique(s,return_inverse=True)
    for u in range(len(su)):
        m=inv==u
        if m.sum()>1: r[m]=r[m].mean()
    npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return float('nan')
    return float((r[y>0].sum()-npos*(npos+1)/2)/(npos*nneg))
out['auc_brake'] = round(auc(p_lon[:,0], (lon==0)),4)
out['auc_steady'] = round(auc(p_lon[:,1], (lon==1)),4)
out['auc_accel'] = round(auc(p_lon[:,2], (lon==2)),4)
out['auc_active_vs_steady'] = round(auc(1-p_lon[:,1], (lon!=1)),4)

# ---- 7. label-side collapse: longitudinal live AND 5-way says a turn ----
destroyed = ((lon!=1) & ((man5==TL)|(man5==TR)))
out['n_destroyed_by_priority'] = int(destroyed.sum())
out['frac_destroyed'] = round(float(destroyed.float().mean()),4)
# and the counterpart: does dropping them change the tau=0 story?
keep = ~destroyed
M0k = torch.zeros(3,3,dtype=torch.long)
for t,q in zip(lon[keep].tolist(), p_lon[keep].argmax(1).tolist()): M0k[t,q]+=1
rk=[(float(M0k[i,i])/float(M0k[i].sum()) if M0k[i].sum()>0 else float('nan')) for i in range(3)]
out['macro_recall_lon_tau0_EXCLUDING_destroyed']=round(sum(rk)/3,4)
out['recall_lon_tau0_EXCLUDING_destroyed']=[round(x,4) for x in rk]
out['n_after_excluding_destroyed']=int(keep.sum())

# ---- 8. NEGATIVE CONTROL: shuffle logits across windows -----------------
g=torch.Generator().manual_seed(1234)
perm=torch.randperm(n,generator=g)
ps=p_lon[perm]
Ms=lon_conf(ps.argmax(1))
rs=[float(Ms[i,i])/float(Ms[i].sum()) for i in range(3)]
out['SHUF_macro_recall_lon_tau0']=round(sum(rs)/3,4)
out['SHUF_auc_active']=round(auc(1-ps[:,1],(lon!=1)),4)
C5s=torch.zeros(5,5,dtype=torch.long)
for t,q in zip(man5.tolist(), pred5[perm].tolist()): C5s[t,q]+=1
out['SHUF_acc5']=round(float(C5s.diag().sum())/n,4)

print(json.dumps(out, indent=1))
