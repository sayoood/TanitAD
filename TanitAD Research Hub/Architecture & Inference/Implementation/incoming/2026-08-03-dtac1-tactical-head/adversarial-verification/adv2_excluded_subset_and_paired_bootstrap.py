import json, torch
D = torch.load('/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt',
               map_location='cpu', weights_only=False)
log5,lat,lon,man5 = D['log5'],D['lat'],D['lon'],D['man5']; n=len(man5)
LK,TL,TR,AC,BS=0,1,2,3,4
out={}
# A. self-consistency: does collapse(lat,lon) reproduce man5 for EVERY window?
coll = torch.full((n,), -1, dtype=torch.long)
coll[lat==1]=TL; coll[lat==2]=TR
keepm = lat==0
coll[keepm & (lon==1)] = LK
coll[keepm & (lon==2)] = AC
coll[keepm & (lon==0)] = BS
out['collapse_eq_man5'] = float((coll==man5).float().mean())
out['n_collapse_mismatch'] = int((coll!=man5).sum())

p = torch.log_softmax(log5.double(),-1).exp()
p_lat = torch.stack([p[:,LK]+p[:,AC]+p[:,BS], p[:,TL], p[:,TR]],1)
p_lon = torch.stack([p[:,BS]/p_lat[:,0].clamp_min(1e-30), p[:,LK]/p_lat[:,0].clamp_min(1e-30),
                     p[:,AC]/p_lat[:,0].clamp_min(1e-30)],1)
prior = torch.tensor([float((lon==i).sum())/n for i in range(3)],dtype=torch.double)

destroyed = (lon!=1) & ((man5==TL)|(man5==TR))
keep = ~destroyed

def sweep(mask,tag):
    rows=[]
    ln=lon[mask]; pl=p_lon[mask]; N=int(mask.sum())
    for tau in [0.0,0.25,0.5,0.75,1.0]:
        pr = pl.clamp_min(1e-30).log() - tau*prior.log()[None,:]
        pd = pr.argmax(1)
        M=torch.zeros(3,3,dtype=torch.long)
        for t,q in zip(ln.tolist(),pd.tolist()): M[t,q]+=1
        r=[(float(M[i,i])/float(M[i].sum()) if M[i].sum()>0 else float('nan')) for i in range(3)]
        rows.append({'tau':tau,'n':N,'acc':round(float(M.diag().sum())/N,4),
                     'macroR':round(sum(r)/3,4),
                     'rec_brake':round(r[0],4),'rec_steady':round(r[1],4),'rec_accel':round(r[2],4),
                     'npred':M.sum(0).tolist(),'ntrue':M.sum(1).tolist()})
    out[tag]=rows
sweep(torch.ones(n,dtype=torch.bool),'sweep_ALL')
sweep(keep,'sweep_EXCLUDING_the_132_label_destroyed')

# B. is the recovered accelerate concentrated on destroyed windows?
pd0 = p_lon.argmax(1)
out['tau0_correct_accel_total'] = int(((pd0==2)&(lon==2)).sum())
out['tau0_correct_accel_on_DESTROYED'] = int(((pd0==2)&(lon==2)&destroyed).sum())
out['tau0_correct_brake_total'] = int(((pd0==0)&(lon==0)).sum())
out['tau0_correct_brake_on_DESTROYED'] = int(((pd0==0)&(lon==0)&destroyed).sum())

# C. the prior used by the adjustment IS the val label marginal -> quantify the leak.
#    re-run tau=0.5 with a TRAIN-LIKE prior perturbed +/-25% and see the swing.
for name,pv in [('val_marginal',prior),
                ('uniform',torch.tensor([1/3,1/3,1/3],dtype=torch.double)),
                ('brake_x0.75',prior*torch.tensor([0.75,1.0,1.0],dtype=torch.double)),
                ('brake_x1.25',prior*torch.tensor([1.25,1.0,1.0],dtype=torch.double))]:
    pv=pv/pv.sum()
    pr = p_lon.clamp_min(1e-30).log() - 0.5*pv.log()[None,:]
    pd=pr.argmax(1); M=torch.zeros(3,3,dtype=torch.long)
    for t,q in zip(lon.tolist(),pd.tolist()): M[t,q]+=1
    r=[float(M[i,i])/float(M[i].sum()) for i in range(3)]
    out.setdefault('prior_sensitivity_tau0.5',{})[name]={
        'macroR':round(sum(r)/3,4),'rec_brake':round(r[0],4),'rec_accel':round(r[2],4),
        'acc':round(float(M.diag().sum())/n,4)}

# D. episode-cluster bootstrap, PAIRED, on tactical-correctness raw vs adjusted(0.5)
eid=torch.tensor([int(e) for e in D['eid']])
ue=torch.unique(eid)
pdr = p_lon.argmax(1)
pra = (p_lon.clamp_min(1e-30).log()-0.5*prior.log()[None,:]).argmax(1)
corr_r=(pdr==lon).double(); corr_a=(pra==lon).double()
diff = corr_a-corr_r
g=torch.Generator().manual_seed(7)
per_ep_d=torch.stack([diff[eid==e].mean() for e in ue])
per_ep_r=torch.stack([corr_r[eid==e].mean() for e in ue])
per_ep_a=torch.stack([corr_a[eid==e].mean() for e in ue])
B=4000; ne=len(ue)
bd=[];br=[];ba=[]
for _ in range(B):
    idx=torch.randint(0,ne,(ne,),generator=g)
    bd.append(per_ep_d[idx].mean()); br.append(per_ep_r[idx].mean()); ba.append(per_ep_a[idx].mean())
bd=torch.stack(bd);br=torch.stack(br);ba=torch.stack(ba)
q=lambda t:(round(float(torch.quantile(t,0.025)),4),round(float(torch.quantile(t,0.975)),4))
out['PAIRED_bootstrap_lon_accuracy'] = {
  'raw_mean':round(float(per_ep_r.mean()),4),'raw_ci':q(br),
  'adj0.5_mean':round(float(per_ep_a.mean()),4),'adj0.5_ci':q(ba),
  'PAIRED_delta_mean':round(float(per_ep_d.mean()),4),'PAIRED_delta_ci':q(bd),
  'n_episodes':ne,'n_boot':B}
# same, on MACRO-RECALL (the metric actually quoted) -- paired
def macro(mask_pred,mask_idx):
    r=[]
    for c in range(3):
        m=(lon[mask_idx]==c)
        r.append(float(((mask_pred[mask_idx]==c)&m).sum())/max(1,int(m.sum())) if m.sum()>0 else float('nan'))
    return r
bm=[]
for _ in range(B):
    idx=torch.randint(0,ne,(ne,),generator=g)
    sel=torch.cat([torch.nonzero(eid==ue[i],as_tuple=True)[0] for i in idx])
    rr=macro(pdr,sel); ra=macro(pra,sel)
    import math
    if any(math.isnan(x) for x in rr+ra): continue
    bm.append(sum(ra)/3-sum(rr)/3)
bm=torch.tensor(bm)
out['PAIRED_bootstrap_lon_MACRO_RECALL_delta']={
  'mean':round(float(bm.mean()),4),'ci':q(bm),'n_boot_used':len(bm)}
print(json.dumps(out,indent=1))
