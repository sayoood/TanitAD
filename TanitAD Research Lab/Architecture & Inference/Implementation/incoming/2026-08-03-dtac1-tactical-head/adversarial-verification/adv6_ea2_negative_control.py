"""E-A2 adversarial re-run: does the linear probe DISCRIMINATE, and is the
pooled-across-folds AUC an artifact?  Adds the negative control the probe omits."""
import json, numpy as np, torch
D=torch.load('/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt',
             map_location='cpu', weights_only=False)
pooled=D['pooled'].numpy(); v0=D['v0'].numpy()[:,None]; lon=D['lon'].numpy()
eid=np.array([int(e) for e in D['eid']]); n=len(lon)
out={}

def fit(X,y,ncls,epochs=400,l2=1e-3,seed=0):
    torch.manual_seed(seed)
    X=torch.as_tensor(X,dtype=torch.float32); y=torch.as_tensor(y,dtype=torch.long)
    mu,sd=X.mean(0,keepdim=True),X.std(0,keepdim=True).clamp_min(1e-6)
    Xs=(X-mu)/sd
    W=torch.zeros(Xs.shape[1],ncls,requires_grad=True); b=torch.zeros(ncls,requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=0.05); losses=[]
    for i in range(epochs):
        opt.zero_grad()
        L=torch.nn.functional.cross_entropy(Xs@W+b,y)+l2*(W*W).sum()
        L.backward(); opt.step()
        if i in (0,epochs//2,epochs-1): losses.append(round(float(L),5))
    return (W.detach(),b.detach(),mu,sd), losses

def auc(score,pos):
    s=np.asarray(score,dtype=float); y=np.asarray(pos,dtype=bool)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    u,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    if (cnt>1).any():
        sums=np.zeros(len(u)); np.add.at(sums,inv,r); r=(sums/cnt)[inv]
    npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return float('nan')
    return float((r[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def run(X,y,tag,folds=2,seed=0):
    uniq=sorted(set(eid.tolist())); assign={e:i%folds for i,e in enumerate(uniq)}
    fo=np.array([assign[e] for e in eid])
    prob=np.zeros((n,3)); pred=np.zeros(n,dtype=int); per_fold_auc=[]; loss_trace=[]
    for f in range(folds):
        tr,te=fo!=f,fo==f
        F,ls=fit(X[tr],y[tr],3,seed=seed); loss_trace.append(ls)
        Xs=(torch.as_tensor(X[te],dtype=torch.float32)-F[2])/F[3]
        lg=Xs@F[0]+F[1]; p=torch.softmax(lg,-1).numpy()
        prob[te]=p; pred[te]=p.argmax(1)
        per_fold_auc.append([auc(p[:,k], y[te]==k) for k in range(3)])
    rec=[float(((pred==c)&(y==c)).sum())/max(1,int((y==c).sum())) for c in range(3)]
    res={'macro_recall':round(sum(rec)/3,4),'recall':[round(x,4) for x in rec],
         'AUC_pooled_across_folds':[round(auc(prob[:,k],y==k),4) for k in range(3)],
         'AUC_averaged_per_fold':[round(float(np.mean([pf[k] for pf in per_fold_auc])),4) for k in range(3)],
         'per_fold_auc':[[round(x,4) for x in pf] for pf in per_fold_auc],
         'train_loss_first_mid_last':loss_trace,'n_features':X.shape[1]}
    out[tag]=res

run(pooled,lon,'pooled_only')
run(v0,lon,'v0_only')
run(np.concatenate([pooled,v0],1),lon,'pooled_plus_v0')

# ---- THE MISSING NEGATIVE CONTROL: shuffled labels (episode-preserving shuffle
#      would leak; use a full permutation, which is the honest null for "does
#      this feature set predict the label at all")
rng=np.random.default_rng(0)
run(pooled,lon[rng.permutation(n)],'NEGCTRL_pooled_SHUFFLED_LABELS')
run(np.concatenate([pooled,v0],1),lon[rng.permutation(n)],'NEGCTRL_pooled_plus_v0_SHUFFLED_LABELS')
# seed sensitivity of the +0.051 headline
deltas=[]
for s in range(5):
    run(pooled,lon,f'_s{s}_p',seed=s); run(np.concatenate([pooled,v0],1),lon,f'_s{s}_pv',seed=s)
    deltas.append(round(out[f'_s{s}_pv']['macro_recall']-out[f'_s{s}_p']['macro_recall'],4))
    del out[f'_s{s}_p'], out[f'_s{s}_pv']
out['SEED_SENSITIVITY_of_the_+0.051_delta']=deltas
print(json.dumps(out,indent=1))
