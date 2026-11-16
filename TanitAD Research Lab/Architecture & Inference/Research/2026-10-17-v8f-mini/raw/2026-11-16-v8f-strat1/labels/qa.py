import numpy as np, pandas as pd, json
import build_strat_labels as B
st=np.load('F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz')
V=B.VARS
tr=[l.strip() for l in open('F:/TanitAD-scratch/datascale/navtok/raw/train4572.txt')]
ev=[l.strip() for l in open('F:/TanitAD-scratch/datascale/navtok/raw/clips141.txt')]
Q={}
vocab={v:list(st['vocab_'+v]) for v in V}
def counts(s,v):
    ids=st[f'{s}_{v}_id'];ok=st[f'{s}_{v}_valid']
    c=np.bincount(ids[ok],minlength=len(vocab[v]));return {vocab[v][i]:int(c[i]) for i in range(len(c))}
Q['class_counts_steps']={s:{v:counts(s,v) for v in V} for s in ('train','eval')}
base={}
for v in V:
    ce=Q['class_counts_steps']['eval'][v];ct=Q['class_counts_steps']['train'][v]
    n=sum(ce.values());m=max(ce,key=ce.get);mt=max(ct,key=ct.get)
    base[v]={'eval_majority':m,'acc_eval_majority':ce[m]/n,'train_majority':mt,'acc_eval_when_predicting_train_majority':ce[mt]/n,'n_eval_valid_steps':n,'n_classes':len(ce),'uniform_chance':1/len(ce)}
Q['constant_baseline_eval']=base
cov={}
for s,cl in (('train',tr),('eval',ev)):
    off=st[f'{s}_off'];n=st[f'{s}_n'];fl=st[f'{s}_flag']
    per=np.array([(fl[o:o+k]==0).mean() for o,k in zip(off,n)])
    inrel=np.array([(fl[o:o+k]!=1).any() for o,k in zip(off,n)])
    cov[s]={'clips':len(cl),'clips_in_release':int(inrel.sum()),'clips_absent':int((~inrel).sum()),
      'steps':int(n.sum()),'steps_ok':int((fl==0).sum()),'steps_clip_absent':int((fl==1).sum()),'steps_out_of_tol':int((fl==2).sum()),
      'step_coverage_all':float((fl==0).mean()),'step_coverage_clips_in_release':float((fl[fl!=1]==0).mean()),
      'per_episode_coverage_in_release_pct_quantiles_min_5_50_max':[float(x) for x in np.percentile(per[inrel],[0,5,50,100])],
      'episodes_with_coverage_lt_1_in_release':int((per[inrel]<1).sum())}
Q['coverage']=cov
# sensitivity to a 1-frame (0.2 s) shift of the time base
J=dict(np.load(B.JOIN_DIR+'join_eval141.npz'));lab,vocab_=B.load_release()
T0=B.build(J,lab,{});sens={}
for sh in (0.2,-0.2,0.4):
    T1=B.build(J,lab,{},t_shift=sh);both=(T0['flag']==0)&(T1['flag']==0)
    sens[str(sh)]={v:float((T0[v+'_id'][both]!=T1[v+'_id'][both]).mean()) for v in V}
Q['eval_label_change_fraction_under_time_shift_s']=sens
# metadata
md=pd.read_parquet('F:/TanitAD-scratch/strat1/labels/meta_dl/metadata/data_collection.parquet')
md=md[~md.index.duplicated()] if md.index.name else md
if 'clip_id' in md.columns: md=md.set_index('clip_id')
def cramers(a,b):
    ct=pd.crosstab(a,b).to_numpy().astype(float);n=ct.sum()
    e=ct.sum(1,keepdims=True)*ct.sum(0,keepdims=True)/n;chi=((ct-e)**2/e).sum()
    phi=chi/n;r,k=ct.shape
    v=np.sqrt(phi/(min(r,k)-1)) 
    phic=max(0,phi-(k-1)*(r-1)/(n-1));rc=r-(r-1)**2/(n-1);kc=k-(k-1)**2/(n-1)
    return float(v),float(np.sqrt(phic/(min(rc,kc)-1))),int(n),r,k
res={}
for name,cl in (('train4572',tr),('eval141',ev),('train+eval',tr+ev)):
    m=md.reindex(cl);m=m.dropna(subset=['country','platform_class'])
    v,vc,n,r,k=cramers(m.country,m.platform_class);res[name]={'cramers_v':v,'cramers_v_bias_corrected':vc,'n_clips':n,'n_countries':r,'n_platform_classes':k}
m=md.reindex(tr+ev).dropna(subset=['country','platform_class'])
res['platform_class_counts_ours']=m.platform_class.value_counts().to_dict()
res['platform_class_counts_full_metadata']=md.platform_class.value_counts().to_dict()
ct=pd.crosstab(m.country,m.platform_class);res['crosstab_ours']=ct.to_dict('index')
# countries with a single platform class
res['countries_single_platform_class']=[c for c in ct.index if (ct.loc[c]>0).sum()==1]
# country in release vs metadata agree
Q['country_x_platform_class']=res
# eval-only countries
cs=md.reindex(tr).country.value_counts();ce=md.reindex(ev).country.value_counts()
Q['eval_country_clips']=ce.to_dict();Q['train_country_clips']=cs.to_dict()
Q['countries_only_in_eval']=[c for c in ce.index if cs.get(c,0)==0]
Q['countries_in_train_not_eval']=[c for c in cs.index if ce.get(c,0)==0]
Q['eval_countries_with_train_clips_lt_20']={c:int(cs.get(c,0)) for c in ce.index if cs.get(c,0)<20}
# per-clip modes: clip-level class counts (episodes), eval
cl=pd.read_parquet(B.REL+'clips/clips.parquet').set_index('clip_id')
Q['eval_clip_level_mode_counts']={f:cl.reindex(ev)[f].value_counts().to_dict() for f in ['weather_mode','surface_mode','visibility_mode','illumination_mode','period_of_day']}
Q['train_clip_level_mode_counts']={f:cl.reindex(tr).dropna(subset=[f])[f].value_counts().to_dict() for f in ['weather_mode','surface_mode','visibility_mode','illumination_mode','period_of_day']}
Q['country_agrees_with_nvidia_metadata']=float((cl.reindex(tr+ev).dropna(subset=['country']).country==md.reindex(cl.reindex(tr+ev).dropna(subset=['country']).index).country).mean())
json.dump(Q,open('F:/TanitAD-scratch/strat1/labels/LABEL_QA.json','w'),indent=1)
print(json.dumps({k:Q[k] for k in ['constant_baseline_eval','coverage','eval_label_change_fraction_under_time_shift_s','countries_only_in_eval','eval_countries_with_train_clips_lt_20','country_agrees_with_nvidia_metadata']},indent=1)[:6000])
r=Q['country_x_platform_class'];print({k:r[k] for k in ['train4572','eval141','train+eval','platform_class_counts_ours','countries_single_platform_class']})
