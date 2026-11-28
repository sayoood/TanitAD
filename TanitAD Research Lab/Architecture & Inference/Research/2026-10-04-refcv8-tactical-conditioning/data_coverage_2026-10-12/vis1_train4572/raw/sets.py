import torch, hashlib, json, numpy as np, os, glob
A='C:/Users/Admin/tanitad-data/add203/'
def s12(x): return hashlib.sha256(str(x).encode()).hexdigest()[:12]
mt=torch.load(A+'in/train4572_v2manifest.pt',map_location='cpu',weights_only=False)
me=torch.load(A+'in/eval139_v2manifest.pt',map_location='cpu',weights_only=False)
tc=[str(c) for c in mt['clip_id']]; ec=[str(c) for c in me['clip_id']]
print('train',len(tc),len(set(tc)),'eval',len(ec),'n_stack',sorted(set(mt['n_stack'])),sorted(set(me['n_stack'])))
old=np.load('raw/old/vis1_sidecar_refcv6b1_train4369_eval139.npz')
o12=old['clip_sha12'].tolist(); osp=old['clip_split'].tolist()
otr={s for s,p in zip(o12,osp) if p=='train'}; oev={s for s,p in zip(o12,osp) if p=='eval'}
t12={s12(c):c for c in tc}; e12={s12(c):c for c in ec}
print('sha12 unique train',len(t12),'eval',len(e12))
add=sorted(set(t12)-otr)
print('ADD',len(add),'old train covered',len(otr&set(t12)),'old train not in new view',len(otr-set(t12)))
print('eval set equal old eval:',set(e12)==oev, 'train/eval overlap',len(set(t12)&set(e12)))
tab=json.load(open(A+'out/refcv8_train4572_eval139_extrinsics.json'))
chunks={int(tab[t12[s]].get('chunk',-1)) for s in add}
have={int(os.path.basename(p).split('chunk_')[1][:4]) for p in glob.glob('work/calib_old/calib/camera_intrinsics/*.parquet')}
print('ADD chunks',len(chunks),'missing from calib',sorted(chunks-have))
allch={int(tab[c].get('chunk',-1)) for c in tc+ec}
print('all chunks',len(allch),'missing',sorted(allch-have))
json.dump({'add':add,'old_train':sorted(otr),'eval':sorted(oev)},open('work/sets.json','w'))
# joined status of ADD
print(open(A+'in/join2d.meta.json').read()[:600])
