import json,hashlib,os,sys,pickle,io,tarfile,collections
from PIL import Image
B='F:/TanitAD_tsf_bundle2_20261107/bundle_a6'
CRU='E:/Projects/TanitAD-artifacts/tsf_nuplan_cruise_bundle/a0_model_inputs.jsonl'
SRC='F:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera'
m=json.load(open(B+'/MANIFEST.json'))
out={}
# 1 sha
bad=[];n=0
for p,v in m['files'].items():
    fp=os.path.join(B,p)
    if not os.path.exists(fp): bad.append(('missing',p));continue
    h=hashlib.sha256(open(fp,'rb').read()).hexdigest();n+=1
    if h!=v['sha256'] or os.path.getsize(fp)!=v['bytes']: bad.append(('mismatch',p))
disk=set()
for r,_,fs in os.walk(B):
    for f in fs: disk.add(os.path.relpath(os.path.join(r,f),B).replace(os.sep,'/'))
extra=sorted(disk-set(m['files'])-{'MANIFEST.json'})
out['sha']={'n_manifest':len(m['files']),'checked':n,'bad':bad,'on_disk_not_in_manifest':extra}
# 2 decode + counts
cams=['CAM_B0','CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2']
toks=sorted(os.listdir(B+'/frames'));dec=[]
for t in toks:
    for sub in ('','tm2s/'):
        fs=sorted(os.listdir(f'{B}/frames/{t}/{sub}'))
        js=[f for f in fs if f.endswith('.jpg')]
        if sorted(js)!=sorted(c+'.jpg' for c in cams): dec.append((t,sub,'cams',js))
        for f in js:
            try:
                im=Image.open(f'{B}/frames/{t}/{sub}{f}');im.load()
                if im.size!=(1920,1080): dec.append((t,sub,f,im.size))
            except Exception as e: dec.append((t,sub,f,str(e)))
out['decode']={'n_tokens':len(toks),'problems':dec}
# 3 tm2s offset independent from pkl
pk={}
def meta(log):
    if log not in pk: pk[log]=pickle.load(open(f'{B}/openscene_meta_test/{log}.pkl','rb'))
    return pk[log]
rows=[json.loads(l) for l in open(B+'/a0_model_inputs.jsonl')]
log_of={r['token']:r['log'] for r in rows}
dts={};ok=True;srcpaths={}
for t in toks:
    d=meta(log_of[t]);idx={f['token']:i for i,f in enumerate(d)}
    i=idx[t];f0=d[i];fm=d[i-4]
    dts[t]=(f0['timestamp']-fm['timestamp'])/1e6
    # confirm manifest paths point to those frames, and bytes equal the pkl-named file
    for sub,fr in (('',f0),('tm2s/',fm)):
        for c in cams:
            rel=fr['cams'][c]['data_path']
            fn=f'{B}/frames/{t}/{sub}{c}.jpg'
            ent=m['images'].get(rel) if sub=='' else m['tm2s_frames']['images'].get(rel)
            if ent is None or ent['token']!=t or ent['cam']!=c or ent['sha256']!=hashlib.sha256(open(fn,'rb').read()).hexdigest():
                ok=False;print('PATHFAIL',t,sub,c)
            srcpaths[(t,sub,c)]=rel
v=list(dts.values())
out['tm2s']={'min':min(v),'max':max(v),'outliers_gt_0.05':{k:x for k,x in dts.items() if abs(x-2)>0.05},'index_minus4_path_match_ok':ok,
 'manifest_dt_equal':all(abs(m['tm2s_frames']['dt_to_t0_s'][k]-dts[k])<6e-5 for k in dts)}
# 5 a0 rows
cr=list(json.loads(open(CRU).readline()))
ks=list(rows[0])
out['a0']={'n_rows':len(rows),'one_per_token':sorted(r['token'] for r in rows)==toks and len(set(r['token'] for r in rows))==len(rows),
 'keys_bundle2':ks,'keys_cruise':cr,'extra_vs_cruise':[k for k in ks if k not in cr],'missing_vs_cruise':[k for k in cr if k not in ks],
 'lens_ok':all(len(r['ego7_as_fed'])==7 and len(r['goal_as_fed'])==4 for r in rows),'pick_null':sum(r['pick'] is None for r in rows),
 'log_matches_meta':all(any(f['token']==r['token'] for f in meta(r['log'])) for r in rows)}
# 4 source byte equality: 3 tokens (t0 + tm2s) from the tgz
tg=collections.defaultdict(list)
for t in toks:
    e=m['images'][srcpaths[(t,'','CAM_F0')]];tg[e['tgz']].append(t)
tgz=max(tg,key=lambda k:len(tg[k]));sel=tg[tgz][:3]
if len(sel)<3: sel=toks[:3]
need={}
for t in sel:
    for sub in ('','tm2s/'):
        for c in cams: need[srcpaths[(t,sub,c)]]=f'{B}/frames/{t}/{sub}{c}.jpg'
eq=[];neq=[];found=set()
tgs={m['images'][r]['tgz'] for r in need if r in m['images']}|{m['tm2s_frames']['images'][r]['tgz'] for r in need if r in m['tm2s_frames']['images']}
for tgn in sorted(tgs):
    with tarfile.open(f'{SRC}/{tgn}','r|gz') as tf:
        for mem in tf:
            if not mem.isfile(): continue
            rel=mem.name.split('sensor_blobs/test/',1)[-1]
            if rel in need:
                b=tf.extractfile(mem).read();found.add(rel)
                (eq if b==open(need[rel],'rb').read() else neq).append(rel)
                if len(found)==len(need): break
out['bytes_vs_source']={'tokens':sel,'tgz':sorted(tgs),'n_wanted':len(need),'equal':len(eq),'unequal':neq,'not_found':sorted(set(need)-found)}
print(json.dumps(out,indent=1,default=str))
