import json, os, glob, csv, collections
B='C:/Users/Admin/srttc_s0/run/batches'
out={}
for b in sorted(os.listdir(B)):
    d=os.path.join(B,b)
    row={'unit':b}
    for nm in ('copy_manifest_frames.tsv','copy_manifest_dbs.tsv'):
        p=os.path.join(d,nm)
        if os.path.exists(p):
            rows=[l.rstrip('\n').split('\t') for l in open(p,encoding='utf-8') if l.strip()]
            row[nm]={'n':len(rows),'ok':sum(1 for r in rows if r[-1]=='OK'),'bytes':sum(int(r[1]) for r in rows if r[1].isdigit()) if nm.endswith('frames.tsv') else sum(int(r[1]) for r in rows if r[1].isdigit())}
    a=os.path.join(d,'audit.json')
    if os.path.exists(a):
        j=json.load(open(a))
        row['audit']={'n_tokens':j['n_tokens'],'ok':j['n_tokens_ok'],'blocked':j['n_tokens_blocked'],'logs':len(j['per_log'])}
    out[b]=row
print(len(out))
for b,r in out.items(): print(b, json.dumps({k:v for k,v in r.items() if k!='unit'}))
