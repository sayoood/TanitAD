import json, re, collections, os
BS=chr(92)
R=json.load(open('classes_scanned.json'))['results']
def lp(path):
    fp='E:/Projects/TanitAD/'+path
    if len(fp)>240: fp=BS+BS+'?'+BS+fp.replace('/',BS)
    return fp
kw=re.compile(rb'(?i)(api[_-]?key|apikey|secret|passwd|password|access[_-]?token|auth[_-]?token|hf[_-]?token|api[_-]?token|bearer)')
tail=re.compile(rb'^[A-Za-z0-9_\-]{0,20}["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-./+]{16,}["\']')
keys=collections.Counter(); hit={}
anyflag=collections.Counter()
for p,r in R.items():
    if 'secret_assign' in r.get('flags',[]):
        data=open(lp(p),'rb').read()
        ks=collections.Counter()
        for m in kw.finditer(data):
            seg=data[m.end():m.end()+80]
            if tail.match(seg):
                ks[m.group(1).lower()]+=1
        r['flags']=[f for f in r['flags'] if f!='secret_assign']
        if ks:
            r['flags'].append('secret_assign')
            hit[p]=dict(ks)
        keys.update(ks.keys())
print('files still flagged', len(hit))
print(keys.most_common(30))
json.dump(R, open('R_refined.json','w'))
