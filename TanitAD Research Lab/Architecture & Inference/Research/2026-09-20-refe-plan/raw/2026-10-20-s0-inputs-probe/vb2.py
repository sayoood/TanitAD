import json, os, hashlib
BK='C:/Users/Admin/srttc_s0/work/bank'
m=json.load(open(BK+'/MANIFEST.sha256.json'))['files']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()
n=0;bad=[];miss=[]
for k,v in m.items():
    if k.startswith('work/bank/'):
        p=BK+'/'+k.split('/')[-1]
        if not os.path.exists(p): miss.append(k); continue
        n+=1
        if sha(p)!=v['sha256']: bad.append(k)
on_disk=[x for x in os.listdir(BK) if x!='MANIFEST.sha256.json']
print('bank entries in manifest',n,'bad',bad,'missing',miss,'files on disk',len(on_disk))
json.dump({'bank_entries_checked':n,'bad':bad,'missing':miss,'files_on_disk_excl_manifest':len(on_disk)},open('verify_bank_manifest.json','w'))
