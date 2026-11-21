import os, sys, json, hashlib, random, zipfile, collections
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
L = set(logs)
PS = 'D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19'
FF = 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


out = {}
# 1. frames_full vs its sha list
rows = []
for l in open(PS + '/frames_full_sha256.txt', encoding='utf-8'):
    p = l.rstrip('\n').split(None, 2)
    if len(p) == 3:
        rows.append((p[0], int(p[1]), p[2]))
miss = 0
badsize = 0
for s, n, rel in rows:
    fp = FF + '/' + rel
    if not os.path.exists(fp):
        miss += 1
    elif os.path.getsize(fp) != n:
        badsize += 1
random.seed(20261020)
samp = random.sample(rows, 400)
badsha = sum(1 for s, n, rel in samp if os.path.exists(FF + '/' + rel) and sha(FF + '/' + rel) != s)
disk_files = sum(len(fs) for _, _, fs in os.walk(FF))
tot_bytes = sum(n for _, n, _ in rows)
out['frames_full'] = {'listed': len(rows), 'listed_bytes': tot_bytes, 'files_on_disk': disk_files, 'missing': miss, 'size_mismatch': badsize, 'sha_sampled': len(samp), 'sha_mismatch_in_sample': badsha,
                      'logs': len({r[2].split('/')[0] for r in rows}), 'logs_in_S0': len({r[2].split('/')[0] for r in rows} & L)}
print('frames_full', out['frames_full'])
# 2. DBs
dbr = []
for l in open(PS + '/dbs_76_sha256.txt', encoding='utf-8'):
    p = l.rstrip('\n').split(None, 2)
    if len(p) == 3:
        dbr.append((p[0], int(p[1]), p[2]))
dbpaths = {}
for fn in ('C:/Users/Admin/par18/s0probe/dbfind_D.txt', 'C:/Users/Admin/par18/s0probe/dbfind_C.txt'):
    for l in open(fn, encoding='utf-8'):
        p = l.strip()
        if p:
            dbpaths.setdefault(os.path.basename(p), []).append(p)
nf = 0
bs = 0
loc = collections.Counter()
for s, n, name in dbr:
    ps = dbpaths.get(name)
    if not ps:
        nf += 1
        continue
    loc[os.path.dirname(ps[0]).replace('D:/Projects/TanitAD/data/nuplan/', '')] += 1
    if os.path.getsize(ps[0]) != n:
        bs += 1
small = sorted(dbr, key=lambda r: r[1])[:12]
badsha_db = sum(1 for s, n, name in small if sha(dbpaths[name][0]) != s)
# compare with the S0 staging manifests (sha col 3)
man = {}
B = SR + '/run/batches'
for u in os.listdir(B):
    p = os.path.join(B, u, 'copy_manifest_dbs.tsv')
    if os.path.exists(p):
        for l in open(p, encoding='utf-8'):
            q = l.rstrip('\n').split('\t')
            if len(q) >= 3:
                man[q[0]] = (int(q[1]), q[2])
agree = sum(1 for s, n, name in dbr if name in man and man[name] == (n, s))
out['dbs_76'] = {'listed': len(dbr), 'listed_bytes': sum(n for _, n, _ in dbr), 'not_found': nf, 'size_mismatch': bs, 'sha_checked_smallest': len(small), 'sha_mismatch': badsha_db,
                 'agree_with_S0_staging_manifests(size+sha)': agree, 'S0_staged_db_names': len(man), 'location': dict(loc),
                 'listed_names_in_S0': len({n[:-3] for _, _, n in dbr} & L)}
print('dbs_76', out['dbs_76'])
# 3. Astra anchors identical to proxy zips?
AN = 'D:/TanitAstraResearch/data/e024_anchors_20261010/sensors'
PX = 'D:/Projects/TanitAD/data/refe_proxy/pixels'
same = diff = 0
shared = 0
for lg in os.listdir(AN):
    for cam in ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0'):
        d = os.path.join(AN, lg, cam)
        zp = os.path.join(PX, f'{lg}_{cam}.zip')
        if os.path.isdir(d) and os.path.exists(zp):
            z = zipfile.ZipFile(zp)
            names = set(z.namelist())
            for f in os.listdir(d):
                if f in names:
                    shared += 1
                    if z.read(f) == open(os.path.join(d, f), 'rb').read():
                        same += 1
                    else:
                        diff += 1
out['astra_anchor_vs_proxy_zip'] = {'shared_jpgs': shared, 'byte_identical': same, 'different': diff}
print(out['astra_anchor_vs_proxy_zip'])
json.dump(out, open('C:/Users/Admin/par18/s0probe/verify_bytes.json', 'w'), indent=1)
