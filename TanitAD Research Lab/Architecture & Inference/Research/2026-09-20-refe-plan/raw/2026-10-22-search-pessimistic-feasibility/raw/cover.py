import gzip, csv, os, json, collections, sys
sp = sys.argv[1]
rows = list(csv.DictReader(gzip.open(os.path.join(sp, 'navtrain_token_table.tsv.gz'), 'rt'), delimiter='\t'))
mpdm = {r['token']: r['log'] for r in rows if int(r['served_train_sets']) > 0}
msets = {r['token']: int(r['served_train_sets']) for r in rows}
dbs = set()
for s in ('trainval', 'test'):
    for f in os.listdir('D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/' + s):
        if f.endswith('.db'): dbs.add(f[:-3])
# proxy frames tokens
prox = set()
for line in open('D:/Projects/TanitAD/data/refe_proxy/cache_train_ep015/frames.jsonl'):
    k = json.loads(line)['key']; prox.add(k[1])
proxlogs = set(f.rsplit('_CAM_', 1)[0] for f in os.listdir('D:/Projects/TanitAD/data/refe_proxy/pixels'))
m5logs = set(os.listdir('D:/Projects/TanitAD/data/refe_m5/pixels'))
pixlogs = proxlogs | m5logs
print('proxy cache tokens', len(prox), 'in M-PDM', len(prox & set(mpdm)))
print('pixel logs (proxy zips + m5 dirs)', len(pixlogs), 'proxy', len(proxlogs), 'm5', len(m5logs))
mlogs = set(mpdm.values())
both = mlogs & pixlogs & dbs
print('M-PDM logs with pixel container AND DB on dev box', len(both), 'tokens', sum(1 for t,l in mpdm.items() if l in both), 'sets', sum(msets[t] for t,l in mpdm.items() if l in both))
print('M-PDM logs with pixel container', len(mlogs & pixlogs), 'tokens(upper bound)', sum(1 for t,l in mpdm.items() if l in pixlogs))
print('proxy-cache tokens in M-PDM whose log has a DB', sum(1 for t in prox & set(mpdm) if mpdm[t] in dbs))
