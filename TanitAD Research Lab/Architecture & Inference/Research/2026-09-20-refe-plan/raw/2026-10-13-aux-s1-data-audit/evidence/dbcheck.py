import os, json
logs = open('D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/heldout_logs.txt').read().split()
root = 'D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits'
splits = os.listdir(root)
res = {}
tot = 0
for l in logs:
    found = None
    for s in splits:
        p = os.path.join(root, s, l + '.db')
        if os.path.exists(p):
            found = (s, os.path.getsize(p)); break
    res[l] = found
    if found: tot += found[1]
print(len(logs), 'heldout logs;', sum(1 for v in res.values() if v), 'DBs present on dev;', 'bytes', tot, 'splits', splits)
# sets dir
sd = 'D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets'
fs = [(f, os.path.getsize(os.path.join(sd,f))) for f in sorted(os.listdir(sd))]
print(fs, sum(x[1] for x in fs))
n = 0
for f,_ in fs:
    with open(os.path.join(sd,f), 'rb') as fh:
        n += sum(1 for _ in fh)
print('heldout set lines', n)
