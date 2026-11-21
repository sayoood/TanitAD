import sys, gzip, json
import numpy as np
S='D:/Projects/TanitAD/data/refe_navtest/seams/'
z=np.load(S+'stage2_p2_full.npz', allow_pickle=False)
t=np.load('D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz', allow_pickle=False)
st=[str(x) for x in z['token']]; tt=[str(x) for x in t['token']]
print(len(set(st)),len(set(tt)),set(st)==set(tt))
idx={x:i for i,x in enumerate(tt)}
order=np.array([idx[x] for x in st])
P=t['proposals'][order]; pick=t['pick'][order]
pk=P[np.arange(len(pick)),pick]
print('bitwise equal after token join:', np.array_equal(pk, z['poses']), np.abs(pk-z['poses']).max())
print('fingerprint equal', np.array_equal(t['fingerprint'][order], z['fingerprint']))
