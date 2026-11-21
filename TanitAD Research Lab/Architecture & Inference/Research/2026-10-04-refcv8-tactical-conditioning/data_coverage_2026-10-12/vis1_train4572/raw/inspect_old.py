import numpy as np, json, sys
p = sys.argv[1]
z = np.load(p, allow_pickle=False)
for k in sorted(z.files):
    a = z[k]; print(k, a.dtype, a.shape)
cp = z['clip_frame_ptr']; nf = np.diff(cp)
print('n clips', len(z['clip_sha12']), 'zero-frame clips', int((nf==0).sum()))
print('mask True', int(z['clip_mask_ftheta'].sum()), 'False', int((~z['clip_mask_ftheta']).sum()))
zf = nf==0
print('zero-frame clips mask:', np.unique(z['clip_mask_ftheta'][zf], return_counts=True))
print('zero-frame split:', np.unique(z['clip_split'][zf], return_counts=True))
print('split counts', np.unique(z['clip_split'], return_counts=True))
print('sorted?', bool((z['clip_sha12'][:-1] < z['clip_sha12'][1:]).all()))
print('zero-frame ids', z['clip_sha12'][zf].tolist()[:20])
meta = json.loads(bytes(z['meta_json'].tobytes()).decode())
print(json.dumps({k: (v if k!='sources' else '...') for k,v in meta.items()}, default=str)[:1500])
