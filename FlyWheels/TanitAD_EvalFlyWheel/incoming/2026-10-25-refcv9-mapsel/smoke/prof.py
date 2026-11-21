import sys, time, numpy as np, torch
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code')
import mapsel_lib as L
torch.set_num_threads(4)
z=np.load('F:/refcv9_mapsel/smoke/synth/trace.npz'); fan=z['fan'][50]
r=np.load('F:/refcv9_mapsel/smoke/synth/rast/e000_w0017.npz') if False else None
import glob; f=sorted(glob.glob('F:/refcv9_mapsel/smoke/synth/rast/*.npz'))[50]; r=np.load(f)
def T(label, fn, n=5):
    fn(); t=time.time(); [fn() for _ in range(n)]; print(label, round((time.time()-t)/n*1000,1),'ms')
T('np.load planes', lambda: [np.load(f)[k] for k in ('pdrv','pedge','code','gt')])
g=None
def mk():
    global g; g=L.Geom(torch.from_numpy(fan))
T('Geom', mk)
mo=None
def mm():
    global mo; mo=L.masks_own(r['pdrv'], r['code'])
T('masks_own', mm)
T('masks_gt', lambda: L.masks_gt(r['gt']))
T('flags', lambda: g.flags(mo))
T('own_cont', lambda: g.own_cont(r['pdrv'], r['pedge']))
T('path_samples', lambda: L.path_samples(torch.from_numpy(fan)))
