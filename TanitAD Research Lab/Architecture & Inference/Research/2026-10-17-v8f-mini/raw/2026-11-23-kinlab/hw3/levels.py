import os,sys,json
sys.path.insert(0,'F:/TanitAD-scratch/rollhist/read/code/headway')
os.environ.update(HEADWAY_Q3DIR='F:/TanitAD-scratch/rollhist/read/code/q3',HIERREP_STAGEB='F:/TanitAD-scratch/v8proof/pod/stageB',TANITEVAL_ROOT='F:/TanitAD-scratch/rollhist/read/tree/taniteval',HEADWAY_JOIN='F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz',HEADWAY_BANK='F:/TanitAD-scratch/v8proof/res/cache/frames.npz',CUDA_VISIBLE_DEVICES='-1')
import headway_read_r2 as H
E='F:/TanitAD-scratch/rollhist/read/out/t1'
need=[(a+'_'+f,s) for a in ('I10RH','I10B0') for f in ('R0KHb20','R0C10XHb20') for s in (0,1,2)]
need=[('I10%s_%s'%(a,f),s) for a in ('RH','B0') for f in ('R0KHb20','R0C10XHb20') for s in (0,1,2)]
Wd,missing,where=H.load_world([E],need,141)
ref=list(Wd.values())[0]; J=H.load_join(); L=H.label_windows(J,ref['clip_index'],ref['ws'])
out={}
for (arm,s),d in Wd.items():
    w=H.window_cells(d['g'],d['cl'],L); out.setdefault(arm[3:],[None]*3)[s]=H.level(w['U'])
print(missing, out); json.dump({'levels':out,**out,'missing':missing},open('F:/TanitAD-scratch/rollhist/hw3/hw3_levels.json','w'),indent=1)
