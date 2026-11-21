import json, os, glob, subprocess, sys
sys.path.insert(0,'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs,tok_log,by=s0()
T=set(tok_log)
res={}
dirs={
 'astra_e023_reusable_native_cache':"D:/TanitAstraResearch/experiments/e023_hierarchical_transfer/artifacts/colab_b_20261010/extracted/reusable_native_cache/metric_cache",
 'astra_e024_eval_fixed10':"D:/TanitAstraResearch/experiments/e024_target_distribution/postfeature_batches/eval_fixed10/artifacts/run_20261010_a/extracted/evaluation/metric_cache",
 'astra_e025_teacher_v2':"D:/TanitAstraResearch/experiments/e025_action_consequence_verifier/colab_teacher_v2/run_20261011_b/extracted/evaluation/metric_cache",
 'astra_e026_teacher_v1':"D:/TanitAstraResearch/experiments/e026_negative_experience_acquisition/colab_teacher_v1/run_20261011_a/extracted/evaluation/metric_cache",
 'astra_e031_camera_policy':"D:/TanitAstraResearch/experiments/e031_camera_policy/evaluation_inputs/metric_cache",
}
for k,d in dirs.items():
    names=[n.split('.')[0] for n in os.listdir(d)]
    res[k]={'path':d,'n_entries':len(names),'n_in_S0_tokens':len(set(names)&T)}
# devbox-C warmup (log dirs)
d='D:/Archive/devbox-C/navsim/exp/metric_cache_warmup_two_stage'
ls=[n for n in os.listdir(d) if os.path.isdir(os.path.join(d,n))]
toks=[]
for l in ls:
    for dp,dn,fn in os.walk(os.path.join(d,l)):
        if 'metric_cache.pkl' in fn: toks.append((l,os.path.basename(dp)))
res['devbox_C_warmup_two_stage']={'path':d,'logs':ls,'logs_in_S0':[l for l in ls if l in set(logs)],'n_pkl':len(toks),'n_tokens_in_S0':sum(1 for l,t in toks if t in T)}
# official tar members
tp='D:/TanitAstraResearch/experiments/e023_hierarchical_transfer/artifacts/colab_b_20261010/extracted/official_metric_cache.tar'
out=subprocess.run(['tar','-tf',tp],capture_output=True,text=True,timeout=120).stdout.splitlines()
mem=[o for o in out if o.strip()]
stems=set()
for o in mem:
    b=os.path.basename(o.rstrip('/'))
    stems.add(b.split('.')[0])
res['astra_e023_official_metric_cache.tar']={'path':tp,'n_members':len(mem),'example':mem[:3],'n_stems_in_S0_tokens':len(stems&T),'n_S0_logs_named':sum(1 for l in logs if any(l in o for o in mem))}
json.dump(res,open('mc_overlap.json','w'),indent=1)
print(json.dumps(res,indent=1)[:3500])
