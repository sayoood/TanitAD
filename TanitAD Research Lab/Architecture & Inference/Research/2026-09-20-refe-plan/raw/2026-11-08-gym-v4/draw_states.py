import sys,os,json,hashlib
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
sys.path.insert(0,'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/scorer_gym'); sys.dont_write_bytecode=True
import gym_env as g
sts=g.select_states_v4(exclude_logs=g.used_logs_m_gym_2())
j=json.dumps(sts,indent=1)
open('E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/PREREG_M_GYM_4_STATES.json','w').write(j)
open('E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/PREREG_M_GYM_4_STATES.sha256','w').write(hashlib.sha256(j.encode()).hexdigest()+"  PREREG_M_GYM_4_STATES.json\n")
print(hashlib.sha256(j.encode()).hexdigest())
print(len(sts), sum(s['sampler']=='random' for s in sts), len({s['log'] for s in sts}))
for s in sts: print(s['sampler'], s['log'][:30], s['row'])
