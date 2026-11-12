import sys,time,os
sys.path.insert(0,'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/instruments'); sys.path.insert(0,'F:/tanitad_scratch/instr/w/refe')
import numpy as np, transp_score as T
import navtrain_scenarios as NS, navsim_pdm_targets as PT, navsim_dac as ND
from navsim_pdm.pdm_array_representation import ego_state_to_state_array
z=np.load('plans_T.npz'); toks=[str(t) for t in z['tokens']]; logs=[str(l) for l in z['logs']]
lg=logs[5]; tk=[t for t,l in zip(toks,logs) if l==lg][:6]
dbs=NS.index_dbs(db_root=T.DB_ROOT)
t0=time.time(); scs=list(NS.build_scenarios_for_log(dbs[lg],tk)); print('build',time.time()-t0,len(scs))
for sc in scs[:6]:
    t=time.time(); parts=PT.metric_cache_parts(sc); t1=time.time()-t
    ego=parts[1]; s0=ego_state_to_state_array(ego); i=toks.index(sc._initial_lidar_token); p=z['plans20'][i]
    t=time.time(); tr=ND.to_navsim(p); ts=T.transparent_states(p,ego,s0); t2=time.time()-t
    t=time.time(); res,sim=T.score_arms(parts,[tr],[ts]); t3=time.time()-t
    t=time.time(); h=T.human_rel(sc,ego); l=T.lead_track(sc,ego); t4=time.time()-t
    print(f'cache {t1:.2f} build {t2:.3f} score {t3:.2f} fam {t4:.2f}')
