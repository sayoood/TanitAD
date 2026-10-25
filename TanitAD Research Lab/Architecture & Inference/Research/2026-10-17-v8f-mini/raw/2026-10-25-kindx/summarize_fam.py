import json, sys
r=json.load(open('READ.json'))['beside']
f=r['families']
arms=['kind-own','kind-only-speed','kind-only-angle','kind-only-tte','kind-only-timegap','kind-only-geom','kind-only-leadgap','kind-union','kind-all']
# identical tactical/strategic?
tac_same = all(json.dumps(f[a]['tactical'],sort_keys=True)==json.dumps(f['kind-own']['tactical'],sort_keys=True) for a in arms)
str_same = all(json.dumps(f[a]['strategic'],sort_keys=True)==json.dumps(f['kind-own']['strategic'],sort_keys=True) for a in arms)
print('tactical identical across arms:',tac_same,' strategic identical:',str_same)
t=f['kind-own']['tactical']
print('tactical lat macro_f1',t['lat']['macro_f1'],'n',t['lat']['n'],'lon macro_f1',t['lon']['macro_f1'],'goal_macro_f1',t['goal_macro_f1'])
s=f['kind-own']['strategic']
print('strategic', {k:(v['mean'] if isinstance(v,dict) else v) for k,v in s.items()})
rows=[]
for a in arms[1:]:
    row={'arm':a}
    for fam,blk in (('longitudinal','speed_mae_mps'),('longitudinal','along_mae_m'),('longitudinal','accel_mae_mps2'),('lateral','heading_mae_deg'),('lateral','yaw_rate_mae_degps'),('lateral','curvature_mae_1pm'),('lateral','cross_mae_m')):
        d=f[a][fam]['paired_vs_own'][blk]
        row[blk]=(round(d['delta'],4),round(d['lo'],4),round(d['hi'],4),d['separated'])
    dk=f[a]['longitudinal'].get('distance_keeping_k10',{})
    rows.append(row)
for r_ in rows: print(r_)
print('own abs:', {k:round(v['arm'],4) for k,v in f['kind-only-speed']['longitudinal']['paired_vs_own'].items()}, {k:round(v['arm'],4) for k,v in f['kind-only-speed']['lateral']['paired_vs_own'].items()})
# distance keeping / lambda
for a in arms:
    d=f[a]['longitudinal']['distance_keeping_k10']
    m=d['arm']['metrics']
    print(a,'lambda',round(f[a]['lambda']['mean_lead_m'],4),'headway',round(m['headway_min_m']['mean'],3),'tg',round(m['time_gap_min_s']['mean'],3),'ttc',round(m['min_ttc_s']['mean'],3),'viol_share',d['arm']['lead_viol'].get('share'))
print(list(f['kind-only-speed']['longitudinal']['distance_keeping_k10'].keys()), list(f['kind-only-speed']['longitudinal']['distance_keeping_k10']['arm'].keys()))
print(json.dumps(f['kind-only-speed']['lambda'],indent=0)[:800])
