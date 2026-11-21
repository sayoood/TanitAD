import sys,glob,json,collections
sys.path.insert(0,'.')
import rj5_bank_join as J
R='F:/tanitad_scratch/rj5nt'
bank,_,_=J.load_bank(R+'/bank_p7_rank0_slim.jsonl.gz')
dev={}
for r in J.read_jsonl(R+'/bank_p7_rank0_slim.jsonl.gz'): dev[(r['log_name'],r['token'])]=str(r.get('teacher_device'))
per=collections.defaultdict(lambda:[0,0,0.0])
for src in ('A','B'):
  for fp in glob.glob(f'{R}/run_fetch_{src}/rows/rj5_*.jsonl'):
    for r in J.read_jsonl(fp):
      if r.get('arm')=='R0' and 'traj' in r:
        k=(r['log_name'],r['token']); dx=J.dxy(J.flat(r['traj']),bank[k][0]); p=per[(src,r['log_name'],dev[k])]
        p[0]+=1; p[1]+=dx>1e-6; p[2]=max(p[2],dx)
bad=[(k,v) for k,v in per.items() if v[1]]
print(len(per),'logs with control;',len(bad),'logs with any nonzero')
for k,v in sorted(bad,key=lambda x:-x[1][2]): print(k[0],k[2],k[1],v)
ok=[(k,v) for k,v in per.items() if not v[1]]
out={'n_control_logs':len(per),'n_logs_all_rows_differ':len(bad),'n_logs_exact':len(ok),
 'diverged':[{'src':k[0],'log':k[1],'bank_teacher_device':k[2],'n_control':v[0],'n_over_1e-6':v[1],'max_dxy_m':v[2]} for k,v in sorted(bad,key=lambda x:-x[1][2])],
 'exact':[{'src':k[0],'log':k[1],'bank_teacher_device':k[2],'n_control':v[0]} for k,v in sorted(ok)]}
json.dump(out,open('E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rj5path/control_by_log_bulk.json','w'),indent=1)
