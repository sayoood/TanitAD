import json
raw=json.load(open('probe_s0_raw.json'))['logs']
mine={l for l,r in raw.items() if r.get('tokens_all4cams_resolved')==r['n_tokens']}
pl=json.load(open('D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_full/plan.json'))
sh=set(pl['token_log'].values())
print('mine',len(mine),'sh5',len(sh),'equal',mine==sh)
print('tokens mine',sum(raw[l]['n_tokens'] for l in mine),'sh5',len(pl['tokens']))
pp=json.load(open('D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/plan_partial_5776/plan.json'))
print('partial sh5 tokens with frames', len(pp['tokens']))
