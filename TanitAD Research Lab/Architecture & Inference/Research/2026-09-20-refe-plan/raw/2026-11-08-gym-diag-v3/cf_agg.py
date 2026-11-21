import json,glob,sys
for f in sorted(glob.glob(sys.argv[1]+'/cf_*.json')):
    r=json.load(open(f)); print(f"\n== {r['sampler']} state {r['state']} mode {r['mode']}  (base = stored run: see M_GYM_2 results)")
    for v,rows in r['variants'].items():
        b=max(rows,key=lambda x:x['ret']); nh=[x for x in rows if x['hard']]
        import collections; c=collections.Counter(x['hard'][1] for x in nh)
        print(f" ttc variant {v:7s}: arms with hard event {len(nh)}/24 types {dict(c)}; oracle arm {b['k']} ret {b['ret']:.3f} hard={b['hard']}; best clean ret {max([x['ret'] for x in rows if not x['hard']] or [float('nan')]):.3f}")
