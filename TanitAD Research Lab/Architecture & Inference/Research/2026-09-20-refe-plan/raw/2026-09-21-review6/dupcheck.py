import json,collections
d=json.load(open('probe_120.json'))
seen=collections.defaultdict(list)
for r in d['results']:
    a=r['A_devkit']
    seen[(a.get('ctrl_rear_axle_x'),a.get('ctrl_time_us'))].append(r['path'])
for k,v in seen.items():
    if len(v)>1:
        print("DUPLICATE POSE -> same log present in two dirs:")
        for p in v: print("   ",p)
print("distinct basenames:",len({__import__('os').path.basename(r['path']) for r in d['results']}),"of",len(d['results']))
