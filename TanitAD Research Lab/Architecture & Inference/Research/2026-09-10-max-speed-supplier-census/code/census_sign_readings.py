import gzip, json, sys, re, collections
path=sys.argv[1]; label=sys.argv[2]
# PRECISE: a number immediately bound to a road-speed unit (km/h or mph). NOT m/s.
pat_kmh = re.compile(r'(\d{1,3})\s*(?:km\s*/\s*h|km/h|kmh|kph|km per hour)', re.I)
pat_mph = re.compile(r'(\d{1,3})\s*(?:mph|miles per hour)', re.I)
pat_slctx = re.compile(r'speed[\s_-]*limit|posted', re.I)
n=0; clips_kmh=0; clips_mph=0; clips_any=0; clips_any_in_ctx=0
vals=collections.Counter(); vals_ctx=collections.Counter()
ctrl_digit=0; ctrl_tl=0
sl_bool_true=0; sl_bool_true_and_numeric=0; numeric_and_sl_false=0
per_clip=[]
def gather(r):
    parts=[]
    cs=r.get("cot_source") or {}
    for k in ("chain_of_causation","components_analysis","cot","motion_analysis"):
        v=cs.get(k)
        if isinstance(v,str): parts.append(v)
    sem=r.get("semantics") or {}
    if isinstance(sem.get("cot"),str): parts.append(sem["cot"])
    ct=r.get("cot_tokens") or {}
    if isinstance(ct.get("evidence"),str): parts.append(ct["evidence"])
    alp=r.get("alpamayo") or {}
    cc=alp.get("critical_component") or {}
    for k in ("type","why"):
        if isinstance(cc.get(k),str): parts.append(cc[k])
    return "\n".join(parts)
with gzip.open(path,"rt",encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line); n+=1
        t=gather(r)
        if re.search(r'\d',t): ctrl_digit+=1
        if re.search(r'traffic\s*light',t,re.I): ctrl_tl+=1
        slb=(r.get("cot_tokens") or {}).get("speed_limit") is True
        if slb: sl_bool_true+=1
        k=[int(x) for x in pat_kmh.findall(t)]
        m=[int(x) for x in pat_mph.findall(t)]
        has=bool(k or m)
        if k: clips_kmh+=1
        if m: clips_mph+=1
        if has:
            clips_any+=1
            for v in k: vals[("kmh",v)]+=1
            for v in m: vals[("mph",v)]+=1
            # is at least one hit within +-120 chars of speed-limit context?
            inctx=False
            for mm in list(pat_kmh.finditer(t))+list(pat_mph.finditer(t)):
                w=t[max(0,mm.start()-160):mm.end()+160]
                if pat_slctx.search(w): inctx=True; vals_ctx[int(mm.group(1))]+=1
            if inctx: clips_any_in_ctx+=1
            if slb: sl_bool_true_and_numeric+=1
            else: numeric_and_sl_false+=1
            per_clip.append((r.get("clip_id"), sorted(set(k)), sorted(set(m)), slb, inctx))
print("=== %s ==="%label)
print("n =",n)
print("CONTROL any digit:",ctrl_digit,"/",n,"  CONTROL 'traffic light':",ctrl_tl,"/",n)
print()
print("cot_tokens.speed_limit == True         :", sl_bool_true, "/", n)
print("clips with a km/h number               :", clips_kmh, "/", n)
print("clips with an mph number               :", clips_mph, "/", n)
print("clips with ANY road-unit number        :", clips_any, "/", n, " (%.3f%%)"%(100.0*clips_any/n))
print("  ...of those, in speed-limit context  :", clips_any_in_ctx, "/", n, " (%.3f%%)"%(100.0*clips_any_in_ctx/n))
print()
print("OVERLAP with the boolean flag:")
print("  numeric AND speed_limit==True :", sl_bool_true_and_numeric)
print("  numeric AND speed_limit==False:", numeric_and_sl_false, "  <-- the boolean MISSES these")
print()
print("=== distinct values found (unit,value): count ===")
for (u,v),c in sorted(vals.items(), key=lambda x:(-x[1],x[0])): print("   %-5s %4d  x%d"%(u,v,c))
print()
print("=== values IN speed-limit context ===")
for v,c in sorted(vals_ctx.items()): print("   %4d km/h  x%d"%(v,c))
print()
print("=== per-clip listing (first 40) ===")
for row in per_clip[:40]: print("  ",row)
