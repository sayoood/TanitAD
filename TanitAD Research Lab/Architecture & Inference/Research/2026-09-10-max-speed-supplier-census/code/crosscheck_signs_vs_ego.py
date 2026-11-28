import gzip, json, sys, re
path=sys.argv[1]
pat_kmh=re.compile(r'(\d{1,3})\s*(?:km\s*/\s*h|km/h|kmh|kph)',re.I)
pat_mph=re.compile(r'(\d{1,3})\s*(?:mph|miles per hour)',re.I)
def gather(r):
    p=[]
    cs=r.get("cot_source") or {}
    for k in ("chain_of_causation","components_analysis","cot"):
        if isinstance(cs.get(k),str): p.append(cs[k])
    ct=r.get("cot_tokens") or {}
    if isinstance(ct.get("evidence"),str): p.append(ct["evidence"])
    return "\n".join(p)
rows=[]; n=0
with gzip.open(path,"rt",encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line); n+=1
        t=gather(r)
        k=[int(x) for x in pat_kmh.findall(t)]
        m=[int(x)for x in pat_mph.findall(t)]
        if not (k or m): continue
        sb=((r.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND") or {}
        vhi=sb.get("v_hi_ms")
        st=(r.get("strata") or {})
        sign_kmh = (min(k) if k else round(min(m)*1.609))
        rows.append((r.get("clip_id")[:8], sign_kmh, sorted(set(k)), sorted(set(m)),
                     round(vhi*3.6,1) if isinstance(vhi,(int,float)) else None,
                     st.get("road_class"), st.get("country")))
print("n scanned =", n, " | clips with a sign number =", len(rows))
print()
print("%-9s %-8s %-14s %-8s %-11s %-13s %s"%("clip","sign","kmh_all","mph_all","ego_vhi_kmh","road_class","country"))
viol=0; ok=0
for c,s,k,m,e,rc,co in rows:
    flag=""
    if e is not None:
        if e > s + 5: flag="  <-- ego EXCEEDS sign"; viol+=1
        else: ok+=1
    print("%-9s %-8s %-14s %-8s %-11s %-13s %s%s"%(c,s,k,m,e,rc,co,flag))
print()
print("ego <= sign+5 :", ok, " | ego EXCEEDS sign+5 :", viol, " of", len([r for r in rows if r[4] is not None]))
