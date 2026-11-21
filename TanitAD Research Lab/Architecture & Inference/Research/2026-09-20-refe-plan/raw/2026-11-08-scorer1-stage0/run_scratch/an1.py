import sys,json,glob,os,collections
sys.path.insert(0,"F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-08-scorer1-impl-v3/stage0")
import stage0_gap as g
def rows(d,pat):
    out=[]
    for fp in sorted(glob.glob(os.path.join(d,pat))):
        for ln in open(fp,encoding='utf-8'):
            if ln.startswith("{"): out.append((os.path.basename(fp),json.loads(ln)))
    return out
sets=rows("refe_navtrain/onpolicy/sets","onpolicy_*.jsonl")
tg=rows("refe_navtrain/train_grow","targets_rank*.jsonl")
pd=rows("s1aux/heldout_pdm/heldout_pdm","pdm_*.jsonl")
print("sets rows",len(sets),"unique key_set",len({g.key_set(r) for _,r in sets}))
print("targets rows",len(tg),"unique",len({g.key_set(r) for _,r in tg}))
print("pdm rows",len(pd),"unique key_pdm",len({g.key_pdm(r) for _,r in pd}),"none",sum(g.key_pdm(r) is None for _,r in pd))
S={g.key_set(r) for _,r in sets}; T={g.key_set(r) for _,r in tg}; P={g.key_pdm(r) for _,r in pd}
print("T-S",len(T-S),"S-T",len(S-T),"T&P",len(T&P),"T-P",len(T-P),"P-T",len(P-T),"S&P",len(S&P),"S-P",len(S-P))
print("sample T-S",list(T-S)[:5])
# value-complete
ok=0
pdk={}
for _,r in pd:
    k=g.key_pdm(r)
    pdk.setdefault(k,[]).append(r)
print("dup pdm keys",sum(len(v)>1 for v in pdk.values()))
for k in T:
    r=pdk.get(k,[None])[-1]
    if r and all(len(r.get(c,()))==64 for c in g.PDM): ok+=1
print("targets keys with full PDM (6x64):",ok,"of",len(T))
# ckpt_step comparison sets vs pdm
sk=collections.defaultdict(set)
for _,r in sets: sk[g.key_set(r)].add(r["ckpt_step"])
mm=0;cmp=0
for k,v in pdk.items():
    if k in sk:
        cmp+=1
        if {r["ckpt_step"] for r in v}!=sk[k]: mm+=1
print("ckpt_step compared",cmp,"mismatch",mm)
print("rank dist sets",collections.Counter(k[3] for k in S),"pdm",collections.Counter(k[3] for k in P),"tg",collections.Counter(k[3] for k in T))
print("step dist tg",collections.Counter(k[2] for k in T).most_common(5))
print("logs",len({k[0] for k in T}))
