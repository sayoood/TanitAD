import sys,json,glob,os,collections,hashlib,lzma
sys.path.insert(0,"F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-08-scorer1-impl-v3/stage0")
import stage0_gap as g
def rows(d,pat):
    out=[]
    for fp in sorted(glob.glob(os.path.join(d,pat))):
        for ln in open(fp,encoding='utf-8'):
            if ln.startswith("{"): out.append(json.loads(ln))
    return out
sets=rows("ho/refe_heldout/sets","onpolicy_*.jsonl")
tg=rows("refe_navtrain/train_grow","targets_rank*.jsonl")
pd=rows("s1aux/heldout_pdm/heldout_pdm","pdm_*.jsonl")
S=[g.key_set(r) for r in sets]; T=[g.key_set(r) for r in tg]; P=[g.key_pdm(r) for r in pd]
print("rows sets/tg/pdm",len(S),len(T),len(P),"unique",len(set(S)),len(set(T)),len(set(P)))
print("S==T",set(S)==set(T),"S==P",set(S)==set(P),"|S&T&P|",len(set(S)&set(T)&set(P)))
pk={g.key_pdm(r):r for r in pd}
full=sum(all(len(pk[k].get(c,()))==64 for c in g.PDM) for k in set(S))
print("miss counts stage0_gap would see (sets/targets/pdm):",len(set(S)-set(T)) if 0 else 0, sum(k not in set(T) for k in S), sum(k not in pk or any(len(pk[k].get(c,()))!=64 for c in g.PDM) for k in S))
print("full 6x64 PDM rows among set keys:",full)
sk={g.key_set(r):r for r in sets}
cm=[(sk[k]["ckpt_step"],pk[k]["ckpt_step"]) for k in sk]
print("ckpt_step sets",collections.Counter(a for a,_ in cm),"pdm",collections.Counter(b for _,b in cm),"equal",sum(a==b for a,b in cm),"of",len(cm))
print("rank/step",collections.Counter((k[2],k[3]) for k in S))
print("targets-vs-sets rank match; logs", len({k[0] for k in S}))
hl=[l.strip() for l in open("ho/refe_heldout/heldout_logs.txt",encoding="utf-8") if l.strip()]
sp=[l.strip() for l in open("F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/splits/sft_heldout_logs.txt",encoding="utf-8") if l.strip() and not l.startswith("#")]
print("heldout_logs",len(hl),"splits file",len(sp),"equal",set(hl)==set(sp),"set logs subset",{k[0] for k in S}<=set(hl))
# proposals match: pdm has no traj; check set traj count 64
print("n traj per set",collections.Counter(len(r["traj"]) for r in sets))
# --- Part 1.3 teacher lane vs banked
tl=rows("s1aux/heldout_teacher_lane/heldout_teacher_lane","*.jsonl")
print("s1aux teacher_lane rows",len(tl),"keys",len({g.key_pdm(r) for r in tl}),"ckpt",collections.Counter(r["ckpt_step"] for r in tl))
tlk={g.key_pdm(r):r for r in tl}
print("tl keys==S",set(tlk)==set(S))
bank=None
import glob as gl
cands=gl.glob("E:/Projects/TanitAD/**/heldout_teacher_lane_labels.jsonl.xz",recursive=True)
print(cands)
