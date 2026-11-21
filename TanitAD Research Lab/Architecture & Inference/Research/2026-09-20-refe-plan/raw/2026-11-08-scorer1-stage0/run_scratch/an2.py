import sys,json,glob,os,collections
sys.path.insert(0,"F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-08-scorer1-impl-v3/stage0")
import stage0_gap as g
c=collections.Counter()
for fp in glob.glob("refe_navtrain/onpolicy/sets/*.jsonl"):
    for ln in open(fp,encoding='utf-8'):
        r=json.loads(ln); c[(os.path.basename(fp),r["ckpt_step"])]+=1
print(sorted(c.items()))
c=collections.Counter()
for fp in glob.glob("s1aux/heldout_pdm/heldout_pdm/*.jsonl"):
    for ln in open(fp,encoding='utf-8'):
        r=json.loads(ln); c[(os.path.basename(fp),r["ckpt_step"],r["label_version"])]+=1
print(sorted(c.items()))
