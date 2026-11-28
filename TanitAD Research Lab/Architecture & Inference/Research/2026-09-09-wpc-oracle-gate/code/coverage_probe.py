import gzip, json, lzma, sys
lab_path = "/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz"
ev_path  = "/workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz"
join     = "/workspace/joins/b1_train_plus_eval_agents.jsonl.xz"

def clips(p):
    s = set()
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            c = r.get("clip_id") or r.get("clip") or r.get("id")
            if c: s.add(c)
    return s

train = clips(lab_path); ev = clips(ev_path)
print("ZZLABELS", "train_clips", len(train), "eval_clips", len(ev))

jc = set(); rows = 0
with lzma.open(join, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rows += 1
        r = json.loads(line)
        c = r.get("clip_id")
        if c: jc.add(c)
print("ZZJOIN", "rows", rows, "clips", len(jc))
it = train & jc; ie = ev & jc
print("ZZCOV train_cov %d/%d = %.4f" % (len(it), len(train), len(it)/max(len(train),1)))
print("ZZCOV eval_cov  %d/%d = %.4f" % (len(ie), len(ev), len(ie)/max(len(ev),1)))
print("ZZCTRL join_clips_not_in_either", len(jc - train - ev))
