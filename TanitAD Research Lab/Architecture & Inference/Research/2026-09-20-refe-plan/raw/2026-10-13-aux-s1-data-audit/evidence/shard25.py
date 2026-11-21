import json, re, os
mem = [l.strip() for l in open("D:/Projects/TanitAD/data/openscene_probe/members.txt", encoding="utf-8") if l.strip()]
jp = [m for m in mem if m.endswith(".jpg")]
print("members", len(mem), "jpg", len(jp))
# bank images + needed sets
rows = []
for f in ("tg/train_grow/targets_rank0.jsonl", "tg/train_grow/targets_rank1.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); rows.append(r["image"])
bank_imgs = {x for t in rows for x in t}
base2full = {}
for m in jp:
    parts = m.split("/")
    if len(parts) >= 4:
        base2full[parts[-3] + "/" + parts[-2] + "/" + parts[-1]] = m     # log/CAM/hash.jpg
inb = [b for b in base2full if b in bank_imgs]
cams4 = [b for b in base2full if b.split("/")[1] in ("CAM_F0","CAM_L0","CAM_R0","CAM_B0")]
logs25 = {b.split("/")[0] for b in base2full}
bank_logs = {x.split("/")[0] for t in rows for x in t}
ho = set(open("D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/heldout_logs.txt").read().split())
print("shard25 4cam jpgs", len(cams4), "logs", len(logs25), "logs in bank", len(logs25 & bank_logs), "held-out logs in shard25", len(logs25 & ho), "bank images present in shard25", len(inb), "of", len(bank_imgs))
json.dump({"members": len(mem), "jpg": len(jp), "jpg_4cam": len(cams4), "logs": len(logs25), "logs_in_bank": len(logs25 & bank_logs), "heldout_logs_in_shard": len(logs25 & ho), "bank_images_in_shard": len(inb), "bank_images_total": len(bank_imgs)}, open("shard25.json", "w"))
