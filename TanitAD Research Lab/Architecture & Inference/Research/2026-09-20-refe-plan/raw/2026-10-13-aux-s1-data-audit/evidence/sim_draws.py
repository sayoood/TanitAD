"""Reconstruct S1's data order from banked inputs and count the distinct frames one run (and the nine) read.
Inputs (all banked, nothing downloaded): tg/train_grow/targets_rank{0,1}.jsonl (extracted from the dev-box refe_train_grow.tar.zst),
raw/rl_pool/served_keys.tsv.gz (P2's 174,912 PDM-labelled train keys = the pdm_keys of --require-pdm), heldout_logs.txt, the 7 held-out set files."""
import gzip, json, os, random, sys
import torch
from torch.utils.data import DataLoader, Dataset
REPO = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
HO = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout"
ho_logs = set(open(HO + "/heldout_logs.txt").read().split())
rows = []
for f in ("tg/train_grow/targets_rank0.jsonl", "tg/train_grow/targets_rank1.jsonl"):     # sorted glob order: rank0 then rank1 (train.py:TargetBank)
    with open(f, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            rows.append((r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0)), tuple(r["image"])))
print("bank rows", len(rows), "distinct (log,token,step)", len({r[:3] for r in rows}), "distinct image tuples", len({r[4] for r in rows}))
served = set()
with gzip.open(REPO + "/raw/rl_pool/served_keys.tsv.gz", "rt") as f:
    next(f)
    for l in f:
        t = l.rstrip("\n").split("\t")
        if t[6] == "0":
            served.add((t[0], int(t[1]), int(t[2])))
print("served non-held-out keys", len(served))
tr_idx = [i for i, r in enumerate(rows) if (r[1], r[2], r[3]) in served and r[0] not in ho_logs]
print("tr_idx", len(tr_idx), " served keys with no bank row:", len(served) - len({(rows[i][1], rows[i][2], rows[i][3]) for i in tr_idx}))
# held-out eval rows: keys of the 3,137 held-out sets
hkeys = set()
sd = HO + "/sets"
for fn in sorted(os.listdir(sd)):
    if not fn.endswith(".jsonl"): continue
    with open(os.path.join(sd, fn), encoding="utf-8") as fh:
        for l in fh:
            try: d = json.loads(l)
            except Exception: continue
            if d.get("kind") != "onpolicy_set": continue
            hkeys.add((d["log_name"], d.get("token", ""), int(d["step"]), int(d.get("rank", 0))))
print("held-out set keys", len(hkeys))
ho_idx = [i for i, r in enumerate(rows) if (r[0], r[1], r[2], r[3]) in hkeys]
print("ho_idx", len(ho_idx), "(sets with a bank row)", " held-out sets WITHOUT a bank row:", len(hkeys) - len(ho_idx))

class Idx(Dataset):
    def __init__(s, n): s.n = n
    def __len__(s): return s.n
    def __getitem__(s, i): return i

def draw(seed, n, updates=400, accum=8, batch=8):
    g = torch.Generator().manual_seed(seed)          # scorer_finetune.py:main -- g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(Idx(n), batch_size=batch, shuffle=True, generator=g, num_workers=0, drop_last=True)
    out = []
    for k, b in enumerate(dl):
        out += b.tolist()
        if len(out) >= updates * accum * batch: break
    return out

res = {"torch": torch.__version__, "bank_rows": len(rows), "tr_idx": len(tr_idx), "ho_sets": len(hkeys), "ho_rows": len(ho_idx)}
allimg = {}
per_seed = {}
ho_imgs = {rows[i][4][c] for i in ho_idx for c in range(4)}
for s in (0, 1, 2):
    pos = draw(s, len(tr_idx))
    sel = [tr_idx[p] for p in pos]
    imgs = {rows[i][4][c] for i in sel for c in range(4)}
    tuples = {rows[i][4] for i in sel}
    per_seed[s] = {"draws": len(pos), "distinct_rows": len(set(sel)), "distinct_frame_tuples": len(tuples), "distinct_images": len(imgs), "logs": len({rows[i][0] for i in sel}), "first5_rows": sel[:5]}
    allimg[s] = imgs
res["per_seed"] = per_seed
u3 = set().union(*allimg.values())
res["union3_images"] = len(u3); res["union3_frame_tuples"] = len(u3) // 4          # images come 4 per frame tuple (no cross-tuple sharing: per-seed tuples x 4 == images)
res["ho_images"] = len(ho_imgs); res["ho_logs"] = len({rows[i][0] for i in ho_idx})
res["union3_plus_ho_images"] = len(u3 | ho_imgs)
res["one_seed_plus_ho_images"] = {s: len(allimg[s] | ho_imgs) for s in allimg}
res["bank_distinct_images_all_rows"] = len({x for r in rows for x in r[4]})
res["bank_logs"] = len({r[0] for r in rows})
res["overlap_seed_pairs_images"] = {"0&1": len(allimg[0] & allimg[1]), "0&2": len(allimg[0] & allimg[2]), "1&2": len(allimg[1] & allimg[2])}
json.dump(res, open("sim_draws.json", "w"), indent=1)
# a stratified sample for the pod byte measurement: 40 logs from the union (+ held-out), 3 distinct frame tuples each
rnd = random.Random(20261013)
need = u3 | ho_imgs
bylog = {}
for r in rows:
    if all(x in need for x in r[4]):
        bylog.setdefault(r[0], set()).add(r[4])
logs = sorted(bylog); rnd.shuffle(logs)
samp = []
for l in logs[:40]:
    ts = sorted(bylog[l]); rnd.shuffle(ts)
    samp += [list(t) for t in ts[:3]]
json.dump(samp, open("byte_sample.json", "w"))
print(json.dumps(res, indent=1)); print("sample tuples", len(samp), "files", 4 * len(samp))
