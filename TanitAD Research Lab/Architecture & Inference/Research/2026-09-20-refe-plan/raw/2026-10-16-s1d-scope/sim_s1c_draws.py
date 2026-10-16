"""S1d scope probe 3: what did S1 and S1c's head actually SEE?  (extends raw/2026-10-13-aux-s1-data-audit/evidence/sim_draws.py)

Reconstructs the trainer's data order (scorer_finetune.py: Generator().manual_seed(seed), DataLoader shuffle, drop_last, loader batch 8) from
banked inputs and counts, for S1 (400 updates x 8 micro x 8 rows = 25,600 draws) and S1c (10,000 updates x 8 rows = 80,000 draws), seed 0:
distinct rows / frames (rank-0 and rank-1 twins share a frame) / LOGS / per-city frames, and the label-store coverage of the drawn frames.
Inputs (read-only): bank targets_rank{0,1}.jsonl (sorted glob order: rank0 then rank1), raw/rl_pool/served_keys.tsv.gz (the 174,912 PDM-labelled
train keys), heldout_logs.txt, the label store shards (headers only) and log_map.json.  Run with -I.   Same torch build as the audit => same draw."""
import gzip, json, os, sys, time
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset

PK = "C:/Users/Admin/s1d_scope/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TG = sys.argv[1]                                             # dir holding targets_rank0.jsonl + targets_rank1.jsonl
HO = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout"
STORE = "D:/Projects/TanitAD-artifacts/aux_drivable/label_store"
LOGMAP = "D:/Projects/TanitAD-artifacts/aux_drivable/log_map.json"
OUT = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/sim_s1c_draws.json"
sys.path.insert(0, PK + "/refe")
import aux_label_store as ALS  # noqa: E402  (read_shard only; header reads)

ho_logs = set(open(HO + "/heldout_logs.txt").read().split())
rows = []
for f in ("targets_rank0.jsonl", "targets_rank1.jsonl"):
    with open(os.path.join(TG, f), encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            rows.append((r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0)), tuple(r["image"])))
served = set()
with gzip.open(PK + "/raw/rl_pool/served_keys.tsv.gz", "rt") as f:
    next(f)
    for l in f:
        t = l.rstrip("\n").split("\t")
        if t[6] == "0":
            served.add((t[0], int(t[1]), int(t[2])))
tr_idx = [i for i, r in enumerate(rows) if (r[1], r[2], r[3]) in served and r[0] not in ho_logs]
print("bank rows", len(rows), "tr_idx", len(tr_idx), "(audit: 189,888 / 174,912)")
assert len(rows) == 189888 and len(tr_idx) == 174912, "inputs differ from the S1 audit: stop"

# label store: token set per log (header only) + city of every log
lm = json.load(open(LOGMAP))
lmap = lm.get("logs", lm)
def city_of(log):
    v = lmap.get(log)
    return (v.get("map") if isinstance(v, dict) else v) or "?"
store_keys = set()
t0 = time.time()
for fn in os.listdir(STORE + "/shards"):
    head, _ = ALS.read_shard(__import__("pathlib").Path(STORE + "/shards/" + fn), with_blobs=False)
    for e in head["entries"]:
        store_keys.add((head["log"] if "log" in head else fn[:-5], e["t"]))
print("store keys", len(store_keys), "read s", round(time.time() - t0, 1))

class Idx(Dataset):
    def __init__(s, n): s.n = n
    def __len__(s): return s.n
    def __getitem__(s, i): return i

def draw(seed, n, total, batch=8):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(Idx(n), batch_size=batch, shuffle=True, generator=g, num_workers=0, drop_last=True)
    out = []
    for b in dl:
        out += b.tolist()
        if len(out) >= total: break
    return out[:total]

def summarize(sel):
    frames = {(rows[i][0], rows[i][1]) for i in sel}               # a frame = (log, token); rank-0/1 twins share it
    logs = {f[0] for f in frames}
    cities = Counter(city_of(f[0]) for f in frames)
    in_store = sum(1 for f in frames if f in store_keys)
    return dict(draws=len(sel), distinct_rows=len(set(sel)), distinct_frames=len(frames), logs=len(logs),
                frames_by_city=dict(cities), logs_by_city=dict(Counter(city_of(l) for l in logs)),
                frames_with_label=in_store, frames_without_label=len(frames) - in_store)

res = {"torch": torch.__version__, "tr_idx": len(tr_idx)}
all_frames = {(rows[i][0], rows[i][1]) for i in tr_idx}
res["train_pool"] = dict(rows=len(tr_idx), frames=len(all_frames), logs=len({f[0] for f in all_frames}),
                         frames_by_city=dict(Counter(city_of(f[0]) for f in all_frames)),
                         logs_by_city=dict(Counter(city_of(l) for l in {f[0] for f in all_frames})),
                         frames_with_label=sum(1 for f in all_frames if f in store_keys))
pos80 = draw(0, len(tr_idx), 80000)
sel80 = [tr_idx[p] for p in pos80]
sel26 = sel80[:25600]
res["S1_seed0_25600"] = summarize(sel26)
res["S1c_seed0_80000"] = summarize(sel80)
# prefix property: is S1's draw literally the first 25,600 of S1c's?  (S1: loader batch 8, accum 8, 400 updates)
pos26_direct = draw(0, len(tr_idx), 25600)
res["S1_draw_is_prefix_of_S1c"] = (pos26_direct == pos80[:25600])
# the audit's number for S1 seed 0 (23,936 frame tuples / 1,133 logs) must reproduce
res["audit_check_S1_seed0"] = {"frames": res["S1_seed0_25600"]["distinct_frames"], "logs": res["S1_seed0_25600"]["logs"], "audit": [23936, 1133]}
# frames of S1c never seen at S1
f26 = {(rows[i][0], rows[i][1]) for i in sel26}
f80 = {(rows[i][0], rows[i][1]) for i in sel80}
res["S1c_new_frames_vs_S1"] = len(f80 - f26)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
