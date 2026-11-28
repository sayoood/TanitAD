"""S1d: build the REGISTERED log subsets for the dose-response arms (S1d-q50, S1d-q25) as `--heldout-logs` files for the UNCHANGED trainer.

Mechanism (MEASURED from source, scorer_finetune.py:663-666): tr_idx drops every bank row whose log is in --heldout-logs, while ho_idx (the eval set) comes from the --heldout
SETS dir, independent of that file.  So passing  (the 24 real held-out logs) U (the train logs to DROP)  shrinks the training pool and leaves the evaluation set byte-identical.
Selection: per city (map), logs ordered by sha256(SEED|log); keep the first round(q * n_city).  NESTED: q25 is a subset of q50 (same order, shorter prefix), so the design is monotone.
Pool = the 1,167 logs of S1/S1c's tr_idx (sim_s1c_draws.py).  Checks written to the json: kept U real_heldout = {}, kept U navtest logs = {}, the file contains all 24 held-out logs,
rows/frames kept, and the --epochs the arm needs so that min(ceil(rows/8*epochs), 10000) == 10000 (the TL-1 silent-early-stop guard).   Run with -I.
usage: make_dose_logs.py <dir with targets_rank0.jsonl + targets_rank1.jsonl> <out dir>"""
import gzip, hashlib, json, math, os, sys
from collections import Counter, defaultdict
PK = "C:/Users/Admin/s1d_scope/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TG, OUT = sys.argv[1], sys.argv[2]
HO = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/heldout_logs.txt"
LOGMAP = "D:/Projects/TanitAD-artifacts/aux_drivable/log_map.json"
NAVTEST = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"
SEED = "20261016"
LEVELS = {"q50": 0.50, "q25": 0.25}
os.makedirs(OUT, exist_ok=True)

ho_text = open(HO, encoding="utf-8").read()
ho_logs = ho_text.split()
rows = []
for f in ("targets_rank0.jsonl", "targets_rank1.jsonl"):
    with open(os.path.join(TG, f), encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            rows.append((r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0))))
served = set()
with gzip.open(PK + "/raw/rl_pool/served_keys.tsv.gz", "rt") as f:
    next(f)
    for l in f:
        t = l.rstrip("\n").split("\t")
        if t[6] == "0":
            served.add((t[0], int(t[1]), int(t[2])))
hs = set(ho_logs)
tr = [r for r in rows if (r[1], r[2], r[3]) in served and r[0] not in hs]
assert len(rows) == 189888 and len(tr) == 174912, (len(rows), len(tr))
lm = json.load(open(LOGMAP))["logs"]
city = lambda l: lm[l]["map"]
rows_by_log = Counter(r[0] for r in tr)
frames_by_log = defaultdict(set)
for r in tr: frames_by_log[r[0]].add(r[1])
pool = sorted(rows_by_log)
by_city = defaultdict(list)
for l in pool: by_city[city(l)].append(l)
order = {c: sorted(ls, key=lambda l: hashlib.sha256(f"{SEED}|{l}".encode()).hexdigest()) for c, ls in by_city.items()}
nt_logs = set(json.load(open(NAVTEST))["token_to_log_name"].values())

res = {"seed": SEED, "pool_logs": len(pool), "pool_rows": len(tr), "pool_frames": sum(len(v) for v in frames_by_log.values()),
       "pool_logs_by_city": {c: len(v) for c, v in by_city.items()}, "real_heldout_logs": len(ho_logs),
       "heldout_logs_file_sha256_real": hashlib.sha256(ho_text.encode()).hexdigest(), "levels": {}}
for name, q in LEVELS.items():
    kept = []
    for c, ls in order.items():
        kept += ls[:int(round(q * len(ls)))]
    kset = set(kept)
    drop = sorted(set(pool) - kset)
    fileset = list(ho_logs) + drop                                     # the real 24 first (their order is the original file's), then the dropped train logs
    body = "\n".join(fileset) + "\n"
    p = os.path.join(OUT, f"heldout_logs_S1D_{name}.txt")
    open(p, "w", encoding="utf-8", newline="\n").write(body)
    kr = sum(rows_by_log[l] for l in kset); kf = sum(len(frames_by_log[l]) for l in kset)
    ep_need = math.ceil((10000 * 8 / kr) * 100) / 100                 # smallest 2-decimal epochs with rows/8*epochs >= 10000 updates
    ep = max(1.0, ep_need)
    res["levels"][name] = dict(
        q=q, kept_logs=len(kset), kept_logs_by_city={c: sum(1 for l in kset if city(l) == c) for c in by_city}, dropped_logs=len(drop),
        kept_rows=kr, kept_frames=kf, kept_row_share=round(kr / len(tr), 4), kept_frame_share=round(kf / res["pool_frames"], 4),
        kept_frames_by_city={c: sum(len(frames_by_log[l]) for l in kset if city(l) == c) for c in by_city},
        draws_per_row_at_80000=round(80000 / kr, 3), epochs_needed_min=ep_need, epochs_to_pin=ep,
        updates_at_pinned_epochs=min(math.ceil(kr / 8 * ep), 10000),
        updates_if_epochs_left_at_1_0=min(math.ceil(kr / 8 * 1.0), 10000),
        file=p, file_sha256=hashlib.sha256(body.encode()).hexdigest(), file_lines=len(fileset),
        checks=dict(kept_inter_real_heldout=len(kset & hs), kept_inter_navtest_logs=len(kset & nt_logs), file_contains_all_real_heldout=hs <= set(fileset),
                    train_rows_in_file_logs=sum(rows_by_log[l] for l in kset if l in set(fileset))))
    res["levels"][name]["_kept_list_sha256"] = hashlib.sha256("\n".join(sorted(kset)).encode()).hexdigest()
    open(os.path.join(OUT, f"kept_logs_S1D_{name}.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(sorted(kset)) + "\n")
res["nested_q25_subset_of_q50"] = set(open(os.path.join(OUT, "kept_logs_S1D_q25.txt")).read().split()) <= set(open(os.path.join(OUT, "kept_logs_S1D_q50.txt")).read().split())
json.dump(res, open(os.path.join(OUT, "dose_logs.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
