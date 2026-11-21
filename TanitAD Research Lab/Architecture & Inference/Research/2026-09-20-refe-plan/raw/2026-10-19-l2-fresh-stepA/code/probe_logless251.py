"""The 251 navtrain tokens whose log the token table does not record (log == '': not served, not in train_grow).  Find their nuPlan DB by token, on every DB on this box; then test frame resolvability.  Read-only, CPU."""
import collections, glob, gzip, json, os, sqlite3, sys, time
PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TREE = "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
EXTRA = "C:/Users/Admin/p2heldout_work/dbs"
ROOTS = ["C:/Users/Admin/p2heldout_work/frames_heldout", "D:/Projects/TanitAD/data/refe_proxy/pixels", "D:/Projects/TanitAD/data/refe_m5/pixels"]
out = sys.argv[1]
def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for l in f:
            yield dict(zip(h, l.rstrip("\n").split("\t")))
toks = sorted(t for t, r in ((r["token"], r) for r in rd(PK + "/raw/rl_pool/navtrain_token_table.tsv.gz")) if r["log"] == "")
print("logless tokens", len(toks), flush=True)
dbs = {}
for pat in (DB + "/*/*.db", DB + "/*.db", EXTRA + "/*.db"):
    for p in glob.glob(pat):
        dbs.setdefault(os.path.basename(p)[:-3], p.replace("\\", "/"))
print("dbs", len(dbs), flush=True)
found, t0 = {}, time.time()
want = ",".join("X'%s'" % t for t in toks)
for i, (lg, p) in enumerate(sorted(dbs.items())):
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        for tok, ts in c.execute(f"SELECT lower(hex(token)), timestamp FROM lidar_pc WHERE token IN ({want})"):
            found[tok] = (lg, int(ts), p)
        c.close()
    except Exception as e:
        print("ERR", lg, e, flush=True)
    if i % 500 == 0:
        print("ZZ", i, len(found), f"{time.time()-t0:.0f}s", flush=True)
res = {"n_tokens": len(toks), "found_in_a_db": len(found), "logs": len({v[0] for v in found.values()})}
held = set(open(PK + "/splits/sft_heldout_logs.txt").read().split())
res["in_the_24_heldout_logs"] = sum(1 for v in found.values() if v[0] in held)
sys.path.insert(0, TREE + "/refe")
import planner as PL
fr = {r: PL.FrameResolver(db_dir="", images_root=r) for r in ROOTS if os.path.isdir(r)}
ok = []
for tok, (lg, ts, p) in found.items():
    for r, res_ in fr.items():
        res_.db_dir = os.path.dirname(p)
        if res_.resolve(lg, ts):
            ok.append({"token": tok, "log": lg, "root": r})
            break
needed = {}
r0 = next(iter(fr.values()))
okset = {x["token"] for x in ok}
for tok, (lg, ts, p_) in found.items():
    if tok in okset:
        continue
    r0.db_dir = os.path.dirname(p_)
    per = r0._index(lg)
    nf = {}
    for ch, (ts_arr, names) in per.items():
        if ts_arr.size:
            j = int(abs(ts_arr - ts).argmin())
            nf[ch] = {"filename_jpg": names[j], "dt_ms": abs(int(ts_arr[j]) - ts) / 1000.0}
    needed[tok] = {"log": lg, "ts_us": ts, "frames": nf}
res["needed_frames_tokens"] = len(needed)
res["needed_frames_jpgs"] = sum(len(v["frames"]) for v in needed.values())
json.dump({"what": "the 251 log-less navtrain tokens that have a DB on this box and no resolvable frame: the nearest filename_jpg per camera (as p2ho_audit.py lists them), for a targeted fetch", "tokens": needed}, open(out.replace(".json", "_needed_frames.json"), "w"))
res["frames_resolvable"] = len(ok)
res["frames_resolvable_logs"] = len({x["log"] for x in ok})
res["by_split_dir"] = dict(collections.Counter(os.path.basename(os.path.dirname(v[2])) for v in found.values()))
json.dump({"summary": res, "found": {t: list(v) for t, v in found.items()}, "resolvable": ok}, open(out, "w"))
print("ZZDONE", json.dumps(res), flush=True)
