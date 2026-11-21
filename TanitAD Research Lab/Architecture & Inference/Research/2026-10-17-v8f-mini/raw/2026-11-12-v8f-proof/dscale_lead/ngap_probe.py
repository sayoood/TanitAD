import json, sys, glob
root = sys.argv[1]
for run in ("runC", "runCSH", "runU"):
    for p in sorted(glob.glob(root + "/" + run + "/TK16_s*/train_log.jsonl")):
        rec = [json.loads(l) for l in open(p) if l.strip()]
        body = [r for r in rec if "n_gap" in r]
        z = [r for r in body if r["n_gap"] <= 0]
        keys = sorted(body[0].keys()) if body else []
        nk = [k for k in keys if k.startswith("n_")]
        print(run, p.split("/")[-2], "records", len(body), "n_gap==0", len(z), "steps", [r.get("step") for r in z][:8],
              "n_gap min/med/max", min(r["n_gap"] for r in body), sorted(r["n_gap"] for r in body)[len(body)//2], max(r["n_gap"] for r in body))
        if z:
            print("   zero-record n_* fields:", {k: z[0][k] for k in nk}, "first", z[0].get("first"))
            i = body.index(z[0])
            print("   neighbours:", [(r.get("step"), r["n_gap"]) for r in body[max(0, i-2):i+3]])
    print()
hdr = json.loads(open(sorted(glob.glob(root + "/runC/TK16_s0/train_log.jsonl"))[0]).readline())
print("header keys", sorted(hdr.keys())[:30])
for k in ("batch", "bs", "batch_size", "n_train", "n_windows", "steps_per_epoch", "clip_list", "log_every"):
    if k in hdr: print(" ", k, str(hdr[k])[:160])
