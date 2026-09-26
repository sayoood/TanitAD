import hashlib, os, json, sys, time
A = r"D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139"
B = r"D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
def s12(c): return hashlib.sha256(c.encode()).hexdigest()[:12]
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b''):
            h.update(ch)
    return h.hexdigest()
fa = sorted(f for f in os.listdir(A) if f.endswith('.v2ep.pt'))
fb = sorted(f for f in os.listdir(B) if f.endswith('.v2ep.pt'))
t0 = time.time()
rows = []
for f in sorted(set(fa) | set(fb)):
    cid = f[:-len('.v2ep.pt')]
    ra = md5(os.path.join(A, f)) if f in fa else None
    rb = md5(os.path.join(B, f)) if f in fb else None
    rows.append({"sha12": s12(cid), "kit_md5": ra, "local_md5": rb, "equal": (ra == rb and ra is not None)})
n_eq = sum(r["equal"] for r in rows)
out = {"kit": A, "local": B, "n_kit": len(fa), "n_local": len(fb), "n_union": len(rows),
       "n_equal": n_eq, "only_kit": [r["sha12"] for r in rows if r["local_md5"] is None],
       "only_local": [r["sha12"] for r in rows if r["kit_md5"] is None],
       "differ": [r["sha12"] for r in rows if r["kit_md5"] and r["local_md5"] and not r["equal"]],
       "wall_s": round(time.time() - t0, 1), "rows": rows}
json.dump(out, open(r"C:/Users/Admin/ev6_battery/raw/kit_checks/eval139_md5_compare.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
