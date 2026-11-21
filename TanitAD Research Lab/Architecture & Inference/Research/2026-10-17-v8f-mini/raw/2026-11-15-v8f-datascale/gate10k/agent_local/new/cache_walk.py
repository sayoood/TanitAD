"""C-CACHE walk (pod, CPU + disk only): recompute the md5 of every clip file of the union view and compare with the union manifest's recorded md5.
usage: cache_walk.py <union_manifest.jsonl> <view_dir_with_clips/> <out.json> [--threads N]
A union-view symlink is followed (os.path.realpath is recorded). Output: {n, n_equal, mismatches[:50], absent[:50], bytes, wall_s, expected_sha256_of_pairs}."""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
man, view, out = sys.argv[1], sys.argv[2], sys.argv[3]
nthr = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 6
rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
t0 = time.time()
def one(r):
    p = os.path.join(view, os.path.basename(r["file"]))
    if not os.path.exists(p):
        return r["clip"], None, "absent", 0
    h = hashlib.md5()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b); n += len(b)
    return r["clip"], h.hexdigest(), os.path.realpath(p), n
res, tot = {}, 0
with ThreadPoolExecutor(nthr) as ex:
    for i, (c, md5, rp, n) in enumerate(ex.map(one, rows)):
        res[c] = (md5, rp); tot += n
        if i % 500 == 0:
            print(i, round(time.time() - t0), "s", round(tot / 1e9, 1), "GB", flush=True)
exp = {r["clip"]: r["md5"] for r in rows}
mism = [c for c, (m, _) in res.items() if m is not None and m != exp[c]]
absent = [c for c, (m, _) in res.items() if m is None]
rec = {"n": len(rows), "n_equal": sum(1 for c, (m, _) in res.items() if m == exp[c]), "mismatches": mism[:50], "absent": absent[:50], "bytes": tot, "wall_s": round(time.time() - t0, 1),
       "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
       "actual_sha256": hashlib.sha256(json.dumps({c: m for c, (m, _) in sorted(res.items())}).encode()).hexdigest(), "pass": not mism and not absent}
json.dump(rec, open(out, "w"), indent=1)
print("CACHE-WALK-DONE", json.dumps({k: v for k, v in rec.items() if k not in ("mismatches", "absent")}), flush=True)
