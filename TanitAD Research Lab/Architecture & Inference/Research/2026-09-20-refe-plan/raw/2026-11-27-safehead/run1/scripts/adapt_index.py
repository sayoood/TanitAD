# ADAPTER (documented): data agent's index.csv 'file' column is a bare name relative to the cache root, but the npz live in <root>/<sub>/.
# This rewrites 'file' to the absolute path of the existing file (sha256 column kept; seal re-verifies it).  Never writes into the cache.
import csv, sys, os
src, sub, out = sys.argv[1:4]
root = os.path.dirname(os.path.abspath(src))
rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
for r in rows:
    f = r["file"]
    cands = [f] if os.path.isabs(f) else [os.path.join(root, f), os.path.join(root, sub, f)]
    p = next((c for c in cands if os.path.isfile(c)), None)
    if p is None: sys.exit(f"ADAPT_FAIL missing {f}")
    r["file"] = os.path.abspath(p).replace("\\", "/")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("ADAPT_OK", len(rows))
