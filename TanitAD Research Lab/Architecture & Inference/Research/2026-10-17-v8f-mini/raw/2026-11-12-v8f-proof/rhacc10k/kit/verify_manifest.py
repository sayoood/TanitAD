import hashlib, os, sys
# usage: verify_manifest.py <manifest sha256sum format> <base dir>
man, base = sys.argv[1], sys.argv[2]
bad = n = 0
for line in open(man, encoding="utf-8"):
    line = line.rstrip("\r\n")
    if not line:
        continue
    h, p = line.split("  ", 1)
    p = p.lstrip("*")
    fp = os.path.join(base, p)
    n += 1
    try:
        got = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    except OSError:
        got = "MISSING"
    if got != h:
        bad += 1
        print("MISMATCH", p)
print("files", n, "mismatch", bad)
sys.exit(1 if bad else 0)
