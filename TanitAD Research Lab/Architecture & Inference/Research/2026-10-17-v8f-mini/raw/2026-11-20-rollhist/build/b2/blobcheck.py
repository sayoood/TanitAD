import hashlib, sys, os
root, lst = sys.argv[1], sys.argv[2]
bad = []; n = 0
for line in open(lst):
    h, p = line.rstrip("\n").split(" ", 1)
    fp = os.path.join(root, p)
    try:
        d = open(fp, "rb").read()
    except OSError:
        bad.append(("missing", p)); continue
    if hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest() != h:
        bad.append(("differs", p))
    n += 1
print("checked", n, "of", sum(1 for _ in open(lst)), "bad", len(bad), bad[:5])
sys.exit(1 if bad else 0)
