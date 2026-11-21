import json, os, sys
d = json.load(open(sys.argv[1]))
a = d["argv"] if isinstance(d, dict) else d
bad = 0
for i, t in enumerate(a):
    if t.startswith("/") and "," not in t:
        ok = os.path.exists(t)
        sz = os.path.getsize(t) if ok and os.path.isfile(t) else -1
        print(("OK  " if ok else "MISSING ") + a[i - 1] + " " + t + (" size=%d" % sz if sz >= 0 else ""))
        bad += (not ok)
    elif t.startswith("/") and "," in t:
        for p in t.split(","):
            ok = os.path.exists(p)
            print(("OK  " if ok else "MISSING ") + a[i - 1] + " " + p)
            bad += (not ok)
print("N_MISSING", bad)
