import sys, tarfile, time
import compression.zstd as z
src = sys.argv[1]
t0 = time.time(); n = 0
with z.open(src, "rb") as f:
    with tarfile.open(fileobj=f, mode="r|") as tf:
        for m in tf:
            n += 1
            if n <= 40: print(m.name, m.size, flush=True)
print("members", n, "secs", round(time.time()-t0,1))
