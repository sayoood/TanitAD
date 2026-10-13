import sys, tarfile, os, time
import compression.zstd as z
src, dst = sys.argv[1], sys.argv[2]
want = ("train_grow/targets_rank0.jsonl", "train_grow/targets_rank1.jsonl", "train_grow/calib_table.json")
os.makedirs(dst, exist_ok=True)
t0 = time.time()
with z.open(src, "rb") as f:
    with tarfile.open(fileobj=f, mode="r|") as tf:
        for m in tf:
            if m.name in want:
                tf.extract(m, dst); print("extracted", m.name, m.size, round(time.time()-t0,1), flush=True)
print("done", round(time.time()-t0,1))
