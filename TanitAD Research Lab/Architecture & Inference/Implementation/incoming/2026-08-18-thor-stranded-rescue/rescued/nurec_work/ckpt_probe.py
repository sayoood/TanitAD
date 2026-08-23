import zipfile, sys

P = "/home/nvidia/nurec_work/x/checkpoint.ckpt"
z = zipfile.ZipFile(P)
names = z.namelist()
print("n entries:", len(names))
for n in names[:20]:
    print("  ", n, z.getinfo(n).file_size)

# torch checkpoints keep the pickle in <root>/data.pkl
pk = [n for n in names if n.endswith("data.pkl")]
print("data.pkl:", pk)

import torch

obj = torch.load(P, map_location="cpu", weights_only=False, mmap=False)
print("type:", type(obj))
if isinstance(obj, dict):
    print("top keys:", list(obj.keys())[:40])


def show(o, path="", depth=0):
    import torch as T

    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{path}: dict(n={len(o)})")
        if depth < 2:
            for k in list(o.keys())[:400]:
                show(o[k], str(k), depth + 1)
    elif isinstance(o, T.Tensor):
        print(
            f"{pad}{path}: Tensor {tuple(o.shape)} {o.dtype} "
            f"min={o.float().min().item():.6g} max={o.float().max().item():.6g}"
        )
    elif isinstance(o, list):
        print(f"{pad}{path}: list(n={len(o)})")
    else:
        s = repr(o)
        print(f"{pad}{path}: {type(o).__name__} {s[:120]}")


show(obj, "<ckpt>")
