import msgpack, sys, io

PATH = "/home/nvidia/nurec_work/x/volume.msgpack"

with open(PATH, "rb") as f:
    raw = f.read()
print("raw bytes:", len(raw))

obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
print("top type:", type(obj))


def desc(v):
    if isinstance(v, dict):
        return f"dict(n={len(v)})"
    if isinstance(v, list):
        inner = ""
        if v:
            t0 = type(v[0]).__name__
            same = all(type(x).__name__ == t0 for x in v[:50])
            inner = f" of {t0}{'' if same else '(mixed)'}"
        return f"list(n={len(v)}){inner}"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes(len={len(v)})"
    if isinstance(v, str):
        s = v if len(v) < 80 else v[:77] + "..."
        return f"str {s!r}"
    return f"{type(v).__name__} {v!r}"


def walk(v, path="", depth=0, maxdepth=8):
    pad = "  " * depth
    print(f"{pad}{path}: {desc(v)}")
    if depth >= maxdepth:
        return
    if isinstance(v, dict):
        for k in list(v.keys()):
            walk(v[k], f"{k}", depth + 1, maxdepth)
    elif isinstance(v, list):
        # only descend into first element to keep output bounded
        if v and isinstance(v[0], (dict, list, bytes, bytearray)):
            walk(v[0], "[0]", depth + 1, maxdepth)


walk(obj, "<root>", 0, 4)
