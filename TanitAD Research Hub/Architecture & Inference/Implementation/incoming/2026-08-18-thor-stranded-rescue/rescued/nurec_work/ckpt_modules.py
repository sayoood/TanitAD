"""Probe 2: the .ckpt is a torch zip; its pickle carries MODULE PATHS (GLOBAL opcodes)
and hyper_parameters. Those name the code that implements the ppisp."""
import zipfile, pickletools, io, re, sys

CK = "/home/nvidia/nurec_work/x/checkpoint.ckpt"
z = zipfile.ZipFile(CK)
names = z.namelist()
print("members (first 20):", names[:20], "...n=", len(names))
pk = [n for n in names if n.endswith("data.pkl") or n.endswith(".pkl")]
print("pickles:", pk)

mods = set()
for p in pk:
    buf = z.read(p)
    try:
        for op, arg, pos in pickletools.genops(io.BytesIO(buf)):
            if op.name in ("GLOBAL", "STACK_GLOBAL", "SHORT_BINUNICODE", "BINUNICODE"):
                if isinstance(arg, str):
                    mods.add(arg)
    except Exception as e:
        print("genops stopped:", e)

pat = re.compile(r"ppisp|isp|post_process|vignet|crf|expos|colou?r|nurec|nre|3dg", re.I)
print("\n=== strings in the pickle matching ISP/module pattern ===")
for m in sorted(mods):
    if pat.search(m) and len(m) < 200:
        print("  ", m)

print("\n=== all dotted module-ish strings ===")
for m in sorted(mods):
    if "." in m and " " not in m and len(m) < 120 and not m.startswith("."):
        print("  ", m)
