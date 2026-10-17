import sys, os
a, b = sys.argv[1], sys.argv[2]
def rd(p):
    with open(p, 'rb') as f: return f.read().replace(b'\r\n', b'\n')
skip = {'__pycache__'}
out = []
fa, fb = {}, {}
for root, ds, fs in os.walk(a):
    ds[:] = [d for d in ds if d not in skip]
    for f in fs: fa[os.path.relpath(os.path.join(root, f), a)] = os.path.join(root, f)
for root, ds, fs in os.walk(b):
    ds[:] = [d for d in ds if d not in skip]
    for f in fs: fb[os.path.relpath(os.path.join(root, f), b)] = os.path.join(root, f)
for k in sorted(set(fa) | set(fb)):
    if k not in fa: out.append("ONLY-IN-B " + k)
    elif k not in fb: out.append("ONLY-IN-A " + k)
    elif rd(fa[k]) != rd(fb[k]): out.append("DIFF " + k)
print("\n".join(out)); print("n_diff", len(out))
