"""Why does native_tracks_union.pt differ in sha256 from the pod's while its size is equal? Read-only.
1. List the zip members of our file; is there a .data/serialization_id member?
2. Save the SAME loaded object twice: do the two files differ, and in which members only?
3. Content: every track and the meta of the union equal the two (sha-verified) inputs, bitwise."""
import hashlib, io, json, sys, zipfile
import torch

U = "C:/Users/Admin/lever3/merged/native_tracks_union.pt"
C = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
T = "F:/TanitAD-scratch/rollhist/lever3/labels/in/tenk/labels/native_tracks_datascale10k.pt"
OUT = "F:/TanitAD-scratch/rollhist/lever3/build/TRACKS_IDENTITY.json"


def members(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    return {i.filename: hashlib.sha256(z.read(i.filename)).hexdigest() for i in z.infolist()}


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and \
            bool(torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))) if a.numel() else (a.shape == b.shape)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    except ImportError:
        pass
    return a == b


raw = open(U, "rb").read()
mem = members(raw)
res = {"union_sha256": hashlib.sha256(raw).hexdigest(), "union_bytes": len(raw), "members": sorted(mem)}
u = torch.load(U, map_location="cpu", weights_only=False)
b1, b2 = io.BytesIO(), io.BytesIO()
torch.save(u, b1); torch.save(u, b2)
m1, m2 = members(b1.getvalue()), members(b2.getvalue())
res["resave_twice_sha_equal"] = hashlib.sha256(b1.getvalue()).hexdigest() == hashlib.sha256(b2.getvalue()).hexdigest()
res["resave_members_differing"] = sorted(k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k))
res["resave_sizes"] = [len(b1.getvalue()), len(b2.getvalue())]
# content against the inputs
c = torch.load(C, map_location="cpu", weights_only=False)
t = torch.load(T, map_location="cpu", weights_only=False)
res["types"] = [type(u).__name__, type(c).__name__, type(t).__name__]
res["top_keys"] = {"union": sorted(u)[:10] if isinstance(u, dict) else None, "corpus": sorted(c)[:10] if isinstance(c, dict) else None}


def tracks_of(o):
    return o.get("tracks", o) if isinstance(o, dict) else o


tu, tc, tt = tracks_of(u), tracks_of(c), tracks_of(t)
n_eq = n_c = n_t = 0
bad = []
for k, v in tu.items():
    src = tc.get(k) if k in tc else tt.get(k)
    if k in tc:
        n_c += 1
    elif k in tt:
        n_t += 1
    if src is not None and same(v, src):
        n_eq += 1
    else:
        bad.append(k)
res.update({"n_union_tracks": len(tu), "n_from_corpus": n_c, "n_from_tenk": n_t, "n_equal_to_input": n_eq, "n_bad": len(bad), "bad_first": bad[:5]})
print(json.dumps(res, indent=1)[:3000])
json.dump(res, open(OUT, "w"), indent=1)
