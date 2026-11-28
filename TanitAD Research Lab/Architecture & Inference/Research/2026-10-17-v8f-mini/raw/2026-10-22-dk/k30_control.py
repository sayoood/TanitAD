import sys, numpy as np, hashlib
a = np.load(sys.argv[1], allow_pickle=False); b = np.load(sys.argv[2], allow_pickle=False)
print("k10 keys", sorted(a.files)); print("k30 keys", sorted(b.files))
for f in (sys.argv[1], sys.argv[2]):
    print(f[-30:], hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
print("ts k10", a["ts_rel_s"][:3], a["ts_rel_s"].shape, "ts k30", b["ts_rel_s"][:3], b["ts_rel_s"].shape)
ka = {(str(c), int(fr)): i for i, (c, fr) in enumerate(zip(a["clip_id"], a["frame"]))}
kb = {(str(c), int(fr)): i for i, (c, fr) in enumerate(zip(b["clip_id"], b["frame"]))}
print("rows", len(ka), len(kb), "same keys", set(ka) == set(kb))
sa, sb = a["state"], b["state"]
both = [k for k in ka if k in kb and str(sa[ka[k]]) == "LEAD" and str(sb[kb[k]]) == "LEAD"]
print("LEAD in both", len(both))
ia = np.array([ka[k] for k in both]); ib = np.array([kb[k] for k in both])
la, lb = a["leads"][ia], b["leads"][ib]
print("leads shapes", a["leads"].shape, b["leads"].shape)
K = a["leads"].shape[1]
d = la[:, :K] - lb[:, :K]
print("first-K steps identical:", bool(np.array_equal(la[:, :K], lb[:, :K], equal_nan=True)), "max|d|", np.nanmax(np.abs(d)) if np.isfinite(d).any() else "allnan")
print("lead_lens identical:", bool(np.array_equal(a["lead_lens"][ia], b["lead_lens"][ib], equal_nan=True)))
print("speeds identical (all rows):", bool(np.array_equal(a["speeds"][[ka[k] for k in ka]], b["speeds"][[kb[k] for k in ka]])))
from collections import Counter
ch = Counter((str(sa[ka[k]]), str(sb[kb[k]])) for k in ka if k in kb)
print("state transitions k10->k30", {k: v for k, v in ch.items() if k[0] != k[1]})
