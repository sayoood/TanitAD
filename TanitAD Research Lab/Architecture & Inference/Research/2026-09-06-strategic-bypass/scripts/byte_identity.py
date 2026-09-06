# -*- coding: utf-8 -*-
"""OFF-IS-BYTE-IDENTICAL PROOF (P3).

Compares a PRE-PATCH tree against the POST-PATCH tree with the flag OFF. Both
trees run the SAME `dump_forward.py` (md5-checked) in their own interpreter, so
neither can import the other's modules. A single bitwise difference fails.

Usage:  python byte_identity.py BASE.pt PATCHED.pt
ASCII-only prints (cp1252 dev box).
"""
import sys

import torch

a = torch.load(sys.argv[1], weights_only=True)
b = torch.load(sys.argv[2], weights_only=True)
ka, kb = set(a), set(b)

print(f"keys only in BASE    : {len(ka - kb)}")
print(f"keys only in PATCHED : {len(kb - ka)}")

diff = []
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        diff.append((k, "SHAPE"))
    elif not torch.equal(x, y):
        diff.append((k, float((x.float() - y.float()).abs().max())))

sd = [k for k in ka & kb if k.startswith("sd/")]
fw = [k for k in ka & kb if not k.startswith("sd/")]
print(f"compared             : {len(ka & kb)} tensors "
      f"({len(sd)} state_dict entries, {len(fw)} forward outputs "
      f"over 4 nav_cmd values)")
print(f"BITWISE DIFFERENT    : {len(diff)}")
for k, v in diff[:20]:
    print("   ", k, v)

ok = (not diff) and ka == kb
print()
print(f"P3 OFF-is-byte-identical: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
