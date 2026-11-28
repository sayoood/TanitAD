"""wire_off.py (MM 4a3d05, 2026-11-18; Thor, CPU) -- WIRE-OFF weights, PREREG_E2WISE_2026-11-17_ADDENDUM4.md (landed 81d409d9).
Sets to exactly 0 every tensor of the run file's model state dict that is one of K2DC's two planner wires:
  * a ``box_gate`` (name ends with ``.box_gate``; zero-init gate per consuming planner / tactical layer, refcv6_tactical.py:411 "g = 0 is
    the base bit-for-bit"), and
  * ``_perception.bev_pool.map_chan_proj.*`` (the zero-init 1x1 map-channel projection added to the planner BEV pool).
Everything else is untouched. With NOTHING to zero (K0 has neither wire) the output is a byte copy of the run file (same md5).
Writes <out> and <out>.provenance.json; verifies by RE-LOADING the output: zeroed tensors all 0, every other tensor torch.equal to the run
file's, same top-level keys and non-tensor values. REFUSES if the zeroed count differs from --expect.
usage: python -I wire_off.py --run <model_ema_tail5k6k.pt> --out <model_wireoff.pt> --expect <N>"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch


def is_wire(n: str) -> bool:
    return n.endswith(".box_gate") or n.startswith("_perception.bev_pool.map_chan_proj.")


def h(path, algo):
    d = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            d.update(b)
    return d.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--expect", type=int, required=True)
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit("REFUSED: output exists: " + a.out)
ck = torch.load(a.run, map_location="cpu", weights_only=False)
assert isinstance(ck, dict) and isinstance(ck.get("model"), dict), "run file has no `model` state dict"
sd = ck["model"]
wires = sorted(n for n, v in sd.items() if torch.is_tensor(v) and is_wire(n))
if len(wires) != a.expect:
    sys.exit("REFUSED: %d wire tensors, expected %d: %s" % (len(wires), a.expect, wires))
prior = {n: {"shape": list(sd[n].shape), "max_abs": float(sd[n].float().abs().max()),
             "values": [float(x) for x in sd[n].float().reshape(-1)[:8]]} for n in wires}
if not wires:
    shutil.copyfile(a.run, a.out)
    how = "byte copy (nothing to zero)"
else:
    for n in wires:
        sd[n] = torch.zeros_like(sd[n])
    torch.save(ck, a.out)
    how = "torch.save of the run dict with %d tensors zeroed" % len(wires)
# verify by re-loading both files
o = torch.load(a.out, map_location="cpu", weights_only=False)
r = torch.load(a.run, map_location="cpu", weights_only=False)
assert sorted(o) == sorted(r), "top-level keys differ"
for k in r:
    if k != "model":
        assert (o[k] == r[k]) if not torch.is_tensor(r[k]) else torch.equal(o[k], r[k]), "top-level value differs: " + k
om, rm = o["model"], r["model"]
assert sorted(om) == sorted(rm), "state-dict keys differ"
n_eq = n_zero = 0
for n, v in rm.items():
    if not torch.is_tensor(v):
        continue
    if n in prior:
        assert torch.count_nonzero(om[n]) == 0 and om[n].dtype == v.dtype and om[n].shape == v.shape, "not zeroed: " + n
        n_zero += 1
    else:
        assert torch.equal(om[n], v), "changed: " + n
        n_eq += 1
rec = {"tool": "wire_off.py", "prereg": "e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM4.md (81d409d9)", "run": a.run, "out": a.out, "how": how,
       "n_zeroed": n_zero, "zeroed": prior, "n_untouched_bit_equal": n_eq, "all_others_bit_equal": True,
       "md5_run": h(a.run, "md5"), "md5_out": h(a.out, "md5"), "sha256_run": h(a.run, "sha256"), "sha256_out": h(a.out, "sha256"),
       "step": r.get("step"), "kind": r.get("kind")}
json.dump(rec, open(a.out + ".provenance.json", "w", encoding="utf-8"), indent=1)
print("WIRE-OFF-OK zeroed %d untouched %d md5_out %s (%s)" % (n_zero, n_eq, rec["md5_out"], how))
