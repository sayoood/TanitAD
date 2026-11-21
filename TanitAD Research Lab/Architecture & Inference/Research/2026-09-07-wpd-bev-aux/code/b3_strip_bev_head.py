# -*- coding: utf-8 -*-
"""WP-D step 3 -- strip the training-only BEV aux head out of D1's checkpoint.

The pre-registration's central design claim is that the head is **removable with
BIT-IDENTICAL planner output** (`test_planner_output_bit_identical`).  For the
section 5B planner arm that claim stops being a unit test and becomes the actual
procedure: D0 and D1 must load through the SAME rebuilt model, so that any
difference the four families see is a difference in the TRUNK'S WEIGHTS and not
a difference in what was constructed.

⛔ This does NOT modify anything on Thor and does not touch the original file.
It writes a new checkpoint and reports, by CONTENT, exactly which keys it removed
and that every remaining key is byte-identical to the source.
"""
import argparse
import hashlib
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--out-json", required=True)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model"]
drop = sorted(k for k in sd if "bev_aux" in k)
keep = {k: v for k, v in sd.items() if k not in drop}
print(f"[strip] {len(sd)} -> {len(keep)} keys, dropping {len(drop)}: {drop}")

# ⛔ CONTENT assertion: every kept tensor must be the SAME OBJECT VALUE as the
# source. A strip that silently re-cast or re-ordered would make D0/D1 differ in
# something other than training.
same = all(torch.equal(keep[k], sd[k]) for k in keep)
n_par = sum(int(v.numel()) for v in keep.values() if hasattr(v, "numel"))
n_drop = sum(int(sd[k].numel()) for k in drop)
print(f"[strip] kept params {n_par}  dropped params {n_drop}  identical={same}")
assert same

torch.save({"model": keep, "step": ck["step"]}, a.dst)
rec = {"_evidence_class": "MEASURED (ours; artifact = this file + the two ckpts)",
       "src": a.src, "dst": a.dst,
       "src_md5": hashlib.md5(open(a.src, "rb").read()).hexdigest(),
       "dst_md5": hashlib.md5(open(a.dst, "rb").read()).hexdigest(),
       "n_keys_src": len(sd), "n_keys_dst": len(keep),
       "dropped_keys": drop, "dropped_params": n_drop, "kept_params": n_par,
       "kept_tensors_bit_identical_to_src": bool(same),
       "step": int(ck["step"]),
       "why": ("PREREG section 3: the BEV aux head is TRAINING-ONLY and its removal "
               "leaves the planner's output bit-identical. For the section 5B arm "
               "D0 and D1 must be rebuilt by the SAME code path, so the head is "
               "removed here rather than tolerated by a non-strict load.")}
json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1)
print("WROTE", a.dst, os.path.getsize(a.dst), "bytes; json ->", a.out_json)
