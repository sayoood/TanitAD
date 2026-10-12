#!/usr/bin/env python3
"""Is the on-policy dump's generator (ckpt_last.pt at step 10,075) the SAME generator as model_final.pt?  (RL-S1 priority 4 / pool measurement.)

Compares every tensor of the partial checkpoint's `model_partial` (the trainable LoRA + heads) with the same-named tensor in the full model's
`model`, bit for bit (torch.equal), and the frozen-trunk fingerprints both files record.  The frozen trunk is not loaded: both files carry
`meta.frozen_sha256`, and a partial can only be loaded onto a trunk with that fingerprint (ckpt_io.load_partial refuses otherwise).
    python eval/rl_ckpt_equality.py --partial <ckpt_last.pt> --full <model_final.pt> --out raw/rl_pool/ckpt_last_vs_model_final.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import torch


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def compare(partial, full):
    a = torch.load(partial, map_location="cpu", weights_only=False, mmap=True)
    b = torch.load(full, map_location="cpu", weights_only=False, mmap=True)
    pa, fb = a["model_partial"], (b["model"] if "model" in b else b)
    ident = diff = missing = 0
    maxd = 0.0
    for k, v in pa.items():
        if k not in fb:
            missing += 1
        elif torch.equal(v, fb[k]):
            ident += 1
        else:
            diff += 1
            maxd = max(maxd, float((v.float() - fb[k].float()).abs().max()))
    fa = (a.get("meta") or {}).get("frozen_sha256")
    fm = (b.get("meta") or {}).get("frozen_sha256")
    return {"partial_tensors": len(pa), "identical": ident, "different": diff, "missing_in_full": missing, "max_abs_diff": maxd,
            "partial_step": int(a["state"]["step"]), "full_meta_step": (b.get("meta") or {}).get("step"),
            "frozen_sha256_partial": fa, "frozen_sha256_full": fm, "frozen_fingerprints_equal": bool(fa and fa == fm),
            "same_generator": bool(diff == 0 and missing == 0 and fa and fa == fm)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = compare(a.partial, a.full)
    r["partial_md5"], r["full_md5"] = md5(a.partial), md5(a.full)
    json.dump(r, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(json.dumps(r, indent=1))
    return 0 if r["same_generator"] else 1


if __name__ == "__main__":
    sys.exit(main())
