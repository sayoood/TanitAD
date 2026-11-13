#!/usr/bin/env python3
"""Fills the PENDING arm-file sha256 of an A5 manifest from an INDEPENDENT source (HF LFS sha256 listing, or the pod's sha256sum), after checking that the pulled file really hashes to it.
Never overwrites the input.  Reads no array: it hashes bytes only.  Fails closed: a key without a supplied sha, or a pulled file whose hash differs, aborts.
usage: python fill_manifest_sha.py --manifest manifest_a5_hb.json --shas shas.json --out manifest_a5_hb.filled.json
shas.json = {"hb_s0_a": "<64 hex>", ...}   (the LFS sha256 per pulled npz; for scores/head_pred_step0.npz the pod-side sha256sum, because it is not pushed to HF)"""
import argparse, hashlib, json, os, sys

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fill(man, shas):
    out = json.loads(json.dumps(man))
    for k, e in out["files"].items():
        if e["sha256"] != "PENDING":
            continue
        if k not in shas:
            raise SystemExit(f"REFUSED: no independent sha256 supplied for {k}")
        got = sha(e["path"])
        if got != shas[k]:
            raise SystemExit(f"REFUSED: {k} {e['path']} hashes {got}, not the supplied {shas[k]}")
        e["sha256"] = got
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--shas", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.abspath(a.out) == os.path.abspath(a.manifest):
        raise SystemExit("REFUSED: --out must not be the input manifest")
    out = fill(json.load(open(a.manifest)), json.load(open(a.shas)))
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1); fh.write("\n")
    print("filled", sum(1 for e in out["files"].values()), "files ->", a.out)
