#!/usr/bin/env python3
"""Pull ONE P5-C arm (model_final.pt + model_final_sha256_<ARM>.txt) from HF into a work dir and verify the sha256 against the pod's sha file.

  hf_pull.py --arm C1 --prefix p5c/c1 --out <dir> [--repo Sayood/tanitad-refe-p5c]

* Lists the repo, resolves the two files under the prefix (case-insensitive; `model_final.pt` and `model_final_sha256_<ARM>.txt`). Anything
  ambiguous or absent is a REFUSAL listing what the repo does hold (the exact HF layout is UNVERIFIED until C0's push lands).
* Verifies the pulled file's sha256 == the pod's sha file (first 64-hex token). A mismatch deletes nothing and exits 3.
* The HF token is read IN PLACE from the repo's Keys.txt (never printed, never put on a command line); HF_HUB_DISABLE_XET=1 (xet hangs on this box).
* Offline/local staging (the dry run): --local-dir <dir holding model_final.pt + model_final_sha256_<ARM>.txt> copies nothing, it only verifies in place.
Exit: 0 ok, 2 refusal (layout), 3 sha mismatch.
"""
import argparse
import hashlib
import os
import re
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_sha_file(p):
    m = re.search(r"\b([0-9a-f]{64})\b", open(p, encoding="utf-8").read().lower())
    if not m:
        raise SystemExit(f"REFUSING: no 64-hex sha256 in {p}")
    return m.group(1)


def token():
    for k in ("E:/Projects/TanitAD/Keys.txt", "D:/Projects/TanitAD/Keys.txt"):
        if os.path.isfile(k):
            m = re.search(r"hf_[A-Za-z0-9]+", open(k, encoding="utf-8", errors="ignore").read())
            if m:
                return m.group(0)
    raise SystemExit("REFUSING: no HF token found in Keys.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Sayood/tanitad-refe-p5c")
    ap.add_argument("--local-dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shaname = f"model_final_sha256_{a.arm}.txt"
    if a.local_dir:
        ck, sf = os.path.join(a.local_dir, "model_final.pt"), os.path.join(a.local_dir, shaname)
    else:
        from huggingface_hub import HfApi, hf_hub_download
        tok = token()
        files = HfApi(token=tok).list_repo_files(a.repo)
        pre = (a.prefix or f"p5c/{a.arm.lower()}").strip("/").lower() + "/"
        pick = lambda leaf: [f for f in files if f.lower().startswith(pre) and f.lower().endswith("/" + leaf.lower())]
        m_ck, m_sf = pick("model_final.pt"), pick(shaname)
        if len(m_ck) != 1 or len(m_sf) != 1:
            raise SystemExit(f"REFUSING: under {pre} found model_final.pt x{len(m_ck)}, {shaname} x{len(m_sf)}; repo holds: {sorted(files)[:40]}")
        sf = hf_hub_download(a.repo, m_sf[0], token=tok, local_dir=a.out)
        ck = hf_hub_download(a.repo, m_ck[0], token=tok, local_dir=a.out)
        # hf_hub_download(local_dir=) keeps the repo sub-path (<out>/p5c/c1/model_final.pt); the chain reads <out>/model_final.pt and <out>/<shaname>.
        # MEASURED 2026-11-19 on the first real run: both pulls verified, then arms_verified refused "missing model". Move both files up (same drive).
        flat = []
        for src, leaf in ((sf, shaname), (ck, "model_final.pt")):
            dst = os.path.join(a.out, leaf)
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)
            flat.append(dst)
        sf, ck = flat
    want, got = read_sha_file(sf), sha256_file(ck)
    ok = want == got
    print(f"ZZP5C_HF_PULL {a.arm} {'OK' if ok else 'SHA-MISMATCH'} sha256 {got[:12]} (pod file {want[:12]}) size {os.path.getsize(ck)} path {ck}")
    with open(os.path.join(a.out, f"pulled_{a.arm}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{ck}\n{got}\n")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
