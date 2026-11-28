#!/usr/bin/env python3
"""Pull ONE REFe checkpoint from HF and verify its sha256 against an EXPECTED value given on the command line (registry 14.6).

    ens_pull.py --hf-path p5c/c1r/model_final.pt --expect-sha256 <64 hex> --out <dir> [--repo Sayood/tanitad-refe-p5c]

* C1r has no sha file on HF (its pod egress pushed none; c1r/read/HANDOFF.md:9), so the expected value is the registry's
  (MODEL_REGISTRY 14.6: C1 f09949e2..., C1r 6e39a57c...), passed explicitly. A mismatch exits 3 and keeps the file for inspection.
* The HF token is read IN PLACE from Keys.txt with a regex (never printed, never on a command line); HF_HUB_DISABLE_XET=1.
* An existing <out>/model_final.pt with the right sha256 is accepted without a download (prints SKIP).
* Prints ZZENS_HF_PULL <hf-path> OK|MISMATCH sha256=<got> bytes=<n>. Exit 0 ok, 2 refusal, 3 mismatch.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def token():
    for k in ("E:/Projects/TanitAD/Keys.txt", "D:/Projects/TanitAD/Keys.txt"):
        if os.path.isfile(k):
            m = re.search(r"hf_[A-Za-z0-9]+", open(k, encoding="utf-8", errors="ignore").read())
            if m:
                return m.group(0)
    raise SystemExit("REFUSING: no HF token found in Keys.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Sayood/tanitad-refe-p5c")
    a = ap.parse_args()
    want = a.expect_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        print("REFUSING: --expect-sha256 is not 64 hex")
        return 2
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "model_final.pt")
    if os.path.isfile(dst):
        got = sha256_file(dst)
        if got == want:
            print(f"ZZENS_HF_PULL {a.hf_path} OK sha256={got} bytes={os.path.getsize(dst)} SKIP (already present)", flush=True)
            return 0
        print(f"REFUSING: {dst} exists with sha256 {got} != expected (move it aside)")
        return 2
    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    files = HfApi(token=tok).list_repo_files(a.repo)
    m = [f for f in files if f.lower() == a.hf_path.lower()]
    if len(m) != 1:
        print(f"REFUSING: {a.hf_path} resolves to {m} in {a.repo}")
        return 2
    p = hf_hub_download(a.repo, m[0], token=tok, local_dir=a.out)
    if os.path.abspath(p) != os.path.abspath(dst):
        os.replace(p, dst)
    cache = os.path.join(a.out, ".cache")
    sub = os.path.join(a.out, m[0].split("/")[0])
    for d in (cache, sub):                                   # hf_hub_download's own bookkeeping under the out dir (re-creatable)
        if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(a.out):
            shutil.rmtree(d, ignore_errors=True)
    got = sha256_file(dst)
    ok = got == want
    with open(os.path.join(a.out, "pulled_sha256.txt"), "w", newline="\n") as f:
        f.write(f"{got}  model_final.pt  (HF {a.repo} {m[0]}; expected {want}; {'OK' if ok else 'MISMATCH'})\n")
    print(f"ZZENS_HF_PULL {a.hf_path} {'OK' if ok else 'MISMATCH'} sha256={got} bytes={os.path.getsize(dst)}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
