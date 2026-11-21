"""Pod4 <- HF staging download (relay leg 2). Polls for MANIFEST.json (uploaded
LAST by pod5), downloads shards+extras, md5-verifies EVERYTHING, reassembles
the corpus, places extras at their pod5-identical paths."""
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = "Sayood/tanitad-flagship-v5f-w120"
PREFIX = "staging/val40-w120-256x640cyl"
DEST_DATA = "/workspace/data"


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    tok = open("/root/.cache/huggingface/token").read().strip()
    api = HfApi(token=tok)
    print("[relay] polling for MANIFEST.json ...", flush=True)
    while True:
        try:
            files = api.list_repo_files(REPO, token=tok)
            if f"{PREFIX}/MANIFEST.json" in files:
                break
        except Exception as e:
            print(f"[relay] poll error {type(e).__name__}: {e}", flush=True)
        time.sleep(120)
    mp = hf_hub_download(REPO, f"{PREFIX}/MANIFEST.json", token=tok,
                         force_download=True)
    man = json.load(open(mp))
    parts = []
    for s in sorted(man["shards"]):
        p = hf_hub_download(REPO, f"{PREFIX}/{s}", token=tok)
        got = md5(p)
        if got != man["shards"][s]["md5"]:
            sys.exit(f"MD5 MISMATCH shard {s}: {got}")
        print(f"[relay] shard {s} verified", flush=True)
        parts.append(p)
    os.makedirs(DEST_DATA, exist_ok=True)
    rc = subprocess.call("cat " + " ".join(parts) +
                         f" | tar -C {DEST_DATA} -xf -", shell=True)
    if rc != 0:
        sys.exit(f"untar failed rc={rc}")
    corpus = os.path.join(DEST_DATA, man["corpus_dirname"])
    n = len(os.listdir(corpus))
    if n != man["expected_files"]:
        sys.exit(f"corpus file count {n} != {man['expected_files']}")
    print(f"[relay] corpus OK ({n} files)", flush=True)
    for name, meta in man["extras"].items():
        p = hf_hub_download(REPO, f"{PREFIX}/{name}", token=tok)
        got = md5(p)
        if got != meta["md5"]:
            sys.exit(f"MD5 MISMATCH extra {name}: {got}")
        dst = meta["src"]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.check_call(["cp", p, dst])
        print(f"[relay] extra {name} -> {dst}", flush=True)
    print("RELAY_DOWN_DONE", flush=True)


if __name__ == "__main__":
    main()
