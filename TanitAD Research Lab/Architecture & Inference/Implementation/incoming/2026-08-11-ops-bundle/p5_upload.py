"""Pod5 -> HF staging upload (relay leg 1). Token read in place, never printed.

Shards the val corpus tar into 2000 MB pieces + uploads ckpts/join, writes
MANIFEST.json LAST (the pod4 chain polls for it). Sequential I/O only —
GPU-bound W4r training is unaffected.
"""
import hashlib
import json
import os
import subprocess
import sys

REPO = "Sayood/tanitad-flagship-v5f-w120"
PREFIX = "staging/val40-w120-256x640cyl"
WORK = "/workspace/hf_stage"
CORPUS = "/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl"
EXTRAS = [
    "/workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt",
    "/workspace/experiments/flagship-v5f-w120-30k/probe_vocab.pt",
    "/workspace/experiments/flagship-v5f-w120-30k/config.json",
    "/workspace/experiments/stage-a-predictor/ckpt_stage_a.pt",
    "/workspace/experiments/flagship_v4_anchors_dense.pt",
    "/workspace/data/p8_join/combined140.jsonl",
]


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
    from huggingface_hub import HfApi
    tok = open("/root/.cache/huggingface/token").read().strip()
    api = HfApi(token=tok)
    os.makedirs(WORK, exist_ok=True)
    shard_base = os.path.join(WORK, "corpus_shard_")
    if not os.path.exists(shard_base + "aa"):
        print("[relay] tarring+splitting corpus ...", flush=True)
        rc = subprocess.call(
            f"tar -C {os.path.dirname(CORPUS)} -cf - {os.path.basename(CORPUS)}"
            f" | split -b 2000m - {shard_base}", shell=True)
        if rc != 0:
            sys.exit(f"tar|split failed rc={rc}")
    shards = sorted(p for p in os.listdir(WORK) if p.startswith("corpus_shard_"))
    manifest = {"corpus_dirname": os.path.basename(CORPUS),
                "expected_files": 603, "shards": {}, "extras": {}}
    for s in shards:
        p = os.path.join(WORK, s)
        manifest["shards"][s] = {"md5": md5(p), "bytes": os.path.getsize(p)}
        print(f"[relay] upload {s} ({os.path.getsize(p)>>20} MiB)", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{PREFIX}/{s}",
                        repo_id=REPO)
    for p in EXTRAS:
        name = "__".join(p.split("/")[-2:])
        manifest["extras"][name] = {"src": p, "md5": md5(p),
                                    "bytes": os.path.getsize(p)}
        print(f"[relay] upload extra {name}", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{PREFIX}/{name}",
                        repo_id=REPO)
    mp = os.path.join(WORK, "MANIFEST.json")
    json.dump(manifest, open(mp, "w"), indent=1)
    api.upload_file(path_or_fileobj=mp, path_in_repo=f"{PREFIX}/MANIFEST.json",
                    repo_id=REPO)
    print("RELAY_UP_DONE", flush=True)


if __name__ == "__main__":
    main()
