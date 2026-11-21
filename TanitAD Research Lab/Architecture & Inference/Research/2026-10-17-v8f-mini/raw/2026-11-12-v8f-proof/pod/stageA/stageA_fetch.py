# pod-side: fetch DINOv3-L/16 into the standard HF cache (HF_HUB_CACHE), verify safetensors sha256, write the path for the encoder.
# Token read in place from /root/.hf_token, never printed. Exit 0 only on a sha match.
import os, sys, hashlib, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
WANT = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
HR = "/workspace/v8proof/hierrep"
p = snapshot_download(REPO, token=tok)          # honours HF_HUB_CACHE
print("snapshot at", p, flush=True)
sf = os.path.join(p, "model.safetensors")
h = hashlib.sha256()
with open(sf, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        h.update(b)
got = h.hexdigest()
res = {"snapshot": p, "safetensors": os.path.realpath(sf), "bytes": os.path.getsize(sf), "sha256": got, "want": WANT, "match": got == WANT}
json.dump(res, open(f"{HR}/dinov3_sha_check.json", "w"), indent=1)
open(f"{HR}/dino_safetensors.path", "w").write(sf)
print(json.dumps(res, indent=1))
sys.exit(0 if got == WANT else 3)
