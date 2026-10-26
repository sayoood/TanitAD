"""Stream A (RG note): count parameters per name prefix from safetensors HEADERS fetched with HTTP Range (no weight download).
Usage: python3 rg_audio_tower_params.py <hf_repo> <prefix> [<prefix> ...]   (result of the 2026-10-25 run: rg_audio_tower_params_result.json)
"""
import json, struct, subprocess, sys, collections
def hdr(repo, fn):
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    b = subprocess.run(["curl", "-sS", "-L", "-r", "0-7", url], capture_output=True).stdout
    n = struct.unpack("<Q", b[:8])[0]
    h = subprocess.run(["curl", "-sS", "-L", "-r", f"8-{8+n-1}", url], capture_output=True).stdout
    return json.loads(h)
def files(repo):
    r = subprocess.run(["curl", "-sS", "-L", f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"], capture_output=True).stdout
    try: return sorted(set(json.loads(r)["weight_map"].values()))
    except Exception: return ["model.safetensors"]
repo = sys.argv[1]; prefixes = sys.argv[2:]
cnt = collections.Counter(); tot = 0
for fn in files(repo):
    for k, v in hdr(repo, fn).items():
        if k == "__metadata__": continue
        n = 1
        for s in v["shape"]: n *= s
        tot += n
        for p in prefixes:
            if k.startswith(p): cnt[p] += n
print(repo, "total", tot, dict(cnt))
