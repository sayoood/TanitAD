import os, re, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    repo, fn = spec.split("::")
    p = hf_hub_download(repo, fn, token=tok, local_dir=out)
    print("OK", repo, fn, os.path.getsize(p))
