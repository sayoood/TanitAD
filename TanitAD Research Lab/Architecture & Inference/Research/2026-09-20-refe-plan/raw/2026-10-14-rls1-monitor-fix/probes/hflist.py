import os, re, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
api = HfApi(token=tok)
for repo in sys.argv[1:]:
    try:
        info = api.model_info(repo, files_metadata=True)
        print("REPO", repo, "private=", info.private)
        for s in sorted(info.siblings, key=lambda s: s.rfilename):
            print("  ", s.rfilename, s.size)
    except Exception as e:
        print("REPO", repo, "ERR", type(e).__name__, str(e)[:120])
