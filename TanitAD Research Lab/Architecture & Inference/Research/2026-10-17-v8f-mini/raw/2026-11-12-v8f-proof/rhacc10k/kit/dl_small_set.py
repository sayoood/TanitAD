import os, re, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from huggingface_hub import HfApi, hf_hub_download
txt = open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()
tok = re.search(r"hf_[A-Za-z0-9]+", txt).group(0)
api = HfApi(token=tok)
user = api.whoami()["name"]
repo = user + "/tanitad-v8proof-fork-bank"
out = "F:/TanitAD-scratch/v8proof/rhacc10k/tmp_dl"
os.makedirs(out, exist_ok=True)
files = api.list_repo_files(repo, repo_type="dataset")
print("repo files:", [f for f in files if not f.startswith("integ10k_t1_part")])
for f in ("MANIFEST_FILES.txt", "small_set.tar"):
    if f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=out, token=tok)
        print("got", p, os.path.getsize(p))
# RH repo listing
rr = user + "/tanitad-v8f-rollhist-runs"
try:
    print("rollhist files:", [f for f in api.list_repo_files(rr, repo_type="dataset") if "ckpt_last" in f])
except Exception as e:
    print("rollhist list failed", type(e).__name__)
