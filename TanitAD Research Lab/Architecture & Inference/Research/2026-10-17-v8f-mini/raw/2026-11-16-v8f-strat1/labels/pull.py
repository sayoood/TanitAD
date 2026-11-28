import os,re,json,hashlib
os.environ["HF_HUB_DISABLE_XET"]="1"
tok=re.search(r'hf_[A-Za-z0-9]+',open('E:/Projects/TanitAD/Keys.txt',encoding='utf-8',errors='ignore').read()).group(0)
from huggingface_hub import HfApi, snapshot_download
api=HfApi(token=tok)
info=api.dataset_info("Sayood/tanitad-context-labels-v1",revision="100f03da",files_metadata=True)
print(info.sha)
for s in info.siblings: print(s.rfilename,s.size)
p=snapshot_download("Sayood/tanitad-context-labels-v1",repo_type="dataset",revision="100f03da",token=tok,local_dir="F:/TanitAD-scratch/strat1/labels/rel")
print(p)
