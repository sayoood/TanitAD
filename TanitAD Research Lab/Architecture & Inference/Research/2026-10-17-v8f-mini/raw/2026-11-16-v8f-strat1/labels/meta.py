import os,re
os.environ["HF_HUB_DISABLE_XET"]="1"
tok=re.search(r'hf_[A-Za-z0-9]+',open('E:/Projects/TanitAD/Keys.txt',encoding='utf-8',errors='ignore').read()).group(0)
from huggingface_hub import hf_hub_download
p=hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles","metadata/data_collection.parquet",repo_type="dataset",token=tok,local_dir="F:/TanitAD-scratch/strat1/labels/meta_dl")
import pandas as pd
d=pd.read_parquet(p);print(p,d.shape,os.path.getsize(p));print(d.dtypes);print(d.head(3).T)
