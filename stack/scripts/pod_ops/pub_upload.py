import sys
sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.keys import load_keys
load_keys()
from huggingface_hub import HfApi

REPO = "Sayood/tanitad-comma2k19-episodes"
README = """---
license: mit
---
# TanitAD comma2k19 episode caches

Preprocessed episode tensors derived from [comma2k19](https://github.com/commaai/comma2k19)
(comma.ai, MIT license — attribution to the original authors). Each episode: uint8 frame
stacks [T,9,256,256] (3 RGB frames @100ms, focal-canonicalized to f_eff=266px), real CAN
actions [T,2] (steer rad, accel m/s2), ego poses [T,4] (x,y,yaw,v) at 10 Hz, route-level
train/val split. Built by the TanitAD project (https://github.com/sayoood/TanitAD) —
`stack/tanitad/data/comma2k19.py`. Tars unpack to per-episode `ep_*.pt` (torch.load).
"""

api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
print("repo + README ready", flush=True)
api.upload_file(path_or_fileobj="/workspace/tmp_val.tar", path_in_repo="comma_val.tar",
                repo_id=REPO, repo_type="dataset")
print("VAL_UP", flush=True)
api.upload_file(path_or_fileobj="/workspace/tmp_train.tar", path_in_repo="comma_train.tar",
                repo_id=REPO, repo_type="dataset")
print("TRAIN_UP", flush=True)
print("ALL_DONE", flush=True)
