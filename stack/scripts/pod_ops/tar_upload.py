import sys
sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.keys import load_keys
load_keys()
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(path_or_fileobj="/workspace/tmp_val.tar",
                path_in_repo="comma_val.tar",
                repo_id="Sayood/tanitad-internal-data", repo_type="dataset")
print("VAL_TAR_UP", flush=True)
api.upload_file(path_or_fileobj="/workspace/tmp_train.tar",
                path_in_repo="comma_train.tar",
                repo_id="Sayood/tanitad-internal-data", repo_type="dataset")
print("TRAIN_TAR_UP", flush=True)
