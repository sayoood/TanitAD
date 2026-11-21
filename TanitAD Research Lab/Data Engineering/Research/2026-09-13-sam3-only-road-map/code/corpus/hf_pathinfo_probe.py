import json
from huggingface_hub import HfApi
api = HfApi(); repo = "Sayood/tanitad-v7-training-corpus"
tab = json.load(open("/home/nvidia/sam3map/corpus/hfmeta/camera/camera_sha256.json"))
c = next(iter(tab))
for f in api.get_paths_info(repo, [f"camera/{c}.mp4", "labels/s2_labels_v8_train.jsonl.gz", "DATACARD.md"], repo_type="dataset", expand=True):
    lfs = getattr(f, "lfs", None)
    print(f.path, "| size", f.size, "| lfs.sha256", (lfs.sha256[:16] if lfs else None), "| xet", (getattr(f, "xet_hash", None) or "")[:12], "| last_commit", (f.last_commit.oid[:10] if getattr(f, "last_commit", None) else None))
print("table sha256", tab[c]["sha256"][:16])
import huggingface_hub; print("hub", huggingface_hub.__version__)
