import glob, os, hashlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
tok = open(glob.glob("/root/.hf_*")[0]).read().strip()
D = "/root/geo10k/ref"
os.makedirs(D, exist_ok=True)
items = [("Sayood/tanitad-refcv8-v9-labels", f) for f in
         ("release/v9_labels_train.npz", "release/v9_labels_train.manifest.json", "inputs/lead_train.npz", "inputs/lc_train.npz", "MD5SUMS.txt")]
items += [("Sayood/tanitad-v7-training-corpus", "labels/s2_labels_v8_train.jsonl.gz"),
          ("Sayood/tanitad-context-labels-v1", "clips/clips.parquet"), ("Sayood/tanitad-context-labels-v1", "README.md")]
for repo, f in items:
    p = hf_hub_download(repo, f, repo_type="dataset", token=tok, local_dir=D + "/" + repo.split("/")[1])
    m = hashlib.md5(open(p, "rb").read()).hexdigest()
    print(repo.split("/")[1], f, os.path.getsize(p), "md5", m, flush=True)
print("ZZ GEO10K-G0 DONE ZZ")
