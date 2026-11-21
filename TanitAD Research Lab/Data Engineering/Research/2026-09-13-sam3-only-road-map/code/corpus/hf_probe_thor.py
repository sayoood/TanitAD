from huggingface_hub import HfApi
import collections
api = HfApi()
me = api.whoami()
print("user:", me.get("name"), "| orgs:", [o.get("name") for o in me.get("orgs", [])])
repo = "Sayood/tanitad-v7-training-corpus"
info = api.repo_info(repo, repo_type="dataset")
print("private:", info.private, "| sha:", info.sha)
for t in api.list_repo_tree(repo, repo_type="dataset"):
    print("TOP", t.path, type(t).__name__)
cnt = collections.Counter(); ex = {}
for t in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="camera", recursive=True):
    if type(t).__name__ == "RepoFile":
        k = "/".join(t.path.split("/")[:-1]) + "/*" + (t.path.rsplit(".", 1)[-1] if "." in t.path else "")
        cnt[k] += 1; ex.setdefault(k, (t.path, t.size))
for k, v in cnt.items():
    print("CAM", k, v, "e.g.", ex[k])
