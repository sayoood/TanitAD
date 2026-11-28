"""strat1_bank_pod_s1d.py: bank the 4 STRAT1 S1d arms (S1d_s0, s1, s10, s11; commit 4dedac45) PRIVATE on HF (run ON the pod; reads the token in place, never prints it).
Writes BANK_MANIFEST.json per arm, uploads <tag>/<file>, verifies the hub's LFS sha256 against the manifest."""
import hashlib, json, os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
R = "/root/strat1/runs"
TAGS = ["S1d_s0", "S1d_s1", "S1d_s10", "S1d_s11"]
FILES = {"ckpt_last.pt": "train/ckpt_last.pt", "argv.json": "argv.json", "train_argv.json": "train/argv.json",
         "selfcheck.json": "gate/selfcheck.json", "gate_token.json": "gate/gate_token.json",
         "summary.json": "train/summary.json", "train_log.jsonl": "train/train_log.jsonl", "ARM_DONE": "ARM_DONE"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
user = api.whoami()["name"]
repo = f"{user}/tanitad-v8f-strat1-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(repo, repo_type="dataset").private is True
bad = 0
out = {}
for tag in TAGS:
    d = f"{R}/{tag}"
    files = {k: {"bytes": os.path.getsize(f"{d}/{v}"), "sha256": sha(f"{d}/{v}")} for k, v in FILES.items() if os.path.exists(f"{d}/{v}")}
    tok_j = json.load(open(f"{d}/gate/gate_token.json"))
    man = {"tag": tag, "commit": tok_j["commit"], "branch": "agent/v8f-strat1-20261116", "pod_run_dir": d, "files": files}
    mp = f"/root/strat1/tmp/BANK_MANIFEST_{tag}.json"
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    json.dump(man, open(mp, "w"), indent=1)
    for k, v in FILES.items():
        if k in files:
            api.upload_file(path_or_fileobj=f"{d}/{v}", path_in_repo=f"{tag}/{k}", repo_id=repo, repo_type="dataset", commit_message=f"STRAT1 bank {tag}/{k}")
    api.upload_file(path_or_fileobj=mp, path_in_repo=f"{tag}/BANK_MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message=f"STRAT1 bank {tag} manifest")
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    hub = {s.rfilename: s for s in info.siblings}
    ok = True
    for k, m in files.items():
        s = hub.get(f"{tag}/{k}")
        hs = (s.lfs.sha256 if s is not None and s.lfs else None)
        if hs is None and s is not None and k != "ckpt_last.pt":
            hs = "nolfs"  # small file stored inline: verified below by size
            ok = ok and s.size == m["bytes"]
        else:
            ok = ok and hs == m["sha256"]
    out[tag] = {"verified": ok, "rev": info.sha, "ckpt_sha256": files["ckpt_last.pt"]["sha256"]}
    bad += (not ok)
    print(tag, ok, files["ckpt_last.pt"]["sha256"][:12], flush=True)
print(json.dumps({"repo": repo, "private": api.repo_info(repo, repo_type="dataset").private, "rev": info.sha, "arms": out}))
sys.exit(1 if bad else 0)
