"""strat1_bank_pod_navrb2.py: bank the 4 NAVRB2 base arms (commit e71782a3) PRIVATE on HF (run ON the pod; reads the token in place, never prints it).
Light work only: sha256 + one upload at a time, no GPU, no model load.  Writes BANK_MANIFEST_<tag>.json, uploads <tag>/<file>, verifies the hub's LFS sha256 against the manifest."""
import hashlib, json, os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
R = "/root/strat1/navrb2/runs"
TAGS = ["N1R_s0", "S1R_s0", "N1R_s1", "S1R_s1"]
FILES = {"ckpt_last.pt": "train/ckpt_last.pt", "argv.json": "argv.json", "train_argv.json": "train/argv.json",
         "selfcheck.json": "gate/selfcheck.json", "gate_token.json": "gate/gate_token.json", "checks.json": "gate/checks.json",
         "summary.json": "train/summary.json", "train_log.jsonl": "train/train_log.jsonl", "zerograd_navrb.json": "zerograd_navrb.json",
         "ARM_DONE": "ARM_DONE"}
INPUTS = {"navtext_nav_rb.npz": "/root/strat1/inputs/strat1/navtext_nav_rb.npz",
          "navtext_nav_rb_report.json": "/root/strat1/inputs/strat1/navtext_nav_rb_report.json"}
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
os.makedirs("/root/strat1/tmp", exist_ok=True)
bad = 0
out = {}
for tag in TAGS:
    d = f"{R}/{tag}"
    files = {k: {"bytes": os.path.getsize(f"{d}/{v}"), "sha256": sha(f"{d}/{v}")} for k, v in FILES.items() if os.path.exists(f"{d}/{v}")}
    tok_j = json.load(open(f"{d}/gate/gate_token.json"))
    man = {"tag": tag, "commit": tok_j["commit"], "branch": "agent/v8f-navrb2-20261117", "pod_run_dir": d, "files": files}
    mp = f"/root/strat1/tmp/BANK_MANIFEST_{tag}.json"
    json.dump(man, open(mp, "w"), indent=1)
    for k, v in FILES.items():
        if k in files:
            api.upload_file(path_or_fileobj=f"{d}/{v}", path_in_repo=f"{tag}/{k}", repo_id=repo, repo_type="dataset", commit_message=f"NAVRB2 bank {tag}/{k}")
    api.upload_file(path_or_fileobj=mp, path_in_repo=f"{tag}/BANK_MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message=f"NAVRB2 bank {tag} manifest")
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    hub = {s.rfilename: s for s in info.siblings}
    ok = True
    for k, m in files.items():
        s = hub.get(f"{tag}/{k}")
        hs = (s.lfs.sha256 if s is not None and s.lfs else None)
        if hs is None and s is not None and k != "ckpt_last.pt":
            ok = ok and s.size == m["bytes"]  # small file stored inline: size check
        else:
            ok = ok and hs == m["sha256"]
    out[tag] = {"verified": ok, "rev": info.sha, "ckpt_sha256": files["ckpt_last.pt"]["sha256"]}
    bad += (not ok)
    print(tag, ok, files["ckpt_last.pt"]["sha256"][:12], flush=True)
# the channel's table (single copy otherwise): bank under navrb2_inputs/
inp = {}
for k, p in INPUTS.items():
    inp[k] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
    api.upload_file(path_or_fileobj=p, path_in_repo=f"navrb2_inputs/{k}", repo_id=repo, repo_type="dataset", commit_message=f"NAVRB2 bank inputs/{k}")
info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
hub = {s.rfilename: s for s in info.siblings}
for k, m in inp.items():
    s = hub.get(f"navrb2_inputs/{k}")
    hs = (s.lfs.sha256 if s is not None and s.lfs else None)
    good = (hs == m["sha256"]) if hs else (s is not None and s.size == m["bytes"])
    out["navrb2_inputs/" + k] = {"verified": good, "sha256": m["sha256"]}
    bad += (not good)
print(json.dumps({"repo": repo, "private": api.repo_info(repo, repo_type="dataset").private, "rev": info.sha, "arms": out}))
sys.exit(1 if bad else 0)
