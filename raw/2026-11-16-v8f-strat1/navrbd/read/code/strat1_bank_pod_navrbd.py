"""strat1_bank_pod_navrbd.py: bank the 3 NAVRBD arms (N1D_s0, N1D_s1, N1Dx_s0) PRIVATE on HF (run ON the pod; reads the token in place, never prints it).
Copy of strat1_bank_pod_navrb2.py with the tags, run dir and zerograd file name changed. Light work only: sha256 + one upload at a time, no GPU.
Writes BANK_MANIFEST_<tag>.json, uploads <tag>/<file>, verifies the hub's LFS sha256 (or size for small inline files) against the manifest."""
import hashlib, json, os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
R = "/root/strat1/navrbd/runs"
TAGS = ["N1D_s0", "N1D_s1", "N1Dx_s0"]
FILES = {"ckpt_last.pt": "train/ckpt_last.pt", "argv.json": "argv.json", "train_argv.json": "train/argv.json",
         "selfcheck.json": "gate/selfcheck.json", "gate_token.json": "gate/gate_token.json", "checks.json": "gate/checks.json",
         "summary.json": "train/summary.json", "train_log.jsonl": "train/train_log.jsonl", "zerograd_navrbd.json": "zerograd_navrbd.json",
         "ARM_DONE": "ARM_DONE"}
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
assert api.repo_info(repo, repo_type="dataset").private is True
os.makedirs("/root/strat1/tmp", exist_ok=True)
bad = 0
out = {}
for tag in TAGS:
    d = f"{R}/{tag}"
    files = {k: {"bytes": os.path.getsize(f"{d}/{v}"), "sha256": sha(f"{d}/{v}")} for k, v in FILES.items() if os.path.exists(f"{d}/{v}")}
    assert "ckpt_last.pt" in files and "gate_token.json" in files, (tag, sorted(files))
    tok_j = json.load(open(f"{d}/gate/gate_token.json"))
    man = {"tag": tag, "commit": tok_j.get("commit"), "card": "PREREG_V8F_NAVRBD.md sha256 881d7f3d", "pod_run_dir": d, "files": files}
    mp = f"/root/strat1/tmp/BANK_MANIFEST_{tag}.json"
    json.dump(man, open(mp, "w"), indent=1)
    for k, v in FILES.items():
        if k in files:
            api.upload_file(path_or_fileobj=f"{d}/{v}", path_in_repo=f"{tag}/{k}", repo_id=repo, repo_type="dataset", commit_message=f"NAVRBD bank {tag}/{k}")
    api.upload_file(path_or_fileobj=mp, path_in_repo=f"{tag}/BANK_MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message=f"NAVRBD bank {tag} manifest")
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    hub = {s.rfilename: s for s in info.siblings}
    ok = True
    for k, m in files.items():
        s = hub.get(f"{tag}/{k}")
        hs = (s.lfs.sha256 if s is not None and s.lfs else None)
        if hs is None and s is not None and k != "ckpt_last.pt":
            ok = ok and s.size == m["bytes"]
        else:
            ok = ok and hs == m["sha256"]
    out[tag] = {"verified": ok, "rev": info.sha, "commit": man["commit"], "ckpt_sha256": files["ckpt_last.pt"]["sha256"], "n_files": len(files)}
    bad += (not ok)
    print(tag, ok, files["ckpt_last.pt"]["sha256"][:12], flush=True)
print(json.dumps({"repo": repo, "private": api.repo_info(repo, repo_type="dataset").private, "arms": out}))
print("ZZ NAVRBD BANK", "OK" if bad == 0 else "FAIL", "ZZ", flush=True)
sys.exit(1 if bad else 0)
