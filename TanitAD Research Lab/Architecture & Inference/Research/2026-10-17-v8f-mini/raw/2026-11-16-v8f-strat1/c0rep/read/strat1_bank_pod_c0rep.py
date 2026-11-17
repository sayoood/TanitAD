"""strat1_bank_pod_c0rep.py: bank the 2 STRAT1 C0 same-seed replicate arms PRIVATE on HF under DISTINCT tags (run ON the pod; reads the token in place, never prints it).
Pod run dirs /root/strat1/runs_c0rep/C0_s10 and C0_s11 are banked as C0_s10r and C0_s11r.  The originals C0_s10 / C0_s11 are already in the repo and must never be overwritten:
guards (1) the destination tag must differ from the source dir name and end in 'r', (2) the destination prefix must be empty in the repo before any upload,
(3) the hub sha256 of every original file is read before and after and must be equal.  Writes BANK_MANIFEST_<tag>.json per arm, verifies the hub LFS sha256 against it."""
import hashlib, json, os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi
R = "/root/strat1/runs_c0rep"
MAP = {"C0_s10": "C0_s10r", "C0_s11": "C0_s11r"}          # pod dir name -> HF tag
ORIG = tuple(MAP)
FILES = {"ckpt_last.pt": "train/ckpt_last.pt", "argv.json": "argv.json", "train_argv.json": "train/argv.json",
         "selfcheck.json": "gate/selfcheck.json", "gate_token.json": "gate/gate_token.json",
         "summary.json": "train/summary.json", "train_log.jsonl": "train/train_log.jsonl", "ARM_DONE": "ARM_DONE"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
for src, dst in MAP.items():
    assert dst != src and dst.endswith("r") and dst not in ORIG, (src, dst)
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
user = api.whoami()["name"]
repo = f"{user}/tanitad-v8f-strat1-runs"
assert api.repo_info(repo, repo_type="dataset").private is True
def snapshot():
    info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    return info, {s.rfilename: (s.size, (s.lfs.sha256 if s.lfs else None)) for s in info.siblings}
info0, before = snapshot()
orig_before = {k: v for k, v in before.items() if k.split("/")[0] in ORIG}
assert any(k.startswith("C0_s10/") for k in orig_before) and any(k.startswith("C0_s11/") for k in orig_before), "the originals are not in the repo: wrong repo?"
for dst in MAP.values():
    assert not any(k.startswith(dst + "/") for k in before), f"destination {dst}/ is not empty in the repo: refuse"
print("repo listing ok: originals present", len(orig_before), "files; destinations empty", flush=True)
bad, out = 0, {}
for src, tag in MAP.items():
    d = f"{R}/{src}"
    files = {k: {"bytes": os.path.getsize(f"{d}/{v}"), "sha256": sha(f"{d}/{v}")} for k, v in FILES.items() if os.path.exists(f"{d}/{v}")}
    tok_j = json.load(open(f"{d}/gate/gate_token.json"))
    man = {"tag": tag, "same_seed_replicate_of": src, "commit": tok_j["commit"], "branch": "agent/v8f-strat1-20261116", "pod_run_dir": d, "files": files}
    mp = f"/root/strat1/tmp/BANK_MANIFEST_{tag}.json"
    json.dump(man, open(mp, "w"), indent=1)
    for k, v in FILES.items():
        if k in files:
            api.upload_file(path_or_fileobj=f"{d}/{v}", path_in_repo=f"{tag}/{k}", repo_id=repo, repo_type="dataset", commit_message=f"STRAT1 bank {tag}/{k}")
    api.upload_file(path_or_fileobj=mp, path_in_repo=f"{tag}/BANK_MANIFEST.json", repo_id=repo, repo_type="dataset", commit_message=f"STRAT1 bank {tag} manifest")
    info, now = snapshot()
    ok = True
    for k, m in files.items():
        size, hs = now.get(f"{tag}/{k}", (None, None))
        if hs is None and size is not None and k != "ckpt_last.pt":
            ok = ok and size == m["bytes"]
        else:
            ok = ok and hs == m["sha256"]
    out[tag] = {"verified": ok, "ckpt_sha256": files["ckpt_last.pt"]["sha256"]}
    bad += (not ok)
    print(tag, ok, files["ckpt_last.pt"]["sha256"][:12], flush=True)
info1, after = snapshot()
unchanged = {k: v for k, v in after.items() if k.split("/")[0] in ORIG} == orig_before
print("originals unchanged:", unchanged, flush=True)
print(json.dumps({"repo": repo, "private": info1.private, "rev": info1.sha, "arms": out, "originals_unchanged": unchanged}))
sys.exit(1 if (bad or not unchanged) else 0)
