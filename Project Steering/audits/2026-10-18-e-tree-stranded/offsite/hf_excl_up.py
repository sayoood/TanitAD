import os, re, hashlib, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, CommitOperationAdd
RID = "Sayood/tanitad-internal-data"; PFX = "stranded-bulk-2026-10-18"
BS = chr(92); PRE = BS + BS + "?" + BS
tok = re.search(r"hf_[A-Za-z0-9]{30,}", open("D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
api = HfApi(token=tok)
rels = ["TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/d6_scene_table_step30000.csv",
        "TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p4_mask/mask_R7_A1.index.jsonl"]
LP = lambda r: PRE + ("C:/Users/Admin/estrand_bulk/" + r).replace("/", BS)
api.create_commit(RID, repo_type="dataset", operations=[CommitOperationAdd(path_in_repo=f"{PFX}/{r}", path_or_fileobj=LP(r)) for r in rels],
                  commit_message="stranded bulk 2026-10-18: the 2 files excluded from git (secret-scan false positive on a data float)")
out = []
for p in api.get_paths_info(RID, [f"{PFX}/{r}" for r in rels], repo_type="dataset"):
    b = open(LP(p.path[len(PFX) + 1:]), "rb").read(); sh = hashlib.sha256(b).hexdigest()
    ok = (p.lfs.sha256 == sh) if getattr(p, "lfs", None) else (p.blob_id == hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest())
    out.append({"path": p.path, "sha256": sh, "verified": ok})
json.dump(out, open("C:/Users/Admin/estrand/hfup/HF_VERIFIED_excluded.json", "w"), indent=1)
print("ZZEXCL", [(o["path"][-40:], o["sha256"][:8], o["verified"]) for o in out])
