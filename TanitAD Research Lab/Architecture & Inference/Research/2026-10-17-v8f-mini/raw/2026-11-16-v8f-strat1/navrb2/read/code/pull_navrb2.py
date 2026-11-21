"""pull_navrb2.py: pull the 4 banked NAVRB2 arms from the PRIVATE HF dataset (token read in place from Keys.txt, never printed) into ../ckpts/<tag>/ and check the local sha256
against the pod-side BANK_MANIFEST.json (the hub's own copy was checked on the pod by the uploader).  Third copy of the three-copy check."""
import hashlib, json, os, re, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.normpath(HERE + "/../ckpts")
REPO = "Sayood/tanitad-v8f-strat1-runs"
TAGS = ["N1R_s0", "S1R_s0", "N1R_s1", "S1R_s1"]
KEYS = "E:/Projects/TanitAD/Keys.txt"
tok = re.search(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="ignore").read()).group(0)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
out, bad = {}, 0
for t in TAGS:
    d = f"{CK}/{t}"
    os.makedirs(d, exist_ok=True)
    for fn in ("BANK_MANIFEST.json", "argv.json", "ckpt_last.pt"):
        p = hf_hub_download(REPO, f"{t}/{fn}", repo_type="dataset", token=tok, local_dir=d + "/_hf")
        os.replace(p, f"{d}/{fn}")
    m = json.load(open(f"{d}/BANK_MANIFEST.json", encoding="utf-8"))
    ok = True
    for fn in ("ckpt_last.pt", "argv.json"):
        ok = ok and sha(f"{d}/{fn}") == m["files"][fn]["sha256"]
    out[t] = {"local_sha256_equals_manifest": ok, "ckpt_sha256": m["files"]["ckpt_last.pt"]["sha256"], "commit": m["commit"]}
    bad += (not ok)
    print(t, ok, m["commit"][:8], flush=True)
print(json.dumps(out))
sys.exit(1 if bad else 0)
