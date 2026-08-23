"""P1 — pull the TRAIN-corpus obstacle join (2,308 eps) + its meta sidecar from HF."""
import json, os, re, sys, time
from pathlib import Path
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
KEYS = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
TOK = re.search(r"hf_[A-Za-z0-9]+", KEYS.read_text(encoding="utf-8", errors="ignore")).group(0)
REPO = "Sayood/tanitad-ph0-aug120"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rec = {}
for rel in ["joins/train2400_agents.jsonl.xz.meta.json", "joins/train2400_agents.jsonl.xz"]:
    t0 = time.time()
    p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(out), token=TOK)
    n = os.path.getsize(p); dt = time.time() - t0
    rec[rel] = {"path": p, "bytes": n, "wall_s": round(dt, 1), "MB_per_s": round(n/1e6/max(dt,1e-9), 2)}
    print(json.dumps({rel: rec[rel]}), flush=True)
import hashlib
h = hashlib.md5()
with open(rec["joins/train2400_agents.jsonl.xz"]["path"], "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
rec["md5_xz"] = h.hexdigest()
rec["md5_expected"] = "24cbdca8c3b23aafc2fb17e6bf99cf76"
rec["md5_match"] = rec["md5_xz"] == rec["md5_expected"]
print(json.dumps(rec, indent=1), flush=True)
(out / "p1_pull_join.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
