"""R0c read, exactly PREREG_R0C_A1 section 1, on the dev box from R0c's HF egress (the pod is training S1c).
1. download Sayood/tanitad-refe-rls1-r0c final/* (everything but scorer_states_last.pt), sha256 vs the LFS listing;
2. arm dir = the recorded sft.jsonl, kill_checks.jsonl, KILLED.json, monitor.log + rlval_BvsA_update{N}.json from r0c_ba_read.py per existing check;
3. r0c_read.py <arm> --policy eval/r0c_policy_A1.json --out <arm>/R0C_READ.json.  The token is read in place from Keys.txt, never printed.
argv: PKG OUT"""
import hashlib
import json
import os
import re
import shutil
import sys

PKG, OUT = sys.argv[1:3]
RD = os.path.join(PKG, "raw", "2026-10-15-r0c-reader")
sys.path[:0] = [os.path.join(PKG, "eval"), os.path.join(PKG, "refe"), RD]
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

X = "C:/Users/Admin/rls1b_dl/data/x"
PDM, TERMS = f"{X}/pdm_val/pdm_val", f"{X}/terms_val/terms_val"
REPO = "Sayood/tanitad-refe-rls1-r0c"
tok = None
for kf in ("D:/Projects/TanitAD/Keys.txt", "E:/Projects/TanitAD/Keys.txt"):
    if os.path.isfile(kf):
        m = re.search(r"hf_[A-Za-z0-9]+", open(kf, encoding="utf-8", errors="replace").read())
        if m:
            tok = m.group(0)
            break
assert tok, "no HF token found in place"

api = HfApi(token=tok)
listing = {e.path: e for e in api.list_repo_tree(REPO, recursive=True, expand=True) if getattr(e, "lfs", None) is not None or e.path.startswith("final/")}
dl = os.path.join(OUT, "dl")
want = [p for p in listing if p.startswith("final/") and not p.endswith("scorer_states_last.pt")]
for p in sorted(want):
    f = hf_hub_download(REPO, p, token=tok, local_dir=dl)
    lfs = getattr(listing[p], "lfs", None)
    if lfs is not None:
        sha = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        assert h == sha, f"sha256 mismatch {p}"
print(f"downloaded {len(want)} files, LFS sha256 verified", flush=True)
F = os.path.join(dl, "final")

import r0c_ba_read  # noqa: E402
import r0c_read  # noqa: E402
import rl_val_eval as RV  # noqa: E402

arm = os.path.join(OUT, "R0c")
shutil.rmtree(arm, ignore_errors=True)
os.makedirs(arm)
for n in ("sft.jsonl", "kill_checks.jsonl", "KILLED.json", "monitor.log"):
    shutil.copyfile(os.path.join(F, n), os.path.join(arm, n))
pdm, _ = RV.load_pdm(PDM)
tsv = os.path.join(OUT, "rl_val_scenes_rebuilt_from_pdm_keys.tsv")
with open(tsv, "w", encoding="utf-8", newline="\n") as f:
    f.write("token\tlog\n")
    for k in sorted(pdm):
        f.write(f"{k[1]}\t{k[0]}\n")
for u in (250, 500, 750, 1000):
    a, b = os.path.join(F, f"scores_update{u}_A.npz"), os.path.join(F, f"scores_update{u}_B.npz")
    if not (os.path.isfile(a) and os.path.isfile(b)):
        continue
    rc = r0c_ba_read.main(["--a", a, "--b", b, "--pdm", PDM, "--terms", TERMS, "--rl-val-scenes", tsv, "--out", os.path.join(arm, f"rlval_BvsA_update{u}.json")])
    assert rc == 0, f"r0c_ba_read update{u}: rc {rc}"
pol = os.path.join(PKG, "eval", "r0c_policy_A1.json")
print("policy sha256", hashlib.sha256(open(pol, "rb").read()).hexdigest(), flush=True)
rc = r0c_read.main([arm, "--policy", pol, "--out", os.path.join(arm, "R0C_READ.json")])
print("r0c_read rc", rc, flush=True)
