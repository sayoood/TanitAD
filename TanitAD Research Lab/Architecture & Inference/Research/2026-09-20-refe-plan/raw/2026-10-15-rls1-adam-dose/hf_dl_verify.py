import re, os, sys, json, hashlib
from huggingface_hub import HfApi, hf_hub_download
tok = re.findall(r'hf_[A-Za-z0-9]+', open(r'E:/Projects/TanitAD/Keys.txt', encoding='utf-8', errors='ignore').read())[0]
api = HfApi(token=tok)
OUT = r'C:/Users/Admin/rls1_adam_dl'
WANT = {
 "Sayood/tanitad-refe-rls1": ["base/model_sft4_B_u400.pt", "data/rlfan_sets_train.tar.gz", "data/rlfan_tlane_train.tar.gz"],
 "Sayood/tanitad-refe-rls1-c0": ["scorer_states/scorer_states_u250.pt", "scorer_states/scorer_states_u500.pt", "scorer_states/scorer_states_u1000.pt"],
 "Sayood/tanitad-refe-rls1-r0": ["scorer_states/scorer_states_u250.pt", "scorer_states/scorer_states_u500.pt", "scorer_states/scorer_states_u1000.pt", "final/sft.jsonl"],
 "Sayood/tanitad-refe-rls1-r0b": ["scorer_states/scorer_states_u250.pt", "scorer_states/scorer_states_u500.pt", "final/sft.jsonl", "final/PASS_aa49ff9773a4.json"],
}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()
res = {}
for repo, files in WANT.items():
    info = api.repo_info(repo, repo_type="model", files_metadata=True)
    meta = {s.rfilename: (getattr(getattr(s, 'lfs', None), 'sha256', None), s.size) for s in info.siblings}
    sub = repo.split('/')[1].replace('tanitad-refe-', '')
    for fn in files:
        p = hf_hub_download(repo, fn, repo_type="model", local_dir=os.path.join(OUT, sub), token=tok)
        want = meta[fn][0]
        g = sha(p) if want else None
        ok = (g == want) if want else None
        res[f"{sub}/{fn}"] = {"repo": repo, "revision": info.sha, "bytes": os.path.getsize(p), "sha256": g, "hf_lfs_sha256": want, "match": ok}
        print(sub, fn, "OK" if ok else ("NO-LFS-SHA(small file)" if ok is None else "MISMATCH"), flush=True)
json.dump(res, open(os.path.join(OUT, "inputs_sha256.json"), "w"), indent=1)
print("ZZDL_DONE")
