import os, re, sys, hashlib, json
from huggingface_hub import HfApi, hf_hub_download
tok = re.findall(r'hf_[A-Za-z0-9]+', open(r'E:/Projects/TanitAD/Keys.txt', encoding='utf-8', errors='ignore').read())[0]
api = HfApi(token=tok)
DL = r'C:/Users/Admin/rls1b_dl'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
out = {}
for sub, repo in (('c0', 'Sayood/tanitad-refe-rls1-c0'), ('r0', 'Sayood/tanitad-refe-rls1-r0'), ('r0b', 'Sayood/tanitad-refe-rls1-r0b')):
    info = api.model_info(repo, files_metadata=True)
    want = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.rfilename.startswith('final/scores_') and getattr(s, 'lfs', None)}
    for fn, w in sorted(want.items()):
        if '_base' in fn and 'step0' not in fn:
            continue
        p = os.path.join(DL, sub, fn)
        if not os.path.exists(p):
            hf_hub_download(repo, fn, local_dir=os.path.join(DL, sub), token=tok)
        g = sha(p)
        out[f'{sub}/{fn}'] = {'repo': repo, 'revision': info.sha, 'sha256': g, 'hf_lfs_sha256': w, 'match': g == w}
        print(sub, fn, 'OK' if g == w else 'MISMATCH', flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
