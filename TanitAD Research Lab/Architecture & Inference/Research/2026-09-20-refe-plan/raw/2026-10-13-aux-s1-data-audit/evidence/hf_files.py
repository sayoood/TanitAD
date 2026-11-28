import re, json
import truststore; truststore.inject_into_ssl()
tok = re.search(r'hf_[A-Za-z0-9]+', open('D:/Projects/TanitAD/Keys.txt', encoding='utf-8', errors='ignore').read()).group(0)
from huggingface_hub import HfApi
api = HfApi(token=tok)
res = {}
for rid, kind in (("Sayood/tanitad-refe-navtrain-banks","dataset"),("Sayood/tanitad-refe-p2-joint-pdm","model"),("Sayood/tanitad-refe-rls1","model"),("Sayood/tanitad-refe-vitl16-navtrain10-grow-final","model")):
    try:
        info = api.repo_info(rid, repo_type=kind, files_metadata=True)
        files = [(s.rfilename, s.size) for s in info.siblings]
        res[rid] = {"kind": kind, "private": info.private, "n_files": len(files), "total_bytes": sum(x[1] or 0 for x in files),
                    "files": files if len(files) <= 60 else [f for f in files if "snapshots" not in f[0]]}
    except Exception as e:
        res[rid] = {"error": repr(e)[:300]}
json.dump(res, open("hf_files.json","w"), indent=1)
for k,v in res.items():
    print("==", k, {a:b for a,b in v.items() if a!="files"})
    for f in v.get("files", [])[:70]:
        print("   ", f[1], f[0])
