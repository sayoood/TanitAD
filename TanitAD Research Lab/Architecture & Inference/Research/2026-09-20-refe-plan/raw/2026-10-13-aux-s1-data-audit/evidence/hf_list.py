import os, re, sys, json
import truststore; truststore.inject_into_ssl()
tok = re.search(r'hf_[A-Za-z0-9]+', open('D:/Projects/TanitAD/Keys.txt', encoding='utf-8', errors='ignore').read()).group(0)
from huggingface_hub import HfApi
api = HfApi(token=tok)
out = {}
# every repo of the author (listing only)
for kind, fn in (("model", api.list_models), ("dataset", api.list_datasets)):
    names = [getattr(r, "id") for r in fn(author="Sayood")]
    out[kind + "_repos"] = names
json.dump(out, open("hf_repos.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
