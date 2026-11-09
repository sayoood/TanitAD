"""model_sha: sha256 over the sorted (name, sha256) of each weight file of a model dir. Writes F:/strat_out/MODEL_SHA.json."""
import glob, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C
out = {}
for key, d in (("siglip2", "F:/strat_models/siglip2"), ("qwen25vl3b", "F:/strat_models/qwen25vl3b")):
    parts = [(os.path.basename(p), C.sha256_file(p)) for p in sorted(glob.glob(d + "/*.safetensors"))]
    out[key] = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
    out[key + "_files"] = dict(parts)
os.makedirs("F:/strat_out", exist_ok=True)
json.dump(out, open("F:/strat_out/MODEL_SHA.json", "w"), indent=1)
print({k: v for k, v in out.items() if not k.endswith("_files")})
