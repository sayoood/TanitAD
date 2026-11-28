"""Independent check of the published semantic maps from the dev box: download the manifest and every published file listed in it
(or a sample), recompute sha256 locally, and open each npz (keys, shapes, fraction sums)."""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
REPO = "Sayood/tanitad-v7-training-corpus"; D = Path(sys.argv[1]); lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
man = json.load(open(hf_hub_download(REPO, "semantic_maps/SEMANTIC_MAPS_MANIFEST.json", repo_type="dataset", token=tok, local_dir=str(D))))
print("manifest status", man["status"], "| counts", man["counts"]["published"], man["counts"]["by_status"])
n_ok = n_bad = 0
for clip, e in list(man["clips"].items())[:lim]:
    for path, want in e["files"].items():
        p = Path(hf_hub_download(REPO, path, repo_type="dataset", token=tok, local_dir=str(D)))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        good = h == want["sha256"] and p.stat().st_size == want["bytes"]
        if path.endswith(".sam3mapgt.npz"):
            z = np.load(p, allow_pickle=True); meta = json.loads(str(z["meta_json"]))
            cf = z["cart_frac"].astype(np.int64); sums = cf.sum(axis=1)
            good &= meta["schema"] == "tanitad.sam3_map_gt/2" and z["cart_frac"].shape[1:] == (9, 120, 64) and len(z["t_query_us"]) == e["frames_v2ep"] \
                and int((np.abs(sums - 255) > 5).sum()) == 0
            print(f"  {clip[:8]} gt {z['cart_frac'].shape} {z['cart_frac'].dtype} frames {len(z['t_query_us'])} seen {1 - cf[:, 8].mean() / 255:.3f} sha {'OK' if h == want['sha256'] else 'MISMATCH'}")
        n_ok += good; n_bad += not good
print(f"ZZPUBVERIFY-{n_ok}-{n_bad}ZZ")
