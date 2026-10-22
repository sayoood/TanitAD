#!/usr/bin/env python3
"""sha256 of the two local OpenScene metadata tgz files vs the LFS oid HF lists for them (raw/hf_tree_main_openscene_v11.json). Read-only."""
import hashlib, json, sys
H = {i["path"].rsplit("/", 1)[-1]: i["lfs"]["oid"] for i in json.load(open("raw/hf_tree_main_openscene_v11.json")) if i["type"] == "file" and "lfs" in i}
out = {}
for name, path in (("openscene_metadata_test.tgz", "D:/Archive/devbox-C/navsim/data/openscene_metadata_test.tgz"), ("openscene_metadata_trainval.tgz", "D:/TanitAstraResearch/data/e024_corpus_20261010/openscene_metadata_trainval.tgz"), ("openscene_metadata_mini.tgz", "D:/TanitAstraResearch/research/feasibility/20261003_navsim/mini_cohort256/openscene_metadata_mini.tgz")):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    out[name] = {"path": path, "sha256": h.hexdigest(), "hf_lfs_oid": H.get(name), "match": h.hexdigest() == H.get(name)}
    print(name, out[name]["match"], flush=True)
json.dump(out, open("raw/sha_check.json", "w"), indent=1)
