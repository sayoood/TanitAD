"""BYTE-LEVEL episode disjointness across every val deployment vs the parity train.

The key is sha256 of the raw `poses` bytes -- NOT the directory name and NOT
episode_id. Two builds of the same clip produce identical poses bytes; a name
collision across different clips does not. This is the check CLAUDE.md asks for:
"verify at the byte level, do not assume".
"""
import json, hashlib
from pathlib import Path

M = Path(__file__).resolve().parent / "manifests"

SETS = {
    "PARITY-TRAIN physicalai-train-e438721ae894 (pod3 /workspace/pai_epcache)":
        "manifest_train_parity.json",
    "physicalai-val-0c5f7dac3b11 @600 (pod2 /workspace/data/physicalai_phase0/_epcache)":
        "manifest_val_clean600.json",
    "physicalai-val-0c5f7dac3b11 @40 (tanitad-eval /root/valdata) [PUBLISHED]":
        "manifest_EVALPOD_val40.json",
    "physicalai-val-0c5f7dac3b11 @12 (pod1 /root/valdata)":
        "manifest_POD1_val12.json",
    "physicalai-val-f1b378f295ae @80 (pod3 /workspace/pai_epcache) [LEAKY]":
        "manifest_physicalai-val-f1b378f295ae.json",
    "physicalai-val-heldout-79d4e3d2d4c6 @44 (pod3 /workspace/v4run/valcache)":
        "manifest_physicalai-val-heldout-79d4e3d2d4c6.json",
    "physicalai-train-51f40f5ebc21 @320 (pod3 /workspace/pai_epcache)":
        "manifest_physicalai-train-51f40f5ebc21.json",
}

data = {}
for label, fn in SETS.items():
    m = json.loads((M / fn).read_text())
    data[label] = {
        "n": m["n_episodes"],
        "hashes": {e["poses_sha256"] for e in m["episodes"]},
        "eids": {e["episode_id"] for e in m["episodes"]},
        "T_min": min(e["T"] for e in m["episodes"]),
        "T_max": max(e["T"] for e in m["episodes"]),
        "source": m["source_dir"],
    }

TRAIN = "PARITY-TRAIN physicalai-train-e438721ae894 (pod3 /workspace/pai_epcache)"
tr = data[TRAIN]

out = {"method": "sha256 over raw poses[T,4] float32 bytes; overlap = set "
                 "intersection of those digests. episode_id compared separately.",
       "parity_train": {"name": TRAIN, "n": tr["n"],
                        "T_range": [tr["T_min"], tr["T_max"]],
                        "n_unique_hashes": len(tr["hashes"])},
       "val_deployments": {}}

print(f"{'set':<78} {'n':>5} {'uniq':>5} {'ovl':>5} {'%leak':>7}  T-range  verdict")
print("-" * 130)
for label, d in data.items():
    ov = d["hashes"] & tr["hashes"]
    eov = d["eids"] & tr["eids"]
    pct = 100.0 * len(ov) / max(1, d["n"])
    if label == TRAIN:
        verdict = "(is the parity train)"
    elif len(ov) == 0:
        verdict = "CLEAN - episode-disjoint from parity train"
    else:
        verdict = f"LEAKS {len(ov)}/{d['n']} into parity train"
    print(f"{label:<78} {d['n']:>5} {len(d['hashes']):>5} {len(ov):>5} "
          f"{pct:>6.1f}%  [{d['T_min']},{d['T_max']}]  {verdict}")
    out["val_deployments"][label] = {
        "n_episodes": d["n"], "n_unique_poses_hashes": len(d["hashes"]),
        "source_dir": d["source"],
        "T_range": [d["T_min"], d["T_max"]],
        "overlap_with_parity_train_BYTE_LEVEL": len(ov),
        "overlap_pct": round(pct, 2),
        "overlap_by_episode_id": len(eov),
        "verdict": verdict,
    }

# --- Is the published 40 a SUBSET of the 600? and the 12? ---
print()
V600 = "physicalai-val-0c5f7dac3b11 @600 (pod2 /workspace/data/physicalai_phase0/_epcache)"
V40 = "physicalai-val-0c5f7dac3b11 @40 (tanitad-eval /root/valdata) [PUBLISHED]"
V12 = "physicalai-val-0c5f7dac3b11 @12 (pod1 /root/valdata)"
rel = {}
for sub in (V40, V12):
    inter = data[sub]["hashes"] & data[V600]["hashes"]
    is_sub = inter == data[sub]["hashes"]
    # prefix check: are they the FIRST n of the 600, in file order?
    m600 = json.loads((M / SETS[V600]).read_text())["episodes"]
    msub = json.loads((M / SETS[sub]).read_text())["episodes"]
    prefix = [e["poses_sha256"] for e in m600[:len(msub)]] == \
             [e["poses_sha256"] for e in msub]
    print(f"{sub}\n   subset of the 600? {is_sub} ({len(inter)}/{data[sub]['n']})"
          f"   |   is it the FIRST-{len(msub)} PREFIX of the 600? {prefix}")
    rel[sub] = {"n": data[sub]["n"], "overlap_with_600": len(inter),
                "is_subset_of_600": bool(is_sub),
                "is_first_n_prefix_of_600": bool(prefix)}
out["published_subsets_vs_the_600_build"] = rel

# leaky split cross-check against the CLEAN 600 too
lk = "physicalai-val-f1b378f295ae @80 (pod3 /workspace/pai_epcache) [LEAKY]"
out["leaky_split_cross_checks"] = {
    "overlap_with_parity_train": len(data[lk]["hashes"] & tr["hashes"]),
    "overlap_with_clean_val_600": len(data[lk]["hashes"] & data[V600]["hashes"]),
    "n": data[lk]["n"]}
print(f"\nLEAKY f1b378f295ae: {out['leaky_split_cross_checks']}")

hv = "physicalai-val-heldout-79d4e3d2d4c6 @44 (pod3 /workspace/v4run/valcache)"
out["heldout_val_cross_checks"] = {
    "overlap_with_parity_train": len(data[hv]["hashes"] & tr["hashes"]),
    "overlap_with_clean_val_600": len(data[hv]["hashes"] & data[V600]["hashes"]),
    "n": data[hv]["n"]}
print(f"HELDOUT 79d4e3d2d4c6: {out['heldout_val_cross_checks']}")

Path(__file__).resolve().parent.joinpath("disjointness_result.json").write_text(
    json.dumps(out, indent=2))
print("\nwrote disjointness_result.json")
