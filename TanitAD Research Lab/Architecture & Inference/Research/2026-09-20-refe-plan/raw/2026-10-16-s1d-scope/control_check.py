"""S1d scope control: the DB-origin build path must reproduce the existing navtrain label for a navtrain log whose labels came from the BANK origin path.
Compares every stored blob (zlib int8 SDF) of the control log between the production store and the sample store; plus a deliberate-regression arm
(compare against a DIFFERENT token's blob) that must read as 'differing'.  Also: sample-store sanity (keys, maps, leak vs navtest)."""
import sys, json, pathlib
PK = "C:/Users/Admin/s1d_scope/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PK + "/refe")
import aux_label_store as A
S = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/sample"
ctrl = open(S + "/control_log.txt").read().strip()
hp, bp = A.read_shard(pathlib.Path(f"D:/Projects/TanitAD-artifacts/aux_drivable/label_store/shards/{ctrl}.aux1"))
hs, bs = A.read_shard(pathlib.Path(f"{S}/store/shards/{ctrl}.aux1"))
same = sum(1 for t in hp["entries"] if bp[t["t"]] == bs.get(t["t"]))
src = {e["s"] for e in hs["entries"]}
pose_dmax = max(max(abs(a - b) for a, b in zip(e1["p"], e2["p"])) for e1, e2 in zip(hp["entries"], hs["entries"]))
toks = [e["t"] for e in hp["entries"]]
reg = sum(1 for i in range(len(toks) - 1) if bp[toks[i]] == bs.get(toks[i + 1]))     # DELIBERATE REGRESSION: shifted pairing must NOT match
print(f"control {ctrl}: map {hp['map']}/{hs['map']}  tokens prod {len(hp['entries'])} sample {len(hs['entries'])}  blobs identical {same}/{len(hp['entries'])}  "
      f"sample src {src}  max |pose diff| prod-vs-DB {pose_dmax:.3g}   regression arm (shifted pairing) identical {reg}/{len(toks)-1}")
# sample extra logs: keys by map, navtest leak
nt = json.load(open("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"))["token_to_log_name"]
ntl = set(nt.values()); ntt = {t.lower() for t in nt}
man = json.load(open(S + "/store/manifest.json"))
sample_logs = open(S + "/sample_logs.txt").read().split(",")
leak_logs = [l for l in sample_logs + [ctrl] if l in ntl]
allk = 0; leak_tok = 0
for l in sample_logs:
    h, _ = A.read_shard(pathlib.Path(f"{S}/store/shards/{l}.aux1"), with_blobs=False)
    allk += len(h["entries"]); leak_tok += sum(1 for e in h["entries"] if e["t"] in ntt)
print("sample extra keys", allk, "| navtest leak: logs", len(leak_logs), "tokens", leak_tok, "| manifest counts", man["counts"]["by_map"], "src", man["counts"]["by_src"])
json.dump({"control_log": ctrl, "blobs_identical": same, "n": len(hp["entries"]), "regression_arm_identical": reg, "sample_extra_keys": allk,
           "navtest_leak_logs": len(leak_logs), "navtest_leak_tokens": leak_tok, "by_map": man["counts"]["by_map"]}, open(S + "/../control_check.json", "w"), indent=1)
