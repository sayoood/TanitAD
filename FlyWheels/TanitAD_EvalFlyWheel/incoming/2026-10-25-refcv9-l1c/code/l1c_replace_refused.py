"""l1c_replace_refused.py -- PREREG_L1C AMENDMENT 2 A2.1, mechanical: a fresh clip that fails a chain step is logged by sha12 with the reason, REMOVED, and the next
fetched candidate in sha256 order takes its place.  This script handles the clock-sidecar step (the only step that has refused any clip so far).
    python l1c_replace_refused.py [--round N]
Reads  F:/l1c/kit_fresh/data/clock_sidecar.jsonl.meta.json ; rewrites F:/l1c/select/final_ids_fresh.json (raw ids; F: only) ; moves the removed clips' episode
files to F:/l1c/kit_fresh_discarded/v2 ; appends to refcv9_l1c/select/chain_failures_sha12.json (sha12 + step + reason only).  Prints NOREFUSED when nothing is left to replace.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack")
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, default=0)
a = ap.parse_args()
K = "F:/l1c/kit_fresh/data/refcv6-b1-416x1024-eval139"
ids = json.load(open("F:/l1c/select/final_ids_fresh.json"))
m = json.load(open("F:/l1c/kit_fresh/data/clock_sidecar.jsonl.meta.json"))
ref = m["refused_by_sid"]
sid2 = {str(stable_episode_id(r["clip_id"])): r["clip_id"] for r in ids}
gone = [(sid2[s], why) for s, why in ref.items() if s in sid2]
if not gone:
    print("NOREFUSED")
    raise SystemExit(0)
json.dump(ids, open(f"F:/l1c/select/final_ids_fresh.round{a.round}.json", "w"))
fetched = {json.loads(l)["sha12"] for l in open("F:/l1c/select/fetch_manifest.jsonl", encoding="utf-8") if json.loads(l).get("ok")}
cand = [json.loads(l) for l in open("F:/l1c/select/candidates.jsonl", encoding="utf-8")]
cand = [c for c in cand if c["sha12"] in fetched]            # only candidates that were fetched (330)
dd = "F:/l1c/kit_fresh_discarded/v2"
os.makedirs(dd, exist_ok=True)
for c, _w in gone:
    shutil.move(f"{K}/{c}.v2ep.pt", f"{dd}/{c}.v2ep.pt")
if os.path.exists(f"{K}/_v2manifest.pt"):
    os.remove(f"{K}/_v2manifest.pt")
gset = {c for c, _ in gone}
keep = [r for r in ids if r["clip_id"] not in gset]
used = {r["clip_id"] for r in keep} | gset | {r["clip_id"] for r in json.load(open("F:/l1c/select/final_ids_fresh.round0.json"))}
nxt = [c for c in cand if c["clip_id"] not in used][:len(gone)]
if len(nxt) < len(gone):
    print("EXHAUSTED the fetched candidates: n < 300 will be reported as n")
new = keep + [{"clip_id": c["clip_id"], "chunk": int(c["chunk"])} for c in nxt]
json.dump(new, open("F:/l1c/select/final_ids_fresh.json", "w"))
logp = "C:/Users/Admin/refcv8_pf/refcv9_l1c/select/chain_failures_sha12.json"
log = json.load(open(logp)) if os.path.exists(logp) else []
for c, why in gone:
    log.append({"sha12": hashlib.sha256(c.encode()).hexdigest()[:12], "step": "clock sidecar", "why": why, "round": a.round})
json.dump(log, open(logp, "w"), indent=1)
print(f"REPLACED {len(gone)} -> {[hashlib.sha256(c['clip_id'].encode()).hexdigest()[:12] for c in nxt]}; split now {len(new)}")
