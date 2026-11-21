"""Reconstruct the aug120 todo list and measure label coverage per batch.

Answers, from the data:
  1. len(todo) and whether the far-side batch tags partition it (8-run vs 40-run)
  2. union v2 / sam3 coverage; the exact clips with no labels anywhere
  3. whether batch_00184's clips have SAM3 elsewhere (batch_00160)
  4. duplicate-record equality across overlapping batch files
  5. bridge-failure overlap (ego npz availability for the union)
"""
import glob
import json
import os
from collections import OrderedDict

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
AUX = os.path.join(ROOT, "aux")

rec = set(pd.read_parquet(os.path.join(AUX, "records.parquet"))["clip_id"]
          .astype(str).unique())
loc = json.load(open(os.path.join(AUX, "w120_loc.json")))
done = set(json.load(open(os.path.join(AUX, "w120val_600__clips.json"))))
todo = sorted((rec & set(loc)) - done)
print(f"rec={len(rec)} loc={len(loc)} done={len(done)} "
      f"with_w120={len(rec & set(loc))} todo={len(todo)}")

# --- load every batch file ------------------------------------------------- #
v2_files, s3_files = OrderedDict(), OrderedDict()
for p in sorted(glob.glob(os.path.join(ROOT, "labels", "batch_*", "v2",
                                       "ph0_v2.json"))):
    tag = p.replace("\\", "/").split("/")[-3]
    d = json.load(open(p, encoding="utf-8"))
    v2_files[tag] = {c["clip_id"]: c for c in d["clips"]}
for p in sorted(glob.glob(os.path.join(ROOT, "labels", "batch_*", "sam3",
                                       "sam3.json"))):
    tag = p.replace("\\", "/").split("/")[-3]
    d = json.load(open(p, encoding="utf-8"))
    s3_files[tag] = {c["clip_id"]: c for c in d["clips"]}

print("\nper-batch (tag: n_v2 n_sam3 | todo-slice match)")
for tag in sorted(set(v2_files) | set(s3_files)):
    b0 = int(tag.split("_")[1])
    nv = len(v2_files.get(tag, {}))
    ns = len(s3_files.get(tag, {}))
    v2ids = set(v2_files.get(tag, {}))
    m8 = "==todo[%d:%d]" % (b0, b0 + 8) if v2ids == set(todo[b0:b0 + 8]) else ""
    m40 = "==todo[%d:%d]" % (b0, b0 + 40) if v2ids == set(todo[b0:b0 + 40]) else ""
    print(f"  {tag}: v2={nv:3d} sam3={ns:3d}  {m8}{m40}")

# --- union coverage -------------------------------------------------------- #
v2_union, s3_union = {}, {}
v2_src, s3_src = {}, {}
for tag, m in v2_files.items():
    for cid, r in m.items():
        v2_union.setdefault(cid, []).append((tag, r))
for tag, m in s3_files.items():
    for cid, r in m.items():
        s3_union.setdefault(cid, []).append((tag, r))

no_v2 = [c for c in todo if c not in v2_union]
no_s3 = [c for c in todo if c not in s3_union]
extra_v2 = [c for c in v2_union if c not in todo]
print(f"\nunion: v2={len(v2_union)} sam3={len(s3_union)} of todo={len(todo)}")
print(f"todo clips with NO v2 anywhere: {len(no_v2)} {no_v2}")
print(f"todo clips with NO sam3 anywhere: {len(no_s3)} {no_s3}")
print(f"v2 clips outside todo: {len(extra_v2)} {extra_v2[:5]}")

# batch_00184's 8 clips: sam3 elsewhere?
b184 = sorted(v2_files.get("batch_00184", {}))
b184_s3 = {c: [t for t, _ in s3_union.get(c, [])] for c in b184}
print(f"\nbatch_00184 clips ({len(b184)}) sam3 coverage: "
      f"{sum(1 for v in b184_s3.values() if v)}/{len(b184)}")
for c, tags in b184_s3.items():
    print(f"  {c[:13]}… sam3 in {tags or 'NOWHERE'}")

# --- duplicate equality ---------------------------------------------------- #
def eq_stats(union, label):
    dup = {c: v for c, v in union.items() if len(v) > 1}
    same = diff = 0
    diffs = []
    for c, v in dup.items():
        s = {json.dumps(r, sort_keys=True) for _, r in v}
        if len(s) == 1:
            same += 1
        else:
            diff += 1
            diffs.append((c, [t for t, _ in v]))
    print(f"{label}: {len(dup)} clips in >1 file -> identical {same}, "
          f"different {diff}")
    for c, tags in diffs[:6]:
        print(f"   differs: {c[:13]}… in {tags}")
    return dup, diffs


v2_dup, v2_diffs = eq_stats(v2_union, "v2 duplicates")
s3_dup, s3_diffs = eq_stats(s3_union, "sam3 duplicates")

# --- ego availability ------------------------------------------------------ #
bridged = set(json.load(open(os.path.join(
    AUX, "bridged_w120train_2400__clips.json"))))
fails = json.load(open(os.path.join(
    AUX, "bridged_w120train_2400__failures.json")))
fail_ids = {f["clip"] for f in fails}
need_ego = sorted(v2_union)
no_ego = [c for c in need_ego if c not in bridged]
print(f"\nbridged ego: {len(bridged)} clips, failures {len(fail_ids)}")
print(f"labelled clips missing bridged ego: {len(no_ego)} {no_ego[:10]}")
in_fail = sorted(set(need_ego) & fail_ids)
print(f"labelled clips in bridge FAILURES list: {len(in_fail)} {in_fail[:10]}")

# --- baseline -------------------------------------------------------------- #
print("\nfused_w120val baseline summary:",
      open(os.path.join(AUX, "fused_w120val___summary.json")).read())

json.dump({"todo": todo,
           "v2_union_ids": sorted(v2_union),
           "s3_union_ids": sorted(s3_union),
           "no_v2": no_v2, "no_s3": no_s3, "no_ego": no_ego,
           "b184": b184},
          open(os.path.join(ROOT, "coverage.json"), "w"), indent=1)
print("\nANALYZE_DONE")
