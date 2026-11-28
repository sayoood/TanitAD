#!/usr/bin/env python3
"""DATASCALE selection (pod-side, CPU, metadata only). Rule frozen in SCOPE_DATASCALE.md.
  census : counts per exclusion step, writes NO clip list.
  select : writes the global order (every eligible clip, stratified-permutation rank) + the pilot list.
No address, port, pod id or token in this file."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/tad/stack")
META = "/root/datascale/meta"
OUT = "/root/datascale/selection"
SEED = 20261115
PILOT_N = 200
N_CONTROL = 3
STRATA = ["country", "platform_class"]


def dig(c):
    return hashlib.sha256(str(c).encode("utf-8")).hexdigest()


def read_ids(p):
    return {ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()}


def digest_set(path):
    d = json.load(open(path, encoding="utf-8"))
    best = None
    for k, v in d.items():
        if isinstance(v, (list, dict)) and len(v) >= 10:
            s = set(v if isinstance(v, list) else v.keys())
            if all(isinstance(x, str) and len(x) == 64 for x in list(s)[:5]):
                if best is None or len(s) > len(best):
                    best = s
    return best or set()


def exclusion_sets():
    D = "/root/tad/stack/tanitad/data/"
    ex = {}
    ex["our_train4572"] = read_ids("/root/ops/train4572.txt")
    ex["our_eval141"] = read_ids("/root/ops/clips141.txt")
    cs = json.load(open("/root/data/_hf/camera/camera_sha256.json", encoding="utf-8"))
    cs = cs.get("files", cs) if isinstance(cs, dict) else cs
    ex["hf_corpus_camera_ids"] = {k.split("/")[-1].replace(".mp4", "") for k in cs}
    ex["pod_v2ep_ledger"] = {json.loads(ln)["clip"] for ln in open("/root/data/v2ep_ledger.jsonl", encoding="utf-8") if ln.strip()}
    chal = pd.read_parquet(f"{META}/splits/challenging_scenarios.parquet")
    ex["nvidia_challenging_scenarios"] = set(map(str, chal.index))
    ood = pd.read_parquet(f"{META}/reasoning/ood_reasoning.parquet")
    ex["nvidia_ood_reasoning_val"] = set(map(str, ood.index[ood["split"].astype(str) == "val"]))
    dx = {"parity_train2400_digests": digest_set(D + "parity_train_clip_digests.json"),
          "deployed_val40_digests": digest_set(D + "deployed_val40_clip_digests.json"),
          "v72_eval147_digests": digest_set(D + "v72_eval_clip_digests.json")}
    return ex, dx


def eligible_table():
    ci = pd.read_parquet(f"{META}/clip_index.parquet")
    dc = pd.read_parquet(f"{META}/metadata/data_collection.parquet")
    fp = pd.read_parquet(f"{META}/metadata/feature_presence.parquet")
    sd = pd.read_parquet(f"{META}/splits/default.parquet")["split"].astype(str).rename("split_default")
    sa = pd.read_parquet(f"{META}/splits/alpamayo_aligned.parquet")["split"].astype(str).rename("split_alpamayo")
    t = ci.join(dc).join(sd).join(sa)
    for c in ["egomotion", "camera_front_wide_120fov", "camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions", "obstacle.offline"]:
        t["has_" + c] = fp[c].astype(bool)
    t.index = t.index.astype(str)
    return t


def steps(t):
    ex, dx = exclusion_sets()
    rows = []
    state = {"m": pd.Series(True, index=t.index)}

    def step(name, keep):
        before = int(state["m"].sum())
        state["m"] = state["m"] & keep
        rows.append({"step": name, "removed": before - int(state["m"].sum()), "left": int(state["m"].sum())})
    rows.append({"step": "all clips in clip_index.parquet", "removed": 0, "left": len(t)})
    step("clip_is_valid", t["clip_is_valid"].astype(bool))
    step("required features present (egomotion, front_wide camera, intrinsics, extrinsics, vehicle_dimensions)",
         t[["has_egomotion", "has_camera_front_wide_120fov", "has_camera_intrinsics", "has_sensor_extrinsics", "has_vehicle_dimensions"]].all(axis=1))
    step("NVIDIA default split == train", t["split_default"] == "train")
    step("NVIDIA alpamayo_aligned split == train", t["split_alpamayo"] == "train")
    for k, s in ex.items():
        step("exclude " + k, ~t.index.isin(s))
    dg = pd.Series([dig(c) for c in t.index], index=t.index)
    for k, s in dx.items():
        step("exclude " + k, ~dg.isin(s))
    return rows, state["m"], ex, dx


def census():
    t = eligible_table()
    rows, m, ex, dx = steps(t)
    el = t[m]
    ood_idx = set(pd.read_parquet(f"{META}/reasoning/ood_reasoning.parquet").index.astype(str))
    rep = {"hf_dataset_sha": "14ed7a77897edafb113dc86fde8b8173ae7fb028", "steps": rows, "n_eligible": int(m.sum()),
           "exclusion_set_sizes": {k: len(v) for k, v in {**ex, **dx}.items()},
           "eligible_by_country": el["country"].value_counts().to_dict(),
           "eligible_by_platform": el["platform_class"].value_counts().to_dict(),
           "eligible_with_obstacle_offline": int(el["has_obstacle.offline"].sum()),
           "our_train4572_split_default": t.loc[t.index.isin(ex["our_train4572"]), "split_default"].value_counts().to_dict(),
           "our_eval141_split_default": t.loc[t.index.isin(ex["our_eval141"]), "split_default"].value_counts().to_dict(),
           "our_train4572_split_alpamayo": t.loc[t.index.isin(ex["our_train4572"]), "split_alpamayo"].value_counts().to_dict(),
           "our_eval141_split_alpamayo": t.loc[t.index.isin(ex["our_eval141"]), "split_alpamayo"].value_counts().to_dict(),
           "eligible_in_ood_reasoning_any_split": int(sum(1 for c in el.index if c in ood_idx)),
           "n_chunks_eligible": int(el["chunk"].nunique())}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(f"{OUT}/CENSUS_ELIGIBLE.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


def select():
    t = eligible_table()
    rows, m, ex, dx = steps(t)
    el = t[m].copy().sort_index()
    rng = np.random.default_rng(SEED)
    # stratified permutation: random order inside each stratum, key=(rank+u)/size, global sort by key ->
    # every prefix is proportionally stratified (200 < 10k < 45k < all are NESTED)
    key = np.empty(len(el))
    strat = el[STRATA].astype(str).agg("|".join, axis=1)
    for s, idx in sorted(strat.groupby(strat).indices.items()):
        n = len(idx)
        perm = rng.permutation(n)
        key[idx] = (perm + rng.random(n)) / n
    el["key"] = key
    el["stratum"] = strat
    el = el.sort_values(["key"], kind="stable")
    el["order"] = np.arange(len(el))
    os.makedirs(OUT, exist_ok=True)
    el[["order", "chunk", "country", "platform_class", "month", "hour_of_day", "has_obstacle.offline", "stratum"]].to_parquet(f"{OUT}/order_all_eligible.parquet")
    pilot = el.index[:PILOT_N].tolist()
    open(f"{OUT}/pilot_clips.txt", "w", encoding="utf-8").write("\n".join(pilot) + "\n")
    led = [json.loads(ln) for ln in open("/root/data/v2ep_ledger.jsonl", encoding="utf-8") if ln.strip()]
    cand = sorted({d["clip"] for d in led if d["kind"] == "train"})
    r2 = np.random.default_rng(SEED + 1)
    ctl = [cand[i] for i in r2.choice(len(cand), N_CONTROL, replace=False)]
    open(f"{OUT}/control_clips.txt", "w", encoding="utf-8").write("\n".join(ctl) + "\n")
    from tanitad.data import parity
    kept, rec = parity.guard_corpus_build(pilot, label="datascale pilot", role="train", mode="refuse")
    assert len(kept) == len(pilot)
    pil = el.iloc[:PILOT_N]
    rep = {"seed": SEED, "pilot_n": PILOT_N, "n_distinct_chunks": int(pil["chunk"].nunique()), "by_country": pil["country"].value_counts().to_dict(),
           "by_platform": pil["platform_class"].value_counts().to_dict(), "with_obstacle_offline": int(pil["has_obstacle.offline"].sum()),
           "guard_record": rec, "n_eligible": len(el)}
    json.dump(rep, open(f"{OUT}/SELECTION_REPORT.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "guard_record"}, indent=1))


if __name__ == "__main__":
    {"census": census, "select": select}[sys.argv[1]]()
