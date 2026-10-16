#!/usr/bin/env python3
"""H-a: is the LABEL CONSTRUCTED differently in the navtrain store and the navtest store?  (CPU, read-only)

Evidence in four layers, each of which could have shown a difference and is reported either way:
  L1  provenance: code sha256, raster spec, map root/version, schema, builder argv, origin source mix, log->map source  (manifests + run_meta + shard headers)
  L2  the code that produced them: sha256 of refe/aux_label_store.py at the tip vs the code_sha256 both builds recorded
  L3  the banked path-A (NuPlanScenario) vs path-B equivalence gates of each store (cells differing, origin deviation)
  L4  data level: per-map marginal label statistics on EVERY key of both stores (what a construction difference would shift
      even if the code hashes agreed: coordinate frame, cell offset, sign, scale, origin)
Writes a1_label_construction.json (+ a per-frame feature table in the scratch dir for the later scripts).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C  # noqa: E402

AUXDIR, PK, HERE = C.AUXDIR, C.PK, C.HERE
ALS = None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def l1_l2_l3():
    out = {}
    man = {s: json.load(open(f"{AUXDIR}/{d}/manifest.json", encoding="utf-8")) for s, d in (("train", "label_store"), ("navtest", "label_store_navtest"))}
    rm = {s: json.load(open(f"{AUXDIR}/{d}/run_meta_all.json", encoding="utf-8")) for s, d in (("train", "label_store"), ("navtest", "label_store_navtest"))}
    out["schema"] = {s: man[s]["schema"] for s in man}
    out["raster_equal"] = man["train"]["raster"] == man["navtest"]["raster"]
    out["raster"] = man["train"]["raster"]
    out["run_meta_raster_equal"] = rm["train"]["raster"] == rm["navtest"]["raster"]
    ins = {s: rm[s]["inputs"] for s in rm}
    out["code_sha256"] = {s: ins[s]["code_sha256"] for s in ins}
    out["code_sha256_equal"] = ins["train"]["code_sha256"] == ins["navtest"]["code_sha256"]
    out["maps_json_sha256"] = {s: ins[s]["maps_json_sha256"] for s in ins}
    out["maps_root"] = {s: ins[s]["maps_root"] for s in ins}
    out["maps_json_sha256_equal"] = ins["train"]["maps_json_sha256"] == ins["navtest"]["maps_json_sha256"]
    out["db_root"] = {s: ins[s]["db_root"] for s in ins}
    out["origin_source_mix"] = {s: man[s]["counts"]["by_src"] for s in man}
    out["by_map"] = {s: man[s]["counts"]["by_map"] for s in man}
    out["workers"] = {s: rm[s]["workers"] for s in rm}
    out["origin_xcheck_bank_vs_db"] = rm["train"]["origin_xcheck"]
    # L2: the file at the tip, LF and CRLF forms
    p = os.path.join(C.PK, "refe", "aux_label_store.py")
    b = open(p, "rb").read()
    lf = b.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    cand = {"tip_as_checked_out": sha256_bytes(b), "tip_LF": sha256_bytes(lf), "tip_CRLF": sha256_bytes(crlf)}
    out["code_sha_candidates"] = cand
    out["code_sha_matches_a_form_of_tip_file"] = [k for k, v in cand.items() if v == ins["train"]["code_sha256"]]
    # L3: the banked gates
    pre = json.load(open(os.path.join(C.PK, "raw", "2026-10-12-aux-label-store", "s0_gate_pre.json"), encoding="utf-8"))
    nt = json.load(open(os.path.join(C.PK, "raw", "2026-10-13-aux-navtest-store", "s0_gate_navtest.json"), encoding="utf-8"))
    def pick(g):
        m = g.get("metrics", {})
        return {k: m.get(k) for k in ("samples", "proposals", "ab_cells_differing", "ab_samples", "origin_max_m", "origin_max_rad",
                                       "corner_agree_bilinear", "verdict_agree_bilinear", "target_inside_frac")}
    out["gate_train_pre"] = pick(pre)
    out["gate_navtest"] = pick(nt)
    out["gate_train_pre_pass"] = pre.get("verdict", {}).get("pass")
    out["gate_navtest_pass"] = nt.get("verdict", {}).get("pass")
    out["gate_train_post_exists"] = os.path.exists(os.path.join(C.PK, "raw", "2026-10-12-aux-label-store", "s0_gate_post.json"))
    # log->map source
    lm_tr = json.load(open(f"{AUXDIR}/log_map.json", encoding="utf-8"))
    lm_nt = json.load(open(f"{AUXDIR}/label_store_navtest/inputs/log_map_navtest.json", encoding="utf-8"))
    out["logmap_train_summary"] = lm_tr["summary"]
    out["logmap_navtest_summary"] = lm_nt.get("summary")
    from collections import Counter
    out["logmap_train_src"] = dict(Counter(v["src"] for v in lm_tr["logs"].values()))
    out["logmap_navtest_src"] = dict(Counter(v["src"] for v in lm_nt["logs"].values()))
    return out, lm_tr, lm_nt


def shard_headers(store_dir):
    import aux_label_store as ALS_
    rows = []
    for f in sorted(os.listdir(f"{store_dir}/shards")):
        head, _ = ALS_.read_shard(__import__("pathlib").Path(f"{store_dir}/shards/{f}"), with_blobs=False)
        for e in head["entries"]:
            rows.append((head["log"], e["t"], head["map"], e["s"], e["p"][0], e["p"][1], e["p"][2], (e.get("q") or [-1, -1])[0], (e.get("q") or [-1, -1])[1]))
    return rows


def label_features(src, keys, chunk=512):
    """per-frame near-field features from the label, on the head's own 2 m target grid (AUX.build_targets), band x in [0,40)."""
    band = C.RO.band_cells(0.0, 40.0)
    n = len(keys)
    F = {k: np.zeros(n, np.float32) for k in ("near_frac", "near_bound", "sdf_origin_m", "sdf_origin_fine_m", "left_frac", "right_frac", "far_frac")}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lab, valid = src.labels(keys[s:e])
        t = C.AUX.build_targets(lab, valid)
        m = (t.mask_frac.numpy() >= 0.5)
        mb = m & band[None]
        F["near_frac"][s:e] = mb.sum((1, 2)) / band.sum()
        # boundary cells: a drivable cell with at least one non-drivable 4-neighbour inside the band
        pad = np.pad(m, ((0, 0), (1, 1), (1, 1)), constant_values=True)
        nb = ~(pad[:, :-2, 1:-1] & pad[:, 2:, 1:-1] & pad[:, 1:-1, :-2] & pad[:, 1:-1, 2:])
        F["near_bound"][s:e] = (m & nb & band[None]).sum((1, 2))
        lab_np = lab.numpy().astype(np.float32) * 0.1
        F["sdf_origin_fine_m"][s:e] = lab_np[:, 16, 64]                      # fine cell x in [0,0.5), y in [0,0.5): the rear axle
        F["sdf_origin_m"][s:e] = t.sdf_m.numpy()[:, 4, 16]                   # coarse cell x in [0,2), y in [0,2)
        F["left_frac"][s:e] = (m[:, 4:24, 16:] & True).sum((1, 2)) / (20 * 16)
        F["right_frac"][s:e] = (m[:, 4:24, :16]).sum((1, 2)) / (20 * 16)
        F["far_frac"][s:e] = (m[:, 24:, :]).sum((1, 2)) / (16 * 32)
    return F


def main():
    t0 = time.time()
    res, lm_tr, lm_nt = l1_l2_l3()
    print("L1-L3 done", time.time() - t0, flush=True)
    # shard headers
    ds = {}
    for s, d in (("train", "label_store"), ("navtest", "label_store_navtest")):
        rows = shard_headers(f"{AUXDIR}/{d}")
        ds[s] = rows
    res["shard_header_counts"] = {s: len(r) for s, r in ds.items()}
    res["origin_source_by_map_from_shards"] = {}
    from collections import Counter
    for s, rows in ds.items():
        c = Counter((r[2], r[3]) for r in rows)
        res["origin_source_by_map_from_shards"][s] = {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}
    print("shard headers", time.time() - t0, flush=True)
    # per-frame features, all keys
    held = set(C.heldout_logs())
    feats = {}
    for s, src in (("train", C.open_train_labels()), ("navtest", C.open_navtest_labels())):
        keys = [(r[0], r[1]) for r in ds[s]]
        F = label_features(src, keys)
        maps = np.array([r[2] for r in ds[s]])
        srcs = np.array([r[3] for r in ds[s]])
        feats[s] = dict(keys=keys, maps=maps, srcs=srcs, **F)
        print("features", s, len(keys), time.time() - t0, flush=True)
    np.savez_compressed(f"{C.SCRATCH}/label_feats.npz",
                        **{f"{s}_{k}": (np.array([x[0] + "|" + x[1] for x in v]) if k == "keys" else v) for s, d in feats.items() for k, v in d.items()})
    # tables
    c3 = C.c3()
    c3set = set(c3["tokens"])
    sets = {
        "train_all": ("train", np.ones(len(feats["train"]["keys"]), bool)),
        "train_heldout_logs": ("train", np.array([k[0] in held for k in feats["train"]["keys"]])),
        "train_nonheldout": ("train", np.array([k[0] not in held for k in feats["train"]["keys"]])),
        "navtest_all": ("navtest", np.ones(len(feats["navtest"]["keys"]), bool)),
        "navtest_c3_1058": ("navtest", np.array([k[1] in c3set for k in feats["navtest"]["keys"]])),
    }
    tab = {}
    for name, (s, msk) in sets.items():
        d = feats[s]
        row = {"n": int(msk.sum()), "logs": len({k[0] for k, m in zip(d["keys"], msk) if m})}
        for f in ("near_frac", "near_bound", "sdf_origin_m", "sdf_origin_fine_m", "left_frac", "right_frac", "far_frac"):
            v = d[f][msk]
            row[f] = {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        row["map_share"] = {m: float((d["maps"][msk] == m).mean()) for m in sorted(set(d["maps"]))}
        row["origin_inside_share"] = float((d["sdf_origin_fine_m"][msk] > 0).mean())
        tab[name] = row
    res["label_stats"] = tab
    # per-map, train_all vs navtest_all (the construction-difference test: same map, same code => same marginal law)
    pm = {}
    for m in sorted(set(feats["train"]["maps"])):
        r = {}
        for s in ("train", "navtest"):
            d = feats[s]
            k = d["maps"] == m
            r[s] = {"n": int(k.sum()), **{f: float(d[f][k].mean()) for f in ("near_frac", "near_bound", "sdf_origin_fine_m", "left_frac", "right_frac", "far_frac")},
                    "origin_inside": float((d["sdf_origin_fine_m"][k] > 0).mean())}
        pm[m] = r
    res["per_map_train_vs_navtest"] = pm
    C.jdump(res, os.path.join(HERE, "a1_label_construction.json"))
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
