#!/usr/bin/env python3
"""inventory.py -- machine-checked inventory for the aux-raster veto feasibility package (read-only, CPU, numpy only).

Writes INVENTORY.json next to itself. Computes NO pick outcome, NO PDMS delta and NO veto outcome on any raster:
the only table statistic reported is the base rate of DAC = 0 over ALL proposals (the denominator of the Task-2 agreement),
and the scorer-only reproduction of B's pick from the stored logits (no raster, no sub-score lookup at the pick).

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I inventory.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent                                   # .../2026-09-20-refe-plan
sys.path.insert(0, str(PK / "refe"))
sys.path.insert(0, str(PK / "eval"))
import aux_label_store as A  # noqa: E402
import rl_val_eval as RV  # noqa: E402

ART = Path("D:/Projects/TanitAD-artifacts")
C3 = PK / "raw/2026-10-11-c3-selection-headroom"
STORE_TEST = ART / "aux_drivable/label_store_navtest"
STORE_TRAIN = ART / "aux_drivable/label_store"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fileinfo(path):
    p = Path(path)
    return {"path": str(path).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}


def npz_shapes(path):
    z = np.load(path, allow_pickle=False)
    return {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}


import time
_T0 = time.time()


def stage(n):
    print(f"[{time.time() - _T0:7.1f}s] {n}", flush=True)


out = {"tool": "inventory/1"}

# ---------------------------------------------------------------------------------------------- C3 table (u400)
tab = np.load(C3 / "table.npz", allow_pickle=False)
prp = np.load(C3 / "proposals.npz", allow_pickle=False)
tok = [str(t) for t in tab["token"]]
tj = json.load(open(C3 / "tokens.json"))
token_log = {str(k): v for k, v in tj["token_log"].items()}
out["c3"] = {
    "files": {n: fileinfo(C3 / n) for n in ("table.npz", "proposals.npz", "tokens.json", "gate_fidelity.json", "gate_table.json", "STATUS.json")},
    "table_shapes": npz_shapes(C3 / "table.npz"),
    "proposals_shapes": npz_shapes(C3 / "proposals.npz"),
    "n_tokens": len(tok), "n_unique_tokens": len(set(tok)), "n_logs": len(set(token_log.values())),
    "sub_names": [str(s) for s in tab["sub_names"]],
    "scorer": {"ckpt": str(prp["ckpt"]), "rule": str(prp["rule"]), "select": str(prp["select"]),
               "repair_last_heading": bool(prp["repair_last_heading"]), "sampling_horizon_s_and_dt_s": [float(x) for x in prp["sampling"]]},
    "valid_all": bool(tab["valid"].all()),
    "tokens_json_order_equals_table_order": [str(t) for t in tj["tokens"]] == tok,
    "tokens_json_set_equals_table_set": set(map(str, tj["tokens"])) == set(tok),
    "proposals_identical_between_the_two_npz": bool(np.array_equal(tab["proposals"], prp["proposals"])),
    "logits_identical_between_the_two_npz": bool(np.array_equal(tab["logits"], prp["logits"])),
    "pick_identical_between_the_two_npz": bool(np.array_equal(tab["pick"], prp["pick"])),
    "pose_extent": {"x": [float(tab["proposals"][..., 0].min()), float(tab["proposals"][..., 0].max())],
                    "y": [float(tab["proposals"][..., 1].min()), float(tab["proposals"][..., 1].max())],
                    "heading_rad": [float(tab["proposals"][..., 2].min()), float(tab["proposals"][..., 2].max())]},
}
sub = tab["sub"]
dac = sub[:, :, 1]
out["c3"]["dac_values"] = [float(v) for v in np.unique(dac)]
out["c3"]["dac_zero_share_over_ALL_proposals"] = float((dac == 0).mean())
out["c3"]["dac_zero_n_over_ALL_proposals"] = int((dac == 0).sum())
out["c3"]["n_proposals_total"] = int(dac.size)
out["c3"]["tokens_with_every_proposal_dac1"] = int((dac == 1).all(1).sum())
out["c3"]["tokens_with_every_proposal_dac0"] = int((dac == 0).all(1).sum())

# (c) scorer-only reproduction of B's pick from the stored logits (NO raster, NO sub-score lookup at the pick)
stage("pick reproduction")
rep = 0
for i in range(len(tok)):
    j, _ = RV.select(tab["logits"][i].astype(np.float64), "navsim_v1_lane", space="logit")
    rep += int(j == int(tab["pick"][i]))
out["c3"]["pick_reproduced_from_logits_by_rl_val_eval_select_navsim_v1_lane"] = {"n": len(tok), "reproduced": rep}

stage("GT label store (navtest)")
# ---------------------------------------------------------------------------------------------- GT label store (navtest)
man = json.load(open(STORE_TEST / "manifest.json"))
out["gt_store_navtest"] = {
    "path": str(STORE_TEST).replace("\\", "/"), "manifest": fileinfo(STORE_TEST / "manifest.json"),
    "store_sha256": man["store_sha256"], "raster": man["raster"], "counts": man["counts"],
    "n_shard_files_on_disk": len(list((STORE_TEST / "shards").glob("*.aux1"))),
}
heads, miss, origin = {}, [], {}
for t in tok:
    lg = token_log[t]
    if lg not in heads:
        hd, _ = A.read_shard(STORE_TEST / "shards" / f"{lg}.aux1", with_blobs=False)
        heads[lg] = ({e["t"]: e for e in hd["entries"]}, hd["map"])
    ent = heads[lg][0].get(t)
    if ent is None:
        miss.append(t)
    else:
        origin[t] = ent["p"]
out["gt_store_navtest"]["c3_tokens_missing_from_store"] = len(miss)
out["gt_store_navtest"]["c3_tokens_found"] = len(tok) - len(miss)
out["gt_store_navtest"]["c3_maps"] = {m: sum(1 for t in tok if heads[token_log[t]][1] == m) for m in sorted({h[1] for h in heads.values()})}
out["gt_store_navtest"]["entry_sources_for_c3"] = sorted({heads[token_log[t]][0][t]["s"] for t in tok if t in origin})
# one decode to pin the dtype/shape/range
s0 = A.read_label(STORE_TEST, token_log[tok[0]], tok[0])
out["gt_store_navtest"]["decoded_example"] = {"token": tok[0], "shape": list(s0.shape), "dtype": str(s0.dtype), "min": int(s0.min()), "max": int(s0.max())}
man_tr = json.load(open(STORE_TRAIN / "manifest.json"))
out["gt_store_train"] = {"path": str(STORE_TRAIN).replace("\\", "/"), "store_sha256": man_tr["store_sha256"], "counts": man_tr["counts"],
                         "coverage_vs_navtrain_yaml": man_tr["coverage_vs_navtrain_yaml"]}

stage("predicted rasters")
# ---------------------------------------------------------------------------------------------- predicted rasters
folds = json.load(open(PK / "raw/2026-10-16-scene-tap-a2/folds.json"))
out["predicted"] = {"folds_json": fileinfo(PK / "raw/2026-10-16-scene-tap-a2/folds.json"),
                    "folds_rows_order_equals_table_order": [r[0] for r in folds["rows"]] == tok,
                    "fold_rule": folds["rule"], "logs": folds["n_logs"], "tokens": folds["n_tokens"]}


def probe_dir(d):
    res = {"dir": str(d).replace("\\", "/"), "units": {}}
    cover = {}
    for p in sorted(Path(d).glob("*_f?.npz")):
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta_json"]))
        res["units"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p), "updates": m["info"].get("updates"), "n_held": int(len(z["held_idx"])),
                                "logit_shape": list(z["logit"].shape) if "logit" in z.files else None,
                                "sdf_shape": list(z["sdf"].shape) if "sdf" in z.files else None, "arm": m.get("arm"), "seed": m.get("seed")}
        cover.setdefault(p.name.rsplit("_f", 1)[0], set()).update(int(i) for i in z["held_idx"])
    res["held_out_coverage_by_arm_seed"] = {k: len(v) for k, v in cover.items()}
    return res


out["predicted"]["A2_V_s0_10000_updates"] = probe_dir(ART / "scene_tap_probe_a2/units")
out["predicted"]["A3R_V_s0_400_updates"] = probe_dir(ART / "scene_tap_probe_a3r/units")
out["predicted"]["PROBE_2000_updates_V_S_C_seeds"] = probe_dir(ART / "scene_tap_probe/units")
# the units that are pure init (no held_idx) are skipped by the glob; list them for completeness
out["predicted"]["PROBE_2000_updates_INIT_files"] = sorted(p.name for p in (ART / "scene_tap_probe/units").glob("INIT_*.npz"))

stage("ctx caches (the input of any NEW head)")
# ctx caches (the input of any NEW head)
def count(d, pat):
    return len(list(Path(d).glob(pat)))


out["ctx_caches"] = {
    "visual_ctx_fp32_[7680,256]": {"dir": str(ART / "aux_navtest_forward/c3_full_ctx").replace("\\", "/"), "ctx_files": count(ART / "aux_navtest_forward/c3_full_ctx", "ctx_*.npy"),
                                   "logits_files": count(ART / "aux_navtest_forward/c3_full_ctx", "logits_*.npy")},
    "scene_ctx_fp32_[64,256]": {"dir": str(ART / "aux_navtest_forward/c3_full_scene_ctx").replace("\\", "/"), "ctx_files": count(ART / "aux_navtest_forward/c3_full_scene_ctx", "ctx_*.npy"),
                                "logits_files": count(ART / "aux_navtest_forward/c3_full_scene_ctx", "logits_*.npy")},
}
sm = ART / "aux_navtest_forward/c3_full_scene_ctx/scene_ctx_manifest.json"
if sm.exists():
    out["ctx_caches"]["scene_ctx_manifest"] = fileinfo(sm)
vc = [p.stem[4:] for p in (ART / "aux_navtest_forward/c3_full_ctx").glob("ctx_*.npy")]
out["ctx_caches"]["visual_ctx_tokens_equal_c3"] = set(vc) == set(tok)
x = np.load(ART / "aux_navtest_forward/c3_full_ctx" / f"ctx_{tok[0]}.npy", mmap_mode="r")
out["ctx_caches"]["visual_ctx_example"] = {"shape": list(x.shape), "dtype": str(x.dtype), "bytes_per_token": int(x.nbytes)}

stage("S1 heads (trained on navtrain)")
# S1 heads (trained on navtrain)
s1 = ART / "s1_read_cache/arms"
out["s1_heads"] = {a.name: {"aux_drv.pt": ((a / "aux_drv.pt").stat().st_size if (a / "aux_drv.pt").exists() else None)} for a in sorted(s1.glob("*"))}
scratch = Path("C:/Users/Admin/AppData/Local/Temp/shf_scratch")
out["s1_navtest_predictions_scratch"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(scratch.glob("navtest_head_pred_H1-s*.npz"))}
# the fixture (never trained) head dump that sits in c3_full/ -- flag, do not use
fx = ART / "aux_navtest_forward/c3_full/head_pred_H2-s0.npz"
out["fixture_untrained_head_dump"] = {"path": str(fx).replace("\\", "/"), "present": fx.exists(), "note": "FIXTURE head, never trained (aux_navtest_forward RESULT section 3); plumbing only, NOT a prediction"}

stage("P2 table")
# ---------------------------------------------------------------------------------------------- P2 table
P2 = ART / "p2_zero_tail_table"
t2 = np.load(P2 / "table.npz", allow_pickle=False)
t2tok = [str(t) for t in t2["token"]]
out["p2_table"] = {"dir": str(P2).replace("\\", "/"), "table.npz": fileinfo(P2 / "table.npz"), "proposals.npz": fileinfo(P2 / "proposals.npz"),
                   "shapes": {k: [list(t2[k].shape), str(t2[k].dtype)] for k in t2.files}, "n_tokens": len(t2tok),
                   "overlap_with_c3": len(set(t2tok) & set(tok)), "not_in_c3": len(set(t2tok) - set(tok)),
                   "full_navtest_p2_table": "p2_full_table/dumps/dump_main_c00.npz only (dump in progress, no P2FULL_DONE marker at read time)"}
pf = ART / "p2_full_table/dumps/dump_main_c00.npz"
if pf.exists():
    zz = np.load(pf, allow_pickle=False)
    out["p2_table"]["p2_full_dump_c00"] = {"file": fileinfo(pf), "shapes": {k: [list(zz[k].shape), str(zz[k].dtype)] for k in zz.files}}

stage("RL-val")
# ---------------------------------------------------------------------------------------------- RL-val
RL = Path("C:/Users/Admin/AppData/Local/Temp/rls1dl")
rl = {"present": RL.exists()}
if RL.exists():
    rl["tarballs"] = {n: fileinfo(RL / "data" / n) for n in ("rlfan_pdm_val.tar.gz", "rlfan_sets_val.tar.gz", "rlfan_terms_val.tar.gz")}
    keys = []
    with tarfile.open(RL / "data/rlfan_pdm_val.tar.gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                for line in tf.extractfile(m):
                    r = json.loads(line)
                    keys.append((r["key"][0], r["key"][1]))
                    nprop = len(r["dac"])
    rl["n_sets"] = len(keys)
    rl["n_logs"] = len({k[0] for k in keys})
    rl["proposals_per_set"] = nprop
    # coverage in the TRAIN label store (RL-val is navtrain-held-out)
    cov = 0
    hd_cache = {}
    for lg, t in keys:
        if lg not in hd_cache:
            p = STORE_TRAIN / "shards" / f"{lg}.aux1"
            hd_cache[lg] = {e["t"] for e in A.read_shard(p, with_blobs=False)[0]["entries"]} if p.exists() else set()
        cov += int(t in hd_cache[lg])
    rl["sets_with_gt_raster_in_train_store"] = cov
    rl["keys_unique"] = len(set(keys))
    sc = RL / "final/scores_step0_A.npz"
    zs = np.load(sc, allow_pickle=True)
    rl["scores_step0_A"] = {**fileinfo(sc), "equals_FIT_scores_sha256_6332fcc6": sha256(sc).startswith("6332fcc6"),
                            "shapes": {k: [list(zs[k].shape), str(zs[k].dtype)] for k in zs.files},
                            "keys_equal_pdm_keys": set(zip(map(str, zs["log"]), map(str, zs["token"]))) == set(keys)}
    ho = [l.strip() for l in open(ART / "_tlane_work/ho/refe_heldout/heldout_logs.txt") if l.strip()]
    rl["s1_heldout_logs_n"] = len(ho)
    rl["rl_val_logs_in_common_with_s1_heldout_logs"] = len({k[0] for k in keys} & set(ho))
out["rl_val"] = rl

json.dump(out, open(HERE / "INVENTORY.json", "w"), indent=1, default=str)
print("wrote", HERE / "INVENTORY.json")
print(json.dumps({k: out[k] for k in ("c3",) if k in out}, indent=1, default=str)[:1500])
