#!/usr/bin/env python3
"""Generates manifest_a5.json (both pairs), manifest_a5_hb.json and manifest_a5_hc.json (one arm pair each, the per-arm manifests of POD_RUN_PLAN step F).
Pure text generation: opens no data file.  Arm-file sha256 values are the sentinel PENDING (the files do not exist yet); fill_manifest_sha.py replaces them from the HF LFS listing / pod sha256sum
BEFORE the reader runs, and the reader REFUSES a sentinel (sha mismatch), so an unfilled manifest fails closed.
usage: python make_manifests_a5.py <out_dir>"""
import json, os, sys

PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ROOT = "F:/tanitad_scratch/a5read"
CODE = f"{ROOT}/code/{PKR}"
VDAC = f"{CODE}/raw/2026-11-09-scorer1p6-prereg/vdac"
PEND = "PENDING"

def f(path, sha):
    return {"path": path, "sha256": sha}

HA = {   # fixed values, copied from manifest_s1f_v2.json (the Stage 0 / S1a precedent); paths are the A5 layout (the old scratch trees were deleted: re-pull, the reader checks the sha)
    "export": f(f"{ROOT}/pulled/ha/export/h48_proposals_subscores.npz", "14560686490a2edbb0a9611d69f1285f84b396bad1c28138cf192964b2809345"),
    "export_json": f(f"{ROOT}/pulled/ha/export/h48_proposals_subscores.npz.json", "0f3fbeed3c8a21d08d574e2c68c94ac38ea55e0e42ea5fb8187015b2b7678802"),
    "head_s0_a": f(f"{ROOT}/pulled/ha/ha-s0/head_pred_update10000.npz", "6b9ac2bc1bc5989e7904e1c1b1d12eb6e4c2356fe7ad5658cc6fecc5196c03eb"),
    "head_s0_b": f(f"{ROOT}/pulled/ha/ha-s0/head_pred_h48x.npz", "64d2f71be45154bd36d71d3955a9151023d6f10365f9bac5376c6e3ce02aac3b"),
    "head_shuf_a": f(f"{ROOT}/pulled/ha/ha-shuf/head_pred_update2000.npz", "5cb8c49255de67b8073bfc45cb75f2356c91bdc381cb222cfe608d9d79fb5beb"),
    "head_shuf_b": f(f"{ROOT}/pulled/ha/ha-shuf/head_pred_h48x.npz", "4ee7816cc7a19cacd8a13101f51e6b7ae151dfcc90f2000edcf4b2c627161ba3"),
    "h48_logs": f(f"{VDAC}/s1/h48/h48_logs.txt", "2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8"),
    "head_init0_a": f(f"{ROOT}/pulled/ha/stage0/head_pred_step0.npz", "2ebe4b540b163e2638532a71d0a006a35e7ce9e44ee0c0cb1bce8fd10903f94a"),
    "head_init0_b": f(f"{ROOT}/pulled/ha/stage0/fwd/head_init0_b.npz", "8b556eb23c842732ec91837781f7181c1f90406f3a1aa2316e2e9493bb542ed4"),
    # the repo copy at tip e262e2cc hashes 1547bd30..., NOT the 5ab0c72c... of the Stage 0 manifest (UNVERIFIED why; the value 0.7864227917638612 equals s1f_read_v2.json)
    "in_sample": f(f"{CODE}/raw/2026-11-12-vdac-stage0-read/in_sample_iou.json", "1547bd30ea6fba32f2a499da186b41766c1daa04ce23a542e0df804b7e1d6f9c"),
}
HA_ARMS = [
    {"name": "HA-s0", "role": "trained", "files": ["head_s0_a", "head_s0_b"], "cell": 2.0, "shuf": "HA-SHUF", "init": "HA-s0-initial", "in_sample_iou_file": "in_sample"},
    {"name": "HA-SHUF", "role": "shuf", "files": ["head_shuf_a", "head_shuf_b"], "cell": 2.0},
    {"name": "HA-s0-initial", "role": "init", "files": ["head_init0_a", "head_init0_b"], "cell": 2.0},
]

def arm_files(x):      # x = "hb" | "hc"; pod-relative names kept under pulled/<arm lower>/
    return {f"{x}_s0_a": f(f"{ROOT}/pulled/{x}-s0/scores/head_pred_update10000.npz", PEND),
            f"{x}_s0_b": f(f"{ROOT}/pulled/{x}-s0/head_pred_h48x.npz", PEND),
            f"{x}_shuf_a": f(f"{ROOT}/pulled/{x}-shuf/scores/head_pred_update2000.npz", PEND),
            f"{x}_shuf_b": f(f"{ROOT}/pulled/{x}-shuf/head_pred_h48x.npz", PEND),
            f"{x}_init_a": f(f"{ROOT}/pulled/{x}-s0/scores/head_pred_step0.npz", PEND)}

def hb():
    files = arm_files("hb")
    files["hb_init_b"] = dict(HA["head_init0_b"])            # PREREG 7.5: HA's head_init0_b (identical seed-0 head, G21); same file, same fixed sha
    arms = [{"name": "HB-s0", "role": "trained", "files": ["hb_s0_a", "hb_s0_b"], "cell": 2.0, "shuf": "HB-SHUF", "init": "HB-s0-initial"},
            {"name": "HB-SHUF", "role": "shuf", "files": ["hb_shuf_a", "hb_shuf_b"], "cell": 2.0},
            {"name": "HB-s0-initial", "role": "init", "files": ["hb_init_a", "hb_init_b"], "cell": 2.0}]
    return files, arms

def hc(with_init_arm):
    files = arm_files("hc")            # hc_init_a is a FILE entry (sha-checked, read by the S1-F wrapper); there is no hc_init_b (PREREG 7.5: not built)
    arms = [{"name": "HC-s0", "role": "trained", "files": ["hc_s0_a", "hc_s0_b"], "cell": 1.0, "shuf": "HC-SHUF"},
            {"name": "HC-SHUF", "role": "shuf", "files": ["hc_shuf_a", "hc_shuf_b"], "cell": 1.0}]
    if with_init_arm:
        arms[0]["init"] = "HC-s0-initial"
        arms.append({"name": "HC-s0-initial", "role": "init", "files": ["hc_init_a"], "cell": 1.0})
    return files, arms

def build(which):
    files = {k: dict(v) for k, v in HA.items()}
    arms = [dict(a) for a in HA_ARMS]
    pairs = []
    if which in ("both", "hb"):
        fl, ar = hb(); files.update(fl); arms += ar; pairs.append({"x": "HB-s0", "ref": "HA-s0"})
    if which in ("both", "hc"):
        fl, ar = hc(with_init_arm=False); files.update(fl); arms += ar; pairs.append({"x": "HC-s0", "ref": "HA-s0"})
    return {"cards_dir": VDAC, "store": "E:/Projects/TanitAD-artifacts/aux_drivable/label_store", "files": files,
            "note": "A5 reader manifest (PREREG_VDAC_P6_A5 7.5). HA rows = the Stage 0 manifest values; arm-file sha256 = PENDING until fill_manifest_sha.py (HF LFS sha256 for the npz pushed to Sayood/tanitad-refe-scorer1p6; pod sha256sum for scores/head_pred_step0.npz, which is not pushed). aux_drv.pt not read by the reader.",
            "arms": arms, "pairs": pairs}

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for which, name in (("both", "manifest_a5.json"), ("hb", "manifest_a5_hb.json"), ("hc", "manifest_a5_hc.json")):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(build(which), fh, indent=1)
            fh.write("\n")
        print("wrote", name)
