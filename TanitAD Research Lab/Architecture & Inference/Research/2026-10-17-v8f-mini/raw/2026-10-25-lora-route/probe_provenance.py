"""PREREG_V8F_LORA route question, probe 1 (zero GPU, zero box 2, reads only files already on this box): WHAT DOES THE PRODUCTION TRAIN CACHE RECORD ABOUT ITS OWN ROUTE,
and how far is each LOCAL reference cache from it?

Reads (all existing, read-only):
  * box 2's production cache manifests, pulled earlier by the chains: box2_ship/logs/v8f_dscale/manifest_train.jsonl (4,572 train clips), box2_ship/logs/v8f_grid/manifest_eval.jsonl (141 eval clips)
  * the two dev-box test caches the identity control used as its "train cache": refav1_hj_cache/test_runscript (thor416, GPU bf16) and test_cpu_fp32 (thor416, CPU fp32)
  * box 2's control result (ra1_full/ctl_result.json) and the dev-box S4b / S4c v2ep products (sha256)
Writes only <out>.json / stdout.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import torch

BOX2 = Path("C:/Users/Admin/box2_ship/logs")
TRAIN_MAN = BOX2 / "v8f_dscale" / "manifest_train.jsonl"
EVAL_MAN = BOX2 / "v8f_grid" / "manifest_eval.jsonl"
CTL = BOX2 / "ra1_full" / "ctl_result.json"
TEST_BF16 = Path("C:/Users/Admin/refav1_hj_cache/test_runscript/train")
TEST_FP32 = Path("C:/Users/Admin/refav1_hj_cache/test_cpu_fp32/train")
S4B = Path("C:/Users/Admin/tanitad-data/refav1-train600-256")
S4C = Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb")
AUG30 = Path("C:/Users/Admin/tanitad-data/refav1-eval141/eps")
CLIPS = ["8f387977-4086-466d-bba4-b04f8cdc0743", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"]
EVAL_CLIP = "dca157a0-20a9-464b-a74c-084b678e7be1"


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def stats(mine: torch.Tensor, ref: torch.Tensor) -> dict:
    mine, ref = mine.float().reshape(-1), ref.float().reshape(-1)
    d = (mine - ref).abs()
    return {"n_values": int(d.numel()), "rel_l2": float((mine - ref).norm() / ref.norm().clamp_min(1e-12)), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "ref_mean_abs": float(ref.abs().mean()), "frac_bitwise_equal": float((mine == ref).float().mean())}


def main() -> int:
    out: dict = {}
    tm, em = rows(TRAIN_MAN), rows(EVAL_MAN)
    key = lambda r: (r.get("source"), r.get("enc_dtype"), r.get("device"), r.get("pipeline"), tuple(r.get("grids", [])))
    out["production_manifest"] = {
        "train": {"file": str(TRAIN_MAN), "n": len(tm), "n_distinct_clips": len({r["clip"] for r in tm}),
                  "census(source,enc_dtype,device,pipeline,grids)": [[list(k), v] for k, v in collections.Counter(map(key, tm)).items()]},
        "eval": {"file": str(EVAL_MAN), "n": len(em), "census(source,enc_dtype,device,pipeline,grids)": [[list(k), v] for k, v in collections.Counter(map(key, em)).items()]},
        "any_thor416_row": sum(1 for r in tm + em if "416" in str(r.get("source"))),
    }
    byclip = {r["clip"]: r for r in tm}
    # 1. are the dev-box test caches the production cache? (compare file md5 and tensor content_md5 against the production manifest)
    out["test_cache_vs_production_manifest"] = {}
    for c in CLIPS:
        prod = byclip[c]
        rec = {"production": {"md5": prod["md5"], "content_md5": prod["content_md5"], "source": prod["source"], "enc_dtype": prod["enc_dtype"], "device": prod["device"]}}
        for name, root in (("test_runscript(thor416,cuda,bf16)", TEST_BF16), ("test_cpu_fp32(thor416,cpu,fp32)", TEST_FP32)):
            m = {r["clip"]: r for r in rows(root / "manifest.jsonl")}[c]
            rec[name] = {"md5": md5(root / "clips" / f"{c}.pt"), "content_md5": m["content_md5"], "source": m["source"], "enc_dtype": m["enc_dtype"], "device": m["device"],
                         "md5_equals_production": md5(root / "clips" / f"{c}.pt") == prod["md5"], "content_md5_equals_production": m["content_md5"] == prod["content_md5"]}
        out["test_cache_vs_production_manifest"][c[:8]] = rec
    # 2. the S4b v2ep on this box is the file box 2 rebuilt byte-identically (ctl_result.json: expected = got = ref)
    ctl = json.load(open(CTL, encoding="utf-8"))
    out["s4b_v2ep_vs_box2_control"] = {}
    for c in CLIPS:
        p = S4B / f"{c}.v2ep.pt"
        s = sha256(p)
        out["s4b_v2ep_vs_box2_control"][c[:8]] = {"path": str(p), "sha256": s, "box2_ctl_expected": ctl["clips"][c]["expected"], "box2_ctl_got": ctl["clips"][c]["got"], "box2_ctl_verdict": ctl["clips"][c]["verdict"],
                                                  "local_equals_box2": s == ctl["clips"][c]["got"]}
    # 3. DEVICE-ONLY floor: the SAME frames (thor416 source) encoded CPU fp32 vs GPU bf16, pooled fp8 [T,40,1024]
    out["device_only_floor_same_frames"] = {"what": "test_cpu_fp32 (CPU fp32) vs test_runscript (dev-box GPU bf16), both source=thor416-resize, same clips, pool4x10 fp8, every frame", "clips": {}}
    for c in CLIPS:
        a = torch.load(TEST_FP32 / "clips" / f"{c}.pt", map_location="cpu", weights_only=True)["pool4x10"]
        b = torch.load(TEST_BF16 / "clips" / f"{c}.pt", map_location="cpu", weights_only=True)["pool4x10"]
        assert a.shape == b.shape and a.dtype == torch.float8_e4m3fn
        s = stats(a, b)
        s["n_frames"] = int(a.shape[0])
        out["device_only_floor_same_frames"]["clips"][c[:8]] = s
    # 4. the eval side: Aug-30 eps vs the S4c rebuild that box 2's eval cache was encoded from (ground-truth-free: byte compare)
    out["eval_products"] = {"clip": EVAL_CLIP, "aug30_eps_sha256": sha256(AUG30 / f"{EVAL_CLIP}.v2ep.pt"), "s4c_rebuild_sha256": sha256(S4C / f"{EVAL_CLIP}.v2ep.pt")}
    out["eval_products"]["byte_identical"] = out["eval_products"]["aug30_eps_sha256"] == out["eval_products"]["s4c_rebuild_sha256"]
    json.dump(out, open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
