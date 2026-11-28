"""Dry-run STAND-IN for theory/wmpool/zcells.py (dev box only; named by WMPOOL_TEST_ZCELLS, honoured only with WMPOOL_DRY=1). No torch, no model, no raw feature cache: it writes tiny files in the layout the real builder writes
(<out>/M2b_s{k}/{train,eval}/<clip>.npz, CELL_MANIFEST.json, and the check3 JSON with the keys zcells.py check3 writes), steered by the `cells_mode` of the file in PLANT_CONTROL (json).
Modes: ok | build_crash | check3_fail | wrongseed_passes | stored_off | short_counts | same_adapter | tmp_left | no_manifest.   Counts come from WMPOOL_TEST_CELL_COUNTS ("train=6,eval=3")."""
import hashlib
import json
import os
import sys

BS = chr(92)


def adapter_sha(seed):
    return hashlib.sha256(("adapter-%d" % seed).encode()).hexdigest()


def arg(argv, flag, n=1):
    i = argv.index(flag)
    return argv[i + 1:i + 1 + n] if n else argv[i + 1:]


def main():
    mode_, argv = sys.argv[1], sys.argv[2:]
    ctl = json.load(open(os.environ["PLANT_CONTROL"])) if os.environ.get("PLANT_CONTROL") and os.path.exists(os.environ["PLANT_CONTROL"]) else {}
    cm = ctl.get("cells_mode", "ok")
    counts = {k: int(v) for k, v in (x.split("=") for x in os.environ["WMPOOL_TEST_CELL_COUNTS"].split(","))}
    ck = []
    i = argv.index("--ckpt") + 1
    while i < len(argv) and not argv[i].startswith("--"):
        ck.append(argv[i])
        i += 1
    seeds = [int(x.split("=")[0]) for x in ck]
    if mode_ == "build":
        out = arg(argv, "--out")[0]
        if cm == "build_crash":
            return 3
        for k in seeds:
            for split, n in counts.items():
                if cm == "short_counts" and k == 1 and split == "train":
                    n -= 1
                d = os.path.join(out, f"M2b_s{k}", split)
                os.makedirs(d, exist_ok=True)
                for j in range(n):
                    open(os.path.join(d, f"clip{j:04d}.npz"), "wb").write(b"cells")
                if cm == "tmp_left" and k == 0 and split == "eval":
                    open(os.path.join(d, "clip9999.npz.tmp"), "wb").write(b"x")
        print("ZZBUILD-OK")
        return 0
    if mode_ == "manifest":
        out = arg(argv, "--out")[0]
        if cm == "no_manifest":
            return 0
        for k in seeds:
            for split in counts:
                d = os.path.join(out, f"M2b_s{k}", split)
                sha = adapter_sha(0 if (cm == "same_adapter" and k == 1) else k)
                json.dump({"kind": "zcell", "grid": "pool4x10->2x5", "n_cells": 10, "d": 256, "dtype": "float16", "adapter_state_sha256": sha, "n_files": len([x for x in os.listdir(d) if x.endswith('.npz')])},
                          open(os.path.join(d, "CELL_MANIFEST.json"), "w"))
        print("ZZMANIFEST-OK")
        return 0
    if mode_ == "check3":
        oj = arg(argv, "--out-json")[0]
        rows = {f"{sp}/clip{j}/s{k}": {"T": 100} for k in seeds for sp, n in (("eval", 6), ("train", 2)) for j in range(n)}
        res = {"tol_identity_fp32": 1e-5, "clips": rows, "max_identity_fp32": 4e-7, "max_stored_vs_cached": 1.2e-4, "min_wrongseed_diff": 0.3,
               "identity_pass": True, "wrongseed_fails_identity": True}
        if cm == "check3_fail":
            res.update(max_identity_fp32=0.5, identity_pass=False)
        elif cm == "wrongseed_passes":
            res.update(min_wrongseed_diff=1e-7, wrongseed_fails_identity=False)
        elif cm == "stored_off":
            res.update(max_stored_vs_cached=0.2)
        json.dump(res, open(oj, "w"), indent=1)
        print("ZZCHECK3-OK" if res["identity_pass"] else "ZZCHECK3-FAIL")
        return 0 if cm != "check3_fail" else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
