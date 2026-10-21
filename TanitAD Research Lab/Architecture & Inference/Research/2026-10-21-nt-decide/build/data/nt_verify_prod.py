#!/usr/bin/env python3
"""nt_verify_prod.py -- end-to-end check of PRODUCTION token-episodes (not the smoke's): a random sample of MANIFEST_A1 rows,
symlinked into a scratch dir, opened by the PATCHED loader + the trainer's own ``V3Dataset`` with the INTERFACE R2 filter.
Per item asserts: file sha256 == manifest; one window; frames [8,9,416,1024] all equal to the t0 stitch (sha16 == manifest);
LAW row (future row 4) == ST of the LAW frame (sha16 == manifest law sha, or t0 on fallback); pose_last == (0,0,0,v0);
future_valid_ext.sum() == min(60, n_future_ticks); future_poses_ext[4] == poses at +0.5 s.  Writes VERIFY_PROD.json."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nt_e1 as E  # noqa: E402

NT = "/home/nvidia/r8ns/nt"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    torch.set_num_threads(1)
    rows = [json.loads(ln) for ln in open(f"{NT}/MANIFEST_A1.jsonl") if ln.strip()]
    random.Random(0).shuffle(rows)
    rows = sorted(rows[:n], key=lambda r: r["file"])
    tmp = f"{NT}/scratch_verify"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for r in rows:
        os.symlink(f"{NT}/v2ep_a1/{r['file']}", f"{tmp}/{r['file']}")
    spec = importlib.util.spec_from_file_location("v2_dataset_patched", f"{HERE}/loader_patch/v2_dataset_patched.py")
    P = importlib.util.module_from_spec(spec)
    sys.modules["v2_dataset_patched"] = P
    spec.loader.exec_module(P)
    eps = P.build_v2_providers(tmp, lru_size=2, rebuild=True, verbose=False)
    sys.path.insert(0, f"{E.C}/tree/stack/scripts")
    import refc_v3_train as RV3
    ds = RV3.V3Dataset(eps, window=8, max_horizon=20, channels=9)
    ds.u8_frames = True
    n_before = len(ds.index)
    ds.index = [(e, t) for (e, t) in ds.index if t == 0]
    import gzip
    recs = json.load(gzip.open(f"{NT}/inputs/a1_inputs.json.gz", "rt"))["tokens"]
    out = []
    for i, (e_i, _t) in enumerate(ds.index):
        r = rows[e_i]
        it = ds[i]
        fsha = hashlib.sha256(open(f"{NT}/v2ep_a1/{r['file']}", "rb").read()).hexdigest()
        fr = it["frames"]
        t0_ok = all(E.sha16(fr[j][3 * c:3 * c + 3].permute(1, 2, 0).numpy()) == r["t0_frame_sha16"] for j in range(8) for c in range(3))
        law_want = r["law_frame_sha16"] or r["t0_frame_sha16"]
        law_ok = all(E.sha16(it["future_frames"][4][3 * c:3 * c + 3].permute(1, 2, 0).numpy()) == law_want for c in range(3))
        d = torch.load(f"{NT}/v2ep_a1/{r['file']}", map_location="cpu", weights_only=False)
        pl = it["pose_last"]
        hf = np.asarray(recs[r["token"]]["human_future_poses"], np.float64)
        kk = [k for k in range(1, 9) if 9 + 5 * k < d["poses"].shape[0]]
        mine = d["poses"].numpy()[[9 + 5 * k for k in kk]].astype(np.float64)
        k2 = float(np.abs(mine[:, :2] - hf[[k - 1 for k in kk], :2]).max())
        out.append({"token": r["token"], "file_sha_ok": fsha == r["file_sha256"], "t0_frames_ok": t0_ok, "law_row_ok": law_ok,
                    "law_src": r["law_src"],
                    "pose_last_ok": bool(torch.equal(pl[:3], torch.zeros(3)) and abs(float(pl[3]) - d["nt"]["v0"]) < 1e-5),
                    "valid_ext_ok": int(it["future_valid_ext"].sum()) == min(60, r["n_future_ticks"]),
                    "fut_ext_05s_ok": bool(torch.equal(it["future_poses_ext"][4], d["poses"][14])),
                    "k2_vs_devkit_ok": k2 <= 1e-3, "k2_max_dxy_m": k2, "schema": d["nt"]["schema"],
                    "shapes_ok": list(fr.shape) == [8, 9, 416, 1024] and list(it["future_frames"].shape) == [20, 9, 416, 1024]})
    keys = [k for k in out[0] if k.endswith("_ok")]
    res = {"n_files": len(rows), "windows_before_filter": n_before, "windows_after_filter": len(ds.index),
           "all_ok": {k: all(o[k] for o in out) for k in keys}, "n_law_next_token": sum(o["law_src"] == "next_token" for o in out),
           "items": out}
    res["k2_max_dxy_m"] = max(o["k2_max_dxy_m"] for o in out)
    res["schemas"] = sorted({o["schema"] for o in out})
    res["pass"] = len(ds.index) == len(rows) and all(res["all_ok"].values())
    json.dump(res, open(f"{NT}/VERIFY_PROD.json", "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}))
    print(f"ZZVERIFY_{'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
