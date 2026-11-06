"""G3 pin: frozen_in_proxy fingerprint (sha256 over every tensor the proxy does NOT train) of model_final.pt vs u400, computed with proxy_cache.frozen_in_proxy_fingerprint on a FULL REFe
(strict load). Expected: EQUAL (SFT-4 trained the scorer only; YF cmp_ckpt.out). A cache built from a different snapshot would carry a different value. CPU only."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC
sys.path.insert(0, str(YC.REFE_DIR))
import ckpt_io
import torch
from model import REFe, REFeConfig
import proxy_cache_yf2 as PCY

out = {}
for name, path in (("u400", YC.U400), ("model_final", YC.DATA / "refe_runs_eval" / "model_final.pt")):
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone("vitl16"), str(path))
    m = REFe(cfg).eval()
    ckpt_io.load_for_inference(m, str(path))
    out[name] = {"path": str(path), "sha256": YC.sha256_file(path), "frozen_in_proxy_sha256": PCY.frozen_in_proxy_fingerprint(m), "n_score_components": cfg.n_score_components}
    del m
out["equal"] = out["u400"]["frozen_in_proxy_sha256"] == out["model_final"]["frozen_in_proxy_sha256"]
smoke = HERE / "_smoke_cache" / "manifest.json"
if smoke.exists():
    out["smoke_cache_manifest_matches_u400"] = json.load(open(smoke))["frozen_in_proxy_sha256"] == out["u400"]["frozen_in_proxy_sha256"]
YC.jdump(out, HERE / "g3_fingerprint.json")
print(json.dumps(out, indent=1))
print("ZZG3_OK" if out["equal"] else "ZZG3_RED")
