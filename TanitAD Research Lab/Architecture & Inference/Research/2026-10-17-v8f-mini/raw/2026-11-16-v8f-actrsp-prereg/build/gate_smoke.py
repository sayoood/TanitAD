import sys, os, json, tempfile, shutil
import torch
torch.set_num_threads(1)
import tanitad
import v8f_mini_train as TR
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]
root = tempfile.mkdtemp(prefix="gs_", dir=os.environ.get("TMP"))
assert TR.main(["--out", root + "/m2b", *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
try:
    rc = TR.main(["--out", root + "/l", *BASE, "--init-ckpt", root + "/m2b/ckpt_last.pt", *LADDER, "--wm-rsp-w", "1.0", "--selfcheck-only"])
except SystemExit as e:
    rc = e.code
print("RC", rc)
sc = json.load(open(root + "/l/selfcheck.json", encoding="utf-8"))
print(json.dumps(sc["levers"]["wm_rsp_readback"]))
print({k: v for k, v in sc["live"]["terms"].items() if "rsp" in k or k.startswith("wm/level/L1")})
print({k: v for k, v in sc["live_first"].items() if "rsp" in k or k.startswith("wm/level/L1")})
print(sc["live"]["dead_wm_groups"], sc["live"]["missing_terms"], sc["live"]["control_nonwm_grad_sum"])
shutil.rmtree(root, ignore_errors=True)
