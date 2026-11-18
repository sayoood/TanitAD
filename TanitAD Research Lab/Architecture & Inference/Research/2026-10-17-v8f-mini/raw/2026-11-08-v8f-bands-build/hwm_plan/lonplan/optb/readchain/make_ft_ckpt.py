"""Make a REAL FT-format checkpoint with the 4c396aff trainer (optb/tree, CPU, synthetic recipe of test_v8f_optb.py), with the REAL R35_s0 probe at a simulated box-2 path
that is DELETED afterwards, so the checkpoint's cfg.wm_rsp_probe names a path that does not exist. Usage: make_ft_ckpt.py <out_dir> [steps]"""
import hashlib, json, os, shutil, sys
sys.dont_write_bytecode = True
TREE = "F:/TanitAD-scratch/actrsp/optb/tree"
for p in (TREE + "/stack", TREE + "/taniteval", TREE + "/stack/scripts"):
    sys.path.insert(0, p)
import torch
torch.set_num_threads(1)
import v8f_mini_train as TR

out = os.path.abspath(sys.argv[1]); steps = sys.argv[2] if len(sys.argv) > 2 else "2"
REAL = "F:/TanitAD-scratch/actrsp/amend1/out/readout_K10_R35_s0.pt"
sha = hashlib.sha256(open(REAL, "rb").read()).hexdigest()
sim = os.path.join(out, "box2sim", "optb", "inputs"); os.makedirs(sim, exist_ok=True)
probe = os.path.join(sim, "readout_K10_R35_s0.pt"); shutil.copy(REAL, probe)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
WMF = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "2", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]
RSP = ["--wm-init-from-ckpt-wm", "on", "--wm-freeze-proj", "on", "--wm-rsp-w", "1.0", "--wm-rsp-readout", "fixed_k", "--wm-rsp-probe", probe, "--wm-rsp-probe-sha256", sha]
assert TR.main(["--out", out + "/m2b", *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
assert TR.main(["--out", out + "/src", *BASE, "--init-ckpt", out + "/m2b/ckpt_last.pt", *WMF, "--log-every", "1", "--ckpt-every", "100", "--steps", "3"]) == 0
assert TR.main(["--out", out + "/FT", *BASE, "--init-ckpt", out + "/src/ckpt_last.pt", *WMF, *RSP, "--log-every", "1", "--ckpt-every", "100", "--steps", steps]) == 0
shutil.rmtree(os.path.join(out, "box2sim"))
assert not os.path.exists(probe)
ck = torch.load(out + "/FT/ckpt_last.pt", map_location="cpu", weights_only=False)
print(json.dumps({"ckpt": out + "/FT/ckpt_last.pt", "wm_rsp_probe": ck["cfg"]["wm_rsp_probe"], "sha": ck["cfg"]["wm_rsp_probe_sha256"], "probe_exists": os.path.exists(ck["cfg"]["wm_rsp_probe"]), "n_keys": len(ck["model"])}))
