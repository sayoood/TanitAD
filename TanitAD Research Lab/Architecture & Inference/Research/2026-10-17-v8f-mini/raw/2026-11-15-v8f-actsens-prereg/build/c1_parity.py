"""C1 (dev-box CPU, synthetic world): run the SAME flag-absent argv under the tree on PYTHONPATH and write its fingerprints to <out.json>:
  - sha256 over the model state_dict of the 2-step M2b checkpoint (names, dtypes, bytes), and of the WM-only ladder arm's checkpoint,
  - the selfcheck.json live.terms of the WM-only ladder arm (--selfcheck-only), bit for bit as json floats,
  - the realised first epoch order of the sampler is covered by test_v8f_actsens.py.
usage: python c1_parity.py <tree_root> <out.json> <scratch_dir>   (run once per tree, then compare the two json files)"""
import hashlib
import json
import os
import sys

tree, outp, scratch = sys.argv[1:4]
sys.path[:0] = [os.path.join(tree, "stack", "scripts"), os.path.join(tree, "stack"), os.path.join(tree, "taniteval")]
import torch  # noqa: E402
import tanitad  # noqa: E402
assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "0", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]


def sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes())
    return h.hexdigest()


res = {"tanitad_file": tanitad.__file__}
os.makedirs(scratch, exist_ok=True)
m2b = os.path.join(scratch, "m2b")
assert TR.main(["--out", m2b, *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
ck = torch.load(os.path.join(m2b, "ckpt_last.pt"), map_location="cpu", weights_only=False)
res["m2b_state_sha256"] = sha_state(ck["model"])
res["m2b_n_tensors"] = len(ck["model"])
wo = os.path.join(scratch, "wo")
assert TR.main(["--out", wo, *BASE, "--init-ckpt", os.path.join(m2b, "ckpt_last.pt"), *LADDER, "--log-every", "1", "--ckpt-every", "100", "--steps", "3"]) == 0
ck2 = torch.load(os.path.join(wo, "ckpt_last.pt"), map_location="cpu", weights_only=False)
res["wmonly_state_sha256"] = sha_state(ck2["model"])
g = os.path.join(scratch, "gate")
assert TR.main(["--out", g, *BASE, "--init-ckpt", os.path.join(m2b, "ckpt_last.pt"), *LADDER, "--selfcheck-only"]) == 0
sc = json.load(open(os.path.join(g, "selfcheck.json"), encoding="utf-8"))
res["live_terms"] = sc["live"]["terms"]
res["wm_groups_grad"] = sc["live"]["wm_groups_grad"]
res["steps_total"] = sc["steps_total"]
res["levers_keys"] = sorted(sc["levers"])
res["info_wm_windows"] = sc["info"].get("wm_windows")
json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)
print("wrote", outp, res["m2b_state_sha256"][:12], res["wmonly_state_sha256"][:12])
